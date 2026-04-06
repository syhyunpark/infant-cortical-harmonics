#!/usr/bin/env python3
# analyze_geometry_head_decomposition.py
# One-at-a-time substitution comparisons for neighboring-age sensor-space mismatch.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cortical-basis vs forward-operator substitution analysis"
    )
    parser.add_argument("--dictdir", required=True, help="Directory with saved dictionaries")
    parser.add_argument("--subjects-dir", required=True, help="Subjects directory")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--target-k", type=int, default=50, help="Saved dictionary K")
    parser.add_argument("--analysis-k", type=int, default=5, help="Modes per hemisphere to analyze")

    parser.add_argument("--surface", default="white", help="Filter by surface")
    parser.add_argument("--combine", default="block", help="Filter by combine")
    parser.add_argument("--montage-kind", default="template1020", help="Filter by montage_kind")

    parser.add_argument(
        "--prior-kind",
        choices=["balanced", "decay"],
        default="balanced",
        help="Pattern prior",
    )
    parser.add_argument("--prior-power", type=float, default=1.0, help="Decay power")
    parser.add_argument("--n-patterns", type=int, default=1000, help="Number of simulated patterns")
    parser.add_argument("--seed", type=int, default=20260315, help="Random seed")

    parser.add_argument("--neighbor-only", dest="neighbor_only", action="store_true", help="Only adjacent ages")
    parser.add_argument("--all-pairs", dest="neighbor_only", action="store_false", help="All age pairs")
    parser.set_defaults(neighbor_only=True)

    parser.add_argument("--figscale", type=float, default=1.0, help="Overall figure scale")
    parser.add_argument("--dpi", type=int, default=350, help="Output DPI")
    return parser.parse_args()


def age_to_months(age: str) -> float:
    age = str(age).strip().lower()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(wk|mo|yr)", age)
    if m is None:
        return float("inf")

    value = float(m.group(1))
    unit = m.group(2)

    if unit == "wk":
        return value / 4.34524
    if unit == "mo":
        return value
    if unit == "yr":
        return value * 12.0

    return float("inf")


def scalar_from_npz(x, key):
    arr = x[key]
    if getattr(arr, "shape", ()) == ():
        return arr.item()

    if len(arr) == 1:
        value = arr[0]
        return value.item() if hasattr(value, "item") else value

    return arr


def filter_tag(surface: str, combine: str, montage_kind: str, target_k: int, analysis_k: int) -> str:
    return f"K{target_k}_AK{analysis_k}_{surface}_{combine}_{montage_kind}"


def sphere_unit(rr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rr, axis=1, keepdims=True)
    return rr / np.maximum(norm, EPS)


def _extract_vertno(x, left: bool):
    if left:
        for key in ("vertno_lh", "lh_vertno"):
            if key in x:
                return x[key].astype(int)
    else:
        for key in ("vertno_rh", "rh_vertno"):
            if key in x:
                return x[key].astype(int)
    return None


def load_rows(dictdir: Path, target_k: int, surface: str, combine: str, montage_kind: str):
    rows = []

    for path in sorted(dictdir.glob("*_D_*.npz")):
        x = np.load(path, allow_pickle=True)

        file_K = int(np.array(x["K"]).ravel()[0])
        file_surface = scalar_from_npz(x, "surface")
        file_combine = scalar_from_npz(x, "combine")
        file_montage = scalar_from_npz(x, "montage_kind")

        if file_K != target_k:
            continue
        if file_surface != surface:
            continue
        if file_combine != combine:
            continue
        if file_montage != montage_kind:
            continue

        vertno_lh = _extract_vertno(x, left=True)
        vertno_rh = _extract_vertno(x, left=False)

        if vertno_lh is None or vertno_rh is None:
            raise RuntimeError(
                f"{path.name} does not contain source vertex indices.\n"
                f"Expected one of: vertno_lh/lh_vertno and vertno_rh/rh_vertno.\n"
                f"Please regenerate the matching dictionary files with source-vertex indices saved."
            )

        rows.append(
            {
                "file": path,
                "age": scalar_from_npz(x, "age"),
                "subject": scalar_from_npz(x, "subject"),
                "G": x["G"].astype(float),
                "Phi": x["Phi"].astype(float),
                "combine": file_combine,
                "surface": file_surface,
                "spacing": scalar_from_npz(x, "spacing"),
                "montage_kind": file_montage,
                "K": file_K,
                "vertno_lh": vertno_lh,
                "vertno_rh": vertno_rh,
            }
        )

    rows.sort(key=lambda r: age_to_months(r["age"]))

    if not rows:
        raise RuntimeError(
            f"No files matched filters: K={target_k}, surface={surface}, combine={combine}, montage_kind={montage_kind}"
        )

    return rows


def load_source_sphere_coords(subjects_dir: Path, subject: str, hemi: str, vertno: np.ndarray) -> np.ndarray:
    surf_path = subjects_dir / subject / "surf" / f"{hemi}.sphere.reg"
    rr, _ = nib.freesurfer.read_geometry(str(surf_path))
    return rr[vertno].astype(float)


def map_basis_between_source_spaces(
    Phi_src_hemi: np.ndarray,
    sphere_src: np.ndarray,
    sphere_tgt: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(sphere_unit(sphere_src))
    _, idx = tree.query(sphere_unit(sphere_tgt), k=1)
    return Phi_src_hemi[idx, :]


def get_analysis_parts(row, subjects_dir: Path, analysis_k: int):
    if row["combine"] != "block":
        raise ValueError("This script currently assumes combine='block'.")

    K_file = row["K"]
    if analysis_k > K_file:
        raise ValueError(f"analysis_k={analysis_k} exceeds saved K={K_file}")

    Phi = row["Phi"]
    G = row["G"]

    n_lh = len(row["vertno_lh"])
    n_rh = len(row["vertno_rh"])

    if n_lh + n_rh != Phi.shape[0]:
        raise RuntimeError(f"{row['file'].name}: Phi row count does not match vertno counts")
    if n_lh + n_rh != G.shape[1]:
        raise RuntimeError(f"{row['file'].name}: G column count does not match vertno counts")

    Phi_lh_all = Phi[:n_lh, :]
    Phi_rh_all = Phi[n_lh:, :]

    Phi_lh = Phi_lh_all[:, :analysis_k]
    Phi_rh = Phi_rh_all[:, K_file:K_file + analysis_k]

    sph_lh = load_source_sphere_coords(subjects_dir, row["subject"], "lh", row["vertno_lh"])
    sph_rh = load_source_sphere_coords(subjects_dir, row["subject"], "rh", row["vertno_rh"])

    Phi_native = np.block(
        [
            [Phi_lh, np.zeros((n_lh, analysis_k))],
            [np.zeros((n_rh, analysis_k)), Phi_rh],
        ]
    )
    D_native = G @ Phi_native

    return {
        "age": row["age"],
        "subject": row["subject"],
        "G": G,
        "Phi_lh": Phi_lh,
        "Phi_rh": Phi_rh,
        "Phi_native": Phi_native,
        "D_native": D_native,
        "sphere_lh": sph_lh,
        "sphere_rh": sph_rh,
        "n_lh": n_lh,
        "n_rh": n_rh,
    }


def build_basis_substitution_dictionary(head_row, basis_from_other, analysis_k: int) -> np.ndarray:
    Phi_lh_map = map_basis_between_source_spaces(
        basis_from_other["Phi_lh"],
        basis_from_other["sphere_lh"],
        head_row["sphere_lh"],
    )
    Phi_rh_map = map_basis_between_source_spaces(
        basis_from_other["Phi_rh"],
        basis_from_other["sphere_rh"],
        head_row["sphere_rh"],
    )

    Phi_hybrid = np.block(
        [
            [Phi_lh_map, np.zeros((head_row["n_lh"], analysis_k))],
            [np.zeros((head_row["n_rh"], analysis_k)), Phi_rh_map],
        ]
    )
    return head_row["G"] @ Phi_hybrid


def orthonormal_basis(D: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    if s.size == 0:
        return np.zeros((D.shape[0], 0), dtype=float)

    rank = int(np.sum(s > rtol * s[0]))
    return U[:, :rank]


def projector(Q: np.ndarray) -> np.ndarray:
    return Q @ Q.T


def projector_distance(Qa: np.ndarray, Qb: np.ndarray) -> float:
    return float(np.linalg.norm(projector(Qa) - projector(Qb), ord="fro"))


def principal_angles(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.array([], dtype=float)

    M = Qa.T @ Qb
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def make_mode_orders(analysis_k: int) -> np.ndarray:
    return np.concatenate([np.arange(1, analysis_k + 1), np.arange(1, analysis_k + 1)])


def make_prior_variances(orders: np.ndarray, kind: str, power: float) -> np.ndarray:
    if kind == "balanced":
        return np.ones_like(orders, dtype=float)
    if kind == "decay":
        return orders.astype(float) ** (-power)
    raise ValueError("Unknown prior kind")


def center_patterns(Y: np.ndarray) -> np.ndarray:
    return Y - Y.mean(axis=0, keepdims=True)


def r2_projection(Y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Yc = center_patterns(Y)
    Yhat = Q @ (Q.T @ Y)
    Yhat_c = center_patterns(Yhat)

    num = np.sum((Yc - Yhat_c) ** 2, axis=0)
    den = np.maximum(np.sum(Yc ** 2, axis=0), EPS)

    return 1.0 - num / den


def simulate_reference_patterns(
    D_ref: np.ndarray,
    analysis_k: int,
    prior_kind: str,
    prior_power: float,
    n_patterns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    orders = make_mode_orders(analysis_k)
    tau2 = make_prior_variances(orders, kind=prior_kind, power=prior_power)
    W = rng.normal(size=(2 * analysis_k, n_patterns)) * np.sqrt(tau2)[:, None]
    return D_ref @ W


def evaluate_same_reference_patterns(Y_ref: np.ndarray, D_ref: np.ndarray, D_alt: np.ndarray):
    Q_ref = orthonormal_basis(D_ref)
    Q_alt = orthonormal_basis(D_alt)

    r2_within = r2_projection(Y_ref, Q_ref)
    r2_cross = r2_projection(Y_ref, Q_alt)
    delta = r2_within - r2_cross

    ang = principal_angles(Q_ref, Q_alt)

    return {
        "within_r2_mean": float(np.mean(r2_within)),
        "cross_r2_mean": float(np.mean(r2_cross)),
        "delta_r2_mean": float(np.mean(delta)),
        "within_r2_median": float(np.median(r2_within)),
        "cross_r2_median": float(np.median(r2_cross)),
        "delta_r2_median": float(np.median(delta)),
        "mean_angle_deg": float(np.mean(np.degrees(ang))) if len(ang) else np.nan,
        "max_angle_deg": float(np.max(np.degrees(ang))) if len(ang) else np.nan,
        "projector_distance": projector_distance(Q_ref, Q_alt),
        "rank_ref": int(Q_ref.shape[1]),
        "rank_alt": int(Q_alt.shape[1]),
    }


def save_main_figure(
    labels,
    full_vals,
    basis_vals,
    fwd_vals,
    ylabel,
    title,
    out_path: Path,
    figscale: float,
    dpi: int,
):
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.8 * figscale, 4.8 * figscale))
    ax.plot(x, full_vals, marker="o", label="full age-specific mismatch")
    ax.plot(x, basis_vals, marker="s", label="cortical-basis substitution")
    ax.plot(x, fwd_vals, marker="^", label="forward-operator substitution")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    dictdir = Path(args.dictdir).expanduser().resolve()
    subjects_dir = Path(args.subjects_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(
        dictdir=dictdir,
        target_k=args.target_k,
        surface=args.surface,
        combine=args.combine,
        montage_kind=args.montage_kind,
    )
    parts = [get_analysis_parts(row, subjects_dir, args.analysis_k) for row in rows]

    if args.neighbor_only:
        pair_idx = [(i, i + 1) for i in range(len(parts) - 1)]
    else:
        pair_idx = [(i, j) for i in range(len(parts)) for j in range(i + 1, len(parts))]

    rng = np.random.default_rng(args.seed)
    tag = filter_tag(args.surface, args.combine, args.montage_kind, args.target_k, args.analysis_k)

    out_csv = outdir / f"basis_forwardoperator_decomposition_{tag}.csv"

    labels = []

    full_delta_sym = []
    basis_delta_sym = []
    fwdop_delta_sym = []

    full_proj_sym = []
    basis_proj_sym = []
    fwdop_proj_sym = []

    with out_csv.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "age_1",
                "age_2",
                "age_gap_months",
                "full_delta_r2_sym",
                "corticalbasis_delta_r2_sym",
                "forwardoperator_delta_r2_sym",
                "full_projector_distance_sym",
                "corticalbasis_projector_distance_sym",
                "forwardoperator_projector_distance_sym",
                "full_mean_angle_deg_sym",
                "corticalbasis_mean_angle_deg_sym",
                "forwardoperator_mean_angle_deg_sym",
            ]
        )

        for i, j in pair_idx:
            a = parts[i]
            b = parts[j]

            age1, age2 = a["age"], b["age"]
            gap = abs(age_to_months(age2) - age_to_months(age1))
            labels.append(f"{age1}–{age2}")

            Da = a["D_native"]
            Db = b["D_native"]

            D_basis_b_on_a = build_basis_substitution_dictionary(a, b, args.analysis_k)
            D_basis_a_on_b = build_basis_substitution_dictionary(b, a, args.analysis_k)

            Y_a = simulate_reference_patterns(
                D_ref=Da,
                analysis_k=args.analysis_k,
                prior_kind=args.prior_kind,
                prior_power=args.prior_power,
                n_patterns=args.n_patterns,
                rng=rng,
            )
            Y_b = simulate_reference_patterns(
                D_ref=Db,
                analysis_k=args.analysis_k,
                prior_kind=args.prior_kind,
                prior_power=args.prior_power,
                n_patterns=args.n_patterns,
                rng=rng,
            )

            full_a_to_b = evaluate_same_reference_patterns(Y_ref=Y_a, D_ref=Da, D_alt=Db)
            full_b_to_a = evaluate_same_reference_patterns(Y_ref=Y_b, D_ref=Db, D_alt=Da)

            basis_on_a = evaluate_same_reference_patterns(Y_ref=Y_a, D_ref=Da, D_alt=D_basis_b_on_a)
            basis_on_b = evaluate_same_reference_patterns(Y_ref=Y_b, D_ref=Db, D_alt=D_basis_a_on_b)

            fwd_on_a = evaluate_same_reference_patterns(Y_ref=Y_a, D_ref=Da, D_alt=D_basis_a_on_b)
            fwd_on_b = evaluate_same_reference_patterns(Y_ref=Y_b, D_ref=Db, D_alt=D_basis_b_on_a)

            full_delta = 0.5 * (full_a_to_b["delta_r2_mean"] + full_b_to_a["delta_r2_mean"])
            basis_delta = 0.5 * (basis_on_a["delta_r2_mean"] + basis_on_b["delta_r2_mean"])
            fwdop_delta = 0.5 * (fwd_on_a["delta_r2_mean"] + fwd_on_b["delta_r2_mean"])

            full_proj = 0.5 * (full_a_to_b["projector_distance"] + full_b_to_a["projector_distance"])
            basis_proj = 0.5 * (basis_on_a["projector_distance"] + basis_on_b["projector_distance"])
            fwdop_proj = 0.5 * (fwd_on_a["projector_distance"] + fwd_on_b["projector_distance"])

            full_ang = 0.5 * (full_a_to_b["mean_angle_deg"] + full_b_to_a["mean_angle_deg"])
            basis_ang = 0.5 * (basis_on_a["mean_angle_deg"] + basis_on_b["mean_angle_deg"])
            fwdop_ang = 0.5 * (fwd_on_a["mean_angle_deg"] + fwd_on_b["mean_angle_deg"])

            full_delta_sym.append(full_delta)
            basis_delta_sym.append(basis_delta)
            fwdop_delta_sym.append(fwdop_delta)

            full_proj_sym.append(full_proj)
            basis_proj_sym.append(basis_proj)
            fwdop_proj_sym.append(fwdop_proj)

            writer.writerow(
                [
                    age1,
                    age2,
                    gap,
                    full_delta,
                    basis_delta,
                    fwdop_delta,
                    full_proj,
                    basis_proj,
                    fwdop_proj,
                    full_ang,
                    basis_ang,
                    fwdop_ang,
                ]
            )

            print(
                f"{age1:>5} vs {age2:<5} | "
                f"ΔR2 full={full_delta:.3f}, basis={basis_delta:.3f}, fwdop={fwdop_delta:.3f}"
            )

    print(f"Saved -> {out_csv}")

    save_main_figure(
        labels=labels,
        full_vals=full_delta_sym,
        basis_vals=basis_delta_sym,
        fwd_vals=fwdop_delta_sym,
        ylabel=r"Symmetrized $\Delta R^2$",
        title="Cortical-basis and forward-operator contributions to sensor-space mismatch",
        out_path=outdir / f"basis_forwardoperator_decomposition_deltaR2_{tag}.png",
        figscale=args.figscale,
        dpi=args.dpi,
    )

    save_main_figure(
        labels=labels,
        full_vals=full_proj_sym,
        basis_vals=basis_proj_sym,
        fwd_vals=fwdop_proj_sym,
        ylabel="Symmetrized projector distance",
        title="Cortical-basis and forward-operator contributions to sensor-subspace distance",
        out_path=outdir / f"basis_forwardoperator_decomposition_projdist_{tag}.png",
        figscale=args.figscale,
        dpi=args.dpi,
    )

if __name__ == "__main__":
    main()