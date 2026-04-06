#!/usr/bin/env python3
# analyze_age_mismatch_sensor.py
# Sensor-space age-mismatch analysis using age-specific dictionaries D^(a).

from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="Sensor-space age-mismatch analysis")
    parser.add_argument("--outdir", required=True, help="Directory with saved D files")
    parser.add_argument("--figdir", required=True, help="Output directory")

    parser.add_argument("--target-k", type=int, default=50, help="File K to use")
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

    parser.add_argument("--noise-snr-db", type=float, default=None, help="Optional sensor SNR in dB")
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


def filter_tag(surface, combine, montage_kind, target_k: int, analysis_k: int) -> str:
    s = surface if surface is not None else "anysurf"
    c = combine if combine is not None else "anycombine"
    m = montage_kind if montage_kind is not None else "anymontage"
    return f"K{target_k}_AK{analysis_k}_{s}_{c}_{m}"


def load_rows(outdir: Path, target_k: int, surface, combine, montage_kind):
    rows = []

    for path in sorted(outdir.glob("*_D_*.npz")):
        x = np.load(path, allow_pickle=True)

        row = {
            "file": path,
            "age": scalar_from_npz(x, "age"),
            "subject": scalar_from_npz(x, "subject"),
            "D": x["D"].astype(float),
            "combine": scalar_from_npz(x, "combine"),
            "surface": scalar_from_npz(x, "surface"),
            "spacing": scalar_from_npz(x, "spacing"),
            "montage_kind": scalar_from_npz(x, "montage_kind"),
            "K": int(np.array(x["K"]).ravel()[0]),
        }

        if row["K"] != target_k:
            continue
        if surface is not None and row["surface"] != surface:
            continue
        if combine is not None and row["combine"] != combine:
            continue
        if montage_kind is not None and row["montage_kind"] != montage_kind:
            continue

        rows.append(row)

    rows.sort(key=lambda r: age_to_months(r["age"]))

    if not rows:
        raise RuntimeError(
            f"No dictionary files matched filters: "
            f"K={target_k}, surface={surface}, combine={combine}, montage_kind={montage_kind}"
        )

    return rows


def subset_dictionary(D: np.ndarray, combine: str, K_file: int, analysis_k: int) -> np.ndarray:
    if analysis_k <= 0:
        raise ValueError("analysis_k must be positive")

    if combine == "block":
        if analysis_k > K_file:
            raise ValueError(f"analysis_k={analysis_k} exceeds file K={K_file}")

        idx = np.r_[0:analysis_k, K_file:K_file + analysis_k]
        return D[:, idx]

    if analysis_k > D.shape[1]:
        raise ValueError(f"analysis_k={analysis_k} exceeds available columns {D.shape[1]}")

    return D[:, :analysis_k]


def make_mode_orders(n_cols: int, combine: str, analysis_k: int) -> np.ndarray:
    if combine == "block":
        if n_cols != 2 * analysis_k:
            raise ValueError(f"Expected {2 * analysis_k} columns, got {n_cols}")
        return np.concatenate([np.arange(1, analysis_k + 1), np.arange(1, analysis_k + 1)])

    return np.arange(1, n_cols + 1)


def make_prior_variances(orders: np.ndarray, kind: str = "balanced", power: float = 1.0) -> np.ndarray:
    if kind == "balanced":
        return np.ones_like(orders, dtype=float)
    if kind == "decay":
        return orders.astype(float) ** (-power)
    raise ValueError("kind must be 'balanced' or 'decay'")


def orthonormal_basis(D: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    U, s, _ = np.linalg.svd(D, full_matrices=False)

    if s.size == 0:
        return np.zeros((D.shape[0], 0), dtype=float)

    rank = int(np.sum(s > rtol * s[0]))
    return U[:, :rank]


def projector(Q: np.ndarray) -> np.ndarray:
    return Q @ Q.T


def principal_angles(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return np.array([], dtype=float)

    M = Qa.T @ Qb
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def projector_distance(Qa: np.ndarray, Qb: np.ndarray) -> float:
    return float(np.linalg.norm(projector(Qa) - projector(Qb), ord="fro"))


def center_patterns(Y: np.ndarray) -> np.ndarray:
    return Y - Y.mean(axis=0, keepdims=True)


def r2_projection(Y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Yc = center_patterns(Y)
    Yhat = Q @ (Q.T @ Y)
    Yhat_c = center_patterns(Yhat)

    num = np.sum((Yc - Yhat_c) ** 2, axis=0)
    den = np.maximum(np.sum(Yc ** 2, axis=0), EPS)

    return 1.0 - num / den


def expected_sensor_power(D: np.ndarray, tau2: np.ndarray) -> float:
    M = D.shape[0]
    col_norm2 = np.sum(D ** 2, axis=0)
    return float(np.sum(tau2 * col_norm2) / M)


def save_neighbor_loss_plot(labels, vals, out_path: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))

    ax.plot(x, vals, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_matrix(mat: np.ndarray, ages, out_path: Path, title: str, cbar_label: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, origin="lower", aspect="auto")

    ax.set_xticks(np.arange(len(ages)))
    ax.set_yticks(np.arange(len(ages)))
    ax.set_xticklabels(ages, rotation=45, ha="right")
    ax.set_yticklabels(ages)

    ax.set_xlabel("Target dictionary age")
    ax.set_ylabel("Source pattern age")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_angles_plot(labels, mean_angles_deg, max_angles_deg, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))

    ax.plot(x, mean_angles_deg, marker="o", label="mean principal angle")
    ax.plot(x, max_angles_deg, marker="s", label="max principal angle")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    figdir = Path(args.figdir).expanduser().resolve()
    figdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(
        outdir=outdir,
        target_k=args.target_k,
        surface=args.surface,
        combine=args.combine,
        montage_kind=args.montage_kind,
    )

    tag = filter_tag(
        surface=args.surface,
        combine=args.combine,
        montage_kind=args.montage_kind,
        target_k=args.target_k,
        analysis_k=args.analysis_k,
    )

    print(f"Loaded {len(rows)} dictionaries with filters:")
    print(
        f"  K={args.target_k}, AK={args.analysis_k}, surface={args.surface}, "
        f"combine={args.combine}, montage_kind={args.montage_kind}\n"
    )

    if args.neighbor_only:
        pair_indices = [(i, i + 1) for i in range(len(rows) - 1)]
    else:
        pair_indices = list(itertools.combinations(range(len(rows)), 2))

    rng = np.random.default_rng(args.seed)

    summary_csv = figdir / f"sensor_age_mismatch_summary_{tag}.csv"

    ages = [row["age"] for row in rows]
    n_age = len(rows)

    mat_cross = np.full((n_age, n_age), np.nan)
    mat_delta = np.full((n_age, n_age), np.nan)
    mat_projdist = np.full((n_age, n_age), np.nan)

    neighbor_labels = []
    neighbor_delta = []
    neighbor_cross = []
    neighbor_projdist = []
    neighbor_mean_angle = []
    neighbor_max_angle = []

    with summary_csv.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "age_a",
                "age_b",
                "direction",
                "age_gap_months",
                "rank_a",
                "rank_b",
                "mean_angle_deg",
                "max_angle_deg",
                "projector_distance",
                "within_r2_mean",
                "cross_r2_mean",
                "delta_r2_mean",
                "within_r2_median",
                "cross_r2_median",
                "delta_r2_median",
                "noise_snr_db",
                "surface",
                "combine",
                "montage_kind",
            ]
        )

        for i, j in pair_indices:
            ra = rows[i]
            rb = rows[j]

            age_a = ra["age"]
            age_b = rb["age"]
            age_gap_months = abs(age_to_months(age_b) - age_to_months(age_a))

            Da = subset_dictionary(ra["D"], ra["combine"], ra["K"], args.analysis_k)
            Db = subset_dictionary(rb["D"], rb["combine"], rb["K"], args.analysis_k)

            Qa = orthonormal_basis(Da)
            Qb = orthonormal_basis(Db)

            angles = principal_angles(Qa, Qb)
            mean_angle_deg = float(np.mean(np.degrees(angles))) if angles.size else np.nan
            max_angle_deg = float(np.max(np.degrees(angles))) if angles.size else np.nan
            pdist = projector_distance(Qa, Qb)

            orders_a = make_mode_orders(Da.shape[1], ra["combine"], args.analysis_k)
            tau2_a = make_prior_variances(
                orders_a,
                kind=args.prior_kind,
                power=args.prior_power,
            )

            W = rng.normal(size=(Da.shape[1], args.n_patterns)) * np.sqrt(tau2_a)[:, None]
            Ya = Da @ W

            if args.noise_snr_db is not None:
                sig_power = expected_sensor_power(Da, tau2_a)
                sigma2 = sig_power / (10.0 ** (args.noise_snr_db / 10.0))
                Ya = Ya + rng.normal(scale=np.sqrt(sigma2), size=Ya.shape)

            r2_within = r2_projection(Ya, Qa)
            r2_cross = r2_projection(Ya, Qb)
            delta = r2_within - r2_cross

            writer.writerow(
                [
                    age_a,
                    age_b,
                    "a_to_b",
                    age_gap_months,
                    Qa.shape[1],
                    Qb.shape[1],
                    mean_angle_deg,
                    max_angle_deg,
                    pdist,
                    float(np.mean(r2_within)),
                    float(np.mean(r2_cross)),
                    float(np.mean(delta)),
                    float(np.median(r2_within)),
                    float(np.median(r2_cross)),
                    float(np.median(delta)),
                    args.noise_snr_db if args.noise_snr_db is not None else "",
                    args.surface,
                    args.combine,
                    args.montage_kind,
                ]
            )

            orders_b = make_mode_orders(Db.shape[1], rb["combine"], args.analysis_k)
            tau2_b = make_prior_variances(
                orders_b,
                kind=args.prior_kind,
                power=args.prior_power,
            )

            Wb = rng.normal(size=(Db.shape[1], args.n_patterns)) * np.sqrt(tau2_b)[:, None]
            Yb = Db @ Wb

            if args.noise_snr_db is not None:
                sig_power_b = expected_sensor_power(Db, tau2_b)
                sigma2_b = sig_power_b / (10.0 ** (args.noise_snr_db / 10.0))
                Yb = Yb + rng.normal(scale=np.sqrt(sigma2_b), size=Yb.shape)

            r2_within_b = r2_projection(Yb, Qb)
            r2_cross_b = r2_projection(Yb, Qa)
            delta_b = r2_within_b - r2_cross_b

            writer.writerow(
                [
                    age_b,
                    age_a,
                    "b_to_a",
                    age_gap_months,
                    Qb.shape[1],
                    Qa.shape[1],
                    mean_angle_deg,
                    max_angle_deg,
                    pdist,
                    float(np.mean(r2_within_b)),
                    float(np.mean(r2_cross_b)),
                    float(np.mean(delta_b)),
                    float(np.median(r2_within_b)),
                    float(np.median(r2_cross_b)),
                    float(np.median(delta_b)),
                    args.noise_snr_db if args.noise_snr_db is not None else "",
                    args.surface,
                    args.combine,
                    args.montage_kind,
                ]
            )

            print(
                f"{age_a:>5} -> {age_b:<5} | "
                f"mean angle={mean_angle_deg:6.2f}° | "
                f"max angle={max_angle_deg:6.2f}° | "
                f"projdist={pdist:6.3f} | "
                f"R2 within={np.mean(r2_within):.3f} "
                f"cross={np.mean(r2_cross):.3f} "
                f"delta={np.mean(delta):.3f}"
            )

            mat_cross[i, j] = float(np.mean(r2_cross))
            mat_cross[j, i] = float(np.mean(r2_cross_b))

            mat_delta[i, j] = float(np.mean(delta))
            mat_delta[j, i] = float(np.mean(delta_b))

            mat_projdist[i, j] = pdist
            mat_projdist[j, i] = pdist

            if args.neighbor_only:
                neighbor_labels.append(f"{age_a}→{age_b}")
                neighbor_delta.append(float(np.mean(delta)))
                neighbor_cross.append(float(np.mean(r2_cross)))
                neighbor_projdist.append(pdist)
                neighbor_mean_angle.append(mean_angle_deg)
                neighbor_max_angle.append(max_angle_deg)

    print(f"\nSaved summary CSV -> {summary_csv}")

    pair_tag = "neighbors" if args.neighbor_only else "allpairs"
    noise_tag = (
        f"_snr{str(args.noise_snr_db).replace('.', 'p')}dB"
        if args.noise_snr_db is not None
        else "_noisefree"
    )

    save_pairwise_matrix(
        mat=mat_cross,
        ages=ages,
        out_path=figdir / f"sensor_crossR2_matrix_{tag}_{pair_tag}{noise_tag}.png",
        title=f"Cross-age sensor representation ({tag})",
        cbar_label="Cross-age mean R^2",
    )

    save_pairwise_matrix(
        mat=mat_delta,
        ages=ages,
        out_path=figdir / f"sensor_deltaR2_matrix_{tag}_{pair_tag}{noise_tag}.png",
        title=f"Cross-age mismatch loss ({tag})",
        cbar_label="ΔR^2 = within-age R^2 - cross-age R^2",
    )

    save_pairwise_matrix(
        mat=mat_projdist,
        ages=ages,
        out_path=figdir / f"sensor_projdist_matrix_{tag}_{pair_tag}{noise_tag}.png",
        title=f"Sensor-subspace projector distance ({tag})",
        cbar_label="Projector distance",
    )

    if args.neighbor_only:
        save_neighbor_loss_plot(
            labels=neighbor_labels,
            vals=neighbor_delta,
            out_path=figdir / f"sensor_neighbor_deltaR2_{tag}{noise_tag}.png",
            title=f"Neighbor age-mismatch loss ({tag})",
            ylabel="ΔR^2",
        )

        save_neighbor_loss_plot(
            labels=neighbor_labels,
            vals=neighbor_cross,
            out_path=figdir / f"sensor_neighbor_crossR2_{tag}{noise_tag}.png",
            title=f"Neighbor cross-age R^2 ({tag})",
            ylabel="Cross-age mean R^2",
        )

        save_neighbor_loss_plot(
            labels=neighbor_labels,
            vals=neighbor_projdist,
            out_path=figdir / f"sensor_neighbor_projdist_{tag}{noise_tag}.png",
            title=f"Neighbor projector distance ({tag})",
            ylabel="Projector distance",
        )

        save_angles_plot(
            labels=neighbor_labels,
            mean_angles_deg=neighbor_mean_angle,
            max_angles_deg=neighbor_max_angle,
            out_path=figdir / f"sensor_neighbor_angles_{tag}{noise_tag}.png",
            title=f"Neighbor principal angles ({tag})",
        )

    print(f"Saved figures -> {figdir}")


if __name__ == "__main__":
    main()