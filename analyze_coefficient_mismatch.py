#!/usr/bin/env python3
# analyze_coefficient_mismatch.py
# Coefficient-space mismatch analysis for developmental cortical harmonic bases.

from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Coefficient-space mismatch analysis"
    )
    parser.add_argument("--eigdir-source", required=True, help="Source eigensystem directory")
    parser.add_argument("--eigdir-target", default=None, help="Target eigensystem directory")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--K", type=int, default=30, help="Modes per hemisphere")
    parser.add_argument("--hemi", choices=["lh", "rh", "both"], default="both", help="Hemisphere(s)")

    parser.add_argument("--neighbor-only", dest="neighbor_only", action="store_true", help="Only adjacent ages")
    parser.add_argument("--all-pairs", dest="neighbor_only", action="store_false", help="All age pairs")
    parser.set_defaults(neighbor_only=True)

    parser.add_argument(
        "--families",
        default="onehot,packet,random",
        help="Families: onehot,packet,random",
    )
    parser.add_argument("--packet-width", type=float, default=1.0, help="Gaussian packet width")
    parser.add_argument("--n-random", type=int, default=200, help="Random draws")
    parser.add_argument("--random-decay", type=float, default=1.0, help="Random variance ~ k^{-p}")
    parser.add_argument("--seed", type=int, default=20260315, help="Random seed")

    parser.add_argument("--save-heatmaps", action="store_true", help="Save coefficient-transfer heatmaps")
    parser.add_argument("--plot-top-k", type=int, default=20, help="Top-k shown in heatmaps")

    parser.add_argument(
        "--heatmap-metric",
        choices=["energy", "amplitude"],
        default=None,
        help="Heatmap quantity: energy uses |P|^2; amplitude uses |P|",
    )
    parser.add_argument(
        "--heatmap-squared",
        action="store_true",
        help="Deprecated alias for --heatmap-metric energy",
    )

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


def sphere_unit(rr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rr, axis=1, keepdims=True)
    return rr / np.maximum(norm, EPS)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx < EPS or ny < EPS:
        return np.nan

    return float(np.dot(x, y) / (nx * ny))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)

    if nx < EPS or ny < EPS:
        return np.nan

    return float(np.dot(x, y) / (nx * ny))


def load_eigs(eigdir: Path, K: int):
    files = sorted(eigdir.glob(f"*_fullmesh_lb_*_K{K}.npz"))
    if not files:
        raise RuntimeError(f"No eigensystem files found in {eigdir.resolve()} for K={K}")

    out = {}
    for path in files:
        x = np.load(path, allow_pickle=True)
        age = scalar_from_npz(x, "age")

        out[age] = {
            "file": path,
            "age": age,
            "subject": scalar_from_npz(x, "subject"),
            "evals_lh": x["evals_lh"].astype(float),
            "evals_rh": x["evals_rh"].astype(float),
            "evecs_lh": x["evecs_lh"].astype(float),
            "evecs_rh": x["evecs_rh"].astype(float),
            "area_lh": x["area_lh"].astype(float),
            "area_rh": x["area_rh"].astype(float),
            "sphere_lh": x["sphere_lh"].astype(float),
            "sphere_rh": x["sphere_rh"].astype(float),
        }

    return out


def map_field_or_basis_via_sphere_nn(
    X_src: np.ndarray,
    sphere_src: np.ndarray,
    sphere_tgt: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(sphere_unit(sphere_src))
    _, idx = tree.query(sphere_unit(sphere_tgt), k=1)
    return X_src[idx, ...]


def coefficient_transfer_operator(
    B_src: np.ndarray,
    sphere_src: np.ndarray,
    B_tgt: np.ndarray,
    area_tgt: np.ndarray,
    sphere_tgt: np.ndarray,
) -> np.ndarray:
    B_src_on_tgt = map_field_or_basis_via_sphere_nn(B_src, sphere_src, sphere_tgt)
    return B_tgt.T @ (area_tgt[:, None] * B_src_on_tgt)


def make_onehot(K: int, center: int) -> np.ndarray:
    c = np.zeros(K, dtype=float)
    c[center - 1] = 1.0
    return c


def make_packet(K: int, center: int, width: float = 1.0) -> np.ndarray:
    idx = np.arange(1, K + 1, dtype=float)
    c = np.exp(-0.5 * ((idx - center) / width) ** 2)
    c /= np.linalg.norm(c) + EPS
    return c


def make_random_lowpass(K: int, p: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(1, K + 1, dtype=float)
    sd = idx ** (-0.5 * p)

    c = rng.normal(size=K) * sd
    c /= np.linalg.norm(c) + EPS
    return c


def energy_weights(c: np.ndarray) -> np.ndarray:
    w = np.abs(c) ** 2
    total = w.sum()

    if total < EPS:
        return np.zeros_like(w)

    return w / total


def primary_metrics(c_tgt: np.ndarray, center: int):
    K = len(c_tgt)
    w = energy_weights(c_tgt)

    exact_retained = float(w[center - 1])

    lo1 = max(1, center - 1)
    hi1 = min(K, center + 1)
    near1_retained = float(w[lo1 - 1 : hi1].sum())

    lo2 = max(1, center - 2)
    hi2 = min(K, center + 2)
    near2_retained = float(w[lo2 - 1 : hi2].sum())

    peak_idx = int(np.argmax(np.abs(c_tgt)) + 1)
    abs_peak_shift = int(abs(peak_idx - center))

    mode_idx = np.arange(1, K + 1, dtype=float)
    centroid = float(np.sum(mode_idx * w))
    abs_centroid_shift = abs(centroid - center)

    return {
        "exact_retained": exact_retained,
        "near1_retained": near1_retained,
        "near2_retained": near2_retained,
        "peak_idx": peak_idx,
        "abs_peak_shift": abs_peak_shift,
        "centroid": centroid,
        "abs_centroid_shift": abs_centroid_shift,
    }


def secondary_metrics(c_src: np.ndarray, c_tgt: np.ndarray, center: int | None = None):
    out = {
        "cosine": cosine_similarity(c_src, c_tgt),
        "pearson": pearson_corr(c_src, c_tgt),
        "mse": float(np.mean((c_src - c_tgt) ** 2)),
        "center_coeff": np.nan,
    }

    if center is not None:
        out["center_coeff"] = float(c_tgt[center - 1])

    return out


def save_transfer_heatmap(
    P: np.ndarray,
    age_src: str,
    age_tgt: str,
    hemi: str,
    out_path: Path,
    plot_top_k: int,
    metric: str,
):
    block = np.abs(P[:plot_top_k, :plot_top_k])

    if metric == "energy":
        block = block ** 2
        cbar_label = r"$|P|^2$ (energy transfer)"
    else:
        cbar_label = r"$|P|$"

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(
        block,
        origin="lower",
        aspect="auto",
        extent=(0.5, plot_top_k + 0.5, 0.5, plot_top_k + 0.5),
    )

    ax.plot(
        np.arange(1, plot_top_k + 1),
        np.arange(1, plot_top_k + 1),
        linestyle="--",
        linewidth=1.0,
        color="white",
        alpha=0.8,
    )
    ax.set_xlabel(f"Source mode index ({age_src})")
    ax.set_ylabel(f"Target mode index ({age_tgt})")
    ax.set_title(f"{hemi.upper()}: coefficient transfer {age_src} → {age_tgt}")

    ticks = np.arange(1, plot_top_k + 1) if plot_top_k <= 10 else np.arange(1, plot_top_k + 1, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    if args.heatmap_metric is None:
        heatmap_metric = "energy" if args.heatmap_squared else "amplitude"
    else:
        heatmap_metric = args.heatmap_metric

    eigdir_source = Path(args.eigdir_source).expanduser().resolve()
    eigdir_target = (
        Path(args.eigdir_target).expanduser().resolve()
        if args.eigdir_target is not None
        else eigdir_source
    )

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    families = [f.strip().lower() for f in args.families.split(",") if f.strip()]
    for family in families:
        if family not in {"onehot", "packet", "random"}:
            raise ValueError(f"Unknown family: {family}")

    src_rows = load_eigs(eigdir_source, args.K)
    tgt_rows = load_eigs(eigdir_target, args.K)

    common_ages = sorted(set(src_rows) & set(tgt_rows), key=age_to_months)
    if len(common_ages) < 2:
        raise RuntimeError("Need at least two common ages across source and target eigensystems.")

    if args.neighbor_only:
        pairs = [(common_ages[i], common_ages[i + 1]) for i in range(len(common_ages) - 1)]
    else:
        pairs = list(itertools.combinations(common_ages, 2))

    hemis = ["lh", "rh"] if args.hemi == "both" else [args.hemi]
    rng = np.random.default_rng(args.seed)

    detail_csv = outdir / f"coefficient_mismatch_detail_K{args.K}.csv"
    summary_csv = outdir / f"coefficient_mismatch_summary_K{args.K}.csv"

    npz_dir = outdir / f"coefficient_transfer_matrices_K{args.K}"
    npz_dir.mkdir(parents=True, exist_ok=True)

    heat_dir = None
    if args.save_heatmaps:
        heat_dir = outdir / f"coefficient_transfer_heatmaps_K{args.K}"
        heat_dir.mkdir(parents=True, exist_ok=True)

    with detail_csv.open("w", newline="") as fdet, summary_csv.open("w", newline="") as fsum:
        writer_det = csv.writer(fdet)
        writer_sum = csv.writer(fsum)

        writer_det.writerow(
            [
                "age_src",
                "age_tgt",
                "age_gap_months",
                "hemi",
                "family",
                "center",
                "exact_retained",
                "near1_retained",
                "near2_retained",
                "abs_peak_shift",
                "abs_centroid_shift",
                "cosine",
                "pearson",
                "mse",
                "peak_idx",
                "centroid",
                "center_coeff",
            ]
        )

        writer_sum.writerow(
            [
                "age_src",
                "age_tgt",
                "age_gap_months",
                "hemi",
                "family",
                "exact_retained_mean",
                "near1_retained_mean",
                "near2_retained_mean",
                "abs_peak_shift_mean",
                "abs_centroid_shift_mean",
                "cosine_mean",
                "pearson_mean",
                "mse_mean",
                "center_coeff_mean",
            ]
        )

        for age_a, age_b in pairs:
            for age_src, age_tgt in [(age_a, age_b), (age_b, age_a)]:
                gap = abs(age_to_months(age_tgt) - age_to_months(age_src))

                row_src = src_rows[age_src]
                row_tgt = tgt_rows[age_tgt]

                save_npz = {
                    "age_src": np.array([age_src], dtype=object),
                    "age_tgt": np.array([age_tgt], dtype=object),
                    "age_gap_months": np.array([gap], dtype=float),
                }

                for hemi in hemis:
                    B_src = row_src[f"evecs_{hemi}"][:, : args.K]
                    B_tgt = row_tgt[f"evecs_{hemi}"][:, : args.K]
                    area_tgt = row_tgt[f"area_{hemi}"]
                    sphere_src = row_src[f"sphere_{hemi}"]
                    sphere_tgt = row_tgt[f"sphere_{hemi}"]

                    P = coefficient_transfer_operator(
                        B_src=B_src,
                        sphere_src=sphere_src,
                        B_tgt=B_tgt,
                        area_tgt=area_tgt,
                        sphere_tgt=sphere_tgt,
                    )
                    save_npz[f"P_{hemi}"] = P

                    if heat_dir is not None:
                        heat_path = heat_dir / f"P_{hemi}_{age_src}_to_{age_tgt}_top{args.plot_top_k}.png"
                        save_transfer_heatmap(
                            P=P,
                            age_src=age_src,
                            age_tgt=age_tgt,
                            hemi=hemi,
                            out_path=heat_path,
                            plot_top_k=min(args.plot_top_k, args.K),
                            metric=heatmap_metric,
                        )

                    for family in families:
                        fam_rows = []

                        if family == "onehot":
                            for center in range(1, args.K + 1):
                                c_src = make_onehot(args.K, center)
                                c_tgt = P @ c_src

                                prim = primary_metrics(c_tgt, center=center)
                                sec = secondary_metrics(c_src, c_tgt, center=center)

                                writer_det.writerow(
                                    [
                                        age_src,
                                        age_tgt,
                                        gap,
                                        hemi,
                                        family,
                                        center,
                                        prim["exact_retained"],
                                        prim["near1_retained"],
                                        prim["near2_retained"],
                                        prim["abs_peak_shift"],
                                        prim["abs_centroid_shift"],
                                        sec["cosine"],
                                        sec["pearson"],
                                        sec["mse"],
                                        prim["peak_idx"],
                                        prim["centroid"],
                                        sec["center_coeff"],
                                    ]
                                )
                                fam_rows.append({**prim, **sec})

                        elif family == "packet":
                            for center in range(1, args.K + 1):
                                c_src = make_packet(args.K, center=center, width=args.packet_width)
                                c_tgt = P @ c_src

                                prim = primary_metrics(c_tgt, center=center)
                                sec = secondary_metrics(c_src, c_tgt, center=center)

                                writer_det.writerow(
                                    [
                                        age_src,
                                        age_tgt,
                                        gap,
                                        hemi,
                                        family,
                                        center,
                                        prim["exact_retained"],
                                        prim["near1_retained"],
                                        prim["near2_retained"],
                                        prim["abs_peak_shift"],
                                        prim["abs_centroid_shift"],
                                        sec["cosine"],
                                        sec["pearson"],
                                        sec["mse"],
                                        prim["peak_idx"],
                                        prim["centroid"],
                                        sec["center_coeff"],
                                    ]
                                )
                                fam_rows.append({**prim, **sec})

                        else:
                            for _ in range(args.n_random):
                                c_src = make_random_lowpass(args.K, p=args.random_decay, rng=rng)
                                c_tgt = P @ c_src
                                sec = secondary_metrics(c_src, c_tgt, center=None)

                                writer_det.writerow(
                                    [
                                        age_src,
                                        age_tgt,
                                        gap,
                                        hemi,
                                        family,
                                        "",
                                        "",
                                        "",
                                        "",
                                        "",
                                        "",
                                        sec["cosine"],
                                        sec["pearson"],
                                        sec["mse"],
                                        "",
                                        "",
                                        "",
                                    ]
                                )
                                fam_rows.append(sec)

                        if family in {"onehot", "packet"}:
                            writer_sum.writerow(
                                [
                                    age_src,
                                    age_tgt,
                                    gap,
                                    hemi,
                                    family,
                                    np.mean([r["exact_retained"] for r in fam_rows]),
                                    np.mean([r["near1_retained"] for r in fam_rows]),
                                    np.mean([r["near2_retained"] for r in fam_rows]),
                                    np.mean([r["abs_peak_shift"] for r in fam_rows]),
                                    np.mean([r["abs_centroid_shift"] for r in fam_rows]),
                                    np.mean([r["cosine"] for r in fam_rows]),
                                    np.mean([r["pearson"] for r in fam_rows]),
                                    np.mean([r["mse"] for r in fam_rows]),
                                    np.mean([r["center_coeff"] for r in fam_rows]),
                                ]
                            )
                        else:
                            writer_sum.writerow(
                                [
                                    age_src,
                                    age_tgt,
                                    gap,
                                    hemi,
                                    family,
                                    "",
                                    "",
                                    "",
                                    "",
                                    "",
                                    np.mean([r["cosine"] for r in fam_rows]),
                                    np.mean([r["pearson"] for r in fam_rows]),
                                    np.mean([r["mse"] for r in fam_rows]),
                                    "",
                                ]
                            )

                    print(f"{age_src:>5} -> {age_tgt:<5} | {hemi.upper()} | saved summaries")

                npz_path = npz_dir / f"coefficient_transfer_{age_src}_to_{age_tgt}_K{args.K}.npz"
                np.savez_compressed(npz_path, **save_npz)

    print(f"\nSaved detailed CSV -> {detail_csv}")
    print(f"Saved summary CSV -> {summary_csv}")
    print(f"Saved transfer matrices -> {npz_dir}")
    if heat_dir is not None:
        print(f"Saved heatmaps -> {heat_dir}")


if __name__ == "__main__":
    main()