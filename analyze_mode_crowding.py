#!/usr/bin/env python3
# analyze_mode_crowding.py
# Relate local eigenvalue crowding to cross-age matching instability.

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
        description="Mode crowding and cross-age instability analysis"
    )
    parser.add_argument("--eigdir", required=True, help="Directory with full-mesh eigensystems")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--K", type=int, default=30, help="Modes per hemisphere")

    parser.add_argument("--neighbor-only", dest="neighbor_only", action="store_true", help="Only adjacent ages")
    parser.add_argument("--all-pairs", dest="neighbor_only", action="store_false", help="All age pairs")
    parser.set_defaults(neighbor_only=True)

    parser.add_argument("--nbins", type=int, default=6, help="Number of bins for probability curves")
    parser.add_argument("--figscale", type=float, default=1.0, help="Figure scale")
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


def load_eigs(eigdir: Path, K: int):
    files = sorted(eigdir.glob(f"*_fullmesh_lb_*_K{K}.npz"))
    if not files:
        raise RuntimeError(f"No eigensystem files found in {eigdir.resolve()} for K={K}")

    rows = []
    for path in files:
        x = np.load(path, allow_pickle=True)
        rows.append(
            {
                "file": path,
                "age": scalar_from_npz(x, "age"),
                "subject": scalar_from_npz(x, "subject"),
                "surface": scalar_from_npz(x, "surface"),
                "sphere_surf": scalar_from_npz(x, "sphere_surf"),
                "K": int(np.array(x["K"]).ravel()[0]),
                "evals_lh": x["evals_lh"].astype(float),
                "evals_rh": x["evals_rh"].astype(float),
                "evecs_lh": x["evecs_lh"].astype(float),
                "evecs_rh": x["evecs_rh"].astype(float),
                "area_lh": x["area_lh"].astype(float),
                "area_rh": x["area_rh"].astype(float),
                "sphere_lh": x["sphere_lh"].astype(float),
                "sphere_rh": x["sphere_rh"].astype(float),
            }
        )

    rows.sort(key=lambda r: age_to_months(r["age"]))
    return rows


def local_spacing_measures(evals: np.ndarray):
    evals = np.asarray(evals, dtype=float)
    K = len(evals)

    spacing_abs = np.full(K, np.nan, dtype=float)

    for i in range(K):
        gaps = []
        if i > 0:
            gaps.append(evals[i] - evals[i - 1])
        if i < K - 1:
            gaps.append(evals[i + 1] - evals[i])

        spacing_abs[i] = np.min(gaps) if gaps else np.nan

    spacing_rel = spacing_abs / np.maximum(evals, EPS)
    crowding = -np.log10(np.maximum(spacing_rel, EPS))

    return spacing_abs, spacing_rel, crowding


def sphere_unit(rr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rr, axis=1, keepdims=True)
    return rr / np.maximum(norm, EPS)


def normalize_cols_area(U: np.ndarray, area: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.sum((U ** 2) * area[:, None], axis=0))
    return U / np.maximum(norms, EPS)[None, :]


def map_modes_via_sphere_nn(U_src: np.ndarray, sphere_src: np.ndarray, sphere_tgt: np.ndarray) -> np.ndarray:
    tree = cKDTree(sphere_unit(sphere_src))
    _, idx = tree.query(sphere_unit(sphere_tgt), k=1)
    return U_src[idx, :]


def overlap_matrix(U_eval: np.ndarray, area_eval: np.ndarray, U_map: np.ndarray) -> np.ndarray:
    return U_eval.T @ (area_eval[:, None] * U_map)


def quantile_binned_probability(x: np.ndarray, y: np.ndarray, nbins: int):
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([]), np.array([]), np.array([]), np.array([])

    centers, probs, ses, counts = [], [], [], []

    for i in range(len(edges) - 1):
        if i < len(edges) - 2:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x <= edges[i + 1])

        if np.any(mask):
            yy = y[mask]
            p = np.mean(yy)
            n = len(yy)
            se = np.sqrt(max(p * (1.0 - p) / max(n, 1), 0.0))

            centers.append(0.5 * (edges[i] + edges[i + 1]))
            probs.append(p)
            ses.append(se)
            counts.append(n)

    return (
        np.asarray(centers),
        np.asarray(probs),
        np.asarray(ses),
        np.asarray(counts),
    )


def save_main_figure(
    heat_avg: np.ndarray,
    ages,
    crowding: np.ndarray,
    exact: np.ndarray,
    out_path: Path,
    nbins: int,
    figscale: float,
    dpi: int,
):
    centers, probs, ses, _ = quantile_binned_probability(crowding, exact, nbins)

    fig, axes = plt.subplots(1, 2, figsize=(12.8 * figscale, 4.9 * figscale))

    ax = axes[0]
    K = heat_avg.shape[1]
    im = ax.imshow(
        heat_avg,
        origin="lower",
        aspect="auto",
        extent=(0.5, K + 0.5, -0.5, len(ages) - 0.5),
    )
    ax.set_yticks(np.arange(len(ages)))
    ax.set_yticklabels(ages)
    xticks = np.arange(1, K + 1) if K <= 10 else np.arange(1, K + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Age")
    ax.set_title("Local crowding relative to neighboring modes")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Local crowding (higher = more crowded)")

    ax = axes[1]
    if len(centers):
        ax.errorbar(centers, probs, yerr=ses, marker="o", linewidth=2, capsize=3)
    ax.set_xlabel("Local crowding score")
    ax.set_ylabel("Exact-match probability")
    ax.set_title("Exact-match probability versus local crowding score")
    ax.set_ylim(0.2, 0.85)

    fig.suptitle("Local mode crowding and cross-age instability", fontsize=17, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_supp_figure(
    heat_lh: np.ndarray,
    heat_rh: np.ndarray,
    ages,
    crowding: np.ndarray,
    exact: np.ndarray,
    near1: np.ndarray,
    near2: np.ndarray,
    out_path: Path,
    nbins: int,
    figscale: float,
    dpi: int,
):
    c_e, p_e, se_e, _ = quantile_binned_probability(crowding, exact, nbins)
    c_1, p_1, se_1, _ = quantile_binned_probability(crowding, near1, nbins)
    c_2, p_2, se_2, _ = quantile_binned_probability(crowding, near2, nbins)

    fig, axes = plt.subplots(1, 3, figsize=(15.0 * figscale, 4.8 * figscale))

    ax = axes[0]
    K = heat_lh.shape[1]
    im1 = ax.imshow(
        heat_lh,
        origin="lower",
        aspect="auto",
        extent=(0.5, K + 0.5, -0.5, len(ages) - 0.5),
    )
    ax.set_yticks(np.arange(len(ages)))
    ax.set_yticklabels(ages)
    xticks = np.arange(1, K + 1) if K <= 10 else np.arange(1, K + 1, 2)
    ax.set_xticks(xticks)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Age")
    ax.set_title("LH local crowding score")
    cbar = plt.colorbar(im1, ax=ax)
    cbar.set_label("Larger = more crowded")

    ax = axes[1]
    im2 = ax.imshow(
        heat_rh,
        origin="lower",
        aspect="auto",
        extent=(0.5, K + 0.5, -0.5, len(ages) - 0.5),
    )
    ax.set_yticks(np.arange(len(ages)))
    ax.set_yticklabels(ages)
    ax.set_xticks(xticks)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Age")
    ax.set_title("RH local crowding score")
    cbar = plt.colorbar(im2, ax=ax)
    cbar.set_label("Larger = more crowded")

    ax = axes[2]
    if len(c_e):
        ax.errorbar(c_e, p_e, yerr=se_e, marker="o", linewidth=2, capsize=3, label="exact")
    if len(c_1):
        ax.errorbar(c_1, p_1, yerr=se_1, marker="s", linewidth=2, capsize=3, label=r"within $\pm 1$")
    if len(c_2):
        ax.errorbar(c_2, p_2, yerr=se_2, marker="^", linewidth=2, capsize=3, label=r"within $\pm 2$")
    ax.set_xlabel("Local crowding score")
    ax.set_ylabel("Match probability")
    ax.set_title("Match probability versus local crowding score")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Local mode crowding and cross-age matching", fontsize=17, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    eigdir = Path(args.eigdir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_eigs(eigdir, args.K)
    ages = [row["age"] for row in rows]

    if args.neighbor_only:
        pairs = [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
    else:
        pairs = list(itertools.combinations(rows, 2))

    heat_lh = []
    heat_rh = []

    crowd_csv = outdir / f"mode_crowding_agewise_K{args.K}.csv"
    with crowd_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["age", "hemi", "mode", "lambda", "spacing_abs", "spacing_rel", "crowding_score"]
        )

        for row in rows:
            for hemi in ["lh", "rh"]:
                evals = row[f"evals_{hemi}"][: args.K]
                spacing_abs, spacing_rel, crowding = local_spacing_measures(evals)

                for k in range(args.K):
                    writer.writerow(
                        [
                            row["age"],
                            hemi,
                            k + 1,
                            evals[k],
                            spacing_abs[k],
                            spacing_rel[k],
                            crowding[k],
                        ]
                    )

                if hemi == "lh":
                    heat_lh.append(crowding)
                else:
                    heat_rh.append(crowding)

    heat_lh = np.vstack(heat_lh)
    heat_rh = np.vstack(heat_rh)
    heat_avg = 0.5 * (heat_lh + heat_rh)

    mode_csv = outdir / f"mode_crowding_mismatch_K{args.K}.csv"

    pooled_crowding = []
    pooled_exact = []
    pooled_near1 = []
    pooled_near2 = []

    with mode_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "age_eval",
                "age_map",
                "age_gap_months",
                "hemi",
                "mode",
                "crowding_score_eval",
                "spacing_rel_eval",
                "best_match",
                "abs_shift",
                "exact_match",
                "near1_match",
                "near2_match",
            ]
        )

        for a, b in pairs:
            gap_months = abs(age_to_months(a["age"]) - age_to_months(b["age"]))

            for hemi in ["lh", "rh"]:
                for eval_row, map_row in [(a, b), (b, a)]:
                    age_eval = eval_row["age"]
                    age_map = map_row["age"]

                    U_eval = eval_row[f"evecs_{hemi}"][:, : args.K]
                    U_map = map_row[f"evecs_{hemi}"][:, : args.K]
                    area_eval = eval_row[f"area_{hemi}"]
                    sphere_eval = eval_row[f"sphere_{hemi}"]
                    sphere_map = map_row[f"sphere_{hemi}"]

                    _, spacing_rel, crowding = local_spacing_measures(
                        eval_row[f"evals_{hemi}"][: args.K]
                    )

                    U_map_on_eval = map_modes_via_sphere_nn(U_map, sphere_map, sphere_eval)
                    U_eval_n = normalize_cols_area(U_eval, area_eval)
                    U_map_on_eval_n = normalize_cols_area(U_map_on_eval, area_eval)

                    S = overlap_matrix(U_eval_n, area_eval, U_map_on_eval_n)
                    absS = np.abs(S)
                    best_match = np.argmax(absS, axis=1) + 1

                    mode_idx = np.arange(1, args.K + 1)
                    abs_shift = np.abs(best_match - mode_idx)
                    exact = (best_match == mode_idx).astype(float)
                    near1 = (np.abs(best_match - mode_idx) <= 1).astype(float)
                    near2 = (np.abs(best_match - mode_idx) <= 2).astype(float)

                    for k in range(args.K):
                        writer.writerow(
                            [
                                age_eval,
                                age_map,
                                gap_months,
                                hemi,
                                k + 1,
                                crowding[k],
                                spacing_rel[k],
                                int(best_match[k]),
                                int(abs_shift[k]),
                                exact[k],
                                near1[k],
                                near2[k],
                            ]
                        )

                        pooled_crowding.append(crowding[k])
                        pooled_exact.append(exact[k])
                        pooled_near1.append(near1[k])
                        pooled_near2.append(near2[k])

    pooled_crowding = np.asarray(pooled_crowding)
    pooled_exact = np.asarray(pooled_exact)
    pooled_near1 = np.asarray(pooled_near1)
    pooled_near2 = np.asarray(pooled_near2)

    main_fig = outdir / f"mode_crowding_main_K{args.K}.png"
    save_main_figure(
        heat_avg=heat_avg,
        ages=ages,
        crowding=pooled_crowding,
        exact=pooled_exact,
        out_path=main_fig,
        nbins=args.nbins,
        figscale=args.figscale,
        dpi=args.dpi,
    )

    supp_fig = outdir / f"mode_crowding_supp_K{args.K}.png"
    save_supp_figure(
        heat_lh=heat_lh,
        heat_rh=heat_rh,
        ages=ages,
        crowding=pooled_crowding,
        exact=pooled_exact,
        near1=pooled_near1,
        near2=pooled_near2,
        out_path=supp_fig,
        nbins=args.nbins,
        figscale=args.figscale,
        dpi=args.dpi,
    )

    print(f"Saved -> {crowd_csv}")
    print(f"Saved -> {mode_csv}")
    print(f"Saved -> {main_fig}")
    print(f"Saved -> {supp_fig}")


if __name__ == "__main__":
    main()