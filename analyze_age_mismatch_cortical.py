#!/usr/bin/env python3
# analyze_age_mismatch_cortical.py
# Cross-age cortical-mode comparability analysis for infant LB eigensystems.

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
        description="Analyze cross-age cortical LB mode mismatch"
    )
    parser.add_argument("--eigdir", required=True, help="Directory with saved full-mesh eigensystems")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--K", type=int, default=30, help="Modes per hemisphere to analyze")

    parser.add_argument("--neighbor-only", dest="neighbor_only", action="store_true", help="Only adjacent ages")
    parser.add_argument("--all-pairs", dest="neighbor_only", action="store_false", help="All age pairs")
    parser.set_defaults(neighbor_only=True)

    parser.add_argument("--plot-top-k", type=int, default=None, help="Plot only first top-k modes")
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


def overlap_matrix(U_eval: np.ndarray, area_eval: np.ndarray, U_map_on_eval: np.ndarray) -> np.ndarray:
    return U_eval.T @ (area_eval[:, None] * U_map_on_eval)


def count_inversions(seq: np.ndarray) -> int:
    n = len(seq)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] > seq[j]:
                inv += 1
    return inv


def summarize_overlap(absS: np.ndarray):
    K = absS.shape[0]

    diag = np.diag(absS)
    total = np.sum(absS)
    offdiag = total - np.sum(diag)

    best_match = np.argmax(absS, axis=1) + 1
    k_idx = np.arange(1, K + 1)

    exact_match = np.mean(best_match == k_idx)
    near1_match = np.mean(np.abs(best_match - k_idx) <= 1)
    near2_match = np.mean(np.abs(best_match - k_idx) <= 2)
    mean_abs_shift = float(np.mean(np.abs(best_match - k_idx)))

    inv = count_inversions(best_match)
    inv_rate = inv / max(K * (K - 1) / 2, 1)

    adjacent_reorders = int(np.sum(np.diff(best_match) < 0))

    return {
        "diag_mean": float(np.mean(diag)),
        "diag_median": float(np.median(diag)),
        "offdiag_share": float(offdiag / max(total, EPS)),
        "exact_match_rate": float(exact_match),
        "near1_match_rate": float(near1_match),
        "near2_match_rate": float(near2_match),
        "mean_abs_shift": mean_abs_shift,
        "adjacent_reorders": adjacent_reorders,
        "inversion_rate": float(inv_rate),
        "best_match": best_match,
    }


def symmetrize_scalar_metrics(m1: dict, m2: dict):
    keys = [
        "diag_mean",
        "diag_median",
        "offdiag_share",
        "exact_match_rate",
        "near1_match_rate",
        "near2_match_rate",
        "mean_abs_shift",
        "adjacent_reorders",
        "inversion_rate",
    ]

    out = {}
    for key in keys:
        out[f"{key}_sym"] = float(0.5 * (m1[key] + m2[key]))

    return out


def _overlay_best_match(ax, best_match: np.ndarray, plot_top_k: int, color: str = "cyan"):
    rows = np.arange(1, plot_top_k + 1)
    bm = best_match[:plot_top_k]

    inside = bm <= plot_top_k
    off = ~inside

    if np.any(inside):
        ax.plot(
            bm[inside],
            rows[inside],
            color=color,
            linewidth=1.3,
            alpha=0.9,
            label="best match",
        )

    if np.any(off):
        ax.scatter(
            np.full(np.sum(off), plot_top_k + 0.20),
            rows[off],
            marker=">",
            s=24,
            color=color,
            alpha=0.9,
            label="best match off-window",
            clip_on=False,
        )


def plot_pair_heatmaps_2x2(
    absS_lh_b_to_a: np.ndarray,
    absS_rh_b_to_a: np.ndarray,
    best_lh_b_to_a: np.ndarray,
    best_rh_b_to_a: np.ndarray,
    absS_lh_a_to_b: np.ndarray,
    absS_rh_a_to_b: np.ndarray,
    best_lh_a_to_b: np.ndarray,
    best_rh_a_to_b: np.ndarray,
    age_a: str,
    age_b: str,
    out_path: Path,
    plot_top_k: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    panel_specs = [
        (axes[0, 0], absS_lh_b_to_a, best_lh_b_to_a, f"LH (evaluate on {age_a})"),
        (axes[0, 1], absS_rh_b_to_a, best_rh_b_to_a, f"RH (evaluate on {age_a})"),
        (axes[1, 0], absS_lh_a_to_b, best_lh_a_to_b, f"LH (evaluate on {age_b})"),
        (axes[1, 1], absS_rh_a_to_b, best_rh_a_to_b, f"RH (evaluate on {age_b})"),
    ]

    im = None
    tick_vals = np.arange(1, plot_top_k + 1)

    for ax, absS, best, title in panel_specs:
        block = absS[:plot_top_k, :plot_top_k]

        im = ax.imshow(
            block,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            extent=(0.5, plot_top_k + 0.5, 0.5, plot_top_k + 0.5),
        )

        ax.plot(
            tick_vals,
            tick_vals,
            linestyle="--",
            linewidth=1.0,
            color="white",
            alpha=0.8,
            label="diagonal",
        )
        _overlay_best_match(ax, best, plot_top_k=plot_top_k, color="cyan")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mapped-mode index")
        ax.set_ylabel("Evaluation-mode index")

        show_ticks = tick_vals if plot_top_k <= 10 else np.arange(1, plot_top_k + 1, 2)
        ax.set_xticks(show_ticks)
        ax.set_yticks(show_ticks)
        ax.set_xlim(0.5, plot_top_k + 0.5)
        ax.set_ylim(0.5, plot_top_k + 0.5)

        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7)

    plt.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.08, wspace=0.22, hspace=0.28)
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.58])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$|S|$")

    fig.suptitle(
        f"Cross-age cortical mode overlap: {age_a} and {age_b}\n"
        f"Top row: map {age_b} onto {age_a}; bottom row: map {age_a} onto {age_b}; plotted block = first {plot_top_k} modes",
        fontsize=12,
        y=0.96,
    )

    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    eigdir = Path(args.eigdir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_eigs(eigdir, K=args.K)
    print(f"Loaded {len(rows)} age-specific eigensystems with K={args.K}.")

    if args.neighbor_only:
        pairs = [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
    else:
        pairs = list(itertools.combinations(rows, 2))

    plot_top_k = args.K if args.plot_top_k is None else min(args.plot_top_k, args.K)

    directional_csv = outdir / f"cortical_age_mismatch_directional_K{args.K}.csv"
    sym_csv = outdir / f"cortical_age_mismatch_symmetrized_K{args.K}.csv"
    npz_dir = outdir / f"overlap_matrices_K{args.K}"
    fig_dir = outdir / f"overlap_figures_K{args.K}"

    npz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with directional_csv.open("w", newline="") as fdir, sym_csv.open("w", newline="") as fsym:
        writer_dir = csv.writer(fdir)
        writer_sym = csv.writer(fsym)

        writer_dir.writerow(
            [
                "age_eval",
                "age_map",
                "hemi",
                "diag_mean",
                "diag_median",
                "offdiag_share",
                "exact_match_rate",
                "near1_match_rate",
                "near2_match_rate",
                "mean_abs_shift",
                "adjacent_reorders",
                "inversion_rate",
            ]
        )

        writer_sym.writerow(
            [
                "age_1",
                "age_2",
                "hemi",
                "diag_mean_sym",
                "diag_median_sym",
                "offdiag_share_sym",
                "exact_match_rate_sym",
                "near1_match_rate_sym",
                "near2_match_rate_sym",
                "mean_abs_shift_sym",
                "adjacent_reorders_sym",
                "inversion_rate_sym",
            ]
        )

        for a, b in pairs:
            age_a = a["age"]
            age_b = b["age"]

            out_npz = npz_dir / f"overlap_pair_{age_a}_and_{age_b}_K{args.K}.npz"

            save_pair = {
                "age_a": np.array([age_a], dtype=object),
                "age_b": np.array([age_b], dtype=object),
                "subject_a": np.array([a["subject"]], dtype=object),
                "subject_b": np.array([b["subject"]], dtype=object),
            }

            plot_inputs = {}

            for hemi in ["lh", "rh"]:
                U_a = a[f"evecs_{hemi}"][:, : args.K]
                U_b = b[f"evecs_{hemi}"][:, : args.K]

                area_a = a[f"area_{hemi}"]
                area_b = b[f"area_{hemi}"]

                sphere_a = a[f"sphere_{hemi}"]
                sphere_b = b[f"sphere_{hemi}"]

                U_b_on_a = map_modes_via_sphere_nn(U_b, sphere_b, sphere_a)
                U_a_norm = normalize_cols_area(U_a, area_a)
                U_b_on_a_norm = normalize_cols_area(U_b_on_a, area_a)

                S_b_to_a = overlap_matrix(U_a_norm, area_a, U_b_on_a_norm)
                absS_b_to_a = np.abs(S_b_to_a)
                summ_b_to_a = summarize_overlap(absS_b_to_a)

                writer_dir.writerow(
                    [
                        age_a,
                        age_b,
                        hemi,
                        summ_b_to_a["diag_mean"],
                        summ_b_to_a["diag_median"],
                        summ_b_to_a["offdiag_share"],
                        summ_b_to_a["exact_match_rate"],
                        summ_b_to_a["near1_match_rate"],
                        summ_b_to_a["near2_match_rate"],
                        summ_b_to_a["mean_abs_shift"],
                        summ_b_to_a["adjacent_reorders"],
                        summ_b_to_a["inversion_rate"],
                    ]
                )

                print(
                    f"eval {age_a:>5} | map {age_b:<5} | {hemi.upper()} | "
                    f"diag={summ_b_to_a['diag_mean']:.3f} | "
                    f"offdiag={summ_b_to_a['offdiag_share']:.3f} | "
                    f"shift={summ_b_to_a['mean_abs_shift']:.2f} | "
                    f"inv={summ_b_to_a['inversion_rate']:.3f}"
                )

                U_a_on_b = map_modes_via_sphere_nn(U_a, sphere_a, sphere_b)
                U_b_norm = normalize_cols_area(U_b, area_b)
                U_a_on_b_norm = normalize_cols_area(U_a_on_b, area_b)

                S_a_to_b = overlap_matrix(U_b_norm, area_b, U_a_on_b_norm)
                absS_a_to_b = np.abs(S_a_to_b)
                summ_a_to_b = summarize_overlap(absS_a_to_b)

                writer_dir.writerow(
                    [
                        age_b,
                        age_a,
                        hemi,
                        summ_a_to_b["diag_mean"],
                        summ_a_to_b["diag_median"],
                        summ_a_to_b["offdiag_share"],
                        summ_a_to_b["exact_match_rate"],
                        summ_a_to_b["near1_match_rate"],
                        summ_a_to_b["near2_match_rate"],
                        summ_a_to_b["mean_abs_shift"],
                        summ_a_to_b["adjacent_reorders"],
                        summ_a_to_b["inversion_rate"],
                    ]
                )

                print(
                    f"eval {age_b:>5} | map {age_a:<5} | {hemi.upper()} | "
                    f"diag={summ_a_to_b['diag_mean']:.3f} | "
                    f"offdiag={summ_a_to_b['offdiag_share']:.3f} | "
                    f"shift={summ_a_to_b['mean_abs_shift']:.2f} | "
                    f"inv={summ_a_to_b['inversion_rate']:.3f}"
                )

                summ_sym = symmetrize_scalar_metrics(summ_b_to_a, summ_a_to_b)
                writer_sym.writerow(
                    [
                        age_a,
                        age_b,
                        hemi,
                        summ_sym["diag_mean_sym"],
                        summ_sym["diag_median_sym"],
                        summ_sym["offdiag_share_sym"],
                        summ_sym["exact_match_rate_sym"],
                        summ_sym["near1_match_rate_sym"],
                        summ_sym["near2_match_rate_sym"],
                        summ_sym["mean_abs_shift_sym"],
                        summ_sym["adjacent_reorders_sym"],
                        summ_sym["inversion_rate_sym"],
                    ]
                )

                save_pair[f"S_{hemi}_eval_{age_a}_map_{age_b}"] = S_b_to_a
                save_pair[f"absS_{hemi}_eval_{age_a}_map_{age_b}"] = absS_b_to_a
                save_pair[f"best_match_{hemi}_eval_{age_a}_map_{age_b}"] = summ_b_to_a["best_match"]

                save_pair[f"S_{hemi}_eval_{age_b}_map_{age_a}"] = S_a_to_b
                save_pair[f"absS_{hemi}_eval_{age_b}_map_{age_a}"] = absS_a_to_b
                save_pair[f"best_match_{hemi}_eval_{age_b}_map_{age_a}"] = summ_a_to_b["best_match"]

                plot_inputs[f"absS_{hemi}_b_to_a"] = absS_b_to_a
                plot_inputs[f"best_{hemi}_b_to_a"] = summ_b_to_a["best_match"]

                plot_inputs[f"absS_{hemi}_a_to_b"] = absS_a_to_b
                plot_inputs[f"best_{hemi}_a_to_b"] = summ_a_to_b["best_match"]

            np.savez_compressed(out_npz, **save_pair)

            out_fig = fig_dir / f"overlap_pair_{age_a}_and_{age_b}_top{plot_top_k}_K{args.K}.png"
            plot_pair_heatmaps_2x2(
                absS_lh_b_to_a=plot_inputs["absS_lh_b_to_a"],
                absS_rh_b_to_a=plot_inputs["absS_rh_b_to_a"],
                best_lh_b_to_a=plot_inputs["best_lh_b_to_a"],
                best_rh_b_to_a=plot_inputs["best_rh_b_to_a"],
                absS_lh_a_to_b=plot_inputs["absS_lh_a_to_b"],
                absS_rh_a_to_b=plot_inputs["absS_rh_a_to_b"],
                best_lh_a_to_b=plot_inputs["best_lh_a_to_b"],
                best_rh_a_to_b=plot_inputs["best_rh_a_to_b"],
                age_a=age_a,
                age_b=age_b,
                out_path=out_fig,
                plot_top_k=plot_top_k,
            )

    print(f"\nSaved directional CSV -> {directional_csv}")
    print(f"Saved symmetrized CSV -> {sym_csv}")
    print(f"Saved overlap matrices -> {npz_dir}")
    print(f"Saved figures -> {fig_dir}")


if __name__ == "__main__":
    main()