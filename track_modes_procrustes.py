#!/usr/bin/env python3
# track_modes_procrustes.py
# Simple orthogonal subspace-alignment baseline for developmental cortical harmonics.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree


EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential blockwise Procrustes tracking baseline"
    )
    parser.add_argument("--eigdir", required=True, help="Directory with full-mesh eigensystems")
    parser.add_argument("--tracked-outdir", required=True, help="Output directory for tracked eigensystems")
    parser.add_argument("--summary-outdir", required=True, help="Output directory for summary CSVs and figures")
    parser.add_argument("--K", type=int, default=30, help="Modes per hemisphere to track")

    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Contiguous block size for blockwise alignment",
    )
    parser.add_argument(
        "--blocks",
        default=None,
        help='Optional explicit blocks, e.g. "1-3,4-6,7-9,10-12"',
    )

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


def sphere_unit(rr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rr, axis=1, keepdims=True)
    return rr / np.maximum(norm, EPS)


def normalize_cols_area(U: np.ndarray, area: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.sum((U ** 2) * area[:, None], axis=0))
    return U / np.maximum(norms, EPS)[None, :]


def parse_blocks(K: int, block_size: int, blocks_str: str | None):
    if blocks_str is not None:
        blocks = []
        for chunk in blocks_str.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue

            m = re.fullmatch(r"(\d+)-(\d+)", chunk)
            if m is None:
                raise ValueError(f"Could not parse block specifier: {chunk}")

            a = int(m.group(1))
            b = int(m.group(2))
            if a < 1 or b < a or b > K:
                raise ValueError(f"Invalid block range: {chunk}")

            blocks.append(list(range(a - 1, b)))

        return blocks

    blocks = []
    start = 0
    while start < K:
        stop = min(start + block_size, K)
        blocks.append(list(range(start, stop)))
        start = stop

    return blocks


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
                "surface_basis": scalar_from_npz(x, "surface"),
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


def map_modes_via_sphere_nn(U_src: np.ndarray, sphere_src: np.ndarray, sphere_tgt: np.ndarray) -> np.ndarray:
    tree = cKDTree(sphere_unit(sphere_src))
    _, idx = tree.query(sphere_unit(sphere_tgt), k=1)
    return U_src[idx, :]


def overlap_matrix(U_eval: np.ndarray, area_eval: np.ndarray, U_map: np.ndarray) -> np.ndarray:
    return U_eval.T @ (area_eval[:, None] * U_map)


def weighted_block_procrustes(
    U_prev: np.ndarray,
    U_curr_mapped: np.ndarray,
    area_prev: np.ndarray,
) -> np.ndarray:
    w = np.sqrt(np.maximum(area_prev, EPS))[:, None]
    A = w * U_curr_mapped
    B = w * U_prev
    R, _ = orthogonal_procrustes(A, B)
    return R


def track_one_hemi_sequential(rows, hemi: str, blocks, K: int):
    key_evecs = f"evecs_{hemi}"
    key_area = f"area_{hemi}"
    key_sphere = f"sphere_{hemi}"

    tracked = {}

    ref = rows[0]
    tracked[ref["age"]] = ref[key_evecs][:, :K].copy()

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]

        U_prev_tr = tracked[prev["age"]]
        U_curr_native = curr[key_evecs][:, :K].copy()

        area_prev = prev[key_area]
        sphere_prev = prev[key_sphere]
        sphere_curr = curr[key_sphere]

        U_curr_tr = U_curr_native.copy()

        for block in blocks:
            U_prev_block = U_prev_tr[:, block]

            U_curr_block_mapped = map_modes_via_sphere_nn(
                U_curr_native[:, block],
                sphere_curr,
                sphere_prev,
            )

            U_prev_block_n = normalize_cols_area(U_prev_block, area_prev)
            U_curr_block_mapped_n = normalize_cols_area(U_curr_block_mapped, area_prev)

            R = weighted_block_procrustes(
                U_prev=U_prev_block_n,
                U_curr_mapped=U_curr_block_mapped_n,
                area_prev=area_prev,
            )

            U_curr_tr[:, block] = U_curr_native[:, block] @ R

        U_curr_tr_mapped = map_modes_via_sphere_nn(U_curr_tr, sphere_curr, sphere_prev)
        U_prev_n = normalize_cols_area(U_prev_tr, area_prev)
        U_curr_tr_mapped_n = normalize_cols_area(U_curr_tr_mapped, area_prev)

        for k in range(K):
            ov = np.sum(area_prev * U_prev_n[:, k] * U_curr_tr_mapped_n[:, k])
            if ov < 0:
                U_curr_tr[:, k] *= -1.0

        tracked[curr["age"]] = U_curr_tr

    return tracked


def directional_metrics(eval_row, map_row, hemi: str, K: int):
    key_evecs = f"evecs_{hemi}"
    key_area = f"area_{hemi}"
    key_sphere = f"sphere_{hemi}"

    U_eval = eval_row[key_evecs][:, :K]
    U_map = map_row[key_evecs][:, :K]

    area_eval = eval_row[key_area]
    sphere_eval = eval_row[key_sphere]
    sphere_map = map_row[key_sphere]

    U_eval_n = normalize_cols_area(U_eval, area_eval)
    U_map_on_eval = map_modes_via_sphere_nn(U_map, sphere_map, sphere_eval)
    U_map_on_eval_n = normalize_cols_area(U_map_on_eval, area_eval)

    S = overlap_matrix(U_eval_n, area_eval, U_map_on_eval_n)
    absS = np.abs(S)

    mode_idx = np.arange(1, K + 1)
    best_match = np.argmax(absS, axis=1) + 1
    abs_shift = np.abs(best_match - mode_idx)

    exact = (best_match == mode_idx).astype(float)
    near1 = (np.abs(best_match - mode_idx) <= 1).astype(float)
    near2 = (np.abs(best_match - mode_idx) <= 2).astype(float)

    return {
        "diag_mean": float(np.mean(np.diag(absS))),
        "exact_rate": float(np.mean(exact)),
        "near1_rate": float(np.mean(near1)),
        "near2_rate": float(np.mean(near2)),
        "mean_abs_shift": float(np.mean(abs_shift)),
    }


def save_tracked_eigensystems(rows, tracked_lh, tracked_rh, tracked_outdir: Path, blocks):
    tracked_outdir.mkdir(parents=True, exist_ok=True)

    block_str = ",".join(f"{block[0] + 1}-{block[-1] + 1}" for block in blocks)
    ref_age = rows[0]["age"]

    for row in rows:
        out_path = tracked_outdir / row["file"].name

        np.savez_compressed(
            out_path,
            age=np.array([row["age"]], dtype=object),
            subject=np.array([row["subject"]], dtype=object),
            surface=np.array([row["surface_basis"]], dtype=object),
            sphere_surf=np.array([row["sphere_surf"]], dtype=object),
            K=np.array([row["K"]], dtype=int),
            evals_lh=row["evals_lh"],
            evals_rh=row["evals_rh"],
            evecs_lh=tracked_lh[row["age"]],
            evecs_rh=tracked_rh[row["age"]],
            area_lh=row["area_lh"],
            area_rh=row["area_rh"],
            sphere_lh=row["sphere_lh"],
            sphere_rh=row["sphere_rh"],
            tracking_method=np.array(["blockwise_weighted_orthogonal_procrustes"], dtype=object),
            tracking_reference_age=np.array([ref_age], dtype=object),
            tracking_blocks=np.array([block_str], dtype=object),
        )


def save_summary_figure(rows_avg, outfig: Path, figscale: float, dpi: int):
    labels = [f"{row['age_1']}–{row['age_2']}" for row in rows_avg]
    x = np.arange(len(labels))

    exact_before = [row["exact_rate_before"] for row in rows_avg]
    exact_after = [row["exact_rate_after"] for row in rows_avg]

    near1_before = [row["near1_rate_before"] for row in rows_avg]
    near1_after = [row["near1_rate_after"] for row in rows_avg]

    shift_before = [row["mean_abs_shift_before"] for row in rows_avg]
    shift_after = [row["mean_abs_shift_after"] for row in rows_avg]

    fig, axes = plt.subplots(1, 3, figsize=(14.0 * figscale, 4.5 * figscale))

    ax = axes[0]
    ax.plot(x, exact_before, marker="o", label="before")
    ax.plot(x, exact_after, marker="s", label="after")
    ax.set_title("Exact-match rate")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(x, near1_before, marker="o", label="before")
    ax.plot(x, near1_after, marker="s", label="after")
    ax.set_title(r"Near-$\pm 1$ match rate")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.plot(x, shift_before, marker="o", label="before")
    ax.plot(x, shift_after, marker="s", label="after")
    ax.set_title("Mean absolute mode-index shift")
    ax.set_ylabel("Shift")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Sequential Procrustes tracking improves neighboring-age consistency", fontsize=17, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outfig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    eigdir = Path(args.eigdir).expanduser().resolve()
    tracked_outdir = Path(args.tracked_outdir).expanduser().resolve()
    summary_outdir = Path(args.summary_outdir).expanduser().resolve()
    summary_outdir.mkdir(parents=True, exist_ok=True)

    rows = load_eigs(eigdir, args.K)

    blocks = parse_blocks(args.K, args.block_size, args.blocks)
    print("Tracking blocks:", ", ".join(f"{b[0] + 1}-{b[-1] + 1}" for b in blocks))

    tracked_lh = track_one_hemi_sequential(rows, hemi="lh", blocks=blocks, K=args.K)
    tracked_rh = track_one_hemi_sequential(rows, hemi="rh", blocks=blocks, K=args.K)

    save_tracked_eigensystems(rows, tracked_lh, tracked_rh, tracked_outdir, blocks)
    print(f"Saved tracked eigensystems -> {tracked_outdir}")

    tracked_rows = []
    for row in rows:
        tracked_row = dict(row)
        tracked_row["evecs_lh"] = tracked_lh[row["age"]]
        tracked_row["evecs_rh"] = tracked_rh[row["age"]]
        tracked_rows.append(tracked_row)

    byhemi_rows = []
    avghemi_rows = []

    for i in range(len(rows) - 1):
        raw_a = rows[i]
        raw_b = rows[i + 1]
        tr_a = tracked_rows[i]
        tr_b = tracked_rows[i + 1]

        age1, age2 = raw_a["age"], raw_b["age"]
        gap = abs(age_to_months(age2) - age_to_months(age1))

        hemi_rows = []

        for hemi in ["lh", "rh"]:
            raw_ab = directional_metrics(raw_a, raw_b, hemi=hemi, K=args.K)
            raw_ba = directional_metrics(raw_b, raw_a, hemi=hemi, K=args.K)

            tr_ab = directional_metrics(tr_a, tr_b, hemi=hemi, K=args.K)
            tr_ba = directional_metrics(tr_b, tr_a, hemi=hemi, K=args.K)

            row = {
                "age_1": age1,
                "age_2": age2,
                "age_gap_months": gap,
                "hemi": hemi,
                "diag_mean_before": 0.5 * (raw_ab["diag_mean"] + raw_ba["diag_mean"]),
                "diag_mean_after": 0.5 * (tr_ab["diag_mean"] + tr_ba["diag_mean"]),
                "exact_rate_before": 0.5 * (raw_ab["exact_rate"] + raw_ba["exact_rate"]),
                "exact_rate_after": 0.5 * (tr_ab["exact_rate"] + tr_ba["exact_rate"]),
                "near1_rate_before": 0.5 * (raw_ab["near1_rate"] + raw_ba["near1_rate"]),
                "near1_rate_after": 0.5 * (tr_ab["near1_rate"] + tr_ba["near1_rate"]),
                "near2_rate_before": 0.5 * (raw_ab["near2_rate"] + raw_ba["near2_rate"]),
                "near2_rate_after": 0.5 * (tr_ab["near2_rate"] + tr_ba["near2_rate"]),
                "mean_abs_shift_before": 0.5 * (raw_ab["mean_abs_shift"] + raw_ba["mean_abs_shift"]),
                "mean_abs_shift_after": 0.5 * (tr_ab["mean_abs_shift"] + tr_ba["mean_abs_shift"]),
            }

            byhemi_rows.append(row)
            hemi_rows.append(row)

        avghemi_rows.append(
            {
                "age_1": age1,
                "age_2": age2,
                "age_gap_months": gap,
                "diag_mean_before": np.mean([r["diag_mean_before"] for r in hemi_rows]),
                "diag_mean_after": np.mean([r["diag_mean_after"] for r in hemi_rows]),
                "exact_rate_before": np.mean([r["exact_rate_before"] for r in hemi_rows]),
                "exact_rate_after": np.mean([r["exact_rate_after"] for r in hemi_rows]),
                "near1_rate_before": np.mean([r["near1_rate_before"] for r in hemi_rows]),
                "near1_rate_after": np.mean([r["near1_rate_after"] for r in hemi_rows]),
                "near2_rate_before": np.mean([r["near2_rate_before"] for r in hemi_rows]),
                "near2_rate_after": np.mean([r["near2_rate_after"] for r in hemi_rows]),
                "mean_abs_shift_before": np.mean([r["mean_abs_shift_before"] for r in hemi_rows]),
                "mean_abs_shift_after": np.mean([r["mean_abs_shift_after"] for r in hemi_rows]),
            }
        )

    byhemi_csv = summary_outdir / f"tracking_neighbor_metrics_byhemi_K{args.K}.csv"
    avghemi_csv = summary_outdir / f"tracking_neighbor_metrics_avghemi_K{args.K}.csv"

    with byhemi_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(byhemi_rows[0].keys()))
        writer.writeheader()
        writer.writerows(byhemi_rows)

    with avghemi_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(avghemi_rows[0].keys()))
        writer.writeheader()
        writer.writerows(avghemi_rows)

    print(f"Saved -> {byhemi_csv}")
    print(f"Saved -> {avghemi_csv}")

    outfig = summary_outdir / f"tracking_summary_main_K{args.K}.png"
    save_summary_figure(avghemi_rows, outfig=outfig, figscale=args.figscale, dpi=args.dpi)
    print(f"Saved -> {outfig}")


if __name__ == "__main__":
    main()