#!/usr/bin/env python3
# summarize_transfer_heatmap.py
# Summarize forward-projected scalp gain across age:
#   - normalized paired scalp-gain heatmap
#   - summary CSV with 50/70/90% gain-order cutoffs
#   - long CSV of paired gains by age and harmonic order

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize forward-projected scalp gain across age"
    )
    parser.add_argument("--dictdir", required=True, help="Directory containing dictionary .npz files")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--target-k", type=int, default=50, help="Modes per hemisphere")
    parser.add_argument("--surface", default="white", help="Filter by surface")
    parser.add_argument("--combine", default="block", help="Filter by combine")
    parser.add_argument("--montage-kind", default="template1020", help="Filter by montage kind")

    parser.add_argument(
        "--max-order-plot",
        type=int,
        default=25,
        help="Number of paired harmonic orders to show in the heatmap",
    )
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--figscale", type=float, default=1.0)

    return parser.parse_args()


def infer_scalar(x, key, default=None):
    if key not in x:
        return default

    arr = x[key]
    if getattr(arr, "shape", ()) == ():
        return arr.item()

    if len(arr) == 1:
        value = arr[0]
        return value.item() if hasattr(value, "item") else value

    return arr


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


def load_rows(dictdir: Path, target_k: int, surface: str, combine: str, montage_kind: str):
    rows = []

    for path in sorted(dictdir.glob("*.npz")):
        x = np.load(path, allow_pickle=True)

        K = int(np.array(x["K"]).ravel()[0]) if "K" in x else None
        surf = infer_scalar(x, "surface")
        comb = infer_scalar(x, "combine")
        mont = infer_scalar(x, "montage_kind")

        if K != target_k:
            continue
        if surf is not None and surf != surface:
            continue
        if comb is not None and comb != combine:
            continue
        if mont is not None and mont != montage_kind:
            continue

        D = x["D"].astype(float)
        if D.shape[1] < 2 * target_k:
            raise RuntimeError(
                f"{path.name} has only {D.shape[1]} columns but expected at least {2 * target_k}"
            )

        rows.append(
            {
                "file": path,
                "age": infer_scalar(x, "age", path.stem),
                "subject": infer_scalar(x, "subject", path.stem),
                "K": target_k,
                "D": D[:, : 2 * target_k],
            }
        )

    rows.sort(key=lambda r: age_to_months(r["age"]))

    if not rows:
        raise RuntimeError("No matching dictionary files found.")

    return rows


def paired_gain_from_D(D: np.ndarray, K: int) -> np.ndarray:
    gain = np.linalg.norm(D, axis=0)
    return gain[:K] + gain[K : 2 * K]


def first_reaching(cum: np.ndarray, threshold: float) -> int:
    idx = np.where(cum >= threshold)[0]
    if len(idx) == 0:
        return len(cum)
    return int(idx[0] + 1)


def save_long_csv(rows, outcsv: Path):
    with outcsv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "age",
                "subject",
                "paired_mode_order",
                "paired_gain",
                "paired_gain_norm",
                "paired_gain_cum",
            ]
        )

        for row in rows:
            for k in range(row["K"]):
                writer.writerow(
                    [
                        row["age"],
                        row["subject"],
                        k + 1,
                        row["paired_gain"][k],
                        row["paired_gain_norm"][k],
                        row["paired_gain_cum"][k],
                    ]
                )


def save_summary_csv(rows, outcsv: Path):
    with outcsv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "age",
                "subject",
                "gain_order_50",
                "gain_order_70",
                "gain_order_90",
                "total_paired_gain",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row["age"],
                    row["subject"],
                    row["gain_order_50"],
                    row["gain_order_70"],
                    row["gain_order_90"],
                    row["total_paired_gain"],
                ]
            )


def make_heatmap(rows, outfig: Path, dpi: int, figscale: float, max_order_plot: int):
    ages = [row["age"] for row in rows]
    K = rows[0]["K"]
    max_order_plot = min(max_order_plot, K)

    heat = np.vstack([row["paired_gain_norm"][:max_order_plot] for row in rows])

    fig, ax = plt.subplots(figsize=(8.2 * figscale, 4.8 * figscale))
    im = ax.imshow(heat, aspect="auto", origin="lower")

    ax.set_title("Normalized forward-projected scalp gain across age")
    ax.set_ylabel("Age")
    ax.set_xlabel("Mode order")

    ax.set_yticks(np.arange(len(ages)))
    ax.set_yticklabels(ages)

    step = 2 if max_order_plot <= 25 else 4
    xticks = np.arange(0, max_order_plot, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i + 1) for i in xticks])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized scalp gain")

    plt.tight_layout()
    plt.savefig(outfig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    dictdir = Path(args.dictdir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(
        dictdir=dictdir,
        target_k=args.target_k,
        surface=args.surface,
        combine=args.combine,
        montage_kind=args.montage_kind,
    )

    for row in rows:
        paired_gain = paired_gain_from_D(row["D"], row["K"])
        paired_gain_norm = paired_gain / np.sum(paired_gain)
        paired_gain_cum = np.cumsum(paired_gain_norm)

        row["paired_gain"] = paired_gain
        row["paired_gain_norm"] = paired_gain_norm
        row["paired_gain_cum"] = paired_gain_cum

        row["gain_order_50"] = first_reaching(paired_gain_cum, 0.50)
        row["gain_order_70"] = first_reaching(paired_gain_cum, 0.70)
        row["gain_order_90"] = first_reaching(paired_gain_cum, 0.90)
        row["total_paired_gain"] = float(np.sum(paired_gain))

    long_csv = outdir / "forward_gain_long.csv"
    summary_csv = outdir / "forward_gain_summary.csv"
    heatmap_fig = outdir / f"forward_gain_heatmap_norm_top{min(args.max_order_plot, args.target_k)}.png"

    save_long_csv(rows, long_csv)
    save_summary_csv(rows, summary_csv)
    make_heatmap(rows, heatmap_fig, args.dpi, args.figscale, args.max_order_plot)

    print(f"Saved -> {long_csv}")
    print(f"Saved -> {summary_csv}")
    print(f"Saved -> {heatmap_fig}")


if __name__ == "__main__":
    main()