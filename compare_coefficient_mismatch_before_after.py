#!/usr/bin/env python3
# compare_coefficient_mismatch_before_after.py
# Compare coefficient-mismatch summaries before vs after Sequential Procrustes tracking.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare coefficient mismatch before vs after Sequential Procrustes tracking"
    )
    parser.add_argument("--before-csv", required=True, help="Symmetrized avg-hemi CSV for raw basis")
    parser.add_argument("--after-csv", required=True, help="Symmetrized avg-hemi CSV for tracked basis")
    parser.add_argument("--outdir", required=True, help="Output directory")
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


def to_float(x: str) -> float:
    if x is None or x == "":
        return np.nan
    return float(x)


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    return rows


def key_of(row: dict):
    return row["age_1"], row["age_2"], row["family"]


def pair_sort_key(age1: str, age2: str):
    return age_to_months(age1), age_to_months(age2)


def merge_rows(before_rows, after_rows):
    after_map = {key_of(row): row for row in after_rows}

    merged = []
    for row_before in before_rows:
        key = key_of(row_before)
        if key not in after_map:
            continue

        row_after = after_map[key]

        row = {
            "age_1": row_before["age_1"],
            "age_2": row_before["age_2"],
            "age_gap_months": to_float(row_before["age_gap_months"]),
            "family": row_before["family"],

            "exact_retained_before": to_float(row_before["exact_retained_mean"]),
            "exact_retained_after": to_float(row_after["exact_retained_mean"]),

            "near1_retained_before": to_float(row_before.get("near1_retained_mean", "")),
            "near1_retained_after": to_float(row_after.get("near1_retained_mean", "")),

            "near2_retained_before": to_float(row_before.get("near2_retained_mean", "")),
            "near2_retained_after": to_float(row_after.get("near2_retained_mean", "")),

            "abs_peak_shift_before": to_float(row_before.get("abs_peak_shift_mean", "")),
            "abs_peak_shift_after": to_float(row_after.get("abs_peak_shift_mean", "")),

            "abs_centroid_shift_before": to_float(row_before.get("abs_centroid_shift_mean", "")),
            "abs_centroid_shift_after": to_float(row_after.get("abs_centroid_shift_mean", "")),
        }

        row["exact_retained_gain"] = row["exact_retained_after"] - row["exact_retained_before"]
        row["near1_retained_gain"] = row["near1_retained_after"] - row["near1_retained_before"]
        row["near2_retained_gain"] = row["near2_retained_after"] - row["near2_retained_before"]

        merged.append(row)

    merged.sort(key=lambda r: (pair_sort_key(r["age_1"], r["age_2"]), r["family"]))
    return merged


def rows_for_family(rows, family: str):
    out = [row for row in rows if row["family"] == family]
    out.sort(key=lambda row: pair_sort_key(row["age_1"], row["age_2"]))
    return out


def labels_from_rows(rows):
    return [f"{row['age_1']}–{row['age_2']}" for row in rows]


def save_main_figure(merged, outfig: Path, figscale: float, dpi: int):
    rows_onehot = rows_for_family(merged, "onehot")
    rows_packet = rows_for_family(merged, "packet")

    labels = labels_from_rows(rows_onehot)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.8 * figscale, 4.8 * figscale))

    ax = axes[0]
    ax.plot(x, [row["exact_retained_before"] for row in rows_onehot], marker="o", label="before")
    ax.plot(x, [row["exact_retained_after"] for row in rows_onehot], marker="s", label="after")
    ax.set_title("Single-mode test patterns: exact retained energy")
    ax.set_ylabel("Retained energy fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(x, [row["exact_retained_before"] for row in rows_packet], marker="o", label="before")
    ax.plot(x, [row["exact_retained_after"] for row in rows_packet], marker="s", label="after")
    ax.set_title("Neighboring-mode test patterns: exact retained energy")
    ax.set_ylabel("Retained energy fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Sequential Procrustes tracking increases same-index coefficient retention", fontsize=17, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outfig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(merged, outcsv: Path):
    rows_onehot = rows_for_family(merged, "onehot")
    rows_packet = rows_for_family(merged, "packet")

    summary = [
        {
            "family": "single-mode",
            "median_exact_before": np.median([row["exact_retained_before"] for row in rows_onehot]),
            "median_exact_after": np.median([row["exact_retained_after"] for row in rows_onehot]),
            "mean_exact_before": np.mean([row["exact_retained_before"] for row in rows_onehot]),
            "mean_exact_after": np.mean([row["exact_retained_after"] for row in rows_onehot]),
            "improved_exact_in_pairs": int(
                np.sum([row["exact_retained_after"] > row["exact_retained_before"] for row in rows_onehot])
            ),
            "n_pairs": len(rows_onehot),
        },
        {
            "family": "neighboring-mode",
            "median_exact_before": np.median([row["exact_retained_before"] for row in rows_packet]),
            "median_exact_after": np.median([row["exact_retained_after"] for row in rows_packet]),
            "mean_exact_before": np.mean([row["exact_retained_before"] for row in rows_packet]),
            "mean_exact_after": np.mean([row["exact_retained_after"] for row in rows_packet]),
            "improved_exact_in_pairs": int(
                np.sum([row["exact_retained_after"] > row["exact_retained_before"] for row in rows_packet])
            ),
            "n_pairs": len(rows_packet),
        },
    ]

    with outcsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def main():
    args = parse_args()

    before_csv = Path(args.before_csv).expanduser().resolve()
    after_csv = Path(args.after_csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    before_rows = load_csv(before_csv)
    after_rows = load_csv(after_csv)

    merged = merge_rows(before_rows, after_rows)
    if not merged:
        raise RuntimeError("No overlapping rows found between before and after CSVs.")

    merged_csv = outdir / "coefficient_mismatch_before_after_merged.csv"
    with merged_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    summary_csv = outdir / "coefficient_retention_before_after_summary.csv"
    save_summary_csv(merged, summary_csv)

    main_fig = outdir / "coefficient_retention_before_after_main.png"
    save_main_figure(merged, main_fig, figscale=args.figscale, dpi=args.dpi)

    print(f"Saved -> {merged_csv}")
    print(f"Saved -> {summary_csv}")
    print(f"Saved -> {main_fig}")


if __name__ == "__main__":
    main()