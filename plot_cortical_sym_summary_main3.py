#!/usr/bin/env python3
# plot_cortical_sym_summary_main3.py
# Create a 3-panel main-manuscript summary figure from the symmetrized cortical CSV.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 3-panel symmetrized cortical mismatch summary"
    )
    parser.add_argument("--csv", required=True, help="Symmetrized cortical CSV")
    parser.add_argument("--outfig", required=True, help="Output figure path")
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
    return float(x)


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    return rows


def sort_key(row: dict):
    return age_to_months(row["age_1"]), age_to_months(row["age_2"])


def aggregate_over_hemis(rows):
    grouped = {}
    for row in rows:
        key = (row["age_1"], row["age_2"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (age_1, age_2), group in grouped.items():
        out.append(
            {
                "age_1": age_1,
                "age_2": age_2,
                "exact_match_rate_sym": np.mean([to_float(g["exact_match_rate_sym"]) for g in group]),
                "near1_match_rate_sym": np.mean([to_float(g["near1_match_rate_sym"]) for g in group]),
                "near2_match_rate_sym": np.mean([to_float(g["near2_match_rate_sym"]) for g in group]),
                "mean_abs_shift_sym": np.mean([to_float(g["mean_abs_shift_sym"]) for g in group]),
                "adjacent_reorders_sym": np.mean([to_float(g["adjacent_reorders_sym"]) for g in group]),
            }
        )

    out.sort(key=sort_key)
    return out


def pair_labels(rows):
    return [f"{row['age_1']}–{row['age_2']}" for row in rows]


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outfig = Path(args.outfig).expanduser().resolve()
    outfig.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    rows.sort(key=sort_key)

    rows_avg = aggregate_over_hemis(rows)

    labels = pair_labels(rows_avg)
    x = np.arange(len(labels))

    exact = np.array([row["exact_match_rate_sym"] for row in rows_avg], dtype=float)
    near1 = np.array([row["near1_match_rate_sym"] for row in rows_avg], dtype=float)
    near2 = np.array([row["near2_match_rate_sym"] for row in rows_avg], dtype=float)
    shift = np.array([row["mean_abs_shift_sym"] for row in rows_avg], dtype=float)
    reorders = np.array([row["adjacent_reorders_sym"] for row in rows_avg], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.8 * args.figscale, 4.3 * args.figscale))

    ax = axes[0]
    ax.plot(x, exact, marker="o", label="exact")
    ax.plot(x, near1, marker="s", label=r"within $\pm 1$")
    ax.plot(x, near2, marker="^", label=r"within $\pm 2$")
    ax.set_ylabel("Match rate")
    ax.set_title("Best-match agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(x, shift, marker="o")
    ax.set_ylabel("Mode-index shift")
    ax.set_title("Mean absolute best-match shift")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax = axes[2]
    ax.plot(x, reorders, marker="o")
    ax.set_ylabel("Count")
    ax.set_title("Adjacent reorderings")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    fig.suptitle("Symmetrized neighboring-age cortical mismatch summary", fontsize=17, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(outfig, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {outfig}")


if __name__ == "__main__":
    main()