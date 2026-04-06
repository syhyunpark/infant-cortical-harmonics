#!/usr/bin/env python3
# symmetrize_coefficient_mismatch.py
# Symmetrize coefficient-space mismatch outputs.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Symmetrize coefficient-space mismatch summary"
    )
    parser.add_argument("--csv", required=True, help="Detailed coefficient-mismatch CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--figscale", type=float, default=1.0, help="Overall figure scale")
    parser.add_argument("--dpi", type=int, default=350, help="Output DPI")
    parser.add_argument(
        "--save-hemi-figure",
        action="store_true",
        help="Also save hemisphere-specific summary figure",
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


def ordered_pair(age_a: str, age_b: str):
    if age_to_months(age_a) <= age_to_months(age_b):
        return age_a, age_b
    return age_b, age_a


def to_float(x: str) -> float:
    if x is None or x == "":
        return np.nan
    return float(x)


def get_first_existing(row: dict, keys):
    for key in keys:
        if key in row:
            return to_float(row[key])
    return np.nan


def load_rows(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    return rows


def pair_sort_key(age1: str, age2: str):
    return age_to_months(age1), age_to_months(age2)


def aggregate_by_hemi(rows):
    grouped = {}

    for row in rows:
        age1, age2 = ordered_pair(row["age_src"], row["age_tgt"])
        key = (age1, age2, row["hemi"].lower(), row["family"].lower())
        grouped.setdefault(key, []).append(row)

    out = []
    for (age1, age2, hemi, family), group in sorted(
        grouped.items(),
        key=lambda x: (age_to_months(x[0][0]), age_to_months(x[0][1]), x[0][2], x[0][3]),
    ):
        age_gap = abs(age_to_months(age2) - age_to_months(age1))

        exact_vals = [get_first_existing(g, ["exact_retained", "exact_energy"]) for g in group]
        near1_vals = [get_first_existing(g, ["near1_retained", "near1_energy"]) for g in group]
        near2_vals = [get_first_existing(g, ["near2_retained", "near2_energy"]) for g in group]

        peak_vals = []
        for g in group:
            v = get_first_existing(g, ["abs_peak_shift", "peak_shift"])
            peak_vals.append(abs(v) if np.isfinite(v) else np.nan)

        centroid_vals = []
        for g in group:
            v = get_first_existing(g, ["abs_centroid_shift", "centroid_shift"])
            centroid_vals.append(abs(v) if np.isfinite(v) else np.nan)

        out.append(
            {
                "age_1": age1,
                "age_2": age2,
                "age_gap_months": age_gap,
                "hemi": hemi,
                "family": family,
                "exact_retained_mean": np.nanmean(exact_vals),
                "near1_retained_mean": np.nanmean(near1_vals),
                "near2_retained_mean": np.nanmean(near2_vals),
                "abs_peak_shift_mean": np.nanmean(peak_vals),
                "abs_centroid_shift_mean": np.nanmean(centroid_vals),
                "cosine_mean": np.nanmean([to_float(g["cosine"]) for g in group]) if "cosine" in group[0] else np.nan,
                "pearson_mean": np.nanmean([to_float(g["pearson"]) for g in group]) if "pearson" in group[0] else np.nan,
                "mse_mean": np.nanmean([to_float(g["mse"]) for g in group]) if "mse" in group[0] else np.nan,
                "center_coeff_mean": np.nanmean([to_float(g["center_coeff"]) for g in group]) if "center_coeff" in group[0] else np.nan,
                "n_rows": len(group),
            }
        )

    return out


def aggregate_over_hemi(rows_by_hemi):
    grouped = {}
    for row in rows_by_hemi:
        key = (row["age_1"], row["age_2"], row["family"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (age1, age2, family), group in sorted(
        grouped.items(),
        key=lambda x: (age_to_months(x[0][0]), age_to_months(x[0][1]), x[0][2]),
    ):
        age_gap = abs(age_to_months(age2) - age_to_months(age1))

        out.append(
            {
                "age_1": age1,
                "age_2": age2,
                "age_gap_months": age_gap,
                "family": family,
                "exact_retained_mean": np.nanmean([r["exact_retained_mean"] for r in group]),
                "near1_retained_mean": np.nanmean([r["near1_retained_mean"] for r in group]),
                "near2_retained_mean": np.nanmean([r["near2_retained_mean"] for r in group]),
                "abs_peak_shift_mean": np.nanmean([r["abs_peak_shift_mean"] for r in group]),
                "abs_centroid_shift_mean": np.nanmean([r["abs_centroid_shift_mean"] for r in group]),
                "cosine_mean": np.nanmean([r["cosine_mean"] for r in group]),
                "pearson_mean": np.nanmean([r["pearson_mean"] for r in group]),
                "mse_mean": np.nanmean([r["mse_mean"] for r in group]),
                "center_coeff_mean": np.nanmean([r["center_coeff_mean"] for r in group]),
            }
        )

    return out


def labels_from_rows(rows):
    return [f"{row['age_1']}–{row['age_2']}" for row in rows]


def rows_for_family(rows, family: str):
    out = [row for row in rows if row["family"] == family]
    out.sort(key=lambda row: pair_sort_key(row["age_1"], row["age_2"]))
    return out


def make_main_figure(rows_avg, outfig: Path, figscale: float, dpi: int):
    rows_onehot = rows_for_family(rows_avg, "onehot")
    rows_packet = rows_for_family(rows_avg, "packet")

    labels = labels_from_rows(rows_onehot)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12.8 * figscale, 8.0 * figscale))

    ax = axes[0, 0]
    ax.plot(x, [r["exact_retained_mean"] for r in rows_onehot], marker="o", label="exact mode")
    ax.plot(x, [r["near1_retained_mean"] for r in rows_onehot], marker="s", label=r"within $\pm 1$")
    ax.plot(x, [r["near2_retained_mean"] for r in rows_onehot], marker="^", label=r"within $\pm 2$")
    ax.set_title("Single-mode test patterns: retained coefficient energy")
    ax.set_ylabel("Retained energy fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(x, [r["abs_peak_shift_mean"] for r in rows_onehot], marker="o", label="peak shift")
    ax.plot(x, [r["abs_centroid_shift_mean"] for r in rows_onehot], marker="s", label="centroid shift")
    ax.set_title("Single-mode test patterns: coefficient shifts")
    ax.set_ylabel("Shift in mode index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(x, [r["exact_retained_mean"] for r in rows_packet], marker="o", label="exact center mode")
    ax.plot(x, [r["near1_retained_mean"] for r in rows_packet], marker="s", label=r"within $\pm 1$")
    ax.plot(x, [r["near2_retained_mean"] for r in rows_packet], marker="^", label=r"within $\pm 2$")
    ax.set_title("Neighboring-mode test patterns: retained coefficient energy")
    ax.set_ylabel("Retained energy fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(x, [r["abs_peak_shift_mean"] for r in rows_packet], marker="o", label="peak shift")
    ax.plot(x, [r["abs_centroid_shift_mean"] for r in rows_packet], marker="s", label="centroid shift")
    ax.set_title("Neighboring-mode test patterns: coefficient shifts")
    ax.set_ylabel("Shift in mode index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Coefficient-space mismatch across neighboring ages", fontsize=17, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outfig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_hemi_figure(rows_by_hemi, outfig: Path, figscale: float, dpi: int):
    rows_onehot_lh = sorted(
        [r for r in rows_by_hemi if r["family"] == "onehot" and r["hemi"] == "lh"],
        key=lambda r: pair_sort_key(r["age_1"], r["age_2"]),
    )
    rows_onehot_rh = sorted(
        [r for r in rows_by_hemi if r["family"] == "onehot" and r["hemi"] == "rh"],
        key=lambda r: pair_sort_key(r["age_1"], r["age_2"]),
    )
    rows_packet_lh = sorted(
        [r for r in rows_by_hemi if r["family"] == "packet" and r["hemi"] == "lh"],
        key=lambda r: pair_sort_key(r["age_1"], r["age_2"]),
    )
    rows_packet_rh = sorted(
        [r for r in rows_by_hemi if r["family"] == "packet" and r["hemi"] == "rh"],
        key=lambda r: pair_sort_key(r["age_1"], r["age_2"]),
    )

    labels = labels_from_rows(rows_onehot_lh)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12.8 * figscale, 8.0 * figscale))

    ax = axes[0, 0]
    ax.plot(x, [r["exact_retained_mean"] for r in rows_onehot_lh], marker="o", label="LH")
    ax.plot(x, [r["exact_retained_mean"] for r in rows_onehot_rh], marker="s", label="RH")
    ax.set_title("Single-mode: exact retained energy")
    ax.set_ylabel("Retained energy fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(x, [r["abs_peak_shift_mean"] for r in rows_onehot_lh], marker="o", label="LH peak")
    ax.plot(x, [r["abs_peak_shift_mean"] for r in rows_onehot_rh], marker="s", label="RH peak")
    ax.set_title("Single-mode: mean absolute peak shift")
    ax.set_ylabel("Shift in mode index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(x, [r["exact_retained_mean"] for r in rows_packet_lh], marker="o", label="LH")
    ax.plot(x, [r["exact_retained_mean"] for r in rows_packet_rh], marker="s", label="RH")
    ax.set_title("Neighboring-mode: exact retained energy")
    ax.set_ylabel("Retained energy fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(x, [r["abs_peak_shift_mean"] for r in rows_packet_lh], marker="o", label="LH peak")
    ax.plot(x, [r["abs_peak_shift_mean"] for r in rows_packet_rh], marker="s", label="RH peak")
    ax.set_title("Neighboring-mode: mean absolute peak shift")
    ax.set_ylabel("Shift in mode index")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Coefficient-space mismatch by hemisphere", fontsize=17, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outfig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    rows_by_hemi = aggregate_by_hemi(rows)
    rows_avg = aggregate_over_hemi(rows_by_hemi)

    out_csv_hemi = outdir / f"{csv_path.stem}_symmetrized_by_hemi.csv"
    out_csv_avg = outdir / f"{csv_path.stem}_symmetrized_avghemi.csv"

    with out_csv_hemi.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_by_hemi[0].keys()))
        writer.writeheader()
        writer.writerows(rows_by_hemi)

    with out_csv_avg.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_avg[0].keys()))
        writer.writeheader()
        writer.writerows(rows_avg)

    print(f"Saved -> {out_csv_hemi}")
    print(f"Saved -> {out_csv_avg}")

    outfig = outdir / "coefficient_mismatch_summary_main.png"
    make_main_figure(rows_avg, outfig=outfig, figscale=args.figscale, dpi=args.dpi)
    print(f"Saved -> {outfig}")

    if args.save_hemi_figure:
        hemi_fig = outdir / "coefficient_mismatch_summary_by_hemi.png"
        make_hemi_figure(rows_by_hemi, outfig=hemi_fig, figscale=args.figscale, dpi=args.dpi)
        print(f"Saved -> {hemi_fig}")


if __name__ == "__main__":
    main()