#!/usr/bin/env python3
# symmetrize_sensor_mismatch.py
# Read directional sensor-age-mismatch CSV and create a symmetrized summary.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Symmetrize directional sensor mismatch summary"
    )
    parser.add_argument("--csv", required=True, help="Input directional CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
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


def to_float(x: str):
    if x is None or x == "":
        return np.nan
    return float(x)


def ordered_pair(age_a: str, age_b: str):
    if age_to_months(age_a) <= age_to_months(age_b):
        return age_a, age_b
    return age_b, age_a


def save_line(outdir: Path, labels, vals, ylabel: str, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))

    ax.plot(x, vals, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    for row in rows:
        key = ordered_pair(row["age_a"], row["age_b"])
        grouped.setdefault(key, []).append(row)

    out_rows = []

    for (age1, age2), group in sorted(
        grouped.items(),
        key=lambda x: (age_to_months(x[0][0]), age_to_months(x[0][1])),
    ):
        mean_angle = np.nanmean([to_float(g["mean_angle_deg"]) for g in group])
        max_angle = np.nanmean([to_float(g["max_angle_deg"]) for g in group])
        projdist = np.nanmean([to_float(g["projector_distance"]) for g in group])

        within_mean = np.nanmean([to_float(g["within_r2_mean"]) for g in group])
        cross_mean = np.nanmean([to_float(g["cross_r2_mean"]) for g in group])
        delta_mean = np.nanmean([to_float(g["delta_r2_mean"]) for g in group])

        within_med = np.nanmean([to_float(g["within_r2_median"]) for g in group])
        cross_med = np.nanmean([to_float(g["cross_r2_median"]) for g in group])
        delta_med = np.nanmean([to_float(g["delta_r2_median"]) for g in group])

        age_gap_months = abs(age_to_months(age2) - age_to_months(age1))

        surface_vals = sorted({g["surface"] for g in group})
        combine_vals = sorted({g["combine"] for g in group})
        montage_vals = sorted({g["montage_kind"] for g in group})

        noise_vals = [g["noise_snr_db"] for g in group if g["noise_snr_db"] != ""]
        noise_snr_db = noise_vals[0] if noise_vals and len(set(noise_vals)) == 1 else ""

        out_rows.append(
            {
                "age_1": age1,
                "age_2": age2,
                "age_gap_months": age_gap_months,
                "mean_angle_deg_sym": mean_angle,
                "max_angle_deg_sym": max_angle,
                "projector_distance_sym": projdist,
                "within_r2_mean_sym": within_mean,
                "cross_r2_mean_sym": cross_mean,
                "delta_r2_mean_sym": delta_mean,
                "within_r2_median_sym": within_med,
                "cross_r2_median_sym": cross_med,
                "delta_r2_median_sym": delta_med,
                "noise_snr_db": noise_snr_db,
                "surface": ",".join(surface_vals),
                "combine": ",".join(combine_vals),
                "montage_kind": ",".join(montage_vals),
            }
        )

    out_csv = outdir / f"{csv_path.stem}_symmetrized.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved -> {out_csv}")

    labels = [f"{row['age_1']}→{row['age_2']}" for row in out_rows]

    save_line(
        outdir,
        labels,
        [row["delta_r2_mean_sym"] for row in out_rows],
        r"Symmetrized $\Delta R^2$",
        "Symmetrized neighbor mismatch loss",
        "neighbor_deltaR2_sym.png",
    )

    save_line(
        outdir,
        labels,
        [row["cross_r2_mean_sym"] for row in out_rows],
        "Symmetrized cross-age mean $R^2$",
        "Symmetrized neighbor cross-age $R^2$",
        "neighbor_crossR2_sym.png",
    )

    save_line(
        outdir,
        labels,
        [row["projector_distance_sym"] for row in out_rows],
        "Projector distance",
        "Symmetrized neighbor projector distance",
        "neighbor_projectordist_sym.png",
    )

    save_line(
        outdir,
        labels,
        [row["mean_angle_deg_sym"] for row in out_rows],
        "Mean principal angle (deg)",
        "Symmetrized neighbor mean principal angle",
        "neighbor_meanangle_sym.png",
    )


if __name__ == "__main__":
    main()