#!/usr/bin/env python3
# compare_recoverability_basis_mismatch.py
# Compare matched vs adult-mismatch recoverability.

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare matched vs adult-mismatch recoverability"
    )
    parser.add_argument("--matched-csv", required=True)
    parser.add_argument("--mismatch-csv", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument(
        "--metric",
        default="mean_corr_first10",
        choices=["mean_corr_first5", "mean_corr_first10", "mean_corr_all", "Krec_largest"],
    )
    parser.add_argument(
        "--table-snr",
        type=float,
        default=20.0,
        help="SNR (dB) for the age-by-age table and optional heatmap",
    )
    parser.add_argument(
        "--matched-raw-csv",
        default=None,
        help="Optional raw mode-wise CSV from matched simulation",
    )
    parser.add_argument(
        "--mismatch-raw-csv",
        default=None,
        help="Optional raw mode-wise CSV from adult-mismatch simulation",
    )
    parser.add_argument("--dpi", type=int, default=350)

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


def load_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def key_of_summary(row: dict):
    return (row["age"], row["snr_db"], row["analysis_k"], row["prior_kind"])


def key_of_raw(row: dict):
    return (row["age"], row["snr_db"], row["analysis_k"], row["prior_kind"], row["mode_order"])


def metric_label(metric: str) -> str:
    return {
        "mean_corr_first5": "Mean recoverability\n(first 5 paired orders)",
        "mean_corr_first10": "Mean recoverability\n(first 10 paired orders)",
        "mean_corr_all": "Mean recoverability\n(all paired orders)",
        "Krec_largest": r"$K_{\mathrm{rec},0.8}$",
    }[metric]


def metric_short_label(metric: str) -> str:
    return {
        "mean_corr_first5": "Mean recoverability (first 5 paired orders)",
        "mean_corr_first10": "Mean recoverability (first 10 paired orders)",
        "mean_corr_all": "Mean recoverability (all paired orders)",
        "Krec_largest": r"$K_{\mathrm{rec},0.8}$",
    }[metric]


def fmt_num(x: float, ndigits: int = 3) -> str:
    if np.isnan(x):
        return ""
    return f"{x:.{ndigits}f}"


def merge_rows(matched, mismatch, metric: str):
    matched_map = {key_of_summary(row): row for row in matched}
    mismatch_map = {key_of_summary(row): row for row in mismatch}

    common_keys = sorted(
        set(matched_map) & set(mismatch_map),
        key=lambda k: (age_to_months(k[0]), float(k[1]), int(k[2]), k[3]),
    )

    out = []
    for key in common_keys:
        r1 = matched_map[key]
        r2 = mismatch_map[key]

        matched_val = float(r1[metric])
        mismatch_val = float(r2[metric])

        out.append(
            {
                "age": r1["age"],
                "snr_db": float(r1["snr_db"]),
                "analysis_k": int(r1["analysis_k"]),
                "prior_kind": r1["prior_kind"],
                "matched": matched_val,
                "adult_mismatch": mismatch_val,
                "drop": matched_val - mismatch_val,
            }
        )

    return out


def summarize_by_snr(merged):
    out = []

    for snr in sorted({row["snr_db"] for row in merged}):
        rows = [row for row in merged if row["snr_db"] == snr]

        matched_vals = np.array([row["matched"] for row in rows], dtype=float)
        mismatch_vals = np.array([row["adult_mismatch"] for row in rows], dtype=float)
        drop_vals = np.array([row["drop"] for row in rows], dtype=float)

        out.append(
            {
                "snr_db": snr,
                "median_matched": float(np.median(matched_vals)),
                "median_adult_mismatch": float(np.median(mismatch_vals)),
                "median_drop": float(np.median(drop_vals)),
                "n_ages": len(rows),
            }
        )

    return out


def save_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_main_figure(merged, metric: str, outfig: Path, dpi: int):
    snrs = sorted({row["snr_db"] for row in merged})
    n_panels = len(snrs)

    all_vals = []
    for row in merged:
        all_vals.extend([row["matched"], row["adult_mismatch"]])

    ymin = min(all_vals)
    ymax = max(all_vals)
    pad = 0.05 * (ymax - ymin + 1e-12)
    ymin -= pad
    ymax += pad

    fig, axes = plt.subplots(1, n_panels, figsize=(4.9 * n_panels, 4.9), sharey=True)
    if n_panels == 1:
        axes = [axes]

    handles = None
    labels = None

    for ax, snr in zip(axes, snrs):
        rows = [row for row in merged if row["snr_db"] == snr]
        rows.sort(key=lambda row: age_to_months(row["age"]))

        x = np.arange(len(rows))
        age_labels = [row["age"] for row in rows]

        h1, = ax.plot(
            x,
            [row["matched"] for row in rows],
            marker="o",
            linewidth=1.8,
            linestyle="-",
            label="Age-specific infant basis",
        )
        h2, = ax.plot(
            x,
            [row["adult_mismatch"] for row in rows],
            marker="s",
            linewidth=1.8,
            linestyle=":",
            label="Adult-derived basis",
        )

        ax.axhline(0.0, linestyle="--", linewidth=0.9)
        ax.set_title(f"{int(snr)} dB")
        ax.set_xticks(x)
        ax.set_xticklabels(age_labels, rotation=45, ha="right")
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.25)

        handles = [h1, h2]
        labels = ["Age-specific infant basis", "Adult-derived basis"]

    axes[0].set_ylabel(metric_label(metric))

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    fig.suptitle(
        "Recoverability under matched infant vs adult-derived analysis bases",
        fontsize=17,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.savefig(outfig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_age_table_rows(merged, snr: float):
    rows = [row for row in merged if row["snr_db"] == snr]
    rows.sort(key=lambda row: age_to_months(row["age"]))
    return rows


def write_age_table_latex(rows, metric: str, snr: float, outpath: Path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{\textbf{{Age-by-age recoverability under the age-specific infant and adult-derived analysis bases at {int(snr)} dB.}} "
        rf"Values are reported for {metric_short_label(metric).lower()}. "
        r"The drop is defined as matched recoverability minus adult-derived recoverability.}",
        rf"\label{{tab:recoverability_basis_mismatch_{metric}_{int(snr)}dB}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Age & Age-specific infant basis & Adult-derived basis & Drop \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['age']} & {fmt_num(row['matched'])} & {fmt_num(row['adult_mismatch'])} & {fmt_num(row['drop'])} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    outpath.write_text("\n".join(lines))


def merge_raw_rows(matched_raw, mismatch_raw):
    matched_map = {key_of_raw(row): row for row in matched_raw}
    mismatch_map = {key_of_raw(row): row for row in mismatch_raw}

    common_keys = sorted(
        set(matched_map) & set(mismatch_map),
        key=lambda k: (float(k[1]), age_to_months(k[0]), int(k[4])),
    )

    out = []
    for key in common_keys:
        r1 = matched_map[key]
        r2 = mismatch_map[key]

        matched_val = float(r1["paired_corr"])
        mismatch_val = float(r2["paired_corr"])

        out.append(
            {
                "age": r1["age"],
                "snr_db": float(r1["snr_db"]),
                "analysis_k": int(r1["analysis_k"]),
                "prior_kind": r1["prior_kind"],
                "mode_order": int(r1["mode_order"]),
                "matched": matched_val,
                "adult_mismatch": mismatch_val,
                "drop": matched_val - mismatch_val,
            }
        )

    return out


def make_heatmap(raw_merged, snr: float, outfig: Path, dpi: int):
    rows = [row for row in raw_merged if row["snr_db"] == snr]
    if not rows:
        return

    ages = sorted({row["age"] for row in rows}, key=age_to_months)
    modes = sorted({row["mode_order"] for row in rows})

    age_to_i = {age: i for i, age in enumerate(ages)}
    mode_to_j = {mode: j for j, mode in enumerate(modes)}

    mat = np.full((len(ages), len(modes)), np.nan, dtype=float)
    for row in rows:
        mat[age_to_i[row["age"]], mode_to_j[row["mode_order"]]] = row["drop"]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(mat, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_yticks(np.arange(len(ages)))
    ax.set_yticklabels(ages)
    ax.set_xlabel("Paired mode order")
    ax.set_ylabel("Age")
    ax.set_title("Age-specific vs. adult-derived basis recoverability drop")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Recoverability drop")

    plt.tight_layout()
    plt.savefig(outfig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    matched = load_csv(Path(args.matched_csv).expanduser().resolve())
    mismatch = load_csv(Path(args.mismatch_csv).expanduser().resolve())

    merged = merge_rows(matched, mismatch, args.metric)
    if not merged:
        raise RuntimeError("No common rows between matched and adult-mismatch CSVs.")

    summary_rows = summarize_by_snr(merged)
    summary_csv = outdir / "recoverability_basis_mismatch_summary_by_snr.csv"
    save_csv(summary_rows, summary_csv)

    figfile = outdir / f"recoverability_compare_{args.metric}.png"
    make_main_figure(merged, args.metric, figfile, args.dpi)

    age_rows = make_age_table_rows(merged, args.table_snr)
    if age_rows:
        age_csv = outdir / f"recoverability_basis_mismatch_age_by_age_{int(args.table_snr)}dB.csv"
        age_tex = outdir / f"recoverability_basis_mismatch_age_by_age_{int(args.table_snr)}dB.tex"

        save_csv(age_rows, age_csv)
        write_age_table_latex(age_rows, args.metric, args.table_snr, age_tex)

        print(f"Saved -> {age_csv}")
        print(f"Saved -> {age_tex}")

    if args.matched_raw_csv and args.mismatch_raw_csv:
        matched_raw = load_csv(Path(args.matched_raw_csv).expanduser().resolve())
        mismatch_raw = load_csv(Path(args.mismatch_raw_csv).expanduser().resolve())

        raw_merged = merge_raw_rows(matched_raw, mismatch_raw)
        heatmap_file = outdir / f"recoverability_drop_heatmap_{int(args.table_snr)}dB.png"
        make_heatmap(raw_merged, args.table_snr, heatmap_file, args.dpi)

        print(f"Saved -> {heatmap_file}")

    print(f"Saved -> {summary_csv}")
    print(f"Saved -> {figfile}")


if __name__ == "__main__":
    main()