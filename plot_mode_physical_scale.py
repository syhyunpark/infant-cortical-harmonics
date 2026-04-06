#!/usr/bin/env python3
# plot_mode_physical_scale.py
# Plot age-specific physical scale of cortical LB modes:
#   ell_k(a) = 2*pi / sqrt(lambda_k(a))
# Main figure: physical wavelength-like scale in cm
# Optional supplementary row: area-normalized wavelength

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_AGES = [
    "2wk", "1mo", "2mo", "3mo", "4.5mo", "6mo", "7.5mo",
    "9mo", "10.5mo", "12mo", "15mo", "18mo", "2yr",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot age-specific physical scale of LB modes"
    )
    parser.add_argument("--eigdir", required=True, help="Directory with saved full-mesh eigensystems")
    parser.add_argument("--outfig", required=True, help="Output figure path")
    parser.add_argument("--outcsv", default=None, help="Optional output CSV path")
    parser.add_argument("--K", type=int, default=30, help="K used in saved eigensystems")
    parser.add_argument("--modes", default="2,3,4,5,8,12", help="Comma-separated nominal mode indices")
    parser.add_argument("--ages", default="all", help="Comma-separated ages or 'all'")
    parser.add_argument("--hemi", choices=["lh", "rh", "both"], default="both", help="Hemisphere(s)")
    parser.add_argument("--show-normalized", action="store_true", help="Also show area-normalized wavelength")
    parser.add_argument("--figscale", type=float, default=1.1, help="Overall figure scale")
    parser.add_argument("--dpi", type=int, default=350, help="Output DPI")
    parser.add_argument(
        "--null-tol",
        type=float,
        default=1e-10,
        help="Treat lambda <= null_tol as invalid and plot as NaN",
    )
    return parser.parse_args()


def parse_ages(text: str):
    text = text.strip()
    if text.lower() == "all":
        return DEFAULT_AGES
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_modes(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def load_eigensystems(eigdir: Path, K: int):
    files = sorted(eigdir.glob(f"*_fullmesh_lb_*_K{K}.npz"))
    if not files:
        raise RuntimeError(f"No eigensystem files found in {eigdir.resolve()} for K={K}")

    rows = []
    for path in files:
        x = np.load(path, allow_pickle=True)
        rows.append(
            {
                "age": scalar_from_npz(x, "age"),
                "subject": scalar_from_npz(x, "subject"),
                "evals_lh": x["evals_lh"].astype(float),
                "evals_rh": x["evals_rh"].astype(float),
                "area_lh": x["area_lh"].astype(float),
                "area_rh": x["area_rh"].astype(float),
            }
        )
    rows.sort(key=lambda r: age_to_months(r["age"]))
    return rows


def wavelength_from_lambda(lam: float, null_tol: float) -> float:
    if lam <= null_tol:
        return np.nan
    return float(2.0 * np.pi / np.sqrt(lam))


def main():
    args = parse_args()

    eigdir = Path(args.eigdir).expanduser().resolve()
    outfig = Path(args.outfig).expanduser().resolve()
    outfig.parent.mkdir(parents=True, exist_ok=True)

    outcsv = (
        outfig.with_suffix(".csv")
        if args.outcsv is None
        else Path(args.outcsv).expanduser().resolve()
    )
    outcsv.parent.mkdir(parents=True, exist_ok=True)

    ages_requested = set(parse_ages(args.ages))
    modes = parse_modes(args.modes)

    rows = [r for r in load_eigensystems(eigdir, args.K) if r["age"] in ages_requested]
    rows.sort(key=lambda r: age_to_months(r["age"]))

    if not rows:
        raise RuntimeError("No matching ages found.")

    data = []
    bad_modes = []

    for row in rows:
        area = {
            "lh": float(np.sum(row["area_lh"])),
            "rh": float(np.sum(row["area_rh"])),
        }

        for hemi in ("lh", "rh"):
            evals = row[f"evals_{hemi}"]
            for k in modes:
                lam = float(evals[k - 1])
                ell_mm = wavelength_from_lambda(lam, args.null_tol)

                if np.isnan(ell_mm):
                    bad_modes.append((row["age"], hemi, k, lam))
                    ell_cm = np.nan
                    ell_norm = np.nan
                else:
                    ell_cm = ell_mm / 10.0
                    ell_norm = ell_mm / np.sqrt(area[hemi])

                data.append(
                    {
                        "age": row["age"],
                        "subject": row["subject"],
                        "hemi": hemi,
                        "mode": k,
                        "lambda": lam,
                        "ell_cm": ell_cm,
                        "ell_norm": ell_norm,
                        "surface_area_mm2": area[hemi],
                    }
                )

    if bad_modes:
        print("[plot_mode_physical_scale] warning: near-null eigenvalues encountered")
        for age, hemi, k, lam in bad_modes:
            print(f"  age={age}, hemi={hemi}, mode={k}, lambda={lam:.3e}")

    with outcsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved CSV -> {outcsv}")

    if args.hemi == "both":
        hemis = ["lh", "rh"]
    else:
        hemis = [args.hemi]

    ncols = len(hemis)
    nrows = 2 if args.show_normalized else 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols * args.figscale, 3.8 * nrows * args.figscale),
        squeeze=False,
    )

    ages = [row["age"] for row in rows]
    x = np.arange(len(ages))

    cmap = plt.get_cmap("tab10")
    colors = {k: cmap(i % 10) for i, k in enumerate(modes)}
    line_styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    for col, hemi in enumerate(hemis):
        hemi_data = [d for d in data if d["hemi"] == hemi]

        ax = axes[0, col]
        for i, k in enumerate(modes):
            y = [d["ell_cm"] for d in hemi_data if d["mode"] == k]
            ax.plot(
                x,
                y,
                marker="o",
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                color=colors[k],
                label=f"mode {k}",
            )
        ax.set_title(f"{hemi.upper()}: effective wavelength")
        ax.set_ylabel("effective wavelength (cm)")
        ax.set_xlabel("Age")
        ax.set_xticks(x)
        ax.set_xticklabels(ages, rotation=45, ha="right")

        if args.show_normalized:
            ax = axes[1, col]
            for i, k in enumerate(modes):
                y = [d["ell_norm"] for d in hemi_data if d["mode"] == k]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2,
                    color=colors[k],
                    label=f"mode {k}",
                )
            ax.set_title(f"{hemi.upper()}: normalized wavelength")
            ax.set_ylabel("normalized wavelength")
            ax.set_xlabel("Age")
            ax.set_xticks(x)
            ax.set_xticklabels(ages, rotation=45, ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=min(len(modes), 6),
        frameon=False,
        fontsize=11,
    )

    fig.text(
        0.5,
        0.992,
        "Age-specific physical scale of cortical LB modes",
        ha="center",
        va="top",
        fontsize=18,
    )

    top = 0.80 if args.show_normalized else 0.84
    plt.subplots_adjust(
        top=top,
        bottom=0.12,
        left=0.10,
        right=0.98,
        wspace=0.22,
        hspace=0.55,
    )
    plt.savefig(outfig, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure -> {outfig}")


if __name__ == "__main__":
    main()