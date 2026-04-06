#!/usr/bin/env python3
# plot_developmental_mode_atlas.py
# Developmental cortical LB atlas on age-specific cortical geometry.

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
from matplotlib.colors import TwoSlopeNorm


DEFAULT_AGES = [
    "2wk", "1mo", "2mo", "3mo", "4.5mo", "6mo", "7.5mo",
    "9mo", "10.5mo", "12mo", "15mo", "18mo", "2yr",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot developmental cortical LB mode atlas"
    )
    parser.add_argument("--eigdir", required=True, help="Directory with saved full-mesh eigensystems")
    parser.add_argument("--subjects-dir", required=True, help="MNE subjects dir")
    parser.add_argument("--outfig", required=True, help="Output figure path")
    parser.add_argument("--K", type=int, default=30, help="K used in saved eigensystems")
    parser.add_argument("--hemi", choices=["lh", "rh"], default="lh", help="Hemisphere to plot")
    parser.add_argument("--modes", default="2,3,4,5,8,12", help="Comma-separated nominal mode indices (1-based)")
    parser.add_argument(
        "--ages",
        default="2wk,2mo,4.5mo,7.5mo,12mo,18mo,2yr",
        help="Comma-separated ages or 'all'",
    )
    parser.add_argument(
        "--surface",
        default="pial",
        choices=["white", "pial", "inflated"],
        help="Surface for display",
    )
    parser.add_argument(
        "--sphere-surf",
        default="sphere.reg",
        choices=["sphere.reg", "sphere"],
        help="Sphere used for sign alignment",
    )
    parser.add_argument("--reference-age", default=None, help="Reference age for sign alignment")
    parser.add_argument("--elev", type=float, default=8.0, help="View elevation")
    parser.add_argument("--azim", type=float, default=180.0, help="View azimuth")
    parser.add_argument("--figscale", type=float, default=1.0, help="Overall figure scale")
    parser.add_argument("--dpi", type=int, default=350, help="Output DPI")
    parser.add_argument(
        "--columnwise-scale",
        action="store_true",
        help="Use separate symmetric color scale for each mode column",
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


def read_fs_surf(subjects_dir: Path, subject: str, hemi: str, surface: str):
    path = subjects_dir / subject / "surf" / f"{hemi}.{surface}"
    rr, tris = nib.freesurfer.read_geometry(str(path))
    return rr.astype(np.float64), tris.astype(np.int32)


def triangle_areas(rr: np.ndarray, tris: np.ndarray) -> np.ndarray:
    tri_rr = rr[tris]
    cross = np.cross(tri_rr[:, 1] - tri_rr[:, 0], tri_rr[:, 2] - tri_rr[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def sphere_unit(rr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rr, axis=1, keepdims=True)
    return rr / np.maximum(norm, 1e-12)


def nearest_neighbor_map(src_xyz: np.ndarray, tgt_xyz: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(sphere_unit(src_xyz))
    _, idx = tree.query(sphere_unit(tgt_xyz), k=1)
    return idx


def load_eigensystems(eigdir: Path, K: int):
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


def sign_align_to_reference(rows, hemi: str, modes, reference_age: str):
    row_by_age = {row["age"]: row for row in rows}
    if reference_age not in row_by_age:
        raise ValueError(f"reference_age={reference_age} not found in loaded rows")

    ref = row_by_age[reference_age]
    ref_vecs = ref[f"evecs_{hemi}"]
    ref_area = ref[f"area_{hemi}"]
    ref_sphere = ref[f"sphere_{hemi}"]

    signs = {}
    for row in rows:
        age = row["age"]
        if age == reference_age:
            for k in modes:
                signs[(age, k)] = 1.0
            continue

        idx = nearest_neighbor_map(row[f"sphere_{hemi}"], ref_sphere)
        vecs_on_ref = row[f"evecs_{hemi}"][idx, :]

        for k in modes:
            j = k - 1
            overlap = np.sum(ref_vecs[:, j] * vecs_on_ref[:, j] * ref_area)
            signs[(age, k)] = 1.0 if overlap >= 0 else -1.0

    return signs


def compute_columnwise_vmax(rows, hemi: str, modes, signs, robust_q: float = 99.0):
    vmax = {}
    for k in modes:
        vals = []
        for row in rows:
            age = row["age"]
            vals.append(signs[(age, k)] * row[f"evecs_{hemi}"][:, k - 1])
        vals = np.concatenate(vals)
        vmax[k] = max(np.percentile(np.abs(vals), robust_q), 1e-8)
    return vmax


def set_equal_limits(ax, xyz_limits):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = xyz_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))


def main():
    args = parse_args()

    eigdir = Path(args.eigdir).expanduser().resolve()
    subjects_dir = Path(args.subjects_dir).expanduser().resolve()
    outfig = Path(args.outfig).expanduser().resolve()
    outfig.parent.mkdir(parents=True, exist_ok=True)

    ages_requested = set(parse_ages(args.ages))
    modes = parse_modes(args.modes)

    rows = [row for row in load_eigensystems(eigdir, args.K) if row["age"] in ages_requested]
    rows.sort(key=lambda r: age_to_months(r["age"]))

    if not rows:
        raise RuntimeError("No matching ages found in eigensystem directory.")

    hemi = args.hemi
    reference_age = rows[0]["age"] if args.reference_age is None else args.reference_age

    for row in rows:
        rr_plot, tris_plot = read_fs_surf(subjects_dir, row["subject"], hemi, args.surface)
        rr_basis, tris_basis = read_fs_surf(
            subjects_dir, row["subject"], hemi, row["surface_basis"]
        )

        n_vertices = row[f"evecs_{hemi}"].shape[0]
        if rr_plot.shape[0] != n_vertices:
            raise RuntimeError(
                f"Vertex count mismatch for {row['subject']} {hemi}: "
                f"{args.surface} has {rr_plot.shape[0]} vertices but eigensystem has {n_vertices}"
            )

        row["rr_plot"] = rr_plot
        row["tris_plot"] = tris_plot
        row["surf_area_mm2"] = float(np.sum(triangle_areas(rr_basis, tris_basis)))

    all_xyz = np.vstack([row["rr_plot"] for row in rows])
    xmin, xmax = np.min(all_xyz[:, 0]), np.max(all_xyz[:, 0])
    ymin, ymax = np.min(all_xyz[:, 1]), np.max(all_xyz[:, 1])
    zmin, zmax = np.min(all_xyz[:, 2]), np.max(all_xyz[:, 2])

    mx = 0.03 * (xmax - xmin + 1e-9)
    my = 0.03 * (ymax - ymin + 1e-9)
    mz = 0.03 * (zmax - zmin + 1e-9)
    xyz_limits = (
        (xmin - mx, xmax + mx),
        (ymin - my, ymax + my),
        (zmin - mz, zmax + mz),
    )

    signs = sign_align_to_reference(rows, hemi=hemi, modes=modes, reference_age=reference_age)
    vmax_by_mode = compute_columnwise_vmax(rows, hemi=hemi, modes=modes, signs=signs)

    n_rows = len(rows)
    n_cols = len(modes)

    fig = plt.figure(figsize=(2.0 * n_cols * args.figscale, 1.75 * n_rows * args.figscale))
    cmap = plt.get_cmap("coolwarm")

    for i, row in enumerate(rows):
        age = row["age"]
        rr = row["rr_plot"]
        tris = row["tris_plot"]
        tri = mtri.Triangulation(rr[:, 0], rr[:, 1], triangles=tris)
        area_cm2 = row["surf_area_mm2"] / 100.0

        for j, k in enumerate(modes):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection="3d")

            vals = signs[(age, k)] * row[f"evecs_{hemi}"][:, k - 1]
            face_vals = vals[tris].mean(axis=1)

            vmax = vmax_by_mode[k] if args.columnwise_scale else max(vmax_by_mode.values())
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

            surf = ax.plot_trisurf(
                tri,
                rr[:, 2],
                linewidth=0.0,
                antialiased=False,
                shade=False,
            )
            surf.set_facecolors(cmap(norm(face_vals)))
            surf.set_edgecolor("none")

            try:
                ax.set_proj_type("ortho")
            except Exception:
                pass

            ax.view_init(elev=args.elev, azim=args.azim)
            set_equal_limits(ax, xyz_limits)
            ax.set_axis_off()

            if i == 0:
                ax.set_title(f"mode {k}", fontsize=10, pad=2)

            if j == 0:
                ax.text2D(
                    -0.18,
                    0.5,
                    f"{age}\n{hemi.upper()} area={area_cm2:.1f} cm$^2$",
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=9,
                )

    title = f"Developmental cortical LB mode atlas ({hemi.upper()}; sign-aligned to {reference_age} for display)"
    subtitle = "Common physical scale across ages; colors scaled within mode columns"

    fig.text(0.5, 0.992, title, ha="center", va="top", fontsize=16)
    fig.text(0.5, 0.962, subtitle, ha="center", va="top", fontsize=11)

    plt.subplots_adjust(
        left=0.10,
        right=0.995,
        top=0.88,
        bottom=0.03,
        wspace=0.02,
        hspace=0.03,
    )

    plt.savefig(outfig, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {outfig}")


if __name__ == "__main__":
    main()