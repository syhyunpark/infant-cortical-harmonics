#!/usr/bin/env python3
# compute_infant_fullmesh_lapy.py
# Compute full-mesh cortical LB eigensystems for MNE infant templates.

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import nibabel as nib
import numpy as np
from lapy import Solver, TriaMesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


DEFAULT_AGES = [
    "2wk", "1mo", "2mo", "3mo", "4.5mo", "6mo", "7.5mo",
    "9mo", "10.5mo", "12mo", "15mo", "18mo", "2yr",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute full-mesh infant LB eigensystems"
    )
    parser.add_argument("--ages", default="all", help="Comma-separated ages or 'all'")
    parser.add_argument("--subjects-dir", required=True, help="MNE subjects dir")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--K", type=int, default=30, help="Modes per hemisphere")
    parser.add_argument("--surface", default="white", choices=["white", "pial"], help="Surface")
    parser.add_argument("--sphere-surf", default="sphere.reg", choices=["sphere.reg", "sphere"], help="Sphere")

    parser.add_argument("--lump", action="store_true", help="Use lumped mass in LaPy")
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="Keep only the largest connected mesh component",
    )
    parser.add_argument(
        "--null-tol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for filtering null eigenvalues",
    )
    parser.add_argument(
        "--extra-eigs",
        type=int,
        default=8,
        help="Extra eigenpairs to compute before filtering null modes",
    )

    return parser.parse_args()


def parse_ages(text: str):
    text = text.strip()
    if text.lower() == "all":
        return DEFAULT_AGES
    return [x.strip() for x in text.split(",") if x.strip()]


def fetch_infant_subject(age: str, subjects_dir: Path, verbose: bool = True) -> str:
    subjects_dir.mkdir(parents=True, exist_ok=True)
    return mne.datasets.fetch_infant_template(
        age=age,
        subjects_dir=str(subjects_dir),
        verbose=verbose,
    )


def read_fs_surf(subjects_dir: Path, subject: str, hemi: str, surf_name: str):
    path = subjects_dir / subject / "surf" / f"{hemi}.{surf_name}"
    rr, tris = nib.freesurfer.read_geometry(str(path))
    return rr.astype(np.float64), tris.astype(np.int32)


def vertex_area_weights(rr: np.ndarray, tris: np.ndarray) -> np.ndarray:
    tri_rr = rr[tris]
    tri_area = 0.5 * np.linalg.norm(
        np.cross(tri_rr[:, 1] - tri_rr[:, 0], tri_rr[:, 2] - tri_rr[:, 0]),
        axis=1,
    )

    area = np.zeros(rr.shape[0], dtype=np.float64)
    np.add.at(area, tris[:, 0], tri_area / 3.0)
    np.add.at(area, tris[:, 1], tri_area / 3.0)
    np.add.at(area, tris[:, 2], tri_area / 3.0)

    return area


def normalize_cols_area(U: np.ndarray, area: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.sqrt(np.sum((U ** 2) * area[:, None], axis=0))
    return U / np.maximum(norms, eps)[None, :]


def largest_connected_component(rr: np.ndarray, tris: np.ndarray):
    n = rr.shape[0]

    edges = np.vstack(
        [
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
        ]
    )
    data = np.ones(edges.shape[0], dtype=np.int8)

    A = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = A + A.T

    ncomp, labels = connected_components(A, directed=False)
    if ncomp == 1:
        return rr, tris, np.arange(n, dtype=int)

    counts = np.bincount(labels)
    keep_lab = np.argmax(counts)
    keep = labels == keep_lab
    keep_idx = np.where(keep)[0]

    reindex = -np.ones(n, dtype=int)
    reindex[keep_idx] = np.arange(keep_idx.size)

    mask = np.all(keep[tris], axis=1)
    tris_new = reindex[tris[mask]]
    rr_new = rr[keep_idx]

    return rr_new, tris_new.astype(np.int32), keep_idx


def eigs_lapy_filtered(
    rr: np.ndarray,
    tris: np.ndarray,
    K: int,
    extra_eigs: int,
    lump: bool,
    null_tol: float,
):
    n_try = K + 1 + extra_eigs

    mesh = TriaMesh(rr, tris)
    solver = Solver(mesh, lump=lump)
    evals, evecs = solver.eigs(n_try)

    evals = np.asarray(evals, dtype=np.float64)
    evecs = np.asarray(evecs, dtype=np.float64)

    keep = evals > null_tol
    evals = evals[keep]
    evecs = evecs[:, keep]

    if evals.size < K:
        raise RuntimeError(
            f"Only {evals.size} non-null eigenvalues remained after filtering; "
            f"increase --extra-eigs or inspect the mesh."
        )

    return evals[:K], evecs[:, :K]


def main():
    args = parse_args()

    subjects_dir = Path(args.subjects_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for age in parse_ages(args.ages):
        subject = fetch_infant_subject(age, subjects_dir, verbose=True)

        save_dict = {
            "age": np.array([age], dtype=object),
            "subject": np.array([subject], dtype=object),
            "surface": np.array([args.surface], dtype=object),
            "sphere_surf": np.array([args.sphere_surf], dtype=object),
            "K": np.array([args.K], dtype=int),
            "null_tol": np.array([args.null_tol], dtype=float),
        }

        for hemi in ["lh", "rh"]:
            rr, tris = read_fs_surf(subjects_dir, subject, hemi, args.surface)
            sphere_rr, _ = read_fs_surf(subjects_dir, subject, hemi, args.sphere_surf)

            if args.keep_largest_component:
                rr, tris, keep_idx = largest_connected_component(rr, tris)
                sphere_rr = sphere_rr[keep_idx]
            else:
                keep_idx = np.arange(rr.shape[0], dtype=int)

            area = vertex_area_weights(rr, tris)

            evals, evecs = eigs_lapy_filtered(
                rr=rr,
                tris=tris,
                K=args.K,
                extra_eigs=args.extra_eigs,
                lump=args.lump,
                null_tol=args.null_tol,
            )
            evecs = normalize_cols_area(evecs, area)

            save_dict[f"evals_{hemi}"] = evals
            save_dict[f"evecs_{hemi}"] = evecs
            save_dict[f"area_{hemi}"] = area
            save_dict[f"sphere_{hemi}"] = sphere_rr.astype(np.float32)
            save_dict[f"keep_idx_{hemi}"] = keep_idx.astype(int)

        out_path = outdir / f"{subject}_fullmesh_lb_{args.surface}_K{args.K}.npz"
        np.savez_compressed(out_path, **save_dict)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()