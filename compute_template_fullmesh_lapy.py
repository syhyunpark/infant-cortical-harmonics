#!/usr/bin/env python3
# compute_template_fullmesh_lapy.py
# Compute full-mesh Laplace--Beltrami eigensystems for a FreeSurfer template subject.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from lapy import Solver, TriaMesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute full-mesh LB eigensystem for a template subject"
    )
    parser.add_argument("--subjects-dir", default=None, help="FreeSurfer SUBJECTS_DIR")
    parser.add_argument("--subject", default="fsaverage", help="Template subject name, e.g. fsaverage")
    parser.add_argument(
        "--subject-path",
        default=None,
        help="Direct path to the subject directory (overrides --subjects-dir/--subject)",
    )
    parser.add_argument(
        "--surface",
        default="white",
        choices=["white", "pial", "inflated"],
        help="Surface for LB basis",
    )
    parser.add_argument(
        "--sphere-surf",
        default="sphere.reg",
        choices=["sphere.reg", "sphere"],
        help="Sphere surface to save",
    )
    parser.add_argument("--K", type=int, default=50, help="Number of non-constant modes per hemisphere")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def resolve_subject_dir(subjects_dir: str | None, subject: str, subject_path: str | None) -> Path:
    if subject_path is not None:
        path = Path(subject_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"--subject-path does not exist: {path}")

    candidates = []

    if subjects_dir is not None:
        candidates.append(Path(subjects_dir).expanduser().resolve() / subject)

    env_subjects = os.environ.get("SUBJECTS_DIR")
    if env_subjects:
        candidates.append(Path(env_subjects).expanduser().resolve() / subject)

    home = Path.home()
    candidates.extend(
        [
            home / "mne_data" / "MNE-fsaverage-data" / subject,
            home / "mne_data" / "MNE-fsaverage-data" / "fsaverage",
            home / "mne_data" / "subjects" / subject,
            Path("./subjects") / subject,
            Path("./subjects_dir") / subject,
        ]
    )

    checked = []
    for path in candidates:
        path = path.resolve()
        checked.append(str(path))

        surf_dir = path / "surf"
        if surf_dir.exists() and (surf_dir / "lh.white").exists():
            return path

    msg = (
        f"Could not locate subject '{subject}'.\n"
        f"Searched these candidate directories:\n  - " + "\n  - ".join(checked) + "\n\n"
        f"Fix by either:\n"
        f"  1) passing the correct --subjects-dir that contains '{subject}', or\n"
        f"  2) passing --subject-path directly to the subject directory.\n\n"
        f"For example, the script expects a directory like:\n"
        f"  <subject_dir>/surf/lh.white\n"
    )
    raise FileNotFoundError(msg)


def read_surf(subject_dir: Path, hemi: str, surf_name: str):
    path = subject_dir / "surf" / f"{hemi}.{surf_name}"
    if not path.exists():
        raise FileNotFoundError(f"Missing surface file: {path}")

    rr, tris = nib.freesurfer.read_geometry(str(path))
    return rr.astype(np.float64), tris.astype(np.int32)


def vertex_areas(rr: np.ndarray, tris: np.ndarray) -> np.ndarray:
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


def solve_lb(rr: np.ndarray, tris: np.ndarray, K: int):
    mesh = TriaMesh(rr, tris)
    fem = Solver(mesh)

    evals, evecs = fem.eigs(k=K + 1)
    evals = np.asarray(evals, dtype=np.float64)
    evecs = np.asarray(evecs, dtype=np.float64)

    return evals[1 : K + 1], evecs[:, 1 : K + 1]


def main():
    args = parse_args()

    subject_dir = resolve_subject_dir(args.subjects_dir, args.subject, args.subject_path)
    subject_name = subject_dir.name

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    out = {
        "subject": np.array([subject_name], dtype=object),
        "surface": np.array([args.surface], dtype=object),
        "sphere_surf": np.array([args.sphere_surf], dtype=object),
        "K": np.array([args.K], dtype=np.int32),
    }

    for hemi in ["lh", "rh"]:
        rr, tris = read_surf(subject_dir, hemi, args.surface)
        sphere_rr, _ = read_surf(subject_dir, hemi, args.sphere_surf)

        evals, evecs = solve_lb(rr,