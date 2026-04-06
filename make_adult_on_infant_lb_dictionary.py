#!/usr/bin/env python3
# make_adult_on_infant_lb_dictionary.py
#   Map an adult full-mesh LB basis onto each infant template mesh,
# restrict to the infant source space, sign-align to the infant basis,
# and build an adult-derived analysis dictionary using the infant forward operator.

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build adult-on-infant analysis dictionaries"
    )
    parser.add_argument("--adult-eig", required=True, help="Adult full-mesh eigensystem .npz")
    parser.add_argument("--infant-dictdir", required=True, help="Directory containing infant dictionary .npz files")
    parser.add_argument("--subjects-dir", required=True, help="FreeSurfer SUBJECTS_DIR")
    parser.add_argument("--target-k", type=int, default=50, help="Modes per hemisphere")
    parser.add_argument("--surface", default="white", help="Filter infant dictionaries by surface")
    parser.add_argument("--combine", default="block", help="Filter infant dictionaries by combine")
    parser.add_argument("--montage-kind", default="template1020", help="Filter infant dictionaries by montage kind")
    parser.add_argument("--outdir", required=True, help="Output directory")
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


def read_sphere(subjects_dir: Path, subject: str, hemi: str, sphere_surf: str = "sphere.reg"):
    path = subjects_dir / subject / "surf" / f"{hemi}.{sphere_surf}"
    rr, _ = nib.freesurfer.read_geometry(str(path))
    return rr.astype(np.float64)


def sphere_unit(rr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(rr, axis=1, keepdims=True)
    return rr / np.maximum(norm, 1e-12)


def nearest_neighbor_map(src_xyz: np.ndarray, tgt_xyz: np.ndarray) -> np.ndarray:
    tree = cKDTree(sphere_unit(src_xyz))
    _, idx = tree.query(sphere_unit(tgt_xyz), k=1)
    return idx


def load_adult_eig(path: Path):
    x = np.load(path, allow_pickle=True)

    subject = x["subject"][0]
    surface = x["surface"][0]
    sphere_surf = x["sphere_surf"][0]

    if hasattr(subject, "item"):
        subject = subject.item()
    if hasattr(surface, "item"):
        surface = surface.item()
    if hasattr(sphere_surf, "item"):
        sphere_surf = sphere_surf.item()

    return {
        "subject": subject,
        "surface": surface,
        "sphere_surf": sphere_surf,
        "K": int(np.array(x["K"]).ravel()[0]),
        "evals_lh": x["evals_lh"].astype(float),
        "evals_rh": x["evals_rh"].astype(float),
        "evecs_lh": x["evecs_lh"].astype(float),
        "evecs_rh": x["evecs_rh"].astype(float),
        "sphere_lh": x["sphere_lh"].astype(float),
        "sphere_rh": x["sphere_rh"].astype(float),
    }


def get_vertno(x, hemi: str):
    if f"vertno_{hemi}" in x:
        return x[f"vertno_{hemi}"].astype(int)
    if f"{hemi}_vertno" in x:
        return x[f"{hemi}_vertno"].astype(int)
    raise KeyError(f"Could not find source vertex indices for {hemi}")


def get_channels(x):
    for key in ["channels", "ch_names"]:
        if key in x:
            arr = x[key]
            return [str(v.item() if hasattr(v, "item") else v) for v in arr]
    return []


def get_infant_phi_src(x):
    if "Phi_src" in x:
        return x["Phi_src"].astype(float)
    return None


def norm_cols(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    norm = np.linalg.norm(X, axis=0, keepdims=True)
    return X / np.maximum(norm, 1e-12)


def sign_align_cols(X: np.ndarray, Xref: np.ndarray):
    X = X.copy()
    signs = np.ones(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        if float(np.dot(X[:, j], Xref[:, j])) < 0:
            X[:, j] *= -1.0
            signs[j] = -1.0

    return X, signs


def load_infant_rows(dictdir: Path, target_k: int, surface: str, combine: str, montage_kind: str):
    files = []

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

        files.append(path)

    if not files:
        raise RuntimeError("No matching infant dictionary files found.")

    return files


def main():
    args = parse_args()

    adult = load_adult_eig(Path(args.adult_eig).expanduser().resolve())

    dictdir = Path(args.infant_dictdir).expanduser().resolve()
    subjects_dir = Path(args.subjects_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    infant_files = load_infant_rows(
        dictdir,
        args.target_k,
        args.surface,
        args.combine,
        args.montage_kind,
    )

    for path in infant_files:
        x = np.load(path, allow_pickle=True)

        age = infer_scalar(x, "age", path.stem)
        subject = infer_scalar(x, "subject")
        if subject is None:
            raise RuntimeError(f"{path.name} does not contain subject metadata")

        G = x["G"].astype(float)
        D_infant = x["D"].astype(float)
        K = int(np.array(x["K"]).ravel()[0])
        M = G.shape[0]

        vertno_lh = get_vertno(x, "lh")
        vertno_rh = get_vertno(x, "rh")
        phi_infant_src = get_infant_phi_src(x)

        sphere_lh_infant = read_sphere(subjects_dir, subject, "lh", adult["sphere_surf"])
        sphere_rh_infant = read_sphere(subjects_dir, subject, "rh", adult["sphere_surf"])

        idx_lh = nearest_neighbor_map(adult["sphere_lh"], sphere_lh_infant)
        idx_rh = nearest_neighbor_map(adult["sphere_rh"], sphere_rh_infant)

        adult_lh_on_infant_full = adult["evecs_lh"][idx_lh, :K]
        adult_rh_on_infant_full = adult["evecs_rh"][idx_rh, :K]

        phi_lh_src = norm_cols(adult_lh_on_infant_full[vertno_lh, :])
        phi_rh_src = norm_cols(adult_rh_on_infant_full[vertno_rh, :])

        if phi_infant_src is not None:
            phi_infant_lh = phi_infant_src[: len(vertno_lh), :K]
            phi_infant_rh = phi_infant_src[len(vertno_lh) :, K : 2 * K]

            phi_lh_src, signs_lh = sign_align_cols(phi_lh_src, phi_infant_lh)
            phi_rh_src, signs_rh = sign_align_cols(phi_rh_src, phi_infant_rh)
            sign_align_reference = "infant_source_basis"

        else:
            D_lh_ref = D_infant[:, :K]
            D_rh_ref = D_infant[:, K : 2 * K]

            D_lh_tmp = G[:, : len(vertno_lh)] @ phi_lh_src
            D_rh_tmp = G[:, len(vertno_lh) :] @ phi_rh_src

            _, signs_lh = sign_align_cols(D_lh_tmp, D_lh_ref)
            _, signs_rh = sign_align_cols(D_rh_tmp, D_rh_ref)

            phi_lh_src = phi_lh_src * signs_lh[np.newaxis, :]
            phi_rh_src = phi_rh_src * signs_rh[np.newaxis, :]
            sign_align_reference = "infant_sensor_dictionary"

        phi_src = np.zeros((len(vertno_lh) + len(vertno_rh), 2 * K), dtype=float)
        phi_src[: len(vertno_lh), :K] = phi_lh_src
        phi_src[len(vertno_lh) :, K:] = phi_rh_src

        D = G @ phi_src

        out = {
            "age": np.array([age], dtype=object),
            "subject": np.array([subject], dtype=object),
            "analysis_basis_kind": np.array(["adult_on_infant"], dtype=object),
            "analysis_basis_subject": np.array([adult["subject"]], dtype=object),
            "surface": np.array([args.surface], dtype=object),
            "combine": np.array([args.combine], dtype=object),
            "montage_kind": np.array([args.montage_kind], dtype=object),
            "K": np.array([K], dtype=np.int32),
            "G": G,
            "D": D,
            "Phi_src": phi_src,
            "vertno_lh": vertno_lh,
            "vertno_rh": vertno_rh,
            "channels": np.array(get_channels(x), dtype=object),
            "sign_align_reference": np.array([sign_align_reference], dtype=object),
            "signs_lh": signs_lh,
            "signs_rh": signs_rh,
        }

        outfile = outdir / (
            f"{subject}_D_{args.surface}_{args.combine}_K{K}_M{M}_"
            f"{args.montage_kind}_adult_on_{adult['subject']}.npz"
        )
        np.savez_compressed(outfile, **out)
        print(f"Saved -> {outfile}")


if __name__ == "__main__":
    main()