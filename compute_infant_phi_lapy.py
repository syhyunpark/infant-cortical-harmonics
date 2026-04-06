# compute_infant_phi_lapy.py
# Compute cortical LB eigenmodes for MNE infant templates and save Phi aligned to src.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from infant_lb_common import DEFAULT_AGES, get_phi_infant_template


def parse_ages(text: str):
    text = text.strip()
    if text.lower() == "all":
        return DEFAULT_AGES
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Compute age-specific infant template LB modes."
    )
    parser.add_argument("--ages", default="all", help="Comma-separated ages or 'all'")
    parser.add_argument("--subjects-dir", required=True, help="MNE subjects_dir for infant templates")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--K", type=int, default=60, help="Per-hemisphere non-DC modes")
    parser.add_argument("--spacing", default="oct6", help="Source-space spacing")
    parser.add_argument(
        "--surface",
        default="white",
        choices=["white", "pial"],
        help="Surface for LB calculation",
    )
    parser.add_argument("--combine", default="block", choices=["block", "sym", "sym+antisym"])
    parser.add_argument("--lump", action="store_true", help="Use lumped mass matrix in LaPy")
    parser.add_argument(
        "--no-precomputed-src",
        action="store_true",
        help="Force recomputation of src instead of using template-provided src",
    )
    parser.add_argument(
        "--no-normalize-after-restrict",
        action="store_true",
        help="Do not renormalize restricted source-space mode columns",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for age in parse_ages(args.ages):
        phi_src, evals_lh, evals_rh, src, subject, meta = get_phi_infant_template(
            age=age,
            K=args.K,
            subjects_dir=args.subjects_dir,
            spacing=args.spacing,
            surface=args.surface,
            combine=args.combine,
            use_precomputed_src=not args.no_precomputed_src,
            lump=args.lump,
            normalize_after_restrict=not args.no_normalize_after_restrict,
            verbose=True,
        )

        out_file = outdir / f"{subject}_phi_{args.surface}_{args.combine}_K{args.K}_{args.spacing}.npz"

        np.savez_compressed(
            out_file,
            Phi=phi_src,
            evals_lh=evals_lh,
            evals_rh=evals_rh,
            lh_vertno=meta["lh_vertno"],
            rh_vertno=meta["rh_vertno"],
            src_nuse_lh=np.array([meta["src_nuse_lh"]], dtype=int),
            src_nuse_rh=np.array([meta["src_nuse_rh"]], dtype=int),
            phi_colnorms_raw=meta["phi_colnorms_raw"],
            phi_colnorms=meta["phi_colnorms"],
            normalize_after_restrict=np.array(
                [meta["normalize_after_restrict"]], dtype=bool
            ),
            subject=np.array([subject], dtype=object),
            age=np.array([age], dtype=object),
            spacing=np.array([args.spacing], dtype=object),
            surface=np.array([args.surface], dtype=object),
            combine=np.array([args.combine], dtype=object),
            K=np.array([args.K], dtype=int),
        )

        print(f"Saved Phi for {age} ({subject}) -> {out_file}")


if __name__ == "__main__":
    main()