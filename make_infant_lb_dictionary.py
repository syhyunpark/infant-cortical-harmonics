# make_infant_lb_dictionary.py
# Build age-specific EEG forward models and forward-projected dictionaries D = G @ Phi
# for MNE infant templates.

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

from infant_lb_common import (
    DEFAULT_AGES,
    extract_eeg_xyz,
    fetch_infant_subject,
    get_phi_infant_template,
    load_template_src_bem,
    make_info_for_standard_montage,
    make_info_for_template1020,
    read_canonical_list,
)


def parse_ages(text: str):
    text = text.strip()
    if text.lower() == "all":
        return DEFAULT_AGES
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Make age-specific infant EEG LB dictionaries D = G @ Phi."
    )
    parser.add_argument("--ages", default="all", help="Comma-separated ages or 'all'")
    parser.add_argument("--subjects-dir", required=True, help="MNE subjects_dir for infant templates")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--K", type=int, default=60, help="Per-hemisphere non-DC modes")
    parser.add_argument("--spacing", default="oct6", help="Source-space spacing")
    parser.add_argument("--surface", default="white", choices=["white", "pial"])
    parser.add_argument("--combine", default="block", choices=["block", "sym", "sym+antisym"])
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--mindist", type=float, default=5.0)
    parser.add_argument(
        "--montage-kind",
        default="template1020",
        choices=["template1020", "standard"],
        help="Primary recommendation for this paper: template1020",
    )
    parser.add_argument(
        "--standard-montage",
        default="standard_1005",
        help="Used only if --montage-kind standard",
    )
    parser.add_argument(
        "--canonical-file",
        default=None,
        help="Required if --montage-kind standard; optional otherwise",
    )
    parser.add_argument("--no-precomputed-src", action="store_true")
    parser.add_argument("--no-precomputed-bem", action="store_true")
    parser.add_argument(
        "--no-normalize-after-restrict",
        action="store_true",
        help="Do not renormalize restricted source-space mode columns",
    )
    args = parser.parse_args()

    if args.montage_kind == "standard" and args.canonical_file is None:
        raise ValueError("--canonical-file is required when --montage-kind standard")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    standard_channels = None
    if args.canonical_file is not None:
        standard_channels = read_canonical_list(args.canonical_file)

    for age in parse_ages(args.ages):
        subject, subjects_dir = fetch_infant_subject(age, args.subjects_dir, verbose=True)

        src, bem = load_template_src_bem(
            subject=subject,
            subjects_dir=subjects_dir,
            spacing=args.spacing,
            use_precomputed_src=not args.no_precomputed_src,
            use_precomputed_bem=not args.no_precomputed_bem,
            verbose=True,
        )

        if args.montage_kind == "template1020":
            info, trans, _ = make_info_for_template1020(
                subject=subject,
                subjects_dir=subjects_dir,
                sfreq=args.sfreq,
            )
        else:
            _, trans, _ = make_info_for_template1020(
                subject=subject,
                subjects_dir=subjects_dir,
                sfreq=args.sfreq,
            )
            info = make_info_for_standard_montage(
                ch_names=standard_channels,
                sfreq=args.sfreq,
                montage_name=args.standard_montage,
            )

        ch_names = info.ch_names
        montage_ch_names, montage_xyz = extract_eeg_xyz(info)
        trans_mat = trans["trans"].copy()

        fwd = mne.make_forward_solution(
            info=info,
            trans=trans,
            src=src,
            bem=bem,
            meg=False,
            eeg=True,
            mindist=args.mindist,
            n_jobs=1,
            verbose=True,
        )
        fwd_fixed = mne.convert_forward_solution(
            fwd,
            surf_ori=True,
            force_fixed=True,
            verbose=False,
        )

        G = fwd_fixed["sol"]["data"]
        src_fwd = fwd_fixed["src"]

        phi_src, evals_lh, evals_rh, _, _, meta = get_phi_infant_template(
            age=age,
            K=args.K,
            subjects_dir=subjects_dir,
            spacing=args.spacing,
            surface=args.surface,
            combine=args.combine,
            src=src_fwd,
            use_precomputed_src=not args.no_precomputed_src,
            lump=False,
            normalize_after_restrict=not args.no_normalize_after_restrict,
            verbose=True,
        )

        if G.shape[1] != phi_src.shape[0]:
            raise RuntimeError(
                f"Leadfield columns ({G.shape[1]}) do not match Phi rows ({phi_src.shape[0]})."
            )

        D = G @ phi_src
        d_colnorms = np.linalg.norm(D, axis=0)

        out_file = outdir / (
            f"{subject}_D_{args.surface}_{args.combine}_K{args.K}_"
            f"M{len(ch_names)}_{args.spacing}_{args.montage_kind}.npz"
        )

        np.savez_compressed(
            out_file,
            D=D,
            G=G,
            Phi=phi_src,
            evals_lh=evals_lh,
            evals_rh=evals_rh,
            lh_vertno=meta["lh_vertno"],
            rh_vertno=meta["rh_vertno"],
            src_nuse_lh=np.array([meta["src_nuse_lh"]], dtype=int),
            src_nuse_rh=np.array([meta["src_nuse_rh"]], dtype=int),
            phi_colnorms_raw=meta["phi_colnorms_raw"],
            phi_colnorms=meta["phi_colnorms"],
            d_colnorms=d_colnorms,
            channels=np.array(ch_names, dtype=object),
            montage_ch_names=montage_ch_names,
            montage_xyz=montage_xyz,
            trans_mat=trans_mat,
            subject=np.array([subject], dtype=object),
            age=np.array([age], dtype=object),
            spacing=np.array([args.spacing], dtype=object),
            surface=np.array([args.surface], dtype=object),
            combine=np.array([args.combine], dtype=object),
            montage_kind=np.array([args.montage_kind], dtype=object),
            normalize_after_restrict=np.array(
                [meta["normalize_after_restrict"]], dtype=bool
            ),
            K=np.array([args.K], dtype=int),
            mindist=np.array([args.mindist], dtype=float),
        )

        print(f"Saved dictionary for {age} ({subject}) -> {out_file}")


if __name__ == "__main__":
    main()