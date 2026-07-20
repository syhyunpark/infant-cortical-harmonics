#!/usr/bin/env bash
set -euo pipefail

K=30
EIGDIR="./out_fullmesh_phi"
BLOCK_SIZES=(1 2 3 5)

RAW_DIR="./figures_coeff_mismatch_raw_block_sensitivity"
RAW_SYM_DIR="./figures_coeff_mismatch_raw_block_sensitivity_sym"
RAW_DETAIL_CSV="${RAW_DIR}/coefficient_mismatch_detail_K30.csv"
RAW_SYM_CSV="${RAW_SYM_DIR}/coefficient_mismatch_detail_K30_symmetrized_avghemi.csv"

LOGDIR="./logs_block_sensitivity"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOGFILE}") 2>&1

echo "Starting Sequential Procrustes block-size sensitivity"
echo "Working directory: $(pwd)"
echo "K=${K}"
echo "EIGDIR=${EIGDIR}"
echo "Block sizes: ${BLOCK_SIZES[*]}"
echo

# ---------------------------------------------------------------------
# 1. Raw basis coefficient mismatch
# ---------------------------------------------------------------------

if [ ! -f "${RAW_DETAIL_CSV}" ]; then
  echo "Computing raw-basis coefficient mismatch..."
  python3 analyze_coefficient_mismatch.py \
    --eigdir-source "${EIGDIR}" \
    --outdir "${RAW_DIR}" \
    --K "${K}" \
    --hemi both \
    --neighbor-only \
    --families onehot,packet \
    --packet-width 1.0 \
    --plot-top-k 20
else
  echo "Raw-basis coefficient mismatch already exists."
fi

if [ ! -f "${RAW_SYM_CSV}" ]; then
  echo "Symmetrizing raw-basis coefficient mismatch..."
  python3 symmetrize_coefficient_mismatch.py \
    --csv "${RAW_DETAIL_CSV}" \
    --outdir "${RAW_SYM_DIR}"
else
  echo "Raw-basis symmetrized coefficient mismatch already exists."
fi

# ---------------------------------------------------------------------
# 2. Run tracking and coefficient validation for each block size
# ---------------------------------------------------------------------

for BS in "${BLOCK_SIZES[@]}"; do
  echo
  echo "============================================================"
  echo "Block size ${BS}"
  echo "============================================================"

  TRACKED_DIR="./out_fullmesh_phi_tracked_bs${BS}"
  TRACK_SUMMARY_DIR="./figures_mode_tracking_bs${BS}"
  COEFF_DIR="./figures_coeff_mismatch_tracked_bs${BS}"
  COEFF_SYM_DIR="./figures_coeff_mismatch_tracked_bs${BS}_sym"
  COMPARE_DIR="./figures_coeff_mismatch_before_after_bs${BS}"

  TRACK_CSV="${TRACK_SUMMARY_DIR}/tracking_neighbor_metrics_avghemi_K30.csv"
  COEFF_DETAIL_CSV="${COEFF_DIR}/coefficient_mismatch_detail_K30.csv"
  COEFF_SYM_CSV="${COEFF_SYM_DIR}/coefficient_mismatch_detail_K30_symmetrized_avghemi.csv"
  COMPARE_SUMMARY_CSV="${COMPARE_DIR}/coefficient_retention_before_after_summary.csv"

  if [ ! -f "${TRACK_CSV}" ]; then
    python3 track_modes_procrustes.py \
      --eigdir "${EIGDIR}" \
      --tracked-outdir "${TRACKED_DIR}" \
      --summary-outdir "${TRACK_SUMMARY_DIR}" \
      --K "${K}" \
      --block-size "${BS}"
  else
    echo "Tracking output exists for block size ${BS}."
  fi

  if [ ! -f "${COEFF_DETAIL_CSV}" ]; then
    python3 analyze_coefficient_mismatch.py \
      --eigdir-source "${TRACKED_DIR}" \
      --outdir "${COEFF_DIR}" \
      --K "${K}" \
      --hemi both \
      --neighbor-only \
      --families onehot,packet \
      --packet-width 1.0 \
      --plot-top-k 20
  else
    echo "Coefficient mismatch output exists for block size ${BS}."
  fi

  if [ ! -f "${COEFF_SYM_CSV}" ]; then
    python3 symmetrize_coefficient_mismatch.py \
      --csv "${COEFF_DETAIL_CSV}" \
      --outdir "${COEFF_SYM_DIR}"
  else
    echo "Symmetrized coefficient mismatch exists for block size ${BS}."
  fi

  if [ ! -f "${COMPARE_SUMMARY_CSV}" ]; then
    python3 compare_coefficient_mismatch_before_after.py \
      --before-csv "${RAW_SYM_CSV}" \
      --after-csv "${COEFF_SYM_CSV}" \
      --outdir "${COMPARE_DIR}"
  else
    echo "Before/after comparison exists for block size ${BS}."
  fi
done

# ---------------------------------------------------------------------
# 3. Create compact summary CSV  
# ---------------------------------------------------------------------

python3 - <<'PY'
import pandas as pd
import glob
import re

def block_size_from_path(path):
    m = re.search(r'bs(\d+)', path)
    return int(m.group(1)) if m else None

tracking_rows = [
    pd.read_csv(f).assign(block_size=block_size_from_path(f))
    for f in sorted(glob.glob("./figures_mode_tracking_bs*/tracking_neighbor_metrics_avghemi_K30.csv"))
]

coef_rows = [
    pd.read_csv(f).assign(block_size=block_size_from_path(f))
    for f in sorted(glob.glob("./figures_coeff_mismatch_before_after_bs*/coefficient_retention_before_after_summary.csv"))
]

tracking = pd.concat(tracking_rows, ignore_index=True)
coef = pd.concat(coef_rows, ignore_index=True)

tracking.to_csv("procrustes_block_sensitivity_tracking_all.csv", index=False)
coef.to_csv("procrustes_block_sensitivity_coefficient_all.csv", index=False)

summary_rows = []
for bs, t in tracking.groupby("block_size"):
    c = coef[coef["block_size"] == bs]
    single = c[c["family"] == "single-mode"].iloc[0]
    packet = c[c["family"] == "neighboring-mode"].iloc[0]

    summary_rows.append({
        "block_size": bs,
        "basis_exact_before_median": t["exact_rate_before"].median(),
        "basis_exact_after_median": t["exact_rate_after"].median(),
        "basis_exact_improved_pairs": int((t["exact_rate_after"] > t["exact_rate_before"]).sum()),
        "basis_shift_before_median": t["mean_abs_shift_before"].median(),
        "basis_shift_after_median": t["mean_abs_shift_after"].median(),
        "basis_shift_improved_pairs": int((t["mean_abs_shift_after"] < t["mean_abs_shift_before"]).sum()),
        "n_pairs": len(t),
        "single_exact_before_median": single["median_exact_before"],
        "single_exact_after_median": single["median_exact_after"],
        "single_improved_pairs": int(single["improved_exact_in_pairs"]),
        "packet_exact_before_median": packet["median_exact_before"],
        "packet_exact_after_median": packet["median_exact_after"],
        "packet_improved_pairs": int(packet["improved_exact_in_pairs"]),
    })

summary = pd.DataFrame(summary_rows).sort_values("block_size")
summary.to_csv("procrustes_block_sensitivity_summary.csv", index=False)

print("\nCompact summary:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nSaved:")
print("  procrustes_block_sensitivity_tracking_all.csv")
print("  procrustes_block_sensitivity_coefficient_all.csv")
print("  procrustes_block_sensitivity_summary.csv")
PY

echo
echo "Finished successfully."
echo "Log saved to: ${LOGFILE}"