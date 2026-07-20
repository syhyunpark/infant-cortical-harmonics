#!/usr/bin/env bash
set -euo pipefail

TARGET_K=50
DICTDIR="./out_dict"
ANALYSIS_KS=(3 5 7)

SURFACE="white"
COMBINE="block"
MONTAGE="template1020"
PRIOR="balanced"
N_PATTERNS=1000

LOGDIR="./logs_sensor_mismatch_sensitivity"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOGFILE}") 2>&1

echo "Starting sensor-space mismatch sensitivity analysis"
echo "Working directory: $(pwd)"
echo "TARGET_K=${TARGET_K}"
echo "ANALYSIS_KS=${ANALYSIS_KS[*]}"
echo "DICTDIR=${DICTDIR}"
echo

for AK in "${ANALYSIS_KS[@]}"; do

  echo
  echo "============================================================"
  echo "Running analysis-k ${AK} modes per hemisphere"
  echo "============================================================"

  FIGDIR="./figures_age_mismatch_sensor_AK${AK}"
  SYMDIR="./figures_age_mismatch_sensor_AK${AK}_sym"

  RAW_CSV="${FIGDIR}/sensor_age_mismatch_summary_K${TARGET_K}_AK${AK}_${SURFACE}_${COMBINE}_${MONTAGE}.csv"
  SYM_CSV="${SYMDIR}/sensor_age_mismatch_summary_K${TARGET_K}_AK${AK}_${SURFACE}_${COMBINE}_${MONTAGE}_symmetrized.csv"

  if [ ! -f "${RAW_CSV}" ]; then
    python3 analyze_age_mismatch_sensor.py \
      --outdir "${DICTDIR}" \
      --figdir "${FIGDIR}" \
      --target-k "${TARGET_K}" \
      --analysis-k "${AK}" \
      --surface "${SURFACE}" \
      --combine "${COMBINE}" \
      --montage-kind "${MONTAGE}" \
      --prior-kind "${PRIOR}" \
      --n-patterns "${N_PATTERNS}" \
      --neighbor-only
  else
    echo "Raw sensor mismatch CSV already exists for AK=${AK}: ${RAW_CSV}"
  fi

  if [ ! -f "${SYM_CSV}" ]; then
    python3 symmetrize_sensor_mismatch.py \
      --csv "${RAW_CSV}" \
      --outdir "${SYMDIR}"
  else
    echo "Symmetrized sensor mismatch CSV already exists for AK=${AK}: ${SYM_CSV}"
  fi

done

echo
echo "============================================================"
echo "Combining sensitivity-analysis outputs"
echo "============================================================"

python3 - <<'PY'
import glob
import re
from pathlib import Path
import pandas as pd


def analysis_k_from_path(path):
    m = re.search(r"AK(\d+)", path)
    return int(m.group(1)) if m else None


def pick_col(cols, candidates, contains_all=None):
    cols_lower = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    if contains_all is not None:
        for c in cols:
            low = c.lower()
            if all(s in low for s in contains_all):
                return c

    return None


files = sorted(glob.glob("./figures_age_mismatch_sensor_AK*_sym/*symmetrized*.csv"))

if not files:
    raise RuntimeError("No symmetrized sensor mismatch CSV files found.")

all_rows = [
    pd.read_csv(f).assign(analysis_k=analysis_k_from_path(f), total_modes=lambda df: 2 * analysis_k_from_path(f))
    for f in files
]

all_df = pd.concat(all_rows, ignore_index=True)
all_df.to_csv("sensor_mismatch_AK_sensitivity_all.csv", index=False)

cols = list(all_df.columns)
print("\nDetected columns:")
print(cols)

gap_col = pick_col(cols, ["age_gap_months", "gap_months"], contains_all=["gap"])
angle_col = pick_col(
    cols,
    ["principal_angle_deg", "max_principal_angle_deg", "principal_angle_max_deg", "angle_deg"],
    contains_all=["angle"]
)
cross_r2_col = pick_col(
    cols,
    ["cross_r2", "cross_r2_mean", "mean_cross_r2", "r2_cross", "r2_cross_mean"],
    contains_all=["cross", "r2"]
)
delta_r2_col = pick_col(
    cols,
    ["delta_r2", "delta_r2_mean", "mean_delta_r2", "r2_drop", "drop_r2"],
    contains_all=["r2"]
)

if delta_r2_col is not None:
    possible_delta = [c for c in cols if ("delta" in c.lower() or "drop" in c.lower()) and "r2" in c.lower()]
    if possible_delta:
        delta_r2_col = possible_delta[0]

projector_col = pick_col(
    cols,
    ["projector_distance", "proj_dist", "projector_dist"],
    contains_all=["projector"]
)

print("\nSelected columns:")
print("  gap_col      =", gap_col)
print("  angle_col    =", angle_col)
print("  cross_r2_col =", cross_r2_col)
print("  delta_r2_col =", delta_r2_col)
print("  projector_col=", projector_col)

summary_rows = []
for ak, d in all_df.groupby("analysis_k"):
    row = {
        "analysis_k_per_hemi": ak,
        "total_modes": 2 * ak,
        "n_pairs": len(d),
    }

    if angle_col is not None:
        row["median_angle"] = d[angle_col].median()
        row["min_angle"] = d[angle_col].min()
        row["max_angle"] = d[angle_col].max()

    if cross_r2_col is not None:
        row["median_cross_r2"] = d[cross_r2_col].median()
        row["min_cross_r2"] = d[cross_r2_col].min()
        row["max_cross_r2"] = d[cross_r2_col].max()

    if delta_r2_col is not None:
        row["median_delta_r2"] = d[delta_r2_col].median()
        row["min_delta_r2"] = d[delta_r2_col].min()
        row["max_delta_r2"] = d[delta_r2_col].max()

    if projector_col is not None:
        row["median_projector_distance"] = d[projector_col].median()

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).sort_values("analysis_k_per_hemi")
summary.to_csv("sensor_mismatch_AK_sensitivity_summary.csv", index=False)

print("\nCompact summary:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nSaved:")
print("  sensor_mismatch_AK_sensitivity_all.csv")
print("  sensor_mismatch_AK_sensitivity_summary.csv")
PY

echo
echo "Finished sensor-space mismatch sensitivity analysis"
echo "Log saved to: ${LOGFILE}"