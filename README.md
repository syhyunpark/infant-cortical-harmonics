# infant-cortical-harmonics
Code and files for reproducing the main figures and tables from the paper "Age-dependent spatial transfer of cortical harmonics to scalp EEG in infancy." 
 
The repository is designed to reproduce the core results of the paper. It includes code for reproducing:

1. Age-specific cortical harmonic coordinates and their physical scale
2. Forward-projected scalp gain across harmonic order
3. Developmental basis mismatch in sensor space
4. Neighboring-age cortical harmonic mismatch in basis and coefficient space
5. Local eigenvalue crowding and cross-age instability
6. Cortical-basis versus forward-operator-side contributions to neighboring-age sensor mismatch
7. Sequential Procrustes tracking of neighboring-age harmonic bases

## Repository design

The main scripts expect the following precomputed directories to be present in this folder:

- `out_fullmesh_phi/`
- `out_dict/`
- `out_adult_fullmesh_phi/`
- `out_dict_adult_on_infant/`


These contain: 1) the age-specific full-mesh eigensystems; 2) age-specific forward-projected dictionaries; 3) the adult full-mesh eigensystem; and 4) the adult-on-infant analysis dictionaries used in the manuscript.

## Installation

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

## Download template inputs

The analyses use infant templates distributed through MNE-Python and the fsaverage template. MNE documents fetch_infant_template(age, subjects_dir=...) for the infant templates and fetch_fsaverage(subjects_dir=...) for fsaverage.  

Create a subjects_dir/ folder and download the required templates:

```
python3 - <<'PY'
import mne
from mne.datasets import fetch_infant_template, fetch_fsaverage

subjects_dir = "./subjects_dir"
fetch_fsaverage(subjects_dir=subjects_dir)

ages = [
    "2wk", "1mo", "2mo", "3mo", "4.5mo", "6mo", "7.5mo",
    "9mo", "10.5mo", "12mo", "15mo", "18mo", "2yr"
]

for age in ages:
    fetch_infant_template(age, subjects_dir=subjects_dir)
PY
```

## Generate required precomputed directories

### 1. Infant full-mesh eigensystems
```
python3 compute_infant_fullmesh_lapy.py \
  --ages all \
  --subjects-dir ./subjects_dir \
  --outdir ./out_fullmesh_phi \
  --K 30 \
  --surface white \
  --sphere-surf sphere.reg \
  --keep-largest-component \
  --null-tol 1e-10
```

### 2. Infant forward-projected dictionaries

```
python3 make_infant_lb_dictionary.py \
  --ages all \
  --subjects-dir ./subjects_dir \
  --outdir ./out_dict \
  --K 50 \
  --spacing oct6 \
  --surface white \
  --combine block \
  --montage-kind template1020
```

### 3. Adult full-mesh eigensystem
```
python3 compute_template_fullmesh_lapy.py \
  --subjects-dir ./subjects_dir \
  --subject fsaverage \
  --surface white \
  --sphere-surf sphere.reg \
  --K 50 \
  --outdir ./out_adult_fullmesh_phi
```


### 4. Adult-on-infant analysis dictionaries

```
python3 make_adult_on_infant_lb_dictionary.py \
  --adult-eig ./out_adult_fullmesh_phi/fsaverage_fullmesh_lb_white_sphere.reg_K50.npz \
  --infant-dictdir ./out_dict \
  --subjects-dir ./subjects_dir \
  --target-k 50 \
  --surface white \
  --combine block \
  --montage-kind template1020 \
  --outdir ./out_dict_adult_on_infant
```

## Reproduce results

### Age-specific cortical harmonic coordinates

```bash
python3 plot_mode_physical_scale.py \
  --eigdir ./out_fullmesh_phi \
  --K 30 \
  --outfig ./figures_mode_atlas/mode_physical_scale.png \
  --outcsv ./figures_mode_atlas/mode_physical_scale.csv \
  --modes 2,3,4,5,8,12 \
  --show-normalized


python3 plot_developmental_mode_atlas.py \
  --eigdir ./out_fullmesh_phi \
  --subjects-dir ./subjects_dir \
  --outfig ./figures_mode_atlas/developmental_mode_atlas_lh_full.png \
  --K 30 \
  --hemi lh \
  --figscale 1.3 \
  --modes 2,3,4,5 \
  --ages 2wk,2mo,6mo,2yr \
  --surface pial \
  --sphere-surf sphere.reg \
  --columnwise-scale \
  --dpi 400
```


### Forward-projected scalp gain

```
python3 summarize_transfer_heatmap.py \
  --dictdir ./out_dict \
  --outdir ./figures_transfer \
  --target-k 50 \
  --surface white \
  --combine block \
  --montage-kind template1020 \
  --max-order-plot 25
```

### Neighboring-age sensor mismatch
```
python3 analyze_age_mismatch_sensor.py \
  --outdir ./out_dict \
  --figdir ./figures_age_mismatch_sensor \
  --target-k 50 \
  --analysis-k 5 \
  --surface white \
  --combine block \
  --montage-kind template1020 \
  --prior-kind balanced \
  --n-patterns 1000 \
  --neighbor-only

python3 symmetrize_sensor_mismatch.py \
  --csv ./figures_age_mismatch_sensor/sensor_age_mismatch_summary_K50_AK5_white_block_template1020.csv \
  --outdir ./figures_age_mismatch_sensor_sym
```

### Adult-derived versus infant basis recoverability

```
python3 simulate_recoverability_basis_mismatch.py \
  --gendir ./out_dict \
  --analysisdir ./out_dict \
  --analysis-mode matched \
  --target-k 50 \
  --analysis-k 10 \
  --surface white \
  --combine block \
  --montage-kind template1020 \
  --snr-db-list 20,10,0 \
  --prior-kind equal \
  --n-reps 100 \
  --n-time 600 \
  --pair-hemispheres \
  --outdir ./figures_recoverability_matched

python3 simulate_recoverability_basis_mismatch.py \
  --gendir ./out_dict \
  --analysisdir ./out_dict_adult_on_infant \
  --analysis-mode adult_mismatch \
  --target-k 50 \
  --analysis-k 10 \
  --surface white \
  --combine block \
  --montage-kind template1020 \
  --snr-db-list 20,10,0 \
  --prior-kind equal \
  --n-reps 100 \
  --n-time 600 \
  --pair-hemispheres \
  --outdir ./figures_recoverability_adult_mismatch

python3 compare_recoverability_basis_mismatch.py \
  --matched-csv ./figures_recoverability_matched/recoverability_summary.csv \
  --mismatch-csv ./figures_recoverability_adult_mismatch/recoverability_summary.csv \
  --matched-raw-csv ./figures_recoverability_matched/recoverability_detail.csv \
  --mismatch-raw-csv ./figures_recoverability_adult_mismatch/recoverability_detail.csv \
  --outdir ./figures_recoverability_basis_mismatch \
  --metric mean_corr_first10
```

### Neighboring-age cortical mismatch in basis and coefficient space
```
python3 analyze_age_mismatch_cortical.py \
  --eigdir ./out_fullmesh_phi \
  --outdir ./figures_age_mismatch_cortical \
  --K 30 \
  --neighbor-only \
  --plot-top-k 20

python3 plot_cortical_sym_summary_main3.py \
  --csv ./figures_age_mismatch_cortical/cortical_age_mismatch_symmetrized_K30.csv \
  --outfig ./figures_age_mismatch_cortical/cortical_sym_summary_main3.png

python3 analyze_coefficient_mismatch.py \
  --eigdir-source ./out_fullmesh_phi \
  --outdir ./figures_coeff_mismatch \
  --K 30 \
  --hemi both \
  --neighbor-only \
  --families onehot,packet \
  --packet-width 1.0 \
  --save-heatmaps \
  --plot-top-k 20


python3 symmetrize_coefficient_mismatch.py \
  --csv ./figures_coeff_mismatch/coefficient_mismatch_detail_K30.csv \
  --outdir ./figures_coeff_mismatch_sym \
  --save-hemi-figure
```

### Local eigenvalue crowding
```
python3 analyze_mode_crowding.py \
  --eigdir ./out_fullmesh_phi \
  --outdir ./figures_mode_crowding \
  --K 30 \
  --neighbor-only \
  --nbins 6
```

### Cortical-basis versus forward-operator-side contributions
```
python3 analyze_geometry_head_decomposition.py \
  --dictdir ./out_dict \
  --subjects-dir ./subjects_dir \
  --outdir ./figures_geometry_head_decomp \
  --target-k 50 \
  --analysis-k 5 \
  --surface white \
  --combine block \
  --montage-kind template1020 \
  --prior-kind balanced \
  --n-patterns 1000 \
  --neighbor-only
```

### Sequential Procrustes tracking
```
python3 track_modes_procrustes.py \
  --eigdir ./out_fullmesh_phi \
  --tracked-outdir ./out_fullmesh_phi_tracked \
  --summary-outdir ./figures_mode_tracking \
  --K 30 \
  --block-size 3

# raw basis
python3 analyze_coefficient_mismatch.py \
  --eigdir-source ./out_fullmesh_phi \
  --outdir ./figures_coeff_mismatch_raw \
  --K 30 \
  --hemi both \
  --neighbor-only \
  --families onehot,packet \
  --packet-width 1.0 \
  --plot-top-k 20

python3 symmetrize_coefficient_mismatch.py \
  --csv ./figures_coeff_mismatch_raw/coefficient_mismatch_detail_K30.csv \
  --outdir ./figures_coeff_mismatch_raw_sym

# tracked basis
python3 analyze_coefficient_mismatch.py \
  --eigdir-source ./out_fullmesh_phi_tracked \
  --outdir ./figures_coeff_mismatch_tracked \
  --K 30 \
  --hemi both \
  --neighbor-only \
  --families onehot,packet \
  --packet-width 1.0 \
  --plot-top-k 20

python3 symmetrize_coefficient_mismatch.py \
  --csv ./figures_coeff_mismatch_tracked/coefficient_mismatch_detail_K30.csv \
  --outdir ./figures_coeff_mismatch_tracked_sym

# comparison between tracked vs. original basis
python3 compare_coefficient_mismatch_before_after.py \
  --before-csv ./figures_coeff_mismatch_raw_sym/coefficient_mismatch_detail_K30_symmetrized_avghemi.csv \
  --after-csv ./figures_coeff_mismatch_tracked_sym/coefficient_mismatch_detail_K30_symmetrized_avghemi.csv \
  --outdir ./figures_coeff_mismatch_before_after
```
