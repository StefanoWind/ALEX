# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ALEX (Automated Logging of EXtremes) detects extreme weather events in the AWAKEN dataset — a comprehensive atmospheric dataset collected over a 30×30 km domain in Oklahoma. The goal is to produce a catalogue of weather-driven hazards relevant to energy infrastructure (power lines, wind turbines, etc.).

### Research Phases

1. **Atmospheric driver identification** — correlate power outages with atmospheric signals to determine which variables are most hazardous in this region *(current phase)*
2. **Automated hazard detection** — develop algorithms to detect identified hazards from the atmospheric data
3. **Extreme events catalogue** — compile a systematic record of hazard events

### Phase 1 in Detail

`power_outages.py` identifies which atmospheric variables best predict power outages in the AWAKEN area using data from `data/siteA1_met_outages_15min_v2.nc`:

1. Define target as a binary flag (outage count exceeding a threshold) and predictors as all other atmospheric quantities
2. Plot 2-D histograms of each predictor vs. the target
3. Remove seasonal and diurnal variability from predictors
4. Find the lag and averaging window that maximizes predictor–target correlation
5. Remove collinear features
6. Rank remaining features by Random Forest permutation importance

## Running the Code

```bash
# Real data run
python power_outages.py

# Smoke test with synthetic data
python test.py
```

Each run creates a timestamped output directory (e.g., `data/20240501.123456/`) containing:
- `config.yaml` — snapshot of all parameters used
- CSV tables: `importance_results.csv`, `collinearity_report.csv`, `climatology.csv`
- PNG visualizations for each pipeline stage

## Architecture

The codebase is split into three layers:

| File | Role |
|------|------|
| `utils.py` | Generic pipeline functions and `run_pipeline()` orchestrator |
| `power_outages.py` | Dataset-specific: `load_data()`, `qc_data()`, main block |
| `test.py` | Synthetic smoke test: `generate_synthetic_data()`, main block |
| `configs/power_outages.yaml` | All parameters for the real data run |
| `configs/test.yaml` | Reduced parameters for the synthetic test |

**To add a new dataset:** create a new application script modeled on `power_outages.py` with its own `load_data()`, `qc_data()`, and a `configs/<name>.yaml`. The `run_pipeline()` in `utils.py` requires no changes.

### Pipeline Stages (in `utils.run_pipeline()`)

1. **Target conversion** — binary flag when count exceeds `outage_threshold`
2. **Seasonal detrending** — `remove_seasonal_cycle()`: removes climatological/diurnal cycles; mode `"inplace"` overwrites columns, `"anomaly"` adds `<col>_anom` columns
3. **Cross-lag correlation** — `cross_lag_correlation()`, `select_best_lag()`: sweeps `lag_list` × `window_list` combinations
4. **Feature matrix** — `build_feature_matrix()`: lags + rolling mean/std per predictor
5. **Collinearity check** — `check_collinearity()`: Spearman r + VIF; drops or warns based on `collinearity_action`
6. **Random Forest** — `train_rf_importance()`: StratifiedKFold CV with balanced class weights, permutation importance
7. **SHAP** (optional) — `compute_shap_importance()`: disabled by default
8. **Results** — `build_results_table()`, `plot_importance_comparison()`

`run_pipeline()` creates the timestamped output directory and saves `config.yaml` internally — importing `utils` has no side effects.

### Key Config Parameters

| Key | Effect |
|-----|--------|
| `prelim_rf` | Run a quick OOB RF importance sweep across lags before full pipeline |
| `shap` | Compute SHAP values (slow; requires `shap` package) |
| `collinearity_action` | `"warn"` or `"drop"` redundant features |
| `mode` | `"binary"` (classification) or `"regression"` |
| `detrend_mode` | `"inplace"` or `"anomaly"` |

### Predictors (default)

`wind_speed`, `temperature`, `relative_humidity`, `pressure`, `shortwave_radiation`

## Dependencies

Core: `numpy`, `pandas`, `xarray`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`
Optional: `shap` (for SHAP importance)

No `requirements.txt` exists; install manually as needed.

## Data

Input NetCDF and all outputs are excluded from git (`*.nc`, `*.csv`, `*.png`). Config files in `configs/` are tracked (gitignore exception for `configs/*.yaml`). The NetCDF input must be present at `data/siteA1_met_outages_15min_v2.nc` for a real run.
