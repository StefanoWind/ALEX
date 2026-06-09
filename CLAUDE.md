# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ALEX (Automated Logging of EXtremes) detects extreme weather events in the AWAKEN dataset — a comprehensive atmospheric dataset collected over a 30×30 km domain in Oklahoma. The goal is to produce a catalogue of weather-driven hazards relevant to energy infrastructure (power lines, wind turbines, etc.).

### Research Phases

1. **Atmospheric driver identification** — correlate power outages with atmospheric signals to determine which variables are most hazardous in this region *(complete)*
2. **Automated hazard detection** — develop algorithms to detect identified hazards from the atmospheric data *(current phase)*
3. **Extreme events catalogue** — compile a systematic record of hazard events

### Phase 1 in Detail

`outage_rf_events.py` identifies which atmospheric variables best predict power outages using data from `data/brec.outages_mesonet.nc`:

1. Define a list of large outage events
2. Select a window around the outage start and extract atmopsheric signals as predictors
3. Remove seasonal and diurnal variability from predictors
4. Calculate dynamic features for each predictor as windowed statistics
5. Remove collinear features
6. Rank remaining features by Random Forest permutation importance
7. Explain Random Forest prediction thorugh SHAP

## Running the Code

```bash
python outage_rf_events.py
```

Each run creates a timestamped output directory under `results/` containing:
- `outage_rf_events.yaml` — snapshot of all parameters used
- CSV tables and PNG visualizations for each pipeline stage

## Architecture

The codebase is split into three layers:

| File | Role |
|------|------|
| `utils.py` | Generic pipeline functions and `run_pipeline()` orchestrator |
| `outage_rf_events.py` | Dataset-specific: `load_data()`, `qc_data()`, main block |
| `configs/outage_rf_events.yaml` | All parameters for the run |

**To add a new dataset:** create a new application script modeled on `outage_rf_events.py` with its own `load_data()`, `qc_data()`, and a `configs/<name>.yaml`. The `run_pipeline()` in `utils.py` requires no changes.

### Pipeline Stages (in `utils.run_pipeline()`)

1. **Target conversion** — binary flag when count exceeds `outage_threshold`
2. **Seasonal detrending** — `remove_seasonal_cycle()`: removes climatological/diurnal cycles; mode `"inplace"` overwrites columns, `"anomaly"` adds `<col>_anom` columns
3. **Feature matrix** — `build_feature_matrix()`: lags + rolling mean/std per predictor
4. **Collinearity check** — `check_collinearity()`: Spearman r + VIF; drops or warns based on `collinearity_action`
5. **Random Forest** — `train_rf_importance()`: StratifiedKFold CV with balanced class weights, permutation importance
6. **SHAP** (optional) — `compute_shap_importance()`: disabled by default
7. **Results** — `build_results_table()`, `plot_importance_comparison()`

`run_pipeline()` creates the timestamped output directory and saves `config.yaml` internally — importing `utils` has no side effects.

### Key Config Parameters

| Key | Effect |
|-----|--------|
| `weather_event_flag` | Restrict analysis to timesteps with an active NWS weather event flag |
| `shap` | Compute SHAP values (slow; requires `shap` package) |
| `shap_clustering_cutoff` | Minimum mean absolute SHAP value to include a feature in clustering |
| `collinearity_action` | `"warn"` or `"drop"` redundant features |
| `mode` | Target aggregation mode (e.g. `TOT` for total customers out) |
| `detrend_mode` | `"inplace"` or `"anomaly"` |

### Predictors (default)

`relh` (relative humidity), `tair` (air temperature), `aavi` (wind speed), `wmax` (wind gust), `rain`, `pres` (pressure), `srad` (shortwave radiation)

## Phase 2 Design — Outage Probability and Event Taxonomy

### Scientific narrative

Phase 1 (RF regression + SHAP) identified which atmospheric variables drive power outages. Phase 2 produces a catalogue with two outputs per timestep: **outage probability** and **event type**. These are computed by two separate models and joined at output — they are not co-trained.

```
Mesonet data
    ├── Step 1: RF classifier → outage probability
    ├── Step 2: SHAP ranking → relevant channel identification
    └── Step 3: Clustering on SHAP-selected channels → event type

Output table: [timestamp, outage_probability, event_type]
```

### Step 1 — RF Classifier for Outage Probability

Convert the Phase 1 RF regressor to a **binary RF classifier** predicting whether an outage occurs. Wrap in `Fix t` (isotonic regression, nested inside StratifiedKFold) to produce well-calibrated probabilities. Evaluate with Brier score and reliability diagram — raw RF probabilities are overconfident near 0 and 1 and must be calibrated for catalogue use.

### Step 2 — SHAP Ranking for Channel Identification

Run SHAP on the trained RF classifier to rank which meteorological channels carry the most predictive signal. Current Phase 1 findings point to: sustained high winds, wind speed peaks, precipitation, extreme temperatures, and high wind variability. This step selects the channels for clustering — SHAP is used for *feature selection*, not as clustering coordinates.

### Step 3 — Event Taxonomy Clustering

Cluster outage-associated events using the **meteorological feature vectors** of the SHAP-selected channels (not SHAP values). Feature vectors are summary descriptors per event window: mean, peak, ramp rate, onset steepness, duration of peak. Clustering method: k-means or SOM on this meteorological feature space (synoptic-climatology tradition).

Key design choices:
- Cluster on detrended anomalies + temporal shape descriptors (ramp rate, time-to-peak) — these distinguish sustained wind from wind ramp from thunderstorm outflow
- NWS event flags serve as **external validation** to anchor cluster naming, not as training labels
- SHAP attribution patterns are checked post-hoc for consistency with cluster membership

### Validation gate (mandatory before claiming success)

1. Calibrated RF probability must show reliability (Brier score, reliability diagram) comparable to Phase 1 RF performance
2. Clusters must be meteorologically nameable by composite analysis and broadly consistent with NWS flag co-occurrence
3. Top-K detected events must correspond to recognizable synoptic features (front, dryline, MCS, wind ramp)

### Planned file structure

| File | Role |
|------|------|
| `outage_rf_events.py` | RF classifier + SHAP ranking (extends Phase 1) |
| `event_catalogue.py` | Event taxonomy clustering and catalogue output |
| `configs/event_catalogue.yaml` | Parameters for clustering step |

Reuse from `utils.py`: `remove_seasonal_cycle()`, `build_feature_matrix()`.

## Dependencies

Core: `numpy`, `pandas`, `xarray`, `scipy`, `scikit-learn`, `matplotlib`, `pyyaml`
Optional: `shap` (for SHAP importance)

No `requirements.txt` exists; install manually as needed.

## Data

Input NetCDF and all outputs are excluded from git (`*.nc`, `*.csv`, `*.png`). Config files in `configs/` are tracked (gitignore exception for `configs/*.yaml`). The NetCDF input must be present at `data/brec.outages_mesonet.nc` for a real run.
