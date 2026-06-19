# Trains the final calibrated RF on the full pooled dataset (all counties)
# using the best config selected from sensitivity_analysis.py, then writes
# a self-contained inference bundle to BUNDLE_DIR.
#
# Bundle contents:
#   model.joblib                  CalibratedClassifierCV (RF + isotonic calibration)
#   climatology_<station>.csv     seasonal/diurnal climatology per source station
#   config.yaml                   feature-construction parameters consumed by predict.py
#
# Run once after sensitivity_analysis.py identifies the best setup.

import copy
import os
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

from outage_rf_events import load_data, qc_data, _get_event_matrix
from utils import remove_seasonal_cycle

# ── Edit these after sensitivity_analysis.py ──────────────────────────────
BEST_CONFIG_OVERRIDES = {
    'outage_threshold':       37,
    'min_outage_duration':    16,
    'min_peak_customers_out': 200,
}

BUNDLE_DIR = Path('model_bundle')

# ── Load base config ───────────────────────────────────────────────────────
with open('configs/outage_rf_events.yaml') as f:
    cfg = yaml.safe_load(f)

cfg.update(BEST_CONFIG_OVERRIDES)
cfg['shap'] = False

raw_sources = cfg.get('sources', cfg.get('source'))
sources = [raw_sources] if isinstance(raw_sources, str) else list(raw_sources)
station_names = [Path(s).name.split('.')[0] for s in sources]

os.makedirs(BUNDLE_DIR, exist_ok=True)

detrend_cols = [c for c in cfg['predictor_cols']
                if c not in set(cfg.get('detrend_exclude', []))]
inplace_mode = cfg.get('detrend_mode', 'anomaly') == 'inplace'

# cfg_run passed to _get_event_matrix must have detrend_seasonal=False
# (detrending is already applied to df_det before calling it)
cfg_run = copy.deepcopy(cfg)
cfg_run['detrend_seasonal'] = False

# ── Per-station: load → QC → detrend → save climatology → build events ────
X_parts, y_parts = [], []

for station, src in zip(station_names, sources):
    print(f"\n── {station} ──")
    df    = load_data(cfg, source=src)
    df_qc = qc_data(df, cfg)

    clim_path = BUNDLE_DIR / f'climatology_{station}.csv'
    df_det, _ = remove_seasonal_cycle(
        df_qc,
        columns=detrend_cols,
        window_days=cfg.get('detrend_window_days', 7),
        min_periods=cfg.get('detrend_min_periods', 3),
        inplace=inplace_mode,
        save_climatology_path=clim_path,
    )
    print(f"  Saved climatology → {clim_path}")

    X_i, y_i, _, _ = _get_event_matrix(df_det, cfg_run, Path(src))
    X_parts.append(X_i)
    y_parts.append(y_i)

# Drop county indicator — not meaningful for prediction on arbitrary new data
X = pd.concat(X_parts).drop(columns=['county'], errors='ignore')
y = pd.concat(y_parts)
y_bin = (y > 0).astype(float)

n_out = int(y_bin.sum())
n_non = int((y_bin == 0).sum())
print(f"\nFull dataset: {n_out} outage + {n_non} non-outage = {len(y)} total")

# ── Train final calibrated RF on all data ─────────────────────────────────
# CalibratedClassifierCV (cv folds) fits RF on each train fold and calibrates
# isotonic regression on the held-out fold predictions, using all available data.
# [Niculescu-Mizil & Caruana, 2005, ICML; Zadrozny & Elkan, 2002, KDD]
print("\nTraining calibrated RF ...")
rf_base = RandomForestClassifier(
    n_estimators=cfg['n_estimators'],
    max_features=cfg['max_features'],
    n_jobs=cfg['n_jobs'],
    random_state=cfg['random_state'],
)
sw = compute_sample_weight('balanced', y_bin.values)
model = CalibratedClassifierCV(rf_base, method='isotonic', cv=cfg['n_cv_folds'])
model.fit(X.values, y_bin.values, sample_weight=sw)

# ── Save bundle ────────────────────────────────────────────────────────────
model_path = BUNDLE_DIR / 'model.joblib'
joblib.dump(model, model_path)
print(f"Saved → {model_path}")

bundle_cfg = {
    'predictor_cols': cfg['predictor_cols'],
    'detrend_cols':   detrend_cols,
    'pre_window':     cfg.get('pre_window', 8),
    'post_window':    cfg.get('post_window', 8),
    'feature_names':  list(X.columns),
    'station_names':  station_names,
}
with open(BUNDLE_DIR / 'config.yaml', 'w') as f:
    yaml.dump(bundle_cfg, f)
print(f"Saved → {BUNDLE_DIR / 'config.yaml'}")

print(f"\nBundle ready in {BUNDLE_DIR}/")
for station in station_names:
    print(f"  climatology_{station}.csv")
print(f"  config.yaml")
print(f"  model.joblib")
