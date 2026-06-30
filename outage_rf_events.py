# Event-based approach: one row per outage episode (or non-outage day).
# Deseasonalization and lag optimization are performed on the full 15-min
# time series; features are then aggregated over each event window before
# training the Random Forest.
#
# Motivation: avoids within-episode temporal autocorrelation present in the
# series approach, and naturally handles class imbalance in a regression
# framework (non-outage days have target = 0).
#
# References:
# [Wanik et al., 2015, IEEE Trans. Power Del.] — event-based outage prediction
# [Cerrai et al., 2019, Weather Climate Extremes] — ice storm impact models
# [Prahl et al., 2015, Nat. Haz. Earth Sys. Sci.] — wind damage event framing
# [Ploton et al., 2020, Nat. Commun.] — autocorrelation inflation in ML eval.
# [Boulesteix et al., 2012, WIREs Data Mining] — RF performance with small N

import copy
import itertools
import os
import joblib
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from utils import (load_config, _load_and_detrend, _compute_additional_means,
                   _get_event_matrix, _run_pipeline, _build_groups_arr,
                   _total_duration_h, plot_sensitivity)

#%% Inputs
# Scalar → single run (saves events.nc for post-processing).
# List   → sensitivity sweep (saves CSV + diagnostic plots, no events.nc).
OUTAGE_THRESHOLDS       = 20
MIN_OUTAGE_DURATIONS    = 24  # timesteps (1 ts = 15 min)
MIN_PEAK_CUSTOMERS_OUTS = 100
MODE = 'serial'  # 'serial' | 'parallel'

#%% Initialization
if __name__ == "__main__":
    cfg = load_config()

    raw_sources = cfg.get('sources', cfg.get('source'))
    sources = [raw_sources] if isinstance(raw_sources, str) else list(raw_sources)
    station_names = [Path(s).name.split('.')[0] for s in sources]

    ts_run = datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
    base = Path(cfg['output_dir']) / ts_run
    os.makedirs(base, exist_ok=True)

    loaded = [_load_and_detrend(s, cfg, base) for s in sources]
    dt      = loaded[0][0].index.to_series().diff().median()
    pre_dt  = cfg.get('pre_window',  0) * dt
    post_dt = cfg.get('post_window', 0) * dt

    def _pool(cfg_local):
        """Build pooled event matrix, grouped CV array, and per-run metadata for all sources."""
        X_parts, y_parts, county_rows = [], [], []
        ep_per_county, ep_ends, pk_ends = [], {}, {}
        det_paths, add_parts, dur_h = [], [], 0.0
        for ci, (df_qc, det_path, src_path) in enumerate(loaded):
            X_i, y_i, ep_i, pk_i = _get_event_matrix(df_qc, cfg_local, src_path)
            X_i = X_i.copy()
            X_i['county'] = ci
            X_parts.append(X_i);  y_parts.append(y_i)
            county_rows.extend([ci] * len(X_i))
            ep_per_county.append(ep_i)
            ep_ends.update(ep_i);  pk_ends.update(pk_i)
            det_paths.append(det_path)
            add_parts.append(_compute_additional_means(X_i.index, src_path, cfg_local, dt))
            dur_h += _total_duration_h(ep_i, dt)
        X = pd.concat(X_parts)
        y = pd.concat(y_parts)
        groups = _build_groups_arr(ep_per_county, county_rows, X.index, pre_dt, post_dt)
        add_df = pd.concat(add_parts) if add_parts else pd.DataFrame(index=X.index)
        return X, y, groups, ep_ends, pk_ends, det_paths, add_df, dur_h

    sensitivity_mode = any(isinstance(p, (list, np.ndarray))
                           for p in [OUTAGE_THRESHOLDS, MIN_OUTAGE_DURATIONS,
                                     MIN_PEAK_CUSTOMERS_OUTS])

    #%% Main
    if sensitivity_mode:
        thresholds = list(OUTAGE_THRESHOLDS)       if not isinstance(OUTAGE_THRESHOLDS,       list) else OUTAGE_THRESHOLDS
        durations  = list(MIN_OUTAGE_DURATIONS)    if not isinstance(MIN_OUTAGE_DURATIONS,    list) else MIN_OUTAGE_DURATIONS
        peaks      = list(MIN_PEAK_CUSTOMERS_OUTS) if not isinstance(MIN_PEAK_CUSTOMERS_OUTS, list) else MIN_PEAK_CUSTOMERS_OUTS

        sens_dir = base / 'sensitivity'
        os.makedirs(sens_dir, exist_ok=True)
        with open(sens_dir / 'sensitivity_grid.yaml', 'w') as f:
            yaml.dump({'OUTAGE_THRESHOLDS': thresholds,
                       'MIN_OUTAGE_DURATIONS': durations,
                       'MIN_PEAK_CUSTOMERS_OUTS': peaks}, f)

        # Baseline: total outage hours with no filtering
        MIN_OUTAGES = 10
        cfg_base = copy.deepcopy(cfg)
        cfg_base.update({'outage_threshold': MIN_OUTAGES, 'min_outage_duration': 0,
                         'min_peak_customers_out': 0, 'shap': False, 'importance_reps': 1})
        print("Computing baseline outage duration ...")
        *_, dur_baseline_h = _pool(cfg_base)
        print(f"  dur_baseline = {dur_baseline_h:.1f} h\n")

        def _run_combo(idx, total, thr, dur, peak, n_jobs):
            print(f"[{idx}/{total}]  threshold={thr}  min_dur={dur}  min_peak={peak}")
            cfg_c = copy.deepcopy(cfg)
            cfg_c.update({'outage_threshold': thr, 'min_outage_duration': dur,
                          'min_peak_customers_out': peak, 'shap': False,
                          'importance_reps': 1, 'n_jobs': n_jobs})
            X, y, groups, *_, dur_h = _pool(cfg_c)
            n_out = int((y > 0).sum())
            n_non = int((y == 0).sum())
            cov   = dur_h / dur_baseline_h if dur_baseline_h > 0 else np.nan
            rec   = {'outage_threshold': thr, 'min_outage_duration': dur,
                     'min_peak_customers_out': peak, 'n_outage': n_out,
                     'n_total': len(y), 'coverage': cov}
            if n_out < cfg_c['n_cv_folds'] or n_non < cfg_c['n_cv_folds']:
                print(f"  [{idx}] Skipped: too few samples (outage={n_out}, non-outage={n_non})")
                return {**rec, 'brier_score': np.nan, 'roc_auc': np.nan}
            metrics = _run_pipeline(X, y, cfg_c, groups_arr=groups)
            print(f"  [{idx}] Brier={metrics['brier_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
                  f"coverage={cov:.3f}  n_outage={n_out}  n_total={len(y)}")
            return {**rec, **metrics}

        combos = list(itertools.product(thresholds, durations, peaks))
        print(f"Running {len(combos)} configurations ({MODE}) ...\n")
        if MODE == 'parallel':
            from joblib import Parallel, delayed
            records = Parallel(n_jobs=-1)(
                delayed(_run_combo)(i, len(combos), thr, dur, peak, 1)
                for i, (thr, dur, peak) in enumerate(combos, 1))
        else:
            records = [_run_combo(i, len(combos), thr, dur, peak, cfg.get('n_jobs', -1))
                       for i, (thr, dur, peak) in enumerate(combos, 1)]

        df_res = pd.DataFrame(records)
        df_res.to_csv(sens_dir / 'sensitivity_results.csv', index=False)
        print(f"Saved → {sens_dir / 'sensitivity_results.csv'}")

        valid = df_res.dropna(subset=['brier_score', 'coverage'])
        if not valid.empty:
            b = valid.nsmallest(1, 'brier_score').iloc[0]
            print(f"\nBest (lowest Brier):   thr={int(b['outage_threshold'])}  "
                  f"dur={int(b['min_outage_duration'])}  peak={int(b['min_peak_customers_out'])}  "
                  f"Brier={b['brier_score']:.4f}  coverage={b['coverage']:.3f}")
            vc = valid[valid['coverage'] > 0].copy()
            vc['composite'] = vc['brier_score'] / vc['coverage']
            c = vc.nsmallest(1, 'composite').iloc[0]
            print(f"Best (Brier/coverage): thr={int(c['outage_threshold'])}  "
                  f"dur={int(c['min_outage_duration'])}  peak={int(c['min_peak_customers_out'])}  "
                  f"composite={c['composite']:.4f}  Brier={c['brier_score']:.4f}  "
                  f"coverage={c['coverage']:.3f}")

        plot_sensitivity(df_res, thresholds, durations, peaks, sens_dir)
        print(f"\nAll sensitivity outputs in {sens_dir}")

    else:
        cfg['outage_threshold']       = OUTAGE_THRESHOLDS
        cfg['min_outage_duration']    = MIN_OUTAGE_DURATIONS
        cfg['min_peak_customers_out'] = MIN_PEAK_CUSTOMERS_OUTS

        X, y, groups, ep_ends, pk_ends, det_paths, add_df, _ = _pool(cfg)
        print(f"  Event groups: {len(np.unique(groups))} total for {len(X)} events")

        _run_pipeline(X, y, cfg, groups_arr=groups, out_dir=base,
                      ep_ends=ep_ends, pk_ends=pk_ends,
                      det_paths=det_paths, add_df=add_df)

        # Save model bundle
        bundle_dir = base/Path('model_bundle')
        os.makedirs(bundle_dir, exist_ok=True)
        detrend_cols = [c for c in cfg['predictor_cols']
                        if c not in set(cfg.get('detrend_exclude', []))]

        X_bundle = X.drop(columns=['county'], errors='ignore')
        y_bin = (y > 0).astype(float)
        rf_base = RandomForestClassifier(
            n_estimators=cfg['n_estimators'],
            max_features=cfg['max_features'],
            n_jobs=cfg['n_jobs'],
            random_state=cfg['random_state'],
        )
        sw = compute_sample_weight('balanced', y_bin.values)
        # [Niculescu-Mizil & Caruana, 2005, ICML; Zadrozny & Elkan, 2002, KDD]
        model = CalibratedClassifierCV(rf_base, method='isotonic', cv=cfg['n_cv_folds'])
        model.fit(X_bundle.values, y_bin.values, sample_weight=sw)
        joblib.dump(model, bundle_dir / 'model.joblib')
        print(f"  Saved → {bundle_dir / 'model.joblib'}")

        bundle_cfg = {
            'predictor_cols':      cfg['predictor_cols'],
            'detrend_cols':        detrend_cols,
            'pre_window':          cfg.get('pre_window', 8),
            'post_window':         cfg.get('post_window', 8),
            'feature_names':       list(X_bundle.columns),
            'station_names':       station_names,
            'detrend_window_days': cfg.get('detrend_window_days', 7),
            'detrend_min_periods': cfg.get('detrend_min_periods', 3),
        }
        with open(bundle_dir / 'config.yaml', 'w') as f:
            yaml.dump(bundle_cfg, f)
        print(f"  Saved → {bundle_dir / 'config.yaml'}")
        print(f"\nBundle ready in {bundle_dir}/")
