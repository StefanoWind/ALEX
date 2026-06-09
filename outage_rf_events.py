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
import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from datetime import datetime
from pathlib import Path
from utils import (remove_seasonal_cycle, make_segment_target,
                   build_event_matrix, check_collinearity,
                   train_rf_importance, save_event_dataset)


def load_config(path: str = "configs/outage_rf_events.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    ds = xr.open_dataset(config["source"])
    if cfg.get('weather_event_flag', False):
        cols = config['predictor_cols'] + [config['target_col'], 'weather_event_buffer']
    else:
        cols = config['predictor_cols'] + [config['target_col']]

    if 'TURB' in cols:
        ds['TURB'] = ds['WSSD'] / (10**-10 + ds['WSPD']) * 100
    if 'aavi' in cols:
        ds['aavi'] = ds['wssd'] / ds['wssd'].mean() * ds['wdsd'] / ds['wdsd'].mean()

    df = ds[cols].to_dataframe()
    df.index = ds.time.values
    return df


def qc_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df_qc = pd.DataFrame(index=df.index)
    for v in config['predictor_cols']:
        df_qc[v] = (df[v]
                    .where(df[v] >= config['limits'][v][0])
                    .where(df[v] <= config['limits'][v][1]))
    df_qc[config['target_col']] = df[config['target_col']]
    if 'weather_event_buffer' in df.columns:
        df_qc['weather_event_buffer'] = df['weather_event_buffer']
    return df_qc


def _run_subset(subset: pd.DataFrame, df_raw_subset: pd.DataFrame,
                cfg_run: dict, out_dir: Path,
                detrended_path: Path, source_path: Path):
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg_run, f)

    predictors = cfg_run['predictor_cols']

    # ── Event matrix cache ────────────────────────────────────────────
    thr  = cfg_run['outage_threshold']
    dur  = cfg_run.get('min_outage_duration', 1)
    pk   = cfg_run.get('min_peak_customers_out', 0.0)
    pre  = cfg_run.get('pre_window', 0)
    post = cfg_run.get('post_window', 0)
    matrix_path = source_path.with_suffix('').with_suffix('').with_name(
        f"{source_path.stem}.{thr}.{dur}.{pk}.{pre}.{post}.events_matrix.nc"
    )

    if matrix_path.exists():
        print(f"  Loading event matrix cache ← {matrix_path}")
        ds_m = xr.open_dataset(matrix_path)
        X = pd.DataFrame(ds_m['features'].values,
                         index=pd.DatetimeIndex(ds_m.event.values),
                         columns=list(ds_m.feature.values))
        y = pd.Series(ds_m['target'].values,
                      index=pd.DatetimeIndex(ds_m.event.values), name='__target__')
        is_out = ds_m['is_outage'].values.astype(bool)
        t_ends = ds_m['t_end'].values
        episode_ends = {pd.Timestamp(ts): pd.Timestamp(te)
                        for ts, te, flag in zip(ds_m.event.values, t_ends, is_out)
                        if flag and not pd.isnull(te)}
        pk_vals = ds_m['peak_customers_out'].values
        peak_customers = {pd.Timestamp(ts): float(pv)
                          for ts, pv, flag in zip(ds_m.event.values, pk_vals, is_out)
                          if flag}
        ds_m.close()
    else:
        print(f"\n[4] Building event-level feature matrix "
              f"(pre_window={pre}, post_window={post}) ...")
        X, y, episode_ends, peak_customers = build_event_matrix(
            subset[predictors],
            subset[cfg_run['target_col']],
            threshold=thr,
            mode=cfg_run['mode'],
            pre_window=pre,
            post_window=post,
            min_duration=dur,
            min_auc=cfg_run.get('min_event_auc', 0.0),
            min_peak=pk,
            non_outages_ratio=cfg_run.get('non_outages_ratio', None),
            random_state=cfg_run.get('random_state', None),
        )
        print(f"    Feature matrix: {X.shape[0]} events x {X.shape[1]} features")

        t_end_vals = pd.DatetimeIndex([episode_ends.get(ts, pd.NaT) for ts in X.index])
        ds_m = xr.Dataset(
            {
                'features':           (['event', 'feature'], X.values.astype(np.float32)),
                'target':             (['event'], y.values.astype(np.float32)),
                'is_outage':          (['event'], (y.values > 0).astype(np.int8)),
                't_end':              (['event'], t_end_vals.values),
                'peak_customers_out': (['event'], np.array(
                    [peak_customers.get(ts, np.nan) for ts in X.index],
                    dtype=np.float32)),
            },
            coords={'event': X.index, 'feature': list(X.columns)},
        )
        ds_m.to_netcdf(matrix_path)
        print(f"  Saved event matrix cache → {matrix_path}")

    # ── Collinearity ──────────────────────────────────────────────────
    print("\n[5] Checking feature collinearity ...")
    X_clean, dropped_features, collinearity_report = check_collinearity(
        X,
        r_threshold=cfg_run["collinearity_r_threshold"],
        vif_threshold=cfg_run["collinearity_vif_threshold"],
        action=cfg_run["collinearity_action"],
        save_path=out_dir / "collinearity_heatmap.png",
    )
    collinearity_report.to_csv(out_dir / "collinearity_report.csv")
    if dropped_features:
        print(f"    Proceeding with {X_clean.shape[1]} features "
              f"(dropped {len(dropped_features)}: {dropped_features})")
    else:
        print(f"    No features dropped; proceeding with {X_clean.shape[1]} features.")
    X = X[list(X_clean.columns)]

    # ── RF ────────────────────────────────────────────────────────────
    print("\n[6] Training RF + permutation importance (CV) ...")
    rf_result = train_rf_importance(X, y, cfg_run)

    oof = rf_result['oof_pred']
    events_df = pd.DataFrame({
        't_end':              pd.DatetimeIndex([episode_ends.get(ts, pd.NaT) for ts in X.index]),
        'target':             y.reindex(X.index).values.astype(float),
        'is_outage':          (y.reindex(X.index).values > 0).astype(int),
        'rf_prediction':      oof.reindex(X.index).values.astype(float),
        'peak_customers_out': [peak_customers.get(ts, np.nan) for ts in X.index],
    }, index=X.index)

    # ── SHAP ──────────────────────────────────────────────────────────
    shap_df   = rf_result.get('shap_df',   pd.DataFrame())
    shap_base = rf_result.get('shap_base', None)
    shap_imp  = rf_result.get('shap_imp',  pd.Series(dtype=float))

    # ── Save events.nc ────────────────────────────────────────────────
    print("\n[9] Saving event dataset ...")
    save_event_dataset(
        events_df, X, shap_df, cfg_run,
        save_path=out_dir / "events.nc",
        rf_result=rf_result,
        attrs_extra={'detrended_source': str(detrended_path)},
        shap_base_value=shap_base,
    )
    print("\nPipeline complete.")


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    df_qc = qc_data(df, cfg)
    df_raw = df_qc.copy()

    source_path = Path(cfg['source'])
    detrended_path = source_path.with_suffix('').with_name(
        source_path.stem + '.detrended.nc'
    )

    ts = datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
    base = Path(cfg['output_dir']) / ts
    os.makedirs(base, exist_ok=True)

    # ── Detrended cache ───────────────────────────────────────────────
    if detrended_path.exists():
        print(f"  Loading detrended cache ← {detrended_path}")
        ds_det = xr.open_dataset(detrended_path)
        df_qc = ds_det.to_dataframe()
        df_qc.index = pd.DatetimeIndex(ds_det.time.values)
        ds_det.close()
    else:
        if cfg.get('detrend_seasonal', False):
            inplace = cfg.get('detrend_mode', 'anomaly') == 'inplace'
            exclude = set(cfg.get('detrend_exclude', []))
            detrend_cols = [c for c in cfg['predictor_cols'] if c not in exclude]
            df_qc, _ = remove_seasonal_cycle(
                df_qc,
                columns=detrend_cols,
                window_days=cfg.get('detrend_window_days', 7),
                min_periods=cfg.get('detrend_min_periods', 3),
                inplace=inplace,
                save_climatology_path=base / 'climatology.csv',
            )
        df_qc.index.name = 'time'
        xr.Dataset.from_dataframe(df_qc).to_netcdf(detrended_path)
        print(f"  Saved detrended cache → {detrended_path}")

    if cfg.get('mode') in ('AUC', 'TOT'):
        df_qc['__target__'] = make_segment_target(
            df_qc[cfg['target_col']], cfg['outage_threshold'], cfg['mode']
        )

    cfg_run = copy.deepcopy(cfg)
    cfg_run['detrend_seasonal'] = False

    if cfg.get('weather_event_flag', False):
        for flag, label in [(True, 'NWS_true'), (None, 'all')]:
            if flag is None:
                subset = df_qc.drop(columns=['weather_event_buffer'])
            else:
                subset = (df_qc[df_qc['weather_event_buffer'] == flag]
                          .drop(columns=['weather_event_buffer']))
            _run_subset(subset, None, cfg_run, base / label,
                        detrended_path, source_path)
    else:
        subset = df_qc
        label = 'all'
        _run_subset(subset, None, cfg_run, base / label,
                    detrended_path, source_path)
