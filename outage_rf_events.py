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
                   train_rf_importance, train_rf_full, save_event_dataset)

def load_config(path: str = "configs/outage_rf_events.yaml") -> dict:
    '''
    Load configuration
    '''
    with open(path) as f:
        return yaml.safe_load(f)

def load_data(config: dict, source: str = None) -> pd.DataFrame:
    '''
    Load dataset
    '''
    src = source or config["source"]
    ds = xr.open_dataset(src)
    if config.get('weather_event_flag', False):
        cols = config['predictor_cols'] + [config['target_col'], 'weather_event_buffer']
    else:
        cols = config['predictor_cols'] + [config['target_col']]

    if 'aavi' in cols:
        ds['aavi'] = ds['wssd'] / ds['wssd'].mean() * ds['wdsd'] / ds['wdsd'].mean()

    df = ds[cols].to_dataframe()
    df.index = ds.time.values
    ds.close()
    return df


def qc_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    '''
    Quality-control of the signals
    '''
    df_qc = pd.DataFrame(index=df.index)
    for v in config['predictor_cols']:
        df_qc[v] = (df[v]
                    .where(df[v] >= config['limits'][v][0])
                    .where(df[v] <= config['limits'][v][1]))
    df_qc[config['target_col']] = df[config['target_col']]
    if 'weather_event_buffer' in df.columns:
        df_qc['weather_event_buffer'] = df['weather_event_buffer']
    return df_qc


def _load_and_detrend(source_str: str, cfg: dict, base: Path) -> tuple:
    '''
    Detrend atmopsheric data from seasonal and daily patterns
    '''
    source_path = Path(source_str)
    detrended_path = source_path.with_name(source_path.stem + '.detrended.nc')

    df = load_data(cfg, source=source_str)
    df_qc = qc_data(df, cfg)

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
                save_climatology_path=base / f'climatology_{source_path.stem}.csv',
            )
        df_qc.index.name = 'time'
        xr.Dataset.from_dataframe(df_qc).to_netcdf(detrended_path)
        print(f"  Saved detrended cache → {detrended_path}")

    return df_qc, detrended_path, source_path


def _compute_additional_means(event_idx: pd.DatetimeIndex, source_path: Path,
                              cfg: dict, dt) -> pd.DataFrame:
    '''
    
    '''
    additional_cols = cfg.get('additional_cols', [])
    if not additional_cols:
        return pd.DataFrame(index=event_idx)
    ds_raw = xr.open_dataset(str(source_path))
    avail = [c for c in additional_cols if c in ds_raw]
    if not avail:
        ds_raw.close()
        return pd.DataFrame(index=event_idx)
    df_add = ds_raw[avail].to_dataframe()
    df_add.index = pd.DatetimeIndex(ds_raw.time.values)
    ds_raw.close()
    for v in avail:
        lims = cfg.get('limits', {}).get(v)
        if lims:
            df_add[v] = df_add[v].where(df_add[v] >= lims[0]).where(df_add[v] <= lims[1])
    pre_dt  = cfg.get('pre_window',  0) * dt
    post_dt = cfg.get('post_window', 0) * dt
    rows = {ts: df_add.loc[ts - pre_dt : ts + post_dt].mean() for ts in event_idx}
    return pd.DataFrame(rows).T[avail]


def _get_event_matrix(subset: pd.DataFrame, cfg_run: dict,
                      source_path: Path) -> tuple:
    predictors = cfg_run['predictor_cols']
    thr  = cfg_run['outage_threshold']
    dur  = cfg_run.get('min_outage_duration', 1)
    pk   = cfg_run.get('min_peak_customers_out', 0.0)
    pre  = cfg_run.get('pre_window', 0)
    post = cfg_run.get('post_window', 0)
    mode = cfg_run.get('mode', 'binary')
    matrix_dir = source_path.parent / 'event_matrices'
    matrix_dir.mkdir(exist_ok=True)
    matrix_path = matrix_dir / f"{source_path.stem}.{thr}.{dur}.{pk}.{pre}.{post}.{mode}.events_matrix.nc"

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
            mode=mode,
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

    return X, y, episode_ends, peak_customers


def _run_subset(X: pd.DataFrame, y: pd.Series,
                episode_ends: dict, peak_customers: dict,
                cfg_run: dict, out_dir: Path,
                groups_arr: np.ndarray = None,
                detrended_paths: list = None,
                additional_means_df: pd.DataFrame = None):
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg_run, f)

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

    # ── Groups array for grouped CV ───────────────────────────────────
    if groups_arr is not None:
        # Align groups_arr to X after collinearity drop (index unchanged, only columns dropped)
        print(f"  Using grouped CV: {len(np.unique(groups_arr))} groups for {len(X)} events")

    # ── RF ────────────────────────────────────────────────────────────
    print("\n[6] Training RF + permutation importance (CV) ...")
    rf_result = train_rf_importance(X, y, cfg_run, groups=groups_arr)

    print("\n[7] Training RF on full dataset (global importance + SHAP) ...")
    global_rf_result = train_rf_full(X, y, cfg_run)

    oof = rf_result['oof_pred']
    events_df = pd.DataFrame({
        't_end':              pd.DatetimeIndex([episode_ends.get(ts, pd.NaT) for ts in X.index]),
        'target':             y.reindex(X.index).values.astype(float),
        'is_outage':          (y.reindex(X.index).values > 0).astype(int),
        'rf_prediction':      oof.reindex(X.index).values.astype(float),
        'peak_customers_out': [peak_customers.get(ts, np.nan) for ts in X.index],
    }, index=X.index)

    shap_df   = rf_result.get('shap_df',   pd.DataFrame())
    shap_base = rf_result.get('shap_base', None)

    det_attr = ';'.join(str(p) for p in detrended_paths) if detrended_paths else ''

    print("\n[9] Saving event dataset ...")
    save_event_dataset(
        events_df, X, shap_df, cfg_run,
        save_path=out_dir / "events.nc",
        rf_result=rf_result,
        attrs_extra={'detrended_source': det_attr},
        shap_base_value=shap_base,
        global_rf_result=global_rf_result,
        global_shap_base_value=global_rf_result.get('shap_base', None),
        additional_means_df=additional_means_df,
    )
    print("\nPipeline complete.")


#%% Main
if __name__ == "__main__":
    cfg = load_config()

    # Accept source as string (single) or list (multi-county)
    raw_sources = cfg.get('sources', cfg.get('source'))
    sources = [raw_sources] if isinstance(raw_sources, str) else list(raw_sources)
    station_names = [Path(s).name.split('.')[0] for s in sources]

    #filesystem
    ts_run = datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
    base = Path(cfg['output_dir']) / ts_run
    os.makedirs(base, exist_ok=True)

    cfg_run = copy.deepcopy(cfg)
    cfg_run['detrend_seasonal'] = False

    # ── Load, QC, detrend each source independently ───────────────────
    loaded = [_load_and_detrend(s, cfg, base) for s in sources]

    # ── Build event matrices per source, add county indicator, pool ───
    X_parts, y_parts, county_rows = [], [], []
    episode_ends, peak_customers = {}, {}
    episode_ends_per_county = []
    detrended_paths = []
    add_means_parts = []

    dt = loaded[0][0].index.to_series().diff().median()

    for county_i, (df_qc, detrended_path, source_path) in enumerate(loaded):
        if cfg.get('mode') in ('AUC', 'TOT'):
            df_qc = df_qc.copy()
            df_qc['__target__'] = make_segment_target(
                df_qc[cfg['target_col']], cfg['outage_threshold'], cfg['mode']
            )

        print(f"\n── Source {county_i+1}/{len(sources)}: {source_path.name} ──")
        X_i, y_i, ep_i, pk_i = _get_event_matrix(df_qc, cfg_run, source_path)

        X_i = X_i.copy()
        X_i['county'] = county_i          # binary county indicator [Roberts et al., 2017, Ecography]

        X_parts.append(X_i)
        y_parts.append(y_i)
        county_rows.extend([county_i] * len(X_i))
        episode_ends.update(ep_i)
        episode_ends_per_county.append(ep_i)
        peak_customers.update(pk_i)
        detrended_paths.append(detrended_path)
        add_means_parts.append(_compute_additional_means(X_i.index, source_path, cfg, dt))

    X = pd.concat(X_parts)
    y = pd.concat(y_parts)
    additional_means_df = (pd.concat(add_means_parts) if add_means_parts
                           else pd.DataFrame(index=X.index))

    # ── Compute event group IDs via union-find on the overlap graph ───
    # [Roberts et al., 2017, Ecography]
    pre_dt  = cfg_run.get('pre_window', 0) * dt
    post_dt = cfg_run.get('post_window', 0) * dt

    parent = {}

    def _uf_find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = _uf_find(parent[x])
        return parent[x]

    def _uf_union(x, y):
        px, py = _uf_find(x), _uf_find(y)
        if px != py:
            parent[px] = py

    for ci, ep in enumerate(episode_ends_per_county):
        for ts in ep:
            _uf_find((ci, ts))

    for ci in range(len(sources)):
        for cj in range(ci + 1, len(sources)):
            for ts_i, te_i in episode_ends_per_county[ci].items():
                for ts_j, te_j in episode_ends_per_county[cj].items():
                    if (te_i + post_dt > ts_j - pre_dt and
                            ts_i - pre_dt < te_j + post_dt):
                        _uf_union((ci, ts_i), (cj, ts_j))

    root_to_id, next_id = {}, 0
    outage_group = {}
    for ci, ep in enumerate(episode_ends_per_county):
        for ts in ep:
            root = _uf_find((ci, ts))
            if root not in root_to_id:
                root_to_id[root] = next_id
                next_id += 1
            outage_group[(ci, ts)] = root_to_id[root]

    singleton = next_id
    groups_list = []
    for ci, ts in zip(county_rows, X.index):
        key = (ci, pd.Timestamp(ts))
        if key in outage_group:
            groups_list.append(outage_group[key])
        else:
            groups_list.append(singleton)
            singleton += 1
    groups_arr = np.array(groups_list)
    print(f"  Event groups: {next_id} cross-county groups, "
          f"{len(np.unique(groups_arr))} total for {len(X)} events")

    # ── weather_event_flag subsetting (single-source only) ────────────
    if cfg.get('weather_event_flag', False):
        if len(sources) > 1:
            raise ValueError("weather_event_flag subsetting is not supported with multiple sources.")
        df_qc_single = loaded[0][0]
        for flag, label in [(True, 'NWS_true'), (None, 'all')]:
            if flag is None:
                sub = df_qc_single.drop(columns=['weather_event_buffer'])
            else:
                sub = (df_qc_single[df_qc_single['weather_event_buffer'] == flag]
                       .drop(columns=['weather_event_buffer']))
            X_s, y_s, ep_s, pk_s = _get_event_matrix(sub, cfg_run, loaded[0][2])
            X_s = X_s.copy()
            X_s['county'] = 0
            add_means_s = _compute_additional_means(X_s.index, loaded[0][2], cfg, dt)
            _run_subset(X_s, y_s, ep_s, pk_s, cfg_run, base / label,
                        groups_arr=groups_arr, detrended_paths=detrended_paths,
                        additional_means_df=add_means_s)
    else:
        _run_subset(X, y, episode_ends, peak_customers, cfg_run, base / 'all',
                    groups_arr=groups_arr, detrended_paths=detrended_paths,
                    additional_means_df=additional_means_df)
