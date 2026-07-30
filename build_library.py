'''
Builds a per-window event library: outage probability, predictor features,
SHAP attributions, and outage statistics (peak customers out, duration, AUC).
One row per non-overlapping evaluation window.
'''

import re
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import tkinter
import tkinter.filedialog
import pandas as pd
from utils import OutagePredictor, load_data, qc_data, build_event_matrix, load_turbine_points

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print('Warning: shap not installed; SHAP columns will be omitted.')

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source_data_files = tkinter.filedialog.askopenfilenames(
    title='Select atmospheric + outage data NetCDF (one or more sites)',
    initialdir='./data',
    filetypes=[('NetCDF files', '*.nc')],
)
if not source_data_files:
    print('No data file selected. Exiting.')
    sys.exit()

source_model = tkinter.filedialog.askdirectory(
    title='Select model folder',
    initialdir='./',
)
if not source_model:
    print('No model folder selected. Exiting.')
    sys.exit()

source_clim = tkinter.filedialog.askopenfilename(
    title='Select climatology CSV',
    initialdir='./data',
    filetypes=[('CSV files', '*.csv')],
)
if not source_clim:
    print('No climatology file selected. Exiting.')
    sys.exit()

AVAIL_FILE = Path(__file__).parent / 'data' / 'awaken_data_availability.csv'

#%% Turbine locations (shared across all sites)
turbine_df = load_turbine_points(Path(__file__).parent / 'map_data')
print(f"Turbine locations: {len(turbine_df)} turbines across {turbine_df['farm'].nunique() if not turbine_df.empty else 0} farms")


def _read_attr(ds, keys, default=np.nan):
    for key in keys:
        if key in ds.attrs:
            try:
                return float(ds.attrs[key])
            except (TypeError, ValueError):
                continue
    return default

#%% Load model (shared across all sites)
predictor = OutagePredictor(source_model)
cfg = predictor.cfg

#%% Load HRRR grid (shared across all sites; nearest point is extracted per site below)
_hrrr_files = sorted((Path(__file__).parent / 'data' / 'hrrr').glob('*.nc'))
ds_hrrr = None
_fcst_sfx = None
if _hrrr_files:
    ds_hrrr = xr.open_mfdataset(_hrrr_files, combine='nested', concat_dim='time',
                                  coords='minimal', compat='override')

    for _hv in ds_hrrr.data_vars:
        _m = re.search(r'(_f\d+)$', _hv)
        if _m:
            _fcst_sfx = _m.group(1)
            break

    # Unit conversions — analysis
    ds_hrrr['wspd'] = (ds_hrrr['u10']**2 + ds_hrrr['v10']**2)**0.5
    ds_hrrr['pres'] = ds_hrrr['pres'] / 100        # Pa → hPa
    ds_hrrr['tair'] = ds_hrrr['tair'] - 273.15     # K → °C

    # Unit conversions — forecast
    if _fcst_sfx:
        if f'u10{_fcst_sfx}' in ds_hrrr and f'v10{_fcst_sfx}' in ds_hrrr:
            ds_hrrr[f'wspd{_fcst_sfx}'] = (ds_hrrr[f'u10{_fcst_sfx}']**2
                                            + ds_hrrr[f'v10{_fcst_sfx}']**2) ** 0.5
        for _base, _op in (('pres', lambda x: x / 100), ('tair', lambda x: x - 273.15)):
            _fv = f'{_base}{_fcst_sfx}'
            if _fv in ds_hrrr:
                ds_hrrr[_fv] = _op(ds_hrrr[_fv])

    _anl_base = ('wspd', 'pres', 'srad', 'relh', 'tair', 'gust', 'prate')
    _anl_vars = [v for v in _anl_base if v in ds_hrrr]
    _fct_vars = ([f'{v}{_fcst_sfx}' for v in _anl_vars if f'{v}{_fcst_sfx}' in ds_hrrr]
                 if _fcst_sfx else [])
    _hrrr_keep = _anl_vars + _fct_vars
    _lat2d = ds_hrrr['latitude'].values
    _lon2d = ds_hrrr['longitude'].values
else:
    print("Warning: no HRRR files found in data/hrrr/ — HRRR sheet and RMSE will be omitted")

#%% Main — process each site independently, reusing the model/climatology/HRRR grid
for source_data in source_data_files:
    print(f"\n=== Processing {source_data} ===")

    #%% Load
    df = load_data(cfg, source=source_data)
    df_qc = qc_data(df, cfg)

    with xr.open_dataset(source_data) as _ds_meta0:
        _src_lat = _read_attr(_ds_meta0, ['latitude', 'lat', 'station_lat'], default=np.nan)
        _src_lon = _read_attr(_ds_meta0, ['longitude', 'lon', 'station_lon'], default=np.nan)%360
        # wdir is display-only (dashboard wind-direction arrows), not a model
        # predictor, so it's pulled straight from the source file rather than
        # through load_data()/qc_data() (which only keep predictor_cols).
        wdir = (_ds_meta0['wdir'].to_dataframe()['wdir']
                if 'wdir' in _ds_meta0.data_vars else pd.Series(dtype=float))

    #%% Nearest HRRR grid point for this site
    hrrr_pt = pd.DataFrame()
    if ds_hrrr is not None:
        _sta_lat = float(_src_lat) if np.isfinite(_src_lat) else 36.412010
        _sta_lon = float(_src_lon) if np.isfinite(_src_lon) else (360 - 97.693940)
        _dist  = np.hypot(_lat2d - _sta_lat, _lon2d - _sta_lon)
        _iy, _ix = np.unravel_index(np.argmin(_dist), _dist.shape)
        print(f"HRRR grid point: lat={_lat2d[_iy,_ix]:.4f}  lon={_lon2d[_iy,_ix]:.4f}")
        hrrr_pt = ds_hrrr[_hrrr_keep].isel(y=_iy, x=_ix).load().to_dataframe()[_hrrr_keep]
        print(f"HRRR loaded: {len(hrrr_pt)} hourly steps, "
              f"anl={_anl_vars}, fct={_fct_vars}")

    #%% Bias-correct HRRR against met observations (removes median[HRRR - met] per signal)
    if not hrrr_pt.empty:
        _meso_h_bias = df_qc.copy()
        _meso_h_bias.index = df_qc.index + pd.Timedelta(minutes=7.5)
        _meso_h_bias = _meso_h_bias.resample('1h').nearest()
        for v in hrrr_pt.columns:
            met_v = v[:-len(_fcst_sfx)] if (_fcst_sfx and v.endswith(_fcst_sfx)) else v
            if met_v not in _meso_h_bias.columns:
                continue
            _both = pd.concat([hrrr_pt[v].rename('h'), _meso_h_bias[met_v].rename('m')],
                               axis=1).dropna()
            if len(_both) < 2:
                continue
            _bias = float((_both['h'] - _both['m']).median())
            hrrr_pt[v] = hrrr_pt[v] - _bias
            print(f"  Bias correction [{v}]: median(HRRR - met) = {_bias:+.3f} (removed)")

    #%% Predict, features, detrended data
    probs, feat_df, df_det = predictor.predict(
        df_qc, climatology_path=source_clim, non_overlapping=True, return_features=True,
    )
    print(f"Windows: {len(probs)}  |  valid: {probs.notna().sum()}")

    #%% SHAP [Lundberg & Lee, 2017]
    shap_base_val = np.nan
    if HAS_SHAP:
        valid_mask = feat_df.notna().all(axis=1)
        shap_arr = np.full((len(feat_df), feat_df.shape[1]), np.nan)
        if valid_mask.any():
            explainer = shap_lib.TreeExplainer(predictor.rf)
            sv = explainer.shap_values(feat_df.loc[valid_mask].values)
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            shap_arr[valid_mask.values] = sv
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                shap_base_val = float(base[1]) if len(base) > 1 else float(base[0])
            else:
                shap_base_val = float(base)
        shap_df = pd.DataFrame(
            shap_arr, index=feat_df.index,
            columns=[f'shap_{c}' for c in feat_df.columns],
        )

    #%% Dimensional (non-detrended) window features
    # Same rolling aggregation as in predict() but applied to QC'd raw values
    _pre_w    = cfg['pre_window']
    _post_w   = cfg['post_window']
    _w_label  = _pre_w + _post_w
    _roll     = _pre_w + _post_w          # half-open right: [t-pre, t+post)
    _frames_raw = {}
    for _col in cfg['predictor_cols']:
        _x  = df_qc[_col]
        _g  = _x.diff()
        _r  = _x.rolling(_roll)
        _gr = _g.rolling(_roll)
        _frames_raw[f'{_col}_mean_W{_w_label}_raw']      = _r.mean().shift(1 - _post_w)
        _frames_raw[f'{_col}_std_W{_w_label}_raw']        = _r.std().shift(1 - _post_w)
        _frames_raw[f'{_col}_grad_mean_W{_w_label}_raw']  = _gr.mean().shift(1 - _post_w)
        _frames_raw[f'{_col}_grad_std_W{_w_label}_raw']   = _gr.std().shift(1 - _post_w)
    feat_raw_df = pd.DataFrame(_frames_raw, index=df_qc.index).reindex(feat_df.index)

    #%% Outage episodes — same significance criteria as training
    raw_target = df[cfg['target_col']]
    dt = raw_target.index.to_series().diff().median()

    _, y_auc, episode_ends, peak_customers = build_event_matrix(
        df_det[cfg['predictor_cols']],
        raw_target,
        threshold=cfg['outage_threshold'],
        mode='AUC',
        pre_window=cfg['pre_window'],
        post_window=cfg['post_window'],
        min_duration=cfg.get('min_outage_duration', 1),
        min_auc=cfg.get('min_event_auc', 0.0),
        min_peak=cfg.get('min_peak_customers_out', 0.0),
    )

    # Match each episode start to the nearest prediction window [Wilks, 2011]
    pre_w  = cfg['pre_window']
    post_w = cfg['post_window']
    ep_idx = pd.DatetimeIndex(list(episode_ends.keys()))

    stats = pd.DataFrame(
        {'peak_customers_out': np.nan, 'duration_h': np.nan, 'auc_customer_h': np.nan},
        index=probs.index, dtype=float,
    )
    if len(ep_idx) > 0:
        matched_pos = probs.index.get_indexer(ep_idx, method='nearest')
        t_centers   = probs.index[matched_pos]
        offsets     = ep_idx - t_centers
        in_window   = (offsets >= -pre_w * dt) & (offsets < post_w * dt)
        for t0, pos, ok in zip(ep_idx, matched_pos, in_window):
            if not ok:
                continue
            t1 = episode_ends[t0]
            stats.loc[probs.index[pos], 'peak_customers_out'] = peak_customers[t0]
            stats.loc[probs.index[pos], 'duration_h']         = (t1 - t0 + dt).total_seconds() / 3600
            stats.loc[probs.index[pos], 'auc_customer_h']     = float(y_auc.loc[t0])

    print(f"Outage episodes matched: {int(stats['peak_customers_out'].notna().sum())} / {len(episode_ends)}")

    #%% HRRR–mesonet RMSE per event window
    _RMSE_VARS = ('wspd', 'pres', 'srad', 'relh', 'tair')
    _anl_rmse  = [v for v in _RMSE_VARS if not hrrr_pt.empty and v in hrrr_pt.columns and v in df_qc.columns]
    _fct_rmse  = ([f'{v}{_fcst_sfx}' for v in _anl_rmse if f'{v}{_fcst_sfx}' in hrrr_pt.columns]
                  if _fcst_sfx else [])
    _all_rmse  = _anl_rmse + _fct_rmse
    rmse_df = pd.DataFrame({f'rmse_{v}': np.nan for v in _all_rmse},
                           index=probs.index, dtype=float)

    if not hrrr_pt.empty:
        _meso_shifted = df_qc.copy()
        _meso_shifted.index = df_qc.index + pd.Timedelta(minutes=7.5)
        for t in probs.index:
            w_start = t - pre_w * dt
            w_end   = t + post_w * dt - dt
            hrrr_w  = hrrr_pt.loc[w_start:w_end]
            met_w   = _meso_shifted.loc[w_start:w_end].resample('1h').nearest()
            for v in _all_rmse:
                met_v = v[:-len(_fcst_sfx)] if (_fcst_sfx and v.endswith(_fcst_sfx)) else v
                both = pd.concat([hrrr_w[v].rename('h'), met_w[met_v].rename('m')],
                                 axis=1).dropna()
                if len(both) < 2:
                    continue
                rmse_df.loc[t, f'rmse_{v}'] = float(np.sqrt(((both['h'] - both['m'])**2).mean()))

    #%% WDH data availability
    avail_raw = pd.read_csv(AVAIL_FILE) if AVAIL_FILE.exists() else pd.DataFrame(columns=['channel', 'date_time'])
    channels = avail_raw['channel'].unique().tolist() if not avail_raw.empty else []
    avail = pd.DataFrame(False, index=probs.index, columns=channels, dtype=bool)

    if avail_raw.empty:
        print(f"Warning: {AVAIL_FILE} not found — run awaken_data_availability.py first")
    else:
        avail_raw['date_time'] = pd.to_datetime(avail_raw['date_time'],
                                                 format='%Y%m%d%H%M%S', errors='coerce')
        ch_timestamps = {ch: pd.DatetimeIndex(grp['date_time'].dropna())
                         for ch, grp in avail_raw.groupby('channel')}
        for ch in channels:
            ts = ch_timestamps[ch]
            for t in probs.index:
                avail.loc[t, ch] = bool(((ts >= t - pre_w * dt) & (ts < t + post_w * dt)).any())

    #%% Assemble and save
    frames = [probs.rename('outage_probability'), feat_df, feat_raw_df, stats, rmse_df]
    if HAS_SHAP:
        frames.insert(2, shap_df)
    out = pd.concat(frames, axis=1)
    out = out[out['outage_probability'].notna()]
    out = out.apply(pd.to_numeric, errors='coerce').astype(float)
    out.index.name = 'timestamp'

    out_path = Path(source_data).with_suffix('.library.xlsx')

    # Library sheet: prediction table with metadata rows (string index for compatibility)
    out_str = out.copy()
    out_str.index = out_str.index.strftime('%Y-%m-%d %H:%M:%S')
    out_str.index.name = 'timestamp'

    meta = pd.DataFrame(
        [[shap_base_val] + [np.nan] * (len(out.columns) - 1),
         [float(cfg['outage_threshold'])] + [np.nan] * (len(out.columns) - 1)],
        index=pd.Index(['shap_base_value', 'outage_threshold'], name='timestamp'),
        columns=out.columns,
    )

    # Site sheet: station geolocation, kept separate from the Library meta rows
    # so the dashboard can read it directly without matching a companion .nc file
    site_df = pd.DataFrame({
        'latitude':  [float(_src_lat) if np.isfinite(_src_lat) else np.nan],
        'longitude': [float(_src_lon) if np.isfinite(_src_lon) else np.nan],
    })

    # Data sheet: full raw QC'd + detrended predictor time series for dashboard plots
    _pred_cols = cfg['predictor_cols']
    _target_col = cfg['target_col']
    _raw_part = df_qc[[c for c in _pred_cols + [_target_col] if c in df_qc.columns]]
    _det_part = df_det[[c for c in _pred_cols if c in df_det.columns]].rename(
        columns={c: c + '_det' for c in _pred_cols if c in df_det.columns}
    )
    data_sheet = pd.concat([_raw_part, _det_part], axis=1)
    if not wdir.empty:
        data_sheet['wdir'] = wdir.reindex(data_sheet.index)
    data_str = data_sheet.copy()
    data_str.index = data_str.index.strftime('%Y-%m-%d %H:%M:%S')
    data_str.index.name = 'timestamp'

    avail_str = avail.copy()
    avail_str.index = avail_str.index.strftime('%Y-%m-%d %H:%M:%S')
    avail_str.index.name = 'timestamp'

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        pd.concat([out_str, meta]).to_excel(writer, sheet_name='Library')
        site_df.to_excel(writer, sheet_name='Site', index=False)
        turbine_df.to_excel(writer, sheet_name='Wind farms', index=False)
        data_str.to_excel(writer, sheet_name='Data')
        avail_str.to_excel(writer, sheet_name='wdh_avail')
        if not hrrr_pt.empty:
            hrrr_str = hrrr_pt.copy()
            hrrr_str.index = hrrr_str.index.strftime('%Y-%m-%d %H:%M:%S')
            hrrr_str.index.name = 'timestamp'
            hrrr_str.to_excel(writer, sheet_name='HRRR data')

    print(f"Saved → {out_path}  "
        f"(Library: {len(out)} rows + 4 metadata, "
          f"Data: {len(data_sheet)} rows × {data_sheet.shape[1]} cols)")

if ds_hrrr is not None:
    ds_hrrr.close()
