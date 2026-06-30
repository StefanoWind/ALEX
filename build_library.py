'''
Builds a per-window event library: outage probability, predictor features,
SHAP attributions, and outage statistics (peak customers out, duration, AUC).
One row per non-overlapping evaluation window.
'''

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tkinter
import tkinter.filedialog
from utils import OutagePredictor, load_data, qc_data, build_event_matrix

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
source_data = tkinter.filedialog.askopenfilename(
    title='Select atmospheric + outage data NetCDF',
    initialdir='./data',
    filetypes=[('NetCDF files', '*.nc')],
)
if not source_data:
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

#%% Load
predictor = OutagePredictor(source_model)
cfg = predictor.cfg

df = load_data(cfg, source=source_data)
df_qc = qc_data(df, cfg)

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
_roll     = _pre_w + _post_w + 1
_frames_raw = {}
for _col in cfg['predictor_cols']:
    _x  = df_qc[_col]
    _g  = _x.diff()
    _r  = _x.rolling(_roll)
    _gr = _g.rolling(_roll)
    _frames_raw[f'{_col}_mean_W{_w_label}_raw']      = _r.mean().shift(-_post_w)
    _frames_raw[f'{_col}_std_W{_w_label}_raw']        = _r.std().shift(-_post_w)
    _frames_raw[f'{_col}_grad_mean_W{_w_label}_raw']  = _gr.mean().shift(-_post_w)
    _frames_raw[f'{_col}_grad_std_W{_w_label}_raw']   = _gr.std().shift(-_post_w)
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
    in_window   = (offsets >= -pre_w * dt) & (offsets <= post_w * dt)
    for t0, pos, ok in zip(ep_idx, matched_pos, in_window):
        if not ok:
            continue
        t1 = episode_ends[t0]
        stats.loc[probs.index[pos], 'peak_customers_out'] = peak_customers[t0]
        stats.loc[probs.index[pos], 'duration_h']         = (t1 - t0 + dt).total_seconds() / 3600
        stats.loc[probs.index[pos], 'auc_customer_h']     = float(y_auc.loc[t0])

print(f"Outage episodes matched: {int(stats['peak_customers_out'].notna().sum())} / {len(episode_ends)}")

#%% Assemble and save
frames = [probs.rename('outage_probability'), feat_df, feat_raw_df, stats]
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

# Data sheet: full raw QC'd + detrended predictor time series for dashboard plots
_pred_cols = cfg['predictor_cols']
_target_col = cfg['target_col']
_raw_part = df_qc[[c for c in _pred_cols + [_target_col] if c in df_qc.columns]]
_det_part = df_det[[c for c in _pred_cols if c in df_det.columns]].rename(
    columns={c: c + '_det' for c in _pred_cols if c in df_det.columns}
)
data_sheet = pd.concat([_raw_part, _det_part], axis=1)
data_str = data_sheet.copy()
data_str.index = data_str.index.strftime('%Y-%m-%d %H:%M:%S')
data_str.index.name = 'timestamp'

with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    pd.concat([out_str, meta]).to_excel(writer, sheet_name='Library')
    data_str.to_excel(writer, sheet_name='Data')

print(f"Saved → {out_path}  "
      f"(Library: {len(out)} rows + 2 metadata, "
      f"Data: {len(data_sheet)} rows × {data_sheet.shape[1]} cols)")
