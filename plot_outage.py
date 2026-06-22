import re
import sys
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import tkinter
import tkinter.filedialog
from utils import plot_episode_ts, plot_shap_waterfall, _VAR_LABELS

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
plt.close('all')

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

# ── Args ───────────────────────────────────────────────────────────────────

source = tkinter.filedialog.askopenfilename(
    title='Select events.nc',
    initialdir='./results/',
    filetypes=[('NetCDF files', '*.nc')],
)


if not source:
    print('No file selected. Exiting.')
    sys.exit()
source=Path(source)

date_str=input('Date [Y-m-d H:M:S]: ')

# ── Load events.nc ─────────────────────────────────────────────────────────
ds               = xr.open_dataset(source)
out_dir          = source.parent
events           = pd.DatetimeIndex(ds.event.values)
feat_names       = list(ds.feature.values)
feat_arr         = ds['features'].values.astype(float)
target           = ds['target'].values.astype(float)
is_outage        = ds['is_outage'].values.astype(bool)
rf_pred          = ds['rf_prediction'].values.astype(float)  if 'rf_prediction'   in ds else None
shap_vals        = ds['shap_values'].values.astype(float)    if 'shap_values'     in ds else None
global_shap_vals = ds['global_shap_values'].values.astype(float) if 'global_shap_values' in ds else None
t_end_arr        = ds['t_end'].values                        if 't_end'           in ds else None
mode             = ds.attrs.get('mode', 'binary')
shap_base        = float(ds.attrs.get('shap_base_value',
                         np.nanmean(rf_pred) if rf_pred is not None else 0.0))
global_shap_base = float(ds.attrs.get('global_shap_base_value', shap_base))
colors = {'aavi': 'yellow', 'pres': 'k', 'rain': 'gray', 'relh': 'g',
          'srad': 'orange', 'tair': 'r', 'wmax': 'b'}

# ── Find nearest event ─────────────────────────────────────────────────────
target_ts = pd.Timestamp(date_str)
idx       = int(np.argmin(np.abs(events - target_ts)))
ts_ep     = events[idx]
lag_min   = abs((ts_ep - target_ts).total_seconds() / 60)
print(f"Matched event: {ts_ep}  (lag = {lag_min:.0f} min from '{target_ts}')")
if not is_outage[idx]:
    print("Warning: matched event is NOT an outage event.")

# ── Load config ────────────────────────────────────────────────────────────
cfg = {}
config_path = out_dir / 'config.yaml'
if config_path.exists():
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
else:
    print(f"Warning: config.yaml not found in {out_dir}; episode plot may be incomplete.")

# ── Load time-series sources ───────────────────────────────────────────────
det_srcs = ([s.strip() for s in ds.attrs['detrended_source'].split(';')]
            if 'detrended_source' in ds.attrs else [])
raw_sources = cfg.get('sources', cfg.get('source', []))
if isinstance(raw_sources, str):
    raw_sources = [raw_sources]
while len(raw_sources) < len(det_srcs):
    raw_sources.append('')

pred_cols = cfg.get('predictor_cols', feat_names)
tgt_col   = cfg.get('target_col', 'customers_out')
load_cols = pred_cols + [tgt_col]
station_names = [Path(p).name.split('.')[0] for p in det_srcs]

df_det_list, df_raw_list = [], []
for det_src, raw_src in zip(det_srcs, raw_sources):
    if Path(det_src).exists():
        ds_det = xr.open_dataset(det_src)
        avail  = [c for c in load_cols if c in ds_det]
        df_det = ds_det[avail].to_dataframe()
        df_det.index = pd.DatetimeIndex(ds_det.time.values)
        ds_det.close()
    else:
        df_det = pd.DataFrame()
    df_det_list.append(df_det)

    if raw_src and Path(raw_src).exists():
        ds_orig = xr.open_dataset(raw_src)
        if 'aavi' in load_cols:
            ds_orig['aavi'] = (ds_orig['wssd'] / ds_orig['wssd'].mean()
                               * ds_orig['wdsd'] / ds_orig['wdsd'].mean())
        avail_raw = [c for c in load_cols if c in ds_orig]
        df_raw_c  = ds_orig[avail_raw].to_dataframe()
        df_raw_c.index = pd.DatetimeIndex(ds_orig.time.values)
        ds_orig.close()
    else:
        df_raw_c = df_det
    df_raw_list.append(df_raw_c)

# ── County indicator ───────────────────────────────────────────────────────
county_feat_idx = (list(feat_names).index('county')
                   if 'county' in list(feat_names) else None)
ev_county = (feat_arr[:, county_feat_idx].astype(int)
             if county_feat_idx is not None
             else np.zeros(len(events), dtype=int))

n_counties = max(len(det_srcs), 1)
ci         = int(ev_county[idx])

def _county_series(vals, c):
    mask = ev_county == c
    return pd.Series(vals[mask], index=events[mask])

target_by_county = [_county_series(target, c) for c in range(n_counties)]
oof_by_county    = ([_county_series(rf_pred, c) for c in range(n_counties)]
                    if rf_pred is not None else [None] * n_counties)

# ── Episode descriptor ─────────────────────────────────────────────────────
t_end_ep = pd.Timestamp(t_end_arr[idx]) if t_end_arr is not None else ts_ep
ep = pd.Series({'t_start':  ts_ep,
                't_end':    t_end_ep,
                't_center': ts_ep + (t_end_ep - ts_ep) / 2})

label1  = station_names[0] if len(station_names) > 0 else ''
label2  = station_names[1] if len(station_names) > 1 else ''
df_raw2 = df_raw_list[1]   if len(df_raw_list) > 1 else None
df_det2 = df_det_list[1]   if len(df_det_list) > 1 else None

# ── Plot 1: episode time series ────────────────────────────────────────────
if df_det_list and not df_det_list[0].empty:
    plot_episode_ts(
        df_raw_list[0], df_det_list[0], ep, cfg,
        target=target_by_county[ci], oof_pred=oof_by_county[ci],
        df_raw2=df_raw2, X2=df_det2,
        label1=label1, label2=label2,
    )
else:
    print("No detrended time-series data found; episode plot skipped.")

# ── Plot 1b: simplified raw time series (top-3 SHAP features) ────────────
if shap_vals is not None:
    shap_df_ci = pd.DataFrame(shap_vals[ev_county == ci],
                              index=events[ev_county == ci], columns=feat_names)
    if ts_ep in shap_df_ci.index:
        shap_ev    = shap_df_ci.loc[ts_ep].values
        df_raw_ci  = df_raw_list[ci]
        buf_h      = pd.Timedelta(hours=cfg.get('episode_buffer_hours', 12))
        t0_s, t1_s = ts_ep - buf_h, ep['t_end'] + buf_h

        seen_b, top3_bases = set(), []
        for _idx in np.argsort(shap_ev)[::-1]:
            base = re.sub(r'_(grad_std|grad_mean|std|mean)_W\d+$', '', feat_names[_idx])
            if base not in seen_b and base in df_raw_ci.columns:
                seen_b.add(base)
                top3_bases.append(base)
            if len(top3_bases) == 3:
                break

        dt       = df_raw_ci.index.to_series().diff().median()
        pre_w    = cfg.get('pre_window', 0)
        post_w   = cfg.get('post_window', 0)
        ev_start = ts_ep - pre_w * dt
        ev_end   = ts_ep + post_w * dt
        has_ev_win = pre_w > 0 or post_w > 0

        n_s = len(top3_bases) + 1
        fig_s, axes_s = plt.subplots(n_s, 1, figsize=(12, 2.5 * n_s), sharex=True)
        if n_s == 1:
            axes_s = [axes_s]

        for j, base in enumerate(top3_bases):
            ax_s = axes_s[j]
            win = df_raw_ci[base].loc[t0_s:t1_s]
            ax_s.plot(win.index, win.values, color='k', lw=1.5)
            ax_s.axvspan(ts_ep, ep['t_end'], alpha=0.12, color='red', linewidth=0)
            if has_ev_win:
                ax_s.axvspan(ev_start, ev_end, alpha=0.12, color='royalblue', linewidth=0)
            ax_s.set_ylabel(_VAR_LABELS.get(base, base))
            ax_s.grid(True, alpha=0.3)

        ax_tgt = axes_s[-1]
        if tgt_col in df_raw_ci.columns:
            ax_tgt.plot(df_raw_ci[tgt_col].loc[t0_s:t1_s], color='firebrick', lw=1.5)
        ax_tgt.axhline(cfg.get('outage_threshold', 0), color='firebrick', ls='--', lw=1, alpha=0.6)
        ax_tgt.axvspan(ts_ep, ep['t_end'], alpha=0.12, color='red', linewidth=0)
        if has_ev_win:
            ax_tgt.axvspan(ev_start, ev_end, alpha=0.12, color='royalblue', linewidth=0)
        ax_tgt.set_ylabel(_VAR_LABELS.get(tgt_col, tgt_col))
        ax_tgt.grid(True, alpha=0.3)
        ax_tgt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        fig_s.autofmt_xdate(rotation=0, ha='center')
        station_ci = station_names[ci] if ci < len(station_names) else ''
        axes_s[0].set_title(f"Top-3 SHAP features (raw) | {ts_ep.strftime('%Y%m%d_%H%M')} | {station_ci}")
        fig_s.tight_layout()
        plt.show()
    else:
        print("SHAP values not available for this event; simplified plot skipped.")

# ── SHAP feature stats (shared across waterfall plots) ────────────────────
X_full    = pd.DataFrame(feat_arr, index=events, columns=feat_names)
feat_mean = X_full.mean().values
feat_std  = np.where(X_full.std().values > 0, X_full.std().values, 1.0)

oof_ep  = oof_by_county[ci]
ts_str  = ts_ep.strftime('%Y%m%d_%H%M')
pred_str = (f"  |  RF pred = {float(oof_ep.loc[ts_ep]):.2f}"
            if oof_ep is not None and ts_ep in oof_ep.index else '')

# ── Plot 2: OOF SHAP waterfall ────────────────────────────────────────────
if shap_vals is not None:
    shap_df = pd.DataFrame(shap_vals[ev_county == ci],
                           index=events[ev_county == ci], columns=feat_names)
    if ts_ep in shap_df.index:
        plot_shap_waterfall(
            shap_vals  = shap_df.loc[ts_ep].values,
            feat_vals  = X_full.loc[ts_ep].values,
            feat_names = feat_names,
            base_value = shap_base,
            feat_mean  = feat_mean,
            feat_std   = feat_std,
            title      = f"OOF SHAP  |  {ts_str}{pred_str}",
        )
    else:
        print("OOF SHAP values not available for this event.")
else:
    print("No OOF SHAP values in dataset.")

# ── Plot 3: global SHAP waterfall ─────────────────────────────────────────
if global_shap_vals is not None:
    global_shap_df = pd.DataFrame(global_shap_vals[ev_county == ci],
                                  index=events[ev_county == ci], columns=feat_names)
    if ts_ep in global_shap_df.index:
        plot_shap_waterfall(
            shap_vals  = global_shap_df.loc[ts_ep].values,
            feat_vals  = X_full.loc[ts_ep].values,
            feat_names = feat_names,
            base_value = global_shap_base,
            feat_mean  = feat_mean,
            feat_std   = feat_std,
            title      = f"Global SHAP  |  {ts_str}{pred_str}",
        )
    else:
        print("Global SHAP values not available for this event.")
else:
    print("No global SHAP values in dataset.")
