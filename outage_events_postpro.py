import re
import os
import sys
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter
import tkinter.filedialog
from pathlib import Path
from utils import (_VAR_LABELS, _AGG_LABELS, plot_importance_comparison,
                   plot_reliability_diagram, plot_shap_dependence,
                   plot_shap_waterfall, plot_episode_ts)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source = tkinter.filedialog.askopenfilename(
    title='Select events.nc',
    initialdir='./results/',
    filetypes=[('NetCDF files', '*.nc')],
)
MIN_RF_PRED=0.5#minimum value of RF prediction in scatter plot
RELIABILITY_BINS=20

if not source:
    print('No file selected. Exiting.')
    sys.exit()

# ── load ──────────────────────────────────────────────────────────

ds = xr.open_dataset(source)
out_dir =  Path(os.path.dirname(source))

events     = pd.DatetimeIndex(ds.event.values)
target     = ds['target'].values.astype(float)
tot        = (ds['t_end']-ds.event)/np.timedelta64(60,'s')
is_outage  = ds['is_outage'].values.astype(bool)
feat_names = list(ds.feature.values)
n_feat     = len(feat_names)
mode       = ds.attrs.get('mode', 'binary')
units      = 'min'
colors={'aavi':'yellow', 'pres':'k', 'rain':'gray', 'relh':'g', 'srad':'orange', 'tair':'r', 'wmax':'b'}

feat_arr  = ds['features'].values.astype(float)
rf_pred   = ds['rf_prediction'].values.astype(float) if 'rf_prediction' in ds else None
shap_vals = ds['shap_values'].values.astype(float)   if 'shap_values'   in ds else None

global_shap_vals     = ds['global_shap_values'].values.astype(float)     if 'global_shap_values'     in ds else None
global_rf_importance = ds['global_rf_importance'].values.astype(float)    if 'global_rf_importance'    in ds else None
global_rf_imp_std    = ds['global_rf_importance_std'].values.astype(float) if 'global_rf_importance_std' in ds else None

rain_means = None
if 'additional_col_means' in ds and 'additional_col' in ds.coords:
    add_cols = list(ds.additional_col.values)
    if 'rain' in add_cols:
        rain_means = ds['additional_col_means'].values[:, add_cols.index('rain')].astype(float)

# ── plot 1: bar chart observed vs. predicted ──────────────────────
events_num = mdates.date2num(events.to_pydatetime())
bar_w = 0.35 * np.median(np.diff(events_num)) if len(events_num) > 1 else 1.0

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(events_num, target,  width=bar_w, color='steelblue', alpha=0.8, label='Observed')
if rf_pred is not None:
    ax.bar(events_num, rf_pred, width=bar_w, color='firebrick', alpha=0.5, label='RF prediction')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate(rotation=45, ha='right')
ax.set_ylabel(f'{mode} ({units})')
ax.set_title('Outage severity: observed vs. RF prediction (OOF)')
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()
fig.savefig(out_dir / 'bar_target_vs_pred.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'bar_target_vs_pred.png'}")

def _dominant_shap_scatter(axes, shap_v, feat_arr, tot, is_outage, rf_pred, min_ref_pred, feat_names, colors,
                           rain_means=None):
    from matplotlib.lines import Line2D

    def _feat_type(name):
        if '_grad_std_'  in name: return 'grad_std'
        if '_grad_mean_' in name: return 'grad_mean'
        if '_std_'       in name: return 'std'
        if '_mean_'      in name: return 'mean'
        return 'other'

    def _base_var(name):
        return re.sub(r'_(grad_std|grad_mean|std|mean)_W\d+$', '', name)

    mask     = is_outage * (rf_pred > min_ref_pred) * (np.max(shap_v, axis=1) > 0)
    tot_out  = tot[mask]
    feat_out = feat_arr[mask]
    shap_out = shap_v[mask]
    n_out    = mask.sum()
    rain_out = rain_means[mask] if rain_means is not None else None

    feat_mean  = feat_arr.mean(axis=0)
    feat_std   = np.where(feat_arr.std(axis=0) > 0, feat_arr.std(axis=0), 1.0)
    feat_norm  = (feat_out - feat_mean) / feat_std

    # Rank features by SHAP descending for each event
    rank_idx  = np.argsort(shap_out, axis=1)[:, ::-1]
    n_ranks   = min(len(axes), shap_out.shape[1])
    type_marker  = {'mean': 'o', 'std': '*', 'grad_mean': 's', 'grad_std': 'D'}
    rank_labels  = ['1st', '2nd', '3rd'] + [f'{k+1}th' for k in range(3, n_ranks)]

    # Collect bases and types across all ranks for a complete legend
    all_unique_bases, all_active_types = set(), set()
    for k in range(n_ranks):
        dom_idx_k  = rank_idx[:, k]
        dom_names_k = [feat_names[i] for i in dom_idx_k]
        all_unique_bases.update(_base_var(f) for f in dom_names_k)
        all_active_types.update(t for f in dom_names_k
                                for t in [_feat_type(f)] if t in type_marker)
    all_unique_bases = sorted(all_unique_bases)
    all_active_types = sorted(all_active_types)

    for k, ax in enumerate(axes[:n_ranks]):
        dom_idx      = rank_idx[:, k]
        dom_names    = [feat_names[i] for i in dom_idx]
        dom_types    = [_feat_type(f) for f in dom_names]
        dom_bases    = [_base_var(f)  for f in dom_names]
        unique_bases = sorted(set(dom_bases))

        y_vals   = np.array([feat_norm[i, dom_idx[i]] for i in range(n_out)])
        dom_shap = shap_out[np.arange(n_out), dom_idx]
        sizes    = 10 + 1000 * dom_shap

        for base in unique_bases:
            for ftype, marker in type_marker.items():
                sel = [i for i, (b, t, s) in enumerate(zip(dom_bases, dom_types,sizes))
                       if b == base and t == ftype and s>0]
                if not sel:
                    continue
                if rain_out is not None:
                    ec = ['green' if rain_out[i] > 0 else 'k' for i in sel]
                    lw = [2      if rain_out[i] > 0 else 0.4 for i in sel]
                else:
                    ec, lw = 'k', 0.4
                ax.scatter(
                    tot_out[sel], y_vals[sel],
                    s=sizes[sel], marker=marker,
                    color=colors[base], alpha=0.75,
                    edgecolors=ec, linewidths=lw, zorder=3,
                )

        ax.set_ylabel(f'{rank_labels[k]} driver\n(z-score)')
        ax.grid(alpha=0.3)

        # Legends on first panel only, using all ranks' bases and types
        if k == 0:
            color_handles = [
                Line2D([0], [0], marker='o', linestyle='none',
                       markerfacecolor=colors[b], markeredgecolor='k',
                       markersize=8,
                       label=re.sub(r'\s*\[.*?\]', '', _VAR_LABELS.get(b, b)))
                for b in all_unique_bases
            ]
            marker_handles = [
                Line2D([0], [0], marker=type_marker[t], linestyle='none',
                       markerfacecolor='gray', markeredgecolor='k',
                       markersize=8, label=_AGG_LABELS.get(t, t).capitalize())
                for t in all_active_types
            ]
            if rain_out is not None:
                marker_handles.append(
                    Line2D([0], [0], marker='o', linestyle='none',
                           markerfacecolor='gray', markeredgecolor='green',
                           markeredgewidth=1.2, markersize=8, label='rain > 0')
                )
            leg1 = ax.legend(handles=color_handles, title='Variable',
                             bbox_to_anchor=(1.00, 1), loc='upper left',
                             fontsize=8, framealpha=0.8)
            ax.add_artist(leg1)
            ax.legend(handles=marker_handles, title='Type',
                      bbox_to_anchor=(1.00, 0), loc='lower left',
                      fontsize=8, framealpha=0.8)

    axes[-1].set_xlabel('Duration (min)')

# ── plot 2: scatter – TOT vs. dominant-SHAP feature ───────────
_have_oof    = shap_vals is not None
_have_global = global_shap_vals is not None

if _have_oof or _have_global:
    _panels = []
    if _have_oof:    _panels.append(('OOF SHAP',    'oof',    shap_vals))
    if _have_global: _panels.append(('Global SHAP', 'global', global_shap_vals))

    for title, tag, sv in _panels:
        fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
        _dominant_shap_scatter(axes, sv, feat_arr, tot, is_outage, rf_pred,MIN_RF_PRED, feat_names, colors,
                               rain_means=rain_means)
        axes[0].set_title(f'Dominant SHAP feature — {title}')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
        png = out_dir / f'scatter_dominant_feature_{tag}.png'
        fig.savefig(png, dpi=300)
        # plt.close(fig)
        print(f"Saved → {png}")
else:
    print("SHAP values not in dataset; scatter plot skipped.")

# ── plot 3: importance comparison ─────────────────────────────────
rf_importance     = ds['rf_importance'].values     if 'rf_importance'     in ds else None
rf_importance_std = ds['rf_importance_std'].values if 'rf_importance_std' in ds else None

_importance_cases = []
if rf_importance is not None:
    _importance_cases.append(('oof', rf_importance, rf_importance_std, shap_vals))
if global_rf_importance is not None:
    _importance_cases.append(('global', global_rf_importance, global_rf_imp_std, global_shap_vals))

for _tag, _imp, _imp_std, _sv in _importance_cases:
    _rf_result_pp = {
        'feature_names':   feat_names,
        'importance_mean': _imp,
        'importance_std':  _imp_std if _imp_std is not None else np.zeros(len(feat_names)),
        'cv_scores':       [ds.attrs.get('cv_score_mean', np.nan)],
        'score_name':      ds.attrs.get('score_name', ''),
    }
    _results_pp = pd.DataFrame({
        'RF_perm_importance': _rf_result_pp['importance_mean'],
        'RF_perm_std':        _rf_result_pp['importance_std'],
    }, index=feat_names).sort_values('RF_perm_importance', ascending=False)
    _results_pp['RF_rank'] = range(1, len(_results_pp) + 1)
    if _sv is not None:
        _mean_shap = pd.Series(np.abs(_sv).mean(axis=0), index=feat_names)
        _results_pp['SHAP_mean_abs'] = _mean_shap
        _results_pp['SHAP_rank'] = _results_pp['SHAP_mean_abs'].rank(ascending=False).astype(int)
    plot_importance_comparison(_results_pp, _rf_result_pp,
                               save_path=out_dir / f'importance_comparison_{_tag}.png')

# ── plot 4: reliability diagram ───────────────────────────────────
if rf_pred is not None:
    y_true_pp = pd.Series(target, index=events)
    y_prob_pp = pd.Series(rf_pred, index=events)
    plot_reliability_diagram(y_true_pp, y_prob_pp,
                             save_path=Path(out_dir) / 'reliability_diagram.png')

# ── plot 5: SHAP dependence ───────────────────────────────────────
if shap_vals is not None:
    X_pp    = pd.DataFrame(feat_arr, index=events, columns=feat_names)
    shap_pp = pd.DataFrame(shap_vals, index=events, columns=feat_names)
    plot_shap_dependence(X_pp, shap_pp,
                         save_path=Path(out_dir) / 'shap_dependence.png')

# ── plot 6: episode time series ───────────────────────────────────
if rf_pred is not None and 'detrended_source' in ds.attrs:
    config_path = Path(out_dir) / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            cfg_pp = yaml.safe_load(f)
    else:
        cfg_pp = {}

    det_srcs = [s.strip() for s in ds.attrs['detrended_source'].split(';')]
    raw_sources = cfg_pp.get('sources', cfg_pp.get('source', []))
    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]
    while len(raw_sources) < len(det_srcs):
        raw_sources.append('')

    pred_cols = cfg_pp.get('predictor_cols', list(feat_names))
    tgt_col = cfg_pp.get('target_col', 'customers_out')
    load_cols = pred_cols + [tgt_col]
    station_names = [Path(p).name.split('.')[0] for p in det_srcs]

    df_det_list, df_raw_list = [], []
    for det_src, raw_src in zip(det_srcs, raw_sources):
        if Path(det_src).exists():
            ds_det = xr.open_dataset(det_src)
            avail = [c for c in load_cols if c in ds_det]
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
            df_raw_c = ds_orig[avail_raw].to_dataframe()
            df_raw_c.index = pd.DatetimeIndex(ds_orig.time.values)
            ds_orig.close()
        else:
            df_raw_c = df_det
        df_raw_list.append(df_raw_c)

    # County indicator per event (matches county column added during pooling)
    county_feat_idx = list(feat_names).index('county') if 'county' in list(feat_names) else None
    ev_county = (feat_arr[:, county_feat_idx].astype(int)
                 if county_feat_idx is not None
                 else np.zeros(len(events), dtype=int))

    t_end_arr = ds['t_end'].values              if 't_end'              in ds else None
    pk_arr    = ds['peak_customers_out'].values  if 'peak_customers_out' in ds else None

    if t_end_arr is not None:
        all_t_ends = pd.DatetimeIndex(t_end_arr)
        ev_df_all = pd.DataFrame({
            't_end':              all_t_ends,
            'target':             target,
            'rf_prediction':      rf_pred,
            'is_outage':          is_outage,
            'county':             ev_county,
            'peak_customers_out': pk_arr if pk_arr is not None
                                  else np.full(len(events), np.nan),
        }, index=pd.DatetimeIndex(events))
        ev_df_all.index.name = 't_start'

        # FP events have no real t_end in the dataset; fall back to t_start
        nat_mask = pd.isnull(ev_df_all['t_end'])
        ev_df_all.loc[nat_mask, 't_end'] = ev_df_all.index[nat_mask]

        oo_mask     = ev_df_all['is_outage']
        episodes_tp = ev_df_all[oo_mask].nlargest(5, 'rf_prediction')
        episodes_fp = ev_df_all[~oo_mask].nlargest(5, 'rf_prediction')
        episodes_fn = ev_df_all[oo_mask].nsmallest(5, 'rf_prediction')

        # Per-county Series to avoid duplicate-timestamp issues when two counties
        # share an identical t_start (positional filter ensures scalar .loc access)
        def _county_series(vals, ci):
            mask = (ev_county == ci)
            return pd.Series(vals[mask], index=events[mask])

        target_by_county = [_county_series(target,  ci) for ci in range(len(det_srcs))]
        oof_by_county    = [_county_series(rf_pred, ci) for ci in range(len(det_srcs))]

        label1  = station_names[0] if len(station_names) > 0 else ''
        label2  = station_names[1] if len(station_names) > 1 else ''
        df_raw2 = df_raw_list[1]   if len(df_raw_list)  > 1 else None
        df_det2 = df_det_list[1]   if len(df_det_list)  > 1 else None

        shap_base_pp = ds.attrs.get('shap_base_value', float(np.mean(rf_pred)))
        if shap_vals is not None:
            X_full_df    = pd.DataFrame(feat_arr,  index=events, columns=feat_names)
            feat_mean_pp = X_full_df.mean().values
            feat_std_pp  = np.where(X_full_df.std().values > 0, X_full_df.std().values, 1.0)
            shap_by_county = [
                pd.DataFrame(shap_vals[ev_county == ci],
                             index=events[ev_county == ci], columns=feat_names)
                for ci in range(len(det_srcs))
            ]
            X_by_county = [
                pd.DataFrame(feat_arr[ev_county == ci],
                             index=events[ev_county == ci], columns=feat_names)
                for ci in range(len(det_srcs))
            ]

        groups = [
            ('tp', episodes_tp, 'TP (outage, high pred)'),
            ('fp', episodes_fp, 'FP (no outage, high pred)'),
            ('fn', episodes_fn, 'FN (outage, low pred)'),
        ]
        for group_tag, subset, group_label in groups:
            ep_dir_grp = Path(out_dir) / 'episodes' / group_tag
            os.makedirs(ep_dir_grp, exist_ok=True)

            for ts_ep, row in subset.iterrows():
                ep = pd.Series({
                    't_start':  ts_ep,
                    't_end':    row['t_end'],
                    't_center': ts_ep + (row['t_end'] - ts_ep) / 2,
                })
                ci        = int(row['county'])
                oof_ep    = oof_by_county[ci]
                target_ep = target_by_county[ci]

                pred = float(oof_ep.loc[ts_ep]) if ts_ep in oof_ep.index else np.nan

                plot_episode_ts(
                    df_raw_list[0], df_det_list[0], ep, cfg_pp, ep_dir_grp,
                    target=target_ep, oof_pred=oof_ep,
                    df_raw2=df_raw2, X2=df_det2,
                    label1=label1, label2=label2,
                    title_extra=f"{group_label} | RF pred = {pred:.2f}",
                )

                if shap_vals is not None and ts_ep in shap_by_county[ci].index:
                    ts_str = pd.Timestamp(ts_ep).strftime('%Y%m%d_%H%M')
                    if mode == 'binary':
                        plot_shap_waterfall(
                            shap_vals  = shap_by_county[ci].loc[ts_ep].values,
                            feat_vals  = X_by_county[ci].loc[ts_ep].values,
                            feat_names = feat_names,
                            base_value = shap_base_pp,
                            feat_mean  = feat_mean_pp,
                            feat_std   = feat_std_pp,
                            save_path  = ep_dir_grp / f"shap_waterfall_{ts_str}.png",
                            title      = f"{ts_str} | {group_label} | RF pred = {pred:.2f}",
                        )

# 