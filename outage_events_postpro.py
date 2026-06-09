import re
import os
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from utils import (_VAR_LABELS, _AGG_LABELS, plot_importance_comparison,
                   plot_reliability_diagram, plot_shap_dependence,
                   plot_shap_waterfall, plot_episode_ts)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

#%% Inputs
source='./results/20260608.171511/all/events.nc'

# ── load ──────────────────────────────────────────────────────────

ds = xr.open_dataset(source)
out_dir = os.path.dirname(source)

events     = pd.DatetimeIndex(ds.event.values)
target     = ds['target'].values.astype(float)
tot        = (ds['t_end']-ds.event)/np.timedelta64(60,'s')
is_outage  = ds['is_outage'].values.astype(bool)
feat_names = list(ds.feature.values)
n_feat     = len(feat_names)
mode       = ds.attrs.get('mode', 'target')
units      = {'AUC': 'customer·h', 'TOT': 'min'}.get(mode, '')
colors={'aavi':'yellow', 'pres':'k', 'rain':'gray', 'relh':'g', 'srad':'orange', 'tair':'r', 'wmax':'b'}

feat_arr  = ds['features'].values.astype(float)
rf_pred   = ds['rf_prediction'].values.astype(float) if 'rf_prediction' in ds else None
shap_vals = ds['shap_values'].values.astype(float)   if 'shap_values'   in ds else None

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

# ── plot 2: scatter – TOT vs. dominant-SHAP feature ───────────
if shap_vals is not None:
    mask       = is_outage*(np.max(shap_vals, axis=1)>0)
    tot_out      = tot[mask]
    feat_out   = feat_arr[mask]
    shap_out   = shap_vals[mask]
    n_out      = mask.sum()

    abs_shap  = np.abs(shap_out)
    dom_idx   = np.argmax(abs_shap, axis=1)
    dom_names = [feat_names[i] for i in dom_idx]

    feat_mean = feat_arr.mean(axis=0)
    feat_std  = np.where(feat_arr.std(axis=0) > 0, feat_arr.std(axis=0), 1.0)
    feat_norm = (feat_out - feat_mean) / feat_std

    y_vals = np.array([feat_norm[i, dom_idx[i]] for i in range(n_out)])

    dom_shap = abs_shap[np.arange(n_out), dom_idx]
    sizes = 10 + 270 * (dom_shap / dom_shap.max())

    unique_feats = sorted(set(dom_names))
    n_uf = len(unique_feats)
    cmap = plt.get_cmap('tab10' if n_uf <= 10 else 'tab20' if n_uf <= 20 else 'hsv', n_uf)
    color_map = {f: cmap(i) for i, f in enumerate(unique_feats)}

    def _feat_type(name):
        if '_grad_std_'  in name: return 'grad_std'
        if '_grad_mean_' in name: return 'grad_mean'
        if '_std_'       in name: return 'std'
        if '_mean_'      in name: return 'mean'
        return 'other'

    def _base_var(name):
        return re.sub(r'_(grad_std|grad_mean|std|mean)_W\d+$', '', name)

    type_marker = {'mean': 'o', 'std': '*', 'grad_mean': 's', 'grad_std': 'D'}
    dom_types = [_feat_type(f) for f in dom_names]
    dom_bases = [_base_var(f) for f in dom_names]

    unique_bases = sorted(set(dom_bases))

    fig, ax = plt.subplots(figsize=(9, 6))
    for base in unique_bases:
        for ftype, marker in type_marker.items():
            sel = [i for i, (b, t) in enumerate(zip(dom_bases, dom_types))
                   if b == base and t == ftype]
            if not sel:
                continue
            ax.scatter(
                tot_out[sel], y_vals[sel],
                s=sizes[sel],
                marker=marker,
                color=colors[base],
                alpha=0.75,
                edgecolors='k', linewidths=0.4,
                zorder=3,
            )

    from matplotlib.lines import Line2D
    color_handles = [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=colors[b], markeredgecolor='k',
               markersize=8,
               label=re.sub(r'\s*\[.*?\]', '', _VAR_LABELS.get(b, b)))
        for b in unique_bases
    ]
    active_types = sorted({t for t in dom_types if t in type_marker})
    marker_handles = [
        Line2D([0], [0], marker=type_marker[t], linestyle='none',
               markerfacecolor='gray', markeredgecolor='k',
               markersize=8, label=_AGG_LABELS.get(t, t).capitalize())
        for t in active_types
    ]
    leg1 = ax.legend(handles=color_handles, title='Variable', bbox_to_anchor=(1.00, 1),
                     loc='upper left', fontsize=8, framealpha=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title='Type', bbox_to_anchor=(1.00, 0),
              loc='lower left', fontsize=8, framealpha=0.8)

    ax.set_xlabel('TOT (min)')
    ax.set_ylabel('Dominant feature value (z-score)')
    ax.set_title('Target vs. dominant SHAP feature\n'
                 '(size = |SHAP|, colour = variable, marker = aggregation type)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.72)
    plt.show()
    fig.savefig(out_dir / 'scatter_dominant_feature.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_dir / 'scatter_dominant_feature.png'}")
else:
    print("SHAP values not in dataset; scatter plot skipped.")

# ── plot 3: importance comparison ─────────────────────────────────
rf_importance     = ds['rf_importance'].values     if 'rf_importance'     in ds else None
rf_importance_std = ds['rf_importance_std'].values if 'rf_importance_std' in ds else None

if rf_importance is not None:
    rf_result_pp = {
        'feature_names':   feat_names,
        'importance_mean': rf_importance,
        'importance_std':  rf_importance_std if rf_importance_std is not None
                           else np.zeros(len(feat_names)),
        'cv_scores':       [ds.attrs.get('cv_score_mean', np.nan)],
        'score_name':      ds.attrs.get('score_name', ''),
    }
    results_pp = pd.DataFrame({
        'RF_perm_importance': rf_result_pp['importance_mean'],
        'RF_perm_std':        rf_result_pp['importance_std'],
    }, index=feat_names).sort_values('RF_perm_importance', ascending=False)
    results_pp['RF_rank'] = range(1, len(results_pp) + 1)
    if shap_vals is not None:
        mean_shap_pp = pd.Series(np.abs(shap_vals).mean(axis=0), index=feat_names)
        results_pp['SHAP_mean_abs'] = mean_shap_pp
        results_pp['SHAP_rank'] = results_pp['SHAP_mean_abs'].rank(ascending=False).astype(int)
    plot_importance_comparison(results_pp, rf_result_pp,
                               save_path=Path(out_dir) / 'importance_comparison.png')

# ── plot 4: reliability diagram ───────────────────────────────────
if mode == 'binary' and rf_pred is not None:
    config_pp = {'reliability_bins': ds.attrs.get('reliability_bins', 10)}
    y_true_pp = pd.Series(target, index=events)
    y_prob_pp = pd.Series(rf_pred, index=events)
    plot_reliability_diagram(y_true_pp, y_prob_pp, config_pp,
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

    det_src = ds.attrs['detrended_source']
    if Path(det_src).exists():
        ds_raw = xr.open_dataset(det_src)
        pred_cols = cfg_pp.get('predictor_cols', feat_names)
        available = [c for c in pred_cols if c in ds_raw]
        tgt_col = cfg_pp.get('target_col', 'customers_out')
        load_cols = available + ([tgt_col] if tgt_col in ds_raw else [])
        df_detrended_pp = ds_raw[load_cols].to_dataframe()
        df_detrended_pp.index = pd.DatetimeIndex(ds_raw.time.values)
        ds_raw.close()

        # Load original undetrended data for the black raw signal
        raw_src = cfg_pp.get('source', '')
        if Path(raw_src).exists():
            ds_orig = xr.open_dataset(raw_src)
            if 'aavi' in load_cols:
                ds_orig['aavi'] = ds_orig['wssd'] / ds_orig['wssd'].mean() * ds_orig['wdsd'] / ds_orig['wdsd'].mean()

            df_raw_pp = ds_orig[load_cols].to_dataframe()
            df_raw_pp.index = pd.DatetimeIndex(ds_orig.time.values)
            ds_orig.close()
        else:
            df_raw_pp = df_detrended_pp

        t_end_arr = ds['t_end'].values              if 't_end'              in ds else None
        pk_arr    = ds['peak_customers_out'].values  if 'peak_customers_out' in ds else None

        if t_end_arr is not None:
            out_mask     = is_outage
            t_starts_out = events[out_mask]
            t_ends_out   = pd.DatetimeIndex(t_end_arr[out_mask])
            durations    = (t_ends_out - t_starts_out).total_seconds() / 60

            ev_df = pd.DataFrame({
                't_start':            t_starts_out,
                't_end':              t_ends_out,
                'duration':           durations,
                'peak_customers_out': pk_arr[out_mask] if pk_arr is not None
                                      else np.full(out_mask.sum(), np.nan),
                'rf_prediction':      rf_pred[out_mask],
            }).set_index('t_start')

            n_ep    = cfg_pp.get('n_episode_plots', 10)
            top_dur = ev_df.nlargest(n_ep, 'duration').index
            top_pk  = ev_df.nlargest(n_ep, 'peak_customers_out').index
            top_ev  = ev_df.loc[top_dur.union(top_pk)]

            oof_pp    = pd.Series(rf_pred, index=events)
            ep_dir_pp = Path(out_dir) / 'episodes'
            os.makedirs(ep_dir_pp, exist_ok=True)

            shap_base_pp = ds.attrs.get('shap_base_value', float(np.mean(rf_pred)))
            if shap_vals is not None:
                shap_pp_df = pd.DataFrame(shap_vals, index=events, columns=feat_names)
                X_ev_pp    = pd.DataFrame(feat_arr, index=events, columns=feat_names)
                feat_mean_pp = X_ev_pp.mean().values
                feat_std_pp  = np.where(X_ev_pp.std().values > 0, X_ev_pp.std().values, 1.0)

            for ts_ep, row in top_ev.iterrows():
                ep = pd.Series({
                    't_start':  ts_ep,
                    't_end':    row['t_end'],
                    't_center': ts_ep + (row['t_end'] - ts_ep) / 2,
                    'metric':   row['duration'],
                })
                # df_raw_pp = original signal (black); df_detrended_pp = full detrended time series (blue)
                plot_episode_ts(df_raw_pp, df_detrended_pp, ep, cfg_pp, ep_dir_pp, oof_pred=oof_pp)

                if shap_vals is not None and ts_ep in shap_pp_df.index:
                    ts_str = pd.Timestamp(ts_ep).strftime('%Y%m%d_%H%M')
                    prob   = float(oof_pp.loc[ts_ep])
                    plot_shap_waterfall(
                        shap_vals   = shap_pp_df.loc[ts_ep].values,
                        feat_vals   = X_ev_pp.loc[ts_ep].values,
                        feat_names  = feat_names,
                        base_value  = shap_base_pp,
                        feat_mean   = feat_mean_pp,
                        feat_std    = feat_std_pp,
                        save_path   = ep_dir_pp / f"shap_waterfall_{ts_str}.png",
                        title       = f"{ts_str}  |  RF prob = {prob:.2f}",
                    )
# 