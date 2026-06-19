# Sweeps outage_threshold, min_outage_duration, min_peak_customers_out and
# reports OOF Brier scores, ROC-AUC, and coverage to identify the best
# event-definition setup.
#
# Coverage = total outage hours(config) / total outage hours(baseline),
# where baseline uses threshold=0, min_dur=0, min_peak=0.
# Measures what fraction of total outage duration each config retains.
# The Pareto frontier of (coverage, Brier/AUC) identifies configs that balance
# predictive skill with data representativeness.
#
# Event matrices are cached per configuration; repeated runs load from cache.
# Detrending is done once (threshold-independent).
# Permutation importance is reduced to 1 rep since importances are not used here.
# Grouped cross-validation is skipped (consistent relative comparison across configs).

import copy
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml
from pathlib import Path
from datetime import datetime

from outage_rf_events import _load_and_detrend, _get_event_matrix
from utils import train_rf_importance

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

# ── Parameter grid ─────────────────────────────────────────────────────────
OUTAGE_THRESHOLDS       = [10, 20, 50, 100]
MIN_OUTAGE_DURATIONS    = [0*4, 1*4, 2*4, 3*4, 6*4, 12*4, 24*4]       # timesteps (1 ts = 15 min)
MIN_PEAK_CUSTOMERS_OUTS = [0, 100, 200, 500, 1000]

MIN_OUTAGES = 10 # minimum outages threshold (for data quality ensurance)

# ── Config ─────────────────────────────────────────────────────────────────
with open('configs/outage_rf_events.yaml') as f:
    cfg = yaml.safe_load(f)

raw_sources = cfg.get('sources', cfg.get('source'))
sources = [raw_sources] if isinstance(raw_sources, str) else list(raw_sources)
station_names = [Path(s).name.split('.')[0] for s in sources]

ts_run = datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
out_dir = Path(cfg['output_dir']) / f'sensitivity_{ts_run}'
os.makedirs(out_dir, exist_ok=True)

with open(out_dir / 'sensitivity_grid.yaml', 'w') as f:
    yaml.dump({'OUTAGE_THRESHOLDS': OUTAGE_THRESHOLDS,
               'MIN_OUTAGE_DURATIONS': MIN_OUTAGE_DURATIONS,
               'MIN_PEAK_CUSTOMERS_OUTS': MIN_PEAK_CUSTOMERS_OUTS}, f)

# ── Load and detrend once (threshold-independent) ──────────────────────────
print("Loading and detrending data ...")
loaded = [_load_and_detrend(s, cfg, out_dir) for s in sources]
print("Data ready.\n")

dt = loaded[0][0].index.to_series().diff().median()


def _total_duration_h(episode_ends: dict) -> float:
    """Sum of outage episode durations in hours (inclusive of both endpoints)."""
    return sum((t_end - t_start + dt).total_seconds() / 3600
               for t_start, t_end in episode_ends.items())


# ── Baseline: total outage hours at (threshold=0, min_dur=0, min_peak=0) ───
print("Computing baseline outage duration (threshold=0, min_dur=0, min_peak=0) ...")
cfg_base = copy.deepcopy(cfg)
cfg_base.update({'outage_threshold': MIN_OUTAGES, 'min_outage_duration': 0,
                 'min_peak_customers_out': 0, 'detrend_seasonal': False,
                 'shap': False, 'importance_reps': 1})

dur_baseline_h = 0.0
for df_qc, _, source_path in loaded:
    _, _, ep_base, _ = _get_event_matrix(df_qc, cfg_base, source_path)
    dur_baseline_h += _total_duration_h(ep_base)
print(f"  dur_baseline = {dur_baseline_h:.1f} h\n")

# ── Grid sweep ─────────────────────────────────────────────────────────────
records = []
combos = list(itertools.product(OUTAGE_THRESHOLDS, MIN_OUTAGE_DURATIONS, MIN_PEAK_CUSTOMERS_OUTS))
print(f"Running {len(combos)} configurations ...\n")

for idx, (thr, dur, peak) in enumerate(combos, 1):
    print(f"[{idx}/{len(combos)}]  threshold={thr}  min_dur={dur}  min_peak={peak}")

    cfg_run = copy.deepcopy(cfg)
    cfg_run['outage_threshold']       = thr
    cfg_run['min_outage_duration']    = dur
    cfg_run['min_peak_customers_out'] = peak
    cfg_run['detrend_seasonal']       = False
    cfg_run['shap']                   = False
    cfg_run['importance_reps']        = 1

    X_parts, y_parts = [], []
    dur_config_h = 0.0
    for county_i, (df_qc, _, source_path) in enumerate(loaded):
        X_i, y_i, ep_i, _ = _get_event_matrix(df_qc, cfg_run, source_path)
        X_i = X_i.copy()
        X_i['county'] = county_i
        X_parts.append(X_i)
        y_parts.append(y_i)
        dur_config_h += _total_duration_h(ep_i)

    X = pd.concat(X_parts)
    y = pd.concat(y_parts)

    n_outage     = int((y > 0).sum())
    n_non_outage = int((y == 0).sum())
    coverage     = dur_config_h / dur_baseline_h if dur_baseline_h > 0 else np.nan

    base_record = {'outage_threshold': thr, 'min_outage_duration': dur,
                   'min_peak_customers_out': peak,
                   'n_outage': n_outage, 'n_total': len(y),
                   'coverage': coverage}

    if n_outage < cfg_run['n_cv_folds'] or n_non_outage < cfg_run['n_cv_folds']:
        print(f"  Skipped: too few samples "
              f"(outage={n_outage}, non-outage={n_non_outage})\n")
        records.append({**base_record, 'brier_score': np.nan, 'roc_auc': np.nan})
        continue

    rf    = train_rf_importance(X, y, cfg_run)
    oof   = rf['oof_pred'].reindex(y.index)
    y_bin = (y > 0).astype(float)
    # [Brier, 1950; Murphy, 1973]
    brier   = float(np.mean((oof.values - y_bin.values) ** 2))
    roc_auc = float(np.mean(rf['cv_scores']))

    print(f"  Brier={brier:.4f}  ROC-AUC={roc_auc:.4f}  coverage={coverage:.3f}  "
          f"n_outage={n_outage}  n_total={len(y)}\n")
    records.append({**base_record, 'brier_score': brier, 'roc_auc': roc_auc})

# ── Save results ───────────────────────────────────────────────────────────
df_res = pd.DataFrame(records)
csv_path = out_dir / 'sensitivity_results.csv'
df_res.to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")

valid = df_res.dropna(subset=['brier_score', 'coverage'])
if not valid.empty:
    best_brier = valid.nsmallest(1, 'brier_score').iloc[0]
    print(f"\nBest configuration (lowest Brier score):")
    print(f"  thr={int(best_brier['outage_threshold'])}  "
          f"dur={int(best_brier['min_outage_duration'])}  "
          f"peak={int(best_brier['min_peak_customers_out'])}  "
          f"Brier={best_brier['brier_score']:.4f}  coverage={best_brier['coverage']:.3f}")

    valid_c = valid[valid['coverage'] > 0].copy()
    valid_c['composite'] = valid_c['brier_score'] / valid_c['coverage']
    best_comp = valid_c.nsmallest(1, 'composite').iloc[0]
    print(f"\nBest configuration (lowest Brier/coverage composite):")
    print(f"  thr={int(best_comp['outage_threshold'])}  "
          f"dur={int(best_comp['min_outage_duration'])}  "
          f"peak={int(best_comp['min_peak_customers_out'])}  "
          f"composite={best_comp['composite']:.4f}  "
          f"Brier={best_comp['brier_score']:.4f}  coverage={best_comp['coverage']:.3f}")


# ── Helper: Pareto frontier ─────────────────────────────────────────────────
def _pareto_frontier(df, x_col, y_col, minimize_y=True):
    """Non-dominated set: maximise x_col and minimise/maximise y_col."""
    sub = df.dropna(subset=[x_col, y_col]).sort_values(x_col, ascending=False)
    frontier, best_y = [], np.inf if minimize_y else -np.inf
    for _, row in sub.iterrows():
        y = row[y_col]
        if (minimize_y and y <= best_y) or (not minimize_y and y >= best_y):
            frontier.append(row)
            best_y = y
    return pd.DataFrame(frontier).sort_values(x_col)


# ── Heatmaps ───────────────────────────────────────────────────────────────
_METRICS = [
    ('brier_score', 'Brier score', 'RdYlGn_r', True),
    ('roc_auc',     'ROC-AUC',     'RdYlGn',   False),
    ('coverage',    'Coverage',    'RdYlGn',   False),
]

for metric, label, cmap, lower_better in _METRICS:
    if df_res[metric].isna().all():
        continue

    vmin = df_res[metric].min()
    vmax = df_res[metric].max()
    n_thr = len(OUTAGE_THRESHOLDS)

    fig, axes = plt.subplots(1, n_thr, figsize=(4.5 * n_thr, 4), sharey=True)
    if n_thr == 1:
        axes = [axes]

    for ax, thr in zip(axes, OUTAGE_THRESHOLDS):
        sub = df_res[df_res['outage_threshold'] == thr]
        mat = (sub.pivot(index='min_outage_duration',
                         columns='min_peak_customers_out',
                         values=metric)
               .sort_index(ascending=False))

        im = ax.imshow(mat.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel('min_peak_customers_out')
        if ax is axes[0]:
            ax.set_ylabel('min_outage_duration [timesteps]')
        ax.set_title(f'threshold = {thr}')

        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                val = mat.values[ri, ci]
                if not np.isnan(val):
                    ax.text(ci, ri, f'{val:.3f}',
                            ha='center', va='center', fontsize=7)

    cbar = plt.colorbar(im, ax=axes[-1])
    direction = 'lower' if lower_better else 'higher'
    cbar.set_label(f'{label} ({direction} = better)')
    fig.suptitle(f'Sensitivity analysis — {label}', fontweight='bold')
    plt.tight_layout()
    tag = metric.replace('_score', '')
    png_path = out_dir / f'sensitivity_{tag}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {png_path}")


# ── Pareto scatter plots ────────────────────────────────────────────────────
dur_vals  = sorted(df_res['min_outage_duration'].unique())
peak_vals = sorted(df_res['min_peak_customers_out'].unique())
thr_vals  = sorted(df_res['outage_threshold'].unique())

_MARKER_POOL = ['o', 's', '^', 'D', 'P', 'X', 'v', '<']
peak_to_marker = {p: _MARKER_POOL[i] for i, p in enumerate(peak_vals)}

thr_sizes = np.linspace(40, 220, max(len(thr_vals), 2))
thr_to_size = dict(zip(thr_vals, thr_sizes))

# Discrete color mapping for min_outage_duration
_DUR_COLORS = plt.cm.tab10.colors
dur_to_color = {d: _DUR_COLORS[i % len(_DUR_COLORS)] for i, d in enumerate(dur_vals)}

_PARETO_METRICS = [
    ('brier_score', 'Brier score', True),
    ('roc_auc',     'ROC-AUC',    False),
]

for metric, label, minimize_y in _PARETO_METRICS:
    plot_df = df_res.dropna(subset=[metric, 'coverage'])
    if plot_df.empty:
        continue

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(right=0.65)

    for _, row in plot_df.iterrows():
        ax.scatter(
            row['coverage'], row[metric],
            c=[dur_to_color[row['min_outage_duration']]],
            marker=peak_to_marker[row['min_peak_customers_out']],
            s=thr_to_size[row['outage_threshold']],
            edgecolors='k', linewidths=0.5, zorder=3,
        )

    # Pareto frontier line
    pareto = _pareto_frontier(plot_df, 'coverage', metric, minimize_y=minimize_y)
    if len(pareto) > 1:
        ax.plot(pareto['coverage'], pareto[metric],
                color='k', lw=1.5, zorder=4)

    # ── Legends outside the plot area ──────────────────────────────────────
    # Color legend: min_outage_duration (discrete)
    color_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=dur_to_color[d],
               markeredgecolor='k', markersize=9, label=str(d))
        for d in dur_vals
    ]
    leg1 = ax.legend(handles=color_handles, title='min_outage_duration',
                     loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, framealpha=0.8)
    ax.add_artist(leg1)

    # Marker legend: min_peak_customers_out
    marker_handles = [
        Line2D([0], [0], marker=peak_to_marker[p], color='gray', linestyle='none',
               markersize=8, markeredgecolor='k', label=str(p))
        for p in peak_vals
    ]
    leg2 = ax.legend(handles=marker_handles, title='min_peak_customers_out',
                     loc='upper left', bbox_to_anchor=(1.02, 0.65),
                     fontsize=8, framealpha=0.8)
    ax.add_artist(leg2)

    # Size legend: outage_threshold; append Pareto line entry if applicable
    size_handles = [
        Line2D([0], [0], marker='o', color='gray', linestyle='none',
               markeredgecolor='k', markersize=np.sqrt(thr_to_size[t]), label=str(t))
        for t in thr_vals
    ]
    if len(pareto) > 1:
        size_handles.append(Line2D([0], [0], color='k', lw=1.5, label='Pareto frontier'))
    ax.legend(handles=size_handles, title='outage_threshold',
              loc='upper left', bbox_to_anchor=(1.02, 0.3),
              fontsize=8, framealpha=0.8)

    direction = 'lower' if minimize_y else 'higher'
    ax.set_xlabel('Coverage  (total outage hours / baseline total outage hours)')
    ax.set_ylabel(f'{label}  ({direction} = better)')
    ax.set_title(f'Coverage vs {label} — Pareto frontier')
    ax.grid(alpha=0.3)

    tag = metric.replace('_score', '')
    png_path = out_dir / f'sensitivity_pareto_{tag}.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {png_path}")

print(f"\nAll outputs in {out_dir}")
