import os
import sys
import tkinter
import tkinter.filedialog
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from utils import _VAR_LABELS

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 150
plt.close('all')

CFG = {
    'k_range':        range(2, 12),
    'n_feat_range':   range(2, 21),
    'random_state':   42,
    'rain_threshold': 0.1,   # events with mean rain > this are "rainy"
}

# ── File picker ───────────────────────────────────────────────────────────────
root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()
source = tkinter.filedialog.askopenfilename(
    title='Select events.nc', initialdir='./results/', filetypes=[('NetCDF files', '*.nc')])
if not source:
    sys.exit()
ds = xr.open_dataset(source)
out_dir = Path(os.path.dirname(source)) / 'clusters'
out_dir.mkdir(exist_ok=True)

# ── Extract outage events ─────────────────────────────────────────────────────
events     = pd.DatetimeIndex(ds.event.values)
is_outage  = ds['is_outage'].values.astype(bool)
feat_names = list(ds.feature.values)
feat_arr   = ds['features'].values.astype(float)
peak_pk    = ds['peak_customers_out'].values.astype(float)

X_out      = feat_arr[is_outage]
events_out = events[is_outage]
peak_out   = peak_pk[is_outage]

# ── Global SHAP importance (feature selection) ────────────────────────────────
shap_arr = (ds['global_shap_values'].values if 'global_shap_values' in ds
            else ds['shap_values'].values)
shap_out = shap_arr[is_outage]

# exclude 'county' — site label, not a meteorological predictor
met_idx = [i for i, n in enumerate(feat_names) if n != 'county']

# ── County indicator ──────────────────────────────────────────────────────────
county_idx = feat_names.index('county') if 'county' in feat_names else None
county_out = (X_out[:, county_idx].astype(int)
              if county_idx is not None
              else np.zeros(len(X_out), dtype=int))

# ── Stratify by precipitation ─────────────────────────────────────────────────
# Identify the rain mean feature; fall back to any rain feature
rain_feat = next((n for n in feat_names if n.startswith('rain') and '_mean_' in n),
                 next((n for n in feat_names if n.startswith('rain')), None))
if rain_feat is not None:
    rain_vals  = X_out[:, feat_names.index(rain_feat)]
    rainy_mask = rain_vals > CFG['rain_threshold']
else:
    rainy_mask = np.zeros(len(X_out), dtype=bool)
    print("Warning: no rain feature found; all events classified as dry")

dry_mask = ~rainy_mask
print(f"Stratification on '{rain_feat}' > {CFG['rain_threshold']}: "
      f"{rainy_mask.sum()} rainy, {dry_mask.sum()} dry outage events")

strata = [('rainy', rainy_mask), ('dry', dry_mask)]

# ── Per-stratum clustering ────────────────────────────────────────────────────
for stratum_name, stratum_mask in strata:
    n_s = int(stratum_mask.sum())
    if n_s < min(CFG['k_range']) + 1:
        print(f"\nSkipping '{stratum_name}': only {n_s} events (need ≥ {min(CFG['k_range']) + 1})")
        continue

    stratum_dir = out_dir / stratum_name
    stratum_dir.mkdir(exist_ok=True)
    print(f"\n══ Stratum: {stratum_name}  ({n_s} events) ══")

    X_s      = X_out[stratum_mask]
    peak_s   = peak_out[stratum_mask]
    county_s = county_out[stratum_mask]
    shap_s   = shap_out[stratum_mask]

    # SHAP importance within stratum (features most relevant to this regime)
    shap_imp_s = np.mean(np.abs(shap_s[:, met_idx]), axis=0)

    # ── 2-D silhouette sweep (n_features × K) ────────────────────────────────
    # [Rousseeuw, 1987, J. Comput. Appl. Math.]
    n_feat_list = list(CFG['n_feat_range'])
    k_list      = list(CFG['k_range'])
    sil_matrix  = pd.DataFrame(np.nan, index=n_feat_list, columns=k_list)

    for n_top in n_feat_list:
        n_top_eff  = min(n_top, len(met_idx))
        top_idx_n  = np.argsort(shap_imp_s)[::-1][:n_top_eff]
        sel_idx_n  = [met_idx[i] for i in top_idx_n]
        X_scaled_n = StandardScaler().fit_transform(X_s[:, sel_idx_n])
        for k in k_list:
            if k >= n_s:
                continue
            km_n     = KMeans(n_clusters=k, random_state=CFG['random_state'], n_init=20)
            labels_n = km_n.fit_predict(X_scaled_n)
            if len(set(labels_n)) > 1:
                sil_matrix.loc[n_top, k] = silhouette_score(X_scaled_n, labels_n)

    sil_matrix.to_csv(stratum_dir / 'silhouette_matrix.csv')
    print(f"  Saved → {stratum_dir / 'silhouette_matrix.csv'}")

    # ── Auto-select best (n_features, K) from silhouette heatmap ─────────────
    best_n_top, best_k = sil_matrix.stack().idxmax()
    best_score = sil_matrix.loc[best_n_top, best_k]
    print(f"  Best silhouette: n_features={best_n_top}, K={best_k}, score={best_score:.3f}")

    # ── Final k-means with auto-selected parameters ───────────────────────────
    K        = best_k
    n_top_f  = min(best_n_top, len(met_idx))
    top_idx_f  = np.argsort(shap_imp_s)[::-1][:n_top_f]
    sel_idx_f  = [met_idx[i] for i in top_idx_f]
    sel_names  = [feat_names[i] for i in sel_idx_f]
    print(f"  Selected features: {sel_names}")

    X_sel_f   = X_s[:, sel_idx_f]
    scaler_f  = StandardScaler()
    X_scaled_f = scaler_f.fit_transform(X_sel_f)

    km_f     = KMeans(n_clusters=K, random_state=CFG['random_state'], n_init=20)
    labels   = km_f.fit_predict(X_scaled_f)
    centroids_scaled = km_f.cluster_centers_
    centroids        = scaler_f.inverse_transform(centroids_scaled)

    unique_labs, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster sizes: { {f'C{k}': int(c) for k, c in zip(unique_labs, counts)} }")

    crosstab = pd.crosstab(pd.Series(labels, name='cluster'),
                           pd.Series(county_s, name='county'))

    # ── Plot 1: silhouette score heatmap ─────────────────────────────────────
    mat_vals = sil_matrix.values.astype(float)
    vmin_h   = np.nanmin(mat_vals)
    vmax_h   = np.nanmax(mat_vals)

    fig, ax = plt.subplots(figsize=(max(5, len(k_list) * 0.7),
                                    max(4, len(n_feat_list) * 0.4)))
    im = ax.imshow(mat_vals, cmap='RdYlGn', vmin=vmin_h, vmax=vmax_h,
                   aspect='auto', origin='lower')
    ax.set_xticks(range(len(k_list)))
    ax.set_xticklabels(k_list)
    ax.set_yticks(range(len(n_feat_list)))
    ax.set_yticklabels(n_feat_list)
    ax.set_xlabel('Number of clusters K')
    ax.set_ylabel('Number of top SHAP features')
    ax.set_title(f'Silhouette score — {stratum_name}')
    for ri, n_top in enumerate(n_feat_list):
        for ci, k in enumerate(k_list):
            v = sil_matrix.loc[n_top, k]
            if not np.isnan(v):
                ax.text(ci, ri, f'{v:.2f}', ha='center', va='center', fontsize=6)
    chosen_xi = k_list.index(best_k)     if best_k     in k_list      else None
    chosen_yi = n_feat_list.index(best_n_top) if best_n_top in n_feat_list else None
    if chosen_xi is not None and chosen_yi is not None:
        ax.add_patch(plt.Rectangle((chosen_xi - 0.5, chosen_yi - 0.5), 1, 1,
                                   fill=False, edgecolor='black', lw=2))
    plt.colorbar(im, ax=ax, label='Silhouette score')
    plt.tight_layout()
    fig.savefig(stratum_dir / 'silhouette_matrix.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {stratum_dir / 'silhouette_matrix.png'}")

    # ── Plot 2: cluster sizes ─────────────────────────────────────────────────
    n_total = len(labels)
    clabels = [f'C{k}' for k in unique_labs]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(clabels, counts, color='steelblue', alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{cnt}  ({100 * cnt / n_total:.0f}%)', va='center', fontsize=10)
    ax.set_xlabel('Count')
    ax.set_title(f'Cluster sizes — {stratum_name}')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(stratum_dir / 'cluster_sizes.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {stratum_dir / 'cluster_sizes.png'}")

    # ── Plot 3: peak customers boxplot ────────────────────────────────────────
    box_data  = [peak_s[labels == k] for k in range(K)]
    box_ticks = [f'C{k}' for k in range(K)]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(box_data, labels=box_ticks, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6),
               medianprops=dict(color='firebrick', lw=2))
    ax.set_ylabel('Peak customers out')
    ax.set_title(f'Outage severity — {stratum_name}')
    ax.grid(axis='y', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(stratum_dir / 'cluster_peak_customers.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {stratum_dir / 'cluster_peak_customers.png'}")

    # ── Plot 4: centroid heatmap ──────────────────────────────────────────────
    feat_global_mean = np.nanmean(X_sel_f, axis=0)
    centroid_rel     = centroids - feat_global_mean
    vmax_c = np.abs(centroid_rel).max()
    col_labels = [_VAR_LABELS.get(n, n) for n in sel_names]
    row_labels = [f'C{k}' for k in range(K)]

    fig, ax = plt.subplots(figsize=(max(6, len(sel_names) * 1.1), K * 0.8 + 1.5))
    im = ax.imshow(centroid_rel, cmap='coolwarm', vmin=-vmax_c, vmax=vmax_c, aspect='auto')
    ax.set_xticks(range(len(sel_names)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(K))
    ax.set_yticklabels(row_labels)
    ax.set_title(f'Cluster centroids — {stratum_name}')
    for ki in range(K):
        for fi in range(len(sel_names)):
            ax.text(fi, ki, f'{centroids[ki, fi]:.2g}',
                    ha='center', va='center', fontsize=7, color='black')
    plt.colorbar(im, ax=ax, label='Deviation from mean')
    plt.tight_layout()
    fig.savefig(stratum_dir / 'centroid_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {stratum_dir / 'centroid_heatmap.png'}")

    # ── Plot 5: silhouette analysis ───────────────────────────────────────────
    # [Rousseeuw, 1987, J. Comput. Appl. Math.]
    sil_samples    = silhouette_samples(X_scaled_f, labels)
    mean_sil       = sil_samples.mean()
    cluster_colors = plt.cm.tab10(np.linspace(0, 0.9, K))

    fig, ax = plt.subplots(figsize=(7, max(4, n_s * 0.06)))
    y_lower = 0
    for k in range(K):
        vals_k  = np.sort(sil_samples[labels == k])
        y_upper = y_lower + vals_k.shape[0]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_k,
                         facecolor=cluster_colors[k], alpha=0.75, label=f'C{k}')
        y_lower = y_upper + 2
    ax.axvline(mean_sil, color='firebrick', lw=1.5, ls='--',
               label=f'mean = {mean_sil:.3f}')
    ax.set_xlabel('Silhouette coefficient')
    ax.set_ylabel('Event (sorted by cluster)')
    ax.set_title(f'Silhouette analysis — {stratum_name}')
    ax.set_yticks([])
    ax.legend(fontsize=9, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(stratum_dir / 'silhouette_analysis.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {stratum_dir / 'silhouette_analysis.png'}")

    # ── Plot 6: county composition ────────────────────────────────────────────
    county_colors = {0: 'steelblue', 1: 'firebrick'}
    county_labels = {0: 'Garfield', 1: 'Noble'}

    fig, ax = plt.subplots(figsize=(5, 4))
    bottom = np.zeros(K)
    for ci in sorted(county_labels.keys()):
        if ci not in crosstab.columns:
            continue
        fracs = np.array([
            crosstab.loc[k, ci] / crosstab.loc[k].sum()
            if k in crosstab.index else 0.0
            for k in range(K)
        ])
        ax.bar(range(K), fracs, bottom=bottom,
               color=county_colors[ci], alpha=0.85, label=county_labels[ci])
        bottom += fracs
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'C{k}' for k in range(K)])
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 1)
    ax.set_title(f'County composition — {stratum_name}')
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(stratum_dir / 'county_composition.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {stratum_dir / 'county_composition.png'}")
