import os
import sys
import tkinter
import tkinter.filedialog
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
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
    'k_range':      range(2, 12),
    'n_feat_range': range(2, 12),
    'random_state': 42,
    'min_rf_pred':  0.5,   # minimum OOF RF probability to include an event
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

if 'rf_prediction' in ds:
    rf_pred   = ds['rf_prediction'].values.astype(float)
    sel_mask  = is_outage & (rf_pred >= CFG['min_rf_pred'])
    print(f"RF threshold >= {CFG['min_rf_pred']}: "
          f"{sel_mask.sum()} / {is_outage.sum()} outage events kept")
else:
    sel_mask = is_outage
    print("Warning: rf_prediction not found; using all outage events")

X_out      = feat_arr[sel_mask]
events_out = events[sel_mask]
peak_out   = peak_pk[sel_mask]

# ── Global SHAP importance (feature selection) ────────────────────────────────
shap_arr = ds['shap_values'].values
shap_out = shap_arr[sel_mask]

# ── County indicator ──────────────────────────────────────────────────────────
county_idx = feat_names.index('county') if 'county' in feat_names else None
county_out = (X_out[:, county_idx].astype(int)
              if county_idx is not None
              else np.zeros(len(X_out), dtype=int))

# ── Additional columns (e.g. rain) — reported per cluster, not used for clustering
if 'additional_col_means' in ds and 'additional_col' in ds.coords:
    add_col_names = list(ds.additional_col.values)
    add_col_arr   = ds['additional_col_means'].values.astype(float)  # (event, add_col)
    add_col_out   = add_col_arr[sel_mask]
else:
    add_col_names = []
    add_col_out   = np.empty((int(sel_mask.sum()), 0))

# exclude 'county' — site label, not a meteorological predictor
met_idx = [i for i, n in enumerate(feat_names) if n != 'county']

# ── SHAP importance ───────────────────────────────────────────────────────────
shap_imp = np.mean(np.abs(shap_out[:, met_idx]), axis=0)

n_s = len(X_out)
print(f"\n══ Clustering {n_s} outage events ══")

# ── 2-D silhouette sweep (n_features × K) ────────────────────────────────────
# [Rousseeuw, 1987, J. Comput. Appl. Math.]
n_feat_list = list(CFG['n_feat_range'])
k_list      = list(CFG['k_range'])
sil_matrix  = pd.DataFrame(np.nan, index=n_feat_list, columns=k_list)

for n_top in n_feat_list:
    n_top_eff  = min(n_top, len(met_idx))
    top_idx_n  = np.argsort(shap_imp)[::-1][:n_top_eff]
    sel_idx_n  = [met_idx[i] for i in top_idx_n]
    X_scaled_n = StandardScaler().fit_transform(X_out[:, sel_idx_n])
    for k in k_list:
        if k >= n_s:
            continue
        km_n     = KMeans(n_clusters=k, random_state=CFG['random_state'], n_init=20)
        labels_n = km_n.fit_predict(X_scaled_n)
        if len(set(labels_n)) > 1:
            sil_matrix.loc[n_top, k] = silhouette_score(X_scaled_n, labels_n)

sil_matrix.to_csv(out_dir / 'silhouette_matrix.csv')
print(f"Saved → {out_dir / 'silhouette_matrix.csv'}")

# ── Auto-select best (n_features, K) from silhouette heatmap ─────────────────
best_n_top, best_k = sil_matrix.stack().idxmax()
best_score = sil_matrix.loc[best_n_top, best_k]
print(f"Best silhouette: n_features={best_n_top}, K={best_k}, score={best_score:.3f}")

# ── Final k-means with auto-selected parameters ───────────────────────────────
K         = best_k
n_top_f   = min(best_n_top, len(met_idx))
top_idx_f = np.argsort(shap_imp)[::-1][:n_top_f]
sel_idx_f = [met_idx[i] for i in top_idx_f]
sel_names = [feat_names[i] for i in sel_idx_f]
print(f"Selected features: {sel_names}")

X_sel_f    = X_out[:, sel_idx_f]
scaler_f   = StandardScaler()
X_scaled_f = scaler_f.fit_transform(X_sel_f)

km_f             = KMeans(n_clusters=K, random_state=CFG['random_state'], n_init=20)
labels           = km_f.fit_predict(X_scaled_f)
centroids_scaled = km_f.cluster_centers_
centroids        = scaler_f.inverse_transform(centroids_scaled)

unique_labs, counts = np.unique(labels, return_counts=True)
print(f"Cluster sizes: { {f'C{k}': int(c) for k, c in zip(unique_labs, counts)} }")

crosstab = pd.crosstab(pd.Series(labels, name='cluster'),
                       pd.Series(county_out, name='county'))

# ── Plot 1: silhouette score heatmap ─────────────────────────────────────────
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
ax.set_title('Silhouette score')
for ri, n_top in enumerate(n_feat_list):
    for ci, k in enumerate(k_list):
        v = sil_matrix.loc[n_top, k]
        if not np.isnan(v):
            ax.text(ci, ri, f'{v:.2f}', ha='center', va='center', fontsize=6)
chosen_xi = k_list.index(best_k)      if best_k      in k_list      else None
chosen_yi = n_feat_list.index(best_n_top) if best_n_top in n_feat_list else None
if chosen_xi is not None and chosen_yi is not None:
    ax.add_patch(plt.Rectangle((chosen_xi - 0.5, chosen_yi - 0.5), 1, 1,
                               fill=False, edgecolor='black', lw=2))
plt.colorbar(im, ax=ax, label='Silhouette score')
plt.tight_layout()
fig.savefig(out_dir / 'silhouette_matrix.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'silhouette_matrix.png'}")

# ── Plot 2: cluster sizes ─────────────────────────────────────────────────────
n_total = len(labels)
clabels = [f'C{k}' for k in unique_labs]

fig, ax = plt.subplots(figsize=(5, 3))
bars = ax.barh(clabels, counts, color='steelblue', alpha=0.85)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{cnt}  ({100 * cnt / n_total:.0f}%)', va='center', fontsize=10)
ax.set_xlabel('Count')
ax.set_title('Cluster sizes')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(out_dir / 'cluster_sizes.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'cluster_sizes.png'}")

# ── Plot 3: peak customers boxplot ────────────────────────────────────────────
box_data  = [peak_out[labels == k] for k in range(K)]
box_ticks = [f'C{k}' for k in range(K)]

fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot(box_data, labels=box_ticks, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.6),
           medianprops=dict(color='firebrick', lw=2))
ax.set_ylabel('Peak customers out')
ax.set_title('Outage severity')
ax.grid(axis='y', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(out_dir / 'cluster_peak_customers.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'cluster_peak_customers.png'}")

# ── Plot 4: centroid heatmap ──────────────────────────────────────────────────
feat_global_mean = np.nanmean(X_sel_f, axis=0)
centroid_rel     = centroids - feat_global_mean
col_labels       = [_VAR_LABELS.get(n, n) for n in sel_names]
n_clust_cols     = len(sel_names)

# append additional column means per cluster (deviation from global mean)
if add_col_names:
    add_global_mean  = np.nanmean(add_col_out, axis=0)
    add_per_cluster  = np.array([add_col_out[labels == k].mean(axis=0) for k in range(K)])
    centroid_rel     = np.column_stack([centroid_rel, add_per_cluster - add_global_mean])
    centroids        = np.column_stack([centroids, add_per_cluster])
    col_labels       = col_labels + [_VAR_LABELS.get(n, n) for n in add_col_names]

vmax_c     = np.abs(centroid_rel).max()
row_labels = [f'C{k}' for k in range(K)]
n_cols     = centroid_rel.shape[1]

fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.1), K * 0.8 + 1.5))
im = ax.imshow(centroid_rel, cmap='coolwarm', vmin=-vmax_c, vmax=vmax_c, aspect='auto')
ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(K))
ax.set_yticklabels(row_labels)
ax.set_title('Cluster centroids')
for ki in range(K):
    for fi in range(n_cols):
        ax.text(fi, ki, f'{centroids[ki, fi]:.2g}',
                ha='center', va='center', fontsize=7, color='black')
# vertical separator between clustering features and additional columns
if add_col_names:
    ax.axvline(n_clust_cols - 0.5, color='black', lw=1.5, ls='--')
plt.colorbar(im, ax=ax, label='Deviation from mean')
plt.tight_layout()
fig.savefig(out_dir / 'centroid_heatmap.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'centroid_heatmap.png'}")

# ── Plot 5: silhouette analysis ───────────────────────────────────────────────
# [Rousseeuw, 1987, J. Comput. Appl. Math.]
sil_samples    = silhouette_samples(X_scaled_f, labels)
mean_sil       = sil_samples.mean()
cluster_colors = ['r','k','b','c']

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
ax.set_title('Silhouette analysis')
ax.set_yticks([])
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(out_dir / 'silhouette_analysis.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'silhouette_analysis.png'}")

# ── Plot 6: county composition ────────────────────────────────────────────────
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
ax.set_title('County composition')
ax.legend()
ax.grid(axis='y', alpha=0.4)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(out_dir / 'county_composition.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'county_composition.png'}")

# ── Plot 7: pair plot (upper triangle) ───────────────────────────────────────
n_f      = len(sel_names)
f_labels = [_VAR_LABELS.get(n, n) for n in sel_names]

rain_add_idx = next((i for i, n in enumerate(add_col_names) if 'rain' in n.lower()), None)
rainy_flag   = (add_col_out[:, rain_add_idx] > 0
                if rain_add_idx is not None
                else np.zeros(len(X_sel_f), dtype=bool))

fig, axes = plt.subplots(n_f, n_f, figsize=(n_f * 2.2, n_f * 2.2))
for i in range(n_f):
    for j in range(n_f):
        ax = axes[i, j]
        if j < i:
            ax.set_visible(False)
            continue
        if i == j:
            for k in range(K):
                ax.hist(X_sel_f[labels == k, i], bins=12, alpha=0.5,
                        color=cluster_colors[k], density=True)
            ax.set_xlabel(f_labels[i], fontsize=9)
        else:  # upper triangle
            for k in range(K):
                mask_k = labels == k
                ax.scatter(X_sel_f[mask_k, j], X_sel_f[mask_k, i],
                           s=20, alpha=0.6, color=cluster_colors[k])
                rain_mask = mask_k & rainy_flag
                if rain_mask.any():
                    ax.scatter(X_sel_f[rain_mask, j], X_sel_f[rain_mask, i],
                               s=70, facecolors='none', edgecolors='green',
                               linewidths=1.2, zorder=5)
        ax.tick_params(labelsize=7)

handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=cluster_colors[k], markersize=8, label=f'C{k}')
           for k in range(K)]
if rainy_flag.any():
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                              markeredgecolor='green', markeredgewidth=0.5,
                              markersize=5, label='rain > 0'))
fig.legend(handles=handles, loc='lower left', fontsize=9)
fig.suptitle('Pair plot — clustering features', fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(out_dir / 'pair_plot.png', bbox_inches='tight')
plt.close(fig)
print(f"Saved → {out_dir / 'pair_plot.png'}")


