import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from pathlib import Path
import yaml
import joblib
import matplotlib
import matplotlib.dates as mdates
import matplotlib.animation as mpl_animation
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (StratifiedKFold, train_test_split,
                                     StratifiedGroupKFold)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from nexradaws import NexradAwsInterface
from datetime import timedelta, timezone
import pyart
from doe_dap_dl import DAP
from functools import lru_cache

plt.close('all')
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Run `pip install shap` for SHAP importance.")

import re
from xml.etree import ElementTree as ET

_VAR_LABELS = {
    'relh': 'Relative humidity [%]',
    'tair': 'Air temperature [°C]',
    'wspd': 'Wind speed [m/s]',
    'wssd': 'Wind speed std [m/s]',
    'wdsd': 'Wind dir. std [°]',
    'aavi': 'Wind variability \n index',
    'wmax': 'Maximum \n wind speed [m/s]',
    'rain': 'Precipitation [mm]',
    'pres': 'Pressure [hPa]',
    'srad': 'Solar radiation [W/m²]',
    'TURB': 'Turbulence \n intensity [%]',
    'customers_out':' Customers out',
}

_AGG_LABELS = {
    'grad_std':  'peak',
    'grad_mean': 'ramp',
    'std':       'variability',
    'mean':      'value',
}


def _feat_label(name: str) -> str:
    """Convert a dynamic feature name to a human-readable label."""
    m = re.match(r'^(.+?)_(grad_std|grad_mean|std|mean)_W\d+$', name)
    if m:
        base, agg = m.group(1), m.group(2)
        return f'{_VAR_LABELS.get(base, base)} ({_AGG_LABELS[agg]})'
    return _VAR_LABELS.get(name, name)


def _kml_text(kml_el, default=''):
    if kml_el is None or kml_el.text is None:
        return default
    return str(kml_el.text).strip()


def load_turbine_points(map_dir) -> pd.DataFrame:
    map_dir_p = Path(map_dir)
    rows = []
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    for kml_path in sorted(map_dir_p.glob('*.kml')):
        farm_name = kml_path.stem
        try:
            root = ET.parse(kml_path).getroot()
        except Exception:
            continue

        placemarks = root.findall('.//kml:Placemark', ns)
        for pm in placemarks:
            coord_el = pm.find('.//kml:Point/kml:coordinates', ns)
            coord_text = _kml_text(coord_el)
            if not coord_text:
                continue

            first_coord = coord_text.split()[0]
            parts = [p.strip() for p in first_coord.split(',')]
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except Exception:
                continue

            turbine_id = _kml_text(pm.find('.//kml:SimpleData[@name="turbine_id"]', ns), default='')
            if not turbine_id:
                turbine_id = _kml_text(pm.find('kml:name', ns), default='')

            rows.append({
                'farm': farm_name,
                'turbine_id': turbine_id,
                'latitude': lat,
                'longitude': lon,
            })

    if not rows:
        return pd.DataFrame(columns=['farm', 'turbine_id', 'latitude', 'longitude'])

    df_turb = pd.DataFrame(rows)
    df_turb['latitude'] = pd.to_numeric(df_turb['latitude'], errors='coerce')
    df_turb['longitude'] = pd.to_numeric(df_turb['longitude'], errors='coerce')
    df_turb = df_turb[
        df_turb['latitude'].between(-90, 90) &
        df_turb['longitude'].between(-180, 180)
    ].reset_index(drop=True)
    return df_turb


def _uf_find(x, parent):
    """Path-compressed union-find lookup."""
    if x not in parent:
        parent[x] = x
    if parent[x] != x:
        parent[x] = _uf_find(parent[x], parent)
    return parent[x]

def _uf_union(x, y, parent):
    """Union-find merge of two elements."""
    px, py = _uf_find(x, parent), _uf_find(y, parent)
    if px != py:
        parent[px] = py


def _build_groups_arr(episode_ends_per_county: list, county_rows: list,
                      events_index: pd.DatetimeIndex, pre_dt, post_dt) -> np.ndarray:
    """Assign a group ID to every event so temporally overlapping cross-county episodes share a fold. [Roberts et al., 2017, Ecography]"""
    parent = {}
    for ci, ep in enumerate(episode_ends_per_county):
        for ts in ep:
            _uf_find((ci, ts), parent)

    for ci in range(len(episode_ends_per_county)):
        for cj in range(ci + 1, len(episode_ends_per_county)):
            for ts_i, te_i in episode_ends_per_county[ci].items():
                for ts_j, te_j in episode_ends_per_county[cj].items():
                    if te_i + post_dt > ts_j - pre_dt and ts_i - pre_dt < te_j + post_dt:
                        _uf_union((ci, ts_i), (cj, ts_j), parent)

    root_to_id, next_id = {}, 0
    outage_group = {}
    for ci, ep in enumerate(episode_ends_per_county):
        for ts in ep:
            root = _uf_find((ci, ts), parent)
            if root not in root_to_id:
                root_to_id[root] = next_id
                next_id += 1
            outage_group[(ci, ts)] = root_to_id[root]

    singleton = next_id
    groups_list = []
    for ci, ts in zip(county_rows, events_index):
        key = (ci, pd.Timestamp(ts))
        if key in outage_group:
            groups_list.append(outage_group[key])
        else:
            groups_list.append(singleton)
            singleton += 1
    return np.array(groups_list)


def _find_episodes(series: pd.Series, threshold: float) -> list[tuple]:
    """Find contiguous episodes where series > threshold."""
    flag = series > threshold
    block_id = (flag != flag.shift()).cumsum()[flag]
    return [(g.index[0], g.index[-1]) for _, g in flag.groupby(block_id)]


def _merge_close_episodes(episodes: list[tuple], merge_gap: pd.Timedelta) -> list[tuple]:
    """Merge consecutive episodes whose inter-event gap is smaller than merge_gap."""
    if not episodes:
        return episodes
    merged = [[episodes[0][0], episodes[0][1]]]
    for t0, t1 in episodes[1:]:
        if t0 - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [(t0, t1) for t0, t1 in merged]


def plot_segment_zoom(series: pd.Series, target: pd.Series, threshold: float,
                      mode: str, save_path: Path = None):
    """Zoom into the densest outage period, showing raw signal, threshold, and metric."""
    above = series > threshold
    dt = series.index.to_series().diff().median()
    pts_per_day = max(1, int(pd.Timedelta(days=1) / dt))
    win = min(30 * pts_per_day, len(series) // 3)
    center_idx = above.rolling(win, center=True).sum().idxmax()
    half = pd.Timedelta(days=15)
    t0 = max(series.index[0], center_idx - half)
    t1 = min(series.index[-1], center_idx + half)
    mask = (series.index >= t0) & (series.index <= t1)

    s = series[mask]
    t = target[mask]
    above_zoom = above[mask]
    seg_id = (above_zoom != above_zoom.shift()).cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(s.index, s.values, color='gray', lw=0.5)
    axes[0].axhline(threshold, color='black', lw=1, ls='--',
                    label=f'threshold = {threshold}')
    for _, idx in above_zoom.groupby(seg_id).groups.items():
        if above_zoom[idx[0]]:
            axes[0].axvspan(idx[0], idx[-1], alpha=0.3, color='red')
    axes[0].set_ylabel('Customers out')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].grid()

    units = 'customer·h' if mode == 'AUC' else 'min'
    axes[1].plot(t.index, t.values, color='steelblue', lw=1)
    axes[1].set_ylabel(f'{mode} ({units})')
    axes[1].grid()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved segment zoom plot → {save_path}")


def plot_time_series(df: pd.DataFrame, config: dict, save_path: Path = None):
    """Time series scatter plot of predictors colored by target."""
    plt.figure(figsize=(18, 10))
    ctr = 1
    flag = df[config['target_col']] > config.get('outage_threshold', 0)
    for col in config['predictor_cols']:
        plt.subplot(len(config['predictor_cols']), 1, ctr)
        plt.scatter(df.index[~flag], df[col][~flag], s=1, c='green')
        plt.scatter(df.index[flag], df[col][flag], s=1, c='red')
        plt.ylabel(col)
        plt.xlim([df.index[0], df.index[-1]])
        plt.grid()
        ctr += 1
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved time series plot → {save_path}")


def remove_seasonal_cycle(df: pd.DataFrame,
                          columns: list,
                          window_days: int = 7,
                          min_periods: int = 3,
                          inplace: bool = False,
                          save_climatology_path: Path = None
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove seasonal and diurnal cycles by subtracting a smoothed climatological
    mean grouped by (day-of-year, time-of-day). [Wilks, 2011]
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a pd.DatetimeIndex.")

    df_out = df if inplace else df.copy()

    doy = df_out.index.day_of_year
    tod = df_out.index.hour * 3600 + df_out.index.minute * 60 + df_out.index.second

    df_out["__doy__"] = doy
    df_out["__tod__"] = tod

    raw_clim = (
        df_out.groupby(["__doy__", "__tod__"])[columns]
        .mean()
    )

    tod_levels = raw_clim.index.get_level_values("__tod__").unique()
    smoothed_parts = []

    for t in tod_levels:
        subset = raw_clim.xs(t, level="__tod__")
        full_doy = pd.RangeIndex(1, 367, name="__doy__")
        subset = subset.reindex(full_doy)

        smoothed = subset.rolling(
            window=window_days, center=True, min_periods=min_periods,
        ).mean()

        half = window_days // 2
        padded = pd.concat([subset.iloc[-half:], subset, subset.iloc[:half]])
        smoothed_wrap = padded.rolling(
            window=window_days, center=True, min_periods=min_periods,
        ).mean().iloc[half: half + 366]
        smoothed_wrap.index = full_doy

        smoothed = smoothed.combine_first(smoothed_wrap)
        smoothed["__tod__"] = t
        smoothed_parts.append(smoothed.reset_index())

    climatology = (
        pd.concat(smoothed_parts, ignore_index=True)
        .set_index(["__doy__", "__tod__"])
        .sort_index()
    )

    nan_cols = climatology.columns[climatology.isna().any()].tolist()
    if nan_cols:
        print(f"  Warning: climatology NaN in {nan_cols} for some (doy, tod) bins "
              f"(min_periods={min_periods} not met). Original values retained.")

    keys = list(zip(doy, tod))
    key_index = pd.MultiIndex.from_tuples(keys, names=["__doy__", "__tod__"])
    clim_aligned = climatology.reindex(key_index).values

    out_col_names = columns if inplace else [f"{c}_anom" for c in columns]

    for i, (orig_col, out_col) in enumerate(zip(columns, out_col_names)):
        raw_vals = df_out[orig_col].values.astype(float)
        clim_vals = clim_aligned[:, i]
        anom = raw_vals - clim_vals
        nan_mask = np.isnan(clim_vals)
        anom[nan_mask] = raw_vals[nan_mask]
        df_out[out_col] = anom

    df_out.drop(columns=["__doy__", "__tod__"], inplace=True)

    if save_climatology_path is not None:
        climatology.to_csv(save_climatology_path)
        print(f"  Saved climatology → {save_climatology_path}")

    return df_out, climatology


def apply_climatology(df: pd.DataFrame, climatology: pd.DataFrame,
                      columns: list, inplace: bool = False) -> pd.DataFrame:
    """Apply a pre-computed climatology to detrend a DataFrame. [Wilks, 2011]"""
    df_out = df if inplace else df.copy()
    cols = [c for c in columns if c in climatology.columns]
    doy = df_out.index.day_of_year
    tod = df_out.index.hour * 3600 + df_out.index.minute * 60 + df_out.index.second
    key_index = pd.MultiIndex.from_tuples(zip(doy, tod), names=['__doy__', '__tod__'])
    clim_vals = climatology[cols].reindex(key_index).values
    for i, col in enumerate(cols):
        raw = df_out[col].values.astype(float)
        clim = clim_vals[:, i]
        anom = raw - clim
        anom[np.isnan(clim)] = raw[np.isnan(clim)]
        df_out[col] = anom
    return df_out


def plot_seasonal_detrending(df_raw: pd.DataFrame,
                             df_anom: pd.DataFrame,
                             climatology: pd.DataFrame,
                             column: str,
                             save_path: Path = None):
    anom_col = column if column in df_anom.columns else f"{column}_anom"

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

    ax = axes[0]
    ax.plot(df_raw.index, df_raw[column], lw=0.4, alpha=0.6,
            color="#2980b9", label="Raw signal")
    doy = df_raw.index.day_of_year
    tod = (df_raw.index.hour * 3600
           + df_raw.index.minute * 60
           + df_raw.index.second)
    keys = list(zip(doy, tod))
    key_index = pd.MultiIndex.from_tuples(keys, names=["__doy__", "__tod__"])
    clim_vals = climatology[column].reindex(key_index).values
    ax.plot(df_raw.index, clim_vals, lw=1.2, color="#e74c3c", label="Climatology")
    ax.set_ylabel(column)
    ax.set_title(f"{column} — Raw signal and climatology")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(df_anom.index, df_anom[anom_col], lw=0.4, alpha=0.6, color="#27ae60")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel(f"{column} anomaly")
    ax.set_title("Anomaly (detrended signal)")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    tod_levels = climatology.index.get_level_values("__tod__").unique()
    noon_sec = 12 * 3600
    nearest_noon = tod_levels[np.argmin(np.abs(tod_levels - noon_sec))]
    clim_doy = climatology.xs(nearest_noon, level="__tod__")[column]
    ax.plot(clim_doy.index, clim_doy.values, color="#e67e22", lw=1.5)
    ax.set_xlabel("Day of year")
    ax.set_ylabel(f"Climatological mean\n({column})")
    ax.set_title("Smoothed climatology vs day-of-year  (noon values)")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved detrending diagnostic → {save_path}")


def plot_histograms(df: pd.DataFrame, config: dict, lag: dict = None,
                    window: dict = None, save_path: Path = None):
    """Scatter plot of top predictors vs target."""
    y_series = df[config['target_col']]
    y_label = config['target_col']

    y_lim = (np.nanpercentile(y_series, config['perc_bins'][0]),
             np.nanpercentile(y_series, config['perc_bins'][1]))

    fig, axes = plt.subplots(config['nrow'], config['ncol'],
                             figsize=(5 * config['ncol'], 4 * config['nrow']),
                             sharey=True, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ctr, col in enumerate(config['predictor_cols']):
        if window is not None and lag is not None:
            if window[col] > 0:
                shifted = df[col].rolling(window[col], center=True).mean().shift(lag[col])
            else:
                shifted = df[col].shift(lag[col])
        else:
            shifted = df[col]

        ax = axes[ctr]
        ax.scatter(shifted, y_series, s=2, alpha=0.1, color='k')
        ax.set_ylim(y_lim)
        if window is not None and lag is not None:
            if window[col] > 0:
                ax.set_xlabel(f"{col}_avg (roll={window[col]}, lag={lag[col]})")
            else:
                ax.set_xlabel(f"{col} (lag={lag[col]})")
        else:
            ax.set_xlabel(f"{col}")
        if ctr % config['ncol'] == 0:
            ax.set_ylabel(y_label)
        ax.grid()

    for ax in axes[len(config['predictor_cols']):]:
        ax.set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved scatter plot → {save_path}")


def cross_lag_correlation(df: pd.DataFrame, predictors: list, target: pd.Series,
                          lag_list: list) -> dict:
    """Compute Spearman correlation at each lag. [von Storch & Zwiers, 1999]"""
    mirror_lags = np.unique(np.concat([-np.array(lag_list), np.array(lag_list)]))
    records = []
    for lag in mirror_lags:
        row = {"lag": lag}
        for col in predictors:
            shifted = df[col].shift(lag)
            valid = ~(shifted.isna() | target.isna())
            row[col] = shifted[valid].corr(target[valid])
        records.append(row)
    return {0: pd.DataFrame(records).set_index("lag")}


def select_best_lag(corr_df: dict, config: dict) -> dict[str, int]:
    """Select lag with maximum absolute correlation for each predictor."""
    subset = corr_df[0]
    lag_max = {}
    for col in subset.columns:
        if subset[col].abs().max() >= config['min_corr']:
            lag_max[col] = int(subset[col].abs().idxmax())
        else:
            lag_max[col] = 0
    return lag_max


def plot_lag_correlation(corr_df: dict, save_path: Path = None,
                         target_name: str = "target"):
    """Heatmap of Spearman r vs lag."""
    subset = corr_df[0]
    max_corr = subset.abs().max().max()

    fig, ax = plt.subplots(figsize=(max(18, len(subset.columns) * 0.8), 10))
    im = ax.imshow(subset.T.values, aspect="auto", cmap="RdBu_r",
                   vmin=-max_corr, vmax=max_corr)
    for j in range(len(subset.columns)):
        for i in range(len(subset.index)):
            color = 'g' if subset.iloc[i, j] > 0 else 'orange'
            plt.text(i - 0.25, j, f"{subset.iloc[i, j]:02.3f}",
                     fontsize=8, color=color)
    ax.set_xticks(range(len(subset.index)))
    ax.set_xticklabels(subset.index)
    ax.set_yticks(range(len(subset.columns)))
    ax.set_yticklabels(subset.columns)
    ax.set_xlabel("Lag (time steps)")
    ax.set_title(f"Cross-Lag Correlation: Predictor[t-lag] vs {target_name}[t]")
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved lag correlation plot → {save_path}")


def build_feature_matrix(df: pd.DataFrame, predictors: list, target_col: str,
                         lag: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix with lag-shifted predictors."""
    frames = {f"{col} (lag={lag[col]})": df[col].shift(lag[col]) for col in predictors}
    X = pd.DataFrame(frames, index=df.index)
    y = df[target_col]
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


def build_dynamic_features(df: pd.DataFrame, predictors: list, target_col: str = None,
                            window: int = 0, rolling: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate features over a window (mean, std, gradient). [Bossavy et al., 2013; Bianco et al., 2016]"""
    frames = {}
    w = window
    for col in predictors:
        x = df[col]
        grad = x.diff()
        frames[f"{col}_mean_W{w}"]      = x.mean()  # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
        frames[f"{col}_std_W{w}"]       = x.std()                            
        frames[f"{col}_grad_mean_W{w}"] = grad.mean()
        frames[f"{col}_grad_std_W{w}"]  = grad.std()
    y = df[target_col] if target_col is not None else None
    if rolling:
        return pd.DataFrame(frames, index=df.index), y
    else:
        return pd.DataFrame([frames]), y

def train_rf_importance(X: pd.DataFrame, y: pd.Series, config: dict,
                        groups: np.ndarray = None) -> dict:
    """Train RF classifier with CV and return permutation importances. [Breiman, 2001]"""
    n_folds = config["n_cv_folds"]
    model_cls = RandomForestClassifier
    score_name = "ROC-AUC"
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                  random_state=config["random_state"])
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=config["random_state"])

    X_arr = X.values
    y_arr = y.values
    imp_list = []
    cv_scores = []
    oof_pred = np.full(len(y_arr), np.nan)
    do_shap = config.get('shap', False) and HAS_SHAP
    oof_shap = np.full((len(y_arr), X_arr.shape[1]), np.nan) if do_shap else None
    shap_base_vals = []

    split_kwargs = {'groups': groups} if groups is not None else {}
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr, **split_kwargs)):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        model = model_cls(
            n_estimators=config["n_estimators"],
            max_features=config["max_features"],
            n_jobs=config["n_jobs"],
            random_state=config["random_state"],
        )

        cal_frac = config.get("calibration_fraction", 0.3)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr, y_tr, test_size=cal_frac, stratify=y_tr,
            random_state=config["random_state"],
        )
        sw_fit = compute_sample_weight("balanced", y_fit)
        model.fit(X_fit, y_fit, sample_weight=sw_fit)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(model.predict_proba(X_cal)[:, 1], y_cal)
        val_prob = iso.predict(model.predict_proba(X_val)[:, 1])
        cv_scores.append(roc_auc_score(y_val, val_prob))
        oof_pred[val_idx] = val_prob

        perm = permutation_importance(model, X_val, y_val,
                                      n_repeats=config["importance_reps"],
                                      random_state=config["random_state"],
                                      n_jobs=config["n_jobs"])
        imp_list.append(perm.importances_mean)
        print(f"    Fold {fold_i + 1}/{n_folds} | {score_name}: {cv_scores[-1]:.4f}")

        # SHAP on validation fold using the same fold model [Lundberg & Lee, 2017]
        # For binary mode, SHAP sums to the uncalibrated RF probability (not isotonic-calibrated).
        if do_shap:
            explainer_fold = shap.TreeExplainer(model)
            sv = explainer_fold.shap_values(X_val)
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            oof_shap[val_idx] = sv
            base = explainer_fold.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = float(base[1]) if len(base) > 1 else float(base[0])
            shap_base_vals.append(float(base))

    mean_imp = np.mean(imp_list, axis=0)
    std_imp = np.std(imp_list, axis=0)
    print(f"  Mean CV {score_name}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    result = {
        "feature_names": list(X.columns),
        "importance_mean": mean_imp,
        "importance_std": std_imp,
        "cv_scores": cv_scores,
        "score_name": score_name,
        "oof_pred": pd.Series(oof_pred, index=y.index),
    }
    if do_shap:
        result["shap_df"]   = pd.DataFrame(oof_shap, index=y.index, columns=X.columns)
        result["shap_base"] = float(np.mean(shap_base_vals))
        result["shap_imp"]  = pd.Series(np.abs(oof_shap).mean(axis=0),
                                        index=X.columns, name="mean_abs_SHAP")
    return result


def train_rf_full(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    """Train RF classifier on full data and compute importance. [Breiman, 2001; Lundberg & Lee, 2017]"""
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_features=config["max_features"],
        n_jobs=config["n_jobs"],
        random_state=config["random_state"],
    )
    sw = compute_sample_weight("balanced", y.values)
    X_arr, y_arr = X.values, y.values
    model.fit(X_arr, y_arr, sample_weight=sw)

    perm = permutation_importance(
        model, X_arr, y_arr,
        n_repeats=config["importance_reps"],
        random_state=config["random_state"],
        n_jobs=config["n_jobs"],
    )

    result = {
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }

    if config.get("shap", False) and HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_arr)
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(base[1]) if len(base) > 1 else float(base[0])
        result["shap_df"]   = pd.DataFrame(sv, index=X.index, columns=X.columns)
        result["shap_base"] = float(base)
        result["shap_imp"]  = pd.Series(np.abs(sv).mean(axis=0),
                                        index=X.columns, name="global_mean_abs_SHAP")

    return result


def compute_shap_importance(X: pd.DataFrame, y: pd.Series,
                            config: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Compute SHAP importances via TreeExplainer. [Lundberg & Lee, 2017]"""
    if not HAS_SHAP:
        print("  SHAP skipped (not installed).")
        return pd.Series(dtype=float), pd.DataFrame()

    model = RandomForestClassifier(n_estimators=config["n_estimators"],
                                   n_jobs=config["n_jobs"],
                                   random_state=config["random_state"])
    sw = compute_sample_weight("balanced", y)

    model.fit(X, y, sample_weight=sw)
    explainer = shap.TreeExplainer(model)

    shap_values_all = explainer.shap_values(X)
    if isinstance(shap_values_all, list):
        shap_values_all = shap_values_all[1]
    elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
        shap_values_all = shap_values_all[:, :, 1]

    shap_df = pd.DataFrame(shap_values_all, index=X.index, columns=X.columns)
    mean_shap = pd.Series(np.abs(shap_values_all).mean(axis=0),
                          index=X.columns, name="mean_abs_SHAP")

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

    return mean_shap, shap_df, float(base_val)


def plot_shap_dependence(X: pd.DataFrame, shap_df: pd.DataFrame,
                         save_path: Path = None):
    """Dependence plot grid of top SHAP features."""
    mean_shap = shap_df.abs().mean(axis=0)
    sorted_feats = mean_shap.sort_values(ascending=False).index.tolist()
    shap_values = shap_df.values

    n_feats = len(sorted_feats)
    n_cols = 6
    n_feat_rows = int(np.ceil(n_feats / n_cols))
    height_ratios = [4, 1] * n_feat_rows
    fig, axes = plt.subplots(n_feat_rows * 2, n_cols,
                             figsize=(5 * n_cols, 7 * n_feat_rows),
                             gridspec_kw={"height_ratios": height_ratios},
                             constrained_layout=True)
    axes = np.array(axes).reshape(n_feat_rows * 2, n_cols)

    sv_min, sv_max = shap_values.min(), shap_values.max()
    col_order = list(X.columns)

    for feat_idx, feat in enumerate(sorted_feats):
        feat_row = feat_idx // n_cols
        feat_col = feat_idx % n_cols
        orig_idx = col_order.index(feat)
        feat_vals = X.iloc[:, orig_idx]

        ax_shap = axes[feat_row * 2,     feat_col]
        ax_hist = axes[feat_row * 2 + 1, feat_col]

        ax_shap.scatter(feat_vals, shap_values[:, orig_idx], s=5, alpha=0.5, color='k')
        ax_shap.set_ylabel("SHAP value")
        ax_shap.set_ylim(sv_min, sv_max)
        ax_shap.axhline(0, color='black', lw=0.8, ls='--')
        ax_shap.grid(True)

        ax_hist.hist(feat_vals, bins=30, color='b', alpha=0.7)
        ax_hist.set_xlabel(feat)
        ax_hist.grid(True)
        ax_hist.set_yticklabels([])

    for k in range(n_feats, n_feat_rows * n_cols):
        r = (k // n_cols) * 2
        c = k % n_cols
        for row_offset in range(2):
            axes[r + row_offset, c].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved SHAP dependence plot → {save_path}")
    else:
        plt.show()


def plot_shap_waterfall(shap_vals: np.ndarray, feat_vals: np.ndarray,
                        feat_names: list, base_value: float,
                        feat_mean: np.ndarray, feat_std: np.ndarray,
                        save_path: Path = None, title: str = ''):
    if not HAS_SHAP:
        print("  SHAP waterfall skipped (shap not installed).")
        return
    z_scores = (feat_vals - feat_mean) / feat_std
    labels = [re.sub(r'\s*\[.*?\]', '', _feat_label(c)) for c in feat_names]
    expl = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=z_scores,
        feature_names=labels,
    )
    fig = plt.figure()
    shap.plots.waterfall(expl, show=False)
    fig = plt.gcf()  # shap may replace the figure internally
    if title:
        fig.suptitle(title, fontsize=10, y=1.01)
    fig.text(0.5, -0.01, 'Feature values shown as z-scores', ha='center', fontsize=8, color='gray')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved SHAP waterfall → {save_path}")
    else:
        plt.tight_layout()
        plt.show()




def build_results_table(rf_result: dict, shap_imp: pd.Series,
                        predictors: list) -> pd.DataFrame:
    """Merge RF permutation and SHAP importances and rank by RF importance."""
    rf_imp = pd.Series(rf_result["importance_mean"],
                       index=rf_result["feature_names"], name="RF_perm_imp")
    rf_std = pd.Series(rf_result["importance_std"],
                       index=rf_result["feature_names"], name="RF_perm_std")

    result = pd.DataFrame({"RF_perm_importance": rf_imp, "RF_perm_std": rf_std})

    if len(shap_imp) > 0:
        result["SHAP_mean_abs"] = shap_imp

    result = result.sort_values("RF_perm_importance", ascending=False)
    result["RF_rank"] = range(1, len(result) + 1)
    if "SHAP_mean_abs" in result.columns:
        result["SHAP_rank"] = result["SHAP_mean_abs"].rank(ascending=False).astype(int)

    return result


def plot_importance_comparison(results: pd.DataFrame, rf_result: dict,
                               save_path: Path = None, top_n: int = 20):
    """Horizontal bar chart of RF and SHAP importances for top-N features."""
    subset = results.head(top_n)
    n = len(subset)
    cols = [c for c in ["RF_perm_importance", "SHAP_mean_abs"] if c in subset.columns]
    n_cols = len(cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(9 * n_cols, max(5, n * 0.4)),
                             sharey=True)
    if n_cols == 1:
        axes = [axes]

    labels = {"RF_perm_importance": "RF Permutation Importance",
               "SHAP_mean_abs": "Mean |SHAP| Value"}
    colors = ["#2980b9", "#e67e22"]

    for ax, col, color in zip(axes, cols, colors):
        vals = subset[col].values[::-1]
        errs = (subset.get("RF_perm_std", pd.Series(0, index=subset.index))
                .values[::-1] if col == "RF_perm_importance" else None)
        ypos = np.arange(n)
        ax.barh(ypos, vals, xerr=errs, color=color, alpha=0.85,
                error_kw=dict(elinewidth=1, capsize=3))
        ax.set_yticks(ypos)
        ax.set_yticklabels(subset.index[::-1], fontsize=9)
        ax.set_xlabel(labels[col])
        ax.set_title(labels[col])
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Variable Importance\n"
        f"CV {rf_result['score_name']}: {np.mean(rf_result['cv_scores']):.4f}"
        f" ± {np.std(rf_result['cv_scores']):.4f}",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved importance comparison plot → {save_path}")


def plot_top_feature_histograms(X: pd.DataFrame, y: pd.Series, results: pd.DataFrame,
                                config: dict, n_top: int = 5, save_path: Path = None):
    """Scatter plot of top features vs target."""
    top_features = results.index[:n_top].tolist()
    top_features = [f for f in top_features if f in X.columns]

    y_label = config.get('target_col', 'target')
    y_lim = (np.nanpercentile(y, config['perc_bins'][0]),
             np.nanpercentile(y, config['perc_bins'][1]))

    fig, axes = plt.subplots(1, len(top_features),
                             figsize=(4 * len(top_features), 4),
                             sharey=True, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(top_features):
        axes[i].scatter(X[feat], y, s=2, alpha=0.3, color='gray')
        axes[i].set_ylim(y_lim)
        axes[i].set_xlabel(feat)
        axes[i].grid()
        if i == 0:
            axes[i].set_ylabel(y_label)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_episode_ts(df_raw: pd.DataFrame, X: pd.DataFrame,
                    episode: pd.Series, config: dict, out_dir: Path = None,
                    target: pd.Series = None, oof_pred: pd.Series = None,
                    df_raw2: pd.DataFrame = None, X2: pd.DataFrame = None,
                    label1: str = '', label2: str = '',
                    title_extra: str = '',
                    figsize: tuple = None,
                    fontsize: int = 18):

    buffer = pd.Timedelta(hours=config.get('episode_buffer_hours', 12))
    t0 = episode['t_start'] - buffer
    t1 = episode['t_end'] + buffer

    orig_preds = config['predictor_cols']
    target_col = config['target_col']
    n = len(orig_preds) + 1

    dt = df_raw.index.to_series().diff().median()
    pre_w = config.get('pre_window', 0)
    post_w = config.get('post_window', 0)
    ev_start = episode['t_start'] - pre_w * dt
    ev_end = episode['t_start'] + post_w * dt
    has_event_window = pre_w > 0 or post_w > 0

    lbl_raw1 = f'met data ({label1})' if label1 else 'met data'
    lbl_det1 = f'detrended ({label1})' if label1 else 'detrended'
    lbl_raw2 = f'met data ({label2})' if label2 else 'met data'
    lbl_det2 = f'detrended ({label2})' if label2 else 'detrended'

    if figsize is None:
        figsize = (18, 3 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)

    for i, col in enumerate(orig_preds):
        ax = axes[i]
        ax2 = None

        if col in df_raw.columns:
            raw_win = df_raw[col].loc[t0:t1]
            ax.plot(raw_win.index, raw_win.values, color='k', linewidth=1.5, label=lbl_raw1)

        if df_raw2 is not None and col in df_raw2.columns:
            raw_win2 = df_raw2[col].loc[t0:t1]
            ax.plot(raw_win2.index, raw_win2.values, color='k', linewidth=1.5,
                    linestyle='--', label=lbl_raw2)

        feat_col = next((c for c in X.columns if c.startswith(col + '_') or c == col), None)
        if feat_col is not None:
            feat_win = X[feat_col].loc[t0:t1]
            ax2 = ax.twinx()
            ax2.plot(feat_win.index, feat_win.values, color='steelblue',
                     linewidth=1.5, alpha=0.85, label=lbl_det1)

        feat_col2 = (next((c for c in X2.columns if c.startswith(col + '_') or c == col), None)
                     if X2 is not None else None)
        if feat_col2 is not None:
            feat_win2 = X2[feat_col2].loc[t0:t1]
            if ax2 is None:
                ax2 = ax.twinx()
            ax2.plot(feat_win2.index, feat_win2.values, color='steelblue',
                     linewidth=1.5, alpha=0.85, linestyle='--', label=lbl_det2)

        if ax2 is not None:
            ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=fontsize)
            ax2.spines['right'].set_edgecolor('steelblue')

        ax.set_ylabel(_feat_label(col), fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.grid(True, alpha=0.3)
        ax.axvspan(episode['t_start'], episode['t_end'],
                   alpha=0.12, color='red', linewidth=0, label='outage')
        if has_event_window:
            ax.axvspan(ev_start, ev_end, alpha=0.12, color='royalblue',
                       linewidth=0, label='event window' if i == 0 else '_')
        if i == 0:
            h, l = ax.get_legend_handles_labels()
            if ax2 is not None:
                h2, l2 = ax2.get_legend_handles_labels()
                h, l = h + h2, l + l2
            ax.legend(h, l, loc='upper left', fontsize=fontsize, framealpha=0.7)

    ax_t = axes[-1]
    if target_col in df_raw.columns:
        tgt_win = df_raw[target_col].loc[t0:t1]
        lbl_tgt1 = f'{target_col} ({label1})' if label1 else target_col
        ax_t.plot(tgt_win.index, tgt_win.values, color='firebrick', lw=1.5, label=lbl_tgt1)
    if df_raw2 is not None and target_col in df_raw2.columns:
        tgt_win2 = df_raw2[target_col].loc[t0:t1]
        lbl_tgt2 = f'{target_col} ({label2})' if label2 else target_col
        ax_t.plot(tgt_win2.index, tgt_win2.values, color='firebrick', lw=1.5,
                  linestyle='--', label=lbl_tgt2)
    ax_t.axhline(config['outage_threshold'], color='firebrick', ls='--', lw=1.5, alpha=0.6)
    ax_t.set_ylabel(target_col, fontsize=fontsize)
    ax_t.tick_params(axis='y', labelsize=fontsize)
    ax_t.axvspan(episode['t_start'], episode['t_end'],
                 alpha=0.12, color='red', linewidth=0)
    if has_event_window:
        ax_t.axvspan(ev_start, ev_end, alpha=0.12, color='royalblue', linewidth=0)
    ax_t.grid(True, alpha=0.3)
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax_t.tick_params(axis='x', labelsize=fontsize)
    fig.autofmt_xdate(rotation=0, ha='center')

    mode = config.get('mode', '')
    units = {'AUC': 'customer·h', 'TOT': 'min', 'binary': 'min'}.get(mode, '')
    ts_str = pd.Timestamp(episode['t_start']).strftime('%Y-%m-%d %H:%M')
    if mode == 'binary':
        if oof_pred is not None:
            title = f"RF prob = {float(oof_pred.loc[episode['t_start']]):.2f}  |  {ts_str}"
        else:
            title = f"{ts_str}"
    else:
        title = f"{mode} = {float(target.loc[episode['t_start']]):.1f} {units} | RF pred = {float(oof_pred.loc[episode['t_start']]):.1f} {units} | {ts_str}"
    if title_extra:
        title = f"{title}  |  {title_extra}"
    axes[0].set_title(title, fontsize=fontsize)
    fig.tight_layout()
    if out_dir is not None:
        fname = out_dir / f"episode_{pd.Timestamp(episode['t_start']).strftime('%Y%m%d_%H%M')}.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved episode plot → {fname}")
    return fig

def download_plot_nexrad(episode: pd.Series,
                         radar_id: str = 'KVNX',
                         radar_dir: Path = 'data/',
                         min_lon: int = -101,
                         max_lon: int = -94,
                         min_lat: int = 33,
                         max_lat: int = 39,
                         vmin: int = -10,
                         vmax: int = 70,
                         map_resolution: str = '10m',
                         site_lat: float = None,
                         site_lon: float = None,
                         site_label: str = None,
                         show_site_label: bool = True,
                         save_fig = False):
    
    target_time = episode['t_start'].replace(tzinfo=timezone.utc)

    radar_dir = Path(radar_dir)
    radar_dir.mkdir(parents=True, exist_ok=True)

    # Find nearest scan
    conn = _get_nexrad_conn()

    start = target_time - timedelta(minutes=10)
    end = target_time + timedelta(minutes=10)

    scans = list(_cached_nexrad_scan_query(
        radar_id,
        start.isoformat(),
        end.isoformat(),
    ))

    if len(scans) == 0:
        raise ValueError(
            f"No scans found for {radar_id} between {start} and {end}"
        )

    nearest_scan = min(
        scans,
        key=lambda s: abs(s.scan_time - target_time)
    )
    filename = _ensure_local_scan_file(nearest_scan, radar_dir, conn)

    # Read radar
    radar = pyart.io.read_nexrad_archive(filename)

    # Plot reflectivity (dark background scoped to this figure only, so it
    # doesn't leak into subsequently drawn time series / SHAP plots)
    with plt.style.context("dark_background"):
        display = pyart.graph.RadarMapDisplay(radar)

        fig = plt.figure(figsize=(10,8), facecolor="black")

        display.plot_ppi_map(
            "reflectivity",
            sweep=0,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
            resolution=map_resolution,
            vmin=vmin,
            vmax=vmax,
        )

        # Overlay the selected site location if coordinates are available.
        if site_lat is not None and site_lon is not None and np.isfinite(site_lat) and np.isfinite(site_lon):
            try:
                display.plot_point(site_lon, site_lat, symbol='wo', markersize=11)
                display.plot_point(site_lon, site_lat, symbol='k+', markersize=10)
            except Exception:
                ax = plt.gca()
                ax.plot(site_lon, site_lat, marker='o', markersize=8,
                        markerfacecolor='white', markeredgecolor='black', zorder=6)
                ax.plot(site_lon, site_lat, marker='+', markersize=9,
                        color='black', zorder=7)

            if show_site_label and site_label:
                try:
                    ax = plt.gca()
                    ax.text(site_lon + 0.03, site_lat + 0.02, site_label,
                            color='white', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.18', facecolor='black', alpha=0.5, linewidth=0),
                            zorder=8)
                except Exception:
                    pass

        if save_fig:
            radar_file = Path(filename)

            output_png = radar_file.with_suffix(".png")

            plt.savefig(
                output_png,
                dpi=300,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

    return fig


def list_nexrad_scans_in_range(start_time,
                               end_time,
                               radar_id: str = 'KVNX',
                               cadence_minutes: int | None = None,
                               max_scans: int | None = None) -> list:
    """List available scans in [start_time, end_time], optionally sampled by cadence."""
    start_utc = _to_utc_timestamp(start_time)
    end_utc = _to_utc_timestamp(end_time)
    if end_utc <= start_utc:
        raise ValueError("end_time must be later than start_time.")

    scans = list(_cached_nexrad_scan_query(
        radar_id,
        start_utc.isoformat(),
        end_utc.isoformat(),
    ))
    scans = [s for s in scans if not str(getattr(s, 'filename', '')).endswith('_MDM')]
    if not scans:
        return []

    scans = sorted(scans, key=_scan_timestamp_utc)

    if cadence_minutes is not None and cadence_minutes > 0:
        picks = []
        used_ids = set()
        targets = pd.date_range(start_utc, end_utc, freq=f"{int(cadence_minutes)}min", tz='UTC')
        for target in targets:
            best = min(scans, key=lambda s: abs(_scan_timestamp_utc(s) - target))
            sid = _scan_identity(best)
            if sid not in used_ids:
                used_ids.add(sid)
                picks.append(best)
        scans = sorted(picks, key=_scan_timestamp_utc)

    if max_scans is not None and max_scans > 0:
        scans = scans[:int(max_scans)]

    return scans


def download_nexrad_scans_in_range(start_time,
                                   end_time,
                                   radar_id: str = 'KVNX',
                                   radar_dir: Path = 'data/',
                                   cadence_minutes: int | None = None,
                                   max_scans: int | None = None) -> list[Path]:
    """Download and return local file paths for scans in a time window."""
    radar_dir = Path(radar_dir)
    radar_dir.mkdir(parents=True, exist_ok=True)
    scans = list_nexrad_scans_in_range(
        start_time,
        end_time,
        radar_id=radar_id,
        cadence_minutes=cadence_minutes,
        max_scans=max_scans,
    )
    conn = _get_nexrad_conn()
    paths = _resolve_or_bulk_download_scans(scans, radar_dir, conn)
    return [Path(p) for p in paths]


def _crop_radar_range_to_bbox(radar, min_lon, max_lon, min_lat, max_lat, margin=1.15):
    """Drop gates beyond the range needed to cover a lat/lon bbox, shrinking the plotted mesh.

    plot_ppi_map reprojects and rasterizes every gate out to the radar's full
    unambiguous range even though only gates inside the map extent are visible,
    so this cuts render time roughly in proportion to the gate count removed.
    """
    lat0 = float(radar.latitude['data'][0])
    lon0 = float(radar.longitude['data'][0])

    corner_lats = [min_lat, min_lat, max_lat, max_lat]
    corner_lons = [min_lon, max_lon, min_lon, max_lon]

    R = 6371000.0
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    dists = []
    for lat, lon in zip(corner_lats, corner_lons):
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        dlat = lat_r - lat0_r
        dlon = lon_r - lon0_r
        a = np.sin(dlat / 2) ** 2 + np.cos(lat0_r) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
        dists.append(2 * R * np.arcsin(np.sqrt(a)))
    max_range_m = max(dists) * margin

    range_data = radar.range['data']
    n_gates_keep = int(np.searchsorted(range_data, max_range_m)) + 1
    n_gates_keep = max(1, min(n_gates_keep, radar.ngates))
    if n_gates_keep >= radar.ngates:
        return radar

    radar.range['data'] = range_data[:n_gates_keep]
    radar.ngates = n_gates_keep
    for fdict in radar.fields.values():
        fdict['data'] = fdict['data'][:, :n_gates_keep]
    return radar


def animate_nexrad_reflectivity(start_time,
                                end_time,
                                output_path: str | Path,
                                radar_id: str = 'KVNX',
                                radar_dir: Path = 'data/',
                                cadence_minutes: int = 5,
                                fps: int = 6,
                                dpi: int = 120,
                                min_lon: int = -101,
                                max_lon: int = -94,
                                min_lat: int = 33,
                                max_lat: int = 39,
                                vmin: int = -10,
                                vmax: int = 70,
                                map_resolution: str = '50m',
                                site_lat: float = None,
                                site_lon: float = None,
                                site_label: str = None,
                                show_site_label: bool = True,
                                sweep: int = 0,
                                interval_ms: int = None) -> Path:
    """Create a reflectivity animation over a time range and save it as MP4 or GIF."""
    scans = list_nexrad_scans_in_range(
        start_time,
        end_time,
        radar_id=radar_id,
        cadence_minutes=cadence_minutes,
    )
    if not scans:
        raise ValueError("No NEXRAD scans found for the requested time window.")

    radar_dir = Path(radar_dir)
    radar_dir.mkdir(parents=True, exist_ok=True)
    conn = _get_nexrad_conn()

    frame_paths = _resolve_or_bulk_download_scans(scans, radar_dir, conn)
    frame_times = [_scan_timestamp_utc(scan) for scan in scans]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if interval_ms is None:
        interval_ms = int(1000 / max(int(fps), 1))

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(10, 8), facecolor="black")

        def _draw_frame(frame_idx):
            fig.clf()
            plt.figure(fig.number)
            radar = pyart.io.read_nexrad_archive(frame_paths[frame_idx])
            radar = _crop_radar_range_to_bbox(radar, min_lon, max_lon, min_lat, max_lat)
            display = pyart.graph.RadarMapDisplay(radar)
            display.plot_ppi_map(
                "reflectivity",
                sweep=sweep,
                min_lon=min_lon,
                max_lon=max_lon,
                min_lat=min_lat,
                max_lat=max_lat,
                resolution=map_resolution,
                vmin=vmin,
                vmax=vmax,
                title_flag=False,
            )
            ax = plt.gca()
            if site_lat is not None and site_lon is not None and np.isfinite(site_lat) and np.isfinite(site_lon):
                try:
                    display.plot_point(site_lon, site_lat, symbol='wo', markersize=11)
                    display.plot_point(site_lon, site_lat, symbol='k+', markersize=10)
                except Exception:
                    ax.plot(site_lon, site_lat, marker='o', markersize=8,
                        markerfacecolor='white', markeredgecolor='black', zorder=10)
                    ax.plot(site_lon, site_lat, marker='+', markersize=9, color='black', zorder=11)
                if show_site_label and site_label:
                    ax.text(site_lon + 0.03, site_lat + 0.02, site_label,
                            color='white', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.18', facecolor='black', alpha=0.5, linewidth=0),
                            zorder=12)
            ax.set_title(
                f"{radar_id} reflectivity | {frame_times[frame_idx].strftime('%Y-%m-%d %H:%M UTC')}",
                pad=10,
            )
            return ax

        ani = mpl_animation.FuncAnimation(
            fig,
            _draw_frame,
            frames=len(frame_paths),
            interval=interval_ms,
            repeat=False,
            blit=False,
        )

        suffix = out_path.suffix.lower()
        if suffix == '.gif':
            writer = mpl_animation.PillowWriter(fps=max(int(fps), 1))
            ani.save(str(out_path), writer=writer, dpi=dpi)
        else:
            try:
                writer = mpl_animation.FFMpegWriter(fps=max(int(fps), 1), bitrate=2000)
                ani.save(str(out_path), writer=writer, dpi=dpi)
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(
                    "FFmpeg is required for MP4 output. Use a .gif output_path or install ffmpeg."
                ) from exc
        plt.close(fig)

    print(f"  Saved reflectivity animation -> {out_path}")
    return out_path


_NEXRAD_CONN = None
_DOWNLOADED_SCAN_PATHS: dict[str, str] = {}


def _get_nexrad_conn() -> NexradAwsInterface:
    """Reuse one nexradaws client per Python process."""
    global _NEXRAD_CONN
    if _NEXRAD_CONN is None:
        _NEXRAD_CONN = NexradAwsInterface()
    return _NEXRAD_CONN


@lru_cache(maxsize=4096)
def _cached_nexrad_scan_query(radar_id: str, start_iso: str, end_iso: str) -> tuple:
    """Cache scan availability lookups by radar and time window."""
    conn = _get_nexrad_conn()
    start = pd.Timestamp(start_iso).to_pydatetime()
    end = pd.Timestamp(end_iso).to_pydatetime()
    scans = conn.get_avail_scans_in_range(start, end, radar_id)
    return tuple(scans)


def _scan_identity(scan) -> str:
    """Build a stable in-session identity string for a NEXRAD scan object."""
    for attr in ('filename', 'key', 'filepath'):
        val = getattr(scan, attr, None)
        if isinstance(val, str) and val:
            return val
    scan_time = getattr(scan, 'scan_time', None)
    radar = getattr(scan, 'radar_id', 'unknown')
    if scan_time is not None:
        return f"{radar}:{pd.Timestamp(scan_time).isoformat()}"
    return f"{radar}:{repr(scan)}"


def _find_local_scan_file(scan, radar_dir: Path) -> Path | None:
    """Try to match an already-downloaded local radar file for a scan."""
    names = []
    for attr in ('filename', 'filepath', 'key'):
        val = getattr(scan, attr, None)
        if isinstance(val, str) and val:
            names.append(Path(val).name)

    for name in names:
        candidate = radar_dir / name
        if candidate.exists():
            return candidate

    scan_time = getattr(scan, 'scan_time', None)
    if scan_time is not None:
        ts = pd.Timestamp(scan_time)
        # Use minute-level pattern as a fallback when exact names are unknown.
        pattern = f"*{ts.strftime('%Y%m%d_%H%M')}*"
        matches = sorted(radar_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def _to_utc_timestamp(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize('UTC')
    return t.tz_convert('UTC')


def _scan_timestamp_utc(scan) -> pd.Timestamp:
    return _to_utc_timestamp(getattr(scan, 'scan_time'))


def _ensure_local_scan_file(scan, radar_dir: Path, conn: NexradAwsInterface) -> str:
    """Resolve or download a scan file and return a local path."""
    scan_id = _scan_identity(scan)
    filename = _DOWNLOADED_SCAN_PATHS.get(scan_id)
    if filename is not None and Path(filename).exists():
        return filename

    local_file = _find_local_scan_file(scan, radar_dir)
    if local_file is not None:
        filename = str(local_file)
        _DOWNLOADED_SCAN_PATHS[scan_id] = filename
        return filename

    downloaded_files = conn.download([scan], str(radar_dir))
    if not downloaded_files.success:
        raise ValueError(
            f"Download failed for scan {scan_id}."
        )
    filename = downloaded_files.success[0].filepath
    _DOWNLOADED_SCAN_PATHS[scan_id] = filename
    return filename


def _resolve_or_bulk_download_scans(scans, radar_dir: Path, conn: NexradAwsInterface) -> list[str]:
    """Resolve local paths for scans, bulk-downloading any missing files."""
    resolved: dict[str, str] = {}
    missing = []

    for scan in scans:
        scan_id = _scan_identity(scan)

        filename = _DOWNLOADED_SCAN_PATHS.get(scan_id)
        if filename is not None and Path(filename).exists():
            resolved[scan_id] = filename
            continue

        local_file = _find_local_scan_file(scan, radar_dir)
        if local_file is not None:
            filename = str(local_file)
            _DOWNLOADED_SCAN_PATHS[scan_id] = filename
            resolved[scan_id] = filename
            continue

        missing.append(scan)

    if missing:
        downloaded = conn.download(missing, str(radar_dir))
        if not downloaded.success:
            raise ValueError(f"Bulk download failed for {len(missing)} scan(s).")

        by_name = {Path(f.filepath).name: f.filepath for f in downloaded.success}

        for scan in missing:
            scan_id = _scan_identity(scan)
            filename = None

            for attr in ('filename', 'filepath', 'key'):
                val = getattr(scan, attr, None)
                if isinstance(val, str) and val:
                    candidate = by_name.get(Path(val).name)
                    if candidate:
                        filename = candidate
                        break

            if filename is None:
                local_file = _find_local_scan_file(scan, radar_dir)
                if local_file is not None:
                    filename = str(local_file)

            if filename is None:
                raise ValueError(f"Downloaded scan could not be resolved locally for {scan_id}.")

            _DOWNLOADED_SCAN_PATHS[scan_id] = filename
            resolved[scan_id] = filename

    return [resolved[_scan_identity(scan)] for scan in scans]


def build_event_matrix(df_preds: pd.DataFrame,
                       raw_target: pd.Series,
                       threshold: float,
                       mode: str,
                       pre_window: int,
                       post_window: int,
                       min_duration: int = 1,
                       min_auc: float = 0.0,
                       min_peak: float = 0.0,
                       non_outages_ratio: float = None,
                       random_state: int = None) -> tuple[pd.DataFrame, pd.Series, dict, dict]:
    """Extract outage and non-outage event windows with aggregated features."""
    window_len = pre_window + post_window
    if window_len <= 0:
        raise ValueError("pre_window + post_window must be > 0.")

    predictors = list(df_preds.columns)
    dt = raw_target.index.to_series().diff().median()
    episodes = _find_episodes(raw_target, threshold)

    # Merge episodes whose inter-event gap is less than the event window width
    merge_gap = (pre_window + post_window) * dt
    episodes = _merge_close_episodes(episodes, merge_gap)
    print(f"    Episodes after merging nearby events: {len(episodes)}")
    
    # Drop individually insignificant episodes 
    dt_h = dt.total_seconds() / 3600
    dt_min = dt.total_seconds() / 60
    episodes = [
        (t0, t1) for t0, t1 in episodes
        if (t1 - t0) / dt + 1 >= min_duration and raw_target.loc[t0:t1].max() >= min_peak
    ]
    print(f"    Episodes after pre-merge filtering "
          f"(min_duration={min_duration}, min_peak={min_peak}): {len(episodes)}")

    # Add target
    filtered = []
    for t0, t1 in episodes:
        seg_vals = raw_target.loc[t0:t1]
        metric = float(seg_vals.sum() * dt_h if mode == 'AUC' else len(seg_vals) * dt_min)
        filtered.append((t0, t1, metric))

    def _agg(window):
        X_dyn, _ = build_dynamic_features(window, predictors, window=window_len, rolling=False)
        return X_dyn.iloc[-1]

    # Outage events: half-open window [t_start - pre_window, t_start + post_window).
    # Require all predictor channels to have complete data within the window.
    expected_steps = window_len
    outage_X, outage_y, episode_ends, peak_customers = {}, {}, {}, {}
    skipped = 0
    for t_start, t_end, metric in filtered:
        window = df_preds.loc[t_start - pre_window * dt : t_start + post_window * dt - dt]
        if len(window) < expected_steps or window.isna().any().any():
            skipped += 1
            continue
        outage_X[t_start] = _agg(window)
        outage_y[t_start] = 1.0 if mode == 'binary' else metric
        episode_ends[t_start] = t_end
        peak_customers[t_start] = float(raw_target.loc[t_start:t_end].max())
    if skipped:
        print(f"    Skipped {skipped} episode(s): incomplete data in event window")

    outage_df = pd.DataFrame(outage_X).T
    outage_df.index = pd.DatetimeIndex(outage_df.index)
    outage_s = pd.Series(outage_y, name='__target__')
    outage_s.index = pd.DatetimeIndex(outage_s.index)

    # Non-outage: non-overlapping tiles of same window_len, skipping any tile that
    # contains an outage timestep [Wanik 2015, Cerrai 2019]
    outage_mask = raw_target > threshold
    t_idx = raw_target.index
    n = len(t_idx)

    non_outage_X, non_outage_y = {}, {}
    i = 0
    while i + window_len <= n:
        w_start = t_idx[i]
        w_end = t_idx[i + window_len - 1]
        if not outage_mask.loc[w_start : w_end].any():
            window = df_preds.loc[w_start : w_end]
            if not window.empty and not window.isna().all().all():
                non_outage_X[w_start] = _agg(window)
                non_outage_y[w_start] = 0.0
        i += window_len

    if non_outages_ratio is not None and len(non_outage_X) > 0:
        n_target = int(len(outage_X) * non_outages_ratio)
        if n_target < len(non_outage_X):
            rng = np.random.default_rng(random_state)
            keys = rng.choice(list(non_outage_X.keys()), size=n_target, replace=False)
            non_outage_X = {k: non_outage_X[k] for k in keys}
            non_outage_y = {k: 0.0 for k in keys}
            print(f"    Non-outage windows subsampled to {n_target} "
                  f"({non_outages_ratio}x outage count)")

    non_outage_df = pd.DataFrame(non_outage_X).T
    non_outage_df.index = pd.DatetimeIndex(non_outage_df.index)
    non_outage_s = pd.Series(non_outage_y, name='__target__')
    non_outage_s.index = pd.DatetimeIndex(non_outage_s.index)

    X = pd.concat([outage_df, non_outage_df])
    y = pd.concat([outage_s, non_outage_s])

    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]

    n_outage = int((y > 0).sum())
    n_non = int((y == 0).sum())
    print(f"  Event matrix: {n_outage} outage events + {n_non} non-outage windows = {len(X)} total rows")

    return X, y, episode_ends, peak_customers

def _feature_priority(name: str) -> int:
    """Rank feature types for collinearity drop priority."""
    if '_mean_' in name:
        return 1
    if '_max_' in name:
        return 2
    if '_min_' in name:
        return 3
    return 4

def check_collinearity(X: pd.DataFrame,
                       r_threshold: float = 0.85,
                       action: str = "drop",
                       save_path: Path = None
                       ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Flag and optionally remove collinear features using Spearman r. [Dormann et al., 2013]"""
    features = list(X.columns)
    n_feat = len(features)

    print(f"  Computing Spearman correlation matrix ({n_feat} x {n_feat}) ...")
    spearman_mat = X.rank().corr()

    collinear_pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            r = spearman_mat.iloc[i, j]
            if abs(r) >= r_threshold:
                collinear_pairs.append((features[i], features[j], r))

    if collinear_pairs:
        print(f"  Warning: {len(collinear_pairs)} collinear pair(s) found "
              f"(|r| >= {r_threshold}):")
        for a, b, r in collinear_pairs:
            print(f"      {a}  <->  {b}   r = {r:+.3f}")
    else:
        print(f"  No collinear pairs found at |r| >= {r_threshold}.")

    mean_abs_r = spearman_mat.abs().mean(axis=1)

    flagged_by = []
    for i, feat in enumerate(features):
        flags = []
        if any(a == feat or b == feat for a, b, _ in collinear_pairs):
            flags.append("spearman")
        flagged_by.append(flags[0] if flags else "")

    report = pd.DataFrame({
        "feature": features,
        "mean_abs_r": mean_abs_r.values,
        "flagged_by": flagged_by,
    }).set_index("feature")

    dropped = []
    if action == "drop" and collinear_pairs:
        pairs_sorted = sorted(collinear_pairs, key=lambda x: -abs(x[2]))
        already_dropped = set()
        for a, b, r in pairs_sorted:
            if a in already_dropped or b in already_dropped:
                continue
            pa, pb = _feature_priority(a), _feature_priority(b)
            if pa != pb:
                to_drop = b if pa < pb else a
            else:
                to_drop = a if mean_abs_r[a] >= mean_abs_r[b] else b
            already_dropped.add(to_drop)
            dropped.append(to_drop)
            print(f"  Dropping '{to_drop}'  (mean |r| = {mean_abs_r[to_drop]:.3f}, "
                  f"pair r = {r:+.3f} with '{b if to_drop == a else a}')")

        if dropped:
            print(f"  -> {len(dropped)} feature(s) dropped; "
                  f"{n_feat - len(dropped)} remain.")

    report["dropped"] = report.index.isin(dropped)

    if save_path is not None:
        _plot_collinearity_heatmap(spearman_mat, r_threshold,
                                   dropped=dropped, save_path=save_path)

    X_out = X.drop(columns=dropped) if dropped else X.copy()
    return X_out, dropped, report


def _plot_collinearity_heatmap(spearman_mat: pd.DataFrame,
                                r_threshold: float,
                                dropped: list[str],
                                save_path: Path):
    """Spearman correlation heatmap with dropped features in red."""
    n = len(spearman_mat)
    size = max(6, n * 0.45)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    mat = spearman_mat.values
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    labels = list(spearman_mat.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for tick, label in zip(ax.get_xticklabels(), labels):
        if label in dropped:
            tick.set_color("red")
    for tick, label in zip(ax.get_yticklabels(), labels):
        if label in dropped:
            tick.set_color("red")

    for i in range(n):
        for j in range(n):
            if i != j and abs(mat[i, j]) >= r_threshold:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=1.2
                ))

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman r")
    ax.set_title(
        f"Feature Collinearity (Spearman r)  |  threshold = {r_threshold}\n"
        f"Red labels = dropped  |  boxed = |r| >= {r_threshold}",
        fontsize=9
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved collinearity heatmap → {save_path}")

def plot_reliability_diagram(y_true: pd.Series, y_prob: pd.Series,
                             n_bins: int=20, save_path: Path = None):
    """Calibration plot and Brier score. [DeGroot & Fienberg, 1983; Wilks, 2011; Brier, 1950]"""
    y_t = y_true.values.astype(float)
    y_p = y_prob.reindex(y_true.index).values.astype(float)

    bin_edges = np.unique(np.percentile(y_p, np.linspace(0, 100, n_bins + 1)))
    mean_pred, frac_pos = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_p >= lo) & (y_p < hi)
        if mask.sum() > 0:
            mean_pred.append(y_p[mask].mean())
            frac_pos.append(y_t[mask].mean())

    brier = float(np.mean((y_p - y_t) ** 2))

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax_cal.plot(mean_pred, frac_pos, 'o-', color='steelblue', lw=1.5,
                label='RF (isotonic calibration)')
    ax_cal.set_ylabel('Observed outage fraction')
    ax_cal.set_title(f'Reliability diagram  |  Brier score = {brier:.4f}')
    ax_cal.legend(fontsize=9)
    ax_cal.set_ylim([0, 1])
    ax_cal.grid()

    ax_hist.hist(y_p, bins=bin_edges, color='steelblue', alpha=0.7)
    ax_hist.set_xlabel('Predicted probability')
    ax_hist.set_ylabel('Count')
    ax_hist.grid()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved reliability diagram → {save_path}")


def save_event_dataset(events_df: pd.DataFrame, X: pd.DataFrame,
                       shap_df: pd.DataFrame, config: dict,
                       save_path: Path, rf_result: dict = None,
                       attrs_extra: dict = None,
                       shap_base_value: float = None,
                       global_rf_result: dict = None,
                       global_shap_base_value: float = None,
                       additional_means_df: pd.DataFrame = None) -> xr.Dataset:
    """Save event-level features, targets, and importances to NetCDF."""
    data_vars = {
        'features':  (['event', 'feature'], X.values.astype(np.float32)),
        'target':    (['event'], events_df['target'].values.astype(np.float32)),
        'is_outage': (['event'], events_df['is_outage'].values.astype(np.int8)),
        't_end':     (['event'], events_df['t_end'].values),
    }
    if 'rf_prediction' in events_df.columns:
        data_vars['rf_prediction'] = (
            ['event'],
            events_df['rf_prediction'].values.astype(np.float32),
        )
    if 'peak_customers_out' in events_df.columns:
        data_vars['peak_customers_out'] = (
            ['event'],
            events_df['peak_customers_out'].values.astype(np.float32),
        )
    if not shap_df.empty:
        data_vars['shap_values'] = (
            ['event', 'feature'],
            shap_df.reindex(X.index).values.astype(np.float32),
        )
    if rf_result is not None:
        data_vars['rf_importance'] = (
            ['feature'],
            rf_result['importance_mean'].astype(np.float32),
        )
        data_vars['rf_importance_std'] = (
            ['feature'],
            rf_result['importance_std'].astype(np.float32),
        )
    if global_rf_result is not None:
        data_vars['global_rf_importance'] = (
            ['feature'],
            global_rf_result['importance_mean'].astype(np.float32),
        )
        data_vars['global_rf_importance_std'] = (
            ['feature'],
            global_rf_result['importance_std'].astype(np.float32),
        )
        if 'shap_df' in global_rf_result:
            data_vars['global_shap_values'] = (
                ['event', 'feature'],
                global_rf_result['shap_df'].reindex(X.index).values.astype(np.float32),
            )

    attrs = {
        'mode':             config['mode'],
        'target_col':       config['target_col'],
        'outage_threshold': config['outage_threshold'],
        'pre_window':       config.get('pre_window', 0),
        'post_window':      config.get('post_window', 0),
        'predictor_cols':   config.get('predictor_cols', []),
    }
    if rf_result is not None:
        attrs['cv_score_mean'] = float(np.mean(rf_result['cv_scores']))
        attrs['cv_score_std']  = float(np.std(rf_result['cv_scores']))
        attrs['score_name']    = rf_result['score_name']
    if shap_base_value is not None:
        attrs['shap_base_value'] = float(shap_base_value)
    if global_shap_base_value is not None:
        attrs['global_shap_base_value'] = float(global_shap_base_value)
    if attrs_extra:
        attrs.update(attrs_extra)

    coords_dict = {'event': X.index, 'feature': list(X.columns)}
    if additional_means_df is not None and not additional_means_df.empty:
        add_cols = list(additional_means_df.columns)
        data_vars['additional_col_means'] = (
            ['event', 'additional_col'],
            additional_means_df.reindex(X.index).values.astype(np.float32),
        )
        coords_dict['additional_col'] = add_cols

    ds = xr.Dataset(data_vars, coords=coords_dict, attrs=attrs)
    ds.to_netcdf(save_path)
    print(f"  Saved event dataset → {save_path}")
    return ds


def load_config(path: str = "configs/outage_rf_events.yaml") -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict, source: str = None) -> pd.DataFrame:
    """Load and extract predictor and target columns from NetCDF."""
    src = source or config["source"]
    ds = xr.open_dataset(src)
    cols = config['predictor_cols'] + [config['target_col']]

    if 'aavi' in cols:
        ds['aavi'] = ds['wssd'] / ds['wssd'].mean() * ds['wdsd'] / ds['wdsd'].mean()

    df = ds[cols].to_dataframe()
    df.index = ds.time.values
    ds.close()
    return df


def qc_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply min/max limits to each predictor column."""
    df_qc = pd.DataFrame(index=df.index)
    for v in config['predictor_cols']:
        df_qc[v] = (df[v]
                    .where(df[v] >= config['limits'][v][0])
                    .where(df[v] <= config['limits'][v][1]))
    df_qc[config['target_col']] = df[config['target_col']]
    return df_qc


def _load_and_detrend(source_str: str, cfg: dict, base: Path) -> tuple:
    """Load NetCDF, apply QC, optionally detrend and cache."""
    source_path = Path(source_str)
    detrended_path = source_path.with_name(source_path.stem + '.detrended.nc')
    climatology_path = source_path.with_name(source_path.stem + '.climatology.csv')
    
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
                save_climatology_path=climatology_path,
            )
        df_qc.index.name = 'time'
        xr.Dataset.from_dataframe(df_qc).to_netcdf(detrended_path)
        print(f"  Saved detrended cache → {detrended_path}")

    return df_qc, detrended_path, source_path


def _compute_additional_means(event_idx: pd.DatetimeIndex, source_path: Path,
                              cfg: dict, dt) -> pd.DataFrame:
    """Extract and aggregate additional columns from raw source at event times."""
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
    if not rows:
        return pd.DataFrame(index=event_idx, columns=avail, dtype=float)
    return pd.DataFrame(rows).T[avail]


def _get_event_matrix(subset: pd.DataFrame, cfg_run: dict,
                      source_path: Path) -> tuple:
    """Build cached event matrix from detrended data."""
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


def _run_pipeline(X: pd.DataFrame, y: pd.Series, cfg_run: dict,
                  groups_arr: np.ndarray = None,
                  out_dir: Path = None,
                  ep_ends: dict = None, pk_ends: dict = None,
                  det_paths: list = None, add_df: pd.DataFrame = None) -> dict:
    """Collinearity + RF CV for both modes. With out_dir: also global RF and full save; without: return CV metrics only."""
    import os
    import yaml

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir / 'config.yaml', 'w') as f:
            yaml.dump(cfg_run, f)

    print("\n[5] Checking feature collinearity ...")
    X_clean, dropped, report = check_collinearity(
        X,
        r_threshold=cfg_run["collinearity_r_threshold"],
        action=cfg_run["collinearity_action"],
        save_path=(out_dir / "collinearity_heatmap.png") if out_dir else None,
    )
    if out_dir:
        report.to_csv(out_dir / "collinearity_report.csv")
    if dropped:
        print(f"    Proceeding with {X_clean.shape[1]} features "
              f"(dropped {len(dropped)}: {dropped})")
    else:
        print(f"    No features dropped; proceeding with {X_clean.shape[1]} features.")
    X = X[list(X_clean.columns)]

    if groups_arr is not None:
        print(f"  Using grouped CV: {len(np.unique(groups_arr))} groups for {len(X)} events")

    print("\n[6] Training RF + permutation importance (CV) ...")
    rf_result = train_rf_importance(X, y, cfg_run, groups=groups_arr)

    if out_dir is None:
        # Sensitivity mode: return CV metrics only
        oof   = rf_result['oof_pred'].reindex(y.index)
        y_bin = (y > 0).astype(float)
        # [Brier, 1950; Murphy, 1973]
        return {
            'brier_score': float(np.mean((oof.values - y_bin.values) ** 2)),
            'roc_auc':     float(np.mean(rf_result['cv_scores'])),
        }

    # Single-run mode: global RF + save
    print("\n[7] Training RF on full dataset (global importance + SHAP) ...")
    global_rf = train_rf_full(X, y, cfg_run)

    oof = rf_result['oof_pred']
    events_df = pd.DataFrame({
        't_end':              pd.DatetimeIndex([ep_ends.get(ts, pd.NaT) for ts in X.index]),
        'target':             y.reindex(X.index).values.astype(float),
        'is_outage':          (y.reindex(X.index).values > 0).astype(int),
        'rf_prediction':      oof.reindex(X.index).values.astype(float),
        'peak_customers_out': [pk_ends.get(ts, np.nan) for ts in X.index],
    }, index=X.index)

    shap_df   = rf_result.get('shap_df',   pd.DataFrame())
    shap_base = rf_result.get('shap_base', None)
    det_attr  = ';'.join(str(p) for p in det_paths) if det_paths else ''

    print("\n[9] Saving event dataset ...")
    save_event_dataset(
        events_df, X, shap_df, cfg_run,
        save_path=out_dir / "events.nc",
        rf_result=rf_result,
        attrs_extra={'detrended_source': det_attr},
        shap_base_value=shap_base,
        global_rf_result=global_rf,
        global_shap_base_value=global_rf.get('shap_base', None),
        additional_means_df=add_df,
    )
    print("\nPipeline complete.")
    return X


def _total_duration_h(episode_ends: dict, dt) -> float:
    """Sum of outage episode durations in hours (inclusive of both endpoints)."""
    return sum((t_end - t_start + dt).total_seconds() / 3600
               for t_start, t_end in episode_ends.items())


def _pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str,
                     minimize_y: bool = True) -> pd.DataFrame:
    """Non-dominated set: maximise x_col and minimise/maximise y_col."""
    sub = df.dropna(subset=[x_col, y_col]).sort_values(x_col, ascending=False)
    frontier, best_y = [], np.inf if minimize_y else -np.inf
    for _, row in sub.iterrows():
        y_val = row[y_col]
        if (minimize_y and y_val <= best_y) or (not minimize_y and y_val >= best_y):
            frontier.append(row)
            best_y = y_val
    return pd.DataFrame(frontier).sort_values(x_col)


def plot_sensitivity(df_res: pd.DataFrame, thresholds: list, durations: list,
                     peaks: list, out_dir: Path):
    """Heatmap and Pareto scatter plots for sensitivity analysis results."""
    from matplotlib.lines import Line2D

    _METRICS = [
        ('brier_score', 'Brier score', 'RdYlGn_r', True),
        ('roc_auc',     'ROC-AUC',     'RdYlGn',   False),
        ('coverage',    'Coverage',    'RdYlGn',   False),
    ]

    for metric, label, cmap, lower_better in _METRICS:
        if df_res[metric].isna().all():
            continue
        vmin, vmax = df_res[metric].min(), df_res[metric].max()
        n_thr = len(thresholds)

        fig, axes = plt.subplots(1, n_thr, figsize=(4.5 * n_thr, 4), sharey=True)
        if n_thr == 1:
            axes = [axes]

        for ax, thr in zip(axes, thresholds):
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
                        ax.text(ci, ri, f'{val:.3f}', ha='center', va='center', fontsize=7)

        plt.colorbar(im, ax=axes[-1]).set_label(
            f'{label} ({"lower" if lower_better else "higher"} = better)')
        fig.suptitle(f'Sensitivity analysis — {label}', fontweight='bold')
        plt.tight_layout()
        png_path = out_dir / f'sensitivity_{metric.replace("_score", "")}.png'
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {png_path}")

    # Pareto scatter plots
    dur_vals  = sorted(df_res['min_outage_duration'].unique())
    peak_vals = sorted(df_res['min_peak_customers_out'].unique())
    thr_vals  = sorted(df_res['outage_threshold'].unique())

    _MARKER_POOL = ['o', 's', '^', 'D', 'P', 'X', 'v', '<']
    peak_to_marker = {p: _MARKER_POOL[i % len(_MARKER_POOL)] for i, p in enumerate(peak_vals)}
    thr_sizes      = np.linspace(40, 220, max(len(thr_vals), 2))
    thr_to_size    = dict(zip(thr_vals, thr_sizes))
    dur_to_color   = {d: plt.cm.tab10.colors[i % 10] for i, d in enumerate(dur_vals)}

    for metric, label, minimize_y in [('brier_score', 'Brier score', True),
                                       ('roc_auc',     'ROC-AUC',    False)]:
        plot_df = df_res.dropna(subset=[metric, 'coverage'])
        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.subplots_adjust(right=0.65)

        for _, row in plot_df.iterrows():
            ax.scatter(row['coverage'], row[metric],
                       c=[dur_to_color[row['min_outage_duration']]],
                       marker=peak_to_marker[row['min_peak_customers_out']],
                       s=thr_to_size[row['outage_threshold']],
                       edgecolors='k', linewidths=0.5, zorder=3)

        pareto = _pareto_frontier(plot_df, 'coverage', metric, minimize_y=minimize_y)
        if len(pareto) > 1:
            ax.plot(pareto['coverage'], pareto[metric], color='k', lw=1.5, zorder=4)

        leg1 = ax.legend(
            handles=[Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=dur_to_color[d], markeredgecolor='k',
                            markersize=9, label=str(d)) for d in dur_vals],
            title='min_outage_duration', loc='upper left',
            bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.8)
        ax.add_artist(leg1)

        leg2 = ax.legend(
            handles=[Line2D([0], [0], marker=peak_to_marker[p], color='gray',
                            linestyle='none', markersize=8, markeredgecolor='k',
                            label=str(p)) for p in peak_vals],
            title='min_peak_customers_out', loc='upper left',
            bbox_to_anchor=(1.02, 0.65), fontsize=8, framealpha=0.8)
        ax.add_artist(leg2)

        size_handles = [
            Line2D([0], [0], marker='o', color='gray', linestyle='none',
                   markeredgecolor='k', markersize=np.sqrt(thr_to_size[t]), label=str(t))
            for t in thr_vals
        ]
        if len(pareto) > 1:
            size_handles.append(Line2D([0], [0], color='k', lw=1.5, label='Pareto frontier'))
        ax.legend(handles=size_handles, title='outage_threshold',
                  loc='upper left', bbox_to_anchor=(1.02, 0.3), fontsize=8, framealpha=0.8)

        direction = 'lower' if minimize_y else 'higher'
        ax.set_xlabel('Coverage  (total outage hours / baseline total outage hours)')
        ax.set_ylabel(f'{label}  ({direction} = better)')
        ax.set_title(f'Coverage vs {label} — Pareto frontier')
        ax.grid(alpha=0.3)
        png_path = out_dir / f'sensitivity_pareto_{metric.replace("_score", "")}.png'
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        print(f"  Saved → {png_path}")


class OutagePredictor:
    """
    Loads a saved model bundle and returns calibrated outage probabilities for
    any input atmospheric time series via a sliding event window.

    The event window [t - pre_window, t + post_window] is slid over every
    timestep t. Features are window-level summary statistics (mean, std,
    mean gradient, std gradient) computed with vectorised rolling operations,
    matching the aggregation used during training.

    Note: gradient features include one extra boundary observation compared to
    the per-window computation during training (trivial effect on predictions).
    """

    def __init__(self, bundle_dir: str):
        bundle_dir = Path(bundle_dir)
        with open(bundle_dir / 'config.yaml') as f:
            self.cfg = yaml.safe_load(f)
        bundle = joblib.load(bundle_dir / 'model.joblib')
        self.rf  = bundle['rf']
        self.iso = bundle['iso']
        feat_names = self.cfg.get('feature_names', [])
        if len(feat_names) != self.rf.n_features_in_:
            raise ValueError(
                f"Bundle mismatch: config has {len(feat_names)} feature names "
                f"but RF expects {self.rf.n_features_in_}."
            )

    def predict(self, df: pd.DataFrame,
                climatology_path: str | Path = None,
                non_overlapping: bool = False,
                return_features: bool = False) -> pd.Series:
        """

        Parameters
        ----------
        df               : DataFrame with DatetimeIndex; must contain all predictor_cols.
        station          : name of the station whose climatology is used for detrending;
                           defaults to the first station in the bundle.
        climatology_path : optional path to a climatology CSV produced by
                           remove_seasonal_cycle(); takes priority over the bundle
                           climatology.  If neither is available the seasonal cycle
                           is removed directly from df.
        non_overlapping  : if True, tile the series into non-overlapping windows of
                           roll_size steps and return one probability per tile, indexed
                           at the center timestamp.  If False (default), return a
                           probability at every timestep (sliding window).

        Returns
        -------
        pd.Series of float.  When non_overlapping=False: same index as df, NaN at
        boundary timesteps and wherever input data are missing.  When
        non_overlapping=True: one value per tile, indexed at the tile center.
        """
        predictors   = self.cfg['predictor_cols']
        pre_w        = self.cfg['pre_window']
        post_w       = self.cfg['post_window']
        feat_names   = self.cfg['feature_names']
        w_label      = pre_w + post_w   # matches the W<n> suffix in feature names
        roll_size    = pre_w + post_w + 1
        detrend_cols = [c for c in self.cfg['predictor_cols'] if c not in self.cfg['detrend_exclude']]
        
        missing = [c for c in predictors if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing columns: {missing}")


        if climatology_path is not None:
            clim = pd.read_csv(climatology_path, index_col=[0, 1])
            clim.index.names = ['__doy__', '__tod__']
        else:
            clim=None

        # Detrend with saved climatology [Wilks, 2011]; fall back to on-the-fly detrending
       
        if clim is not None:
            df_det = apply_climatology(df[predictors].copy(), clim,
                                       columns=detrend_cols, inplace=True)
        else:
            df_det, _ = remove_seasonal_cycle(
                df[predictors].copy(),
                columns=detrend_cols,
                window_days=self.cfg.get('detrend_window_days', 7),
                min_periods=self.cfg.get('detrend_min_periods', 3),
                inplace=True,
            )

        # Vectorised rolling feature matrix.
        # rolling(roll_size).shift(-post_w) aligns each value so that at index t
        # the window covers [t - pre_w, t + post_w], matching the training windows.
        # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
        #
        # min_periods < roll_size lets the statistic be computed on whatever
        # valid (non-nan) samples remain in the window, up to max_nan_ratio
        # missing — otherwise a single QC'd-out sample nulls every feature for
        # the entire window (pandas rolling default is min_periods=roll_size).
        max_nan_ratio = self.cfg.get('max_nan_ratio', 0.25)
        min_periods = max(1, int(np.ceil((1 - max_nan_ratio) * roll_size)))
        frames = {}
        for col in predictors:
            x = df_det[col]
            g = x.diff()
            r = x.rolling(roll_size, min_periods=min_periods)
            gr = g.rolling(roll_size, min_periods=min_periods)
            frames[f'{col}_mean_W{w_label}']     = r.mean().shift(-post_w)
            frames[f'{col}_std_W{w_label}']      = r.std().shift(-post_w)
            frames[f'{col}_grad_mean_W{w_label}'] = gr.mean().shift(-post_w)
            frames[f'{col}_grad_std_W{w_label}']  = gr.std().shift(-post_w)

        feat_df = pd.DataFrame(frames, index=df_det.index).reindex(columns=feat_names)

        if non_overlapping:
            # Anchor at pre_w within each tile so [t-pre_w, t+post_w] = tile [k*(roll_size-1), k*(roll_size-1)+roll_size-1]
            # step = roll_size-1 so consecutive windows share one boundary timestep
            sel_pos = np.arange(pre_w, len(feat_df) - post_w, roll_size - 1)
            feat_sel = feat_df.iloc[sel_pos].reindex(columns=feat_names)
            valid = feat_sel.notna().all(axis=1).values
            # Center of each tile (works for unequal pre_w / post_w)
            center_pos = (sel_pos - pre_w + roll_size // 2).clip(0, len(df) - 1)
            center_times = df.index[center_pos]
            probs = pd.Series(np.nan, index=center_times, name='outage_probability')
            if valid.any():
                probs.iloc[valid] = self.iso.predict(self.rf.predict_proba(feat_sel.values[valid])[:, 1])
            if return_features:
                feat_out = feat_sel.set_axis(center_times)
                return probs, feat_out, df_det
            return probs

        probs = pd.Series(np.nan, index=df_det.index, name='outage_probability')
        valid = feat_df.notna().all(axis=1)
        if valid.any():
            probs[valid] = self.iso.predict(self.rf.predict_proba(feat_df[valid].values)[:, 1])
        if return_features:
            return probs, feat_df, df_det
        return probs

def wdh_awaken_search(a2e, channel, sdate, edate):
    t0 = pd.Timestamp(sdate).strftime('%Y%m%d%H%M%S')
    t1 = pd.Timestamp(edate).strftime('%Y%m%d%H%M%S')
    try:
        results = a2e.search({'Dataset': channel,
                              'date_time': {'between': [t0, t1]}}, table='inventory')
        if not results:
            return pd.DatetimeIndex([])
        df = pd.DataFrame(results)
        return pd.to_datetime(df['date_time'], format='%Y%m%d%H%M%S', errors='coerce').dropna()
    except Exception:
        # retry month by month
        print(f"  Timeout for {channel} over full period — retrying by month")
        months = pd.date_range(pd.Timestamp(sdate).to_period('M').to_timestamp(),
                               pd.Timestamp(edate),
                               freq='MS')
        all_timestamps = []
        for m in months:
            m0 = m.strftime('%Y%m%d%H%M%S')
            m1 = (m + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59).strftime('%Y%m%d%H%M%S')
            try:
                results = a2e.search({'Dataset': channel,
                                      'date_time': {'between': [m0, m1]}}, table='inventory')
                if results:
                    all_timestamps.append(pd.to_datetime(
                        pd.DataFrame(results)['date_time'],
                        format='%Y%m%d%H%M%S', errors='coerce').dropna())
            except Exception as exc:
                print(f"  Warning — {channel} {m.strftime('%Y-%m')}: {exc}")
        if all_timestamps:
            return pd.DatetimeIndex(pd.concat(all_timestamps))
        return pd.DatetimeIndex([])