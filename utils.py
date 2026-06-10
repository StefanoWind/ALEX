import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import stats
import matplotlib
import matplotlib.dates as mdates
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (StratifiedKFold, KFold, train_test_split,
                                     StratifiedGroupKFold, GroupKFold)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight

plt.close('all')
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Run `pip install shap` for SHAP importance.")

import re

_VAR_LABELS = {
    'relh': 'Relative humidity [%]',
    'tair': 'Air temperature [°C]',
    'wspd': 'Wind speed [m/s]',
    'wssd': 'Wind speed std [m/s]',
    'wdsd': 'Wind dir. std [°]',
    'aavi': 'Wind variability index',
    'wmax': 'Maximum wind speed [m/s]',
    'rain': 'Precipitation [mm]',
    'pres': 'Pressure [hPa]',
    'srad': 'Solar radiation [W/m²]',
    'TURB': 'Turbulence intensity [%]',
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


def make_binary_target(series: pd.Series, threshold: float) -> pd.Series:
    """Convert count column to binary flag."""
    return (series > threshold).astype(int)


def make_segment_target(series: pd.Series, threshold: float, mode: str) -> pd.Series:
    """
    For each contiguous segment where series > threshold compute a severity metric
    and broadcast it to every timestep in that segment; assign 0 elsewhere.

    mode='AUC': integral of the outage curve (customer·h) over the segment.
    mode='TOT': total duration of the segment in minutes.
    """
    result = pd.Series(0.0, index=series.index)
    above = series > threshold
    segment_id = (above != above.shift()).cumsum()
    dt_minutes = series.index.to_series().diff().median().total_seconds() / 60

    for seg_id, idx in above.groupby(segment_id).groups.items():
        if not above[idx[0]]:
            continue
        seg = series[idx]
        value = seg.sum() * dt_minutes / 60 if mode == 'AUC' else len(seg) * dt_minutes
        result[idx] = value

    return result


def _find_episodes(series: pd.Series, threshold: float) -> list[tuple]:
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


def _select_top_episodes(series: pd.Series, threshold: float,
                          mode: str, n: int) -> pd.DataFrame:
    episodes = _find_episodes(series, threshold)
    if not episodes:
        return pd.DataFrame()
    seg = make_segment_target(series, threshold, mode)
    rows = [{'t_start': t0, 't_end': t1,
             't_center': t0 + (t1 - t0) / 2,
             'metric': float(seg.loc[t0])}
            for t0, t1 in episodes]
    return (pd.DataFrame(rows)
            .sort_values('metric', ascending=False)
            .head(n)
            .reset_index(drop=True))


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
    plt.figure(figsize=(18, 10))
    ctr = 1
    mode = config.get('mode')
    if mode == 'binary':
        flag = df[config['target_col']] > config.get('outage_threshold', 0)
    elif mode in ('AUC', 'TOT'):
        c_vals = df['__target__'] if '__target__' in df.columns else df[config['target_col']]
    for col in config['predictor_cols']:
        plt.subplot(len(config['predictor_cols']), 1, ctr)
        if mode == 'binary':
            plt.scatter(df.index[~flag], df[col][~flag], s=1, c='green')
            plt.scatter(df.index[flag], df[col][flag], s=1, c='red')
        elif mode in ('AUC', 'TOT'):
            plt.scatter(df.index, df[col], s=1, c=c_vals, cmap='RdYlGn_r')
        else:
            plt.scatter(df.index, df[col], s=1,
                        c=np.log10(df[config['target_col']]), cmap='RdYlGn_r')
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
    mode = config.get('mode')
    if mode in ('AUC', 'TOT') and '__target__' in df.columns:
        y_series = df['__target__']
        y_label = mode
    else:
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
    # [von Storch & Zwiers, 1999]
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
    frames = {f"{col} (lag={lag[col]})": df[col].shift(lag[col]) for col in predictors}
    X = pd.DataFrame(frames, index=df.index)
    y = df[target_col]
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


# [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
def build_dynamic_features(df: pd.DataFrame, predictors: list, target_col: str = None,
                            window: int = 0, rolling: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    frames = {}
    w = window
    for col in predictors:
        x = df[col]
        grad = x.diff()
        if rolling:
            frames[f"{col}_mean_W{w}"]      = x.rolling(w, center=True).mean()     # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
            # frames[f"{col}_max_W{w}"]       = x.rolling(w, center=True).max()
            # frames[f"{col}_min_W{w}"]       = x.rolling(w, center=True).min()
            # frames[f"{col}_std_W{w}"]       = x.rolling(w, center=True).std()    # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
            # frames[f"{col}_grad_mean_W{w}"] = grad.rolling(w, center=True).mean()
            # frames[f"{col}_grad_max_W{w}"]  = grad.rolling(w, center=True).max()
            # frames[f"{col}_grad_min_W{w}"]  = grad.rolling(w, center=True).min()
            # frames[f"{col}_grad_std_W{w}"]  = grad.rolling(w, center=True).std()
        else:
            # Whole-window aggregation — scalar statistics over the full input slice
            frames[f"{col}_mean_W{w}"]      = x.mean()                             # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
            # frames[f"{col}_max_W{w}"]       = x.max()
            # frames[f"{col}_min_W{w}"]       = x.min()
            frames[f"{col}_std_W{w}"]       = x.std()                            # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
            frames[f"{col}_grad_mean_W{w}"] = grad.mean()
            # frames[f"{col}_grad_max_W{w}"]  = grad.max()
            # frames[f"{col}_grad_min_W{w}"]  = grad.min()
            frames[f"{col}_grad_std_W{w}"]  = grad.std()
    y = df[target_col] if target_col is not None else None
    if rolling:
        return pd.DataFrame(frames, index=df.index), y
    else:
        return pd.DataFrame([frames]), y


def _mode_label(config: dict) -> str:
    mode = config.get('mode', '')
    units = {'AUC': 'customer·h', 'TOT': 'min'}
    return f"{mode} ({units[mode]})" if mode in units else mode


def plot_rf_scatter(y_true: pd.Series, y_pred: pd.Series, config: dict,
                    save_path: Path = None):
    """Scatter of observed vs OOF RF prediction with 1:1 line and linear regression."""
    valid = ~(y_true.isna() | y_pred.isna())
    yt_s, yp_s = y_true[valid], y_pred[valid]
    yt, yp = yt_s.values, yp_s.values
    slope, intercept, r, _, _ = stats.linregress(yt, yp)
    x_line = np.array([yp.min(), yp.max()])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=10, alpha=0.4, color='steelblue')
    ax.plot(x_line, x_line, 'k--', lw=1, label='1:1')
    ax.plot(x_line, slope * x_line + intercept, 'r-', lw=1,
            label=f'linear fit  r={r:.2f}')

    error = yp_s - yt_s
    annotate_idx = list(error.nlargest(2).index) + list(error.nsmallest(2).index)
    offsets = [(10, 10), (10, -20), (10, 10), (10, -20)]
    for idx, (dx, dy) in zip(annotate_idx, offsets):
        label_txt = pd.Timestamp(idx).strftime('%Y-%m-%d\n%H:%M')
        ax.annotate(label_txt,
                    xy=(yt_s[idx], yp_s[idx]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=6, color='dimgray',
                    arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

    label = _mode_label(config)
    ax.set_ylabel(f'RF prediction [{label}]')
    ax.set_xlabel(f'Observed [{label}]')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved RF scatter plot → {save_path}")


def plot_rf_timeseries(y_true: pd.Series, y_pred: pd.Series, config: dict,
                       save_path: Path = None):
    """Time series of observed vs OOF RF prediction."""
    valid = ~(y_true.isna() | y_pred.isna())
    label = _mode_label(config)

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(y_true[valid].index, y_true[valid].values,
            '.-k', lw=0.8, label='Observed')
    ax.plot(y_pred[valid].index, y_pred[valid].values,
            '.-r', lw=0.8, label='RF prediction')
    ax.set_ylabel(label)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved RF time series plot → {save_path}")


def train_rf_importance(X: pd.DataFrame, y: pd.Series, config: dict,
                        groups: np.ndarray = None) -> dict:
    """
    Train RF with cross-validation and return permutation importances. [Breiman, 2001]
    """
    mode = config["mode"]
    n_folds = config["n_cv_folds"]

    if mode == "binary":
        model_cls = RandomForestClassifier
        score_fn = lambda m, Xt, yt: roc_auc_score(yt, m.predict_proba(Xt)[:, 1])
        score_name = "ROC-AUC"
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                      random_state=config["random_state"])
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=config["random_state"])
    else:
        model_cls = RandomForestRegressor
        score_fn = lambda m, Xt, yt: -mean_squared_error(yt, m.predict(Xt))
        score_name = "neg-RMSE"
        if groups is not None:
            cv = GroupKFold(n_splits=n_folds)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True,
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

        if mode == "binary":
            cal_frac = config.get("calibration_fraction", 0.3)
            X_fit, X_cal, y_fit, y_cal = train_test_split(
                X_tr, y_tr, test_size=cal_frac, stratify=y_tr,
                random_state=config["random_state"],
            )
            sw_fit = compute_sample_weight("balanced", y_fit)
            model.fit(X_fit, y_fit, sample_weight=sw_fit)
            # [Zadrozny & Elkan, 2002, KDD; Niculescu-Mizil & Caruana, 2005, ICML]
            calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            calibrated.fit(X_cal, y_cal)
            val_prob = calibrated.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, val_prob))
            oof_pred[val_idx] = val_prob
        else:
            model.fit(X_tr, y_tr)
            cv_scores.append(score_fn(model, X_val, y_val))
            oof_pred[val_idx] = model.predict(X_val)

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
    # [Breiman, 2001; Lundberg & Lee, 2017]
    mode = config["mode"]
    if mode == "binary":
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_features=config["max_features"],
            n_jobs=config["n_jobs"],
            random_state=config["random_state"],
        )
        sw = compute_sample_weight("balanced", y.values)
    else:
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_features=config["max_features"],
            n_jobs=config["n_jobs"],
            random_state=config["random_state"],
        )
        sw = None

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


def _plot_shap_waterfalls(explainer, X: pd.DataFrame,
                          episode_centers: list, out_dir: Path):
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

    feat_mean = X.mean().values
    feat_std = np.where(X.std().values > 0, X.std().values, 1.0)

    for center in episode_centers:
        nearest_idx = int(np.abs((X.index - center).total_seconds()).argmin())
        if nearest_idx < 0 or nearest_idx >= len(X):
            continue
        x_row = X.iloc[[nearest_idx]]
        sv = explainer.shap_values(x_row)
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        z_scores = (x_row.values[0] - feat_mean) / feat_std
        expl = shap.Explanation(
            values=sv[0],
            base_values=base_val,
            data=z_scores,
            feature_names=[re.sub(r'\s*\[.*?\]', '', _feat_label(c)) for c in X.columns],
        )
        shap.plots.waterfall(expl, show=False)
        fig = plt.gcf()
        fig.text(0.5, 0.01, 'Feature values shown as z-scores relative to full dataset',
                 ha='center', fontsize=8, color='gray')
        ts_str = pd.Timestamp(X.index[nearest_idx]).strftime('%Y%m%d_%H%M')
        fname = out_dir / f"shap_waterfall_{ts_str}.png"
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved SHAP waterfall → {fname}")


def compute_shap_importance(X: pd.DataFrame, y: pd.Series,
                            config: dict) -> tuple[pd.Series, pd.DataFrame]:
    # [Lundberg & Lee, 2017]
    if not HAS_SHAP:
        print("  SHAP skipped (not installed).")
        return pd.Series(dtype=float), pd.DataFrame()

    mode = config["mode"]
    if mode == "binary":
        model = RandomForestClassifier(n_estimators=config["n_estimators"],
                                       n_jobs=config["n_jobs"],
                                       random_state=config["random_state"])
        sw = compute_sample_weight("balanced", y)
    else:
        model = RandomForestRegressor(n_estimators=config["n_estimators"],
                                      n_jobs=config["n_jobs"],
                                      random_state=config["random_state"])
        sw = None

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
    plt.close('all')
    z_scores = (feat_vals - feat_mean) / feat_std
    labels = [re.sub(r'\s*\[.*?\]', '', _feat_label(c)) for c in feat_names]
    expl = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=z_scores,
        feature_names=labels,
    )
    shap.plots.waterfall(expl, show=False)
    fig = plt.gcf()
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


def save_event_dataset(events_df: pd.DataFrame, X: pd.DataFrame,
                       shap_df: pd.DataFrame, config: dict,
                       save_path: Path, rf_result: dict = None,
                       attrs_extra: dict = None,
                       shap_base_value: float = None,
                       global_rf_result: dict = None,
                       global_shap_base_value: float = None) -> xr.Dataset:
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

    ds = xr.Dataset(
        data_vars,
        coords={'event': X.index, 'feature': list(X.columns)},
        attrs=attrs,
    )
    ds.to_netcdf(save_path)
    print(f"  Saved event dataset → {save_path}")
    return ds


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
                    episode: pd.Series, config: dict, out_dir: Path,
                    target: pd.Series = None, oof_pred: pd.Series = None,
                    df_raw2: pd.DataFrame = None, X2: pd.DataFrame = None,
                    label1: str = '', label2: str = ''):

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

    lbl_raw1 = f'raw ({label1})' if label1 else 'raw'
    lbl_det1 = f'detrended ({label1})' if label1 else 'detrended'
    lbl_raw2 = f'raw ({label2})' if label2 else 'raw'
    lbl_det2 = f'detrended ({label2})' if label2 else 'detrended'

    fig, axes = plt.subplots(n, 1, figsize=(18, 3 * n), sharex=True)

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
            ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=18)
            ax2.spines['right'].set_edgecolor('steelblue')

        ax.set_ylabel(_feat_label(col), fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
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
            ax.legend(h, l, loc='upper left', fontsize=18, framealpha=0.7)

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
    ax_t.set_ylabel(target_col, fontsize=18)
    ax_t.tick_params(axis='y', labelsize=18)
    ax_t.axvspan(episode['t_start'], episode['t_end'],
                 alpha=0.12, color='red', linewidth=0)
    if has_event_window:
        ax_t.axvspan(ev_start, ev_end, alpha=0.12, color='royalblue', linewidth=0)
    ax_t.grid(True, alpha=0.3)
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax_t.tick_params(axis='x', labelsize=18)
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
    axes[0].set_title(title)
    fig.tight_layout()
    fname = out_dir / f"episode_{pd.Timestamp(episode['t_start']).strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved episode plot → {fname}")


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
    
    # Drop individually insignificant episodes before merging so they cannot
    # anchor a merged block that inherits a large peak from a later sub-episode.
    dt_h = dt.total_seconds() / 3600
    dt_min = dt.total_seconds() / 60
    episodes = [
        (t0, t1) for t0, t1 in episodes
        if (t1 - t0) / dt + 1 >= min_duration and raw_target.loc[t0:t1].max() >= min_peak
    ]
    print(f"    Episodes after pre-merge filtering "
          f"(min_duration={min_duration}, min_peak={min_peak}): {len(episodes)}")

    # Apply AUC/TOT severity filter on the (possibly merged) episode
    filtered = []
    for t0, t1 in episodes:
        seg_vals = raw_target.loc[t0:t1]
        metric = float(seg_vals.sum() * dt_h if mode == 'AUC' else len(seg_vals) * dt_min)
        filtered.append((t0, t1, metric))

    def _agg(window):
        X_dyn, _ = build_dynamic_features(window, predictors, window=window_len, rolling=False)
        return X_dyn.iloc[-1]

    # Outage events: fixed window [t_start - pre_window, t_start + post_window].
    # Require all predictor channels to have complete data within the window.
    expected_steps = pre_window + post_window + 1
    outage_X, outage_y, episode_ends, peak_customers = {}, {}, {}, {}
    skipped = 0
    for t_start, t_end, metric in filtered:
        window = df_preds.loc[t_start - pre_window * dt : t_start + post_window * dt]
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

def _vif_series(X_arr: np.ndarray) -> np.ndarray:
    """Variance Inflation Factor for each column via OLS. [Montgomery et al., 2012]"""
    n, p = X_arr.shape
    vif = np.full(p, np.nan)
    for j in range(p):
        y_j = X_arr[:, j]
        X_oth = np.delete(X_arr, j, axis=1)
        X_oth = np.column_stack([np.ones(n), X_oth])
        try:
            beta = np.linalg.lstsq(X_oth, y_j, rcond=None)[0]
            y_hat = X_oth @ beta
            ss_res = np.sum((y_j - y_hat) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        except np.linalg.LinAlgError:
            vif[j] = np.inf
    return vif


def _feature_priority(name: str) -> int:
    if '_mean_' in name:
        return 1
    if '_max_' in name:
        return 2
    if '_min_' in name:
        return 3
    return 4

def check_collinearity(X: pd.DataFrame,
                       r_threshold: float = 0.85,
                       vif_threshold: float | None = 10.0,
                       action: str = "drop",
                       save_path: Path = None
                       ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Flag and optionally remove collinear features using Spearman r and VIF.
    [Dormann et al., 2013]
    """
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

    vif_values = np.full(n_feat, np.nan)
    if vif_threshold is not None:
        print(f"  Computing VIF for {n_feat} features ...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values.astype(float))
        vif_values = _vif_series(X_scaled)
        high_vif = [(features[i], vif_values[i])
                    for i in range(n_feat) if vif_values[i] > vif_threshold]
        if high_vif:
            print(f"  Warning: {len(high_vif)} feature(s) with VIF > {vif_threshold}:")
            for name, v in sorted(high_vif, key=lambda x: -x[1]):
                print(f"      {name}   VIF = {v:.1f}")
        else:
            print(f"  All VIF values <= {vif_threshold}.")

    mean_abs_r = spearman_mat.abs().mean(axis=1)

    flagged_by = []
    for i, feat in enumerate(features):
        flags = []
        if any(a == feat or b == feat for a, b, _ in collinear_pairs):
            flags.append("spearman")
        if not np.isnan(vif_values[i]) and vif_values[i] > (vif_threshold or np.inf):
            flags.append("vif")
        flagged_by.append("|".join(flags) if flags else "")

    report = pd.DataFrame({
        "feature": features,
        "mean_abs_r": mean_abs_r.values,
        "vif": vif_values,
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
    """Annotated Spearman correlation heatmap; dropped features labeled in red."""
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
                              config: dict, save_path: Path = None):
    # [DeGroot & Fienberg, 1983; Wilks, 2011; Brier, 1950]
    n_bins = config.get("reliability_bins", 20)
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


def mid(x):
    return (x[1:] + x[:-1]) / 2
