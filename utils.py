import numpy as np
import os
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import stats
from matplotlib.gridspec import GridSpec
import matplotlib
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime
import yaml

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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    """Plot 2-D histogram of feature vs. target."""
    mode = config.get('mode')
    if mode in ('AUC', 'TOT') and '__target__' in df.columns:
        y_series = df['__target__']
        y_label = mode
        bins_y = np.linspace(0,
                             np.nanpercentile(y_series, config['perc_bins'][1] + 0.1),
                             config['n_bins'])
    else:
        y_series = df[config['target_col']]
        y_label = config['target_col']
        bins_y = np.linspace(config['outage_threshold'],
                             np.nanpercentile(y_series, config['perc_bins'][1] + 0.1),
                             config['n_bins'])

    fig = plt.figure(figsize=(18, 4 * config['nrow']))
    gs = GridSpec(config['nrow'], config['ncol'] + 1, figure=fig,
                  width_ratios=config['ncol'] * [1] + [0.05])
    ctr = 0
    for col in config['predictor_cols']:
        if window is not None and lag is not None:
            if window[col] > 0:
                shifted = df[col].rolling(window[col], center=True).mean().shift(lag[col])
            else:
                shifted = df[col].shift(lag[col])
        else:
            shifted = df[col]

        bins_x = np.linspace(np.nanpercentile(shifted, config['perc_bins'][0]),
                              np.nanpercentile(shifted, config['perc_bins'][1] + 0.1),
                              config['n_bins'])

        N = stats.binned_statistic_2d(shifted.values,
                                      y_series.values,
                                      shifted.values,
                                      statistic='count',
                                      bins=(bins_x, bins_y))[0]

        ax = fig.add_subplot(gs[ctr + int(ctr / config['ncol'])])
        pc = plt.pcolor(mid(bins_x), mid(bins_y), np.log10(N.T / N.max()),
                        cmap='inferno', vmin=-3, vmax=0)

        if window is not None and lag is not None:
            if window[col] > 0:
                plt.xlabel(f"{col}_avg (roll={window[col]}, lag={lag[col]})")
            else:
                plt.xlabel(f"{col} (lag={lag[col]})")
        else:
            plt.xlabel(f"{col}")
        plt.ylabel(y_label)
        ax.set_facecolor('k')
        plt.grid()
        ctr += 1

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(pc, cax=cax, label='Normalized count')
    cbar.set_ticks(np.arange(-3, 1))
    cbar.set_ticklabels([r'$10^{' + str(i) + '}$' for i in np.arange(-3, 1)])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved histograms plot → {save_path}")


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

    fig, ax = plt.subplots(figsize=(max(18, len(subset.columns) * 0.8), 5))
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
        fig.savefig(save_path, dpi=300)
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
def build_dynamic_features(df: pd.DataFrame, predictors: list, target_col: str,
                            lag: dict, windows: list) -> tuple[pd.DataFrame, pd.Series]:
    frames = {}
    for col in predictors:
        L = lag[col]
        x = df[col]
        frames[f"{col}_raw (lag={L})"] = x.shift(L)
        frames[f"{col}_grad (lag={L})"] = x.diff(1).shift(L)
        for w in windows:
            frames[f"{col}_mean_W{w} (lag={L})"] = x.rolling(w).mean().shift(L)
            frames[f"{col}_std_W{w} (lag={L})"] = x.rolling(w).std().shift(L)
            frames[f"{col}_maxgrad_W{w} (lag={L})"] = x.diff(1).rolling(w).max().shift(L)
            rolling_med = x.rolling(w).median()
            rolling_mad = (x - rolling_med).abs().rolling(w).median()
            spike_z = (x - rolling_med) / rolling_mad.replace(0, np.nan)
            frames[f"{col}_spikez_W{w} (lag={L})"] = spike_z.shift(L)

    X = pd.DataFrame(frames, index=df.index)
    y = df[target_col]
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


def _mode_label(config: dict) -> str:
    mode = config.get('mode', '')
    units = {'AUC': 'customer·h', 'TOT': 'min'}
    return f"{mode} ({units[mode]})" if mode in units else mode


def plot_rf_scatter(y_true: pd.Series, y_pred: pd.Series, config: dict,
                    save_path: Path = None):
    """Scatter of observed vs OOF RF prediction with 1:1 line and linear regression."""
    valid = ~(y_true.isna() | y_pred.isna())
    yt, yp = y_true[valid].values, y_pred[valid].values
    slope, intercept, r, _, _ = stats.linregress(yt, yp)
    x_line = np.array([yp.min(), yp.max()])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=2, alpha=0.4, color='steelblue')
    ax.plot(x_line, x_line, 'k--', lw=1, label='1:1')
    ax.plot(x_line, slope * x_line + intercept, 'r-', lw=1,
            label=f'linear fit  r={r:.2f}')
    label = _mode_label(config)
    ax.set_ylabel(f'RF prediction [{label}]')
    ax.set_xlabel(f'Observed [{label}]')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved RF time series plot → {save_path}")


def train_rf_importance(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    """
    Train RF with cross-validation and return permutation importances. [Breiman, 2001]
    """
    mode = config["mode"]
    n_folds = config["n_cv_folds"]

    if mode == "binary":
        model_cls = RandomForestClassifier
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=config["random_state"])
        score_fn = lambda m, Xt, yt: roc_auc_score(yt, m.predict_proba(Xt)[:, 1])
        score_name = "ROC-AUC"
    else:
        model_cls = RandomForestRegressor
        cv = KFold(n_splits=n_folds, shuffle=True,
                   random_state=config["random_state"])
        score_fn = lambda m, Xt, yt: -mean_squared_error(yt, m.predict(Xt))
        score_name = "neg-RMSE"

    X_arr = X.values
    y_arr = y.values
    imp_list = []
    cv_scores = []
    oof_pred = np.full(len(y_arr), np.nan)

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        sw = compute_sample_weight("balanced", y_tr) if mode == "binary" else None

        model = model_cls(
            n_estimators=config["n_estimators"],
            max_features=config["max_features"],
            n_jobs=config["n_jobs"],
            random_state=config["random_state"],
        )
        model.fit(X_tr, y_tr, sample_weight=sw)
        cv_scores.append(score_fn(model, X_val, y_val))

        if mode != "binary":
            oof_pred[val_idx] = model.predict(X_val)

        perm = permutation_importance(model, X_val, y_val,
                                      n_repeats=config["importance_reps"],
                                      random_state=config["random_state"],
                                      n_jobs=config["n_jobs"])
        imp_list.append(perm.importances_mean)
        print(f"    Fold {fold_i + 1}/{n_folds} | {score_name}: {cv_scores[-1]:.4f}")

    mean_imp = np.mean(imp_list, axis=0)
    std_imp = np.std(imp_list, axis=0)
    print(f"  Mean CV {score_name}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return {
        "feature_names": list(X.columns),
        "importance_mean": mean_imp,
        "importance_std": std_imp,
        "cv_scores": cv_scores,
        "score_name": score_name,
        "oof_pred": pd.Series(oof_pred, index=y.index) if mode != "binary" else None,
    }


def compute_shap_importance(X: pd.DataFrame, y: pd.Series,
                            config: dict, n_sample: int = 2000,
                            save_path: Path = None) -> pd.Series:
    """
    Compute SHAP feature importances using TreeExplainer. [Lundberg & Lee, 2017]
    """
    if not HAS_SHAP:
        print("  SHAP skipped (not installed).")
        return pd.Series(dtype=float)

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

    rng = np.random.default_rng(config["random_state"])
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_samp = X.iloc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_samp)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_shap = pd.Series(np.abs(shap_values).mean(axis=0),
                          index=X.columns, name="mean_abs_SHAP")

    if save_path:
        shap.summary_plot(shap_values, X_samp, show=False,
                          max_display=20, plot_size=(10, 7))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP summary plot → {save_path}")

    return mean_shap


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
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved importance comparison plot → {save_path}")


def lag_sweep_importance(df: pd.DataFrame, predictors: list,
                         target_col: str, lag_list: list,
                         config: dict) -> pd.DataFrame:
    """Quick RF importance at each lag (OOB only, no CV). [Breiman, 2001]"""
    records = []
    for lag in lag_list:
        print(f"  Lag sweep: lag={lag} ...")
        X_lag = pd.DataFrame({col: df[col].shift(lag) for col in predictors},
                             index=df.index)
        y_lag = df[target_col]
        valid = X_lag.notna().all(axis=1) & y_lag.notna()
        X_v, y_v = X_lag[valid].values, y_lag[valid].values

        if config["mode"] == "binary":
            sw = compute_sample_weight("balanced", y_v)
            rf = RandomForestClassifier(n_estimators=200, oob_score=True,
                                        n_jobs=config["n_jobs"],
                                        random_state=config["random_state"])
        else:
            sw = None
            rf = RandomForestRegressor(n_estimators=200, oob_score=True,
                                       n_jobs=config["n_jobs"],
                                       random_state=config["random_state"])
        rf.fit(X_v, y_v, sample_weight=sw)

        row = {"lag": lag, "oob_score": rf.oob_score_}
        row.update(dict(zip(predictors, rf.feature_importances_)))
        records.append(row)

    return pd.DataFrame(records).set_index("lag")


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


def plot_lag_sweep(sweep_df: pd.DataFrame, top_n: int = 8,
                   save_path: Path = None):
    """Line plot of RF impurity importance vs lag for top-N variables."""
    imp_cols = [c for c in sweep_df.columns if c != "oob_score"]
    top_vars = sweep_df[imp_cols].max(axis=0).nlargest(top_n).index.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for var in top_vars:
        axes[0].plot(sweep_df.index, sweep_df[var], marker="o",
                     label=var, linewidth=1.5)
    axes[0].set_ylabel("RF Impurity Importance")
    axes[0].set_title("Feature Importance vs. Lag")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(sweep_df.index, sweep_df["oob_score"], color="black",
                 marker="s", linewidth=1.5)
    axes[1].set_ylabel("OOB Score")
    axes[1].set_xlabel("Lag (time steps)")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved lag sweep plot → {save_path}")


def mid(x):
    return (x[1:] + x[:-1]) / 2


def run_pipeline(config: dict, df: pd.DataFrame, out_dir: Path = None,
                 best_lag: dict | None = None):
    """Orchestrate the full variable importance pipeline."""
    if out_dir is None:
        OUT = Path(config['output_dir']) / datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
    else:
        OUT = Path(out_dir)
    os.makedirs(OUT, exist_ok=True)
    with open(OUT / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print("=" * 65)
    print("  Atmospheric Variable Importance Pipeline")
    print("=" * 65)
    print("\n[1] Using provided DataFrame.")

    plot_time_series(df, config, save_path=OUT / "time_series.png")
    plot_histograms(df, config, lag=None, save_path=OUT / "histograms.png")

    predictors = config["predictor_cols"] or [
        c for c in df.columns if c != config["target_col"]
    ]
    print(f"    Predictors ({len(predictors)}): {predictors}")

    if config["mode"] == "binary":
        df["__target__"] = make_binary_target(df[config["target_col"]],
                                              config["outage_threshold"])
        print(f"    Outage rate: {df['__target__'].mean():.3%}")
    elif config["mode"] in ("AUC", "TOT"):
        if "__target__" not in df.columns:
            df["__target__"] = make_segment_target(df[config["target_col"]],
                                                   config["outage_threshold"],
                                                   config["mode"])
            plot_segment_zoom(df[config["target_col"]], df["__target__"],
                              config["outage_threshold"], config["mode"],
                              save_path=OUT / "segment_zoom.png")
        nonzero = (df["__target__"] > 0)
        print(f"    {config['mode']} target: non-zero fraction = {nonzero.mean():.3%}, "
              f"mean (non-zero) = {df['__target__'][nonzero].mean():.2f}")
    else:
        df["__target__"] = df[config["target_col"]]

    if config.get("detrend_seasonal", False):
        print("\n[2] Removing seasonal cycle ...")
        inplace = config.get("detrend_mode", "anomaly") == "inplace"
        df, climatology = remove_seasonal_cycle(
            df,
            columns=predictors,
            window_days=config.get("detrend_window_days", 7),
            min_periods=config.get("detrend_min_periods", 3),
            inplace=inplace,
            save_climatology_path=OUT / "climatology.csv",
        )
        if not inplace:
            predictors = [f"{c}_anom" for c in predictors]
            print("    Predictor columns updated to anomaly variants.")

        first_orig = (config["predictor_cols"][0] if config["predictor_cols"]
                      else [c for c in df.columns
                            if c not in ("__target__", config["target_col"])][0])
        if not inplace and first_orig in df.columns:
            plot_seasonal_detrending(
                df_raw=df, df_anom=df, climatology=climatology,
                column=first_orig,
                save_path=OUT / f"detrending_{first_orig}.png",
            )
        print("    Seasonal detrending complete.")
    else:
        print("\n[2] Seasonal detrending skipped.")

    print("\n[2] Cross-lag correlation analysis ...")
    if best_lag is None:
        corr_df = cross_lag_correlation(df, predictors, df["__target__"],
                                        config["lag_list"])
        plot_lag_correlation(corr_df, save_path=OUT / "lag_correlation.png",
                             target_name=config['target_col'])
        best_lag = select_best_lag(corr_df, config)
        print(f"    Best lag: \n{best_lag} time steps")
    else:
        corr_df = None
        print(f"    Using pre-supplied lag: {best_lag}")
    plot_histograms(df, config, lag=best_lag,
                    window={col: 0 for col in best_lag},
                    save_path=OUT / "histograms_lag.png")

    if config['prelim_rf']:
        print("\n[3] Lag sweep (quick RF at each lag) ...")
        sweep_df = lag_sweep_importance(df, predictors, "__target__",
                                        config["lag_list"], config)
        plot_lag_sweep(sweep_df, save_path=OUT / "lag_sweep.png")
    else:
        sweep_df = None

    print(f"\n[4] Building feature matrix at lag\n{best_lag} ...")
    dynamic_windows = config.get('dynamic_windows', [])
    if dynamic_windows:
        X, y = build_dynamic_features(df, predictors, "__target__",
                                       lag=best_lag, windows=dynamic_windows)
    else:
        X, y = build_feature_matrix(df, predictors, "__target__", lag=best_lag)
    print(f"    Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    print("\n[5] Checking feature collinearity ...")
    X, dropped_features, collinearity_report = check_collinearity(
        X,
        r_threshold=config["collinearity_r_threshold"],
        vif_threshold=config["collinearity_vif_threshold"],
        action=config["collinearity_action"],
        save_path=OUT / "collinearity_heatmap.png",
    )
    collinearity_report.to_csv(OUT / "collinearity_report.csv")
    print(f"    Saved collinearity report → {OUT / 'collinearity_report.csv'}")
    if dropped_features:
        print(f"    Proceeding with {X.shape[1]} features "
              f"(dropped {len(dropped_features)}: {dropped_features})")
    else:
        print(f"    No features dropped; proceeding with {X.shape[1]} features.")

    print("\n[5] Training RF + permutation importance (CV) ...")
    rf_result = train_rf_importance(X, y, config)

    if config["mode"] != "binary" and rf_result["oof_pred"] is not None:
        plot_rf_scatter(y, rf_result["oof_pred"], config,
                        save_path=OUT / "rf_scatter.png")
        plot_rf_timeseries(y, rf_result["oof_pred"], config,
                           save_path=OUT / "rf_timeseries.png")

    if config['shap']:
        print("\n[6] Computing SHAP importance ...")
        shap_imp = compute_shap_importance(X, y, config,
                                           save_path=OUT / "shap_summary.png")
    else:
        shap_imp = []

    print("\n[7] Compiling results table ...")
    results = build_results_table(rf_result, shap_imp, predictors)
    results.to_csv(OUT / "importance_results.csv")
    print(f"    Saved importance table → {OUT / 'importance_results.csv'}")
    print("\n  Top 10 variables:")
    print(results.head(10).to_string())

    plot_importance_comparison(results, rf_result,
                               save_path=OUT / "importance_comparison.png")

    print("\nPipeline complete.")
    return results, corr_df, sweep_df
