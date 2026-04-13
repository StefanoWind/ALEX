'''
Find relationaship between power outage and met signals at AWAKEN
'''

"""
Atmospheric Variable Importance Ranking for Power Outage Prediction
====================================================================
Handles:
  - Lagged predictors (atmospheric signal may precede outage)
  - Class imbalance (outages are rare)
  - Multiple importance methods: RF permutation, SHAP, cross-lag correlation
  - Both binary (outage flag) and continuous (customers out) targets

Usage:
    python atmo_importance_pipeline.py

Inputs (edit CONFIG section below):
    - CSV with datetime index, atmospheric columns, and a target column
    - Or pass DataFrames directly if using as a module

Outputs:
    - importance_results.csv    : ranked feature importances across methods
    - lag_correlation.png       : cross-correlation plot per variable
    - shap_summary.png          : SHAP beeswarm plot
    - importance_comparison.png : bar chart comparing RF vs SHAP ranks
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
import numpy as np
import os
import pandas as pd
cd=os.path.dirname(__file__)
import xarray as xr
from matplotlib import pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
import matplotlib
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             mean_squared_error)
from sklearn.utils.class_weight import compute_sample_weight

plt.close('all')
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi']=300

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Run `pip install shap` for SHAP importance.")
    
    
#%% Inputs
source='./data/siteA1_met_outages_15min.nc'
# target = 'outages'
# min_outages=10
# vars_=['wind_speed','wind_direction','wind_direction_std','temperature',
#        'relative_humidity','pressure','shortwave_radiation']


# n_bins=50
# perc_bins=[1,99]

# #graphics
# nrow=2
# ncol=4


# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = dict(
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path        = os.path.abspath("data/siteA1_met_outages_15min.nc"),   # path to input CSV
    datetime_col     = "time",        # column name or index
    target_col       = "outages",   # outage column (count or binary)

    # Atmospheric predictor columns (None = all non-target columns)
    predictor_cols   = ['wind_speed','wind_direction','wind_direction_std','temperature',
           'relative_humidity','pressure','shortwave_radiation'],

    # ── Target mode ───────────────────────────────────────────────────────────
    # "binary"     : predict outage yes/no (threshold applied to target_col)
    # "regression" : predict customers_out directly
    mode             = "binary",
    outage_threshold = 10,                 # customers_out > this → outage=1

    # ── Lag settings ──────────────────────────────────────────────────────────
    # Lags to explore (in time steps). E.g. if data is 1-min, lag=60 → 1 hour.
    lag_list         = [0,1,2,3,6,12],

    # Best single lag to use for model training (None = auto-selected by XCorr)
    best_lag         = None,
    min_corr = 0.1,

    # Also include rolling statistics as features?
    add_rolling      = True,
    rolling_windows  = [6],      # window sizes in time steps

    # ── Model settings ─────────────────────────────────────────────────────────
    n_estimators     = 500,
    max_features     = "sqrt",
    n_jobs           = -1,
    random_state     = 42,
    n_cv_folds       = 2,
    
    #stats
    limits={'wind_speed':[0,20],
            'wind_direction':[0,360],
            'wind_direction_std':[0,30],
            'temperature':[-10,45],
            'relative_humidity':[0,100],
            'pressure':[90,100],
            'shortwave_radiation':[0,0.2]},

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir       = os.path.abspath("data")
)

OUT = Path(CONFIG["output_dir"])


#%% Functions
def load_data(config: dict) -> pd.DataFrame:
    """Load CSV, parse datetime, return tidy DataFrame."""
    ds = xr.open_dataset(config["data_path"], parse_dates=[config["datetime_col"]])
    df=ds.to_dataframe()
    df = df.set_index(config["datetime_col"]).sort_index()
    return df

def qc_data(df: pd.DataFrame(),config: dict):
    '''
    Apply thresholds
    '''
    df_qc=df.DataFrame()
    for v in config.predictor_cols:
        df_qc[v]=df[v].where(df[v]>=config['limits'][v][0]).where(df[v]<=config['limits'][v][1])
        
    return df_qc
    
def make_binary_target(series: pd.Series, threshold: float) -> pd.Series:
    """Convert count column to binary outage flag."""
    return (series > threshold).astype(int)

def cross_lag_correlation(df: pd.DataFrame, predictors: list, target: pd.Series,
                          lag_list: list) -> pd.DataFrame:
    """
    For each predictor and each lag, compute Pearson correlation between
    predictor[t - lag] and target[t].

    Returns DataFrame: rows=lags, columns=predictors.
    """
    records = []
    for lag in lag_list:
        row = {"lag": lag}
        for col in predictors:
            shifted = df[col].shift(lag)          # shift predictor forward in time
            valid   = ~(shifted.isna() | target.isna())
            r       = shifted[valid].corr(target[valid])
            row[col] = r
        records.append(row)

    corr_df = pd.DataFrame(records).set_index("lag")
    return corr_df


def select_best_lag(corr_df: pd.DataFrame,config=dict) -> int:
    """
    Return the lag at which the mean |correlation| across all predictors
    is maximised.
    """
    id_max=corr_df.abs().idxmax()
    corr_max=corr_df.abs().max()
    id_max.loc[corr_max<config['min_corr']]=0
    return id_max

def plot_lag_correlation(corr_df: pd.DataFrame, top_n: int = 10,
                         save_path: Path = None):
    """Heatmap of cross-lag correlation for top-N predictors (by max |r|)."""
    # Select top_n most correlated predictors
    max_abs = corr_df.abs().max(axis=0).sort_values(ascending=False)
    top_cols = max_abs.head(top_n).index.tolist()
    subset   = corr_df[top_cols]

    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.8), 5))
    im = ax.imshow(subset.T.values, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(subset.index)))
    ax.set_xticklabels(subset.index, rotation=45, ha="right")
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols)
    ax.set_xlabel("Lag (time steps)")
    ax.set_title("Cross-Lag Correlation: Predictor[t−lag] vs Outage[t]")
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved lag correlation plot → {save_path}")

def build_feature_matrix(df: pd.DataFrame, predictors: list, target_col: str,
                         lag: dict, rolling_windows: list,
                         add_rolling: bool) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X and y with:
      - Each predictor shifted by `lag` time steps (predictor[t−lag])
      - Optional rolling mean and std features for each predictor
    """
    frames = {}

    for col in predictors:
        # Lagged value
        frames[f"{col}_lag{lag}"] = df[col].shift(lag[col])

        if add_rolling:
            for w in rolling_windows:
                # Rolling statistics computed BEFORE the lag offset
                frames[f"{col}_rollmean{w}_lag{lag}"] = (
                    df[col].rolling(w).mean().shift(lag[col])
                )
                frames[f"{col}_rollstd{w}_lag{lag}"]  = (
                    df[col].rolling(w).std().shift(lag[col])
                )

    X = pd.DataFrame(frames, index=df.index)
    y = df[target_col]

    # Drop rows with NaNs introduced by shifting/rolling
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]

def train_rf_importance(X: pd.DataFrame, y: pd.Series, config: dict) -> dict:
    """
    Train RF with cross-validation. Return:
      - mean permutation importance per feature
      - std permutation importance per feature
      - mean CV score
    """
    mode = config["mode"]
    n_folds = config["n_cv_folds"]

    if mode == "binary":
        model_cls = RandomForestClassifier
        cv        = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                    random_state=config["random_state"])
        score_fn  = lambda m, Xt, yt: roc_auc_score(yt, m.predict_proba(Xt)[:,1])
        score_name = "ROC-AUC"
    else:
        model_cls = RandomForestRegressor
        cv        = KFold(n_splits=n_folds, shuffle=True,
                          random_state=config["random_state"])
        score_fn  = lambda m, Xt, yt: -mean_squared_error(yt, m.predict(Xt),
                                                           squared=False)
        score_name = "neg-RMSE"

    X_arr = X.values
    y_arr = y.values

    imp_list   = []
    cv_scores  = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        # Handle class imbalance via sample weights
        if mode == "binary":
            sw = compute_sample_weight("balanced", y_tr)
        else:
            sw = None

        model = model_cls(
            n_estimators = config["n_estimators"],
            max_features = config["max_features"],
            n_jobs       = config["n_jobs"],
            random_state = config["random_state"],
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

        # Validation score
        cv_scores.append(score_fn(model, X_val, y_val))

        # Permutation importance on validation fold
        perm = permutation_importance(model, X_val, y_val,
                                      n_repeats=10,
                                      random_state=config["random_state"],
                                      n_jobs=config["n_jobs"])
        imp_list.append(perm.importances_mean)

        print(f"    Fold {fold_i+1}/{n_folds} | {score_name}: "
              f"{cv_scores[-1]:.4f}")

    mean_imp = np.mean(imp_list, axis=0)
    std_imp  = np.std(imp_list, axis=0)

    print(f"  Mean CV {score_name}: {np.mean(cv_scores):.4f} "
          f"± {np.std(cv_scores):.4f}")

    return {
        "feature_names" : list(X.columns),
        "importance_mean": mean_imp,
        "importance_std" : std_imp,
        "cv_scores"      : cv_scores,
        "score_name"     : score_name,
    }

def compute_shap_importance(X: pd.DataFrame, y: pd.Series,
                            config: dict, n_sample: int = 2000,
                            save_path: Path = None) -> pd.Series:
    """
    Fit one RF on the full dataset, compute SHAP values,
    return mean |SHAP| per feature.
    """
    if not HAS_SHAP:
        print("  SHAP skipped (not installed).")
        return pd.Series(dtype=float)

    mode = config["mode"]
    if mode == "binary":
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            n_jobs=config["n_jobs"],
            random_state=config["random_state"])
        sw = compute_sample_weight("balanced", y)
    else:
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            n_jobs=config["n_jobs"],
            random_state=config["random_state"])
        sw = None

    model.fit(X, y, sample_weight=sw)

    # Subsample for speed
    rng     = np.random.default_rng(config["random_state"])
    idx     = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_samp  = X.iloc[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_samp)

    # For classifier, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # use class-1 (outage) SHAP values

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
        name="mean_abs_SHAP"
    )

    # Beeswarm plot
    if save_path:
        shap.summary_plot(shap_values, X_samp, show=False,
                          max_display=20, plot_size=(10, 7))
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP summary plot → {save_path}")

    return mean_shap

def aggregate_to_original_vars(importance_series: pd.Series,
                                predictors: list) -> pd.Series:
    """
    Sum importance across all features derived from the same original variable
    (lag + rolling variants). Returns importance indexed by original var name.
    """
    agg = {}
    for var in predictors:
        mask = importance_series.index.str.startswith(var + "_")
        agg[var] = importance_series[mask].sum()
    return pd.Series(agg).sort_values(ascending=False)

def build_results_table(rf_result: dict, shap_imp: pd.Series,
                        predictors: list) -> pd.DataFrame:
    """Merge RF permutation and SHAP importances, aggregate by original var."""
    rf_imp = pd.Series(rf_result["importance_mean"],
                       index=rf_result["feature_names"], name="RF_perm_imp")
    rf_std = pd.Series(rf_result["importance_std"],
                       index=rf_result["feature_names"], name="RF_perm_std")

    # Aggregate by original variable
    rf_agg   = aggregate_to_original_vars(rf_imp,  predictors)
    rf_std_agg = aggregate_to_original_vars(rf_std, predictors)

    result = pd.DataFrame({
        "RF_perm_importance": rf_agg,
        "RF_perm_std":        rf_std_agg,
    })

    if len(shap_imp) > 0:
        shap_agg = aggregate_to_original_vars(shap_imp, predictors)
        result["SHAP_mean_abs"] = shap_agg

    result = result.sort_values("RF_perm_importance", ascending=False)
    result["RF_rank"]   = range(1, len(result)+1)
    if "SHAP_mean_abs" in result.columns:
        result["SHAP_rank"] = result["SHAP_mean_abs"].rank(
            ascending=False).astype(int)

    return result

def plot_importance_comparison(results: pd.DataFrame,
                               save_path: Path = None, top_n: int = 20):
    """Horizontal bar chart of RF and SHAP importances for top-N variables."""
    subset = results.head(top_n)
    n      = len(subset)
    cols   = [c for c in ["RF_perm_importance", "SHAP_mean_abs"]
              if c in subset.columns]
    n_cols = len(cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, max(5, n * 0.4)),
                             sharey=True)
    if n_cols == 1:
        axes = [axes]

    labels = {
        "RF_perm_importance": "RF Permutation Importance",
        "SHAP_mean_abs":      "Mean |SHAP| Value",
    }
    colors = ["#2980b9", "#e67e22"]

    for ax, col, color in zip(axes, cols, colors):
        vals = subset[col].values[::-1]
        errs = subset.get("RF_perm_std", pd.Series(0, index=subset.index)
                          ).values[::-1] if col == "RF_perm_importance" else None
        ypos = np.arange(n)
        ax.barh(ypos, vals, xerr=errs, color=color, alpha=0.85,
                error_kw=dict(elinewidth=1, capsize=3))
        ax.set_yticks(ypos)
        ax.set_yticklabels(subset.index[::-1], fontsize=9)
        ax.set_xlabel(labels[col])
        ax.set_title(labels[col])
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Atmospheric Variable Importance for Power Outage Prediction",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved importance comparison plot → {save_path}")

def lag_sweep_importance(df: pd.DataFrame, predictors: list,
                         target_col: str, lag_list: list,
                         config: dict) -> pd.DataFrame:
    """
    Train a quick RF at each lag (no rolling, no CV, just OOB score) and
    return a DataFrame of feature importances vs lag.

    Useful for identifying at which lag each variable is most predictive.
    """
    records = []
    for lag in lag_list:
        print(f"  Lag sweep: lag={lag} ...")
        X_lag = pd.DataFrame(
            {col: df[col].shift(lag) for col in predictors}, index=df.index
        )
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

def plot_lag_sweep(sweep_df: pd.DataFrame, top_n: int = 8,
                   save_path: Path = None):
    """Line plot of RF impurity importance vs lag for top-N variables."""
    imp_cols = [c for c in sweep_df.columns if c != "oob_score"]
    # Pick top_n by max importance across lags
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
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved lag sweep plot → {save_path}")


def run_pipeline(config: dict = CONFIG, df: pd.DataFrame = None):
    """
    Full pipeline. Pass df directly (with datetime index) or set
    config["data_path"] to load from CSV.
    """
    print("=" * 65)
    print("  Atmospheric Variable Importance Pipeline")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    if df is None:
        print("\n[1] Loading data ...")
        df = load_data(config)
    else:
        print("\n[1] Using provided DataFrame.")

    # Resolve predictor columns
    predictors = config["predictor_cols"] or [
        c for c in df.columns if c != config["target_col"]
    ]
    print(f"    Predictors ({len(predictors)}): {predictors}")

    # Build target
    if config["mode"] == "binary":
        df["__target__"] = make_binary_target(df[config["target_col"]],
                                              config["outage_threshold"])
        outage_rate = df["__target__"].mean()
        print(f"    Outage rate: {outage_rate:.3%}")
    else:
        df["__target__"] = df[config["target_col"]]

    # ── Cross-lag correlation ─────────────────────────────────────────────
    print("\n[2] Cross-lag correlation analysis ...")
    corr_df = cross_lag_correlation(df, predictors, df["__target__"],
                                    config["lag_list"])
    plot_lag_correlation(corr_df, top_n=min(10, len(predictors)),
                         save_path=OUT / "lag_correlation.png")

    best_lag = config["best_lag"]
    if best_lag is None:
        best_lag = select_best_lag(corr_df,config)
    print(f"    Best lag (auto-selected): \n {best_lag} time steps")

    # ── Lag sweep (RF impurity, fast) ─────────────────────────────────────
    print("\n[3] Lag sweep (quick RF at each lag) ...")
    sweep_df = lag_sweep_importance(df, predictors, "__target__",
                                    config["lag_list"], config)
    plot_lag_sweep(sweep_df, save_path=OUT / "lag_sweep.png")

    # ── Build feature matrix at best lag ─────────────────────────────────
    print(f"\n[4] Building feature matrix at lag={best_lag} ...")
    X, y = build_feature_matrix(df, predictors, "__target__",
                                 lag=best_lag,
                                 rolling_windows=config["rolling_windows"],
                                 add_rolling=config["add_rolling"])
    print(f"    Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # ── RF permutation importance (cross-validated) ───────────────────────
    print("\n[5] Training RF + permutation importance (CV) ...")
    rf_result = train_rf_importance(X, y, config)

    # ── SHAP importance ───────────────────────────────────────────────────
    print("\n[6] Computing SHAP importance ...")
    shap_imp = compute_shap_importance(X, y, config,
                                       save_path=OUT / "shap_summary.png")

    # ── Compile results ───────────────────────────────────────────────────
    print("\n[7] Compiling results table ...")
    results = build_results_table(rf_result, shap_imp, predictors)
    results.to_csv(OUT / "importance_results.csv")
    print(f"    Saved importance table → {OUT / 'importance_results.csv'}")
    print("\n  Top 10 variables:")
    print(results.head(10).to_string())

    plot_importance_comparison(results,
                               save_path=OUT / "importance_comparison.png")

    print("\n✓ Pipeline complete.")
    return results, corr_df, sweep_df

def generate_synthetic_data(n: int = 100_000,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Synthetic atmospheric + outage dataset for testing.
    Wind gust at lag=30 is the main driver; temperature at lag=10 is secondary.
    """
    rng   = np.random.default_rng(random_state)
    dates = pd.date_range("2020-01-01", periods=n, freq="1min")

    wind_speed  = np.abs(rng.normal(8, 4, n))
    wind_direction   =  rng.uniform(0, 360, n)
    wind_direction_std   =  rng.normal(10, 5, n)
    temperature = rng.normal(15, 8, n)
    pressure    = rng.normal(95, 5, n)
    relative_humidity    = rng.uniform(20, 100, n)
    shortwave_radiation=rng.normal(0.1,0.05,n)
   
    # Outage probability driven by lagged wind_gust and lagged temperature
    lag_w = np.roll(wind_speed, 2)
    lag_t = np.roll(temperature, 6)
    logit = -6 + 0.5 * lag_w - 0.5 * lag_t
    prob  = 1 / (1 + np.exp(-logit))
    outages = (rng.random(n) < prob).astype(int) * rng.integers(1, 500, n)

    df = pd.DataFrame({
        "wind_speed":   wind_speed,
        "wind_direction":    wind_direction,
        "wind_direction_std":    wind_direction_std,
        "temperature":  temperature,
        "pressure":     pressure,
        "relative_humidity":     relative_humidity,
        "shortwave_radiation":    shortwave_radiation,
        "outages": outages,
    }, index=dates)

    return df

if __name__ == "__main__":
    print("Running with SYNTHETIC data (replace with your CSV path).\n")

    df_synth = generate_synthetic_data(n=100_000)

    cfg = CONFIG.copy()
  
    results, corr_df, sweep_df = run_pipeline(config=cfg, df=df_synth)
    

def mid(x):
    '''
    Midpoint in array
    '''
    
    return (x[1:]+x[:-1])/2

    



# bins_y=np.linspace(np.nanpercentile(Data[target],perc_bins[0]), 
#                  np.nanpercentile(Data[target],perc_bins[1]+0.1),n_bins)
                 
# #%% Plots

# #all 2-D histograms
# fig = plt.figure(figsize=(18,8))
# gs = GridSpec(nrow, ncol+1,
# figure=fig,
# width_ratios=ncol*[1]+[0.05])
# ctr=0
# for v in vars_:
#         bins_x=np.linspace(np.nanpercentile(Data[v],perc_bins[0]), 
#                          np.nanpercentile(Data[v],perc_bins[1]+0.1),n_bins)
#         N=stats.binned_statistic_2d(Data[v].values, 
#                                     Data[target].values,
#                                     Data[v].values,
#                                     statistic='count',
#                                     bins=(bins_x,bins_y))[0]
        
#         ax = fig.add_subplot(gs[ctr+int(ctr/ncol)])
        
#         pc=plt.pcolor(mid(bins_x),mid(bins_y),np.log10(N.T/N.max()),cmap='inferno',vmin=-3,vmax=0)
#         plt.xlabel(v)
#         plt.ylabel(target)
#         ax.set_facecolor('k')
#         plt.grid()
#         ctr+=1
        
# cax = fig.add_subplot(gs[:, -1])
# cbar = fig.colorbar(pc, cax=cax,label='Normalized count')
# cbar.set_ticks(np.arange(-3,1))
# cbar.set_ticklabels([r'$10^{'+str(i)+'}$' for i in np.arange(-3,1)])
# plt.tight_layout()
        



