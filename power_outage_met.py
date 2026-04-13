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
from pathlib import Path
from scipy import stats
from matplotlib.gridspec import GridSpec
import matplotlib
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime

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


# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = dict(
    
    # __ User ──────────────────────────────────────────────────────────────────
    synthetic = False,
    prelim_rf = False,
    shap = False,
    add_rolling      = True,
    
    # ── Data ──────────────────────────────────────────────────────────────────
    source        = os.path.abspath("data/siteA1_met_outages_15min_v2.nc"),   # path to input CSV
    datetime_col     = "time",        # column name or index
    target_col       = "outages",   # outage column (count or binary)

    # Atmospheric predictor columns (None = all non-target columns)
    predictor_cols   = ['wind_speed','wind_direction_uv','wind_direction_std','temperature',
           'relative_humidity','pressure','shortwave_radiation'],

    # ── Target mode ───────────────────────────────────────────────────────────
    # "binary"     : predict outage yes/no (threshold applied to target_col)
    # "regression" : predict customers_out directly
    mode             = "binary",
    outage_threshold = 10,                 # customers_out > this → outage=1

    # ── Lag settings ──────────────────────────────────────────────────────────
    # Lags to explore (in time steps). 
    lag_list         = [0,1,3,6,12,36,144],

    # Best single lag to use for model training (None = auto-selected by XCorr)
    best_lag         = None,
    min_corr = 0.1,

    # Also include rolling statistics as features?
    rolling_windows  = [3,6,144],      # window sizes in time steps
    
    # ── Collinearity settings ──────────────────────────────────────────────────
    # Spearman |r| above this → flag as collinear pair
    collinearity_r_threshold  = 0.85,
    # VIF above this → flag feature as redundant (10 is conventional; use 5 for
    # stricter control). Set to None to skip VIF (slow for wide feature matrices).
    collinearity_vif_threshold = 10.0,
    # "warn"  → report collinear features but keep all of them
    # "drop"  → automatically drop the lower-importance member of each pair
    collinearity_action       = "drop",

    # ── RF settings ─────────────────────────────────────────────────────────
    n_estimators     = 500,
    max_features     = "sqrt",
    n_jobs           = -1,
    random_state     = 42,
    n_cv_folds       = 2,
    importance_reps  = 10,
    
    #── Stats ─────────────────────────────────────────────────────────
    limits={'wind_speed':[0,20],
            'wind_direction_uv':[0,360],
            'wind_direction_std':[0,30],
            'temperature':[-10,45],
            'relative_humidity':[0,100],
            'pressure':[90,100],
            'shortwave_radiation':[0,0.2]},

    # ── Output ─────────────────────────────────────────────────────────────────
    output_dir       = os.path.abspath("data"),
    
    # ── Graphics ─────────────────────────────────────────────────────────────────
    n_bins=50,
    perc_bins=[1,99],
    nrow=2,
    ncol=4,

)

OUT = Path(os.path.join(CONFIG["output_dir"],datetime.strftime(datetime.now(),'%Y%m%d.%H%M%S')))
os.makedirs(OUT,exist_ok=True)

#%% Functions
def load_data(config: dict) -> pd.DataFrame:
    """Load CSV, parse datetime, return tidy DataFrame."""
    ds = xr.open_dataset(config["source"])
    df=ds[config['predictor_cols']+[config['target_col']]].to_dataframe()
    return df

def qc_data(df: pd.DataFrame(),config: dict):
    '''
    Apply thresholds
    '''
    df_qc=pd.DataFrame()
    for v in config['predictor_cols']:
        df_qc[v]=df[v].where(df[v]>=config['limits'][v][0]).where(df[v]<=config['limits'][v][1])
    df_qc[config['target_col']]=df[config['target_col']]
    
    return df_qc
    
def make_binary_target(series: pd.Series, threshold: float) -> pd.Series:
    """Convert count column to binary outage flag."""
    return (series > threshold).astype(int)

def plot_histograms(df: pd.DataFrame(),config: dict, lag: dict = None, save_path: Path = None):
    """ Plot 2-D histogram of feature vs. target"""
    
    #bins of target
    bins_y=np.linspace(config['outage_threshold'], 
                       np.nanpercentile(df[config['target_col']],config['perc_bins'][1]+0.1),config['n_bins'])
    
    #loop through features
    fig = plt.figure(figsize=(18,8))
    gs = GridSpec(config['nrow'], config['ncol']+1,
    figure=fig,
    width_ratios=config['ncol']*[1]+[0.05])
    ctr=0
    for col in config['predictor_cols']:
        if lag is not None:
            shifted = df[col].shift(lag[col])
        else:
            shifted = df[col].copy()
            
        bins_x=np.linspace(np.nanpercentile(shifted,config['perc_bins'][0]), 
                           np.nanpercentile(shifted,config['perc_bins'][1]+0.1),config['n_bins'])
        
        N=stats.binned_statistic_2d(shifted.values, 
                                    df[config['target_col']].values,
                                    shifted.values,
                                    statistic='count',
                                    bins=(bins_x,bins_y))[0]
        
        ax = fig.add_subplot(gs[ctr+int(ctr/config['ncol'])])
        
        pc=plt.pcolor(mid(bins_x),mid(bins_y),np.log10(N.T/N.max()),cmap='inferno',vmin=-3,vmax=0)
        plt.xlabel(col)
        plt.ylabel(config['target_col'])
        ax.set_facecolor('k')
        plt.grid()
        ctr+=1
            
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(pc, cax=cax,label='Normalized count')
    cbar.set_ticks(np.arange(-3,1))
    cbar.set_ticklabels([r'$10^{'+str(i)+'}$' for i in np.arange(-3,1)])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved histograms plot → {save_path}")
    
def cross_lag_correlation(df: pd.DataFrame, predictors: list, target: pd.Series,
                          lag_list: list) -> pd.DataFrame:
    """
    For each predictor and each lag, compute Pearson correlation between
    predictor[t - lag] and target[t].

    Returns DataFrame: rows=lags, columns=predictors.
    """
    records = []
    mirror_lags=np.unique(np.concat([-np.array(lag_list),np.array(lag_list)]))
    for lag in mirror_lags:
        row = {"lag": lag}
        for col in predictors:
            shifted = df[col].shift(lag)          # shift predictor forward in time
            valid   = ~(shifted.isna() | target.isna())
            r       = shifted[valid].corr(target[valid])
            row[col] = r
        records.append(row)

    corr_df = pd.DataFrame(records).set_index("lag")
    return corr_df

def select_best_lag(corr_df: pd.DataFrame,config=dict) -> dict:
    """
    Return the lag at which the mean |correlation| across all predictors
    is maximised.
    """
    id_max=corr_df.abs().idxmax()
    corr_max=corr_df.abs().max()
    id_max.loc[corr_max<config['min_corr']]=0
    return dict(id_max)

def plot_lag_correlation(corr_df: pd.DataFrame, top_n: int = 10,
                         save_path: Path = None):
    """Heatmap of cross-lag correlation for top-N predictors (by max |r|)."""
    # Select top_n most correlated predictors
    max_abs = corr_df.abs().max(axis=0).sort_values(ascending=False)
    top_cols = max_abs.head(top_n).index.tolist()
    subset   = corr_df[top_cols]
    max_corr=subset.abs().max().max()

    fig, ax = plt.subplots(figsize=(max(18, top_n * 0.8), 5))
    im = ax.imshow(subset.T.values, aspect="auto", cmap="RdBu_r",
                   vmin=-max_corr, vmax=max_corr)
    for j in range(len(subset.columns)):
        for i in range(len(subset.index)):
            if subset.iloc[i,j]>0:
                color='g'
            else:
                color='orange'
            plt.text(i-0.25,j,f"{subset.iloc[i,j]:02.3f}",fontsize=8,color=color)
    ax.set_xticks(range(len(subset.index)))
    ax.set_xticklabels(subset.index, rotation=45, ha="right")
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols)
    ax.set_xlabel("Lag (time steps)")
    ax.set_title("Cross-Lag Correlation: Predictor[t−lag] vs Outage[t]")
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
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
        frames[f"{col}_lag{lag[col]}"] = df[col].shift(lag[col])

        if add_rolling:
            for w in rolling_windows:
                # Rolling statistics computed BEFORE the lag offset
                frames[f"{col}_rollmean{w}_lag{lag[col]}"] = (
                    df[col].rolling(w).mean().shift(lag[col])
                )
                frames[f"{col}_rollstd{w}_lag{lag[col]}"]  = (
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
                                      n_repeats=config["importance_reps"],
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

def _vif_series(X_arr: np.ndarray) -> np.ndarray:
    """
    Compute Variance Inflation Factor for each column of X_arr.
 
    VIF_j = 1 / (1 - R²_j), where R²_j is the R² from regressing column j
    on all other columns.  Uses the normal-equations directly to avoid a
    statsmodels dependency.
 
    Parameters
    ----------
    X_arr : ndarray, shape (n_samples, n_features)
        Feature matrix (should be standardised before calling).
 
    Returns
    -------
    vif : ndarray, shape (n_features,)
    """
    n, p = X_arr.shape
    vif  = np.full(p, np.nan)
    for j in range(p):
        y_j   = X_arr[:, j]
        X_oth = np.delete(X_arr, j, axis=1)
        # Add intercept column
        X_oth = np.column_stack([np.ones(n), X_oth])
        try:
            # OLS via pseudo-inverse
            beta  = np.linalg.lstsq(X_oth, y_j, rcond=None)[0]
            y_hat = X_oth @ beta
            ss_res = np.sum((y_j - y_hat) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
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
    Identify and optionally remove collinear features from the feature matrix.
 
    Two complementary diagnostics are run:
 
    1. **Spearman correlation matrix** — catches monotonic linear and nonlinear
       dependencies without assuming normality.  Pairs with |r| ≥ r_threshold
       are flagged.
 
    2. **Variance Inflation Factor (VIF)** — measures how much the variance of
       a regression coefficient is inflated due to collinearity with the other
       features.  VIF > 10 (or > 5 for stricter control) conventionally
       indicates a problematic feature.
 
    When ``action="drop"``, the function resolves collinear pairs greedily:
    within each pair it drops the feature with the higher mean |r| to all
    other features (i.e., the more "redundant" one), keeping the one that
    carries more unique information.
 
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix produced by ``build_feature_matrix``.
    r_threshold : float
        Spearman |r| above which a pair is considered collinear (default 0.85).
    vif_threshold : float or None
        VIF above which a feature is flagged.  Set to None to skip VIF
        (recommended for very wide matrices, >200 features, where VIF is slow).
    action : {"warn", "drop"}
        "warn"  → report findings, return X unchanged.
        "drop"  → remove redundant features and return reduced X.
    save_path : Path or None
        If provided, save the Spearman correlation heatmap here.
 
    Returns
    -------
    X_out : pd.DataFrame
        Feature matrix after (optional) collinear feature removal.
    dropped : list[str]
        Names of features that were dropped (empty if action="warn").
    report : pd.DataFrame
        Summary table with columns:
        feature | mean_abs_r | vif | flagged_by | dropped
    """
    features = list(X.columns)
    n_feat   = len(features)
 
    # ── 1. Spearman correlation matrix ────────────────────────────────────────
    print(f"  Computing Spearman correlation matrix ({n_feat} × {n_feat}) ...")
    spearman_mat = X.rank().corr()                   # rank-transform then Pearson = Spearman
 
    # Identify collinear pairs (upper triangle only, exclude diagonal)
    collinear_pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            r = spearman_mat.iloc[i, j]
            if abs(r) >= r_threshold:
                collinear_pairs.append((features[i], features[j], r))
 
    if collinear_pairs:
        print(f"  ⚠  {len(collinear_pairs)} collinear pair(s) found "
              f"(|r| ≥ {r_threshold}):")
        for a, b, r in collinear_pairs:
            print(f"      {a}  ↔  {b}   r = {r:+.3f}")
    else:
        print(f"  ✓  No collinear pairs found at |r| ≥ {r_threshold}.")
 
    # ── 2. VIF ────────────────────────────────────────────────────────────────
    vif_values = np.full(n_feat, np.nan)
    if vif_threshold is not None:
        print(f"  Computing VIF for {n_feat} features ...")
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X.values.astype(float))
        vif_values = _vif_series(X_scaled)
        high_vif  = [(features[i], vif_values[i])
                     for i in range(n_feat)
                     if vif_values[i] > vif_threshold]
        if high_vif:
            print(f"  ⚠  {len(high_vif)} feature(s) with VIF > {vif_threshold}:")
            for name, v in sorted(high_vif, key=lambda x: -x[1]):
                print(f"      {name}   VIF = {v:.1f}")
        else:
            print(f"  ✓  All VIF values ≤ {vif_threshold}.")
 
    # ── 3. Build report table ─────────────────────────────────────────────────
    mean_abs_r = spearman_mat.abs().mean(axis=1)    # mean |r| to all other features
 
    flagged_by = []
    for i, feat in enumerate(features):
        flags = []
        if any(a == feat or b == feat for a, b, _ in collinear_pairs):
            flags.append("spearman")
        if not np.isnan(vif_values[i]) and vif_values[i] > (vif_threshold or np.inf):
            flags.append("vif")
        flagged_by.append("|".join(flags) if flags else "")
 
    report = pd.DataFrame({
        "feature":     features,
        "mean_abs_r":  mean_abs_r.values,
        "vif":         vif_values,
        "flagged_by":  flagged_by,
    }).set_index("feature")
 
    # ── 4. Resolve pairs: drop the more redundant member ─────────────────────
    dropped = []
    if action == "drop" and collinear_pairs:
        # Greedy: sort pairs by |r| descending so strongest collinearity resolved first
        pairs_sorted = sorted(collinear_pairs, key=lambda x: -abs(x[2]))
        already_dropped = set()
        for a, b, r in pairs_sorted:
            if a in already_dropped or b in already_dropped:
                continue
            # Drop whichever has the higher mean |r| (more redundant globally)
            to_drop = a if mean_abs_r[a] >= mean_abs_r[b] else b
            already_dropped.add(to_drop)
            dropped.append(to_drop)
            print(f"  ✂  Dropping '{to_drop}'  (mean |r| = {mean_abs_r[to_drop]:.3f}, "
                  f"pair r = {r:+.3f}  with "
                  f"'{b if to_drop == a else a}')")
 
        if dropped:
            print(f"  → {len(dropped)} feature(s) dropped; "
                  f"{n_feat - len(dropped)} remain.")
 
    report["dropped"] = report.index.isin(dropped)
 
    # ── 5. Heatmap ────────────────────────────────────────────────────────────
    if save_path is not None:
        _plot_collinearity_heatmap(spearman_mat, r_threshold,
                                   dropped=dropped, save_path=save_path)
 
    X_out = X.drop(columns=dropped) if dropped else X.copy()
    return X_out, dropped, report
 
 
def _plot_collinearity_heatmap(spearman_mat: pd.DataFrame,
                                r_threshold: float,
                                dropped: list[str],
                                save_path: Path):
    """
    Annotated heatmap of the Spearman correlation matrix.
 
    Dropped features are marked with a red border on their row/column labels.
    Cells that exceed r_threshold are outlined with a black rectangle.
    """
    n    = len(spearman_mat)
    size = max(6, n * 0.45)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
 
    mat  = spearman_mat.values
    im   = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
 
    # Axis labels — colour dropped features red
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
 
    # Outline collinear cells
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
        f"Red labels = dropped features  |  boxed cells = |r| ≥ {r_threshold}",
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
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved lag sweep plot → {save_path}")

def mid(x):
    '''
    Midpoint in array
    '''
    return (x[1:]+x[:-1])/2

def run_pipeline(config: dict = CONFIG, df: pd.DataFrame = None):
    """
    Full pipeline. Pass df directly (with datetime index) or set
    config["data_path"] to load from CSV.
    """
    print("=" * 65)
    print("  Atmospheric Variable Importance Pipeline")
    print("=" * 65)

    # ── Initial assessment ──────────────────────────────────────────────────────────────
    if df is None:
        print("\n[1] Loading data ...")
        df = load_data(config)
    else:
        print("\n[1] Using provided DataFrame.")
        
    # Histograms
    plot_histograms(df, config, lag=None, save_path=OUT / "histograms.png")

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
    print(f"    Best lag (auto-selected): \n{best_lag} time steps")
    
    # Histograms
    plot_histograms(df, config, lag=best_lag, save_path=OUT / "histograms_lag.png")

    # ── Lag sweep (RF impurity, fast) ─────────────────────────────────────
    if config['prelim_rf']:
        print("\n[3] Lag sweep (quick RF at each lag) ...")
        sweep_df = lag_sweep_importance(df, predictors, "__target__",
                                        config["lag_list"], config)
        plot_lag_sweep(sweep_df, save_path=OUT / "lag_sweep.png")
    else:
        sweep_df=None

    # ── Build feature matrix at best lag ─────────────────────────────────
    print(f"\n[4] Building feature matrix at lag\n{best_lag} ...")
    X, y = build_feature_matrix(df, predictors, "__target__",
                                 lag=best_lag,
                                 rolling_windows=config["rolling_windows"],
                                 add_rolling=config["add_rolling"])
    print(f"    Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    
    # ── Collinearity check ────────────────────────────────────────────────
    print("\n[5] Checking feature collinearity ...")
    X, dropped_features, collinearity_report = check_collinearity(
        X,
        r_threshold   = config["collinearity_r_threshold"],
        vif_threshold = config["collinearity_vif_threshold"],
        action        = config["collinearity_action"],
        save_path     = OUT / "collinearity_heatmap.png",
    )
    collinearity_report.to_csv(OUT / "collinearity_report.csv")
    print(f"    Saved collinearity report → {OUT / 'collinearity_report.csv'}")
    if dropped_features:
        print(f"    Proceeding with {X.shape[1]} features "
              f"(dropped {len(dropped_features)}: {dropped_features})")
    else:
        print(f"    No features dropped; proceeding with {X.shape[1]} features.")


    # ── RF permutation importance (cross-validated) ───────────────────────
    print("\n[5] Training RF + permutation importance (CV) ...")
    rf_result = train_rf_importance(X, y, config)

    # ── SHAP importance ───────────────────────────────────────────────────
    if config['shap']:
        print("\n[6] Computing SHAP importance ...")
        shap_imp = compute_shap_importance(X, y, config,
                                           save_path=OUT / "shap_summary.png")
    else:
        shap_imp=[]

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
    
    cfg = CONFIG.copy()
    
    if cfg==True:
        print("Running with SYNTHETIC data (replace with your CSV path).\n")
    
        df_qc = generate_synthetic_data(n=100_000)
    else:
        df=load_data(cfg)
    
        df_qc=qc_data(df, cfg)
    
    
    results, corr_df, sweep_df = run_pipeline(config=cfg, df=df_qc)





        



