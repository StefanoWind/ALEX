import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from pathlib import Path

from utils import make_segment_target, cross_lag_correlation, select_best_lag


def normalize(s: pd.Series, p0: float, p100: float) -> pd.Series:
    denom = p100 - p0
    if denom == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - p0) / denom


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300


def load_config(path: str = "configs/outage_episodes.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    ds = xr.open_dataset(config["source"])
    cols = config['predictor_cols'] + [config['target_col'], 'weather_event']
    return ds[cols].to_dataframe()


def qc_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df_qc = pd.DataFrame(index=df.index)
    for v in config['predictor_cols']:
        df_qc[v] = (df[v]
                    .where(df[v] >= config['limits'][v][0])
                    .where(df[v] <= config['limits'][v][1]))
    df_qc[config['target_col']] = df[config['target_col']]
    df_qc['weather_event'] = df['weather_event']
    return df_qc


def find_episodes(series: pd.Series, threshold: float) -> list[tuple]:
    flag = series > threshold
    block_id = (flag != flag.shift()).cumsum()[flag]
    return [(g.index[0], g.index[-1]) for _, g in flag.groupby(block_id)]


def _shade_boolean(ax, series: pd.Series, color: str, alpha: float, label: str):
    blocks = (series != series.shift()).cumsum()[series]
    first = True
    for _, grp in series.groupby(blocks):
        ax.axvspan(grp.index[0], grp.index[-1],
                   alpha=alpha, color=color,
                   label=label if first else None, linewidth=0)
        first = False


def plot_episode(df: pd.DataFrame, t_start, t_end, auc, tot, config: dict,
                 out_dir: Path, dyn: dict = None, pct: dict = None, best_lag: dict = None):
    buffer = pd.Timedelta(hours=config['buffer_hours'])
    t0 = t_start - buffer
    t1 = t_end + buffer
    win = df.loc[t0:t1]

    predictors = config['predictor_cols']
    target = config['target_col']
    w = config.get('dynamic_window', 12)
    n = len(predictors) + 1

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)

    for i, col in enumerate(predictors):
        ax = axes[i]

        if pct is not None and col in pct:
            p0r, p100r = pct[col]['raw']
            raw_line = normalize(win[col], p0r, p100r)
        else:
            raw_line = win[col]
        ax.plot(win.index, raw_line, color='k', linewidth=0.8, label=col)

        if dyn is not None and col in dyn:
            lag = best_lag[col] if best_lag is not None else 0
            mean_series = dyn[col].shift(lag).loc[win.index[0]:win.index[-1]]
            p0m, p100m = pct[col]['mean']
            feat_label = f"{col}_mean_W{w} (lag={lag})"
            ax.plot(win.index,
                    normalize(mean_series, p0m, p100m),
                    color='steelblue', linewidth=0.7, alpha=0.8, label=feat_label)

        ax.set_ylabel(f"{col}\n(normalised)")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        if win['weather_event'].any():
            _shade_boolean(ax, win['weather_event'], color='gold', alpha=0.35,
                           label='weather event')
        ax.axvspan(t_start, t_end, alpha=0.15, color='red',
                   label='outage period', linewidth=0)

        if i == 0:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.7)

    ax_target = axes[-1]
    ax_target.plot(win.index, win[target], color='firebrick', linewidth=0.8)
    ax_target.axhline(config['outage_threshold'], color='firebrick',
                      linestyle='--', linewidth=0.8, alpha=0.6)
    ax_target.set_ylabel(target)
    ax_target.grid(True, alpha=0.3)

    if win['weather_event'].any():
        _shade_boolean(ax_target, win['weather_event'], color='steelblue', alpha=0.25,
                       label=None)
    ax_target.axvspan(t_start, t_end, alpha=0.15, color='red', linewidth=0)

    ax_target.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    fig.autofmt_xdate(rotation=0, ha='center')

    axes[0].set_title(f"AUC = {auc:.1f} customer·h   |   TOT = {tot:.1f} min")
    fig.tight_layout()

    fname = out_dir / f"outage_{pd.Timestamp(t_start).strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(fname)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    df_qc = qc_data(df, cfg)

    w = cfg.get('dynamic_window', 12)

    dyn = {}
    pct = {}
    for col in cfg['predictor_cols']:
        mean_series = df_qc[col].rolling(w, center=True).mean()  # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
        dyn[col] = mean_series
        pct[col] = {
            'raw': (
                float(np.nanpercentile(df_qc[col], 0)),
                float(np.nanpercentile(df_qc[col], 100)),
            ),
            'mean': (
                float(np.nanpercentile(mean_series, 0)),
                float(np.nanpercentile(mean_series, 100)),
            ),
        }
        print(f"{col} completed.")

    X_mean = pd.DataFrame({col: dyn[col] for col in cfg['predictor_cols']})
    target_auc = make_segment_target(df_qc[cfg['target_col']], cfg['outage_threshold'], 'AUC')
    corr_df = cross_lag_correlation(X_mean, cfg['predictor_cols'], target_auc, cfg['lag_list'])
    best_lag = select_best_lag(corr_df, cfg)
    print(f"Best lag per predictor: {best_lag}")

    episodes = find_episodes(df_qc[cfg['target_col']], cfg['outage_threshold'])
    print(f"Found {len(episodes)} outage episode(s)")

    auc_series = make_segment_target(df_qc[cfg['target_col']], cfg['outage_threshold'], 'AUC')
    tot_series = make_segment_target(df_qc[cfg['target_col']], cfg['outage_threshold'], 'TOT')

    metrics = pd.DataFrame({
        't_start': [t0 for t0, _ in episodes],
        't_end': [t1 for _, t1 in episodes],
        'AUC': [auc_series.loc[t0] for t0, _ in episodes],
        'TOT': [tot_series.loc[t0] for t0, _ in episodes],
    })

    n_plots = cfg['n_plots']
    top_auc = set(metrics['AUC'].nlargest(n_plots).index)
    top_tot = set(metrics['TOT'].nlargest(n_plots).index)
    selected = sorted(top_auc | top_tot)
    print(f"Plotting {len(selected)} episode(s) (top {n_plots} by AUC or TOT)")

    out_dir = Path(cfg['output_dir'])
    os.makedirs(out_dir, exist_ok=True)

    for i in selected:
        row = metrics.iloc[i]
        plot_episode(df_qc, row['t_start'], row['t_end'], row['AUC'], row['TOT'],
                     cfg, out_dir, dyn=dyn, pct=pct, best_lag=best_lag)
