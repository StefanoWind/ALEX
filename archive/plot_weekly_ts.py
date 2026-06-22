import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from pathlib import Path

from utils import remove_seasonal_cycle

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300


def load_config(path: str = "configs/outage_episodes.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    ds = xr.open_dataset(config["source"])
    if config.get('weather_event_flag', False):
        cols = config['predictor_cols'] + [config['target_col'], 'weather_event_buffer']
    else: 
        cols = config['predictor_cols'] + [config['target_col']]
    
    if 'TURB' in cols:
        ds['TURB']=ds['WSSD']/(10**-10+ds['WSPD'])*100
    df=ds[cols].to_dataframe()
    df.index=ds.time.values
    return df

def qc_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df_qc = pd.DataFrame(index=df.index)
    for v in config['predictor_cols']:
        df_qc[v] = (df[v]
                    .where(df[v] >= config['limits'][v][0])
                    .where(df[v] <= config['limits'][v][1]))
    df_qc[config['target_col']] = df[config['target_col']]
    if 'weather_event_buffer' in df.columns:
        df_qc['weather_event_buffer'] = df['weather_event_buffer']
    return df_qc

def _shade_boolean(ax, series: pd.Series, color: str, alpha: float, label: str):
    blocks = (series != series.shift()).cumsum()[series]
    first = True
    for _, grp in series.groupby(blocks):
        ax.axvspan(grp.index[0], grp.index[-1],
                   alpha=alpha, color=color,
                   label=label if first else None, linewidth=0)
        first = False


def plot_week(win: pd.DataFrame, win_anom: pd.DataFrame,
             week_start: pd.Timestamp, config: dict, out_dir: Path):
    predictors = config['predictor_cols']
    target = config['target_col']
    threshold = config['outage_threshold']
    n = len(predictors) + 1

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)

    outage_flag = win[target] > threshold
    has_outage = outage_flag.any()
    if config.get('weather_event_flag', False):
        has_weather = win['weather_event'].any()
    else:
        has_weather=False

    for i, col in enumerate(predictors):
        ax = axes[i]
        ax.plot(win.index, win[col], color='k', linewidth=0.8, label='raw')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

        anom_col = f"{col}_anom"
        if anom_col in win_anom.columns:
            ax2 = ax.twinx()
            ax2.plot(win_anom.index, win_anom[anom_col],
                     color='steelblue', linewidth=0.8, alpha=0.85, label='deseasonalized')
            ax2.set_ylabel(anom_col, fontsize=7, color='steelblue')
            ax2.tick_params(axis='y', labelcolor='steelblue', labelsize=7)
            ax2.spines['right'].set_edgecolor('steelblue')

        if has_weather:
            _shade_boolean(ax, win['weather_event'], color='gold', alpha=0.35,
                           label='weather event' if i == 0 else None)
        if has_outage:
            _shade_boolean(ax, outage_flag, color='red', alpha=0.12,
                           label='outage' if i == 0 else None)
        if i == 0:
            ax.legend(loc='upper left', fontsize=7, framealpha=0.7)

    ax_t = axes[-1]
    ax_t.plot(win.index, win[target], color='firebrick', linewidth=0.8, label=target)
    ax_t.axhline(threshold, color='firebrick', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_t.set_ylabel(target)
    ax_t.grid(True, alpha=0.3)

    if has_weather:
        _shade_boolean(ax_t, win['weather_event'], color='gold', alpha=0.35, label=None)
    if has_outage:
        _shade_boolean(ax_t, outage_flag, color='red', alpha=0.12, label=None)

    ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    fig.autofmt_xdate(rotation=0, ha='center')

    axes[0].set_title(week_start.strftime('Week of %Y-%m-%d'))
    fig.tight_layout()

    fname = out_dir / f"week_{week_start.strftime('%Y%m%d')}.png"
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    df_qc = qc_data(df, cfg)

    print("Computing seasonal anomalies ...")
    df_anom, _ = remove_seasonal_cycle(
        df_qc, cfg['predictor_cols'],
        window_days=cfg.get('detrend_window_days', 7),
        min_periods=cfg.get('detrend_min_periods', 3),
    )

    out_dir = Path(cfg['output_dir'])
    os.makedirs(out_dir, exist_ok=True)

    t_start = df_qc.index[0].normalize()
    t_end = df_qc.index[-1]

    week_starts = pd.date_range(start=t_start, end=t_end, freq='W-MON', inclusive='left')
    if week_starts.empty or week_starts[0] > t_start:
        week_starts = pd.DatetimeIndex([t_start]).append(week_starts)

    for ws in week_starts:
        we = ws + pd.Timedelta(weeks=1)
        win = df_qc.loc[ws:we]
        if win.empty or win[cfg['predictor_cols']].isna().all().all():
            continue
        win_anom = df_anom.loc[ws:we]
        plot_week(win, win_anom, ws, cfg, out_dir)
