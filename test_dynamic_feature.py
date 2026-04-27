# -*- coding: utf-8 -*-
"""
Dynamic feature test
"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import yaml
from matplotlib import pyplot as plt
import matplotlib
plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300

def load_config(path: str = "configs/test_dynamic_features.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
    
def ar1(phi: float=0.8, n: int=1000, mean: float=0, sigma: float=1, random_state: int=42):
    f=np.zeros(n)
    rng = np.random.default_rng(random_state)
    for i in range(1,n):
        f[i]=f[i-1]*phi+rng.normal(0,sigma)
    
    return f+mean

def build_dynamic_features(config: dict={}, n: int=1000, random_state: int=42):
    
    dates = pd.date_range("2020-01-01", periods=n, freq="15min")
    
    wind_speed=ar1(config['phi'],n,config['mean']['wind_speed'],config['sigma']['wind_speed'])
    temperature=ar1(config['phi'],n,config['mean']['temperature'],config['sigma']['temperature'])
    pressure=ar1(config['phi'],n,config['mean']['pressure'],config['sigma']['pressure'])
    baseline=ar1(config['phi'],n,0,1)
    
    t=np.arange(n)
    peak=np.exp(-(t-n/2)**2/(2*config['peak_amplitude']))*config['peak_height']
    wind_speed+=peak
    
    ramp= config['drop'] / (1 + ((t) / (n/2))**8)
    temperature+=ramp
    
    wave=config['wave_amplitude']*np.sin(t*2*np.pi/config['wave_period'])
    pressure+=wave
    
    return pd.DataFrame({
        "wind_speed": wind_speed,
        "temperature": temperature,
        "pressure": pressure,
        "baseline": baseline,
    }, index=dates)
   

def compute_full_period_metrics(x: np.ndarray) -> dict:
    mu = np.mean(x)
    sd = np.std(x)

    if sd == 0:
        return {k: np.nan for k in ("gust_factor", "mann_kendall_tau",
                                    "dominant_freq_power_ratio", "acf_oscillation_score", "residual_rms")}

    # [Durst, 1960; IEC 61400-1, 2019]
    gust_factor = (np.max(x) - mu) / sd

    # [Mann, 1945; Kendall, 1975]
    tau = scipy.stats.kendalltau(np.arange(len(x)), x)[0]
    mann_kendall_tau = np.abs(tau)

    # [Welch, 1967; Wang et al., 2006]
    xd = scipy.signal.detrend(x, type='linear')
    xw = xd * np.hanning(len(xd))
    fft_w = np.fft.rfft(xw)
    power_w = np.abs(fft_w[1:]) ** 2
    top2 = np.sort(power_w)[-2:].sum()
    dominant_freq_power_ratio = top2 / power_w.sum()

    # [Box, Jenkins & Reinsel, 2008]
    acf = np.correlate(x - mu, x - mu, mode='full')
    acf = acf[len(x) - 1:] / acf[len(x) - 1]
    lags = acf[1:len(x) // 2 + 1]
    acf_oscillation_score = -np.min(lags)

    # residual standard deviation after linear detrend
    residuals = scipy.signal.detrend(x, type='linear')
    residual_rms = np.std(residuals) / sd

    return {
        "gust_factor": gust_factor,
        "mann_kendall_tau": mann_kendall_tau,
        "dominant_freq_power_ratio": dominant_freq_power_ratio,
        "acf_oscillation_score": acf_oscillation_score,
        "residual_rms": residual_rms,
    }


def plot_metric_comparison(df: pd.DataFrame, cfg: dict, save_path=None):
    colors = {"wind_speed": "steelblue", "temperature": "darkorange", "pressure": "forestgreen", "baseline": "gray"}
    signals = list(df.columns)

    metrics_raw = {col: compute_full_period_metrics(df[col].values) for col in signals}
    metric_names = list(next(iter(metrics_raw.values())).keys())

    # normalize each metric so max across signals = 1
    metric_matrix = np.array([[metrics_raw[col][m] for col in signals] for m in metric_names])
    col_max = np.nanmax(np.abs(metric_matrix), axis=1, keepdims=True)
    col_max[col_max == 0] = 1
    metric_norm = metric_matrix / col_max

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3)

    for j, col in enumerate(signals):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(df[col].values, color=colors[col])
        ax.set_title(col)
        ax.grid(True)

    ax_bar = fig.add_subplot(gs[1, :])
    n_metrics = len(metric_names)
    n_signals = len(signals)
    x = np.arange(n_metrics)
    width = 0.18
    offsets = np.linspace(-(n_signals - 1) / 2, (n_signals - 1) / 2, n_signals) * width

    for i, col in enumerate(signals):
        ax_bar.bar(x + offsets[i], metric_norm[:, i], width=width, color=colors[col], label=col)

    ax_bar.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax_bar.grid(True, axis='y')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylabel("Normalized metric value")
    ax_bar.legend(draggable=True)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)


if __name__ == "__main__":
    cfg = load_config()
    df = build_dynamic_features(cfg, n=100)

    metrics = {col: compute_full_period_metrics(df[col].values) for col in df.columns}
    metrics_df = pd.DataFrame(metrics)   # rows = metrics, columns = signals

    plot_metric_comparison(df, cfg, save_path="data/dynamic_feature_comparison.png")
    print(metrics_df.to_string())
