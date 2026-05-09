# Series approach: Random Forest feature importance on 15-min timestep data.
# Each row of the feature matrix is one 15-minute timestep; the target (AUC or
# TOT) is broadcast to all timesteps within each outage episode.
#
# NOTE: this design creates within-episode temporal autocorrelation. Use
# event-based approach (outage_rf_events.py) to avoid this.
# [Ploton et al., 2020, Nat. Commun. — autocorrelation inflation in ML]
# [Boulesteix et al., 2012, WIREs Data Mining — RF with small samples]

import copy
import os
import pandas as pd
import xarray as xr
import yaml
from datetime import datetime
from pathlib import Path
from utils import run_pipeline, remove_seasonal_cycle, make_segment_target, plot_segment_zoom


def load_config(path: str = "configs/outage_rf_series.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    ds = xr.open_dataset(config["source"])
    cols = config['predictor_cols'] + [config['target_col'], 'weather_event_buffer']
    return ds[cols].to_dataframe()


def qc_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df_qc = pd.DataFrame(index=df.index)
    for v in config['predictor_cols']:
        df_qc[v] = (df[v]
                    .where(df[v] >= config['limits'][v][0])
                    .where(df[v] <= config['limits'][v][1]))
    df_qc[config['target_col']] = df[config['target_col']]
    df_qc['weather_event_buffer'] = df['weather_event_buffer']
    return df_qc


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    df_qc = qc_data(df, cfg)
    df_raw = df_qc.copy()

    ts = datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
    base = Path(cfg['output_dir']) / ts
    os.makedirs(base, exist_ok=True)

    if cfg.get('detrend_seasonal', False):
        inplace = cfg.get('detrend_mode', 'anomaly') == 'inplace'
        df_qc, _ = remove_seasonal_cycle(
            df_qc,
            columns=cfg['predictor_cols'],
            window_days=cfg.get('detrend_window_days', 7),
            min_periods=cfg.get('detrend_min_periods', 3),
            inplace=inplace,
            save_climatology_path=base / 'climatology.csv',
        )

    if cfg.get('mode') in ('AUC', 'TOT'):
        df_qc['__target__'] = make_segment_target(
            df_qc[cfg['target_col']], cfg['outage_threshold'], cfg['mode']
        )
        plot_segment_zoom(df_qc[cfg['target_col']], df_qc['__target__'],
                          cfg['outage_threshold'], cfg['mode'],
                          save_path=base / 'segment_zoom.png')

    cfg_run = copy.deepcopy(cfg)
    cfg_run['detrend_seasonal'] = False

    for flag, label in [(True, 'NWS_true'), (None, 'all')]:
        if flag is None:
            subset = df_qc.drop(columns=['weather_event_buffer'])
            raw_subset = df_raw.drop(columns=['weather_event_buffer'])
        else:
            subset = (df_qc[df_qc['weather_event_buffer'] == flag]
                      .drop(columns=['weather_event_buffer']))
            raw_subset = (df_raw[df_raw['weather_event_buffer'] == flag]
                          .drop(columns=['weather_event_buffer']))
        results = run_pipeline(config=cfg_run, df=subset, out_dir=base / label,
                               df_raw=raw_subset)
