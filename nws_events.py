import pandas as pd
import xarray as xr
import yaml
from datetime import datetime
from pathlib import Path
from utils import run_pipeline


def load_config(path: str = "configs/nws_events.yaml") -> dict:
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

    ts = datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
    base = Path(cfg['output_dir']) / ts

    for flag, label in [(True, 'NWS_true'), (False, 'NWS_false')]:
        subset = (df_qc[df_qc['weather_event_buffer'] == flag]
                  .drop(columns=['weather_event_buffer']))
        run_pipeline(config=cfg, df=subset, out_dir=base / label)
