"""
Phase 1 analysis: correlate power outages with atmospheric signals (AWAKEN site A1).
Loads config from configs/power_outages.yaml.
"""

import pandas as pd
import xarray as xr
import yaml
from pathlib import Path
from utils import run_pipeline


def load_config(path: str = "configs/power_outages.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    ds = xr.open_dataset(config["source"])
    df = ds[config['predictor_cols'] + [config['target_col']]].to_dataframe()
    return df


def qc_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply physical limits; values outside bounds become NaN."""
    df_qc = pd.DataFrame(index=df.index)
    for v in config['predictor_cols']:
        df_qc[v] = (df[v]
                    .where(df[v] >= config['limits'][v][0])
                    .where(df[v] <= config['limits'][v][1]))
    df_qc[config['target_col']] = df[config['target_col']]
    return df_qc


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    df_qc = qc_data(df, cfg)
    results, corr_df, sweep_df = run_pipeline(config=cfg, df=df_qc)
