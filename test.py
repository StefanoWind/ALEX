"""
Smoke test using synthetic data. Loads config from configs/test.yaml.
Synthetic outage probability is driven by lagged wind speed and temperature,
so those variables should rank highest in the importance results.
"""

import numpy as np
import pandas as pd
import yaml
from utils import run_pipeline


def load_config(path: str = "configs/test.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_synthetic_data(n: int = 100_000, random_state: int = 42) -> pd.DataFrame:
    """
    Synthetic atmospheric dataset where outage probability is driven by
    wind speed (lag=2) and temperature (lag=6).
    """
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2020-01-01", periods=n, freq="1min")

    wind_speed = np.abs(rng.normal(8, 4, n))
    temperature = rng.normal(15, 8, n)
    pressure = rng.normal(95, 5, n)
    relative_humidity = rng.uniform(20, 100, n)
    shortwave_radiation = rng.normal(0.1, 0.05, n)

    lag_w = np.roll(wind_speed, 2)
    lag_t = np.roll(temperature, 6)
    logit = -6 + 0.5 * lag_w - 0.5 * lag_t
    prob = 1 / (1 + np.exp(-logit))
    outages = (rng.random(n) < prob).astype(int) * rng.integers(1, 500, n)

    return pd.DataFrame({
        "wind_speed": wind_speed,
        "temperature": temperature,
        "pressure": pressure,
        "relative_humidity": relative_humidity,
        "shortwave_radiation": shortwave_radiation,
        "outages": outages,
    }, index=dates)


if __name__ == "__main__":
    cfg = load_config()
    df = generate_synthetic_data(n=100_000)
    results, corr_df, sweep_df = run_pipeline(config=cfg, df=df)
