# Self-contained inference module. Loads the bundle produced by save_model.py
# and predicts outage probability via a sliding event window over any
# atmospheric time series.
#
# Usage:
#   from predict import OutagePredictor
#   predictor = OutagePredictor('model_bundle')
#   prob = predictor.predict(df)               # df: DataFrame with DatetimeIndex
#   prob = predictor.predict(df, station='redr')  # use redr climatology

import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from utils import apply_climatology


class OutagePredictor:
    """
    Loads a saved model bundle and returns calibrated outage probabilities for
    any input atmospheric time series via a sliding event window.

    The event window [t - pre_window, t + post_window] is slid over every
    timestep t. Features are window-level summary statistics (mean, std,
    mean gradient, std gradient) computed with vectorised rolling operations,
    matching the aggregation used during training.

    Note: gradient features include one extra boundary observation compared to
    the per-window computation during training (trivial effect on predictions).
    """

    def __init__(self, bundle_dir: str | Path = 'model_bundle'):
        bundle_dir = Path(bundle_dir)
        with open(bundle_dir / 'config.yaml') as f:
            self.cfg = yaml.safe_load(f)
        self.model = joblib.load(bundle_dir / 'model.joblib')
        self.climatologies = {}
        for station in self.cfg['station_names']:
            p = bundle_dir / f'climatology_{station}.csv'
            if p.exists():
                clim = pd.read_csv(p, index_col=[0, 1])
                clim.index.names = ['__doy__', '__tod__']
                self.climatologies[station] = clim

    def predict(self, df: pd.DataFrame, station: str = None) -> pd.Series:
        """
        Return calibrated outage probability at every timestep in df.

        Parameters
        ----------
        df      : DataFrame with DatetimeIndex; must contain all predictor_cols.
        station : name of the station whose climatology is used for detrending;
                  defaults to the first station in the bundle.

        Returns
        -------
        pd.Series of float, same index as df. NaN at the first pre_window and
        last post_window timesteps and wherever input data are missing.
        """
        predictors   = self.cfg['predictor_cols']
        detrend_cols = self.cfg['detrend_cols']
        pre_w        = self.cfg['pre_window']
        post_w       = self.cfg['post_window']
        feat_names   = self.cfg['feature_names']
        w_label      = pre_w + post_w   # matches the W<n> suffix in feature names
        roll_size    = pre_w + post_w + 1

        missing = [c for c in predictors if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing columns: {missing}")

        clim = (self.climatologies.get(station)
                or next(iter(self.climatologies.values()), None))
        if clim is None:
            raise ValueError("No climatology found in bundle directory.")

        # Detrend the full time series once with the saved climatology [Wilks, 2011]
        df_det = apply_climatology(df[predictors].copy(), clim,
                                   columns=detrend_cols, inplace=True)

        # Vectorised rolling feature matrix.
        # rolling(roll_size).shift(-post_w) aligns each value so that at index t
        # the window covers [t - pre_w, t + post_w], matching the training windows.
        # [Bossavy et al., 2013; Bianco et al., 2016; Vickers & Mahrt, 1997]
        frames = {}
        for col in predictors:
            x = df_det[col]
            g = x.diff()
            r = x.rolling(roll_size)
            gr = g.rolling(roll_size)
            frames[f'{col}_mean_W{w_label}']     = r.mean().shift(-post_w)
            frames[f'{col}_std_W{w_label}']      = r.std().shift(-post_w)
            frames[f'{col}_grad_mean_W{w_label}'] = gr.mean().shift(-post_w)
            frames[f'{col}_grad_std_W{w_label}']  = gr.std().shift(-post_w)

        feat_df = pd.DataFrame(frames, index=df_det.index).reindex(columns=feat_names)

        probs = pd.Series(np.nan, index=df_det.index, name='outage_probability')
        valid = feat_df.notna().all(axis=1)
        if valid.any():
            probs[valid] = self.model.predict_proba(feat_df[valid].values)[:, 1]
        return probs


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import yaml

    from outage_rf_events import load_data, qc_data

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.size'] = 12

    with open('configs/outage_rf_events.yaml') as f:
        cfg = yaml.safe_load(f)

    raw_sources = cfg.get('sources', cfg.get('source'))
    sources = [raw_sources] if isinstance(raw_sources, str) else list(raw_sources)

    predictor = OutagePredictor('model_bundle')

    fig, axes = plt.subplots(len(sources), 2, figsize=(18, 4 * len(sources)),
                             sharex='row')
    if len(sources) == 1:
        axes = axes[np.newaxis, :]

    for row, src in enumerate(sources):
        df_raw = load_data(cfg, source=src)
        df_qc  = qc_data(df_raw, cfg)
        station = Path(src).name.split('.')[0]

        prob = predictor.predict(df_qc, station=station)

        ax_out, ax_prob = axes[row]

        ax_out.plot(df_qc.index, df_qc[cfg['target_col']], color='firebrick', lw=0.6)
        ax_out.axhline(cfg['outage_threshold'], color='firebrick',
                       ls='--', lw=1, alpha=0.5)
        ax_out.set_ylabel('Customers out')
        ax_out.set_title(station)
        ax_out.grid(alpha=0.3)

        ax_prob.plot(prob.index, prob.values, color='steelblue', lw=0.6)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel('Outage probability')
        ax_prob.set_title(f'{station} — model output')
        ax_prob.grid(alpha=0.3)

        for ax in (ax_out, ax_prob):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()
    out_path = Path('model_bundle') / 'predict_verification.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")
