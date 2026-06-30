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

from utils import apply_climatology, remove_seasonal_cycle


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

    def __init__(self, bundle_dir: str):
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

    def predict(self, df: pd.DataFrame, station: str = None,
                climatology_path: str | Path = None,
                non_overlapping: bool = False) -> pd.Series:
        """
        Return calibrated outage probability for df.

        Parameters
        ----------
        df               : DataFrame with DatetimeIndex; must contain all predictor_cols.
        station          : name of the station whose climatology is used for detrending;
                           defaults to the first station in the bundle.
        climatology_path : optional path to a climatology CSV produced by
                           remove_seasonal_cycle(); takes priority over the bundle
                           climatology.  If neither is available the seasonal cycle
                           is removed directly from df.
        non_overlapping  : if True, tile the series into non-overlapping windows of
                           roll_size steps and return one probability per tile, indexed
                           at the center timestamp.  If False (default), return a
                           probability at every timestep (sliding window).

        Returns
        -------
        pd.Series of float.  When non_overlapping=False: same index as df, NaN at
        boundary timesteps and wherever input data are missing.  When
        non_overlapping=True: one value per tile, indexed at the tile center.
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

        if climatology_path is not None:
            clim = pd.read_csv(climatology_path, index_col=[0, 1])
            clim.index.names = ['__doy__', '__tod__']
        else:
            clim = (self.climatologies.get(station)
                    or next(iter(self.climatologies.values()), None))

        # Detrend with saved climatology [Wilks, 2011]; fall back to on-the-fly detrending
        if clim is not None:
            df_det = apply_climatology(df[predictors].copy(), clim,
                                       columns=detrend_cols, inplace=True)
        else:
            df_det, _ = remove_seasonal_cycle(
                df[predictors].copy(),
                columns=detrend_cols,
                window_days=self.cfg.get('detrend_window_days', 7),
                min_periods=self.cfg.get('detrend_min_periods', 3),
                inplace=True,
            )

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

        if non_overlapping:
            # Anchor at pre_w within each tile so [t-pre_w, t+post_w] = tile [k*roll_size, (k+1)*roll_size-1]
            sel_pos = np.arange(pre_w, len(feat_df) - post_w, roll_size)
            feat_sel = feat_df.iloc[sel_pos].reindex(columns=feat_names)
            valid = feat_sel.notna().all(axis=1).values
            # Center of each tile (works for unequal pre_w / post_w)
            center_pos = (sel_pos - pre_w + roll_size // 2).clip(0, len(df) - 1)
            center_times = df.index[center_pos]
            probs = pd.Series(np.nan, index=center_times, name='outage_probability')
            if valid.any():
                probs.iloc[valid] = self.model.predict_proba(feat_sel.values[valid])[:, 1]
            return probs

        probs = pd.Series(np.nan, index=df_det.index, name='outage_probability')
        valid = feat_df.notna().all(axis=1)
        if valid.any():
            probs[valid] = self.model.predict_proba(feat_df[valid].values)[:, 1]
        return probs


if __name__ == '__main__':
    import sys
    import tkinter
    import xarray as xr
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.size'] = 12

    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.update()

    source_data = tkinter.filedialog.askopenfilename(
        title='Select atmospheric NetCDF',
        initialdir='./',
        filetypes=[('NetCDF files', '*.nc')],
    )
    if not source_data:
        print('No file selected. Exiting.')
        sys.exit()
        
    source_model = tkinter.filedialog.askdirectory(
        title='Select model folder',
        initialdir='./',
    )
    if not source_model:
        print('No file selected. Exiting.')
        sys.exit()

    predictor = OutagePredictor(source_model)
    pred_cols = predictor.cfg['predictor_cols']

    ds = xr.open_dataset(source_data)

    # Compute aavi from wssd and wdsd if required [Bianco et al., 2016]
    if 'aavi' in pred_cols and 'aavi' not in ds:
        ds['aavi'] = ds['wssd'] / ds['wssd'].mean() * ds['wdsd'] / ds['wdsd'].mean()

    df_raw = ds[pred_cols].to_dataframe()
    df_raw.index = pd.DatetimeIndex(ds['time'].values)
    ds.close()

    # Apply physical QC limits from training config
    with open('configs/outage_rf_events.yaml') as f:
        cfg_train = yaml.safe_load(f)
    lims = cfg_train.get('limits', {})
    df_qc = df_raw.copy()
    for col in pred_cols:
        if col in lims:
            lo, hi = lims[col]
            df_qc[col] = df_qc[col].where(df_qc[col] >= lo).where(df_qc[col] <= hi)

    prob = predictor.predict(df_qc, non_overlapping=True)
    print(f"Windows: {len(prob)}  |  valid: {prob.notna().sum()}")

    # Save to NetCDF
    out_nc = Path(source_data).with_suffix('.outage_prob.nc')
    ds_out = xr.Dataset(
        {'outage_probability': ('time', prob.values)},
        coords={'time': prob.index.values},
    )
    ds_out.to_netcdf(out_nc)
    print(f"Saved → {out_nc}")

    # Plot
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(prob.index, prob.values, color='steelblue', lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Outage probability')
    ax.set_title(Path(source_data).name)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()
    out_png = Path(source_data).with_suffix('.outage_prob.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_png}")
