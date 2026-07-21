# -*- coding: utf-8 -*-
"""
Build input for RF outage predictor
"""

from pathlib import Path
import tkinter
import tkinter.filedialog
import yaml
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source_data_files = tkinter.filedialog.askopenfilenames(
    title='Select met data',
    initialdir='./data/',
    filetypes=[('NetCDF files', '*.nc')])

source_config = tkinter.filedialog.askopenfilename(
    title='Select configuration file',
    initialdir='./',
    filetypes=[('YAML files', '*.yaml')])


MAPPING={'wspd':'wind_speed',
         'wdir':'wind_direction',
         'tair':'temperature',
         'relh':'relative_humidity',
         'pres':'pressure',
         'srad':'shortwave_radiation',
         'wssd':'wind_speed',
         'wdsd':'wind_direction'}

STATS= {'wspd':'mean',
        'wdir':'mean',
        'tair':'mean',
        'relh':'mean',
        'pres':'mean',
        'srad':'mean',
        'wssd':'std',
        'wdsd':'std'}

CONVERSION={'wspd':1,
            'wdir':1,
            'tair':1,
            'relh':1,
            'pres':10,
            'srad':1, #1000/0.2,
            'wssd':1,
            'wdsd':1}

#%% Initialization
with open(source_config) as f:
    config=yaml.safe_load(f)

limits = config.get('limits', {})
max_flat_points = config['max_flat_points']


def _read_attr(ds, keys):
    for key in keys:
        if key in ds.attrs:
            try:
                return float(ds.attrs[key])
            except (TypeError, ValueError):
                continue
    return np.nan


def _apply_limits(da, lims):
    lo, hi = lims
    return da.where(da >= lo).where(da <= hi)


def _flag_flat_mask(values, max_flat):
    """True where a sample belongs to a run of more than `max_flat` consecutive
    identical values (a stuck/flat sensor segment)."""
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=bool)
    valid = ~np.isnan(values)
    same_as_prev = np.zeros(n, dtype=bool)
    same_as_prev[1:] = valid[1:] & valid[:-1] & (values[1:] == values[:-1])
    run_id = np.cumsum(~same_as_prev)
    run_len = np.bincount(run_id)[run_id]
    return valid & (run_len > max_flat)


def _qc_signal(da, lims, max_flat):
    """Apply physical-limit thresholds, then null out stuck/flat runs."""
    qc = _apply_limits(da, lims) if lims is not None else da
    values = qc.values.astype(float)
    values[_flag_flat_mask(values, max_flat)] = np.nan
    return qc.copy(data=values)


def _plot_qc(raw_signals, qc_signals, out_path):
    variables = list(raw_signals.keys())
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 2.2 * len(variables)), sharex=True)
    if len(variables) == 1:
        axes = [axes]
    for ax, v in zip(axes, variables):
        raw_da = raw_signals[v]
        qc_da  = qc_signals[v]
        t = raw_da[raw_da.dims[0]].values
        ax.plot(t, raw_da.values, color='red', linewidth=1, label='before QC')
        ax.plot(t, qc_da.values, color='green', linewidth=1, label='after QC')
        ax.set_ylabel(v)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('time')
    fig.suptitle(out_path.stem)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

predictors=config['predictor_cols']
if 'aavi' in predictors:
    predictors+=['wssd','wdsd']

#%% Main
for source_data in source_data_files:
    ds_in=xr.open_dataset(source_data)
    ds_out_qc  = xr.Dataset()
    ds_out_raw = xr.Dataset()

    raw_signals = {}
    qc_signals  = {}
    for v in predictors:
        if v in MAPPING.keys():
            if MAPPING[v] in ds_in.data_vars:
                raw_da = ds_in[MAPPING[v]].sel(stat=STATS[v])*CONVERSION[v]
                qc_da  = _qc_signal(raw_da, limits.get(v), max_flat_points)
                raw_signals[v] = raw_da
                qc_signals[v]  = qc_da
                ds_out_qc[v]  = qc_da
                ds_out_raw[v] = raw_da
            else:
                print(f'Missing {v} in input')

    ds_out_qc['customers_out']  = ds_in.outages
    ds_out_raw['customers_out'] = ds_in.outages

    # Preserve station geolocation metadata so downstream products can export it.
    lat_val = _read_attr(ds_in, ['latitude', 'lat', 'station_lat'])
    lon_val = _read_attr(ds_in, ['longitude', 'lon', 'station_lon'])

    for ds_out in (ds_out_qc, ds_out_raw):
        if np.isfinite(lat_val):
            ds_out.attrs['latitude'] = float(lat_val)
        if np.isfinite(lon_val):
            ds_out.attrs['longitude'] = float(lon_val)

        for key in ['station', 'county', 'state', 'source', 'description']:
            if key in ds_in.attrs:
                ds_out.attrs[key] = ds_in.attrs[key]

    ds_in.close()

    #%% QC plot
    if raw_signals:
        _plot_qc(raw_signals, qc_signals, Path(source_data.replace('.nc', '.qc.png')))

    #%% Output — QC'd and raw versions saved separately for dashboards with/without QC
    ds_out_qc.drop('stat').to_netcdf(source_data.replace('.nc', '.input.nc'))
