# -*- coding: utf-8 -*-
"""
Compare outage predictions
"""

import re
import tkinter
import tkinter.filedialog
from pathlib import Path
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source_train = tkinter.filedialog.askopenfilename(
    title='Select event.nc',
    initialdir='./results/',
    filetypes=[('NetCDF files', '*.nc')])

source_libraries = tkinter.filedialog.askopenfilenames(
    title='Select library.xlsx file(s) (one per met station)',
    initialdir='./data/',
    filetypes=[('Library Excel files', '*.library.xlsx'), ('Excel files', '*.xlsx')])


def _station_label(xlsx_path):
    """Derive a short station label from a library.xlsx filename."""
    stem = Path(xlsx_path).stem  # strips .xlsx
    m = re.search(r'(met[\w\-]+?only)', stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return re.sub(r'\.input\.library$', '', stem, flags=re.IGNORECASE)


def _load_outage_probability(xlsx_path):
    """Read the outage-probability time series from a build_library.py library.xlsx."""
    df = pd.read_excel(xlsx_path, sheet_name='Library', index_col=0)
    # 'latitude'/'longitude' are dropped for compatibility with library.xlsx files
    # generated before geolocation moved to its own 'Site' sheet.
    meta_rows = [k for k in ['shap_base_value', 'outage_threshold', 'latitude', 'longitude']
                 if k in df.index]
    if meta_rows:
        df = df.drop(meta_rows)
    df.index = pd.to_datetime(df.index)
    return df['outage_probability'].astype(float)


#%% Initialization
ds_train = xr.open_dataset(source_train)

inferences = {_station_label(p): _load_outage_probability(p) for p in source_libraries}

#%% Plots
plt.figure(figsize=(18, 10))

for label, series in inferences.items():
    plt.plot(series.index, series.values,'.-b', label=f'Inference ({label})')

plt.plot(ds_train.event, ds_train.rf_prediction, '.k', markersize=20, label='Training')
plt.plot(ds_train.event[ds_train.is_outage == 1], ds_train.is_outage[ds_train.is_outage == 1],
          'xg', markersize=20, label='Real outage')
plt.grid()
plt.xlabel('Time (UTC)')
plt.legend()
plt.ylabel('Outage probability')
