'''
Self-contained inference module. Loads the bundle produced by save_model.py
and predicts outage probability via a sliding event window over any
atmospheric time series.

'''

import pandas as pd
import yaml
from pathlib import Path
import sys
import tkinter
import tkinter.filedialog
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import OutagePredictor

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source_data = tkinter.filedialog.askopenfilename(
    title='Select atmospheric data NetCDF',
    initialdir='./data',
    filetypes=[('NetCDF files', '*.nc')],
)
if not source_data:
    print('No atmospheric data file selected. Exiting.')
    sys.exit()
    
source_model = tkinter.filedialog.askdirectory(
    title='Select model folder',
    initialdir='./',
)
if not source_model:
    print('No model folder selected. Exiting.')
    sys.exit()
    
source_clim = tkinter.filedialog.askopenfilename(
    title='Select climatology csv',
    initialdir='./data',
    filetypes=[('csv files', '*.csv')],
)
if not source_clim:
    print('No climatology file selected. Exiting.')
    sys.exit()

#%% Initialization
predictor = OutagePredictor(source_model)
pred_cols = predictor.cfg['predictor_cols']

ds = xr.open_dataset(source_data)

# Compute aavi from wssd and wdsd if required [Bianco et al., 2016]
if 'aavi' in pred_cols and 'aavi' not in ds:
    ds['aavi'] = ds['wssd'] / ds['wssd'].mean() * ds['wdsd'] / ds['wdsd'].mean()

df_raw = ds[pred_cols].to_dataframe()
df_raw.index = pd.DatetimeIndex(ds['time'].values)
ds.close()
        
#%% Main

# Apply physical QC limits from training config
with open('configs/outage_rf_events.yaml') as f:
    cfg_train = yaml.safe_load(f)
lims = cfg_train.get('limits', {})
df_qc = df_raw.copy()
for col in pred_cols:
    if col in lims:
        lo, hi = lims[col]
        df_qc[col] = df_qc[col].where(df_qc[col] >= lo).where(df_qc[col] <= hi)

prob = predictor.predict(df_qc, climatology_path=source_clim, non_overlapping=True)
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
