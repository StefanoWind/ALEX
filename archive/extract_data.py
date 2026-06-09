import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% Inputs
county= "Garfield"
EAGLEI_DIR = 'data/eaglei'
MESONET_DIR = 'data/mesonet/brec'
OUTPUT_FILE = f'data/{county}_outages_mesonet.nc'
MESONET_VARS = ['RELH', 'TAIR', 'WSPD', 'WVEC', 'WDIR', 'WDSD', 'WSSD', 'WMAX', 'RAIN', 'PRES', 'SRAD']
MISSING_VALS = [-994,-995,-996,-997,-998,-999]
threshold = 37   # minimum customers_out to count as an outage event
n_top     = 20   # number of largest events to plot

#metadata
var_attrs = {
    'RELH': {'long_name': 'Relative humidity',            'units': '%'},
    'TAIR': {'long_name': 'Air temperature',              'units': 'degC'},
    'WSPD': {'long_name': 'Wind speed',                   'units': 'm/s'},
    'WVEC': {'long_name': 'Vector-averaged wind speed',   'units': 'm/s'},
    'WDIR': {'long_name': 'Wind direction',               'units': 'deg'},
    'WDSD': {'long_name': 'Wind direction std dev',       'units': 'deg'},
    'WSSD': {'long_name': 'Wind speed std dev',           'units': 'm/s'},
    'WMAX': {'long_name': 'Maximum wind speed',           'units': 'm/s'},
    'RAIN': {'long_name': 'Rainfall',                     'units': 'mm'},
    'PRES': {'long_name': 'Atmospheric pressure',         'units': 'mb'},
    'SRAD': {'long_name': 'Solar radiation',              'units': 'W/m^2'},
}

#%% Main

# --- Eagle-I: load and filter for {county} County, OK ---
eaglei_files = sorted(glob.glob(os.path.join(EAGLEI_DIR, '*.csv')))
outage_frames = []
for f in eaglei_files:
    df = pd.read_csv(f, parse_dates=['run_start_time'])
    df = df[(df['county'] == f'{county}') & (df['state'] == 'Oklahoma')]
    if 'sum' in df.columns:
        df = df.rename(columns={'sum': 'customers_out'})
    elif 'customers_out' not in df.columns:
        raise ValueError(f"Neither 'sum' nor 'customers_out' found in {f}")
    outage_frames.append(df[['run_start_time', 'customers_out']])
    print(f'{os.path.basename(f)} appended')

outages = pd.concat(outage_frames, ignore_index=True)
outages = outages.groupby('run_start_time')['customers_out'].sum().sort_index()
print(f'Eagle-I: {len(outages)} {county} County records '
      f'({outages.index.min()} — {outages.index.max()})')

# Use the native Eagle-I reporting timestamps as the time axis
outages = outages.resample('15min').asfreq(fill_value=0)
time_grid = outages.index

# --- Mesonet: load all daily BREC files ---
mesonet_files = sorted(glob.glob(os.path.join(MESONET_DIR, '*.csv')))
meso_frames = []
for f in mesonet_files:
    date_str = os.path.basename(f).replace('brec_', '').replace('.csv', '')
    df = pd.read_csv(f, usecols=['TIME'] + MESONET_VARS)
    df.index = pd.to_datetime(date_str + ' ' + df['TIME'])
    df.index.name = 'time'
    meso_frames.append(df[MESONET_VARS])

mesonet = pd.concat(meso_frames).sort_index()
mesonet = mesonet.replace(dict.fromkeys(MISSING_VALS, np.nan))
print(f'Mesonet: {len(mesonet)} records '
      f'({mesonet.index.min()} — {mesonet.index.max()})')

# --- Align mesonet to outage timestamps ---
# Direct reindex onto the Eagle-I time axis; timestamps outside the mesonet
# record or flagged as missing remain NaN.
mesonet_aligned = mesonet.reindex(time_grid)

# --- Drop timestamps where any variable is NaN ---
df_combined = pd.DataFrame({'outages': outages}, index=time_grid)
for var in MESONET_VARS:
    df_combined[var] = mesonet_aligned[var]
df_combined     = df_combined.dropna()
outages         = df_combined['outages']
mesonet_aligned = df_combined[MESONET_VARS]
time_grid       = df_combined.index
print(f'After dropping NaN rows: {len(df_combined)} records remaining '
      f'({time_grid.min()} — {time_grid.max()})')

#%% Output
data_vars = {
    'outages': (
        'time',
        outages.values.astype(float),
        {'long_name': 'Customers without power', 'units': 'count'},
    )
}
for var in MESONET_VARS:
    data_vars[var] = ('time', mesonet_aligned[var].values, var_attrs[var])

ds = xr.Dataset(data_vars, coords={'time': time_grid})
ds.attrs['description'] = (
    f'Power outages (Eagle-I) and surface meteorology (Oklahoma Mesonet, BREC station) '
    f'for {county} County, OK at 15-minute resolution'
)
ds.attrs['county'] = county
ds.attrs['state'] = 'Oklahoma'
ds.attrs['mesonet_station'] = 'BREC — Breckinridge, OK'
ds.attrs['outage_source'] = 'Eagle-I Outages dataset'

ds.to_netcdf(OUTPUT_FILE)
print(f'\nSaved: {OUTPUT_FILE}')
print(ds)

#%% Plots
os.makedirs('figures/time_series_raw', exist_ok=True)
os.makedirs('figures/histograms', exist_ok=True)

plot_cols = ['outages'] + MESONET_VARS
df_plot = pd.DataFrame({'outages': outages.values}, index=time_grid)
for var in MESONET_VARS:
    df_plot[var] = mesonet_aligned[var].values

# --- Identify top-N outage events above threshold ---
above = df_plot['outages'].fillna(0) > threshold
event_id = (above != above.shift()).cumsum()
events = []
for _, grp in df_plot.groupby(event_id):
    if above[grp.index[0]]:
        events.append({'t_start': grp.index[0], 'peak': grp['outages'].max()})

top_events = sorted(events, key=lambda x: x['peak'], reverse=True)[:n_top]
print(f'\nTop {len(top_events)} outage events (threshold={threshold} customers):')
for i, ev in enumerate(top_events):
    print(f'  {i+1:2d}. start={ev["t_start"]}  peak={ev["peak"]:.0f}')

# --- Time series: 2-day window centred on the start of each top event ---
half_window = pd.Timedelta(days=2)
for rank, ev in enumerate(top_events):
    t0 = max(time_grid[0], ev['t_start'] - half_window)
    t1 = min(time_grid[-1], ev['t_start'] + half_window)
    chunk = df_plot.loc[t0:t1]
    flag = chunk['outages'].fillna(0) > threshold

    fig, axes = plt.subplots(len(plot_cols), 1,
                             figsize=(18, 2 * len(plot_cols)), sharex=True)
    for ax, col in zip(axes, plot_cols):
        ax.plot(chunk.index, chunk.loc[:,col],'-k',lw=0.5)
        ax.plot(chunk.index[~flag], chunk.loc[~flag, col],'.g')
        ax.plot(chunk.index[flag],  chunk.loc[flag,  col], '.r')
        if col == 'outages':
            ax.axhline(threshold, color='black', lw=1, ls='--')
        ax.set_ylabel(col)
        ax.set_xlim([t0, t1])
        ax.grid()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H'))
    plt.tight_layout()
    fname = (f'figures/time_series_raw/'
             f'outage_{rank+1:02d}_{ev["t_start"].strftime("%Y%m%d_%H%M")}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')

# --- Histograms of all variables ---
ncols = 3
nrows = (len(plot_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()
for ax, col in zip(axes, plot_cols):
    ax.hist(df_plot[col].dropna(), bins=50, color='b', edgecolor='none')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.grid()
for ax in axes[len(plot_cols):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig('figures/histograms/histograms.png', dpi=300, bbox_inches='tight')
plt.close()
print('  Saved figures/histograms/histograms.png')
