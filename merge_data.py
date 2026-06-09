import os
import numpy as np
import pandas as pd
import xarray as xr

#%% Inputs
EAGLEI_DIR  = 'data/eaglei'
MESONET_DIR = 'data/mesonet'
OUTPUT_DIR  = 'data'

SITE_COUNTY = {
    'BREC': 'Garfield',
    'REDR': 'Noble',
}

MESONET_VARS = ['relh', 'tair', 'wspd', 'wvec', 'wdir', 'wdsd', 'wssd', 'wmax', 'pres', 'srad', 'rain']

var_attrs = {
    'customers_out': {'long_name': 'Customers without power', 'units': 'count'},
    'relh': {'long_name': 'Relative humidity',           'units': '%'},
    'tair': {'long_name': 'Air temperature',             'units': 'degC'},
    'wspd': {'long_name': 'Wind speed',                  'units': 'm/s'},
    'wvec': {'long_name': 'Vector-averaged wind speed',  'units': 'm/s'},
    'wdir': {'long_name': 'Wind direction',              'units': 'deg'},
    'wdsd': {'long_name': 'Wind direction std dev',      'units': 'deg'},
    'wssd': {'long_name': 'Wind speed std dev',          'units': 'm/s'},
    'wmax': {'long_name': 'Maximum wind speed',          'units': 'm/s'},
    'pres': {'long_name': 'Atmospheric pressure',        'units': 'mb'},
    'srad': {'long_name': 'Solar radiation',             'units': 'W/m^2'},
    'rain': {'long_name': 'Rainfall',                    'units': 'mm'},
}

#%% Load Eagle-I (all years, all counties at once)
eaglei_files = sorted(
    os.path.join(EAGLEI_DIR, f)
    for f in os.listdir(EAGLEI_DIR)
    if f.endswith('.csv')
)
frames = []
for f in eaglei_files:
    df = pd.read_csv(f, parse_dates=['run_start_time'])
    if 'sum' in df.columns:
        df = df.rename(columns={'sum': 'customers_out'})
    df = df[df['state'] == 'Oklahoma'][['county', 'customers_out', 'run_start_time']]
    frames.append(df)
    print(f'  loaded {os.path.basename(f)}')

eaglei_ok = pd.concat(frames, ignore_index=True)

#%% Process each site
for site, county in SITE_COUNTY.items():
    print(f'\n--- {site} / {county} ---')

    # --- Eagle-I: sum customers_out per timestamp for this county ---
    county_data = (
        eaglei_ok[eaglei_ok['county'] == county]
        .groupby('run_start_time')['customers_out']
        .sum()
        .sort_index()
        .rename_axis('time')
    )
    # fill gaps in the reporting grid with 0 (no outage records = no outage)
    county_data = county_data.resample('15min').asfreq(fill_value=0)
    print(f'  Eagle-I: {len(county_data)} records '
          f'({county_data.index.min()} — {county_data.index.max()})')

    # --- Mesonet: find the NetCDF for this site ---
    nc_files = [
        f for f in os.listdir(MESONET_DIR)
        if f.startswith(f'mesonet.{site.lower()}.') and f.endswith('.nc')
    ]
    if not nc_files:
        print(f'  No mesonet NetCDF found for {site}, skipping')
        continue
    nc_path = os.path.join(MESONET_DIR, sorted(nc_files)[-1])
    ds_meso = xr.open_dataset(nc_path)
    meso = ds_meso.to_dataframe()[MESONET_VARS]
    print(f'  Mesonet: {len(meso)} records '
          f'({meso.index.min()} — {meso.index.max()})')

    # --- Align on the Eagle-I time grid (primary axis) ---
    meso_aligned = meso.reindex(county_data.index)

    # --- Build combined DataFrame and drop rows missing any variable ---
    combined = pd.DataFrame({'customers_out': county_data}, index=county_data.index)
    for var in MESONET_VARS:
        combined[var] = meso_aligned[var]
    combined = combined.dropna()
    print(f'  After dropping NaN rows: {len(combined)} records '
          f'({combined.index.min()} — {combined.index.max()})')

    # --- Build xarray Dataset ---
    data_vars = {
        col: ('time', combined[col].values, var_attrs[col])
        for col in combined.columns
    }
    ds = xr.Dataset(data_vars, coords={'time': combined.index})
    ds.attrs['station']     = site
    ds.attrs['county']      = county
    ds.attrs['state']       = 'Oklahoma'
    ds.attrs['source']      = 'Eagle-I outages + Oklahoma Mesonet'
    ds.attrs['description'] = (
        f'Power outages (Eagle-I, {county} County OK) and surface meteorology '
        f'(Oklahoma Mesonet, {site} station) at 15-minute resolution'
    )

    out_path = os.path.join(OUTPUT_DIR, f'{site.lower()}.outages_mesonet.nc')
    ds.to_netcdf(out_path)
    print(f'  Saved {out_path}')
    print(ds)
