import os
import pandas as pd
import xarray as xr

#%% Inputs
SITES = ['BREC', 'REDR']
WEATHER_DIR = 'data/mesonet/raw/weather'
RAIN_DIR    = 'data/mesonet/raw/rain'
OUTPUT_DIR  = 'data/mesonet'

var_attrs = {
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

#%% Main
for site in SITES:
    # --- Weather ---
    # os.walk forces OneDrive on-demand directories to expand their listings
    weather_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(WEATHER_DIR)
        for f in files
        if f.startswith(site) and f.endswith('.csv')
    )
    if not weather_files:
        print(f'{site}: no weather files found, skipping')
        continue

    wx_frames = []
    for f in weather_files:
        df = pd.read_csv(f, parse_dates=['time'])
        df = df.set_index('time').drop(columns=['stid'])
        # strip the ".mean" suffix from column names
        df.columns = [c.replace('.mean', '') for c in df.columns]
        wx_frames.append(df)
        print(f'  loaded {os.path.basename(f)}')

    weather = pd.concat(wx_frames).sort_index()
    weather.index.name = 'time'

    # --- Rain ---
    rain_dir = os.path.join(RAIN_DIR, site)
    rain_files = sorted(
        os.path.join(rain_dir, f)
        for f in os.listdir(rain_dir)
        if f.startswith(site) and f.endswith('_15min.csv')
    ) if os.path.isdir(rain_dir) else []
    if not rain_files:
        print(f'{site}: no rain files found, merging without rainfall')
        rain = None
    else:
        rain_frames = []
        for f in rain_files:
            df = pd.read_csv(f, parse_dates=['timestamp'])
            df = df.set_index('timestamp').rename(columns={'rain_mm': 'rain'})
            rain_frames.append(df)
            print(f'  loaded {os.path.basename(f)}')
        rain = pd.concat(rain_frames).sort_index()
        rain.index.name = 'time'

    # --- Merge on shared time axis ---
    if rain is not None:
        merged = weather.join(rain, how='outer')
    else:
        merged = weather

    merged = merged.sort_index()

    # --- Build xarray Dataset ---
    data_vars = {}
    for col in merged.columns:
        key = col  # already stripped to base name
        attrs = var_attrs.get(key, {})
        data_vars[key] = ('time', merged[col].values, attrs)

    ds = xr.Dataset(data_vars, coords={'time': merged.index})
    ds.attrs['station']     = site
    ds.attrs['source']      = 'Oklahoma Mesonet'
    ds.attrs['description'] = (f'Oklahoma Mesonet surface meteorology and rainfall '
                                f'at station {site} at 15-minute resolution')

    t0 = merged.index.min().strftime('%Y%m%d')
    t1 = merged.index.max().strftime('%Y%m%d')
    out_path = os.path.join(OUTPUT_DIR, f'mesonet.{site.lower()}.{t0}.{t1}.nc')
    ds.to_netcdf(out_path)
    print(f'\nSaved {out_path}')
    print(ds)
    print()
