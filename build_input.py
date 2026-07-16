# -*- coding: utf-8 -*-
"""
Build input for RF outage predictor
"""

import tkinter
import tkinter.filedialog
import yaml
import xarray as xr
import numpy as np

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source_data = tkinter.filedialog.askopenfilename(
    title='Select met data',
    initialdir='./data/',
    filetypes=[('NetCDF files', '*.nc')])

source_config = tkinter.filedialog.askopenfilename(
    title='Select configuration file',
    initialdir='./',
    filetypes=[('YAML files', '*.yaml')])


MAPPING={'wspd':'wind_speed',
         'tair':'temperature',
         'relh':'relative_humidity',
         'pres':'pressure',
         'srad':'shortwave_radiation',
         'wssd':'wind_speed',
         'wdsd':'wind_direction'}

STATS={'wspd':'mean',
        'tair':'mean',
        'relh':'mean',
        'pres':'mean',
        'srad':'mean',
        'wssd':'std',
        'wdsd':'std'}

CONVERSION={'wspd':1,
        'tair':1,
        'relh':1,
        'pres':10,
        'srad':1, #1000/0.2,
        'wssd':1,
        'wdsd':1}

#%% Initialization
ds_in=xr.open_dataset(source_data)

with open(source_config) as f:
    config=yaml.safe_load(f)

ds_out=xr.Dataset()


def _read_attr(ds, keys):
    for key in keys:
        if key in ds.attrs:
            try:
                return float(ds.attrs[key])
            except (TypeError, ValueError):
                continue
    return np.nan

predictors=config['predictor_cols']
if 'aavi' in predictors:
    predictors+=['wssd','wdsd']
    
#%% Main
for v in predictors:
    if v in MAPPING.keys():
        if MAPPING[v] in ds_in.data_vars:
            ds_out[v]=ds_in[MAPPING[v]].sel(stat=STATS[v])*CONVERSION[v]
        else:
            print(f'Missing {v} in input')
        
ds_out['customers_out']=ds_in.outages

# Preserve station geolocation metadata so downstream products can export it.
lat_val = _read_attr(ds_in, ['latitude', 'lat', 'station_lat'])
lon_val = _read_attr(ds_in, ['longitude', 'lon', 'station_lon'])

if np.isfinite(lat_val):
    ds_out.attrs['latitude'] = float(lat_val)
if np.isfinite(lon_val):
    ds_out.attrs['longitude'] = float(lon_val)

for key in ['station', 'county', 'state', 'source', 'description']:
    if key in ds_in.attrs:
        ds_out.attrs[key] = ds_in.attrs[key]

ds_in.close()

#%% Output
ds_out.drop('stat').to_netcdf(source_data.replace('.nc','.input.nc'))

