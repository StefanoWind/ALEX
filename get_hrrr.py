import sys
import numpy as np
import pandas as pd
import xarray as xr
xr.set_options(use_new_combine_kwarg_defaults=True)    
from pathlib import Path
from herbie import FastHerbie
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# HRRR: [Benjamin et al., 2016, Mon. Wea. Rev.]; v4: [Dowell et al., 2022, Bull. Amer. Meteor. Soc.]

#%% Inputs
if len(sys.argv)==1:
    mode='serial'
    sdate='2022-09-01'
    edate='2025-10-01'
else:
    mode=sys.argv[0]
    sdate=sys.argv[1]
    edate=sys.argv[2]
    
CFG = {'start':sdate,
       'end': edate,
        'freq':             '1h',
        'fxx_fcst':         18,               # forecast lead (hours); 18h is the max for all HRRR cycles
        'fcst_cycle_hours': list(range(24)),  # all cycles support fxx=18h → every hour is filled
        'product':          'sfc',
        'lat_range':        [36.0, 36.5],   # AWAKEN domain — northern Oklahoma
        'lon_range':        [360-97.75, 360-97.25],
        'variables': {                       # output name → HRRR GRIB searchstring
            'u10':  ':UGRD:10 m above ground:',
            'v10':  ':VGRD:10 m above ground:',
            'gust': ':GUST:surface:',
            'tair': ':TMP:2 m above ground:',
            'relh': ':RH:2 m above ground:',
            'pres': ':PRES:surface:',
            'srad': ':DSWRF:surface:',
            'prate': ':PRATE:surface:'  
        },
        'max_threads': 10,
        'chunk_days':  1,
        'output':     str(Path(__file__).parent / 'data/hrrr/{year:04d}{month:02d}01.000000.hrrr.nc'),
}

# metadata variables that cfgrib adds alongside the actual data
_META_VARS = {
    'latitude', 'longitude', 'gribfile_projection', 'spatial_ref',
    'step', 'valid_time', 'time', 'heightAboveGround', 'surface',
    'meanSea', 'entireAtmosphere', 'unknown',
}

#%% Functions
def _main_var(ds: xr.Dataset) -> str:
    candidates = [v for v in ds.data_vars if v not in _META_VARS]
    if not candidates:
        raise KeyError(f"No data variable found; got {list(ds.data_vars)}")
    return candidates[0]


def _subset(ds: xr.Dataset, lat_range: list, lon_range: list) -> xr.Dataset:
    lat  = ds.latitude.values
    lon  = ds.longitude.values
    mask = ((lat >= lat_range[0]) & (lat <= lat_range[1]) &
            (lon >= lon_range[0]) & (lon <= lon_range[1]))
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError(f"No HRRR grid points in domain "
                         f"lat={lat_range}, lon={lon_range}")
    return ds.isel(y=slice(int(rows.min()), int(rows.max()) + 1),
                   x=slice(int(cols.min()), int(cols.max()) + 1))


def _ensure_time_dim(da: xr.DataArray) -> xr.DataArray:
    if 'valid_time' in da.dims and 'time' not in da.dims:
        return da.rename({'valid_time': 'time'})
    if 'time' not in da.dims:
        return da.expand_dims('time')
    # For forecasts cfgrib sets time=cycle_time and valid_time=cycle+fxx as an aux coord.
    # Replace the cycle-time index with valid_time so analysis and forecast share the same axis.
    if 'valid_time' in da.coords:
        da = da.assign_coords(time=da['valid_time']).drop_vars('valid_time')
    return da


def _fetch_vars(dates: pd.DatetimeIndex, fxx: int, cfg: dict) -> dict:
    fh = FastHerbie(dates, model='hrrr', product=cfg['product'],
                    fxx=[fxx], max_threads=cfg['max_threads'])
    arrays = {}
    for name, searchstr in cfg['variables'].items():
        try:
            ds_v = fh.xarray(searchstr, remove_grib=True)
            if 'heightAboveGround' in ds_v.coords:
                ds_v = ds_v.drop_vars('heightAboveGround')
            var  = _main_var(ds_v)
            da   = _ensure_time_dim(ds_v[var])
            arrays[name] = da
            print(f"  fxx={fxx:3d}  {name}: {da.dims}  shape={da.shape}")
        except Exception as exc:
            print(f"  Warning — '{name}' (fxx={fxx}): {exc}")
    return arrays


def fetch_chunk(dates: pd.DatetimeIndex, cfg: dict) -> xr.Dataset | None:
    fcst_h = cfg['fxx_fcst']

    # ── Analysis (fxx=0, all dates) ──────────────────────────────────────────
    anl_arrays = _fetch_vars(dates, 0, cfg)
    if not anl_arrays:
        return None
    ds_anl = _subset(xr.Dataset(anl_arrays), cfg['lat_range'], cfg['lon_range'])

    # ── Forecast (fxx=fcst_h) ────────────────────────────────────────────────
    # Back-compute cycle times from target valid times so that cycle + fcst_h = valid.
    # With fxx=18h and all 24 cycle hours included, every hourly slot is covered.
    target_valid = dates[np.isin(dates.hour, cfg['fcst_cycle_hours'])]
    if len(target_valid) > 0:
        fcst_cycles = target_valid - pd.Timedelta(hours=fcst_h)
        fcst_arrays = _fetch_vars(fcst_cycles, fcst_h, cfg)
        if fcst_arrays:
            ds_fcst_sub = _subset(xr.Dataset(fcst_arrays), cfg['lat_range'], cfg['lon_range'])
            # reindex to guard against any edge-case timestamp mismatch
            ds_fcst = ds_fcst_sub.reindex(time=ds_anl.time)
            ds_fcst = ds_fcst.rename({v: f'{v}_f{fcst_h}' for v in ds_fcst.data_vars})
            return xr.merge([ds_anl, ds_fcst], compat='override')

    return ds_anl


def _run_chunk(args: tuple) -> tuple:
    idx, dates_list, cfg = args
    chunk = pd.DatetimeIndex(dates_list)
    try:
        return idx, fetch_chunk(chunk, cfg)
    except Exception as exc:
        print(f"  Chunk {idx} failed: {exc}")
        return idx, None

#%% Main
if __name__ == '__main__':
    dates   = pd.date_range(CFG['start'], CFG['end'], freq=CFG['freq'])
    n_chunk = CFG['chunk_days'] * 24
    chunks  = [dates[i:i + n_chunk] for i in range(0, len(dates), n_chunk)]

    # group chunks by (year, month) of their first date
    monthly_chunks: dict = defaultdict(list)
    for i, chunk in enumerate(chunks):
        key = (chunk[0].year, chunk[0].month)
        monthly_chunks[key].append((i, chunk))

    n_workers = CFG.get('n_workers', 1)

    print(f"Date range : {dates[0]} → {dates[-1]}  ({len(dates)} hours)")
    print(f"Variables  : {list(CFG['variables'].keys())}")
    print(f"Domain     : lat={CFG['lat_range']}, lon={CFG['lon_range']}")
    print(f"Forecast   : fxx={CFG['fxx_fcst']}h from {CFG['fcst_cycle_hours']}z cycles only")
    print(f"Chunks     : {len(chunks)} × {CFG['chunk_days']} days  |  workers={n_workers}\n")

    for (year, month), indexed_chunks in sorted(monthly_chunks.items()):
        out_path = Path(CFG['output'].format(year=year, month=month))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {year}-{month:02d}  ({len(indexed_chunks)} chunks) → {out_path}")

        args_list = [(i, list(chunk), CFG) for i, chunk in indexed_chunks]
        results: dict = {}

        if mode=='parallel':
            with ProcessPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(_run_chunk, a): a[0] for a in args_list}
                for future in as_completed(futures):
                    idx, ds = future.result()
                    if ds is not None:
                        results[idx] = ds
        elif mode=='serial':
            for args in args_list:
                idx, ds = _run_chunk(args)
                if ds is not None:
                    results[idx] = ds
        else:
            raise ValueError(f'Unknown processing mode: {mode}')

        if not results:
            print("  No data retrieved, skipping.")
            continue

        ds_out = xr.concat([results[i] for i in sorted(results)], dim='time')
        encoding = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
        ds_out.to_netcdf(out_path, encoding=encoding)
        print(f"Saved → {out_path}   {dict(ds_out.dims)}")
