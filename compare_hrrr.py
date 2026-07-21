import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
from utils import _VAR_LABELS

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

CFG = {
    'met_source':    'data/merged_outages_gf_15min/merged_outages_gf_15min_metA1only.input.nc',
    'hrrr_source':       'data/hrrr/*nc',
    'sample_week':       '2023-06-15',
    'met_resample':  '1h',    # resample met to match HRRR temporal resolution
    'nearest_tolerance': '1h',    # max gap allowed for nearest-neighbor time matching; farther → nan
    'met_sources':       ['data/merged_outages_gf_15min/merged_outages_gf_15min_metA1only.input.nc',
                          'data/merged_outages_gf_15min/merged_outages_gf_15min_metA2only.input.nc',
                          'data/merged_outages_gf_15min/merged_outages_gf_15min_metA5only.input.nc',
                          'data/merged_outages_gf_15min/merged_outages_gf_15min_metA7only.input.nc',
                          'data/merged_outages_gf_15min/merged_outages_gf_15min_metBonly.input.nc',
                          'data/merged_outages_gf_15min/merged_outages_gf_15min_metC1only.input.nc',
                          'data/merged_outages_gf_15min/merged_outages_gf_15min_metGonly.input.nc'],    # list of met NetCDFs to scatter on the domain maps; None → just met_source
    'n_snapshot':        24,       # number of evenly spaced domain-map snapshots across the first day of sample_week
    'output_dir':        'figures/hrrr_comparison',
    'min':{'tair':0,'relh':0,'pres':955,'wspd':0,'srad':0},
    'max':{'tair':30,'relh':100,'pres':975,'wspd':10,'srad':1000},
}

# ── Load ──────────────────────────────────────────────────────────────────────
ds_met = xr.open_dataset(CFG['met_source'])

_hrrr_files = sorted(Path(__file__).parent.glob(CFG['hrrr_source']))
if not _hrrr_files:
    raise FileNotFoundError(f"No HRRR files found matching {CFG['hrrr_source']}")
print(f"Loading {len(_hrrr_files)} HRRR files …")
ds_hrrr = xr.open_mfdataset(
    _hrrr_files,
    combine='nested',
    concat_dim='time',
    coords='minimal',
    compat='override',
)

ds_hrrr['wspd']=(ds_hrrr.u10**2+ds_hrrr.v10**2)**0.5
ds_hrrr['wspd_f18']=(ds_hrrr.u10_f18**2+ds_hrrr.v10_f18**2)**0.5

station_lat = float(ds_met.attrs.get('latitude'))
station_lon = float(ds_met.attrs.get('longitude'))%360

# ── Detect forecast suffix (e.g. '_f18') from HRRR variable names ─────────────
fcst_suffix = None
for hv in ds_hrrr.data_vars:
    m = re.search(r'(_f\d+)$', hv)
    if m:
        fcst_suffix = m.group(1)
        break
fcst_lead = fcst_suffix[2:] if fcst_suffix else None   # '18' from '_f18'
print(f"Forecast suffix: {fcst_suffix!r}")

# ── Unit conversions (applied to both analysis and forecast variables) ─────────
_unit_ops = {
    'pres': lambda x: x / 100,
    'tair': lambda x: x - 273.15,
}
for base, op in _unit_ops.items():
    if base in ds_hrrr:
        ds_hrrr[base] = op(ds_hrrr[base])
    if fcst_suffix and f'{base}{fcst_suffix}' in ds_hrrr:
        ds_hrrr[f'{base}{fcst_suffix}'] = op(ds_hrrr[f'{base}{fcst_suffix}'])

# ── Common variables ──────────────────────────────────────────────────────────
common_vars = sorted(set(ds_met.data_vars) & set(ds_hrrr.data_vars))
if not common_vars:
    raise RuntimeError(
        f"No common variable names.\n"
        f"  Mesonet: {sorted(ds_met.data_vars)}\n"
        f"  HRRR:    {sorted(ds_hrrr.data_vars)}"
    )
print(f"Common variables ({len(common_vars)}): {common_vars}")

fcst_hrrr_vars = [f'{v}{fcst_suffix}' for v in common_vars] if fcst_suffix else []
fcst_hrrr_vars = [v for v in fcst_hrrr_vars if v in ds_hrrr.data_vars]
have_fcst = bool(fcst_hrrr_vars)

# ── Interpolate HRRR at station location (nearest grid point) ─────────────────
lat2d = ds_hrrr['latitude'].values
lon2d = ds_hrrr['longitude'].values
dist  = np.hypot(lat2d - station_lat, lon2d - station_lon)
iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
print(f"Nearest HRRR grid point: lat={lat2d[iy, ix]:.4f}, lon={lon2d[iy, ix]:.4f}  "
      f"(≈{dist[iy, ix] * 111:.1f} km from station)")

all_hrrr_pt = common_vars + fcst_hrrr_vars
hrrr_all = (ds_hrrr[all_hrrr_pt]
            .isel(y=iy, x=ix)
            .load()
            .to_dataframe()[all_hrrr_pt])

hrrr_anl = hrrr_all[common_vars]
hrrr_fct = (hrrr_all[fcst_hrrr_vars]
            .rename(columns={f'{v}{fcst_suffix}': v for v in common_vars})
            if have_fcst else pd.DataFrame(index=hrrr_anl.index))

# ── Resample met to HRRR temporal resolution ──────────────────────────────
# Mesonet timestamps mark the beginning of each 15-min averaging window;
# shift by half the interval to align with the window midpoint before resampling.
met_df = ds_met[common_vars].to_dataframe()[common_vars]
met_df.index = pd.DatetimeIndex(ds_met.time.values) + pd.Timedelta(minutes=7.5)

# Nearest-neighbor match onto the resampled grid, but only within
# `nearest_tolerance` of an actual sample — otherwise nan. Plain
# resample(...).nearest() has no such limit and silently reuses whatever
# sample is closest even across large data gaps, producing flat
# interpolation artifacts.
_met_grid = met_df.resample(CFG['met_resample']).asfreq().index
met_h = met_df.reindex(_met_grid, method='nearest',
                          tolerance=pd.Timedelta(CFG['nearest_tolerance']))

# ── Align on common time axis ─────────────────────────────────────────────────
met_al, hrrr_anl_al = met_h.align(hrrr_anl, join='inner')
if have_fcst:
    _, hrrr_fct_al = met_h.align(hrrr_fct, join='inner')

out_dir = Path(CFG['output_dir'])
out_dir.mkdir(parents=True, exist_ok=True)

n = len(common_vars)

COLORS = {
    'met': 'steelblue',
    'anl':  'firebrick',
    'fct':  'darkorange',
}


def _label(var):
    return _VAR_LABELS.get(var, var)


# ── Plot 1: sample-week time series (all three streams in one figure) ──────────
t0 = pd.Timestamp(CFG['sample_week'])
t1 = t0 + pd.Timedelta(weeks=1)
mask_w = (met_al.index >= t0) & (met_al.index < t1)
t_idx  = met_al.index[mask_w]

fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
if n == 1:
    axes = [axes]

for ax, var in zip(axes, common_vars):
    ax.plot(t_idx, met_al.loc[mask_w, var],'-',
            color=COLORS['met'], lw=1.5, label='Met')
    ax.scatter(t_idx, hrrr_anl_al.loc[mask_w, var],
               color=COLORS['anl'], s=25, zorder=3, label='HRRR anl')
    if have_fcst:
        ax.scatter(t_idx, hrrr_fct_al.loc[mask_w, var],
                   color=COLORS['fct'], s=25, zorder=3,
                   marker='^', label=f'HRRR f{fcst_lead}h')
    ax.set_ylabel(_label(var), fontsize=10)
    ax.grid(alpha=0.3)

axes[0].legend(fontsize=10)
axes[0].set_title(
    f"Met vs HRRR  —  "
    f"{t0.strftime('%Y-%m-%d')} to {(t1 - pd.Timedelta(days=1)).strftime('%Y-%m-%d')} \n"
    f"Source: {os.path.basename(CFG['met_source'])}"
)
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.autofmt_xdate(rotation=30, ha='right')
plt.tight_layout()
save_ts = out_dir / 'hrrr_comparison_ts.png'
fig.savefig(save_ts, bbox_inches='tight')
print(f"Saved → {save_ts}")


# ── Plot 2: linear regression (separate figures for analysis and forecast) ─────
# [Wilks, 2011, Statistical Methods in the Atmospheric Sciences] — regression verification
def _reg_figure(met_df_al, hrrr_df_al, stream_label, color, filename):
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, var in zip(axes_flat, common_vars):
        both = pd.concat([met_df_al[var], hrrr_df_al[var]], axis=1,
                         keys=['met', 'hrrr']).dropna()
        if len(both) < 2:
            ax.set_title(f'{var} — no valid pairs', fontsize=10)
            ax.set_visible(True)
            continue
        x, y = both['met'].values, both['hrrr'].values

        slope, intercept, r, _, _ = stats.linregress(x, y)

        ax.scatter(x, y, s=6, alpha=0.35, color=color, rasterized=True)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, slope * x_line + intercept, color='k', lw=1.5,
                label=f'r={r:.2f}, intercept={intercept:.2f}, slope={slope:.2f}')
        ax.plot(x_line, x_line, color='k', lw=0.8, ls='--', label='1:1')
        lbl = _label(var)
        ax.set_xlabel(f'Met — {lbl}', fontsize=10)
        ax.set_ylabel(f'{stream_label} — {lbl}', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle((f'Met vs {stream_label} — linear regression (all data) \n'
                 f"Source: {os.path.basename(CFG['met_source'])}"),
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved → {filename}")


_reg_figure(met_al, hrrr_anl_al,
            'HRRR anl', COLORS['anl'],
            out_dir / 'hrrr_comparison_regression_anl.png')

if have_fcst:
    _reg_figure(met_al, hrrr_fct_al,
                f'HRRR f{fcst_lead}h', COLORS['fct'],
                out_dir / 'hrrr_comparison_regression_fct.png')


# ── Plot 3: HRRR domain snapshot maps with station overlay ─────────────────────
# One pcolor map per variable per snapshot, showing the full HRRR analysis
# domain at `n_snapshot` evenly spaced times across the first day of
# `sample_week`. Each station in `met_sources` is overlaid as a scatter point
# using the same colormap and vmin/vmax as the pcolor field, so scatter and
# map colors are directly comparable.
_met_sources = CFG.get('met_sources') or [CFG['met_source']]
_domain_cmap = 'viridis'
ncols = min(3, n)
nrows = (n + ncols - 1) // ncols

_snapshot_hours = np.linspace(0, 24, CFG['n_snapshot'], endpoint=False)
snapshot_times = [pd.Timestamp(ds_hrrr.time.sel(
    time=t0 + pd.Timedelta(hours=float(h)), method='nearest').values)
    for h in _snapshot_hours]

for _i, snapshot_time in enumerate(snapshot_times):
    print(f"Domain map snapshot {_i + 1}/{len(snapshot_times)}: {snapshot_time}")

    _stations = []
    for _src in _met_sources:
        _ds_st = xr.open_dataset(_src)
        _st_lat = float(_ds_st.attrs.get('latitude'))
        _st_lon = float(_ds_st.attrs.get('longitude')) % 360   # HRRR grid uses 0-360 convention
        _st_label = str(_ds_st.attrs.get('station', Path(_src).stem))
        _st_vars = [v for v in common_vars if v in _ds_st.data_vars]
        _st_df = _ds_st[_st_vars].to_dataframe()[_st_vars]
        _st_df.index = pd.DatetimeIndex(_ds_st.time.values) + pd.Timedelta(minutes=7.5)
        _st_row = _st_df.reindex([snapshot_time], method='nearest',
                                  tolerance=pd.Timedelta(CFG['nearest_tolerance'])).iloc[0]
        _stations.append({'label': _st_label, 'lat': _st_lat, 'lon': _st_lon, 'row': _st_row})
        _ds_st.close()

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, var in zip(axes_flat, common_vars):
        field = ds_hrrr[var].sel(time=snapshot_time, method='nearest').load().values
        vmin, vmax = CFG['min'][var],CFG['max'][var]

        pcm = ax.pcolormesh(lon2d, lat2d, field, cmap=_domain_cmap, vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(pcm, ax=ax, label=_label(var))

        for st in _stations:
            val = st['row'].get(var, np.nan)
            if pd.notna(val):
                ax.scatter(st['lon'], st['lat'], c=[val], cmap=_domain_cmap, vmin=vmin, vmax=vmax,
                           s=100, edgecolor='k', linewidth=1.3, zorder=5)

        ax.set_title(_label(var), fontsize=11)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"HRRR domain — {snapshot_time.strftime('%Y-%m-%d %H:%M')}", fontsize=13)
    plt.tight_layout()
    save_map = out_dir / f"hrrr_domain_maps_{snapshot_time.strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(save_map, bbox_inches='tight')
    print(f"Saved → {save_map}")
    plt.close(fig)
