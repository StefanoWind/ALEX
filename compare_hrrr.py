import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
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
    'mesonet_source':    'data/brec.outages_mesonet.nc',
    'hrrr_source':       'data/2023-01-01.2023-01-07.hrrr.nc',
    'station_lat':       36.412010,   # used if latitude not in mesonet NC attributes
    'station_lon':       360-97.693940,
    'sample_week':       '2023-01-01',
    'mesonet_resample':  '1h',    # resample mesonet to match HRRR temporal resolution
    'output_dir':        'figures/hrrr_comparison',
}

# ── Load ──────────────────────────────────────────────────────────────────────
ds_meso = xr.open_dataset(CFG['mesonet_source'])
ds_hrrr = xr.open_dataset(CFG['hrrr_source'])

ds_hrrr['wspd']=(ds_hrrr.u10**2+ds_hrrr.v10**2)**0.5
ds_hrrr['wspd_f18']=(ds_hrrr.u10_f18**2+ds_hrrr.v10_f18**2)**0.5

station_lat = float(ds_meso.attrs.get('latitude', CFG['station_lat']))
station_lon = float(ds_meso.attrs.get('longitude', CFG['station_lon']))

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
common_vars = sorted(set(ds_meso.data_vars) & set(ds_hrrr.data_vars))
if not common_vars:
    raise RuntimeError(
        f"No common variable names.\n"
        f"  Mesonet: {sorted(ds_meso.data_vars)}\n"
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
            .to_dataframe()[all_hrrr_pt])
hrrr_all.index = pd.DatetimeIndex(ds_hrrr.time.values)

hrrr_anl = hrrr_all[common_vars]
hrrr_fct = (hrrr_all[fcst_hrrr_vars]
            .rename(columns={f'{v}{fcst_suffix}': v for v in common_vars})
            if have_fcst else pd.DataFrame(index=hrrr_anl.index))

# ── Resample mesonet to HRRR temporal resolution ──────────────────────────────
meso_df = ds_meso[common_vars].to_dataframe()[common_vars]
meso_df.index = pd.DatetimeIndex(ds_meso.time.values)
meso_h = meso_df.resample(CFG['mesonet_resample']).mean()

# ── Align on common time axis ─────────────────────────────────────────────────
meso_al, hrrr_anl_al = meso_h.align(hrrr_anl, join='inner')
if have_fcst:
    _, hrrr_fct_al = meso_h.align(hrrr_fct, join='inner')

out_dir = Path(CFG['output_dir'])
out_dir.mkdir(parents=True, exist_ok=True)

n = len(common_vars)

COLORS = {
    'meso': 'steelblue',
    'anl':  'firebrick',
    'fct':  'darkorange',
}


def _label(var):
    return _VAR_LABELS.get(var, var)


# ── Plot 1: sample-week time series (all three streams in one figure) ──────────
t0 = pd.Timestamp(CFG['sample_week'])
t1 = t0 + pd.Timedelta(weeks=1)
mask_w = (meso_al.index >= t0) & (meso_al.index < t1)
t_idx  = meso_al.index[mask_w]

fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
if n == 1:
    axes = [axes]

for ax, var in zip(axes, common_vars):
    ax.plot(t_idx, meso_al.loc[mask_w, var],
            color=COLORS['meso'], lw=1.5, label='Mesonet')
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
    f"Mesonet vs HRRR  —  "
    f"{t0.strftime('%Y-%m-%d')} to {(t1 - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
)
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.autofmt_xdate(rotation=30, ha='right')
plt.tight_layout()
save_ts = out_dir / 'hrrr_comparison_ts.png'
fig.savefig(save_ts, bbox_inches='tight')
plt.close(fig)
print(f"Saved → {save_ts}")


# ── Plot 2: linear regression (separate figures for analysis and forecast) ─────
# [Wilks, 2011, Statistical Methods in the Atmospheric Sciences] — regression verification
def _reg_figure(meso_df_al, hrrr_df_al, stream_label, color, filename):
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, var in zip(axes_flat, common_vars):
        both = pd.concat([meso_df_al[var], hrrr_df_al[var]], axis=1,
                         keys=['meso', 'hrrr']).dropna()
        x, y = both['meso'].values, both['hrrr'].values

        slope, intercept, r, _, _ = stats.linregress(x, y)

        ax.scatter(x, y, s=6, alpha=0.35, color=color, rasterized=True)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, slope * x_line + intercept, color='k', lw=1.5,
                label=f'r={r:.2f},  slope={slope:.2f}')
        ax.plot(x_line, x_line, color='k', lw=0.8, ls='--', label='1:1')
        lbl = _label(var)
        ax.set_xlabel(f'Mesonet — {lbl}', fontsize=10)
        ax.set_ylabel(f'{stream_label} — {lbl}', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f'Mesonet vs {stream_label} — linear regression (all data)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {filename}")


_reg_figure(meso_al, hrrr_anl_al,
            'HRRR anl', COLORS['anl'],
            out_dir / 'hrrr_comparison_regression_anl.png')

if have_fcst:
    _reg_figure(meso_al, hrrr_fct_al,
                f'HRRR f{fcst_lead}h', COLORS['fct'],
                out_dir / 'hrrr_comparison_regression_fct.png')
