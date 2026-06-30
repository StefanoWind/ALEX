'''
Local Streamlit dashboard for the ALEX event library.
Run with: streamlit run dashboard.py
'''

import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from utils import plot_episode_ts, apply_climatology

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

st.set_page_config(page_title='ALEX Event Library', layout='wide')
st.title('ALEX — Outage Event Library')

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load(path: str):
    df = pd.read_excel(path, index_col=0)
    df.index = df.index.astype(str)

    if 'shap_base_value' in df.index:
        shap_base = float(df.loc['shap_base_value', 'outage_probability'])
        df = df.drop('shap_base_value')
    else:
        shap_base = np.nan

    df.index = pd.to_datetime(df.index)

    stat_cols = ['peak_customers_out', 'duration_h', 'auc_customer_h']
    shap_cols = [c for c in df.columns if c.startswith('shap_')]
    raw_cols  = [c for c in df.columns if c.endswith('_raw')]
    feat_cols = [c for c in df.columns
                 if c not in stat_cols + shap_cols + raw_cols + ['outage_probability']]

    means = df[feat_cols + raw_cols].mean()
    stds  = df[feat_cols + raw_cols].std().replace(0, np.nan)

    return df, feat_cols, raw_cols, shap_cols, stat_cols, means, stds, shap_base


@st.cache_data
def load_ts_data(nc_path: str, pred_cols: tuple, target_col: str):
    """Load raw met data and apply climatology for detrended version if available."""
    from pathlib import Path
    p = Path(nc_path)
    ds = xr.open_dataset(nc_path)

    # Compute aavi from directional variability if not directly available [Bianco et al., 2016]
    if 'aavi' in pred_cols and 'aavi' not in ds:
        ds['aavi'] = ds['wssd'] / ds['wssd'].mean() * ds['wdsd'] / ds['wdsd'].mean()

    avail = [c for c in list(pred_cols) + [target_col] if c in ds]
    df_raw = ds[avail].to_dataframe()
    df_raw.index = pd.DatetimeIndex(ds['time'].values)
    ds.close()

    # Apply pre-computed climatology if saved alongside the NetCDF [Wilks, 2011]
    clim_path = p.with_name(p.stem + '.climatology.csv')
    df_det = None
    if clim_path.exists():
        clim = pd.read_csv(str(clim_path), index_col=[0, 1])
        clim.index.names = ['__doy__', '__tod__']
        df_det = apply_climatology(
            df_raw[[c for c in pred_cols if c in df_raw.columns]].copy(),
            clim, columns=list(pred_cols), inplace=True,
        )

    return df_raw, df_det


path = st.sidebar.text_input(
    'Library XLSX path',
    value='data/merged_outages_15min_metA1only.input.library.xlsx',
)
try:
    df, feat_cols, raw_cols, shap_cols, stat_cols, means, stds, shap_base = load(path)
except FileNotFoundError:
    st.error(f'File not found: {path}')
    st.stop()
except Exception as e:
    st.error(f'Could not load file: {e}')
    st.stop()

out_mask = df['peak_customers_out'].notna()
out_df   = df[out_mask]

# ── Sidebar: filters ──────────────────────────────────────────────────────────

st.sidebar.header('Filters')

prob_range  = st.sidebar.slider('RF probability', 0.0, 1.0, (0.0, 1.0), step=0.01)
outage_only = st.sidebar.checkbox('Outage events only', value=False)

st.sidebar.subheader('Outage characteristics')
peak_max = float(out_df['peak_customers_out'].max()) if not out_df.empty else 1.0
dur_max  = float(out_df['duration_h'].max())         if not out_df.empty else 24.0

peak_range = st.sidebar.slider(
    'Peak customers out', 0.0, peak_max, (0.0, peak_max), disabled=not outage_only,
)
dur_range = st.sidebar.slider(
    'Duration (h)', 0.0, dur_max, (0.0, dur_max), disabled=not outage_only,
)

st.sidebar.subheader('Dominant SHAP feature')
shap_opts = ['(none)'] + shap_cols
shap_feat = st.sidebar.selectbox(
    'Feature',
    shap_opts,
    format_func=lambda x: x.replace('shap_', '').replace('_W16', '') if x != '(none)' else x,
)
shap_thresh = st.sidebar.slider(
    '|SHAP| ≥', 0.0, 0.5, 0.0, step=0.005,
    disabled=(shap_feat == '(none)'),
)

st.sidebar.subheader('SHAP settings')
_default_base = shap_base if not np.isnan(shap_base) else 0.5
base_val = st.sidebar.number_input(
    'Base value', value=_default_base, step=0.01, format='%.4f',
    help='Expected RF output before calibration, read from the library file.',
)

st.sidebar.subheader('Time series data')
nc_path = st.sidebar.text_input(
    'NetCDF data file',
    value='data/brec.outages_mesonet.nc',
)
outage_threshold = st.sidebar.number_input(
    'Outage threshold (customers)', value=20, min_value=0, step=1,
)

# ── Apply filters ─────────────────────────────────────────────────────────────

mask = df['outage_probability'].between(prob_range[0], prob_range[1])
if outage_only:
    mask &= out_mask
    mask &= df['peak_customers_out'].between(peak_range[0], peak_range[1])
    mask &= df['duration_h'].between(dur_range[0], dur_range[1])
if shap_feat != '(none)':
    mask &= df[shap_feat].abs() >= shap_thresh

df_filt = df[mask].sort_values('outage_probability', ascending=False)

# ── Event table ───────────────────────────────────────────────────────────────

st.markdown(f'**{len(df_filt)} events** match filters (of {len(df)} total)')

table_cols  = ['outage_probability'] + stat_cols
fmt         = {c: '{:.3f}' for c in ['outage_probability', 'duration_h', 'auc_customer_h']}
fmt['peak_customers_out'] = '{:.0f}'

ev  = st.dataframe(
    df_filt[table_cols].style.format(fmt, na_rep='—'),
    use_container_width=True,
    selection_mode='single-row',
    on_select='rerun',
)

sel = ev.selection.rows
if not sel or sel[0] >= len(df_filt):
    st.caption('Click a row to inspect the event.')
    st.stop()

row = df_filt.iloc[sel[0]]
ts  = row.name

st.subheader(f'Event: {ts.strftime("%Y-%m-%d %H:%M")}')

# ── Time series / SHAP buttons ────────────────────────────────────────────────

if 'view' not in st.session_state:
    st.session_state['view'] = None

c1, c2 = st.columns(2)
if c1.button('Time series', use_container_width=True):
    st.session_state['view'] = 'ts'
if c2.button('SHAP', use_container_width=True, disabled=not HAS_SHAP):
    st.session_state['view'] = 'shap'

view = st.session_state['view']
if view is None:
    st.stop()

# ── Time series plot ──────────────────────────────────────────────────────────

if view == 'ts':
    # Infer predictor names (original order) from feat_cols: 'relh_mean_W16' → 'relh'
    pred_names = list(dict.fromkeys(
        re.split(r'_(mean|std|grad)', c)[0] for c in feat_cols
    ))
    # Infer window half-size from _W{n} suffix, assuming symmetric pre/post split
    w_match = re.search(r'_W(\d+)', feat_cols[0]) if feat_cols else None
    w_total = int(w_match.group(1)) if w_match else 16
    pre_w = post_w = w_total // 2

    try:
        df_raw_ts, df_det_ts = load_ts_data(nc_path, tuple(pred_names), 'customers_out')
    except Exception as e:
        st.error(f'Could not load {nc_path}: {e}')
        st.stop()

    dur_h = float(row.get('duration_h', np.nan))
    t_end = ts + pd.Timedelta(hours=dur_h) if not np.isnan(dur_h) else ts
    episode = pd.Series({'t_start': ts, 't_end': t_end})

    config_ts = {
        'predictor_cols': pred_names,
        'target_col': 'customers_out',
        'outage_threshold': int(outage_threshold),
        'episode_buffer_hours': 48,
        'pre_window': pre_w,
        'post_window': post_w,
        'mode': 'binary',
    }

    # Pass empty DataFrame if no climatology found: detrended line will be omitted
    X_det = df_det_ts if df_det_ts is not None else pd.DataFrame(index=df_raw_ts.index)
    oof_pred = pd.Series([float(row['outage_probability'])], index=[ts])

    fig = plot_episode_ts(df_raw_ts, X_det, episode, config_ts, oof_pred=oof_pred)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── SHAP waterfall ────────────────────────────────────────────────────────────

elif view == 'shap':
    sv          = row[shap_cols].values.astype(float)
    fv          = row[feat_cols].values.astype(float)
    feat_labels = [c.replace('shap_', '').replace('_W16', '') for c in shap_cols]

    expl = shap_lib.Explanation(
        values      = sv,
        base_values = float(base_val),
        data        = fv,
        feature_names = feat_labels,
    )

    fig = plt.figure(figsize=(7, max(4, len(shap_cols) * 0.35 + 2)))
    shap_lib.plots.waterfall(expl, show=False)
    plt.title(f'SHAP — {ts.strftime("%Y-%m-%d %H:%M")}', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
