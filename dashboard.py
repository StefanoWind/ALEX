'''
Local Streamlit dashboard for the ALEX event library.
Run with: streamlit run dashboard.py
'''

import re
import io
import shutil
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from utils import (
    plot_episode_ts,
    download_plot_nexrad,
    list_nexrad_scans_in_range,
    _VAR_LABELS,
)

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'

st.set_page_config(page_title='ALEX Event Library', layout='wide')
st.title('ALEX — Outage Event Library')

# ── Constants ─────────────────────────────────────────────────────────────────

_COLORS = {
    'aavi': '#DAA520', 'pres': '#555555', 'rain': '#999999',
    'relh': '#2ca02c', 'srad': '#ff7f0e', 'tair': '#d62728',
    'wmax': '#1f77b4', 'wspd': '#1f77b4',
}
_PLOTLY_MARKERS = {
    'mean': 'circle', 'std': 'star', 'grad_mean': 'square', 'grad_std': 'diamond',
}
_RANK_LABELS = ['1st dominant driver', '2nd dominant driver', '3rd dominant driver']
_BASE_DIR = Path(__file__).resolve().parent
_SOURCE_DATA_DIR = _BASE_DIR / 'data' / 'merged_outages_gf_15min'
_DEFAULT_LIBRARY_PATH = str(_SOURCE_DATA_DIR / 'merged_outages_15min_metA1only.input.library.xlsx')
_TERRAIN_TILE_URLS = [
    f'https://{sub}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png' for sub in ('a', 'b', 'c')
]
_TERRAIN_ATTRIBUTION = 'Map data: OpenStreetMap contributors, SRTM | Map display: OpenTopoMap (CC-BY-SA)'


@st.cache_data
def load_turbine_points(xlsx_path: str, file_sig: float = 0.0):
    """Read turbine locations from a library.xlsx's 'Wind farms' sheet."""
    try:
        df_turb = pd.read_excel(xlsx_path, sheet_name='Wind farms')
    except Exception:
        return pd.DataFrame(columns=['farm', 'turbine_id', 'latitude', 'longitude'])

    df_turb['latitude'] = pd.to_numeric(df_turb['latitude'], errors='coerce')
    df_turb['longitude'] = pd.to_numeric(df_turb['longitude'], errors='coerce')
    df_turb = df_turb[
        df_turb['latitude'].between(-90, 90) &
        df_turb['longitude'].between(-180, 180)
    ].reset_index(drop=True)
    return df_turb


def _format_source_label(source_key: str, fallback_label: str):
    m = re.match(r'^met([a-z]\d*).*$' , str(source_key).lower())
    if m:
        return f"Site {m.group(1).upper()}"
    return fallback_label


def _feat_type(name):
    if '_grad_std_'  in name: return 'grad_std'
    if '_grad_mean_' in name: return 'grad_mean'
    if '_std_'       in name: return 'std'
    if '_mean_'      in name: return 'mean'
    return 'other'


def _base_var(name):
    return re.sub(r'_(grad_std|grad_mean|std|mean)_W\d+.*$', '', name)


def _var_label(base):
    raw = _VAR_LABELS.get(base, base)
    return re.sub(r'\s*\[.*?\]', '', raw).replace('\n', ' ').strip()


def _source_key_from_filename(filename: str):
    stem = Path(filename).stem
    m = re.search(r'(met[\w\-]+?only)', stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return re.sub(r'\.input\.library$', '', stem, flags=re.IGNORECASE).lower()


@st.cache_data
def _read_xlsx_coords(xlsx_path: str):
    """Read station latitude/longitude from a library.xlsx's dedicated 'Site' sheet."""
    lat = np.nan
    lon = np.nan
    try:
        site_df = pd.read_excel(xlsx_path, sheet_name='Site')
        if not site_df.empty:
            if 'latitude' in site_df.columns:
                lat = float(site_df['latitude'].iloc[0])
            if 'longitude' in site_df.columns:
                lon = float(site_df['longitude'].iloc[0])
    except Exception:
        return np.nan, np.nan

    if np.isfinite(lon) and lon > 180:
        lon = lon - 360
    return lat, lon


@st.cache_data
def discover_library_sources(data_dir: str, data_sig: float = 0.0):
    data_dir_p = Path(data_dir)
    files = sorted(data_dir_p.glob('*.library.xlsx'))
    all_nc = list(data_dir_p.glob('*.nc'))
    src_rows = []
    for fp in files:
        source_stem = re.sub(r'\.input\.library\.xlsx$', '', fp.name, flags=re.IGNORECASE)
        candidate_nc = data_dir_p / f'{source_stem}.nc'
        nc_path = None
        if candidate_nc.exists():
            nc_path = candidate_nc
        else:
            matches = [p for p in all_nc if _source_key_from_filename(p.name) == _source_key_from_filename(fp.name)]
            if len(matches) == 1:
                nc_path = matches[0]

        lat, lon = _read_xlsx_coords(str(fp))

        label = _source_key_from_filename(fp.name)
        if nc_path is not None:
            try:
                with xr.open_dataset(str(nc_path)) as ds_nc:
                    label = str(ds_nc.attrs.get('station', label))
            except Exception:
                pass

        src_rows.append({
            'source_key': _source_key_from_filename(fp.name),
            'file_name': fp.name,
            'path': str(fp).replace('\\', '/'),
            'label': label,
            'latitude': lat,
            'longitude': lon,
            'nc_path': str(nc_path).replace('\\', '/') if nc_path is not None else np.nan,
        })
    src_df = pd.DataFrame(src_rows)
    if src_df.empty:
        return src_df

    src_df['label'] = src_df['label'].fillna(src_df['source_key'])
    src_df['label'] = src_df.apply(
        lambda r: _format_source_label(r['source_key'], r['label']), axis=1
    )
    src_df['latitude'] = pd.to_numeric(src_df['latitude'], errors='coerce')
    src_df['longitude'] = pd.to_numeric(src_df['longitude'], errors='coerce')
    src_df['has_coords'] = src_df['latitude'].notna() & src_df['longitude'].notna()
    src_df = src_df.sort_values(['source_key', 'file_name']).reset_index(drop=True)
    return src_df


def _data_dir_signature(data_dir: str):
    p = Path(data_dir)
    if not p.exists():
        return 0.0
    latest = p.stat().st_mtime
    for f in p.glob('*'):
        if f.is_file() and f.suffix.lower() in {'.xlsx', '.nc'}:
            latest = max(latest, f.stat().st_mtime)
    return float(latest)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load(path: str, file_sig: float = 0.0):
    df = pd.read_excel(path, sheet_name='Library', index_col=0)
    df.index = df.index.astype(str)

    def _meta_value(meta_key, default=np.nan):
        if meta_key in df.index:
            try:
                return float(df.loc[meta_key, 'outage_probability'])
            except Exception:
                return default
        return default

    shap_base = _meta_value('shap_base_value', np.nan)
    outage_threshold = int(_meta_value('outage_threshold', 20))
    lat_meta, lon_meta = _read_xlsx_coords(path)

    _meta_rows = [k for k in ['shap_base_value', 'outage_threshold', 'latitude', 'longitude'] if k in df.index]
    if _meta_rows:
        df = df.drop(_meta_rows)

    df.index = pd.to_datetime(df.index)

    stat_cols = ['peak_customers_out', 'duration_h', 'auc_customer_h']
    shap_cols = [c for c in df.columns if c.startswith('shap_')]
    raw_cols  = [c for c in df.columns if c.endswith('_raw')]
    rmse_cols = [c for c in df.columns if c.startswith('rmse_')]
    feat_cols = [c for c in df.columns
                 if c not in stat_cols + shap_cols + raw_cols + rmse_cols + ['outage_probability']]

    df_data = pd.read_excel(path, sheet_name='Data', index_col=0)
    df_data.index = pd.to_datetime(df_data.index)

    try:
        df_hrrr = pd.read_excel(path, sheet_name='HRRR data', index_col=0)
        df_hrrr.index = pd.to_datetime(df_hrrr.index)
    except Exception:
        df_hrrr = pd.DataFrame()

    fcst_sfx = None
    for col in df_hrrr.columns:
        m = re.search(r'(_f\d+)$', col)
        if m:
            fcst_sfx = m.group(1)
            break

    return (df, feat_cols, raw_cols, shap_cols, stat_cols, rmse_cols,
            shap_base, outage_threshold, df_data, df_hrrr, fcst_sfx, lat_meta, lon_meta)


def _file_mtime(path: str):
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0


def _as_utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


@st.cache_data(ttl=3600, show_spinner=False)
def list_nexrad_scan_times(start_iso: str, end_iso: str, radar_id: str = 'KVNX') -> list[str]:
    """List available NEXRAD scan timestamps in a window (metadata only, no download)."""
    scans = list_nexrad_scans_in_range(
        pd.Timestamp(start_iso), pd.Timestamp(end_iso), radar_id=radar_id, cadence_minutes=None,
    )
    return [pd.Timestamp(s.scan_time).isoformat() for s in scans]


@st.cache_data(ttl=3600, show_spinner=False)
def render_nexrad_scan_png(scan_time_iso: str,
                           marker_lat: float | None,
                           marker_lon: float | None,
                           site_label: str | None,
                           radar_id: str = 'KVNX') -> bytes:
    """Render and cache a single NEXRAD scan as PNG bytes; downloads only this one scan."""
    scan_time = pd.Timestamp(scan_time_iso)
    episode = pd.Series({'t_start': scan_time, 't_end': scan_time})
    fig = download_plot_nexrad(
        episode,
        radar_id=radar_id,
        radar_dir=_BASE_DIR / 'data' / 'nexrad_cache',
        map_resolution='10m',
        site_lat=marker_lat,
        site_lon=marker_lon,
        site_label=site_label,
        show_site_label=True,
    )
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format='png',
        dpi=150,
        bbox_inches='tight',
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return buf.getvalue()


for key, default in [
    ('selected_source_key', None),
    ('prev_source_key', None),
    ('prev_table_sel', []),
    ('sel_ts', None),
    ('scatter_gen', 0),
    ('view', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

sources_df = discover_library_sources(str(_SOURCE_DATA_DIR), data_sig=_data_dir_signature(str(_SOURCE_DATA_DIR)))
if not sources_df.empty:
    _turbine_source_path = sources_df.iloc[0]['path']
    turbine_df = load_turbine_points(_turbine_source_path, file_sig=_file_mtime(_turbine_source_path))
else:
    turbine_df = pd.DataFrame(columns=['farm', 'turbine_id', 'latitude', 'longitude'])

if not sources_df.empty:
    valid_keys = set(sources_df['source_key'])
    sel_key = st.session_state.get('selected_source_key')
    if sel_key not in valid_keys:
        st.session_state.selected_source_key = sources_df.iloc[0]['source_key']

    if st.session_state.prev_source_key is None:
        st.session_state.prev_source_key = st.session_state.selected_source_key

    st.markdown('### Input Source Map')
    map_df = sources_df.copy()
    map_df['plot_lat'] = map_df['latitude']
    map_df['plot_lon'] = map_df['longitude']

    if not map_df['has_coords'].all():
        missing_idx = map_df.index[~map_df['has_coords']].tolist()
        if map_df['has_coords'].any():
            lat0 = float(map_df.loc[map_df['has_coords'], 'latitude'].mean())
            lon0 = float(map_df.loc[map_df['has_coords'], 'longitude'].mean())
        else:
            lat0, lon0 = 36.4, -97.6

        for offset, idx in enumerate(missing_idx):
            map_df.loc[idx, 'plot_lat'] = lat0 + 0.20 * offset
            map_df.loc[idx, 'plot_lon'] = lon0 + 0.30 * offset

        st.info(
            'Some sources are missing latitude/longitude attributes in matched .nc files. '
            'Using temporary fallback locations for map selection.'
        )

    map_event = None

    if not map_df.empty:
        selected_key = st.session_state.selected_source_key
        selected_pts = [
            i for i, k in enumerate(map_df['source_key'].tolist()) if k == selected_key
        ]

        lon_span = float(map_df['plot_lon'].max() - map_df['plot_lon'].min())
        lat_span = float(map_df['plot_lat'].max() - map_df['plot_lat'].min())
        span = max(lon_span, lat_span)
        if span < 0.3:
            zoom = 9
        elif span < 0.8:
            zoom = 8
        elif span < 1.5:
            zoom = 7
        elif span < 3.0:
            zoom = 6
        else:
            zoom = 5

        site_lon = map_df['plot_lon'].tolist()
        site_lat = map_df['plot_lat'].tolist()
        site_text = map_df['label'].tolist()
        site_customdata = map_df['source_key'].tolist()

        site_marker_trace_kwargs = dict(
            lon=site_lon,
            lat=site_lat,
            mode='markers+text',
            marker=dict(size=11, color='#00A6FF', opacity=0.95),
            customdata=site_customdata,
            text=site_text,
            textposition='top center',
            textfont=dict(color='#111111', size=12, weight="bold"),
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Map lat: %{lat:.3f}<br>'
                'Map lon: %{lon:.3f}<extra></extra>'
            ),
            selectedpoints=selected_pts,
            selected=dict(marker=dict(size=14, color='#FF3B30', opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.45)),
            showlegend=False,
            name='Sites',
        )

        center_cfg = dict(
            lat=float(map_df['plot_lat'].mean()),
            lon=float(map_df['plot_lon'].mean()),
        )

        map_layers = [
            dict(
                sourcetype='raster',
                type='raster',
                source=_TERRAIN_TILE_URLS,
                sourceattribution=_TERRAIN_ATTRIBUTION,
                below='traces',
                opacity=1.0,
            )
        ]

        map_style = 'white-bg'

        if hasattr(go, 'Scattermap'):
            fig_map = go.Figure()
            if not turbine_df.empty:
                fig_map.add_trace(
                    go.Scattermap(
                        lon=turbine_df['longitude'].tolist(),
                        lat=turbine_df['latitude'].tolist(),
                        mode='markers',
                        name='Turbines',
                        marker=dict(size=5, color='#111111', opacity=0.75),
                        text=[
                            f"{farm} {tid}".strip()
                            for farm, tid in zip(turbine_df['farm'], turbine_df['turbine_id'])
                        ],
                        hovertemplate=(
                            '<b>%{text}</b><br>'
                            'Lat: %{lat:.4f}<br>'
                            'Lon: %{lon:.4f}<extra></extra>'
                        ),
                        showlegend=True,
                    )
                )
            fig_map.add_trace(go.Scattermap(**site_marker_trace_kwargs))
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=320,
                map=dict(style=map_style, center=center_cfg, zoom=zoom, layers=map_layers),
                showlegend=False,
            )
        else:
            fig_map = go.Figure()
            if not turbine_df.empty:
                fig_map.add_trace(
                    go.Scattermapbox(
                        lon=turbine_df['longitude'].tolist(),
                        lat=turbine_df['latitude'].tolist(),
                        mode='markers',
                        name='Turbines',
                        marker=dict(size=5, color="#646464", opacity=0.75),
                        text=[
                            f"{farm} {tid}".strip()
                            for farm, tid in zip(turbine_df['farm'], turbine_df['turbine_id'])
                        ],
                        hovertemplate=(
                            '<b>%{text}</b><br>'
                            'Lat: %{lat:.4f}<br>'
                            'Lon: %{lon:.4f}<extra></extra>'
                        ),
                        showlegend=True,
                    )
                )
            fig_map.add_trace(go.Scattermapbox(**site_marker_trace_kwargs))
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=320,
                mapbox=dict(style=map_style, center=center_cfg, zoom=zoom, layers=map_layers),
                showlegend=False,
            )
        map_event = st.plotly_chart(
            fig_map,
            key='source_map',
            use_container_width=True,
            on_select='rerun',
            selection_mode='points',
        )

        if turbine_df.empty:
            st.info("No turbine locations found in the active library.xlsx's 'Wind farms' sheet.")
    map_pts = []
    try:
        if map_event and map_event.selection and map_event.selection.points:
            map_pts = map_event.selection.points
    except (AttributeError, TypeError):
        pass

    if map_pts:
        clicked_key = map_pts[0].get('customdata')
        if clicked_key and clicked_key != st.session_state.selected_source_key:
            st.session_state.selected_source_key = clicked_key
            st.rerun()

    source_label = sources_df.loc[
        sources_df['source_key'] == st.session_state.selected_source_key, 'label'
    ].iloc[0]
    st.caption(f'Active source: {source_label}')

    source_row = sources_df[sources_df['source_key'] == st.session_state.selected_source_key].iloc[0]
    selected_site_lat = float(source_row['latitude']) if pd.notna(source_row.get('latitude', np.nan)) else np.nan
    selected_site_lon = float(source_row['longitude']) if pd.notna(source_row.get('longitude', np.nan)) else np.nan
    selected_site_label = str(source_label)
    auto_path = source_row['path']

    st.sidebar.subheader('Input source')
    st.sidebar.text_input('Selected library path', value=auto_path, disabled=True)
    manual_override = st.sidebar.checkbox('Manual path override', value=False)
    if manual_override:
        path = st.sidebar.text_input('Library XLSX path (manual)', value=auto_path)
    else:
        path = auto_path
else:
    st.warning(
        f'No *.library.xlsx files were found in {_SOURCE_DATA_DIR}. '
        'Using manual path input fallback.'
    )
    path = st.sidebar.text_input('Library XLSX path', value=_DEFAULT_LIBRARY_PATH)
    selected_site_lat = np.nan
    selected_site_lon = np.nan
    selected_site_label = None

if st.session_state.prev_source_key != st.session_state.selected_source_key:
    st.session_state.sel_ts = None
    st.session_state.prev_table_sel = []
    st.session_state.scatter_gen += 1
    st.session_state.view = None
    st.session_state.prev_source_key = st.session_state.selected_source_key
    st.session_state.pop('event_table', None)

try:
    (df, feat_cols, raw_cols, shap_cols, stat_cols, rmse_cols,
    shap_base, outage_threshold, df_data, df_hrrr, fcst_sfx,
    lat_meta, lon_meta) = load(path, file_sig=_file_mtime(path))
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
    'Feature', shap_opts,
    format_func=lambda x: x.replace('shap_', '').replace('_W16', '') if x != '(none)' else x,
)
shap_thresh = st.sidebar.slider(
    '|SHAP| ≥', 0.0, 0.5, 0.0, step=0.005, disabled=(shap_feat == '(none)'),
)

st.sidebar.subheader('SHAP settings')
_default_base = shap_base if not np.isnan(shap_base) else 0.5
base_val = st.sidebar.number_input(
    'Base value', value=_default_base, step=0.01, format='%.4f',
    help='Expected RF output before calibration, read from the library file.',
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

# Reset selection when filter widgets change.
# Also bump scatter_gen so the Plotly widget gets a fresh key → clears its
# internal frontend selection state (avoids stale highlighted markers).
_filter_sig = (prob_range, outage_only, peak_range, dur_range, shap_feat, shap_thresh)
if 'prev_filter_sig' in st.session_state and st.session_state.prev_filter_sig != _filter_sig:
    st.session_state.sel_ts = None
    st.session_state.scatter_gen += 1
    st.session_state.pop('event_table', None)
st.session_state.prev_filter_sig = _filter_sig

# ── Header row: event count + Clear button ────────────────────────────────────

_col_count, _col_clear = st.columns([6, 1])
_col_count.markdown(f'**{len(df_filt)} events** match filters (of {len(df)} total)')
if _col_clear.button('Clear selection', use_container_width=True):
    st.session_state.sel_ts = None
    st.session_state.prev_table_sel = []
    st.session_state.scatter_gen += 1  # force fresh Plotly widget → no stale highlights
    st.session_state.pop('event_table', None)

# ── SHAP scatter (outage events only, shown above table) ──────────────────────

df_out_filt = df_filt[df_filt['peak_customers_out'].notna()]
scatter_event = None

if not df_out_filt.empty and shap_cols:
    shap_v  = df_out_filt[shap_cols].values.astype(float)
    feat_v  = df_out_filt[feat_cols].values.astype(float)
    x_vals  = df_out_filt['duration_h'].values * 60          # minutes
    ts_strs = df_out_filt.index.strftime('%Y-%m-%d %H:%M:%S').tolist()

    rain_col  = next((c for c in df_out_filt.columns if re.match(r'^rain_mean_W\d+', c)), None)
    rain_vals = df_out_filt[rain_col].fillna(0).values if rain_col else None

    feat_mean = feat_v.mean(axis=0)
    feat_std  = np.where(feat_v.std(axis=0) > 0, feat_v.std(axis=0), 1.0)
    feat_norm = (feat_v - feat_mean) / feat_std

    valid = np.max(shap_v, axis=1) > 0
    shap_v    = shap_v[valid];    feat_v    = feat_v[valid]
    feat_norm = feat_norm[valid]; x_vals    = x_vals[valid]
    ts_strs   = [ts_strs[i] for i in np.where(valid)[0]]
    if rain_vals is not None:
        rain_vals = rain_vals[valid]

    _raw_sel = st.session_state.get('sel_ts')
    try:
        sel_ts_str = pd.Timestamp(_raw_sel).strftime('%Y-%m-%d %H:%M:%S') if _raw_sel else None
    except Exception:
        sel_ts_str = None
    ts_strs_set     = set(ts_strs)
    scatter_has_sel = sel_ts_str is not None and sel_ts_str in ts_strs_set

    rank_idx = np.argsort(shap_v, axis=1)[:, ::-1]
    n_events = len(ts_strs)
    n_ranks  = min(3, shap_v.shape[1])

    all_bases = sorted({
        _base_var(feat_cols[rank_idx[i, k]])
        for i in range(n_events) for k in range(n_ranks)
    })

    fig_sc = make_subplots(
        rows=n_ranks, cols=1, shared_xaxes=True,
        subplot_titles=_RANK_LABELS[:n_ranks],
        vertical_spacing=0.07,
    )

    legend_shown = set()
    for k in range(n_ranks):
        dom_idx   = rank_idx[:, k]
        dom_names = [feat_cols[dom_idx[i]] for i in range(n_events)]
        dom_types = [_feat_type(f) for f in dom_names]
        dom_bases = [_base_var(f)  for f in dom_names]

        y_vals   = np.array([feat_norm[i, dom_idx[i]] for i in range(n_events)])
        dom_shap = shap_v[np.arange(n_events), dom_idx]
        sizes    = np.clip(6 + 80 * dom_shap, 6, 50).tolist()

        for base in all_bases:
            color = _COLORS.get(base, '#888888')
            for ftype, marker_sym in _PLOTLY_MARKERS.items():
                idx_sel = [i for i, (b, t) in enumerate(zip(dom_bases, dom_types))
                           if b == base and t == ftype]
                if not idx_sel:
                    continue

                if rain_vals is not None:
                    ec = ['green' if rain_vals[i] > 0 else '#333' for i in idx_sel]
                    ew = [2.0    if rain_vals[i] > 0 else 0.5    for i in idx_sel]
                else:
                    ec, ew = '#333333', 0.5

                label       = f'{_var_label(base)} / {ftype}'
                trace_cdata = [ts_strs[i] for i in idx_sel]

                group_key   = (base, ftype)
                show_legend = group_key not in legend_shown
                if show_legend:
                    legend_shown.add(group_key)

                if scatter_has_sel:
                    sel_pts = [j for j, c in enumerate(trace_cdata) if c == sel_ts_str]
                    sel_kw  = dict(
                        selectedpoints=sel_pts,
                        selected=dict(marker=dict(opacity=1.0)),
                        unselected=dict(marker=dict(opacity=0.15)),
                    )
                else:
                    # No active selection: omit selected/unselected so Plotly
                    # renders all markers at the base opacity without any dimming.
                    sel_kw = {}

                fig_sc.add_trace(
                    go.Scatter(
                        x=[x_vals[i] for i in idx_sel],
                        y=[y_vals[i] for i in idx_sel],
                        mode='markers',
                        name=label,
                        legendgroup=f'{base}_{ftype}',
                        showlegend=show_legend,
                        marker=dict(
                            symbol=marker_sym,
                            size=[sizes[i] for i in idx_sel],
                            color=color,
                            line=dict(color=ec, width=ew),
                            opacity=0.80,
                        ),
                        customdata=trace_cdata,
                        hovertemplate=(
                            '<b>%{customdata}</b><br>'
                            'Duration: %{x:.0f} min<br>'
                            f'Driver: {_var_label(base)} / {ftype}<br>'
                            'z-score: %{y:.2f}<extra></extra>'
                        ),
                        **sel_kw,
                    ),
                    row=k + 1, col=1,
                )

        fig_sc.update_yaxes(
            title_text=f'{_RANK_LABELS[k]} (z-score)', title_font_size=8,
            tickfont_size=8, row=k + 1, col=1,
        )

    fig_sc.update_xaxes(
        title_text='Duration (min)', title_font_size=8, tickfont_size=8,
        row=n_ranks, col=1,
    )
    fig_sc.update_layout(
        height=550,
        margin=dict(l=60, r=160, t=30, b=40),
        legend=dict(orientation='v', x=1.02, y=1, xanchor='left', font_size=8),
        font=dict(size=8),
    )

    scatter_event = st.plotly_chart(
        fig_sc, on_select='rerun', selection_mode='points',
        key=f'shap_scatter_{st.session_state.scatter_gen}',
        use_container_width=True,
    )

scatter_pts = []
try:
    if scatter_event and scatter_event.selection and scatter_event.selection.points:
        scatter_pts = scatter_event.selection.points
except (AttributeError, TypeError):
    pass

# Streamlit does not allow programmatically writing a dataframe widget's
# native selection state, so the scatter → table link is done visually via
# row highlighting instead: whichever event is "current" (latest scatter
# click, falling back to the last table selection) gets its row shaded.
_pending_cdata = scatter_pts[0].get('customdata') if scatter_pts else None
_highlight_ts = _pending_cdata or st.session_state.get('sel_ts')

# ── Event table ───────────────────────────────────────────────────────────────

table_cols = ['outage_probability'] + stat_cols + rmse_cols
fmt        = {c: '{:.3f}' for c in ['outage_probability', 'duration_h', 'auc_customer_h']}
fmt['peak_customers_out'] = '{:.0f}'
for _rc in rmse_cols:
    fmt[_rc] = '{:.3f}'


_ZSCORE_CMAP = plt.get_cmap('coolwarm')


def _zscore_colors(col):
    """Color each cell by (value - median) / std, blue (low) to red (high)."""
    med, std = col.median(), col.std()
    if not np.isfinite(std) or std == 0:
        return [''] * len(col)
    z = (col - med) / std
    max_abs = np.nanmax(np.abs(z.values))
    max_abs = max_abs if np.isfinite(max_abs) and max_abs > 0 else 1.0
    norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
    styles = []
    for v in z:
        if pd.isna(v):
            styles.append('')
        else:
            r, g, b, _ = _ZSCORE_CMAP(norm(v))
            styles.append(f'background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.85)')
    return styles


def _highlight_selected_row(row):
    is_sel = _highlight_ts is not None and row.name == pd.Timestamp(_highlight_ts)
    return ['background-color: #ffcc66' if is_sel else ''] * len(row)


ev = st.dataframe(
    df_filt[table_cols].style.format(fmt, na_rep='—')
        .apply(_zscore_colors, axis=0)
        .apply(_highlight_selected_row, axis=1),
    use_container_width=True,
    selection_mode='single-row',
    on_select='rerun',
    key='event_table',
)
table_sel = list(ev.selection.rows) if ev.selection else []

# ── Selection resolution ──────────────────────────────────────────────────────

table_changed = (table_sel != st.session_state.prev_table_sel)

if table_changed:
    if table_sel and table_sel[0] < len(df_filt):
        st.session_state.sel_ts = df_filt.index[table_sel[0]].isoformat()
    else:
        st.session_state.sel_ts = None
    st.session_state.prev_table_sel = table_sel
elif scatter_pts:
    cdata = scatter_pts[0].get('customdata')
    if cdata:
        changed = (cdata != st.session_state.get('sel_ts'))
        st.session_state.sel_ts = cdata
        st.session_state.prev_table_sel = table_sel
        if changed:
            st.rerun()
    else:
        st.session_state.prev_table_sel = table_sel

# Resolve current row
row, ts = None, None
sel_ts_str = st.session_state.get('sel_ts')
if sel_ts_str:
    try:
        candidate = pd.Timestamp(sel_ts_str)
        if candidate in df_filt.index:
            row = df_filt.loc[candidate]
            ts  = candidate
        else:
            st.session_state.sel_ts = None  # stale after filter change
    except Exception:
        pass

if row is None:
    st.caption('Click a table row or scatter point to inspect the event.')
    st.stop()

st.subheader(f'Event: {ts.strftime("%Y-%m-%d %H:%M")}')

# ── Time series / SHAP / NEXRAD buttons ────────────────────────────────────────────────

if 'view' not in st.session_state:
    st.session_state['view'] = None

c1, c2, c3 = st.columns(3)
if c1.button('Time series', use_container_width=True):
    st.session_state['view'] = 'ts'
if c2.button('SHAP', use_container_width=True, disabled=not HAS_SHAP):
    st.session_state['view'] = 'shap'
if c3.button('NEXRAD', use_container_width=True):
    st.session_state['view'] = 'nexrad'

view = st.session_state['view']
if view is None:
    st.stop()

# ── Time series plot ──────────────────────────────────────────────────────────

if view == 'ts':
    pred_names = list(dict.fromkeys(
        re.split(r'_(mean|std|grad)', c)[0] for c in feat_cols
    ))
    available_preds = [c for c in pred_names if c in df_data.columns and c != 'customers_out']
    if 'wdir' in df_data.columns and 'wdir' not in available_preds:
        available_preds.append('wdir')

    # Requested panel order (top to bottom), then any remaining predictors.
    preferred_order = ['wspd', 'wdir', 'aavi', 'tair', 'relh', 'pres']
    pred_names_ts = [c for c in preferred_order if c in available_preds]
    pred_names_ts += [c for c in available_preds if c not in pred_names_ts]
    w_match = re.search(r'_W(\d+)', feat_cols[0]) if feat_cols else None
    w_total = int(w_match.group(1)) if w_match else 16
    pre_w = post_w = w_total // 2

    df_raw_ts = df_data[[c for c in df_data.columns if not c.endswith('_det')]]
    X_det = pd.DataFrame(index=df_raw_ts.index)  # detrended overlay disabled

    dur_h = float(row.get('duration_h', np.nan))
    t_end = ts + pd.Timedelta(hours=dur_h) if not np.isnan(dur_h) else ts
    episode = pd.Series({'t_start': ts, 't_end': t_end})

    config_ts = {
        'predictor_cols': pred_names_ts,
        'target_col': 'customers_out',
        'outage_threshold': outage_threshold,
        'episode_buffer_hours': 48,
        'pre_window': pre_w,
        'post_window': post_w,
        'mode': 'binary',
    }

    n_panels = len(pred_names_ts) + 1
    oof_pred = pd.Series([float(row['outage_probability'])], index=[ts])
    with plt.style.context('default'):
        fig = plot_episode_ts(df_raw_ts, X_det, episode, config_ts, oof_pred=oof_pred,
                              figsize=(10, 2 * n_panels), fontsize=9)

    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
    fig.axes[-1].set_ylabel('Customers out of power', fontsize=9)

    if 'wdir' in pred_names_ts:
        ax_wdir = fig.axes[pred_names_ts.index('wdir')]
        ax_wdir.set_ylim(0, 360)
        ax_wdir.set_yticks([0, 90, 180, 270, 360])
        ax_wdir.set_yticklabels(['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)', 'N (360°)'])
        ax_wdir.set_ylabel('Wind direction', fontsize=9)

    buffer_h = pd.Timedelta(hours=config_ts['episode_buffer_hours'])
    t0_plot  = ts - buffer_h
    t1_plot  = t_end + buffer_h

    if not df_hrrr.empty:
        fcst_lead = fcst_sfx[2:] if fcst_sfx else None   # '18' from '_f18'

        for i, col in enumerate(pred_names_ts):
            ax = fig.axes[i]
            if col in df_hrrr.columns:
                hw = df_hrrr[col].loc[t0_plot:t1_plot].dropna()
                ax.scatter(hw.index, hw.values, color='firebrick', s=14,
                           zorder=3, alpha=0.85, label='HRRR anl')
            if fcst_sfx:
                hrrr_fcol = f'{col}{fcst_sfx}'
                if hrrr_fcol in df_hrrr.columns:
                    hw = df_hrrr[hrrr_fcol].loc[t0_plot:t1_plot].dropna()
                    ax.scatter(hw.index, hw.values, color='darkorange', s=14,
                               marker='^', zorder=3, alpha=0.85,
                               label=f'HRRR f{fcst_lead}h')

        # Refresh legend on first panel to include HRRR entries
        ax0 = fig.axes[0]
        h, l = ax0.get_legend_handles_labels()
        ax0.legend(h, l, loc='upper left', fontsize=9, framealpha=0.7)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── SHAP waterfall ────────────────────────────────────────────────────────────

elif view == 'shap':
    sv          = row[shap_cols].values.astype(float)
    fv          = row[feat_cols].values.astype(float)
    feat_labels = [c.replace('shap_', '').replace('_W16', '') for c in shap_cols]

    expl = shap_lib.Explanation(
        values        = sv,
        base_values   = float(base_val),
        data          = fv,
        feature_names = feat_labels,
    )

    with plt.style.context('default'):
        fig = plt.figure(figsize=(6, max(3, len(shap_cols) * 0.28 + 1.5)))
        shap_lib.plots.waterfall(expl, show=False)
        plt.title(f'SHAP — {ts.strftime("%Y-%m-%d %H:%M")}', fontsize=8)
        plt.tight_layout()

    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── NEXRAD reflectivity ────────────────────────────────────────────────────────────

if view == 'nexrad':

    marker_lat = selected_site_lat if np.isfinite(selected_site_lat) else lat_meta
    marker_lon = selected_site_lon if np.isfinite(selected_site_lon) else lon_meta
    marker_lat_val = float(marker_lat) if np.isfinite(marker_lat) else None
    marker_lon_val = float(marker_lon) if np.isfinite(marker_lon) else None

    st.markdown('#### Reflectivity')

    cache_col, _ = st.columns([1, 3])
    if cache_col.button('Clear NEXRAD cache', use_container_width=True, key='nexrad_clear_cache'):
        cache_dir = _BASE_DIR / 'data' / 'nexrad_cache'
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        render_nexrad_scan_png.clear()
        list_nexrad_scan_times.clear()
        st.session_state.pop('nexrad_scan_idx', None)
        st.success(f"Cleared cache: {cache_dir.name}")

    browse_start = ts - pd.Timedelta(hours=2)
    browse_end = ts + pd.Timedelta(hours=4)

    try:
        scan_times = list_nexrad_scan_times(
            browse_start.isoformat(), browse_end.isoformat(), radar_id='KVNX',
        )
    except Exception as e:
        scan_times = []
        st.error(f'Unable to list NEXRAD scans: {e}')

    if not scan_times:
        st.warning('No NEXRAD scans found for this window.')
    else:
        current_event_ts = ts.isoformat()
        if st.session_state.get('nexrad_scan_event_ts') != current_event_ts:
            nearest_idx = min(
                range(len(scan_times)),
                key=lambda i: abs(_as_utc(scan_times[i]) - _as_utc(ts)),
            )
            st.session_state['nexrad_scan_idx'] = nearest_idx
            st.session_state['nexrad_scan_event_ts'] = current_event_ts

        idx = st.session_state.get('nexrad_scan_idx', 0)
        idx = max(0, min(idx, len(scan_times) - 1))

        c_prev, c_label, c_next = st.columns([1, 3, 1])
        if c_prev.button('< Prev', use_container_width=True, disabled=(idx <= 0), key='nexrad_scan_prev'):
            st.session_state['nexrad_scan_idx'] = idx - 1
            st.rerun()
        if c_next.button('Next >', use_container_width=True, disabled=(idx >= len(scan_times) - 1), key='nexrad_scan_next'):
            st.session_state['nexrad_scan_idx'] = idx + 1
            st.rerun()

        scan_time = _as_utc(scan_times[idx])
        c_label.markdown(
            f"<div style='text-align:center; padding-top:0.4em'>Scan {idx + 1} / {len(scan_times)} "
            f"&nbsp;—&nbsp; {scan_time.strftime('%Y-%m-%d %H:%M UTC')}</div>",
            unsafe_allow_html=True,
        )

        try:
            with st.spinner('Loading NEXRAD reflectivity...'):
                nexrad_png = render_nexrad_scan_png(
                    scan_times[idx],
                    marker_lat_val,
                    marker_lon_val,
                    selected_site_label,
                )
            st.image(nexrad_png, use_container_width=True)
        except Exception as e:
            st.error(f'Unable to load NEXRAD plot: {e}')