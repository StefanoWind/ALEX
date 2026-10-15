'''
Local Streamlit dashboard for the ALEX event library.
Run with: streamlit run dashboard.py
'''

import re
import io
import shutil
import threading
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
    _get_nexrad_conn,
    _resolve_or_bulk_download_scans,
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
_FTYPE_LABELS = {
    'mean': 'mean', 'std': 'std', 'grad_mean': 'gradient mean', 'grad_std': 'gradient std',
}
_TABLE_LABELS = {
    'timestamp':'Time (UTC)',
    'outage_probability': 'Outage probability',
    'peak_customers_out': 'Peak customers out',
    'duration_h': 'Duration [h]',
    'auc_customer_h': 'AUC [customer·h]',
}
_AWAKEN_SITE_LABELS = {
    'sc1': 'Site C1a',
    'rmgc': 'Rolling Meadows Golf Course',
    'rad1': 'Radar Site 1',
    'rad2': 'Radar Site 2',
    'rad': 'Radar Sites',
}
_AWAKEN_INSTRUMENT_LABELS = {
    'aeri': 'AERI',
    'assist': 'ASSIST',
    'assist.tropoe': 'TROPoe retrieval',
    'ceil': 'Ceilometer',
    'ld': 'Laser disdrometer',
    'lidar': 'Doppler lidar',
    'met': 'Meteorological station',
    'mwr': 'Microwave radiometer',
    'radar': 'Radar',
    'sonic': 'Sonic anemometer',
    'vtower': 'Virtual tower',
}

_WDH_URL = 'https://wdh.energy.gov/ds/'
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
    return ' '.join(re.sub(r'\s*\[.*?\]', '', raw).split())


def _var_units(base):
    m = re.search(r'\[(.*?)\]', _VAR_LABELS.get(base, ''))
    return m.group(1) if m else ''


def _feat_plain(name):
    return f'{_var_label(_base_var(name))} {_FTYPE_LABELS.get(_feat_type(name), "")}'.strip()


def _table_col_label(col):
    if col in _TABLE_LABELS:
        return _TABLE_LABELS[col]
    m = re.match(r'^rmse_([a-zA-Z]+)(?:_f(\d+))?$', col)
    if not m:
        return col
    var, lead = m.groups()
    name = _var_label(var)
    source = f'HRRR f{lead}' if lead else 'HRRR analysis'
    units = _var_units(var)
    label = f'RMSE of {name[0].lower() + name[1:]} ({source})'
    return f'{label} [{units}]' if units else label


def _awaken_site_label(site):
    if site in _AWAKEN_SITE_LABELS:
        return _AWAKEN_SITE_LABELS[site]
    m = re.match(r'^s([a-z]\d*)$', site)
    return f'Site {m.group(1).upper()}' if m else site


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

    try:
        df_avail = pd.read_excel(path, sheet_name='wdh_avail', index_col=0)
        df_avail.index = pd.to_datetime(df_avail.index)
    except Exception:
        df_avail = pd.DataFrame()

    fcst_sfxs = sorted(
        {m.group(1) for col in df_hrrr.columns for m in [re.search(r'(_f\d+)$', col)] if m},
        key=lambda s: int(s[2:]),
    )

    return (df, feat_cols, raw_cols, shap_cols, stat_cols, rmse_cols,
            shap_base, outage_threshold, df_data, df_hrrr, fcst_sfxs, df_avail, lat_meta, lon_meta)


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


def _prefetch_nearest_nexrad_scans(event_ts_iso: str, radar_dir: Path, radar_id: str = 'KVNX', n: int = 5):
    """
    Background download of the n NEXRAD scans nearest the event window center.
    Runs in a plain thread (not st.cache_data — that ties to the Streamlit
    script-run context, which doesn't exist off the main thread) and only
    touches pure I/O helpers, so it's safe to overlap with the main script.
    Populates utils._DOWNLOADED_SCAN_PATHS, which render_nexrad_scan_png's
    download_plot_nexrad call also consults, so a later click on the NEXRAD
    view finds the files already local.
    """
    try:
        t_center = pd.Timestamp(event_ts_iso)
        browse_start = t_center - pd.Timedelta(hours=2)
        browse_end = t_center + pd.Timedelta(hours=4)
        scans = list_nexrad_scans_in_range(
            browse_start, browse_end, radar_id=radar_id, cadence_minutes=None,
        )
        if not scans:
            return
        nearest = sorted(scans, key=lambda s: abs(_as_utc(s.scan_time) - _as_utc(t_center)))[:n]
        conn = _get_nexrad_conn()
        _resolve_or_bulk_download_scans(nearest, Path(radar_dir), conn)
    except Exception:
        pass  # best-effort warmup; failures surface normally when the user opens NEXRAD


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
    ('prev_scatter_cdata', None),
    ('sel_ts', None),
    ('scatter_gen', 0),
    ('table_gen', 0),
    ('view', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# Neither the Plotly nor the data_editor selection can be written from code, so
# both widgets are re-created under a fresh key whenever sel_ts changes; they then
# render the selection from sel_ts instead of their own stale frontend state.
def _refresh_selection_widgets():
    st.session_state.scatter_gen += 1
    st.session_state.table_gen += 1
    st.session_state.prev_scatter_cdata = None

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

    st.markdown('### Input source map')
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
            width='stretch',
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
    _refresh_selection_widgets()
    st.session_state.view = None
    st.session_state.prev_source_key = st.session_state.selected_source_key

try:
    (df, feat_cols, raw_cols, shap_cols, stat_cols, rmse_cols,
    shap_base, outage_threshold, df_data, df_hrrr, fcst_sfxs, df_avail,
    lat_meta, lon_meta) = load(path, file_sig=_file_mtime(path))
except FileNotFoundError:
    st.error(f'File not found: {path}')
    st.stop()
except Exception as e:
    st.error(f'Could not load file: {e}')
    st.stop()

out_mask = df['peak_customers_out'].notna()
out_df   = df[out_mask]

# ── Sidebar: HRRR horizon ───────────────────────────────────────────────────────

st.sidebar.header('HRRR horizon')
if not df_hrrr.empty:
    horizon_opts = ['anl'] + fcst_sfxs
    selected_horizon = st.sidebar.selectbox(
        'Horizon', horizon_opts,
        format_func=lambda s: 'Analysis' if s == 'anl' else f'+{s[2:]}h forecast',
        help='Used for the time series overlay and for the RMSE column shown in the event table.',
    )
else:
    selected_horizon = None
    st.sidebar.caption('No HRRR data in this library.')

# ── Sidebar: filters ──────────────────────────────────────────────────────────

st.sidebar.header('Filters')

prob_range  = st.sidebar.slider('Outage probability', 0.0, 1.0, (0.0, 1.0), step=0.01)
outage_only = st.sidebar.checkbox('Outage events only', value=False)

st.sidebar.subheader('Outage characteristics')
peak_max = int(np.ceil(out_df['peak_customers_out'].max())) if not out_df.empty else 1
dur_max  = float(out_df['duration_h'].max())         if not out_df.empty else 24.0

peak_range = st.sidebar.slider(
    'Peak customers out', 0, peak_max, (0, peak_max), disabled=not outage_only,
)
dur_range = st.sidebar.slider(
    'Duration [h]', 0.0, dur_max, (0.0, dur_max), disabled=not outage_only,
)

base_val = shap_base if not np.isnan(shap_base) else 0.5

# ── Apply filters ─────────────────────────────────────────────────────────────

mask = df['outage_probability'].between(prob_range[0], prob_range[1])
if outage_only:
    mask &= out_mask
    mask &= df['peak_customers_out'].between(peak_range[0], peak_range[1])
    mask &= df['duration_h'].between(dur_range[0], dur_range[1])

df_filt = df[mask].sort_values('outage_probability', ascending=False)

# Reset selection when filter widgets change.
# Also bump scatter_gen so the Plotly widget gets a fresh key → clears its
# internal frontend selection state (avoids stale highlighted markers).
_filter_sig = (prob_range, outage_only, peak_range, dur_range)
if 'prev_filter_sig' in st.session_state and st.session_state.prev_filter_sig != _filter_sig:
    st.session_state.sel_ts = None
    _refresh_selection_widgets()
st.session_state.prev_filter_sig = _filter_sig

# ── Header row: event count + Clear button ────────────────────────────────────

_col_count, _col_clear = st.columns([6, 1])
_col_count.markdown(f'**{len(df_filt)} events** match filters (of {len(df)} total)')
if _col_clear.button('Clear selection', width='stretch'):
    st.session_state.sel_ts = None
    _refresh_selection_widgets()

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

                label       = f'{_var_label(base)} {_FTYPE_LABELS[ftype]}'
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
                            f'Driver: {label}<br>'
                            'z-score: %{y:.2f}<extra></extra>'
                        ),
                        **sel_kw,
                    ),
                    row=k + 1, col=1,
                )

        fig_sc.update_yaxes(
            title_text='z-score', title_font_size=14,
            tickfont_size=14, row=k + 1, col=1,
        )

    fig_sc.update_xaxes(
        title_text='Duration [min]', title_font_size=14, tickfont_size=14,
        row=n_ranks, col=1,
    )
    fig_sc.update_layout(
        height=550,
        margin=dict(l=80, r=220, t=30, b=50),
        legend=dict(orientation='v', x=1.02, y=1, xanchor='left', font_size=14),
        font=dict(size=8),
    )
    fig_sc.update_annotations(font_size=16)

    scatter_event = st.plotly_chart(
        fig_sc, on_select='rerun', selection_mode='points',
        key=f'shap_scatter_{st.session_state.scatter_gen}',
        width='stretch',
    )

scatter_pts = []
try:
    if scatter_event and scatter_event.selection and scatter_event.selection.points:
        scatter_pts = scatter_event.selection.points
except (AttributeError, TypeError):
    pass

# The Plotly selection persists across reruns, so only a click on a different
# point than last time counts as a new scatter selection.
_scatter_cdata = scatter_pts[0].get('customdata') if scatter_pts else None
if _scatter_cdata and _scatter_cdata != st.session_state.prev_scatter_cdata:
    st.session_state.prev_scatter_cdata = _scatter_cdata
    st.session_state.sel_ts = pd.Timestamp(_scatter_cdata).isoformat()
    st.session_state.table_gen += 1
    st.rerun()

# ── Event table ───────────────────────────────────────────────────────────────

# Show only the RMSE columns for the horizon selected in the sidebar (analysis or one forecast lead).
if selected_horizon == 'anl':
    rmse_cols_shown = [c for c in rmse_cols if not re.search(r'_f\d+$', c)]
elif selected_horizon:
    rmse_cols_shown = [c for c in rmse_cols if c.endswith(selected_horizon)]
else:
    rmse_cols_shown = []

table_cols = ['outage_probability'] + stat_cols + rmse_cols_shown
fmt        = {c: '{:.3f}' for c in ['outage_probability', 'duration_h', 'auc_customer_h']}
fmt['peak_customers_out'] = '{:.0f}'
for _rc in rmse_cols_shown:
    fmt[_rc] = '{:.3f}'
fmt = {_table_col_label(c): f for c, f in fmt.items()}


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


_cur_sel = pd.Timestamp(st.session_state.sel_ts) if st.session_state.get('sel_ts') else None
df_table = df_filt[table_cols].rename(columns=_table_col_label).rename_axis('Time (UTC)')
data_labels = list(df_table.columns)
df_table.insert(0, 'Select', df_table.index == _cur_sel)

ev_df = st.data_editor(
    df_table.style.format(fmt, na_rep='—').apply(_zscore_colors, axis=0, subset=data_labels),
    width='stretch',
    disabled=['_index'] + data_labels,
    column_config={'Select': st.column_config.CheckboxColumn('Select', width='small')},
    key=f'event_table_{st.session_state.table_gen}',
)

# ── Selection resolution ──────────────────────────────────────────────────────

_checked = ev_df.index[ev_df['Select']]
_newly_checked = [t for t in _checked if t != _cur_sel]
if _newly_checked:
    st.session_state.sel_ts = _newly_checked[0].isoformat()
elif _cur_sel is not None and _cur_sel in df_table.index and _cur_sel not in _checked:
    st.session_state.sel_ts = None
if (pd.Timestamp(st.session_state.sel_ts) if st.session_state.sel_ts else None) != _cur_sel:
    _refresh_selection_widgets()
    st.rerun()

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
    st.caption('Check a table row or click a scatter point to inspect the event.')
    st.stop()

st.subheader(f'Event: {ts.strftime("%Y-%m-%d %H:%M")} UTC')

# Warm the NEXRAD cache for this event as soon as it's selected, before the user
# clicks NEXRAD. Streamlit reruns the whole script on every interaction, so a
# per-event session_state flag keeps this from re-launching a thread each rerun.
_prefetch_flag = f'nexrad_prefetched_{ts.isoformat()}'
if not st.session_state.get(_prefetch_flag):
    st.session_state[_prefetch_flag] = True
    threading.Thread(
        target=_prefetch_nearest_nexrad_scans,
        args=(ts.isoformat(), _BASE_DIR / 'data' / 'nexrad_cache'),
        daemon=True,
    ).start()

# ── Time series / SHAP / NEXRAD buttons ────────────────────────────────────────────────

if 'view' not in st.session_state:
    st.session_state['view'] = None

c1, c2, c3, c4, c5 = st.columns(5)
if c1.button('Time series', width='stretch'):
    st.session_state['view'] = 'ts'
if c2.button('SHAP', width='stretch', disabled=not HAS_SHAP):
    st.session_state['view'] = 'shap'
if c3.button('NEXRAD', width='stretch'):
    st.session_state['view'] = 'nexrad'
if c4.button('AWAKEN Data', width='stretch'):
    st.session_state['view'] = 'awaken'
if c5.button('Histograms', width='stretch'):
    st.session_state['view'] = 'hist'

view = st.session_state['view']
if view is None:
    st.stop()

# ── Time series plot ──────────────────────────────────────────────────────────

if view == 'ts':
    pred_names = list(dict.fromkeys(
        re.split(r'_(mean|std|grad)', c)[0] for c in feat_cols
    ))
    available_preds = [c for c in pred_names if c in df_data.columns and c != 'customers_out']
    # wdir isn't its own panel — it's overlaid as arrows on the wind-speed panel below.

    # Requested panel order (top to bottom), then any remaining predictors.
    preferred_order = ['wspd', 'aavi', 'tair', 'relh', 'pres']
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
    
    st.markdown('#### Event time series')

    n_panels = len(pred_names_ts) + 1
    oof_pred = pd.Series([float(row['outage_probability'])], index=[ts])
    with plt.style.context('default'):
        fig = plot_episode_ts(df_raw_ts, X_det, episode, config_ts, oof_pred=oof_pred,
                              figsize=(10, 2 * n_panels), fontsize=9)

    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
    fig.axes[-1].set_ylabel('Customers out of power', fontsize=9)

    buffer_h = pd.Timedelta(hours=config_ts['episode_buffer_hours'])
    t0_plot  = ts - buffer_h
    t1_plot  = t_end + buffer_h

    if not df_hrrr.empty and selected_horizon:
        is_anl = selected_horizon == 'anl'
        fcst_lead = None if is_anl else selected_horizon[2:]   # '18' from '_f18'

        for i, col in enumerate(pred_names_ts):
            ax = fig.axes[i]
            if is_anl:
                if col in df_hrrr.columns:
                    hw = df_hrrr[col].loc[t0_plot:t1_plot].dropna()
                    ax.scatter(hw.index, hw.values, color='firebrick', s=14,
                               zorder=3, alpha=0.85, label='HRRR analysis')
            else:
                hrrr_fcol = f'{col}{selected_horizon}'
                if hrrr_fcol in df_hrrr.columns:
                    hw = df_hrrr[hrrr_fcol].loc[t0_plot:t1_plot].dropna()
                    ax.scatter(hw.index, hw.values, color='darkorange', s=14,
                               marker='^', zorder=3, alpha=0.85,
                               label=f'HRRR f{fcst_lead}')

    for ax in fig.axes:
        for patch in ax.patches:
            patch.set_alpha(0.3)

    # Rebuild legend on first panel to include HRRR entries and capitalize labels
    ax0 = fig.axes[0]
    h, l = ax0.get_legend_handles_labels()
    with plt.style.context('default'):
        ax0.legend(h, [s[0].upper() + s[1:] for s in l], loc='upper left', fontsize=9, framealpha=0.7)
    ax0.set_title(f"Predicted outage probability = {float(row['outage_probability']):.2f}", fontsize=9)

    # Wind direction as small arrows on top of the wind-speed panel, rather than
    # its own panel — direction only matters in the context of speed/timing.
    wspd_col = next((c for c in ('wspd', 'aavi') if c in pred_names_ts), None)
    if wspd_col is not None and 'wdir' in df_raw_ts.columns:
        ax_wspd = fig.axes[pred_names_ts.index(wspd_col)]
        wdir_win = df_raw_ts['wdir'].loc[t0_plot:t1_plot].dropna()
        wspd_win = df_raw_ts[wspd_col].loc[t0_plot:t1_plot]

        n_arrows = 24
        step = max(1, len(wdir_win) // n_arrows)
        wdir_sub = wdir_win.iloc[::step]
        # Anchor each arrow to the wind-speed value at the same time (nan → skip).
        y_sub = wspd_win.reindex(wdir_sub.index, method='nearest', tolerance=pd.Timedelta(minutes=30))
        valid = y_sub.notna()
        wdir_sub, y_sub = wdir_sub[valid], y_sub[valid]

        if not wdir_sub.empty:
            theta = np.radians(wdir_sub.values)
            u, v = -np.sin(theta), -np.cos(theta)   # points where the wind is blowing toward
            ax_wspd.quiver(
                wdir_sub.index, y_sub.values, u, v,
                angles='uv', scale_units='inches', scale=6, width=0.004,
                pivot='mid', color='b', alpha=0.8, zorder=5,
            )

    st.pyplot(fig, width='stretch')
    plt.close(fig)

# ── SHAP waterfall ────────────────────────────────────────────────────────────

elif view == 'shap':
    sv          = row[shap_cols].values.astype(float)
    fv          = row[feat_cols].values.astype(float)
    feat_labels = [_feat_plain(c.replace('shap_', '', 1)) for c in shap_cols]

    expl = shap_lib.Explanation(
        values        = sv,
        base_values   = float(base_val),
        data          = fv,
        feature_names = feat_labels,
    )
    
    st.markdown('#### SHAP waterfall')
    
    with plt.style.context('default'):
        fig = plt.figure(figsize=(6, max(3, len(shap_cols) * 0.28 + 1.5)))
        shap_lib.plots.waterfall(expl, show=False)
        for txt in fig.findobj(matplotlib.text.Text):
            txt.set_fontsize(txt.get_fontsize() * 0.75)
        plt.tight_layout()

    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
    shap_col, _ = st.columns([4, 1])
    shap_col.pyplot(fig, width='stretch')
    plt.close(fig)

# ── NEXRAD reflectivity ────────────────────────────────────────────────────────────

if view == 'nexrad':

    marker_lat = selected_site_lat if np.isfinite(selected_site_lat) else lat_meta
    marker_lon = selected_site_lon if np.isfinite(selected_site_lon) else lon_meta
    marker_lat_val = float(marker_lat) if np.isfinite(marker_lat) else None
    marker_lon_val = float(marker_lon) if np.isfinite(marker_lon) else None

    st.markdown('#### Radar reflectivity')

    cache_col, _ = st.columns([1, 3])
    if cache_col.button('Clear NEXRAD cache', width='stretch', key='nexrad_clear_cache'):
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
        if c_prev.button('< Prev', width='stretch', disabled=(idx <= 0), key='nexrad_scan_prev'):
            st.session_state['nexrad_scan_idx'] = idx - 1
            st.rerun()
        if c_next.button('Next >', width='stretch', disabled=(idx >= len(scan_times) - 1), key='nexrad_scan_next'):
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
            st.image(nexrad_png, width='stretch')
        except Exception as e:
            st.error(f'Unable to load NEXRAD plot: {e}')

# ── AWAKEN data availability ────────────────────────────────────────────────────

if view == 'awaken':
    st.markdown('#### AWAKEN channels available in this event window')

    if df_avail.empty:
        st.warning("No 'wdh_avail' sheet found in this library — run build_library.py "
                   "with awaken_data_availability.py output present.")
    elif ts not in df_avail.index:
        st.warning('No availability record for this event window.')
    else:
        avail_row = df_avail.loc[ts].fillna(False).astype(bool)
        avail_channels = sorted(avail_row.index[avail_row].tolist())
        st.caption(f'{len(avail_channels)} of {len(avail_row)} channels available.')

        if avail_channels:
            def _channel_row(ch):
                parts = ch.split('/')[-1].split('.')
                instrument = '.'.join(parts[1:-2])
                return {
                    'Site': _awaken_site_label(parts[0]),
                    'Instrument': _AWAKEN_INSTRUMENT_LABELS.get(instrument, instrument),
                    'Data Level': parts[-1],
                    'Channel': _WDH_URL + ch,
                }

            st.dataframe(
                pd.DataFrame([_channel_row(ch) for ch in avail_channels]),
                width='stretch', hide_index=True,
                column_config={
                    'Channel': st.column_config.LinkColumn(
                        'Channel', display_text=r'https://wdh\.energy\.gov/ds/(.*)',
                    ),
                },
            )
        else:
            st.info('No AWAKEN channels flagged available for this event window.')

# ── Feature histograms ──────────────────────────────────────────────────────────

if view == 'hist':
    st.markdown('#### Feature distributions — event window vs. full dataset')

    _HIST_VARS = ['wspd', 'aavi', 'pres', 'relh']
    hist_vars = [v for v in _HIST_VARS if v in df_data.columns]

    if not hist_vars:
        st.warning('None of the expected weather features (wind speed, wind variability '
                   'index, pressure, relative humidity) are present in this library.')
    else:
        w_match = re.search(r'_W(\d+)', feat_cols[0]) if feat_cols else None
        w_total = int(w_match.group(1)) if w_match else 16
        pre_w = post_w = w_total // 2
        dt = df_data.index.to_series().diff().median()
        w_start = ts - pre_w * dt
        w_end   = ts + post_w * dt
        st.caption(f'Event window: {w_start.strftime("%Y-%m-%d %H:%M")} to '
                  f'{w_end.strftime("%Y-%m-%d %H:%M")} UTC')

        ncols = 2
        nrows = (len(hist_vars) + ncols - 1) // ncols
        with plt.style.context('default'):
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2 * nrows), squeeze=False)
        axes_flat = axes.ravel()

        for ax, var in zip(axes_flat, hist_vars):
            full_vals = df_data[var].dropna()
            ax.hist(full_vals, bins=40, color='steelblue', alpha=0.8)

            win_vals = df_data[var][(df_data.index >= w_start) & (df_data.index < w_end)].dropna()
            if not win_vals.empty:
                lo, hi = float(win_vals.min()), float(win_vals.max())
                if lo == hi:
                    span = float(full_vals.max() - full_vals.min())
                    pad = span * 0.01 if span > 0 else 0.5
                    lo, hi = lo - pad, hi + pad
                ax.axvspan(lo, hi, color='red', alpha=0.3, zorder=3)
                ax.axvline(float(win_vals.mean()), color='red', linewidth=1.5, zorder=4)

            ax.set_xlabel(' '.join(_VAR_LABELS.get(var, var).split()), fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.grid(alpha=0.3)

        for ax in axes_flat[len(hist_vars):]:
            ax.set_visible(False)

        legend_patch = plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3)
        legend_line = plt.Line2D([0], [0], color='red', linewidth=1.5)
        with plt.style.context('default'):
            fig.legend([legend_patch, legend_line], ['Event window range', 'Event window mean'],
                       loc='upper right', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        fig.patch.set_facecolor('white')
        for ax in fig.axes:
            ax.set_facecolor('white')

        st.pyplot(fig, width='stretch')
        plt.close(fig)