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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from utils import plot_episode_ts, _VAR_LABELS

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 7

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


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load(path: str):
    df = pd.read_excel(path, sheet_name='Library', index_col=0)
    df.index = df.index.astype(str)

    shap_base = np.nan
    if 'shap_base_value' in df.index:
        shap_base = float(df.loc['shap_base_value', 'outage_probability'])
        df = df.drop('shap_base_value')

    outage_threshold = 20
    if 'outage_threshold' in df.index:
        outage_threshold = int(df.loc['outage_threshold', 'outage_probability'])
        df = df.drop('outage_threshold')

    df.index = pd.to_datetime(df.index)

    stat_cols = ['peak_customers_out', 'duration_h', 'auc_customer_h']
    shap_cols = [c for c in df.columns if c.startswith('shap_')]
    raw_cols  = [c for c in df.columns if c.endswith('_raw')]
    feat_cols = [c for c in df.columns
                 if c not in stat_cols + shap_cols + raw_cols + ['outage_probability']]

    df_data = pd.read_excel(path, sheet_name='Data', index_col=0)
    df_data.index = pd.to_datetime(df_data.index)

    return df, feat_cols, raw_cols, shap_cols, stat_cols, shap_base, outage_threshold, df_data


path = st.sidebar.text_input(
    'Library XLSX path',
    value='data/merged_outages_15min_metA1only.input.library.xlsx',
)
try:
    df, feat_cols, raw_cols, shap_cols, stat_cols, shap_base, outage_threshold, df_data = load(path)
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

# ── Session state init ────────────────────────────────────────────────────────

for key, default in [('prev_table_sel', []), ('sel_ts', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# Reset selection when filter widgets change (must run before scatter is drawn)
_filter_sig = (prob_range, outage_only, peak_range, dur_range, shap_feat, shap_thresh)
if 'prev_filter_sig' in st.session_state and st.session_state.prev_filter_sig != _filter_sig:
    st.session_state.sel_ts = None
st.session_state.prev_filter_sig = _filter_sig

# ── Event table ───────────────────────────────────────────────────────────────

_col_count, _col_clear = st.columns([6, 1])
_col_count.markdown(f'**{len(df_filt)} events** match filters (of {len(df)} total)')
if _col_clear.button('Clear selection', use_container_width=True):
    st.session_state.sel_ts = None
    st.session_state.prev_table_sel = []

table_cols = ['outage_probability'] + stat_cols
fmt        = {c: '{:.3f}' for c in ['outage_probability', 'duration_h', 'auc_customer_h']}
fmt['peak_customers_out'] = '{:.0f}'

ev = st.dataframe(
    df_filt[table_cols].style.format(fmt, na_rep='—'),
    use_container_width=True,
    selection_mode='single-row',
    on_select='rerun',
)
table_sel = list(ev.selection.rows) if ev.selection else []

# ── SHAP scatter (always shown for outage events in current filter) ────────────

df_out_filt = df_filt[df_filt['peak_customers_out'].notna()]
scatter_event = None

if not df_out_filt.empty and shap_cols:
    shap_v  = df_out_filt[shap_cols].values.astype(float)   # (n, n_feat)
    feat_v  = df_out_filt[feat_cols].values.astype(float)   # (n, n_feat)
    x_vals  = df_out_filt['duration_h'].values * 60          # minutes
    ts_strs = df_out_filt.index.strftime('%Y-%m-%d %H:%M:%S').tolist()

    # Optional rain indicator (skipped if column absent)
    rain_col  = next((c for c in df_out_filt.columns if re.match(r'^rain_mean_W\d+', c)), None)
    rain_vals = df_out_filt[rain_col].fillna(0).values if rain_col else None

    feat_mean = feat_v.mean(axis=0)
    feat_std  = np.where(feat_v.std(axis=0) > 0, feat_v.std(axis=0), 1.0)
    feat_norm = (feat_v - feat_mean) / feat_std

    # Max SHAP per event > 0 guard
    valid = np.max(shap_v, axis=1) > 0
    shap_v    = shap_v[valid];    feat_v    = feat_v[valid]
    feat_norm = feat_norm[valid]; x_vals    = x_vals[valid]
    ts_strs   = [ts_strs[i] for i in np.where(valid)[0]]
    if rain_vals is not None:
        rain_vals = rain_vals[valid]

    # Which event is currently selected (from table or previous scatter click)?
    # Used to highlight that event consistently across all three subplots.
    _raw_sel = st.session_state.get('sel_ts')
    try:
        sel_ts_str = pd.Timestamp(_raw_sel).strftime('%Y-%m-%d %H:%M:%S') if _raw_sel else None
    except Exception:
        sel_ts_str = None
    ts_strs_set         = set(ts_strs)
    scatter_has_sel     = sel_ts_str is not None and sel_ts_str in ts_strs_set

    rank_idx  = np.argsort(shap_v, axis=1)[:, ::-1]
    n_events  = len(ts_strs)
    n_ranks   = min(3, shap_v.shape[1])

    # Unique bases across all ranks (for legend consistency)
    all_bases = sorted({
        _base_var(feat_cols[rank_idx[i, k]])
        for i in range(n_events) for k in range(n_ranks)
    })

    fig_sc = make_subplots(
        rows=n_ranks, cols=1, shared_xaxes=True,
        subplot_titles=_RANK_LABELS[:n_ranks],
        vertical_spacing=0.07,
    )

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

                label = f'{_var_label(base)} / {ftype}'
                trace_cdata = [ts_strs[i] for i in idx_sel]
                # selectedpoints: highlight this event across ALL subplots
                # (None = no dimming; list = only those indices at full opacity)
                sel_pts = (
                    [j for j, c in enumerate(trace_cdata) if c == sel_ts_str]
                    if scatter_has_sel else None
                )
                fig_sc.add_trace(
                    go.Scatter(
                        x=[x_vals[i] for i in idx_sel],
                        y=[y_vals[i] for i in idx_sel],
                        mode='markers',
                        name=label,
                        legendgroup=f'{base}_{ftype}',
                        showlegend=(k == 0),
                        marker=dict(
                            symbol=marker_sym,
                            size=[sizes[i] for i in idx_sel],
                            color=color,
                            line=dict(color=ec, width=ew),
                            opacity=0.80,
                        ),
                        selectedpoints=sel_pts,
                        selected=dict(marker=dict(opacity=1.0)),
                        unselected=dict(marker=dict(opacity=0.15)),
                        customdata=trace_cdata,
                        hovertemplate=(
                            '<b>%{customdata}</b><br>'
                            'Duration: %{x:.0f} min<br>'
                            f'Driver: {_var_label(base)} / {ftype}<br>'
                            'z-score: %{y:.2f}<extra></extra>'
                        ),
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
        fig_sc, on_select='rerun', selection_mode='points', key='shap_scatter',
        use_container_width=True,
    )

# ── Selection resolution ──────────────────────────────────────────────────────

scatter_pts = []
try:
    if scatter_event and scatter_event.selection and scatter_event.selection.points:
        scatter_pts = scatter_event.selection.points
except (AttributeError, TypeError):
    pass

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
    pred_names = list(dict.fromkeys(
        re.split(r'_(mean|std|grad)', c)[0] for c in feat_cols
    ))
    w_match = re.search(r'_W(\d+)', feat_cols[0]) if feat_cols else None
    w_total = int(w_match.group(1)) if w_match else 16
    pre_w = post_w = w_total // 2

    det_col_map = {c: c[:-4] for c in df_data.columns if c.endswith('_det')}
    df_raw_ts = df_data[[c for c in df_data.columns if not c.endswith('_det')]]
    X_det = (df_data[[c for c in df_data.columns if c.endswith('_det')]]
             .rename(columns=det_col_map))
    if X_det.empty:
        X_det = pd.DataFrame(index=df_raw_ts.index)

    dur_h = float(row.get('duration_h', np.nan))
    t_end = ts + pd.Timedelta(hours=dur_h) if not np.isnan(dur_h) else ts
    episode = pd.Series({'t_start': ts, 't_end': t_end})

    config_ts = {
        'predictor_cols': pred_names,
        'target_col': 'customers_out',
        'outage_threshold': outage_threshold,
        'episode_buffer_hours': 48,
        'pre_window': pre_w,
        'post_window': post_w,
        'mode': 'binary',
    }

    n_panels = len(pred_names) + 1
    oof_pred = pd.Series([float(row['outage_probability'])], index=[ts])
    fig = plot_episode_ts(df_raw_ts, X_det, episode, config_ts, oof_pred=oof_pred,
                          figsize=(10, 2 * n_panels), fontsize=9)
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

    fig = plt.figure(figsize=(6, max(3, len(shap_cols) * 0.28 + 1.5)))
    shap_lib.plots.waterfall(expl, show=False)
    plt.title(f'SHAP — {ts.strftime("%Y-%m-%d %H:%M")}', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
