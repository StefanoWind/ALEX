import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
from utils import _VAR_LABELS, build_event_matrix

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 150
plt.close('all')

counties = {
    'Garfield (BREC)': 'data/brec.outages_mesonet.nc',
    'Noble (REDR)':    'data/redr.outages_mesonet.nc',
}
colors = {'Garfield (BREC)': 'steelblue', 'Noble (REDR)': 'firebrick'}

PREDICTORS = ['relh', 'tair', 'aavi', 'wmax', 'rain', 'pres', 'srad']
TARGET = 'customers_out'
OUTAGE_THRESHOLD = 37

out_dir = Path('results')
out_dir.mkdir(exist_ok=True)


def load_county(path: str) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    ds['aavi'] = ds['wssd'] / ds['wssd'].mean() * ds['wdsd'] / ds['wdsd'].mean()
    cols = [TARGET] + PREDICTORS
    df = ds[cols].to_dataframe()
    df.index = pd.DatetimeIndex(ds.time.values)
    return df


# ── Load ──────────────────────────────────────────────────────────────────────
data = {name: load_county(path) for name, path in counties.items()}

# Aligned (inner join on common time range) for correlation
aligned = pd.concat(
    [df.add_suffix(f'__{name}') for name, df in data.items()],
    axis=1,
    join='inner',
)

# ── Plot 1: time series per variable ─────────────────────────────────────────
all_vars = [TARGET] + PREDICTORS
years = list(range(2015, 2026))

for var in all_vars:
    var_label = _VAR_LABELS.get(var, var)

    fig, axes = plt.subplots(len(years), 1, figsize=(16, 2.2 * len(years)),
                             sharex=False)
    fig.suptitle(var_label, fontsize=13)

    legend_drawn = False
    any_visible = False

    for row_i, yr in enumerate(years):
        ax = axes[row_i]
        yr_data = {}
        for name, df in data.items():
            mask = df.index.year == yr
            if var in df.columns:
                yr_data[name] = df.loc[mask, var]
            else:
                yr_data[name] = pd.Series(dtype=float)

        # Skip subplot if both counties are all-NaN for this year
        all_nan = all(s.isna().all() for s in yr_data.values())
        if all_nan:
            ax.set_visible(False)
            continue

        any_visible = True
        for name, s in yr_data.items():
            ax.plot(s.index, s.values, color=colors[name], alpha=0.7, lw=0.5,
                    label=name if not legend_drawn else '_')

        if var == TARGET:
            ax.axhline(OUTAGE_THRESHOLD, color='black', lw=1, ls='--')

        if not legend_drawn:
            ax.legend(fontsize=9, loc='upper right')
            legend_drawn = True

        ax.set_ylabel(str(yr), fontsize=9)
        ax.set_xlim(pd.Timestamp(f'{yr}-01-01'), pd.Timestamp(f'{yr}-12-31'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        # Show x-tick labels only on the bottom visible subplot (handled after loop)
        ax.tick_params(labelbottom=False)

    # Enable x-tick labels on the last visible subplot
    for row_i in range(len(years) - 1, -1, -1):
        ax = axes[row_i]
        if ax.get_visible():
            ax.tick_params(labelbottom=True)
            break

    save_path = out_dir / f'comparison_ts_{var}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {save_path}')


# ── Plot 2: Spearman correlation bar chart (county A vs county B per variable) ─
# [von Storch & Zwiers, 1999]
county_names = list(counties.keys())
corr_vars = [TARGET] + PREDICTORS
corr_r = {}
for var in corr_vars:
    col_a = f'{var}__{county_names[0]}'
    col_b = f'{var}__{county_names[1]}'
    valid = aligned[[col_a, col_b]].dropna()
    r, _ = stats.spearmanr(valid[col_a], valid[col_b])
    corr_r[var] = r

corr_series = pd.Series(corr_r, name='spearman_r').reindex(corr_vars)

print(f'\nSpearman correlation ({county_names[0]} vs {county_names[1]}):')
print(corr_series.to_string())

n_vars = len(corr_vars)
y_pos = np.arange(n_vars)

fig, ax = plt.subplots(figsize=(7, 0.55 * n_vars + 2))
ax.barh(y_pos, corr_series.values, color='steelblue', alpha=0.85)
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([_VAR_LABELS.get(v, v) for v in corr_vars])
ax.set_xlabel(f'Spearman r  ({county_names[0]} vs {county_names[1]})')
ax.set_xlim(-1, 1)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

save_path = out_dir / 'comparison_correlation.png'
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Saved → {save_path}')

# ── Outage event overlap analysis ───────────────────────────────────────────
OUTAGE_MIN_DURATION = 16    # timesteps
OUTAGE_MIN_PEAK     = 200   # customers
PRE_WINDOW          = 8     # timesteps
POST_WINDOW         = 8     # timesteps
MODE                = 'binary'

events_by_county = {}
for name, df in data.items():
    df_preds_empty = pd.DataFrame(index=df.index)
    X, y, episode_ends, _ = build_event_matrix(
        df_preds_empty,
        df[TARGET],
        threshold=OUTAGE_THRESHOLD,
        mode=MODE,
        pre_window=PRE_WINDOW,
        post_window=POST_WINDOW,
        min_duration=OUTAGE_MIN_DURATION,
        min_peak=OUTAGE_MIN_PEAK,
        non_outages_ratio=0,
        random_state=42,
    )
    events_by_county[name] = [(ts, episode_ends[ts]) for ts in X.index if ts in episode_ends]

dt = data[county_names[0]].index.to_series().diff().median()
pre_dt  = PRE_WINDOW  * dt
post_dt = POST_WINDOW * dt

def _overlaps(t_start_a, t_end_a, t_start_b, t_end_b, pre_dt, post_dt):
    
    return (t_end_a + post_dt > t_start_b - pre_dt and
            t_start_a - pre_dt < t_end_b + post_dt)


evs_a = events_by_county[county_names[0]]
evs_b = events_by_county[county_names[1]]

n_a = len(evs_a)
n_b = len(evs_b)

# Build the full set of overlapping pairs (symmetric by construction)
overlap_pairs = [
    (i, j)
    for i, (sa, ea) in enumerate(evs_a)
    for j, (sb, eb) in enumerate(evs_b)
    if _overlaps(sa, ea, sb, eb, pre_dt, post_dt)
]
n_pairs = len(overlap_pairs)

# A events / B events involved in at least one overlapping pair
a_matched = len({i for i, _ in overlap_pairs})
b_matched = len({j for _, j in overlap_pairs})

print(f'\n── Outage event overlap ({county_names[0]} vs {county_names[1]}) ──')
print(f'  Overlapping pairs:           {n_pairs}')
print(f'  {county_names[0]}: {n_a} events,  {a_matched} involved in ≥1 overlap ({100*a_matched/n_a:.0f}%),  {n_a - a_matched} non-overlapping')
print(f'  {county_names[1]}: {n_b} events,  {b_matched} involved in ≥1 overlap ({100*b_matched/n_b:.0f}%),  {n_b - b_matched} non-overlapping')

# ── Assign event group IDs via union-find on the overlap graph ───────────────
parent = {}

def _find(x):
    if x not in parent:
        parent[x] = x
    if parent[x] != x:
        parent[x] = _find(parent[x])
    return parent[x]

def _union(x, y):
    px, py = _find(x), _find(y)
    if px != py:
        parent[px] = py

# Initialise all nodes
for name, evs in events_by_county.items():
    for ts, _ in evs:
        _find((name, ts))

# Add edges for overlapping pairs
for i, (sa, ea) in enumerate(evs_a):
    for j, (sb, eb) in enumerate(evs_b):
        if _overlaps(sa, ea, sb, eb, pre_dt, post_dt):
            _union((county_names[0], sa), (county_names[1], sb))

# Map roots to integer IDs
root_to_id = {}
next_id = 0
for name in counties:
    for ts, _ in events_by_county[name]:
        root = _find((name, ts))
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1

print(f'\n  {next_id} event groups ({sum(len(v) for v in events_by_county.values())} total outage events)')
