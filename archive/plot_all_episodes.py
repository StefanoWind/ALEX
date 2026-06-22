import os
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from utils import plot_episode_ts, _find_episodes, _merge_close_episodes
from outage_rf_events import load_data, qc_data


with open('configs/outage_rf_events.yaml') as f:
    cfg = yaml.safe_load(f)

raw_sources = cfg.get('sources', cfg.get('source'))
sources = [raw_sources] if isinstance(raw_sources, str) else list(raw_sources)
station_names = [Path(s).name.split('.')[0] for s in sources]

ts_run = datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S')
out_dir = Path(cfg['output_dir']) / f'all_episodes_{ts_run}'
os.makedirs(out_dir, exist_ok=True)

threshold = cfg['outage_threshold']
target_col = cfg['target_col']

df_raw_list = []
for src in sources:
    df = load_data(cfg, source=src)
    df = qc_data(df, cfg)
    df_raw_list.append(df)

dt = df_raw_list[0].index.to_series().diff().median()
dt_h = dt.total_seconds() / 3600

# Union of episodes from all counties; merge those within one timestep (same storm)
all_episodes = []
for df_raw in df_raw_list:
    all_episodes.extend(_find_episodes(df_raw[target_col], threshold))

all_episodes.sort(key=lambda x: x[0])
all_episodes = _merge_close_episodes(all_episodes, merge_gap=dt)
print(f"Found {len(all_episodes)} episodes to plot.")

empty_X = pd.DataFrame()
label1 = station_names[0] if len(station_names) > 0 else ''
label2 = station_names[1] if len(station_names) > 1 else ''
df_raw2 = df_raw_list[1] if len(df_raw_list) > 1 else None

for t_start, t_end in all_episodes:
    auc_parts = []
    for name, df_raw in zip(station_names, df_raw_list):
        seg = df_raw[target_col].loc[t_start:t_end].fillna(0).clip(lower=0)
        auc = float(seg.sum() * dt_h)
        auc_parts.append(f'{name}={auc:.0f}')
    title_extra = 'AUC [cust·h]: ' + ' | '.join(auc_parts)

    ep = pd.Series({'t_start': t_start, 't_end': t_end,
                    't_center': t_start + (t_end - t_start) / 2})

    plot_episode_ts(
        df_raw_list[0], empty_X, ep, cfg, out_dir,
        df_raw2=df_raw2, X2=None,
        label1=label1, label2=label2,
        title_extra=title_extra,
    )

print(f"Done. {len(all_episodes)} plots saved to {out_dir}")
