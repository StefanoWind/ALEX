"""
Discover all AWAKEN channels on the WDH and record available file timestamps.

Two-step approach to keep API calls manageable:
  1. Channel discovery — test each (site, instrument, zid, data_level) combination
     against a short reference window; mark as valid if the stats table returns results.
  2. Data collection  — for each valid channel, search the inventory week by week
     (weekly chunks avoid the timeout that hits monthly or full-period queries).

Output: data/awaken_data_availability.csv  (columns: channel, date_time)
Run once; build_library.py loads the result for fast local window checks.
"""
import sys
import pandas as pd
from pathlib import Path
from doe_dap_dl import DAP
from tqdm import tqdm

#%% Inputs

if len(sys.argv)==1:
    sdate = '2022-09-01'
    edate = '2025-10-01'
    username = input('WDH username: ')
    password = input('WDH password: ')
else:
    sdate = sys.argv[1]
    edate = sys.argv[2]
    username = sys.argv[3]
    password = sys.argv[4]

# reference window used only for channel existence check (pick a mid-campaign date)
ref_t0 = '20230101000000'
ref_t1 = '20230102000000'

sites        = ['rad', 'rad1', 'rad2', 'rmgc',
                'sa1', 'sa2', 'sa5', 'sa7', 'sb', 'sc1', 'sd', 'se', 'se36', 'sg', 'sh']
instruments  = ['aeri', 'assist', 'assist.tropoe', 'ceil', 'cup', 'dts', 'imetxq2',
                'ld', 'lidar', 'met', 'mwr', 'radar', 'sirs', 'sonic', 'tsi', 'vtower']
max_zid      = 14
data_levels  = ['00', 'a0', 'a1', 'b0', 'c0', 'c1', 'c2']

OUTPUT = Path(__file__).parent / 'data' / 'awaken_data_availability.csv'

#%% Login


a2e = DAP('a2e.energy.gov', confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

weeks = pd.date_range(sdate, edate, freq='W-MON')
print(f"Period : {sdate} → {edate}  ({len(weeks)} weeks)\n")

#%% Discover channels and collect timestamps
records = []

for site in sites:
    for instrument in instruments:
        for zid in range(1, max_zid + 1):
            for data_level in data_levels:
                channel = f'awaken/{site}.{instrument}.z{zid:02d}.{data_level}'

                # ── Step 1: existence check (stats, 1 day) ──────────────────
                try:
                    result = a2e.search({'Dataset': channel,
                                         'date_time': {'between': [ref_t0, ref_t1]}},
                                        table='stats')
                    if not result:
                        continue
                except Exception:
                    print(f"{channel} not found", flush=True)
                    continue

                print(f"Found: {channel} — collecting timestamps", flush=True)

                # ── Step 2: weekly inventory search, fall back to daily ──────
                for w in tqdm(weeks, desc=channel, unit='wk'):
                    t0 = w.strftime('%Y%m%d%H%M%S')
                    t1 = (w + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)).strftime('%Y%m%d%H%M%S')
                    try:
                        inv = a2e.search({'Dataset': channel,
                                          'date_time': {'between': [t0, t1]}},
                                         table='inventory')
                        if inv:
                            df_inv = pd.DataFrame(inv)
                            if 'date_time' in df_inv.columns:
                                for dt_str in df_inv['date_time']:
                                    records.append({'channel': channel, 'date_time': dt_str})
                    except Exception:
                        tqdm.write(f"  Weekly timeout — {channel} {w.date()}, retrying daily")
                        for day in tqdm(pd.date_range(w, periods=7, freq='D'),
                                        desc=f'  {w.date()}', unit='d', leave=False):
                            d0 = day.strftime('%Y%m%d%H%M%S')
                            d1 = (day + pd.Timedelta(hours=23, minutes=59, seconds=59)).strftime('%Y%m%d%H%M%S')
                            try:
                                inv = a2e.search({'Dataset': channel,
                                                  'date_time': {'between': [d0, d1]}},
                                                 table='inventory')
                                if inv:
                                    df_inv = pd.DataFrame(inv)
                                    if 'date_time' in df_inv.columns:
                                        for dt_str in df_inv['date_time']:
                                            records.append({'channel': channel, 'date_time': dt_str})
                            except Exception as exc:
                                tqdm.write(f"  Daily timeout — {channel} {day.date()}: {exc}")

#%% Save
out = (pd.DataFrame(records)
         .drop_duplicates()
         .sort_values(['channel', 'date_time'])
         .reset_index(drop=True))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}  ({len(out)} records, {out['channel'].nunique()} channels)")
