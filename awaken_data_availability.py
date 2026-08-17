"""
Discover all AWAKEN channels on the WDH and record available file timestamps.

Two-step approach to keep API calls manageable:
  1. Channel discovery — test each (site, instrument, zid, data_level) combination
     against a short reference window; mark as valid if the stats table returns results.
  2. Data collection  — for each valid channel, search the inventory week by week
     (weekly chunks avoid the timeout that hits monthly or full-period queries).

One CSV is written per channel as soon as its collection finishes (or as soon as
it is confirmed absent), so a crash mid-run does not lose completed channels. On
resume, channels that already have an output file are skipped.

Processing mode:
  serial   — sites processed one at a time.
  parallel — sites processed concurrently (one thread per site). The bottleneck
             here is API round-trip latency, not local compute, so threads (not
             processes/nodes) are the right tool; each site's channel files are
             independent so there is no cross-thread contention.

Output: data/awaken_data_availability/<channel>.csv  (columns: channel, date_time)
Run once (rerun to resume); build_library.py loads the per-channel files for
fast local window checks.
"""
import sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from doe_dap_dl import DAP
from tqdm import tqdm

#%% Inputs

if len(sys.argv)==1:
    sdate = '2022-09-01'
    edate = '2025-10-01'
    username = input('WDH username: ')
    password = input('WDH password: ')
    mode = (input("Processing mode [serial/parallel] (default serial): ").strip().lower()
            or 'serial')
else:
    sdate = sys.argv[1]
    edate = sys.argv[2]
    username = sys.argv[3]
    password = sys.argv[4]
    mode = sys.argv[5] if len(sys.argv) > 5 else 'serial'

if mode not in ('serial', 'parallel'):
    raise ValueError(f"mode must be 'serial' or 'parallel', got {mode!r}")

# reference window used only for channel existence check (pick a mid-campaign date)
ref_t0 = '20230101000000'
ref_t1 = '20230102000000'

sites        = ['rad', 'rad1', 'rad2', 'rmgc',
                'sa1', 'sa2', 'sa5', 'sa7', 'sb', 'sc1', 'sd', 'se', 'se36', 'sg', 'sh']
instruments  = ['aeri', 'assist', 'assist.tropoe', 'ceil', 'cup', 'dts', 'imetxq2',
                'ld', 'lidar', 'met', 'mwr', 'radar', 'sirs', 'sonic', 'tsi', 'vtower']
max_zid      = 14
data_levels  = ['00', 'a0', 'a1', 'b0', 'c0', 'c1', 'c2']

OUTPUT_DIR = Path(__file__).parent / 'data' / 'awaken_data_availability'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#%% Login

a2e = DAP('a2e.energy.gov', confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

weeks = pd.date_range(sdate, edate, freq='W-MON')
print(f"Period : {sdate} → {edate}  ({len(weeks)} weeks)  [{mode}]\n")

#%% Discover channels and collect timestamps, one site at a time

def process_site(site):
    for instrument in instruments:
        for zid in range(1, max_zid + 1):
            for data_level in data_levels:
                channel = f'awaken/{site}.{instrument}.z{zid:02d}.{data_level}'
                channel_file = OUTPUT_DIR / (channel.replace('/', '_') + '.csv')

                if channel_file.exists():
                    print(f"Skipping {channel} — output already exists", flush=True)
                    continue

                records = []

                # ── Step 1: existence check (stats, 1 day) ──────────────────
                try:
                    result = a2e.search({'Dataset': channel,
                                         'date_time': {'between': [ref_t0, ref_t1]}},
                                        table='stats')
                    if not result:
                        pd.DataFrame(columns=['channel', 'date_time']).to_csv(channel_file, index=False)
                        continue
                except Exception:
                    print(f"{channel} not found", flush=True)
                    pd.DataFrame(columns=['channel', 'date_time']).to_csv(channel_file, index=False)
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

                # ── Save this channel's results immediately ──────────────────
                channel_out = (pd.DataFrame(records, columns=['channel', 'date_time'])
                                 .drop_duplicates()
                                 .sort_values('date_time')
                                 .reset_index(drop=True))
                channel_out.to_csv(channel_file, index=False)
                print(f"  Saved → {channel_file}  ({len(channel_out)} records)", flush=True)

#%% Run

if mode == 'parallel':
    with ThreadPoolExecutor(max_workers=len(sites)) as ex:
        list(ex.map(process_site, sites))
else:
    for site in sites:
        process_site(site)

print(f"\nDone. Per-channel files in {OUTPUT_DIR}")
