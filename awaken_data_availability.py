"""
Discover all AWAKEN channels on the WDH and record available file timestamps.

Two-step approach to keep API calls manageable:
  1. Channel discovery — test each (site, instrument, zid, data_level) combination
     against a short reference window; mark as valid if the stats table returns results.
  2. Data collection  — for each valid channel, search the inventory week by week
     (weekly chunks avoid the timeout that hits monthly or full-period queries).

Every search (discovery and collection) is restricted server-side to file_type in
`formats`, so files in extensions we can't/don't want to read never count toward
availability.

A CSV is written per channel only if it has real inventory entries, as soon as
its collection finishes, so a crash mid-run does not lose completed channels.
On resume, channels that already have an output file are skipped; channels with
no confirmed data (never found, or found but empty) are re-checked on the next
run since that check is a single cheap API call.

Rather than listing every individual file timestamp (which can run into the
hundreds of thousands for high-frequency instruments like radar), each file
is assumed to cover the hours from its start time through start + duration,
where duration is estimated per search (week, or day on fallback) as the
median gap between consecutive file start times in that search's results —
e.g. ~1h for hourly files, ~24h for daily files. This avoids the false-negative
of a multi-hour file being flagged available only in its start hour. date_time
marks every hour covered this way by at least one file.

Processing mode:
  serial   — sites processed one at a time.
  parallel — sites processed concurrently (one thread per site). The bottleneck
             here is API round-trip latency, not local compute, so threads (not
             processes/nodes) are the right tool; each site's channel files are
             independent so there is no cross-thread contention.

Output: data/awaken_data_availability/<channel>.csv  (columns: channel, date_time)
date_time is the start of each hour flagged available (a file starts in it, or
an earlier file's estimated duration covers it) — not individual file times.
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

sites        = ['sa1','rad', 'rad1', 'rad2', 'rmgc',
                'sa1', 'sa2', 'sa5', 'sa7', 'sb', 'sc1', 'sd', 'se', 'se36', 'sg', 'sh']
instruments  = ['met','aeri', 'assist', 'assist.tropoe', 'ceil', 'cup', 'dts', 'imetxq2',
                'ld', 'lidar', 'met', 'mwr', 'radar', 'sirs', 'sonic', 'tsi', 'vtower']
max_zid      = 14
data_levels  = ['b0','00', 'a0', 'a1', 'b0', 'c0', 'c1', 'c2']

formats=['nc','cdf','txt','dat','csv','hpl']

OUTPUT_DIR = Path(__file__).parent / 'data' / 'awaken_data_availability'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#%% Login

a2e = DAP('a2e.energy.gov', confirm_downloads=False)
a2e.setup_cert_auth(username=username, password=password)

weeks = pd.date_range(sdate, edate, freq='W-MON')
print(f"Period : {sdate} → {edate}  ({len(weeks)} weeks)  [{mode}]\n")

#%% Discover channels and collect timestamps, one site at a time

def _record_hours(inv: list, hour_set: set) -> None:
    """
    Flag every hour each file in this search likely covers, not just its start
    hour. Files don't report their own duration, so it's estimated as the
    median gap between consecutive file start times returned by this same
    search (e.g. ~1h for hourly files, ~24h for daily files) — otherwise a
    daily file would only be flagged available in the single hour it starts.
    """
    df_inv = pd.DataFrame(inv)
    if 'date_time' not in df_inv.columns:
        return
    starts = pd.DatetimeIndex(
        pd.to_datetime(df_inv['date_time'], format='%Y%m%d%H%M%S', errors='coerce').dropna().unique()
    ).sort_values()
    if starts.empty:
        return

    if len(starts) > 1:
        duration = pd.Series(starts).diff().median()
    else:
        duration = pd.Timedelta(hours=1)
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        duration = pd.Timedelta(hours=1)

    for t0 in starts:
        hrs = pd.date_range(t0.floor('h'), (t0 + duration).floor('h'), freq='h')
        hour_set.update(hrs.strftime('%Y%m%d%H%M%S'))


def process_site(site):
    for instrument in instruments:
        for zid in range(1, max_zid + 1):
            for data_level in data_levels:
                channel = f'awaken/{site}.{instrument}.z{zid:02d}.{data_level}'
                channel_file = OUTPUT_DIR / (channel.replace('/', '_') + '.csv')

                if channel_file.exists():
                    print(f"Skipping {channel} — output already exists", flush=True)
                    continue

                hour_set: set = set()

                # ── Step 1: existence check (stats, 1 day) ──────────────────
                try:
                    result = a2e.search({'Dataset': channel,
                                         'date_time': {'between': [ref_t0, ref_t1]},
                                         'file_type': formats},
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
                                          'date_time': {'between': [t0, t1]},
                                          'file_type': formats},
                                         table='inventory')
                        if inv:
                            _record_hours(inv, hour_set)
                    except Exception:
                        tqdm.write(f"  Weekly timeout — {channel} {w.date()}, retrying daily")
                        for day in tqdm(pd.date_range(w, periods=7, freq='D'),
                                        desc=f'  {w.date()}', unit='d', leave=False):
                            d0 = day.strftime('%Y%m%d%H%M%S')
                            d1 = (day + pd.Timedelta(hours=23, minutes=59, seconds=59)).strftime('%Y%m%d%H%M%S')
                            try:
                                inv = a2e.search({'Dataset': channel,
                                                  'date_time': {'between': [d0, d1]},
                                                  'file_type': formats},
                                                 table='inventory')
                                if inv:
                                    _record_hours(inv, hour_set)
                            except Exception as exc:
                                tqdm.write(f"  Daily timeout — {channel} {day.date()}: {exc}")

                # ── Save this channel's results immediately, only if real entries exist ──
                if not hour_set:
                    print(f"  No inventory records for {channel} — not saving", flush=True)
                    continue

                channel_out = pd.DataFrame({
                    'channel': channel,
                    'date_time': sorted(hour_set),
                })
                channel_out.to_csv(channel_file, index=False)
                print(f"  Saved → {channel_file}  ({len(channel_out)} hourly flags)", flush=True)

#%% Run

if mode == 'parallel':
    with ThreadPoolExecutor(max_workers=len(sites)) as ex:
        list(ex.map(process_site, sites))
else:
    for site in sites:
        process_site(site)

print(f"\nDone. Per-channel files in {OUTPUT_DIR}")
