"""Post-run health check for the scheduled weekly pipeline.

Piped into the container by ``run_weekly_pipeline.sh``:

    docker exec -i finance_vibe python - < scripts/check_weekly_outputs.py

A run can exit 0 and still be wrong (a stale week, a Yahoo outage, an empty
stage), so this checks the outputs themselves. It prints one line per problem
and exits 1 if there are any.
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime

import pandas as pd

from finance_vibe import config
from finance_vibe.data import quality
from finance_vibe.data.ingest import MARKET_TZ, last_complete_week
from finance_vibe.web.playbook import MARKET_SYMBOLS

MODE = "weekly"
MAX_INGEST_ERROR_SHARE = 0.20
BENCHMARKS = ("SPY", "QQQ")
EXPECTED_PREFIXES = (
    "coiled_cobra_setups_",
    "breakout_setups_",
)


def main() -> int:
    now = datetime.now(MARKET_TZ)
    stamp = now.strftime("%Y-%m-%d")
    cfg = config.get_mode_config(MODE)
    logs, raw = cfg["logs_dir"], cfg["raw_dir"]
    problems: list[str] = []

    for prefix in EXPECTED_PREFIXES:
        path = os.path.join(logs, f"{prefix}{stamp}.csv")
        if not os.path.exists(path):
            problems.append(f"missing output: {os.path.basename(path)}")

    want = pd.Timestamp(last_complete_week(now))
    for sym in BENCHMARKS:
        hits = glob.glob(os.path.join(raw, f"{sym}_*.csv"))
        if not hits:
            problems.append(f"no raw file for benchmark {sym}")
            continue
        last = pd.to_datetime(pd.read_csv(hits[0], usecols=["Date"])["Date"]).max().normalize()
        if last != want:
            problems.append(f"{sym} newest weekly bar is {last.date()}, expected {want.date()}")

    tickers = pd.read_csv(config.TICKER_LIST_PATH)["Ticker"].dropna().nunique()
    err_path = os.path.join(logs, f"ingest_errors_{stamp}.csv")
    errors = len(pd.read_csv(err_path)) if os.path.exists(err_path) else 0
    if tickers and errors / tickers > MAX_INGEST_ERROR_SHARE:
        problems.append(
            f"ingest errors {errors}/{tickers} tickers (> {MAX_INGEST_ERROR_SHARE:.0%})"
        )

    # Bar-quality errors on the Market tab ETFs fail the run; others are counted.
    dq_path = quality.report_path(logs, stamp)
    if not os.path.exists(dq_path):
        problems.append(f"missing output: {os.path.basename(dq_path)}")
        dq_errors = 0
    else:
        dq = pd.read_csv(dq_path, dtype=str)
        dq = dq[dq["Severity"] == quality.ERROR]
        dq_errors = len(dq)
        for r in dq[dq["Symbol"].isin(MARKET_SYMBOLS)].itertuples(index=False):
            problems.append(f"{r.Symbol} data quality {r.Check} ({r.Date}): {r.Detail}")

    if problems:
        print("HEALTH CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(
        f"health check ok: outputs for {stamp}, newest bar {want.date()}, ingest errors {errors}/{tickers}, "
        f"data-quality errors {dq_errors}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
