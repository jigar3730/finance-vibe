"""Top 3 digest, step 1: candidates from the daily run, 12-1 momentum rank, ledger.

Runs after a daily run that passed its health check (``scripts/run_daily_pipeline.sh``).
It reads that run's ``breakout_setups_<date>.csv`` and ``data_quality_<date>.csv`` and
the daily raw files, cut at the run's as-of date. Nothing in the pipeline or the
dashboard reads its output.

1. **Candidates**: Status BREAKOUT_CONFIRMED or BREAKOUT_TRIGGERED, or PRE_BREAKOUT with
   ``Distance Resistance ATR`` <= ``DIGEST_PRE_BREAKOUT_MAX_ATR``; Trend BULLISH; MTF
   ALIGNED or PARTIAL; not a fund.
2. **Rank** by ``MOM_12_1 = C[t-21] / C[t-252] - 1`` (RESEARCH.md §13's baseline), highest
   first, ties by Symbol. No 252-bar history: dropped (``no_momentum``).
3. **Drops**, in rank order: any flag in that day's bar-quality report (``data_quality``),
   then earnings on t or within ``DIGEST_EARNINGS_SESSIONS`` NYSE sessions after t
   (``earnings``, the date from Yahoo). An unknown earnings date is not a drop. Earnings are
   looked up only until ``DIGEST_REVIEW_N`` candidates survive.
4. The first ``DIGEST_REVIEW_N`` survivors go on to the event-risk review (step 3, not yet
   built); the first ``DIGEST_TOP_N`` are the digest.

Writes ``data/digest/digest_<date>.csv`` (every candidate, with its drop reason) and
replaces that date's rows in ``data/digest/ledger.csv``. The ledger is how the picks get
measured later. None of this is a tested edge (RESEARCH.md §9, §13).

Usage::

    python -m finance_vibe.digest                  # newest daily run
    python -m finance_vibe.digest --date 2026-11-13
    python -m finance_vibe.digest --no-earnings    # skip the Yahoo lookups (offline)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import Callable
from datetime import date

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from finance_vibe import config
from finance_vibe.data import quality, raw
from finance_vibe.data.daily_ingest import NYSEHolidayCalendar
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

MODE = "daily"
BREAKOUT_PREFIX = "breakout_setups_"
PRE_BREAKOUT = "PRE_BREAKOUT"
MOMENTUM = "MOM_12_1"
DROP_NO_MOMENTUM, DROP_QUALITY, DROP_EARNINGS = "no_momentum", "data_quality", "earnings"
CARRY = [
    "Symbol",
    "Status",
    "Close",
    "Resistance",
    "Support",
    "Distance Resistance ATR",
    "ATR",
    "RVOL20",
    "Trend",
    "MTF",
]
DIGEST_COLUMNS = [
    "Date",
    *CARRY,
    MOMENTUM,
    "Momentum Rank",
    "Earnings Date",
    "Dropped",
    "Rank",
    "Review",
    "Selected",
]
_SESSION = CustomBusinessDay(calendar=NYSEHolidayCalendar())

EarningsLookup = Callable[[str], date | None]


def momentum_12_1(close: pd.Series) -> float:
    """``C[t-21] / C[t-252] - 1`` on the last bar of ``close``; NaN without 253 bars."""
    n, skip = config.DIGEST_MOMENTUM_BARS, config.DIGEST_MOMENTUM_SKIP
    if len(close) <= n:
        return float("nan")
    return float(close.iloc[-1 - skip] / close.iloc[-1 - n] - 1.0)


def sessions_after(t: date, d: date) -> int:
    """NYSE sessions strictly after ``t`` up to and including ``d`` (0 when d <= t)."""
    if d <= t:
        return 0
    return len(
        pd.date_range(pd.Timestamp(t) + pd.Timedelta(days=1), pd.Timestamp(d), freq=_SESSION)
    )


def earnings_soon(t: date, d: date | None) -> bool:
    """Earnings on ``t`` or within ``DIGEST_EARNINGS_SESSIONS`` sessions after it."""
    if d is None or d < t:
        return False
    return sessions_after(t, d) <= config.DIGEST_EARNINGS_SESSIONS


def yahoo_earnings(symbol: str) -> date | None:
    """The next earnings date Yahoo lists for ``symbol`` (earliest of a range), or None."""
    import yfinance as yf

    try:
        cal = yf.Ticker(symbol).calendar
    except Exception:  # yfinance raises many types for a missing or odd quote page
        logger.warning("earnings lookup failed for %s", symbol, exc_info=True)
        return None
    dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
    if not dates:
        return None
    return min(pd.Timestamp(x).date() for x in dates)


def latest_stamp(logs_dir: str) -> str:
    """The date suffix of the newest ``breakout_setups_<date>.csv`` in ``logs_dir``."""
    pattern = re.compile(rf"^{BREAKOUT_PREFIX}(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
    stamps = sorted(m.group(1) for n in os.listdir(logs_dir) if (m := pattern.match(n)))
    if not stamps:
        raise FileNotFoundError(f"no {BREAKOUT_PREFIX}<date>.csv in {logs_dir}")
    return stamps[-1]


def candidates(setups: pd.DataFrame) -> pd.DataFrame:
    """Rows of a breakout scan that meet the digest filter (funds left out)."""
    status = setups["Status"]
    near = pd.to_numeric(setups["Distance Resistance ATR"], errors="coerce")
    state_ok = status.isin(config.DIGEST_BREAKOUT_STATUSES) | (
        status.eq(PRE_BREAKOUT) & (near <= config.DIGEST_PRE_BREAKOUT_MAX_ATR)
    )
    keep = (
        state_ok
        & setups["Trend"].eq(config.DIGEST_TREND)
        & setups["MTF"].isin(config.DIGEST_MTF)
        & ~setups["Symbol"].isin(config.UNIVERSE_FUNDS)
    )
    return setups.loc[keep, CARRY].reset_index(drop=True)


def build(
    setups: pd.DataFrame,
    closes: dict[str, pd.Series],
    flagged: set[str],
    as_of: date,
    earnings: EarningsLookup | None,
) -> pd.DataFrame:
    """The digest table: every candidate, its momentum rank, drop reason and final rank.

    ``closes`` maps Symbol to its daily closes up to ``as_of``. ``flagged`` holds the
    symbols with a bar-quality flag. ``earnings`` None skips the earnings check.
    """
    c = candidates(setups)
    c[MOMENTUM] = [momentum_12_1(closes.get(s, pd.Series(dtype="float64"))) for s in c["Symbol"]]
    c = c.sort_values([MOMENTUM, "Symbol"], ascending=[False, True], na_position="last")
    c = c.reset_index(drop=True)
    c["Momentum Rank"] = pd.array(
        [i + 1 if pd.notna(m) else pd.NA for i, m in enumerate(c[MOMENTUM])], dtype="Int64"
    )
    c["Earnings Date"] = ""
    c["Dropped"] = ""
    c.loc[c[MOMENTUM].isna(), "Dropped"] = DROP_NO_MOMENTUM
    c.loc[(c["Dropped"] == "") & c["Symbol"].isin(flagged), "Dropped"] = DROP_QUALITY

    c["Rank"] = pd.Series(pd.NA, index=c.index, dtype="Int64")
    survivors = 0
    for i in c.index[c["Dropped"] == ""]:
        if survivors >= config.DIGEST_REVIEW_N:
            break
        if earnings is not None:
            d = earnings(str(c.at[i, "Symbol"]))
            c.at[i, "Earnings Date"] = d.isoformat() if d else "unknown"
            if earnings_soon(as_of, d):
                c.at[i, "Dropped"] = DROP_EARNINGS
                continue
        survivors += 1
        c.at[i, "Rank"] = survivors

    c["Review"] = c["Rank"].le(config.DIGEST_REVIEW_N).fillna(False).astype(bool)
    c["Selected"] = c["Rank"].le(config.DIGEST_TOP_N).fillna(False).astype(bool)
    c.insert(0, "Date", as_of.isoformat())
    return c[DIGEST_COLUMNS]


def update_ledger(table: pd.DataFrame, path: str) -> pd.DataFrame:
    """Replace the ledger's rows for ``table``'s date with ``table``; returns the ledger."""
    day = table["Date"].iloc[0] if len(table) else None
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"Date": str})
        old = old[old["Date"] != day] if day else old
        ledger = pd.concat([old, table], ignore_index=True) if len(table) else old
    else:
        ledger = table
    ledger = ledger.sort_values(
        ["Date", "Momentum Rank", "Symbol"], kind="stable", na_position="last"
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ledger.to_csv(path, index=False)
    return ledger


def run(
    stamp: str | None = None, *, earnings: EarningsLookup | None = yahoo_earnings
) -> pd.DataFrame:
    """Build and save the digest for one daily run (the newest when ``stamp`` is None)."""
    cfg = config.get_mode_config(MODE)
    logs_dir, raw_dir = cfg["logs_dir"], cfg["raw_dir"]
    stamp = stamp or latest_stamp(logs_dir)
    setups = pd.read_csv(os.path.join(logs_dir, f"{BREAKOUT_PREFIX}{stamp}.csv"))
    for col in [*CARRY, "AsOf Date"]:
        if col not in setups.columns:
            raise ValueError(f"breakout_setups_{stamp}.csv has no {col!r} column")
    as_of_values = setups["AsOf Date"].dropna().unique()
    if len(as_of_values) != 1:
        raise ValueError(f"breakout_setups_{stamp}.csv has {len(as_of_values)} as-of dates")
    as_of = pd.Timestamp(as_of_values[0]).date()

    report = quality.report_path(logs_dir, stamp)
    flagged: set[str] = set()
    if os.path.exists(report):
        flagged = set(pd.read_csv(report, dtype=str)["Symbol"].dropna())
    else:
        logger.warning("no bar-quality report %s; no candidate dropped for quality", report)

    wanted = set(candidates(setups)["Symbol"])
    files = {f.ticker: f.path for f in raw.raw_files(raw_dir) if f.ticker in wanted}
    closes = {
        s: raw.load_raw(p, as_of=as_of.isoformat(), weekly=False)["Close"] for s, p in files.items()
    }

    table = build(setups, closes, flagged, as_of, earnings)
    os.makedirs(config.DIGEST_DIR, exist_ok=True)
    table.to_csv(os.path.join(config.DIGEST_DIR, f"digest_{stamp}.csv"), index=False)
    update_ledger(table, config.DIGEST_LEDGER)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="the daily run's date stamp (default: newest)")
    parser.add_argument("--no-earnings", action="store_true", help="skip the Yahoo lookups")
    args = parser.parse_args(argv)
    table = run(args.date, earnings=None if args.no_earnings else yahoo_earnings)
    dropped = table["Dropped"].value_counts().to_dict()
    top = table[table["Selected"]]
    logger.info(
        "digest %s: %d candidates, %d dropped %s, top %d: %s",
        table["Date"].iloc[0] if len(table) else "-",
        len(table),
        sum(dropped.get(k, 0) for k in (DROP_NO_MOMENTUM, DROP_QUALITY, DROP_EARNINGS)),
        {k: v for k, v in dropped.items() if k},
        len(top),
        ", ".join(top["Symbol"]) or "none",
    )
    if len(table):
        cols = ["Rank", "Symbol", "Status", "Close", MOMENTUM, "Earnings Date", "Dropped"]
        print(table[cols].to_markdown(index=False, floatfmt=".4f"))
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
