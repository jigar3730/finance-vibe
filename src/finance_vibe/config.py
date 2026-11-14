"""Central paths and timeframe profiles for Finance Vibe.

Use ``get_mode_config(mode)`` to resolve raw/log directories and yfinance
download parameters for ``weekly`` or ``daily`` pipeline runs.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "../.."))

TIMEFRAME_PROFILES = {
    "weekly": {
        "period": "10y",
        "interval": "1wk",
        "raw_dir": os.path.join(PROJECT_ROOT, "data", "raw", "weekly"),
        "logs_dir": os.path.join(PROJECT_ROOT, "data", "logs", "weekly"),
    },
    "daily": {
        "period": "5y",
        "interval": "1d",
        "raw_dir": os.path.join(PROJECT_ROOT, "data", "raw", "daily"),
        "logs_dir": os.path.join(PROJECT_ROOT, "data", "logs", "daily"),
    },
}

DEFAULT_MODE = "weekly"

# Coiled Cobra rubric version (docs/coiled_cobra.md).
# Bump whenever gate/pillar logic or thresholds change in a way that alters
# ``Score`` or which setups qualify (e.g. Gate D 5/6 -> 4/6).
RUBRIC_VERSION = "4.0"

# Breakout scanner rubric version (docs/breakout.md), stamped on scan_history()
# rows and every research output. Bump whenever a state, Status or Readiness
# rule or threshold changes; tests/test_rubric_docs_sync.py fingerprints the rules.
BREAKOUT_RUBRIC_VERSION = "1.0"


def get_mode_config(mode: str | None = None) -> dict[str, Any]:
    """Return download settings and directory paths for ``weekly`` or ``daily``.

    Creates ``raw_dir`` and ``logs_dir`` if they do not exist.
    Falls back to ``DEFAULT_MODE`` when ``mode`` is missing or invalid.
    """
    target_mode = mode if mode in TIMEFRAME_PROFILES else DEFAULT_MODE
    cfg = TIMEFRAME_PROFILES[target_mode].copy()
    cfg["mode"] = target_mode

    os.makedirs(cfg["raw_dir"], exist_ok=True)
    os.makedirs(cfg["logs_dir"], exist_ok=True)

    return cfg


BASE_DIR = os.path.join(PROJECT_ROOT, "data")
TICKER_LIST_PATH = os.path.join(BASE_DIR, "active_tickers.csv")

# =====================================================================
# SCAN UNIVERSE (finance_vibe.data.universe; replaces the screener list)
# =====================================================================
# Candidates: S&P 500 + S&P 400 (Wikipedia, scripts/refresh_index_lists.py, dated
# in index_lists.json) plus the stock watchlist in ticker_manifest.csv. Every
# candidate, watchlist included, must pass the same rules. Rebuilt once a quarter,
# on the first run after the third Friday of a rebalance month; between rebuilds
# the members do not change. Hysteresis: a non-member joins at the JOIN levels, a
# member stays while it holds the STAY levels, so names near a line don't flicker.
_DATA_PKG_DIR = os.path.join(_CONFIG_DIR, "data")
SP500_TICKERS_PATH = os.path.join(_DATA_PKG_DIR, "sp500_tickers.csv")
SP400_TICKERS_PATH = os.path.join(_DATA_PKG_DIR, "sp400_tickers.csv")
INDEX_LISTS_META_PATH = os.path.join(_DATA_PKG_DIR, "index_lists.json")
WATCHLIST_PATH = os.path.join(_DATA_PKG_DIR, "ticker_manifest.csv")
# Funds: always members (RS uses QQQ; the market strip reads SPY / QQQ / SCHG / SCHD
# rows from the scan), tagged "fund" and left out of stock counts and breadth.
UNIVERSE_FUNDS = ("SPY", "QQQ", "IWM", "SCHD", "SCHG", "IBIT", "GDX")
UNIVERSE_JOIN_DOLLAR_VOLUME = 20_000_000.0  # median Close x Volume over the window
UNIVERSE_STAY_DOLLAR_VOLUME = 15_000_000.0
UNIVERSE_JOIN_CLOSE = 5.0
UNIVERSE_STAY_CLOSE = 4.0
UNIVERSE_DV_WINDOW = 63  # daily bars
UNIVERSE_MIN_BARS = 252  # daily bars of listing history
UNIVERSE_REBALANCE_MONTHS = (3, 6, 9, 12)
UNIVERSE_HISTORY_PERIOD = "2y"  # daily bars downloaded per candidate at a rebuild
# A rebuild with more failed downloads than this share keeps the previous members.
UNIVERSE_MAX_FAILED_SHARE = 0.10
UNIVERSE_DIR = os.path.join(BASE_DIR, "universe")  # members, changes, rejected, meta, bars/

# =====================================================================
# RESEARCH UNIVERSE (finance_vibe.research only; scanners and dashboard ignore it)
# =====================================================================
# Point-in-time membership at date t uses only bars on or before t: a ticker is
# in the universe if it is not excluded, has >= RESEARCH_MIN_HISTORY_BARS
# sessions of listing history, closes >= RESEARCH_MIN_CLOSE, and ranks in the
# top RESEARCH_UNIVERSE_TOP_N by trailing RESEARCH_DV_WINDOW-bar median
# Close x Volume. Changing a value after results exist is a RESEARCH.md variant.
RESEARCH_DIR = os.path.join(BASE_DIR, "research")
RESEARCH_UNIVERSE_TOP_N = 200
RESEARCH_DV_WINDOW = 20
RESEARCH_MIN_HISTORY_BARS = 252
RESEARCH_MIN_CLOSE = 5.0
# Funds and ETPs in the raw pool, reviewed by hand on 2026-10-26 (no quote type is
# stored offline): SPY, QQQ, IWM, SCHD, SCHG, IBIT and GDX (now UNIVERSE_FUNDS). Reviewed and kept as companies: SPCX
# (2026 mega-cap IPO), PURR / MSTR / BMNR / ASST (crypto-treasury corporations),
# ARCC (BDC), BULL (SPAC history before 2025), ADRs, preferreds, REITs, ET (MLP).
# Unverified, kept on their price/volume profile: SKYD, INIO, FPS.
# Review again whenever the manifest or screener adds symbols.
RESEARCH_EXCLUDED_SYMBOLS = frozenset({"SPY", "QQQ", "IWM", "SCHD", "SCHG", "IBIT", "GDX"})
# Forward labels (finance_vibe.research.labels, RESEARCH.md §4-§7). Signal dates
# after RESEARCH_IN_SAMPLE_END are the sealed holdout: labelling one raises.
RESEARCH_IN_SAMPLE_START = "2022-01-01"
RESEARCH_IN_SAMPLE_END = "2026-03-31"
RESEARCH_HORIZON = 10
RESEARCH_RANDOM_DRAWS = 20
RESEARCH_SEED = 20261027
RESEARCH_BENCHMARK = "SPY"
# Weekly addendum (RESEARCH.md §11): weekly bars, same seal on the week-end date.
# The in-sample start is "first available" (before the 2016-10 weekly files begin).
RESEARCH_WEEKLY_IN_SAMPLE_START = "2016-01-01"
RESEARCH_WEEKLY_HORIZON = 4
RESEARCH_WEEKLY_MIN_HISTORY_BARS = 52
RESEARCH_WEEKLY_DV_WINDOW = 4
# RESEARCH.md §12 (pre-breakout addendum, daily). Same seal (RESEARCH_IN_SAMPLE_END),
# horizon, draws, seed and benchmark as §4-§7. Cobra's daily backfill starts 2020-01-13
# and a first bar needs the bar before it.
RESEARCH_PREBREAKOUT_IN_SAMPLE_START = "2020-01-14"
RESEARCH_PREBREAKOUT_SNAPSHOT = "2026-11-09"

# =====================================================================
# BACKFILL SNAPSHOT (finance_vibe.research.snapshot; scanners and dashboard ignore it)
# =====================================================================
# One frozen 10-year download per dated folder; the universe rule below picks
# tickers from it without looking at how they performed. Thresholds apply to
# the whole history. A ticker must pass on both daily and weekly bars.
BACKFILL_SNAPSHOT_DIR = os.path.join(RESEARCH_DIR, "backfill", "snapshots")
DATABASE_PATH = os.path.join(BASE_DIR, "db", "finance_vibe.sqlite")  # derived; see data.database
ANALYSIS_SUMMARY_PATH = os.path.join(BASE_DIR, "db", "analysis_summary.json")  # for /scan-stats
BACKFILL_HISTORY_DIR = os.path.join(RESEARCH_DIR, "backfill", "history")
BACKFILL_WORKERS = 3  # leaves one of the host's four cores for everything else
# S&P 500 members from Wikipedia (first taken 2026-11-09 for the backfill; refreshed by
# scripts/refresh_index_lists.py, dated in index_lists.json).
# Same file as the scan universe. The snapshot's manifest.json keeps the exact list it
# requested, so refreshing it later does not change the frozen backfill.
BACKFILL_SP500_PATH = SP500_TICKERS_PATH
BACKFILL_PERIOD = "10y"
BACKFILL_INTERVALS = {"daily": "1d", "weekly": "1wk"}
BACKFILL_BENCHMARKS = ("QQQ", "SPY")  # stored for RS, never universe members
BACKFILL_MIN_YEARS = 10
BACKFILL_HISTORY_GRACE_DAYS = 14  # Yahoo's "10y" starts within days of the 10-year mark
BACKFILL_MIN_CLOSE = 5.0
BACKFILL_DV_WINDOW = 63  # daily bars for the median dollar volume
BACKFILL_MIN_DOLLAR_VOLUME = 20_000_000.0
BACKFILL_MAX_MISSING_BARS = 5  # bars absent versus SPY's calendar, after the first bar
BACKFILL_MAX_INCOHERENT_BARS = 0
BACKFILL_MAX_ZERO_VOLUME_BARS = 5
BACKFILL_MAX_FLAT_RUN = 10  # identical closes in a row; shorter runs are thin old trading
# A big move only counts as bad data on ordinary volume: real news trades heavily.
BACKFILL_QUIET_VOLUME_X = 2.0  # at most this times the prior 20-bar median volume
BACKFILL_VOLUME_WINDOW = 20
BACKFILL_REVERSAL_PCT = 0.10  # a big move undone to within this of the prior close
BACKFILL_SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 50)
BACKFILL_SPLIT_TOL = 0.03  # relative distance from a whole split ratio


def _get_raw_filename(ticker: str, cfg: dict[str, Any]) -> str:
    """Build a standardized raw CSV name, e.g. ``AAPL_10y_1wk.csv``."""
    return f"{ticker}_{cfg['period']}_{cfg['interval']}.csv"


def get_raw_path(ticker: str, cfg: dict[str, Any]) -> str:
    """Absolute path to one ticker's raw CSV inside the active mode directory."""
    return os.path.join(cfg["raw_dir"], _get_raw_filename(ticker, cfg))


def resolve_pipeline_mode(mode: str = "weekly") -> tuple[str, str]:
    """Map a CLI/pipeline mode to (data_timeframe, signal_profile).

    Both are the mode itself; unknown modes fall back to ``DEFAULT_MODE``.
    """
    key = (mode or DEFAULT_MODE).strip().lower()
    if key in TIMEFRAME_PROFILES:
        return key, key
    return DEFAULT_MODE, DEFAULT_MODE


# =====================================================================
# HISTORICAL "AS-OF" RUNS
# =====================================================================
# ``--as-of YYYY-MM-DD`` re-runs a scan as if it were that date: every bar not
# yet complete on that date is dropped, and outputs are stamped with the as-of
# date instead of today's. CLI only -- deliberately no environment fallback, so
# a stray variable can never turn a live scan into a historical one.


def parse_as_of(argv: list[str] | None = None) -> str | None:
    """Return the ``--as-of`` date (ISO ``YYYY-MM-DD``) from ``argv``, or None.

    Accepts ``--as-of DATE`` and ``--as-of=DATE``. Raises ``ValueError`` for a
    missing value, a malformed date, or a date in the future.
    """
    args = sys.argv[1:] if argv is None else argv
    raw: str | None = None
    for i, arg in enumerate(args):
        if arg == "--as-of":
            if i + 1 >= len(args):
                raise ValueError("--as-of requires a date (YYYY-MM-DD)")
            raw = args[i + 1]
        elif arg.startswith("--as-of="):
            raw = arg.split("=", 1)[1]
    if raw is None:
        return None
    try:
        parsed = date.fromisoformat(raw.strip())
    except ValueError:
        raise ValueError(f"--as-of must be a date like 2025-11-07, got {raw!r}") from None
    if parsed > date.today():
        raise ValueError(f"--as-of {parsed.isoformat()} is in the future")
    return parsed.isoformat()


def run_stamp(as_of: str | None = None) -> str:
    """Date stamp for output filenames: the as-of date, else today."""
    return as_of or datetime.now().strftime("%Y-%m-%d")


# Anything pd.to_datetime turns into a single Timestamp.
type AsOf = str | date | np.datetime64


def cut_to_as_of(df: pd.DataFrame, as_of: str | None, *, weekly: bool) -> pd.DataFrame:
    """Keep only bars that are complete on ``as_of`` (no lookahead).

    A weekly bar is dated at the start of its week (Monday) but holds the whole
    week's OHLCV, so it only counts once that week's Friday is on or before
    ``as_of`` -- otherwise a mid-week as-of date would see the rest of the
    week. A daily bar counts on its own date (scans are assumed to run after
    the close). ``as_of=None`` returns ``df`` unchanged.
    """
    if as_of is None:
        return df
    col = next((c for c in df.columns if str(c).strip().lower() in ("date", "datetime")), None)
    if col is None:
        # Never fall through with uncut data on a historical run.
        raise ValueError("cannot apply --as-of: frame has no Date column")
    dates = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    if weekly:
        ends = dates - pd.to_timedelta(dates.dt.weekday, unit="D") + pd.Timedelta(days=4)
    else:
        ends = dates
    return df.loc[ends <= pd.Timestamp(as_of)].reset_index(drop=True)


# =====================================================================
# DATA QUALITY CONTRACT
# =====================================================================

# Columns every raw OHLCV CSV must contain after ingestion/normalization.
REQUIRED_OHLCV = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Minimum usable rows before a raw CSV is worth saving/scanning.
MIN_SAVE_ROWS = 60

# Bar-level quality flags (data.quality), written to data_quality_<date>.csv after
# ingest. They flag suspect bars for review; they never change what is scanned.
DQ_RECENT_BARS = 20  # window for the per-bar checks (newest bars only)
DQ_FLAT_RUN = 5  # this many identical closes in a row looks like placeholder data
DQ_JUMP_PCT = 0.40  # a close-to-close move this large may be a split or a bad print
DQ_PRICE_TOL = 1e-6  # relative slack when checking High/Low against Open/Close

# Market context (home page): an index and funds downloaded only for the market facts,
# by data.market_ingest at 17:05 ET and at the end of each daily run. They live in their
# own silo so neither scanner ever sees them (and no scan CSV changes), even for symbols
# also in the scanned universe. Same OHLCV contract and bar-quality checks as that universe.
MARKET_CONTEXT_PROFILE = {
    "period": "2y",
    "interval": "1d",
    "raw_dir": os.path.join(PROJECT_ROOT, "data", "raw", "market"),
    "logs_dir": os.path.join(PROJECT_ROOT, "data", "logs", "market"),
}
# The 11 S&P 500 sector funds, for breadth (sectors above their 50-day average).
MARKET_SECTOR_FUNDS = {
    "XLB": "Materials",
    "XLC": "Communication services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer staples",
    "XLRE": "Real estate",
    "XLU": "Utilities",
    "XLV": "Health care",
    "XLY": "Consumer discretionary",
}
# Short tile labels for the same funds (market home breadth tiles; 9 characters at most so
# all 11 fit on one row).
MARKET_SECTOR_SHORT = {
    "XLB": "Materials",
    "XLC": "Comms",
    "XLE": "Energy",
    "XLF": "Finance",
    "XLI": "Industr.",
    "XLK": "Tech",
    "XLP": "Staples",
    "XLRE": "Real est.",
    "XLU": "Utilities",
    "XLV": "Health",
    "XLY": "Discr.",
}
# Broad market funds, for the home page's table and the growth-vs-dividend label.
MARKET_INDEX_FUNDS = ("SPY", "QQQ", "IWM", "RSP", "SCHG", "SCHD")
MARKET_HOME_FUNDS = ("SPY", "QQQ", "IWM", "RSP")  # the home table's rows
# File symbol -> Yahoo symbol. VIX is an index (``^VIX``); its file is ``VIX_...csv``.
MARKET_CONTEXT_TICKERS = {
    **{s: s for s in MARKET_INDEX_FUNDS},
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",  # 3-month volatility index (VIX / VIX3M)
    "TNX": "^TNX",  # 10-year Treasury yield, in percent
    "DXY": "DX-Y.NYB",  # US dollar index
    "HYG": "HYG",  # high-yield corporate bonds
    "IEF": "IEF",  # 7-10 year Treasuries (credit ratio HYG / IEF)
    **{s: s for s in MARKET_SECTOR_FUNDS},
}
# Indexes Yahoo reports without volume: the zero-volume quality check is skipped.
MARKET_NO_VOLUME = frozenset({"VIX", "VIX3M", "TNX", "DXY"})

# Live quotes for the market home (data.market_snapshot). Their own file, never data/raw/.
MARKET_SNAPSHOT_PATH = os.path.join(PROJECT_ROOT, "data", "live", "market_snapshot.json")
MARKET_SNAPSHOT_TICKERS = MARKET_CONTEXT_TICKERS
MARKET_SNAPSHOT_PERIOD = "2d"
MARKET_SNAPSHOT_INTERVAL = "5m"
# The page warns past this; pre-market slots are 15 minutes apart.
MARKET_SNAPSHOT_STALE_MINUTES = 20

# Scheduled releases for the market home, hand-copied on 2026-11-05 and updated about once
# a year. Name -> (release time ET, dates). FOMC dates are the decision day (the meeting's
# second day), and its 2027 dates are tentative until confirmed at the meeting before.
# Sources: federalreserve.gov/monetarypolicy/fomccalendars.htm,
# bls.gov/schedule/news_release/empsit.htm, bls.gov/schedule/news_release/cpi.htm
MARKET_EVENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Jobs report": (
        "08:30",
        (
            "2026-11-06", "2026-12-04", "2027-01-08", "2027-02-05", "2027-03-05", "2027-04-02",
            "2027-05-07", "2027-06-04", "2027-07-02", "2027-08-06", "2027-09-03", "2027-10-08",
            "2027-11-05", "2027-12-03",
        ),
    ),
    "CPI": (
        "08:30",
        (
            "2026-11-10", "2026-12-10", "2027-01-13", "2027-02-11", "2027-03-10", "2027-04-13",
            "2027-05-12", "2027-06-10", "2027-07-14", "2027-08-11", "2027-09-14", "2027-10-14",
            "2027-11-10", "2027-12-10",
        ),
    ),
    "FOMC": (
        "14:00",
        (
            "2026-12-09", "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28",
            "2027-09-15", "2027-10-27", "2027-12-08",
        ),
    ),
}  # fmt: skip

# Breadth across the scanned tickers (data.universe_breadth), rewritten by each daily run.
UNIVERSE_BREADTH_PATH = os.path.join(MARKET_CONTEXT_PROFILE["logs_dir"], "universe_breadth.json")

# Market tab growth-vs-dividend rotation (web.insights.rotation): SCHG / SCHD close
# ratio vs its SMA, plus the spread of their returns over the lookback. Unvalidated.
ROTATION_GROWTH, ROTATION_DIVIDEND = "SCHG", "SCHD"
ROTATION_MA_BARS = 50
ROTATION_LOOKBACK = 63

# Operating docs served in the dashboard under /docs (web.docs): URL name -> (path from the
# repo root, one-line contents), in index order. Research notes stay out. The image copies
# these files (.dockerignore).
DOCS_DIR = PROJECT_ROOT
DOCS_PAGES: dict[str, tuple[str, str]] = {
    "runbook": (
        "docs/runbook.md",
        "Daily check, what to do when a run fails, deploying, recovery, recurring tasks",
    ),
    "operations": (
        "docs/operations.md",
        "Pipeline stages, schedule, health checks, database, every dashboard page, maintenance",
    ),
    "coiled-cobra": (
        "docs/coiled_cobra.md",
        "Coiled Cobra rules: hard gates, 100-point scorecard, grades, output columns",
    ),
    "breakout": (
        "docs/breakout.md",
        "Breakout readiness rules: states, status rules, 100-point score, display rule",
    ),
    "ux-standards": (
        "docs/UX_STANDARDS.md",
        "Rules for every dashboard page: facts not verdicts, colors, layout, number formats",
    ),
    "readme": ("README.md", "Project overview, quick start, pipeline and layout"),
}

# Breakout confirmation volume: RVOL20 at or above this on a close through
# resistance confirms a breakout. Shared by scanners.breakout and the
# dashboard's trigger rule (web.playbook).
BREAKOUT_RVOL_CONFIRM = 1.50

# Setup-row schema emitted by scanners.coiled_cobra: a single stable contract for
# coiled_cobra_setups_<date>.csv. Column order is significant for CSV output.
#
# Fill rules:
#   - ``coiled_cobra``: sets Source="coiled_cobra"; fills the macro-only
#     fields plus RSI/Notes. Setup Type is currently always SETUP_LONG.
#   - ``AsOf Date``: the signal bar's date.
SETUP_ROW_COLUMNS = [
    "Symbol",
    "Setup Type",
    "Source",
    "Mode",
    "AsOf Date",
    "Close",
    "EMA20",
    "EMA50",
    "ATR",
    "RSI",
    "Swing Low",
    "Swing High",
    "Risk Per Share",
    "Regime OK",
    "RS 63d",
    "Notes",
    "Score",
    "Grade",
    "Tier",
    "Checks Met",
    "Fib 61.8%",
    "Fib 78.6%",
    "Fib Score",
    "MACD",
    "MACD Signal",
    # Coiled Cobra v3 attribution
    "RVOL",
    "Market Gate",
]

# Coiled Cobra scan_history(): one row per scored bar, passes and rejects alike.
COBRA_HISTORY_COLUMNS = [
    "Date",
    *SETUP_ROW_COLUMNS,
    "Passed",
    "BBWidth Pctile",
    "Vol Contraction Points",
    "Relative Strength Points",
    "Structure Points",
    "Volume Shelf Points",
    "Overhead Clearance Points",
    "RVOL Trigger Points",
    "Rubric Version",
]


def blank_setup_row() -> dict[str, Any]:
    """Return a setup-row dict with every schema key present and set to None."""
    return {col: None for col in SETUP_ROW_COLUMNS}


def validate_and_clean_ohlcv(df: pd.DataFrame, *, require_volume: bool = True) -> pd.DataFrame:
    """Normalize and validate an OHLCV frame against :data:`REQUIRED_OHLCV`.

    Standardizes a ``Date`` column (promoting a DatetimeIndex when needed),
    capitalizes OHLCV headers, coerces price/volume columns to numeric, and
    drops rows with NaN in any of Open/High/Low/Close.

    Raises ``ValueError`` if required columns are missing after normalization.
    """
    out = df.copy()

    # Promote an index-based Date (yfinance downloads) into a real column.
    if "Date" not in out.columns and "date" not in {str(c).lower() for c in out.columns}:
        out = out.reset_index()

    # Standardize header casing (Open, High, Low, Close, Volume, Date).
    out.columns = [str(c).strip().capitalize() for c in out.columns]

    required = (
        list(REQUIRED_OHLCV) if require_volume else [c for c in REQUIRED_OHLCV if c != "Volume"]
    )
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    numeric_cols = ["Open", "High", "Low", "Close"]
    if "Volume" in out.columns:
        numeric_cols.append("Volume")
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return out
