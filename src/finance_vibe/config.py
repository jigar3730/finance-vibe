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


# Always-included symbols merged with manifest and screener output
STATIC_TICKERS = ["SPY", "QQQ", "IWM", "SCHD", "IBIT", "SCHG"]

# Universe size for data.tickers → data/active_tickers.csv
ACTIVE_TICKER_CAP = 1000
# yahooquery Screener IDs merged after the manifest (deduped, then capped)
SCREENER_IDS = [
    "most_actives",
]
# Per-screener quote count (Yahoo typically caps near 250)
SCREENER_COUNT = 250

BASE_DIR = os.path.join(PROJECT_ROOT, "data")
TICKER_LIST_PATH = os.path.join(BASE_DIR, "active_tickers.csv")

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
# stored offline): STATIC_TICKERS plus GDX. Reviewed and kept as companies: SPCX
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
