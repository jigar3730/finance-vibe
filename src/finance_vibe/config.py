"""Central paths and timeframe profiles for Finance Vibe.

Use ``get_mode_config(mode)`` to resolve raw/log directories and yfinance
download parameters for ``weekly`` or ``daily`` pipeline runs.
"""
import os

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
        "period": "2y",
        "interval": "1d",
        "raw_dir": os.path.join(PROJECT_ROOT, "data", "raw", "daily"),
        "logs_dir": os.path.join(PROJECT_ROOT, "data", "logs", "daily"),
    },
}

DEFAULT_MODE = "weekly"


def get_mode_config(mode: str = None) -> dict:
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
STATIC_TICKERS = ["SPY", "QQQ", "IWM", "SCHD"]

BASE_DIR = os.path.join(PROJECT_ROOT, "data")
TICKER_LIST_PATH = os.path.join(BASE_DIR, "active_tickers.csv")


def get_raw_filename(ticker: str, cfg: dict) -> str:
    """Build a standardized raw CSV name, e.g. ``AAPL_10y_1wk.csv``."""
    return f"{ticker}_{cfg['period']}_{cfg['interval']}.csv"


def get_raw_path(ticker: str, cfg: dict) -> str:
    """Absolute path to one ticker's raw CSV inside the active mode directory."""
    return os.path.join(cfg["raw_dir"], get_raw_filename(ticker, cfg))


# Pipeline backtest defaults (offline validation; not part of run_vibe.py)
BACKTEST_LONG_MIN_SCORE = 7
BACKTEST_SHORT_MAX_SCORE = -2
BACKTEST_WARMUP_BARS = 60
BACKTEST_ENTRY_VALID_BARS = 4
BACKTEST_MAX_HOLD_BARS = 12
