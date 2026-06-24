import os

# --- 1. PROJECT PATH LOGIC (The Root Fix) ---
# This finds the absolute path to the 'finance-vibe' folder
# This file is in /src/finance_vibe/, so root is 2 levels up
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "../.."))

# --- 2. TIMEFRAME PROFILES ---
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
    }
}

DEFAULT_MODE = "weekly"


def get_mode_config(mode: str = None):
    """
    Returns runtime parameters and safe directory paths for a given mode.
    Automatically initializes directories on access.
    """
    target_mode = mode if mode in TIMEFRAME_PROFILES else DEFAULT_MODE
    cfg = TIMEFRAME_PROFILES[target_mode].copy()
    cfg["mode"] = target_mode
    
    # Ensure mode-specific folders exist automatically on-demand
    os.makedirs(cfg["raw_dir"], exist_ok=True)
    os.makedirs(cfg["logs_dir"], exist_ok=True)
    
    return cfg


# --- 3. TICKER LISTS ---
STATIC_TICKERS = ["SPY", "QQQ", "IWM", "SCHD"]

# --- 4. SHARED PATH CONFIGURATIONS ---
BASE_DIR = os.path.join(PROJECT_ROOT, "data")
TICKER_LIST_PATH = os.path.join(BASE_DIR, "active_tickers.csv")


# --- 5. FILENAME LOGIC (Updated to require profile context) ---
def get_raw_filename(ticker, cfg):
    """Generates standardized filename based on current profile context: e.g., AAPL_10y_1wk.csv"""
    return f"{ticker}_{cfg['period']}_{cfg['interval']}.csv"


def get_raw_path(ticker, cfg):
    """Returns absolute path to a specific ticker's raw data file within the active profile directory."""
    return os.path.join(cfg["raw_dir"], get_raw_filename(ticker, cfg))


# --- 6. BACKTEST SETTINGS ---
BACKTEST_START_DATE = "2020-01-01"
BACKTEST_INITIAL_CAPITAL = 10000
BACKTEST_BUY_SCORE = 7   # "🟢 STARTER POSITION"
BACKTEST_SELL_SCORE = 1  # Exit when it hits "NO EDGE" or "REDUCE"