import os

# --- 1. PROJECT PATH LOGIC (The Root Fix) ---
# This finds the absolute path to the 'finance-vibe' folder
# This file is in /src/finance_vibe/, so root is 2 levels up
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "../.."))

# --- 2. API PARAMETERS ---
PERIOD = "10y"
INTERVAL = "1wk"

# --- 3. TICKER LISTS ---
STATIC_TICKERS = ["SPY", "QQQ", "IWM", "SCHD"]

# --- 4. FOLDER STRUCTURE (Absolute Paths) ---
BASE_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(BASE_DIR, "raw")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
TICKER_LIST_PATH = os.path.join(BASE_DIR, "active_tickers.csv")

# --- 5. FILENAME LOGIC ---


def get_raw_filename(ticker):
    """Generates standardized filename: e.g., AAPL_2y_1d.csv"""
    return f"{ticker}_{PERIOD}_{INTERVAL}.csv"


def get_raw_path(ticker):
    """Returns absolute path to a specific ticker's raw data file."""
    return os.path.join(RAW_DIR, get_raw_filename(ticker))


# --- 6. BACKTEST SETTINGS ---
BACKTEST_START_DATE = "2020-01-01"
BACKTEST_INITIAL_CAPITAL = 10000
BACKTEST_BUY_SCORE = 7   # "🟢 STARTER POSITION"
BACKTEST_SELL_SCORE = 1  # Exit when it hits "NO EDGE" or "REDUCE"

# --- 7. DIRECTORY INITIALIZATION ---
# This ensures folders exist relative to the project root
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
