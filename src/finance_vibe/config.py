import os

# --- API Parameters ---
PERIOD = "5y"
INTERVAL = "1wk"

# --- Ticker Lists ---
# These are always included regardless of market activity
STATIC_TICKERS = ["SPY", "QQQ", "IWM", "SCHD"] 

# --- Folder Structure ---
BASE_DIR = "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")
LOGS_DIR = os.path.join(BASE_DIR, "logs")  # Changed DATA_DIR to BASE_DIR
TICKER_LIST_PATH = os.path.join(BASE_DIR, "active_tickers.csv")

# --- Filename Logic ---
def get_raw_filename(ticker):
    return f"{ticker}_{PERIOD}_{INTERVAL}.csv"

def get_raw_path(ticker):
    return os.path.join(RAW_DIR, get_raw_filename(ticker))

# --- Directory Initialization ---
# This ensures all folders exist before any script tries to save to them
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)