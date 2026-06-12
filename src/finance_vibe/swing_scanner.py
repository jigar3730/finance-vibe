import os
import logging
from datetime import datetime

import pandas as pd
import pandas_ta as ta

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ACTIVE_TICKERS_PATH = os.path.join(BASE_DIR, "data", "active_tickers.csv")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =========================
# INDICATORS
# =========================


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)

    macd = ta.macd(df["Close"])
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Critical hygiene
    df.dropna(inplace=True)
    return df

# =========================
# MOMENTUM FILTER
# =========================


def momentum_ready_long(df: pd.DataFrame) -> bool:
    h = df["MACD_Hist"].tail(3)
    if len(h) < 3:
        return False

    is_rising = h.iloc[-1] > h.iloc[-2]
    was_rising = h.iloc[-2] > h.iloc[-3]

    hist_std = df["MACD_Hist"].rolling(20).std().iloc[-1]
    not_overextended = h.iloc[-1] < hist_std * 2

    return is_rising and was_rising and not_overextended


def momentum_ready_short(df: pd.DataFrame) -> bool:
    h = df["MACD_Hist"].tail(3)
    if len(h) < 3:
        return False

    is_falling = h.iloc[-1] < h.iloc[-2]
    was_falling = h.iloc[-2] < h.iloc[-3]

    hist_std = df["MACD_Hist"].rolling(20).std().iloc[-1]
    not_overextended = h.iloc[-1] > -hist_std * 2

    return is_falling and was_falling and not_overextended

# =========================
# SETUP LOGIC
# =========================


def evaluate_setup(df: pd.DataFrame):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest["Close"]
    ema20 = latest["EMA20"]
    ema50 = latest["EMA50"]

    # -----------------
    # LONG SETUP
    # -----------------
    if (
        ema20 > ema50 and
        ema50 > prev["EMA50"] and                 # EMA50 slope UP
        ema20 <= close <= ema20 * 1.02 and        # Buy zone
        45 <= latest["RSI"] <= 60 and
        momentum_ready_long(df)
    ):
        return {
            "Setup Type": "SETUP_LONG",
            "Notes": "Pullback into 20EMA"
        }

    # -----------------
    # SHORT SETUP
    # -----------------
    if (
        ema20 < ema50 and
        ema50 < prev["EMA50"] and                 # EMA50 slope DOWN
        ema20 * 0.98 <= close <= ema20 and        # Sell zone
        50 <= latest["RSI"] <= 65 and
        momentum_ready_short(df)
    ):
        return {
            "Setup Type": "SETUP_SHORT",
            "Notes": "Pullback into 20EMA"
        }

    return None

# =========================
# SCANNER
# =========================


def run_scanner():
    active_tickers = set(pd.read_csv(ACTIVE_TICKERS_PATH)
                         ["Ticker"].str.upper())
    logger.info(f"Loaded {len(active_tickers)} active tickers")

    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    logger.info(f"Found {len(raw_files)} raw data files")

    results = []
    rejection_counts = {}

    for file in raw_files:
        symbol = file.split("_")[0].upper()

        if symbol not in active_tickers:
            rejection_counts["inactive_ticker"] = rejection_counts.get(
                "inactive_ticker", 0) + 1
            continue

        path = os.path.join(RAW_DATA_DIR, file)
        df = pd.read_csv(path)

        if len(df) < 60:
            rejection_counts["insufficient_data"] = rejection_counts.get(
                "insufficient_data", 0) + 1
            continue

        df = add_indicators(df)

        setup = evaluate_setup(df)
        if not setup:
            rejection_counts["IGNORE"] = rejection_counts.get("IGNORE", 0) + 1
            continue

        latest = df.iloc[-1]

        results.append({
            "Symbol": symbol,
            "Setup Type": setup["Setup Type"],
            "Close": round(latest["Close"], 2),
            "EMA20": round(latest["EMA20"], 2),
            "EMA50": round(latest["EMA50"], 2),
            "RSI": round(latest["RSI"], 2),
            "ATR": round(latest["ATR"], 2),
            "Notes": setup["Notes"]
        })

    if results:
        df_out = pd.DataFrame(results).sort_values("Symbol")
        print(df_out.to_markdown(index=False))

        today = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(LOG_DIR, f"swing_setups_{today}.csv")
        df_out.to_csv(out_path, index=False)

        logger.info(f"Archive created: {out_path}")
    else:
        logger.warning("No valid swing setups found.")

    logger.info("Scanner rejection summary:")
    for k, v in rejection_counts.items():
        logger.info(f"  {k}: {v}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_scanner()
