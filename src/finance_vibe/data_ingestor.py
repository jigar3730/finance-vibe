import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime

# --- 1. PACKAGE IMPORT (Upgraded for Multi-Environment Paths) ---
try:
    # 1st Priority: Docker module structure (python -m src.finance_vibe.data_ingestor)
    from src.finance_vibe import config
except ImportError:
    try:
        # 2nd Priority: Flat direct package fallback
        from finance_vibe import config
    except ImportError:
        # 3rd Priority: Manual repository root insertion for local direct execution
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from src.finance_vibe import config


def _log_ingest_error(logs_dir: str, ticker: str, message: str) -> None:
    """Append a structured ingestion failure to ``ingest_errors_<date>.csv``."""
    stamp = datetime.now().strftime("%Y-%m-%d")
    err_path = os.path.join(logs_dir, f"ingest_errors_{stamp}.csv")
    row = pd.DataFrame(
        [{
            "Ticker": ticker,
            "Error": message,
            "Timestamp": datetime.now().isoformat(timespec="seconds"),
        }]
    )
    header = not os.path.exists(err_path)
    row.to_csv(err_path, mode="a", header=header, index=False)


def ingest_market_data(mode="weekly"):
    # --- 2. EXTRACT SETTINGS DYNAMICALLY FROM PROFILE ---
    mode_cfg = config.get_mode_config(mode)

    csv_path = config.TICKER_LIST_PATH
    raw_dir = mode_cfg['raw_dir']
    logs_dir = mode_cfg['logs_dir']
    PERIOD = mode_cfg['period']
    INTERVAL = mode_cfg['interval']

    if not os.path.exists(csv_path):
        print(f"❌ Could not find ticker list at {csv_path}. Please run ticker_provider.py first.")
        return

    # Ensure targeted sub-silo raw data directory exists
    os.makedirs(raw_dir, exist_ok=True)

    # Read tickers and drop any duplicates/NaNs
    tickers = pd.read_csv(csv_path)['Ticker'].dropna().unique().tolist()

    print(f"\n--- STEP 2: Ingesting [{mode.upper()}] {PERIOD} {INTERVAL} data ---")
    print(f"Target Directory: {raw_dir}")

    saved = 0
    rejected = 0

    for ticker in tickers:
        print(f"Processing {ticker:6}...", end=" ", flush=True)
        try:
            # --- 3. DOWNLOAD ---
            # auto_adjust=True handles splits/dividends for cleaner backtesting
            df = yf.download(ticker, period=PERIOD,
                             interval=INTERVAL, progress=False, auto_adjust=True)

            if df.empty:
                print("⚠️ No data found.")
                _log_ingest_error(logs_dir, ticker, "empty_download")
                rejected += 1
                continue

            # Flatten MultiIndex columns (common in newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # --- 4. DATA CLEANING ---
            # Remove the last row if it's an incomplete weekly candle (only for 1wk)
            if INTERVAL == "1wk" and len(df) > 0:
                last_date = df.index[-1]
                if hasattr(last_date, "weekday") and last_date.weekday() != 4:
                    df = df.iloc[:-1]

            # --- 4b. VALIDATE OHLCV CONTRACT (reject, never save partial) ---
            clean = config.validate_and_clean_ohlcv(df, require_volume=True)

            if len(clean) < config.MIN_SAVE_ROWS:
                print(f"⚠️ Only {len(clean)} valid rows (< {config.MIN_SAVE_ROWS}). Skipped.")
                _log_ingest_error(
                    logs_dir, ticker,
                    f"insufficient_rows:{len(clean)}<{config.MIN_SAVE_ROWS}"
                )
                rejected += 1
                continue

            # --- 5. SAVE ---
            save_path = config.get_raw_path(ticker, mode_cfg)
            clean.to_csv(save_path, index=False)
            print(f"✅ {os.path.basename(save_path)}")
            saved += 1

        except ValueError as e:
            # Schema/validation failure from validate_and_clean_ohlcv
            print(f"❌ Validation: {e}")
            _log_ingest_error(logs_dir, ticker, f"validation:{e}")
            rejected += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            _log_ingest_error(logs_dir, ticker, f"exception:{e}")
            rejected += 1

    print(f"\n📊 Ingestion summary: {saved} saved, {rejected} rejected.")
    if rejected:
        print(f"   Failure log: {os.path.join(logs_dir, 'ingest_errors_<date>.csv')}")


if __name__ == "__main__":
    # Check for CLI argument, otherwise default to weekly execution
    selected_mode = "weekly"
    if len(sys.argv) > 1:
        arg_mode = sys.argv[1].lower()
        if arg_mode in ["weekly", "daily"]:
            selected_mode = arg_mode
        else:
            print(f"⚠️ Unknown mode '{arg_mode}'. Defaulting to 'weekly'.")

    ingest_market_data(mode=selected_mode)