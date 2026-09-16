import time
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


# Batching cuts network round-trips from one-per-ticker to one-per-batch;
# the retry/backoff + timeout keep a single hung or throttled request from
# stalling the whole ingestion run instead of failing (and moving on) fast.
BATCH_SIZE = 50
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = 2.0
DOWNLOAD_TIMEOUT_SECONDS = 15


def _download_batch(tickers, period, interval, *, retries=DOWNLOAD_RETRIES,
                     backoff=DOWNLOAD_BACKOFF_SECONDS, timeout=DOWNLOAD_TIMEOUT_SECONDS):
    """Fetch many tickers in one yfinance call, with retry + backoff.

    Raises the last exception once retries are exhausted; the caller logs
    that as a batch-level failure for every ticker in the batch.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return yf.download(
                tickers, period=period, interval=interval,
                group_by="ticker", threads=True, progress=False,
                auto_adjust=True, timeout=timeout,
            )
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            sleep_s = backoff ** attempt
            print(f"⚠️ Batch download failed ({e}); retry {attempt}/{retries} in {sleep_s:.0f}s")
            time.sleep(sleep_s)
    raise last_err


def _ticker_frame(batch, ticker: str):
    """Extract one ticker's OHLCV frame from a batch keyed by ticker.

    Falls back to treating ``batch`` itself as the frame when yfinance
    collapses a single-ticker batch to flat (non-grouped) columns.
    """
    if isinstance(batch.columns, pd.MultiIndex):
        if ticker not in batch.columns.get_level_values(0):
            return None
        return batch[ticker]
    return batch


def ingest_market_data(mode="weekly", batch_size=BATCH_SIZE):
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

    for chunk_start in range(0, len(tickers), batch_size):
        chunk = tickers[chunk_start: chunk_start + batch_size]
        try:
            # --- 3. DOWNLOAD (whole batch, one round-trip) ---
            # auto_adjust=True handles splits/dividends for cleaner backtesting
            batch = _download_batch(chunk, PERIOD, INTERVAL)
        except Exception as e:
            # Batch failed after all retries: log once per ticker and move on
            # rather than letting one throttled/hung batch stall ingestion.
            print(f"❌ Batch [{chunk_start}:{chunk_start + len(chunk)}] failed after retries: {e}")
            for ticker in chunk:
                _log_ingest_error(logs_dir, ticker, f"batch_exception:{e}")
            rejected += len(chunk)
            continue

        for ticker in chunk:
            print(f"Processing {ticker:6}...", end=" ", flush=True)
            try:
                df = _ticker_frame(batch, ticker)

                if df is None or df.empty:
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