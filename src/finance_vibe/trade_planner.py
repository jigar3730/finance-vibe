"""Trade plan generator for Coiled Cobra expansion levels.

Reads ``coiled_cobra_setups_<date>.csv`` and writes ``trade_plan_<date>.csv``
with Close / Coil_Low / 2R-3R geometry (spec: ``Coiled Cobra Rubric .MD``).
Quality-swing rows remain supported for offline ``pipeline_backtest`` only.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

try:
    from finance_vibe import config
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
    from finance_vibe import config

# =========================
# PROFILE CONFIGURATION
# =========================
if len(sys.argv) > 1 and sys.argv[1].lower() in ["weekly", "daily", "high_beta"]:
    mode = sys.argv[1].lower()
else:
    print(f"⚠️ Unknown mode parsed to trade planner. Defaulting to '{config.DEFAULT_MODE}'.")
    mode = config.DEFAULT_MODE

# --------- CONFIG ----------
DELTA_LONG = (0.65, 0.80)
DELTA_SHORT = (-0.80, -0.65)

# Short-dated options profiles vs long-dated LEAPS profiles.
_SHORT_DATED_MODES = {"daily", "high_beta"}

# Dynamic path resolution according to isolation architecture.
# high_beta gets its own log silo via config.get_log_dir.
BASE_DIR = Path(__file__).resolve().parents[2]
SCANNER_DIR = Path(config.get_log_dir(mode))
COILED_PREFIX = "coiled_cobra_setups_"
OUTPUT_PREFIX = "trade_plan_"

# --------- HELPER FUNCTIONS ----------


def _resolve_row_mode(row, mode: str | None) -> str:
    """Resolve which swing profile governs a row.

    Row ``Mode`` is authoritative when present so a high_beta setup keeps its
    geometry even if the planner is invoked under a different CLI mode. An
    explicit ``mode`` argument (used by the backtest) still takes precedence.
    """
    candidate = (
        (mode or "").strip().lower()
        or str(row.get("Mode", "")).strip().lower()
        or globals().get("mode", config.DEFAULT_MODE)
    )
    return candidate if candidate in config.SWING_PROFILES else config.DEFAULT_MODE


def calculate_stock_levels(row, mode: str | None = None):
    """Derive entry, stop, targets, option side, and delta band from one setup row.

    Coiled Cobra uses expansion geometry: enter at Close, protect Coil_Low
    (else Swing Low) with a triple-constraint stop, targets at 2R / 3R.
    Quality-swing rows still use ``config.compute_swing_levels`` for offline studies.
    """
    atr = float(row["ATR"])
    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    swing_low = row.get("Swing Low", None)
    swing_high = row.get("Swing High", None)
    coil_low = row.get("Coil_Low", None)
    source = str(row.get("Source", "swing")).strip().lower()
    setup_type = row["Setup Type"]

    resolved_mode = _resolve_row_mode(row, mode)
    sp = config.get_swing_params(resolved_mode)

    options_type = "CALL"
    delta_range = DELTA_LONG

    if setup_type == "SETUP_LONG" and source in {"coiled_cobra", "cobra"}:
        # Expansion: buy the close of a passing coil. Fib is context only.
        entry = close
        buf = 0.25 * atr
        if coil_low is not None and pd.notna(coil_low):
            floor = float(coil_low)
        elif swing_low is not None and pd.notna(swing_low):
            floor = float(swing_low)
        else:
            floor = entry - 1.5 * atr
        structural = floor - buf
        vol_floor = entry - 1.5 * atr
        price_floor = entry - config.MAX_RISK_PCT_OF_CLOSE * close
        stop = min(max(structural, vol_floor, price_floor), entry - buf)
        if stop >= entry:
            stop = entry - buf
        risk = abs(entry - stop)
        target1 = entry + 2.0 * risk
        target2 = entry + 3.0 * risk
        return entry, stop, target1, target2, options_type, delta_range

    if setup_type not in ("SETUP_LONG", "SETUP_SHORT"):
        raise ValueError(f"Unknown Setup Type: {setup_type}")

    levels = config.compute_swing_levels(
        setup_type=setup_type, close=close, ema20=ema20, ema50=ema50, atr=atr,
        swing_low=swing_low, swing_high=swing_high, sp=sp,
    )
    if setup_type == "SETUP_SHORT":
        options_type = "PUT"
        delta_range = DELTA_SHORT

    return (
        levels["entry"], levels["stop"], levels["target1"], levels["target2"],
        options_type, delta_range,
    )


def _export_levels(entry: float, stop: float, t1: float, t2: float, close: float, setup_type: str):
    """Round levels for CSV export while keeping risk ≤ MAX_RISK_PCT_OF_CLOSE × close."""
    entry_r = round(entry, 2)
    close_v = float(close) if pd.notna(close) else entry_r
    max_risk = config.MAX_RISK_PCT_OF_CLOSE * close_v if close_v > 0 else None
    is_long = str(setup_type).upper() != "SETUP_SHORT"

    if max_risk is not None and max_risk > 0:
        if is_long:
            stop_floor = entry_r - max_risk
            stop_r = round(max(stop, stop_floor), 2)
            if entry_r - stop_r > max_risk + 1e-9:
                stop_r = round(entry_r - max_risk, 2)
            if entry_r - stop_r > max_risk + 1e-9:
                stop_r = round(stop_r + 0.01, 2)
            # Preserve R-multiple targets when geometry was risk-based (T1 ≈ entry+2R).
            risk = entry_r - stop_r
            if risk > 0 and abs((t1 - entry) - 2.0 * (entry - stop)) < 1e-6:
                t1, t2 = entry_r + 2.0 * risk, entry_r + 3.0 * risk
        else:
            stop_ceil = entry_r + max_risk
            stop_r = round(min(stop, stop_ceil), 2)
            if stop_r - entry_r > max_risk + 1e-9:
                stop_r = round(entry_r + max_risk, 2)
            if stop_r - entry_r > max_risk + 1e-9:
                stop_r = round(stop_r - 0.01, 2)
            risk = stop_r - entry_r
            if risk > 0 and abs((entry - t1) - 2.0 * (stop - entry)) < 1e-6:
                t1, t2 = entry_r - 2.0 * risk, entry_r - 3.0 * risk
    else:
        stop_r = round(stop, 2)

    return entry_r, stop_r, round(t1, 2), round(t2, 2), round(abs(entry_r - stop_r), 2)


def calculate_options_expiry():
    """
    Dynamically adjusts structural contracts option timeline based on profile timeframe mode.
    Weekly pulls long-term LEAPS setups; daily pulls agile swing cycles.
    """
    today = datetime.today()
    if mode in _SHORT_DATED_MODES:
        # Standard swing option cycle boundaries (1 to 3 months forward lookahead)
        expiry_min = today + pd.DateOffset(months=1)
        expiry_max = today + pd.DateOffset(months=3)
    else:
        # Legacy LEAPS macro cycles (12 to 24 months forward lookahead)
        expiry_min = today + pd.DateOffset(months=12)
        expiry_max = today + pd.DateOffset(months=24)

    return expiry_min.strftime("%b-%Y"), expiry_max.strftime("%b-%Y")


# --------- MAIN FUNCTION ----------


def generate_trade_plan(scanner_csv_path=None):
    """Build and export a trade plan CSV from the latest or provided scanner output."""
    print(
        f"--- STEP 5: Drafting Trade Execution Architectures [{mode.upper()} MODE] ---"
    )

    # Auto-detect latest Coiled Cobra scanner CSV. Explicit path still accepted
    # (offline swing studies can pass a swing_setups file directly).
    if scanner_csv_path is None:
        if not SCANNER_DIR.exists():
            print(f"⚠️ Target scanner directory empty or non-existent: {SCANNER_DIR}")
            return None

        cobra_files = list(SCANNER_DIR.glob(f"{COILED_PREFIX}*.csv"))
        cobra_files.sort(key=lambda f: f.stem.split("_")[-1], reverse=True)
        if not cobra_files:
            print(
                f"⚠️ No {COILED_PREFIX}*.csv archive in {SCANNER_DIR}. Exiting plan generation."
            )
            return None
        scanner_file = cobra_files[0]
        print(f"Using Coiled Cobra scanner file: {scanner_file}")
    else:
        scanner_file = Path(scanner_csv_path)
        print(f"Using provided scanner file: {scanner_file}")

    df = pd.read_csv(scanner_file)
    if "Source" not in df.columns:
        df["Source"] = "coiled_cobra"
    if df.empty:
        print("⚠️ Setup archive is empty. Skipping calculations.")
        return None
    print(f"Loaded {len(df)} setup(s).")

    plan_rows = []
    short_dated = mode in _SHORT_DATED_MODES
    expiry_label_min = "Options Expiry Min" if short_dated else "LEAPS Expiry Min"
    expiry_label_max = "Options Expiry Max" if short_dated else "LEAPS Expiry Max"
    contract_label = "Options Type" if short_dated else "LEAPS Type"

    # Expiry window depends only on mode, so compute it once per run.
    expiry_min, expiry_max = calculate_options_expiry()

    for _, row in df.iterrows():
        # Row Mode is authoritative (mode=None) so high_beta setups keep their
        # geometry even when the planner runs under a different CLI mode.
        entry, stop, t1, t2, opt_type, delta_range = calculate_stock_levels(row, mode=None)
        entry_r, stop_r, t1_r, t2_r, risk_per_share = _export_levels(
            entry, stop, t1, t2, row.get("Close", entry), row["Setup Type"],
        )

        plan_rows.append(
            {
                "Symbol": row["Symbol"],
                "Setup Type": row["Setup Type"],
                "Source": row.get("Source", None),
                "Mode": row.get("Mode", None),
                "AsOf Date": row.get("AsOf Date", None),
                "Stock Entry": entry_r,
                "Stock Stop": stop_r,
                "Target 1": t1_r,
                "Target 2": t2_r,
                "Risk Per Share": risk_per_share,
                # Pass-through structural context that justified the levels
                "Close": row.get("Close", None),
                "EMA20": row.get("EMA20", None),
                "EMA50": row.get("EMA50", None),
                "ATR": row.get("ATR", None),
                "RSI": row.get("RSI", None),
                "Swing Low": row.get("Swing Low", None),
                "Swing High": row.get("Swing High", None),
                "Fib 61.8%": row.get("Fib 61.8%", None),
                "Fib 78.6%": row.get("Fib 78.6%", None),
                "Score": row.get("Score", None),
                "Grade": row.get("Grade", None),
                "Checks Met": row.get("Checks Met", None),
                "Volume_Shelf": row.get("Volume_Shelf", None),
                "MACD_Compression": row.get("MACD_Compression", None),
                "Structure": row.get("Structure", None),
                "RS_Score": row.get("RS_Score", None),
                "Coil_Width": row.get("Coil_Width", None),
                "MACD_Cross": row.get("MACD_Cross", None),
                "Fib_Bonus": row.get("Fib_Bonus", None),
                "MACD_Spread_ATR": row.get("MACD_Spread_ATR", None),
                "Coil_Width_ATR": row.get("Coil_Width_ATR", None),
                "Coil_High": row.get("Coil_High", None),
                "Coil_Low": row.get("Coil_Low", None),
                "ML_Pred_Return": row.get("ML_Pred_Return", None),
                "ML_Rank": row.get("ML_Rank", None),
                "Notes": row.get("Notes", None),
                # Options / LEAPS metadata (now persisted, previously dropped)
                contract_label: opt_type,
                "Delta Min": delta_range[0],
                "Delta Max": delta_range[1],
                expiry_label_min: expiry_min,
                expiry_label_max: expiry_max,
            }
        )

    plan_df = pd.DataFrame(plan_rows)

    date_str = scanner_file.stem.split("_")[-1]
    output_csv_path = SCANNER_DIR / f"{OUTPUT_PREFIX}{date_str}.csv"

    os.makedirs(SCANNER_DIR, exist_ok=True)
    plan_df.to_csv(output_csv_path, index=False)
    print(f"✅ Trade plan exported successfully to: {output_csv_path}")
    return plan_df


# --------- USAGE ----------
if __name__ == "__main__":
    generate_trade_plan()
