"""Trade plan generator: stock levels and options metadata from swing setups.

Reads ``swing_setups_<date>.csv`` and writes ``trade_plan_<date>.csv`` with
entry, stop, ATR targets, and LEAPS (weekly) or short-dated options (daily) fields.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

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
    print("⚠️ Unknown mode parsed to trade planner. Defaulting to 'weekly'.")
    mode = "weekly"

# --------- CONFIG ----------
DELTA_LONG = (0.65, 0.80)
DELTA_SHORT = (-0.80, -0.65)

# Short-dated options profiles vs long-dated LEAPS profiles.
_SHORT_DATED_MODES = {"daily", "high_beta"}

# Dynamic path resolution according to isolation architecture.
# high_beta gets its own log silo via config.get_log_dir.
SCANNER_DIR = Path(config.get_log_dir(mode))
SCANNER_PREFIX = "swing_setups_"
COILED_PREFIX = "coiled_cobra_setups_"
OUTPUT_PREFIX = "trade_plan_"

# --------- HELPER FUNCTIONS ----------


def _resolve_row_mode(row: Mapping[str, Any] | pd.Series, mode: str | None) -> str:
    """Resolve which swing profile governs a row.

    Row ``Mode`` is authoritative when present so a high_beta setup keeps its
    geometry even if the planner is invoked under a different CLI mode. An
    explicit ``mode`` argument (used by the backtest) still takes precedence.
    """
    candidate = (
        (mode or "").strip().lower()
        or str(row.get("Mode", "")).strip().lower()
        or globals().get("mode", "weekly")
    )
    return candidate if candidate in config.SWING_PROFILES else "weekly"


def calculate_stock_levels(
    row: Mapping[str, Any] | pd.Series, mode: str | None = None
) -> tuple[float, float, float, float, str, tuple[float, float]]:
    """Derive entry, stop, targets, option side, and delta band from one setup row.

    Quality swing geometry is mode-aware via ``config.get_swing_params``.
    The high_beta profile uses dual-constraint stops and true 1R/2R targets.
    Row ``Mode`` is authoritative unless an explicit ``mode`` is passed.
    """
    atr = float(row["ATR"])
    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    fib786 = row.get("Fib 78.6%", None)
    swing_low = row.get("Swing Low", None)
    swing_high = row.get("Swing High", None)
    source = str(row.get("Source", "swing")).strip().lower()
    setup_type = row["Setup Type"]

    resolved_mode = _resolve_row_mode(row, mode)
    sp = config.get_swing_params(resolved_mode)

    options_type = "CALL"
    delta_range = DELTA_LONG

    if setup_type == "SETUP_LONG" and source in {"coiled_cobra", "cobra"} and pd.notna(fib786):
        # Fib-anchored entry; stop uses triple-constraint rule:
        # local 10-session floor vs 1.5×ATR vs 5% of close (highest / tightest wins).
        # Year-long Fib levels must NOT widen the stop — only entry context.
        entry = max(float(fib786), close - 0.25 * atr)
        buf = 0.25 * atr
        if swing_low is not None and pd.notna(swing_low):
            structural = float(swing_low) - buf
        else:
            structural = entry - 1.5 * atr
        vol_floor = entry - 1.5 * atr
        price_floor = entry - config.MAX_RISK_PCT_OF_CLOSE * close
        stop = min(max(structural, vol_floor, price_floor), entry - buf)
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


def _export_levels(
    entry: float, stop: float, t1: float, t2: float, close: float, setup_type: str
) -> tuple[float, float, float, float, float]:
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


def _calculate_options_expiry() -> tuple[str, str]:
    """Return (min, max) expiry labels for the active planner mode.

    Weekly uses LEAPS (12–24 months); daily/high_beta use 1–3 month swings.
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


def generate_trade_plan(scanner_csv_path: str | Path | None = None) -> pd.DataFrame | None:
    """Build and export a trade plan CSV from the latest or provided scanner output."""
    print(
        f"--- STEP 5: Drafting Trade Execution Architectures [{mode.upper()} MODE] ---"
    )

    # Auto-detect latest scanner CSVs inside isolated subdirectory if none provided
    if scanner_csv_path is None:
        if not SCANNER_DIR.exists():
            print(f"⚠️ Target scanner directory empty or non-existent: {SCANNER_DIR}")
            return None

        # Find latest swing scanner file
        swing_files = list(SCANNER_DIR.glob(f"{SCANNER_PREFIX}*.csv"))
        swing_files.sort(key=lambda f: f.stem.split("_")[-1], reverse=True)
        swing_csv_path = swing_files[0] if swing_files else None

        # Find latest Coiled Cobra scanner file
        cobra_files = list(SCANNER_DIR.glob(f"{COILED_PREFIX}*.csv"))
        cobra_files.sort(key=lambda f: f.stem.split("_")[-1], reverse=True)
        cobra_csv_path = cobra_files[0] if cobra_files else None

        if swing_csv_path is None and cobra_csv_path is None:
            print(
                f"⚠️ No active setup archives discovered in {SCANNER_DIR}. Exiting plan generation."
            )
            return None

        print(f"Using swing scanner file: {swing_csv_path}")
        print(f"Using Coiled Cobra scanner file: {cobra_csv_path}")
    else:
        # Explicit single-source path provided (treated as a swing-style file)
        swing_csv_path = Path(scanner_csv_path)
        cobra_csv_path = None
        print(f"Using provided scanner file: {swing_csv_path}")

    # Load swing setups
    df_swing = None
    if swing_csv_path:
        df_swing = pd.read_csv(swing_csv_path)
        if "Source" not in df_swing.columns:
            df_swing["Source"] = "swing"
        print(f"Loaded {len(df_swing)} swing setups.")

    # Load Coiled Cobra setups
    df_cobra = None
    if cobra_csv_path:
        df_cobra = pd.read_csv(cobra_csv_path)
        if "Source" not in df_cobra.columns:
            df_cobra["Source"] = "coiled_cobra"
        print(f"Loaded {len(df_cobra)} Coiled Cobra setups.")

    # Combine into one DataFrame
    dfs = [df for df in [df_swing, df_cobra] if df is not None and not df.empty]
    if not dfs:
        print("⚠️ All setup archives are empty. Skipping calculations.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined total setups: {len(df)}")

    plan_rows = []
    short_dated = mode in _SHORT_DATED_MODES
    expiry_label_min = "Options Expiry Min" if short_dated else "LEAPS Expiry Min"
    expiry_label_max = "Options Expiry Max" if short_dated else "LEAPS Expiry Max"
    contract_label = "Options Type" if short_dated else "LEAPS Type"

    # Expiry window depends only on mode, so compute it once per run.
    expiry_min, expiry_max = _calculate_options_expiry()

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
                # Offline-model ranking signal (soft; null when no model ran)
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

    # Auto-generate output filename within isolated directory block context
    # Use the latest of the two detected files (swing or cobra)
    latest_file = swing_csv_path or cobra_csv_path
    if cobra_csv_path and swing_csv_path:
        swing_date = swing_csv_path.stem.split("_")[-1]
        cobra_date = cobra_csv_path.stem.split("_")[-1]
        latest_file = swing_csv_path if swing_date >= cobra_date else cobra_csv_path

    scanner_file = Path(latest_file)
    date_str = scanner_file.stem.split("_")[-1]
    output_csv_path = SCANNER_DIR / f"{OUTPUT_PREFIX}{date_str}.csv"

    os.makedirs(SCANNER_DIR, exist_ok=True)
    plan_df.to_csv(output_csv_path, index=False)
    print(f"✅ Trade plan exported successfully to: {output_csv_path}")
    return plan_df


# --------- USAGE ----------
if __name__ == "__main__":
    generate_trade_plan()
