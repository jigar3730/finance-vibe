"""Trade plan generator for Coiled Cobra expansion levels.

Reads ``coiled_cobra_setups_<date>.csv`` and writes ``trade_plan_<date>.csv``
with Close / Coil_Low / 2R-3R geometry (spec: ``Coiled Cobra Rubric .MD``).
Quality-swing rows remain supported for offline ``pipeline_backtest`` only.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
    print(f"⚠️ Unknown mode parsed to trade planner. Defaulting to '{config.DEFAULT_MODE}'.")
    mode = config.DEFAULT_MODE

# high_beta gets its own log silo via config.get_log_dir.
SCANNER_DIR = Path(config.get_log_dir(mode))
COILED_PREFIX = "coiled_cobra_setups_"
OUTPUT_PREFIX = "trade_plan_"

# Kept for calculate_stock_levels return tuple (tests + swing backtest unpack).
# Not written to live trade_plan CSVs — stock expansion only.
DELTA_LONG = (0.65, 0.80)
DELTA_SHORT = (-0.80, -0.65)


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


# --------- RANK / GUARDRAILS (formerly trade_plan_helper) ----------

MAX_RISK_PCT_OF_CLOSE = config.MAX_RISK_PCT_OF_CLOSE
TIGHT_COIL_PROPENSITY = 1.25
TIGHT_COIL_WIDTH_ATR = 4.0
TIGHT_RISK_PCT = 0.03

PLAN_EXPORT_COLUMNS = [
    "Symbol",
    "Setup Type",
    "Source",
    "Mode",
    "AsOf Date",
    "Score",
    "Grade",
    "Checks Met",
    "Close",
    "Stock Entry",
    "Stock Stop",
    "Target 1",
    "Target 2",
    "Risk Per Share",
    "R:R T1",
    "R:R T2",
    "ML_Pred_Return",
    "ML_Rank",
    "Expected Value",
    "Priority",
    "ATR",
    "RSI",
    "Fib 78.6%",
    "Coil_High",
    "Coil_Low",
    "Coil_Width_ATR",
    "MACD_Spread_ATR",
    "Volume_Shelf",
    "MACD_Compression",
    "Structure",
    "RS_Score",
    "Coil_Width",
    "Proximity_Highs",
    "MACD_Cross",
    "MACD_Crossed",
    "Weekly_Coil_Pass",
    "Weekly_Score",
    "Fib_Bonus",
]


def _count_true(mask: pd.Series) -> int:
    if mask is None or len(mask) == 0:
        return 0
    return int(np.asarray(mask.fillna(False), dtype=bool).sum())


def add_rr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Direction-aware R:R from entry / stop / targets."""
    out = df.copy()
    if "Setup Type" in out.columns:
        is_long = out["Setup Type"].astype(str).str.upper() != "SETUP_SHORT"
    else:
        is_long = pd.Series(True, index=out.index)

    out["Risk Per Share"] = (out["Stock Entry"] - out["Stock Stop"]).abs()
    reward_t1 = np.where(
        is_long, out["Target 1"] - out["Stock Entry"], out["Stock Entry"] - out["Target 1"]
    )
    reward_t2 = np.where(
        is_long, out["Target 2"] - out["Stock Entry"], out["Stock Entry"] - out["Target 2"]
    )
    safe_risk = out["Risk Per Share"].replace(0, pd.NA)
    out["R:R T1"] = (pd.Series(reward_t1, index=out.index, dtype="float") / safe_risk.astype(float)).round(2)
    out["R:R T2"] = (pd.Series(reward_t2, index=out.index, dtype="float") / safe_risk.astype(float)).round(2)
    return out


def apply_ingestion_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows whose risk exceeds 5% of close."""
    n0 = len(df)
    stats = {"input": n0, "risk_pct": 0, "kept": 0}

    out = df.copy()
    if out.empty:
        return out, stats

    price = pd.to_numeric(
        out["Close"] if "Close" in out.columns else out.get("Stock Entry"),
        errors="coerce",
    )
    if "Stock Entry" in out.columns and price is not None:
        entry = pd.to_numeric(out["Stock Entry"], errors="coerce")
        price = price.fillna(entry) if hasattr(price, "fillna") else entry

    risk = pd.to_numeric(out.get("Risk Per Share"), errors="coerce")
    if price is not None and risk is not None:
        risk_pct = risk / price.replace(0, np.nan)
        mask_risk = (risk_pct > MAX_RISK_PCT_OF_CLOSE).fillna(False)
        stats["risk_pct"] = _count_true(mask_risk)
        out = out.loc[~mask_risk].copy()

    stats["kept"] = len(out)
    return out, stats


def rank_by_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by ML predicted return, else Score, with a tight-coil boost."""
    out = df.copy()
    rr2 = pd.to_numeric(out.get("R:R T2"), errors="coerce").fillna(0.0)
    score = pd.to_numeric(out.get("Score"), errors="coerce").fillna(0.0)
    out["Expected Value"] = (rr2 * score).round(2)

    price = pd.to_numeric(
        out["Close"] if "Close" in out.columns else out.get("Stock Entry"),
        errors="coerce",
    )
    risk = pd.to_numeric(out.get("Risk Per Share"), errors="coerce")
    tight_risk = (risk / price.replace(0, np.nan)) <= TIGHT_RISK_PCT if price is not None else False
    if "Coil_Width_ATR" in out.columns:
        width = pd.to_numeric(out["Coil_Width_ATR"], errors="coerce")
        tight_width = width <= TIGHT_COIL_WIDTH_ATR
    else:
        tight_width = pd.Series(False, index=out.index)
    propensity = np.where(
        np.asarray(tight_width.fillna(False)) | np.asarray(
            tight_risk.fillna(False) if hasattr(tight_risk, "fillna") else tight_risk
        ),
        TIGHT_COIL_PROPENSITY,
        1.0,
    )

    if "ML_Pred_Return" in out.columns:
        ml_pred = pd.to_numeric(out["ML_Pred_Return"], errors="coerce")
    else:
        ml_pred = pd.Series(np.nan, index=out.index, dtype="float64")

    if ml_pred.notna().any():
        weekly_boost = np.ones(len(out), dtype=float)
        if "Weekly_Coil_Pass" in out.columns:
            weekly_flag = pd.to_numeric(out["Weekly_Coil_Pass"], errors="coerce").fillna(0)
            weekly_boost = np.where(weekly_flag.eq(1), TIGHT_COIL_PROPENSITY, 1.0)
        out["Priority"] = (ml_pred * propensity * weekly_boost).round(4)
    else:
        out["Priority"] = (score * propensity).round(2)

    return out.sort_values(
        "Priority", ascending=False, na_position="last", kind="mergesort"
    ).reset_index(drop=True)


def slim_plan_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in PLAN_EXPORT_COLUMNS if c in df.columns]
    return df[keep].copy() if keep else df.copy()


def finalize_trade_plan(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """R:R, 5% risk filter, ML/Score rank, slim columns."""
    out = add_rr_columns(df)
    out, stats = apply_ingestion_filters(out)
    if not out.empty:
        out = rank_by_expected_value(out)
    return slim_plan_columns(out), stats


def resolve_trade_plan_path(mode: str | None = None, *, today: str | None = None) -> tuple[Path, Path]:
    """Locate ``trade_plan_{date}.csv`` (ignores legacy ``trade_plan_clean_*``)."""
    mode = mode or config.DEFAULT_MODE
    today_str = today or datetime.now().strftime("%Y-%m-%d")
    filename = f"trade_plan_{today_str}.csv"
    base_dir = Path(__file__).resolve().parents[2]

    possible_dirs = [
        base_dir / "data" / "logs" / mode,
        Path(f"./data/logs/{mode}"),
        Path("/app/data/logs") / mode,
        Path("data/logs") / mode,
        base_dir / "data" / "logs",
        Path("./data/logs"),
        Path("/app/data/logs"),
        Path("data/logs"),
    ]

    for p_dir in possible_dirs:
        check_path = p_dir / filename
        if check_path.exists():
            return p_dir, check_path

    for p_dir in possible_dirs:
        if not p_dir.exists():
            continue
        candidates = sorted(
            (f for f in p_dir.glob("trade_plan_*.csv") if "clean" not in f.stem),
            key=lambda f: f.stem.split("_")[-1],
            reverse=True,
        )
        if candidates:
            return p_dir, candidates[0]

    raise FileNotFoundError(
        f"{filename} not found (mode={mode}); checked data/logs/{mode}/ and legacy data/logs/"
    )


def process_trade_plan(mode: str | None = None, *, today: str | None = None) -> Path:
    """Re-finalize an existing plan CSV in place (tests / one-off repair)."""
    mode = mode or config.DEFAULT_MODE
    today_str = today or datetime.now().strftime("%Y-%m-%d")
    trade_plan_dir, scanner_csv = resolve_trade_plan_path(mode, today=today_str)
    df = pd.read_csv(scanner_csv)
    df.columns = df.columns.str.strip()
    numeric_cols = ["Stock Entry", "Stock Stop", "Target 1", "Target 2", "Close", "Score"]
    for col in [c for c in numeric_cols if c in df.columns]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r"[$,]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df, stats = finalize_trade_plan(df)
    print(
        f"   kept {stats['kept']}/{stats['input']} "
        f"(dropped risk={stats['risk_pct']})"
    )
    df.to_csv(scanner_csv, index=False)
    return scanner_csv


# --------- MAIN FUNCTION ----------


def generate_trade_plan(scanner_csv_path=None):
    """Build ranked expansion plan CSV from the latest or provided scanner output."""
    print(f"--- Trade plan [{mode.upper()}] ---")

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

    for _, row in df.iterrows():
        # Row Mode is authoritative (mode=None) so high_beta setups keep their
        # geometry even when the planner runs under a different CLI mode.
        entry, stop, t1, t2, _opt_type, _delta_range = calculate_stock_levels(row, mode=None)
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
                "MACD_Crossed": row.get("MACD_Crossed", None),
                "Weekly_Coil_Pass": row.get("Weekly_Coil_Pass", None),
                "Weekly_Score": row.get("Weekly_Score", None),
                "Fib_Bonus": row.get("Fib_Bonus", None),
                "MACD_Spread_ATR": row.get("MACD_Spread_ATR", None),
                "Coil_Width_ATR": row.get("Coil_Width_ATR", None),
                "Coil_High": row.get("Coil_High", None),
                "Coil_Low": row.get("Coil_Low", None),
                "ML_Pred_Return": row.get("ML_Pred_Return", None),
                "ML_Rank": row.get("ML_Rank", None),
                "Notes": row.get("Notes", None),
            }
        )

    plan_df = pd.DataFrame(plan_rows)
    plan_df, stats = finalize_trade_plan(plan_df)
    print(
        f"   ranked {stats['kept']}/{stats['input']} "
        f"(dropped risk={stats['risk_pct']})"
    )

    date_str = scanner_file.stem.split("_")[-1]
    output_csv_path = SCANNER_DIR / f"{OUTPUT_PREFIX}{date_str}.csv"

    os.makedirs(SCANNER_DIR, exist_ok=True)
    plan_df.to_csv(output_csv_path, index=False)
    print(f"Trade plan: {output_csv_path}")
    return plan_df


# --------- USAGE ----------
if __name__ == "__main__":
    generate_trade_plan()
