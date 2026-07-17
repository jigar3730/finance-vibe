"""Quality swing setup scanner for Finance Vibe.

Detects high-probability SETUP_LONG / SETUP_SHORT pullbacks: regime filter
(EMA100), tight EMA20 location, RSI band, early MACD-histogram turn, held
swing structure, and next-bar confirmation. Output:
``data/logs/{mode}/swing_setups_<date>.csv``.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import pandas_ta as ta

try:
    from finance_vibe import config
    from finance_vibe.analysis_engine import (
        build_features,
        load_benchmark_frame,
        relative_strength,
        market_regime_ok,
        score_last_row,
    )
except ImportError:
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")))
    from finance_vibe import config
    from finance_vibe.analysis_engine import (
        build_features,
        load_benchmark_frame,
        relative_strength,
        market_regime_ok,
        score_last_row,
    )

# =========================
# PROFILE CONFIGURATION
# =========================
if len(sys.argv) > 1 and sys.argv[1].lower() in ["weekly", "daily", "high_beta"]:
    mode = sys.argv[1].lower()
else:
    print("⚠️ Unknown mode parsed to scanner. Defaulting to 'weekly'.")
    mode = "weekly"

# Data timeframe may differ from swing profile (high_beta → daily OHLCV)
_data_mode, _swing_profile = config.resolve_pipeline_mode(mode)
mode = _swing_profile  # scanner/planner Mode column = swing profile

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", _data_mode)
ACTIVE_TICKERS_PATH = os.path.join(BASE_DIR, "data", "active_tickers.csv")
# high_beta writes to its own log silo (config.get_log_dir) to avoid colliding
# with the ETF daily pipeline that shares the same raw data.
LOG_DIR = config.get_log_dir(mode)

os.makedirs(LOG_DIR, exist_ok=True)

# Cache of benchmark frames keyed by (benchmark, data_mode) so relative-strength
# and market-regime checks do not reload the CSV per bar during scans/backtests.
_BENCH_CACHE: dict = {}


def get_benchmark_frame(sp: dict, data_mode: str):
    """Return the cached indicator-enriched benchmark frame for a profile."""
    bench = sp.get("benchmark")
    if not bench:
        return None
    key = (bench, data_mode)
    if key not in _BENCH_CACHE:
        _BENCH_CACHE[key] = load_benchmark_frame(bench, data_mode)
    return _BENCH_CACHE[key]

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

def add_indicators(df: pd.DataFrame, mode: str = "weekly") -> pd.DataFrame:
    """EMA trend stack, MACD hist, RSI, ATR; drop rows with NaN core indicators."""
    sp = config.get_swing_params(mode)
    out = df.copy()
    out["EMA20"] = ta.ema(out["Close"], length=20)
    out["EMA50"] = ta.ema(out["Close"], length=50)
    out["EMA100"] = ta.ema(out["Close"], length=100)

    macd = ta.macd(out["Close"])
    out["MACD_Hist"] = macd["MACDh_12_26_9"]

    out["RSI"] = ta.rsi(out["Close"], length=14)
    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=14)

    n = sp["structure_bars"]
    out["SwingLow"] = out["Low"].rolling(n, min_periods=n).min()
    out["SwingHigh"] = out["High"].rolling(n, min_periods=n).max()

    out.dropna(subset=["EMA20", "EMA50", "EMA100", "MACD_Hist", "RSI", "ATR"], inplace=True)
    return out


# =========================
# MOMENTUM FILTER
# =========================

def momentum_ready_long(df: pd.DataFrame) -> bool:
    """MACD hist rising two bars while still <= 0 (early recovery)."""
    h = df["MACD_Hist"].tail(3)
    if len(h) < 3:
        return False

    is_rising = h.iloc[-1] > h.iloc[-2]
    was_rising = h.iloc[-2] > h.iloc[-3]
    still_compressed = h.iloc[-1] <= 0

    hist_std = df["MACD_Hist"].rolling(20).std().iloc[-1]
    if pd.isna(hist_std):
        return False
    not_overextended = h.iloc[-1] < hist_std * 2

    return is_rising and was_rising and still_compressed and not_overextended


def momentum_ready_short(df: pd.DataFrame) -> bool:
    """MACD hist falling two bars while still >= 0 (early fade)."""
    h = df["MACD_Hist"].tail(3)
    if len(h) < 3:
        return False

    is_falling = h.iloc[-1] < h.iloc[-2]
    was_falling = h.iloc[-2] < h.iloc[-3]
    still_elevated = h.iloc[-1] >= 0

    hist_std = df["MACD_Hist"].rolling(20).std().iloc[-1]
    if pd.isna(hist_std):
        return False
    not_overextended = h.iloc[-1] > -hist_std * 2

    return is_falling and was_falling and still_elevated and not_overextended


def _structure_tolerance(latest, structure_slack_atr) -> float:
    """Absolute price tolerance for the structure hold.

    ATR-normalized when ``structure_slack_atr`` is set (scales across low- and
    high-volatility names); otherwise a fixed 0.2% band (legacy behavior).
    """
    if structure_slack_atr is not None and "ATR" in latest:
        return float(structure_slack_atr) * float(latest["ATR"])
    return None


def _structure_held_long(df: pd.DataFrame, structure_bars: int, structure_slack_atr=None) -> bool:
    """Pullback held above the prior swing low (no breakdown of local structure)."""
    if len(df) < structure_bars + 1:
        return False
    latest = df.iloc[-1]
    prior = df.iloc[-(structure_bars + 1):-1]
    swing_low = float(prior["Low"].min())
    tol = _structure_tolerance(latest, structure_slack_atr)
    floor = swing_low - tol if tol is not None else swing_low * 0.998
    return float(latest["Low"]) >= floor


def _structure_held_short(df: pd.DataFrame, structure_bars: int, structure_slack_atr=None) -> bool:
    """Rally held below the prior swing high (no breakout of local structure)."""
    if len(df) < structure_bars + 1:
        return False
    latest = df.iloc[-1]
    prior = df.iloc[-(structure_bars + 1):-1]
    swing_high = float(prior["High"].max())
    tol = _structure_tolerance(latest, structure_slack_atr)
    ceil = swing_high + tol if tol is not None else swing_high * 1.002
    return float(latest["High"]) <= ceil


# =========================
# SETUP LOGIC
# =========================

def _near_ema20_long(close: float, ema20: float, atr: float, sp: dict) -> bool:
    """Price at/above EMA20 within proximity band (ATR or %)."""
    if sp.get("prox_atr") is not None:
        return ema20 <= close <= ema20 + float(sp["prox_atr"]) * atr
    prox = float(sp["prox_pct"])
    return ema20 <= close <= ema20 * (1.0 + prox)


def _near_ema20_short(close: float, ema20: float, atr: float, sp: dict) -> bool:
    """Price at/below EMA20 within proximity band (ATR or %)."""
    if sp.get("prox_atr") is not None:
        return ema20 - float(sp["prox_atr"]) * atr <= close <= ema20
    prox = float(sp["prox_pct"])
    return ema20 * (1.0 - prox) <= close <= ema20


def evaluate_setup(df: pd.DataFrame, mode: str = "weekly") -> dict | None:
    """Return quality SETUP_LONG/SHORT dict or None (evaluated on *df*'s last bar)."""
    if len(df) < 3:
        return None

    sp = config.get_swing_params(mode)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema100 = float(latest["EMA100"])
    atr = float(latest["ATR"])
    rsi = float(latest["RSI"])
    n = sp["structure_bars"]
    slack = sp.get("structure_slack_atr")
    stack_ok_long = (not sp.get("require_ema_stack")) or (ema20 > ema50 > ema100)
    stack_ok_short = (not sp.get("require_ema_stack")) or (ema20 < ema50 < ema100)

    # --- QUALITY LONG ---
    if (
        ema20 > ema50
        and ema50 > float(prev["EMA50"])
        and close > ema100
        and stack_ok_long
        and _near_ema20_long(close, ema20, atr, sp)
        and sp["rsi_min_long"] <= rsi <= sp["rsi_max_long"]
        and momentum_ready_long(df)
        and _structure_held_long(df, n, slack)
    ):
        swing_low = float(df.iloc[-(n + 1):-1]["Low"].min())
        return {
            "Setup Type": "SETUP_LONG",
            "Notes": "Quality pullback: regime+EMA20+MACD turn+structure",
            "Swing Low": round(swing_low, 2),
            "Swing High": None,
        }

    # Long-only profiles (e.g. high_beta) never emit shorts.
    if sp.get("long_only"):
        return None

    # --- QUALITY SHORT ---
    if (
        ema20 < ema50
        and ema50 < float(prev["EMA50"])
        and close < ema100
        and stack_ok_short
        and _near_ema20_short(close, ema20, atr, sp)
        and sp["rsi_min_short"] <= rsi <= sp["rsi_max_short"]
        and momentum_ready_short(df)
        and _structure_held_short(df, n, slack)
    ):
        swing_high = float(df.iloc[-(n + 1):-1]["High"].max())
        return {
            "Setup Type": "SETUP_SHORT",
            "Notes": "Quality pullback short: regime+EMA20+MACD fade+structure",
            "Swing Low": None,
            "Swing High": round(swing_high, 2),
        }

    return None


def _confirm_setup(
    setup: dict,
    confirm_bar: pd.Series,
    setup_bar: pd.Series,
    *,
    confirm_slack_atr: float = 0.0,
) -> bool:
    """Next-bar confirmation: hold the pullback and reclaim direction."""
    close = float(confirm_bar["Close"])
    ema20 = float(confirm_bar["EMA20"])
    atr = float(confirm_bar["ATR"])
    slack = float(confirm_slack_atr) * atr
    if setup["Setup Type"] == "SETUP_LONG":
        # Confirm: close back above EMA20; allow limited undercut of setup low
        return close >= ema20 and close >= float(setup_bar["Low"]) - slack
    # Short: close back below EMA20; allow limited overshoot of setup high
    return close <= ema20 and close <= float(setup_bar["High"]) + slack


def _soft_vibe_gate(setup_type: str, sp: dict, df: pd.DataFrame):
    """Directional soft Vibe gate. Returns (passed, score).

    Longs require ``score >= vibe_min``; shorts require ``score <= short_max_vibe``
    (a bullish floor must never gate a short). Profiles without a ``vibe_min``
    skip the gate entirely.
    """
    if sp.get("vibe_min") is None:
        return True, None
    try:
        feat = build_features(df)
        score = score_last_row(feat.iloc[-1])
    except Exception:
        return False, None

    if setup_type == "SETUP_LONG":
        return score >= sp["vibe_min"], score
    # SETUP_SHORT
    short_max = sp.get("short_max_vibe")
    if short_max is None:
        return False, score  # shorts disabled under this vibe profile
    return score <= short_max, score


def detect_setup_at_bar(
    df: pd.DataFrame,
    symbol: str,
    mode: str = "weekly",
    benchmark_df: "pd.DataFrame | None" = None,
) -> dict | None:
    """Quality swing row: setup on prior bar + confirmation on latest bar.

    ``benchmark_df`` (indicator-enriched) enables market-regime and
    relative-strength gating for profiles that require them (e.g. high_beta).
    """
    if len(df) < 60:
        return None

    sp = config.get_swing_params(mode)
    indicated = add_indicators(df.copy(), mode=mode)
    if len(indicated) < 3:
        return None

    setup_slice = indicated.iloc[:-1]
    setup = evaluate_setup(setup_slice, mode)
    if not setup:
        return None

    setup_bar = setup_slice.iloc[-1]
    confirm_bar = indicated.iloc[-1]
    if not _confirm_setup(
        setup, confirm_bar, setup_bar,
        confirm_slack_atr=float(sp.get("confirm_slack_atr") or 0.0),
    ):
        return None

    # Directional soft Vibe gate (daily / high_beta).
    passed, vibe_score = _soft_vibe_gate(setup["Setup Type"], sp, df)
    if not passed:
        return None

    confirm_date = confirm_bar["Date"] if "Date" in indicated.columns else None

    # Market regime + relative strength (real cross-asset context, no lookahead).
    regime_ok = None
    rs_63d = None
    if benchmark_df is not None and (
        sp.get("require_market_regime") or sp.get("require_relative_strength")
    ):
        if sp.get("require_market_regime"):
            regime_ok = market_regime_ok(benchmark_df, confirm_date)
            if not regime_ok:
                return None
        if sp.get("require_relative_strength"):
            rs_ok, rs_63d = relative_strength(
                df, benchmark_df, as_of=confirm_date,
                lookback=int(sp["rs_lookback"]),
                ratio_ma_bars=int(sp["rs_ratio_ma_bars"]),
            )
            if not rs_ok:
                return None

    # Structural-risk rejection: reject setups whose risk is out of bounds
    # instead of squeezing the stop into invalid structure.
    levels = config.compute_swing_levels(
        setup_type=setup["Setup Type"],
        close=float(confirm_bar["Close"]),
        ema20=float(confirm_bar["EMA20"]),
        ema50=float(confirm_bar["EMA50"]),
        atr=float(confirm_bar["ATR"]),
        swing_low=setup.get("Swing Low"),
        swing_high=setup.get("Swing High"),
        sp=sp,
    )
    if levels["reject_reason"]:
        return None

    asof = str(confirm_date)[:10] if confirm_date is not None and pd.notna(confirm_date) else None

    notes = setup["Notes"] + " (confirmed)"
    if vibe_score is not None:
        notes += f" vibe={vibe_score}"
    if rs_63d is not None:
        notes += f" rs63={rs_63d}"

    row = config.blank_setup_row()
    row.update({
        "Symbol": symbol.upper(),
        "Setup Type": setup["Setup Type"],
        "Source": "swing",
        "Mode": mode,
        "AsOf Date": asof,
        "Close": round(float(confirm_bar["Close"]), 2),
        "EMA20": round(float(confirm_bar["EMA20"]), 2),
        "EMA50": round(float(confirm_bar["EMA50"]), 2),
        "ATR": round(float(confirm_bar["ATR"]), 2),
        "RSI": round(float(confirm_bar["RSI"]), 2),
        "Swing Low": setup.get("Swing Low"),
        "Swing High": setup.get("Swing High"),
        "Risk Per Share": round(levels["risk"], 2),
        "Regime OK": regime_ok,
        "RS 63d": rs_63d,
        "Score": vibe_score,
        "Notes": notes,
    })
    return row


# =========================
# SCANNER
# =========================

def run_scanner():
    """Scan active tickers and archive quality setups to the mode log directory."""
    logger.info(f"--- STEP 4: Quality Swing Scan [{mode.upper()} MODE] ---")

    if not os.path.exists(ACTIVE_TICKERS_PATH):
        logger.error(f"Missing active tickers inventory file at {ACTIVE_TICKERS_PATH}")
        sys.exit(1)

    active_tickers = set(pd.read_csv(ACTIVE_TICKERS_PATH)["Ticker"].str.upper())
    logger.info(f"Loaded {len(active_tickers)} active tickers")

    if not os.path.exists(RAW_DATA_DIR):
        logger.warning(f"Target raw directory empty or non-existent: {RAW_DATA_DIR}")
        return

    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    logger.info(f"Found {len(raw_files)} raw data files in target silo")

    sp = config.get_swing_params(mode)
    benchmark_df = get_benchmark_frame(sp, _data_mode)
    if sp.get("benchmark") and benchmark_df is None:
        logger.warning(
            f"Benchmark {sp.get('benchmark')} unavailable; regime/RS gates will reject all setups."
        )

    results = []
    rejection_counts: dict[str, int] = {}

    for file in raw_files:
        symbol = file.split("_")[0].upper()

        if symbol not in active_tickers:
            rejection_counts["inactive_ticker"] = rejection_counts.get(
                "inactive_ticker", 0) + 1
            continue

        path = os.path.join(RAW_DATA_DIR, file)
        df = pd.read_csv(path)

        try:
            df = config.validate_and_clean_ohlcv(df, require_volume=False)
        except ValueError:
            rejection_counts["missing_columns"] = rejection_counts.get(
                "missing_columns", 0) + 1
            continue

        if len(df) < config.MIN_SAVE_ROWS:
            rejection_counts["insufficient_data"] = rejection_counts.get(
                "insufficient_data", 0) + 1
            continue

        setup_row = detect_setup_at_bar(df, symbol, mode, benchmark_df)
        if not setup_row:
            rejection_counts["IGNORE"] = rejection_counts.get("IGNORE", 0) + 1
            continue

        results.append(setup_row)

    if results:
        df_out = pd.DataFrame(results).reindex(
            columns=config.SETUP_ROW_COLUMNS).sort_values("Symbol")
        print(df_out.to_markdown(index=False))

        today = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(LOG_DIR, f"swing_setups_{today}.csv")
        df_out.to_csv(out_path, index=False)

        logger.info(f"Archive created: {out_path}")
    else:
        logger.warning("No quality swing setups found for this timeframe window.")

    logger.info("Scanner rejection summary:")
    for k, v in rejection_counts.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    run_scanner()
