"""Central paths and timeframe profiles for Finance Vibe.

Use ``get_mode_config(mode)`` to resolve raw/log directories and yfinance
download parameters for ``weekly`` or ``daily`` pipeline runs.
"""
import os

import pandas as pd

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_CONFIG_DIR, "../.."))

TIMEFRAME_PROFILES = {
    "weekly": {
        "period": "10y",
        "interval": "1wk",
        "raw_dir": os.path.join(PROJECT_ROOT, "data", "raw", "weekly"),
        "logs_dir": os.path.join(PROJECT_ROOT, "data", "logs", "weekly"),
    },
    "daily": {
        "period": "5y",
        "interval": "1d",
        "raw_dir": os.path.join(PROJECT_ROOT, "data", "raw", "daily"),
        "logs_dir": os.path.join(PROJECT_ROOT, "data", "logs", "daily"),
    },
}

DEFAULT_MODE = "weekly"


def get_mode_config(mode: str = None) -> dict:
    """Return download settings and directory paths for ``weekly`` or ``daily``.

    Creates ``raw_dir`` and ``logs_dir`` if they do not exist.
    Falls back to ``DEFAULT_MODE`` when ``mode`` is missing or invalid.
    """
    target_mode = mode if mode in TIMEFRAME_PROFILES else DEFAULT_MODE
    cfg = TIMEFRAME_PROFILES[target_mode].copy()
    cfg["mode"] = target_mode

    os.makedirs(cfg["raw_dir"], exist_ok=True)
    os.makedirs(cfg["logs_dir"], exist_ok=True)

    return cfg


# Always-included symbols merged with manifest and screener output
STATIC_TICKERS = ["SPY", "QQQ", "IWM", "SCHD","IBIT","SCHG"]

# Universe size for ticker_provider → data/active_tickers.csv
ACTIVE_TICKER_CAP = 1000
# yahooquery Screener IDs merged after the manifest (deduped, then capped)
SCREENER_IDS = [
    "most_actives",
    #"day_gainers",
    #"day_losers",
    #"undervalued_growth_stocks",
    #"growth_technology_stocks",
    #"most_shorted_stocks",
    #"aggressive_small_caps",
    #"small_cap_gainers",
    #"undervalued_large_caps",
]
# Per-screener quote count (Yahoo typically caps near 250)
SCREENER_COUNT = 250

BASE_DIR = os.path.join(PROJECT_ROOT, "data")
TICKER_LIST_PATH = os.path.join(BASE_DIR, "active_tickers.csv")


def get_raw_filename(ticker: str, cfg: dict) -> str:
    """Build a standardized raw CSV name, e.g. ``AAPL_10y_1wk.csv``."""
    return f"{ticker}_{cfg['period']}_{cfg['interval']}.csv"


def get_raw_path(ticker: str, cfg: dict) -> str:
    """Absolute path to one ticker's raw CSV inside the active mode directory."""
    return os.path.join(cfg["raw_dir"], get_raw_filename(ticker, cfg))


# Pipeline backtest defaults (offline validation; not part of run_vibe.py)
BACKTEST_LONG_MIN_SCORE = 7
BACKTEST_SHORT_MAX_SCORE = -2
BACKTEST_WARMUP_BARS = 60
BACKTEST_ENTRY_VALID_BARS = 4
BACKTEST_MAX_HOLD_BARS = 12
# Skip new signals within N bars of the prior accepted signal (cuts clustered re-entries)
BACKTEST_COOLDOWN_BARS = 4

# Realistic execution assumptions for the scaled-out swing simulator.
# Adverse per-fill slippage applied to entries and market (stop) exits.
BACKTEST_SLIPPAGE_PCT = 0.0005
# Fraction of the position taken off at the first (1R) target.
BACKTEST_PARTIAL_FRACTION = 0.5

# Default benchmark for high-beta relative-strength / market-regime gating.
BENCHMARK_TICKER = "QQQ"

# Quality swing profiles (weekly = swing leg; daily = tighter T1 / wider stop / soft vibe)
_SWING_WEEKLY = {
    "entry_atr": 0.25,
    "stop_buffer_atr": 0.25,
    "stop_atr_cap": 1.5,       # dual-constraint vol floor: entry − 1.5×ATR
    "t1_atr": 1.25,
    "t2_atr": 2.25,
    "prox_pct": 0.015,
    "prox_atr": None,
    "rsi_min_long": 45,
    "rsi_max_long": 55,
    "rsi_min_short": 50,
    "rsi_max_short": 60,
    "structure_bars": 10,       # local consolidation lookback (sessions)
    "vibe_min": None,           # no soft vibe gate on weekly
    "cooldown_bars": 4,
    "entry_valid_bars": 4,
    "max_hold_bars": 12,
    "confirm_slack_atr": 0.0,
    "require_ema_stack": False,
}

# Tuned on 5y daily QQQ/SPY/IWM: ~75% WR, +0.19 avg R vs weekly-like −0.33
_SWING_DAILY = {
    "entry_atr": 0.25,
    "stop_buffer_atr": 0.25,
    "stop_atr_cap": 1.5,
    "t1_atr": 0.85,
    "t2_atr": 1.6,
    "prox_pct": 0.02,
    "prox_atr": None,           # fixed % proximity when None
    "rsi_min_long": 40,
    "rsi_max_long": 55,
    "rsi_min_short": 50,
    "rsi_max_short": 60,
    "structure_bars": 10,       # local consolidation lookback (sessions)
    "vibe_min": 5,              # soft macro gate inside scanner
    "cooldown_bars": 8,
    "entry_valid_bars": 6,
    "max_hold_bars": 20,
    "confirm_slack_atr": 0.0,
    "require_ema_stack": False,
}

# High-beta single names (PLTR/TSLA/HOOD-class): long-only, ATR proximity,
# wider RSI, QQQ regime + relative-strength gating, dual-constraint stops
# (local structure vs 1.5×ATR floor), and true 1R/2R targets.
# Uses daily OHLCV via resolve_pipeline_mode().
_SWING_HIGH_BETA = {
    "entry_atr": 0.25,
    "stop_buffer_atr": 0.25,
    "stop_atr_cap": 1.5,        # volatility floor: stop >= entry - 1.5×ATR
    "t1_atr": 1.0,              # unused when use_r_targets is True
    "t2_atr": 1.8,             # unused when use_r_targets is True
    "prox_pct": 0.03,           # fallback only if prox_atr unset
    "prox_atr": 0.5,            # |close-EMA20| band in ATR units
    "rsi_min_long": 35,
    "rsi_max_long": 58,
    "rsi_min_short": 42,
    "rsi_max_short": 65,
    "structure_bars": 10,       # local consolidation lookback (sessions)
    "vibe_min": 5,
    "cooldown_bars": 10,
    "entry_valid_bars": 6,
    "max_hold_bars": 20,
    "confirm_slack_atr": 0.35,  # allow undercut of setup low/high
    "require_ema_stack": True,  # EMA20 > EMA50 > EMA100 (long)
    # --- hardening additions ---
    "long_only": True,
    "structure_slack_atr": 0.25,   # ATR-normalized structure tolerance
    "min_risk_atr": 0.5,           # reject setups with risk below this
    # Dual-constraint binds risk at stop_atr_cap; max_risk_atr is a safety net
    # for anything that still lands outside the band after flooring.
    "max_risk_atr": 1.5,
    "use_r_targets": True,         # T1/T2 measured in R (stop distance)
    "t1_r": 2.0,                   # hard floor: T1 R:R >= 2:1
    "t2_r": 3.0,
    "benchmark": BENCHMARK_TICKER,
    "require_market_regime": True,
    "require_relative_strength": True,
    "rs_lookback": 63,
    "rs_ratio_ma_bars": 20,
}

SWING_PROFILES = {
    "weekly": _SWING_WEEKLY,
    "daily": _SWING_DAILY,
    "high_beta": _SWING_HIGH_BETA,
}

# Optional keys applied to every profile so callers can read them uniformly.
# Existing weekly/daily behavior is preserved because these defaults are inert.
_SWING_DEFAULTS = {
    "prox_atr": None,
    "confirm_slack_atr": 0.0,
    "require_ema_stack": False,
    "long_only": False,
    "short_max_vibe": None,        # directional short soft-gate; None disables shorts under a vibe profile
    "structure_slack_atr": None,   # None -> legacy percentage structure tolerance
    "min_risk_atr": None,
    "max_risk_atr": None,          # None -> cap stop via stop_atr_cap (legacy)
    "use_r_targets": False,
    "t1_r": 1.0,
    "t2_r": 2.0,
    "partial_fraction": BACKTEST_PARTIAL_FRACTION,
    "benchmark": None,
    "require_market_regime": False,
    "require_relative_strength": False,
    "rs_lookback": 63,
    "rs_ratio_ma_bars": 20,
}

# Backward-compatible aliases (weekly defaults)
SWING_ENTRY_ATR = _SWING_WEEKLY["entry_atr"]
SWING_STOP_BUFFER_ATR = _SWING_WEEKLY["stop_buffer_atr"]
SWING_STOP_ATR_CAP = _SWING_WEEKLY["stop_atr_cap"]
SWING_T1_ATR = _SWING_WEEKLY["t1_atr"]
SWING_T2_ATR = _SWING_WEEKLY["t2_atr"]
SWING_PROX_PCT_WEEKLY = _SWING_WEEKLY["prox_pct"]
SWING_PROX_PCT_DAILY = _SWING_DAILY["prox_pct"]
SWING_RSI_MIN_LONG_WEEKLY = _SWING_WEEKLY["rsi_min_long"]
SWING_RSI_MIN_LONG_DAILY = _SWING_DAILY["rsi_min_long"]
SWING_RSI_MAX_LONG = _SWING_WEEKLY["rsi_max_long"]
SWING_RSI_MIN_SHORT = _SWING_WEEKLY["rsi_min_short"]
SWING_RSI_MAX_SHORT = _SWING_WEEKLY["rsi_max_short"]
SWING_STRUCTURE_BARS = _SWING_WEEKLY["structure_bars"]


def get_swing_params(mode: str = "weekly") -> dict:
    """Return quality-swing geometry + filter params for a swing profile.

    Optional keys from :data:`_SWING_DEFAULTS` are always present so callers
    can read them uniformly. Profiles: ``weekly``, ``daily``, ``high_beta``.
    Unknown → weekly.
    """
    key = (mode or "weekly").strip().lower()
    merged = dict(_SWING_DEFAULTS)
    merged.update(SWING_PROFILES.get(key, _SWING_WEEKLY))
    return merged


def resolve_pipeline_mode(mode: str = "weekly") -> tuple[str, str]:
    """Map CLI/pipeline mode → (data_timeframe, swing_profile).

    ``high_beta`` reads daily OHLCV but uses the high-beta swing profile.
    """
    key = (mode or DEFAULT_MODE).strip().lower()
    if key == "high_beta":
        return "daily", "high_beta"
    if key in TIMEFRAME_PROFILES:
        return key, key
    return DEFAULT_MODE, DEFAULT_MODE


def get_log_dir(mode: str = "weekly") -> str:
    """Absolute log directory for a pipeline mode, isolated per swing profile.

    ``high_beta`` gets its own ``data/logs/high_beta`` silo so its outputs do
    not collide with the ETF ``daily`` pipeline that shares the same raw data.
    """
    _, profile = resolve_pipeline_mode(mode)
    d = os.path.join(PROJECT_ROOT, "data", "logs", profile)
    os.makedirs(d, exist_ok=True)
    return d


# =====================================================================
# SHARED SWING GEOMETRY (used by scanner, planner, and backtest)
# =====================================================================

# Hard ceiling on risk as a fraction of close (ingestion guardrail mirror).
MAX_RISK_PCT_OF_CLOSE = 0.05


def structural_stop_long(
    entry: float, atr: float, ema50: float, swing_low,
    *, stop_buffer_atr: float, stop_atr_cap: float | None = None,
    close: float | None = None, max_risk_pct: float | None = MAX_RISK_PCT_OF_CLOSE,
) -> float:
    """Stop below local structure with dual-constraint + price risk cap.

    Local structure is the tighter of swing low / EMA50 (minus ATR buffer).
    When ``stop_atr_cap`` is set, the stop is also floored at
    ``entry - stop_atr_cap × ATR`` so distant macro lows cannot widen risk.
    When ``max_risk_pct`` is set, risk is also capped at ``max_risk_pct × close``.
    The final stop is the higher (tighter) of those floors.
    """
    buf = stop_buffer_atr * atr
    candidates = [ema50 - buf]
    if swing_low is not None and pd.notna(swing_low):
        candidates.append(float(swing_low) - buf)
    structural = min(candidates)
    if stop_atr_cap is not None:
        structural = max(structural, entry - stop_atr_cap * atr)
    if max_risk_pct is not None and close is not None and close > 0:
        structural = max(structural, entry - max_risk_pct * float(close))
    return min(structural, entry - buf)


def structural_stop_short(
    entry: float, atr: float, ema50: float, swing_high,
    *, stop_buffer_atr: float, stop_atr_cap: float | None = None,
    close: float | None = None, max_risk_pct: float | None = MAX_RISK_PCT_OF_CLOSE,
) -> float:
    """Stop above local structure with dual-constraint + price risk cap."""
    buf = stop_buffer_atr * atr
    candidates = [ema50 + buf]
    if swing_high is not None and pd.notna(swing_high):
        candidates.append(float(swing_high) + buf)
    structural = max(candidates)
    if stop_atr_cap is not None:
        structural = min(structural, entry + stop_atr_cap * atr)
    if max_risk_pct is not None and close is not None and close > 0:
        structural = min(structural, entry + max_risk_pct * float(close))
    return max(structural, entry + buf)


def compute_swing_levels(
    *, setup_type: str, close: float, ema20: float, ema50: float, atr: float,
    swing_low=None, swing_high=None, sp: dict,
) -> dict:
    """Compute entry/stop/targets + risk for the non-cobra swing path.

    Returns a dict with ``entry``, ``stop``, ``target1``, ``target2``,
    ``risk``, and ``reject_reason`` (None when the setup passes risk bounds).

    Behavior is profile-driven:
      * All profiles use a dual-constraint stop: local structure vs
        ``entry ± stop_atr_cap × ATR``, picking the tighter bound.
      * ``max_risk_atr`` set  → also reject when risk is outside
        ``[min_risk_atr, max_risk_atr]`` ATR after the floor is applied.
      * ``use_r_targets``     → targets at ``t1_r``/``t2_r`` × risk; else ATR.
    """
    is_long = setup_type == "SETUP_LONG"
    structural_mode = sp.get("max_risk_atr") is not None
    buf = sp["stop_buffer_atr"] * atr
    cap = sp["stop_atr_cap"]

    max_risk_pct = sp.get("max_risk_pct", MAX_RISK_PCT_OF_CLOSE)

    if is_long:
        entry = max(ema20, close - sp["entry_atr"] * atr)
        if structural_mode:
            # Anchor on the local swing low (consolidation), fall back to EMA50.
            # Triple constraint: structure vs ATR floor vs % of close (tightest wins).
            anchor = float(swing_low) if (swing_low is not None and pd.notna(swing_low)) else ema50
            structural = anchor - buf
            vol_floor = entry - cap * atr
            stop = max(structural, vol_floor)
            if max_risk_pct is not None and close > 0:
                stop = max(stop, entry - max_risk_pct * close)
            stop = min(stop, entry - buf)
        else:
            stop = structural_stop_long(
                entry, atr, ema50, swing_low,
                stop_buffer_atr=sp["stop_buffer_atr"], stop_atr_cap=cap,
                close=close, max_risk_pct=max_risk_pct,
            )
    else:
        entry = min(ema20, close + sp["entry_atr"] * atr)
        if structural_mode:
            anchor = float(swing_high) if (swing_high is not None and pd.notna(swing_high)) else ema50
            structural = anchor + buf
            vol_ceil = entry + cap * atr
            stop = min(structural, vol_ceil)
            if max_risk_pct is not None and close > 0:
                stop = min(stop, entry + max_risk_pct * close)
            stop = max(stop, entry + buf)
        else:
            stop = structural_stop_short(
                entry, atr, ema50, swing_high,
                stop_buffer_atr=sp["stop_buffer_atr"], stop_atr_cap=cap,
                close=close, max_risk_pct=max_risk_pct,
            )

    risk = abs(entry - stop)
    risk_atr = risk / atr if atr > 0 else 0.0

    reject_reason = None
    if sp.get("min_risk_atr") is not None and risk_atr < sp["min_risk_atr"]:
        reject_reason = f"risk_too_tight:{risk_atr:.2f}ATR<{sp['min_risk_atr']}"
    elif sp.get("max_risk_atr") is not None and risk_atr > sp["max_risk_atr"]:
        reject_reason = f"risk_too_wide:{risk_atr:.2f}ATR>{sp['max_risk_atr']}"

    if sp.get("use_r_targets"):
        t1_off = sp["t1_r"] * risk
        t2_off = sp["t2_r"] * risk
    else:
        t1_off = sp["t1_atr"] * atr
        t2_off = sp["t2_atr"] * atr

    if is_long:
        target1, target2 = entry + t1_off, entry + t2_off
    else:
        target1, target2 = entry - t1_off, entry - t2_off

    return {
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "risk": risk,
        "reject_reason": reject_reason,
    }


# =====================================================================
# DATA QUALITY CONTRACT
# =====================================================================

# Columns every raw OHLCV CSV must contain after ingestion/normalization.
REQUIRED_OHLCV = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Minimum usable rows before a raw CSV is worth saving/scanning.
MIN_SAVE_ROWS = 60

# Shared setup-row schema emitted by BOTH scanners so trade_planner can rely
# on a single stable contract. Column order is significant for CSV output.
#
# Fill rules:
#   - ``swing_scanner``: sets Source="swing"; fills Symbol, Setup Type, AsOf
#     Date, Close, EMA20, EMA50, ATR, RSI, Swing Low, Swing High, Notes.
#     Leaves macro-only fields empty.
#   - ``coiled_cobra``: sets Source="coiled_cobra"; fills the macro-only
#     fields plus RSI/Notes. Setup Type is currently always SETUP_LONG.
#   - ``AsOf Date``: confirmation bar date (quality swing) or signal bar.
SETUP_ROW_COLUMNS = [
    "Symbol",
    "Setup Type",
    "Source",
    "Mode",
    "AsOf Date",
    "Close",
    "EMA20",
    "EMA50",
    "ATR",
    "RSI",
    "Swing Low",
    "Swing High",
    "Risk Per Share",
    "Regime OK",
    "RS 63d",
    "Notes",
    "Score",
    "Grade",
    "Checks Met",
    "Fib 61.8%",
    "Fib 78.6%",
    "Fib Score",
    "MACD",
    "MACD Signal",
    # Pre-signal ML features (parity with coiled_cobra_ml_training FEATURE_COLS)
    "Pct_From_EMA20",
    "Pct_From_EMA50",
    "Pct_From_Fib618",
    "Pct_From_Fib786",
    "ATR_Pct",
    # Offline-model ranking outputs (soft signal; null when no model available)
    "ML_Pred_Return",
    "ML_Rank",
]


def blank_setup_row() -> dict:
    """Return a setup-row dict with every schema key present and set to None."""
    return {col: None for col in SETUP_ROW_COLUMNS}


def validate_and_clean_ohlcv(df: pd.DataFrame, *, require_volume: bool = True) -> pd.DataFrame:
    """Normalize and validate an OHLCV frame against :data:`REQUIRED_OHLCV`.

    Standardizes a ``Date`` column (promoting a DatetimeIndex when needed),
    capitalizes OHLCV headers, coerces price/volume columns to numeric, and
    drops rows with NaN in any of Open/High/Low/Close.

    Raises ``ValueError`` if required columns are missing after normalization.
    """
    out = df.copy()

    # Promote an index-based Date (yfinance downloads) into a real column.
    if "Date" not in out.columns and "date" not in {str(c).lower() for c in out.columns}:
        out = out.reset_index()

    # Standardize header casing (Open, High, Low, Close, Volume, Date).
    out.columns = [str(c).strip().capitalize() for c in out.columns]

    required = list(REQUIRED_OHLCV) if require_volume else [
        c for c in REQUIRED_OHLCV if c != "Volume"
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    numeric_cols = ["Open", "High", "Low", "Close"]
    if "Volume" in out.columns:
        numeric_cols.append("Volume")
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return out
