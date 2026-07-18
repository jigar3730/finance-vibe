"""Data-quality contract tests: OHLCV validation, setup schema, level math, R:R."""

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe import trade_planner
from finance_vibe.trade_planner import calculate_stock_levels, generate_trade_plan
from finance_vibe.trade_plan_helper import process_trade_plan


# ---------------------------------------------------------------------------
# config.validate_and_clean_ohlcv
# ---------------------------------------------------------------------------

def _ohlcv(n=5):
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="W"),
        "Open": np.linspace(10, 14, n),
        "High": np.linspace(11, 15, n),
        "Low": np.linspace(9, 13, n),
        "Close": np.linspace(10.5, 14.5, n),
        "Volume": np.linspace(1000, 1400, n),
    })


def test_validate_ohlcv_happy_path():
    cleaned = config.validate_and_clean_ohlcv(_ohlcv())
    assert list(config.REQUIRED_OHLCV) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in config.REQUIRED_OHLCV:
        assert col in cleaned.columns


def test_validate_ohlcv_missing_volume_raises():
    df = _ohlcv().drop(columns=["Volume"])
    with pytest.raises(ValueError):
        config.validate_and_clean_ohlcv(df, require_volume=True)


def test_validate_ohlcv_missing_volume_allowed_when_optional():
    df = _ohlcv().drop(columns=["Volume"])
    cleaned = config.validate_and_clean_ohlcv(df, require_volume=False)
    assert "Close" in cleaned.columns


def test_validate_ohlcv_drops_nan_rows_and_coerces():
    df = _ohlcv()
    # Mirror a CSV with a stray non-numeric cell: the column loads as object.
    df["Close"] = df["Close"].astype(object)
    df.loc[2, "Close"] = "not_a_number"
    cleaned = config.validate_and_clean_ohlcv(df)
    assert len(cleaned) == len(df) - 1
    assert pd.api.types.is_numeric_dtype(cleaned["Close"])


def test_validate_ohlcv_promotes_date_index():
    df = _ohlcv().set_index("Date")
    cleaned = config.validate_and_clean_ohlcv(df)
    assert "Date" in cleaned.columns


# ---------------------------------------------------------------------------
# setup schema
# ---------------------------------------------------------------------------

def test_blank_setup_row_matches_schema():
    row = config.blank_setup_row()
    assert set(row.keys()) == set(config.SETUP_ROW_COLUMNS)
    assert all(v is None for v in row.values())


def test_setup_schema_includes_ml_feature_and_rank_columns():
    from finance_vibe.coiled_cobra_ml_training import FEATURE_COLS
    # Every ML feature (except Score, already present) must be an emitted column.
    for col in FEATURE_COLS:
        assert col in config.SETUP_ROW_COLUMNS, f"missing feature col {col}"
    assert "ML_Pred_Return" in config.SETUP_ROW_COLUMNS
    assert "ML_Rank" in config.SETUP_ROW_COLUMNS


# ---------------------------------------------------------------------------
# calculate_stock_levels
# ---------------------------------------------------------------------------

def _base_row(**overrides):
    row = {
        "ATR": 2.0,
        "Close": 100.0,
        "EMA20": 105.0,
        "EMA50": 110.0,
        "Setup Type": "SETUP_LONG",
        "Source": "swing",
    }
    row.update(overrides)
    return row


def test_cobra_valid_fib_uses_structural_entry():
    row = _base_row(Source="coiled_cobra")
    row["Fib 78.6%"] = 95.0
    row["Swing Low"] = 97.0
    entry, stop, t1, t2, opt_type, delta = calculate_stock_levels(row)
    # entry = max(fib786, close - 0.25*atr) = max(95, 99.5)
    assert entry == pytest.approx(99.5)
    assert stop < entry
    risk = entry - stop
    assert t1 == pytest.approx(entry + 2.0 * risk)
    assert t2 == pytest.approx(entry + 3.0 * risk)
    assert opt_type == "CALL"


def test_cobra_distant_fib_uses_local_swing_not_macro_floor():
    """Year-scale Fib must not place stops 40–50% away; local swing + 1.5×ATR wins."""
    row = _base_row(Source="coiled_cobra", Close=100.0, ATR=2.0)
    row["Fib 78.6%"] = 50.0          # macro Fib far below market
    row["Swing Low"] = 96.0          # 10-session consolidation floor
    entry, stop, t1, t2, *_ = calculate_stock_levels(row)
    assert entry == pytest.approx(99.5)  # close - 0.25*ATR
    # Dual-constraint: max(swing-buf, entry-1.5ATR) = max(95.5, 96.5) = 96.5
    assert stop == pytest.approx(entry - 1.5 * 2.0)
    assert (entry - stop) / entry < 0.05
    risk = entry - stop
    assert t1 == pytest.approx(entry + 2.0 * risk)
    assert t2 == pytest.approx(entry + 3.0 * risk)


def test_cobra_price_risk_cap_binds_when_atr_wide():
    """Weekly-scale ATR×1.5 can exceed 5% of close; price cap must tighten the stop."""
    row = _base_row(Source="coiled_cobra", Close=100.0, ATR=10.0)
    row["Fib 78.6%"] = 50.0
    row["Swing Low"] = 80.0
    entry, stop, t1, t2, *_ = calculate_stock_levels(row)
    # 1.5×ATR = 15% of close; cap forces risk ≤ 5% of close.
    assert (entry - stop) <= config.MAX_RISK_PCT_OF_CLOSE * 100.0 + 1e-9
    assert stop == pytest.approx(entry - config.MAX_RISK_PCT_OF_CLOSE * 100.0)
    risk = entry - stop
    assert t1 == pytest.approx(entry + 2.0 * risk)


def test_cobra_nan_fib_falls_back_to_swing():
    row = _base_row(Source="coiled_cobra")
    row["Fib 78.6%"] = np.nan
    entry, stop, *_ = calculate_stock_levels(row)
    # Swing long entry = max(ema20, close - 0.25*atr) = max(105, 99.5)
    assert entry == pytest.approx(105.0)
    assert stop < entry


def test_long_stop_below_entry():
    entry, stop, *_ = calculate_stock_levels(_base_row())
    assert stop < entry


def test_quality_swing_targets_use_config_multiples():
    from finance_vibe import config
    row = _base_row()
    row["Swing Low"] = 100.0
    entry, stop, t1, t2, *_ = calculate_stock_levels(row, mode="weekly")
    atr = 2.0
    wp = config.get_swing_params("weekly")
    assert entry == pytest.approx(105.0)
    assert t1 == pytest.approx(entry + wp["t1_atr"] * atr)
    assert t2 == pytest.approx(entry + wp["t2_atr"] * atr)
    assert stop < entry
    assert (entry - stop) <= wp["stop_atr_cap"] * atr + 1e-9


def test_daily_swing_params_tighter_than_weekly():
    from finance_vibe import config
    w = config.get_swing_params("weekly")
    d = config.get_swing_params("daily")
    assert d["t1_atr"] < w["t1_atr"]
    assert d["stop_atr_cap"] >= w["stop_atr_cap"]
    assert d["vibe_min"] == 5
    assert d["cooldown_bars"] > w["cooldown_bars"]


def test_high_beta_profile_uses_atr_proximity():
    from finance_vibe import config
    hb = config.get_swing_params("high_beta")
    d = config.get_swing_params("daily")
    assert hb["prox_atr"] is not None
    assert hb["prox_atr"] > 0
    assert hb["stop_atr_cap"] == pytest.approx(1.5)
    assert hb["structure_bars"] == 10
    assert hb["t1_atr"] > d["t1_atr"]
    assert hb["rsi_min_long"] < d["rsi_min_long"]
    assert hb["require_ema_stack"] is True
    data_mode, profile = config.resolve_pipeline_mode("high_beta")
    assert data_mode == "daily"
    assert profile == "high_beta"


def test_high_beta_stock_levels_use_profile():
    from finance_vibe import config
    row = _base_row()
    row["Swing Low"] = 100.0
    row["Mode"] = "high_beta"
    entry, stop, t1, t2, *_ = calculate_stock_levels(row, mode="high_beta")
    hb = config.get_swing_params("high_beta")
    # high_beta uses true R targets (stop distance), not ATR multiples.
    risk = entry - stop
    assert risk > 0
    assert t1 == pytest.approx(entry + hb["t1_r"] * risk)
    assert t2 == pytest.approx(entry + hb["t2_r"] * risk)


def test_high_beta_dual_constraint_floors_wide_structure():
    """Distant swing lows are floored at entry − stop_atr_cap × ATR."""
    from finance_vibe import config
    row = _base_row(EMA50=110.0)
    row["Swing Low"] = 90.0
    row["Mode"] = "high_beta"
    entry, stop, t1, t2, *_ = calculate_stock_levels(row, mode="high_beta")
    hb = config.get_swing_params("high_beta")
    atr = 2.0
    assert stop == pytest.approx(entry - hb["stop_atr_cap"] * atr)
    risk = entry - stop
    assert t1 == pytest.approx(entry + hb["t1_r"] * risk)
    assert t2 == pytest.approx(entry + hb["t2_r"] * risk)


def test_short_stop_above_entry():
    row = _base_row(Setup_Type="SETUP_SHORT")
    row["Setup Type"] = "SETUP_SHORT"
    row["Swing High"] = 112.0
    entry, stop, t1, t2, opt_type, delta = calculate_stock_levels(row)
    assert stop > entry
    assert opt_type == "PUT"


# ---------------------------------------------------------------------------
# structural-risk rejection + compute_swing_levels
# ---------------------------------------------------------------------------

def test_compute_swing_levels_floors_too_wide_structure():
    """Wide swing lows are normalized by the volatility floor, not rejected."""
    sp = config.get_swing_params("high_beta")
    lv = config.compute_swing_levels(
        setup_type="SETUP_LONG", close=100.0, ema20=100.0, ema50=99.0, atr=1.0,
        swing_low=90.0, swing_high=None, sp=sp,
    )
    assert lv["reject_reason"] is None
    assert lv["stop"] == pytest.approx(lv["entry"] - sp["stop_atr_cap"] * 1.0)
    assert lv["risk"] == pytest.approx(sp["stop_atr_cap"] * 1.0)


def test_compute_swing_levels_rejects_too_tight_risk():
    sp = config.get_swing_params("high_beta")
    # Swing low essentially at entry -> risk below min_risk_atr.
    lv = config.compute_swing_levels(
        setup_type="SETUP_LONG", close=100.0, ema20=100.0, ema50=99.9, atr=1.0,
        swing_low=99.95, swing_high=None, sp=sp,
    )
    assert lv["reject_reason"] is not None
    assert "too_tight" in lv["reject_reason"]


def test_compute_swing_levels_accepts_in_bounds_with_r_targets():
    sp = config.get_swing_params("high_beta")
    lv = config.compute_swing_levels(
        setup_type="SETUP_LONG", close=100.0, ema20=100.0, ema50=99.0, atr=1.0,
        swing_low=99.0, swing_high=None, sp=sp,
    )
    assert lv["reject_reason"] is None
    risk = lv["risk"]
    assert lv["target1"] == pytest.approx(lv["entry"] + sp["t1_r"] * risk)
    assert lv["target2"] == pytest.approx(lv["entry"] + sp["t2_r"] * risk)


def test_weekly_profile_has_no_risk_rejection():
    sp = config.get_swing_params("weekly")
    lv = config.compute_swing_levels(
        setup_type="SETUP_LONG", close=100.0, ema20=100.0, ema50=95.0, atr=2.0,
        swing_low=90.0, swing_high=None, sp=sp,
    )
    assert lv["reject_reason"] is None  # weekly never rejects on structural risk


# ---------------------------------------------------------------------------
# row-Mode precedence for the planner
# ---------------------------------------------------------------------------

def test_row_mode_authoritative_when_mode_none():
    # Row carries Mode=high_beta; planner called with mode=None must honor it.
    row = _base_row(EMA50=99.0, ATR=1.0, Close=100.0, EMA20=100.0)
    row["Swing Low"] = 99.0
    row["Mode"] = "high_beta"
    entry, stop, t1, t2, *_ = calculate_stock_levels(row, mode=None)
    risk = entry - stop
    hb = config.get_swing_params("high_beta")
    # R-based targets prove the high_beta profile was applied.
    assert t1 == pytest.approx(entry + hb["t1_r"] * risk)


# ---------------------------------------------------------------------------
# ATR structure tolerance
# ---------------------------------------------------------------------------

def test_structure_tolerance_atr_vs_pct():
    from finance_vibe import swing_scanner as ss
    n = 5
    # Latest low undercuts the prior swing low by ~0.4 ATR.
    df = pd.DataFrame({
        "Low": [100.0, 100.0, 100.0, 100.0, 100.0, 99.6],
        "High": [101.0] * 6,
        "ATR": [1.0] * 6,
    })
    # Legacy 0.2% band (~0.2) rejects a 0.4 undercut...
    assert ss._structure_held_long(df, n, structure_slack_atr=None) is False
    # ...but a 0.5-ATR slack (0.5) tolerates it.
    assert ss._structure_held_long(df, n, structure_slack_atr=0.5) is True


# ---------------------------------------------------------------------------
# benchmark: no-lookahead regime + relative strength
# ---------------------------------------------------------------------------

def _bench_frame(closes, start="2023-01-01"):
    from finance_vibe.analysis_engine import ema
    df = pd.DataFrame({
        "Date": pd.date_range(start, periods=len(closes), freq="D"),
        "Close": closes,
    })
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA100"] = ema(df["Close"], 100)
    df["EMA50_rising"] = df["EMA50"] > df["EMA50"].shift(1)
    return df


def test_market_regime_causal_lookup():
    from finance_vibe.analysis_engine import market_regime_ok
    # Decline then recovery. A causal (no-lookahead) lookup sees only past bars,
    # so mid-decline is unfavorable even though the series later recovers.
    closes = list(np.linspace(200, 120, 90)) + list(np.linspace(120, 260, 90))
    f = _bench_frame(closes)
    assert market_regime_ok(f, f["Date"].iloc[80]) is False
    assert market_regime_ok(f, f["Date"].iloc[-1]) is True


def test_market_regime_rejects_downtrend():
    from finance_vibe.analysis_engine import market_regime_ok
    down = _bench_frame(list(np.linspace(200, 100, 160)))
    assert market_regime_ok(down, down["Date"].iloc[-1]) is False


def test_relative_strength_positive_and_negative():
    from finance_vibe.analysis_engine import relative_strength
    n = 120
    bench = _bench_frame(list(np.linspace(100, 120, n)))
    dates = bench["Date"]
    # Strong stock: outpaces benchmark.
    strong = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 200, n)})
    ok, rel = relative_strength(strong, bench, as_of=dates.iloc[-1], lookback=63, ratio_ma_bars=20)
    assert ok is True and rel > 0
    # Weak stock: lags benchmark.
    weak = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 101, n)})
    ok2, rel2 = relative_strength(weak, bench, as_of=dates.iloc[-1], lookback=63, ratio_ma_bars=20)
    assert ok2 is False


def test_relative_strength_no_lookahead_alignment():
    from finance_vibe.analysis_engine import relative_strength
    n = 120
    bench = _bench_frame(list(np.linspace(100, 120, n)))
    dates = bench["Date"]
    stock = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 200, n)})
    # Insufficient history before the lookback window -> cannot assess (no peeking ahead).
    ok, rel = relative_strength(
        stock, bench, as_of=dates.iloc[30], lookback=63, ratio_ma_bars=20
    )
    assert ok is False and rel is None


# ---------------------------------------------------------------------------
# high-beta routing
# ---------------------------------------------------------------------------

def test_high_beta_routing_and_log_isolation():
    data_mode, profile = config.resolve_pipeline_mode("high_beta")
    assert (data_mode, profile) == ("daily", "high_beta")
    log_dir = config.get_log_dir("high_beta")
    assert log_dir.endswith("logs/high_beta")
    # daily and high_beta share raw data but have separate log silos.
    assert config.get_log_dir("daily").endswith("logs/daily")


# ---------------------------------------------------------------------------
# trade_plan_helper R:R sign correctness
# ---------------------------------------------------------------------------

def _write_plan(mode_dir, date_str, rows):
    mode_dir.mkdir(parents=True, exist_ok=True)
    plan = mode_dir / f"trade_plan_{date_str}.csv"
    pd.DataFrame(rows).to_csv(plan, index=False)
    return plan


def test_helper_rr_positive_for_long_and_short(tmp_path, monkeypatch):
    date_str = "2099-02-02"
    mode_dir = tmp_path / "data" / "logs" / "weekly"
    _write_plan(mode_dir, date_str, [
        {
            "Symbol": "LONGY", "Setup Type": "SETUP_LONG",
            "Close": 100.0, "Score": 80, "Checks Met": "6/6",
            "Stock Entry": 100.0, "Stock Stop": 96.0,
            "Target 1": 108.0, "Target 2": 112.0,
        },
        {
            "Symbol": "SHORTY", "Setup Type": "SETUP_SHORT",
            "Close": 100.0, "Score": 75, "Checks Met": "6/6",
            "Stock Entry": 100.0, "Stock Stop": 104.0,
            "Target 1": 92.0, "Target 2": 88.0,
        },
    ])

    monkeypatch.chdir(tmp_path)
    out_path = process_trade_plan("weekly", today=date_str)
    clean = pd.read_csv(out_path)

    assert "Setup Type" in clean.columns
    assert len(clean) == 2
    assert (clean["Risk Per Share"] > 0).all()
    assert (clean["R:R T1"] >= 2.0).all()
    assert (clean["R:R T2"] > 0).all()
    assert "Expected Value" in clean.columns
    assert "Priority" in clean.columns


def test_helper_ingestion_survives_when_all_rows_fail_risk(tmp_path, monkeypatch):
    """Empty survivors must not crash (pandas empty-string .sum() → '')."""
    date_str = "2099-02-04"
    mode_dir = tmp_path / "data" / "logs" / "weekly"
    _write_plan(mode_dir, date_str, [
        {
            "Symbol": "WIDE", "Setup Type": "SETUP_LONG",
            "Close": 100.0, "Score": 90, "Checks Met": "6/6",
            "Stock Entry": 100.0, "Stock Stop": 90.0,
            "Target 1": 120.0, "Target 2": 130.0,
        },
    ])
    monkeypatch.chdir(tmp_path)
    out_path = process_trade_plan("weekly", today=date_str)
    clean = pd.read_csv(out_path)
    assert len(clean) == 0


def test_helper_ingestion_guardrails_drop_bad_rows(tmp_path, monkeypatch):
    date_str = "2099-02-03"
    mode_dir = tmp_path / "data" / "logs" / "weekly"
    _write_plan(mode_dir, date_str, [
        {  # wide risk > 5% of close
            "Symbol": "WIDE", "Setup Type": "SETUP_LONG", "Source": "swing",
            "Close": 100.0, "Score": 90, "Checks Met": "6/6",
            "Stock Entry": 100.0, "Stock Stop": 90.0,
            "Target 1": 120.0, "Target 2": 130.0,
        },
        {  # incomplete checklist (< 5/6)
            "Symbol": "FAILCHK", "Setup Type": "SETUP_LONG", "Source": "coiled_cobra",
            "Close": 100.0, "Score": 90, "Checks Met": "4/6",
            "Stock Entry": 100.0, "Stock Stop": 97.0,
            "Target 1": 106.0, "Target 2": 109.0,
        },
        {  # R:R T1 < 2
            "Symbol": "LOWRR", "Setup Type": "SETUP_LONG", "Source": "swing",
            "Close": 100.0, "Score": 90, "Checks Met": "6/6",
            "Stock Entry": 100.0, "Stock Stop": 96.0,
            "Target 1": 104.0, "Target 2": 108.0,
        },
        {  # survivor — soft baseline 5/6 is enough
            "Symbol": "KEEP", "Setup Type": "SETUP_LONG", "Source": "coiled_cobra",
            "Close": 100.0, "Score": 85, "Checks Met": "5/6",
            "Stock Entry": 100.0, "Stock Stop": 97.0,
            "Target 1": 106.0, "Target 2": 109.0,
        },
    ])

    monkeypatch.chdir(tmp_path)
    out_path = process_trade_plan("weekly", today=date_str)
    clean = pd.read_csv(out_path)
    assert list(clean["Symbol"]) == ["KEEP"]
    assert clean.iloc[0]["R:R T1"] >= 2.0
    assert clean.iloc[0]["Expected Value"] == pytest.approx(85 * 3.0)


# ---------------------------------------------------------------------------
# generate_trade_plan explicit path (regression: previously raised NameError)
# ---------------------------------------------------------------------------

def test_generate_trade_plan_with_explicit_path(tmp_path, monkeypatch):
    scanner_dir = tmp_path / "logs" / "weekly"
    scanner_dir.mkdir(parents=True)
    src = scanner_dir / "swing_setups_2099-03-03.csv"
    pd.DataFrame([
        {
            "Symbol": "AAPL", "Setup Type": "SETUP_LONG", "Source": "swing",
            "Close": 100.0, "EMA20": 98.0, "EMA50": 95.0, "ATR": 2.0, "RSI": 55.0,
        }
    ]).to_csv(src, index=False)

    monkeypatch.setattr(trade_planner, "SCANNER_DIR", scanner_dir)

    plan_df = generate_trade_plan(scanner_csv_path=str(src))
    assert plan_df is not None
    assert len(plan_df) == 1
    assert plan_df.iloc[0]["Symbol"] == "AAPL"
