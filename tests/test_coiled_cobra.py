"""Unit tests for the Coiled Cobra coil → expansion scorecard."""

import numpy as np
import pandas as pd

from finance_vibe.analysis_engine import check_coiled_cobra_market_gate
from finance_vibe.coiled_cobra import (
    coil_width_score,
    evaluate_coiled_cobra,
    macd_compression_score,
    overhead_clearance_score,
    rvol_trigger_score,
    rs_score,
    structure_score,
)


def _ohlc(n=80, *, start=100.0, drift=0.3, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] + drift + rng.normal(0, noise))
    close = np.array(closes, dtype=float)
    high = close + 1.0
    low = close - 1.0
    open_ = close.copy()
    vol = np.full(n, 1_000_000.0)
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="W"),
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })


def _aligned_structure_frame(*, close=102.0, ema20=100.0, ema50=None, atr=4.0):
    """Three-bar frame with a full 10>20>50 stack; extension set by close/ema50."""
    if ema50 is None:
        ema50 = ema20 - 1.0
    return pd.DataFrame({
        "Close": [ema20, ema20, close],
        "EMA10": [ema20 + 1.0, ema20 + 1.0, ema20 + 1.0],
        "EMA20": [ema20, ema20, ema20],
        "EMA50": [ema50, ema50, ema50],
        "SMA50": [ema50, ema50, ema50],
        "ATR": [atr, atr, atr],
    })


def test_macd_compression_no_negative_macd_required():
    # Positive MACD with tight spread still compresses (uptrend coil).
    assert macd_compression_score(macd=2.0, macd_signal=1.95, atr=10.0) == 15
    # Wide spread scores zero.
    assert macd_compression_score(macd=5.0, macd_signal=0.0, atr=10.0) == 0


def test_coil_width_rewards_tight_range():
    df = _ohlc(40, drift=0.0, noise=0.05, seed=1)
    # Flat coil → high score relative to ATR≈1
    assert coil_width_score(df, atr=1.0, coil_bars=8) >= 10


def test_structure_score_rising_stack():
    from finance_vibe.coiled_cobra import add_macro_indicators
    df = add_macro_indicators(_ohlc(120, drift=0.8, noise=0.2, seed=2))
    # Drift series is typically > 1.5 ATR from EMA20, so the v3 cap is 8.
    assert structure_score(df) >= 8
    assert structure_score(_aligned_structure_frame()) == 15


def test_evaluate_rejects_without_compression_or_structure():
    # Falling market: structure fails hard gate even if other noise scores.
    from finance_vibe.coiled_cobra import add_macro_indicators
    df = add_macro_indicators(_ohlc(120, start=200.0, drift=-1.0, noise=0.3, seed=3))
    assert evaluate_coiled_cobra(df, benchmark_df=None) is None


def test_evaluate_rejects_negative_rs_even_if_structure_ok(monkeypatch):
    """RS hard gate: lagging QQQ must not pass (BA/DG false-positive class)."""
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.coiled_cobra import add_macro_indicators

    df = add_macro_indicators(_ohlc(120, drift=0.6, noise=0.15, seed=5))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 16)
    monkeypatch.setattr(cc, "macd_compression_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 15)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    # Partial RS credit only (rel > 0 but not a full pass) — must still reject.
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (5, -0.15))

    assert evaluate_coiled_cobra(df, benchmark_df=None) is None


def test_evaluate_can_pass_coiled_uptrend(monkeypatch):
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.coiled_cobra import add_macro_indicators

    df = add_macro_indicators(_ohlc(120, drift=0.6, noise=0.15, seed=4))

    # Force favorable pillars so the test is deterministic across TA noise.
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 16)
    monkeypatch.setattr(cc, "macd_compression_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 15)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (15, 0.12))

    setup = evaluate_coiled_cobra(df, benchmark_df=None)
    assert setup is not None
    assert setup["Score"] >= 70
    assert "Coil" in setup["Grade"]
    assert setup["Parts"]["relative_strength"] >= 12
    assert "macd_cross" not in setup["Parts"]
    assert "fib_bonus" not in setup["Parts"]
    assert "rvol_trigger" in setup["Parts"]
    assert "overhead_clearance" in setup["Parts"]
    assert setup["Fib Score"] == 0.0
    assert "RVOL" in setup
    assert setup["Market Gate"] is True


def test_rvol_full_points_at_2x():
    df = _ohlc(40)
    df["RVOL"] = 1.0
    df.loc[df.index[-1], "RVOL"] = 2.0
    pts, rvol = rvol_trigger_score(df)
    assert pts == 10
    assert rvol == 2.0
    df.loc[df.index[-1], "RVOL"] = 1.5
    assert rvol_trigger_score(df)[0] == 8
    df.loc[df.index[-1], "RVOL"] = 1.2
    assert rvol_trigger_score(df)[0] == 6
    df.loc[df.index[-1], "RVOL"] = 1.0
    assert rvol_trigger_score(df)[0] == 4
    df.loc[df.index[-1], "RVOL"] = 0.7
    assert rvol_trigger_score(df)[0] == 0


def test_structure_penalizes_overextension():
    coiled = _aligned_structure_frame(close=102.0, ema20=100.0, ema50=99.0)
    # (102-99)/99 ≈ 0.03 ≤ 0.25 → full 15
    assert structure_score(coiled) == 15
    # 40% above EMA50 with RS leadership → scaled haircut, not a zero
    extended = _aligned_structure_frame(close=140.0, ema20=105.0, ema50=100.0)
    leader = structure_score(extended, rs_rel=0.15)
    laggard = structure_score(extended, rs_rel=0.0)
    assert 8 <= leader < 15
    assert 0 < laggard <= leader
    # Beyond 0.50 is still a soft deduction, never a binary 0 on an aligned stack
    blown = _aligned_structure_frame(close=160.0, ema20=110.0, ema50=100.0)
    assert 0 < structure_score(blown, rs_rel=0.20) < 15


def test_macd_squeeze_prefers_hist_near_zero_above_zero_line():
    above = macd_compression_score(macd=2.0, macd_signal=1.95, atr=10.0)
    below = macd_compression_score(macd=-0.10, macd_signal=-0.15, atr=10.0)
    assert above == 15
    assert below == 10
    assert above > below


def test_market_gate_fail_open_without_frames():
    assert check_coiled_cobra_market_gate() is True
    assert check_coiled_cobra_market_gate(spy_df=None, qqq_df=None) is True


def test_market_gate_fails_only_on_trend_break():
    # More than 10% below EMA50 is a total trend failure.
    assert check_coiled_cobra_market_gate(close=89.0, ema50=100.0, rs_63d=0.20) is False
    # A coil sitting just under the 50 is not a drop.
    assert check_coiled_cobra_market_gate(close=97.0, ema50=100.0, rs_63d=0.05) is True
    # Material lag (≤ −15% RS) is a total trend failure.
    assert check_coiled_cobra_market_gate(close=110.0, ema50=100.0, rs_63d=-0.16) is False
    # Slight RS noise and extension do not fail the gate.
    assert check_coiled_cobra_market_gate(close=140.0, ema50=100.0, rs_63d=-0.02) is True
    assert check_coiled_cobra_market_gate(close=110.0, ema50=100.0, rs_63d=None) is True


def test_evaluate_does_not_zero_rvol_or_penalize_on_spy_chop(monkeypatch):
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.coiled_cobra import add_macro_indicators

    df = add_macro_indicators(_ohlc(120, drift=0.6, noise=0.15, seed=4))
    df["RVOL"] = 1.0
    df.loc[df.index[-1], "RVOL"] = 1.5
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 16)
    monkeypatch.setattr(cc, "macd_compression_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 15)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (15, 0.12))

    open_setup = evaluate_coiled_cobra(df, benchmark_df=None, apply_market_gate=False)
    gated = evaluate_coiled_cobra(df, benchmark_df=None)
    assert open_setup is not None and gated is not None
    # Ticker is above EMA50 with positive RS — SPY/QQQ unused, no -15 haircut.
    assert gated["Score"] == open_setup["Score"]
    assert gated["Parts"]["rvol_trigger"] == 8
    assert gated["Market Gate"] is True


def test_evaluate_rejects_trend_fail_but_still_scores(monkeypatch):
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.coiled_cobra import add_macro_indicators

    df = add_macro_indicators(_ohlc(120, drift=0.6, noise=0.15, seed=4))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 16)
    monkeypatch.setattr(cc, "macd_compression_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 15)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (15, -0.20))

    dropped = evaluate_coiled_cobra(df, benchmark_df=None)
    scored = evaluate_coiled_cobra(df, benchmark_df=None, include_rejects=True)
    assert dropped is None
    assert scored is not None
    assert scored["Market Gate"] is False
    assert scored["Score"] >= 70
    assert scored["Grade"] == "Rejected - Trend Fail"


def test_rs_full_score_at_15_pct_regardless_of_chop(monkeypatch):
    from finance_vibe.analysis_engine import relative_strength as _rs

    def _fake_rel(*a, **k):
        return False, 0.18  # ratio failed MA (choppy QQQ) but 63d RS is strong

    monkeypatch.setattr(
        "finance_vibe.coiled_cobra.relative_strength", _fake_rel
    )
    pts, rel = rs_score(_ohlc(80), _ohlc(80))
    assert rel == 0.18
    assert pts == 15
    del _rs


def test_open_sky_awards_full_overhead():
    df = _ohlc(60, start=100.0, drift=0.2, noise=0.1, seed=7)
    price = float(df["Close"].iloc[-1])
    high_52 = float(df["High"].max())
    assert price >= 0.95 * high_52
    assert overhead_clearance_score(df, price, atr=2.0, lookback=52) == 5.0


def test_low_rvol_does_not_fail_market_gate():
    assert check_coiled_cobra_market_gate(close=110.0, ema50=100.0, rs_63d=0.12) is True
