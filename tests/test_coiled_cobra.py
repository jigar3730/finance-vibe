"""Unit tests for the Coiled Cobra coil → expansion scorecard."""

import numpy as np
import pandas as pd

from finance_vibe.analysis_engine import check_coiled_cobra_market_gate
from finance_vibe.coiled_cobra import (
    coil_width_score,
    evaluate_coiled_cobra,
    macd_compression_score,
    rvol_trigger_score,
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


def _aligned_structure_frame(*, close=102.0, ema20=100.0, atr=4.0):
    """Three-bar frame with a full 10>20>50 stack; proximity set by close/atr."""
    return pd.DataFrame({
        "Close": [ema20, ema20, close],
        "EMA10": [ema20 + 1.0, ema20 + 1.0, ema20 + 1.0],
        "EMA20": [ema20, ema20, ema20],
        "SMA50": [ema20 - 1.0, ema20 - 1.0, ema20 - 1.0],
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
    assert "Market Gate" in setup


def test_rvol_full_points_at_2x():
    df = _ohlc(40)
    df["RVOL"] = 1.0
    df.loc[df.index[-1], "RVOL"] = 2.0
    pts, rvol = rvol_trigger_score(df)
    assert pts == 10
    assert rvol == 2.0
    df.loc[df.index[-1], "RVOL"] = 1.5
    assert rvol_trigger_score(df)[0] == 6
    df.loc[df.index[-1], "RVOL"] = 1.2
    assert rvol_trigger_score(df)[0] == 3
    df.loc[df.index[-1], "RVOL"] = 1.0
    assert rvol_trigger_score(df)[0] == 0


def test_structure_penalizes_overextension():
    coiled = _aligned_structure_frame(close=102.0, ema20=100.0, atr=4.0)
    # |102-100| = 2 <= 1.5*4 = 6 → full 15
    assert structure_score(coiled) == 15
    extended = _aligned_structure_frame(close=120.0, ema20=100.0, atr=4.0)
    # |120-100| = 20 > 6 → cap at 8
    assert structure_score(extended) == 8


def test_macd_squeeze_prefers_hist_near_zero_above_zero_line():
    above = macd_compression_score(macd=2.0, macd_signal=1.95, atr=10.0)
    below = macd_compression_score(macd=-0.10, macd_signal=-0.15, atr=10.0)
    assert above == 15
    assert below == 10
    assert above > below


def test_market_gate_fail_open_without_frames():
    assert check_coiled_cobra_market_gate() is True
    assert check_coiled_cobra_market_gate(spy_df=None, qqq_df=None) is True


def test_evaluate_applies_macro_penalty_when_gate_fails(monkeypatch):
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.coiled_cobra import add_macro_indicators

    df = add_macro_indicators(_ohlc(120, drift=0.6, noise=0.15, seed=4))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 16)
    monkeypatch.setattr(cc, "macd_compression_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 15)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (15, 0.12))

    open_setup = evaluate_coiled_cobra(df, benchmark_df=None, apply_market_gate=False)
    assert open_setup is not None

    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: False)
    penalized = evaluate_coiled_cobra(df, benchmark_df=None)
    assert penalized is not None
    assert penalized["Score"] == round(open_setup["Score"] - 15, 2)
    assert penalized["Grade"] == "B - Valid Coil"
    assert penalized["Parts"]["rvol_trigger"] == 0
    assert penalized["Market Gate"] is False
