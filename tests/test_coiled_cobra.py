"""Unit tests for the Coiled Cobra coil → expansion scorecard."""

import numpy as np
import pandas as pd
import pytest

from finance_vibe.coiled_cobra import (
    coil_width_score,
    evaluate_coiled_cobra,
    macd_compression_score,
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


def test_macd_compression_no_negative_macd_required():
    # Positive MACD with tight spread still compresses (uptrend coil).
    assert macd_compression_score(macd=2.0, macd_signal=1.95, atr=10.0) == 20
    # Wide spread scores zero.
    assert macd_compression_score(macd=5.0, macd_signal=0.0, atr=10.0) == 0


def test_coil_width_rewards_tight_range():
    df = _ohlc(40, drift=0.0, noise=0.05, seed=1)
    # Flat coil → high score relative to ATR≈1
    assert coil_width_score(df, atr=1.0, coil_bars=8) >= 10


def test_structure_score_rising_stack():
    from finance_vibe.coiled_cobra import add_macro_indicators
    df = add_macro_indicators(_ohlc(120, drift=0.8, noise=0.2, seed=2))
    assert structure_score(df) >= 12


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