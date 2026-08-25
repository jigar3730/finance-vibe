"""Unit tests for the Coiled Cobra coil → expansion scorecard."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.coiled_cobra import (
    MAX_SCORE,
    _interp_score,
    coil_width_score,
    evaluate_coiled_cobra,
    macd_compression_score,
    structure_score,
)


def test_project_default_horizon_is_daily():
    from finance_vibe import coiled_cobra as cc

    assert config.DEFAULT_MODE == "daily"
    assert cc.mode == "daily"
    assert cc.LOOKBACK == 252
    assert cc.COIL_BARS == 30
    assert cc.RS_LOOKBACK == 63


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


def test_interp_score_linear_and_clamps():
    knots = [(0.05, 20.0), (0.10, 15.0)]
    assert _interp_score(0.00, knots) == 20.0
    assert _interp_score(0.075, knots) == 17.5
    assert _interp_score(0.20, knots) == 15.0


def test_macd_compression_no_negative_macd_required():
    # Positive MACD with tight spread still compresses (uptrend coil).
    assert macd_compression_score(macd=2.0, macd_signal=1.95, atr=10.0) == 20
    # Wide spread scores zero.
    assert macd_compression_score(macd=5.0, macd_signal=0.0, atr=10.0) == 0


def test_macd_compression_interpolates_between_knots():
    # spread 0.075 sits halfway between 20 and 15.
    assert macd_compression_score(macd=2.0, macd_signal=1.25, atr=10.0) == 17.5
    assert macd_compression_score(macd=2.0, macd_signal=1.0, atr=10.0) == 15.0
    assert macd_compression_score(macd=2.0, macd_signal=-2.0, atr=10.0) == 0.0


def test_macd_cross_is_flag_not_score():
    from finance_vibe.coiled_cobra import macd_crossed_this_bar, macd_cross_score

    assert macd_crossed_this_bar(0.0, 0.1, 0.6, 0.1) is True
    assert macd_crossed_this_bar(0.2, 0.1, 0.3, 0.1) is False
    assert macd_cross_score(0.0, 0.1, 0.6, 0.1, atr=10.0) == 0.0


def test_rs_score_full_pass_is_continuous(monkeypatch):
    from finance_vibe import coiled_cobra as cc

    df = _ohlc(80)
    bench = _ohlc(80, start=100.0, drift=0.1, seed=9)
    monkeypatch.setattr(cc, "relative_strength", lambda *a, **k: (True, 0.05))
    pts, rel = cc.rs_score(df, bench)
    assert pts == 13.5
    assert rel == pytest.approx(0.05)


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
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 13)
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
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 13)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (15, 0.12))

    setup = evaluate_coiled_cobra(df, benchmark_df=None)
    assert setup is not None
    assert setup["Score"] >= 70
    assert "Coil" in setup["Grade"]
    assert setup["Parts"]["relative_strength"] >= 12


def test_evaluate_clips_score_at_100(monkeypatch):
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.coiled_cobra import add_macro_indicators

    df = add_macro_indicators(_ohlc(120, drift=0.6, noise=0.15, seed=6))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "macd_compression_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 13)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (15, 0.20))
    monkeypatch.setattr(cc, "proximity_highs_score", lambda *a, **k: 12.0)
    monkeypatch.setattr(cc, "fibonacci_score", lambda *a, **k: 5.0)

    setup = evaluate_coiled_cobra(df, benchmark_df=None)
    assert setup is not None
    assert setup["Score"] == MAX_SCORE
    assert setup["Parts"]["macd_cross"] == 0.0


def test_coil_width_interpolates_between_knots():
    df = _ohlc(40, drift=0.0, noise=0.0, seed=1)
    # range is 2.0 (high-low = 2) so width_atr = 2 / atr
    assert coil_width_score(df, atr=0.5, coil_bars=8) == 13.0  # 4 ATR
    assert coil_width_score(df, atr=2.0 / 6.0, coil_bars=8) == 9.0
    assert coil_width_score(df, atr=0.2, coil_bars=8) == 0.0  # 10 ATR


def test_configure_mode_switches_lookback_and_paths():
    from finance_vibe import coiled_cobra as cc
    weekly = cc.configure_mode("weekly")
    assert weekly == "weekly"
    assert cc.LOOKBACK == 52
    assert cc.COIL_BARS == 8
    assert cc.STRUCTURE_STOP_BARS == 8
    assert cc.RS_LOOKBACK == 13
    assert cc.HIGH_LOOKBACK_BARS == (13, 26, 52)
    assert Path(cc.RAW_DATA_DIR).name == "weekly"
    daily = cc.configure_mode("daily")
    assert daily == "daily"
    assert cc.LOOKBACK == 252
    assert cc.COIL_BARS == 30
    assert cc.STRUCTURE_STOP_BARS == 30
    assert cc.RS_LOOKBACK == 63
    assert cc.RS_RATIO_MA == 20
    assert cc.HIGH_LOOKBACK_BARS == (63, 126, 252)
    cc.configure_mode("weekly")
    assert cc.STRUCTURE_STOP_BARS == 8
    assert cc.RS_LOOKBACK == 13
    cc.configure_mode(config.DEFAULT_MODE)


def test_local_swing_low_follows_configure_mode():
    from finance_vibe import coiled_cobra as cc
    df = _ohlc(40, drift=0.0, noise=0.0, seed=3)
    df.loc[df.index[-1], "Low"] = 50.0
    cc.configure_mode("daily")
    # Daily coil window is 30 bars; the last low of 50 must be inside it.
    assert cc.local_swing_low(df) == pytest.approx(50.0)
    cc.configure_mode(config.DEFAULT_MODE)


def test_coil_geometry_fields_reports_high_low_and_width():
    from finance_vibe.coiled_cobra import coil_geometry_fields
    df = _ohlc(20, drift=0.0, noise=0.0, seed=2)
    geom = coil_geometry_fields(df, atr=2.0)
    assert geom["Coil_High"] == pytest.approx(float(df["High"].max()))
    assert geom["Coil_Low"] == pytest.approx(float(df["Low"].min()))
    assert geom["Coil_Width_ATR"] == pytest.approx(
        (geom["Coil_High"] - geom["Coil_Low"]) / 2.0
    )


def test_proximity_full_score_when_at_highs():
    from finance_vibe.coiled_cobra import (
        proximity_highs_score,
        proximity_to_highs_fields,
    )

    df = _ohlc(260, drift=0.4, noise=0.05, seed=8)
    close = float(df["Close"].iloc[-1])
    fields, dists = proximity_to_highs_fields(df, close, atr=1.0)
    assert fields["Dist_High_63_Pct"] is not None
    # Sitting on the rolling high → full 12.
    score = proximity_highs_score([0.0, 0.0, 0.0])
    assert score == 12.0
    far = proximity_highs_score([5.0, 8.0, 10.0])
    assert far == 0.0


def test_rsi_healthy_zone():
    from finance_vibe.coiled_cobra import rsi_zone_fields

    mid = rsi_zone_fields(55.0)
    assert mid["RSI_Healthy"] == 1
    assert mid["RSI_Zone_Score"] == 5.0
    wash = rsi_zone_fields(28.0)
    assert wash["RSI_Healthy"] == 0
    assert wash["RSI_Zone_Score"] == 0.0


def test_volume_accumulation_up_ratio_and_trend():
    from finance_vibe.coiled_cobra import volume_accumulation_fields

    n = 40
    df = _ohlc(n, drift=0.2, noise=0.0, seed=1)
    df["Volume"] = [100.0] * (n - 8) + [80.0] * 4 + [40.0] * 4
    fields = volume_accumulation_fields(df, coil_bars=8)
    assert fields["Up_Volume_Ratio"] is not None
    assert 0.0 <= fields["Up_Volume_Ratio"] <= 1.0
    assert fields["Volume_Trend_Ratio"] == pytest.approx(0.5, rel=0.05)
    assert fields["OBV_Coil_Slope"] is not None


def test_coil_width_percentile_excludes_signal_bar():
    from finance_vibe.coiled_cobra import coil_width_percentile

    df = _ohlc(80, drift=0.0, noise=0.0, seed=2)
    base = coil_width_percentile(df, coil_bars=8)
    spiked = df.copy()
    spiked.loc[spiked.index[-1], "High"] = float(spiked["High"].iloc[-1]) + 50.0
    after = coil_width_percentile(spiked, coil_bars=8)
    assert after is not None and base is not None
    assert after >= base


def test_attach_weekly_confirmation_boosts_daily_preds(tmp_path, monkeypatch):
    from finance_vibe import coiled_cobra as cc
    from finance_vibe import config

    weekly_dir = tmp_path / "data" / "logs" / "weekly"
    weekly_dir.mkdir(parents=True)
    pd.DataFrame([
        {"Symbol": "AAA", "Score": 88},
        {"Symbol": "BBB", "Score": 72},
    ]).to_csv(weekly_dir / "coiled_cobra_setups_2099-01-01.csv", index=False)

    monkeypatch.setattr(cc, "BASE_DIR", str(tmp_path))
    daily = pd.DataFrame({
        "Symbol": ["AAA", "CCC"],
        "Score": [80.0, 90.0],
        "ML_Pred_Return": [0.04, 0.08],
        "ML_Rank": [2, 1],
    })
    out = cc.attach_weekly_confirmation(daily, "daily")
    assert out.loc[out["Symbol"] == "AAA", "Weekly_Coil_Pass"].iloc[0] == 1
    assert out.loc[out["Symbol"] == "CCC", "Weekly_Coil_Pass"].iloc[0] == 0
    aaa_pred = float(out.loc[out["Symbol"] == "AAA", "ML_Pred_Return"].iloc[0])
    assert aaa_pred == pytest.approx(0.04)
    cc.configure_mode(config.DEFAULT_MODE)


def test_score_excludes_fib_and_macd_cross(monkeypatch):
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.coiled_cobra import add_macro_indicators

    df = add_macro_indicators(_ohlc(120, drift=0.6, noise=0.15, seed=7))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "macd_compression_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "coil_width_score", lambda *a, **k: 13)
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "rs_score", lambda *a, **k: (15, 0.20))
    monkeypatch.setattr(cc, "proximity_highs_score", lambda *a, **k: 0.0)
    monkeypatch.setattr(cc, "fibonacci_score", lambda *a, **k: 5.0)

    setup = evaluate_coiled_cobra(df, benchmark_df=None)
    assert setup is not None
    # 20+20+20+15+13+0 = 88; Fib 5 must not push this to 93.
    assert setup["Score"] == pytest.approx(88.0)
    # Fib may compute 0 if levels are NaN; either way it must not enter Score.
    assert setup["Score"] == pytest.approx(
        20 + 20 + 20 + 15 + 13 + 0
    )
