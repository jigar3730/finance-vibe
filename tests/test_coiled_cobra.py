"""Unit tests for the Coiled Cobra coil → expansion scorecard (rubric v4.0).

Full spec: docs/handbook/coiled_cobra_rubric.md.
"""

import numpy as np
import pandas as pd

from finance_vibe.analysis_engine import check_coiled_cobra_market_gate
from finance_vibe.coiled_cobra import (
    MIN_BARS_FULL_SCORE,
    add_macro_indicators,
    evaluate_coiled_cobra,
    macd_directional_penalty,
    overhead_clearance_score,
    relative_strength_score,
    rvol_trigger_score,
    structure_score,
    vol_contraction_score,
)


def _ohlc(n=220, *, start=100.0, drift=0.3, noise=0.5, seed=0):
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
        "Date": pd.date_range("2018-01-01", periods=n, freq="W"),
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })


def _aligned_structure_frame(*, close=102.0, s2=100.0, fast=None, slow=None):
    """Three-bar frame with a full TT_EMA stack; extension set by close/slow."""
    if slow is None:
        slow = s2 - 1.0
    if fast is None:
        fast = slow + (s2 - slow) * 0.5
    return pd.DataFrame({
        "Close": [s2, s2, close],
        "TT_EMA_S1": [s2 + 1.0, s2 + 1.0, s2 + 1.0],
        "TT_EMA_S2": [s2, s2, s2],
        "TT_EMA_FAST": [fast, fast, fast],
        "TT_EMA_SLOW": [slow, slow, slow],
    })


# ---------------------------------------------------------------------------
# Individual pillars
# ---------------------------------------------------------------------------

def test_macd_directional_penalty():
    assert macd_directional_penalty(2.0) == 0
    assert macd_directional_penalty(0.0) == 8
    assert macd_directional_penalty(-1.0) == 8


def test_vol_contraction_score_bands_and_declining_halving():
    n = 40
    # Tight percentile, declining into it -> full 25.
    df = pd.DataFrame({"BBWidth_Pctile": [50.0] * (n - 1) + [5.0]})
    assert vol_contraction_score(df) == (25, 5.0)

    # Percentile in the 35-50 band but RISING (not declining) -> base 6 halved to 3.
    df2 = pd.DataFrame({"BBWidth_Pctile": [5.0] * (n - 1) + [45.0]})
    pts, pct = vol_contraction_score(df2)
    assert pct == 45.0
    assert pts == 3

    # No data -> zero, not a crash.
    df3 = pd.DataFrame({"BBWidth_Pctile": [np.nan] * n})
    assert vol_contraction_score(df3) == (0, None)


def test_structure_score_requires_stack_and_tightens_extension():
    # Full stack, tight to TT_EMA_S2 -> max 20.
    coiled = _aligned_structure_frame(close=101.0, s2=100.0, slow=99.0)
    assert structure_score(coiled) == 20

    # Stack broken (S1 < 0.98*S2) -> hard 0, not a soft deduction.
    broken = pd.DataFrame({
        "Close": [100.0, 100.0, 100.0],
        "TT_EMA_S1": [90.0, 90.0, 90.0],
        "TT_EMA_S2": [100.0, 100.0, 100.0],
        "TT_EMA_FAST": [97.0, 97.0, 97.0],
        "TT_EMA_SLOW": [95.0, 95.0, 95.0],
    })
    assert structure_score(broken) == 0

    # Hole mid-stack: S1/S2/SLOW alone look aligned, but TT_EMA_FAST dips
    # below TT_EMA_SLOW -- not a clean ascending fan. Must hard-fail even
    # though the old (pre-fix) S1/S2/SLOW-only check would have passed this.
    hole_in_stack = _aligned_structure_frame(close=110.0, s2=105.0, fast=95.0, slow=100.0)
    assert structure_score(hole_in_stack) == 0

    # 25% extension above TT_EMA_SLOW -> partial deduction (v4.0's soft-ext
    # floor is tightened to 20%, vs v3.1's 25%).
    extended = _aligned_structure_frame(close=125.0, s2=105.0, slow=100.0)
    assert 0 < structure_score(extended) < 20

    # >40% extension floors the whole pillar at 0, even with an aligned stack.
    blown = _aligned_structure_frame(close=145.0, s2=105.0, slow=100.0)
    assert structure_score(blown, rs_rel=0.20) == 0


def test_overhead_clearance_new_point_scale():
    df = _ohlc(60, start=100.0, drift=0.2, noise=0.1, seed=7)
    price = float(df["Close"].iloc[-1])
    high_52 = float(df["High"].max())
    assert price >= 0.95 * high_52
    assert overhead_clearance_score(df, price, atr=2.0, lookback=52) == 10.0


def test_rvol_full_points_at_2x():
    df = _ohlc(40)
    df["RVOL"] = 1.0
    df.loc[df.index[-1], "RVOL"] = 2.0
    assert rvol_trigger_score(df)[0] == 10
    df.loc[df.index[-1], "RVOL"] = 1.5
    assert rvol_trigger_score(df)[0] == 8
    df.loc[df.index[-1], "RVOL"] = 1.2
    assert rvol_trigger_score(df)[0] == 6
    df.loc[df.index[-1], "RVOL"] = 1.0
    assert rvol_trigger_score(df)[0] == 4
    df.loc[df.index[-1], "RVOL"] = 0.7
    assert rvol_trigger_score(df)[0] == 0


def test_relative_strength_score_smoothed_bands(monkeypatch):
    from finance_vibe import coiled_cobra as cc

    def _score(rel, ok):
        monkeypatch.setattr(cc, "relative_strength", lambda *a, **k: (ok, rel))
        monkeypatch.setattr(cc, "_rs_line_new_high", lambda *a, **k: False)
        return relative_strength_score(_ohlc(30), _ohlc(30))

    assert _score(0.20, False)[0] == 20    # unconditional full score, even if not "ok"
    assert _score(0.12, True)[0] == 18
    assert _score(0.05, True)[0] == 14
    pts_mid, _ = _score(0.05, False)       # linear 6 -> 12 band
    assert 6 <= pts_mid <= 12
    pts_lag, _ = _score(-0.05, False)      # linear 0 -> 6 band (was a flat 5 in v3.1)
    assert 0 <= pts_lag <= 6
    assert _score(-0.20, False)[0] == 0


def test_relative_strength_score_rs_line_bonus(monkeypatch):
    from finance_vibe import coiled_cobra as cc
    monkeypatch.setattr(cc, "relative_strength", lambda *a, **k: (True, 0.05))
    monkeypatch.setattr(cc, "_rs_line_new_high", lambda *a, **k: True)
    pts, rel = relative_strength_score(_ohlc(30), _ohlc(30))
    assert pts == 16  # 14 base + 2 RS-line-new-high bonus
    assert rel == 0.05


def test_market_gate_fail_open_without_frames():
    assert check_coiled_cobra_market_gate() is True
    assert check_coiled_cobra_market_gate(spy_df=None, qqq_df=None) is True


def test_market_gate_fails_only_on_trend_break():
    assert check_coiled_cobra_market_gate(close=89.0, ema50=100.0, rs_63d=0.20) is False
    assert check_coiled_cobra_market_gate(close=97.0, ema50=100.0, rs_63d=0.05) is True
    assert check_coiled_cobra_market_gate(close=110.0, ema50=100.0, rs_63d=-0.16) is False
    assert check_coiled_cobra_market_gate(close=140.0, ema50=100.0, rs_63d=-0.02) is True


# ---------------------------------------------------------------------------
# evaluate_coiled_cobra: hard gates + tiering (integration)
# ---------------------------------------------------------------------------

def _force_all_pillars_strong(monkeypatch):
    """Monkeypatch every pillar to a guaranteed-strong value so evaluate_
    coiled_cobra's gate/tier/score logic is deterministic across TA noise.
    """
    from finance_vibe import coiled_cobra as cc
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 15)
    monkeypatch.setattr(cc, "vol_contraction_score", lambda *a, **k: (25, 5.0))
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)
    monkeypatch.setattr(cc, "relative_strength_score", lambda *a, **k: (20, 0.18))
    monkeypatch.setattr(cc, "overhead_clearance_score", lambda *a, **k: 10.0)


def test_insufficient_history_returns_none_even_with_include_rejects():
    short_df = add_macro_indicators(_ohlc(MIN_BARS_FULL_SCORE - 10, drift=0.8, noise=0.1, seed=1))
    assert evaluate_coiled_cobra(short_df, benchmark_df=None) is None
    assert evaluate_coiled_cobra(short_df, benchmark_df=None, include_rejects=True) is None


def test_evaluate_passes_and_tiers_actionable(monkeypatch):
    from finance_vibe import coiled_cobra as cc

    df = add_macro_indicators(_ohlc(220, drift=0.8, noise=0.15, seed=4))
    df["RVOL"] = 1.0
    df.loc[df.index[-1], "RVOL"] = 2.0  # breakout-volume trigger
    # Force an unambiguous break above the trailing COIL_BARS-window high
    # (drift alone isn't reliably > the fixed +1.0 High offset in _ohlc()).
    last = df.index[-1]
    prior_high = float(df.iloc[-9:-1]["High"].max())
    df.loc[last, "Close"] = prior_high * 1.05
    df.loc[last, "High"] = prior_high * 1.06
    _force_all_pillars_strong(monkeypatch)
    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: True)

    setup = evaluate_coiled_cobra(df, benchmark_df=None)
    assert setup is not None
    assert setup["Score"] >= 70
    assert setup["Checks Met"] == "6/6"
    assert setup["Tier"] == "Actionable"
    assert "Coil Ready" in setup["Grade"] or "Coil" in setup["Grade"]
    assert setup["Market Gate"] is True


def test_evaluate_rejects_gate_a_on_downtrend(monkeypatch):
    # Falling market: Gate A (trend template) must fail regardless of pillars.
    df = add_macro_indicators(_ohlc(220, start=300.0, drift=-1.0, noise=0.2, seed=3))
    _force_all_pillars_strong(monkeypatch)
    assert evaluate_coiled_cobra(df, benchmark_df=None) is None
    rejected = evaluate_coiled_cobra(df, benchmark_df=None, include_rejects=True)
    assert rejected is not None
    assert "A" in rejected["Grade"]


def test_evaluate_rejects_gate_b_trend_fail_but_still_scores(monkeypatch):
    from finance_vibe import coiled_cobra as cc

    df = add_macro_indicators(_ohlc(220, drift=0.8, noise=0.15, seed=4))
    _force_all_pillars_strong(monkeypatch)
    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: False)

    dropped = evaluate_coiled_cobra(df, benchmark_df=None)
    scored = evaluate_coiled_cobra(df, benchmark_df=None, include_rejects=True)
    assert dropped is None
    assert scored is not None
    assert scored["Market Gate"] is False
    assert scored["Score"] >= 70
    assert "B" in scored["Grade"]


def test_evaluate_rejects_gate_c_when_structure_fails_independently(monkeypatch):
    """High total score but a hard-zero structure pillar must still reject
    (this is the exact v3.1 false-positive class Gate C exists to close)."""
    from finance_vibe import coiled_cobra as cc

    df = add_macro_indicators(_ohlc(220, drift=0.8, noise=0.15, seed=4))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 15)
    monkeypatch.setattr(cc, "vol_contraction_score", lambda *a, **k: (25, 5.0))
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 0)  # fails Gate C
    monkeypatch.setattr(cc, "relative_strength_score", lambda *a, **k: (20, 0.18))
    monkeypatch.setattr(cc, "overhead_clearance_score", lambda *a, **k: 10.0)
    monkeypatch.setattr(cc, "rvol_trigger_score", lambda *a, **k: (10, 2.0))
    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: True)

    assert evaluate_coiled_cobra(df, benchmark_df=None) is None
    rejected = evaluate_coiled_cobra(df, benchmark_df=None, include_rejects=True)
    assert rejected["Score"] >= 70  # would have passed on score alone
    assert "C" in rejected["Grade"]


def test_evaluate_rejects_gate_d_breadth_despite_high_score(monkeypatch):
    """Score >=70 concentrated in few pillars, but fewer than MIN_CHECKS_MET
    (4/6, recalibrated 2026-09-18 from 5/6 -- see coiled_cobra_rubric.md)
    individually clear their own check threshold -> Gate D rejects."""
    from finance_vibe import coiled_cobra as cc

    df = add_macro_indicators(_ohlc(220, drift=0.8, noise=0.15, seed=4))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 5)   # below 8, doesn't count
    monkeypatch.setattr(cc, "vol_contraction_score", lambda *a, **k: (25, 5.0))  # counts
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)               # counts
    monkeypatch.setattr(cc, "relative_strength_score", lambda *a, **k: (20, 0.18))  # counts
    monkeypatch.setattr(cc, "overhead_clearance_score", lambda *a, **k: 2.0)     # below 5, doesn't count
    monkeypatch.setattr(cc, "rvol_trigger_score", lambda *a, **k: (0, 0.5))      # below 6, doesn't count
    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: True)

    rejected = evaluate_coiled_cobra(df, benchmark_df=None, include_rejects=True)
    assert rejected["Checks Met"] == "3/6"
    assert rejected["Score"] >= 70
    assert "D" in rejected["Grade"]
    assert evaluate_coiled_cobra(df, benchmark_df=None) is None


def test_evaluate_passes_gate_d_at_four_of_six_checks(monkeypatch):
    """4/6 checks now clears Gate D (recalibrated from the original 5/6 --
    a 264-ticker/10y walk-forward backtest showed >=5/6 had no measurable
    expectancy edge over >=4/6 and discarded ~65% of equal-or-better
    B/Watchlist-tier signal volume). Same pillar mix as the rejection test
    above, but with overhead_clearance also clearing its own threshold."""
    from finance_vibe import coiled_cobra as cc

    df = add_macro_indicators(_ohlc(220, drift=0.8, noise=0.15, seed=4))
    monkeypatch.setattr(cc, "evaluate_volume_profile_shelf", lambda *a, **k: 5)   # below 8, doesn't count
    monkeypatch.setattr(cc, "vol_contraction_score", lambda *a, **k: (25, 5.0))  # counts
    monkeypatch.setattr(cc, "structure_score", lambda *a, **k: 20)               # counts
    monkeypatch.setattr(cc, "relative_strength_score", lambda *a, **k: (20, 0.18))  # counts
    monkeypatch.setattr(cc, "overhead_clearance_score", lambda *a, **k: 10.0)    # counts
    monkeypatch.setattr(cc, "rvol_trigger_score", lambda *a, **k: (0, 0.5))      # below 6, doesn't count
    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: True)

    passed = evaluate_coiled_cobra(df, benchmark_df=None, include_rejects=True)
    assert passed["Checks Met"] == "4/6"
    assert passed["Score"] >= 70
    assert "D" not in passed["Grade"]
    assert evaluate_coiled_cobra(df, benchmark_df=None) is not None


def test_evaluate_watchlist_when_no_breakout_or_rvol(monkeypatch):
    """Gates pass and score clears threshold, but no RVOL trigger / breakout
    yet -> Watchlist, not Actionable (the v3.1 tiering gap this rubric fixes)."""
    from finance_vibe import coiled_cobra as cc

    df = add_macro_indicators(_ohlc(220, drift=0.8, noise=0.15, seed=4))
    _force_all_pillars_strong(monkeypatch)
    monkeypatch.setattr(cc, "rvol_trigger_score", lambda *a, **k: (0, 0.6))
    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: True)

    setup = evaluate_coiled_cobra(df, benchmark_df=None)
    assert setup is not None
    assert setup["Tier"] == "Watchlist"
    assert "Watch" in setup["Grade"]


def test_macd_penalty_reduces_score(monkeypatch):
    from finance_vibe import coiled_cobra as cc

    df = add_macro_indicators(_ohlc(220, drift=0.8, noise=0.15, seed=4))
    df["RVOL"] = 1.0
    df.loc[df.index[-1], "RVOL"] = 2.0
    _force_all_pillars_strong(monkeypatch)
    monkeypatch.setattr(cc, "check_coiled_cobra_market_gate", lambda **k: True)

    positive_macd = df.copy()
    positive_macd.loc[positive_macd.index[-1], "MACD"] = 1.0
    negative_macd = df.copy()
    negative_macd.loc[negative_macd.index[-1], "MACD"] = -1.0

    pos_setup = evaluate_coiled_cobra(positive_macd, benchmark_df=None)
    neg_setup = evaluate_coiled_cobra(negative_macd, benchmark_df=None)
    assert pos_setup is not None and neg_setup is not None
    assert neg_setup["Score"] == pos_setup["Score"] - 8
