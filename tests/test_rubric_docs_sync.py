"""The scanner rubric docs quote these constants; fail if code and doc drift apart.

If a test here fails after a deliberate rubric change, update the doc (and for
Coiled Cobra, bump config.RUBRIC_VERSION) in the same change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finance_vibe import breakout_scanner as bs
from finance_vibe import coiled_cobra as cc
from finance_vibe import config

DOCS = Path(__file__).resolve().parents[1] / "docs"
COBRA = (DOCS / "handbook" / "coiled_cobra_rubric.md").read_text(encoding="utf-8")
BREAKOUT = (DOCS / "architecture" / "breakout_scanner.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "phrase",
    [
        f"Rubric v{config.RUBRIC_VERSION}",
        f'`config.RUBRIC_VERSION = "{config.RUBRIC_VERSION}"`',
        f"Checks Met ≥ {cc.MIN_CHECKS_MET} of {cc.N_SCORED_PILLARS} scored pillars",
        f"vol_contraction   ≥ {cc.GATE_C_VOL_CONTRACTION_MIN}",
        f"structure_score   ≥ {cc.GATE_C_STRUCTURE_MIN}",
        f"`MIN_PASS_SCORE = {cc.MIN_PASS_SCORE}`",
        f"`GRADE_A_SCORE = {cc.GRADE_A_SCORE}`",
        f"-{cc.MACD_DIRECTIONAL_PENALTY} pt penalty",
        f"trailing {cc.Timeframe.for_mode('weekly').wk_to_bar * 130}-week window",
        f"RVOL ≥ {cc.RVOL_BONUS}×",
        f"RS_13w ≥ +{round(cc.RS_FULL_SCORE * 100)}%",
        f"`RS_13w > +{round(cc.RS_LEADER_EXT * 100)}%`",
        f"≥ {cc.OPEN_SKY_PCT} ×",
    ],
)
def test_cobra_rubric_quotes_code(phrase):
    assert phrase in COBRA


def test_cobra_timeframe_numbers_in_doc():
    weekly, daily = cc.Timeframe.for_mode("weekly"), cc.Timeframe.for_mode("daily")
    assert f"`COIL_BARS` = {daily.coil_bars}" in COBRA
    assert f"RS lookback = {daily.rs_lookback}" in COBRA
    assert f"RS ratio MA = {daily.rs_ratio_ma}" in COBRA
    assert f"overhead\n> lookback = {daily.lookback}" in COBRA
    assert f"bars ago ({weekly.coil_bars} weeks)" in COBRA
    assert f"{weekly.rs_ratio_ma}-week SMA of ratio" in COBRA


@pytest.mark.parametrize(
    "phrase",
    [
        f"fewer than {bs.MIN_PRIMARY_BARS} native bars",
        f"RVOL20 ≥ {bs.RVOL_CONFIRM}",
        f"distance ≤ {bs.PRE_BREAKOUT_ATR_MAX} ATR",
        f"distance ≤ {bs.AT_RESISTANCE_ATR} ATR",
        f"RVOL20 < {bs.RVOL_DRYUP}",
        f"< {bs.VOL_DRYUP_RATIO}",
        f"previous {bs.FAKEOUT_LOOKBACK} bars",
        f"MACD({bs.MACD_FAST}, {bs.MACD_SLOW}, {bs.MACD_SIGNAL})",
        f"Bollinger({bs.BB_LEN}, {int(bs.BB_STD)})",
        f"Keltner({bs.KC_LEN}, {bs.KC_SCALAR})",
        f"readiness ≥ {bs.DISPLAY_SCORE_FLOOR[bs.STATUS_WATCH]}",
        f"readiness ≥ {bs.DISPLAY_SCORE_FLOOR[bs.STATUS_DEV]}",
        "daily 252, weekly 52 or monthly 36",
    ],
)
def test_breakout_doc_quotes_code(phrase):
    assert phrase in BREAKOUT


def test_breakout_percentile_windows_match_doc():
    assert bs.PCTL_WINDOW == {"daily": 252, "weekly": 52, "monthly": 36}
    assert bs.PCTL_MIN_PERIODS == 20 and "at least 20 bars" in BREAKOUT


def test_breakout_factor_score_count_matches_doc():
    feat = bs.BreakoutFeatures(
        **{f: None for f in bs.BreakoutFeatures.__dataclass_fields__ if f != "has_daily"}
    )
    feat.symbol, feat.mode, feat.asof = "X", "weekly", None
    scores = bs.score_readiness(feat, bs.classify_states(feat))
    pillars = {
        "Breakout Readiness",
        "Trend Score",
        "Compression Score",
        "Momentum Score",
        "Volume Score",
        "Structure Score",
        "Penalty",
    }
    n_factors = len(set(scores) - pillars)
    assert f"{n_factors} factor scores" in BREAKOUT
