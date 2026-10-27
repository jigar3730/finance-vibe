"""The scanner rubric docs quote these constants; fail if code and doc drift apart.

If a test here fails after a deliberate rubric change, update the doc (and for
Coiled Cobra, bump config.RUBRIC_VERSION) in the same change.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from finance_vibe import config
from finance_vibe.scanners import breakout as bs
from finance_vibe.scanners import coiled_cobra as cc

DOCS = Path(__file__).resolve().parents[1] / "docs"
COBRA = (DOCS / "coiled_cobra.md").read_text(encoding="utf-8")
BREAKOUT = (DOCS / "breakout.md").read_text(encoding="utf-8")


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
        f"Rubric v{config.BREAKOUT_RUBRIC_VERSION}",
        f'`config.BREAKOUT_RUBRIC_VERSION = "{config.BREAKOUT_RUBRIC_VERSION}"`',
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


# Breakout rule code (everything scan_history's states/status/score depend on)
# and every module-level constant, fingerprinted per BREAKOUT_RUBRIC_VERSION.
# ast.dump output can change between Python minor versions; 3.12 (the container) is the reference.
_BREAKOUT_RULES = [
    "_ta_col", "_rolling_percentile", "_series_or_nan", "_sma", "_native_timeframe",
    "_scalar", "normalize_ohlcv", "resample_ohlcv", "add_indicators",
    "_round_like_builtin", "_nullable", "_trend_flags", "_htf_periods", "_forming_smas",
    "_forming_htf_smas", "_forming_htf_trend", "_native_trend", "_feature_frame",
    "_num", "_is_true", "_is_false", "_is_known", "_pctl_points_v",
    "classify_states_frame", "classify_status_frame", "score_readiness_frame",
    "scan_history",
]  # fmt: skip
BREAKOUT_RULE_FINGERPRINTS = {
    "1.0": "0c3e04cf2e62c62ad6bdc865602c733dabcb41b551d08f091d7b51a4b5ea8868",
}


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return tree


def breakout_rule_fingerprint() -> str:
    """sha256 of the rule functions' AST (no comments, docstrings or formatting)
    plus the values of every UPPER_CASE breakout and BREAKOUT_* config constant
    (paths excluded: they differ between checkouts)."""
    parts = []
    for name in _BREAKOUT_RULES:
        tree = _strip_docstrings(ast.parse(inspect.getsource(getattr(bs, name))))
        parts.append(ast.dump(tree, include_attributes=False))
    consts = {k: getattr(bs, k) for k in dir(bs) if k.isupper() and not k.endswith("_PATH")}
    consts |= {k: getattr(config, k) for k in dir(config) if k.startswith("BREAKOUT_")}
    consts.pop("BREAKOUT_RUBRIC_VERSION")
    for k in sorted(consts):
        v = consts[k]  # sets repr in hash order; sort them so the hash is stable
        parts.append(f"{k}={sorted(v)!r}" if isinstance(v, set | frozenset) else f"{k}={v!r}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def test_breakout_rules_match_rubric_version():
    got = breakout_rule_fingerprint()
    assert BREAKOUT_RULE_FINGERPRINTS.get(config.BREAKOUT_RUBRIC_VERSION) == got, (
        "breakout rule code or constants changed. If states/Status/Readiness can change, "
        "bump config.BREAKOUT_RUBRIC_VERSION and docs/breakout.md and add the new "
        "fingerprint here; for a behaviour-neutral refactor (golden IDENTICAL), replace "
        f"the current version's fingerprint. New fingerprint: {got}"
    )
