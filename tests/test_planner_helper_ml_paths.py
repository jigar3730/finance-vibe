"""Planner / helper tests for the null-ML and active-ML ranking branches.

Current live state is *ML inactive* (``config.ML_RANKING_ENABLED = False`` and no
predictions); ranking must then be exactly raw-Score ranking with no warnings.
The active-ML branch is exercised by enabling the switch explicitly.

Also locks the cross-module contracts: scanner -> planner -> helper schemas, and
that planner geometry always satisfies the helper's guardrails (v4.0 rubric:
T1 = 2R, T2 = 3R, risk <= MAX_RISK_PCT_OF_CLOSE, Gate D 4/6).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config, trade_plan_helper as h, trade_planner as tp
from finance_vibe.coiled_cobra import MIN_CHECKS_MET, N_SCORED_PILLARS
from finance_vibe.trade_plan_helper import (
    CLEAN_EXPORT_COLUMNS,
    _checklist_fully_passed,
    ml_priority_active,
    process_trade_plan,
    rank_by_expected_value,
)

SCORES = [90.0, 80.0, 75.0, 70.0]
SYMS = ["A", "B", "C", "D"]


def _survivors(**extra) -> pd.DataFrame:
    df = pd.DataFrame({
        "Symbol": SYMS, "Source": "coiled_cobra", "Score": SCORES, "Close": 100.0,
        "Risk Per Share": 4.0, "R:R T1": 2.0, "R:R T2": 3.0,
    })
    for k, v in extra.items():
        df[k] = v
    return df


def _no_warnings():
    cm = warnings.catch_warnings()
    cm.__enter__()
    warnings.simplefilter("error")
    return cm


@pytest.fixture
def ml_on(monkeypatch):
    monkeypatch.setattr(config, "ML_RANKING_ENABLED", True)


# ---------------------------------------------------------------------------
# Null-ML branch (the current live state): falls back to raw Score
# ---------------------------------------------------------------------------

NULL_FORMS = {
    "column absent": None,
    "NaN": np.nan,
    "None": None,
    "empty string": "",
    "non-numeric text": "n/a",
}


@pytest.mark.parametrize("switch", [False, True], ids=["switch_off", "switch_on"])
@pytest.mark.parametrize("form", list(NULL_FORMS))
def test_null_ml_falls_back_to_score_without_warnings(form, switch, monkeypatch):
    monkeypatch.setattr(config, "ML_RANKING_ENABLED", switch)
    df = _survivors() if form == "column absent" else _survivors(ML_Pred_Return=NULL_FORMS[form])

    cm = _no_warnings()
    try:
        out = rank_by_expected_value(df)
    finally:
        cm.__exit__(None, None, None)

    assert not ml_priority_active(df)
    assert list(out["Symbol"]) == SYMS                       # descending Score
    assert list(out["Expected Value"]) == [270.0, 240.0, 225.0, 210.0]   # R:R T2 x Score
    assert list(out["Priority"]) == [337.5, 300.0, 281.25, 262.5]        # x 1.25 propensity


def test_priority_order_is_score_order_when_rr_t2_is_constant():
    # Every Coiled Cobra plan has R:R T2 = 3.0, so the "expected value" ranking
    # is exactly raw-Score ranking (ties stable).
    rng = np.random.default_rng(0)
    df = _survivors().iloc[[0] * 40].reset_index(drop=True)
    df["Symbol"] = [f"S{i}" for i in range(40)]
    df["Score"] = rng.uniform(70, 98, 40).round(2)
    out = rank_by_expected_value(df)
    assert list(out["Score"]) == sorted(df["Score"], reverse=True)


def test_switch_off_ignores_populated_predictions(monkeypatch):
    monkeypatch.setattr(config, "ML_RANKING_ENABLED", False)
    df = _survivors(ML_Pred_Return=[0.01, 0.09, 0.02, 0.05])   # would reorder if used
    assert not ml_priority_active(df)
    out = rank_by_expected_value(df)
    assert list(out["Symbol"]) == SYMS
    assert list(out["Priority"]) == [337.5, 300.0, 281.25, 262.5]


def test_missing_rr_t2_column_does_not_crash():
    out = rank_by_expected_value(_survivors().drop(columns=["R:R T2"]))
    assert (out["Expected Value"] == 0).all() and len(out) == 4


def test_missing_source_and_risk_columns_do_not_crash():
    out = rank_by_expected_value(_survivors().drop(columns=["Source", "Risk Per Share"]))
    assert list(out["Symbol"]) == SYMS
    assert (out["Priority"] == out["Expected Value"]).all()   # propensity 1.0: no coil, no risk info


def test_nan_score_sorts_last_without_error():
    out = rank_by_expected_value(_survivors().assign(Score=[90.0, np.nan, 75.0, 70.0]))
    assert list(out["Symbol"]) == ["A", "C", "D", "B"]
    assert out.iloc[-1]["Expected Value"] == 0


@pytest.mark.parametrize("switch", [False, True])
def test_empty_frame_is_safe_in_both_states(switch, monkeypatch):
    monkeypatch.setattr(config, "ML_RANKING_ENABLED", switch)
    out = rank_by_expected_value(_survivors().iloc[0:0])
    assert out.empty and {"Expected Value", "Priority"} <= set(out.columns)
    assert not ml_priority_active(_survivors().iloc[0:0])


def test_propensity_documented_behaviour():
    # Coil source -> 1.25 regardless of risk; non-coil only when risk <= 3% of close.
    df = pd.DataFrame({
        "Symbol": ["coil_wide", "swing_tight", "swing_wide"],
        "Source": ["coiled_cobra", "swing", "swing"],
        "Score": [80.0, 80.0, 80.0], "Close": 100.0,
        "Risk Per Share": [12.0, 2.0, 12.0], "R:R T2": 3.0,
    })
    out = rank_by_expected_value(df).set_index("Symbol")
    mult = (out["Priority"] / out["Expected Value"]).round(3)
    assert mult["coil_wide"] == 1.25 and mult["swing_tight"] == 1.25 and mult["swing_wide"] == 1.0


# ---------------------------------------------------------------------------
# Active-ML branch (switch on)
# ---------------------------------------------------------------------------

def test_active_ml_full_coverage_ranks_by_prediction(ml_on):
    df = _survivors(ML_Pred_Return=[0.01, 0.05, 0.02, 0.03])
    assert ml_priority_active(df)
    out = rank_by_expected_value(df)
    assert list(out["Symbol"]) == ["B", "D", "C", "A"]
    # Priority = R:R T2 x max(pred, 0) x propensity
    assert list(out["Priority"]) == [0.1875, 0.1125, 0.075, 0.0375]
    assert list(out["Expected Value"]) == [240.0, 210.0, 225.0, 270.0]   # still reported


def test_active_ml_partial_coverage_falls_back_to_score(ml_on):
    # The top-Score row has no prediction (e.g. a NaN feature). It must NOT be
    # demoted below every predicted row: incomplete coverage -> Score for all.
    df = _survivors(ML_Pred_Return=[np.nan, 0.05, 0.02, 0.03])
    assert not ml_priority_active(df)
    out = rank_by_expected_value(df)
    assert list(out["Symbol"]) == SYMS
    assert list(out["Priority"]) == [337.5, 300.0, 281.25, 262.5]


def test_active_ml_all_negative_predictions_tie_break_by_score(ml_on):
    # Input deliberately NOT in Score order, so a stable sort alone can't pass.
    df = _survivors(ML_Pred_Return=[-0.01, -0.05, -0.02, -0.03]).iloc[[3, 1, 0, 2]]
    out = rank_by_expected_value(df)
    assert (out["Priority"] == 0).all()                  # negative alpha is never rewarded
    assert list(out["Symbol"]) == SYMS                   # ...but Score still orders the ties


def test_active_ml_mixed_sign_predictions(ml_on):
    df = _survivors(ML_Pred_Return=[0.04, -0.02, 0.0, 0.01]).iloc[[2, 1, 3, 0]]
    out = rank_by_expected_value(df)
    assert list(out["Symbol"]) == ["A", "D", "B", "C"]   # positives by pred; 0-ties (B, C) by Score


def test_active_ml_single_row(ml_on):
    out = rank_by_expected_value(_survivors(ML_Pred_Return=0.02).iloc[:1])
    assert len(out) == 1 and out.iloc[0]["Priority"] == pytest.approx(0.075)


def test_active_ml_accepts_string_numbers_from_csv(ml_on):
    df = _survivors(ML_Pred_Return=["0.01", "0.05", "0.02", "0.03"])
    assert list(rank_by_expected_value(df)["Symbol"]) == ["B", "D", "C", "A"]


# ---------------------------------------------------------------------------
# v4.0 alignment: schema + guardrail contracts
# ---------------------------------------------------------------------------

V4_FIELDS = {"Score", "Grade", "Tier", "Checks Met", "RVOL", "Market Gate",
             "ML_Pred_Return", "ML_Rank"}


def test_checklist_guardrail_tracks_coiled_cobra_gate_d():
    assert h.MIN_CHECKLIST_RATIO == pytest.approx(MIN_CHECKS_MET / N_SCORED_PILLARS)
    assert (MIN_CHECKS_MET, N_SCORED_PILLARS) == (4, 6)
    assert _checklist_fully_passed(f"{MIN_CHECKS_MET}/{N_SCORED_PILLARS}")
    assert not _checklist_fully_passed(f"{MIN_CHECKS_MET - 1}/{N_SCORED_PILLARS}")
    for missing in (None, np.nan, "", "nan", "None", "garbage"):
        assert _checklist_fully_passed(missing)          # non-checklist rows fail open


def test_schemas_line_up_scanner_planner_helper():
    computed_by_planner = {"Stock Entry", "Stock Stop", "Target 1", "Target 2"}
    planner_cols = set(tp._PLAN_EXPORT_COLUMNS)
    helper_derived = {"R:R T1", "R:R T2", "Expected Value", "Priority"}

    assert planner_cols - computed_by_planner <= set(config.SETUP_ROW_COLUMNS)
    assert set(CLEAN_EXPORT_COLUMNS) <= planner_cols | helper_derived
    for cols in (set(config.SETUP_ROW_COLUMNS), planner_cols, set(CLEAN_EXPORT_COLUMNS)):
        assert V4_FIELDS <= cols
    assert len(set(tp._PLAN_EXPORT_COLUMNS)) == len(tp._PLAN_EXPORT_COLUMNS)
    assert len(set(CLEAN_EXPORT_COLUMNS)) == len(CLEAN_EXPORT_COLUMNS)


def _cobra_rows(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        close = float(np.exp(rng.uniform(np.log(3), np.log(600))))
        atr = close * rng.uniform(0.01, 0.09)
        rows.append({
            "Symbol": f"S{i}", "Setup Type": "SETUP_LONG", "Source": "coiled_cobra",
            "Mode": "weekly", "Close": round(close, 2), "ATR": round(atr, 2),
            "EMA20": round(close * rng.uniform(0.94, 1.02), 2),
            "EMA50": round(close * rng.uniform(0.88, 1.0), 2),
            "Fib 78.6%": round(close * rng.uniform(0.85, 1.0), 2),
            "Swing Low": round(close * rng.uniform(0.88, 0.99), 2),
        })
    return rows


def test_planner_geometry_always_satisfies_helper_guardrails():
    """T1 = 2R and T2 = 3R after cent-rounding, risk within the 5% cap.

    The helper's MIN_RR_T1 = 2.0 is *exactly* the planner's T1 multiple, so any
    change to either side (or a rounding regression) would silently reject
    every Coiled Cobra row -- lock the contract.
    """
    for row in _cobra_rows():
        e, s, t1, t2, *_ = tp.calculate_stock_levels(row, mode=None)
        er, sr, t1r, t2r, risk = tp._export_levels(e, s, t1, t2, row["Close"], row["Setup Type"])
        assert risk > 0
        assert round((t1r - er) / risk, 2) >= h.MIN_RR_T1
        assert round((t1r - er) / risk, 2) == 2.0 and round((t2r - er) / risk, 2) == 3.0
        assert risk / row["Close"] <= h.MAX_RISK_PCT_OF_CLOSE + 1e-12


# ---------------------------------------------------------------------------
# Planner ML pass-through + planner -> helper round trip
# ---------------------------------------------------------------------------

DATE = "2099-06-01"


def _scanner_frame(ml_pred=None, ml_rank=None) -> pd.DataFrame:
    rows = []
    for i, (sym, score) in enumerate(zip(SYMS, SCORES)):
        rows.append({
            "Symbol": sym, "Setup Type": "SETUP_LONG", "Source": "coiled_cobra", "Mode": "weekly",
            "AsOf Date": "2099-05-29", "Close": 100.0, "EMA20": 99.0, "EMA50": 96.0, "ATR": 2.0,
            "RSI": 58.0, "Swing Low": 96.5, "Fib 78.6%": 97.5, "Fib 61.8%": 95.0,
            "Score": score, "Grade": "B - Valid Coil", "Tier": "Actionable",
            "Checks Met": "4/6", "RVOL": 1.3, "Market Gate": True, "Notes": "n",
            "ML_Pred_Return": None if ml_pred is None else ml_pred[i],
            "ML_Rank": None if ml_rank is None else ml_rank[i],
        })
    return pd.DataFrame(rows)


@pytest.fixture
def pipeline_dir(tmp_path, monkeypatch):
    logs = tmp_path / "data" / "logs" / "weekly"
    logs.mkdir(parents=True)
    monkeypatch.setattr(tp, "SCANNER_DIR", logs)
    monkeypatch.chdir(tmp_path)
    return logs


def _plan(logs, scanner_df):
    scanner_df.to_csv(logs / f"coiled_cobra_setups_{DATE}.csv", index=False)
    return tp.generate_trade_plan(as_of=DATE)


def test_planner_passes_null_ml_and_v4_fields_through(pipeline_dir):
    plan = _plan(pipeline_dir, _scanner_frame())
    assert list(plan.columns) == tp._PLAN_EXPORT_COLUMNS
    assert plan["ML_Pred_Return"].isna().all() and plan["ML_Rank"].isna().all()
    assert list(plan["Score"]) == SCORES
    assert set(plan["Tier"]) == {"Actionable"} and set(plan["Checks Met"]) == {"4/6"}
    assert (plan["RVOL"] == 1.3).all() and plan["Market Gate"].all()


def test_planner_tolerates_scanner_csv_without_ml_columns(pipeline_dir):
    plan = _plan(pipeline_dir, _scanner_frame().drop(columns=["ML_Pred_Return", "ML_Rank"]))
    assert list(plan.columns) == tp._PLAN_EXPORT_COLUMNS
    assert plan["ML_Pred_Return"].isna().all()


def test_planner_passes_active_ml_values_through_unchanged(pipeline_dir):
    plan = _plan(pipeline_dir, _scanner_frame(ml_pred=[0.01, 0.05, 0.02, 0.03], ml_rank=[4, 1, 3, 2]))
    assert list(plan["ML_Pred_Return"]) == [0.01, 0.05, 0.02, 0.03]
    assert list(plan["ML_Rank"]) == [4, 1, 3, 2]


def test_roundtrip_null_ml_ranks_by_score_quietly(pipeline_dir, capsys):
    _plan(pipeline_dir, _scanner_frame())
    capsys.readouterr()

    cm = _no_warnings()
    try:
        out = process_trade_plan("weekly", today=DATE)
    finally:
        cm.__exit__(None, None, None)
    text = capsys.readouterr().out
    clean = pd.read_csv(out)

    assert list(clean["Symbol"]) == SYMS                       # nothing spuriously dropped
    assert list(clean["Score"]) == SCORES
    assert (clean["R:R T1"] == 2.0).all() and (clean["R:R T2"] == 3.0).all()
    assert clean["Priority"].is_monotonic_decreasing
    assert "Expected Value (R:R T2 × Score)" in text
    assert "Ignoring" not in text                              # no ML noise when the column is empty
    assert list(clean.columns) == [c for c in CLEAN_EXPORT_COLUMNS if c in clean.columns]


def test_roundtrip_predictions_ignored_while_switch_off(pipeline_dir, capsys):
    _plan(pipeline_dir, _scanner_frame(ml_pred=[0.01, 0.09, 0.02, 0.05]))
    capsys.readouterr()
    clean = pd.read_csv(process_trade_plan("weekly", today=DATE))
    text = capsys.readouterr().out
    assert list(clean["Symbol"]) == SYMS
    assert "Ignoring 4 ML prediction(s)" in text and "disabled" in text


def test_roundtrip_active_ml_ranks_by_prediction(pipeline_dir, capsys, ml_on):
    _plan(pipeline_dir, _scanner_frame(ml_pred=[0.01, 0.05, 0.02, 0.03], ml_rank=[4, 1, 3, 2]))
    capsys.readouterr()
    clean = pd.read_csv(process_trade_plan("weekly", today=DATE))
    text = capsys.readouterr().out
    assert list(clean["Symbol"]) == ["B", "D", "C", "A"]
    assert "ML predicted return" in text


def test_roundtrip_incomplete_ml_falls_back_and_says_why(pipeline_dir, capsys, ml_on):
    _plan(pipeline_dir, _scanner_frame(ml_pred=[np.nan, 0.05, 0.02, 0.03]))
    capsys.readouterr()
    clean = pd.read_csv(process_trade_plan("weekly", today=DATE))
    text = capsys.readouterr().out
    assert list(clean["Symbol"]) == SYMS
    assert "incomplete (3/4 rows)" in text


def test_attach_ml_ranks_is_inert_while_switch_off(monkeypatch):
    """Scanner side of the switch: even with a working model the ML columns stay null
    and the existing (Score) order is preserved."""
    from finance_vibe import ml_ranker

    monkeypatch.setattr(config, "ML_RANKING_ENABLED", False)
    monkeypatch.setattr(
        ml_ranker, "predict_returns",
        lambda frame, mode="weekly": pd.Series([0.01, 0.09, 0.02, 0.05], index=frame.index),
    )
    df = _survivors()
    out = ml_ranker.attach_ml_ranks(df, "weekly")
    assert list(out["Symbol"]) == SYMS
    assert out[ml_ranker.ML_PRED_COL].isna().all() and out[ml_ranker.ML_RANK_COL].isna().all()
