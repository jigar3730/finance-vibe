"""Tests for offline-model ranking of Coiled Cobra scan results.

These tests do NOT require real model artifacts: ``predict_returns`` is
monkeypatched with deterministic values so CI stays hermetic.
"""

import numpy as np
import pandas as pd
import pytest

from finance_vibe import ml_ranker
from finance_vibe.ml_ranker import (
    ML_PRED_COL,
    ML_RANK_COL,
    attach_ml_ranks,
    build_feature_frame,
)
from finance_vibe.coiled_cobra_ml_training import FEATURE_COLS
from finance_vibe.trade_plan_helper import rank_by_expected_value


# ---------------------------------------------------------------------------
# build_feature_frame
# ---------------------------------------------------------------------------

def test_build_feature_frame_uses_existing_columns():
    df = pd.DataFrame([{
        "Volume_Shelf": 16.0, "MACD_Compression": 18.0, "Structure": 17.0,
        "RS_Score": 13.5, "Coil_Width": 12.0, "MACD_Cross": 0.0, "Fib_Bonus": 1.2,
        "Pct_From_EMA20": 0.01, "Pct_From_EMA50": 0.05, "ATR_Pct": 0.03,
    }])
    feats = build_feature_frame(df)
    assert list(feats.columns) == FEATURE_COLS
    assert feats.iloc[0]["ATR_Pct"] == pytest.approx(0.03)
    assert feats.iloc[0]["RS_Score"] == pytest.approx(13.5)


def test_build_feature_frame_derives_from_raw_fields():
    df = pd.DataFrame([{
        "Volume_Shelf": 16.0, "MACD_Compression": 18.0, "Structure": 17.0,
        "RS_Score": 13.5, "Coil_Width": 12.0, "MACD_Cross": 0.0, "Fib_Bonus": 1.2,
        "Close": 100.0, "EMA20": 98.0, "EMA50": 95.0, "ATR": 4.0,
    }])
    feats = build_feature_frame(df)
    assert feats.iloc[0]["Pct_From_EMA20"] == pytest.approx((100 - 98) / 98, rel=1e-6)
    assert feats.iloc[0]["Pct_From_EMA50"] == pytest.approx((100 - 95) / 95, rel=1e-6)
    assert feats.iloc[0]["ATR_Pct"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# attach_ml_ranks
# ---------------------------------------------------------------------------

def test_attach_ml_ranks_sorts_best_first(monkeypatch):
    df = pd.DataFrame([
        {"Symbol": "LOW", "Score": 90, "Close": 100.0, "EMA20": 99.0, "EMA50": 98.0,
         "Fib 61.8%": 97.0, "Fib 78.6%": 96.0, "ATR": 2.0},
        {"Symbol": "HIGH", "Score": 70, "Close": 100.0, "EMA20": 99.0, "EMA50": 98.0,
         "Fib 61.8%": 97.0, "Fib 78.6%": 96.0, "ATR": 2.0},
    ])

    def fake_predict(frame, mode="weekly"):
        # HIGH gets the stronger predicted return regardless of Score.
        return pd.Series([0.01 if s == "LOW" else 0.09 for s in frame["Symbol"]], index=frame.index)

    monkeypatch.setattr(ml_ranker, "predict_returns", fake_predict)
    out = attach_ml_ranks(df, "weekly")

    assert list(out["Symbol"]) == ["HIGH", "LOW"]
    assert out.iloc[0][ML_RANK_COL] == 1
    assert out.iloc[1][ML_RANK_COL] == 2
    assert out.iloc[0][ML_PRED_COL] == pytest.approx(0.09)


def test_attach_ml_ranks_no_model_falls_back_null(monkeypatch):
    df = pd.DataFrame([
        {"Symbol": "A", "Score": 80, "Close": 100.0, "EMA20": 99.0, "EMA50": 98.0,
         "Fib 61.8%": 97.0, "Fib 78.6%": 96.0, "ATR": 2.0},
    ])

    def no_model(frame, mode="weekly"):
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    monkeypatch.setattr(ml_ranker, "predict_returns", no_model)
    out = attach_ml_ranks(df, "weekly")

    assert out[ML_PRED_COL].isna().all()
    assert out[ML_RANK_COL].isna().all()
    # Original order preserved when no model ran.
    assert list(out["Symbol"]) == ["A"]


def test_attach_ml_ranks_empty_frame():
    out = attach_ml_ranks(pd.DataFrame(columns=["Symbol", "Score"]))
    assert ML_PRED_COL in out.columns
    assert ML_RANK_COL in out.columns
    assert len(out) == 0


def test_attach_ml_ranks_defaults_to_daily_mode(monkeypatch):
    df = pd.DataFrame([
        {"Symbol": "A", "Score": 80, "Close": 100.0, "EMA20": 99.0, "EMA50": 98.0,
         "Fib 61.8%": 97.0, "Fib 78.6%": 96.0, "ATR": 2.0},
    ])
    captured_mode = []

    def fake_predict(frame, mode=None):
        captured_mode.append(mode)
        return pd.Series([0.05], index=frame.index)

    monkeypatch.setattr(ml_ranker, "predict_returns", fake_predict)
    attach_ml_ranks(df)
    assert captured_mode == [None]


# ---------------------------------------------------------------------------
# helper Priority integration
# ---------------------------------------------------------------------------

def test_priority_uses_ml_prediction_over_score():
    # Same R:R T2; higher Score but lower ML prediction must rank lower.
    df = pd.DataFrame([
        {"Symbol": "HISCORE", "Source": "coiled_cobra", "Score": 95, "Close": 100.0,
         "Risk Per Share": 3.0, "R:R T2": 3.0, "ML_Pred_Return": 0.01},
        {"Symbol": "HIML", "Source": "coiled_cobra", "Score": 70, "Close": 100.0,
         "Risk Per Share": 3.0, "R:R T2": 3.0, "ML_Pred_Return": 0.08},
    ])
    ranked = rank_by_expected_value(df)
    assert list(ranked["Symbol"]) == ["HIML", "HISCORE"]


def test_priority_falls_back_to_score_when_no_ml():
    df = pd.DataFrame([
        {"Symbol": "A", "Source": "coiled_cobra", "Score": 70, "Close": 100.0,
         "Risk Per Share": 3.0, "R:R T2": 3.0},
        {"Symbol": "B", "Source": "coiled_cobra", "Score": 90, "Close": 100.0,
         "Risk Per Share": 3.0, "R:R T2": 3.0},
    ])
    ranked = rank_by_expected_value(df)
    assert list(ranked["Symbol"]) == ["B", "A"]
    assert ranked.iloc[0]["Expected Value"] == pytest.approx(270.0)


def test_priority_keeps_negative_ml_below_zero():
    df = pd.DataFrame([
        {"Symbol": "NEG", "Source": "coiled_cobra", "Score": 95, "Close": 100.0,
         "Risk Per Share": 4.5, "R:R T2": 3.0, "ML_Pred_Return": -0.04},
        {"Symbol": "POS", "Source": "coiled_cobra", "Score": 70, "Close": 100.0,
         "Risk Per Share": 4.5, "R:R T2": 3.0, "ML_Pred_Return": 0.01},
    ])
    ranked = rank_by_expected_value(df)
    assert list(ranked["Symbol"]) == ["POS", "NEG"]


def test_priority_boosts_tight_coil_width_not_every_cobra_row():
    df = pd.DataFrame([
        {"Symbol": "WIDE", "Source": "coiled_cobra", "Score": 90, "Close": 100.0,
         "Risk Per Share": 4.5, "R:R T2": 3.0, "Coil_Width_ATR": 8.0},
        {"Symbol": "TIGHT", "Source": "coiled_cobra", "Score": 80, "Close": 100.0,
         "Risk Per Share": 4.5, "R:R T2": 3.0, "Coil_Width_ATR": 3.0},
    ])
    ranked = rank_by_expected_value(df)
    assert list(ranked["Symbol"]) == ["TIGHT", "WIDE"]
    assert ranked.iloc[0]["Priority"] == pytest.approx(80 * 1.25)
