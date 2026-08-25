"""Tests for offline-model ranking of Coiled Cobra scan results.

These tests do NOT require real model artifacts: ``predict_returns`` is
monkeypatched with deterministic values so CI stays hermetic.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

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
    monkeypatch.setattr(
        ml_ranker, "attach_horizon_probabilities", lambda frame, mode=None: frame
    )
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
    monkeypatch.setattr(
        ml_ranker, "attach_horizon_probabilities", lambda frame, mode=None: frame
    )
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
    monkeypatch.setattr(
        ml_ranker, "attach_horizon_probabilities", lambda frame, mode=None: frame
    )
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


def test_metadata_feature_schema_is_used():
    from finance_vibe.ml_ranker import feature_columns_from_metadata, _iteration_range

    meta = {"feature_columns": ["RSI", "ATR_Pct"], "best_iteration": 12}
    assert feature_columns_from_metadata(meta) == ["RSI", "ATR_Pct"]
    assert _iteration_range(meta) == (0, 13)
    with pytest.raises(ValueError, match="feature_columns"):
        feature_columns_from_metadata({})


def test_predict_returns_does_not_average_lgb_when_unpromoted(monkeypatch):
    df = pd.DataFrame([{"Symbol": "A", "Score": 80, "RSI": 55.0}])
    monkeypatch.setattr(ml_ranker, "_ranking_spec", lambda mode=None: None)
    out = ml_ranker.predict_returns(df, "daily")
    assert out.isna().all()


def test_horizon_predict_passes_best_iteration(monkeypatch):
    from finance_vibe.coiled_cobra_ml_training import HORIZON_SPECS

    spec = HORIZON_SPECS[0]
    captured = {}

    class FakeBooster:
        feature_names = list(FEATURE_COLS)

        def save_config(self):
            return '{"learner":{"objective":{"name":"binary:logistic"}}}'

        def predict(self, dmat, iteration_range=None):
            captured["iteration_range"] = iteration_range
            return np.array([0.42])

    df = pd.DataFrame([{col: 0.1 for col in FEATURE_COLS}])
    monkeypatch.setattr(
        ml_ranker,
        "load_horizon_artifact",
        lambda mode, s: (
            Path("unused.json"),
            {
                "task": "binary",
                "model_type": "XGBClassifier",
                "horizon": spec["key"],
                "target_column": spec["label_col"],
                "prob_column": spec["prob_col"],
                "feature_columns": list(FEATURE_COLS),
                "feature_count": len(FEATURE_COLS),
                "best_iteration": 7,
                "production_model": "xgb",
            },
        ),
    )
    monkeypatch.setattr(ml_ranker, "_load_xgb", lambda path: FakeBooster())
    probs = ml_ranker.predict_horizon_proba(df, spec, "daily")
    assert captured["iteration_range"] == (0, 8)
    assert probs.iloc[0] == pytest.approx(0.42)


def test_horizon_metadata_rejects_degenerate_best_iteration():
    from finance_vibe.coiled_cobra_ml_training import HORIZON_SPECS
    from finance_vibe.ml_ranker import validate_artifact_metadata

    spec = HORIZON_SPECS[1]
    degenerate = {
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": spec["key"],
        "target_column": spec["label_col"],
        "prob_column": spec["prob_col"],
        "feature_columns": list(FEATURE_COLS),
        "feature_count": len(FEATURE_COLS),
        "best_iteration": 2,
        "production_model": "none",
    }
    with pytest.raises(ValueError, match="degenerate"):
        validate_artifact_metadata(degenerate, spec)


def test_horizon_metadata_mismatch_fails_loudly():
    from finance_vibe.coiled_cobra_ml_training import HORIZON_SPECS
    from finance_vibe.ml_ranker import validate_artifact_metadata

    spec = HORIZON_SPECS[0]
    bad = {
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": "42d",
        "target_column": spec["hit_col"],
        "prob_column": spec["prob_col"],
        "feature_columns": list(FEATURE_COLS),
        "feature_count": len(FEATURE_COLS),
        "best_iteration": 12,
        "production_model": "xgb",
    }
    with pytest.raises(ValueError, match="horizon"):
        validate_artifact_metadata(bad, spec)


def test_horizon_target_mismatch_fails_loudly():
    from finance_vibe.coiled_cobra_ml_training import HORIZON_SPECS
    from finance_vibe.ml_ranker import validate_artifact_metadata

    spec = HORIZON_SPECS[1]
    bad = {
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": spec["key"],
        "target_column": "Hit_25Pct_42d",
        "prob_column": spec["prob_col"],
        "feature_columns": list(FEATURE_COLS),
        "feature_count": len(FEATURE_COLS),
        "best_iteration": 12,
        "production_model": "none",
    }
    with pytest.raises(ValueError, match="target_column"):
        validate_artifact_metadata(bad, spec)


def test_horizon_metadata_accepts_research_hit_as_target_column():
    """On-disk I5 files wrote the last research Hit name; those must still load."""
    from finance_vibe.coiled_cobra_ml_training import HORIZON_SPECS
    from finance_vibe.ml_ranker import validate_artifact_metadata

    spec = HORIZON_SPECS[0]
    meta = {
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": spec["key"],
        "target_column": spec["research_targets"][1],
        "prob_column": spec["prob_col"],
        "feature_columns": list(FEATURE_COLS),
        "feature_count": len(FEATURE_COLS),
        "best_iteration": 56,
        "production_model": "none",
    }
    validate_artifact_metadata(meta, spec)


def test_unpromoted_artifact_is_served_when_flag_on(monkeypatch):
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import HORIZON_SPECS

    monkeypatch.setattr(config, "SERVE_ML_RANKER", True)
    spec = HORIZON_SPECS[1]
    meta = {
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": spec["key"],
        "target_column": spec["label_col"],
        "prob_column": spec["prob_col"],
        "feature_columns": list(FEATURE_COLS),
        "feature_count": len(FEATURE_COLS),
        "best_iteration": 18,
        "production_model": "none",
    }

    def fake_load(mode, s):
        if s["key"] != spec["key"]:
            return None, None
        return Path("unused.json"), meta

    def fake_predict(df, s, mode=None):
        return pd.Series(0.61, index=df.index, dtype="float64")

    monkeypatch.setattr(ml_ranker, "load_horizon_artifact", fake_load)
    monkeypatch.setattr(ml_ranker, "predict_horizon_proba", fake_predict)
    df = pd.DataFrame([{"Symbol": "A", "Score": 80}])
    out = ml_ranker.attach_horizon_probabilities(df, "daily")
    assert out[spec["prob_col"]].iloc[0] == pytest.approx(0.61)
    assert out["ML_Prob_Win_10d"].isna().all()


def test_unpromoted_artifact_stays_null_when_flag_off(monkeypatch):
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import HORIZON_SPECS

    monkeypatch.setattr(config, "SERVE_ML_RANKER", False)
    spec = HORIZON_SPECS[1]
    meta = {
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": spec["key"],
        "target_column": spec["label_col"],
        "prob_column": spec["prob_col"],
        "feature_columns": list(FEATURE_COLS),
        "feature_count": len(FEATURE_COLS),
        "best_iteration": 18,
        "production_model": "none",
    }
    called = {"n": 0}

    def fake_load(mode, s):
        if s["key"] != spec["key"]:
            return None, None
        return Path("unused.json"), meta

    def fake_predict(df, s, mode=None):
        called["n"] += 1
        return pd.Series(0.61, index=df.index, dtype="float64")

    monkeypatch.setattr(ml_ranker, "load_horizon_artifact", fake_load)
    monkeypatch.setattr(ml_ranker, "predict_horizon_proba", fake_predict)
    df = pd.DataFrame([{"Symbol": "A", "Score": 80}])
    out = ml_ranker.attach_horizon_probabilities(df, "daily")
    assert called["n"] == 0
    assert out[spec["prob_col"]].isna().all()
