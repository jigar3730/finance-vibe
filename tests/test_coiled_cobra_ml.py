"""Unit tests for Coiled Cobra ML feature/target alignment with rubric v2.2."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe.coiled_cobra_ml_training import (
    DATE_COL,
    EMBARGO_WEEKS,
    FALLBACK_TARGET_COL,
    FEATURE_COLS,
    HORIZON_SPECS,
    PREFERRED_TARGET_COL,
    assert_features_have_no_leakage,
    compare_rankers,
    drop_duplicate_signals,
    embargo_weeks_for_target,
    load_and_prepare,
    select_target_col,
    temporal_split,
    walk_forward_folds,
)


def _feature_row(**overrides):
    row = {col: 0.1 for col in FEATURE_COLS}
    row.update({
        "Volume_Shelf": 16.0,
        "MACD_Compression": 18.0,
        "Structure": 17.0,
        "RS_Score": 13.5,
        "Coil_Width": 12.0,
        "Proximity_Highs": 8.0,
        "Pct_From_EMA20": 0.02,
        "Pct_From_EMA50": 0.05,
        "ATR_Pct": 0.03,
        "MACD_Crossed": 0,
        "RSI": 55.0,
        "RSI_Healthy": 1,
    })
    row.update(overrides)
    return row


def test_feature_cols_are_pillars_and_raw_not_score():
    assert "Score" not in FEATURE_COLS
    assert "Grade" not in FEATURE_COLS
    assert "Pct_From_Fib618" not in FEATURE_COLS
    assert "MACD_Cross" not in FEATURE_COLS
    assert "Fib_Bonus" not in FEATURE_COLS
    for col in (
        "Volume_Shelf",
        "MACD_Compression",
        "Structure",
        "RS_Score",
        "Coil_Width",
        "Proximity_Highs",
        "Volume_Contraction_Ratio",
        "Dist_High_63_Pct",
        "RSI",
        "MACD_Crossed",
        "Pct_From_EMA20",
        "ATR_Pct",
    ):
        assert col in FEATURE_COLS
    assert_features_have_no_leakage(FEATURE_COLS)


def test_features_reject_forward_columns():
    with pytest.raises(ValueError, match="leakage"):
        assert_features_have_no_leakage(["Volume_Shelf", "Forward_Return_10d"])


def test_select_target_prefers_rel_forward_42d():
    df = pd.DataFrame({
        PREFERRED_TARGET_COL: [0.02, None],
        "Rel_Forward_2w": [0.01, 0.03],
        FALLBACK_TARGET_COL: [0.01, 0.03],
    })
    assert select_target_col(df) == PREFERRED_TARGET_COL


def test_select_target_falls_back_to_absolute():
    df = pd.DataFrame({FALLBACK_TARGET_COL: [0.01, 0.03]})
    assert select_target_col(df) == FALLBACK_TARGET_COL


def test_select_target_weekly_prefers_13w():
    df = pd.DataFrame({
        "Mode": ["weekly", "weekly"],
        "Rel_Forward_13w": [0.04, 0.05],
        "Rel_Forward_42d": [0.9, 0.8],
        "Rel_Forward_2w": [0.01, 0.02],
    })
    assert select_target_col(df) == "Rel_Forward_13w"


def test_embargo_matches_training_horizon():
    assert embargo_weeks_for_target("Rel_Forward_42d") == 9
    assert embargo_weeks_for_target("Hit_25Pct_42d") == 9
    assert embargo_weeks_for_target("Hit_15Pct_21d") == 5
    assert embargo_weeks_for_target("Hit_10Pct_10d") == 2
    assert embargo_weeks_for_target("Forward_Return_10d") == 2
    assert embargo_weeks_for_target("Rel_Forward_13w") == 13
    assert embargo_weeks_for_target("Rel_Forward_2w") == 2
    for spec in HORIZON_SPECS:
        assert embargo_weeks_for_target(spec["hit_col"]) == spec["embargo_weeks"]


def test_load_and_prepare_keeps_new_coils_only(tmp_path: Path):
    rows = []
    for i, (new_coil, rel) in enumerate(
        [(True, 0.04), (False, 0.09), (True, 0.01), (True, None)]
    ):
        row = {
            **_feature_row(),
            "Symbol": "AAA",
            "Signal Date": f"2024-01-{i + 1:02d}",
            "Is_New_Coil": new_coil,
            "Rel_Forward_42d": rel,
            "Rel_Forward_2w": 0.02,
            "Forward_Return_2w": 0.02,
            "Forward_Return_10d": 0.02,
            "Max_Return_10d": 0.12,
            "Hit_10Pct_10d": 1,
            "Win_10d": 1,
            "Stock Entry": 100.0,
        }
        rows.append(row)
    csv_path = tmp_path / "trades.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out = load_and_prepare(csv_path)
    assert len(out) == 3
    assert (out["Is_New_Coil"] == True).all()  # noqa: E712
    assert "Stock Entry" not in out.columns
    assert out["Hit_10Pct_10d"].notna().all()


def test_load_and_prepare_aborts_conflicting_duplicate_symbol_date(tmp_path: Path):
    rows = [
        {**_feature_row(), "Symbol": "AAA", "Signal Date": "2024-01-02",
         "Is_New_Coil": True, "Forward_Return_2w": 0.01, "Forward_Return_10d": 0.01,
         "Max_Return_10d": 0.11, "Hit_10Pct_10d": 1, "Rel_Forward_42d": 0.02},
        {**_feature_row(), "Symbol": "AAA", "Signal Date": "2024-01-02",
         "Is_New_Coil": True, "Forward_Return_2w": 0.99, "Forward_Return_10d": 0.99,
         "Max_Return_10d": 0.99, "Hit_10Pct_10d": 1, "Rel_Forward_42d": 0.02},
        {**_feature_row(), "Symbol": "BBB", "Signal Date": "2024-01-02",
         "Is_New_Coil": True, "Forward_Return_2w": 0.03, "Forward_Return_10d": 0.03,
         "Max_Return_10d": 0.03, "Hit_10Pct_10d": 0, "Rel_Forward_42d": 0.02},
    ]
    csv_path = tmp_path / "trades.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        load_and_prepare(csv_path)


def test_drop_duplicate_signals_reports_extras():
    df = pd.DataFrame({
        "Symbol": ["A", "A", "B"],
        "Signal Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
    })
    out = drop_duplicate_signals(df)
    assert len(out) == 2


def test_drop_duplicate_signals_removes_exact_duplicate():
    df = pd.DataFrame({
        "Symbol": ["A", "A", "B"],
        "Signal Date": pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-01-01"]
        ),
        "Score": [80, 80, 75],
    })
    out = drop_duplicate_signals(df)
    assert len(out) == 2


def test_temporal_split_embargoes_label_horizon():
    dates = pd.date_range("2020-01-03", periods=160, freq="W-FRI")
    df = pd.DataFrame({DATE_COL: dates})
    train, val, test, bounds = temporal_split(df)
    assert bounds["embargo_weeks"] == EMBARGO_WEEKS
    assert len(train) and len(val) and len(test)
    train_end = train[DATE_COL].max()
    val_start = val[DATE_COL].min()
    val_end = val[DATE_COL].max()
    test_start = test[DATE_COL].min()
    assert (val_start - train_end) >= pd.Timedelta(weeks=EMBARGO_WEEKS)
    assert (test_start - val_end) >= pd.Timedelta(weeks=EMBARGO_WEEKS)


def test_walk_forward_is_chronological_with_embargo():
    dates = pd.date_range("2018-01-05", periods=400, freq="W-FRI")
    df = pd.DataFrame({DATE_COL: dates, "Hit_10Pct_10d": 1})
    folds = walk_forward_folds(df, embargo_weeks=2, min_train_rows=20, min_val_rows=5, min_test_rows=5)
    assert len(folds) >= 2
    for fold in folds:
        assert fold["train"][DATE_COL].max() < fold["val"][DATE_COL].min()
        assert fold["val"][DATE_COL].max() < fold["test"][DATE_COL].min()
        assert (fold["val"][DATE_COL].min() - fold["train"][DATE_COL].max()) >= pd.Timedelta(weeks=2)
        assert (fold["test"][DATE_COL].min() - fold["val"][DATE_COL].max()) >= pd.Timedelta(weeks=2)
    starts = [fold["test_start"] for fold in folds]
    assert starts == sorted(starts)


def test_promote_requires_higher_return_and_win_rate():
    spec = HORIZON_SPECS[0]
    dates = pd.to_datetime(["2024-01-02"] * 12 + ["2024-01-03"] * 12)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        DATE_COL: dates,
        "Score": np.linspace(70, 95, 24),
        spec["prob_col"]: np.linspace(0.9, 0.1, 24),
        spec["forward_col"]: rng.normal(0.01, 0.02, 24),
        spec["hit_col"]: (rng.random(24) > 0.5).astype(int),
        spec["win_col"]: (rng.random(24) > 0.4).astype(int),
    })
    df[spec["forward_col"]] = np.where(df["Score"] > 85, 0.08, -0.02)
    df[spec["win_col"]] = (df[spec["forward_col"]] > 0).astype(int)
    df[spec["prob_col"]] = 1.0 - (df["Score"] - 70) / 25.0
    cmp = compare_rankers(df, spec)
    assert cmp["promote"] is False


def test_train_horizon_classifier_walk_forward(tmp_path: Path):
    pytest.importorskip("xgboost")
    import json

    from finance_vibe.coiled_cobra_ml_training import train_horizon

    spec = dict(HORIZON_SPECS[0])
    n = 420
    dates = pd.date_range("2019-01-02", periods=n, freq="B")
    rng = np.random.default_rng(7)
    coil = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    hit = ((coil + 0.3 * noise) > 0.4).astype(int)
    fwd = np.where(hit == 1, 0.12 + 0.02 * rng.random(n), -0.03 + 0.02 * rng.random(n))
    rows = []
    for i in range(n):
        row = _feature_row(
            Coil_Width=12.0 + coil[i],
            RSI=50 + 10 * coil[i],
            ATR_Pct=0.03,
        )
        row.update({
            "Symbol": f"T{i % 15}",
            "Signal Date": dates[i],
            "Is_New_Coil": True,
            spec["forward_col"]: fwd[i],
            spec["max_col"]: max(fwd[i], 0.11 if hit[i] else 0.02),
            spec["hit_col"]: hit[i],
            spec["win_col"]: int(fwd[i] > 0),
            "Score": 70 + 8 * coil[i],
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    result = train_horizon(
        df,
        spec,
        tmp_path,
        xgb_params={"n_estimators": 40, "learning_rate": 0.1, "min_child_weight": 1},
    )
    assert (tmp_path / spec["model_filename"]).is_file()
    assert (tmp_path / spec["metadata_filename"]).is_file()
    payload = json.loads((tmp_path / spec["metadata_filename"]).read_text(encoding="utf-8"))
    assert payload["feature_columns"] == list(FEATURE_COLS)
    assert payload["best_iteration"] is not None
    assert payload["best_iteration"] >= 0
    assert "promoted" in payload
    assert payload["production_model"] in {"xgb", "none"}
    assert payload["model_type"] == "XGBClassifier"
    assert payload["feature_count"] == len(FEATURE_COLS)
    assert payload["source_csv_sha256"] is None
    assert set(payload["classification_metrics"]["xgb"]) == {
        "precision", "recall", "f1", "pr_auc", "roc_auc", "brier"
    }
    assert "top_50pct" in payload["model_metrics"]["xgb"]["selections"]
    assert "top_5_per_date" in payload["model_metrics"]["xgb"]["selections"]
    assert result["n_oos"] > 0
    assert "score_top10" in result
