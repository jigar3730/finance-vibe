"""Unit tests for Coiled Cobra ML feature/target alignment with rubric v2.2."""

from pathlib import Path

import pandas as pd

from finance_vibe.coiled_cobra_ml_training import (
    DATE_COL,
    EMBARGO_WEEKS,
    FALLBACK_TARGET_COL,
    FEATURE_COLS,
    PREFERRED_TARGET_COL,
    embargo_weeks_for_target,
    load_and_prepare,
    select_target_col,
    temporal_split,
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
    assert embargo_weeks_for_target("Rel_Forward_13w") == 13
    assert embargo_weeks_for_target("Rel_Forward_2w") == 2


def test_load_and_prepare_keeps_new_coils_only(tmp_path: Path):
    rows = []
    for i, (new_coil, rel) in enumerate(
        [(True, 0.04), (False, 0.09), (True, 0.01), (True, None)]
    ):
        row = {
            **_feature_row(),
            "Signal Date": f"2024-01-{i + 1:02d}",
            "Is_New_Coil": new_coil,
            "Rel_Forward_42d": rel,
            "Rel_Forward_2w": 0.02,
            "Forward_Return_2w": 0.02,
            "Stock Entry": 100.0,
        }
        rows.append(row)
    csv_path = tmp_path / "trades.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out = load_and_prepare(csv_path)
    assert len(out) == 2
    assert (out["Is_New_Coil"] == True).all()  # noqa: E712
    assert out["Rel_Forward_42d"].notna().all()
    assert "Stock Entry" not in out.columns


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
