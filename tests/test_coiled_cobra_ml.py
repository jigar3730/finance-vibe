"""Unit tests for Coiled Cobra ML feature/target alignment with rubric v2.1."""

from pathlib import Path

import pandas as pd

from finance_vibe.coiled_cobra_ml_training import (
    DATE_COL,
    EMBARGO_WEEKS,
    FALLBACK_TARGET_COL,
    FEATURE_COLS,
    PREFERRED_TARGET_COL,
    load_and_prepare,
    select_target_col,
    temporal_split,
)


PILLAR_ROW = {
    "Volume_Shelf": 16.0,
    "MACD_Compression": 18.0,
    "Structure": 17.0,
    "RS_Score": 13.5,
    "Coil_Width": 12.0,
    "MACD_Cross": 0.0,
    "Fib_Bonus": 1.2,
    "Pct_From_EMA20": 0.02,
    "Pct_From_EMA50": 0.05,
    "ATR_Pct": 0.03,
}


def test_feature_cols_are_pillars_not_score_or_fib_pct():
    assert "Score" not in FEATURE_COLS
    assert "Grade" not in FEATURE_COLS
    assert "Pct_From_Fib618" not in FEATURE_COLS
    assert "Pct_From_Fib786" not in FEATURE_COLS
    for col in (
        "Volume_Shelf",
        "MACD_Compression",
        "Structure",
        "RS_Score",
        "Coil_Width",
        "MACD_Cross",
        "Fib_Bonus",
        "Pct_From_EMA20",
        "Pct_From_EMA50",
        "ATR_Pct",
    ):
        assert col in FEATURE_COLS
    assert len(FEATURE_COLS) == 10


def test_select_target_prefers_rel_forward():
    df = pd.DataFrame({
        PREFERRED_TARGET_COL: [0.02, None],
        FALLBACK_TARGET_COL: [0.01, 0.03],
    })
    assert select_target_col(df) == PREFERRED_TARGET_COL


def test_select_target_falls_back_to_absolute():
    df = pd.DataFrame({FALLBACK_TARGET_COL: [0.01, 0.03]})
    assert select_target_col(df) == FALLBACK_TARGET_COL


def test_load_and_prepare_keeps_new_coils_only(tmp_path: Path):
    rows = []
    for i, (new_coil, rel) in enumerate(
        [(True, 0.04), (False, 0.09), (True, 0.01), (True, None)]
    ):
        row = {
            **PILLAR_ROW,
            "Signal Date": f"2024-01-{i + 1:02d}",
            "Is_New_Coil": new_coil,
            "Rel_Forward_2w": rel,
            "Forward_Return_2w": 0.02,
            "Stock Entry": 100.0,
        }
        rows.append(row)
    csv_path = tmp_path / "trades.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out = load_and_prepare(csv_path)
    assert len(out) == 2
    assert (out["Is_New_Coil"] == True).all()  # noqa: E712
    assert out["Rel_Forward_2w"].notna().all()
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
