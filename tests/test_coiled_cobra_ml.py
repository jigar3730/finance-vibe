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
    FOLD_COL,
    HORIZON_SPECS,
    MIN_BEST_ITERATION,
    MIN_TRAIN_ROWS,
    PREFERRED_TARGET_COL,
    TOP_N_PER_DATE,
    TOP_N_PER_DATE_LEVELS,
    _ensure_horizon_targets,
    assert_features_have_no_leakage,
    booster_is_degenerate,
    compare_rankers,
    drop_duplicate_signals,
    embargo_weeks_for_target,
    evaluate_ranker,
    load_and_prepare,
    save_xgb_artifact,
    select_target_col,
    temporal_split,
    top_fraction,
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


def test_hit_labels_never_fall_back_to_the_horizon_close():
    """A close-only CSV must fail, not train on "close >= X%" mislabelled as a hit."""
    df = pd.DataFrame({
        "Forward_Return_10d": [0.12, -0.03],
        "Forward_Return_21d": [0.20, -0.03],
        "Forward_Return_42d": [0.30, -0.03],
    })

    lenient = _ensure_horizon_targets(df, require_max=False)
    for spec in HORIZON_SPECS:
        for hit_col, _ in spec["all_hits"]:
            assert hit_col not in lenient.columns
        assert lenient[spec["win_col"]].tolist() == [1.0, 0.0]

    with pytest.raises(ValueError, match="Max_Return_10d is missing"):
        _ensure_horizon_targets(df, require_max=True)


def test_hit_labels_come_from_the_intra_horizon_high():
    df = pd.DataFrame({
        "Forward_Return_10d": [0.0],
        "Forward_Return_21d": [0.0],
        "Forward_Return_42d": [0.0],
        "Max_Return_10d": [0.12],
        "Max_Return_21d": [0.12],
        "Max_Return_42d": [0.30],
    })

    out = _ensure_horizon_targets(df, require_max=True)

    # Flat closes, but the highs paid: hits fire where the close-based fallback would not.
    assert out["Win_10d"].iloc[0] == 0.0
    assert out["Hit_10Pct_10d"].iloc[0] == 1.0
    assert out["Hit_15Pct_10d"].iloc[0] == 0.0
    assert out["Hit_15Pct_21d"].iloc[0] == 0.0
    assert out["Hit_25Pct_42d"].iloc[0] == 1.0
    assert out["Hit_50Pct_42d"].iloc[0] == 0.0


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
    assert embargo_weeks_for_target("Win_10d") == 2
    assert embargo_weeks_for_target("Win_21d") == 5
    assert embargo_weeks_for_target("Win_42d") == 9
    assert embargo_weeks_for_target("Forward_Return_10d") == 2
    assert embargo_weeks_for_target("Rel_Forward_13w") == 13
    assert embargo_weeks_for_target("Rel_Forward_2w") == 2
    for spec in HORIZON_SPECS:
        assert embargo_weeks_for_target(spec["hit_col"]) == spec["embargo_weeks"]
        assert embargo_weeks_for_target(spec["label_col"]) == spec["embargo_weeks"]


def test_primary_training_target_is_win_not_mfe_hit():
    """T1: MFE hit is a volatility proxy; the classifier trains on close > 0."""
    for spec in HORIZON_SPECS:
        assert spec["label_col"] == spec["win_col"]
        assert spec["label_col"].startswith("Win_")
        assert spec["hit_col"].startswith("Hit_")
        assert spec["label_col"] != spec["hit_col"]
        assert spec["hit_col"] in spec["research_targets"]
        assert spec["prob_col"] == f"ML_Prob_{spec['label_col']}"


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
            "Forward_Return_21d": 0.03,
            "Forward_Return_42d": 0.05,
            "Max_Return_10d": 0.12,
            "Max_Return_21d": 0.18,
            "Max_Return_42d": 0.30,
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
    # Native Max_Return_* in the CSV means no raw-OHLC enrichment was needed.
    assert out["Hit_15Pct_21d"].eq(1.0).all()
    assert out["Hit_25Pct_42d"].eq(1.0).all()
    assert out["Hit_50Pct_42d"].eq(0.0).all()


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


def test_default_min_train_rows_excludes_sub_thousand_windows():
    """C3: the 80-row floor admitted a 505-row fold that stopped at 0 trees."""
    assert MIN_TRAIN_ROWS >= 1000
    dates = pd.date_range("2015-01-01", periods=2500, freq="B")
    df = pd.DataFrame({DATE_COL: dates})
    folds = walk_forward_folds(df, embargo_weeks=2)
    assert folds, "10y of business days should still produce expanding-window folds"
    for fold in folds:
        assert len(fold["train"]) >= MIN_TRAIN_ROWS


def test_booster_is_degenerate_below_four_trees():
    assert MIN_BEST_ITERATION == 3
    assert booster_is_degenerate(None)
    assert booster_is_degenerate(0)
    assert booster_is_degenerate(2)
    assert not booster_is_degenerate(3)
    assert not booster_is_degenerate(44)


def test_save_xgb_artifact_refuses_degenerate_and_deletes_stale(tmp_path: Path):
    path = tmp_path / "coiled_cobra_xgb_21d.json"
    path.write_text("stale leftover", encoding="utf-8")
    saved = save_xgb_artifact(object(), path, best_iteration=2)
    assert saved is False
    assert not path.is_file()


def _two_fold_oos(spec: dict) -> pd.DataFrame:
    """Fold 1 = high probabilities / bad returns, fold 2 = low probabilities / good ones.

    Ranking the pooled frame would hand the whole top decile to fold 1 purely
    because its model emits a higher probability scale.
    """
    rows = []
    for k in range(10):
        rows.append({
            FOLD_COL: 1,
            DATE_COL: pd.Timestamp("2024-01-02"),
            "Score": 70 + k,
            spec["prob_col"]: 0.90 - 0.01 * k,
            spec["forward_col"]: -0.05,
        })
    for k in range(10):
        prob = 0.20 - 0.01 * k
        rows.append({
            FOLD_COL: 2,
            DATE_COL: pd.Timestamp("2025-01-02"),
            "Score": 70 + k,
            spec["prob_col"]: prob,
            # Only the five strongest picks in this fold pay off.
            spec["forward_col"]: 0.10 if prob >= 0.16 else -0.20,
        })
    df = pd.DataFrame(rows)
    df[spec["win_col"]] = (df[spec["forward_col"]] > 0).astype(int)
    df[spec["hit_col"]] = df[spec["win_col"]]
    return df


def test_top_fraction_cuts_within_each_fold():
    spec = HORIZON_SPECS[0]
    oos = _two_fold_oos(spec)

    selected = top_fraction(oos, spec["prob_col"], 0.50)
    assert len(selected) == 10
    assert selected[FOLD_COL].value_counts().to_dict() == {1: 5, 2: 5}
    # Highest-probability half of each fold: fold 2 contributes its winners.
    assert selected[spec["forward_col"]].mean() == pytest.approx(0.025)


def test_top_fraction_without_grouping_is_scale_biased():
    """Documents the behavior the fold cut exists to avoid."""
    spec = HORIZON_SPECS[0]
    oos = _two_fold_oos(spec)

    selected = top_fraction(oos, spec["prob_col"], 0.50, group_col=None)
    assert set(selected[FOLD_COL]) == {1}
    assert selected[spec["forward_col"]].mean() == pytest.approx(-0.05)


def test_evaluate_ranker_reports_per_fold_selection_scope():
    spec = HORIZON_SPECS[0]
    oos = _two_fold_oos(spec)

    result = evaluate_ranker(oos, spec, spec["prob_col"], "xgb")
    assert result["selection_scope"] == "per_fold"
    assert result["n_selection_groups"] == 2
    top_half = result["selections"]["top_50pct"]
    assert top_half["n"] == 10
    assert top_half["avg_fwd"] == pytest.approx(0.025)
    assert result["selections"]["population"]["n"] == 20


def test_evaluate_ranker_single_model_frame_ranks_globally():
    spec = HORIZON_SPECS[0]
    oos = _two_fold_oos(spec).drop(columns=[FOLD_COL])

    result = evaluate_ranker(oos, spec, spec["prob_col"], "xgb")
    assert result["selection_scope"] == "single_model"
    assert result["n_selection_groups"] == 1
    assert result["selections"]["top_50pct"]["avg_fwd"] == pytest.approx(-0.05)


def test_per_date_levels_stay_below_typical_daily_breadth():
    """The pool averages ~9 new coils per date; a top-20 cut would select everything."""
    assert max(TOP_N_PER_DATE_LEVELS) <= 5
    assert TOP_N_PER_DATE == max(TOP_N_PER_DATE_LEVELS)


def test_selected_fraction_reports_how_much_a_cut_discards():
    spec = HORIZON_SPECS[0]
    oos = _two_fold_oos(spec)

    selections = evaluate_ranker(oos, spec, spec["prob_col"], "xgb")["selections"]
    assert selections["population"]["selected_fraction"] == pytest.approx(1.0)
    assert selections["top_50pct"]["selected_fraction"] == pytest.approx(0.5)
    # Both fixture dates hold 10 rows, so a top-5 per-date cut keeps half.
    assert selections["top_5_per_date"]["selected_fraction"] == pytest.approx(0.5)


def _promotion_frame(spec: dict, *, top_decile_return: float) -> pd.DataFrame:
    """Every setup loses except the ones the model ranks highest.

    ``Score`` is deliberately ordered against the model so the two disagree,
    and only the model's top decile gets ``top_decile_return``. A positive value
    therefore beats Score, random, and the population on all three metrics; a
    negative value loses on all three.
    """
    n = 40
    df = pd.DataFrame({
        DATE_COL: pd.to_datetime(["2024-01-02"] * 20 + ["2024-01-03"] * 20),
        "Score": np.linspace(95, 70, n),
        spec["prob_col"]: np.linspace(0.1, 0.9, n),
        spec["forward_col"]: np.linspace(-0.05, -0.01, n),
        "_Random_Rank": np.random.default_rng(0).random(n),
    })
    top_decile = df[spec["prob_col"]].rank(pct=True) > 0.90
    df.loc[top_decile, spec["forward_col"]] = top_decile_return
    df[spec["win_col"]] = (df[spec["forward_col"]] > 0).astype(int)
    df[spec["hit_col"]] = df[spec["win_col"]]
    return df


def test_promote_passes_when_ml_beats_every_baseline():
    """Positive control: the gate must be able to fire, not just always refuse."""
    spec = HORIZON_SPECS[0]
    cmp = compare_rankers(_promotion_frame(spec, top_decile_return=0.14), spec)
    assert cmp["promote"] is True
    assert cmp["promotion"]["failed_metrics"] == []
    assert set(cmp["promotion"]["baselines"]) == {"score", "random", "population"}


def test_promote_requires_higher_return_and_win_rate():
    spec = HORIZON_SPECS[0]
    cmp = compare_rankers(_promotion_frame(spec, top_decile_return=-0.10), spec)
    assert cmp["promote"] is False
    assert cmp["promotion"]["failed_metrics"] == ["avg_fwd", "med_fwd", "win_rate"]


def test_promotion_gate_ignores_mfe_hit_rate():
    """MFE hit rate is a volatility-loaded research label, not a promotion input."""
    spec = HORIZON_SPECS[0]
    cmp = compare_rankers(_promotion_frame(spec, top_decile_return=-0.10), spec)
    assert set(cmp["promotion"]["metrics"]) == {"avg_fwd", "med_fwd", "win_rate"}
    # The model trivially wins its own objective, which must not count.
    xgb = cmp["models"]["xgb"]["selections"]["top_10pct"]
    score = cmp["models"]["score"]["selections"]["top_10pct"]
    assert xgb["hit_rate"] >= score["hit_rate"]


def test_promotion_requires_beating_random_and_population():
    """Matching Score is not enough when Score itself adds nothing over random."""
    spec = HORIZON_SPECS[0]
    flat = _promotion_frame(spec, top_decile_return=0.14)
    # Give every row the same outcome, so no ranker can out-select any other.
    flat[spec["forward_col"]] = 0.01
    flat[spec["win_col"]] = 1
    flat[spec["hit_col"]] = 1
    cmp = compare_rankers(flat, spec)
    assert cmp["promote"] is False
    assert cmp["promotion"]["metrics"]["avg_fwd"]["beats"] == {
        "score": False,
        "population": False,
        "random": False,
    }


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
    assert payload["best_iteration"] >= MIN_BEST_ITERATION
    assert payload["model_saved"] is True
    assert payload["min_best_iteration"] == MIN_BEST_ITERATION
    assert payload["walk_forward"]["n_folds"] == len(
        [f for f in payload["walk_forward"]["folds"] if not f.get("skipped")]
    )
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
    assert payload["promotion_metrics"] == ["avg_fwd", "med_fwd", "win_rate"]
    assert payload["promotion_checks"]["selection"] == "top_10pct"
    assert result["n_oos"] > 0
    assert "score_per_date" in result

    # Batch 0: raw OOS predictions are persisted so later measurement changes
    # can be re-scored offline instead of requiring a retrain.
    oos_path = tmp_path / payload["artifacts"]["oos_predictions"]
    assert oos_path.is_file()
    oos = pd.read_csv(oos_path)
    assert len(oos) == payload["oos_rows"]
    for col in ("Symbol", "Signal Date", FOLD_COL, "Score", spec["prob_col"],
                spec["forward_col"], spec["hit_col"]):
        assert col in oos.columns
    assert oos[FOLD_COL].nunique() == payload["walk_forward"]["n_folds"]


def test_train_horizon_skips_a_stump_fold_and_keeps_the_rest(tmp_path: Path, monkeypatch):
    pytest.importorskip("xgboost")
    import json

    import finance_vibe.coiled_cobra_ml_training as mod
    from finance_vibe.coiled_cobra_ml_training import train_horizon

    monkeypatch.setattr(mod, "MIN_TRAIN_ROWS", 20)
    real_best = mod._best_iteration
    seen = {"n": 0}

    def first_fold_is_a_stump(model):
        seen["n"] += 1
        if seen["n"] == 1:
            return 0
        return real_best(model)

    monkeypatch.setattr(mod, "_best_iteration", first_fold_is_a_stump)

    spec = dict(HORIZON_SPECS[0])
    n = 800
    dates = pd.date_range("2018-01-02", periods=n, freq="B")
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

    result = train_horizon(
        pd.DataFrame(rows),
        spec,
        tmp_path,
        xgb_params={"n_estimators": 40, "learning_rate": 0.1, "min_child_weight": 1},
    )
    payload = json.loads((tmp_path / spec["metadata_filename"]).read_text(encoding="utf-8"))
    skipped = [f for f in payload["walk_forward"]["folds"] if f.get("skipped")]
    assert skipped and skipped[0]["best_iteration"] == 0
    assert payload["walk_forward"]["n_skipped"] >= 1
    assert payload["walk_forward"]["n_folds"] >= 1
    oos = pd.read_csv(tmp_path / payload["artifacts"]["oos_predictions"])
    assert 1 not in set(oos[FOLD_COL].astype(int))
    assert result["n_oos"] == len(oos)


def test_train_horizon_raises_when_every_fold_is_a_stump(tmp_path: Path):
    pytest.importorskip("xgboost")
    from finance_vibe.coiled_cobra_ml_training import train_horizon

    spec = dict(HORIZON_SPECS[0])
    n = 420
    dates = pd.date_range("2019-01-02", periods=n, freq="B")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = _feature_row()
        row.update({
            "Symbol": f"T{i % 15}",
            "Signal Date": dates[i],
            "Is_New_Coil": True,
            spec["forward_col"]: 0.02,
            spec["max_col"]: 0.12,
            spec["hit_col"]: int(i % 5 == 0),
            spec["win_col"]: 1,
            "Score": 80,
        })
        rows.append(row)

    with pytest.raises(RuntimeError, match="degenerate"):
        train_horizon(
            pd.DataFrame(rows),
            spec,
            tmp_path,
            xgb_params={"n_estimators": 1, "learning_rate": 0.01, "min_child_weight": 1},
        )
