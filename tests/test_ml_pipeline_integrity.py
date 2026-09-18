"""Regression tests for ML pipeline integrity patches.

1. Temporal split purges a forward-return-horizon embargo at each boundary.
2. ``ml_ranker`` resolves models strictly per mode (no weekly -> daily fallback).
3. Rubric version is stamped on CSVs/models and validated at train + inference.
4. Training auto-selects the newest trades CSV (no hard-coded filename).

All hermetic: tiny models are trained on random data in ``tmp_path``.
"""
import json
import os
import logging

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe import ml_ranker
from finance_vibe import coiled_cobra_ml_training as trn
from finance_vibe.coiled_cobra_ml_training import FEATURE_COLS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _weekly_frame(n_weeks: int = 200, per_week: int = 3) -> pd.DataFrame:
    """Monday-labelled weekly signals, ``per_week`` rows per date."""
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="7D")
    rows = [{trn.DATE_COL: d, "Symbol": f"T{k}"} for d in dates for k in range(per_week)]
    return pd.DataFrame(rows)


def _trades_csv(path, n_weeks=260, per_week=3, rubric_version=config.RUBRIC_VERSION, seed=0):
    rng = np.random.default_rng(seed)
    df = _weekly_frame(n_weeks, per_week)
    n = len(df)
    df["Score"] = rng.uniform(70, 95, n).round(2)
    for col in ("Pct_From_EMA20", "Pct_From_EMA50", "Pct_From_Fib618", "Pct_From_Fib786"):
        df[col] = rng.normal(0.02, 0.05, n).round(4)
    df["ATR_Pct"] = rng.uniform(0.02, 0.08, n).round(4)
    df[trn.TARGET_COL] = rng.normal(0.0, 0.05, n).round(4)
    df["Outcome"] = "stopped"          # leakage col that must be dropped
    df["R Multiple"] = -1.0
    if rubric_version is not None:
        df[config.RUBRIC_VERSION_COL] = rubric_version
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_tiny_models(art_dir, *, mode="weekly", rubric_version=None,
                       feature_names=None, seed=0):
    """Train + save tiny XGB/LGB models and metadata using the real writer."""
    xgboost = pytest.importorskip("xgboost")
    lightgbm = pytest.importorskip("lightgbm")
    art_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURE_COLS))), columns=FEATURE_COLS)
    y = rng.normal(size=200)

    xgb = xgboost.XGBRegressor(n_estimators=5, max_depth=2, random_state=seed).fit(X, y)
    lgb = lightgbm.LGBMRegressor(
        n_estimators=5, max_depth=2, min_child_samples=5, verbose=-1, random_state=seed
    ).fit(X, y)
    xgb_path = art_dir / trn.XGB_MODEL_FILENAME
    lgb_path = art_dir / trn.LGB_MODEL_FILENAME
    xgb.get_booster().save_model(str(xgb_path))
    lgb.booster_.save_model(str(lgb_path))

    m = {"mae": 0.0, "rmse": 0.0}
    trn._save_model_metadata(
        art_dir, feature_names or list(FEATURE_COLS), m, m, m, m,
        xgb_path, lgb_path, art_dir / "plot.png",
        {
            "rubric_version": rubric_version or config.RUBRIC_VERSION,
            "mode": mode, "trained_at": "2026-01-01T00:00:00+00:00",
            "git_sha": "test", "source_csv": "x.csv", "split": {},
        },
    )
    return art_dir


def _scan_rows(n=4) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(0.02, 0.03, size=(n, len(FEATURE_COLS))), columns=FEATURE_COLS)


@pytest.fixture
def mode_dirs(tmp_path, monkeypatch):
    """Point every mode's log dir at tmp_path/<mode>."""
    monkeypatch.delenv(ml_ranker.MODEL_DIR_ENV, raising=False)

    def fake_log_dir(mode="weekly"):
        _, profile = config.resolve_pipeline_mode(mode)
        d = tmp_path / profile
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(config, "get_log_dir", fake_log_dir)
    return tmp_path


# ---------------------------------------------------------------------------
# Patch 1: temporal split embargo
# ---------------------------------------------------------------------------

def test_split_embargo_no_label_overlap_across_boundaries():
    df = _weekly_frame(n_weeks=200)
    train, val, test, b = trn._temporal_split(df)
    horizon = pd.Timedelta(weeks=trn.TARGET_HORIZON_WEEKS)

    assert b["embargo_weeks"] == trn.TARGET_HORIZON_WEEKS
    # A label at date t is realised by t + horizon; it must land before the
    # next partition's first row.
    assert train[trn.DATE_COL].max() + horizon < val[trn.DATE_COL].min()
    assert val[trn.DATE_COL].max() + horizon < test[trn.DATE_COL].min()


def test_split_purged_rows_are_exactly_the_embargo_windows():
    df = _weekly_frame(n_weeks=200)
    train, val, test, b = trn._temporal_split(df)
    d = df[trn.DATE_COL]
    expected = int(
        ((d >= b["train_end"]) & (d < b["val_start"])).sum()
        + ((d >= b["val_end"]) & (d < b["test_start"])).sum()
    )
    assert b["purged_rows"] == expected > 0
    assert len(train) + len(val) + len(test) + b["purged_rows"] == len(df)
    # Test partition is never shrunk by the embargo.
    assert test[trn.DATE_COL].min() >= b["test_start"]
    assert test[trn.DATE_COL].max() == b["max_date"]


def test_split_zero_embargo_is_contiguous():
    df = _weekly_frame(n_weeks=200)
    train, val, test, b = trn._temporal_split(df, embargo_weeks=0)
    assert b["purged_rows"] == 0
    assert len(train) + len(val) + len(test) == len(df)


def test_split_rejects_negative_embargo():
    with pytest.raises(ValueError):
        trn._temporal_split(_weekly_frame(), embargo_weeks=-1)


# ---------------------------------------------------------------------------
# Patch 2: strict mode-bound model resolution
# ---------------------------------------------------------------------------

def test_weekly_model_is_not_resolved_for_other_modes(mode_dirs):
    _write_tiny_models(mode_dirs / "weekly", mode="weekly")

    weekly = ml_ranker._resolve_model_paths("weekly")
    assert weekly["xgb"] and weekly["lgb"] and weekly["metadata"]

    for other in ("daily", "high_beta"):
        paths = ml_ranker._resolve_model_paths(other)
        assert paths == {"xgb": None, "lgb": None, "metadata": None}, other


def test_predict_daily_does_not_use_weekly_model(mode_dirs):
    _write_tiny_models(mode_dirs / "weekly", mode="weekly")
    X = _scan_rows()
    assert ml_ranker.predict_returns(X, "weekly").notna().all()
    assert ml_ranker.predict_returns(X, "daily").isna().all()
    assert ml_ranker.predict_returns(X, "high_beta").isna().all()


def test_weekly_model_copied_into_daily_dir_is_rejected_by_metadata_mode(mode_dirs, caplog):
    # Even physically placed in the daily silo, a weekly-bound model is refused.
    _write_tiny_models(mode_dirs / "daily", mode="weekly")
    with caplog.at_level(logging.WARNING, logger=ml_ranker.logger.name):
        out = ml_ranker.predict_returns(_scan_rows(), "daily")
    assert out.isna().all()
    assert "bound to mode" in caplog.text


def test_xgb_and_lgb_are_never_mixed_across_directories(mode_dirs):
    # Only the LGB file sits in the mode dir; XGB exists elsewhere -> no XGB.
    _write_tiny_models(mode_dirs / "elsewhere", mode="weekly")
    (mode_dirs / "weekly").mkdir(exist_ok=True)
    (mode_dirs / "elsewhere" / trn.LGB_MODEL_FILENAME).replace(
        mode_dirs / "weekly" / trn.LGB_MODEL_FILENAME
    )
    paths = ml_ranker._resolve_model_paths("weekly")
    assert paths["xgb"] is None and paths["metadata"] is None


def test_model_dir_env_override(mode_dirs, monkeypatch):
    _write_tiny_models(mode_dirs / "custom", mode="weekly")
    monkeypatch.setenv(ml_ranker.MODEL_DIR_ENV, str(mode_dirs / "custom"))
    assert ml_ranker.predict_returns(_scan_rows(), "weekly").notna().all()


# ---------------------------------------------------------------------------
# Patch 3: rubric version stamping + metadata validation
# ---------------------------------------------------------------------------

def test_rubric_version_is_stamped_on_backtest_and_backfill_csvs(tmp_path, monkeypatch):
    from finance_vibe import coiled_cobra_backtest as bt

    raw, logs = tmp_path / "raw", tmp_path / "logs"
    raw.mkdir()
    monkeypatch.setattr(
        config, "get_mode_config",
        lambda mode=None: {"raw_dir": str(raw), "logs_dir": str(logs)},
    )
    trades = bt.run_backtest("weekly")
    backfill = bt.generate_backfill("weekly")
    assert config.RUBRIC_VERSION_COL in trades.columns
    assert config.RUBRIC_VERSION_COL in backfill.columns
    for f in logs.glob("*.csv"):
        assert config.RUBRIC_VERSION_COL in pd.read_csv(f).columns


def test_validate_rubric_version_accepts_live_version(tmp_path):
    df = pd.DataFrame({config.RUBRIC_VERSION_COL: [config.RUBRIC_VERSION] * 3})
    assert trn._validate_rubric_version(df, tmp_path / "t.csv") == config.RUBRIC_VERSION


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"x": [1, 2]}),                                  # pre-stamping CSV
    pd.DataFrame({config.RUBRIC_VERSION_COL: ["3.1", "3.1"]}),    # stale
    pd.DataFrame({config.RUBRIC_VERSION_COL: [None, None]}),      # blank
])
def test_validate_rubric_version_rejects_unversioned_or_stale(tmp_path, frame):
    with pytest.raises(ValueError, match="rubric version"):
        trn._validate_rubric_version(frame, tmp_path / "t.csv")


def test_validate_rubric_version_override_returns_found_version(tmp_path):
    df = pd.DataFrame({config.RUBRIC_VERSION_COL: ["3.1"]})
    assert trn._validate_rubric_version(df, tmp_path / "t.csv", allow_mismatch=True) == "3.1"


def test_validate_rubric_version_rejects_mixed_even_with_override(tmp_path):
    df = pd.DataFrame({config.RUBRIC_VERSION_COL: ["3.1", config.RUBRIC_VERSION]})
    with pytest.raises(ValueError, match="mixes"):
        trn._validate_rubric_version(df, tmp_path / "t.csv", allow_mismatch=True)


def test_inference_rejects_stale_rubric_model(mode_dirs, caplog):
    _write_tiny_models(mode_dirs / "weekly", rubric_version="3.1")
    with caplog.at_level(logging.WARNING, logger=ml_ranker.logger.name):
        out = ml_ranker.predict_returns(_scan_rows(), "weekly")
    assert out.isna().all()
    assert "rubric_version" in caplog.text


def test_inference_rejects_reordered_feature_columns(mode_dirs):
    _write_tiny_models(mode_dirs / "weekly", feature_names=list(reversed(FEATURE_COLS)))
    assert ml_ranker.predict_returns(_scan_rows(), "weekly").isna().all()


def test_inference_refuses_model_without_metadata(mode_dirs, caplog):
    art = _write_tiny_models(mode_dirs / "weekly")
    (art / trn.MODEL_METADATA_FILENAME).unlink()
    with caplog.at_level(logging.WARNING, logger=ml_ranker.logger.name):
        out = ml_ranker.predict_returns(_scan_rows(), "weekly")
    assert out.isna().all()
    assert "unverified" in caplog.text


def test_inference_refuses_unreadable_metadata(mode_dirs):
    art = _write_tiny_models(mode_dirs / "weekly")
    (art / trn.MODEL_METADATA_FILENAME).write_text("{not json", encoding="utf-8")
    assert ml_ranker.predict_returns(_scan_rows(), "weekly").isna().all()


def test_tampered_or_swapped_artifact_is_skipped_by_hash(mode_dirs):
    art = _write_tiny_models(mode_dirs / "weekly", seed=0)
    X = _scan_rows()
    both = ml_ranker.predict_returns(X, "weekly")
    assert both.notna().all()

    # Swap in an LGB binary from a different training run -> hash mismatch.
    other = _write_tiny_models(mode_dirs / "other", seed=7)
    (other / trn.LGB_MODEL_FILENAME).replace(art / trn.LGB_MODEL_FILENAME)

    xgb_only = ml_ranker.predict_returns(X, "weekly")
    assert xgb_only.notna().all()
    assert not np.allclose(both, xgb_only)          # LGB no longer contributes

    # Corrupt XGB too -> nothing verifiable is left.
    with open(art / trn.XGB_MODEL_FILENAME, "ab") as fh:
        fh.write(b" ")
    assert ml_ranker.predict_returns(X, "weekly").isna().all()


def test_metadata_records_rubric_mode_and_hashes(mode_dirs):
    art = _write_tiny_models(mode_dirs / "weekly")
    meta = json.loads((art / trn.MODEL_METADATA_FILENAME).read_text())
    assert meta["rubric_version"] == config.RUBRIC_VERSION
    assert meta["mode"] == "weekly"
    assert meta["feature_columns"] == FEATURE_COLS
    assert meta["artifacts"]["xgb_sha256"] == trn._sha256(art / trn.XGB_MODEL_FILENAME)
    assert meta["artifacts"]["lgb_sha256"] == trn._sha256(art / trn.LGB_MODEL_FILENAME)


# ---------------------------------------------------------------------------
# Patch 4: newest-CSV auto-selection (no hard-coded filename)
# ---------------------------------------------------------------------------

def _stub_roots(monkeypatch, *roots):
    monkeypatch.setattr(trn, "_candidate_roots", lambda mode=trn.TRAIN_MODE: list(roots))


def test_no_hardcoded_source_filename():
    assert not hasattr(trn, "SOURCE_FILENAME")


def test_auto_selects_newest_by_filename_stamp_not_mtime(tmp_path, monkeypatch):
    old = tmp_path / "coiled_cobra_backtest_trades_2026-07-17.csv"
    new = tmp_path / "coiled_cobra_backtest_trades_2026-09-19.csv"
    old.write_text("x"); new.write_text("x")
    os.utime(new, (1, 1))                       # older mtime, newer stamp
    os.utime(old, (2_000_000_000, 2_000_000_000))
    _stub_roots(monkeypatch, tmp_path)
    assert trn._resolve_source_csv() == new


def test_first_root_with_matches_wins(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "coiled_cobra_backtest_trades_2026-08-01.csv").write_text("x")
    (b / "coiled_cobra_backtest_trades_2026-09-01.csv").write_text("x")
    _stub_roots(monkeypatch, a, b)
    assert trn._resolve_source_csv().parent == a


def test_missing_root_is_skipped_and_none_found_gives_hint(tmp_path, monkeypatch):
    _stub_roots(monkeypatch, tmp_path / "nope", tmp_path)
    with pytest.raises(FileNotFoundError, match="coiled_cobra_backtest"):
        trn._resolve_source_csv()


def test_explicit_missing_csv_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    (tmp_path / "coiled_cobra_backtest_trades_2026-09-01.csv").write_text("x")
    _stub_roots(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="--csv"):
        trn._resolve_source_csv(str(tmp_path / "typo.csv"))


# ---------------------------------------------------------------------------
# End to end: CSV -> train (main) -> artifacts -> inference
# ---------------------------------------------------------------------------

def test_train_then_infer_roundtrip(mode_dirs):
    pytest.importorskip("xgboost"); pytest.importorskip("lightgbm")
    pytest.importorskip("matplotlib")
    csv = _trades_csv(mode_dirs / "weekly" / "coiled_cobra_backtest_trades_2026-09-19.csv")

    assert trn.main([]) == 0            # auto-selects the CSV, writes to weekly dir

    meta = json.loads((mode_dirs / "weekly" / trn.MODEL_METADATA_FILENAME).read_text())
    assert meta["source_csv"] == csv.name
    assert meta["rubric_version"] == config.RUBRIC_VERSION
    assert meta["split"]["embargo_weeks"] == trn.EMBARGO_WEEKS
    assert meta["split"]["rows"]["purged"] > 0

    X = _scan_rows()
    assert ml_ranker.predict_returns(X, "weekly").notna().all()
    assert ml_ranker.predict_returns(X, "daily").isna().all()


def test_train_refuses_unversioned_csv(mode_dirs):
    _trades_csv(mode_dirs / "weekly" / "coiled_cobra_backtest_trades_2026-07-17.csv",
                rubric_version=None)
    with pytest.raises(ValueError, match="unversioned"):
        trn.main([])
    assert not (mode_dirs / "weekly" / trn.MODEL_METADATA_FILENAME).exists()


def test_train_refuses_non_weekly_mode(mode_dirs):
    with pytest.raises(ValueError, match="weekly"):
        trn.main(["--mode", "daily"])


def test_failed_retrain_does_not_leave_old_metadata_with_new_binaries(mode_dirs, monkeypatch):
    pytest.importorskip("xgboost"); pytest.importorskip("lightgbm")
    pytest.importorskip("matplotlib")
    art = _write_tiny_models(mode_dirs / "weekly")
    _trades_csv(art / "coiled_cobra_backtest_trades_2026-09-19.csv")

    def boom(*a, **k):
        raise RuntimeError("plot failed")

    monkeypatch.setattr(trn, "_save_importance_plot", boom)
    with pytest.raises(RuntimeError):
        trn.main([])
    # New binaries were written but stale metadata was removed -> inference refuses.
    assert not (art / trn.MODEL_METADATA_FILENAME).exists()
    assert ml_ranker.predict_returns(_scan_rows(), "weekly").isna().all()
