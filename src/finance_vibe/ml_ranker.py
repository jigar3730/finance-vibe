"""Offline-model inference for ranking Coiled Cobra scan results.

Loads per-horizon XGBClassifier boosters produced by
``coiled_cobra_ml_training.py`` and attaches win probabilities
(``ML_Prob_Win_*``). ``ML_Pred_Return`` / ``ML_Rank`` are filled only from a
**promoted** horizon (walk-forward beat Score on avg return AND win rate).
Horizons are never blended; LightGBM is not averaged in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import (
        FEATURE_COLS,
        HORIZON_SPECS,
        MIN_BEST_ITERATION,
        MODEL_METADATA_FILENAME,
        booster_is_degenerate,
        label_col,
    )
except ImportError:  # pragma: no cover - local direct execution
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import (
        FEATURE_COLS,
        HORIZON_SPECS,
        MIN_BEST_ITERATION,
        MODEL_METADATA_FILENAME,
        booster_is_degenerate,
        label_col,
    )

XGB_MODEL_FILENAME = "coiled_cobra_xgb_model.json"
LGB_MODEL_FILENAME = "coiled_cobra_lgb_model.txt"

ML_PRED_COL = "ML_Pred_Return"
ML_RANK_COL = "ML_Rank"

# Prefer the swing-length horizon when more than one model is promoted.
PROMOTE_PRIORITY = ("21d", "10d", "42d")


def _search_dirs(mode: str | None) -> list[Path]:
    mode = mode or config.DEFAULT_MODE
    dirs: list[Path] = []
    try:
        dirs.append(Path(config.get_log_dir(mode)))
    except Exception:  # pragma: no cover
        pass
    project_root = Path(config.PROJECT_ROOT)
    dirs.extend([
        project_root / "data" / "logs" / mode,
        Path("/app/data/logs") / mode,
        Path.cwd() / "data" / "logs" / mode,
    ])
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        key = d.resolve() if d.exists() else d
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def resolve_model_paths(mode: str | None = None) -> dict[str, Path | None]:
    """Locate saved XGB artifacts for ``mode`` (legacy filenames + 21d default)."""
    found: dict[str, Path | None] = {"xgb": None, "lgb": None}
    for d in _search_dirs(mode):
        if found["xgb"] is None:
            for name in (
                "coiled_cobra_xgb_21d.json",
                XGB_MODEL_FILENAME,
                "coiled_cobra_xgb_10d.json",
                "coiled_cobra_xgb_42d.json",
            ):
                cand = d / name
                if cand.is_file():
                    found["xgb"] = cand
                    break
        if found["lgb"] is None:
            cand = d / LGB_MODEL_FILENAME
            if cand.is_file():
                found["lgb"] = cand
        if found["xgb"]:
            break
    return found


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_horizon_artifact(mode: str | None, spec: dict) -> tuple[Path | None, dict | None]:
    """Return (model_path, metadata) for one horizon; metadata supplies feature schema."""
    for d in _search_dirs(mode):
        meta_cand = d / spec["metadata_filename"]
        model_cand = d / spec["model_filename"]
        if not model_cand.is_file():
            continue
        if not meta_cand.is_file():
            raise ValueError(
                f"Model {model_cand.name} has no matching metadata "
                f"{meta_cand.name}"
            )
        meta = _load_json(meta_cand)
        if not meta:
            raise ValueError(f"Unreadable model metadata: {meta_cand}")
        validate_artifact_metadata(meta, spec)
        return model_cand, meta
    return None, None


def load_horizon_index(mode: str | None = None) -> dict | None:
    for d in _search_dirs(mode):
        cand = d / MODEL_METADATA_FILENAME
        if cand.is_file():
            loaded = _load_json(cand)
            if loaded:
                return loaded
    return None


def feature_columns_from_metadata(meta: dict | None) -> list[str]:
    cols = (meta or {}).get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError("Model metadata has no feature_columns schema")
    return [str(c) for c in cols]


def validate_artifact_metadata(meta: dict, spec: dict) -> None:
    """Reject wrong task/model/horizon/target/schema before loading weights."""
    expected = {
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": spec["key"],
        "target_column": label_col(spec),
        "prob_column": spec["prob_col"],
    }
    mismatches = [
        f"{key}={meta.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if meta.get(key) != value
    ]
    features = meta.get("feature_columns")
    if not isinstance(features, list) or not features:
        mismatches.append("feature_columns missing/empty")
    elif meta.get("feature_count") != len(features):
        mismatches.append(
            f"feature_count={meta.get('feature_count')!r} "
            f"(expected {len(features)})"
        )
    if meta.get("production_model") not in {"xgb", "lgb", "ensemble", "none"}:
        mismatches.append(
            f"production_model={meta.get('production_model')!r} is invalid"
        )
    if _iteration_range(meta) is None:
        mismatches.append("best_iteration missing/invalid")
    elif booster_is_degenerate(meta.get("best_iteration")):
        mismatches.append(
            f"best_iteration={meta.get('best_iteration')!r} is degenerate "
            f"(need >= {MIN_BEST_ITERATION})"
        )
    if mismatches:
        raise ValueError(
            f"{spec['key']} artifact metadata mismatch: " + "; ".join(mismatches)
        )


def build_feature_frame(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Return a numeric frame with the training feature schema in order.

    Existing feature columns are coerced to numeric. Missing ``Pct_From_EMA*``
    / ``ATR_Pct`` columns are derived from Close/EMA/ATR when those are present.
    Pillars have no fallback — they must be written by the scanner/backtest.
    """
    cols = list(feature_cols or FEATURE_COLS)
    out = pd.DataFrame(index=df.index)

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

    close = _num("Close")

    for col in cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
            continue

        if col == "Pct_From_EMA20":
            ema = _num("EMA20")
            out[col] = (close - ema) / ema.replace(0, np.nan)
        elif col == "Pct_From_EMA50":
            ema = _num("EMA50")
            out[col] = (close - ema) / ema.replace(0, np.nan)
        elif col == "ATR_Pct":
            atr = _num("ATR")
            out[col] = atr / close.replace(0, np.nan)
        else:
            out[col] = np.nan

    return out[cols]


def _load_xgb(path: Path):
    try:
        import xgboost as xgb
    except ImportError:  # pragma: no cover
        return None
    try:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    except Exception:  # pragma: no cover - corrupt/incompatible artifact
        return None


def _load_lgb(path: Path):
    try:
        import lightgbm as lgb
    except ImportError:  # pragma: no cover
        return None
    try:
        return lgb.Booster(model_file=str(path))
    except Exception:  # pragma: no cover
        return None


def _booster_feature_names(booster, *, xgb: bool) -> list[str] | None:
    try:
        names = booster.feature_names if xgb else list(booster.feature_name())
    except Exception:
        return None
    if not names:
        return None
    return [str(n) for n in names]


def _validate_xgb_booster(booster, feature_cols: list[str]) -> None:
    names = _booster_feature_names(booster, xgb=True)
    if names != feature_cols:
        raise ValueError(
            f"XGBoost feature schema mismatch: model={names}, metadata={feature_cols}"
        )
    try:
        cfg = json.loads(booster.save_config())
        objective = cfg["learner"]["objective"]["name"]
    except Exception as exc:
        raise ValueError("Cannot inspect XGBoost model objective") from exc
    if objective != "binary:logistic":
        raise ValueError(
            f"XGBoost model type mismatch: objective={objective!r}, "
            "expected 'binary:logistic'"
        )


def _iteration_range(meta: dict | None) -> tuple[int, int] | None:
    if not meta:
        return None
    best = meta.get("best_iteration")
    if best is None:
        return None
    try:
        best_i = int(best)
    except (TypeError, ValueError):
        return None
    if best_i < 0:
        return None
    return (0, best_i + 1)


def predict_horizon_proba(
    df: pd.DataFrame,
    spec: dict,
    mode: str | None = None,
) -> pd.Series:
    """XGBClassifier positive-class probability for one horizon."""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    if df.empty:
        return result
    path, meta = load_horizon_artifact(mode, spec)
    if path is None:
        return result
    feature_cols = feature_columns_from_metadata(meta)
    feats = build_feature_frame(df, feature_cols)
    # XGBoost handles NaN splits; only skip rows where every feature is missing.
    valid = feats.notna().any(axis=1)
    if not valid.any():
        return result

    booster = _load_xgb(path)
    if booster is None:
        raise ValueError(f"Could not load XGBoost model: {path}")

    _validate_xgb_booster(booster, feature_cols)

    X = feats.loc[valid]
    try:
        import xgboost as xgb
        dmat = xgb.DMatrix(X, feature_names=list(X.columns))
        kwargs = {}
        irange = _iteration_range(meta)
        if irange is not None:
            kwargs["iteration_range"] = irange
        raw = booster.predict(dmat, **kwargs)
        result.loc[valid] = np.asarray(raw, dtype="float64")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"XGBoost inference failed for {spec['key']}: {exc}"
        ) from exc
    return result


def attach_horizon_probabilities(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Attach separately promoted ``ML_Prob_*`` columns; never blend horizons."""
    out = df.copy()
    for spec in HORIZON_SPECS:
        _, meta = load_horizon_artifact(mode, spec)
        if meta and meta.get("production_model") == "xgb":
            out[spec["prob_col"]] = predict_horizon_proba(
                out, spec, mode
            ).round(4)
        else:
            out[spec["prob_col"]] = np.nan
    return out


def _promoted_spec(mode: str | None) -> dict | None:
    by_key = {s["key"]: s for s in HORIZON_SPECS}
    index = load_horizon_index(mode)
    promoted_keys: list[str] = []
    if index and index.get("horizons"):
        for item in index["horizons"]:
            if (
                item.get("production_model") == "xgb"
                and item.get("key") in by_key
            ):
                promoted_keys.append(item["key"])
    else:
        for spec in HORIZON_SPECS:
            _, meta = load_horizon_artifact(mode, spec)
            if meta and meta.get("production_model") == "xgb":
                promoted_keys.append(spec["key"])
    for key in PROMOTE_PRIORITY:
        if key in promoted_keys:
            return by_key[key]
    return None


def predict_returns(df: pd.DataFrame, mode: str | None = None) -> pd.Series:
    """Live ranking score: probability from the single promoted horizon.

    Returns all-NaN when no horizon beat Score out of sample. Does not average
    XGBoost with LightGBM and does not blend 10d/21d/42d models.
    """
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    if df.empty:
        return result
    spec = _promoted_spec(mode)
    if spec is None:
        return result
    return predict_horizon_proba(df, spec, mode)


def attach_ml_ranks(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Attach horizon probabilities plus optional ``ML_Pred_Return`` / ``ML_Rank``.

    Rank 1 is the highest promoted-horizon probability. When no model is
    promoted, ML rank columns stay null and row order is preserved (Score sort).
    """
    out = df.copy()
    if out.empty:
        out[ML_PRED_COL] = pd.Series(dtype="float64")
        out[ML_RANK_COL] = pd.Series(dtype="float64")
        for spec in HORIZON_SPECS:
            out[spec["prob_col"]] = pd.Series(dtype="float64")
        return out

    out = attach_horizon_probabilities(out, mode)
    preds = predict_returns(out, mode)
    out[ML_PRED_COL] = preds.round(4)

    if preds.notna().any():
        ranks = preds.rank(method="dense", ascending=False)
        out[ML_RANK_COL] = ranks.astype("Int64")
        out = out.sort_values(
            [ML_PRED_COL, "Score"] if "Score" in out.columns else [ML_PRED_COL],
            ascending=False,
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        out[ML_RANK_COL] = pd.Series([pd.NA] * len(out), dtype="Int64")

    return out
