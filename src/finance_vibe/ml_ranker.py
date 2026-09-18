"""Offline-model inference for ranking Coiled Cobra scan results.

Loads the XGBoost / LightGBM boosters produced by
``coiled_cobra_ml_training.py`` and attaches a predicted short-horizon forward
return (``ML_Pred_Return``) plus a dense ``ML_Rank`` (1 = best) to a setup
frame. This is a soft ranking signal only: models never gate or size trades,
and missing artifacts / features fail soft (columns left null) so the live
pipeline keeps running on the rubric ``Score`` sort.

Artifacts are strictly mode-bound and validated: models are only read from the
requested mode's own log directory (no cross-mode fallback), and only served
when the training metadata's rubric version, mode, feature list and artifact
hashes all match the live pipeline. Anything else fails soft with a logged
reason rather than serving an incompatible model.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import (
        FEATURE_COLS,
        LGB_MODEL_FILENAME,
        MODEL_METADATA_FILENAME,
        XGB_MODEL_FILENAME,
        _sha256,
    )
except ImportError:  # pragma: no cover - local direct execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import (
        FEATURE_COLS,
        LGB_MODEL_FILENAME,
        MODEL_METADATA_FILENAME,
        XGB_MODEL_FILENAME,
        _sha256,
    )

logger = logging.getLogger(__name__)

# Optional explicit model directory (e.g. a mounted volume). It replaces the
# mode log dir, but the metadata's ``mode`` check still applies.
MODEL_DIR_ENV = "FINANCE_VIBE_MODEL_DIR"

ML_PRED_COL: str = "ML_Pred_Return"
ML_RANK_COL: str = "ML_Rank"


def _model_dir(mode: str) -> Path:
    override = os.environ.get(MODEL_DIR_ENV)
    return Path(override) if override else Path(config.get_log_dir(mode))


def _resolve_model_paths(mode: str = "weekly") -> dict[str, Path | None]:
    """Locate the XGB/LGB artifacts and metadata for ``mode``.

    Strictly mode-bound: only the mode's own log directory (or the explicit
    ``FINANCE_VIBE_MODEL_DIR`` override) is consulted, so a ``weekly`` model can
    never be picked up by ``daily``/``high_beta``. All three files come from that
    one directory, so their vintages cannot be mixed. Missing files are None.
    """
    d = _model_dir(mode)
    found: dict[str, Path | None] = {}
    for key, name in (
        ("xgb", XGB_MODEL_FILENAME),
        ("lgb", LGB_MODEL_FILENAME),
        ("metadata", MODEL_METADATA_FILENAME),
    ):
        cand = d / name
        found[key] = cand if cand.is_file() else None
    return found


def _metadata_problem(meta: dict, mode: str) -> str | None:
    """Return why ``meta`` is incompatible with the live pipeline, or None."""
    _, profile = config.resolve_pipeline_mode(mode)
    if meta.get("rubric_version") != config.RUBRIC_VERSION:
        return (
            f"rubric_version {meta.get('rubric_version')!r} != live "
            f"{config.RUBRIC_VERSION!r}"
        )
    if meta.get("mode") != profile:
        return f"model bound to mode {meta.get('mode')!r}, requested {profile!r}"
    if list(meta.get("feature_columns") or []) != list(FEATURE_COLS):
        return "feature_columns differ from the live FEATURE_COLS (order matters)"
    return None


def _load_validated_metadata(paths: dict[str, Path | None], mode: str) -> dict | None:
    """Read + validate metadata; None (with a logged reason) when unusable."""
    if paths["metadata"] is None:
        if paths["xgb"] or paths["lgb"]:
            logger.warning(
                "ML model files found for mode %r but no %s; refusing to serve "
                "unverified models.", mode, MODEL_METADATA_FILENAME,
            )
        else:
            logger.info("No ML model artifacts for mode %r.", mode)
        return None
    try:
        meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Unreadable ML metadata %s (%s); skipping ML.", paths["metadata"], exc)
        return None
    problem = _metadata_problem(meta, mode)
    if problem:
        logger.warning("ML model rejected for mode %r: %s. Retrain to refresh.", mode, problem)
        return None
    return meta


def _artifact_verified(meta: dict, key: str, path: Path) -> bool:
    """True when the file's sha256 matches the hash recorded at training time."""
    expected = (meta.get("artifacts") or {}).get(f"{key}_sha256")
    if not expected or _sha256(path) != expected:
        logger.warning(
            "ML %s artifact %s does not match its recorded sha256; skipping it.",
            key, path.name,
        )
        return False
    return True


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric frame with exactly ``FEATURE_COLS`` in training order.

    Existing feature columns are coerced to numeric. Missing ``Pct_From_*`` /
    ``ATR_Pct`` columns are derived from raw Close/EMA/Fib/ATR fields when those
    are present, so both the scanner (raw fields) and pre-featurized frames work.
    """
    out = pd.DataFrame(index=df.index)

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

    close = _num("Close")

    for col in FEATURE_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
            continue

        if col == "Pct_From_EMA20":
            ema = _num("EMA20")
            out[col] = (close - ema) / ema.replace(0, np.nan)
        elif col == "Pct_From_EMA50":
            ema = _num("EMA50")
            out[col] = (close - ema) / ema.replace(0, np.nan)
        elif col == "Pct_From_Fib618":
            fib = _num("Fib 61.8%")
            out[col] = (close - fib) / fib.replace(0, np.nan)
        elif col == "Pct_From_Fib786":
            fib = _num("Fib 78.6%")
            out[col] = (close - fib) / fib.replace(0, np.nan)
        elif col == "ATR_Pct":
            atr = _num("ATR")
            out[col] = atr / close.replace(0, np.nan)
        else:  # Score or any unexpected feature with no fallback
            out[col] = np.nan

    return out[FEATURE_COLS]


def _load_xgb(path: Path) -> Any | None:
    try:
        import xgboost as xgb
    except ImportError:  # pragma: no cover
        return None
    try:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    except Exception as exc:  # pragma: no cover - corrupt/incompatible artifact
        logger.warning("Could not load XGB model %s (%s).", path, exc)
        return None


def _load_lgb(path: Path) -> Any | None:
    try:
        import lightgbm as lgb
    except ImportError:  # pragma: no cover
        return None
    try:
        return lgb.Booster(model_file=str(path))
    except Exception as exc:  # pragma: no cover - corrupt/incompatible artifact
        logger.warning("Could not load LGB model %s (%s).", path, exc)
        return None


def predict_returns(df: pd.DataFrame, mode: str = "weekly") -> pd.Series:
    """Predict forward return per row, averaging available XGB/LGB models.

    Returns a float Series aligned to ``df.index``; all-NaN when no model loads.
    Rows with any NaN feature are left NaN.
    """
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    if df.empty:
        return result

    paths = _resolve_model_paths(mode)
    meta = _load_validated_metadata(paths, mode)
    if meta is None:
        return result
    feats = build_feature_frame(df)
    valid = feats.notna().all(axis=1)
    if not valid.any():
        return result

    X = feats.loc[valid]
    preds: list[np.ndarray] = []

    if paths["xgb"] is not None and _artifact_verified(meta, "xgb", paths["xgb"]):
        booster = _load_xgb(paths["xgb"])
        if booster is not None:
            try:
                import xgboost as xgb
                preds.append(booster.predict(xgb.DMatrix(X, feature_names=list(X.columns))))
            except Exception as exc:  # pragma: no cover
                logger.warning("XGB prediction failed (%s); continuing without it.", exc)

    if paths["lgb"] is not None and _artifact_verified(meta, "lgb", paths["lgb"]):
        booster = _load_lgb(paths["lgb"])
        if booster is not None:
            try:
                preds.append(np.asarray(booster.predict(X)))
            except Exception as exc:  # pragma: no cover
                logger.warning("LGB prediction failed (%s); continuing without it.", exc)

    if not preds:
        return result
    if len(preds) < 2:
        logger.warning("ML ensemble degraded: only %d of 2 models usable.", len(preds))

    stacked = np.vstack(preds)
    result.loc[valid] = np.nanmean(stacked, axis=0)
    return result


def attach_ml_ranks(df: pd.DataFrame, mode: str = "weekly") -> pd.DataFrame:
    """Attach ``ML_Pred_Return`` + dense ``ML_Rank`` and sort best-first.

    Rank 1 is the highest predicted return. Rows without a prediction sort last
    and keep a null rank. When no model is available, the frame is returned with
    null ML columns and its existing order preserved (caller falls back to Score).
    """
    out = df.copy()
    if out.empty:
        out[ML_PRED_COL] = pd.Series(dtype="float64")
        out[ML_RANK_COL] = pd.Series(dtype="float64")
        return out

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
