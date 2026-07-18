"""Offline-model inference for ranking Coiled Cobra scan results.

Loads the XGBoost / LightGBM boosters produced by
``coiled_cobra_ml_training.py`` and attaches a predicted short-horizon forward
return (``ML_Pred_Return``) plus a dense ``ML_Rank`` (1 = best) to a setup
frame. This is a soft ranking signal only: models never gate or size trades,
and missing artifacts / features fail soft (columns left null) so the live
pipeline keeps running on the rubric ``Score`` sort.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import FEATURE_COLS
except ImportError:  # pragma: no cover - local direct execution
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe.coiled_cobra_ml_training import FEATURE_COLS

XGB_MODEL_FILENAME = "coiled_cobra_xgb_model.json"
LGB_MODEL_FILENAME = "coiled_cobra_lgb_model.txt"

ML_PRED_COL = "ML_Pred_Return"
ML_RANK_COL = "ML_Rank"


def resolve_model_paths(mode: str = "weekly") -> dict[str, Path | None]:
    """Locate saved XGB/LGB artifacts for ``mode``.

    Searches the mode log silo first, then common container/data roots. Returns
    a dict with ``xgb`` / ``lgb`` keys set to the first existing path or None.
    """
    search_dirs: list[Path] = []
    try:
        search_dirs.append(Path(config.get_log_dir(mode)))
    except Exception:  # pragma: no cover - defensive; log dir resolution is cheap
        pass

    project_root = Path(config.PROJECT_ROOT)
    search_dirs.extend([
        project_root / "data" / "logs" / mode,
        project_root / "data" / "logs" / "weekly",
        Path("/app/data/logs") / mode,
        Path("/app/data/logs/weekly"),
        Path.cwd() / "data" / "logs" / mode,
    ])

    found: dict[str, Path | None] = {"xgb": None, "lgb": None}
    for d in search_dirs:
        if found["xgb"] is None:
            cand = d / XGB_MODEL_FILENAME
            if cand.is_file():
                found["xgb"] = cand
        if found["lgb"] is None:
            cand = d / LGB_MODEL_FILENAME
            if cand.is_file():
                found["lgb"] = cand
        if found["xgb"] and found["lgb"]:
            break
    return found


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
    except Exception:  # pragma: no cover - corrupt/incompatible artifact
        return None


def predict_returns(df: pd.DataFrame, mode: str = "weekly") -> pd.Series:
    """Predict forward return per row, averaging available XGB/LGB models.

    Returns a float Series aligned to ``df.index``; all-NaN when no model loads.
    Rows with any NaN feature are left NaN.
    """
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    if df.empty:
        return result

    paths = resolve_model_paths(mode)
    feats = build_feature_frame(df)
    valid = feats.notna().all(axis=1)
    if not valid.any():
        return result

    X = feats.loc[valid]
    preds: list[np.ndarray] = []

    if paths["xgb"] is not None:
        booster = _load_xgb(paths["xgb"])
        if booster is not None:
            try:
                import xgboost as xgb
                preds.append(booster.predict(xgb.DMatrix(X, feature_names=list(X.columns))))
            except Exception:  # pragma: no cover
                pass

    if paths["lgb"] is not None:
        booster = _load_lgb(paths["lgb"])
        if booster is not None:
            try:
                preds.append(np.asarray(booster.predict(X)))
            except Exception:  # pragma: no cover
                pass

    if not preds:
        return result

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
