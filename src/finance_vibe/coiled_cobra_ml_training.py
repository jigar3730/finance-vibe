"""Coiled Cobra ML baseline: LightGBM + XGBoost regressors for short-horizon returns.

Standalone training script. Auto-selects the newest
coiled_cobra_backtest_trades_*.csv, refuses data stamped with a different
rubric version, isolates pre-signal features, applies a dynamic relative
temporal split with a purge/embargo equal to the forward-return horizon, and
trains MAE-objective XGBRegressor / LGBMRegressor baselines for the
short-horizon target ``Forward_Return_2w`` with ATR_Pct sample weights to
reduce the impact of heavy-tailed financial outliers.

Model artifacts and a metadata file (rubric version, feature list, mode,
artifact hashes) are written to the mode's log directory; ``ml_ranker``
validates that metadata before serving predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    from finance_vibe import config
except ImportError:  # pragma: no cover - local direct execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config

# ---------------------------------------------------------------------------
# Column zones (strict isolation — no leakage from post-trade metrics)
# ---------------------------------------------------------------------------
# Pre-signal features shared with ``ml_ranker`` inference. Do not add
# post-trade / leakage columns here.
FEATURE_COLS: list[str] = [
    "Score",
    "Pct_From_EMA20",
    "Pct_From_EMA50",
    "Pct_From_Fib618",
    "Pct_From_Fib786",
    "ATR_Pct",
]
# Short tactical horizon (~2 weekly bars) — useful for ranking setups without
# relying on long-horizon return assumptions.
TARGET_COL = "Forward_Return_2w"
TARGET_HORIZON_WEEKS = 2
DATE_COL = "Signal Date"
WEIGHT_COL = "ATR_Pct"
MODEL_METADATA_FILENAME = "coiled_cobra_ml_model_metadata.json"
XGB_MODEL_FILENAME = "coiled_cobra_xgb_model.json"
LGB_MODEL_FILENAME = "coiled_cobra_lgb_model.txt"
# Forward_Return_2w is measured in *bars*; only weekly bars make it a 2-week
# label, so training is weekly-only until the label is made mode-aware.
TRAIN_MODE = "weekly"
# Purge between partitions: a row at date t has a label realised through
# t + TARGET_HORIZON_WEEKS, so rows that close to a boundary would overlap the
# next partition's period.
EMBARGO_WEEKS = TARGET_HORIZON_WEEKS

LEAKAGE_COLS = [
    "Stock Entry",
    "Stock Stop",
    "Target 1",
    "Target 2",
    "Outcome",
    "Exit Date",
    "Exit Price",
    "R Multiple",
    "Target_Label",
    "Target_R_Mult",
]

SOURCE_GLOB = "coiled_cobra_backtest_trades_*.csv"

# Regularization: shallow trees, slow learning, row/feature bagging.
MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.01,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _candidate_roots(mode: str = TRAIN_MODE) -> list[Path]:
    """Directories searched for backtest trades CSVs, most authoritative first.

    The mode's log directory wins; the legacy locations are only consulted when
    it holds no trades CSV at all (e.g. host-side runs against the data volume).
    """
    project_root = Path(__file__).resolve().parents[2]  # src/finance_vibe -> repo root
    cwd = Path.cwd()
    roots = [
        Path(config.get_log_dir(mode)),
        cwd,
        cwd / "data" / "logs" / mode,
        project_root / "data" / "logs" / mode,
        Path("/app/data/logs") / mode,
        Path("/mnt/fast/finance-vibe-data/logs") / mode,
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _csv_recency_key(path: Path) -> tuple[str, float]:
    """Sort key: date stamp embedded in the filename, then mtime as tie-break."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return (m.group(1) if m else "", path.stat().st_mtime)


def _resolve_source_csv(explicit: str | None = None, mode: str = TRAIN_MODE) -> Path:
    """Locate the trades CSV: ``explicit`` if given, else the newest by stamp.

    An explicit path that does not exist is an error (never a silent fall-back
    to some other file). Auto-selection takes the first root holding any
    ``coiled_cobra_backtest_trades_*.csv`` and returns its newest match; the
    caller then validates that file's rubric version, so a stale newest file
    fails loudly instead of quietly yielding to an older one.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--csv path does not exist: {path}")
        return path

    roots = _candidate_roots(mode)
    for root in roots:
        if not root.is_dir():
            continue
        matches = [p for p in root.glob(SOURCE_GLOB) if p.is_file()]
        if matches:
            return max(matches, key=_csv_recency_key)

    tried = "\n  ".join(str(r) for r in roots)
    raise FileNotFoundError(
        f"No {SOURCE_GLOB} found. Searched:\n  {tried}\n"
        f"Generate one with: python -m finance_vibe.coiled_cobra_backtest {mode} --backtest"
    )


def _validate_rubric_version(
    df: pd.DataFrame, csv_path: Path, allow_mismatch: bool = False
) -> str:
    """Return the CSV's rubric version; refuse unversioned/mixed/stale data.

    ``allow_mismatch`` permits *experiments* on other vintages, but the model
    metadata will record the CSV's version, so ``ml_ranker`` still refuses to
    serve such a model against the live rubric.
    """
    col = config.RUBRIC_VERSION_COL
    if col in df.columns:
        versions = sorted(df[col].dropna().astype(str).unique())
    else:
        versions = []

    if len(versions) > 1:
        raise ValueError(f"{csv_path.name} mixes rubric versions {versions}; regenerate it.")

    found = versions[0] if versions else "unversioned"
    if found != config.RUBRIC_VERSION:
        msg = (
            f"{csv_path.name} has rubric version '{found}' but the live rubric is "
            f"'{config.RUBRIC_VERSION}'. Score/feature semantics and the qualifying "
            f"population differ; regenerate with: "
            f"python -m finance_vibe.coiled_cobra_backtest {TRAIN_MODE} --backtest"
        )
        if not allow_mismatch:
            raise ValueError(msg)
        print(f"WARNING (--allow-rubric-mismatch): {msg}")
    return found


def _load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load CSV, drop leakage cols, keep no_fill rows, drop NaN targets."""
    df = pd.read_csv(csv_path)
    print(f"Loaded source: {csv_path}")
    print(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} cols")

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing date column: {DATE_COL}")

    drop_present = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop_present)
    print(f"Dropped leakage columns ({len(drop_present)}): {drop_present}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        n_bad = int(df[DATE_COL].isna().sum())
        raise ValueError(f"{DATE_COL} has {n_bad} unparseable value(s)")

    before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    print(
        f"Dropped {before - len(df)} row(s) with NaN/None {TARGET_COL} "
        f"(kept no_fill and all other outcomes)"
    )
    print(f"Training pool shape: {df.shape[0]} rows")
    return df.sort_values(DATE_COL).reset_index(drop=True)


def _temporal_split(
    df: pd.DataFrame, embargo_weeks: int = EMBARGO_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Rolling temporal split back from the max date, with a purge embargo.

    Layout (oldest -> newest)::

        train | embargo | val | embargo | test

    Rows dated within ``embargo_weeks`` before a boundary are dropped from the
    *earlier* partition: their forward-return label is realised inside the next
    partition's period, so keeping them would leak future information across
    the boundary (and share the cross-sectional market move).
    """
    if embargo_weeks < 0:
        raise ValueError("embargo_weeks must be >= 0")
    max_date = df[DATE_COL].max()
    embargo = pd.Timedelta(weeks=embargo_weeks)

    # Define relative sliding windows (6 Months Test, 6 Months Val, Rest is Train)
    test_start = max_date - pd.Timedelta(weeks=26)
    val_start = test_start - pd.Timedelta(weeks=26)
    train_end = val_start - embargo
    val_end = test_start - embargo

    train = df[df[DATE_COL] < train_end].copy()
    val = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < val_end)].copy()
    test = df[df[DATE_COL] >= test_start].copy()

    bounds = {
        "max_date": max_date,
        "val_start": val_start,
        "test_start": test_start,
        "train_end": train_end,
        "val_end": val_end,
        "embargo_weeks": embargo_weeks,
        "purged_rows": int(len(df) - len(train) - len(val) - len(test)),
    }
    return train, val, test, bounds


def _build_matrices(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Build X / y / sample_weight arrays for each temporal partition."""
    parts = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        X = frame[FEATURE_COLS].copy()
        y = frame[TARGET_COL].astype(float).to_numpy()
        w = frame[WEIGHT_COL].astype(float).to_numpy()
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        parts[name] = {"X": X, "y": y, "w": w, "n": len(frame)}
        
    med = np.nanmedian(parts["train"]["w"])
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    for name in parts:
        w = parts[name]["w"]
        parts[name]["w"] = np.where(np.isfinite(w) & (w > 0), w, med)
    return parts


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(model, X: pd.DataFrame, y: np.ndarray, label: str) -> dict:
    pred = model.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": _rmse(y, pred),
    }
    print(f"  {label}: MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}")
    return metrics


def _print_ascii_importance(names: list[str], importances: np.ndarray, title: str) -> None:
    order = np.argsort(importances)[::-1]
    max_imp = float(importances.max()) if len(importances) else 1.0
    max_imp = max_imp if max_imp > 0 else 1.0
    print(f"\n{title}")
    print("-" * 56)
    for idx in order:
        bar_len = int(30 * float(importances[idx]) / max_imp)
        bar = "#" * bar_len
        print(f"  {names[idx]:<18} {importances[idx]:8.4f}  {bar}")


def _save_importance_plot(
    feature_names: list[str],
    xgb_imp: np.ndarray,
    lgb_imp: np.ndarray,
    out_path: Path,
) -> None:
    """Side-by-side gain/split importance bar chart."""
    order = np.argsort(xgb_imp)[::-1]
    names = [feature_names[i] for i in order]
    xgb_sorted = xgb_imp[order]
    lgb_sorted = lgb_imp[order]

    y_pos = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Coiled Cobra ML — Feature Importances ({TARGET_COL})")

    axes[0].barh(y_pos, xgb_sorted, color="#2c5f7c")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Importance (gain)")
    axes[0].set_title("XGBoost")

    axes[1].barh(y_pos, lgb_sorted, color="#3d7a5a")
    axes[1].set_xlabel("Importance (split/gain)")
    axes[1].set_title("LightGBM")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved feature importance plot: {out_path}")

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    """Best-effort short git SHA of the training code (None outside a checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def _save_model_metadata(
    art_dir: Path,
    feature_names: list[str],
    xgb_val_metrics: dict,
    xgb_test_metrics: dict,
    lgb_val_metrics: dict,
    lgb_test_metrics: dict,
    xgb_model_path: Path,
    lgb_model_path: Path,
    plot_path: Path,
    context: dict,
) -> None:
    """Persist a JSON summary that downstream tooling can consume.

    ``ml_ranker`` refuses to serve a model unless this file exists and its
    ``rubric_version`` / ``mode`` / ``feature_columns`` match the live pipeline
    and the ``sha256`` of each model file matches the binary on disk.
    """
    metadata = {
        "rubric_version": context["rubric_version"],
        "mode": context["mode"],
        "trained_at": context["trained_at"],
        "git_sha": context["git_sha"],
        "source_csv": context["source_csv"],
        "split": context["split"],
        "target_column": TARGET_COL,
        "target_horizon_weeks": TARGET_HORIZON_WEEKS,
        "feature_columns": feature_names,
        "decision_guidance": {
            "use_as": "soft ranking signal for setup selection",
            "combine_with": [
                "macro regime score",
                "risk management rules",
                "market context",
                "liquidity and options constraints",
            ],
            "do_not_use_as": [
                "hard entry/exit gate",
                "position sizing rule",
                "standalone trading system",
            ],
        },
        "artifacts": {
            "xgb_model": xgb_model_path.name,
            "xgb_sha256": _sha256(xgb_model_path),
            "lgb_model": lgb_model_path.name,
            "lgb_sha256": _sha256(lgb_model_path),
            "importance_plot": plot_path.name,
        },
        "metrics": {
            "xgb": {
                "val_mae": xgb_val_metrics["mae"],
                "val_rmse": xgb_val_metrics["rmse"],
                "test_mae": xgb_test_metrics["mae"],
                "test_rmse": xgb_test_metrics["rmse"],
            },
            "lgb": {
                "val_mae": lgb_val_metrics["mae"],
                "val_rmse": lgb_val_metrics["rmse"],
                "test_mae": lgb_test_metrics["mae"],
                "test_rmse": lgb_test_metrics["rmse"],
            },
        },
    }
    metadata_path = art_dir / MODEL_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n[SAVED] ML metadata summary: {metadata_path}")


def _train_and_report(parts: dict, art_dir: Path, labels: dict, context: dict) -> None:
    X_train, y_train, w_train = parts["train"]["X"], parts["train"]["y"], parts["train"]["w"]
    X_val, y_val = parts["val"]["X"], parts["val"]["y"]
    X_test, y_test = parts["test"]["X"], parts["test"]["y"]

    print("\n=== Dataset Shape Integrity ===")
    print(f"  X_train: {X_train.shape[0]} rows x {X_train.shape[1]} cols")
    print(f"  X_val:   {X_val.shape[0]} rows x {X_val.shape[1]} cols")
    print(f"  X_test:  {X_test.shape[0]} rows x {X_test.shape[1]} cols")
    print(f"  Features: {list(X_train.columns)}")

    print("\n=== Training XGBRegressor (reg:absoluteerror) ===")
    xgb = XGBRegressor(
        max_depth=MODEL_PARAMS["max_depth"],
        learning_rate=MODEL_PARAMS["learning_rate"],
        n_estimators=MODEL_PARAMS["n_estimators"],
        subsample=MODEL_PARAMS["subsample"],
        colsample_bytree=MODEL_PARAMS["colsample_bytree"],
        objective="reg:absoluteerror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train, sample_weight=w_train)

    print("XGBoost validation / OOS scores:")
    xgb_val_metrics = _evaluate(xgb, X_val, y_val, f"Val ({labels['val']})")
    xgb_test_metrics = _evaluate(xgb, X_test, y_test, f"Test OOS ({labels['test']})")

    print("\n=== Training LGBMRegressor (regression_l1 / MAE) ===")
    lgb = LGBMRegressor(
        max_depth=MODEL_PARAMS["max_depth"],
        learning_rate=MODEL_PARAMS["learning_rate"],
        n_estimators=MODEL_PARAMS["n_estimators"],
        subsample=MODEL_PARAMS["subsample"],
        colsample_bytree=MODEL_PARAMS["colsample_bytree"],
        objective="regression_l1",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb.fit(X_train, y_train, sample_weight=w_train)

    print("LightGBM validation / OOS scores:")
    lgb_val_metrics = _evaluate(lgb, X_val, y_val, f"Val ({labels['val']})")
    lgb_test_metrics = _evaluate(lgb, X_test, y_test, f"Test OOS ({labels['test']})")

    # --- NEW: Serialize Model Weights for Review and Pega Ingestion ---
    art_dir.mkdir(parents=True, exist_ok=True)
    # Drop the previous metadata first: if this run dies between writing the
    # binaries and the new metadata, inference must see "no metadata" (refuse)
    # rather than old metadata paired with new binaries.
    (art_dir / MODEL_METADATA_FILENAME).unlink(missing_ok=True)

    # 1. Save XGBoost Weights (Standard JSON format, highly readable/parseable)
    xgb_model_path = art_dir / XGB_MODEL_FILENAME
    xgb.get_booster().save_model(str(xgb_model_path))
    print(f"\n[SAVED] XGBoost model weights exported to: {xgb_model_path}")

    # 2. Save LightGBM Weights (Standard text model structure)
    lgb_model_path = art_dir / LGB_MODEL_FILENAME
    lgb.booster_.save_model(str(lgb_model_path))
    print(f"[SAVED] LightGBM model weights exported to: {lgb_model_path}")
    # ------------------------------------------------------------------

    feature_names = list(X_train.columns)
    xgb_imp = np.asarray(xgb.feature_importances_, dtype=float)
    lgb_imp = np.asarray(lgb.feature_importances_, dtype=float)

    _print_ascii_importance(feature_names, xgb_imp, "XGBoost feature importance")
    _print_ascii_importance(feature_names, lgb_imp, "LightGBM feature importance")

    plot_path = art_dir / "coiled_cobra_ml_feature_importance.png"
    _save_importance_plot(feature_names, xgb_imp, lgb_imp, plot_path)
    _save_model_metadata(
        art_dir,
        feature_names,
        xgb_val_metrics,
        xgb_test_metrics,
        lgb_val_metrics,
        lgb_test_metrics,
        xgb_model_path,
        lgb_model_path,
        plot_path,
        context,
    )

def main(argv: list[str] | None = None) -> int:
    """Train XGB/LGB baselines and write model artifacts to the mode's log dir."""
    parser = argparse.ArgumentParser(
        description="Coiled Cobra ML baseline (XGBoost + LightGBM) with Dynamic Windows"
    )
    parser.add_argument(
        "--mode",
        default=TRAIN_MODE,
        help=f"Pipeline mode the model is bound to (only '{TRAIN_MODE}' is supported: "
        f"{TARGET_COL} is a bar-count label)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"Path to trades CSV (default: newest {SOURCE_GLOB} in the mode's log dir)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for model artifacts (default: the mode's log dir, where "
        "ml_ranker looks for them)",
    )
    parser.add_argument(
        "--allow-rubric-mismatch",
        action="store_true",
        help="Train on a CSV from a different rubric version (experiments only; the "
        "resulting model will be refused by ml_ranker)",
    )
    args = parser.parse_args(argv)

    if args.mode != TRAIN_MODE:
        raise ValueError(
            f"Training supports mode '{TRAIN_MODE}' only (got '{args.mode}'): "
            f"{TARGET_COL} counts bars, so it is not a 2-week label on other timeframes."
        )

    csv_path = _resolve_source_csv(args.csv, args.mode)
    art_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(config.get_log_dir(args.mode))

    df = _load_and_prepare(csv_path)
    rubric_version = _validate_rubric_version(df, csv_path, args.allow_rubric_mismatch)
    train, val, test, bounds = _temporal_split(df)

    fmt = lambda ts: ts.strftime("%Y-%m-%d")
    v_str = f"{fmt(bounds['val_start'])} .. {fmt(bounds['val_end'])}"
    t_str = f"{fmt(bounds['test_start'])} .. {fmt(bounds['max_date'])}"

    print(f"\n=== Temporal Split Bounds (Dynamic Rolling Windows, {bounds['embargo_weeks']}w embargo) ===")
    print(f"  Train:  Signal Date < {fmt(bounds['train_end'])} -> {len(train)} rows")
    print(f"  Val:    {v_str} -> {len(val)} rows")
    print(f"  Test:   {t_str} -> {len(test)} rows")
    print(f"  Purged: {bounds['purged_rows']} row(s) inside the embargo windows")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Empty partition(s): train={len(train)} val={len(val)} test={len(test)}"
        )

    parts = _build_matrices(train, val, test)
    labels = {"val": v_str, "test": t_str}
    context = {
        "rubric_version": rubric_version,
        "mode": args.mode,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "source_csv": csv_path.name,
        "split": {
            "embargo_weeks": bounds["embargo_weeks"],
            "train_end": fmt(bounds["train_end"]),
            "val_start": fmt(bounds["val_start"]),
            "val_end": fmt(bounds["val_end"]),
            "test_start": fmt(bounds["test_start"]),
            "max_date": fmt(bounds["max_date"]),
            "rows": {"train": len(train), "val": len(val), "test": len(test),
                     "purged": bounds["purged_rows"]},
        },
    }

    _train_and_report(parts, art_dir, labels, context)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)