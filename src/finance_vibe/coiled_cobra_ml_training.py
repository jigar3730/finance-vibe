"""Coiled Cobra ML baseline: LightGBM + XGBoost regressors for short-horizon returns.

Standalone training script. Looks for coiled_cobra_backtest_trades_*.csv,
isolates pre-signal features, applies a dynamic relative temporal split,
and trains MAE-objective XGBRegressor / LGBMRegressor baselines for the
short-horizon target ``Forward_Return_2w`` with ATR_Pct sample weights to
reduce the impact of heavy-tailed financial outliers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Column zones (strict isolation — no leakage from post-trade metrics)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
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

SOURCE_FILENAME = "coiled_cobra_backtest_trades_2026-07-17.csv"

# Regularization: shallow trees, slow learning, row/feature bagging.
MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.01,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _candidate_csv_paths(explicit: str | None = None) -> list[Path]:
    """Resolve likely locations for the backtest trades CSV."""
    if explicit:
        return [Path(explicit).expanduser().resolve()]

    here = Path(__file__).resolve().parent
    project_root = here.parents[1]  # src/finance_vibe -> repo root
    cwd = Path.cwd()
    names = [SOURCE_FILENAME]
    
    search_roots = [
        cwd,
        cwd / "data" / "logs" / "weekly",
        project_root / "data" / "logs" / "weekly",
        Path("/app/data/logs/weekly"),
        Path("/mnt/fast/finance-vibe-data/logs/weekly"),
    ]
    paths: list[Path] = []
    for root in search_roots:
        for name in names:
            paths.append(root / name)
    return paths


def resolve_source_csv(explicit: str | None = None) -> Path:
    """Locate the trades CSV; raise with a clear message if missing."""
    for path in _candidate_csv_paths(explicit):
        if path.is_file():
            return path

    project_root = Path(__file__).resolve().parent.parents[1]
    globs: list[Path] = []
    for root in {
        Path.cwd() / "data" / "logs" / "weekly",
        project_root / "data" / "logs" / "weekly",
        Path("/app/data/logs/weekly"),
        Path("/mnt/fast/finance-vibe-data/logs/weekly"),
    }:
        if root.is_dir():
            globs.extend(sorted(root.glob("coiled_cobra_backtest_trades_*.csv")))
    if globs:
        return max(globs, key=lambda p: p.stat().st_mtime)

    tried = "\n  ".join(str(p) for p in _candidate_csv_paths(explicit))
    raise FileNotFoundError(
        f"Could not find {SOURCE_FILENAME}. Tried:\n  {tried}"
    )


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
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


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Applies a dynamic rolling temporal split backward from the max available date."""
    max_date = df[DATE_COL].max()
    
    # Define relative sliding windows (6 Months Test, 6 Months Val, Rest is Train)
    test_start = max_date - pd.Timedelta(weeks=26)
    val_start = test_start - pd.Timedelta(weeks=26)
    
    train = df[df[DATE_COL] < val_start].copy()
    val = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < test_start)].copy()
    test = df[df[DATE_COL] >= test_start].copy()
    
    bounds = {
        "max_date": max_date,
        "val_start": val_start,
        "test_start": test_start
    }
    return train, val, test, bounds


def build_matrices(
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X: pd.DataFrame, y: np.ndarray, label: str) -> dict:
    pred = model.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse(y, pred),
    }
    print(f"  {label}: MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}")
    return metrics


def print_ascii_importance(names: list[str], importances: np.ndarray, title: str) -> None:
    order = np.argsort(importances)[::-1]
    max_imp = float(importances.max()) if len(importances) else 1.0
    max_imp = max_imp if max_imp > 0 else 1.0
    print(f"\n{title}")
    print("-" * 56)
    for idx in order:
        bar_len = int(30 * float(importances[idx]) / max_imp)
        bar = "#" * bar_len
        print(f"  {names[idx]:<18} {importances[idx]:8.4f}  {bar}")


def save_importance_plot(
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

def save_model_metadata(
    art_dir: Path,
    feature_names: list[str],
    xgb_val_metrics: dict,
    xgb_test_metrics: dict,
    lgb_val_metrics: dict,
    lgb_test_metrics: dict,
    xgb_model_path: Path,
    lgb_model_path: Path,
    plot_path: Path,
) -> None:
    """Persist a JSON summary that downstream tooling can consume."""
    metadata = {
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
            "lgb_model": lgb_model_path.name,
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


def train_and_report(parts: dict, art_dir: Path, labels: dict) -> None:
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
    xgb_val_metrics = evaluate(xgb, X_val, y_val, f"Val ({labels['val']})")
    xgb_test_metrics = evaluate(xgb, X_test, y_test, f"Test OOS ({labels['test']})")

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
    lgb_val_metrics = evaluate(lgb, X_val, y_val, f"Val ({labels['val']})")
    lgb_test_metrics = evaluate(lgb, X_test, y_test, f"Test OOS ({labels['test']})")

    # --- NEW: Serialize Model Weights for Review and Pega Ingestion ---
    art_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save XGBoost Weights (Standard JSON format, highly readable/parseable)
    xgb_model_path = art_dir / "coiled_cobra_xgb_model.json"
    xgb.get_booster().save_model(str(xgb_model_path))
    print(f"\n[SAVED] XGBoost model weights exported to: {xgb_model_path}")

    # 2. Save LightGBM Weights (Standard text model structure)
    lgb_model_path = art_dir / "coiled_cobra_lgb_model.txt"
    lgb.booster_.save_model(str(lgb_model_path))
    print(f"[SAVED] LightGBM model weights exported to: {lgb_model_path}")
    # ------------------------------------------------------------------

    feature_names = list(X_train.columns)
    xgb_imp = np.asarray(xgb.feature_importances_, dtype=float)
    lgb_imp = np.asarray(lgb.feature_importances_, dtype=float)

    print_ascii_importance(feature_names, xgb_imp, "XGBoost feature importance")
    print_ascii_importance(feature_names, lgb_imp, "LightGBM feature importance")

    plot_path = art_dir / "coiled_cobra_ml_feature_importance.png"
    save_importance_plot(feature_names, xgb_imp, lgb_imp, plot_path)
    save_model_metadata(
        art_dir,
        feature_names,
        xgb_val_metrics,
        xgb_test_metrics,
        lgb_val_metrics,
        lgb_test_metrics,
        xgb_model_path,
        lgb_model_path,
        plot_path,
    )
    
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coiled Cobra ML baseline (XGBoost + LightGBM) with Dynamic Windows"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"Path to trades CSV (default: search for {SOURCE_FILENAME})",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for plots (default: data/logs/weekly next to CSV or cwd)",
    )
    args = parser.parse_args(argv)

    csv_path = resolve_source_csv(args.csv)
    if args.artifacts_dir:
        art_dir = Path(args.artifacts_dir)
    else:
        art_dir = csv_path.parent if csv_path.parent.is_dir() else Path.cwd()

    df = load_and_prepare(csv_path)
    train, val, test, bounds = temporal_split(df)

    v_str = f"{bounds['val_start'].strftime('%Y-%m-%d')} .. {bounds['test_start'].strftime('%Y-%m-%d')}"
    t_str = f"{bounds['test_start'].strftime('%Y-%m-%d')} .. {bounds['max_date'].strftime('%Y-%m-%d')}"

    print("\n=== Temporal Split Bounds (Dynamic Rolling Windows) ===")
    print(f"  Train:  Signal Date < {bounds['val_start'].strftime('%Y-%m-%d')} -> {len(train)} rows")
    print(f"  Val:    {v_str} -> {len(val)} rows")
    print(f"  Test:   {t_str} -> {len(test)} rows")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Empty partition(s): train={len(train)} val={len(val)} test={len(test)}"
        )

    parts = build_matrices(train, val, test)
    labels = {"val": v_str, "test": t_str}
    
    train_and_report(parts, art_dir, labels)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)