"""Coiled Cobra ML baseline: LightGBM + XGBoost regressors for Forward_Return_13w.

Standalone training script. Looks for coiled_cobra_backtest_trades_*.csv,
isolates 6 pre-signal features (Grade dropped), applies a rigid temporal split,
and trains MAE-objective XGBRegressor / LGBMRegressor baselines with ATR_Pct
sample weights to dampen heavy-tailed financial outliers.
"""
from __future__ import annotations

import argparse
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
# Grade dropped: redundant binned form of Score (multi-collinearity noise).
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Score",
    "Pct_From_EMA20",
    "Pct_From_EMA50",
    "Pct_From_Fib618",
    "Pct_From_Fib786",
    "ATR_Pct",
]
TARGET_COL = "Forward_Return_13w"
DATE_COL = "Signal Date"
WEIGHT_COL = "ATR_Pct"

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

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-07-31")

MODEL_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 300,
}


def _candidate_csv_paths(explicit: str | None = None) -> list[Path]:
    """Resolve likely locations for the backtest trades CSV."""
    if explicit:
        return [Path(explicit).expanduser().resolve()]

    here = Path(__file__).resolve().parent
    project_root = here.parents[1]  # src/finance_vibe -> repo root
    cwd = Path.cwd()
    names = [SOURCE_FILENAME]
    # Prefer the dated source file; fall back to newest matching export.
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

    # Fallback: newest coiled_cobra_backtest_trades_*.csv under common roots
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

    # Drop catastrophic leakage columns when present (do not filter no_fill).
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


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rigid non-random date-boundary split on Signal Date."""
    train = df[df[DATE_COL] <= TRAIN_END].copy()
    val = df[(df[DATE_COL] >= VAL_START) & (df[DATE_COL] <= VAL_END)].copy()
    test = df[(df[DATE_COL] >= TEST_START) & (df[DATE_COL] <= TEST_END)].copy()
    return train, val, test


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
        # Spec: pass ATR_Pct as sample_weight. Clip tiny values for stability.
        w = frame[WEIGHT_COL].astype(float).to_numpy()
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        # Fill any non-positive / NaN weights with train median later for train;
        # for val/test weights are unused at fit time but kept for completeness.
        parts[name] = {"X": X, "y": y, "w": w, "n": len(frame)}
    # Stabilize train weights
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
    fig.suptitle("Coiled Cobra ML — Feature Importances (Forward_Return_13w)")

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


def train_and_report(parts: dict, art_dir: Path) -> None:
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
        objective="reg:absoluteerror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train, sample_weight=w_train)

    print("XGBoost validation / OOS scores:")
    evaluate(xgb, X_val, y_val, "Val 2024")
    evaluate(xgb, X_test, y_test, "Test 2025-2026")

    print("\n=== Training LGBMRegressor (regression_l1 / MAE) ===")
    lgb = LGBMRegressor(
        max_depth=MODEL_PARAMS["max_depth"],
        learning_rate=MODEL_PARAMS["learning_rate"],
        n_estimators=MODEL_PARAMS["n_estimators"],
        objective="regression_l1",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb.fit(X_train, y_train, sample_weight=w_train)

    print("LightGBM validation / OOS scores:")
    evaluate(lgb, X_val, y_val, "Val 2024")
    evaluate(lgb, X_test, y_test, "Test 2025-2026")

    feature_names = list(X_train.columns)
    xgb_imp = np.asarray(xgb.feature_importances_, dtype=float)
    lgb_imp = np.asarray(lgb.feature_importances_, dtype=float)

    print_ascii_importance(feature_names, xgb_imp, "XGBoost feature importance")
    print_ascii_importance(feature_names, lgb_imp, "LightGBM feature importance")

    plot_path = art_dir / "coiled_cobra_ml_feature_importance.png"
    save_importance_plot(feature_names, xgb_imp, lgb_imp, plot_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coiled Cobra ML baseline (XGBoost + LightGBM) for Forward_Return_13w"
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
    train, val, test = temporal_split(df)

    print("\n=== Temporal Split Bounds ===")
    print(f"  Train:  Signal Date <= {TRAIN_END.date()}  -> {len(train)} rows")
    print(
        f"  Val:    {VAL_START.date()} .. {VAL_END.date()}  -> {len(val)} rows"
    )
    print(
        f"  Test:   {TEST_START.date()} .. {TEST_END.date()}  -> {len(test)} rows"
    )

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Empty partition(s): train={len(train)} val={len(val)} test={len(test)}"
        )

    parts = build_matrices(train, val, test)
    train_and_report(parts, art_dir)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
