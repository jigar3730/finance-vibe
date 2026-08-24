"""Coiled Cobra ML baseline: LightGBM + XGBoost on new-coil expansion vs QQQ.

Standalone training script. Looks for coiled_cobra_backtest_trades_*.csv,
keeps ``Is_New_Coil`` rows, isolates pre-signal pillar + raw geometry features,
applies a dynamic relative temporal split, and trains MAE-objective
XGBRegressor / LGBMRegressor baselines for ``Rel_Forward_42d`` (daily) or
``Rel_Forward_13w`` (weekly). Score/Grade are filters and live baselines, not tree features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column zones (strict isolation — no leakage from post-trade metrics)
# ---------------------------------------------------------------------------
# Pillars + raw geometry. Score/Grade excluded (filter + live baseline only).
FEATURE_COLS = [
    "Volume_Shelf",
    "MACD_Compression",
    "Structure",
    "RS_Score",
    "Coil_Width",
    "Proximity_Highs",
    "Volume_Contraction_Ratio",
    "MACD_Spread_ATR",
    "Coil_Width_ATR",
    "Coil_Width_Pctile",
    "Dist_High_63_Pct",
    "Dist_High_63_ATR",
    "Dist_High_126_Pct",
    "Dist_High_126_ATR",
    "Dist_High_252_Pct",
    "Dist_High_252_ATR",
    "OBV_Coil_Slope",
    "Up_Volume_Ratio",
    "Volume_Trend_Ratio",
    "RSI",
    "RSI_Healthy",
    "Pct_From_EMA20",
    "Pct_From_EMA50",
    "ATR_Pct",
    "Distance_To_Pivot_Pct",
    "MACD_Crossed",
]
PREFERRED_TARGET_COL = "Rel_Forward_42d"
FALLBACK_TARGET_COL = "Forward_Return_2w"
DAILY_TARGET_PREFERENCE = (
    "Rel_Forward_42d",
    "Rel_Forward_13w",
    "Rel_Forward_2w",
)
WEEKLY_TARGET_PREFERENCE = (
    "Rel_Forward_13w",
    "Rel_Forward_26w",
    "Rel_Forward_2w",
)
TARGET_COL = PREFERRED_TARGET_COL
TARGET_HORIZON_WEEKS = 9
DATE_COL = "Signal Date"
WEIGHT_COL = "ATR_Pct"
# Experiment switch:
# False = every training row has equal influence.
# True  = quieter stocks receive more influence through inverse ATR weighting.
USE_INVERSE_ATR_WEIGHTS = False
NEW_COIL_COL = "Is_New_Coil"
MODEL_METADATA_FILENAME = "coiled_cobra_ml_model_metadata.json"
EMBARGO_WEEKS = TARGET_HORIZON_WEEKS
EARLY_STOPPING_ROUNDS = 40

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
# LightGBM only applies subsample when bagging_freq > 0.
MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.01,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "bagging_freq": 1,
}


_LOG_SILOS = ("daily", "weekly")  # project primary first; weekly is opt-in confirmation


def _mode_log_dirs() -> list[Path]:
    """Candidate log silos: daily first, then weekly, across cwd / repo / container mounts."""
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]  # src/finance_vibe -> repo root
    cwd = Path.cwd()
    dirs: list[Path] = []
    for silo in _LOG_SILOS:
        dirs.extend(
            [
                cwd / "data" / "logs" / silo,
                project_root / "data" / "logs" / silo,
                Path("/app/data/logs") / silo,
                Path("/mnt/fast/finance-vibe-data/logs") / silo,
            ]
        )
    return dirs


def _candidate_csv_paths(explicit: str | None = None) -> list[Path]:
    """Resolve likely locations for the backtest trades CSV."""
    if explicit:
        return [Path(explicit).expanduser().resolve()]

    names = [SOURCE_FILENAME]
    search_roots = [Path.cwd(), *_mode_log_dirs()]
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

    # Prefer the newest file in the first silo that has any trades CSV (daily, then weekly).
    seen: set[Path] = set()
    for root in _mode_log_dirs():
        resolved = root.resolve() if root.exists() else root
        if resolved in seen:
            continue
        seen.add(resolved)
        if root.is_dir():
            matches = list(root.glob("coiled_cobra_backtest_trades_*.csv"))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)

    tried = "\n  ".join(str(p) for p in _candidate_csv_paths(explicit))
    raise FileNotFoundError(
        f"Could not find {SOURCE_FILENAME}. Tried:\n  {tried}"
    )


def _infer_frame_mode(df: pd.DataFrame) -> str:
    if "Mode" in df.columns:
        modes = df["Mode"].dropna().astype(str).str.strip().str.lower()
        if len(modes) and (modes == "weekly").all():
            return "weekly"
    return "daily"


def embargo_weeks_for_target(target_col: str) -> int:
    mapping = {
        "Rel_Forward_42d": 9,
        "Forward_Return_42d": 9,
        "Rel_Forward_21d": 5,
        "Forward_Return_21d": 5,
        "Rel_Forward_26w": 26,
        "Rel_Forward_13w": 13,
        "Rel_Forward_8w": 8,
        "Rel_Forward_5w": 5,
        "Rel_Forward_4w": 4,
        "Rel_Forward_2w": 2,
        "Forward_Return_2w": 2,
    }
    return mapping.get(target_col, 2)


def select_target_col(df: pd.DataFrame) -> str:
    """Prefer QQQ-relative medium/long horizon; fall back to shorter then absolute."""
    prefs = (
        WEEKLY_TARGET_PREFERENCE
        if _infer_frame_mode(df) == "weekly"
        else DAILY_TARGET_PREFERENCE
    )
    for col in prefs:
        if col in df.columns and df[col].notna().any():
            return col
    if FALLBACK_TARGET_COL in df.columns:
        return FALLBACK_TARGET_COL
    raise ValueError(
        f"Missing target column: need one of {prefs} or {FALLBACK_TARGET_COL}"
    )


def _is_new_coil_mask(series: pd.Series) -> pd.Series:
    """True for boolean True, 1, or the strings 'true' / '1'."""
    if series.dtype == bool:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return (numeric == 1) | text.isin({"true", "1", "yes"})


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load CSV, keep new coils, drop leakage cols, drop NaN targets."""
    global TARGET_COL, EMBARGO_WEEKS, TARGET_HORIZON_WEEKS
    df = pd.read_csv(csv_path)
    print(f"Loaded source: {csv_path}")
    print(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} cols")

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {missing_features}. "
            "Retrain needs a post-v2.1 backtest CSV with rubric pillars."
        )
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing date column: {DATE_COL}")

    TARGET_COL = select_target_col(df)
    EMBARGO_WEEKS = embargo_weeks_for_target(TARGET_COL)
    TARGET_HORIZON_WEEKS = EMBARGO_WEEKS
    print(f"Target column: {TARGET_COL} (embargo={EMBARGO_WEEKS}w)")

    if NEW_COIL_COL in df.columns:
        before_coil = len(df)
        df = df[_is_new_coil_mask(df[NEW_COIL_COL])].copy()
        print(
            f"Kept {len(df)}/{before_coil} row(s) where {NEW_COIL_COL} is True "
            "(aged continuation bars excluded)"
        )
    else:
        print(f"Warning: {NEW_COIL_COL} missing — training on every signal bar.")

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
        f"Dropped {before - len(df)} row(s) with NaN/None {TARGET_COL}"
    )
    print(f"Training pool shape: {df.shape[0]} rows")
    return df.sort_values(DATE_COL).reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Date split with an embargo so train labels do not overlap val/test.

    ``Rel_Forward_2w`` uses the next two bars. Without an embargo, a train
    signal in the last two weeks of the train window has a target that lands
    in validation.
    """
    max_date = df[DATE_COL].max()
    embargo = pd.Timedelta(weeks=EMBARGO_WEEKS)

    test_start = max_date - pd.Timedelta(weeks=26)
    val_start = test_start - pd.Timedelta(weeks=26)

    train = df[df[DATE_COL] < (val_start - embargo)].copy()
    val = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < (test_start - embargo))].copy()
    test = df[df[DATE_COL] >= test_start].copy()

    bounds = {
        "max_date": max_date,
        "val_start": val_start,
        "test_start": test_start,
        "embargo_weeks": EMBARGO_WEEKS,
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
        # Inverse ATR: high-vol names already have noisier Rel_Forward. Weighting
        # *by* ATR_Pct would amplify them; 1/ATR_Pct equalizes return-space MAE.
        if USE_INVERSE_ATR_WEIGHTS:
            # Optional legacy behavior: quieter stocks receive more weight.
            atr = frame[WEIGHT_COL].astype(float).to_numpy()
            w = np.where(
                np.isfinite(atr) & (atr > 0),
                1.0 / atr,
                np.nan,
            )
        else:
            # Default experiment: every setup has equal influence.
             w = np.ones(len(frame), dtype=float)

        parts[name] = {
            "X": X,
            "y": y,
            "w": w,
            "n": len(frame),
        }
        
    med = np.nanmedian(parts["train"]["w"])
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    for name in parts:
        w = parts[name]["w"]
        parts[name]["w"] = np.where(np.isfinite(w) & (w > 0), w, med)
    return parts


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    from sklearn.metrics import ndcg_score

    if len(y_true) < 2:
        return 0.0
    k = min(k, len(y_true))
    # Shift labels so NDCG can handle negatives (rank-only).
    shifted = y_true - float(np.min(y_true)) + 1e-9
    try:
        return float(ndcg_score(shifted.reshape(1, -1), y_pred.reshape(1, -1), k=k))
    except Exception:
        return 0.0


def top_decile_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    n = max(1, len(y_true) // 10)
    order = np.argsort(y_pred)[::-1][:n]
    return float(np.mean(y_true[order]))


def evaluate(model, X: pd.DataFrame, y: np.ndarray, label: str) -> dict:
    from sklearn.metrics import mean_absolute_error
    pred = model.predict(X)
    ic = float(pd.Series(pred).corr(pd.Series(y), method="spearman"))
    if not np.isfinite(ic):
        ic = 0.0
    metrics = {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse(y, pred),
        "spearman": ic,
        "ndcg10": ndcg_at_k(y, pred, k=10),
        "top_decile_mean": top_decile_mean(y, pred),
    }
    print(
        f"  {label}: MAE={metrics['mae']:.6f}  RMSE={metrics['rmse']:.6f}  "
        f"Spearman={metrics['spearman']:.4f}  NDCG@10={metrics['ndcg10']:.4f}  "
        f"TopDecileRel={metrics['top_decile_mean']:.4f}"
    )
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
    import matplotlib.pyplot as plt

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
        "embargo_weeks": EMBARGO_WEEKS,
        "sample_weight": (
            "inverse_ATR_Pct"
            if USE_INVERSE_ATR_WEIGHTS
            else "uniform"
        ),
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
                "val_spearman": xgb_val_metrics["spearman"],
                "val_ndcg10": xgb_val_metrics.get("ndcg10"),
                "val_top_decile_mean": xgb_val_metrics.get("top_decile_mean"),
                "test_mae": xgb_test_metrics["mae"],
                "test_rmse": xgb_test_metrics["rmse"],
                "test_spearman": xgb_test_metrics["spearman"],
                "test_ndcg10": xgb_test_metrics.get("ndcg10"),
                "test_top_decile_mean": xgb_test_metrics.get("top_decile_mean"),
            },
            "lgb": {
                "val_mae": lgb_val_metrics["mae"],
                "val_rmse": lgb_val_metrics["rmse"],
                "val_spearman": lgb_val_metrics["spearman"],
                "val_ndcg10": lgb_val_metrics.get("ndcg10"),
                "val_top_decile_mean": lgb_val_metrics.get("top_decile_mean"),
                "test_mae": lgb_test_metrics["mae"],
                "test_rmse": lgb_test_metrics["rmse"],
                "test_spearman": lgb_test_metrics["spearman"],
                "test_ndcg10": lgb_test_metrics.get("ndcg10"),
                "test_top_decile_mean": lgb_test_metrics.get("top_decile_mean"),
            },
        },
    }
    metadata_path = art_dir / MODEL_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n[SAVED] ML metadata summary: {metadata_path}")


def train_and_report(parts: dict, art_dir: Path, labels: dict) -> None:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from xgboost import XGBRegressor

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
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

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
        bagging_freq=MODEL_PARAMS["bagging_freq"],
        objective="regression_l1",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
    )

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
        help="Directory for plots (default: same directory as the CSV, typically data/logs/daily)",
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
    print(f"  Train:  Signal Date < {bounds['val_start'].strftime('%Y-%m-%d')} minus {bounds['embargo_weeks']}w embargo -> {len(train)} rows")
    print(f"  Val:    {v_str} -> {len(val)} rows")
    print(f"  Test:   {t_str} -> {len(test)} rows")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise RuntimeError(
            f"Empty partition(s): train={len(train)} val={len(val)} test={len(test)}"
        )

    if "Score" in test.columns:
        score = pd.to_numeric(test["Score"], errors="coerce")
        y = test[TARGET_COL].astype(float)
        mask = score.notna() & y.notna()
        if mask.sum() >= 5:
            ic = float(score[mask].corr(y[mask], method="spearman"))
            print(
                f"\nScore-only ranking baseline on test: Spearman={ic:.4f} "
                f"TopDecileRel={top_decile_mean(y[mask].to_numpy(), score[mask].to_numpy()):.4f}"
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