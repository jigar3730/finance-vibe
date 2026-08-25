"""Coiled Cobra ML: separate XGBClassifier models for 10d / 21d / 42d hit targets.

Standalone training script. Looks for coiled_cobra_backtest_trades_*.csv,
keeps ``Is_New_Coil`` rows, isolates pre-signal pillar + raw geometry features,
runs chronological walk-forward with an embargo, and trains one binary
XGBoost model per horizon. Score/Grade are filters and live baselines, not
tree features. Future prices are used only to build targets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
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
SYMBOL_COL = "Symbol"
# Walk-forward fold id stamped on OOS rows. Each fold has its own model, so
# probabilities are only comparable inside a fold — selections are cut per fold.
FOLD_COL = "_Fold"
WEIGHT_COL = "ATR_Pct"
USE_INVERSE_ATR_WEIGHTS = False
NEW_COIL_COL = "Is_New_Coil"
SCORE_COL = "Score"
MODEL_METADATA_FILENAME = "coiled_cobra_ml_model_metadata.json"
EMBARGO_WEEKS = TARGET_HORIZON_WEEKS
EARLY_STOPPING_ROUNDS = 40
WALK_FORWARD_TEST_WEEKS = 26
WALK_FORWARD_VAL_WEEKS = 26
# Expanding-window fold 1 on 10y daily historically has ~250-500 train rows
# and those models stop at a single tree. Keep the floor above that band.
MIN_TRAIN_ROWS = 1000
MIN_VAL_ROWS = 25
MIN_TEST_ROWS = 25
# XGBoost best_iteration is 0-indexed. 0/1/2 trees at lr=0.01 is a near-
# constant equal to the base rate; those predictions must not be pooled or
# shipped. Require at least four trees (best_iteration >= 3).
MIN_BEST_ITERATION = 3
SELECTION_FRACTIONS = (1.0, 0.50, 0.25, 0.10, 0.05)
# The daily pool averages ~9 new coils per signal date (median 8), so a
# "top 20 per date" cut would hand back almost the whole population and
# measure nothing. Keep these well under the typical per-date breadth.
TOP_N_PER_DATE_LEVELS = (5, 3, 1)
# Headline per-date cut used in the promotion table and metadata summary.
TOP_N_PER_DATE = TOP_N_PER_DATE_LEVELS[0]
# Selection cut the promotion gate is judged on.
PROMOTION_SELECTION = "top_10pct"
# A promoted model must win on all of these against every baseline. Hit rate is
# deliberately absent: it is the classifier's own training objective, so beating
# Score on it is guaranteed and says nothing about trading outcomes.
PROMOTION_METRICS = ("avg_fwd", "med_fwd", "win_rate")
RANDOM_SEED = 42
PROMOTION_FOLD_PASS_RATE = 0.60
OOS_PREDICTIONS_TEMPLATE = "coiled_cobra_ml_oos_{key}.csv"

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
LEAKAGE_PREFIXES = (
    "Forward_Return_",
    "Rel_Forward_",
    "MAE_",
    "Max_Return_",
    "Held_Coil_Low_",
    "Win_",
    "Hit_",
    "ML_Prob_",
    "ML_Pred_",
    "ML_Rank",
)

SOURCE_FILENAME = "coiled_cobra_backtest_trades_2026-07-17.csv"

MODEL_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.01,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 16,
    "reg_lambda": 2.0,
}

HORIZON_SPECS = (
    {
        "key": "10d",
        "target": "Win",
        "forward_col": "Forward_Return_10d",
        "max_col": "Max_Return_10d",
        "win_col": "Win_10d",
        "label_col": "Win_10d",
        "hit_col": "Hit_10Pct_10d",
        "prob_col": "ML_Prob_Win_10d",
        "logistic_prob_col": "_Logistic_Prob_Win_10d",
        "threshold": 0.0,
        "embargo_weeks": 2,
        "bars": 10,
        "model_filename": "coiled_cobra_xgb_10d.json",
        "metadata_filename": "coiled_cobra_ml_metadata_10d.json",
        "research_targets": ("Hit_10Pct_10d", "Hit_15Pct_10d"),
        "all_hits": (("Hit_10Pct_10d", 0.10), ("Hit_15Pct_10d", 0.15)),
        "legacy_forward": "Forward_Return_2w",
        "legacy_max": "Max_Return_2w",
    },
    {
        "key": "21d",
        "target": "Win",
        "forward_col": "Forward_Return_21d",
        "max_col": "Max_Return_21d",
        "win_col": "Win_21d",
        "label_col": "Win_21d",
        "hit_col": "Hit_15Pct_21d",
        "prob_col": "ML_Prob_Win_21d",
        "logistic_prob_col": "_Logistic_Prob_Win_21d",
        "threshold": 0.0,
        "embargo_weeks": 5,
        "bars": 21,
        "model_filename": "coiled_cobra_xgb_21d.json",
        "metadata_filename": "coiled_cobra_ml_metadata_21d.json",
        "research_targets": ("Hit_15Pct_21d", "Hit_20Pct_21d"),
        "all_hits": (
            ("Hit_10Pct_21d", 0.10),
            ("Hit_15Pct_21d", 0.15),
            ("Hit_20Pct_21d", 0.20),
        ),
        "legacy_forward": None,
        "legacy_max": None,
    },
    {
        "key": "42d",
        "target": "Win",
        "forward_col": "Forward_Return_42d",
        "max_col": "Max_Return_42d",
        "win_col": "Win_42d",
        "label_col": "Win_42d",
        "hit_col": "Hit_25Pct_42d",
        "prob_col": "ML_Prob_Win_42d",
        "logistic_prob_col": "_Logistic_Prob_Win_42d",
        "threshold": 0.0,
        "embargo_weeks": 9,
        "bars": 42,
        "model_filename": "coiled_cobra_xgb_42d.json",
        "metadata_filename": "coiled_cobra_ml_metadata_42d.json",
        "research_targets": ("Hit_25Pct_42d", "Hit_50Pct_42d"),
        "all_hits": (
            ("Hit_10Pct_42d", 0.10),
            ("Hit_25Pct_42d", 0.25),
            ("Hit_50Pct_42d", 0.50),
        ),
        "legacy_forward": None,
        "legacy_max": None,
    },
)


def label_col(spec: dict) -> str:
    """Binary training target. MFE ``hit_col`` is kept for research reporting."""
    return spec.get("label_col") or spec["hit_col"]


FEATURE_DECISIONS = {
    "Volume_Shelf": "KEEP",
    "MACD_Compression": "KEEP",
    "Structure": "KEEP",
    "RS_Score": "KEEP",
    "Coil_Width": "KEEP",
    "Proximity_Highs": "KEEP",
    "Volume_Contraction_Ratio": "KEEP",
    "MACD_Spread_ATR": "KEEP",
    "Coil_Width_ATR": "KEEP",
    "Coil_Width_Pctile": "KEEP",
    "Dist_High_63_Pct": "KEEP",
    "Dist_High_63_ATR": "KEEP",
    "Dist_High_126_Pct": "KEEP",
    "Dist_High_126_ATR": "KEEP",
    "Dist_High_252_Pct": "KEEP",
    "Dist_High_252_ATR": "KEEP",
    "OBV_Coil_Slope": "KEEP",
    "Up_Volume_Ratio": "KEEP",
    "Volume_Trend_Ratio": "KEEP",
    "RSI": "KEEP",
    "RSI_Healthy": "EXPERIMENTAL",
    "Pct_From_EMA20": "KEEP",
    "Pct_From_EMA50": "KEEP",
    "ATR_Pct": "KEEP",
    "Distance_To_Pivot_Pct": "KEEP",
    "MACD_Crossed": "EXPERIMENTAL",
}

BUCKET_FEATURES = (
    "ATR_Pct",
    "Distance_To_Pivot_Pct",
    "MACD_Compression",
    "RSI",
    "Up_Volume_Ratio",
    "Dist_High_126_Pct",
    "Dist_High_252_Pct",
)

_LOG_SILOS = ("daily", "weekly")


def _mode_log_dirs() -> list[Path]:
    """Candidate log silos: daily first, then weekly, across cwd / repo / container mounts."""
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
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
        "Hit_25Pct_42d": 9,
        "Max_Return_42d": 9,
        "Win_42d": 9,
        "Rel_Forward_21d": 5,
        "Forward_Return_21d": 5,
        "Hit_15Pct_21d": 5,
        "Max_Return_21d": 5,
        "Win_21d": 5,
        "Forward_Return_10d": 2,
        "Hit_10Pct_10d": 2,
        "Max_Return_10d": 2,
        "Win_10d": 2,
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


def assert_features_have_no_leakage(feature_names: list[str] | None = None) -> None:
    names = list(feature_names or FEATURE_COLS)
    leaked = [c for c in names if c in LEAKAGE_COLS or c == SCORE_COL or c == "Grade"]
    for c in names:
        if any(c.startswith(p) or c == p for p in LEAKAGE_PREFIXES):
            leaked.append(c)
    if leaked:
        raise ValueError(f"Feature list includes leakage / future columns: {leaked}")


def drop_duplicate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate episodes and abort on conflicting duplicates."""
    if SYMBOL_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(
            f"Cannot check duplicate signals: missing {SYMBOL_COL} or {DATE_COL}"
        )
    keys = [SYMBOL_COL, DATE_COL]
    dup_mask = df.duplicated(subset=keys, keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_extra = int(df.duplicated(subset=keys, keep="first").sum())
    if not n_dup_rows:
        print("Duplicate count: 0  removed: 0  conflicting: 0")
        return df

    compare_cols = [c for c in df.columns if c not in keys]
    conflicts: list[tuple[str, str]] = []
    for key, group in df.loc[dup_mask].groupby(keys, dropna=False, sort=False):
        if not compare_cols:
            continue
        normalized = group[compare_cols].copy()
        for col in compare_cols:
            normalized[col] = normalized[col].map(
                lambda value: "<NA>" if pd.isna(value) else str(value)
            )
        if len(normalized.drop_duplicates()) > 1:
            conflicts.append((str(key[0]), str(key[1])))

    print(
        f"Duplicate count: {n_dup_rows}  removed: {n_extra}  "
        f"conflicting: {len(conflicts)}"
    )
    if conflicts:
        sample = ", ".join(f"{s}@{d}" for s, d in conflicts[:5])
        raise ValueError(
            f"Conflicting duplicate {SYMBOL_COL}+{DATE_COL} records: "
            f"{len(conflicts)} group(s); sample: {sample}"
        )
    return df.drop_duplicates(subset=keys, keep="first").copy()


def _ensure_horizon_targets(df: pd.DataFrame, require_max: bool = True) -> pd.DataFrame:
    """Fill Forward/Max/Win/Hit columns from aliases when a fresh backtest is missing.

    ``Hit_*`` is only ever derived from ``Max_Return_*`` (intra-horizon high). There
    is no close-based fallback: mixing "close >= X%" and "any high >= X%" rows in one
    training pool silently trains on two different questions. With ``require_max``
    a horizon that cannot supply ``Max_Return_*`` raises instead.
    """
    out = df.copy()
    for spec in HORIZON_SPECS:
        fwd = spec["forward_col"]
        mx = spec["max_col"]
        if fwd not in out.columns and spec.get("legacy_forward") and spec["legacy_forward"] in out.columns:
            out[fwd] = pd.to_numeric(out[spec["legacy_forward"]], errors="coerce")
            print(f"Derived {fwd} from {spec['legacy_forward']}")
        if mx not in out.columns and spec.get("legacy_max") and spec["legacy_max"] in out.columns:
            out[mx] = pd.to_numeric(out[spec["legacy_max"]], errors="coerce")
        if fwd in out.columns:
            fwd_num = pd.to_numeric(out[fwd], errors="coerce")
            if spec["win_col"] not in out.columns:
                out[spec["win_col"]] = np.where(fwd_num.isna(), np.nan, (fwd_num > 0).astype(float))
        if mx in out.columns:
            mx_num = pd.to_numeric(out[mx], errors="coerce")
            for hit_col, threshold in spec["all_hits"]:
                if hit_col not in out.columns:
                    out[hit_col] = np.where(
                        mx_num.isna(),
                        np.nan,
                        (mx_num >= threshold).astype(float),
                    )
        else:
            absent = [h for h, _ in spec["all_hits"] if h not in out.columns]
            if absent and require_max:
                raise ValueError(
                    f"{mx} is missing, so {absent} cannot be labelled. Hit labels "
                    "require the intra-horizon high and must never be approximated "
                    "from the horizon close. Re-run the backtest "
                    "(`python -m finance_vibe.coiled_cobra_backtest daily --backtest`) "
                    "so the source CSV carries native Max_Return_*/Hit_* columns."
                )
    return out


def analyze_feature_quality(df: pd.DataFrame) -> dict:
    """Missingness, variance, outliers, redundancy, and time stability."""
    rows: list[dict] = []
    numeric = _feature_matrix(df)
    midpoint = df[DATE_COL].min() + (df[DATE_COL].max() - df[DATE_COL].min()) / 2
    early = numeric.loc[df[DATE_COL] < midpoint]
    late = numeric.loc[df[DATE_COL] >= midpoint]
    for col in FEATURE_COLS:
        s = numeric[col]
        non_null = s.dropna()
        dominant = float(non_null.value_counts(normalize=True).iloc[0]) if len(non_null) else 1.0
        q1, q3 = non_null.quantile([0.25, 0.75]) if len(non_null) else (np.nan, np.nan)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            outlier_rate = float(((non_null < q1 - 3 * iqr) | (non_null > q3 + 3 * iqr)).mean())
        else:
            outlier_rate = 0.0
        early_med = float(early[col].median()) if early[col].notna().any() else np.nan
        late_med = float(late[col].median()) if late[col].notna().any() else np.nan
        scale = max(abs(early_med), float(non_null.std()) if len(non_null) > 1 else 0.0, 1e-9)
        rows.append(
            {
                "feature": col,
                "decision": FEATURE_DECISIONS[col],
                "missing_count": int(s.isna().sum()),
                "missing_rate": float(s.isna().mean()),
                "unique_count": int(non_null.nunique()),
                "dominant_rate": dominant,
                "constant": bool(non_null.nunique() <= 1),
                "near_constant": bool(dominant >= 0.995),
                "outlier_rate_3iqr": outlier_rate,
                "early_median": early_med,
                "late_median": late_med,
                "median_shift_scaled": float(abs(late_med - early_med) / scale)
                if np.isfinite(early_med) and np.isfinite(late_med)
                else np.nan,
            }
        )

    corr = numeric.corr(method="spearman", min_periods=30).abs()
    correlated: list[dict] = []
    for i, left in enumerate(FEATURE_COLS):
        for right in FEATURE_COLS[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= 0.90:
                correlated.append(
                    {"left": left, "right": right, "abs_spearman": float(value)}
                )
    return {"features": rows, "high_correlation_pairs": correlated}


def report_feature_quality(df: pd.DataFrame) -> dict:
    print("\n=== Feature quality (new-coil training pool) ===")
    quality = analyze_feature_quality(df)
    for row in quality["features"]:
        print(
            f"  {row['feature']:<24} {row['decision']:<12} "
            f"missing={row['missing_rate']:5.1%} unique={row['unique_count']:5d} "
            f"dominant={row['dominant_rate']:5.1%} "
            f"outliers={row['outlier_rate_3iqr']:5.1%}"
        )
    if quality["high_correlation_pairs"]:
        print("  High-correlation pairs (|Spearman| >= 0.90):")
        for pair in quality["high_correlation_pairs"]:
            print(
                f"    {pair['left']} ~ {pair['right']}: "
                f"{pair['abs_spearman']:.3f}"
            )
    return quality


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load CSV, keep new coils, drop leakage cols, drop duplicate episodes."""
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

    assert_features_have_no_leakage(FEATURE_COLS)

    if any(c in df.columns for c in DAILY_TARGET_PREFERENCE) or FALLBACK_TARGET_COL in df.columns:
        TARGET_COL = select_target_col(df)
        EMBARGO_WEEKS = embargo_weeks_for_target(TARGET_COL)
        TARGET_HORIZON_WEEKS = EMBARGO_WEEKS
        print(f"Legacy regression target (reference only): {TARGET_COL}")

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

    df = drop_duplicate_signals(df)
    df = _ensure_horizon_targets(df, require_max=False)
    need_mfe = any(
        spec["max_col"] not in df.columns or pd.to_numeric(df[spec["max_col"]], errors="coerce").isna().all()
        for spec in HORIZON_SPECS
    )
    if need_mfe:
        print("Max_Return_* missing — enriching hit labels from raw OHLC (targets only).")
        from finance_vibe.coiled_cobra_backtest import enrich_mfe_targets

        df = enrich_mfe_targets(df, mode=_infer_frame_mode(df), strict=True)
    df = _ensure_horizon_targets(df, require_max=True)
    report_feature_quality(df)
    print(f"Training pool shape: {df.shape[0]} rows")
    return df.sort_values(DATE_COL).reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    embargo_weeks: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Date split with an embargo so train labels do not overlap val/test."""
    weeks = EMBARGO_WEEKS if embargo_weeks is None else embargo_weeks
    max_date = df[DATE_COL].max()
    embargo = pd.Timedelta(weeks=weeks)

    test_start = max_date - pd.Timedelta(weeks=26)
    val_start = test_start - pd.Timedelta(weeks=26)

    train = df[df[DATE_COL] < (val_start - embargo)].copy()
    val = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < (test_start - embargo))].copy()
    test = df[df[DATE_COL] >= test_start].copy()

    bounds = {
        "max_date": max_date,
        "val_start": val_start,
        "test_start": test_start,
        "embargo_weeks": weeks,
    }
    return train, val, test, bounds


def walk_forward_folds(
    df: pd.DataFrame,
    embargo_weeks: int,
    test_weeks: int = WALK_FORWARD_TEST_WEEKS,
    val_weeks: int = WALK_FORWARD_VAL_WEEKS,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_val_rows: int = MIN_VAL_ROWS,
    min_test_rows: int = MIN_TEST_ROWS,
) -> list[dict]:
    """Expanding-window folds walking backward from max date, then reversed.

    Test windows do not overlap. Train ends embargo weeks before val; val ends
    embargo weeks before test. No random shuffling.
    """
    if df.empty:
        return []
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    embargo = pd.Timedelta(weeks=embargo_weeks)
    folds: list[dict] = []
    test_end = max_date

    while True:
        test_start = test_end - pd.Timedelta(weeks=test_weeks)
        if test_start <= min_date:
            break
        val_end = test_start - embargo
        val_start = val_end - pd.Timedelta(weeks=val_weeks)
        train_end = val_start - embargo
        train = df[df[DATE_COL] < train_end]
        val = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < val_end)]
        test = df[(df[DATE_COL] >= test_start) & (df[DATE_COL] <= test_end)]
        if len(train) < min_train_rows:
            break
        if len(val) >= min_val_rows and len(test) >= min_test_rows:
            folds.append(
                {
                    "train": train,
                    "val": val,
                    "test": test,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "embargo_weeks": embargo_weeks,
                }
            )
        test_end = test_start - pd.Timedelta(days=1)

    folds.reverse()
    return folds


def horizon_frame(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Rows with a defined training label for one horizon (features may still be NaN)."""
    target = label_col(spec)
    if target not in df.columns:
        raise ValueError(f"Missing training target {target}. Re-run daily --backtest.")
    out = df[pd.to_numeric(df[target], errors="coerce").notna()].copy()
    out[target] = pd.to_numeric(out[target], errors="coerce").astype(int)
    if spec["forward_col"] in out.columns:
        out[spec["forward_col"]] = pd.to_numeric(out[spec["forward_col"]], errors="coerce")
    if spec["win_col"] in out.columns:
        out[spec["win_col"]] = pd.to_numeric(out[spec["win_col"]], errors="coerce")
    if spec["hit_col"] in out.columns:
        out[spec["hit_col"]] = pd.to_numeric(out[spec["hit_col"]], errors="coerce")
    return out


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.reindex(columns=FEATURE_COLS)
    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def top_n_per_date(
    frame: pd.DataFrame,
    rank_col: str,
    n: int = TOP_N_PER_DATE,
) -> pd.DataFrame:
    if frame.empty or rank_col not in frame.columns:
        return frame.iloc[0:0].copy()
    ranked = frame.copy()
    ranked["_rank"] = pd.to_numeric(ranked[rank_col], errors="coerce")
    ranked = ranked[ranked["_rank"].notna()]
    if ranked.empty:
        return ranked
    parts = []
    for _, g in ranked.groupby(DATE_COL, sort=True):
        take = min(n, len(g))
        parts.append(g.nlargest(take, "_rank", keep="first"))
    return pd.concat(parts, ignore_index=True) if parts else ranked.iloc[0:0].copy()


def summarize_selection(
    selected: pd.DataFrame,
    forward_col: str,
    hit_col: str,
    win_col: str,
) -> dict:
    if selected.empty or forward_col not in selected.columns:
        return {
            "n": 0,
            "avg_fwd": float("nan"),
            "med_fwd": float("nan"),
            "win_rate": float("nan"),
            "hit_rate": float("nan"),
        }
    fwd = pd.to_numeric(selected[forward_col], errors="coerce")
    valid = selected.loc[fwd.notna()].copy()
    fwd = pd.to_numeric(valid[forward_col], errors="coerce")
    hit = pd.to_numeric(valid[hit_col], errors="coerce") if hit_col in valid.columns else pd.Series(dtype=float)
    win = pd.to_numeric(valid[win_col], errors="coerce") if win_col in valid.columns else (fwd > 0).astype(float)
    return {
        "n": int(len(valid)),
        "avg_fwd": float(fwd.mean()) if len(valid) else float("nan"),
        "med_fwd": float(fwd.median()) if len(valid) else float("nan"),
        "win_rate": float(win.mean()) if len(valid) and win.notna().any() else float("nan"),
        "hit_rate": float(hit.mean()) if len(valid) and hit.notna().any() else float("nan"),
    }


def top_fraction(
    frame: pd.DataFrame,
    rank_col: str,
    fraction: float,
    group_col: str | None = FOLD_COL,
) -> pd.DataFrame:
    """Highest-ranked fraction, cut separately inside each ``group_col`` block.

    Every walk-forward fold is scored by its own model, so a 0.60 probability in
    one fold is not the same conviction as a 0.60 in another. Ranking the pooled
    frame would let whichever folds emit systematically higher probabilities
    monopolize the top decile, turning a period effect into apparent skill. The
    fraction is therefore taken per fold and the picks are pooled afterwards, so
    each fold contributes its own share. Pass ``group_col=None`` to rank a frame
    that already comes from a single model.
    """
    ranked = frame.copy()
    ranked["_rank"] = pd.to_numeric(ranked.get(rank_col), errors="coerce")
    ranked = ranked[ranked["_rank"].notna()]
    if ranked.empty:
        return ranked
    if fraction >= 1:
        return ranked
    if group_col and group_col in ranked.columns:
        parts = [
            group.nlargest(
                max(1, int(np.ceil(len(group) * fraction))), "_rank", keep="first"
            )
            for _, group in ranked.groupby(group_col, sort=True)
        ]
        return pd.concat(parts, ignore_index=True) if parts else ranked.iloc[0:0].copy()
    return ranked.nlargest(
        max(1, int(np.ceil(len(ranked) * fraction))), "_rank", keep="first"
    )


def classification_metrics(y_true: pd.Series, probabilities: pd.Series) -> dict:
    """Standard classifier diagnostics at a fixed 0.5 probability threshold."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = pd.to_numeric(y_true, errors="coerce")
    p = pd.to_numeric(probabilities, errors="coerce")
    mask = y.notna() & p.notna()
    y = y[mask].astype(int).to_numpy()
    p = p[mask].clip(0, 1).to_numpy()
    if len(y) == 0:
        return {k: np.nan for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "brier")}
    pred = (p >= 0.5).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)) if two_classes else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else np.nan,
        "brier": float(brier_score_loss(y, p)),
    }


def evaluate_ranker(
    oos: pd.DataFrame,
    spec: dict,
    rank_col: str,
    model_name: str,
) -> dict:
    """Trading outcomes for population/fractions and per-date top-N selections.

    Fraction cuts are per walk-forward fold and per-date cuts are per signal
    date, so no selection ever compares scores from two different models.
    """
    forward_col = spec["forward_col"]
    hit_col = spec["hit_col"]
    win_col = spec["win_col"]
    fwd = pd.to_numeric(oos[forward_col], errors="coerce") if forward_col in oos.columns else pd.Series(dtype=float)
    usable = oos.loc[fwd.notna()].copy()
    n_folds = int(usable[FOLD_COL].nunique()) if FOLD_COL in usable.columns else 1
    selections: dict[str, dict] = {}
    for fraction in SELECTION_FRACTIONS:
        label = "population" if fraction >= 1 else f"top_{int(fraction * 100)}pct"
        selected = usable if fraction >= 1 else top_fraction(usable, rank_col, fraction)
        selections[label] = summarize_selection(
            selected, forward_col, hit_col, win_col
        )
    for n in TOP_N_PER_DATE_LEVELS:
        selections[f"top_{n}_per_date"] = summarize_selection(
            top_n_per_date(usable, rank_col, n),
            forward_col,
            hit_col,
            win_col,
        )
    # A cut that keeps most of the population cannot demonstrate selection
    # skill, so record how much each one actually discards.
    population_n = selections["population"]["n"]
    for summary in selections.values():
        summary["selected_fraction"] = (
            summary["n"] / population_n if population_n else float("nan")
        )
    return {
        "model": model_name,
        "rank_column": rank_col,
        "selection_scope": "per_fold" if n_folds > 1 else "single_model",
        "n_selection_groups": n_folds,
        "selections": selections,
    }


def _finite(value) -> bool:
    return value is not None and np.isfinite(float(value))


def evaluate_promotion(candidate: dict, baselines: dict[str, dict]) -> dict:
    """Check one selection against every baseline on the promotion metrics.

    Beating ``Score`` alone is far too weak a bar: on this pool Score's own top
    decile is statistically indistinguishable from a random cut, and at 42d it
    trails the population it draws from. A model therefore has to beat Score,
    a random ranker, and the untouched population before it can be promoted.
    """
    per_metric: dict[str, dict] = {}
    for metric in PROMOTION_METRICS:
        value = candidate.get(metric)
        beats = {
            name: bool(_finite(value) and _finite(base.get(metric)) and value > base[metric])
            for name, base in baselines.items()
        }
        per_metric[metric] = {
            "value": float(value) if _finite(value) else None,
            "beats": beats,
            "passed": bool(beats) and all(beats.values()),
        }
    failed = [metric for metric, r in per_metric.items() if not r["passed"]]
    return {
        "selection": PROMOTION_SELECTION,
        "baselines": sorted(baselines),
        "metrics": per_metric,
        "failed_metrics": failed,
        "passed": bool(candidate.get("n", 0) > 0 and not failed),
    }


def compare_rankers(oos: pd.DataFrame, spec: dict) -> dict:
    """Compare Score, Logistic, XGB, and deterministic random OOS ranks."""
    rankers = {
        "score": SCORE_COL,
        "logistic": spec["logistic_prob_col"],
        "xgb": spec["prob_col"],
        "random": "_Random_Rank",
    }
    models = {
        name: evaluate_ranker(oos, spec, rank_col, name)
        for name, rank_col in rankers.items()
        if rank_col in oos.columns
    }
    per_date_key = f"top_{TOP_N_PER_DATE}_per_date"
    score_top = models["score"]["selections"][PROMOTION_SELECTION]
    xgb_top = models["xgb"]["selections"][PROMOTION_SELECTION]
    base_hit = float(pd.to_numeric(oos[label_col(spec)], errors="coerce").mean())

    baselines = {
        "score": score_top,
        # Same rows for every ranker, so this is the "did selecting help at all"
        # test rather than a rival model.
        "population": models["xgb"]["selections"]["population"],
    }
    if "random" in models:
        baselines["random"] = models["random"]["selections"][PROMOTION_SELECTION]
    promotion = evaluate_promotion(xgb_top, baselines)

    return {
        "horizon": spec["key"],
        "target": spec["target"],
        "base_hit_rate": base_hit,
        "n_oos": int(len(oos)),
        "models": models,
        "per_date_n": TOP_N_PER_DATE,
        "score_per_date": models["score"]["selections"][per_date_key],
        "ml_per_date": models["xgb"]["selections"][per_date_key],
        "score_top10pct_avg": score_top["avg_fwd"],
        "score_top5pct_avg": models["score"]["selections"]["top_5pct"]["avg_fwd"],
        "ml_top10pct_avg": xgb_top["avg_fwd"],
        "ml_top5pct_avg": models["xgb"]["selections"]["top_5pct"]["avg_fwd"],
        "promotion": promotion,
        "promote": promotion["passed"],
    }


def _fit_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict | None = None,
):
    from xgboost import XGBClassifier

    cfg = {**MODEL_PARAMS, **(params or {})}
    model = XGBClassifier(
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=cfg["min_child_weight"],
        reg_lambda=cfg["reg_lambda"],
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        # Baseline uses standard weighting. Imbalance experiments must earn
        # their place OOS instead of being enabled automatically.
        scale_pos_weight=float(cfg.get("scale_pos_weight", 1.0)),
        n_jobs=-1,
        random_state=42,
        missing=np.nan,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _fit_logistic_classifier(X_train: pd.DataFrame, y_train: np.ndarray):
    """Simple, deterministic linear baseline with train-only median imputation."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight=None,
            random_state=RANDOM_SEED,
        ),
    )
    model.fit(X_train, y_train)
    return model


def class_balance(series: pd.Series) -> dict:
    y = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    total = positive + negative
    return {
        "positive": positive,
        "negative": negative,
        "positive_rate": positive / total if total else np.nan,
    }


def report_all_target_balances(df: pd.DataFrame) -> dict[str, dict]:
    """Print class balance for every generated binary horizon target."""
    balances: dict[str, dict] = {}
    print("\n=== Binary target class balance ===")
    for spec in HORIZON_SPECS:
        for target in (spec["win_col"], *(name for name, _ in spec["all_hits"])):
            if target not in df.columns:
                continue
            balance = class_balance(df[target])
            balances[target] = balance
            print(
                f"  {target:<20} positive={balance['positive']:6d} "
                f"negative={balance['negative']:6d} "
                f"positive_rate={balance['positive_rate']:.2%}"
            )
    return balances


def _best_iteration(model) -> int:
    best = getattr(model, "best_iteration", None)
    if best is None:
        booster = model.get_booster()
        best = getattr(booster, "best_iteration", None)
    if best is None:
        best = int(model.n_estimators) - 1
    return int(best)


def booster_is_degenerate(best_iteration: int | None) -> bool:
    """True when the booster is a stump / near-constant at the training lr."""
    if best_iteration is None:
        return True
    try:
        best = int(best_iteration)
    except (TypeError, ValueError):
        return True
    return best < MIN_BEST_ITERATION


def save_xgb_artifact(model, path: Path, best_iteration: int) -> bool:
    """Write the booster only if it is more than a near-constant stump.

    Returns True if saved. A degenerate leftover from a previous run is deleted
    so inference cannot load it.
    """
    path = Path(path)
    if booster_is_degenerate(best_iteration):
        if path.is_file():
            path.unlink()
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(path))
    return True


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    best = _best_iteration(model)
    return model.predict_proba(X, iteration_range=(0, best + 1))[:, 1]


def evaluate_research_targets(
    labeled: pd.DataFrame,
    folds: list[dict],
    spec: dict,
    xgb_params: dict | None,
) -> dict:
    """Walk-forward XGB checks for optional stricter hit thresholds."""
    results: dict[str, dict] = {}
    for research_target in spec.get("research_targets", ()):
        if research_target not in labeled.columns:
            continue
        prob_col = f"_Research_Prob_{research_target}"
        parts: list[pd.DataFrame] = []
        for i, fold in enumerate(folds, start=1):
            train = fold["train"].dropna(subset=[research_target])
            val = fold["val"].dropna(subset=[research_target])
            test = fold["test"].dropna(subset=[research_target])
            if min(len(train), len(val), len(test)) == 0:
                continue
            y_train = train[research_target].astype(int).to_numpy()
            y_val = val[research_target].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            model = _fit_xgb_classifier(
                _feature_matrix(train),
                y_train,
                _feature_matrix(val),
                y_val,
                xgb_params,
            )
            if booster_is_degenerate(_best_iteration(model)):
                continue
            out = test.copy()
            out[prob_col] = _predict_proba(model, _feature_matrix(test))
            out["_Random_Rank"] = np.random.default_rng(
                RANDOM_SEED + 1000 + i
            ).random(len(out))
            out[FOLD_COL] = i
            parts.append(out)
        if not parts:
            continue
        oos = pd.concat(parts, ignore_index=True)
        research_spec = {
            **spec,
            "hit_col": research_target,
            "prob_col": prob_col,
            "logistic_prob_col": "_unused",
        }
        xgb = evaluate_ranker(oos, research_spec, prob_col, "xgb")
        score = evaluate_ranker(oos, research_spec, SCORE_COL, "score")
        results[research_target] = {
            "class_balance": class_balance(labeled[research_target]),
            "classification": classification_metrics(
                oos[research_target], oos[prob_col]
            ),
            "score_top_10pct": score["selections"]["top_10pct"],
            "xgb_top_10pct": xgb["selections"]["top_10pct"],
        }
    return results


def feature_bucket_analysis(df: pd.DataFrame) -> dict:
    """Quintile buckets showing horizon-specific feature behavior."""
    analysis: dict[str, list[dict]] = {}
    for feature in BUCKET_FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce")
        valid = values.notna()
        if valid.sum() < 20 or values[valid].nunique() < 2:
            continue
        try:
            buckets = pd.qcut(values[valid], q=5, duplicates="drop")
        except ValueError:
            continue
        rows: list[dict] = []
        for bucket, index in buckets.groupby(buckets, observed=True).groups.items():
            group = df.loc[index]
            row = {"bucket": str(bucket), "count": int(len(group))}
            for spec in HORIZON_SPECS:
                hit = (
                    pd.to_numeric(group[spec["hit_col"]], errors="coerce")
                    if spec["hit_col"] in group.columns
                    else pd.Series(dtype=float)
                )
                fwd = (
                    pd.to_numeric(group[spec["forward_col"]], errors="coerce")
                    if spec["forward_col"] in group.columns
                    else pd.Series(dtype=float)
                )
                row[f"{spec['key']}_hit_rate"] = (
                    float(hit.mean()) if len(hit) else np.nan
                )
                row[f"{spec['key']}_avg_return"] = (
                    float(fwd.mean()) if len(fwd) else np.nan
                )
            rows.append(row)
        analysis[feature] = rows
    return analysis


def normalized_xgb_gain(model) -> dict[str, float]:
    raw = model.get_booster().get_score(importance_type="gain")
    values = {name: float(raw.get(name, 0.0)) for name in FEATURE_COLS}
    total = sum(values.values())
    return {
        name: (value / total if total > 0 else 0.0)
        for name, value in values.items()
    }


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_oos_predictions(oos: pd.DataFrame, spec: dict, art_dir: Path) -> Path:
    """Persist one row per walk-forward OOS signal with its ranks and outcomes.

    Keeping the raw predictions on disk means a later change to how selections
    are measured can be re-scored offline and diffed against this run, instead
    of being tangled up with a retrain that also moved the predictions.
    """
    columns = [
        SYMBOL_COL,
        DATE_COL,
        FOLD_COL,
        SCORE_COL,
        spec["prob_col"],
        spec["logistic_prob_col"],
        "_Random_Rank",
        spec["forward_col"],
        spec["max_col"],
        spec["win_col"],
        *[hit for hit, _ in spec["all_hits"]],
    ]
    present = [c for c in dict.fromkeys(columns) if c in oos.columns]
    out = oos[present].sort_values([FOLD_COL, DATE_COL, SYMBOL_COL])
    art_dir.mkdir(parents=True, exist_ok=True)
    path = art_dir / OOS_PREDICTIONS_TEMPLATE.format(key=spec["key"])
    out.to_csv(path, index=False)
    print(f"[SAVED] {spec['key']} OOS predictions: {path}  rows={len(out)}")
    return path


def print_selection_comparison(pooled: dict, spec: dict) -> None:
    """Per-cut ML vs Score vs random, with median and win rate alongside mean.

    The mean alone is easy to win with a handful of fat-tailed outliers, so the
    median and win rate are printed next to it rather than buried in metadata.
    """
    models = pooled["models"]
    order = ["population", *(f"top_{int(f * 100)}pct" for f in SELECTION_FRACTIONS if f < 1)]
    order += [f"top_{n}_per_date" for n in TOP_N_PER_DATE_LEVELS]
    header = (
        f"  {'Selection':<16}{'N':>7}{'Keep':>7} | "
        f"{'ML avg':>9}{'Score':>9}{'Rand':>9} | "
        f"{'ML med':>9}{'Score':>9} | {'ML wr':>7}{'Score':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in order:
        xgb = models["xgb"]["selections"].get(label)
        if not xgb:
            continue
        score = models["score"]["selections"][label]
        rand = models.get("random", {}).get("selections", {}).get(label, {})
        rand_avg = rand.get("avg_fwd", float("nan"))
        print(
            f"  {label:<16}{xgb['n']:>7}{xgb['selected_fraction']:>7.1%} | "
            f"{xgb['avg_fwd']:>+9.4f}{score['avg_fwd']:>+9.4f}{rand_avg:>+9.4f} | "
            f"{xgb['med_fwd']:>+9.4f}{score['med_fwd']:>+9.4f} | "
            f"{xgb['win_rate']:>7.3f}{score['win_rate']:>7.3f}"
        )


def train_horizon(
    df: pd.DataFrame,
    spec: dict,
    art_dir: Path,
    *,
    xgb_params: dict | None = None,
    source_csv: Path | None = None,
) -> dict:
    """Walk-forward train one XGBClassifier; refit on all pre-final-test data for artifacts."""
    labeled = horizon_frame(df, spec)
    target = label_col(spec)
    balance = class_balance(labeled[target])
    print(f"\n======== Horizon {spec['key']}  target={target}  embargo={spec['embargo_weeks']}w ========")
    print(
        f"Class balance: positive={balance['positive']} "
        f"negative={balance['negative']} "
        f"positive_rate={balance['positive_rate']:.2%}"
    )

    folds = walk_forward_folds(
        labeled,
        spec["embargo_weeks"],
        min_train_rows=MIN_TRAIN_ROWS,
        min_val_rows=MIN_VAL_ROWS,
        min_test_rows=MIN_TEST_ROWS,
    )
    if not folds:
        train, val, test, bounds = temporal_split(labeled, spec["embargo_weeks"])
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise RuntimeError(
                f"{spec['key']}: empty partition(s) train={len(train)} val={len(val)} test={len(test)}"
            )
        folds = [
            {
                "train": train,
                "val": val,
                "test": test,
                "train_end": bounds["val_start"] - pd.Timedelta(weeks=spec["embargo_weeks"]),
                "val_start": bounds["val_start"],
                "val_end": bounds["test_start"] - pd.Timedelta(weeks=spec["embargo_weeks"]),
                "test_start": bounds["test_start"],
                "test_end": bounds["max_date"],
                "embargo_weeks": spec["embargo_weeks"],
            }
        ]
        print("Walk-forward produced no folds; using single chronological split.")

    print(f"Walk-forward folds: {len(folds)}")
    oos_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for i, fold in enumerate(folds, start=1):
        train, val, test = fold["train"], fold["val"], fold["test"]
        y_train = train[target].to_numpy()
        y_val = val[target].to_numpy()
        if y_train.sum() == 0 or (y_train == 0).sum() == 0:
            fold_rows.append(
                {
                    "fold": i,
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "skipped": True,
                    "skip_reason": "single-class train",
                    "promotion_pass": False,
                }
            )
            print(f"  Fold {i}: skipped (single-class train)")
            continue
        model = _fit_xgb_classifier(
            _feature_matrix(train),
            y_train,
            _feature_matrix(val),
            y_val,
            xgb_params,
        )
        best = _best_iteration(model)
        if booster_is_degenerate(best):
            fold_rows.append(
                {
                    "fold": i,
                    "test_start": pd.Timestamp(fold["test_start"]).strftime("%Y-%m-%d"),
                    "test_end": pd.Timestamp(fold["test_end"]).strftime("%Y-%m-%d"),
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "best_iteration": best,
                    "skipped": True,
                    "skip_reason": (
                        f"best_iteration={best} < {MIN_BEST_ITERATION}"
                    ),
                    "promotion_pass": False,
                }
            )
            print(
                f"  Fold {i}: skipped (best_iteration={best} < {MIN_BEST_ITERATION}, "
                f"n_train={len(train)})"
            )
            continue
        logistic = _fit_logistic_classifier(_feature_matrix(train), y_train)
        test_out = test.copy()
        test_out[spec["prob_col"]] = _predict_proba(model, _feature_matrix(test))
        test_out[spec["logistic_prob_col"]] = logistic.predict_proba(
            _feature_matrix(test)
        )[:, 1]
        fold_rng = np.random.default_rng(RANDOM_SEED + i)
        test_out["_Random_Rank"] = fold_rng.random(len(test_out))
        test_out[FOLD_COL] = i
        fold_cmp = compare_rankers(test_out, spec)
        score_decile = fold_cmp["models"]["score"]["selections"][PROMOTION_SELECTION]
        xgb_decile = fold_cmp["models"]["xgb"]["selections"][PROMOTION_SELECTION]
        # Folds and the pooled result share one gate so the pass rate means
        # exactly what the headline promotion decision means.
        fold_pass = fold_cmp["promote"]
        fold_rows.append(
            {
                "fold": i,
                "test_start": pd.Timestamp(fold["test_start"]).strftime("%Y-%m-%d"),
                "test_end": pd.Timestamp(fold["test_end"]).strftime("%Y-%m-%d"),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "best_iteration": best,
                "skipped": False,
                "class_balance": class_balance(train[target]),
                "classification": {
                    "xgb": classification_metrics(
                        test_out[target], test_out[spec["prob_col"]]
                    ),
                    "logistic": classification_metrics(
                        test_out[target],
                        test_out[spec["logistic_prob_col"]],
                    ),
                },
                "trading": fold_cmp["models"],
                "promotion": fold_cmp["promotion"],
                "promotion_pass": fold_pass,
            }
        )
        blocked = ",".join(fold_cmp["promotion"]["failed_metrics"]) or "-"
        print(
            f"  Fold {i}: test {fold_rows[-1]['test_start']}..{fold_rows[-1]['test_end']} "
            f"n={len(test)}  avg ML/Score={xgb_decile['avg_fwd']:+.4f}/{score_decile['avg_fwd']:+.4f}  "
            f"med={xgb_decile['med_fwd']:+.4f}/{score_decile['med_fwd']:+.4f}  "
            f"wr={xgb_decile['win_rate']:.3f}/{score_decile['win_rate']:.3f}  "
            f"pass={fold_pass} blocked_by={blocked}"
        )
        oos_parts.append(test_out)

    if not oos_parts:
        raise RuntimeError(
            f"{spec['key']}: no walk-forward OOS predictions "
            f"(every fold was skipped as single-class or degenerate)"
        )

    oos = pd.concat(oos_parts, ignore_index=True)
    pooled = compare_rankers(oos, spec)
    contributing = [row for row in fold_rows if not row.get("skipped")]
    fold_pass_rate = float(
        np.mean([row["promotion_pass"] for row in contributing])
    )
    pooled["fold_pass_rate"] = fold_pass_rate
    pooled["promote"] = bool(
        pooled["promote"] and fold_pass_rate >= PROMOTION_FOLD_PASS_RATE
    )
    pooled["classification"] = {
        "xgb": classification_metrics(oos[target], oos[spec["prob_col"]]),
        "logistic": classification_metrics(
            oos[target], oos[spec["logistic_prob_col"]]
        ),
    }
    score_decile = pooled["models"]["score"]["selections"][PROMOTION_SELECTION]
    xgb_decile = pooled["models"]["xgb"]["selections"][PROMOTION_SELECTION]
    population = pooled["models"]["xgb"]["selections"]["population"]
    print(
        f"Pooled OOS n={pooled['n_oos']}  base_hit={pooled['base_hit_rate']:.4f}  "
        f"fold_pass_rate={fold_pass_rate:.1%} promote={pooled['promote']}"
    )
    print_selection_comparison(pooled, spec)
    print(
        f"  Population baseline: avg={population['avg_fwd']:+.4f} "
        f"med={population['med_fwd']:+.4f} wr={population['win_rate']:.3f}"
    )
    blocked = ", ".join(pooled["promotion"]["failed_metrics"]) or "none"
    print(f"  Promotion blocked by: {blocked}")
    oos_path = write_oos_predictions(oos, spec, art_dir)
    research_results = evaluate_research_targets(
        labeled, folds, spec, xgb_params
    )
    target = label_col(spec)
    for research_name, result in research_results.items():
        print(
            f"  Research {research_name}: "
            f"ScoreTop10%={result['score_top_10pct']['avg_fwd']:.4f} "
            f"XGBTop10%={result['xgb_top_10pct']['avg_fwd']:.4f}"
        )

    # Final artifact: train on all rows before the last fold's test window (plus val).
    last = folds[-1]
    final_train = labeled[labeled[DATE_COL] < last["val_start"]]
    final_val = labeled[
        (labeled[DATE_COL] >= last["val_start"]) & (labeled[DATE_COL] < last["val_end"])
    ]
    if len(final_train) < MIN_TRAIN_ROWS or len(final_val) < MIN_VAL_ROWS:
        final_train = last["train"]
        final_val = last["val"]
    y_tr = final_train[target].to_numpy()
    y_va = final_val[target].to_numpy()
    final_model = _fit_xgb_classifier(
        _feature_matrix(final_train),
        y_tr,
        _feature_matrix(final_val),
        y_va,
        xgb_params,
    )
    best_iter = _best_iteration(final_model)

    art_dir.mkdir(parents=True, exist_ok=True)
    model_path = art_dir / spec["model_filename"]
    model_saved = save_xgb_artifact(final_model, model_path, best_iter)
    if model_saved:
        print(
            f"[SAVED] {spec['key']} XGBoost classifier: {model_path}  "
            f"best_iteration={best_iter}"
        )
    else:
        print(
            f"[SKIPPED] {spec['key']} artifact not saved: "
            f"best_iteration={best_iter} < {MIN_BEST_ITERATION}"
        )
        pooled["promote"] = False

    gain = normalized_xgb_gain(final_model)
    xgb_imp = np.asarray([gain[name] for name in FEATURE_COLS], dtype=float)
    print_ascii_importance(FEATURE_COLS, xgb_imp, f"XGBoost gain ({spec['key']})")

    production_model = "xgb" if pooled["promote"] else "none"
    source_hash = sha256_file(source_csv)
    feature_quality = analyze_feature_quality(labeled)
    metadata = {
        "schema_version": 5,
        "task": "binary",
        "model_type": "XGBClassifier",
        "horizon": spec["key"],
        "horizon_bars": spec["bars"],
        "target_column": target,
        "mfe_hit_column": spec["hit_col"],
        "forward_column": spec["forward_col"],
        "max_column": spec["max_col"],
        "win_column": spec["win_col"],
        "prob_column": spec["prob_col"],
        "hit_threshold": spec["threshold"],
        "embargo_weeks": spec["embargo_weeks"],
        "feature_columns": list(FEATURE_COLS),
        "feature_count": len(FEATURE_COLS),
        "feature_decisions": FEATURE_DECISIONS,
        "feature_quality": feature_quality,
        "feature_bucket_analysis": feature_bucket_analysis(labeled),
        "normalized_xgb_gain": gain,
        "best_iteration": best_iter,
        "min_best_iteration": MIN_BEST_ITERATION,
        "min_train_rows": MIN_TRAIN_ROWS,
        "model_saved": bool(model_saved),
        "sample_weight": "uniform",
        "scale_pos_weight": float((xgb_params or {}).get("scale_pos_weight", 1.0)),
        "promoted": bool(pooled["promote"]),
        "production_model": production_model,
        "promotion_rule": (
            "Promote only when the walk-forward OOS top-decile beats Score, a "
            "random ranker, and the untouched population on average forward "
            "return, median forward return, and win rate, and at least 60% of "
            "walk-forward folds pass the same test. MFE hit rate is excluded: "
            "it is a volatility-loaded research label, not a trading outcome."
        ),
        "promotion_selection": PROMOTION_SELECTION,
        "promotion_metrics": list(PROMOTION_METRICS),
        "promotion_checks": pooled["promotion"],
        "selection_scope": (
            "Fraction cuts (top_50pct .. top_5pct) are taken within each "
            "walk-forward fold and pooled afterwards; probabilities from "
            "different fold models are never ranked against each other. "
            "Per-date cuts are inherently single-fold."
        ),
        "source_csv": str(source_csv) if source_csv else None,
        "source_csv_sha256": source_hash,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": library_versions(),
        "training_rows": int(len(final_train)),
        "validation_rows": int(len(final_val)),
        "oos_rows": int(len(oos)),
        "class_balance": balance,
        "date_ranges": {
            "all": [
                labeled[DATE_COL].min().strftime("%Y-%m-%d"),
                labeled[DATE_COL].max().strftime("%Y-%m-%d"),
            ],
            "final_train": [
                final_train[DATE_COL].min().strftime("%Y-%m-%d"),
                final_train[DATE_COL].max().strftime("%Y-%m-%d"),
            ],
            "final_validation": [
                final_val[DATE_COL].min().strftime("%Y-%m-%d"),
                final_val[DATE_COL].max().strftime("%Y-%m-%d"),
            ],
        },
        "do_not_average_with_lightgbm": True,
        "artifacts": {
            "xgb_model": model_path.name if model_saved else None,
            "oos_predictions": oos_path.name,
        },
        "classification_metrics": pooled["classification"],
        "research_targets": research_results,
        "score_baseline": pooled["models"]["score"],
        "model_metrics": {
            "xgb": pooled["models"]["xgb"],
            "logistic": pooled["models"].get("logistic"),
            "random": pooled["models"].get("random"),
        },
        "walk_forward": {
            "n_folds": len(contributing),
            "n_skipped": int(sum(1 for row in fold_rows if row.get("skipped"))),
            "n_oos": pooled["n_oos"],
            "folds": fold_rows,
            "pooled": {
                "base_hit_rate": pooled["base_hit_rate"],
                "per_date_n": pooled["per_date_n"],
                "score_per_date_avg_return": pooled["score_per_date"]["avg_fwd"],
                "ml_per_date_avg_return": pooled["ml_per_date"]["avg_fwd"],
                "score_per_date_median_return": pooled["score_per_date"]["med_fwd"],
                "ml_per_date_median_return": pooled["ml_per_date"]["med_fwd"],
                "score_per_date_selected_fraction": pooled["score_per_date"]["selected_fraction"],
                "score_win_rate": score_decile["win_rate"],
                "ml_win_rate": xgb_decile["win_rate"],
                "population_win_rate": population["win_rate"],
                "score_hit_rate": score_decile["hit_rate"],
                "ml_hit_rate": xgb_decile["hit_rate"],
                "score_top10pct_avg": pooled["score_top10pct_avg"],
                "score_top5pct_avg": pooled["score_top5pct_avg"],
                "ml_top10pct_avg": pooled["ml_top10pct_avg"],
                "ml_top5pct_avg": pooled["ml_top5pct_avg"],
                "population_avg": population["avg_fwd"],
                "population_median": population["med_fwd"],
                "fold_pass_rate": fold_pass_rate,
                "models": pooled["models"],
                "promotion_checks": pooled["promotion"],
                "promote": pooled["promote"],
            },
        },
        "decision_guidance": {
            "use_as": "soft ranking signal for setup selection (only if promoted)",
            "combine_with": [
                "macro regime score",
                "risk management rules",
                "market context",
            ],
            "do_not_use_as": [
                "hard entry/exit gate",
                "position sizing rule",
                "blend of 10d/21d/42d models",
                "average of XGBoost and LightGBM",
            ],
        },
    }
    meta_path = art_dir / spec["metadata_filename"]
    meta_path.write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    print(f"[SAVED] {spec['key']} metadata: {meta_path}")
    pooled["metadata"] = metadata
    pooled["model_path"] = str(model_path) if model_saved else None
    pooled["model_saved"] = bool(model_saved)
    pooled["metadata_path"] = str(meta_path)
    pooled["folds"] = fold_rows
    return pooled


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def print_ascii_importance(names: list[str], importances: np.ndarray, title: str) -> None:
    order = np.argsort(importances)[::-1]
    max_imp = float(importances.max()) if len(importances) else 1.0
    max_imp = max_imp if max_imp > 0 else 1.0
    print(f"\n{title}")
    print("-" * 56)
    for idx in order:
        bar_len = int(30 * float(importances[idx]) / max_imp)
        bar = "#" * bar_len
        print(f"  {names[idx]:<24} {importances[idx]:8.4f}  {bar}")


def print_promotion_table(results: list[dict]) -> None:
    print(
        f"\n=== ML vs baselines (walk-forward OOS, {PROMOTION_SELECTION} within fold) ==="
    )
    hdr = (
        f"{'Horizon':<8}{'Target':<8}{'MLAvg':>9}{'ScoreAvg':>9}{'RandAvg':>9}{'PopAvg':>9}"
        f"{'MLMed':>9}{'ScoreMed':>9}{'MLWR':>8}{'ScoreWR':>8}{'PopWR':>8}"
        f"{'FoldPass':>10}{'Promote':>9}  {'Blocked by'}"
    )
    print(hdr)
    print("-" * (len(hdr) + 12))
    for r in results:
        score = r["models"]["score"]["selections"][PROMOTION_SELECTION]
        xgb = r["models"]["xgb"]["selections"][PROMOTION_SELECTION]
        pop = r["models"]["xgb"]["selections"]["population"]
        rand = r["models"].get("random", {}).get("selections", {}).get(
            PROMOTION_SELECTION, {}
        )
        blocked = ",".join(r["promotion"]["failed_metrics"]) or "-"
        if r["promotion"]["passed"] and not r["promote"]:
            blocked = f"fold_pass_rate({r.get('fold_pass_rate', float('nan')):.0%})"
        print(
            f"{r['horizon']:<8}{r['target']:<8}"
            f"{xgb['avg_fwd']:>+9.4f}{score['avg_fwd']:>+9.4f}"
            f"{rand.get('avg_fwd', float('nan')):>+9.4f}{pop['avg_fwd']:>+9.4f}"
            f"{xgb['med_fwd']:>+9.4f}{score['med_fwd']:>+9.4f}"
            f"{xgb['win_rate']:>8.3f}{score['win_rate']:>8.3f}{pop['win_rate']:>8.3f}"
            f"{r.get('fold_pass_rate', float('nan')):>10.1%}"
            f"{'YES' if r['promote'] else 'NO':>9}  {blocked}"
        )


def write_index_metadata(art_dir: Path, results: list[dict]) -> None:
    """Pointer file so inference can discover the three horizon artifacts."""
    payload = {
        "task": "binary_horizons",
        "do_not_average_with_lightgbm": True,
        "do_not_combine_horizons": True,
        "feature_columns": list(FEATURE_COLS),
        "horizons": [
            {
                "key": r["horizon"],
                "promoted": r["promote"],
                "production_model": "xgb" if r["promote"] else "none",
                "metadata_file": Path(r["metadata_path"]).name,
                "xgb_model": (
                    Path(r["model_path"]).name if r.get("model_path") else None
                ),
                "prob_column": next(
                    s["prob_col"] for s in HORIZON_SPECS if s["key"] == r["horizon"]
                ),
            }
            for r in results
        ],
    }
    path = art_dir / MODEL_METADATA_FILENAME
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"\n[SAVED] horizon index: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coiled Cobra ML hit classifiers (10d / 21d / 42d) with walk-forward"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"Path to trades CSV (default: search for {SOURCE_FILENAME})",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory for models/metadata (default: same directory as the CSV)",
    )
    parser.add_argument(
        "--horizons",
        default="10d,21d,42d",
        help="Comma-separated horizons to train (default: 10d,21d,42d)",
    )
    args = parser.parse_args(argv)

    csv_path = resolve_source_csv(args.csv)
    art_dir = Path(args.artifacts_dir) if args.artifacts_dir else (
        csv_path.parent if csv_path.parent.is_dir() else Path.cwd()
    )

    wanted = {h.strip() for h in args.horizons.split(",") if h.strip()}
    specs = [s for s in HORIZON_SPECS if s["key"] in wanted]
    if not specs:
        raise ValueError(f"No matching horizons in {wanted}")

    df = load_and_prepare(csv_path)
    if _infer_frame_mode(df) == "weekly":
        raise RuntimeError(
            "This trainer is for 10y daily (1D) hit models. Pass a daily backtest CSV."
        )
    report_all_target_balances(df)

    results = []
    for spec in specs:
        results.append(train_horizon(df, spec, art_dir, source_csv=csv_path))

    print_promotion_table(results)
    write_index_metadata(art_dir, results)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
