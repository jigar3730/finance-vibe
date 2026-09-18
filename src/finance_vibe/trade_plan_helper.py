"""Ingest a trade plan CSV, apply guardrails, and rank survivors by expected value."""
from __future__ import annotations

import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.errors

try:
    from finance_vibe import config
    from finance_vibe.coiled_cobra import MIN_CHECKS_MET, N_SCORED_PILLARS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from finance_vibe import config
    from finance_vibe.coiled_cobra import MIN_CHECKS_MET, N_SCORED_PILLARS

# Ingestion guardrails (Part 3): drop broken / unprofitable rows before ranking.
MAX_RISK_PCT_OF_CLOSE = config.MAX_RISK_PCT_OF_CLOSE
MIN_RR_T1 = 2.0
# Mirrors coiled_cobra's own Gate D breadth threshold (MIN_CHECKS_MET of
# N_SCORED_PILLARS) instead of a hardcoded ratio, so this guardrail can't
# silently drift out of sync the way the old fixed 5/7 (calibrated for the
# retired v3.1 7-pillar checklist) did once v4.0's 6-pillar Gate D was
# empirically recalibrated to 4/6 -- that stale ratio was re-rejecting
# scanner-admitted 4/6 rows before they ever reached the trade plan. See
# docs/handbook/coiled_cobra_rubric.md's Gate D recalibration note.
MIN_CHECKLIST_RATIO = MIN_CHECKS_MET / N_SCORED_PILLARS
# Static propensity boost for tight coils until adaptive CDH weighting exists.
TIGHT_COIL_PROPENSITY = 1.25
TIGHT_RISK_PCT = 0.03

CLEAN_EXPORT_COLUMNS = [
    "Symbol",
    "Setup Type",
    "Source",
    "Mode",
    "AsOf Date",
    "Score",
    "Grade",
    "Tier",
    "Checks Met",
    "RVOL",
    "Market Gate",
    "Close",
    "Stock Entry",
    "Stock Stop",
    "Target 1",
    "Target 2",
    "Risk Per Share",
    "R:R T1",
    "R:R T2",
    "ML_Pred_Return",
    "ML_Rank",
    "Expected Value",
    "Priority",
    "ATR",
    "RSI",
    "Fib 78.6%",
]


def resolve_trade_plan_path(mode: str = "weekly", *, today: str | None = None) -> tuple[Path, Path]:
    """Locate a trade plan CSV under data/logs/{mode}/ or legacy flat dirs.

    Prefers ``trade_plan_{today}.csv`` (the planner always writes that stamp).
    If it is missing — e.g. a helper-only rerun — fall back to the latest dated
    ``trade_plan_<date>.csv`` (excluding the ``_clean`` variant).
    """
    today_str = today or datetime.now().strftime("%Y-%m-%d")
    filename = f"trade_plan_{today_str}.csv"
    base_dir = Path(__file__).resolve().parents[2]

    possible_dirs = [
        base_dir / "data" / "logs" / mode,
        Path(f"./data/logs/{mode}"),
        Path("/app/data/logs") / mode,
        Path("data/logs") / mode,
        base_dir / "data" / "logs",
        Path("./data/logs"),
        Path("/app/data/logs"),
        Path("data/logs"),
    ]

    for p_dir in possible_dirs:
        check_path = p_dir / filename
        if check_path.exists():
            return p_dir, check_path

    # Fallback: newest dated trade plan in the first directory that has one.
    for p_dir in possible_dirs:
        if not p_dir.exists():
            continue
        candidates = sorted(
            (f for f in p_dir.glob("trade_plan_*.csv") if "clean" not in f.stem),
            key=lambda f: f.stem.split("_")[-1],
            reverse=True,
        )
        if candidates:
            return p_dir, candidates[0]

    raise FileNotFoundError(
        f"{filename} not found (mode={mode}); checked data/logs/{mode}/ and legacy data/logs/"
    )


def _checklist_fully_passed(value: Any) -> bool:
    """True when Checks Met is missing (swing) or meets ``MIN_CHECKLIST_RATIO``
    (mirrors coiled_cobra's Gate D breadth threshold, currently
    ``MIN_CHECKS_MET``/``N_SCORED_PILLARS``)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", ""}:
        return True
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if not match:
        return True
    passed, total = int(match.group(1)), int(match.group(2))
    if total <= 0:
        return True
    return (passed / total) >= MIN_CHECKLIST_RATIO - 1e-12


def _count_true(mask: pd.Series) -> int:
    """Count True values; safe on empty frames (pandas empty-string sum → '')."""
    if mask is None or len(mask) == 0:
        return 0
    return int(np.asarray(mask.fillna(False), dtype=bool).sum())


def _apply_ingestion_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows that fail risk, checklist, or T1 R:R guardrails.

    Returns the filtered frame and a small rejection summary.
    """
    n0 = len(df)
    stats = {"input": n0, "risk_pct": 0, "checklist": 0, "rr_t1": 0}

    out = df.copy()
    if out.empty:
        stats["kept"] = 0
        return out, stats

    price = pd.to_numeric(
        out["Close"] if "Close" in out.columns else out.get("Stock Entry"),
        errors="coerce",
    )
    if "Stock Entry" in out.columns and price is not None:
        # Prefer Close; fall back to entry when Close is absent/NaN.
        entry = pd.to_numeric(out["Stock Entry"], errors="coerce")
        price = price.fillna(entry) if hasattr(price, "fillna") else entry

    risk = pd.to_numeric(out.get("Risk Per Share"), errors="coerce")
    if price is not None and risk is not None:
        risk_pct = risk / price.replace(0, np.nan)
        mask_risk = (risk_pct > MAX_RISK_PCT_OF_CLOSE).fillna(False)
        stats["risk_pct"] = _count_true(mask_risk)
        out = out.loc[~mask_risk].copy()

    if not out.empty and "Checks Met" in out.columns:
        passed = out["Checks Met"].map(_checklist_fully_passed).astype(bool)
        stats["checklist"] = _count_true(~passed)
        out = out.loc[passed].copy()

    if not out.empty and "R:R T1" in out.columns:
        rr1 = pd.to_numeric(out["R:R T1"], errors="coerce")
        mask_rr = (rr1 < MIN_RR_T1).fillna(True)
        stats["rr_t1"] = _count_true(mask_rr)
        out = out.loc[~mask_rr].copy()

    stats["kept"] = len(out)
    return out, stats


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric view of ``name``; an all-NaN Series when the column is absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def ml_priority_active(df: pd.DataFrame) -> bool:
    """True when ``ML_Pred_Return`` should drive ``Priority``.

    Requires ``config.ML_RANKING_ENABLED`` AND a prediction on every row.
    ML values (~0.05) and Score-based EV (~300) are on different scales, so a
    partially covered frame would push every unpredicted row -- possibly the
    highest-Score setup, e.g. one with a NaN feature -- below all predicted
    ones. Incomplete coverage therefore falls back to Score for the whole frame.
    """
    if not config.ML_RANKING_ENABLED or df.empty or "ML_Pred_Return" not in df.columns:
        return False
    return bool(_num_col(df, "ML_Pred_Return").notna().all())


def rank_by_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    """Rank survivors by expected value with a tight-coil propensity boost.

    ``Expected Value = R:R T2 × Score`` is always computed for transparency.
    ``Priority`` is ``Expected Value × propensity`` (raw-Score ranking, since
    R:R T2 is 3.0 for every Coiled Cobra plan) unless :func:`ml_priority_active`
    -- ML enabled and every row predicted -- in which case it is
    ``R:R T2 × max(ML_Pred_Return, 0) × propensity``. Ties (e.g. all-negative
    predictions clipped to 0) are broken by Expected Value, i.e. by Score.
    """
    out = df.copy()
    rr2 = _num_col(out, "R:R T2").fillna(0.0)
    score = _num_col(out, "Score").fillna(0.0)
    out["Expected Value"] = (rr2 * score).round(2)

    if "Source" in out.columns:
        source = out["Source"].astype(str).str.strip().str.lower()
    else:
        source = pd.Series("", index=out.index)
    price = _num_col(out, "Close" if "Close" in out.columns else "Stock Entry")
    risk = _num_col(out, "Risk Per Share")
    tight_risk = (risk / price.replace(0, np.nan)) <= TIGHT_RISK_PCT   # NaN -> False
    is_coil = source.isin(["coiled_cobra", "cobra"])
    propensity = np.where(is_coil | tight_risk, TIGHT_COIL_PROPENSITY, 1.0)

    if ml_priority_active(out):
        # ML-driven expected value: reward per unit risk scaled by predicted alpha.
        ml_pred = _num_col(out, "ML_Pred_Return")
        out["Priority"] = (rr2 * ml_pred.clip(lower=0) * propensity).round(4)
    else:
        out["Priority"] = (out["Expected Value"] * propensity).round(2)

    return out.sort_values(
        ["Priority", "Expected Value"], ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def process_trade_plan(mode: str = "weekly", *, today: str | None = None) -> Path:
    """Load trade plan, compute R:R, apply guardrails, rank by EV. Returns output path."""
    today_str = today or datetime.now().strftime("%Y-%m-%d")
    trade_plan_dir, scanner_csv = resolve_trade_plan_path(mode, today=today_str)
    print(f"🎯 Target trade plan file located: {scanner_csv}")

    # Couple the cleaned-file date to the plan we actually resolved (may be a
    # fallback older than "today").
    resolved_date = scanner_csv.stem.split("_")[-1]

    clean_csv = trade_plan_dir / f"trade_plan_clean_{resolved_date}.csv"

    def _finish_empty() -> Path:
        print("⚠️ Trade plan file is empty. Skipping processing cleanly.")
        pd.DataFrame(columns=CLEAN_EXPORT_COLUMNS).to_csv(clean_csv, index=False)
        print(f"✅ Cleaned trade plan saved: {clean_csv}")
        return clean_csv

    if scanner_csv.exists() and scanner_csv.stat().st_size == 0:
        return _finish_empty()

    try:
        df = pd.read_csv(scanner_csv)
    except pandas.errors.EmptyDataError:
        return _finish_empty()
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        raise SystemExit(1) from e

    if df.empty:
        return _finish_empty()

    df.columns = df.columns.str.strip()
    print("✅ Loaded CSV columns:", df.columns.tolist())

    # Ensure numeric columns are clean
    numeric_cols = ["Stock Entry", "Stock Stop", "Target 1", "Target 2", "Close", "Score"]
    for col in [c for c in numeric_cols if c in df.columns]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r"[$,]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("🧮 Calculating Risk-to-Reward distributions...")
    try:
        # Direction-aware: reward is measured toward the trade's target side and
        # risk is always the absolute entry-to-stop distance.
        if "Setup Type" in df.columns:
            is_long = df["Setup Type"].astype(str).str.upper() != "SETUP_SHORT"
        else:
            is_long = pd.Series(True, index=df.index)

        df["Risk Per Share"] = (df["Stock Entry"] - df["Stock Stop"]).abs()
        reward_t1 = np.where(
            is_long, df["Target 1"] - df["Stock Entry"], df["Stock Entry"] - df["Target 1"]
        )
        reward_t2 = np.where(
            is_long, df["Target 2"] - df["Stock Entry"], df["Stock Entry"] - df["Target 2"]
        )

        safe_risk = df["Risk Per Share"].replace(0, pd.NA)
        df["R:R T1"] = (pd.Series(reward_t1, index=df.index, dtype="float") / safe_risk.astype(float)).round(2)
        df["R:R T2"] = (pd.Series(reward_t2, index=df.index, dtype="float") / safe_risk.astype(float)).round(2)
    except Exception:
        print("❌ Fatal exception caught inside metrics distribution generation engine:")
        traceback.print_exc()
        raise SystemExit(1) from None

    print(
        f"🛡️ Applying ingestion guardrails (risk ≤5%, "
        f"checklist ≥{MIN_CHECKS_MET}/{N_SCORED_PILLARS}, R:R T1 ≥ 2)..."
    )
    df, filter_stats = _apply_ingestion_filters(df)
    print(
        f"   kept {filter_stats['kept']}/{filter_stats['input']} "
        f"(dropped risk={filter_stats['risk_pct']}, "
        f"checklist={filter_stats['checklist']}, rr_t1={filter_stats['rr_t1']})"
    )

    if not df.empty:
        df = rank_by_expected_value(df)
        if ml_priority_active(df):
            print("📊 Ranked survivors by ML predicted return × R:R T2 with coil propensity.")
        else:
            print("📊 Ranked survivors by Expected Value (R:R T2 × Score) with coil propensity.")
            n_pred = int(_num_col(df, "ML_Pred_Return").notna().sum())
            if n_pred:  # silent when the ML column is simply empty (current state)
                reason = (
                    "ML ranking is disabled (config.ML_RANKING_ENABLED)"
                    if not config.ML_RANKING_ENABLED
                    else f"predictions are incomplete ({n_pred}/{len(df)} rows)"
                )
                print(f"   ℹ️ Ignoring {n_pred} ML prediction(s): {reason}.")

    # Select essential columns for the cleaned file (only those that exist).
    # Both LEAPS/Options label variants are listed so mode-specific columns
    # survive the filter.
    # Keep only columns that exist
    keep_cols = [c for c in CLEAN_EXPORT_COLUMNS if c in df.columns]
    df_clean = df[keep_cols].copy()

    print("\n📄 Cleaned Trade Plan Preview:")
    if df_clean.empty:
        print("(no setups survived ingestion guardrails)")
    else:
        print(df_clean.head(10).to_markdown(index=False))

    clean_csv = trade_plan_dir / f"trade_plan_clean_{resolved_date}.csv"
    try:
        df_clean.to_csv(clean_csv, index=False)
        print(f"\n✅ Cleaned trade plan saved: {clean_csv}")
    except Exception as save_err:
        print(f"❌ Error saving cleaned file: {save_err}")
        raise SystemExit(1) from save_err

    return clean_csv


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    mode = "weekly"
    if argv and argv[0].lower() in ("weekly", "daily", "high_beta"):
        mode = argv[0].lower()
    try:
        process_trade_plan(mode)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
