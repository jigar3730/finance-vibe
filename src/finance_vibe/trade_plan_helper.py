# src/finance_vibe/trade_plan_helper.py
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from finance_vibe import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from finance_vibe import config

# Ingestion guardrails (Part 3): drop broken / unprofitable rows before ranking.
MAX_RISK_PCT_OF_CLOSE = config.MAX_RISK_PCT_OF_CLOSE
MIN_RR_T1 = 2.0
# Allow one missed soft pillar (5/6); hard gates already ran in the scanner.
MIN_CHECKLIST_RATIO = 5 / 6
# Static propensity boost for tight coils until adaptive CDH weighting exists.
TIGHT_COIL_PROPENSITY = 1.25
TIGHT_RISK_PCT = 0.03


def resolve_trade_plan_path(mode: str = "weekly", *, today: str | None = None) -> tuple[Path, Path]:
    """Locate a trade plan CSV under data/logs/{mode}/ or legacy flat dirs.

    Prefers ``trade_plan_{today}.csv``; if that is missing, falls back to the
    latest dated ``trade_plan_<date>.csv`` (excluding the ``_clean`` variant) so
    the helper stays coupled to whatever date the planner actually produced.
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


def _checklist_fully_passed(value) -> bool:
    """True when Checks Met is missing (swing) or meets the soft baseline (≥5/6)."""
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


def apply_ingestion_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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


def rank_by_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    """Rank survivors by expected value with a tight-coil propensity boost.

    ``Expected Value = R:R T2 × Score`` is always computed for transparency.
    When ``ML_Pred_Return`` is present (offline model ran), ``Priority`` is
    driven by ``R:R T2 × max(ML_Pred_Return, 0) × propensity`` so setups with
    stronger predicted forward alpha rank first. When the ML column is absent or
    all-null, ``Priority`` falls back to ``Expected Value × propensity``.
    """
    out = df.copy()
    rr2 = pd.to_numeric(out.get("R:R T2"), errors="coerce").fillna(0.0)
    score = pd.to_numeric(out.get("Score"), errors="coerce").fillna(0.0)
    out["Expected Value"] = (rr2 * score).round(2)

    source = out["Source"].astype(str).str.strip().str.lower() if "Source" in out.columns else ""
    price = pd.to_numeric(
        out["Close"] if "Close" in out.columns else out.get("Stock Entry"),
        errors="coerce",
    )
    risk = pd.to_numeric(out.get("Risk Per Share"), errors="coerce")
    tight_risk = (risk / price.replace(0, np.nan)) <= TIGHT_RISK_PCT if price is not None else False
    is_coil = source.isin(["coiled_cobra", "cobra"]) if hasattr(source, "isin") else False
    propensity = np.where(
        np.asarray(is_coil) | np.asarray(tight_risk.fillna(False) if hasattr(tight_risk, "fillna") else tight_risk),
        TIGHT_COIL_PROPENSITY,
        1.0,
    )

    if "ML_Pred_Return" in out.columns:
        ml_pred = pd.to_numeric(out["ML_Pred_Return"], errors="coerce")
    else:
        ml_pred = pd.Series(np.nan, index=out.index, dtype="float64")

    if ml_pred.notna().any():
        # ML-driven expected value: reward per unit risk scaled by predicted alpha.
        ml_ev = rr2 * ml_pred.clip(lower=0).fillna(0.0)
        out["Priority"] = (ml_ev * propensity).round(4)
    else:
        out["Priority"] = (out["Expected Value"] * propensity).round(2)

    return out.sort_values("Priority", ascending=False, kind="mergesort").reset_index(drop=True)


def process_trade_plan(mode: str = "weekly", *, today: str | None = None) -> Path:
    """Load trade plan, compute R:R, apply guardrails, rank by EV. Returns output path."""
    today_str = today or datetime.now().strftime("%Y-%m-%d")
    trade_plan_dir, scanner_csv = resolve_trade_plan_path(mode, today=today_str)
    print(f"🎯 Target trade plan file located: {scanner_csv}")

    # Couple the cleaned-file date to the plan we actually resolved (may be a
    # fallback older than "today").
    resolved_date = scanner_csv.stem.split("_")[-1]

    try:
        df = pd.read_csv(scanner_csv)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        raise SystemExit(1) from e

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

    print("🛡️ Applying ingestion guardrails (risk ≤5%, checklist ≥5/6, R:R T1 ≥ 2)...")
    df, filter_stats = apply_ingestion_filters(df)
    print(
        f"   kept {filter_stats['kept']}/{filter_stats['input']} "
        f"(dropped risk={filter_stats['risk_pct']}, "
        f"checklist={filter_stats['checklist']}, rr_t1={filter_stats['rr_t1']})"
    )

    if not df.empty:
        df = rank_by_expected_value(df)
        ml_active = "ML_Pred_Return" in df.columns and pd.to_numeric(
            df["ML_Pred_Return"], errors="coerce"
        ).notna().any()
        if ml_active:
            print("📊 Ranked survivors by ML predicted return × R:R T2 with coil propensity.")
        else:
            print("📊 Ranked survivors by Expected Value (R:R T2 × Score) with coil propensity.")

    # Select essential columns for the cleaned file (only those that exist).
    # Both LEAPS/Options label variants are listed so mode-specific columns
    # survive the filter.
    essential_cols = [
        "Symbol",
        "Setup Type",
        "Source",
        "Mode",           # swing profile (weekly/daily/high_beta)
        "AsOf Date",
        "Score",          # raw scanner score
        "Grade",          # e.g. "A - Institutional Setup"
        "Checks Met",     # e.g. "4/5"
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
        "LEAPS Type",
        "Options Type",
        "Delta Min",
        "Delta Max",
        "LEAPS Expiry Min",
        "LEAPS Expiry Max",
        "Options Expiry Min",
        "Options Expiry Max",
    ]

    # Keep only columns that exist
    keep_cols = [c for c in essential_cols if c in df.columns]
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
