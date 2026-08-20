# src/finance_vibe/trade_plan_helper.py
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

# Ingestion guardrails: drop broken risk rows before ranking.
# Checklist / T1 R:R filters are not used — the scanner already applied hard gates.
MAX_RISK_PCT_OF_CLOSE = config.MAX_RISK_PCT_OF_CLOSE
# Boost only genuinely tight coils (width ≤ 4 ATR or risk ≤ 3%), not every Cobra row.
TIGHT_COIL_PROPENSITY = 1.25
TIGHT_COIL_WIDTH_ATR = 4.0
TIGHT_RISK_PCT = 0.03


def resolve_trade_plan_path(mode: str | None = None, *, today: str | None = None) -> tuple[Path, Path]:
    """Locate a trade plan CSV under data/logs/{mode}/ or legacy flat dirs.

    Prefers ``trade_plan_{today}.csv``; if that is missing, falls back to the
    latest dated ``trade_plan_<date>.csv`` (excluding the ``_clean`` variant) so
    the helper stays coupled to whatever date the planner actually produced.
    """
    mode = mode or config.DEFAULT_MODE
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


def _count_true(mask: pd.Series) -> int:
    """Count True values; safe on empty frames (pandas empty-string sum → '')."""
    if mask is None or len(mask) == 0:
        return 0
    return int(np.asarray(mask.fillna(False), dtype=bool).sum())


def apply_ingestion_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows whose risk exceeds 5% of close.

    Returns the filtered frame and a small rejection summary.
    """
    n0 = len(df)
    stats = {"input": n0, "risk_pct": 0, "kept": 0}

    out = df.copy()
    if out.empty:
        return out, stats

    price = pd.to_numeric(
        out["Close"] if "Close" in out.columns else out.get("Stock Entry"),
        errors="coerce",
    )
    if "Stock Entry" in out.columns and price is not None:
        entry = pd.to_numeric(out["Stock Entry"], errors="coerce")
        price = price.fillna(entry) if hasattr(price, "fillna") else entry

    risk = pd.to_numeric(out.get("Risk Per Share"), errors="coerce")
    if price is not None and risk is not None:
        risk_pct = risk / price.replace(0, np.nan)
        mask_risk = (risk_pct > MAX_RISK_PCT_OF_CLOSE).fillna(False)
        stats["risk_pct"] = _count_true(mask_risk)
        out = out.loc[~mask_risk].copy()

    stats["kept"] = len(out)
    return out, stats


def rank_by_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    """Rank survivors by ML predicted return, else Score, with a tight-coil boost.

    ``Expected Value = R:R T2 × Score`` is always computed for transparency.
    ``Priority`` is ``ML_Pred_Return × propensity`` when the ML column has any
    values (negatives sort below zero; missing predictions sort last). Otherwise
    ``Score × propensity``. Propensity is 1.25 only when ``Coil_Width_ATR`` ≤ 4
    or risk ≤ 3% of close.
    """
    out = df.copy()
    rr2 = pd.to_numeric(out.get("R:R T2"), errors="coerce").fillna(0.0)
    score = pd.to_numeric(out.get("Score"), errors="coerce").fillna(0.0)
    out["Expected Value"] = (rr2 * score).round(2)

    price = pd.to_numeric(
        out["Close"] if "Close" in out.columns else out.get("Stock Entry"),
        errors="coerce",
    )
    risk = pd.to_numeric(out.get("Risk Per Share"), errors="coerce")
    tight_risk = (risk / price.replace(0, np.nan)) <= TIGHT_RISK_PCT if price is not None else False
    if "Coil_Width_ATR" in out.columns:
        width = pd.to_numeric(out["Coil_Width_ATR"], errors="coerce")
        tight_width = width <= TIGHT_COIL_WIDTH_ATR
    else:
        tight_width = pd.Series(False, index=out.index)
    propensity = np.where(
        np.asarray(tight_width.fillna(False)) | np.asarray(
            tight_risk.fillna(False) if hasattr(tight_risk, "fillna") else tight_risk
        ),
        TIGHT_COIL_PROPENSITY,
        1.0,
    )

    if "ML_Pred_Return" in out.columns:
        ml_pred = pd.to_numeric(out["ML_Pred_Return"], errors="coerce")
    else:
        ml_pred = pd.Series(np.nan, index=out.index, dtype="float64")

    if ml_pred.notna().any():
        out["Priority"] = (ml_pred * propensity).round(4)
    else:
        out["Priority"] = (score * propensity).round(2)

    return out.sort_values(
        "Priority", ascending=False, na_position="last", kind="mergesort"
    ).reset_index(drop=True)


def process_trade_plan(mode: str | None = None, *, today: str | None = None) -> Path:
    """Load trade plan, compute R:R, apply guardrails, rank by EV. Returns output path."""
    mode = mode or config.DEFAULT_MODE
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

    print("🛡️ Applying ingestion guardrails (risk ≤5% of close)...")
    df, filter_stats = apply_ingestion_filters(df)
    print(
        f"   kept {filter_stats['kept']}/{filter_stats['input']} "
        f"(dropped risk={filter_stats['risk_pct']})"
    )

    if not df.empty:
        df = rank_by_expected_value(df)
        ml_active = "ML_Pred_Return" in df.columns and pd.to_numeric(
            df["ML_Pred_Return"], errors="coerce"
        ).notna().any()
        if ml_active:
            print("📊 Ranked survivors by ML predicted return (tight-coil boost when width ≤ 4 ATR or risk ≤ 3%).")
        else:
            print("📊 Ranked survivors by Score (tight-coil boost when width ≤ 4 ATR or risk ≤ 3%).")

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
        "Coil_High",
        "Coil_Low",
        "Coil_Width_ATR",
        "MACD_Spread_ATR",
        "Volume_Shelf",
        "MACD_Compression",
        "Structure",
        "RS_Score",
        "Coil_Width",
        "MACD_Cross",
        "Fib_Bonus",
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
        try:
            print(df_clean.head(10).to_markdown(index=False))
        except (ImportError, AttributeError):
            print(df_clean.head(10).to_string(index=False))

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
    mode = config.DEFAULT_MODE
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
