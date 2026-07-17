# src/finance_vibe/trade_plan_helper.py
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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


def process_trade_plan(mode: str = "weekly", *, today: str | None = None) -> Path:
    """Load trade plan, compute R:R columns, write cleaned CSV. Returns output path."""
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
    numeric_cols = ["Stock Entry", "Stock Stop", "Target 1", "Target 2"]
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
        "Stock Entry",
        "Stock Stop",
        "Target 1",
        "Target 2",
        "Risk Per Share",
        "R:R T1",
        "R:R T2",
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