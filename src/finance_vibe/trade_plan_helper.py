# src/finance_vibe/trade_plan_helper.py
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd


def resolve_trade_plan_path(mode: str = "weekly", *, today: str | None = None) -> tuple[Path, Path]:
    """Locate today's trade plan CSV under data/logs/{mode}/ or legacy flat dirs."""
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

    raise FileNotFoundError(
        f"{filename} not found (mode={mode}); checked data/logs/{mode}/ and legacy data/logs/"
    )


def process_trade_plan(mode: str = "weekly", *, today: str | None = None) -> Path:
    """Load trade plan, compute R:R columns, write cleaned CSV. Returns output path."""
    today_str = today or datetime.now().strftime("%Y-%m-%d")
    trade_plan_dir, scanner_csv = resolve_trade_plan_path(mode, today=today_str)
    print(f"🎯 Target trade plan file located: {scanner_csv}")

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
        df["Risk Per Share"] = df["Stock Entry"] - df["Stock Stop"]
        df["Reward T1"] = df["Target 1"] - df["Stock Entry"]
        df["Reward T2"] = df["Target 2"] - df["Stock Entry"]

        safe_risk = df["Risk Per Share"].replace(0, pd.NA)
        df["R:R T1"] = (df["Reward T1"].astype(float) / safe_risk.astype(float)).round(2)
        df["R:R T2"] = (df["Reward T2"].astype(float) / safe_risk.astype(float)).round(2)

        df.drop(columns=["Reward T1", "Reward T2"], errors="ignore", inplace=True)
    except Exception:
        print("❌ Fatal exception caught inside metrics distribution generation engine:")
        traceback.print_exc()
        raise SystemExit(1) from None

    # Select only essential columns for the cleaned file
    essential_cols = [
        "Symbol",
        "Source",
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
    ]

    # Keep only columns that exist
    keep_cols = [c for c in essential_cols if c in df.columns]
    df_clean = df[keep_cols].copy()

    print("\n📄 Cleaned Trade Plan Preview:")
    print(df_clean.head(10).to_markdown(index=False))

    clean_csv = trade_plan_dir / f"trade_plan_clean_{today_str}.csv"
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
    if argv and argv[0].lower() in ("weekly", "daily"):
        mode = argv[0].lower()
    try:
        process_trade_plan(mode)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())