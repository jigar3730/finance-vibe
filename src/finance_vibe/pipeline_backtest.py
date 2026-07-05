"""Walk-forward backtest of the Finance Vibe pipeline output path.

Replays swing scanner setup detection and trade-planner stock levels on historical
OHLC bars, gated by macro Vibe Score thresholds on the signal bar. Stock simulation
only (no options P&L).

Usage:
    python src/finance_vibe/pipeline_backtest.py weekly
    python src/finance_vibe/pipeline_backtest.py weekly --tickers SPY,QQQ
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    from finance_vibe import config
    from finance_vibe.analysis_engine import build_features, load_ohlc_csv, score_last_row, ticker_from_filename
    from finance_vibe.swing_scanner import detect_setup_at_bar
    from finance_vibe.trade_planner import calculate_stock_levels
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from finance_vibe import config
    from finance_vibe.analysis_engine import build_features, load_ohlc_csv, score_last_row, ticker_from_filename
    from finance_vibe.swing_scanner import detect_setup_at_bar
    from finance_vibe.trade_planner import calculate_stock_levels


def passes_macro_gate(setup_type: str, score: int, long_min: int, short_max: int) -> bool:
    """Return True when macro Vibe Score confirms the tactical setup direction."""
    if setup_type == "SETUP_LONG":
        return score >= long_min
    if setup_type == "SETUP_SHORT":
        return score <= short_max
    return False


def simulate_trade(
    df: pd.DataFrame,
    start_idx: int,
    is_long: bool,
    entry: float,
    stop: float,
    target1: float,
    target2: float,
    entry_valid_bars: int,
    max_hold_bars: int,
) -> tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]]:
    """Forward-simulate entry fill, stop, and targets on High/Low bars after *start_idx*."""
    risk = abs(entry - stop)
    if risk <= 0:
        return "no_fill", None, None, None

    filled = False
    fill_idx = None

    for j in range(start_idx, min(start_idx + entry_valid_bars, len(df))):
        bar = df.iloc[j]
        if is_long and bar["Low"] <= entry:
            filled = True
            fill_idx = j
            break
        if not is_long and bar["High"] >= entry:
            filled = True
            fill_idx = j
            break

    if not filled or fill_idx is None:
        return "no_fill", None, None, None

    for k in range(fill_idx, min(fill_idx + max_hold_bars, len(df))):
        bar = df.iloc[k]
        if is_long:
            if bar["Low"] <= stop:
                return "stopped", bar["Date"], stop, -1.0
            if bar["High"] >= target2:
                return "target2", bar["Date"], target2, (target2 - entry) / risk
            if bar["High"] >= target1:
                return "target1", bar["Date"], target1, (target1 - entry) / risk
        else:
            if bar["High"] >= stop:
                return "stopped", bar["Date"], stop, -1.0
            if bar["Low"] <= target2:
                return "target2", bar["Date"], target2, (entry - target2) / risk
            if bar["Low"] <= target1:
                return "target1", bar["Date"], target1, (entry - target1) / risk

    last_idx = min(fill_idx + max_hold_bars - 1, len(df) - 1)
    exit_price = float(df.iloc[last_idx]["Close"])
    if is_long:
        r_mult = (exit_price - entry) / risk
    else:
        r_mult = (entry - exit_price) / risk
    return "expired", df.iloc[last_idx]["Date"], exit_price, r_mult


def backtest_ticker(
    path: str,
    mode: str,
    long_min: int,
    short_max: int,
    warmup: int,
    entry_valid: int,
    max_hold: int,
) -> tuple[list[dict], dict]:
    """Walk-forward backtest for one ticker CSV. Returns trade rows and counters."""
    symbol = ticker_from_filename(path)
    df = load_ohlc_csv(path)

    if "High" not in df.columns or "Low" not in df.columns:
        return [], {"skipped": 1}

    trades: list[dict] = []
    counts = {
        "signals": 0,
        "macro_pass": 0,
        "filled": 0,
        "no_fill": 0,
        "stopped": 0,
        "target1": 0,
        "target2": 0,
        "expired": 0,
    }

    for i in range(warmup, len(df) - 1):
        window = df.iloc[: i + 1].copy()
        setup_row = detect_setup_at_bar(window, symbol, mode)
        if not setup_row:
            continue

        counts["signals"] += 1

        feat = build_features(window)
        score = score_last_row(feat.iloc[-1])
        if not passes_macro_gate(setup_row["Setup Type"], score, long_min, short_max):
            continue

        counts["macro_pass"] += 1

        entry, stop, t1, t2, _, _ = calculate_stock_levels(setup_row)
        is_long = setup_row["Setup Type"] == "SETUP_LONG"

        outcome, exit_date, exit_price, r_mult = simulate_trade(
            df,
            i + 1,
            is_long,
            entry,
            stop,
            t1,
            t2,
            entry_valid,
            max_hold,
        )

        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome != "no_fill":
            counts["filled"] += 1

        signal_date = df.iloc[i]["Date"]
        trades.append({
            "Symbol": symbol,
            "Signal Date": signal_date.strftime("%Y-%m-%d") if hasattr(signal_date, "strftime") else signal_date,
            "Setup Type": setup_row["Setup Type"],
            "Vibe Score": score,
            "Stock Entry": round(entry, 2),
            "Stock Stop": round(stop, 2),
            "Target 1": round(t1, 2),
            "Target 2": round(t2, 2),
            "Outcome": outcome,
            "Exit Date": exit_date.strftime("%Y-%m-%d") if exit_date is not None and hasattr(exit_date, "strftime") else exit_date,
            "Exit Price": round(exit_price, 2) if exit_price is not None else None,
            "R Multiple": round(r_mult, 2) if r_mult is not None else None,
        })

    return trades, counts


def load_ticker_filter(mode_cfg: dict, tickers_arg: Optional[str]) -> Optional[set[str]]:
    """Resolve optional --tickers list; otherwise use active_tickers.csv if present."""
    if tickers_arg:
        return {t.strip().upper() for t in tickers_arg.split(",") if t.strip()}

    if os.path.exists(config.TICKER_LIST_PATH):
        return set(pd.read_csv(config.TICKER_LIST_PATH)["Ticker"].str.upper())

    return None


def print_summary(all_counts: dict, trades_df: pd.DataFrame) -> None:
    """Print aggregate backtest statistics to stdout."""
    print("\n" + "=" * 60)
    print("PIPELINE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Signals detected:     {all_counts.get('signals', 0)}")
    print(f"Passed macro gate:    {all_counts.get('macro_pass', 0)}")
    print(f"Entries filled:       {all_counts.get('filled', 0)}")
    print(f"  No fill:            {all_counts.get('no_fill', 0)}")
    print(f"  Stopped:            {all_counts.get('stopped', 0)}")
    print(f"  Target 1:           {all_counts.get('target1', 0)}")
    print(f"  Target 2:           {all_counts.get('target2', 0)}")
    print(f"  Expired (max hold): {all_counts.get('expired', 0)}")

    filled = trades_df[trades_df["Outcome"] != "no_fill"] if not trades_df.empty else pd.DataFrame()
    if filled.empty:
        print("\nNo filled trades to analyze.")
        return

    for setup in ("SETUP_LONG", "SETUP_SHORT"):
        subset = filled[filled["Setup Type"] == setup]
        if subset.empty:
            continue
        print(f"\n{setup}:")
        print(f"  Trades: {len(subset)}")
        print(f"  Avg R:  {subset['R Multiple'].mean():.2f}")
        wins_t1 = subset["Outcome"].isin(["target1", "target2"]).mean() * 100
        print(f"  Win rate (T1+): {wins_t1:.1f}%")


def run_backtest(
    mode: str = "weekly",
    tickers: Optional[str] = None,
    long_min: int = config.BACKTEST_LONG_MIN_SCORE,
    short_max: int = config.BACKTEST_SHORT_MAX_SCORE,
    warmup: int = config.BACKTEST_WARMUP_BARS,
    entry_valid: int = config.BACKTEST_ENTRY_VALID_BARS,
    max_hold: int = config.BACKTEST_MAX_HOLD_BARS,
) -> pd.DataFrame:
    """Run walk-forward backtest across raw CSV files for *mode*."""
    mode_cfg = config.get_mode_config(mode)
    raw_dir = mode_cfg["raw_dir"]
    logs_dir = mode_cfg["logs_dir"]
    ticker_filter = load_ticker_filter(mode_cfg, tickers)

    if not os.path.isdir(raw_dir):
        print(f"No raw directory: {raw_dir}")
        return pd.DataFrame()

    paths = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    )

    all_trades: list[dict] = []
    all_counts: dict = {}

    print(f"--- Pipeline Backtest [{mode.upper()}] ---")
    print(f"Macro gate: LONG score >= {long_min}, SHORT score <= {short_max}")
    print(f"Raw dir: {raw_dir}\n")

    for path in paths:
        symbol = ticker_from_filename(path)
        if ticker_filter and symbol not in ticker_filter:
            continue

        trades, counts = backtest_ticker(
            path, mode, long_min, short_max, warmup, entry_valid, max_hold
        )
        all_trades.extend(trades)
        for k, v in counts.items():
            all_counts[k] = all_counts.get(k, 0) + v
        if trades:
            print(f"{symbol}: {len(trades)} signal(s), {sum(1 for t in trades if t['Outcome'] != 'no_fill')} filled")

    trades_df = pd.DataFrame(all_trades)
    print_summary(all_counts, trades_df)

    if not trades_df.empty:
        stamp = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(logs_dir, f"backtest_trades_{stamp}.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return trades_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward pipeline output backtest")
    parser.add_argument("mode", nargs="?", default=config.DEFAULT_MODE, choices=list(config.TIMEFRAME_PROFILES))
    parser.add_argument("--tickers", help="Comma-separated tickers (default: active_tickers.csv)")
    parser.add_argument("--long-min-score", type=int, default=config.BACKTEST_LONG_MIN_SCORE)
    parser.add_argument("--short-max-score", type=int, default=config.BACKTEST_SHORT_MAX_SCORE)
    args = parser.parse_args(argv)

    run_backtest(
        mode=args.mode,
        tickers=args.tickers,
        long_min=args.long_min_score,
        short_max=args.short_max_score,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
