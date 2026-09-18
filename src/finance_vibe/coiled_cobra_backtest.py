"""Coiled Cobra historical backfill and walk-forward backtest.

This module evaluates the Coiled Cobra strategy across historical raw OHLC data,
exports signal archives, and simulates stock trade outcomes using the existing
trade-planner stock level calculator.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    from finance_vibe import config
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.analysis_engine import load_benchmark_frame, load_ohlc_csv, ticker_from_filename
    from finance_vibe.coiled_cobra import (
        BENCHMARK,
        SPY_BENCHMARK,
        add_macro_indicators,
        evaluate_coiled_cobra,
        local_swing_low,
    )
    from finance_vibe.trade_simulator import simulate_trade
    from finance_vibe.trade_planner import calculate_stock_levels
except ImportError:  # pragma: no cover
    # Package lives under src/; repo root alone is not enough for `finance_vibe`.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe import coiled_cobra as cc
    from finance_vibe.analysis_engine import load_benchmark_frame, load_ohlc_csv, ticker_from_filename
    from finance_vibe.coiled_cobra import (
        BENCHMARK,
        SPY_BENCHMARK,
        add_macro_indicators,
        evaluate_coiled_cobra,
        local_swing_low,
    )
    from finance_vibe.trade_simulator import simulate_trade
    from finance_vibe.trade_planner import calculate_stock_levels


def detect_cobra_setup_at_bar(
    df: pd.DataFrame,
    symbol: str,
    benchmark_df=None,
    spy_df=None,
) -> Optional[dict]:
    """Evaluate the latest bar in a history window for a Coiled Cobra coil setup."""
    if len(df) < cc.LOOKBACK // 2 + 15:
        return None

    window = df.copy()
    try:
        window = add_macro_indicators(window)
    except Exception:
        return None

    if len(window) < 2:
        return None

    setup = evaluate_coiled_cobra(
        window, benchmark_df, spy_df=spy_df, qqq_df=benchmark_df
    )
    if not setup:
        return None

    latest = window.iloc[-1]
    close = float(latest["Close"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    atr = float(latest["ATR"])
    fib_618 = float(latest["Fib_618"]) if pd.notna(latest.get("Fib_618")) else None
    fib_786 = float(latest["Fib_786"]) if pd.notna(latest.get("Fib_786")) else None
    rvol_raw = latest.get("RVOL")
    rvol_v = setup.get("RVOL")
    if rvol_v is None and rvol_raw is not None and pd.notna(rvol_raw):
        rvol_v = round(float(rvol_raw), 4)

    return {
        "Symbol": symbol.upper(),
        "Date": latest["Date"],
        "Setup Type": "SETUP_LONG",
        "Close": round(close, 2),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "ATR": round(atr, 2),
        "Swing Low": round(local_swing_low(window), 2),
        "Fib 61.8%": round(fib_618, 2) if fib_618 is not None else None,
        "Fib 78.6%": round(fib_786, 2) if fib_786 is not None else None,
        "Pct_From_EMA20": round((close - ema20) / ema20, 4),
        "Pct_From_EMA50": round((close - ema50) / ema50, 4),
        "Pct_From_Fib618": round((close - fib_618) / fib_618, 4) if fib_618 is not None else None,
        "Pct_From_Fib786": round((close - fib_786) / fib_786, 4) if fib_786 is not None else None,
        "ATR_Pct": round(atr / close, 4),
        "Score": setup["Score"],
        "Grade": setup["Grade"],
        "Tier": setup.get("Tier"),
        "Checks Met": setup["Checks Met"],
        "Source": "coiled_cobra",
        "RS 63d": setup.get("RS 63d"),
        "RVOL": rvol_v,
        "Market Gate": setup.get("Market Gate"),
        "BBWidth Pctile": setup.get("BBWidth Pctile"),
        # Per-pillar sub-scores (the composite Score hides these); research
        # features only -- they are not in the deployed FEATURE_COLS.
        **{f"Part_{k}": v for k, v in (setup.get("Parts") or {}).items()},
    }


def generate_backfill(mode: str = "weekly", tickers: Optional[str] = None) -> pd.DataFrame:
    """Scan historical raw CSVs to produce a Coiled Cobra signal archive."""
    # Calibrate coiled_cobra's mode-derived globals from the explicit `mode`
    # param rather than relying on sys.argv[1] matching this script's own
    # positional CLI arg (fragile if these functions are ever called
    # programmatically with a different mode).
    cc.apply_timeframe(mode)
    mode_cfg = config.get_mode_config(mode)
    raw_dir = mode_cfg["raw_dir"]
    logs_dir = mode_cfg["logs_dir"]
    ticker_filter = None
    if tickers:
        ticker_filter = {t.strip().upper() for t in tickers.split(",") if t.strip()}

    file_paths = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    )
    work_paths = [
        path
        for path in file_paths
        if not ticker_filter or ticker_from_filename(path) in ticker_filter
    ]

    max_workers = os.cpu_count() or 1
    print(f"--- Coiled Cobra Backfill [{mode.upper()}] ---")
    print(f"Raw dir: {raw_dir}")
    print(f"Workers: {max_workers}  Tickers: {len(work_paths)}\n")

    results_by_path: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_cobra_worker,
        initargs=(mode,),
    ) as ex:
        futures = {ex.submit(_backfill_ticker_worker, path): path for path in work_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            symbol = ticker_from_filename(path)
            try:
                symbol, rows = fut.result()
            except Exception as exc:
                print(f"{symbol}: error — {exc}", file=sys.stderr)
                rows = []
            results_by_path[path] = rows
            if rows:
                print(f"{symbol}: {len(rows)} setup(s)")

    # Reassemble in sorted path order so CSV row sequence matches single-threaded runs.
    all_rows: list[dict] = []
    for path in work_paths:
        all_rows.extend(results_by_path.get(path, []))

    out_df = pd.DataFrame(all_rows)
    out_df[config.RUBRIC_VERSION_COL] = config.RUBRIC_VERSION
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(logs_dir, f"coiled_cobra_backfill_{stamp}.csv")
    os.makedirs(logs_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved Coiled Cobra backfill archive: {out_path}")
    return out_df


def _checks_n(checks_met) -> Optional[int]:
    """'5/6' -> 5 (None when missing/malformed)."""
    try:
        return int(str(checks_met).split("/")[0])
    except (TypeError, ValueError):
        return None


def _benchmark_context(benchmark_df) -> Optional[pd.DataFrame]:
    """Date-indexed benchmark Close plus causal regime columns, or None.

    ``Pct_From_EMA50`` and ``Ret_13w`` use only data up to each row's date, so
    they are safe as features. The forward benchmark return used for the
    excess-return *label* is computed separately from later rows.
    """
    if benchmark_df is None or len(benchmark_df) == 0 or "Close" not in benchmark_df.columns:
        return None
    b = benchmark_df.copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values("Date").drop_duplicates("Date").set_index("Date")
    close = b["Close"].astype(float)
    ctx = pd.DataFrame({"Close": close})
    ctx["Pct_From_EMA50"] = (close / b["EMA50"].astype(float) - 1.0) if "EMA50" in b.columns else float("nan")
    ctx["Ret_13w"] = close.pct_change(13)
    return ctx


def backtest_ticker(
    path: str,
    entry_valid: int,
    max_hold: int,
    benchmark_df=None,
    spy_df=None,
) -> tuple[list[dict], dict]:
    """Walk-forward simulate one ticker using Coiled Cobra signals."""
    symbol = ticker_from_filename(path)
    df = load_ohlc_csv(path)
    trades: list[dict] = []
    counts = {
        "signals": 0,
        "filled": 0,
        "no_fill": 0,
        "stopped": 0,
        "target1": 0,
        "target2": 0,
        "expired": 0,
        "errors": 0,
    }

    bench_ctx = _benchmark_context(benchmark_df)

    min_bars = cc.LOOKBACK // 2 + 15
    for idx in range(min_bars, len(df) - 1):
        window = df.iloc[: idx + 1]
        setup_row = detect_cobra_setup_at_bar(
            window, symbol, benchmark_df=benchmark_df, spy_df=spy_df
        )
        if not setup_row:
            continue

        counts["signals"] += 1
        try:
            entry, stop, t1, t2, _, _ = calculate_stock_levels(setup_row)
            outcome, exit_date, exit_price, r_mult = simulate_trade(
                df,
                idx + 1,
                is_long=True,
                entry=entry,
                stop=stop,
                target1=t1,
                target2=t2,
                entry_valid_bars=entry_valid,
                max_hold_bars=max_hold,
            )
        except Exception:
            counts["errors"] += 1
            continue

        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome != "no_fill":
            counts["filled"] += 1

        signal_date = df.iloc[idx]["Date"]
        setup_close = float(setup_row["Close"])
        n_bars = len(df)

        def _forward_return(horizon: int) -> float | None:
            future_idx = idx + horizon
            if future_idx >= n_bars:
                return None
            future_close = float(df.iloc[future_idx]["Close"])
            return round((future_close - setup_close) / setup_close, 4)

        def _bench(col: str, date) -> Optional[float]:
            if bench_ctx is None or date is None:
                return None
            ts = pd.Timestamp(date)
            if ts not in bench_ctx.index:
                return None
            val = bench_ctx.at[ts, col]
            return None if pd.isna(val) else float(val)

        forward_return_2w = _forward_return(2)
        excess_return_2w = None
        if forward_return_2w is not None and idx + 2 < n_bars:
            b0, b2 = _bench("Close", signal_date), _bench("Close", df.iloc[idx + 2]["Date"])
            if b0 and b2:
                excess_return_2w = round(forward_return_2w - (b2 / b0 - 1.0), 4)
        forward_return_5w = _forward_return(5)
        forward_return_13w = _forward_return(13)
        forward_return_26w = _forward_return(26)

        r_multiple = round(r_mult, 2) if r_mult is not None else None
        if outcome == "no_fill":
            target_label = None
        elif outcome in ("target1", "target2"):
            target_label = 1
        else:
            target_label = 0

        trades.append(
            {
                "Symbol": symbol,
                "Signal Date": signal_date.strftime("%Y-%m-%d")
                if hasattr(signal_date, "strftime")
                else signal_date,
                "Setup Type": setup_row["Setup Type"],
                "Score": setup_row["Score"],
                "Grade": setup_row["Grade"],
                "Pct_From_EMA20": setup_row["Pct_From_EMA20"],
                "Pct_From_EMA50": setup_row["Pct_From_EMA50"],
                "Pct_From_Fib618": setup_row["Pct_From_Fib618"],
                "Pct_From_Fib786": setup_row["Pct_From_Fib786"],
                "ATR_Pct": setup_row["ATR_Pct"],
                "RVOL": setup_row.get("RVOL"),
                "Market Gate": setup_row.get("Market Gate"),
                "Tier": setup_row.get("Tier"),
                "Checks_N": _checks_n(setup_row.get("Checks Met")),
                "RS_63d": setup_row.get("RS 63d"),
                "BBWidth_Pctile": setup_row.get("BBWidth Pctile"),
                **{k: v for k, v in setup_row.items() if k.startswith("Part_")},
                # Causal benchmark regime at the signal bar (candidate features).
                "QQQ_Pct_From_EMA50": _bench("Pct_From_EMA50", signal_date),
                "QQQ_Ret_13w": _bench("Ret_13w", signal_date),
                "Stock Entry": round(entry, 2),
                "Stock Stop": round(stop, 2),
                "Target 1": round(t1, 2),
                "Target 2": round(t2, 2),
                "Outcome": outcome,
                "Exit Date": exit_date.strftime("%Y-%m-%d")
                if exit_date is not None and hasattr(exit_date, "strftime")
                else exit_date,
                "Exit Price": round(exit_price, 2) if exit_price is not None else None,
                "R Multiple": r_multiple,
                "Target_Label": target_label,
                "Target_R_Mult": r_multiple,
                "Forward_Return_2w": forward_return_2w,
                "Excess_Return_2w": excess_return_2w,
                "Forward_Return_5w": forward_return_5w,
                "Forward_Return_13w": forward_return_13w,
                "Forward_Return_26w": forward_return_26w,
            }
        )

    return trades, counts


# Per-process cache: loaded once in the worker initializer, never pickled from parent.
_WORKER_BENCHMARK_DF = None
_WORKER_SPY_DF = None


def _init_cobra_worker(mode: str) -> None:
    """Load benchmark OHLC once per worker process.

    Also explicitly (re-)calibrates coiled_cobra's mode-derived globals in
    this worker process -- required under the ``spawn`` start method, where
    the worker re-imports the module fresh instead of inheriting the parent's
    already-calibrated state via fork's copy-on-write memory.
    """
    global _WORKER_BENCHMARK_DF, _WORKER_SPY_DF
    cc.apply_timeframe(mode)
    _WORKER_BENCHMARK_DF = load_benchmark_frame(BENCHMARK, mode)
    _WORKER_SPY_DF = load_benchmark_frame(SPY_BENCHMARK, mode)


def _backfill_ticker_worker(path: str) -> tuple[str, list[dict]]:
    """Process-pool worker: load ticker OHLC locally and collect Coiled Cobra setups."""
    symbol = ticker_from_filename(path)
    try:
        df = load_ohlc_csv(path)
    except Exception as exc:
        print(f"{symbol}: error — {exc}", file=sys.stderr)
        return symbol, []

    rows: list[dict] = []
    min_bars = cc.LOOKBACK // 2 + 15
    try:
        for idx in range(min_bars, len(df)):
            setup = detect_cobra_setup_at_bar(
                df.iloc[: idx + 1],
                symbol,
                benchmark_df=_WORKER_BENCHMARK_DF,
                spy_df=_WORKER_SPY_DF,
            )
            if setup:
                rows.append(setup)
    except Exception as exc:
        print(f"{symbol}: error — {exc}", file=sys.stderr)
        return symbol, []

    return symbol, rows


def _backtest_ticker_worker(
    path: str,
    entry_valid: int,
    max_hold: int,
) -> tuple[str, list[dict], dict]:
    """Process-pool worker: load ticker OHLC locally and run walk-forward backtest."""
    symbol = ticker_from_filename(path)
    empty_counts: dict = {}
    try:
        trades, counts = backtest_ticker(
            path,
            entry_valid,
            max_hold,
            benchmark_df=_WORKER_BENCHMARK_DF,
            spy_df=_WORKER_SPY_DF,
        )
        return symbol, trades, counts
    except Exception as exc:
        print(f"{symbol}: error — {exc}", file=sys.stderr)
        return symbol, [], empty_counts


def run_backtest(
    mode: str = "weekly",
    tickers: Optional[str] = None,
    entry_valid: int = config.BACKTEST_ENTRY_VALID_BARS,
    max_hold: int = config.BACKTEST_MAX_HOLD_BARS,
) -> pd.DataFrame:
    """Run Coiled Cobra backtest across raw CSV files using all CPU cores."""
    cc.apply_timeframe(mode)
    mode_cfg = config.get_mode_config(mode)
    raw_dir = mode_cfg["raw_dir"]
    logs_dir = mode_cfg["logs_dir"]
    ticker_filter = None
    if tickers:
        ticker_filter = {t.strip().upper() for t in tickers.split(",") if t.strip()}

    paths = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    )
    work_paths = [
        path
        for path in paths
        if not ticker_filter or ticker_from_filename(path) in ticker_filter
    ]

    max_workers = os.cpu_count() or 1
    all_trades: list[dict] = []
    all_counts: dict = {}

    print(f"--- Coiled Cobra Backtest [{mode.upper()}] ---")
    print(f"Raw dir: {raw_dir}")
    print(f"Workers: {max_workers}  Tickers: {len(work_paths)}\n")

    results_by_path: dict[str, tuple[list[dict], dict]] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_cobra_worker,
        initargs=(mode,),
    ) as ex:
        futures = {
            ex.submit(_backtest_ticker_worker, path, entry_valid, max_hold): path
            for path in work_paths
        }
        for fut in as_completed(futures):
            path = futures[fut]
            symbol = ticker_from_filename(path)
            try:
                symbol, trades, counts = fut.result()
            except Exception as exc:
                print(f"{symbol}: error — {exc}", file=sys.stderr)
                trades, counts = [], {}
            results_by_path[path] = (trades, counts)
            if trades:
                filled = sum(1 for t in trades if t["Outcome"] != "no_fill")
                print(f"{symbol}: {len(trades)} signal(s), {filled} filled")

    # Reassemble in sorted path order so CSV row sequence matches single-threaded runs.
    for path in work_paths:
        trades, counts = results_by_path.get(path, ([], {}))
        all_trades.extend(trades)
        for key, value in counts.items():
            all_counts[key] = all_counts.get(key, 0) + value

    trades_df = pd.DataFrame(all_trades)
    trades_df[config.RUBRIC_VERSION_COL] = config.RUBRIC_VERSION
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(logs_dir, f"coiled_cobra_backtest_trades_{stamp}.csv")
    os.makedirs(logs_dir, exist_ok=True)
    trades_df.to_csv(out_path, index=False)
    print(f"Saved backtest output: {out_path}")

    return trades_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coiled Cobra historical backfill and backtest")
    parser.add_argument("mode", nargs="?", default=config.DEFAULT_MODE, choices=list(config.TIMEFRAME_PROFILES))
    parser.add_argument("--tickers", help="Comma-separated tickers to include")
    parser.add_argument("--backfill", action="store_true", help="Export historical Coiled Cobra signal archive")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward Coiled Cobra backtest")
    parser.add_argument("--entry-valid", type=int, default=config.BACKTEST_ENTRY_VALID_BARS)
    parser.add_argument("--max-hold", type=int, default=config.BACKTEST_MAX_HOLD_BARS)
    args = parser.parse_args(argv)

    if not args.backfill and not args.backtest:
        args.backtest = True

    if args.backfill:
        generate_backfill(args.mode, args.tickers)

    if args.backtest:
        run_backtest(args.mode, args.tickers, args.entry_valid, args.max_hold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
