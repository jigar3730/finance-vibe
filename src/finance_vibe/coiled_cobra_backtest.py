"""Coiled Cobra historical backfill and expansion backtest.

Backfill archives every valid coil bar (v2.1 scorecard). Backtest measures
forward expansion vs QQQ — not Fib-bounce fills. ``--playbook fib`` is deferred.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd

from finance_vibe import config
from finance_vibe.market import load_benchmark_frame, load_ohlc_csv, ticker_from_filename
from finance_vibe import coiled_cobra as cobra
from finance_vibe.coiled_cobra import (
    BENCHMARK,
    add_macro_indicators,
    build_setup_row,
    configure_mode,
    evaluate_coiled_cobra,
)

def forward_label_specs(scan_mode: str | None = None) -> tuple[tuple[int, str], ...]:
    """(bar_offset, column_suffix) for forward / relative / MAE / held-low labels."""
    m = scan_mode or cobra.mode
    if m == "daily":
        return (
            (10, "2w"),
            (21, "21d"),
            (25, "5w"),
            (42, "42d"),
            (63, "13w"),
            (126, "26w"),
        )
    return (
        (2, "2w"),
        (4, "4w"),
        (5, "5w"),
        (8, "8w"),
        (13, "13w"),
        (26, "26w"),
    )


def forward_horizon_bars(scan_mode: str | None = None) -> tuple[int, ...]:
    """Bar counts for forward labels (daily adds 21d/42d; weekly adds 4w/8w)."""
    return tuple(bars for bars, _ in forward_label_specs(scan_mode))


_FIB_PLAYBOOK_NOTE = (
    "Fib-dip playbook is not active; --entry-valid / --max-hold are ignored. "
    "Backtest records forward expansion vs QQQ."
)


def _close_on_or_before(df: Optional[pd.DataFrame], as_of) -> Optional[float]:
    """Last Close on or before *as_of*, or None if unavailable."""
    if df is None or df.empty or as_of is None or "Date" not in df.columns:
        return None
    as_of_ts = pd.to_datetime(as_of)
    dates = pd.to_datetime(df["Date"])
    mask = dates <= as_of_ts
    if not mask.any():
        return None
    return float(df.loc[mask, "Close"].iloc[-1])


def _pct_change(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None or start == 0:
        return None
    return round((end - start) / start, 4)


def stamp_episode(
    row: Optional[dict],
    prev_valid: bool,
    prev_age: int,
) -> tuple[Optional[dict], bool, int]:
    """Attach Is_New_Coil / Coil_Age_Bars; reset age when the bar is not a coil."""
    if row is None:
        return None, False, 0
    if prev_valid:
        is_new, age = False, prev_age + 1
    else:
        is_new, age = True, 1
    stamped = dict(row)
    stamped["Is_New_Coil"] = is_new
    stamped["Coil_Age_Bars"] = age
    return stamped, True, age


def forward_return_at(
    df: pd.DataFrame,
    idx: int,
    horizon: int,
    setup_close: float,
) -> Optional[float]:
    future_idx = idx + horizon
    if future_idx >= len(df) or setup_close == 0:
        return None
    future_close = float(df.iloc[future_idx]["Close"])
    return round((future_close - setup_close) / setup_close, 4)


def rel_forward_at(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    idx: int,
    horizon: int,
    setup_close: float,
) -> Optional[float]:
    stock_fwd = forward_return_at(df, idx, horizon, setup_close)
    if stock_fwd is None or benchmark_df is None:
        return None
    sig_date = df.iloc[idx]["Date"]
    fut_date = df.iloc[idx + horizon]["Date"]
    bench_fwd = _pct_change(
        _close_on_or_before(benchmark_df, sig_date),
        _close_on_or_before(benchmark_df, fut_date),
    )
    if bench_fwd is None:
        return None
    return round(stock_fwd - bench_fwd, 4)


def mae_at(
    df: pd.DataFrame,
    idx: int,
    horizon: int,
    setup_close: float,
) -> Optional[float]:
    """Max adverse excursion: (close - min Low over the horizon) / close."""
    if setup_close == 0 or idx + 1 >= len(df):
        return None
    end = min(idx + horizon + 1, len(df))
    future = df.iloc[idx + 1 : end]
    if future.empty:
        return None
    low_water = float(pd.to_numeric(future["Low"], errors="coerce").min())
    if not pd.notna(low_water):
        return None
    return round((setup_close - low_water) / setup_close, 4)


def held_coil_low_at(
    df: pd.DataFrame,
    idx: int,
    horizon: int,
    coil_low,
) -> Optional[int]:
    """1 if every Close in the horizon stays at or above Coil_Low."""
    if coil_low is None or (isinstance(coil_low, float) and pd.isna(coil_low)):
        return None
    if idx + 1 >= len(df):
        return None
    end = min(idx + horizon + 1, len(df))
    future = df.iloc[idx + 1 : end]
    if future.empty:
        return None
    floor = float(coil_low)
    closes = pd.to_numeric(future["Close"], errors="coerce")
    if closes.isna().all():
        return None
    return int(bool((closes >= floor).all()))


def _horizon_fields(
    df: pd.DataFrame,
    idx: int,
    setup_close: float,
    benchmark_df: Optional[pd.DataFrame],
    coil_low=None,
) -> dict:
    fields: dict = {}
    for bars, suffix in forward_label_specs():
        fields[f"Forward_Return_{suffix}"] = forward_return_at(
            df, idx, bars, setup_close
        )
        fields[f"Rel_Forward_{suffix}"] = rel_forward_at(
            df, benchmark_df, idx, bars, setup_close
        )
        fields[f"MAE_{suffix}"] = mae_at(df, idx, bars, setup_close)
        fields[f"Held_Coil_Low_{suffix}"] = held_coil_low_at(
            df, idx, bars, coil_low
        )
    return fields


def detect_cobra_setup_at_bar(
    df: pd.DataFrame,
    symbol: str,
    benchmark_df=None,
) -> Optional[dict]:
    """Evaluate the latest bar in a history window for a Coiled Cobra coil setup."""
    if len(df) < cobra.LOOKBACK // 2 + 15:
        return None

    window = df.copy()
    try:
        window = add_macro_indicators(window)
    except Exception:
        return None

    if len(window) < 2:
        return None

    setup = evaluate_coiled_cobra(window, benchmark_df)
    if not setup:
        return None

    return build_setup_row(symbol, window, setup, cobra.mode)


def generate_backfill(mode: str | None = None, tickers: Optional[str] = None) -> pd.DataFrame:
    """Scan historical raw CSVs to produce a Coiled Cobra signal archive."""
    mode = mode or config.DEFAULT_MODE
    configure_mode(mode)
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
                new_n = sum(1 for r in rows if r.get("Is_New_Coil"))
                print(f"{symbol}: {len(rows)} setup(s), {new_n} new coil(s)")

    # Reassemble in sorted path order so CSV row sequence matches single-threaded runs.
    all_rows: list[dict] = []
    for path in work_paths:
        all_rows.extend(results_by_path.get(path, []))

    out_df = pd.DataFrame(all_rows)
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(logs_dir, f"coiled_cobra_backfill_{stamp}.csv")
    os.makedirs(logs_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved Coiled Cobra backfill archive: {out_path}")
    return out_df


def backtest_ticker(
    path: str,
    entry_valid: int = 0,
    max_hold: int = 0,
    benchmark_df=None,
) -> tuple[list[dict], dict]:
    """Walk-forward expansion study for one ticker (no Fib fills)."""
    del entry_valid, max_hold  # reserved for a future --playbook fib
    symbol = ticker_from_filename(path)
    df = load_ohlc_csv(path)
    trades: list[dict] = []
    counts = {
        "signals": 0,
        "new_coils": 0,
        "errors": 0,
    }

    min_bars = cobra.LOOKBACK // 2 + 15
    prev_valid = False
    prev_age = 0
    for idx in range(min_bars, len(df)):
        window = df.iloc[: idx + 1]
        try:
            setup_row = detect_cobra_setup_at_bar(
                window, symbol, benchmark_df=benchmark_df
            )
        except Exception:
            counts["errors"] += 1
            prev_valid, prev_age = False, 0
            continue

        stamped, prev_valid, prev_age = stamp_episode(setup_row, prev_valid, prev_age)
        if not stamped:
            continue

        counts["signals"] += 1
        if stamped["Is_New_Coil"]:
            counts["new_coils"] += 1

        signal_date = df.iloc[idx]["Date"]
        setup_close = float(stamped["Close"])
        row = {
            **stamped,
            "Signal Date": signal_date.strftime("%Y-%m-%d")
            if hasattr(signal_date, "strftime")
            else signal_date,
            **_horizon_fields(
                df,
                idx,
                setup_close,
                benchmark_df,
                coil_low=stamped.get("Coil_Low"),
            ),
        }
        trades.append(row)

    return trades, counts


# Per-process cache: loaded once in the worker initializer, never pickled from parent.
_WORKER_BENCHMARK_DF = None


def _init_cobra_worker(mode: str) -> None:
    """Configure weekly/daily globals and load benchmark OHLC once per worker."""
    global _WORKER_BENCHMARK_DF
    configure_mode(mode)
    _WORKER_BENCHMARK_DF = load_benchmark_frame(BENCHMARK, mode)


def _backfill_ticker_worker(path: str) -> tuple[str, list[dict]]:
    """Process-pool worker: load ticker OHLC locally and collect Coiled Cobra setups."""
    symbol = ticker_from_filename(path)
    try:
        df = load_ohlc_csv(path)
    except Exception as exc:
        print(f"{symbol}: error — {exc}", file=sys.stderr)
        return symbol, []

    rows: list[dict] = []
    min_bars = cobra.LOOKBACK // 2 + 15
    prev_valid = False
    prev_age = 0
    try:
        for idx in range(min_bars, len(df)):
            setup = detect_cobra_setup_at_bar(
                df.iloc[: idx + 1], symbol, benchmark_df=_WORKER_BENCHMARK_DF
            )
            stamped, prev_valid, prev_age = stamp_episode(setup, prev_valid, prev_age)
            if stamped:
                rows.append(stamped)
    except Exception as exc:
        print(f"{symbol}: error — {exc}", file=sys.stderr)
        return symbol, []

    return symbol, rows


def _backtest_ticker_worker(
    path: str,
    entry_valid: int,
    max_hold: int,
) -> tuple[str, list[dict], dict]:
    """Process-pool worker: load ticker OHLC locally and run expansion backtest."""
    symbol = ticker_from_filename(path)
    empty_counts: dict = {}
    try:
        trades, counts = backtest_ticker(
            path,
            entry_valid,
            max_hold,
            benchmark_df=_WORKER_BENCHMARK_DF,
        )
        return symbol, trades, counts
    except Exception as exc:
        print(f"{symbol}: error — {exc}", file=sys.stderr)
        return symbol, [], empty_counts


def _print_expansion_summary(trades_df: pd.DataFrame) -> None:
    if trades_df.empty or "Is_New_Coil" not in trades_df.columns:
        return
    new = trades_df[trades_df["Is_New_Coil"] == True]  # noqa: E712
    print(f"New coils: {len(new)} / {len(trades_df)} signal bars")
    if new.empty:
        return
    for grade in ("A - Coil Ready", "B - Valid Coil"):
        g = new[new["Grade"] == grade]
        if g.empty:
            continue
        m2 = pd.to_numeric(g.get("Forward_Return_2w"), errors="coerce").median()
        r_med = pd.to_numeric(
            g.get("Rel_Forward_42d", g.get("Rel_Forward_13w")), errors="coerce"
        ).median()
        print(f"  {grade}: n={len(g)} median Fwd_2w={m2} median Rel_med={r_med}")


def run_backtest(
    mode: str | None = None,
    tickers: Optional[str] = None,
    entry_valid: int = config.BACKTEST_ENTRY_VALID_BARS,
    max_hold: int = config.BACKTEST_MAX_HOLD_BARS,
) -> pd.DataFrame:
    """Run Coiled Cobra expansion backtest across raw CSV files using all CPU cores."""
    mode = mode or config.DEFAULT_MODE
    configure_mode(mode)
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

    print(f"--- Coiled Cobra Expansion Backtest [{mode.upper()}] ---")
    print(f"Raw dir: {raw_dir}")
    print(f"Workers: {max_workers}  Tickers: {len(work_paths)}")
    print(_FIB_PLAYBOOK_NOTE + "\n")

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
                new_n = counts.get("new_coils", 0)
                print(f"{symbol}: {len(trades)} signal(s), {new_n} new coil(s)")

    # Reassemble in sorted path order so CSV row sequence matches single-threaded runs.
    for path in work_paths:
        trades, counts = results_by_path.get(path, ([], {}))
        all_trades.extend(trades)
        for key, value in counts.items():
            all_counts[key] = all_counts.get(key, 0) + value

    trades_df = pd.DataFrame(all_trades)
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(logs_dir, f"coiled_cobra_backtest_trades_{stamp}.csv")
    os.makedirs(logs_dir, exist_ok=True)
    trades_df.to_csv(out_path, index=False)
    print(f"Saved backtest output: {out_path}")
    if all_counts:
        print(
            f"Totals: signals={all_counts.get('signals', 0)} "
            f"new_coils={all_counts.get('new_coils', 0)} "
            f"errors={all_counts.get('errors', 0)}"
        )
    _print_expansion_summary(trades_df)

    return trades_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coiled Cobra historical backfill and expansion backtest"
    )
    parser.add_argument("mode", nargs="?", default=config.DEFAULT_MODE, choices=list(config.TIMEFRAME_PROFILES))
    parser.add_argument("--tickers", help="Comma-separated tickers to include")
    parser.add_argument("--backfill", action="store_true", help="Export historical Coiled Cobra signal archive")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run walk-forward expansion study (forward returns vs QQQ)",
    )
    parser.add_argument(
        "--entry-valid",
        type=int,
        default=config.BACKTEST_ENTRY_VALID_BARS,
        help="Unused (reserved for a future Fib playbook)",
    )
    parser.add_argument(
        "--max-hold",
        type=int,
        default=config.BACKTEST_MAX_HOLD_BARS,
        help="Unused (reserved for a future Fib playbook)",
    )
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
