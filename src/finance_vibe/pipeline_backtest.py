"""Walk-forward backtest of the Finance Vibe pipeline output path.

Replays swing scanner setup detection and trade-planner stock levels on historical
OHLC bars, gated by macro Vibe Score thresholds on the signal bar. Stock simulation
only (no options P&L).

Updated Strategy Parameters:
    - Target: Full exit at 2.0R (No partial scaling)
    - Stop Loss: Dynamic ATR trailing stop (1.5x - 2.0x ATR below high-water mark)
    - Capital Base: Position risk sized relative to account equity / fixed R risk

Usage:
    python src/finance_vibe/pipeline_backtest.py daily
    python src/finance_vibe/pipeline_backtest.py daily --no-partials --trailing-atr-mult 2.0
    python src/finance_vibe/pipeline_backtest.py high_beta --tickers PLTR,TSLA,HOOD --trailing-atr-mult 1.5
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
    from finance_vibe.analysis_engine import (
        build_features,
        load_benchmark_frame,
        load_ohlc_csv,
        score_last_row,
        ticker_from_filename,
    )
    from finance_vibe.swing_scanner import detect_setup_at_bar
    from finance_vibe.trade_planner import calculate_stock_levels
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe.analysis_engine import (
        build_features,
        load_benchmark_frame,
        load_ohlc_csv,
        score_last_row,
        ticker_from_filename,
    )
    from finance_vibe.swing_scanner import detect_setup_at_bar
    from finance_vibe.trade_planner import calculate_stock_levels


def passes_macro_gate(
    setup_type: str, score: int, long_min: int, short_max: int
) -> bool:
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


# =====================================================================
# UPDATED SIMULATOR (No Partials, Full 2.0R Target, High-Water ATR Trail)
# =====================================================================


def _fmt_date(value):
    return (
        value.strftime("%Y-%m-%d")
        if value is not None and hasattr(value, "strftime")
        else value
    )


def _stop_exit_price(
    is_long: bool, stop: float, open_px: Optional[float], slippage_pct: float
) -> float:
    """Market-stop exit price, worsened by gaps through the stop and slippage."""
    px = stop
    if open_px is not None:
        if is_long and open_px < stop:
            px = open_px  # gapped below the stop -> worse fill
        elif not is_long and open_px > stop:
            px = open_px
    return px * (1 - slippage_pct) if is_long else px * (1 + slippage_pct)


def _target_exit_price(is_long: bool, target: float, open_px: Optional[float]) -> float:
    """Limit-target exit price (no slippage); gaps beyond the target fill better."""
    px = target
    if open_px is not None:
        if is_long and open_px > target:
            px = open_px
        elif not is_long and open_px < target:
            px = open_px
    return px


def simulate_scaled_trade(
    df: pd.DataFrame,
    start_idx: int,
    is_long: bool,
    entry: float,
    stop: float,
    target1: float,
    target2: float,
    entry_valid_bars: int,
    max_hold_bars: int,
    *,
    slippage_pct: float = 0.0,
    partial_fraction: float = 0.0,  # 0.0 = No partials (Full exit at target_r)
    target_r: float = 2.0,  # Full exit target at 2.0R
    trailing_atr_mult: Optional[
        float
    ] = 2.0,  # 2.0 ATR trailing stop below current bar high
) -> dict:
    """Simulates trade execution with options for full exit (no partials) and high-water mark ATR trailing stops.

    Model:
      1. Fill limit order on entry pullback (slippage aware).
      2. If partial_fraction == 0.0: Trade exits entirely at target_r (default 2.0R) or dynamic ATR trailing stop.
      3. Trailing Stop: Continuously ratchets up behind the highest high (for long) or lowest low (for short)
         by `trailing_atr_mult * risk`.
      4. If partial_fraction > 0.0: Scales out partial_fraction at target1 (1.0R), moves stop to BE, runs remainder to 2.0R.
    """
    risk = abs(entry - stop)
    result = {
        "outcome": "no_fill",
        "fill_index": None,
        "exit_index": None,
        "fill_date": None,
        "fill_price": None,
        "gap_entry": False,
        "stop_moved_be": False,
        "partial_date": None,
        "partial_price": None,
        "partial_r": None,
        "runner_date": None,
        "runner_price": None,
        "runner_r": None,
        "blended_r": None,
        "bars_held": 0,
        "mae_r": None,
        "mfe_r": None,
        "risk": risk,
    }
    if risk <= 0:
        return result

    has_open = "Open" in df.columns
    n = len(df)

    # --- Entry fill check ---
    fill_idx = None
    fill_price = None
    gap_entry = False
    for j in range(start_idx, min(start_idx + entry_valid_bars, n)):
        bar = df.iloc[j]
        o = float(bar["Open"]) if has_open and pd.notna(bar["Open"]) else None
        if is_long and float(bar["Low"]) <= entry:
            if o is not None and o <= entry:
                fill_price, gap_entry = o, True
            else:
                fill_price = entry
            fill_idx = j
            break
        if not is_long and float(bar["High"]) >= entry:
            if o is not None and o >= entry:
                fill_price, gap_entry = o, True
            else:
                fill_price = entry
            fill_idx = j
            break

    if fill_idx is None:
        return result

    fill_price = (
        fill_price * (1 + slippage_pct) if is_long else fill_price * (1 - slippage_pct)
    )

    def r_of(price: float) -> float:
        return (price - fill_price) / risk if is_long else (fill_price - price) / risk

    # Effective stop initialized to setup stop loss
    current_stop = stop
    highest_high = fill_price
    lowest_low = fill_price

    # Target calculation for full exit model
    full_target_price = (
        entry + (target_r * risk) if is_long else entry - (target_r * risk)
    )

    partialed = False
    partial_r = partial_price = partial_date = None
    runner_r = runner_price = runner_date = None
    full_r = None
    outcome = None
    exit_index = None
    mae_r = 0.0
    mfe_r = 0.0
    bars_held = 0

    end = min(fill_idx + max_hold_bars, n)
    for k in range(fill_idx, end):
        bar = df.iloc[k]
        hi, lo = float(bar["High"]), float(bar["Low"])
        o = float(bar["Open"]) if has_open and pd.notna(bar["Open"]) else None
        date = bar["Date"] if "Date" in df.columns else k
        bars_held = k - fill_idx + 1

        if is_long:
            mfe_r = max(mfe_r, (hi - fill_price) / risk)
            mae_r = min(mae_r, (lo - fill_price) / risk)
            highest_high = max(highest_high, hi)

            # Dynamic high-water mark trailing stop (1.5x - 2.0x ATR/Risk)
            if trailing_atr_mult is not None and trailing_atr_mult > 0:
                trail_stop_lvl = highest_high - (trailing_atr_mult * risk)
                current_stop = max(current_stop, trail_stop_lvl)
        else:
            mfe_r = max(mfe_r, (fill_price - lo) / risk)
            mae_r = min(mae_r, (fill_price - hi) / risk)
            lowest_low = min(lowest_low, lo)

            # Dynamic low-water mark trailing stop for short positions
            if trailing_atr_mult is not None and trailing_atr_mult > 0:
                trail_stop_lvl = lowest_low + (trailing_atr_mult * risk)
                current_stop = min(current_stop, trail_stop_lvl)

        # ---------------------------------------------------------------
        # NO PARTIALS MODE (100% position exit at target_r or ATR stop)
        # ---------------------------------------------------------------
        if partial_fraction <= 0.0:
            stop_hit = lo <= current_stop if is_long else hi >= current_stop
            target_hit = hi >= full_target_price if is_long else lo <= full_target_price

            if stop_hit and target_hit:
                # Same-bar conflict resolution (Pessimistic: stop checked first)
                px = _stop_exit_price(is_long, current_stop, o, slippage_pct)
                full_r = r_of(px)
                runner_price, runner_date = px, date
                outcome, exit_index = "stopped_full", k
                break
            elif stop_hit:
                px = _stop_exit_price(is_long, current_stop, o, slippage_pct)
                full_r = r_of(px)
                runner_price, runner_date = px, date
                outcome = "stopped_trailing" if current_stop != stop else "stopped_full"
                exit_index = k
                break
            elif target_hit:
                px = _target_exit_price(is_long, full_target_price, o)
                full_r = r_of(px)
                runner_price, runner_date = px, date
                outcome, exit_index = "target_full_2r", k
                break
            continue

        # ---------------------------------------------------------------
        # LEGACY / PARTIAL SCALING MODE (50% at 1R, BE runner to 2R)
        # ---------------------------------------------------------------
        if not partialed:
            stop_hit = lo <= current_stop if is_long else hi >= current_stop
            t1_hit = hi >= target1 if is_long else lo <= target1
            if stop_hit:
                px = _stop_exit_price(is_long, current_stop, o, slippage_pct)
                full_r = r_of(px)
                runner_price, runner_date = px, date
                outcome, exit_index = "stopped_full", k
                break
            if t1_hit:
                partial_price = _target_exit_price(is_long, target1, o)
                partial_r = r_of(partial_price)
                partial_date = date
                partialed = True
                current_stop = fill_price  # Move remaining position stop to Breakeven

                be_hit = lo <= current_stop if is_long else hi >= current_stop
                t2_hit = hi >= target2 if is_long else lo <= target2
                if be_hit:
                    px = _stop_exit_price(is_long, current_stop, o, slippage_pct)
                    runner_price, runner_r, runner_date = px, r_of(px), date
                    outcome, exit_index = "partial_be", k
                    break
                if t2_hit:
                    runner_price = _target_exit_price(is_long, target2, o)
                    runner_r, runner_date = r_of(runner_price), date
                    outcome, exit_index = "partial_t2", k
                    break
                continue
        else:
            be_hit = lo <= current_stop if is_long else hi >= current_stop
            t2_hit = hi >= target2 if is_long else lo <= target2
            if be_hit:
                px = _stop_exit_price(is_long, current_stop, o, slippage_pct)
                runner_price, runner_r, runner_date = px, r_of(px), date
                outcome, exit_index = "partial_be", k
                break
            if t2_hit:
                runner_price = _target_exit_price(is_long, target2, o)
                runner_r, runner_date = r_of(runner_price), date
                outcome, exit_index = "partial_t2", k
                break

    # Max hold duration expiration
    if outcome is None:
        last_idx = min(fill_idx + max_hold_bars - 1, n - 1)
        last_close = float(df.iloc[last_idx]["Close"])
        last_date = df.iloc[last_idx]["Date"] if "Date" in df.columns else last_idx
        exit_index = last_idx
        if partialed:
            runner_price, runner_r, runner_date = (
                last_close,
                r_of(last_close),
                last_date,
            )
            outcome = "partial_expired"
        else:
            full_r = r_of(last_close)
            runner_price, runner_date = last_close, last_date
            outcome = "expired_no_partial"

    if partialed:
        blended_r = partial_fraction * partial_r + (1 - partial_fraction) * runner_r
    else:
        blended_r = full_r

    result.update(
        {
            "outcome": outcome,
            "fill_index": fill_idx,
            "exit_index": exit_index,
            "fill_date": (
                df.iloc[fill_idx]["Date"] if "Date" in df.columns else fill_idx
            ),
            "fill_price": fill_price,
            "gap_entry": gap_entry,
            "stop_moved_be": partialed or (current_stop != stop),
            "partial_date": partial_date,
            "partial_price": partial_price,
            "partial_r": partial_r,
            "runner_date": runner_date,
            "runner_price": runner_price,
            "runner_r": runner_r,
            "blended_r": blended_r,
            "bars_held": bars_held,
            "mae_r": mae_r,
            "mfe_r": mfe_r,
        }
    )
    return result


def backtest_ticker(
    path: str,
    mode: str,
    long_min: int,
    short_max: int,
    warmup: int,
    entry_valid: int,
    max_hold: int,
    cooldown_bars: int = 0,
    benchmark_df: "pd.DataFrame | None" = None,
    slippage_pct: Optional[float] = None,
    partial_fraction: Optional[float] = None,
    target_r: float = 2.0,
    trailing_atr_mult: Optional[float] = 2.0,
) -> tuple[list[dict], dict]:
    """Walk-forward backtest for one ticker CSV. Returns trade rows and counters."""
    symbol = ticker_from_filename(path)
    df = load_ohlc_csv(path)

    if "High" not in df.columns or "Low" not in df.columns:
        return [], {"skipped": 1}

    sp = config.get_swing_params(mode)
    long_only = bool(sp.get("long_only"))
    if slippage_pct is None:
        slippage_pct = config.BACKTEST_SLIPPAGE_PCT
    if partial_fraction is None:
        partial_fraction = 0.0  # Default to No Partials per updated trade plan

    trades: list[dict] = []
    counts = {
        "signals": 0,
        "macro_pass": 0,
        "cooldown_skip": 0,
        "long_only_skip": 0,
        "filled": 0,
        "no_fill": 0,
        "wins": 0,
        "stopped_full": 0,
        "stopped_trailing": 0,
        "target_full_2r": 0,
        "partial_be": 0,
        "partial_t2": 0,
        "partial_expired": 0,
        "expired_no_partial": 0,
    }

    position_exit_idx = -1
    last_exit_idx: Optional[int] = None

    for i in range(warmup, len(df) - 1):
        if i <= position_exit_idx:
            continue  # One active position at a time

        window = df.iloc[: i + 1].copy()
        setup_row = detect_setup_at_bar(window, symbol, mode, benchmark_df)
        if not setup_row:
            continue

        counts["signals"] += 1
        setup_type = setup_row["Setup Type"]
        is_long = setup_type == "SETUP_LONG"

        if long_only and not is_long:
            counts["long_only_skip"] += 1
            continue

        feat = build_features(window)
        score = score_last_row(feat.iloc[-1])
        if not passes_macro_gate(setup_type, score, long_min, short_max):
            continue
        counts["macro_pass"] += 1

        if (
            cooldown_bars
            and last_exit_idx is not None
            and (i - last_exit_idx) < cooldown_bars
        ):
            counts["cooldown_skip"] += 1
            continue

        entry, stop, t1, t2, _, _ = calculate_stock_levels(setup_row, mode=mode)
        res = simulate_scaled_trade(
            df,
            i + 1,
            is_long,
            entry,
            stop,
            t1,
            t2,
            entry_valid,
            max_hold,
            slippage_pct=slippage_pct,
            partial_fraction=partial_fraction,
            target_r=target_r,
            trailing_atr_mult=trailing_atr_mult,
        )

        if res["outcome"] == "no_fill":
            counts["no_fill"] += 1
            continue

        counts["filled"] += 1
        counts[res["outcome"]] = counts.get(res["outcome"], 0) + 1
        position_exit_idx = res["exit_index"]
        last_exit_idx = res["exit_index"]
        blended = res["blended_r"]
        if blended is not None and blended > 0:
            counts["wins"] += 1

        signal_date = df.iloc[i]["Date"]
        risk = res["risk"]
        trades.append(
            {
                "Symbol": symbol,
                "Signal Date": _fmt_date(signal_date),
                "Setup Type": setup_type,
                "Mode": mode,
                "Vibe Score": score,
                "Stock Entry": round(entry, 2),
                "Stock Stop": round(stop, 2),
                "Risk Per Share": round(risk, 2),
                "Target 1R": round(t1, 2),
                "Target 2R": round(t2, 2),
                "Fill Date": _fmt_date(res["fill_date"]),
                "Fill Price": (
                    round(res["fill_price"], 2)
                    if res["fill_price"] is not None
                    else None
                ),
                "Gap Entry": res["gap_entry"],
                "Stop Moved BE": res["stop_moved_be"],
                "Partial Exit Date": _fmt_date(res["partial_date"]),
                "Partial Price": (
                    round(res["partial_price"], 2)
                    if res["partial_price"] is not None
                    else None
                ),
                "Partial R": (
                    round(res["partial_r"], 2) if res["partial_r"] is not None else None
                ),
                "Runner Exit Date": _fmt_date(res["runner_date"]),
                "Runner Price": (
                    round(res["runner_price"], 2)
                    if res["runner_price"] is not None
                    else None
                ),
                "Runner R": (
                    round(res["runner_r"], 2) if res["runner_r"] is not None else None
                ),
                "Outcome": res["outcome"],
                "Blended R Multiple": (
                    round(blended, 2) if blended is not None else None
                ),
                "Bars Held": res["bars_held"],
                "MAE R": round(res["mae_r"], 2) if res["mae_r"] is not None else None,
                "MFE R": round(res["mfe_r"], 2) if res["mfe_r"] is not None else None,
                "Regime OK": setup_row.get("Regime OK"),
                "RS 63d": setup_row.get("RS 63d"),
            }
        )

    return trades, counts


def load_ticker_filter(
    mode_cfg: dict, tickers_arg: Optional[str]
) -> Optional[set[str]]:
    """Resolve optional --tickers list; otherwise use active_tickers.csv if present."""
    if tickers_arg:
        return {t.strip().upper() for t in tickers_arg.split(",") if t.strip()}
    active_path = config.TICKER_LIST_PATH
    if os.path.isfile(active_path):
        try:
            return set(pd.read_csv(active_path)["Ticker"].dropna().str.upper())
        except Exception:
            return None
    return None


def _period_rank(path: str) -> int:
    """Prefer longer lookbacks when multiple CSVs exist for one ticker (5y > 2y)."""
    parts = os.path.basename(path).replace(".csv", "").split("_")
    if len(parts) < 2:
        return 0
    token = parts[1].lower()
    if token.endswith("y") and token[:-1].isdigit():
        return int(token[:-1]) * 365
    if token.endswith("mo") and token[:-2].isdigit():
        return int(token[:-2]) * 30
    if token.endswith("d") and token[:-1].isdigit():
        return int(token[:-1])
    return 0


def select_raw_paths(raw_dir: str, ticker_filter: Optional[set[str]]) -> list[str]:
    """List raw CSVs, optionally filtered, de-duped to one file per symbol."""
    paths = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv")
    )
    best: dict[str, tuple[int, str]] = {}
    for path in paths:
        symbol = ticker_from_filename(path)
        if ticker_filter and symbol not in ticker_filter:
            continue
        rank = _period_rank(path)
        prev = best.get(symbol)
        if prev is None or rank > prev[0]:
            best[symbol] = (rank, path)
    return [best[s][1] for s in sorted(best)]


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a win-rate proportion (as percents)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return max(0.0, center - margin) * 100, min(1.0, center + margin) * 100


def _blended_stats(df: pd.DataFrame) -> dict:
    """Risk-based aggregates over a frame of filled trades (Blended R Multiple)."""
    r = df["Blended R Multiple"].dropna()
    n = len(r)
    wins = int((r > 0).sum())
    winners = r[r > 0]
    losers = r[r <= 0]
    gross_win = float(winners.sum())
    gross_loss = float(-losers.sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    lo, hi = _wilson_interval(wins, n)
    return {
        "n": n,
        "win_rate": 100 * wins / n if n else 0.0,
        "win_lo": lo,
        "win_hi": hi,
        "expectancy": float(r.mean()) if n else 0.0,
        "total_r": float(r.sum()),
        "avg_winner": float(winners.mean()) if len(winners) else 0.0,
        "avg_loser": float(losers.mean()) if len(losers) else 0.0,
        "profit_factor": profit_factor,
    }


def print_summary(all_counts: dict, trades_df: pd.DataFrame) -> None:
    """Print risk-based backtest statistics (centered on filled trades)."""
    print("\n" + "=" * 60)
    print("PIPELINE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Signals detected:     {all_counts.get('signals', 0)}")
    print(f"Passed macro gate:    {all_counts.get('macro_pass', 0)}")
    print(f"Long-only skipped:    {all_counts.get('long_only_skip', 0)}")
    print(f"Cooldown skipped:     {all_counts.get('cooldown_skip', 0)}")
    print(f"Entries filled:       {all_counts.get('filled', 0)}")
    print(f"  No fill:            {all_counts.get('no_fill', 0)}")
    print(f"  Stopped (full):     {all_counts.get('stopped_full', 0)}")
    print(f"  Stopped (trailing): {all_counts.get('stopped_trailing', 0)}")
    print(f"  Target 2.0R Full:   {all_counts.get('target_full_2r', 0)}")
    print(f"  Partial -> BE:      {all_counts.get('partial_be', 0)}")
    print(f"  Partial -> 2R:      {all_counts.get('partial_t2', 0)}")
    print(f"  Partial -> expired: {all_counts.get('partial_expired', 0)}")
    print(f"  Expired:            {all_counts.get('expired_no_partial', 0)}")

    if trades_df.empty:
        print("\nNo filled trades to analyze.")
        return

    def _print_block(label: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        s = _blended_stats(frame)
        pf = (
            "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        )
        print(f"\n{label}:")
        print(f"  Trades:        {s['n']}")
        print(
            f"  Win rate:      {s['win_rate']:.1f}%  (95% CI {s['win_lo']:.0f}-{s['win_hi']:.0f}%)"
        )
        print(f"  Expectancy:    {s['expectancy']:+.2f}R  (total {s['total_r']:+.2f}R)")
        print(
            f"  Avg winner:    {s['avg_winner']:+.2f}R   Avg loser: {s['avg_loser']:+.2f}R"
        )
        print(f"  Profit factor: {pf}")
        if "MAE R" in frame.columns and frame["MAE R"].notna().any():
            print(
                f"  Avg MAE/MFE:   {frame['MAE R'].mean():+.2f}R / {frame['MFE R'].mean():+.2f}R"
            )

    _print_block("ALL FILLED", trades_df)
    for setup in ("SETUP_LONG", "SETUP_SHORT"):
        _print_block(setup, trades_df[trades_df["Setup Type"] == setup])


def run_backtest(
    mode: str = "weekly",
    tickers: Optional[str] = None,
    long_min: int = config.BACKTEST_LONG_MIN_SCORE,
    short_max: int = config.BACKTEST_SHORT_MAX_SCORE,
    warmup: int = config.BACKTEST_WARMUP_BARS,
    entry_valid: Optional[int] = None,
    max_hold: Optional[int] = None,
    cooldown_bars: Optional[int] = None,
    partial_fraction: float = 0.0,
    target_r: float = 2.0,
    trailing_atr_mult: Optional[float] = 2.0,
) -> pd.DataFrame:
    """Run walk-forward backtest across raw CSV files for *mode*."""
    data_mode, swing_profile = config.resolve_pipeline_mode(mode)
    mode_cfg = config.get_mode_config(data_mode)
    swing = config.get_swing_params(swing_profile)
    raw_dir = mode_cfg["raw_dir"]
    logs_dir = config.get_log_dir(mode)
    ticker_filter = load_ticker_filter(mode_cfg, tickers)

    if entry_valid is None:
        entry_valid = swing["entry_valid_bars"]
    if max_hold is None:
        max_hold = swing["max_hold_bars"]
    if cooldown_bars is None:
        cooldown_bars = swing["cooldown_bars"]

    if not os.path.isdir(raw_dir):
        print(f"No raw directory: {raw_dir}")
        return pd.DataFrame()

    paths = select_raw_paths(raw_dir, ticker_filter)

    benchmark_df = None
    if swing.get("benchmark"):
        benchmark_df = load_benchmark_frame(swing["benchmark"], data_mode)
        if benchmark_df is None:
            print(
                f"WARNING: benchmark {swing['benchmark']} unavailable; regime/RS gates will reject all."
            )

    all_trades: list[dict] = []
    all_counts: dict = {}

    exec_desc = (
        f"NO PARTIALS (Full Exit at {target_r:.1f}R, Trailing Stop={trailing_atr_mult:.1f} ATR)"
        if partial_fraction == 0.0
        else f"PARTIAL (50% at 1R, Runner to {target_r:.1f}R)"
    )

    print(f"--- Pipeline Backtest [{mode.upper()}] ---")
    print(f"Data mode: {data_mode} | Swing profile: {swing_profile}")
    print(f"Macro gate: LONG score >= {long_min}, SHORT score <= {short_max}")
    print(
        f"Profile: long_only={swing.get('long_only')} "
        f"benchmark={swing.get('benchmark')} regime={swing.get('require_market_regime')} "
        f"rs={swing.get('require_relative_strength')} vibe_min={swing['vibe_min']} "
        f"cooldown={cooldown_bars} entry_valid={entry_valid} max_hold={max_hold}"
    )
    print(f"Execution: slippage={config.BACKTEST_SLIPPAGE_PCT*100:.2f}% | {exec_desc}")
    print(f"Raw dir: {raw_dir}")
    print(f"Tickers: {len(paths)}\n")

    for path in paths:
        trades, counts = backtest_ticker(
            path,
            swing_profile,
            long_min,
            short_max,
            warmup,
            entry_valid,
            max_hold,
            cooldown_bars=cooldown_bars,
            benchmark_df=benchmark_df,
            partial_fraction=partial_fraction,
            target_r=target_r,
            trailing_atr_mult=trailing_atr_mult,
        )
        all_trades.extend(trades)
        for k, v in counts.items():
            all_counts[k] = all_counts.get(k, 0) + v
        if trades:
            symbol = ticker_from_filename(path)
            print(
                f"{symbol}: {counts.get('signals', 0)} signal(s), {len(trades)} filled"
            )

    trades_df = pd.DataFrame(all_trades)
    print_summary(all_counts, trades_df)

    if not trades_df.empty:
        stamp = datetime.now().strftime("%Y-%m-%d")
        tag = swing_profile if swing_profile != data_mode else data_mode
        out_path = os.path.join(logs_dir, f"backtest_trades_{tag}_{stamp}.csv")
        trades_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return trades_df


def main(argv: list[str] | None = None) -> int:
    choices = list(config.TIMEFRAME_PROFILES) + ["high_beta"]
    parser = argparse.ArgumentParser(
        description="Walk-forward pipeline output backtest"
    )
    parser.add_argument("mode", nargs="?", default=config.DEFAULT_MODE, choices=choices)
    parser.add_argument(
        "--tickers", help="Comma-separated tickers (default: active_tickers.csv)"
    )
    parser.add_argument(
        "--long-min-score",
        type=int,
        default=None,
        help="Hard macro gate for LONG (default: 7 weekly; open/-10 daily+high_beta)",
    )
    parser.add_argument(
        "--short-max-score", type=int, default=config.BACKTEST_SHORT_MAX_SCORE
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=None,
        help="Min bars between accepted signals (default: from swing profile)",
    )
    parser.add_argument(
        "--no-partials",
        action="store_true",
        default=True,
        help="Disable 50%% partial scaling; use full position target exit at 2.0R (default: True)",
    )
    parser.add_argument(
        "--use-partials",
        action="store_false",
        dest="no_partials",
        help="Enable legacy partial scaling at 1.0R",
    )
    parser.add_argument(
        "--target-r",
        type=float,
        default=1.5,
        help="Target R multiple for full exit mode (default: 2.0)",
    )
    parser.add_argument(
        "--trailing-atr-mult",
        type=float,
        default=2.0,
        help="Trailing stop ATR multiplier below current bar high (default: 2.0; set 1.5 or 0.0 to disable)",
    )

    args = parser.parse_args(argv)

    data_mode, _swing = config.resolve_pipeline_mode(args.mode)
    if args.long_min_score is None:
        long_min = -10 if data_mode == "daily" else config.BACKTEST_LONG_MIN_SCORE
    else:
        long_min = args.long_min_score

    partial_fraction = 0.0 if args.no_partials else 0.5

    run_backtest(
        mode=args.mode,
        tickers=args.tickers,
        long_min=long_min,
        short_max=args.short_max_score,
        cooldown_bars=args.cooldown_bars,
        partial_fraction=partial_fraction,
        target_r=args.target_r,
        trailing_atr_mult=args.trailing_atr_mult,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
