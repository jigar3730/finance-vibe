"""Generic OHLC-bar trade simulation primitives.

Given an entry/stop/target geometry and a bar-indexed OHLC frame, these
functions forward-simulate fills, stops, and targets. They have no
dependency on any particular signal source (swing setups, Coiled Cobra,
etc.) -- callers pass in levels computed however they like (see
``trade_planner.calculate_stock_levels``) and a starting bar index.

Extracted from the former ``pipeline_backtest.py`` (which also contained a
swing_scanner-specific walk-forward harness, since removed) so
``coiled_cobra_backtest.py`` can keep using ``simulate_trade`` without a
dependency on swing-scanner code.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


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
