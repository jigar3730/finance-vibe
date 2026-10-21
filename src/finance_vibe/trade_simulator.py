"""Generic OHLC-bar trade simulation primitives.

Given an entry/stop/target geometry and a bar-indexed OHLC frame, these
functions forward-simulate fills, stops, and targets. They have no
dependency on any particular signal source -- callers pass in levels
computed however they like (see ``trade_planner.calculate_stock_levels``)
and a starting bar index.
"""

from __future__ import annotations

import pandas as pd


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
) -> tuple[str, pd.Timestamp | None, float | None, float | None]:
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
