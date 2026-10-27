"""Breakout events as RESEARCH.md §3 defines them (first bar of a run only).

| ID | Event | Definition |
| E1 | PRE_BREAKOUT | Status == PRE_BREAKOUT and previous bar's Status != PRE_BREAKOUT |
| E2 | TRIGGERED | Status == BREAKOUT_TRIGGERED and previous bar not TRIGGERED/CONFIRMED |
| E3 | CONFIRMED | Status == BREAKOUT_CONFIRMED and previous bar not TRIGGERED/CONFIRMED |

"Previous bar" is the ticker's previous ``scan_history`` row. A ticker's first
history row never counts (its previous state is unknown).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from finance_vibe.scanners import breakout as bs

logger = logging.getLogger(__name__)
E1, E2, E3 = "E1", "E2", "E3"
EVENT_STATUS = {E1: bs.STATUS_PRE, E2: bs.STATUS_TRIGGERED, E3: bs.STATUS_CONFIRMED}
_BREAKOUT = [bs.STATUS_TRIGGERED, bs.STATUS_CONFIRMED]


def first_bar_events(history: pd.DataFrame) -> pd.DataFrame:
    """E1/E2/E3 rows of a stacked ``scan_history`` panel, with an ``Event`` column.

    ``history`` needs ``Date``, ``Symbol`` and ``Status``; other columns are
    kept. Output is sorted by Symbol then Date.
    """
    h = history.sort_values(["Symbol", "Date"], kind="stable").reset_index(drop=True)
    status = h["Status"]
    prev = h.groupby("Symbol", sort=False)["Status"].shift(1)
    has_prev = prev.notna()
    after_breakout = prev.isin(_BREAKOUT)
    event = np.select(
        [
            has_prev & status.eq(bs.STATUS_PRE) & prev.ne(bs.STATUS_PRE),
            has_prev & status.eq(bs.STATUS_TRIGGERED) & ~after_breakout,
            has_prev & status.eq(bs.STATUS_CONFIRMED) & ~after_breakout,
        ],
        [E1, E2, E3],
        "",
    )
    keep = event != ""
    out = h.loc[keep].copy()
    out.insert(2, "Event", event[keep])
    return out.reset_index(drop=True)


HISTORY_COLUMNS = ["Date", "Symbol", "Status", "Breakout Readiness", "Rubric Version"]


def history_panel(
    bars: Mapping[str, pd.DataFrame], mode: str = "daily"
) -> tuple[pd.DataFrame, dict[str, str]]:
    """``scan_history`` (``mode``: daily or weekly bars) for every ticker, stacked; plus
    skipped tickers and why."""
    frames, skipped = [], {}
    for symbol, b in bars.items():
        try:
            hist = bs.scan_history(b, symbol, mode, mode)
        except ValueError as exc:
            skipped[symbol] = str(exc)
            continue
        except Exception:  # per ticker: one bad file must not stop the panel
            logger.exception("scan_history failed for %s", symbol)
            skipped[symbol] = "execution_error"
            continue
        frames.append(hist[HISTORY_COLUMNS])
    return pd.concat(frames, ignore_index=True), skipped
