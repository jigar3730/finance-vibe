"""The scan universe: rule-based members, rebuilt once a quarter, with hysteresis.

Candidates are the S&P 500 and S&P 400 lists plus the stock watchlist
(``config.SP500_TICKERS_PATH``, ``SP400_TICKERS_PATH``, ``WATCHLIST_PATH``).
At a rebuild each candidate's last ``UNIVERSE_HISTORY_PERIOD`` of daily bars is
downloaded into ``<out>/bars/`` and judged:

- a non-member **joins** with at least ``UNIVERSE_MIN_BARS`` bars, a newest bar
  on the last complete session, close >= ``UNIVERSE_JOIN_CLOSE``, median
  ``UNIVERSE_DV_WINDOW``-bar Close x Volume >= ``UNIVERSE_JOIN_DOLLAR_VOLUME`` and
  no full-history bar-quality failure (:func:`data.quality.quality_failures`);
- a member **stays** while close >= ``UNIVERSE_STAY_CLOSE`` and dollar volume >=
  ``UNIVERSE_STAY_DOLLAR_VOLUME``. A member that is no longer a candidate (left
  both indexes and the watchlist) is removed. A member whose download failed is
  kept, so one bad Yahoo day never drops it. Bar-quality and stale findings are
  noted for members but don't remove them.

``UNIVERSE_FUNDS`` are always members (role ``fund``), without rules.

A rebuild is due when there is no build yet, or the newest rebalance date (third
Friday of a ``UNIVERSE_REBALANCE_MONTHS`` month, effective the day after) is
after the last build; ``--rebuild`` forces one. Otherwise members stay as they
are. Every run, rebuild or not, writes ``config.TICKER_LIST_PATH`` from the
members, so the scanners and ingest keep reading one ``Ticker`` column.

If more than ``UNIVERSE_MAX_FAILED_SHARE`` of the downloads fail, the previous
members are kept and ``meta.json`` records ``"status": "kept_previous"``.

Files in ``<out>`` (``config.UNIVERSE_DIR`` by default):

- ``members.csv``: one row per member (role, sources, date added, numbers)
- ``changes.csv``: every addition and removal with its date and reason (append-only)
- ``rejected.csv``: candidates that are not members at the newest rebuild, and why
- ``meta.json``: last build and attempt, status, counts, index-list dates

    python -m finance_vibe.data.universe              # rebuild only when due
    python -m finance_vibe.data.universe --rebuild    # rebuild now
    python -m finance_vibe.data.universe --out DIR --rebuild   # trial run elsewhere
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from finance_vibe import config
from finance_vibe.data import daily_ingest, ingest, raw
from finance_vibe.data.quality import bar_stats, quality_failures
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

MEMBERS, CHANGES, REJECTED, META, BARS = (
    "members.csv",
    "changes.csv",
    "rejected.csv",
    "meta.json",
    "bars",
)
MEMBER_COLUMNS = [
    "Symbol",
    "Role",
    "Sources",
    "Added",
    "Close",
    "Dollar Volume",
    "Bars",
    "Notes",
]
CHANGE_COLUMNS = ["Date", "Symbol", "Change", "Reason"]
REJECTED_COLUMNS = ["Symbol", "Sources", "Reasons", "Close", "Dollar Volume", "Bars"]
SOURCE_LISTS = {
    "S&P 500": "SP500_TICKERS_PATH",
    "S&P 400": "SP400_TICKERS_PATH",
    "watchlist": "WATCHLIST_PATH",
}
CALENDAR_SYMBOL = "SPY"  # trading calendar for the missing-bar check
DOWNLOAD_SETTINGS = {"period": config.UNIVERSE_HISTORY_PERIOD, "interval": "1d"}

Downloader = Callable[[list[str], Path, datetime], dict[str, str]]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def third_friday(year: int, month: int) -> date:
    """The third Friday of ``month``: S&P's quarterly rebalance date."""
    fifteenth = date(year, month, 15)
    return fifteenth + timedelta(days=(4 - fifteenth.weekday()) % 7)


def latest_rebalance(today: date) -> date:
    """The newest rebalance date already in effect on ``today`` (strictly before it)."""
    dates = [
        third_friday(y, m)
        for y in (today.year - 1, today.year)
        for m in config.UNIVERSE_REBALANCE_MONTHS
    ]
    return max(d for d in dates if d < today)


def next_rebalance(today: date) -> date:
    """The next rebalance date on or after ``today``."""
    dates = [
        third_friday(y, m)
        for y in (today.year, today.year + 1)
        for m in config.UNIVERSE_REBALANCE_MONTHS
    ]
    return min(d for d in dates if d >= today)


def rebuild_reason(meta: dict[str, Any], today: date, *, force: bool = False) -> str | None:
    """Why a rebuild is due (``initial``, ``quarterly``, ``forced``), or None."""
    if force:
        return "forced"
    built = meta.get("built")
    if not built:
        return "initial"
    if date.fromisoformat(built) <= latest_rebalance(today):
        return "quarterly"
    return None


# ---------------------------------------------------------------------------
# Candidates and rules
# ---------------------------------------------------------------------------


def _read_symbols(path: str) -> list[str]:
    df = pd.read_csv(path)
    col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    return [s for s in df[col].dropna().astype(str).str.strip().str.upper() if s]


def load_candidates() -> dict[str, list[str]]:
    """Stock candidates mapped to their source lists; funds are never candidates."""
    sources: dict[str, list[str]] = {}
    for name, attr in SOURCE_LISTS.items():
        for symbol in _read_symbols(getattr(config, attr)):
            if symbol not in config.UNIVERSE_FUNDS:
                sources.setdefault(symbol, []).append(name)
    return dict(sorted(sources.items()))


def measure(bars: pd.DataFrame | None, calendar: pd.DatetimeIndex) -> dict[str, Any]:
    """The numbers the rules use, from one ticker's daily bars (a ``load_raw`` frame)."""
    if bars is None or bars.empty:
        return {}
    recent = bars.tail(config.UNIVERSE_DV_WINDOW)
    return {
        "Close": float(bars["Close"].iloc[-1]),
        "Dollar Volume": float((recent["Close"] * recent["Volume"]).median()),
        "Bars": len(bars),
        "Last Bar": bars["Date"].iloc[-1].date(),
        "quality": quality_failures(bar_stats(bars, calendar)),
    }


def judge(m: dict[str, Any], *, member: bool, expected_last: date) -> tuple[bool, list[str]]:
    """(in the universe, reasons). For a member, reasons that keep it are notes."""
    if not m:
        return (True, ["no_data_kept"]) if member else (False, ["missing_download"])
    if member:
        reasons = []
        if m["Close"] < config.UNIVERSE_STAY_CLOSE:
            reasons.append(f"close_below_{config.UNIVERSE_STAY_CLOSE:g}")
        if m["Dollar Volume"] < config.UNIVERSE_STAY_DOLLAR_VOLUME:
            reasons.append(f"dollar_volume_below_{config.UNIVERSE_STAY_DOLLAR_VOLUME / 1e6:g}M")
        if reasons:
            return False, reasons
        notes = [f"quality:{q}" for q in m["quality"]]
        if m["Last Bar"] < expected_last:
            notes.append("stale")
        return True, notes
    reasons = []
    if m["Bars"] < config.UNIVERSE_MIN_BARS:
        reasons.append("short_history")
    if m["Last Bar"] < expected_last:
        reasons.append("stale")
    if m["Close"] < config.UNIVERSE_JOIN_CLOSE:
        reasons.append(f"close_below_{config.UNIVERSE_JOIN_CLOSE:g}")
    if m["Dollar Volume"] < config.UNIVERSE_JOIN_DOLLAR_VOLUME:
        reasons.append(f"dollar_volume_below_{config.UNIVERSE_JOIN_DOLLAR_VOLUME / 1e6:g}M")
    reasons += [f"quality:{q}" for q in m["quality"]]
    return not reasons, reasons


def build(
    candidates: dict[str, list[str]],
    measures: dict[str, dict[str, Any]],
    previous: pd.DataFrame,
    *,
    today: date,
    expected_last: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """New (members, changes, rejected) from the candidates' measures and the previous members.

    Pure: no IO. ``previous`` is a members frame (empty on the first build).
    """
    prev_stocks = previous[previous["Role"] == "stock"] if not previous.empty else previous
    added_on = (
        dict(zip(previous["Symbol"], previous["Added"], strict=True)) if len(previous) else {}
    )
    prev_set = set(prev_stocks["Symbol"]) if len(prev_stocks) else set()
    first_build = previous.empty
    stamp = today.isoformat()

    members: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for symbol in config.UNIVERSE_FUNDS:
        m = measures.get(symbol, {})
        members.append(_member_row(symbol, "fund", ["fund"], added_on.get(symbol, stamp), m, []))
        if symbol not in added_on:
            changes.append(_change(stamp, symbol, "added", "fund"))

    for symbol, sources in candidates.items():
        m = measures.get(symbol, {})
        is_member = symbol in prev_set
        keep, reasons = judge(m, member=is_member, expected_last=expected_last)
        if keep:
            members.append(
                _member_row(symbol, "stock", sources, added_on.get(symbol, stamp), m, reasons)
            )
            if not is_member:
                why = "initial build" if first_build else "passed the join rules"
                changes.append(_change(stamp, symbol, "added", why))
        else:
            rejected.append(_rejected_row(symbol, sources, m, reasons))
            if is_member:
                changes.append(_change(stamp, symbol, "removed", ";".join(reasons)))

    for symbol in sorted(prev_set - set(candidates)):
        changes.append(_change(stamp, symbol, "removed", "left_index_and_watchlist"))

    return (
        pd.DataFrame(members, columns=MEMBER_COLUMNS),
        pd.DataFrame(changes, columns=CHANGE_COLUMNS),
        pd.DataFrame(rejected, columns=REJECTED_COLUMNS),
    )


def _numbers(m: dict[str, Any]) -> dict[str, Any]:
    if not m:
        return {"Close": float("nan"), "Dollar Volume": float("nan"), "Bars": pd.NA}
    return {
        "Close": round(m["Close"], 2),
        "Dollar Volume": round(m["Dollar Volume"]),
        "Bars": m["Bars"],
    }


def _member_row(
    symbol: str, role: str, sources: list[str], added: str, m: dict[str, Any], notes: list[str]
) -> dict[str, Any]:
    return {
        "Symbol": symbol,
        "Role": role,
        "Sources": ";".join(sources),
        "Added": added,
        **_numbers(m),
        "Notes": ";".join(notes),
    }


def _rejected_row(
    symbol: str, sources: list[str], m: dict[str, Any], reasons: list[str]
) -> dict[str, Any]:
    return {"Symbol": symbol, "Sources": ";".join(sources), "Reasons": ";".join(reasons)} | (
        _numbers(m)
    )


def _change(day: str, symbol: str, change: str, reason: str) -> dict[str, str]:
    return {"Date": day, "Symbol": symbol, "Change": change, "Reason": reason}


def ticker_list(members: pd.DataFrame) -> pd.Series:
    """The ``Ticker`` column for ``config.TICKER_LIST_PATH``: funds first, then stocks A-Z."""
    funds = [s for s in config.UNIVERSE_FUNDS if s in set(members["Symbol"])]
    stocks = sorted(members.loc[members["Role"] == "stock", "Symbol"])
    return pd.Series([*funds, *stocks], name="Ticker")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _download_pass(
    symbols: list[str], bars_dir: Path, now: datetime, *, min_rows: int
) -> dict[str, str]:
    """One download pass; a result shorter than ``min_rows`` counts as failed (not saved)."""
    failed: dict[str, str] = {}
    period, interval = DOWNLOAD_SETTINGS["period"], DOWNLOAD_SETTINGS["interval"]
    for start in range(0, len(symbols), ingest.BATCH_SIZE):
        chunk = symbols[start : start + ingest.BATCH_SIZE]
        try:
            batch = ingest._download_batch(chunk, period, interval)
        except Exception as exc:  # yfinance raises anything; the batch counts as failed
            logger.exception("Universe batch %d failed", start)
            failed.update(dict.fromkeys(chunk, f"batch_exception:{exc}"))
            continue
        for symbol in chunk:
            try:
                df = ingest._ticker_frame(batch, symbol)
                if df is None or df.empty:
                    failed[symbol] = "empty_download"
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.copy()
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[[daily_ingest.daily_bar_is_complete(d, now=now) for d in df.index]]
                clean = config.validate_and_clean_ohlcv(df, require_volume=True)
                if len(clean) < max(min_rows, 1):
                    failed[symbol] = f"insufficient_rows:{len(clean)}"
                    continue
                clean.to_csv(
                    bars_dir / config._get_raw_filename(symbol, DOWNLOAD_SETTINGS), index=False
                )
            except ValueError as exc:
                failed[symbol] = f"validation:{exc}"
    return failed


def download_bars(symbols: list[str], bars_dir: Path, now: datetime) -> dict[str, str]:
    """Save each symbol's finished daily bars into ``bars_dir``; ``{symbol: failure}``.

    Yahoo sometimes returns nothing or a few bars for a long-listed ticker inside a
    batch, so empty or shorter-than-``MIN_SAVE_ROWS`` results are fetched once more;
    the retry keeps whatever it gets, so a real new listing still lands as
    ``short_history``.
    """
    failed = _download_pass(symbols, bars_dir, now, min_rows=config.MIN_SAVE_ROWS)
    if failed:
        logger.info("Universe: retrying %d downloads", len(failed))
        failed = _download_pass(sorted(failed), bars_dir, now, min_rows=1)
    return failed


def _load_bars(bars_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in raw.raw_files(str(bars_dir)):
        try:
            out[f.ticker] = raw.load_raw(f.path)
        except (OSError, ValueError) as exc:
            logger.warning("Universe bars unreadable for %s: %s", f.ticker, exc)
    return out


def read_members(out: Path) -> pd.DataFrame:
    path = out / MEMBERS
    if not path.exists():
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def read_meta(out: Path) -> dict[str, Any]:
    path = out / META
    return json.loads(path.read_text()) if path.exists() else {}


def _index_list_dates() -> dict[str, str]:
    path = Path(config.INDEX_LISTS_META_PATH)
    if not path.exists():
        return {}
    return {name: info["taken"] for name, info in json.loads(path.read_text()).items()}


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_changes(changes: pd.DataFrame, path: Path) -> None:
    if changes.empty:
        return
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=CHANGE_COLUMNS)
    _write_csv(pd.concat([old, changes], ignore_index=True), path)


def refresh(
    *,
    out: Path,
    ticker_list_path: Path,
    force: bool = False,
    now: datetime | None = None,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Rebuild when due, then write the ticker list from the members. Returns ``meta``."""
    now = now or datetime.now(ingest.MARKET_TZ)
    today = now.date()
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta(out)
    previous = read_members(out)
    reason = rebuild_reason(meta, today, force=force)

    if reason is not None:
        meta = _rebuild(out, previous, meta, reason, now, downloader or download_bars)
        previous = read_members(out)
    else:
        logger.info(
            "Universe: no rebuild due (built %s, next rebalance %s)",
            meta["built"],
            next_rebalance(today),
        )

    if previous.empty:
        logger.error("Universe has no members; %s left unchanged", ticker_list_path)
        return meta
    _write_csv(ticker_list(previous).to_frame(), ticker_list_path)
    counts = previous["Role"].value_counts()
    logger.info(
        "Universe: %d stocks + %d funds -> %s",
        counts.get("stock", 0),
        counts.get("fund", 0),
        ticker_list_path,
    )
    return meta


def _rebuild(
    out: Path,
    previous: pd.DataFrame,
    meta: dict[str, Any],
    reason: str,
    now: datetime,
    downloader: Downloader,
) -> dict[str, Any]:
    today = now.date()
    expected_last = daily_ingest.last_complete_session(now)
    candidates = load_candidates()
    symbols = sorted({*candidates, *config.UNIVERSE_FUNDS, CALENDAR_SYMBOL})
    logger.info("Universe rebuild (%s): %d symbols to download", reason, len(symbols))

    bars_dir = out / BARS
    shutil.rmtree(bars_dir, ignore_errors=True)
    bars_dir.mkdir(parents=True)
    failed = downloader(symbols, bars_dir, now)
    bars = _load_bars(bars_dir)
    lists = _index_list_dates()
    stale_lists = sorted(
        n for n, d in lists.items() if date.fromisoformat(d) < latest_rebalance(today)
    )
    if stale_lists:
        logger.warning("Index lists older than the last rebalance: %s", ", ".join(stale_lists))
    meta = meta | {
        "attempted": now.isoformat(timespec="seconds"),
        "reason": reason,
        "index_lists": lists,
        "index_lists_stale": stale_lists,
        "download_failures": len(failed),
        "candidates": len(candidates),
    }

    share = len(failed) / len(symbols)
    if share > config.UNIVERSE_MAX_FAILED_SHARE or CALENDAR_SYMBOL not in bars:
        logger.error(
            "Universe rebuild: %d of %d downloads failed (%.0f%%); keeping previous members",
            len(failed),
            len(symbols),
            100 * share,
        )
        meta["status"] = "kept_previous"
        (out / META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return meta

    calendar = pd.DatetimeIndex(bars[CALENDAR_SYMBOL]["Date"])
    measures = {s: measure(bars.get(s), calendar) for s in symbols}
    members, changes, rejected = build(
        candidates, measures, previous, today=today, expected_last=expected_last
    )
    _write_csv(members, out / MEMBERS)
    _write_csv(rejected, out / REJECTED)
    _append_changes(changes, out / CHANGES)
    counts = members["Role"].value_counts()
    meta |= {
        "built": today.isoformat(),
        "status": "ok",
        "stocks": int(counts.get("stock", 0)),
        "funds": int(counts.get("fund", 0)),
        "added": int((changes["Change"] == "added").sum()),
        "removed": int((changes["Change"] == "removed").sum()),
        "rejected": len(rejected),
        "next_rebalance": next_rebalance(today + timedelta(days=1)).isoformat(),
    }
    (out / META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Universe rebuilt: %d stocks, %d funds (+%d -%d), %d rejected",
        meta["stocks"],
        meta["funds"],
        meta["added"],
        meta["removed"],
        meta["rejected"],
    )
    return meta


def summarize(members: pd.DataFrame, rejected: pd.DataFrame) -> dict[str, Any]:
    """Counts for logs and reports: members by source, rejections by first reason."""
    stocks = members[members["Role"] == "stock"]
    by_source: dict[str, int] = {}
    for sources in stocks["Sources"]:
        for s in str(sources).split(";"):
            by_source[s] = by_source.get(s, 0) + 1
    first = rejected["Reasons"].astype(str).str.split(";").str[0]
    return {"by_source": by_source, "rejected_by_reason": first.value_counts().to_dict()}


def main(argv: Iterable[str] | None = None) -> int:
    """CLI: ``python -m finance_vibe.data.universe [--rebuild] [--out DIR]``."""
    parser = argparse.ArgumentParser(description="Refresh the scan universe")
    parser.add_argument("--rebuild", action="store_true", help="rebuild now, even if not due")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write everything (including the ticker list) under DIR instead of data/",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = args.out or Path(config.UNIVERSE_DIR)
    tickers = (args.out / "active_tickers.csv") if args.out else Path(config.TICKER_LIST_PATH)
    meta = refresh(out=out, ticker_list_path=tickers, force=args.rebuild)
    # A failed first build leaves the old list in place; only no list at all stops a run.
    return 0 if meta.get("built") or tickers.exists() else 1


def members_on(as_of: str, out: Path | None = None) -> set[str] | None:
    """Members on ``as_of`` from ``changes.csv`` (changes dated on or before it).

    None when there is no change history or ``as_of`` is before the first build:
    no membership was recorded then, so the caller falls back to the current list.
    """
    path = (out or Path(config.UNIVERSE_DIR)) / CHANGES
    if not path.exists():
        return None
    changes = pd.read_csv(path, dtype=str)
    changes = changes[changes["Date"] <= as_of]
    if changes.empty:
        return None
    members: set[str] = set()
    for symbol, change in zip(changes["Symbol"], changes["Change"], strict=True):
        if change == "added":
            members.add(symbol)
        else:
            members.discard(symbol)
    return members


def scan_tickers(ticker_list_path: str, as_of: str | None = None) -> set[str]:
    """Symbols a scanner should scan: the members on ``as_of`` for a replay, else the list.

    The list (``config.TICKER_LIST_PATH``) is the current members. A replay dated
    before the first universe build has no recorded membership and uses the list.
    """
    if as_of:
        members = members_on(as_of)
        if members is not None:
            logger.info("AS-OF %s: %d universe members on that date", as_of, len(members))
            return members
        logger.info("AS-OF %s: before the first universe build; using the current list", as_of)
    return set(pd.read_csv(ticker_list_path)["Ticker"].astype(str).str.strip().str.upper())


def health_problems(today: date, out: Path | None = None) -> list[str]:
    """Problems for the cron health checks: no build yet, or today's rebuild failed."""
    out = out or Path(config.UNIVERSE_DIR)
    meta = read_meta(out)
    if not meta.get("built"):
        return ["universe has no build yet (data/universe/meta.json); scanned the old list"]
    attempted = str(meta.get("attempted", ""))[:10]
    if meta.get("status") == "kept_previous" and attempted == today.isoformat():
        return [
            f"universe rebuild failed ({meta.get('download_failures')} downloads failed); "
            f"kept the {meta['built']} members"
        ]
    return []


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
