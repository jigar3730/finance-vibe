"""Frozen 10-year price snapshot and the fixed universe the backfill runs on.

``download`` pulls daily and weekly bars once into
``config.BACKFILL_SNAPSHOT_DIR/<date>/{daily,weekly}/`` and records a SHA-256
of every file in ``manifest.json``. A dated folder is never overwritten or
appended to: Yahoo restates adjusted prices after splits and dividends, so
newer data means a whole new snapshot.

``evaluate`` checks those hashes, applies the written rule (history, price,
dollar volume, no funds, bar quality over the full history) and writes
``universe.csv`` and ``report.md``. It reads prices only for those checks and
never looks at forward returns, and it can be rerun without downloading.

    python -m finance_vibe.research.snapshot download --commit <hash>
    python -m finance_vibe.research.snapshot evaluate <snapshot-dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from finance_vibe import config
from finance_vibe.data import daily_ingest, ingest, raw
from finance_vibe.data.quality import bar_stats, quality_failures
from finance_vibe.log import setup_logging

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
UNIVERSE = "universe.csv"
REPORT = "report.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_bars(df: pd.DataFrame, mode: str, now: datetime) -> pd.DataFrame:
    """Finished bars only, validated against the OHLCV contract."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if mode == "weekly":
        df = ingest.one_row_per_week(ingest.drop_incomplete_weeks(df, now=now))
    else:
        df = df.loc[[daily_ingest.daily_bar_is_complete(d, now=now) for d in df.index]]
    return config.validate_and_clean_ohlcv(df, require_volume=True)


def _download_mode(
    tickers: list[str], out_dir: Path, mode: str, now: datetime, batch_size: int
) -> dict[str, str]:
    """Save each ticker's bars into ``out_dir``; returns ``{ticker: failure reason}``."""
    settings = {"period": config.BACKFILL_PERIOD, "interval": config.BACKFILL_INTERVALS[mode]}
    failed: dict[str, str] = {}
    for start in range(0, len(tickers), batch_size):
        chunk = tickers[start : start + batch_size]
        try:
            batch = ingest._download_batch(chunk, settings["period"], settings["interval"])
        except Exception as exc:  # batch failed after its retries: move on, retried later
            logger.exception("Batch %d failed", start)
            failed.update(dict.fromkeys(chunk, f"batch_exception:{exc}"))
            continue
        for ticker in chunk:
            try:
                df = ingest._ticker_frame(batch, ticker)
                if df is None or df.empty:
                    failed[ticker] = "empty_download"
                    continue
                clean = _clean_bars(df, mode, now)
                if len(clean) < config.MIN_SAVE_ROWS:
                    failed[ticker] = f"insufficient_rows:{len(clean)}"
                    continue
                clean.to_csv(out_dir / config._get_raw_filename(ticker, settings), index=False)
            except ValueError as exc:
                failed[ticker] = f"validation:{exc}"
    return failed


def download(
    tickers: list[str],
    out_root: Path,
    *,
    now: datetime | None = None,
    commit: str = "unknown",
    batch_size: int = ingest.BATCH_SIZE,
) -> Path:
    """Create one dated snapshot; raises ``FileExistsError`` if that date already has one.

    Files go to ``<date>.partial`` and the folder takes its final name only
    when both timeframes and the manifest are written, so a dated folder is
    always complete. Failed tickers are retried once, then listed in the manifest.
    """
    now = now or datetime.now(ingest.MARKET_TZ)
    final = out_root / now.date().isoformat()
    if final.exists():
        raise FileExistsError(f"snapshot already exists: {final}")
    partial = out_root / f"{final.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)

    symbols = sorted({t.upper() for t in tickers} | set(config.BACKFILL_BENCHMARKS))
    failures: dict[str, dict[str, str]] = {}
    for mode in config.BACKFILL_INTERVALS:
        (partial / mode).mkdir(parents=True)
        failed = _download_mode(symbols, partial / mode, mode, now, batch_size)
        if failed:
            logger.warning("%s: retrying %d failed tickers", mode, len(failed))
            failed = _download_mode(sorted(failed), partial / mode, mode, now, batch_size)
        failures[mode] = failed
        missing = set(failed) & set(config.BACKFILL_BENCHMARKS)
        if missing:
            raise ValueError(f"benchmark download failed ({mode}): {sorted(missing)}")

    files = {p.relative_to(partial).as_posix(): _sha256(p) for p in sorted(partial.rglob("*.csv"))}
    manifest = {
        "stamp": final.name,
        "created_at": now.isoformat(timespec="seconds"),
        "commit": commit,
        "yfinance": yf.__version__,
        "period": config.BACKFILL_PERIOD,
        "adjusted": True,
        "expected_last": {
            "daily": daily_ingest.last_complete_session(now).isoformat(),
            "weekly": ingest.last_complete_week(now).isoformat(),
        },
        "requested": symbols,
        "failures": failures,
        "files": files,
    }
    (partial / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    partial.rename(final)
    logger.info("Snapshot written: %s (%d files)", final, len(files))
    return final


def verify(snapshot: Path) -> dict[str, Any]:
    """The manifest, after checking every file matches it; raises ``ValueError`` if not."""
    manifest: dict[str, Any] = json.loads((snapshot / MANIFEST).read_text())
    on_disk = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*.csv")} - {UNIVERSE}
    if on_disk != set(manifest["files"]):
        raise ValueError(f"snapshot files differ from manifest: {snapshot}")
    changed = [n for n, h in manifest["files"].items() if _sha256(snapshot / n) != h]
    if changed:
        raise ValueError(f"snapshot files changed since download: {changed[:5]}")
    return manifest


def _load_mode(snapshot: Path, mode: str) -> dict[str, pd.DataFrame]:
    return {f.ticker: raw.load_raw(f.path) for f in raw.raw_files(str(snapshot / mode))}


def evaluate(snapshot: Path) -> pd.DataFrame:
    """Apply the universe rule; writes ``universe.csv`` and ``report.md`` and returns the table."""
    manifest = verify(snapshot)
    daily, weekly = _load_mode(snapshot, "daily"), _load_mode(snapshot, "weekly")
    if "SPY" not in daily or "SPY" not in weekly:
        raise ValueError("snapshot has no SPY bars to use as the calendar")
    calendars = {
        "daily": pd.DatetimeIndex(daily["SPY"]["Date"]),
        "weekly": pd.DatetimeIndex(weekly["SPY"]["Date"]),
    }
    expected = {m: pd.Timestamp(d) for m, d in manifest["expected_last"].items()}
    latest_start = (
        expected["daily"]
        - pd.DateOffset(years=config.BACKFILL_MIN_YEARS)
        + pd.Timedelta(days=config.BACKFILL_HISTORY_GRACE_DAYS)
    )

    rows = []
    for symbol in manifest["requested"]:
        row, reasons = _evaluate_symbol(
            symbol, daily.get(symbol), weekly.get(symbol), calendars, expected, latest_start
        )
        benchmark = symbol in config.BACKFILL_BENCHMARKS
        rows.append(
            {
                "Symbol": symbol,
                "Role": "benchmark" if benchmark else "candidate",
                "In Universe": not benchmark and not reasons,
                "Reasons": ";".join(reasons),
                **row,
            }
        )
    table = pd.DataFrame(rows).sort_values("Symbol").reset_index(drop=True)
    table.to_csv(snapshot / UNIVERSE, index=False)
    (snapshot / REPORT).write_text(_report(table, manifest))
    return table


def _evaluate_symbol(
    symbol: str,
    day: pd.DataFrame | None,
    week: pd.DataFrame | None,
    calendars: dict[str, pd.DatetimeIndex],
    expected: dict[str, pd.Timestamp],
    latest_start: pd.Timestamp,
) -> tuple[dict[str, Any], list[str]]:
    """One universe row and the rules this ticker fails (first listed = most basic)."""
    if day is None or week is None:
        return {}, ["missing_download"]
    reasons = []
    if symbol in config.RESEARCH_EXCLUDED_SYMBOLS and symbol not in config.BACKFILL_BENCHMARKS:
        reasons.append("fund")
    first = min(day["Date"].iloc[0], week["Date"].iloc[0])
    if max(day["Date"].iloc[0], week["Date"].iloc[0]) > latest_start:
        reasons.append("short_history")
    close = float(day["Close"].iloc[-1])
    if close < config.BACKFILL_MIN_CLOSE:
        reasons.append("low_price")
    recent = day.tail(config.BACKFILL_DV_WINDOW)
    dollar_volume = float((recent["Close"] * recent["Volume"]).median())
    if dollar_volume < config.BACKFILL_MIN_DOLLAR_VOLUME:
        reasons.append("low_dollar_volume")

    row: dict[str, Any] = {
        "First Bar": first.date().isoformat(),
        "Close": round(close, 2),
        "Dollar Volume": round(dollar_volume),
    }
    for mode, df in (("daily", day), ("weekly", week)):
        if df["Date"].iloc[-1] < expected[mode]:
            reasons.append(f"{mode}:stale")
        stats = bar_stats(df, calendars[mode])
        reasons += [f"{mode}:{name}" for name in quality_failures(stats)]
        row[f"{mode} bars"] = len(df)
        row.update({f"{mode} {k}": v for k, v in stats.items()})
    return row, reasons


def _report(table: pd.DataFrame, manifest: dict[str, Any]) -> str:
    candidates = table[table["Role"] == "candidate"]
    members = candidates[candidates["In Universe"]]
    out = candidates[~candidates["In Universe"]]
    tally = out["Reasons"].str.split(";").explode().value_counts()
    failed = sum(len(v) for v in manifest["failures"].values())
    lines = [
        f"# Backfill snapshot {manifest['stamp']}",
        "",
        f"- Requested: {len(manifest['requested'])} symbols; download failures: {failed}",
        f"- Newest bars: daily {manifest['expected_last']['daily']}, "
        f"weekly {manifest['expected_last']['weekly']}",
        f"- Universe: {len(members)} of {len(candidates)} candidates",
        "",
        "## Exclusions by rule (a ticker can fail several)",
        "",
        *[f"- {name}: {count}" for name, count in tally.items()],
        "",
        "## Excluded tickers",
        "",
        *[f"- {r.Symbol}: {r.Reasons}" for r in out.itertuples(index=False)],
        "",
    ]
    return "\n".join(lines)


def read_tickers(paths: list[str]) -> list[str]:
    """Upper-cased, de-duplicated ``Ticker`` column of every CSV in ``paths``."""
    tickers = pd.concat([pd.read_csv(p)["Ticker"] for p in paths]).dropna().astype(str)
    return sorted(set(tickers.str.upper()))


def main(argv: list[str] | None = None) -> int:
    """CLI: ``download`` or ``evaluate <snapshot-dir>`` (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("download")
    pull.add_argument(
        "--tickers", nargs="+", default=[config.TICKER_LIST_PATH, config.BACKFILL_SP500_PATH]
    )
    pull.add_argument("--out-root", default=config.BACKFILL_SNAPSHOT_DIR)
    pull.add_argument("--commit", default="unknown")
    check = sub.add_parser("evaluate")
    check.add_argument("snapshot")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "download":
        path = download(read_tickers(args.tickers), Path(args.out_root), commit=args.commit)
        print(path)
        return 0
    table = evaluate(Path(args.snapshot))
    logger.info("Universe: %d tickers", int(table["In Universe"].sum()))
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
