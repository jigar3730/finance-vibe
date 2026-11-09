"""SQLite copy of the scan outputs, for questions that span many runs.

The CSVs stay the source of truth: this file can be deleted and rebuilt at any
time, and no scanner reads it. Real scans (``live_*``) and backfilled history
(``backfill_*``) live in separate tables, so a query cannot mix what the
scanner said at the time with what today's rules say about the past by accident.
Column names are the CSV headers in snake_case (see :func:`column_name`).

A rebuild writes a temporary file and replaces the database only after its row
counts match the sources. Malformed input stops it; the old database stays.

    python -m finance_vibe.data.database rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from finance_vibe import config
from finance_vibe.data import analysis
from finance_vibe.log import setup_logging
from finance_vibe.scanners import breakout as bs

logger = logging.getLogger(__name__)

CHUNK_ROWS = 100_000
MODES = ("daily", "weekly")
# Table -> file prefix and expected CSV columns.
LIVE_TABLES = {
    "live_cobra": ("coiled_cobra_setups", config.SETUP_ROW_COLUMNS),
    "live_breakout": ("breakout_setups", bs.OUTPUT_COLUMNS),
}
BACKFILL_TABLES = {
    "backfill_cobra": ("coiled_cobra", config.COBRA_HISTORY_COLUMNS),
    "backfill_breakout": ("breakout", ["Date", *bs.OUTPUT_COLUMNS, "Rubric Version"]),
}
UNIVERSE_TABLE = "backfill_universe"
SOURCES_COLUMNS = ["table_name", "mode", "label", "path", "rows", "sha256"]
BOOL_MAP = {True: 1, False: 0, "True": 1, "False": 0}


@dataclass(frozen=True)
class Locations:
    """Where a rebuild reads from (defaults to the live data folders)."""

    logs: dict[str, Path] = field(
        default_factory=lambda: {m: Path(config.TIMEFRAME_PROFILES[m]["logs_dir"]) for m in MODES}
    )
    history: Path = Path(config.BACKFILL_HISTORY_DIR)
    snapshots: Path = Path(config.BACKFILL_SNAPSHOT_DIR)


def column_name(header: str) -> str:
    """SQL column for a CSV header: ``"Fib 61.8%"`` -> ``fib_61_8_pct``."""
    text = header.replace("%", " pct").lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _column_names(headers: list[str]) -> list[str]:
    names = [column_name(h) for h in headers]
    if len(set(names)) != len(names):
        raise ValueError(f"column names collide: {headers}")
    return names


def _create_table(conn: sqlite3.Connection, table: str, columns: list[str]) -> None:
    """Columns have no declared type, so numbers, text and NULL are stored as they are."""
    conn.execute(f'CREATE TABLE "{table}" ({", ".join(f"{c!r}" for c in columns)})')


def _bools_to_ints(df: pd.DataFrame) -> pd.DataFrame:
    """True/False columns (with or without blanks) become 1/0, so SQL can filter on them."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        values = series.dropna()
        if not values.empty and set(values.unique()) <= set(BOOL_MAP):
            out[col] = series.map(BOOL_MAP)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _check_columns(df: pd.DataFrame, expected: list[str], path: Path) -> None:
    if list(df.columns) != expected:
        raise ValueError(f"{path}: columns differ from the expected schema")


def _record(
    conn: sqlite3.Connection, table: str, mode: str, label: str, path: Path, rows: int
) -> None:
    conn.execute(
        "INSERT INTO sources VALUES (?,?,?,?,?,?)",
        (table, mode, label, str(path), rows, _sha256(path)),
    )


def _load_live(conn: sqlite3.Connection, locations: Locations) -> None:
    for table, (prefix, headers) in LIVE_TABLES.items():
        names = _column_names(headers)
        _create_table(conn, table, ["run_date", *names])
        pattern = re.compile(rf"{prefix}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
        for mode in MODES:
            folder = locations.logs[mode]
            for path in sorted(folder.glob(f"{prefix}_*.csv")) if folder.is_dir() else []:
                match = pattern.search(path.name)
                if match is None:
                    continue
                df = pd.read_csv(path)
                _check_columns(df, headers, path)
                df = _bools_to_ints(df)
                df.columns = pd.Index(names)
                df.insert(0, "run_date", match.group(1))
                df.to_sql(table, conn, if_exists="append", index=False)
                _record(conn, table, mode, match.group(1), path, len(df))
        conn.execute(f'CREATE UNIQUE INDEX "{table}_run" ON "{table}" (mode, run_date, symbol)')
        conn.execute(f'CREATE INDEX "{table}_symbol" ON "{table}" (symbol, asof_date)')


def _load_backfill(conn: sqlite3.Connection, locations: Locations) -> None:
    for table, (prefix, headers) in BACKFILL_TABLES.items():
        names = _column_names(headers)
        _create_table(conn, table, ["snapshot", *names])
        for mode in MODES:
            pattern = f"*/rubric-*/{prefix}_{mode}_history.csv.gz"
            for path in sorted(locations.history.glob(pattern)):
                snapshot = path.parent.parent.name
                rows = 0
                with pd.read_csv(path, chunksize=CHUNK_ROWS, low_memory=False) as reader:
                    for chunk in reader:
                        _check_columns(chunk, headers, path)
                        chunk = _bools_to_ints(chunk)
                        chunk.columns = pd.Index(names)
                        chunk.insert(0, "snapshot", snapshot)
                        chunk.to_sql(table, conn, if_exists="append", index=False)
                        rows += len(chunk)
                _record(conn, table, mode, snapshot, path, rows)
        conn.execute(
            f'CREATE UNIQUE INDEX "{table}_bar" ON "{table}" '
            "(snapshot, rubric_version, mode, symbol, date)"
        )
        conn.execute(f'CREATE INDEX "{table}_date" ON "{table}" (mode, date)')


def _load_universe(conn: sqlite3.Connection, locations: Locations) -> None:
    created = False
    for path in sorted(locations.snapshots.glob("*/universe.csv")):
        df = _bools_to_ints(pd.read_csv(path))
        df.columns = pd.Index(_column_names([str(c) for c in df.columns]))
        df.insert(0, "snapshot", path.parent.name)
        if not created:
            _create_table(conn, UNIVERSE_TABLE, [str(c) for c in df.columns])
            created = True
        df.to_sql(UNIVERSE_TABLE, conn, if_exists="append", index=False)
        _record(conn, UNIVERSE_TABLE, "", path.parent.name, path, len(df))
    if not created:
        _create_table(conn, UNIVERSE_TABLE, ["snapshot", "symbol"])


def _check_counts(conn: sqlite3.Connection) -> None:
    """Every table must hold exactly the rows its source files reported."""
    expected = dict(
        conn.execute("SELECT table_name, SUM(rows) FROM sources GROUP BY table_name").fetchall()
    )
    for table in [*LIVE_TABLES, *BACKFILL_TABLES, UNIVERSE_TABLE]:
        actual = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if actual != expected.get(table, 0):
            raise ValueError(f"{table}: {actual} rows, sources report {expected.get(table, 0)}")
    status = conn.execute("PRAGMA quick_check").fetchone()[0]
    if status != "ok":
        raise ValueError(f"database check failed: {status}")


def rebuild(db_path: Path, locations: Locations | None = None) -> dict[str, int]:
    """Build the database from every source file; returns rows per table."""
    locations = locations or Locations()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = db_path.with_name(f"{db_path.name}.building")
    temporary.unlink(missing_ok=True)
    conn = sqlite3.connect(temporary)
    try:
        conn.execute("PRAGMA synchronous = OFF")
        _create_table(conn, "sources", SOURCES_COLUMNS)
        _load_live(conn, locations)
        _load_backfill(conn, locations)
        _load_universe(conn, locations)
        conn.execute("CREATE TABLE built (built_at)")
        conn.execute(
            "INSERT INTO built VALUES (?)", (datetime.now().isoformat(timespec="seconds"),)
        )
        conn.commit()
        _check_counts(conn)
        counts = {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in [*LIVE_TABLES, *BACKFILL_TABLES, UNIVERSE_TABLE]
        }
    except BaseException:
        conn.close()
        temporary.unlink(missing_ok=True)
        raise
    conn.close()
    os.replace(temporary, db_path)
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI: ``rebuild [--path FILE]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("rebuild")
    build.add_argument("--path", default=config.DATABASE_PATH)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    path = Path(args.path)
    for table, rows in rebuild(path).items():
        logger.info("%s: %d rows", table, rows)
    analysis.write_summary(path, path.with_name(Path(config.ANALYSIS_SUMMARY_PATH).name))
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
