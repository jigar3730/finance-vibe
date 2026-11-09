"""SQLite copy of the scan outputs: complete, labelled, and never half-built."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.data import database as db
from finance_vibe.scanners import breakout as bs


def _cobra_rows(symbols: list[str], mode: str, day: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=config.SETUP_ROW_COLUMNS, index=range(len(symbols)))
    df["Symbol"], df["Mode"], df["AsOf Date"] = symbols, mode, day
    df["Score"], df["Market Gate"], df["Fib 61.8%"] = 80.0, True, 12.5
    return df


def _breakout_rows(symbols: list[str], mode: str, day: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=bs.OUTPUT_COLUMNS, index=range(len(symbols)))
    df["Symbol"], df["Mode"], df["AsOf Date"], df["Status"] = symbols, mode, day, "WATCH"
    return df


def _history(symbols: list[str], mode: str, dates: list[str]) -> pd.DataFrame:
    rows = [
        {"Date": d, "Symbol": s, "Mode": mode, "Passed": i % 2 == 0, "Rubric Version": 4.0}
        for s in symbols
        for i, d in enumerate(dates)
    ]
    return pd.DataFrame(rows).reindex(columns=config.COBRA_HISTORY_COLUMNS)


@pytest.fixture
def locations(tmp_path: Path) -> db.Locations:
    logs = {m: tmp_path / "logs" / m for m in db.MODES}
    for folder in logs.values():
        folder.mkdir(parents=True)
    _cobra_rows(["AAA", "BBB"], "daily", "2026-11-05").to_csv(
        logs["daily"] / "coiled_cobra_setups_2026-11-05.csv", index=False
    )
    _cobra_rows(["AAA"], "weekly", "2026-11-02").to_csv(
        logs["weekly"] / "coiled_cobra_setups_2026-11-06.csv", index=False
    )
    _breakout_rows(["CCC"], "daily", "2026-11-05").to_csv(
        logs["daily"] / "breakout_setups_2026-11-05.csv", index=False
    )
    (logs["daily"] / "data_quality_2026-11-05.csv").write_text("Symbol\nAAA\n")

    history = tmp_path / "history" / "2026-11-09" / "rubric-4.0"
    history.mkdir(parents=True)
    _history(["AAA", "BBB"], "daily", ["2026-11-04", "2026-11-05"]).to_csv(
        history / "coiled_cobra_daily_history.csv.gz", index=False
    )

    snapshot = tmp_path / "snapshots" / "2026-11-09"
    snapshot.mkdir(parents=True)
    (snapshot / "universe.csv").write_text(
        "Symbol,Role,In Universe,daily zero_volume\nAAA,candidate,True,0\nBBB,candidate,False,3\n"
    )
    return db.Locations(logs=logs, history=tmp_path / "history", snapshots=tmp_path / "snapshots")


def _query(path: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Fib 61.8%", "fib_61_8_pct"),
        ("AsOf Date", "asof_date"),
        ("RS 63d", "rs_63d"),
        ("BB Width Pctl", "bb_width_pctl"),
    ],
)
def test_column_name(header, expected):
    assert db.column_name(header) == expected


def test_every_expected_column_name_is_unique():
    for _, headers in [*db.LIVE_TABLES.values(), *db.BACKFILL_TABLES.values()]:
        assert len(set(db._column_names(headers))) == len(headers)


def test_rebuild_loads_every_source(locations, tmp_path):
    path = tmp_path / "x.sqlite"
    counts = db.rebuild(path, locations)
    assert counts == {
        "live_cobra": 3,
        "live_breakout": 1,
        "backfill_cobra": 4,
        "backfill_breakout": 0,
        "backfill_universe": 2,
    }
    assert not path.with_name("x.sqlite.building").exists()
    assert len(_query(path, "SELECT * FROM sources")) == 5  # 3 scans, 1 history, 1 universe
    assert _query(path, "SELECT COUNT(*) FROM built") == [(1,)]


def test_run_date_comes_from_the_file_and_bar_date_from_the_row(locations, tmp_path):
    path = tmp_path / "x.sqlite"
    db.rebuild(path, locations)
    rows = _query(path, "SELECT run_date, asof_date FROM live_cobra WHERE mode='weekly'")
    assert rows == [("2026-11-06", "2026-11-02")]


def test_values_are_queryable(locations, tmp_path):
    path = tmp_path / "x.sqlite"
    db.rebuild(path, locations)
    assert _query(path, "SELECT COUNT(*) FROM backfill_cobra WHERE passed = 1") == [(2,)]
    assert _query(path, "SELECT DISTINCT market_gate, fib_61_8_pct FROM live_cobra") == [(1, 12.5)]
    assert _query(path, "SELECT DISTINCT snapshot FROM backfill_cobra") == [("2026-11-09",)]
    assert _query(path, "SELECT symbol FROM backfill_universe WHERE in_universe=1") == [("AAA",)]
    assert _query(path, "SELECT daily_zero_volume FROM backfill_universe WHERE symbol='BBB'") == [
        (3,)
    ]


def test_blank_cells_are_null(locations, tmp_path):
    path = tmp_path / "x.sqlite"
    db.rebuild(path, locations)
    assert _query(path, "SELECT COUNT(*) FROM live_cobra WHERE tier IS NULL") == [(3,)]


def test_schema_drift_is_rejected_and_old_database_survives(locations, tmp_path):
    path = tmp_path / "x.sqlite"
    db.rebuild(path, locations)
    before = _query(path, "SELECT COUNT(*) FROM live_cobra")
    bad = locations.logs["daily"] / "coiled_cobra_setups_2026-11-06.csv"
    bad.write_text("Symbol,Surprise\nAAA,1\n")
    with pytest.raises(ValueError, match="columns differ"):
        db.rebuild(path, locations)
    assert _query(path, "SELECT COUNT(*) FROM live_cobra") == before
    assert not path.with_name("x.sqlite.building").exists()


def test_duplicate_ticker_in_a_run_is_rejected(locations, tmp_path):
    df = _cobra_rows(["AAA", "AAA"], "daily", "2026-11-06")
    df.to_csv(locations.logs["daily"] / "coiled_cobra_setups_2026-11-06.csv", index=False)
    with pytest.raises(sqlite3.IntegrityError):
        db.rebuild(tmp_path / "x.sqlite", locations)


def test_missing_folders_build_an_empty_database(tmp_path):
    empty = db.Locations(
        logs={m: tmp_path / "no" / m for m in db.MODES},
        history=tmp_path / "no",
        snapshots=tmp_path / "no",
    )
    assert set(db.rebuild(tmp_path / "x.sqlite", empty).values()) == {0}
