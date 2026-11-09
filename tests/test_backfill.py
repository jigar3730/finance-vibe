"""Backfill runner: frozen universe only, one history file, a bad ticker can't stop it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.research import backfill as bf
from finance_vibe.research import snapshot as sn
from finance_vibe.scanners import coiled_cobra as cc


def _weekly(seed: int, n: int = 170) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0.002, 0.015, n)))
    return pd.DataFrame(
        {
            "Date": pd.date_range("2019-01-07", periods=n, freq="W-MON"),
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )


def _snapshot(tmp_path: Path, frames: dict[str, pd.DataFrame], members: list[str]) -> Path:
    folder = tmp_path / "snap"
    (folder / "weekly").mkdir(parents=True)
    names = []
    for symbol, frame in frames.items():
        name = f"weekly/{symbol}_10y_1wk.csv"
        frame.to_csv(folder / name, index=False)
        names.append(name)
    manifest = {"stamp": "2026-11-09", "files": {n: sn._sha256(folder / n) for n in names}}
    (folder / sn.MANIFEST).write_text(json.dumps(manifest))
    pd.DataFrame(
        {"Symbol": sorted(frames), "In Universe": [s in members for s in sorted(frames)]}
    ).to_csv(folder / sn.UNIVERSE, index=False)
    return folder


@pytest.fixture
def snap(tmp_path):
    frames = {"QQQ": _weekly(9), "AAA": _weekly(1), "BBB": _weekly(2), "OUT": _weekly(3)}
    return _snapshot(tmp_path, frames, ["AAA", "BBB"])


def _record(path: Path) -> dict:
    return json.loads(path.with_suffix("").with_suffix(".json").read_text())


def test_writes_only_universe_tickers(snap, tmp_path):
    path = bf.run_mode(snap, "weekly", out_root=tmp_path / "h", workers=1, commit="abc")
    assert path.parent.name == f"rubric-{config.RUBRIC_VERSION}"
    assert path.parent.parent.name == "2026-11-09"
    out = pd.read_csv(path)
    assert list(out.columns) == config.COBRA_HISTORY_COLUMNS
    assert set(out["Symbol"]) == {"AAA", "BBB"}
    assert len(out) == 2 * (170 - 160 + 1)
    record = _record(path)
    assert record["rows"] == len(out)
    assert record["tickers"] == 2
    assert record["failures"] == {}
    assert record["commit"] == "abc"
    assert not list(path.parent.glob("*.partial"))


def test_parallel_output_equals_serial(snap, tmp_path):
    one = bf.run_mode(snap, "weekly", out_root=tmp_path / "a", workers=1)
    two = bf.run_mode(snap, "weekly", out_root=tmp_path / "b", workers=2)
    pd.testing.assert_frame_equal(pd.read_csv(one), pd.read_csv(two))


def test_only_narrows_the_run(snap, tmp_path):
    path = bf.run_mode(snap, "weekly", out_root=tmp_path / "h", workers=1, only=["aaa", "OUT"])
    assert set(pd.read_csv(path)["Symbol"]) == {"AAA"}


def test_refuses_to_overwrite(snap, tmp_path):
    bf.run_mode(snap, "weekly", out_root=tmp_path / "h", workers=1)
    with pytest.raises(FileExistsError):
        bf.run_mode(snap, "weekly", out_root=tmp_path / "h", workers=1)


def test_bad_ticker_is_recorded_and_the_run_continues(snap, tmp_path, monkeypatch):
    real = cc.scan_history

    def flaky(bars, benchmark, symbol, mode):
        if symbol == "AAA":
            raise ValueError("boom")
        return real(bars, benchmark, symbol, mode)

    monkeypatch.setattr(cc, "scan_history", flaky)
    path = bf.run_mode(snap, "weekly", out_root=tmp_path / "h", workers=1)
    assert set(pd.read_csv(path)["Symbol"]) == {"BBB"}
    record = _record(path)
    assert record["failures"] == {"AAA": "ValueError: boom"}
    assert record["tickers"] == 1


def test_changed_snapshot_is_rejected(snap, tmp_path):
    path = snap / "weekly" / "AAA_10y_1wk.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        bf.run_mode(snap, "weekly", out_root=tmp_path / "h", workers=1)


def test_missing_benchmark_is_rejected(tmp_path):
    snap = _snapshot(tmp_path, {"AAA": _weekly(1)}, ["AAA"])
    with pytest.raises(ValueError, match="QQQ"):
        bf.run_mode(snap, "weekly", out_root=tmp_path / "h", workers=1)
