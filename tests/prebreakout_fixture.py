"""Shared builders for the RESEARCH.md §12 tests: synthetic bars, histories, a tiny snapshot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from finance_vibe import config
from finance_vibe.research import snapshot as sn
from finance_vibe.scanners import breakout as bs

DAYS = pd.bdate_range("2025-06-02", periods=240)  # runs past the 2026-03-31 seal
PRE, WATCH = bs.STATUS_PRE, bs.STATUS_WATCH
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SIGNAL_DAY = 100  # C1 and C2 on every snapshot ticker
SEALED_DAY = 230  # a Cobra pass after 2026-03-31: never labelled


def make_bars(seed: int = 0, days: pd.DatetimeIndex = DAYS, volume: float = 1e6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(days))))
    open_ = close * (1 + 0.003 * rng.standard_normal(len(days)))
    return pd.DataFrame(
        {
            "Date": days,
            "Open": open_,
            "High": np.maximum(open_, close) * 1.01,
            "Low": np.minimum(open_, close) * 0.99,
            "Close": close,
            "Volume": volume,
        }
    )


def make_universe(n: int = 30) -> dict[str, pd.DataFrame]:
    out = {f"T{i:02d}": make_bars(i) for i in range(n)}
    out["SPY"] = make_bars(999)
    return out


def make_cobra(symbol: str, passed: list[bool], days: pd.DatetimeIndex = DAYS) -> pd.DataFrame:
    return pd.DataFrame({"Date": days[: len(passed)], "Symbol": symbol, "Passed": passed})


def make_breakout(symbol: str, status: list[str], days: pd.DatetimeIndex = DAYS) -> pd.DataFrame:
    return pd.DataFrame({"Date": days[: len(status)], "Symbol": symbol, "Status": status})


def make_full_breakout(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """WATCH everywhere, Resistance 2% above and Support 2% below each close (no SPY)."""
    rows = [
        pd.DataFrame(
            {
                "Date": b["Date"],
                "Symbol": s,
                "Status": WATCH,
                "Resistance": b["Close"] * 1.02,
                "Support": b["Close"] * 0.98,
            }
        )
        for s, b in frames.items()
        if s != "SPY"
    ]
    return pd.concat(rows, ignore_index=True)


def write_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    """A verified 3-ticker snapshot (plus SPY) and both scanners' daily histories.

    Each ticker: Cobra passes on SIGNAL_DAY and the day after, and on SEALED_DAY;
    Status is PRE_BREAKOUT on SIGNAL_DAY only. So one C1 and one C2 per ticker in-sample.
    """
    frames = {"AAA": make_bars(1), "BBB": make_bars(2), "CCC": make_bars(3), "SPY": make_bars(999)}
    snap = tmp_path / "snap"
    (snap / "daily").mkdir(parents=True)
    names = []
    for s, b in frames.items():
        name = f"daily/{s}_10y_1d.csv"
        b.to_csv(snap / name, index=False)
        names.append(name)
    (snap / sn.MANIFEST).write_text(
        json.dumps({"stamp": "2026-11-09", "files": {n: sn._sha256(snap / n) for n in names}})
    )
    pd.DataFrame(
        {"Symbol": ["AAA", "BBB", "CCC", "SPY"], "In Universe": [True, True, True, False]}
    ).to_csv(snap / sn.UNIVERSE, index=False)

    history = tmp_path / "hist"
    cobra_dir = history / f"rubric-{config.RUBRIC_VERSION}"
    breakout_dir = history / f"rubric-{config.BREAKOUT_RUBRIC_VERSION}"
    cobra_dir.mkdir(parents=True)
    breakout_dir.mkdir(parents=True)
    passed = np.zeros(len(DAYS), dtype=bool)
    passed[[SIGNAL_DAY, SIGNAL_DAY + 1, SEALED_DAY]] = True
    tickers = ["AAA", "BBB", "CCC"]
    cobra = pd.concat([make_cobra(s, list(passed)) for s in tickers], ignore_index=True)
    cobra["Rubric Version"] = config.RUBRIC_VERSION
    breakout = make_full_breakout({s: frames[s] for s in tickers})
    breakout.loc[breakout["Date"] == DAYS[SIGNAL_DAY], "Status"] = PRE
    breakout["Rubric Version"] = config.BREAKOUT_RUBRIC_VERSION
    cobra.to_csv(cobra_dir / "coiled_cobra_daily_history.csv.gz", index=False)
    breakout.to_csv(breakout_dir / "breakout_daily_history.csv.gz", index=False)
    return snap, history


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
