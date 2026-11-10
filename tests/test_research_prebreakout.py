"""RESEARCH.md §12 labels: event rules, point-in-time check, no lookahead, sealed holdout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.research import prebreakout as pb
from finance_vibe.research import snapshot as sn
from finance_vibe.scanners import breakout as bs

DAYS = pd.bdate_range("2025-06-02", periods=240)  # runs past the 2026-03-31 seal
H = config.RESEARCH_HORIZON
PRE, WATCH = bs.STATUS_PRE, bs.STATUS_WATCH


def _bars(seed: int = 0, days: pd.DatetimeIndex = DAYS, volume: float = 1e6) -> pd.DataFrame:
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


def _universe(n: int = 30) -> dict[str, pd.DataFrame]:
    bars = {f"T{i:02d}": _bars(i) for i in range(n)}
    bars["SPY"] = _bars(999)
    return bars


def _cobra(symbol: str, passed: list[bool], days=DAYS) -> pd.DataFrame:
    return pd.DataFrame({"Date": days[: len(passed)], "Symbol": symbol, "Passed": passed})


def _breakout(symbol: str, status: list[str], days=DAYS) -> pd.DataFrame:
    return pd.DataFrame({"Date": days[: len(status)], "Symbol": symbol, "Status": status})


def _full_breakout(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, b in bars.items():
        if s == "SPY":
            continue
        rows.append(
            pd.DataFrame(
                {
                    "Date": b["Date"],
                    "Symbol": s,
                    "Status": WATCH,
                    "Resistance": b["Close"] * 1.02,
                    "Support": b["Close"] * 0.98,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --- events ---------------------------------------------------------------


def test_c1_is_the_first_bar_of_a_cobra_pass():
    cobra = _cobra("AAA", [True, True, False, True, True])
    found = pb.events(cobra, _breakout("AAA", [WATCH] * 5))
    assert found.to_dict("records") == [{"Symbol": "AAA", "Date": DAYS[3], "Event": pb.C1}]


def test_c2_needs_both_and_counts_alongside_c1():
    cobra = _cobra("AAA", [False, True, True, True])
    breakout = _breakout("AAA", [PRE, WATCH, PRE, PRE])
    found = pb.events(cobra, breakout)
    assert list(zip(found["Date"], found["Event"], strict=True)) == [
        (DAYS[1], pb.C1),
        (DAYS[2], pb.C2),
    ]
    both = pb.events(_cobra("BBB", [False, True]), _breakout("BBB", [PRE, PRE]))
    assert list(both["Event"]) == [pb.C1, pb.C2]


def test_first_joined_row_never_counts():
    assert pb.events(_cobra("AAA", [True]), _breakout("AAA", [PRE])).empty


def test_events_are_per_ticker():
    cobra = pd.concat([_cobra("AAA", [True, True]), _cobra("BBB", [False, True])])
    breakout = pd.concat([_breakout("AAA", [WATCH] * 2), _breakout("BBB", [WATCH] * 2)])
    assert list(pb.events(cobra, breakout)["Symbol"]) == ["BBB"]


# --- point-in-time check --------------------------------------------------


def test_point_in_time_check_uses_only_past_bars():
    b = _bars(1)
    full = pb.point_in_time_members({"A": b}, ["A"])
    cut = pb.point_in_time_members({"A": b.iloc[:100]}, ["A"])
    pd.testing.assert_frame_equal(full.iloc[:100], cut)


def test_point_in_time_check_needs_the_full_window_and_thresholds():
    window = config.BACKFILL_DV_WINDOW
    liquid = pb.point_in_time_members({"A": _bars(1)}, ["A"])
    assert not liquid["in_universe"].iloc[: window - 1].any()
    assert liquid["in_universe"].iloc[window - 1 :].all()
    thin = pb.point_in_time_members({"A": _bars(1, volume=1_000)}, ["A"])
    assert not thin["in_universe"].any()
    cheap = _bars(1).assign(Close=4.0)
    assert not pb.point_in_time_members({"A": cheap}, ["A"])["in_universe"].any()


# --- level first ------------------------------------------------------------


def _flat(n: int = 30) -> pd.DataFrame:
    days = pd.bdate_range("2025-06-02", periods=n)
    return pd.DataFrame(
        {"Date": days, "Open": 10.0, "High": 10.5, "Low": 9.5, "Close": 10.0, "Volume": 1e6}
    )


def _level(b: pd.DataFrame, t: int, res: float, sup: float) -> str | None:
    levels = pd.DataFrame(
        {"Symbol": ["A"], "Date": [b["Date"][t]], "Resistance": [res], "Support": [sup]}
    )
    value = pb.level_first({"A": b}, levels)[pb.LEVEL_FIRST].iloc[0]
    return None if pd.isna(value) else value


def test_level_first_cases():
    b = _flat()
    assert _level(b, 5, 12.0, 8.0) == pb.NEITHER
    up = b.copy()
    up.loc[8, "High"] = 12.0
    assert _level(up, 5, 12.0, 8.0) == pb.RESISTANCE
    down = up.copy()
    down.loc[7, "Low"] = 8.0
    assert _level(down, 5, 12.0, 8.0) == pb.SUPPORT
    same = b.copy()
    same.loc[8, ["High", "Low"]] = [12.0, 8.0]
    assert _level(same, 5, 12.0, 8.0) == pb.SUPPORT
    assert _level(b, 5, float("nan"), 8.0) is None


def test_level_first_ignores_bar_t_and_bars_past_the_window():
    b = _flat()
    b.loc[5, "High"] = 12.0  # bar t itself
    b.loc[5 + H + 1, "High"] = 12.0  # one bar after the window
    assert _level(b, 5, 12.0, 8.0) == pb.NEITHER
    b.loc[5 + H, "High"] = 12.0  # last bar of the window
    assert _level(b, 5, 12.0, 8.0) == pb.RESISTANCE


def test_level_first_unknown_date_is_missing():
    b = _flat()
    levels = pd.DataFrame(
        {"Symbol": ["A", "Z"], "Date": [pd.Timestamp("2030-01-01"), b["Date"][3]],
         "Resistance": [12.0, 12.0], "Support": [8.0, 8.0]}
    )  # fmt: skip
    assert pb.level_first({"A": b}, levels)[pb.LEVEL_FIRST].isna().all()


# --- build_labels -----------------------------------------------------------


@pytest.fixture
def world():
    bars = _universe()
    symbols = [s for s in bars if s != "SPY"]
    members = pb.point_in_time_members(bars, symbols)
    return bars, members, _full_breakout(bars)


def _events(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"Symbol": s, "Date": DAYS[t], "Event": e} for s, t, e in rows])


def test_labels_match_the_definition(world):
    bars, members, breakout = world
    ev_out, dr_out = pb.build_labels(_events([("T01", 100, pb.C1)]), members, bars, breakout)
    b = bars["T01"]
    row = ev_out.iloc[0]
    assert row["R10"] == pytest.approx(b["Close"][100 + H] / b["Open"][101] - 1)
    assert row["entry_date"] == DAYS[101] and row["exit_date"] == DAYS[100 + H]
    assert row["n_draws"] == config.RESEARCH_RANDOM_DRAWS and not row["draws_short"]
    assert row[pb.LEVEL_FIRST] in {pb.RESISTANCE, pb.SUPPORT, pb.NEITHER}
    assert row["Rubric Version"] == config.RUBRIC_VERSION
    assert "T01" not in set(dr_out["Symbol"])
    assert dr_out[pb.LEVEL_FIRST].notna().all()
    assert dr_out["REACH10"].notna().all()


def test_c1_and_c2_on_one_bar_each_get_their_own_draws(world):
    bars, members, breakout = world
    ev_out, dr_out = pb.build_labels(
        _events([("T01", 100, pb.C1), ("T01", 100, pb.C2)]), members, bars, breakout
    )
    assert list(ev_out["n_draws"]) == [config.RESEARCH_RANDOM_DRAWS] * 2
    assert dr_out.groupby("Event").size().to_dict() == {pb.C1: 20, pb.C2: 20}


def test_labels_do_not_read_bars_after_the_window(world):
    bars, members, breakout = world
    one = _events([("T01", 100, pb.C1)])
    before, _ = pb.build_labels(one, members, bars, breakout)
    changed = dict(bars)
    later = bars["T01"].copy()
    later.loc[100 + H + 1 :, ["Open", "High", "Low", "Close"]] *= 3
    changed["T01"] = later
    after, _ = pb.build_labels(one, members, changed, breakout)
    cols = ["R10", "REACH10", pb.LEVEL_FIRST, "split_suspect"]
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_holdout_is_sealed(world):
    bars, members, breakout = world
    sealed = int(np.searchsorted(DAYS, pd.Timestamp("2026-04-01")))
    with pytest.raises(ValueError, match="holdout is sealed"):
        pb.build_labels(_events([("T01", sealed, pb.C1)]), members, bars, breakout)


def test_signal_before_the_start_is_rejected(world):
    bars, members, breakout = world
    with pytest.raises(ValueError, match="before the in-sample start"):
        pb.build_labels(_events([("T01", 100, pb.C1)]), members, bars, breakout, start="2026-01-01")


def test_default_start_allows_2020_to_2021():
    days = pd.bdate_range("2021-01-04", periods=150)
    bars = {f"T{i:02d}": _bars(i, days) for i in range(5)}
    bars["SPY"] = _bars(999, days)
    members = pb.point_in_time_members(bars, [s for s in bars if s != "SPY"])
    event = pd.DataFrame([{"Symbol": "T01", "Date": days[100], "Event": pb.C1}])
    ev_out, _ = pb.build_labels(event, members, bars, _full_breakout(bars))
    assert ev_out["R10"].notna().all()


def test_event_must_pass_the_point_in_time_check(world):
    bars, members, breakout = world
    with pytest.raises(ValueError, match="point-in-time"):
        pb.build_labels(_events([("T01", 10, pb.C1)]), members, bars, breakout)


def test_duplicate_events_are_rejected(world):
    bars, members, breakout = world
    with pytest.raises(ValueError, match="duplicate"):
        pb.build_labels(
            _events([("T01", 100, pb.C1), ("T01", 100, pb.C1)]), members, bars, breakout
        )


# --- the blind run script ---------------------------------------------------


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research_prebreakout_labels.py"
    spec = importlib.util.spec_from_file_location("research_prebreakout_labels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blind_run_writes_labels_and_counts_only(tmp_path, capsys):
    bars = {"AAA": _bars(1), "BBB": _bars(2), "CCC": _bars(3), "SPY": _bars(999)}
    snap = tmp_path / "snap"
    (snap / "daily").mkdir(parents=True)
    names = []
    for s, b in bars.items():
        name = f"daily/{s}_10y_1d.csv"
        b.to_csv(snap / name, index=False)
        names.append(name)
    (snap / sn.MANIFEST).write_text(
        json.dumps({"stamp": "2026-11-09", "files": {n: sn._sha256(snap / n) for n in names}})
    )
    pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC", "SPY"], "In Universe": [1, 1, 1, 0]}).astype(
        {"In Universe": bool}
    ).to_csv(snap / sn.UNIVERSE, index=False)

    history = tmp_path / "hist"
    (history / f"rubric-{config.RUBRIC_VERSION}").mkdir(parents=True)
    (history / f"rubric-{config.BREAKOUT_RUBRIC_VERSION}").mkdir(parents=True)
    passed = np.zeros(len(DAYS), dtype=bool)
    passed[[100, 101, 230]] = True  # day 230 is after the seal: never labelled
    status = np.full(len(DAYS), WATCH, dtype=object)
    status[100] = PRE
    cobra = pd.concat(
        [_cobra(s, list(passed)) for s in ["AAA", "BBB", "CCC"]], ignore_index=True
    ).assign(**{"Rubric Version": config.RUBRIC_VERSION})
    breakout = _full_breakout({s: bars[s] for s in ["AAA", "BBB", "CCC"]})
    breakout.loc[breakout["Date"] == DAYS[100], "Status"] = PRE
    breakout = breakout.assign(**{"Rubric Version": config.BREAKOUT_RUBRIC_VERSION})
    cobra.to_csv(
        history / f"rubric-{config.RUBRIC_VERSION}" / "coiled_cobra_daily_history.csv.gz",
        index=False,
    )
    breakout.to_csv(
        history / f"rubric-{config.BREAKOUT_RUBRIC_VERSION}" / "breakout_daily_history.csv.gz",
        index=False,
    )

    script = _load_script()
    out = tmp_path / "out"
    argv = ["--snapshot", str(snap), "--history", str(history), "--out-dir", str(out)]
    assert script.main([*argv, "--stamp", "x", "--commit", "abc"]) == 0
    run = out / "prebreakout_labels_x"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["rows"]["events_labelled"] == {pb.C1: 3, pb.C2: 3}
    assert manifest["max_event_date"] == str(DAYS[100].date())
    assert manifest["commit"] == "abc"
    events = pd.read_csv(run / "events.csv.gz")
    assert len(events) == 6 and set(events["n_draws"]) == {2}
    printed = capsys.readouterr().out
    assert "R10" not in printed and "LEVEL" not in printed and "resistance" not in printed
    with pytest.raises(FileExistsError):
        script.main([*argv, "--stamp", "x", "--commit", "abc"])


def test_blind_run_requires_a_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    with pytest.raises(SystemExit):
        _load_script().main(["--out-dir", str(tmp_path)])
