"""RESEARCH.md §12 in-sample analysis: draws matched per event, §7 test, sealed holdout."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import prebreakout_fixture as pf
from finance_vibe.research import prebreakout as pb
from finance_vibe.research import prebreakout_insample as pi
from finance_vibe.research import stats
from finance_vibe.scanners import breakout as bs

DAYS = pf.DAYS


def _event(symbol, day, event, r10, spy=0.0, reach=True, level=pb.NEITHER, split=False):
    return {
        "Symbol": symbol,
        "Date": day,
        "Event": event,
        "R10": r10,
        "SPY_R10": spy,
        "REACH10": reach,
        pb.LEVEL_FIRST: level,
        "split_suspect": split,
    }


def _draw(event_symbol, day, event, symbol, r10, reach=False, level=pb.SUPPORT, split=False):
    return {
        "event_symbol": event_symbol,
        "Date": day,
        "Event": event,
        "Symbol": symbol,
        "R10": r10,
        "REACH10": reach,
        pb.LEVEL_FIRST: level,
        "split_suspect": split,
    }


def _frames(events: list[dict], draws: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    e, d = pd.DataFrame(events), pd.DataFrame(draws)
    for frame in (e, d):
        pi._clean(frame)
    return e, d


def test_draws_are_matched_to_their_own_event():
    day = DAYS[50]
    e, d = _frames(
        [_event("AAA", day, pb.C1, 0.05), _event("AAA", day, pb.C2, 0.05)],
        [_draw("AAA", day, pb.C1, "X", 0.01), _draw("AAA", day, pb.C2, "Y", 0.03)],
    )
    out = pi.with_benchmarks(e, d).set_index("Event")
    assert out.loc[pb.C1, "ex_rand"] == pytest.approx(0.04)
    assert out.loc[pb.C2, "ex_rand"] == pytest.approx(0.02)
    assert out.loc[pb.C1, "ex_spy"] == pytest.approx(0.05)


def _many(event: str, excess: float, n_days: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    events, draws = [], []
    for i in range(n_days):
        day = DAYS[i]
        r = excess + 0.01 * rng.standard_normal()
        events.append(_event("AAA", day, event, r, reach=i % 2 == 0, level=pb.RESISTANCE))
        draws.append(_draw("AAA", day, event, "X", 0.0))
    return events, draws


def test_scorecard_applies_the_frozen_rule():
    up_e, up_d = _many(pb.C1, 0.02)
    flat_e, flat_d = _many(pb.C2, -0.01, seed=1)
    e, d = _frames(up_e + flat_e, up_d + flat_d)
    card = pi.scorecard(pi.with_benchmarks(e, d))
    assert list(card.index) == list(pi.SIGNALS.values())
    c1, c2 = card.loc[pi.SIGNALS[pb.C1]], card.loc[pi.SIGNALS[pb.C2]]
    assert c1["Result"] == "PASS" and c2["Result"] == "FAIL"
    assert c1["p (larger)"] == max(c1["p vs SPY"], c1["p vs random"])
    assert list(card["p (Holm)"]) == pytest.approx(stats.holm(list(card["p (larger)"])))
    assert c1["Count"] == 60 and c1["Dates"] == 60


def test_reported_shares_against_the_draws():
    events, draws = _many(pb.C1, 0.0)
    events[0][pb.LEVEL_FIRST] = None  # missing level: left out of the shares
    e, d = _frames(events, draws)
    row = pi.reported(pi.with_benchmarks(e, d)).loc[pi.SIGNALS[pb.C1]]
    assert row["Reach breakout %"] == pytest.approx(50.0)
    assert row["Random reach %"] == pytest.approx(0.0)
    assert row["Resistance first %"] == pytest.approx(100.0)
    assert row["Random support first %"] == pytest.approx(100.0)


def test_by_year_is_unadjusted_and_per_signal():
    e, d = _frames(*_many(pb.C1, 0.01))
    years = pi.by_year(pi.with_benchmarks(e, d))
    assert set(years["Year"]) == {2025}
    assert "p vs SPY (unadj.)" in years.columns and "p (Holm)" not in years.columns


def _write_labels(folder, days):
    folder.mkdir()
    events = pd.DataFrame(
        [{**_event("AAA", d, pb.C1, 0.01), "entry_date": d, "exit_date": d} for d in days]
    )
    draws = pd.DataFrame(
        [{**_draw("AAA", d, pb.C1, "X", 0.0), "entry_date": d, "exit_date": d} for d in days]
    )
    events.to_csv(folder / "events.csv.gz", index=False)
    draws.to_csv(folder / "draws.csv.gz", index=False)
    (folder / "manifest.json").write_text(json.dumps({"commit": "x"}))


def test_load_accepts_2020_and_seals_the_holdout(tmp_path):
    _write_labels(tmp_path / "ok", [pd.Timestamp("2020-02-03")])
    events, _, _ = pi.load(tmp_path / "ok")
    assert len(events) == 1
    _write_labels(tmp_path / "sealed", [pd.Timestamp("2026-04-01")])
    with pytest.raises(ValueError, match="holdout is sealed"):
        pi.load(tmp_path / "sealed")


def test_reference_events_are_pre_breakout_first_bars_in_window():
    frames = pf.make_universe(3)
    members = pb.point_in_time_members(frames, [s for s in frames if s != "SPY"])
    breakout = pf.make_full_breakout(frames)
    t00 = breakout["Symbol"] == "T00"
    breakout.loc[t00 & breakout["Date"].isin(DAYS[[100, 101]]), "Status"] = bs.STATUS_PRE
    breakout.loc[t00 & (breakout["Date"] == DAYS[10]), "Status"] = bs.STATUS_PRE  # not a member yet
    breakout.loc[t00 & (breakout["Date"] == DAYS[230]), "Status"] = bs.STATUS_PRE  # sealed
    out = pi.reference_events(breakout, members)
    assert out.to_dict("records") == [
        {"Symbol": "T00", "Date": DAYS[100], "Event": pi.REFERENCE_EVENT}
    ]


def test_report_end_to_end(tmp_path, capsys):
    snap, history = pf.write_snapshot(tmp_path)
    common = ["--snapshot", str(snap), "--history", str(history), "--stamp", "x"]
    labels = pf.load_script("research_prebreakout_labels")
    assert labels.main([*common, "--out-dir", str(tmp_path), "--commit", "a"]) == 0
    capsys.readouterr()
    report = pf.load_script("prebreakout_insample_report")
    argv = [*common, "--labels-dir", str(tmp_path / "prebreakout_labels_x")]
    argv += ["--out-dir", str(tmp_path / "out"), "--commit", "b"]
    assert report.main(argv) == 0
    run = tmp_path / "out" / "prebreakout_insample_x"
    names = {
        "scorecard.csv",
        "reported.csv",
        "by_year.csv",
        "robustness.csv",
        "reference.csv",
        "report.md",
    }
    assert {p.name for p in run.iterdir()} == names
    card = pd.read_csv(run / "scorecard.csv")
    assert list(card["Signal"]) == list(pi.SIGNALS.values())
    assert list(card["Count"]) == [3, 3]
    assert set(card["Result"]) <= {"PASS", "FAIL"}
    assert pd.read_csv(run / "reference.csv")["Count"].iloc[0] == 3
    assert "Report code `b`" in (run / "report.md").read_text()
    with pytest.raises(FileExistsError):
        report.main(argv)


def test_report_requires_a_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    with pytest.raises(SystemExit):
        pf.load_script("prebreakout_insample_report").main(["--out-dir", str(tmp_path)])
