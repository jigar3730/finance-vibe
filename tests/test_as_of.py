"""``--as-of`` historical replay: parsing, no-lookahead cutting, and per-stage plumbing."""
from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from finance_vibe import analysis_engine as ae
from finance_vibe import breakout_scanner as bs
from finance_vibe import coiled_cobra as cc
from finance_vibe import config
from finance_vibe import run_vibe
from finance_vibe import trade_plan_helper as tph


# ---------------------------------------------------------------------------
# parse_as_of / run_stamp
# ---------------------------------------------------------------------------

def test_parse_as_of_accepts_both_forms_and_absent():
    assert config.parse_as_of(["weekly"]) is None
    assert config.parse_as_of([]) is None
    assert config.parse_as_of(["weekly", "--as-of", "2025-11-07"]) == "2025-11-07"
    assert config.parse_as_of(["--as-of=2025-11-07", "weekly"]) == "2025-11-07"
    assert config.parse_as_of(["--as-of", " 2025-11-07 "]) == "2025-11-07"


@pytest.mark.parametrize("bad", ["11/07/2025", "2025-13-01", "yesterday", "2025-11", ""])
def test_parse_as_of_rejects_malformed_dates(bad):
    with pytest.raises(ValueError, match="--as-of"):
        config.parse_as_of(["--as-of", bad])


def test_parse_as_of_rejects_missing_value_and_future_dates():
    with pytest.raises(ValueError, match="requires a date"):
        config.parse_as_of(["weekly", "--as-of"])
    with pytest.raises(ValueError, match="future"):
        config.parse_as_of(["--as-of", (date.today() + timedelta(days=1)).isoformat()])


def test_parse_as_of_ignores_the_environment(monkeypatch):
    monkeypatch.setenv("FINANCE_VIBE_AS_OF", "2025-11-07")
    assert config.parse_as_of([]) is None          # no env fallback: a stray var can't make a live run historical


def test_run_stamp():
    assert config.run_stamp("2025-11-07") == "2025-11-07"
    assert config.run_stamp(None) == date.today().isoformat()


# ---------------------------------------------------------------------------
# cut_to_as_of  (the no-lookahead rule)
# ---------------------------------------------------------------------------

def _weekly_frame(n=6, start="2025-10-06"):
    dates = pd.date_range(start, periods=n, freq="W-MON")      # Monday-dated weekly bars
    return pd.DataFrame({"Date": dates, "Close": np.arange(n, dtype=float)})


def test_weekly_bar_counts_only_once_its_friday_has_passed():
    df = _weekly_frame()                                        # 10-06, 10-13, 10-20, 10-27, 11-03, 11-10
    friday = config.cut_to_as_of(df, "2025-11-07", weekly=True)
    assert friday["Date"].iloc[-1] == pd.Timestamp("2025-11-03")      # week ending Fri 11-07 is complete
    assert len(friday) == 5


@pytest.mark.parametrize("as_of", ["2025-11-03", "2025-11-05", "2025-11-06"])
def test_weekly_mid_week_as_of_excludes_the_week_in_progress(as_of):
    cut = config.cut_to_as_of(_weekly_frame(), as_of, weekly=True)
    assert cut["Date"].iloc[-1] == pd.Timestamp("2025-10-27")         # 11-03 bar holds data through Fri 11-07


def test_weekly_weekend_as_of_includes_the_finished_week():
    cut = config.cut_to_as_of(_weekly_frame(), "2025-11-09", weekly=True)      # Sunday
    assert cut["Date"].iloc[-1] == pd.Timestamp("2025-11-03")


def test_friday_dated_weekly_bars_are_handled_too():
    dates = pd.date_range("2025-10-03", periods=5, freq="W-FRI")
    cut = config.cut_to_as_of(pd.DataFrame({"Date": dates, "Close": range(5)}), "2025-10-24", weekly=True)
    assert cut["Date"].iloc[-1] == pd.Timestamp("2025-10-24")


def test_daily_bar_counts_on_its_own_date():
    dates = pd.bdate_range("2025-11-03", periods=5)
    df = pd.DataFrame({"Date": dates, "Close": range(5)})
    assert config.cut_to_as_of(df, "2025-11-05", weekly=False)["Date"].iloc[-1] == pd.Timestamp("2025-11-05")
    assert len(config.cut_to_as_of(df, "2025-11-02", weekly=False)) == 0


def test_cut_handles_timezone_aware_and_string_dates():
    aware = _weekly_frame().assign(Date=lambda d: d["Date"].dt.tz_localize("America/New_York"))
    assert len(config.cut_to_as_of(aware, "2025-11-07", weekly=True)) == 5
    strings = _weekly_frame().assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"))
    assert len(config.cut_to_as_of(strings, "2025-11-07", weekly=True)) == 5


def test_cut_is_a_noop_without_as_of_and_never_falls_through_without_a_date_column():
    df = _weekly_frame()
    assert config.cut_to_as_of(df, None, weekly=True) is df
    with pytest.raises(ValueError, match="no Date column"):
        config.cut_to_as_of(df.rename(columns={"Date": "when"}), "2025-11-07", weekly=True)


# ---------------------------------------------------------------------------
# Coiled Cobra scanner plumbing
# ---------------------------------------------------------------------------

def _write_weekly_csv(path, n=200, mutate_after=None):
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    if mutate_after is not None:                      # rewrite the future: must never be seen by an earlier replay
        close[mutate_after:] *= 10
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Open": close, "High": close + 1, "Low": close - 1,
                  "Close": close, "Volume": 1_000_000.0}).to_csv(path, index=False)
    return dates


def _last_date(frame: pd.DataFrame) -> pd.Timestamp:
    """Last bar date the scanner handed to the evaluator (Date may still be a string)."""
    return pd.Timestamp(frame["Date"].iloc[-1])


@pytest.fixture
def cobra_env(tmp_path, monkeypatch):
    raw, logs = tmp_path / "raw", tmp_path / "logs"
    raw.mkdir(), logs.mkdir()
    (tmp_path / "active.csv").write_text("Ticker\nTEST\n")
    seen: list[pd.DataFrame] = []

    def fake_indicators(df):
        return df.assign(EMA20=df["Close"], EMA50=df["Close"], ATR=1.0, RSI=50.0, MACD=0.0,
                         MACD_Signal=0.0, Fib_618=np.nan, Fib_786=np.nan)

    def fake_evaluate(df, benchmark_df=None, **kw):
        seen.append(df.copy())
        return {"Score": 80.0, "Grade": "B - Watch", "Tier": "Watchlist", "Checks Met": "5/6",
                "Fib Score": 0.0, "RS 63d": 0.1, "RVOL": 1.0, "Market Gate": True}

    monkeypatch.setattr(cc, "RAW_DATA_DIR", str(raw))
    monkeypatch.setattr(cc, "LOG_DIR", str(logs))
    monkeypatch.setattr(cc, "ACTIVE_TICKERS_PATH", str(tmp_path / "active.csv"))
    monkeypatch.setattr(cc, "MIN_BARS_TO_EVALUATE", 60)
    monkeypatch.setattr(cc, "load_benchmark_frame", lambda *a, **k: None)
    monkeypatch.setattr(cc, "add_macro_indicators", fake_indicators)
    monkeypatch.setattr(cc, "evaluate_coiled_cobra", fake_evaluate)
    return {"raw": raw, "logs": logs, "seen": seen}


def test_cobra_as_of_uses_only_complete_bars_and_stamps_the_archive(cobra_env, monkeypatch):
    dates = _write_weekly_csv(cobra_env["raw"] / "TEST_10y_1wk.csv")

    def no_ml(*a, **k):
        raise AssertionError("ML ranking must be skipped on an as-of replay")

    import finance_vibe.ml_ranker as ml
    monkeypatch.setattr(ml, "attach_ml_ranks", no_ml)

    as_of = (dates[100] + pd.Timedelta(days=4)).date().isoformat()          # Friday of bar 100's week
    cc.run_scanner(as_of=as_of)

    assert _last_date(cobra_env["seen"][0]) == dates[100]
    out = pd.read_csv(cobra_env["logs"] / f"coiled_cobra_setups_{as_of}.csv")
    assert list(out["Symbol"]) == ["TEST"] and out["AsOf Date"].iloc[0] == str(dates[100].date())


def test_cobra_mid_week_as_of_drops_the_week_in_progress(cobra_env):
    dates = _write_weekly_csv(cobra_env["raw"] / "TEST_10y_1wk.csv")
    as_of = (dates[100] + pd.Timedelta(days=2)).date().isoformat()          # Wednesday of bar 100's week
    cc.run_scanner(as_of=as_of)
    assert _last_date(cobra_env["seen"][0]) == dates[99]


def test_cobra_replay_is_invariant_to_data_after_the_as_of_date(cobra_env):
    path = cobra_env["raw"] / "TEST_10y_1wk.csv"
    dates = _write_weekly_csv(path)
    as_of = (dates[120] + pd.Timedelta(days=4)).date().isoformat()
    cc.run_scanner(as_of=as_of)
    baseline = cobra_env["seen"][-1]

    _write_weekly_csv(path, mutate_after=121)                               # rewrite everything after the as-of week
    cc.run_scanner(as_of=as_of)
    pd.testing.assert_frame_equal(cobra_env["seen"][-1], baseline)


def test_cobra_live_scan_is_unchanged_without_as_of(cobra_env):
    dates = _write_weekly_csv(cobra_env["raw"] / "TEST_10y_1wk.csv")
    cc.run_scanner()
    assert _last_date(cobra_env["seen"][0]) == dates[-1]
    assert (cobra_env["logs"] / f"coiled_cobra_setups_{date.today().isoformat()}.csv").exists()


def test_cobra_as_of_before_enough_history_yields_no_setups(cobra_env):
    dates = _write_weekly_csv(cobra_env["raw"] / "TEST_10y_1wk.csv")
    as_of = (dates[30] + pd.Timedelta(days=4)).date().isoformat()            # only 31 bars exist by then (< 60)
    cc.run_scanner(as_of=as_of)
    assert cobra_env["seen"] == []
    assert len(pd.read_csv(cobra_env["logs"] / f"coiled_cobra_setups_{as_of}.csv")) == 0


# ---------------------------------------------------------------------------
# Breakout scanner + vibe report plumbing
# ---------------------------------------------------------------------------

def test_breakout_scan_files_cuts_each_frame_at_as_of(tmp_path, monkeypatch):
    dates = _write_weekly_csv(tmp_path / "TEST_10y_1wk.csv")
    seen = []

    class FakeEngine:
        def __init__(self, native_tf): pass

        def extract(self, raw, symbol, scan_mode):
            seen.append(pd.to_datetime(raw["Date"]).max())
            return symbol

    class FakeScorer:
        def evaluate(self, features): return {"Symbol": features}

    monkeypatch.setattr(bs, "FeatureEngine", FakeEngine)
    monkeypatch.setattr(bs, "ScoringEngine", FakeScorer)
    as_of = (dates[90] + pd.Timedelta(days=4)).date().isoformat()

    rows, _ = bs.scan_files(["TEST_10y_1wk.csv"], {"TEST"}, str(tmp_path), "weekly", "weekly", as_of=as_of)
    assert rows == [{"Symbol": "TEST"}] and seen == [dates[90]]
    bs.scan_files(["TEST_10y_1wk.csv"], {"TEST"}, str(tmp_path), "weekly", "weekly")
    assert seen[-1] == dates[-1]                                              # no as_of -> whole file


def test_vibe_report_scores_the_as_of_bar(tmp_path):
    path = tmp_path / "TEST_10y_1wk.csv"
    dates = _write_weekly_csv(path)
    as_of = (dates[150] + pd.Timedelta(days=4)).date().isoformat()
    expected = pd.read_csv(path)["Close"].iloc[150]
    assert ae.scan_one_file(str(path), as_of, True).price == pytest.approx(float(expected))
    assert ae.scan_one_file(str(path)).price == pytest.approx(float(pd.read_csv(path)["Close"].iloc[-1]))


# ---------------------------------------------------------------------------
# Trade plan helper: strict resolution (no silent fallback to another week)
# ---------------------------------------------------------------------------

def test_helper_strict_lookup_refuses_to_fall_back_to_another_weeks_plan():
    with pytest.raises(FileNotFoundError, match="--as-of"):
        tph.resolve_trade_plan_path("weekly", today="1999-01-01", strict=True)


def test_helper_main_validates_and_reports_missing_plan(capsys):
    assert tph.main(["weekly", "--as-of", "not-a-date"]) == 2
    assert tph.main(["weekly", "--as-of", "1999-01-01"]) == 1
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _run_orchestrator(monkeypatch, argv):
    cmds: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["run_vibe.py", *argv])
    monkeypatch.setattr(run_vibe.subprocess, "run", lambda cmd, **kw: cmds.append(cmd))
    return cmds


def test_orchestrator_as_of_implies_reuse_raw_and_passes_the_date_to_each_stage(monkeypatch):
    monkeypatch.setattr(run_vibe, "clean_raw_folder",
                        lambda *a, **k: pytest.fail("as-of must never wipe the raw data"))
    cmds = _run_orchestrator(monkeypatch, ["--as-of", "2025-11-07"])
    run_vibe.run_workflow()

    scripts = [c[1].rsplit("/", 1)[-1] for c in cmds]
    assert scripts == ["analysis_engine.py", "coiled_cobra.py", "breakout_scanner.py",
                       "trade_planner.py", "trade_plan_helper.py"]           # no ticker refresh / ingest
    assert all(c[-2:] == ["--as-of", "2025-11-07"] and c[2] == "weekly" for c in cmds)


def test_orchestrator_without_as_of_is_unchanged(monkeypatch):
    monkeypatch.setattr(run_vibe, "clean_raw_folder", lambda *a, **k: None)
    cmds = _run_orchestrator(monkeypatch, ["--reuse-raw"])
    run_vibe.run_workflow()
    assert len(cmds) == 5 and not any("--as-of" in c for c in cmds)


@pytest.mark.parametrize("bad", ["not-a-date", (date.today() + timedelta(days=3)).isoformat()])
def test_orchestrator_rejects_a_bad_as_of_before_running_anything(monkeypatch, bad):
    cmds = _run_orchestrator(monkeypatch, ["--as-of", bad])
    with pytest.raises(SystemExit) as exc:
        run_vibe.run_workflow()
    assert exc.value.code == 2 and cmds == []
