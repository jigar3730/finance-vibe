"""Coiled Cobra ``scan_history()``: every scored bar, same rules as the scan, no lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.scanners import coiled_cobra as cc

SIZES = {"weekly": 200, "daily": 830}
FLOORS = {"weekly": 160, "daily": 800}


def _bars(mode: str, seed: int = 1, n: int | None = None) -> pd.DataFrame:
    n = n or SIZES[mode]
    rng = np.random.default_rng(seed)
    if mode == "weekly":
        dates = pd.date_range("2019-01-07", periods=n, freq="W-MON")
    else:
        dates = pd.bdate_range("2019-01-01", periods=n)
    close = 50 * np.exp(np.cumsum(rng.normal(0.002, 0.015, n)))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": close * 1.012,
            "Low": close * 0.988,
            "Close": close,
            "Volume": rng.integers(800_000, 1_500_000, n).astype(float),
        }
    )


@pytest.fixture(autouse=True)
def _weekly_after_each_test():
    yield
    cc.apply_timeframe("weekly")


@pytest.mark.parametrize("mode", ["weekly", "daily"])
def test_shape_and_stamp(mode):
    out = cc.scan_history(_bars(mode), _bars(mode, seed=2), "abc", mode)
    assert list(out.columns) == config.COBRA_HISTORY_COLUMNS
    assert len(out) == SIZES[mode] - FLOORS[mode] + 1
    assert out["Date"].is_monotonic_increasing
    assert set(out["Symbol"]) == {"ABC"}
    assert set(out["Mode"]) == {mode}
    assert set(out["Rubric Version"]) == {config.RUBRIC_VERSION}


@pytest.mark.parametrize("mode", ["weekly", "daily"])
def test_each_row_equals_a_full_recompute_at_that_bar(mode):
    bars, bench = _bars(mode), _bars(mode, seed=2)
    out = cc.scan_history(bars, bench, "ABC", mode).set_index("Date")
    cc.apply_timeframe(mode)
    for end in range(FLOORS[mode], len(bars) + 1, 7):
        cut = bars.iloc[:end]
        day = cut["Date"].iloc[-1]
        expected = cc.evaluate_coiled_cobra(
            cc.add_macro_indicators(cut),
            bench[bench["Date"] <= day],
            include_rejects=True,
        )
        assert expected is not None
        row = out.loc[day]
        for col in ("Score", "Grade", "Checks Met"):
            assert row[col] == expected[col]
        assert row["Structure Points"] == expected["Parts"]["structure"]
        assert row["Relative Strength Points"] == expected["Parts"]["relative_strength"]


@pytest.mark.parametrize("mode", ["weekly", "daily"])
def test_later_bars_cannot_change_earlier_rows(mode):
    bars, bench = _bars(mode), _bars(mode, seed=2)
    full = cc.scan_history(bars, bench, "ABC", mode)
    shocked, shocked_bench = bars.copy(), bench.copy()
    last = len(bars) - 5
    prices = ["Open", "High", "Low", "Close"]
    shocked.loc[last:, prices] *= 3
    shocked.loc[last:, "Volume"] *= 50
    shocked_bench.loc[last:, prices] *= 0.2
    changed = cc.scan_history(shocked, shocked_bench, "ABC", mode)
    cutoff = bars["Date"].iloc[last - 1]
    pd.testing.assert_frame_equal(full[full["Date"] <= cutoff], changed[changed["Date"] <= cutoff])


@pytest.mark.parametrize("mode", ["weekly", "daily"])
def test_rows_match_a_run_scanner_replay(mode, tmp_path, monkeypatch):
    """Written and read back, a history row equals the --as-of scanner row for that date."""
    bars, bench = _bars(mode), _bars(mode, seed=2)
    history = cc.scan_history(bars, bench, "ABC", mode)

    cc.apply_timeframe(mode)
    raw_dir, logs = tmp_path / "raw", tmp_path / "logs"
    raw_dir.mkdir()
    bars.to_csv(raw_dir / "ABC_x_y.csv", index=False)
    (tmp_path / "active.csv").write_text("Ticker\nABC\n")
    real_evaluate = cc.evaluate_coiled_cobra

    def with_rejects(df, benchmark):  # the scanner only writes passes; we want every row
        return real_evaluate(df, benchmark, include_rejects=True)

    monkeypatch.setattr(cc, "RAW_DATA_DIR", str(raw_dir))
    monkeypatch.setattr(cc, "LOG_DIR", str(logs))
    monkeypatch.setattr(cc, "ACTIVE_TICKERS_PATH", str(tmp_path / "active.csv"))
    monkeypatch.setattr(cc, "load_benchmark_frame", lambda *a, **k: bench.copy())
    monkeypatch.setattr(cc, "evaluate_coiled_cobra", with_rejects)

    for i in (FLOORS[mode] + 3, len(bars) - 1):
        day = bars["Date"].iloc[i]
        as_of = (day + pd.Timedelta(days=4 if mode == "weekly" else 0)).strftime("%Y-%m-%d")
        cc.run_scanner(as_of=as_of)
        written = pd.read_csv(logs / f"coiled_cobra_setups_{as_of}.csv")
        assert len(written) == 1

        path = tmp_path / "row.csv"
        history[history["Date"] == day][config.SETUP_ROW_COLUMNS].to_csv(path, index=False)
        ours = pd.read_csv(path)
        columns = [c for c in config.SETUP_ROW_COLUMNS if c != "Setup Type"]
        pd.testing.assert_frame_equal(ours[columns], written[columns], check_dtype=False)


def test_passed_flag_and_setup_type_follow_the_grade(monkeypatch):
    calls = iter(range(10_000))

    def fake(df, benchmark, include_rejects=False):
        assert include_rejects
        passed = next(calls) % 2 == 0
        return {
            "Score": 80.0,
            "Grade": "B - Watch" if passed else "Rejected - Gate Fail (A)",
            "Tier": "Watchlist" if passed else None,
            "Checks Met": "4/6",
            "Fib Score": 0.0,
            "Parts": dict.fromkeys(cc._PILLAR_COLUMNS, 1),
            "RS 63d": 0.1,
            "RVOL": 1.0,
            "Market Gate": True,
            "BBWidth Pctile": 12.5,
        }

    monkeypatch.setattr(cc, "evaluate_coiled_cobra", fake)
    out = cc.scan_history(_bars("weekly"), _bars("weekly", seed=2), "ABC", "weekly")
    assert out["Passed"].iloc[0]
    assert out["Setup Type"].iloc[0] == "SETUP_LONG"
    assert not out["Passed"].iloc[1]
    assert pd.isna(out["Setup Type"].iloc[1])
    assert out["BBWidth Pctile"].iloc[0] == 12.5
    assert out["RVOL Trigger Points"].iloc[0] == 1


def test_unscoreable_bars_are_left_out(monkeypatch):
    monkeypatch.setattr(cc, "evaluate_coiled_cobra", lambda *a, **k: None)
    out = cc.scan_history(_bars("weekly"), _bars("weekly", seed=2), "ABC", "weekly")
    assert out.empty
    assert list(out.columns) == config.COBRA_HISTORY_COLUMNS


def test_bad_input_is_rejected_loudly():
    bars, bench = _bars("weekly"), _bars("weekly", seed=2)
    with pytest.raises(ValueError, match="insufficient bars"):
        cc.scan_history(bars.iloc[:100], bench, "ABC", "weekly")
    with pytest.raises(ValueError, match="benchmark"):
        cc.scan_history(bars, None, "ABC", "weekly")
    with pytest.raises(ValueError, match="unknown mode"):
        cc.scan_history(bars, bench, "ABC", "hourly")


def test_active_timeframe_is_restored():
    cc.apply_timeframe("weekly")
    cc.scan_history(_bars("daily"), _bars("daily", seed=2), "ABC", "daily")
    assert cc.mode == "weekly"
    assert cc.MIN_BARS_FULL_SCORE == 160
    with pytest.raises(ValueError):
        cc.scan_history(_bars("daily").iloc[:100], _bars("daily", seed=2), "ABC", "daily")
    assert cc.mode == "weekly"
