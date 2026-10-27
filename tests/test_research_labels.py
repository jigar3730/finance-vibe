"""Forward labels and benchmarks (RESEARCH.md §4-§5): no lookahead, sealed holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_vibe import config
from finance_vibe.research import events as ev
from finance_vibe.research import labels as lb
from finance_vibe.scanners import breakout as bs

DAYS = pd.bdate_range("2022-01-03", periods=300)
H = config.RESEARCH_HORIZON
T = 100  # signal bar index used throughout


def _bars(seed: int = 0, days: pd.DatetimeIndex = DAYS) -> pd.DataFrame:
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
            "Volume": 1e6,
        }
    )


def _universe(n: int = 30) -> dict[str, pd.DataFrame]:
    bars = {f"T{i:02d}": _bars(i) for i in range(n)}
    bars["SPY"] = _bars(999)
    return bars


def _members(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        pd.DataFrame({"Date": b["Date"], "Symbol": s, "in_universe": s != "SPY"})
        for s, b in bars.items()
    ]
    return pd.concat(rows, ignore_index=True)


def _history(bars: dict[str, pd.DataFrame], status: str = bs.STATUS_WATCH) -> pd.DataFrame:
    rows = [
        pd.DataFrame({"Date": b["Date"], "Symbol": s, "Status": status}) for s, b in bars.items()
    ]
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Entry and no lookahead
# ---------------------------------------------------------------------------


def test_entry_is_never_the_signal_bar():
    b = _bars()
    f = lb.forward_frame(b)
    valid = f["entry_date"].notna()
    assert (f.loc[valid, "entry_date"] > f.loc[valid, "Date"]).all()
    assert np.allclose(f["entry_open"].iloc[:-1], b["Open"].iloc[1:])
    row = f.iloc[T]
    assert row["entry_open"] not in (b["Open"].iloc[T], b["Close"].iloc[T])
    assert row["entry_date"] == b["Date"].iloc[T + 1]
    assert row["exit_date"] == b["Date"].iloc[T + H]
    assert row["R10"] == pytest.approx(b["Close"].iloc[T + H] / b["Open"].iloc[T + 1] - 1)


def test_labels_for_t_use_only_bars_after_t():
    b = _bars()
    base = lb.forward_frame(b).iloc[T]
    cols = ["Open", "High", "Low", "Close"]

    past = b.copy()
    past.loc[:T, cols] *= 3.7  # every bar up to and including t
    after_exit = b.copy()
    after_exit.loc[T + H + 1 :, cols] *= 0.2
    for changed in (past, after_exit):
        row = lb.forward_frame(changed).iloc[T]
        assert row["R10"] == pytest.approx(base["R10"])
        assert row["entry_open"] == pytest.approx(base["entry_open"])

    for idx, col in ((T + 1, "Open"), (T + H, "Close")):
        moved = b.copy()
        moved.loc[idx, col] *= 1.1
        assert lb.forward_frame(moved).iloc[T]["R10"] != pytest.approx(base["R10"])


def test_reach10_window_is_t_plus_1_to_t_plus_10():
    def reach_with_breakout_at(i: int) -> bool:
        hist = pd.DataFrame({"Date": DAYS, "Symbol": "X", "Status": bs.STATUS_WATCH})
        hist.loc[i, "Status"] = bs.STATUS_TRIGGERED
        return bool(lb.reach_frame(hist).iloc[T]["REACH10"])

    assert reach_with_breakout_at(T + 1)
    assert reach_with_breakout_at(T + H)
    assert not reach_with_breakout_at(T)
    assert not reach_with_breakout_at(T + H + 1)
    hist = pd.DataFrame({"Date": DAYS, "Symbol": "X", "Status": bs.STATUS_WATCH})
    hist.loc[T + 3, "Status"] = bs.STATUS_CONFIRMED
    assert lb.reach_frame(hist).iloc[T]["REACH10"]


def test_reach10_does_not_bleed_across_tickers():
    a = pd.DataFrame({"Date": DAYS[:50], "Symbol": "A", "Status": bs.STATUS_WATCH})
    b = pd.DataFrame({"Date": DAYS[:50], "Symbol": "B", "Status": bs.STATUS_TRIGGERED})
    r = lb.reach_frame(pd.concat([b, a], ignore_index=True)).set_index("Symbol")
    assert not r.loc["A", "REACH10"].any()


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------


def test_spy_return_uses_the_events_entry_and_exit_dates():
    bars = _universe(3)
    keys = pd.DataFrame({"Symbol": ["T01"], "Date": [DAYS[T]]})
    out = lb.labels_at(keys, bars, bars["SPY"]).iloc[0]
    spy = bars["SPY"]
    assert out["SPY_R10"] == pytest.approx(spy["Close"].iloc[T + H] / spy["Open"].iloc[T + 1] - 1)
    assert not out["spy_missing"]


def test_spy_missing_when_spy_lacks_the_exit_date():
    bars = _universe(3)
    bars["SPY"] = bars["SPY"].drop(index=T + H).reset_index(drop=True)
    keys = pd.DataFrame({"Symbol": ["T01"], "Date": [DAYS[T]]})
    out = lb.labels_at(keys, bars, bars["SPY"]).iloc[0]
    assert np.isnan(out["SPY_R10"]) and out["spy_missing"]


# ---------------------------------------------------------------------------
# Random draws
# ---------------------------------------------------------------------------


def _events(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": ["T00", "T05", "T07", "T05"],
            "Date": [DAYS[T], DAYS[T], DAYS[T], DAYS[T + 30]],
            "Event": [ev.E1, ev.E2, ev.E3, ev.E3],
        }
    )


def test_same_seed_gives_the_same_draws_in_any_input_order():
    bars = _universe()
    members, events = _members(bars), _events(bars)
    a = lb.draw_random(events, members, n=20, seed=7)
    b = lb.draw_random(events, members, n=20, seed=7)
    shuffled = lb.draw_random(
        events.sample(frac=1, random_state=1), members.sample(frac=1, random_state=2), n=20, seed=7
    )
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, shuffled)
    assert not a.equals(lb.draw_random(events, members, n=20, seed=8))


def test_draws_are_other_members_on_the_date_without_replacement():
    bars = _universe()
    members = _members(bars)
    members.loc[members["Symbol"].eq("T09") & members["Date"].eq(DAYS[T]), "in_universe"] = False
    draws = lb.draw_random(_events(bars), members, n=20, seed=1)
    assert (draws["Symbol"] != draws["event_symbol"]).all()
    assert not draws["Symbol"].isin(["SPY"]).any()
    assert not (draws["Symbol"].eq("T09") & draws["Date"].eq(DAYS[T])).any()
    per_event = draws.groupby(["event_symbol", "Date"])["Symbol"]
    assert (per_event.nunique() == 20).all() and (per_event.size() == 20).all()


def test_too_few_members_takes_all_and_flags_draws_short():
    bars = _universe(6)
    events = _events(bars).query("Symbol in @bars")
    ev_out, dr_out = lb.build_labels(events, _members(bars), bars, _history(bars))
    assert (ev_out["n_draws"] == 5).all() and ev_out["draws_short"].all()
    assert len(dr_out) == 5 * len(ev_out)


# ---------------------------------------------------------------------------
# Holdout seal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["2026-04-01", "2026-10-26", "2021-12-31"])
def test_out_of_sample_dates_raise(bad):
    with pytest.raises(ValueError):
        lb.check_in_sample([pd.Timestamp(bad)])


def test_last_in_sample_date_passes():
    lb.check_in_sample([pd.Timestamp(config.RESEARCH_IN_SAMPLE_END)])


def test_holdout_event_raises_everywhere():
    days = pd.bdate_range("2026-01-02", "2026-06-30")
    bars = {f"T{i:02d}": _bars(i, days) for i in range(25)} | {"SPY": _bars(999, days)}
    event = pd.DataFrame(
        {"Symbol": ["T00"], "Date": [pd.Timestamp("2026-04-01")], "Event": [ev.E3]}
    )
    with pytest.raises(ValueError, match="holdout is sealed"):
        lb.build_labels(event, _members(bars), bars, _history(bars))
    with pytest.raises(ValueError, match="holdout is sealed"):
        lb.draw_random(event, _members(bars))
    with pytest.raises(ValueError, match="holdout is sealed"):
        lb.labels_at(event, bars, bars["SPY"])


def test_late_march_event_may_read_april_bars():
    days = pd.bdate_range("2026-01-02", "2026-06-30")
    bars = {f"T{i:02d}": _bars(i, days) for i in range(25)} | {"SPY": _bars(999, days)}
    event = pd.DataFrame(
        {"Symbol": ["T00"], "Date": [pd.Timestamp("2026-03-31")], "Event": [ev.E3]}
    )
    ev_out, _ = lb.build_labels(event, _members(bars), bars, _history(bars))
    assert ev_out.loc[0, "exit_date"] > pd.Timestamp("2026-04-01")
    assert not ev_out.loc[0, "short_window"]


# ---------------------------------------------------------------------------
# Data-quality flags
# ---------------------------------------------------------------------------


def test_no_next_open_and_short_window_near_the_end():
    f = lb.forward_frame(_bars())
    last, five_left = f.iloc[-1], f.iloc[-6]
    assert last["no_next_open"] and last["short_window"] and np.isnan(last["R10"])
    assert (
        not five_left["no_next_open"] and five_left["short_window"] and five_left["fwd_bars"] == 5
    )
    assert not f.iloc[T][["no_next_open", "short_window"]].any()


def test_split_suspect_flags_a_two_for_one_jump_in_the_window():
    b = _bars()
    b.loc[T + 4 :, ["Open", "High", "Low", "Close"]] /= 2.0
    f = lb.forward_frame(b)
    assert f.iloc[T]["split_suspect"]
    assert f.iloc[T + 3]["split_suspect"]
    assert not f.iloc[T + 4]["split_suspect"]  # jump is on t itself, not after
    assert not f.iloc[T - H]["split_suspect"]


def test_session_gap_when_the_ticker_misses_a_spy_session():
    bars = _universe(3)
    bars["T01"] = bars["T01"].drop(index=T + 4).reset_index(drop=True)
    keys = pd.DataFrame({"Symbol": ["T01", "T02"], "Date": [DAYS[T], DAYS[T]]})
    out = lb.labels_at(keys, bars, bars["SPY"]).set_index("Symbol")
    assert out.loc["T01", "session_gap"] and not out.loc["T02", "session_gap"]


def test_key_without_a_bar_is_flagged_not_dropped():
    bars = _universe(3)
    saturday = pd.Timestamp("2022-05-28")
    keys = pd.DataFrame({"Symbol": ["T01", "T02"], "Date": [saturday, DAYS[T]]})
    out = lb.labels_at(keys, bars, bars["SPY"]).set_index("Symbol")
    assert len(out) == 2 and out.loc["T01", "no_next_open"] and np.isnan(out.loc["T01", "R10"])


# ---------------------------------------------------------------------------
# build_labels
# ---------------------------------------------------------------------------


def test_build_labels_shapes_reach_and_rubric_stamp():
    bars = _universe()
    events = _events(bars)
    ev_out, dr_out = lb.build_labels(events, _members(bars), bars, _history(bars))
    assert len(ev_out) == len(events) and len(dr_out) == 20 * len(events)
    assert (ev_out["Rubric Version"] == config.BREAKOUT_RUBRIC_VERSION).all()
    assert (dr_out["Rubric Version"] == config.BREAKOUT_RUBRIC_VERSION).all()
    assert ev_out.loc[ev_out["Event"] != ev.E1, "REACH10"].isna().all()
    assert ev_out.loc[ev_out["Event"] == ev.E1, "REACH10"].notna().all()
    assert dr_out.loc[dr_out["Event"] != ev.E1, "REACH10"].isna().all()
    assert set(lb.DQ_COLUMNS) <= set(ev_out.columns) and "draws_short" in ev_out


def test_build_labels_rejects_non_members():
    bars = _universe()
    members = _members(bars)
    members.loc[members["Symbol"].eq("T00") & members["Date"].eq(DAYS[T]), "in_universe"] = False
    with pytest.raises(ValueError, match="universe member"):
        lb.build_labels(_events(bars), members, bars, _history(bars))
