"""Point-in-time research universe and RESEARCH.md §3 events."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.holiday import USFederalHolidayCalendar

from finance_vibe import config
from finance_vibe.research import events as ev
from finance_vibe.research import universe as uv
from finance_vibe.scanners import breakout as bs

_HOLIDAYS = USFederalHolidayCalendar().holidays("2020-01-01", "2027-12-31")
DAYS = pd.bdate_range("2022-01-03", periods=520)
DAYS = DAYS[~DAYS.isin(_HOLIDAYS)][:500]
IPO_AT = 150  # IPOX lists on DAYS[150]
GAP = (300, 306)  # GAPY has no bars in DAYS[300:306]


def _panel(seed: int = 0, n: int = 30) -> pd.DataFrame:
    """n random tickers over DAYS, an IPO, a gappy ticker and a sub-$5 drifter."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        close = 20 * np.exp(np.cumsum(0.02 * rng.standard_normal(len(DAYS))))
        volume = rng.lognormal(13 + i * 0.05, 0.5, len(DAYS)).round()
        frames.append(pd.DataFrame({"Date": DAYS, "Symbol": f"T{i:02d}", "Close": close,
                                    "Volume": volume}))  # fmt: skip
    ipo = pd.DataFrame({"Date": DAYS[IPO_AT:], "Symbol": "IPOX", "Close": 50.0, "Volume": 1e9})
    gappy = frames[3].assign(Symbol="GAPY").drop(index=range(*GAP))
    cheap = pd.DataFrame(
        {"Date": DAYS, "Symbol": "LOWP", "Close": np.linspace(3.0, 6.0, len(DAYS)), "Volume": 1e8}
    )
    fund = frames[5].assign(Symbol="SPY", Volume=1e10)
    return pd.concat([*frames, ipo, gappy, cheap, fund], ignore_index=True)


def _build(panel: pd.DataFrame, **kw) -> pd.DataFrame:
    kw.setdefault("top_n", 12)
    return uv.build_membership(panel, **kw)


# ---------------------------------------------------------------------------
# No lookahead
# ---------------------------------------------------------------------------

CUTS = [
    DAYS[251],  # the 30 originals reach bar 252
    DAYS[252],
    DAYS[IPO_AT + 250],  # day before IPOX's 252nd bar
    DAYS[IPO_AT + 251],  # IPOX's 252nd bar
    DAYS[302],  # inside GAPY's gap
    DAYS[320] + pd.Timedelta(days=1),  # a date with no bar for anyone if a weekend
    DAYS[333],
    DAYS[-1],
]


@pytest.mark.parametrize("t", CUTS, ids=lambda t: str(t.date()))
def test_membership_at_t_ignores_later_data(t):
    panel = _panel()
    full = _build(panel)
    cut = _build(panel[panel["Date"] <= t])
    pd.testing.assert_frame_equal(
        full[full["Date"] <= t].reset_index(drop=True), cut, check_exact=True
    )


def test_prior_sessions_ignore_weekly_data_after_t():
    first = {"OLD": pd.Timestamp("2021-10-27")}
    weekly = pd.Series(pd.date_range("2018-01-01", "2026-10-19", freq="W-MON"))
    full = uv.prior_sessions(first, {"OLD": weekly})
    for t in ["2021-10-27", "2022-06-15", "2026-01-07"]:
        cut = uv.prior_sessions(first, {"OLD": weekly[weekly <= t]})
        assert cut == full


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_ties_break_by_symbol_including_at_the_cutoff():
    base = _panel()
    twin = base[base["Symbol"] == "T07"]
    tied = pd.concat([base, twin.assign(Symbol="AAAA"), twin.assign(Symbol="ZZZZ")])
    out = _build(tied, top_n=50)
    day = out[(out["Date"] == DAYS[400]) & out["Symbol"].isin(["AAAA", "T07", "ZZZZ"])]
    ranks = day.set_index("Symbol")["rank"]
    assert ranks["AAAA"] + 1 == ranks["T07"] and ranks["T07"] + 1 == ranks["ZZZZ"]

    cutoff = int(ranks["T07"])  # the cutoff falls inside the tie
    out = _build(tied, top_n=cutoff)
    members = out[out["Date"] == DAYS[400]].set_index("Symbol")["in_universe"]
    assert members["AAAA"] and members["T07"] and not members["ZZZZ"]


def test_input_row_order_does_not_matter():
    panel = _panel()
    shuffled = panel.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pd.testing.assert_frame_equal(_build(panel), _build(shuffled), check_exact=True)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_ipo_enters_only_after_252_bars():
    out = _build(_panel())
    ipo = out[out["Symbol"] == "IPOX"].reset_index(drop=True)
    assert ipo["Date"].iloc[0] == DAYS[IPO_AT]
    assert not ipo["in_universe"].iloc[:251].any()
    assert (ipo["reason"].iloc[:251] == "history").all()
    assert ipo["in_universe"].iloc[251:].all()  # 1e9 volume: rank 1 once eligible
    assert ipo["bars"].iloc[251] == config.RESEARCH_MIN_HISTORY_BARS


def test_weekly_history_before_the_daily_window_counts():
    first_daily = pd.Timestamp("2022-01-03")  # a Monday
    weekly = {
        "OLD": pd.Series(pd.date_range("2019-01-07", "2023-01-02", freq="W-MON")),
        "MIDWK": pd.Series(pd.DatetimeIndex(["2021-12-27", "2022-01-03"])),  # listed late Dec
    }
    prior = uv.prior_sessions({"OLD": first_daily, "MIDWK": first_daily, "NOWK": first_daily},
                              weekly)  # fmt: skip
    weeks_before = len(pd.date_range("2019-01-07", "2021-12-27", freq="W-MON"))
    assert prior["OLD"] == uv.SESSIONS_PER_WEEK * weeks_before
    assert prior["NOWK"] == 0

    # A Wednesday IPO's Monday-dated first weekly bar ends after it: adds nothing.
    wed = pd.Timestamp("2022-01-05")
    assert uv.prior_sessions({"X": wed}, {"X": pd.Series([pd.Timestamp("2022-01-03")])}) == {"X": 0}
    assert prior["MIDWK"] == 5  # the full week ending 2021-12-31 did end before 2022-01-03

    panel = _panel()
    out = _build(panel, prior={"T00": prior["OLD"]})
    t00 = out[out["Symbol"] == "T00"].reset_index(drop=True)
    # Long listing: only the 20-bar dollar-volume window holds it back.
    assert (t00["reason"].iloc[:19] == "dv_window").all()
    assert t00["reason"].iloc[19] in {"member", "rank"}


@pytest.mark.parametrize(("close", "ok"), [(4.99, False), (5.0, True), (5.01, True)])
def test_price_floor_is_inclusive(close, ok):
    panel = _panel()
    day = DAYS[400]
    panel.loc[(panel["Symbol"] == "T29") & (panel["Date"] == day), "Close"] = close
    row = _build(panel, top_n=100).query("Symbol == 'T29' and Date == @day").iloc[0]
    assert (row["reason"] != "price") is ok
    assert row["close"] == close


def test_cheap_stock_enters_when_it_crosses_five_dollars():
    out = _build(_panel(), top_n=100)
    low = out[out["Symbol"] == "LOWP"]
    eligible_age = low["bars"] >= 252
    assert set(low.loc[eligible_age & (low["close"] < 5), "reason"]) == {"price"}
    assert low.loc[eligible_age & (low["close"] >= 5), "in_universe"].all()


def test_dollar_volume_is_the_trailing_median_not_the_mean():
    panel = _panel()
    day = DAYS[400]
    spike = panel["Symbol"].eq("T00") & panel["Date"].eq(day)
    panel.loc[spike, "Volume"] *= 100
    row = _build(panel).query("Symbol == 'T00' and Date == @day").iloc[0]
    t00 = panel[panel["Symbol"].eq("T00") & panel["Date"].le(day)].tail(20)
    window = (t00["Close"] * t00["Volume"]).to_numpy()
    assert row["dv_median20"] == np.median(window)
    assert row["dv_median20"] < window.mean() / 3  # the spike dominates a mean, not a median


def test_excluded_symbols_never_rank():
    out = _build(_panel())
    spy = out[out["Symbol"] == "SPY"]
    assert (spy["reason"] == "excluded").all() and spy["rank"].isna().all()
    assert not spy["in_universe"].any()


def test_gap_days_have_no_rows():
    out = _build(_panel())
    gappy = out[out["Symbol"] == "GAPY"]
    assert not gappy["Date"].isin(DAYS[GAP[0] : GAP[1]]).any()


def test_shape_and_rank_invariants():
    top_n = 12
    out = _build(_panel(), top_n=top_n)
    assert list(out.columns) == uv.MEMBERSHIP_COLUMNS
    assert not out.duplicated(["Date", "Symbol"]).any()
    assert out.equals(out.sort_values(["Date", "Symbol"], kind="stable").reset_index(drop=True))
    assert set(out["reason"]) <= set(uv.REASONS)
    for _, day in out.groupby("Date"):
        ranked = day["rank"].dropna().sort_values().astype(int).tolist()
        assert ranked == list(range(1, len(ranked) + 1))  # contiguous among eligible
        assert day["in_universe"].sum() == min(top_n, len(ranked))
        assert (day["in_universe"] == (day["reason"] == "member")).all()


def test_rejects_malformed_panels():
    panel = _panel()
    with pytest.raises(ValueError, match="missing columns"):
        uv.build_membership(panel.drop(columns="Volume"))
    with pytest.raises(ValueError, match="duplicate"):
        uv.build_membership(pd.concat([panel, panel.iloc[:1]]))


def test_excluded_symbols_constant_covers_static_tickers_and_gdx():
    assert set(config.STATIC_TICKERS) <= config.RESEARCH_EXCLUDED_SYMBOLS
    assert "GDX" in config.RESEARCH_EXCLUDED_SYMBOLS


_GOLDEN_DAILY = Path("/app/data/golden/fixture/raw/daily")


@pytest.mark.skipif(not _GOLDEN_DAILY.is_dir(), reason="golden fixture not mounted")
def test_golden_fixture_membership_has_no_lookahead():
    panel = uv.panel_from_bars(uv.load_daily_bars(str(_GOLDEN_DAILY)))
    full = uv.build_membership(panel)
    for t in ["2023-03-15", "2024-07-03", "2025-12-31"]:
        cut = uv.build_membership(panel[panel["Date"] <= t])
        want = full[full["Date"] <= t].reset_index(drop=True)
        pd.testing.assert_frame_equal(want, cut, check_exact=True)


# ---------------------------------------------------------------------------
# RESEARCH.md §3 events
# ---------------------------------------------------------------------------

W, P, D = bs.STATUS_WATCH, bs.STATUS_PRE, bs.STATUS_DEV
T, C, F = bs.STATUS_TRIGGERED, bs.STATUS_CONFIRMED, bs.STATUS_FAILED


def _history(*tickers: tuple[str, list[str]]) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"Date": DAYS[: len(s)], "Symbol": sym, "Status": s}) for sym, s in tickers
    ]
    return pd.concat(frames, ignore_index=True)


def _events(*tickers):
    out = ev.first_bar_events(_history(*tickers))
    return [(r.Symbol, DAYS.get_loc(r.Date), r.Event) for r in out.itertuples()]


def test_breakout_events_need_a_non_breakout_previous_bar():
    assert _events(("A", [W, T, C, T, W, C, C, F, T])) == [
        ("A", 1, ev.E2),
        ("A", 5, ev.E3),
        ("A", 8, ev.E2),
    ]


def test_pre_breakout_event_is_first_bar_of_a_run():
    assert _events(("A", [W, P, P, D, P, T])) == [("A", 1, ev.E1), ("A", 4, ev.E1), ("A", 5, ev.E2)]


def test_first_history_row_never_counts_and_tickers_do_not_bleed():
    # B's first row is CONFIRMED; A's last row (TRIGGERED) is not B's previous bar.
    assert _events(("A", [W, W, T]), ("B", [C, C, W, C])) == [
        ("A", 2, ev.E2),
        ("B", 3, ev.E3),
    ]


def test_events_keep_other_columns_and_row_order_does_not_matter():
    hist = _history(("A", [W, P, T, C]), ("B", [W, C])).assign(Readiness=range(6))
    out = ev.first_bar_events(hist.sample(frac=1.0, random_state=1))
    assert list(out.columns) == ["Date", "Symbol", "Event", "Status", "Readiness"]
    assert out["Event"].tolist() == [ev.E1, ev.E2, ev.E3]
