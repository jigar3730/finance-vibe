"""RESEARCH.md §7 statistics and the in-sample scorecard."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from finance_vibe.research import events as ev
from finance_vibe.research import insample as ins
from finance_vibe.research import stats

# ---------------------------------------------------------------------------
# t distribution and Holm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "df", "p"),
    [
        (1.812461, 10, 0.05),
        (2.228139, 10, 0.025),
        (2.763262, 28, 0.005),
        (1.660234, 100, 0.05),
        (1.0, 1, 0.25),  # Cauchy
        (0.0, 7, 0.5),
        (-1.812461, 10, 0.95),
        (1.644854, 1e7, 0.05),  # ~normal
    ],
)
def test_t_sf_matches_textbook_values(t, df, p):
    assert stats.t_sf(t, df) == pytest.approx(p, abs=2e-6)


@pytest.mark.parametrize(("t", "df"), [(0.5, 3), (2.1, 9), (2.76, 28), (3.4, 250)])
def test_t_sf_matches_numerical_integration(t, df):
    x = np.linspace(t, t + 400.0, 2_000_001)
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    tail = np.trapezoid(c * (1 + x * x / df) ** (-(df + 1) / 2), x)
    assert stats.t_sf(t, df) == pytest.approx(tail, rel=1e-5, abs=1e-9)


def test_holm_step_down_and_nan():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.2, 0.001, 0.5]) == pytest.approx([0.4, 0.003, 0.5])
    out = stats.holm([0.01, math.nan, 0.04])
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1]) and out[2] == pytest.approx(0.04)


def test_clustered_ttest_works_on_daily_means():
    dates = pd.Series(pd.to_datetime(["2023-01-03"] * 3 + ["2023-01-04", "2023-01-05"]))
    x = pd.Series([0.01, 0.03, 0.02, 0.05, -0.01])  # daily means 0.02, 0.05, -0.01
    res = stats.clustered_ttest(x, dates)
    daily = np.array([0.02, 0.05, -0.01])
    t = daily.mean() / (daily.std(ddof=1) / math.sqrt(3))
    assert res.n_dates == 3 and res.n_rows == 5
    assert res.t == pytest.approx(t) and res.p == pytest.approx(stats.t_sf(t, 2))
    assert res.mean == pytest.approx(x.mean())


def test_clustered_ttest_ignores_extra_rows_on_one_date():
    dates = pd.Series(pd.bdate_range("2023-01-02", periods=30))
    x = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 30))
    base = stats.clustered_ttest(x, dates)
    dup_d = pd.concat([dates, pd.Series([dates[0]] * 50)], ignore_index=True)
    dup_x = pd.concat([x, pd.Series([x[0]] * 50)], ignore_index=True)
    assert stats.clustered_ttest(dup_x, dup_d).p == pytest.approx(base.p)


def test_clustered_ttest_degenerate_inputs_give_nan():
    one = stats.clustered_ttest(
        pd.Series([0.1, 0.2]), pd.Series(pd.to_datetime(["2023-01-03"] * 2))
    )
    flat = stats.clustered_ttest(
        pd.Series([0.1, 0.1]), pd.Series(pd.bdate_range("2023-01-02", periods=2))
    )
    assert math.isnan(one.p) and math.isnan(flat.p)


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

DATES = pd.bdate_range("2023-01-02", periods=60)


def _frames(edge: dict[str, tuple[float, float]], seed: int = 0, n_draws: int = 5):
    """Events with R10 = signal; SPY at signal - edge_spy; draws centred on signal - edge_rand."""
    rng = np.random.default_rng(seed)
    ev_rows, dr_rows = [], []
    for event, (edge_spy, edge_rand) in edge.items():
        for i, d in enumerate(DATES):
            sym = f"{event}{i:02d}"
            r = 0.01 + 0.02 * rng.standard_normal()
            ev_rows.append({"Symbol": sym, "Date": d, "Event": event, "R10": r,
                            "SPY_R10": r - edge_spy + 0.002 * rng.standard_normal(),
                            "REACH10": (i % 2 == 0) if event == ev.E1 else None,
                            "split_suspect": i == 0})  # fmt: skip
            for k in range(n_draws):
                dr_rows.append({"event_symbol": sym, "Date": d, "Event": event, "draw": k,
                                "Symbol": f"R{k}", "split_suspect": False,
                                "R10": r - edge_rand + 0.002 * rng.standard_normal(),
                                "REACH10": (k == 0) if event == ev.E1 else None})  # fmt: skip
    events, draws = pd.DataFrame(ev_rows), pd.DataFrame(dr_rows)
    for f in (events, draws):
        f["REACH10"] = f["REACH10"].astype("boolean")
    return events, draws


def test_pass_needs_both_benchmarks():
    events, draws = _frames({ev.E1: (0.01, 0.01), ev.E2: (0.01, -0.01), ev.E3: (0.0, 0.0)})
    card = ins.scorecard(ins.with_benchmarks(events, draws))
    assert card.loc["PRE_BREAKOUT", "Result"] == "PASS"
    assert card.loc["TRIGGERED", "Result"] == "FAIL"  # beats SPY, loses to random
    assert card.loc["TRIGGERED", "p (larger)"] == card.loc["TRIGGERED", "p vs random"]
    assert card.loc["CONFIRMED", "Result"] == "FAIL"
    assert card.loc["PRE_BREAKOUT", "Mean − random %"] == pytest.approx(1.0, abs=0.1)
    holm = stats.holm(card["p (larger)"].tolist())
    assert card["p (Holm)"].tolist() == pytest.approx(holm, nan_ok=True)


def test_follow_through_only_for_pre_breakout():
    events, draws = _frames({ev.E1: (0.0, 0.0), ev.E2: (0.0, 0.0), ev.E3: (0.0, 0.0)})
    card = ins.scorecard(ins.with_benchmarks(events, draws))
    assert card.loc["PRE_BREAKOUT", "Follow-through %"] == pytest.approx(50.0)
    assert card.loc["PRE_BREAKOUT", "Random follow-through %"] == pytest.approx(20.0)
    assert card.loc[["TRIGGERED", "CONFIRMED"], "Follow-through %"].isna().all()


def test_random_excess_is_event_minus_mean_of_its_own_draws():
    events, draws = _frames({ev.E3: (0.0, 0.0)})
    e = ins.with_benchmarks(events, draws).set_index("Symbol")
    own = draws[draws["event_symbol"] == "E301"]["R10"].mean()
    assert e.loc["E301", "ex_rand"] == pytest.approx(e.loc["E301", "R10"] - own)


def test_by_year_and_robustness_shapes():
    events, draws = _frames({ev.E1: (0.01, 0.01), ev.E2: (0.0, 0.0), ev.E3: (0.0, 0.0)})
    years = ins.by_year(ins.with_benchmarks(events, draws))
    assert set(years["Year"]) == {2023} and "Result" not in years and "p (larger)" not in years
    robust = ins.scorecard(
        ins.with_benchmarks(events[~events["split_suspect"]], draws), decide=False
    )
    assert "Result" not in robust and (robust["Count"] == len(DATES) - 1).all()


def _write_labels(tmp_path, events, draws):
    events.to_csv(tmp_path / "events.csv.gz", index=False)
    draws.to_csv(tmp_path / "draws.csv.gz", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({"commit": "x"}))


def test_load_round_trips_booleans(tmp_path):
    events, draws = _frames({ev.E1: (0.0, 0.0), ev.E2: (0.0, 0.0)})
    events = events.assign(entry_date=events["Date"], exit_date=events["Date"])
    draws = draws.assign(entry_date=draws["Date"], exit_date=draws["Date"])
    _write_labels(tmp_path, events, draws)
    e, _, _ = ins.load(tmp_path)
    assert e["REACH10"].dtype == "boolean" and e["split_suspect"].dtype == bool
    assert e.loc[e["Event"] == ev.E2, "REACH10"].isna().all()
    assert int(e["split_suspect"].sum()) == int(events["split_suspect"].sum())


@pytest.mark.parametrize("which", ["events", "draws"])
def test_load_refuses_holdout_dates(tmp_path, which):
    events, draws = _frames({ev.E3: (0.0, 0.0)})
    frames = {"events": events, "draws": draws}
    frames[which].loc[0, "Date"] = pd.Timestamp("2026-04-01")
    for f in frames.values():
        f["entry_date"] = f["Date"]
        f["exit_date"] = f["Date"]
    _write_labels(tmp_path, frames["events"], frames["draws"])
    with pytest.raises(ValueError, match="holdout is sealed"):
        ins.load(tmp_path)


def _write_weekly(tmp_path, events, draws):
    from finance_vibe.research import labels as lb

    for f in (events, draws):
        f["entry_date"] = f["Date"]
        f["exit_date"] = f["Date"]
    events.rename(columns=lb.WEEKLY_NAMES).to_csv(tmp_path / "events.csv.gz", index=False)
    draws.rename(columns=lb.WEEKLY_NAMES).to_csv(tmp_path / "draws.csv.gz", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({"commit": "x", "mode": "weekly"}))


def test_weekly_load_renames_columns_back(tmp_path):
    events, draws = _frames({ev.E1: (0.01, 0.01)})
    _write_weekly(tmp_path, events, draws)
    e, d, m = ins.load(tmp_path)
    assert m["mode"] == "weekly" and {"R10", "SPY_R10", "REACH10"} <= set(e) and "R4W" not in e
    assert e["R10"].tolist() == pytest.approx(events["R10"].tolist())
    assert ins.scorecard(ins.with_benchmarks(e, d)).loc["PRE_BREAKOUT", "Result"] == "PASS"


def test_weekly_load_seals_on_the_week_end(tmp_path):
    events, draws = _frames({ev.E3: (0.0, 0.0)})
    events.loc[0, "Date"] = pd.Timestamp("2026-03-30")  # Monday; week closes 04-03
    _write_weekly(tmp_path, events, draws)
    with pytest.raises(ValueError, match="holdout is sealed"):
        ins.load(tmp_path)
