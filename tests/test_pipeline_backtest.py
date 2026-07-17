"""Tests for pipeline_backtest walk-forward simulation helpers."""

import pandas as pd
import pytest

import finance_vibe.pipeline_backtest as pb
from finance_vibe.pipeline_backtest import (
    passes_macro_gate,
    simulate_trade,
    simulate_scaled_trade,
)


def _bars(rows):
    """Build an OHLC frame from (open, high, low, close) tuples with a leading pad bar."""
    rows = [(100.0, 100.0, 100.0, 100.0)] + rows
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=len(rows), freq="D"),
        "Open": [r[0] for r in rows],
        "High": [r[1] for r in rows],
        "Low": [r[2] for r in rows],
        "Close": [r[3] for r in rows],
    })


def test_passes_macro_gate_long():
    assert passes_macro_gate("SETUP_LONG", 7, 7, -2) is True
    assert passes_macro_gate("SETUP_LONG", 5, 7, -2) is False


def test_passes_macro_gate_short():
    assert passes_macro_gate("SETUP_SHORT", -2, 7, -2) is True
    assert passes_macro_gate("SETUP_SHORT", 0, 7, -2) is False


def test_simulate_trade_long_stopped():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
        "High": [110.0, 101.0, 99.0, 100.0],
        "Low": [100.0, 100.0, 95.0, 96.0],
        "Close": [105.0, 100.0, 97.0, 98.0],
    })
    outcome, exit_date, exit_price, r_mult = simulate_trade(
        df,
        start_idx=1,
        is_long=True,
        entry=100.0,
        stop=96.0,
        target1=104.0,
        target2=108.0,
        entry_valid_bars=2,
        max_hold_bars=4,
    )
    assert outcome == "stopped"
    assert exit_price == 96.0
    assert r_mult == -1.0
    assert exit_date is not None


def test_simulate_trade_long_target1():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
        "High": [110.0, 101.0, 105.0, 106.0],
        "Low": [100.0, 99.0, 100.0, 101.0],
        "Close": [105.0, 100.0, 103.0, 104.0],
    })
    outcome, _, exit_price, r_mult = simulate_trade(
        df,
        start_idx=1,
        is_long=True,
        entry=100.0,
        stop=96.0,
        target1=104.0,
        target2=108.0,
        entry_valid_bars=2,
        max_hold_bars=4,
    )
    assert outcome == "target1"
    assert exit_price == 104.0
    assert r_mult == 1.0


def test_simulate_trade_short_r_multiple():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=4, freq="W"),
        "High": [100.0, 101.0, 102.0, 103.0],
        "Low": [90.0, 99.0, 94.0, 95.0],
        "Close": [95.0, 100.0, 96.0, 97.0],
    })
    outcome, _, exit_price, r_mult = simulate_trade(
        df,
        start_idx=1,
        is_long=False,
        entry=100.0,
        stop=104.0,
        target1=96.0,
        target2=92.0,
        entry_valid_bars=2,
        max_hold_bars=4,
    )
    assert outcome == "target1"
    assert exit_price == 96.0
    assert r_mult == 1.0


# ---------------------------------------------------------------------------
# simulate_scaled_trade: 50% at 1R, breakeven runner to 2R
# ---------------------------------------------------------------------------

def _run_scaled(rows, is_long, **kw):
    df = _bars(rows)
    return simulate_scaled_trade(
        df, start_idx=1, is_long=is_long,
        entry=kw.get("entry", 100.0), stop=kw.get("stop", 96.0),
        target1=kw.get("target1", 104.0), target2=kw.get("target2", 108.0),
        entry_valid_bars=kw.get("entry_valid_bars", 4),
        max_hold_bars=kw.get("max_hold_bars", 10),
        slippage_pct=kw.get("slippage_pct", 0.0),
        partial_fraction=kw.get("partial_fraction", 0.5),
    )


def test_scaled_partial_then_2r_runner():
    # fill bar, then 1R bar (partial), then 2R bar (runner) — blended +1.5R.
    res = _run_scaled(
        [(101, 102, 100, 101),   # fill at 100
         (102, 105, 101, 104),   # hits 1R (104); BE at 100 not touched
         (106, 109, 105, 108)],  # runner hits 2R (108)
        is_long=True,
    )
    assert res["outcome"] == "partial_t2"
    assert res["partial_r"] == pytest.approx(1.0)
    assert res["runner_r"] == pytest.approx(2.0)
    assert res["blended_r"] == pytest.approx(1.5)
    assert res["stop_moved_be"] is True


def test_scaled_partial_then_breakeven_runner():
    # partial at 1R, runner falls back to breakeven — blended +0.5R.
    res = _run_scaled(
        [(101, 102, 100, 101),   # fill at 100
         (102, 105, 101, 104),   # hits 1R -> partial, BE=100
         (101, 103, 99, 100)],   # runner taps BE (100)
        is_long=True,
    )
    assert res["outcome"] == "partial_be"
    assert res["partial_r"] == pytest.approx(1.0)
    assert res["runner_r"] == pytest.approx(0.0)
    assert res["blended_r"] == pytest.approx(0.5)


def test_scaled_full_stop_before_partial():
    res = _run_scaled(
        [(101, 102, 100, 101),   # fill at 100
         (99, 101, 95, 96)],     # stop 96 hit before any target
        is_long=True,
    )
    assert res["outcome"] == "stopped_full"
    assert res["blended_r"] == pytest.approx(-1.0)


def test_scaled_gap_through_stop_is_worse_than_1r():
    # Bar opens below the stop -> filled at the (worse) open, loss > 1R.
    res = _run_scaled(
        [(101, 102, 100, 101),   # fill at 100
         (94, 95, 93, 94)],      # gap down through 96 stop; open 94
        is_long=True,
    )
    assert res["outcome"] == "stopped_full"
    assert res["blended_r"] == pytest.approx((94 - 100) / 4)  # -1.5R
    assert res["blended_r"] < -1.0


def test_scaled_slippage_worsens_stop_loss():
    res = _run_scaled(
        [(101, 102, 100, 101),
         (99, 101, 95, 96)],
        is_long=True, slippage_pct=0.01,
    )
    # Entry filled higher and stop exit filled lower than planned -> worse than -1R.
    assert res["blended_r"] < -1.0


def test_scaled_same_bar_stop_and_target_is_pessimistic():
    # One bar touches both the stop and 1R target -> assume stop first.
    res = _run_scaled(
        [(101, 102, 100, 101),
         (99, 105, 95, 100)],   # High 105 >= 104 AND Low 95 <= 96
        is_long=True,
    )
    assert res["outcome"] == "stopped_full"
    assert res["blended_r"] == pytest.approx(-1.0)


def test_scaled_no_fill_when_entry_not_reached():
    res = _run_scaled(
        [(102, 103, 101, 102),
         (103, 104, 102, 103)],
        is_long=True,
    )
    assert res["outcome"] == "no_fill"
    assert res["blended_r"] is None


def test_scaled_short_partial_then_2r_runner():
    res = _run_scaled(
        [(99, 100, 98, 99),      # fill at 100 (High >= 100)
         (98, 99, 95, 96),       # hits 1R (96); BE 100 not touched (High 99)
         (94, 95, 91, 92)],      # runner hits 2R (92)
        is_long=False,
        entry=100.0, stop=104.0, target1=96.0, target2=92.0,
    )
    assert res["outcome"] == "partial_t2"
    assert res["blended_r"] == pytest.approx(1.5)


def test_scaled_tracks_mae_mfe():
    res = _run_scaled(
        [(101, 102, 100, 101),
         (102, 105, 101, 104),
         (106, 109, 105, 108)],
        is_long=True,
    )
    assert res["mfe_r"] >= res["blended_r"]
    assert res["mae_r"] <= 0.0


# ---------------------------------------------------------------------------
# backtest_ticker: no-overlap, cooldown-from-exit, long-only enforcement
# ---------------------------------------------------------------------------

def _write_flat_csv(tmp_path, n=120, low=95.0, high=105.0):
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "Open": [100.0] * n, "High": [high] * n, "Low": [low] * n,
        "Close": [100.0] * n, "Volume": [1_000_000] * n,
    })
    path = tmp_path / "XYZ_5y_1d.csv"
    df.to_csv(path, index=False)
    return str(path)


def _force_long_signal(monkeypatch, entry=100.0, stop=96.0, t1=104.0, t2=108.0):
    monkeypatch.setattr(pb, "detect_setup_at_bar",
                        lambda w, s, m, b=None: {"Setup Type": "SETUP_LONG", "Symbol": s,
                                                 "Regime OK": True, "RS 63d": 0.1})
    monkeypatch.setattr(pb, "build_features", lambda w: w)
    monkeypatch.setattr(pb, "score_last_row", lambda r: 10)
    monkeypatch.setattr(pb, "calculate_stock_levels",
                        lambda row, mode=None: (entry, stop, t1, t2, "CALL", (0.65, 0.8)))


def test_backtest_respects_cooldown_between_exits(tmp_path, monkeypatch):
    path = _write_flat_csv(tmp_path)
    _force_long_signal(monkeypatch)
    cooldown = 5
    trades, counts = pb.backtest_ticker(
        path, "high_beta", -10, -2, warmup=60, entry_valid=4, max_hold=10,
        cooldown_bars=cooldown,
    )
    assert counts["filled"] > 0
    # Every filled trade must start at least `cooldown` bars after the previous exit.
    dates = pd.to_datetime([t["Signal Date"] for t in trades]).sort_values()
    gaps = dates.to_series().diff().dropna().dt.days
    assert (gaps >= cooldown).all()


def test_backtest_no_fill_does_not_consume_cooldown(tmp_path, monkeypatch):
    path = _write_flat_csv(tmp_path)
    # Entry far below the flat Low (95) so nothing ever fills.
    _force_long_signal(monkeypatch, entry=50.0, stop=46.0, t1=54.0, t2=58.0)
    trades, counts = pb.backtest_ticker(
        path, "high_beta", -10, -2, warmup=60, entry_valid=4, max_hold=10,
        cooldown_bars=5,
    )
    assert counts["filled"] == 0
    assert counts["no_fill"] > 0
    assert counts["cooldown_skip"] == 0  # unfilled orders never start a cooldown
    assert trades == []


def test_backtest_long_only_skips_shorts(tmp_path, monkeypatch):
    path = _write_flat_csv(tmp_path)
    monkeypatch.setattr(pb, "detect_setup_at_bar",
                        lambda w, s, m, b=None: {"Setup Type": "SETUP_SHORT", "Symbol": s})
    monkeypatch.setattr(pb, "build_features", lambda w: w)
    monkeypatch.setattr(pb, "score_last_row", lambda r: -5)
    trades, counts = pb.backtest_ticker(
        path, "high_beta", -10, -2, warmup=60, entry_valid=4, max_hold=10,
        cooldown_bars=0,
    )
    assert counts["long_only_skip"] > 0
    assert counts["filled"] == 0
    assert trades == []
