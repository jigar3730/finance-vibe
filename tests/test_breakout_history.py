"""Breakout ``scan_history()``: per-bar states/status/score with no lookahead.

Four guarantees, each with its own oracle:

* **Rules port** -- the vectorized rules equal a frozen copy of the original
  scalar rules (``breakout_reference.py``) on random and threshold-edge inputs.
* **Forming higher-timeframe bar** -- on daily data, the weekly/monthly SMAs and
  trend flags at date t equal a brute-force recompute on data cut at t (the
  bar as it was forming at t, never the completed week/month).
* **Parity** -- the last row of ``scan_history()`` equals the row
  ``run_scanner()`` writes, cell for cell and byte for byte in the CSV.
* **No lookahead** -- ``scan_history()`` on full data, truncated at t, equals
  ``scan_history()`` on data cut at t, for mid-week, holiday, month-end and
  halt-gap dates.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.holiday import USFederalHolidayCalendar

import breakout_reference as ref
from finance_vibe import config
from finance_vibe.data import raw
from finance_vibe.scanners import breakout as bs

# ---------------------------------------------------------------------------
# Synthetic market data (seeded; runs anywhere)
# ---------------------------------------------------------------------------

_HOLIDAYS = USFederalHolidayCalendar().holidays("2020-01-01", "2027-12-31")
HALT = ("2024-05-28", "2024-06-07")  # spans a week end and the May month end


def _sessions(start: str, n: int, *, halt: tuple[str, str] | None = HALT) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, periods=int(n * 1.2))
    days = days[~days.isin(_HOLIDAYS)]
    if halt is not None:
        days = days[(days < halt[0]) | (days > halt[1])]
    return days[:n]


def _daily_bars(
    seed: int,
    *,
    n: int = 1150,
    start: str = "2021-03-17",  # a Wednesday, mid-month
    drift: float = 0.0004,
    flat: tuple[int, int] | None = (300, 330),
    coil: tuple[int, int] | None = (700, 760),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _sessions(start, n)
    vol = np.full(n, 0.015)
    if coil is not None:  # quiet stretch, then a burst: exercises compression/breakout states
        vol[coil[0] : coil[1]] = 0.003
        vol[coil[1] : coil[1] + 10] = 0.03
    close = np.round(20.0 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n))), 2)
    if flat is not None:  # identical closes: SMA ties and the SMA20 >= previous edge
        close[flat[0] : flat[1]] = close[flat[0]]
    opn = np.round(np.r_[close[0], close[:-1]] * (1 + 0.003 * rng.standard_normal(n)), 2)
    wick = np.abs(0.006 * rng.standard_normal(n))
    high = np.round(np.maximum(opn, close) * (1 + wick), 2)
    low = np.round(np.minimum(opn, close) * (1 - wick), 2)
    if flat is not None:
        sl = slice(flat[0], flat[1])
        opn[sl] = high[sl] = low[sl] = close[sl]
    volume = rng.integers(100_000, 2_000_000, n).astype(float)
    volume[rng.choice(n, 8, replace=False)] = 0.0
    return pd.DataFrame(
        {"Date": dates, "Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def _weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Monday-dated weekly bars, as the weekly ingest writes them."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    week = daily.set_index("Date").resample("W-FRI").agg(agg).dropna(subset=["Close"])
    week.index = week.index - pd.Timedelta(days=4)
    return week.rename_axis("Date").reset_index()


TICKERS = {
    "BULL": _daily_bars(1, drift=0.0008),
    "BEAR": _daily_bars(2, drift=-0.0004),
    "CHOP": _daily_bars(3, drift=0.0),
    "MIXD": _daily_bars(4, drift=0.0003, coil=(400, 470)),
    "LATE": _daily_bars(5, start="2021-11-03", n=900),
}
SHORT = _daily_bars(6, n=60, start="2025-03-05", flat=None, coil=None)  # always < 80 bars


def _bars(symbol: str, native_tf: str) -> pd.DataFrame:
    daily = SHORT if symbol == "SHRT" else TICKERS[symbol]
    return daily if native_tf == "daily" else _weekly_from_daily(daily)


def _clean(symbol: str, native_tf: str) -> pd.DataFrame:
    return bs.normalize_ohlcv(_bars(symbol, native_tf))


# ---------------------------------------------------------------------------
# Cell-exact comparison
# ---------------------------------------------------------------------------


def _is_missing(v) -> bool:
    return v is None or (not isinstance(v, str) and bool(pd.isna(v)))


def _same_cell(a, b) -> bool:
    if _is_missing(a) or _is_missing(b):
        return _is_missing(a) and _is_missing(b)
    bools = (bool, np.bool_)
    if isinstance(a, bools) or isinstance(b, bools):
        return isinstance(a, bools) and isinstance(b, bools) and bool(a) == bool(b)
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    return float(a) == float(b)  # exact, no tolerance


def _assert_rows_equal(got: pd.Series | dict, want: pd.Series | dict, where: str) -> None:
    diffs = [
        f"{col}: {got[col]!r} != {want[col]!r}"
        for col in bs.OUTPUT_COLUMNS
        if not _same_cell(got[col], want[col])
    ]
    assert not diffs, f"{where}:\n  " + "\n  ".join(diffs)


def _replay_row(clean_cut: pd.DataFrame, symbol: str, mode: str, native_tf: str) -> dict:
    """What run_scanner writes for this ticker on the cut data (the per-row path)."""
    feat = bs.FeatureEngine(native_tf).extract(clean_cut, symbol, mode)
    return bs.ScoringEngine().evaluate(feat)


# ---------------------------------------------------------------------------
# 1. Rules port: vectorized rules == frozen scalar reference
# ---------------------------------------------------------------------------

_EDGES: dict[str, list[float]] = {
    "distance_resistance_atr": [0.0, 0.25, 0.75, 1.25, 1.5, 2.5, -0.35, -0.3, -1.0, 3.0],
    "extension_atr": [0.0, 1.5, 2.0, 2.5, -0.5, 3.0],
    "bb_width_pctl": [15, 20, 25, 30, 35, 40, 50, 70, 0, 100],
    "kc_width_pctl": [15, 30, 31, 50],
    "atr_pctl": [15, 20, 25, 35, 40, 50, 70, 100],
    "range20_pctl": [20, 35, 40, 50, 51],
    "rvol20": [0.0, 0.85, 1.0, 1.2, 1.5, 2.0, 0.5, 3.0],
    "rsi": [40, 45, 50, 65, 70, 75, 30, 80],
    "rsi_slope": [0.0, 2.0, -1.0, 1.0, 5.0],
    "macd_hist_slope": [0.0, 0.1, -0.1],
    "macd_hist": [0.0, 0.2, -0.2],
    "close": [9.5, 10.0, 10.5],
    "support": [9.5, 10.0, 10.5],
    "sma50": [9.5, 10.0, 10.5],
    "sma200": [9.5, 10.0, 10.5],
}
_RANGES: dict[str, tuple[float, float]] = {
    "distance_resistance_atr": (-3, 4),
    "extension_atr": (-3, 4),
    "bb_width_pctl": (0, 100),
    "kc_width_pctl": (0, 100),
    "atr_pctl": (0, 100),
    "range20_pctl": (0, 100),
    "rvol20": (0, 4),
    "rsi": (10, 95),
    "rsi_slope": (-8, 8),
    "macd_hist_slope": (-1, 1),
    "macd_hist": (-1, 1),
    "close": (8, 12),
    "support": (8, 12),
    "sma50": (8, 12),
    "sma200": (8, 12),
}
_BOOLS = [
    f.name
    for f in dataclasses.fields(bs.BreakoutFeatures)
    if f.name.endswith(("_bull", "_bear"))
    or f.name
    in {
        "squeeze",
        "compression",
        "volume_dryup",
        "obv_rising",
        "breakout_triggered",
        "breakout_confirmation",
        "failed_breakout",
        "wick_reject",
    }
]


def _random_features(rng: np.random.Generator) -> bs.BreakoutFeatures:
    values: dict = {f.name: None for f in dataclasses.fields(bs.BreakoutFeatures)}
    values.update(symbol="X", mode="daily", asof="2026-01-02", has_daily=bool(rng.integers(2)))
    for name, edges in _EDGES.items():
        u = rng.random()
        if u < 0.15:
            values[name] = None
        elif u < 0.6:
            values[name] = float(edges[rng.integers(len(edges))])
        else:
            lo, hi = _RANGES[name]
            values[name] = float(np.round(rng.uniform(lo, hi), 4))
    for name in _BOOLS:
        values[name] = [None, True, False][rng.integers(3)]
    return bs.BreakoutFeatures(**values)


# Each condition of PRE_BREAKOUT (an 8-way AND that uniform draws almost never meet).
_PRE_SETUP = [
    {"failed_breakout": False, "breakout_confirmation": False, "breakout_triggered": False},
    {"daily_trend_bull": True, "weekly_trend_bull": True, "monthly_trend_bull": True},
    {"compression": True},
    {"rsi_slope": 2.0, "macd_hist_slope": 0.1},
    {"distance_resistance_atr": 1.25},
    {"rvol20": 1.2},
    {"close": 10.0, "support": 10.0},
]


def _near_pre_breakout(rng: np.random.Generator) -> bs.BreakoutFeatures:
    """Random features with most PRE_BREAKOUT conditions set: hits and near misses."""
    feat = _random_features(rng)
    overrides: dict = {}
    for group in _PRE_SETUP:
        if rng.random() < 0.9:
            overrides.update(group)
    return dataclasses.replace(feat, **overrides)


def _random_cases(n: int, seed: int) -> list[bs.BreakoutFeatures]:
    rng = np.random.default_rng(seed)
    return [_random_features(rng) if i % 3 else _near_pre_breakout(rng) for i in range(n)]


def test_vectorized_rules_match_reference():
    """All cases in one frame through the vectorized rules, row by row vs the reference."""
    cases = _random_cases(6000, 20261026)
    feats = pd.DataFrame([dataclasses.asdict(f) for f in cases])
    states = bs.classify_states_frame(feats)
    status = bs.classify_status_frame(feats, states)
    scores = bs.score_readiness_frame(feats, states)
    for i, feat in enumerate(cases):
        want_states = ref.classify_states(feat)
        assert states.iloc[i].to_dict() == want_states, (i, feat)
        assert status.iloc[i] == ref.classify_status(feat, want_states), (i, feat)
        assert scores.iloc[i].to_dict() == ref.score_readiness(feat, want_states), (i, feat)
    assert set(status) == {
        bs.STATUS_PRE,
        bs.STATUS_WATCH,
        bs.STATUS_DEV,
        bs.STATUS_CONFIRMED,
        bs.STATUS_TRIGGERED,
        bs.STATUS_FAILED,
    }  # every branch exercised


def test_per_row_wrappers_match_reference():
    """The scalar API (1-row frames from a dataclass): same values, plain str/int types."""
    for i, feat in enumerate(_random_cases(300, 7)):
        want_states = ref.classify_states(feat)
        states = bs.classify_states(feat)
        assert states == want_states, (i, feat)
        status = bs.classify_status(feat, states)
        assert status == ref.classify_status(feat, want_states), (i, feat)
        scores = bs.score_readiness(feat, states)
        assert scores == ref.score_readiness(feat, want_states), (i, feat)
        assert type(status) is str and all(type(v) is str for v in states.values())
        assert all(type(v) is int for v in scores.values()), scores


# ---------------------------------------------------------------------------
# 2. Forming higher-timeframe bar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("digits", [2, 4, 6])
def test_round_like_builtin_matches_round(digits):
    rng = np.random.default_rng(digits)
    scale = 10.0**digits
    ties = (np.arange(-5000, 5000) + 0.5) / scale  # x.xx5 values: the hard cases
    values = np.concatenate(
        [ties, rng.uniform(-200, 200, 20_000), rng.normal(0, 1e-3, 2_000), [2.675, 1.0005, 0.125]]
    )
    got = bs._round_like_builtin(values, digits)
    want = np.array([round(float(v), digits) for v in values])
    np.testing.assert_array_equal(got, want)
    edge = bs._round_like_builtin([np.inf, -np.inf, np.nan, 1e17], digits)
    assert np.isnan(edge[:3]).all() and edge[3] == 1e17


def _truncated_htf_smas(clean: pd.DataFrame, t: int, rule: str, lengths) -> list[float]:
    """Brute force: SMAs on the last HTF bar of data cut after native bar t."""
    close = bs.resample_ohlcv(clean.iloc[: t + 1], rule)["Close"].astype(float)
    return [float(bs._sma(close, n).iloc[-1]) for n in lengths]


@pytest.mark.parametrize(
    ("native_tf", "rule", "lengths"),
    [("daily", "W-FRI", (20, 50, 200)), ("daily", "ME", (20, 50)), ("weekly", "ME", (20, 50))],
)
def test_forming_sma_matches_truncated_recompute(native_tf, rule, lengths):
    clean = _clean("BULL", native_tf)
    got = bs._forming_htf_smas(clean, rule, lengths)
    want = np.array([_truncated_htf_smas(clean, t, rule, lengths) for t in range(len(clean))])
    for j, n in enumerate(lengths):
        assert np.isfinite(want[:, j]).any(), f"SMA{n} never warm: fixture too short"
        np.testing.assert_array_equal(got[n], want[:, j], err_msg=f"{rule} SMA{n}")  # bit-exact


def _reference_flags(clean: pd.DataFrame, native_tf: str) -> dict[str, bool | None]:
    """Trend flags the original per-row code computes on ``clean`` (its last bar)."""
    flags: dict[str, bool | None] = {"daily_trend_bull": None, "daily_trend_bear": None}
    frames = bs.FeatureEngine(native_tf).create_timeframes(clean)
    for tf, frame in frames.items():
        enriched = bs.add_indicators(
            frame, pctl_window=bs.PCTL_WINDOW[tf], include_sma200=tf != "monthly"
        )
        flags[f"{tf}_trend_bull"] = ref.trend_bull(enriched, require_sma50=tf != "monthly")
        flags[f"{tf}_trend_bear"] = ref.trend_bear(enriched)
    return flags


@pytest.mark.parametrize("native_tf", ["daily", "weekly"])
@pytest.mark.parametrize("symbol", ["BULL", "CHOP"])
def test_trend_flags_match_truncated_recompute(native_tf, symbol):
    clean = _clean(symbol, native_tf)
    frame = bs._feature_frame(clean, native_tf)
    step = 17 if native_tf == "daily" else 3  # coprime to 5: every weekday is hit
    for t in range(bs.MIN_PRIMARY_BARS - 1, len(clean), step):
        want = _reference_flags(clean.iloc[: t + 1], native_tf)
        for col, value in want.items():
            assert _same_cell(frame[col].iloc[t], value), (symbol, clean["Date"].iloc[t], col)


def _canary_bars(dip: slice, rally: slice) -> pd.DataFrame:
    """Smooth uptrend; closes in ``dip`` drop 20%, closes in ``rally`` make new highs."""
    dates = pd.bdate_range("2021-01-04", periods=1100)
    base = 50.0 * np.exp(0.0005 * np.arange(len(dates)))
    close = base.copy()
    close[dip] *= 0.80
    close[rally] *= 1.02
    close = np.round(close, 2)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": np.round(close * 1.005, 2),
            "Low": np.round(close * 0.995, 2),
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )


def _week_canary():
    dates = pd.bdate_range("2021-01-04", periods=1100)
    mon = int(np.flatnonzero(dates.weekday == 0)[-10])
    bars = _canary_bars(slice(mon, mon + 3), slice(mon + 3, mon + 5))
    return bars, mon + 2, mon + 4  # Wednesday, Friday


def _month_canary():
    dates = pd.bdate_range("2021-01-04", periods=1100)
    month = dates.to_period("M")
    first = int(np.flatnonzero(month == month[-60])[0])
    last = int(np.flatnonzero(month == month[-60])[-1])
    mid = first + 8  # a weekday in the middle of the month
    bars = _canary_bars(slice(first, mid + 1), slice(mid + 1, last + 1))
    return bars, mid, last


@pytest.mark.parametrize(("canary", "flag"), [(_week_canary, "weekly"), (_month_canary, "monthly")])
def test_canary_is_bullish_when_complete_but_not_while_forming(canary, flag):
    """Guards the canary itself, so the forming-bar test below can't pass vacuously."""
    bars, mid, end = canary()
    clean = bs.normalize_ohlcv(bars)
    forming = _reference_flags(clean.iloc[: mid + 1], "daily")[f"{flag}_trend_bull"]
    complete = _reference_flags(clean.iloc[: end + 1], "daily")[f"{flag}_trend_bull"]
    assert forming is False and complete is True


@pytest.mark.parametrize(("canary", "flag"), [(_week_canary, "Weekly"), (_month_canary, "Monthly")])
def test_htf_flags_use_the_forming_bar(canary, flag):
    """Mid-period, the flag must reflect the period so far, not how it ended."""
    bars, mid, end = canary()
    hist = bs.scan_history(bars, "CANARY", "daily", "daily").set_index("Date")
    assert _same_cell(hist.loc[bars["Date"].iloc[mid], f"{flag} Trend Bull"], False)
    assert _same_cell(hist.loc[bars["Date"].iloc[end], f"{flag} Trend Bull"], True)


# ---------------------------------------------------------------------------
# 3. Parity with run_scanner()
# ---------------------------------------------------------------------------

_RAW_NAME = {"daily": "{}_5y_1d.csv", "weekly": "{}_10y_1wk.csv"}


@pytest.fixture
def scanner_env(tmp_path, monkeypatch):
    def setup(mode: str) -> Path:
        raw_dir = tmp_path / "raw" / mode
        raw_dir.mkdir(parents=True)
        for symbol in [*TICKERS, "SHRT"]:
            bars = _bars(symbol, mode).assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"))
            bars.to_csv(raw_dir / _RAW_NAME[mode].format(symbol), index=False)
        active = tmp_path / "active_tickers.csv"
        pd.DataFrame({"Ticker": [*TICKERS, "SHRT"]}).to_csv(active, index=False)
        data_mode, scan_mode = config.resolve_pipeline_mode(mode)
        monkeypatch.setattr(bs, "mode", scan_mode)
        monkeypatch.setattr(bs, "_data_mode", data_mode)
        monkeypatch.setattr(bs, "RAW_DATA_DIR", str(raw_dir))
        monkeypatch.setattr(bs, "LOG_DIR", str(tmp_path / "logs" / mode))
        monkeypatch.setattr(bs, "ACTIVE_TICKERS_PATH", str(active))
        return raw_dir

    return setup


@pytest.mark.parametrize("mode", ["daily", "weekly"])
@pytest.mark.parametrize("as_of", [None, "2025-06-11", "2025-06-13", "2025-04-30"])
def test_last_row_parity_with_run_scanner(scanner_env, mode, as_of):
    raw_dir = scanner_env(mode)
    out = bs.run_scanner(as_of=as_of)
    stamp = config.run_stamp(as_of)
    lines = (Path(bs.LOG_DIR) / f"breakout_setups_{stamp}.csv").read_text().splitlines()
    assert set(out["Symbol"]) == set(TICKERS)  # SHRT rejected as insufficient

    for symbol in TICKERS:
        path = raw_dir / _RAW_NAME[mode].format(symbol)
        bars = raw.load_raw(str(path), as_of=as_of, weekly=mode == "weekly")
        hist = bs.scan_history(bars, symbol, mode, mode)
        last = hist.iloc[-1]
        assert last["Date"].date().isoformat() == last["AsOf Date"]
        _assert_rows_equal(last, out.set_index("Symbol", drop=False).loc[symbol], symbol)
        # Byte-identical to the line run_scanner wrote.
        csv = hist.iloc[[-1]][bs.OUTPUT_COLUMNS].to_csv(index=False).splitlines()
        assert csv[0] == lines[0]
        assert csv[1] in lines[1:], symbol

    short = raw.load_raw(str(raw_dir / _RAW_NAME[mode].format("SHRT")), weekly=mode == "weekly")
    with pytest.raises(ValueError, match="insufficient bars"):
        bs.scan_history(short, "SHRT", mode, mode)


@pytest.mark.parametrize("native_tf", ["daily", "weekly"])
def test_history_rows_match_per_date_replay(native_tf):
    """Every sampled row equals the per-row path run on data cut at that bar."""
    clean = _clean("MIXD", native_tf)
    hist = bs.scan_history(clean, "MIXD", native_tf, native_tf)
    step = 23 if native_tf == "daily" else 3
    offset = bs.MIN_PRIMARY_BARS - 1
    for t in range(offset, len(clean), step):
        row = hist.iloc[t - offset]
        assert row["Date"] == clean["Date"].iloc[t]
        want = _replay_row(clean.iloc[: t + 1], "MIXD", native_tf, native_tf)
        _assert_rows_equal(row, want, str(row["AsOf Date"]))


_GOLDEN = Path("/app/data/golden")


@pytest.mark.skipif(not (_GOLDEN / "fixture").is_dir(), reason="golden fixture not mounted")
@pytest.mark.parametrize("mode", ["daily", "weekly"])
def test_golden_fixture_sample_parity(mode):
    as_of = json.loads((_GOLDEN / "baseline" / "MANIFEST.json").read_text())["as_of"][mode]
    raw_dir = _GOLDEN / "fixture" / "raw" / mode
    files = [f for f in raw.raw_files(str(raw_dir))][::12]
    for f in files:
        rows, _ = bs.scan_files(
            [Path(f.path).name], {f.ticker}, str(raw_dir), mode, mode, as_of=as_of
        )
        if not rows:
            continue
        bars = raw.load_raw(f.path, as_of=as_of, weekly=mode == "weekly")
        _assert_rows_equal(bs.scan_history(bars, f.ticker, mode, mode).iloc[-1], rows[0], f.ticker)


# ---------------------------------------------------------------------------
# 4. No lookahead
# ---------------------------------------------------------------------------

LOOKAHEAD_DATES = [
    "2023-01-04",  # Wed
    "2024-04-15",  # Mon
    "2024-04-16",  # Tue
    "2024-04-17",  # Wed
    "2024-04-18",  # Thu
    "2024-04-19",  # Fri
    "2024-05-30",  # inside the halt gap (no bar), the week before a month end
    "2024-06-01",  # Saturday inside the halt gap, after the May month end
    "2024-07-03",  # Wed before the July 4 holiday (holiday-shortened week)
    "2024-07-31",  # Wed month end
    "2024-10-01",  # Tue, first session of the month
    "2024-11-30",  # Saturday
    "2025-01-15",  # Wed, weekly SMA200 just warm
    "2025-06-11",  # Wed
]


@pytest.mark.parametrize("native_tf", ["daily", "weekly"])
@pytest.mark.parametrize("symbol", ["BULL", "MIXD", "LATE"])
def test_scan_history_has_no_lookahead(native_tf, symbol):
    weekly = native_tf == "weekly"
    bars = _bars(symbol, native_tf)
    full = bs.scan_history(bars, symbol, native_tf, native_tf)
    checked = 0
    for t in LOOKAHEAD_DATES:
        cut = config.cut_to_as_of(bars, t, weekly=weekly)
        if len(cut) < bs.MIN_PRIMARY_BARS:
            continue
        expected = bs.scan_history(cut, symbol, native_tf, native_tf)
        truncated = config.cut_to_as_of(full, t, weekly=weekly)
        pd.testing.assert_frame_equal(truncated, expected, check_exact=True, obj=f"{symbol} {t}")
        checked += 1
    assert checked >= len(LOOKAHEAD_DATES) - 2


# ---------------------------------------------------------------------------
# 5. Shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("native_tf", ["daily", "weekly"])
def test_scan_history_shape(native_tf):
    clean = _clean("BULL", native_tf)
    hist = bs.scan_history(_bars("BULL", native_tf), "bull", native_tf, native_tf)
    assert list(hist.columns) == ["Date", *bs.OUTPUT_COLUMNS]
    assert len(hist) == len(clean) - (bs.MIN_PRIMARY_BARS - 1)
    assert pd.api.types.is_datetime64_any_dtype(hist["Date"])
    assert hist["Date"].is_monotonic_increasing and hist["Date"].is_unique
    assert (hist["AsOf Date"] == hist["Date"].dt.strftime("%Y-%m-%d")).all()
    assert (hist["Symbol"] == "BULL").all() and (hist["Mode"] == native_tf).all()
    assert hist.index.equals(pd.RangeIndex(len(hist)))

    floor = _bars("BULL", native_tf).iloc[: bs.MIN_PRIMARY_BARS]
    assert len(bs.scan_history(floor, "BULL", native_tf, native_tf)) == 1
    with pytest.raises(ValueError, match="insufficient bars"):
        bs.scan_history(floor.iloc[:-1], "BULL", native_tf, native_tf)
