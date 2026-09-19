"""Pre-registered Coiled Cobra vs. Leader-Expansion walk-forward experiment (research only).

Question: the Coiled Cobra rubric (v4.0) is built to catch *coils*, and a manual
review of ten tickers with monster runs (PLTR, HOOD, BBY, NKE, ASTS, MU, MRVL,
COIN, MSTR, STX) found it caught few of them -- volatility contraction scored 0
during most of those runs (Gate C / D), and 8 of 26 runs began before the
160-bar history floor.  Would a separate **leader-expansion** rule (trend gate +
market gate + strong relative strength, *no* coil requirement) catch the runs
without giving back edge -- and does simply loosening the existing gates do
better?  This harness never touches the live scanner or its thresholds.  The
protocol is fixed in code *before* looking at full-universe results:

* **Discovery vs. holdout** -- the ten tickers above are the *discovery* set: the
  leader rules were chosen after looking at them, so they are reported but never
  used for qualification.  Every decision is taken on the *holdout* tickers (the
  rest of the active universe).
* **Signals** -- one pass over every scoreable weekly bar (>= the 160-bar floor)
  computes the full gate/pillar profile once (``evaluate_coiled_cobra(...,
  include_rejects=True)``, causal); each variant in ``VARIANTS`` is a fixed
  predicate over that profile.  ``C0_random`` (random bars, same exits) is the
  bar to beat, not a "no edge" check: long-only entries in a rising market plus a
  trailing exit earn positive R on their own, so a variant only counts as signal
  if it beats the control (Q4).
* **Deduplication** -- consecutive weekly flags per ticker are one *episode*;
  only the first bar is a trade (overlapping re-fires would overstate stats).
* **Exit models** -- ``planner`` = the deployed Fib-78.6% entry / 2R-3R geometry
  (``calculate_stock_levels`` + ``simulate_trade``, apples-to-apples with the
  existing backtest); ``trail`` = market entry at the next bar's open, initial
  stop ``TRAIL_INIT_ATR`` x ATR, chandelier trail ``TRAIL_ATR`` x ATR off the
  highest close, time exit at ``TRAIL_MAX_HOLD`` bars (a fixed target cap turns a
  monster into +3R; the trail is what lets a leader entry pay).  **Primary
  outcome = R multiple of ``trail``**; ``planner`` is reported, never selected on.
* **Periods** -- ``lockbox`` = the ``LOCKBOX_WEEKS`` ending ``CENSOR_WEEKS``
  before the last bar (so trail trades can fully resolve); ``dev`` = everything
  before it; ``live`` = the most recent ``CENSOR_WEEKS`` (censored, reported only).
* **Qualification** (holdout tickers, dev period), all must hold, for a variant
  to reach the lockbox: (Q0) >= ``MIN_EPISODES`` episodes; (Q1) 95% week-cluster
  bootstrap CI of mean trail-R above 0; (Q2) monster capture -- share of
  scoreable monster runs caught with >= 100% still left -- at least
  ``Q2_CAPTURE_MARGIN`` above baseline; (Q3) share of episodes with a >= 25%
  drawdown inside 26 weeks no more than ``Q3_DD_MARGIN`` above baseline; (Q4)
  paired (same-week) mean-R difference vs ``C0_random`` has CI above 0.
  Only the single best qualifier (highest Q1 lower bound) is evaluated **once**
  on the lockbox, where it passes if mean R > 0 with cluster t > 1.645.  If
  nothing qualifies the lockbox is left unspent.

    python -m finance_vibe.coiled_cobra_leader_experiment [--tickers A,B] [--out-dir D] [--out r.json]
    python -m finance_vibe.coiled_cobra_leader_experiment --from-bars bars.csv.gz --runs runs.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from finance_vibe import config
    from finance_vibe import coiled_cobra as cc
    from finance_vibe import coiled_cobra_backtest as cbt
    from finance_vibe.analysis_engine import load_ohlc_csv, ticker_from_filename
    from finance_vibe.coiled_cobra import add_macro_indicators, evaluate_coiled_cobra, local_swing_low
    from finance_vibe.trade_planner import calculate_stock_levels
    from finance_vibe.trade_simulator import simulate_trade
except ImportError:  # pragma: no cover - local direct execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe import coiled_cobra as cc
    from finance_vibe import coiled_cobra_backtest as cbt
    from finance_vibe.analysis_engine import load_ohlc_csv, ticker_from_filename
    from finance_vibe.coiled_cobra import add_macro_indicators, evaluate_coiled_cobra, local_swing_low
    from finance_vibe.trade_planner import calculate_stock_levels
    from finance_vibe.trade_simulator import simulate_trade

MODE = "weekly"          # the rubric is weekly-only; other modes are out of scope
SEED = 20260919

# Tickers the leader rules were derived from -- reported, never used to qualify.
DISCOVERY_TICKERS = frozenset(
    {"PLTR", "HOOD", "BBY", "NKE", "ASTS", "MU", "MRVL", "COIN", "MSTR", "STX", "SNDK"}
)

# --- trail exit model (fixed) -------------------------------------------------
TRAIL_INIT_ATR = 2.5
TRAIL_ATR = 3.0
TRAIL_MAX_HOLD = 52
TRAIL_SLIPPAGE = config.BACKTEST_SLIPPAGE_PCT

# --- periods / controls -------------------------------------------------------
LOCKBOX_WEEKS = 52
CENSOR_WEEKS = TRAIL_MAX_HOLD
P_CONTROL = 0.05         # per-bar probability of a C0_random flag

# --- pre-registered qualification thresholds ----------------------------------
MIN_EPISODES = 100
Q2_CAPTURE_MARGIN = 0.15
Q3_DD_MARGIN = 0.10
LOCKBOX_T = 1.645

# --- monster runs -------------------------------------------------------------
RUN_MIN_GAIN = 1.0       # trough -> peak >= +100%
RUN_MAX_WEEKS = 52
RUN_MAX_PER_TICKER = 3
RUN_LEAD_BARS = 4        # a signal counts from this many bars before the low

BASELINE = "B0_baseline"
CONTROL = "C0_random"
LEADERS = ("L1_trend_rs_struct_sky", "L2_L1_plus_rvol", "L3_trend_rs_rvol", "L4_trend_rs18_sky8")
VARIANTS = (BASELINE, "R1_relaxed_gates", *LEADERS, CONTROL)
QUALIFYING = ("R1_relaxed_gates", *LEADERS)


# ---------------------------------------------------------------------------
# Variant predicates (fixed; inputs are the per-bar gate/pillar profile)
# ---------------------------------------------------------------------------

def variant_flags(rec: Mapping[str, Any]) -> dict[str, bool]:
    """Flag which variants fire on one bar.  ``C0_random`` is set by the caller."""
    fails = set(rec["fails"])
    score = float(rec["score"])
    no_ab = "A" not in fails and "B" not in fails

    # R1: loosen the coil scanner -- gate A needs only close > slow EMA and a
    # rising slow EMA, gate C needs structure only, score floor 60.
    r_fails = set(fails)
    if "A" in r_fails and rec["a_close_gt_slow"] and rec["a_c3_slow_rising"]:
        r_fails.discard("A")
    if "C" in r_fails and rec["struct"] >= cc.GATE_C_STRUCTURE_MIN:
        r_fails.discard("C")

    l1 = no_ab and rec["rs"] >= 14 and rec["struct"] >= 8 and rec["ovhd"] >= 5
    return {
        BASELINE: (not fails) and score >= cc.MIN_PASS_SCORE,
        "R1_relaxed_gates": (not r_fails) and score >= 60,
        "L1_trend_rs_struct_sky": bool(l1),
        "L2_L1_plus_rvol": bool(l1 and rec["rvol"] >= 6),
        "L3_trend_rs_rvol": bool(no_ab and rec["rs"] >= 14 and rec["rvol"] >= 6),
        "L4_trend_rs18_sky8": bool(no_ab and rec["rs"] >= 18 and rec["ovhd"] >= 8),
    }


# ---------------------------------------------------------------------------
# Exit model: market entry + chandelier trail
# ---------------------------------------------------------------------------

def simulate_trail_trade(
    df: pd.DataFrame,
    start_idx: int,
    atr: float,
    *,
    slippage: float = TRAIL_SLIPPAGE,
    init_atr: float = TRAIL_INIT_ATR,
    trail_atr: float = TRAIL_ATR,
    max_hold: int = TRAIL_MAX_HOLD,
) -> Optional[dict]:
    """Enter at bar ``start_idx``'s open; exit on a stop/trail touch or time.

    The stop is only ever ratcheted *after* a bar's low has been tested against
    it, so a bar's own high/close can never rescue a same-bar stop-out.  A stop
    gapped through fills at the open.  ``censored`` is True when the data ends
    before the trade resolves or ``max_hold`` bars pass.
    """
    n = len(df)
    if start_idx >= n or not atr or atr <= 0:
        return None
    entry_raw = float(df["Open"].iloc[start_idx])
    entry = entry_raw * (1.0 + slippage)
    stop0 = entry - init_atr * atr
    if stop0 <= 0 or entry <= stop0:
        return None
    risk = entry - stop0

    stop = stop0
    high_close = entry_raw
    max_high = entry_raw
    last = min(start_idx + max_hold, n) - 1
    for j in range(start_idx, last + 1):
        o, h, low, c = (float(df[k].iloc[j]) for k in ("Open", "High", "Low", "Close"))
        if low <= stop:
            px = min(o, stop) * (1.0 - slippage)
            return {
                "outcome": "stopped" if stop <= stop0 + 1e-12 else "trailed",
                "exit_idx": j, "exit_price": px, "r": (px - entry) / risk,
                "censored": False, "mfe_r": (max_high - entry) / risk,
            }
        max_high = max(max_high, h)
        high_close = max(high_close, c)
        stop = max(stop, high_close - trail_atr * atr)

    px = float(df["Close"].iloc[last]) * (1.0 - slippage)
    ran_full = (last - start_idx + 1) >= max_hold
    return {
        "outcome": "expired" if ran_full else "open",
        "exit_idx": last, "exit_price": px, "r": (px - entry) / risk,
        "censored": not ran_full, "mfe_r": (max_high - entry) / risk,
    }


# ---------------------------------------------------------------------------
# Monster runs
# ---------------------------------------------------------------------------

def find_runs(
    highs: np.ndarray,
    lows: np.ndarray,
    *,
    min_gain: float = RUN_MIN_GAIN,
    max_weeks: int = RUN_MAX_WEEKS,
    max_runs: int = RUN_MAX_PER_TICKER,
) -> list[tuple[float, int, int]]:
    """Greedy non-overlapping trough->peak runs: max ``High[j] / Low[i] - 1``, ``0 < j - i <= max_weeks``."""
    n = len(highs)
    taken = np.zeros(n, dtype=bool)
    runs: list[tuple[float, int, int]] = []
    for _ in range(max_runs):
        best: Optional[tuple[float, int, int]] = None
        for i in range(n):
            if taken[i] or lows[i] <= 0:
                continue
            for j in range(i + 1, min(i + max_weeks, n - 1) + 1):
                if taken[j]:
                    break
                g = highs[j] / lows[i] - 1.0
                if best is None or g > best[0]:
                    best = (g, i, j)
        if best is None or best[0] < min_gain:
            break
        runs.append(best)
        taken[max(0, best[1] - 8): min(n, best[2] + 8)] = True
    return sorted(runs, key=lambda r: r[1])


# ---------------------------------------------------------------------------
# Per-ticker walk-forward pass (process-pool worker)
# ---------------------------------------------------------------------------

def _fails_from_grade(grade: str) -> list[str]:
    if grade.startswith("Rejected - Gate Fail"):
        return grade.split("(")[1].rstrip(")").split("/")
    return []


def _forward(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, idx: int, weeks: int):
    fut_hi, fut_lo = highs[idx + 1: idx + 1 + weeks], lows[idx + 1: idx + 1 + weeks]
    if len(fut_hi) == 0:
        return np.nan, np.nan
    return fut_hi.max() / closes[idx] - 1.0, fut_lo.min() / closes[idx] - 1.0


def _planner_outcome(df: pd.DataFrame, idx: int, window: pd.DataFrame) -> dict:
    """Existing deployed geometry on one bar (same call chain as the live backtest)."""
    last = window.iloc[-1]
    fib = last.get("Fib_786")
    row = {
        "Source": "coiled_cobra", "Setup Type": "SETUP_LONG", "Mode": MODE,
        "Close": float(last["Close"]), "EMA20": float(last["EMA20"]), "EMA50": float(last["EMA50"]),
        "ATR": float(last["ATR"]), "Fib 78.6%": float(fib) if pd.notna(fib) else None,
        "Swing Low": local_swing_low(window),
    }
    try:
        entry, stop, t1, t2, _, _ = calculate_stock_levels(row, mode=MODE)
        outcome, _, _, r = simulate_trade(
            df, idx + 1, is_long=True, entry=entry, stop=stop, target1=t1, target2=t2,
            entry_valid_bars=config.BACKTEST_ENTRY_VALID_BARS, max_hold_bars=config.BACKTEST_MAX_HOLD_BARS,
        )
    except Exception:
        return {"pl_outcome": "error", "pl_r": np.nan}
    return {"pl_outcome": outcome, "pl_r": np.nan if r is None else float(r)}


def _ticker_pass(path: str) -> tuple[str, list[dict], list[dict], dict]:
    """Per-bar records, monster runs and metadata for one ticker.  Causal at every bar."""
    symbol = ticker_from_filename(path)
    try:
        df = load_ohlc_csv(path).reset_index(drop=True)
    except Exception as exc:
        print(f"{symbol}: error - {exc}", file=sys.stderr)
        return symbol, [], [], {}
    if not {"Open", "High", "Low", "Close"} <= set(df.columns):
        return symbol, [], [], {}

    highs, lows, closes = (df[c].to_numpy(dtype=float) for c in ("High", "Low", "Close"))
    runs = [
        {"sym": symbol, "low_idx": i, "peak_idx": j, "gain": g, "low": lows[i], "high": highs[j],
         "low_date": df["Date"].iloc[i], "peak_date": df["Date"].iloc[j]}
        for g, i, j in find_runs(highs, lows)
    ]

    rng = np.random.default_rng([SEED, zlib.crc32(symbol.encode())])
    bench, spy = cbt._WORKER_BENCHMARK_DF, cbt._WORKER_SPY_DF
    prior_col = "TT_EMA_SLOW"
    rows: list[dict] = []
    for idx in range(cc.MIN_BARS_FULL_SCORE - 1, len(df) - 1):     # need a next bar to enter
        control = bool(rng.random() < P_CONTROL)                     # draw every bar: deterministic
        try:
            window = add_macro_indicators(df.iloc[: idx + 1].copy())
            res = evaluate_coiled_cobra(window, bench, spy_df=spy, qqq_df=bench, include_rejects=True)
        except Exception:
            continue
        if res is None:
            continue
        last = window.iloc[-1]
        parts = res["Parts"]
        prior = window[prior_col].iloc[-(cc.TREND_RISING_LOOKBACK + 1)]
        close, slow = float(last["Close"]), float(last[prior_col])
        rec = {
            "sym": symbol, "idx": idx, "date": df["Date"].iloc[idx], "close": close, "atr": float(last["ATR"]),
            "score": float(res["Score"]), "checks": int(res["Checks Met"].split("/")[0]),
            "vol": parts["vol_contraction"], "rs": parts["relative_strength"], "struct": parts["structure"],
            "shelf": parts["volume_shelf"], "ovhd": parts["overhead_clearance"], "rvol": parts["rvol_trigger"],
            "rs63": res["RS 63d"], "mkt": bool(res["Market Gate"]), "bbp": res["BBWidth Pctile"],
            "fails": _fails_from_grade(res["Grade"]),
            "a_close_gt_slow": close > slow,
            "a_c3_slow_rising": bool(pd.notna(prior) and slow > float(prior)),
        }
        rec["mu13"], rec["dd13"] = _forward(highs, lows, closes, idx, 13)
        rec["mu26"], rec["dd26"] = _forward(highs, lows, closes, idx, 26)
        flags = variant_flags(rec)
        flags[CONTROL] = control
        for name in VARIANTS:
            rec[f"f_{name}"] = bool(flags[name])
        if any(flags.values()):
            rec.update(_planner_outcome(df, idx, window))
            tr = simulate_trail_trade(df, idx + 1, rec["atr"])
            if tr:
                rec.update({"tr_outcome": tr["outcome"], "tr_r": tr["r"], "tr_censored": tr["censored"],
                            "tr_mfe_r": tr["mfe_r"], "tr_bars": tr["exit_idx"] - idx})
        rec["fails"] = "/".join(rec["fails"])
        rows.append(rec)
    meta = {"sym": symbol, "n_bars": len(df), "last_date": df["Date"].iloc[-1]}
    return symbol, rows, runs, meta


def collect(paths: list[str], workers: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Run the pass over ``paths`` in a process pool; returns (bars, runs, last_bar_date)."""
    cc.apply_timeframe(MODE)
    rows: list[dict] = []
    runs: list[dict] = []
    last_dates: list[pd.Timestamp] = []
    print(f"--- Leader-Expansion experiment [{MODE}] --- tickers: {len(paths)}", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers or os.cpu_count() or 1,
        initializer=cbt._init_cobra_worker, initargs=(MODE,),
    ) as ex:
        futs = {ex.submit(_ticker_pass, p): p for p in paths}
        for k, fut in enumerate(as_completed(futs), 1):
            sym, r, ru, meta = fut.result()
            rows.extend(r)
            runs.extend(ru)
            if meta:
                last_dates.append(meta["last_date"])
            if k % 20 == 0 or k == len(paths):
                print(f"  {k}/{len(paths)} tickers done ({len(rows)} scoreable bars)", flush=True)
    bars = pd.DataFrame(rows)
    runs_df = pd.DataFrame(runs)
    return bars, runs_df, max(last_dates) if last_dates else pd.NaT


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def dedup_episodes(flagged: pd.DataFrame) -> pd.DataFrame:
    """Keep the first bar of each run of consecutive weekly flags per ticker."""
    if flagged.empty:
        return flagged
    f = flagged.sort_values(["sym", "idx"])
    new = (f["sym"] != f["sym"].shift()) | (f["idx"] != f["idx"].shift() + 1)
    return f[new]


def _week_arrays(df: pd.DataFrame, col: str, weeks: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Per-week (sum, count) of ``col`` aligned on ``weeks`` (cluster-bootstrap units)."""
    g = df.dropna(subset=[col]).groupby("date")[col].agg(["sum", "count"]).reindex(weeks).fillna(0.0)
    return g["sum"].to_numpy(), g["count"].to_numpy()


def _boot_mean(sums: np.ndarray, counts: np.ndarray, draws: np.ndarray) -> np.ndarray:
    num, den = sums[draws].sum(axis=1), counts[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def cluster_stats(df: pd.DataFrame, col: str, n_boot: int, seed: int = 0) -> dict:
    """Mean of ``col`` with a 95% week-cluster bootstrap CI and cluster t-stat."""
    d = df.dropna(subset=[col])
    if d.empty:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "t": np.nan}
    weeks = pd.Index(sorted(d["date"].unique()))
    sums, counts = _week_arrays(d, col, weeks)
    draws = np.random.default_rng(seed).integers(0, len(weeks), size=(n_boot, len(weeks)))
    boots = _boot_mean(sums, counts, draws)
    mean = sums.sum() / counts.sum()
    se = np.sqrt(((sums - mean * counts) ** 2).sum()) / counts.sum()
    return {"n": int(counts.sum()), "mean": float(mean),
            "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5)),
            "t": float(mean / se) if se > 0 else np.nan}


def paired_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int, seed: int = 0) -> dict:
    """Mean(a) - mean(b) with a 95% CI from a shared week resample (paired by calendar week)."""
    a, b = a.dropna(subset=[col]), b.dropna(subset=[col])
    if a.empty or b.empty:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan}
    weeks = pd.Index(sorted(set(a["date"]) | set(b["date"])))
    sa, ca = _week_arrays(a, col, weeks)
    sb, cb = _week_arrays(b, col, weeks)
    draws = np.random.default_rng(seed).integers(0, len(weeks), size=(n_boot, len(weeks)))
    boots = _boot_mean(sa, ca, draws) - _boot_mean(sb, cb, draws)
    return {"diff": float(sa.sum() / ca.sum() - sb.sum() / cb.sum()),
            "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5))}


def _profit_factor(r: pd.Series) -> Optional[float]:
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else None


def summarize(ep: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """Trade + forward-outcome metrics for one set of deduplicated episodes."""
    out: dict[str, Any] = {"episodes": int(len(ep))}
    if ep.empty:
        return out
    trail = ep[ep["tr_r"].notna() & ~ep["tr_censored"].fillna(True).astype(bool)]
    out["trail"] = {
        **cluster_stats(trail, "tr_r", n_boot, seed),
        "median_r": float(trail["tr_r"].median()) if len(trail) else np.nan,
        "win_rate": float((trail["tr_r"] > 0).mean()) if len(trail) else np.nan,
        "profit_factor": _profit_factor(trail["tr_r"]) if len(trail) else None,
        "avg_mfe_r": float(trail["tr_mfe_r"].mean()) if len(trail) else np.nan,
        "big_winner_share": float((trail["tr_r"] >= 5).mean()) if len(trail) else np.nan,
    }
    filled = ep[ep["pl_outcome"].notna() & ~ep["pl_outcome"].isin(["no_fill", "error"])]
    out["planner"] = {
        "fill_rate": float(len(filled) / len(ep)),
        **cluster_stats(filled, "pl_r", n_boot, seed),
        "win_rate": float((filled["pl_r"] > 0).mean()) if len(filled) else np.nan,
        "profit_factor": _profit_factor(filled["pl_r"]) if len(filled) else None,
    }
    out["forward"] = forward_stats(ep)
    return out


def forward_stats(d: pd.DataFrame) -> dict:
    """Forward-outcome shape of a set of bars/episodes (entry = signal close)."""
    m13, m26 = d.dropna(subset=["mu13"]), d.dropna(subset=["mu26"])
    return {
        "hit20_13w": float((m13["mu13"] >= 0.20).mean()) if len(m13) else np.nan,
        "med_maxup_13w": float(m13["mu13"].median()) if len(m13) else np.nan,
        "med_maxdd_13w": float(m13["dd13"].median()) if len(m13) else np.nan,
        "runner_26w": float((m26["mu26"] >= 1.0).mean()) if len(m26) else np.nan,     # doubled within 26w
        "dd26_worse25": float((m26["dd26"] <= -0.25).mean()) if len(m26) else np.nan,
    }


def monster_capture(bars: pd.DataFrame, runs: pd.DataFrame, variant: str) -> dict:
    """Share of scoreable monster runs a variant flags (any bar, no dedup), and how early/useful."""
    flagged = bars[bars[f"f_{variant}"]]
    caught = useful = scoreable = 0
    offsets: list[int] = []
    by_sym = {s: g.sort_values("idx") for s, g in bars.groupby("sym")}
    flag_by_sym = {s: g.sort_values("idx") for s, g in flagged.groupby("sym")}
    for run in runs.itertuples():
        g = by_sym.get(run.sym)
        if g is None or not ((g["idx"] >= run.low_idx - RUN_LEAD_BARS) & (g["idx"] <= run.peak_idx)).any():
            continue                                        # no scoreable bar in the run window
        scoreable += 1
        f = flag_by_sym.get(run.sym)
        if f is None:
            continue
        hit = f[(f["idx"] >= run.low_idx - RUN_LEAD_BARS) & (f["idx"] <= run.peak_idx)]
        if hit.empty:
            continue
        first = hit.iloc[0]
        caught += 1
        offsets.append(int(first["idx"] - run.low_idx))
        useful += int(run.high / first["close"] - 1.0 >= 1.0)
    return {"scoreable_runs": scoreable, "caught": caught, "useful": useful,
            "capture_rate": caught / scoreable if scoreable else np.nan,
            "useful_rate": useful / scoreable if scoreable else np.nan,
            "median_offset_wks": float(np.median(offsets)) if offsets else np.nan}


def split_periods(bars: pd.DataFrame, last_date: pd.Timestamp) -> pd.Series:
    """'dev' | 'lockbox' | 'live' by signal date (see module docstring)."""
    live_start = last_date - pd.Timedelta(weeks=CENSOR_WEEKS)
    lock_start = live_start - pd.Timedelta(weeks=LOCKBOX_WEEKS)
    return pd.Series(
        np.where(bars["date"] >= live_start, "live", np.where(bars["date"] >= lock_start, "lockbox", "dev")),
        index=bars.index,
    )


def qualifies(name: str, res: dict, base: dict, ctrl_diff: dict) -> dict:
    """Pre-registered Q0-Q4 checks for one variant (holdout tickers, dev period)."""
    tr = res.get("trail", {})
    checks = {
        "Q0_episodes": res.get("episodes", 0) >= MIN_EPISODES,
        "Q1_ci_above_0": bool(tr.get("lo", np.nan) > 0),
        "Q2_monster_capture": bool(res["capture"]["useful_rate"] >= base["capture"]["useful_rate"] + Q2_CAPTURE_MARGIN),
        "Q3_drawdown": bool(res.get("forward", {}).get("dd26_worse25", np.inf)
                            <= base.get("forward", {}).get("dd26_worse25", -np.inf) + Q3_DD_MARGIN),
        "Q4_beats_random": bool(ctrl_diff.get("lo", np.nan) > 0),
    }
    return {"checks": checks, "qualified": all(checks.values())}


def run_analysis(bars: pd.DataFrame, runs: pd.DataFrame, last_date: pd.Timestamp, n_boot: int = 2000) -> dict:
    """Full pre-registered analysis over a collected bar table."""
    bars = bars.copy()
    bars["period"] = split_periods(bars, last_date)
    bars["holdout"] = ~bars["sym"].isin(DISCOVERY_TICKERS)
    runs = runs.copy()
    lock_start = last_date - pd.Timedelta(weeks=CENSOR_WEEKS + LOCKBOX_WEEKS)
    runs["holdout"] = ~runs["sym"].isin(DISCOVERY_TICKERS)
    runs["dev"] = runs["peak_date"] < lock_start

    def episodes(sel: pd.DataFrame, v: str) -> pd.DataFrame:
        return dedup_episodes(sel[sel[f"f_{v}"]])

    def block(holdout: bool, period: str) -> dict:
        sel = bars[(bars["holdout"] == holdout) & (bars["period"] == period)]
        rsel = runs[(runs["holdout"] == holdout) & (runs["dev"] if period == "dev" else True)]
        b_sub = bars[(bars["holdout"] == holdout) & (bars["period"] == period)]
        out: dict[str, Any] = {"all_bars": {"bars": int(len(sel)), **forward_stats(sel)}}
        for v in VARIANTS:
            res = summarize(episodes(sel, v), n_boot, SEED)
            res["capture"] = monster_capture(b_sub, rsel, v) if period == "dev" else {}
            out[v] = res
        return out

    result: dict[str, Any] = {
        "protocol": {"seed": SEED, "trail": [TRAIL_INIT_ATR, TRAIL_ATR, TRAIL_MAX_HOLD],
                     "min_episodes": MIN_EPISODES, "q2_margin": Q2_CAPTURE_MARGIN, "q3_margin": Q3_DD_MARGIN,
                     "lockbox_weeks": LOCKBOX_WEEKS, "censor_weeks": CENSOR_WEEKS, "n_boot": n_boot},
        "last_date": str(last_date.date()) if pd.notna(last_date) else None,
        "tickers": {"holdout": int(bars.loc[bars["holdout"], "sym"].nunique()),
                    "discovery": int(bars.loc[~bars["holdout"], "sym"].nunique())},
        "holdout_dev": block(True, "dev"),
        "discovery_dev": block(False, "dev"),
    }

    hd = result["holdout_dev"]
    dev_h = bars[bars["holdout"] & (bars["period"] == "dev")]
    ctrl_ep = episodes(dev_h, CONTROL)
    qual: dict[str, Any] = {}
    for v in QUALIFYING:
        diff = paired_diff(episodes(dev_h, v), ctrl_ep, "tr_r", n_boot, SEED)
        qual[v] = {"vs_random": diff, **qualifies(v, hd[v], hd[BASELINE], diff)}
    result["qualification"] = qual

    winners = [v for v in QUALIFYING if qual[v]["qualified"]]
    result["lockbox"] = {"spent": False, "best": None}
    if winners:
        best = max(winners, key=lambda v: hd[v]["trail"]["lo"])
        lock = bars[bars["holdout"] & (bars["period"] == "lockbox")]
        ep = episodes(lock, best)
        ep = ep[ep["tr_r"].notna() & ~ep["tr_censored"].fillna(True).astype(bool)]
        st = cluster_stats(ep, "tr_r", n_boot, SEED)
        base_st = cluster_stats(
            episodes(lock, BASELINE).pipe(lambda e: e[e["tr_r"].notna() & ~e["tr_censored"].fillna(True).astype(bool)]),
            "tr_r", n_boot, SEED)
        result["lockbox"] = {"spent": True, "best": best, "trail": st, "baseline_trail": base_st,
                             "passed": bool(st["mean"] > 0 and st["t"] > LOCKBOX_T)}
    result["holdout_live_open"] = {   # censored recent weeks: information only
        v: int(len(episodes(bars[bars["holdout"] & (bars["period"] == "live")], v))) for v in VARIANTS}
    return result


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

def _f(x: Any, nd: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{nd}f}"


def format_report(res: dict) -> str:
    lines = [
        f"Leader-Expansion vs Coiled Cobra walk-forward   last bar {res['last_date']}   "
        f"holdout tickers {res['tickers']['holdout']}  discovery {res['tickers']['discovery']}",
        "Primary outcome: mean R of the trail exit, holdout tickers, dev period. Discovery set is reference only.",
    ]
    for label, key in (("HOLDOUT (decision set)", "holdout_dev"), ("DISCOVERY (reference only)", "discovery_dev")):
        blk = res[key]
        ab = blk["all_bars"]
        lines += ["", f"== {label} ==",
                  f"all scoreable bars: {ab['bars']}  hit+20%/13w {_f(ab['hit20_13w'], pct=True)}  "
                  f"runner(2x/26w) {_f(ab['runner_26w'], pct=True)}  medMaxDD13 {_f(ab['med_maxdd_13w'], pct=True)}  "
                  f"DD26<=-25% {_f(ab['dd26_worse25'], pct=True)}",
                  f"{'variant':26s}{'eps':>6s}{'meanR':>7s}{'CI lo':>7s}{'CI hi':>7s}{'win':>6s}{'PF':>6s}"
                  f"{'planR':>7s}{'fill':>6s}{'runner':>8s}{'DD<-25':>8s}{'capt':>6s}{'useful':>7s}{'medOff':>7s}"]
        for v in VARIANTS:
            r = blk[v]
            if r["episodes"] == 0:
                lines.append(f"{v:26s}{0:6d}")
                continue
            t, p, fw, cp = r["trail"], r["planner"], r["forward"], r.get("capture", {})
            lines.append(
                f"{v:26s}{r['episodes']:6d}{_f(t['mean']):>7s}{_f(t['lo']):>7s}{_f(t['hi']):>7s}"
                f"{_f(t['win_rate'], pct=True):>6s}{_f(t['profit_factor']):>6s}{_f(p['mean']):>7s}"
                f"{_f(p['fill_rate'], pct=True):>6s}{_f(fw['runner_26w'], pct=True):>8s}"
                f"{_f(fw['dd26_worse25'], pct=True):>8s}{_f(cp.get('capture_rate'), pct=True):>6s}"
                f"{_f(cp.get('useful_rate'), pct=True):>7s}{_f(cp.get('median_offset_wks'), 0):>7s}")
    lines += ["", "== QUALIFICATION (holdout, dev) =="]
    for v, q in res["qualification"].items():
        marks = " ".join(f"{k.split('_')[0]}={'Y' if ok else 'n'}" for k, ok in q["checks"].items())
        d = q["vs_random"]
        lines.append(f"{v:26s}{'QUALIFIED' if q['qualified'] else 'no':10s}{marks}   "
                     f"vs random: {_f(d['diff'])} [{_f(d['lo'])}, {_f(d['hi'])}]")
    lb = res["lockbox"]
    if lb["spent"]:
        lines += ["", f"LOCKBOX spent once on {lb['best']}: mean R {_f(lb['trail']['mean'])} "
                      f"[{_f(lb['trail']['lo'])}, {_f(lb['trail']['hi'])}] t={_f(lb['trail']['t'])}  "
                      f"baseline {_f(lb['baseline_trail']['mean'])}  -> {'PASS' if lb['passed'] else 'FAIL'}"]
    else:
        lines += ["", "LOCKBOX left unspent: no variant qualified on the development period."]
    lines += ["", "live/open (censored) episodes, holdout: " + ", ".join(f"{k}={v}" for k, v in res["holdout_live_open"].items())]
    return "\n".join(lines)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered Leader-Expansion vs Coiled Cobra experiment")
    ap.add_argument("--tickers", help="Comma-separated tickers (smoke tests); default = whole raw dir")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out-dir", help="Directory for bars/runs/result files (default: the weekly logs dir)")
    ap.add_argument("--out", help="Write the JSON result here (default: <out-dir>/leader_experiment_<date>.json)")
    ap.add_argument("--from-bars", help="Skip the walk-forward: re-run the analysis from a saved bars .csv.gz")
    ap.add_argument("--runs", help="Saved runs .csv (required with --from-bars)")
    args = ap.parse_args(argv)

    logs_dir = args.out_dir or config.get_mode_config(MODE)["logs_dir"]
    stamp = datetime.now().strftime("%Y-%m-%d")
    if args.from_bars:
        if not args.runs:
            ap.error("--from-bars requires --runs")
        bars = pd.read_csv(args.from_bars, parse_dates=["date"])
        runs = pd.read_csv(args.runs, parse_dates=["low_date", "peak_date"])
        last_date = bars["date"].max() + pd.Timedelta(weeks=1)
    else:
        raw_dir = config.get_mode_config(MODE)["raw_dir"]
        wanted = {t.strip().upper() for t in args.tickers.split(",")} if args.tickers else None
        paths = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".csv")
                       if not wanted or ticker_from_filename(os.path.join(raw_dir, f)) in wanted)
        if not paths:
            ap.error(f"no raw CSVs matched in {raw_dir}")
        bars, runs, last_date = collect(paths, args.workers)
        os.makedirs(logs_dir, exist_ok=True)
        bars_path = os.path.join(logs_dir, f"leader_experiment_bars_{stamp}.csv.gz")
        runs_path = os.path.join(logs_dir, f"leader_experiment_runs_{stamp}.csv")
        bars.to_csv(bars_path, index=False)
        runs.to_csv(runs_path, index=False)
        print(f"Saved bars: {bars_path}\nSaved runs: {runs_path}")
        bars["date"] = pd.to_datetime(bars["date"])

    for c in [f"f_{v}" for v in VARIANTS] + ["tr_censored"]:
        if c in bars.columns:
            bars[c] = bars[c].fillna(False).astype(bool)
    res = run_analysis(bars, runs, last_date, n_boot=args.bootstrap)
    out_path = args.out or os.path.join(logs_dir, f"leader_experiment_{stamp}.json")
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, default=_json_default)
    print(format_report(res))
    print(f"\nSaved result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
