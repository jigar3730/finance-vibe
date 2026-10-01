"""Pre-registered walk-forward test of the Breakout Readiness scanner (research only).

Question: ``breakout_scanner.py`` runs in the live pipeline and its dashboard
surfaces ``PRE_BREAKOUT`` / ``BREAKOUT_CONFIRMED`` names, but nothing has shown
that its states or its 100-point readiness score predict anything.  Does any of
its outputs beat a random-entry control on forward trade outcomes, and is the
readiness score informative across names?  This harness never touches the live
scanner or its thresholds.  The protocol is fixed in code *before* looking at
results:

* **Signals** -- every weekly bar from the scanner's own history floor
  (``MIN_PRIMARY_BARS``) onward is re-scanned with the live engines
  (``FeatureEngine.extract`` + ``ScoringEngine.evaluate``) on the data cut at
  that bar, so each row is exactly what an ``--as-of`` run would have shown.
  Each variant in ``VARIANTS`` is a fixed predicate over that row.
* **Variants** -- ``BK_PRE`` (status PRE_BREAKOUT), ``BK_CONFIRMED``,
  ``BK_PRE_OR_CONF``, ``BK_DISPLAY`` (the dashboard's candidate rule minus
  FAILED: PRE, CONFIRMED, WATCH >= 55, DEVELOPING >= 50) and ``BK_READY70``
  (readiness >= 70 whatever the status -- the score on its own).
  ``BK_FAILED`` is reported as a negative check and never qualifies.
  ``C0_random`` (random bars, same exits, same rate as the leader experiment)
  is the bar to beat.
* **Deduplication / exits / periods** -- identical to
  ``coiled_cobra_leader_experiment``: consecutive weekly flags are one episode;
  the primary outcome is the R multiple of the ``trail`` exit (next-open entry,
  2.5 ATR stop, 3 ATR chandelier off the highest close, 52-bar time exit);
  ``lockbox`` = the 52 weeks ending 52 weeks before the last bar, ``dev`` =
  everything before, ``live`` = censored, reported only.  The breakout scanner
  has no planner geometry (no Fib levels), so there is no ``planner`` exit.
* **Qualification** (dev period, all tickers -- the scanner's thresholds were not
  fitted to any ticker set), all must hold: (Q0) >= ``MIN_EPISODES`` episodes;
  (Q1) 95% week-cluster bootstrap CI of mean trail-R above 0; (Q2) paired
  (same-week) mean-R difference vs ``C0_random`` has CI above 0; (Q3) share of
  episodes with a >= 25% drawdown inside 26 weeks no more than ``Q3_DD_MARGIN``
  above the control's.  Only the single best qualifier (highest Q2 lower bound)
  is evaluated **once** on the lockbox: pass = paired difference vs random > 0
  **and** mean R > 0 with cluster t > 1.645.  If nothing qualifies the lockbox
  is left unspent.
* **Report-only (never used to decide)** -- (a) cross-sectional rank IC of
  ``Breakout Readiness`` and each pillar score vs the 13-week close-to-close
  return, mean of weekly Spearman ICs with a week bootstrap CI (dev period);
  (b) forward outcomes for every status, all bars, no dedup; (c) Coiled Cobra
  ``B0_baseline`` on the same ticker-weeks, joined from the newest
  ``leader_experiment_bars_*.csv.gz`` when one exists.

    python -m finance_vibe.breakout_experiment [--tickers A,B] [--out-dir D] [--out r.json]
    python -m finance_vibe.breakout_experiment --from-bars breakout_experiment_bars_<date>.csv.gz
"""
from __future__ import annotations

import argparse
import glob
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
    from finance_vibe import breakout_scanner as bs
    from finance_vibe import coiled_cobra_leader_experiment as lx
    from finance_vibe.analysis_engine import ticker_from_filename
except ImportError:  # pragma: no cover - local direct execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from finance_vibe import config
    from finance_vibe import breakout_scanner as bs
    from finance_vibe import coiled_cobra_leader_experiment as lx
    from finance_vibe.analysis_engine import ticker_from_filename

MODE = "weekly"          # 10y of weekly bars; same horizon and exits as the leader experiment
SEED = 20261001

# --- pre-registered thresholds --------------------------------------------------
MIN_EPISODES = 100
Q3_DD_MARGIN = 0.10
LOCKBOX_T = 1.645
READY_MIN = 70
IC_HORIZON = 13          # weeks, close-to-close, for the report-only rank IC
IC_MIN_NAMES = 10        # weeks with fewer scored names are skipped for IC

CONTROL = "C0_random"
NEGATIVE = "BK_FAILED"
QUALIFYING = ("BK_PRE", "BK_CONFIRMED", "BK_PRE_OR_CONF", "BK_DISPLAY", "BK_READY70")
VARIANTS = (*QUALIFYING, NEGATIVE, CONTROL)
COBRA = "B0_cobra"       # report-only, joined from the leader experiment's bars file

PILLARS = ("Trend Score", "Compression Score", "Momentum Score", "Volume Score", "Structure Score")
STATUSES = (bs.STATUS_PRE, bs.STATUS_CONFIRMED, bs.STATUS_FAILED, bs.STATUS_WATCH, bs.STATUS_DEV)


# ---------------------------------------------------------------------------
# Variant predicates (fixed; inputs are one scanner row)
# ---------------------------------------------------------------------------

def variant_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    """Flag which variants fire on one bar.  ``C0_random`` is set by the caller."""
    status = row.get("Status")
    ready = row.get("Breakout Readiness")
    ready = float(ready) if ready is not None and pd.notna(ready) else -1.0
    pre = status == bs.STATUS_PRE
    conf = status == bs.STATUS_CONFIRMED
    display = bs._is_display_candidate({"Status": status, "Breakout Readiness": ready})
    return {
        "BK_PRE": pre,
        "BK_CONFIRMED": conf,
        "BK_PRE_OR_CONF": pre or conf,
        "BK_DISPLAY": bool(display) and status != bs.STATUS_FAILED,
        "BK_READY70": ready >= READY_MIN and status != bs.STATUS_FAILED,
        NEGATIVE: status == bs.STATUS_FAILED,
    }


# ---------------------------------------------------------------------------
# Per-ticker walk-forward pass (process-pool worker)
# ---------------------------------------------------------------------------

def _ticker_pass(path: str) -> tuple[str, list[dict], dict]:
    """One row per scoreable bar for one ticker.  Causal at every bar."""
    symbol = ticker_from_filename(path)
    try:
        df = bs.normalize_ohlcv(pd.read_csv(path))
    except Exception as exc:
        print(f"{symbol}: error - {exc}", file=sys.stderr)
        return symbol, [], {}

    closes = df["Close"].to_numpy(dtype=float)
    highs, lows = df["High"].to_numpy(dtype=float), df["Low"].to_numpy(dtype=float)
    engine, scorer = bs.FeatureEngine(MODE), bs.ScoringEngine()
    rng = np.random.default_rng([SEED, zlib.crc32(symbol.encode())])
    rows: list[dict] = []
    for idx in range(bs.MIN_PRIMARY_BARS - 1, len(df) - 1):         # need a next bar to enter
        control = bool(rng.random() < lx.P_CONTROL)                  # draw every bar: deterministic
        try:
            res = scorer.evaluate(engine.extract(df.iloc[: idx + 1], symbol, MODE))
        except Exception:
            continue
        rec: dict[str, Any] = {
            "sym": symbol, "idx": idx, "date": df["Date"].iloc[idx], "close": closes[idx],
            "atr": res.get("ATR"), "status": res.get("Status"), "ready": res.get("Breakout Readiness"),
            "penalty": res.get("Penalty"),
            **{p: res.get(p) for p in PILLARS},
            "trend": res.get("Trend"), "volatility": res.get("Volatility"), "momentum": res.get("Momentum"),
            "structure": res.get("Structure"), "mtf": res.get("MTF"),
            "dist_res_atr": res.get("Distance Resistance ATR"), "rvol20": res.get("RVOL20"),
        }
        rec["mu13"], rec["dd13"] = lx._forward(highs, lows, closes, idx, 13)
        rec["mu26"], rec["dd26"] = lx._forward(highs, lows, closes, idx, 26)
        j = idx + IC_HORIZON
        rec["fr13"] = closes[j] / closes[idx] - 1.0 if j < len(df) else np.nan
        flags = variant_flags(res)
        flags[CONTROL] = control
        for name in VARIANTS:
            rec[f"f_{name}"] = bool(flags[name])
        atr = rec["atr"]
        if any(flags.values()) and atr is not None and pd.notna(atr):
            tr = lx.simulate_trail_trade(df, idx + 1, float(atr))
            if tr:
                rec.update({"tr_outcome": tr["outcome"], "tr_r": tr["r"], "tr_censored": tr["censored"],
                            "tr_mfe_r": tr["mfe_r"], "tr_bars": tr["exit_idx"] - idx})
        rows.append(rec)
    return symbol, rows, {"sym": symbol, "last_date": df["Date"].iloc[-1]}


def collect(paths: list[str], workers: Optional[int] = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Run the pass over ``paths`` in a process pool; returns (bars, last_bar_date)."""
    rows: list[dict] = []
    last_dates: list[pd.Timestamp] = []
    print(f"--- Breakout Readiness experiment [{MODE}] --- tickers: {len(paths)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 1) as ex:
        futs = {ex.submit(_ticker_pass, p): p for p in paths}
        for k, fut in enumerate(as_completed(futs), 1):
            _, r, meta = fut.result()
            rows.extend(r)
            if meta:
                last_dates.append(meta["last_date"])
            if k % 20 == 0 or k == len(paths):
                print(f"  {k}/{len(paths)} tickers done ({len(rows)} scoreable bars)", flush=True)
    return pd.DataFrame(rows), max(last_dates) if last_dates else pd.NaT


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _resolved(ep: pd.DataFrame) -> pd.DataFrame:
    """Episodes with a trail trade that ran to an exit (censored trades dropped)."""
    if ep.empty or "tr_r" not in ep.columns:
        return ep.iloc[0:0]
    return ep[ep["tr_r"].notna() & ~ep["tr_censored"].fillna(True).astype(bool)]


def summarize(ep: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """Trail-exit + forward-outcome metrics for one set of deduplicated episodes."""
    out: dict[str, Any] = {"episodes": int(len(ep))}
    if ep.empty:
        return out
    trail = _resolved(ep)
    out["trail"] = {
        **lx.cluster_stats(trail, "tr_r", n_boot, seed),
        "median_r": float(trail["tr_r"].median()) if len(trail) else np.nan,
        "win_rate": float((trail["tr_r"] > 0).mean()) if len(trail) else np.nan,
        "profit_factor": lx._profit_factor(trail["tr_r"]) if len(trail) else None,
        "avg_mfe_r": float(trail["tr_mfe_r"].mean()) if len(trail) else np.nan,
        "big_winner_share": float((trail["tr_r"] >= 5).mean()) if len(trail) else np.nan,
    }
    out["forward"] = lx.forward_stats(ep)
    return out


def weekly_ic(bars: pd.DataFrame, col: str, n_boot: int, seed: int = 0,
              target: str = "fr13", min_names: int = IC_MIN_NAMES) -> dict:
    """Mean per-week Spearman IC of ``col`` vs ``target`` with a week-bootstrap CI."""
    d = bars.dropna(subset=[col, target])
    ics: list[float] = []
    for _, g in d.groupby("date"):
        if len(g) < min_names or g[col].nunique() < 2 or g[target].nunique() < 2:
            continue
        ics.append(float(g[col].rank().corr(g[target].rank())))
    arr = np.array([x for x in ics if pd.notna(x)])
    if len(arr) == 0:
        return {"weeks": 0, "mean_ic": np.nan, "lo": np.nan, "hi": np.nan}
    draws = np.random.default_rng(seed).integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[draws].mean(axis=1)
    return {"weeks": int(len(arr)), "mean_ic": float(arr.mean()),
            "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "share_positive": float((arr > 0).mean())}


def status_ladder(bars: pd.DataFrame) -> dict:
    """Forward outcomes per status over every bar (no dedup)."""
    out: dict[str, Any] = {}
    for s in STATUSES:
        sel = bars[bars["status"] == s]
        fr = sel["fr13"].dropna()
        out[s] = {"bars": int(len(sel)), "mean_fr13": float(fr.mean()) if len(fr) else np.nan,
                  **lx.forward_stats(sel)}
    return out


def qualifies(res: dict, ctrl: dict, ctrl_diff: dict) -> dict:
    """Pre-registered Q0-Q3 checks for one variant (dev period)."""
    tr = res.get("trail", {})
    checks = {
        "Q0_episodes": res.get("episodes", 0) >= MIN_EPISODES,
        "Q1_ci_above_0": bool(tr.get("lo", np.nan) > 0),
        "Q2_beats_random": bool(ctrl_diff.get("lo", np.nan) > 0),
        "Q3_drawdown": bool(res.get("forward", {}).get("dd26_worse25", np.inf)
                            <= ctrl.get("forward", {}).get("dd26_worse25", -np.inf) + Q3_DD_MARGIN),
    }
    return {"checks": checks, "qualified": all(checks.values())}


def cobra_reference(bars: pd.DataFrame, cobra_bars: Optional[pd.DataFrame], n_boot: int) -> dict:
    """Coiled Cobra baseline on the ticker-weeks both experiments scored (dev period)."""
    if cobra_bars is None or cobra_bars.empty:
        return {"available": False}
    keep = ["sym", "date", "idx", "f_B0_baseline", "tr_r", "tr_censored", "tr_mfe_r", "mu13", "dd13", "mu26", "dd26"]
    cb = cobra_bars[[c for c in keep if c in cobra_bars.columns]].copy()
    cb["date"] = pd.to_datetime(cb["date"])
    common = bars[["sym", "date"]].merge(cb, on=["sym", "date"], how="inner")
    dev_dates = set(bars.loc[bars["period"] == "dev", "date"])
    common = common[common["date"].isin(dev_dates)]
    if common.empty:
        return {"available": False}
    common["f_B0_baseline"] = common["f_B0_baseline"].fillna(False).astype(bool)
    common["tr_censored"] = common["tr_censored"].fillna(False).astype(bool)
    ep = lx.dedup_episodes(common[common["f_B0_baseline"]])
    overlap = bars[bars["period"] == "dev"].merge(common[["sym", "date"]], on=["sym", "date"])
    out = {"available": True, "common_bars": int(len(common)), COBRA: summarize(ep, n_boot, SEED)}
    for v in ("BK_PRE_OR_CONF", "BK_DISPLAY", CONTROL):
        out[v] = summarize(lx.dedup_episodes(overlap[overlap[f"f_{v}"]]), n_boot, SEED)
    return out


def run_analysis(bars: pd.DataFrame, last_date: pd.Timestamp, n_boot: int = 2000,
                 cobra_bars: Optional[pd.DataFrame] = None) -> dict:
    """Full pre-registered analysis over a collected bar table."""
    bars = bars.copy()
    bars["period"] = lx.split_periods(bars, last_date)
    dev = bars[bars["period"] == "dev"]

    def episodes(sel: pd.DataFrame, v: str) -> pd.DataFrame:
        return lx.dedup_episodes(sel[sel[f"f_{v}"]])

    result: dict[str, Any] = {
        "protocol": {"seed": SEED, "trail": [lx.TRAIL_INIT_ATR, lx.TRAIL_ATR, lx.TRAIL_MAX_HOLD],
                     "p_control": lx.P_CONTROL, "min_episodes": MIN_EPISODES, "q3_margin": Q3_DD_MARGIN,
                     "ready_min": READY_MIN, "lockbox_weeks": lx.LOCKBOX_WEEKS,
                     "censor_weeks": lx.CENSOR_WEEKS, "n_boot": n_boot},
        "last_date": str(last_date.date()) if pd.notna(last_date) else None,
        "tickers": int(bars["sym"].nunique()),
        "dev": {"all_bars": {"bars": int(len(dev)), **lx.forward_stats(dev)}},
    }
    for v in VARIANTS:
        result["dev"][v] = summarize(episodes(dev, v), n_boot, SEED)

    ctrl_ep = episodes(dev, CONTROL)
    qual: dict[str, Any] = {}
    for v in QUALIFYING:
        diff = lx.paired_diff(_resolved(episodes(dev, v)), _resolved(ctrl_ep), "tr_r", n_boot, SEED)
        qual[v] = {"vs_random": diff, **qualifies(result["dev"][v], result["dev"][CONTROL], diff)}
    result["qualification"] = qual

    winners = [v for v in QUALIFYING if qual[v]["qualified"]]
    result["lockbox"] = {"spent": False, "best": None}
    if winners:
        best = max(winners, key=lambda v: qual[v]["vs_random"]["lo"])
        lock = bars[bars["period"] == "lockbox"]
        ep, ctrl_lock = _resolved(episodes(lock, best)), _resolved(episodes(lock, CONTROL))
        st = lx.cluster_stats(ep, "tr_r", n_boot, SEED)
        diff = lx.paired_diff(ep, ctrl_lock, "tr_r", n_boot, SEED)
        result["lockbox"] = {"spent": True, "best": best, "trail": st, "vs_random": diff,
                             "passed": bool(diff["diff"] > 0 and st["mean"] > 0 and st["t"] > LOCKBOX_T)}

    result["report_only"] = {
        "ic_dev": {c: weekly_ic(dev, c, n_boot, SEED) for c in ("ready", *PILLARS, "penalty")},
        "status_ladder_dev": status_ladder(dev),
        "cobra": cobra_reference(bars, cobra_bars, n_boot),
        "live_open": {v: int(len(episodes(bars[bars["period"] == "live"], v))) for v in VARIANTS},
    }
    return result


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

_f = lx._f


def _variant_line(name: str, r: dict) -> str:
    if r.get("episodes", 0) == 0 or "trail" not in r:
        return f"{name:18s}{r.get('episodes', 0):6d}"
    t, fw = r["trail"], r["forward"]
    return (f"{name:18s}{r['episodes']:6d}{t['n']:6d}{_f(t['mean']):>7s}{_f(t['lo']):>7s}{_f(t['hi']):>7s}"
            f"{_f(t['win_rate'], pct=True):>6s}{_f(t['profit_factor']):>6s}{_f(fw['hit20_13w'], pct=True):>7s}"
            f"{_f(fw['runner_26w'], pct=True):>8s}{_f(fw['dd26_worse25'], pct=True):>8s}")


def format_report(res: dict) -> str:
    ab = res["dev"]["all_bars"]
    header = (f"{'variant':18s}{'eps':>6s}{'res':>6s}{'meanR':>7s}{'CI lo':>7s}{'CI hi':>7s}{'win':>6s}"
              f"{'PF':>6s}{'+20%13':>7s}{'runner':>8s}{'DD<-25':>8s}")
    lines = [
        f"Breakout Readiness walk-forward   last bar {res['last_date']}   tickers {res['tickers']}",
        "Primary outcome: mean R of the trail exit, dev period. C0_random is the bar to beat.",
        "",
        f"all scoreable bars (dev): {ab['bars']}  hit+20%/13w {_f(ab['hit20_13w'], pct=True)}  "
        f"runner(2x/26w) {_f(ab['runner_26w'], pct=True)}  DD26<=-25% {_f(ab['dd26_worse25'], pct=True)}",
        header,
    ]
    lines += [_variant_line(v, res["dev"][v]) for v in VARIANTS]
    lines += ["", "== QUALIFICATION (dev) =="]
    for v, q in res["qualification"].items():
        marks = " ".join(f"{k.split('_')[0]}={'Y' if ok else 'n'}" for k, ok in q["checks"].items())
        d = q["vs_random"]
        lines.append(f"{v:18s}{'QUALIFIED' if q['qualified'] else 'no':10s}{marks}   "
                     f"vs random: {_f(d['diff'])} [{_f(d['lo'])}, {_f(d['hi'])}]")
    lb = res["lockbox"]
    if lb["spent"]:
        lines += ["", f"LOCKBOX spent once on {lb['best']}: mean R {_f(lb['trail']['mean'])} "
                      f"[{_f(lb['trail']['lo'])}, {_f(lb['trail']['hi'])}] t={_f(lb['trail']['t'])}  "
                      f"vs random {_f(lb['vs_random']['diff'])}  -> {'PASS' if lb['passed'] else 'FAIL'}"]
    else:
        lines += ["", "LOCKBOX left unspent: no variant qualified on the development period."]

    ro = res["report_only"]
    lines += ["", "== REPORT ONLY: weekly rank IC vs 13w return (dev) =="]
    for c, ic in ro["ic_dev"].items():
        lines.append(f"{c:18s} IC {_f(ic['mean_ic'], 3)} [{_f(ic['lo'], 3)}, {_f(ic['hi'], 3)}]  "
                     f"weeks {ic['weeks']}  positive {_f(ic.get('share_positive'), pct=True)}")
    lines += ["", "== REPORT ONLY: status ladder, all dev bars (no dedup) ==",
              f"{'status':20s}{'bars':>8s}{'mean13w':>9s}{'+20%13':>8s}{'runner':>8s}{'DD<-25':>8s}"]
    for s, d in ro["status_ladder_dev"].items():
        lines.append(f"{s:20s}{d['bars']:8d}{_f(d['mean_fr13'], pct=True):>9s}{_f(d['hit20_13w'], pct=True):>8s}"
                     f"{_f(d['runner_26w'], pct=True):>8s}{_f(d['dd26_worse25'], pct=True):>8s}")
    cb = ro["cobra"]
    if cb.get("available"):
        lines += ["", f"== REPORT ONLY: vs Coiled Cobra on shared ticker-weeks (dev, {cb['common_bars']} bars) ==",
                  header]
        lines += [_variant_line(v, cb[v]) for v in (COBRA, "BK_PRE_OR_CONF", "BK_DISPLAY", CONTROL)]
    else:
        lines += ["", "Coiled Cobra reference: no leader_experiment_bars_*.csv.gz found (skipped)."]
    lines += ["", "live/open (censored) episodes: " + ", ".join(f"{k}={v}" for k, v in ro["live_open"].items())]
    return "\n".join(lines)


def _newest(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered Breakout Readiness walk-forward experiment")
    ap.add_argument("--tickers", help="Comma-separated tickers (smoke tests); default = whole raw dir")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out-dir", help="Directory for bars/result files (default: the weekly logs dir)")
    ap.add_argument("--out", help="Write the JSON result here (default: <out-dir>/breakout_experiment_<date>.json)")
    ap.add_argument("--from-bars", help="Skip the walk-forward: re-run the analysis from a saved bars .csv.gz")
    ap.add_argument("--cobra-bars", help="Leader experiment bars .csv.gz for the Cobra reference "
                                         "(default: newest in the weekly logs dir)")
    args = ap.parse_args(argv)

    logs_dir = config.get_mode_config(MODE)["logs_dir"]
    out_dir = args.out_dir or logs_dir
    stamp = datetime.now().strftime("%Y-%m-%d")
    if args.from_bars:
        bars = pd.read_csv(args.from_bars, parse_dates=["date"])
        last_date = bars["date"].max() + pd.Timedelta(weeks=1)
    else:
        raw_dir = config.get_mode_config(MODE)["raw_dir"]
        wanted = {t.strip().upper() for t in args.tickers.split(",")} if args.tickers else None
        paths = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".csv")
                       if not wanted or ticker_from_filename(os.path.join(raw_dir, f)) in wanted)
        if not paths:
            ap.error(f"no raw CSVs matched in {raw_dir}")
        bars, last_date = collect(paths, args.workers)
        os.makedirs(out_dir, exist_ok=True)
        bars_path = os.path.join(out_dir, f"breakout_experiment_bars_{stamp}.csv.gz")
        bars.to_csv(bars_path, index=False)
        print(f"Saved bars: {bars_path}")
        bars["date"] = pd.to_datetime(bars["date"])

    for c in [f"f_{v}" for v in VARIANTS] + ["tr_censored"]:
        if c in bars.columns:
            bars[c] = bars[c].fillna(False).astype(bool)

    cobra_path = args.cobra_bars or _newest(os.path.join(logs_dir, "leader_experiment_bars_*.csv.gz"))
    cobra_bars = pd.read_csv(cobra_path, parse_dates=["date"]) if cobra_path and os.path.exists(cobra_path) else None
    if cobra_path and cobra_bars is not None:
        print(f"Cobra reference: {cobra_path}")

    res = run_analysis(bars, last_date, n_boot=args.bootstrap, cobra_bars=cobra_bars)
    out_path = args.out or os.path.join(out_dir, f"breakout_experiment_{stamp}.json")
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2, default=lx._json_default)
    print(format_report(res))
    print(f"\nSaved result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
