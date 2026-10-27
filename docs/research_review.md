# Research review — pass 1 (definitions & leakage)

Reviewed commit `534434d` (2026-10-20). Code and docs only; `data/` was not available,
so every frequency claim below is something to measure, not a finding.

Goal of the project: turn the breakout scanner from a descriptive daily snapshot into a
research system that can produce validated, actionable signals.

## Fatal — blocks any valid test

1. **Universe is selected by same-day activity and rebuilt every run.**
   `data/tickers.py` = 70-name manifest + Yahoo `most_actives` (250), re-pulled daily.
   - Volume feeds RVOL, so the scanner partly selects its own breakouts.
   - Daily CSVs are not a stable panel (names enter/leave by that day's volume).
   - `--as-of` replays use *today's* list → survivorship / look-ahead bias.
   - Fix: frozen research universe with point-in-time membership per date
     (e.g. top N by 20-day dollar volume computed as of date t). Store membership.

2. **No point-in-time raw data.** Raw is wiped and re-downloaded with `auto_adjust=True`.
   Past signals can't be reproduced; later re-adjusted prices won't match logged levels.
   - Fix: append-only raw store (unadjusted OHLCV + splits/dividends), or daily raw
     snapshots. Forward returns from one consistent series, entry at **next open**.

3. **Scanner evaluates only the last bar** (`FeatureEngine.extract` → `iloc[-1]`).
   No state history exists. Indicators are already vectorized; classify/score are not.
   - Fix: `scan_history()` emitting states/status/score for every bar.
   - Parity test: last bar of `scan_history()` == `run_scanner()` output; golden stays IDENTICAL.
   - **Lookahead trap:** weekly/monthly trend flags use the still-forming bar. At date t,
     higher-timeframe features must come from the partial bar *as of t*, never the
     completed week/month. Needs its own test.

## Major

4. **Dashboard contradicts the validation rule.** `docs/breakout.md` says do not rank or
   select; `web/playbook.py` Top Setups ranks by Readiness + status bonus (+20/+15/+10),
   merges with Cobra Score via `max()` on one scale, and prints trigger/stop/2R-3R.
   Either gate it as unvalidated or make its Rank Score a pre-registered hypothesis.

5. **No version stamp on breakout output.** Definitions changed 2026-10-08, -10, -20.
   Add a breakout rubric version column + bump rule (like Cobra `RUBRIC_VERSION`).
   Regenerate history with `scan_history()` rather than trusting old logs.

6. **Statuses are bar states, not events.** Resistance rolls up the day after a breakout,
   so CONFIRMED usually flips to TRIGGERED/WATCH next bar. Define the event as the first
   bar of a run (`run_start` already exists). Add a gap flag and event-time extension.

7. **PRE_BREAKOUT fired 0/276 on 2026-10-26** (8-way AND). Measure its historical base
   rate; if too rare to test, loosen it *before* looking at outcomes.

8. **Readiness score = ~20 hand-set parameters** "kept for later weight tuning" → overfit
   risk. Don't tune first; test raw features as monotone hypotheses and the score as one
   fixed composite.

## Minor

9. Live Price beside close-based states invites intraday decisions on EOD signals.
10. Redundant WATCH branches in `classify_status`.
11. Dotted symbols (BRK.B) silently dropped; no delisting handling.

## Plan (in order)

1. Point-in-time universe + per-date membership.
2. Append-only raw store.
3. `scan_history()` + parity test + forming-bar lookahead test.
4. Event table + forward labels (5/10/20d returns, MFE/MAE in ATR, +2ATR-before-−1ATR),
   entry next open, baselines: random entry same universe/dates, sector ETF, SPY.
5. `RESEARCH.md`: pre-registered hypotheses, frozen holdout, variant log — before
   looking at any results.
