# RESEARCH.md — breakout scanner validation plan

Status: FROZEN — no forward return was computed before the freeze.
Frozen on: 2026-10-27 (commit hash: the commit that sets this line;
`git log -S"Status: FROZEN" --format=%H -- RESEARCH.md`)

Rule: nothing in this file changes after it is frozen except the "Results" and
"Variant log" sections. A changed hypothesis is a new hypothesis, logged as exploratory.

## 1. Question

Do the breakout scanner's events or its Readiness score identify long entries that
beat comparable random entries over 5–20 trading days, after costs?

## 2. Data

- Source: `scan_history()` over `data/raw/daily/` (Yahoo, auto-adjusted), 2022-01 → 2026-10.
- Universe: see §8 (prerequisites). Until fix #1 lands, results carry survivorship bias
  and are labelled PRELIMINARY.
- One row per (Symbol, Date). Signal is known at that bar's close.

## 3. Events (first bar of a run only)

| ID | Event | Definition |
|----|-------|------------|
| E1 | PRE_BREAKOUT | Status == PRE_BREAKOUT and previous bar's Status != PRE_BREAKOUT |
| E2 | TRIGGERED | Status == BREAKOUT_TRIGGERED and previous bar not TRIGGERED/CONFIRMED |
| E3 | CONFIRMED | Status == BREAKOUT_CONFIRMED and previous bar not TRIGGERED/CONFIRMED |

Repeat events on the same ticker are allowed; overlap is handled in §6.
A ticker's first history row never counts as an event (unknown previous state).

## 4. Entry, exit, labels

- Entry: **next session's open** after the signal bar. Never the signal bar's close or the breakout level.
- ATR unit: ATR(14) on the signal bar.
- Labels, all measured from the entry open:
  - R5, R10, R20: close-to-entry return after 5/10/20 sessions, in ATR units and in %.
  - MFE20 / MAE20: max favourable / adverse excursion over 20 sessions, in ATR.
  - HIT: +2 ATR touched before −1 ATR within 20 sessions (same-bar touch of both = loss).
- Costs: subtract 0.10% round trip from every trade, plus an extra 0.20% (½ spread proxy)
  when price < $10.
- Primary label: **R10 in ATR units, net of costs.** Others are secondary.

## 5. Baselines

- B1 (primary), date-matched random: for each event, 20 random ticker-days from the
  same universe on the same date, same entry/label rules. Controls for market regime.
- B2, same-ticker random: 20 random dates for the same ticker within ±250 sessions.
  Controls for "this stock just trended".
- B3, SPY over the same entry/exit window.

Excess = event label − mean label of its matched B1 draws.

## 6. Hypotheses (pre-registered, one-sided)

| ID | Hypothesis | Test statistic |
|----|-----------|----------------|
| H1 | E3 CONFIRMED events have positive mean excess R10 vs B1 | mean excess |
| H2 | E2 TRIGGERED events have positive mean excess R10 vs B1 | mean excess |
| H3 | E1 PRE_BREAKOUT events reach TRIGGERED within 10 sessions more often than B1, **and** have positive excess R20 | rate diff; mean excess |
| H4 | Among non-triggered rows (WATCH/DEVELOPING/PRE_BREAKOUT), Readiness predicts R10 cross-sectionally | mean daily Spearman IC > 0 |
| H5 | Within E3, events whose open was ≤ Resistance beat events that gapped over it | difference in mean excess |
| H6 | The dashboard's Top Setups (top 3 per day by Rank Score) beat B1 | mean excess |

Everything else is **exploratory** and cannot be called a finding without its own holdout test.

## 7. Statistics and decision rules

- Standard errors clustered by signal date (events on one day are correlated);
  confirm with a weekly block bootstrap (2,000 draws).
- Multiple testing: Holm correction across H1–H6 at α = 0.05. H6 (Top Setups) stays in
  the family.
- Economic bar: a hypothesis "passes" only if mean excess R10 ≥ +0.15 ATR per trade, net
  of costs, **and** it is significant after Holm.
- Report every hypothesis, pass or fail. No tuning thresholds on in-sample results.
- In-sample: 2022-01-01 → 2026-03-31. **Holdout: 2026-04-01 onward — not opened until
  in-sample results are written below.** The holdout is run exactly once, only for
  hypotheses that passed in-sample.

## 8. Prerequisites before running

1. Fix #1: point-in-time universe rule (e.g. top N by trailing 20-day dollar volume as of t),
   applied to scan_history rows. Report how many events the rule removes.
2. Fix #2: forward labels computed from one consistent price series, stored with the run.
3. Breakout rubric version recorded with every result (fix #5).
4. Survivorship cannot be fully removed with Yahoo data (no delisted names). Every result
   states this; a later re-test on a survivorship-free source is required before real money.

## 9. Results

(empty until frozen and run)

## 10. Variant log

Every run: date, commit, what changed, which hypotheses, result. Includes failed and abandoned runs.

| Date | Commit | Change | Hypotheses | Result |
|------|--------|--------|------------|--------|
| 2026-10-27 | freeze commit | Freeze completion, formatting only: loose holdout / costs / pass-bar / H6 notes folded into §4 and §7 with no change in meaning; status set to FROZEN. No forward returns computed yet. | — | — |
