# RESEARCH.md — breakout scanner validation plan

Status: FROZEN — no forward return was computed before the freeze.
Frozen on: 2026-10-27, simplified the same day before any forward return (commit hash:
`git log -S"Simplified to 3 signals" --format=%H -- RESEARCH.md`)

Goal (locked 2026-10-27): produce solid daily and weekly signals that are backtested,
data-backed and actionable manually. Not a trading system: no paper trading, position
sizing, automation or execution. “Solid” = passes the frozen in-sample test and the
holdout vs SPY and date-matched random entries. “Actionable” = known by the close, with
entry, invalidation level and holding window.

Rule: nothing in this file changes after it is frozen except the "Results" and
"Variant log" sections. A changed hypothesis is a new hypothesis, logged as exploratory.

## 1. Question

Do the breakout scanner's PRE_BREAKOUT, TRIGGERED and CONFIRMED events (first bar of a
run) beat the market over the next 10 sessions?

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

Repeat events on the same ticker are allowed; same-day correlation is handled in §7.
A ticker's first history row never counts as an event (unknown previous state).
An event counts only if its ticker is in the point-in-time universe on the signal date.

## 4. Entry, exit, label

- Signal known at the close of bar t.
- Entry: **open of session t+1**. Never the signal bar's close or the breakout level.
- Exit: **close of session t+10** (the 10th session, counting the entry session as the 1st).
- Label: `R10 = Close[t+10] / Open[t+1] − 1`. No costs, no stops, no other labels.
- E1 only: `REACH10` = the ticker's Status is BREAKOUT_TRIGGERED or BREAKOUT_CONFIRMED
  on any bar t+1 … t+10.

## 5. Benchmarks

- SPY: `Close[exit date] / Open[entry date] − 1`, on the event's own entry and exit dates.
- Random: for each event, 20 date-matched random stocks drawn without replacement from
  that date's universe members (excluding the event's ticker), fixed seed, same entry,
  exit and REACH10 rules.

## 6. Report (per signal: E1, E2, E3)

- Count.
- % with R10 > 0.
- Mean of (R10 − SPY R10).
- Mean of (R10 − mean R10 of its 20 random draws).
- E1 only: REACH10 rate vs the random draws' REACH10 rate.

Everything else is **exploratory** and cannot be called a finding without its own holdout test.

## 7. Statistics and decision rules

- Date-clustered t-test: average a signal's excess within each signal date, then a
  one-sided one-sample t-test on those daily means (H1: mean > 0). Done separately vs
  SPY and vs random.
- In-sample pass: the signal beats **both** (its p-value is the larger of the two),
  p < 0.05 after Holm across the 3 signals.
- Holdout: run exactly once, only for signals that passed in-sample, with the same test
  (Holm across the signals taken there). A signal passes only if it passes in-sample
  **and** on the holdout.
- If nothing passes in-sample, or a pass fails the holdout, that signal is dropped, not tuned.
- Report every signal, pass or fail. No tuning thresholds on in-sample results.
- In-sample: 2022-01-01 → 2026-03-31. **Holdout: 2026-04-01 onward — not opened until
  in-sample results are written below.** Both periods are defined by signal date.
- Scope: this plan is the daily test. Weekly gets its own pre-registered addendum before
  any weekly returns are computed.

## 8. Prerequisites before running

1. Fix #1: point-in-time universe rule, applied to scan_history rows (done, `951065a`;
   `data/research/universe_report_2026-10-27.md` reports the events it removes).
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
| 2026-10-27 | simplification commit | Simplified to 3 signals, 10-day horizon, changed before any forward returns computed. Dropped H1–H6, Readiness score, gap split, Top Setups, ATR units, MFE/MAE, HIT, R5/R20, costs, B2 and the bootstrap; added SPY as a benchmark. | E1–E3 | — |
| 2026-10-27 | simplification commit | Scope note: locked Goal added; failed signals are dropped, not tuned (§7). Daily test unchanged; weekly gets its own pre-registered addendum before any weekly returns. | — | — |
| 2026-10-27 | 70e30f2 | Blind label run (fix #2), breakout rubric v1.0: `data/research/labels_2026-10-27/` from snapshot `data/research/snapshots/2026-10-27/` (545 raw files hashed in `manifest.json`). In-sample, in-universe events only (max signal date 2026-03-31): E1 927, E2 7468, E3 2372; 20 draws each (none short). DQ: 0 missing next open / short window / session gap / SPY missing; 42 events and 555 draws flagged `split_suspect` (one-bar moves ≥ 40%, kept). Labels only, no results computed. | E1–E3 | — |
