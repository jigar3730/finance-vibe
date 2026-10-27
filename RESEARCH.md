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

### In-sample, daily (run once on 2026-10-27)

Analysis code `f24a4d5` (committed before the run); labels `70e30f2`, breakout rubric v1.0,
snapshot `data/research/snapshots/2026-10-27/`. Full report with every column:
`data/research/insample_2026-10-27/report.md`. Signal dates 2022-02-23 → 2026-03-31,
in-universe events only. **Survivorship is reduced, not removed** (no delisted names, §8.4).

| Signal | Count | Dates | % up | Mean R10 | Mean − SPY | Mean − random | p vs SPY | p vs random | p (Holm) | Result |
|--------|------:|------:|-----:|---------:|-----------:|--------------:|---------:|------------:|---------:|--------|
| PRE_BREAKOUT | 927 | 383 | 51.1% | +0.24% | −0.30% | −0.43% | 0.497 | 0.662 | 1.000 | **FAIL** |
| TRIGGERED | 7,468 | 896 | 53.0% | +0.50% | −0.02% | −0.15% | 0.722 | 0.858 | 1.000 | **FAIL** |
| CONFIRMED | 2,372 | 761 | 54.3% | +1.36% | +0.83% | +0.69% | 0.238 | 0.195 | 0.713 | **FAIL** |

- PRE_BREAKOUT follow-through (TRIGGERED/CONFIRMED within 10 sessions): 61.1% vs 39.5% for
  its random draws. Reported, not a pass criterion.
- Means are per event; the §7 test is on per-date means (each signal date weighted equally),
  which is why PRE_BREAKOUT's t vs SPY is +0.01 while its per-event mean is −0.30%.
- **Decision (§7): no signal passes in-sample. The holdout is not run and stays unopened.
  All three signals are dropped, not tuned.**

By signal year (descriptive, not a pass criterion): no year carries a result. PRE_BREAKOUT
2023 (503 of 927 events) has mean − random −0.49% (unadjusted p 0.92). The best single
cell is CONFIRMED 2024 (mean − random +1.45%, unadjusted p 0.052), and that is one of 15
cells, uncorrected.

Robustness only (42 events and 555 draws flagged `split_suspect` excluded): same picture.
p (Holm) is 1.000 / 1.000 / 0.992 for PRE_BREAKOUT / TRIGGERED / CONFIRMED, all FAIL.

### In-sample, weekly (§11; run once on 2026-10-27)

Analysis code `77e1826` (committed before the run); labels `d29350e`, breakout rubric v1.0,
same snapshot as daily. Full report: `data/research/insample_weekly_2026-10-27/report.md`.
Signal weeks 2018-05-07 → week of 2026-03-23 (389 weeks), in-universe events only.
**Survivorship is reduced, not removed** (§8.4).

| Signal | Count | Weeks | % up | Mean R4W | Mean − SPY | Mean − random | p vs SPY | p vs random | p (Holm) | Result |
|--------|------:|------:|-----:|---------:|-----------:|--------------:|---------:|------------:|---------:|--------|
| PRE_BREAKOUT | 464 | 189 | 56.7% | +1.25% | +0.00% | −0.06% | 0.439 | 0.534 | 1.000 | **FAIL** |
| TRIGGERED | 3,682 | 375 | 53.1% | +0.58% | −0.01% | −0.09% | 0.549 | 0.756 | 1.000 | **FAIL** |
| CONFIRMED | 718 | 283 | 52.0% | +3.26% | +2.30% | +2.03% | 0.011 | 0.016 | 0.049 | **PASS** |

- PRE_BREAKOUT follow-through (TRIGGERED/CONFIRMED within 4 weeks): 48.5% vs 28.1% random.
  Reported, not a pass criterion.
- **Decision (§11): weekly CONFIRMED passes in-sample and is eligible for the one holdout
  run. PRE_BREAKOUT and TRIGGERED are dropped, not tuned.** The holdout has not been
  opened.
- Read with care. These caveats change nothing in the decision:
  - The pass is marginal (0.049 against 0.05).
  - The §11 overlap caveat applies: successive 4-week windows overlap, so the plain
    t-test overstates significance.
  - The excess is concentrated. % up is only 52%, while the mean is +2%, so a minority
    of large winners carries it.
- Robustness only: with the 28 CONFIRMED events (and their draws) flagged `split_suspect`
  (one-week moves ≥ 40%) excluded, mean − random falls to +1.10% and p (Holm) to 0.400.
  The in-sample result depends on a few extreme weeks.
- By year (descriptive, not a pass criterion): CONFIRMED is positive against random in
  7 of 9 years. 2020 is the largest cell (95 events, +5.36%, unadjusted p 0.046). Negative
  years are 2018 (−0.02%) and 2021 (−0.45%).

## 10. Variant log

Every run: date, commit, what changed, which hypotheses, result. Includes failed and abandoned runs.

| Date | Commit | Change | Hypotheses | Result |
|------|--------|--------|------------|--------|
| 2026-10-27 | freeze commit | Freeze completion, formatting only: loose holdout / costs / pass-bar / H6 notes folded into §4 and §7 with no change in meaning; status set to FROZEN. No forward returns computed yet. | — | — |
| 2026-10-27 | simplification commit | Simplified to 3 signals, 10-day horizon, changed before any forward returns computed. Dropped H1–H6, Readiness score, gap split, Top Setups, ATR units, MFE/MAE, HIT, R5/R20, costs, B2 and the bootstrap; added SPY as a benchmark. | E1–E3 | — |
| 2026-10-27 | simplification commit | Scope note: locked Goal added; failed signals are dropped, not tuned (§7). Daily test unchanged; weekly gets its own pre-registered addendum before any weekly returns. | — | — |
| 2026-10-27 | 70e30f2 | Blind label run (fix #2), breakout rubric v1.0: `data/research/labels_2026-10-27/` from snapshot `data/research/snapshots/2026-10-27/` (545 raw files hashed in `manifest.json`). In-sample, in-universe events only (max signal date 2026-03-31): E1 927, E2 7468, E3 2372; 20 draws each (none short). DQ: 0 missing next open / short window / session gap / SPY missing; 42 events and 555 draws flagged `split_suspect` (one-bar moves ≥ 40%, kept). Labels only, no results computed. | E1–E3 | — |
| 2026-10-27 | f24a4d5 | In-sample analysis run once (§4–§7 as frozen) on `labels_2026-10-27`. No threshold, horizon or definition changed. | E1–E3 | All FAIL in-sample (p Holm 1.000 / 1.000 / 0.713). Holdout not opened. Signals dropped. |
| 2026-10-27 | — | **Exploratory, not a finding:** PRE_BREAKOUT anticipates the scanner's own state change (61% vs 40% reach TRIGGERED/CONFIRMED in 10 sessions), but that does not show up in returns. Any test built on this needs its own pre-registration and a fresh holdout; the 2026-04 → holdout cannot be used to pick it. | — | — |
| 2026-10-27 | — | **Exploratory, not a finding:** CONFIRMED has the largest per-event excess (+0.69% vs random) but heavy-tailed noise (t 0.86). A new hypothesis (e.g. a longer horizon or another universe) would be a new pre-registered test, not a re-run of this one. | — | — |
| 2026-10-27 | weekly addendum commit | Weekly addendum pre-registered (§11) before any weekly forward return was computed. Daily sections unchanged. | W-E1–W-E3 | — |

## 11. Weekly addendum (pre-registered 2026-10-27, before any weekly return)

Same question as §1, on weekly bars: do PRE_BREAKOUT, TRIGGERED and CONFIRMED (first bar
of a run) beat the market over the next 4 weeks? Everything not stated here is as in §3–§7.

**Data.** `scan_history(bars, symbol, "weekly", "weekly")` over the weekly raw files of the
same snapshot the daily run used (`data/research/snapshots/2026-10-27/weekly/`, 268 files,
Yahoo auto-adjusted, 10y: 2016-10 → 2026-10). Weekly bars are dated Monday and hold the whole
week. A bar's signal is known at that week's close (Friday = Monday + 4 days).

**Events.** E1/E2/E3 exactly as §3, on consecutive weekly `scan_history` rows.

**Universe, adapted to weekly bars.** At week t, using only weekly bars up to and including t:
- not in `RESEARCH_EXCLUDED_SYMBOLS` (same list);
- ≥ 52 weekly bars of history (≈ the daily 252 sessions). Listing age is the weekly bar
  count; the files start 2016-10, so names listed earlier are counted from there;
- weekly Close ≥ $5;
- top 200 by trailing 4-bar median of weekly Close × weekly Volume (≈ the daily 20-session
  window), ties A→Z.
- Candidates are the 268 weekly raw files. The 46 daily tickers that have no weekly file
  are not in the weekly test.

**Entry, exit, label.** Entry is the open of week t+1 (its first session). Exit is the close
of week t+4 (the 4th week, counting the entry week as the 1st).
`R4W = Close[t+4] / Open[t+1] − 1`. No costs, no stops.
E1 only: `REACH4W` = Status is BREAKOUT_TRIGGERED or BREAKOUT_CONFIRMED on any weekly bar
t+1 … t+4.

**Benchmarks.** SPY over the same weekly entry and exit bars
(`Close[exit week] / Open[entry week] − 1`). Twenty date-matched random stocks per event,
drawn without replacement from that week's universe members (never the event's own
ticker), fixed seed, same rules.

**Report and decision rule.** As §6–§7, with R4W in place of R10. Per-date clustering
uses the signal week. A signal passes in-sample only if it beats both SPY and random:
the larger of the two one-sided p-values, Holm across the 3 weekly signals, p < 0.05.
The holdout runs once, only for signals that passed, with the same test. Failed
signals are dropped, not tuned. This is a separate family from the daily test, and
daily results do not change it.

**Holdout seal.** The seal is on the date the signal is known (week end). In-sample is
signals known **on or before 2026-03-31**: the last in-sample bar is the week of
2026-03-23 (closes 03-27). The week of 2026-03-30 closes 04-03, so it is holdout.
In-sample starts at the first available signal. With 80 bars of `scan_history`
warm-up, that is about 2018-04, roughly 410 signal weeks. Forward bars after
2026-03-31 may be read to label late-March signals, as in the daily test.

**Sample size.** Weekly gives far fewer signal dates than daily: about 410 weeks
against about 1,000 sessions, and fewer events per date. Power is lower, so a FAIL
here is weaker evidence of no effect than the daily FAIL. Each event's 4-week window
overlaps the next 3 signal weeks, so successive weekly means are correlated. The plain
date-clustered t-test does not correct for that, which makes a false PASS more likely,
not less. This is stated, and the rule is not changed for it. (The daily 10-session
windows overlap the same way.)

**Survivorship.** As §8.4: reduced, not removed.
| 2026-10-27 | d29350e | Weekly blind label run (§11), breakout rubric v1.0: `data/research/labels_weekly_2026-10-27/` from the same snapshot `snapshots/2026-10-27/` (weekly files). In-sample, in-universe: E1 464, E2 3682, E3 718 over 389 signal weeks (2018-05-07 → week of 2026-03-23, known 03-27); 20 draws each (none short). DQ: 0 missing next open / short window / gap / SPY missing; 49 events and 952 draws `split_suspect` (kept). scan_history skipped BMNR, CRCL, FIG (< 80 weekly bars). Labels only, no results computed. | W-E1–W-E3 | — |
| 2026-10-27 | 77e1826 | Weekly in-sample analysis run once (§11 as pre-registered) on `labels_weekly_2026-10-27`. No threshold, horizon or definition changed. | W-E1–W-E3 | PRE_BREAKOUT FAIL (p Holm 1.000), TRIGGERED FAIL (1.000), **CONFIRMED PASS (0.049)**. Holdout not opened. |
| 2026-10-27 | — | **Exploratory, not a finding:** weekly CONFIRMED's in-sample pass does not survive excluding the 28 split-flagged events (p Holm 0.400). An overlap-robust test (non-overlapping 4-week windows, or HAC errors) would be a new pre-registered test. It cannot replace the §11 rule, and it cannot be chosen after seeing the holdout. | — | — |
