# PM product review — 2026-10

Reviewer stance: a portfolio manager who checks the market every morning before the open
and acts manually. Goal under review (locked): **solid daily and weekly signals that are
backtested, data-backed and acted on manually.** Not a trading system; paper trading and
execution are out of scope.

What I looked at, on Friday 2026-10-30 after the 18:00 runs:

- The running dashboard: `/`, `/market`, `/changes`, `/runs`, `/breakout/daily/2026-10-30`,
  `/cobra/{daily,weekly}/2026-10-30`.
- Today's CSVs in `data/logs/{daily,weekly}/`: `breakout_setups_2026-10-30.csv` (277 daily /
  267 weekly rows), `coiled_cobra_setups_2026-10-30.csv` (6 / 7 rows),
  `data_quality_2026-10-30.csv`, plus the eight prior daily runs (2026-10-20 → 10-29).
- The research outputs: `data/research/insample_2026-10-27/report.md`,
  `data/research/insample_weekly_2026-10-27/report.md`, `RESEARCH.md` §9–§11.
- Code: `web/app.py`, `web/playbook.py`, `web/insights.py`, `web/templates/*`.

No code was changed for this review.

## The one-paragraph verdict

The project has done the hard, honest part: a pre-registered backtest with a sealed holdout.
It found that **every daily signal fails** and that **one signal, weekly BREAKOUT_CONFIRMED
(first bar of a run), passes in-sample** (+2.3% vs SPY and +2.0% vs random over 4 weeks,
p Holm 0.049, holdout not yet opened). The dashboard does not know any of this. The home
page still ranks by Coiled Cobra Score (never tested) and Breakout Readiness (dropped from
the research plan), shows six cards from untested or failed signals, attaches 2R/3R trade
levels that bear no relation to what was tested, and does not show today's weekly
CONFIRMED names at all. The fastest way to "data-backed signals" is not a new indicator; it
is making the dashboard say what the research already found.

---

## Top 3 enhancements (ranked by impact on the morning read)

### 1. Evidence-first home page: lead with the signal that passed, label everything else

**Problem (PM view).** At 08:30 I want one answer: *is there anything today with evidence
behind it?* The home page can't tell me.

- `/` (Top Setups) today shows weekly **RIG, LLY, KVUE** and daily **AAPL, CCL, MSFT**.
  Ranking is `playbook.candidates()` (`web/playbook.py:124-186`): Cobra Score, or Readiness +
  a +20/+15/+10 status bonus, `max()`-merged plus a +10 agreement bonus. Neither input has
  been tested. Cobra is not in `RESEARCH.md` at all; Readiness was removed from the plan
  ("Dropped H1–H6, Readiness score…", `RESEARCH.md` §10).
- The daily section is built on signals that **failed** in-sample: CCL is a PRE_BREAKOUT
  (daily E1: mean −0.30% vs SPY, −0.43% vs random over 10 sessions, p Holm 1.000).
- The one signal that **passed**, weekly CONFIRMED, fired on five names this week: **GPN,
  AVPT, GMAB, ADT, BHC** (`logs/weekly/breakout_setups_2026-10-30.csv`). None is on `/`.
  You'd only find them via Runs → Breakout weekly → the CONFIRMED chip. GMAB was CONFIRMED last
  week too, so it is a continuation, not the tested first-bar event. The other four were not
  in last week's file at all (universe churn, see #3). So the CSVs alone can't tell a fresh
  event from a stale one.
- The levels on each card aren't what was tested. The test is *entry at next open, exit 4
  weeks later, no stops, no targets* (`RESEARCH.md` §11). The cards show trigger / stop /
  2R / 3R instead. All three weekly cards are flagged `wide stop` (RIG risk 23.5%, LLY 17.4%,
  KVUE 14.8%). RIG's 3R target of 11.13 is +60% from the 6.94 live price, and LLY's is 2,010.40.
  Those numbers look precise but carry no evidence.
- `docs/breakout.md:321-324` still says "Not validated… Do not use it to rank or select
  entries", while the home page ranks and selects. The research has since run, so the doc
  is stale too.

**Proposed change.** Restructure `/` into three bands, top to bottom:

1. **Tested signals.** Weekly CONFIRMED *first-bar events* this week, as cards. Each card has
   one evidence line read from the latest in-sample scorecard, e.g. *"Weekly CONFIRMED, in-sample
   2018-05 → 2026-03: n=718, 52% up, mean +2.3% vs SPY over 4 weeks. Fragile: p 0.40 without
   split-flagged events. Holdout: not run."* Show "extension at signal" (GMAB 6.6 ATR, ADT 4.6
   ATR) as a fact. Don't show targets.
2. **Watchlist: untested.** Coiled Cobra setups (both modes), collapsed, plainly labelled
   untested.
3. **Failed in-sample.** Daily PRE/TRIGGERED/CONFIRMED and weekly PRE/TRIGGERED. A count and a
   link, not cards.

To support band 1, the breakout stage adds a `Prev Status` / `Event` column computed from the
raw file's own previous bar via `scan_history()`. That way first-bar detection doesn't depend
on yesterday's universe. The columns are descriptive only, with no change to Status or score.
Drop the 2R/3R targets and the Rank Score. Keep trigger and support as chart lines.

**Holdout status.** The §11 weekly holdout is scheduled for 2035-09-07 (`RESEARCH.md` §11).
Until then, weekly CONFIRMED is **unproven**: it passed in-sample only, and band 1 should
label it that way.

**Why this beats the alternatives.** Tweaking the Rank Score weights would be tuning an
untested composite, the exact overfit the research plan forbids. Adding indicators or scanners
widens the untested surface. Deleting Cobra and daily outright throws away useful watchlists.
This change does no new modelling. It makes the page's hierarchy match the evidence, and it
turns the research spend into something I see every morning.

**Effort: M.** Files: `scanners/breakout.py` (event columns from `scan_history`),
`config.py` (path to the latest scorecard dir), new small loader in `research/` or
`web/insights.py` (read `scorecard.csv`, never compute), `web/playbook.py` (replace
`candidates`/`top_setups` ranking with band assignment), `web/app.py` (`index`),
`templates/top.html`, `tests/test_dashboard.py`, `docs/breakout.md` (validation status, new
columns), README dashboard table. Golden compare will show the added columns (deliberate
schema change, re-baseline). Scores don't change, so no rubric bump.

**How we'd know it worked.**
- Every card on `/` names its evidence status. Zero cards come from a FAILed signal.
- This week the page would show GPN, AVPT, ADT and BHC as fresh weekly CONFIRMED events (if they
  pass the event rule), with GMAB marked as a continuation.
- In use, the morning question is answered in under a minute without opening Runs.
- `test_dashboard.py` asserts a FAILed signal never renders as a card.

---

### 2. A market-regime strip that reacts to the tape, replacing eight trade plans on `/market`

**Problem (PM view).** Today the tape broke and the product barely noticed.

- `/market` shows SPY, QQQ and SCHG all closing **below daily support and below SMA50** on
  heavy volume (daily RVOL20: SPY 1.97, QQQ 2.16, SCHG 1.59). QQQ is −6.9% over 1M. SCHD is
  +9.6% over 1M and "Dividend/value leading" (ratio 0.959 vs SMA 1.050, spread −8.9%).
- The bias rule (`playbook.market_bias`, `web/playbook.py:347-369`) still says **NEUTRAL** for
  SPY, QQQ and SCHG, because SHORT needs a *weekly* BEARISH trend or MTF. The headwind banner on
  `/` (`app.py:index`, `templates/top.html`) only fires on SHORT. So the home page shows **long
  cards for AAPL and MSFT** with no market caveat on a day QQQ broke support at 2x volume.
- The most important line of the morning, **SPY bias LONG → NEUTRAL**, is buried on
  `/changes` in a 4-row table.
- Each `/market` card renders a long *and* a short plan with stops and 2R/3R targets (8 plans).
  Today they read "Long: INVALIDATED … the setup is broken" and "Short: TRIGGERED" on SPY. That's
  a confusing way to say "the index closed below support". It also invites shorting index ETFs
  off an unvalidated mirror rule ("mirror of the long rule; not a scanner signal").
- Every Cobra row says `Market Gate True`. That gate is per-ticker (`market.py:84`, close vs
  EMA50 and RS floor), not a market gate, so it gives no regime information either.

**Proposed change.** Put one compact **regime strip** at the top of every page (`base.html`).
For each of SPY / QQQ / SCHG / SCHD it shows: the bias pill (same rule, unchanged), a
"changed today" marker, daily close vs SMA50 / SMA200, "below daily support" / "above
resistance" as plain facts, 1M return, and RVOL20 when it is ≥ 1.5. Add the growth-vs-dividend
label at the end. Let the existing banner on `/` also list these *facts* (not a new rule) when
SPY or QQQ is below daily support, so the long cards always sit under the day's tape. On
`/market`, remove the long/short plan blocks and their targets. Keep the chart with support and
resistance lines, the states table and the relative view.

**Why this beats the alternatives.** A faster bias rule (say, daily-only SHORT) would be a new
untested signal. Better to show the raw facts and let the PM judge. A separate "regime page"
is one more click. The strip costs nothing to read and appears where decisions get made.
Removing the plans also removes the product's most trading-system-like surface.

**Effort: S–M.** Files: `web/playbook.py` (expose the facts already computed in
`market_rows`; drop the plan blocks from the Market path), `web/insights.py`
(`bias_changes` feeds the "changed" marker; `etf_relative` feeds 1M), `web/app.py`
(a context processor for the strip), `templates/base.html`, `templates/top.html`,
`templates/market.html`, `static/css`, `tests/test_dashboard.py`, README.

**How we'd know it worked.**
- On a day like 2026-10-30, the first thing on `/` says SPY / QQQ closed below support on high
  volume and SPY's bias changed.
- `/market` drops from 8 plan tables to 0, and nobody misses them after two weeks of use.
- A test fixture with SPY below support but bias NEUTRAL renders the caution on `/`.

---

### 3. `/changes` as events, not state flips, and show who entered and left the universe

**Problem (PM view).** `/changes` should be my "what's new since yesterday" page. Today it's
mostly noise.

- The daily section lists **57 status changes**: 18 "moved toward a breakout", 6 failures and
  33 "other". That's 25% of the 226 tickers present in both runs. Across the last eight daily
  pairs, 11–30% of tickers change status every day.
- Most of it is churn from statuses being bar states (`docs/research_review.md` item 6). Of the
  15 daily CONFIRMED on 2026-10-22, **0** were still CONFIRMED the next day. Of the 9 on 10-29
  that were still in the universe on 10-30, 4 were. The "other" bucket is mostly FAILED → WATCH (KO, PEP, ADBE, ORLY…) and
  TRIGGERED → WATCH, which tells me nothing.
- The "Readiness / Change" columns mislead. Readiness falls when a breakout happens, because
  it scores the coil, so the "moved toward a breakout" list shows LLY −40, XOM −25 and CVX −25.
  Read naively, an upgrade looks like a deterioration.
- **Universe churn is invisible.** The universe is today's Yahoo most-actives plus a manifest.
  The nine daily runs hold **462 distinct tickers, only 178 present in all nine**.
  `insights.status_changes` explicitly drops entries and exits. So 4 of this week's 5 weekly
  CONFIRMED names (GPN, AVPT, ADT, BHC) never appear on `/changes`: they weren't in last
  week's file.

**Proposed change.** Rebuild `/changes` around **first-bar events**, using the same `Event`
column as #1 (same definition as `RESEARCH.md` §3). Group them as: *tested signal events*
(weekly CONFIRMED) first, then *Cobra new / dropped* (already there), then *other events
(failed in-sample)* as a collapsed count. Add **"Entered / left the universe"** as two plain
symbol lists. Remove the "Other status changes" table and the Readiness Change column. The
market-bias table moves to the regime strip (#2).

**Why this beats the alternatives.** Filtering the existing transitions by readiness or a
minimum-change threshold keeps the wrong unit (state flips) and adds a tuned knob. Freezing
the production universe to the research point-in-time list is the "real" fix, but it is L
effort and changes every output (see runner-ups). Showing entries and exits makes the churn
visible today at almost no cost.

**Effort: S** once #1 lands (M on its own). Files: `web/insights.py` (`status_changes` →
event grouping; add universe in/out), `templates/changes.html`, `tests/test_dashboard.py`
(or the insights tests), README.

**How we'd know it worked.**
- The daily section on a normal day lists a handful of events rather than ~50 transitions.
- Every weekly CONFIRMED first-bar event appears on `/changes` the morning after, even when
  the ticker is new to the universe.
- No row on the page has a negative "change" on an upgrade.

---

## Runner-ups considered

1. **Pre-register and backtest Coiled Cobra.** Cobra drives 5 of today's 6 home-page cards
   and has never been tested. This is high value, but it is research work under `RESEARCH.md`
   rules (a new pre-registration and a new holdout), not a dashboard change. #1 already handles
   the product side by labelling Cobra untested. Do it next.
2. **Use the point-in-time research universe in the production pipeline.** This fixes churn at
   the root (462 vs 178 tickers) and aligns live signals with the tested universe (top 200 by
   dollar volume, close ≥ $5, `RESEARCH.md` §11). But it is L effort, changes every output and
   the golden baseline, and #3 gets most of the morning-read benefit for S.
3. **Forward signal log** (realized 4-week return vs SPY for each weekly CONFIRMED event since
   the seal). It would keep the evidence line in #1 live after the holdout. It isn't top 3
   because it only becomes meaningful after the holdout. It must stay a *signal* record (event date,
   R4W, SPY R4W): no positions, no sizing, no P&L.
4. **Remove the Live Price column and the 5% "surge" row highlight** (`app.py:92`, `LIVE_SURGE_PCT`).
   Pre-open, "live" is a stale or after-hours print sitting beside close-based states, and the
   surge highlight nudges toward chasing (`docs/research_review.md` item 9). It's real noise
   but low impact. It's worth doing alongside #1.
5. **Surface data-quality errors on the scanner pages, not just on home cards.** Today BAX,
   SCHW and TGB have `ohlc_incoherent` daily bars (H below O). They'd only be flagged if they
   reached a top card. That's correct but rare, so it has low impact on the morning read.

Also small: `/runs` is a plain date list that I would ignore. It's fine as an archive, but it
doesn't need to be a top-level nav item once `/changes` links each day's files.

## Trading-system drift: flagged

These parts push the product toward a trading system and should be removed or stay out:

- **2R / 3R targets and the WAITING / TRIGGERED / EXTENDED / INVALIDATED plan states** on `/`
  (`playbook.action_plan`). They read like order management, and they don't match the tested
  holding rule. #1 drops the targets. The trigger and support lines can stay as chart context.
- **Short-side plans on `/market`** ("Short: break below support" with a stop and targets) are a
  mirrored rule the scanner never signals. #2 removes them.
- **Live-price surge highlighting** encourages intraday chasing of end-of-day signals (runner-up 4).
- **Don't add:** position sizing, portfolio or P&L tracking, a paper-trade ledger, broker
  links, intraday alerts, or tuning the Rank Score or Readiness weights on backtest results.
  The forward signal log (runner-up 3) is the line not to cross. It records what the signal
  did, never what "we" did.
