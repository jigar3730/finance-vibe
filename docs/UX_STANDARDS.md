# UX standards

Rules for every dashboard template and stylesheet (`src/finance_vibe/web/templates/`,
`static/css/`). Personal-use tool: the goal is reading the market in seconds.
`tests/test_ux_standards.py` enforces the word and link rules.

## Facts, not verdicts

- Show numbers and plain facts: "VIX 19.42, above its 20-day average".
- Never judgment words: Bullish, Bearish, Strong, Weak, Elevated, Risk-On, Risk-Off,
  Fading, Accelerating, Aligned (any case, any page text or tooltip).
- **Exception: the Setups area** (`setups.html`, `breakout_view.html`,
  `cobra_view.html`) may show raw scanner status and state names as data
  (BREAKOUT_CONFIRMED, FAILED_BREAKOUT, Trend BULLISH, MTF ALIGNED). Everything about
  individual stocks lives there, including "Since last run" changes; `/changes` is
  market facts only.
- Comparisons say what was compared and over what window: "Growth outperforming
  (last 63 days)", "Dividend/value outperforming (last 63 days)", "Mixed".

## Colors

- Green and red only for up / down or above / below. Nothing else gets a color
  that reads as a call.
- Orange only for ⚠ warnings (stale or missing data, a close below support on heavy
  volume, bar-quality errors). Nothing else is orange.
- No glowing, pulsing or highlighted cards or rows.
- Numbers white (`--finviz-text-strong`); labels grey (`--finviz-text-muted`).

## No clutter

- No disclaimers or "not a signal" notes.
- One "as of" date per page, next to the title, not per row. Only an item that is
  older than that date gets a short ⚠ note (e.g. "⚠ HYG stale").
- Explanations only as short `?` tooltips (the `title` attribute), and only where a
  term needs explaining (HYG/IEF, RS vs SPY).
- How far to trust a section (backtest result, untested, failed) is one `?` tooltip
  on that section's heading, not a note or a pill.

## Layout

- Merge tables that share rows (one row per ticker, not two tables).
- Headline numbers go in metric tiles with a large number and a small label.
- Several items with a yes/no fact: small color tiles with the name on hover, not a
  list of text.
- No charts unless they clearly add something a number or a bar can't.

## Text

- Large enough to read easily: body text at least 15px, nothing under 12px.
- High contrast: white numbers on the dark background; grey only for labels.

## No external libraries

- Existing CSS (`finviz-theme.css`) and local files only: no Tailwind, Font Awesome,
  CDNs or internet fonts. Vendored files live in `static/vendor/`.
- Outbound `<a href>` links (Finviz ticker pages) are fine.

## Number formats

| Kind | Format | Example |
| ---- | ------ | ------- |
| % change | sign, 2 decimals | +0.48%, −1.77% |
| Price, level, average | 2 decimals | 766.77 |
| Change in points (VIX) | sign, 2 decimals | +2.22 |
| Ratio | 4 decimals | 0.8438 |
| Count | "N of M" | 8 of 11 |
