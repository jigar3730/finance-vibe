Here’s the **README** fully formatted as a proper Markdown file (`README.md`) that you can drop in your project:

---

```markdown
# Swing Scanner - README

## Overview

`swing_scanner.py` is a **rule-based swing trading scanner** that analyzes stock price data and identifies potential **long and short swing trade setups** based on trend, momentum, volatility, and pullback conditions.  

It is designed for weekly (or daily) price data in CSV format and provides a **scored output** along with actionable signals.

---

## Features

- Detects **SETUP_LONG** and **SETUP_SHORT** trades  
- Calculates **SMA20/SMA50, EMA20/EMA50, RSI, MACD, CCI, ATR**  
- Assigns **composite scores** to summarize trend + momentum + volatility  
- Generates **Notes** to explain why each ticker triggered  
- Archives results in `/data/logs`  
- Full logging for debugging and audit purposes

---

## Folder Structure

```

finance-vibe/
├── data/
│   ├── raw/                # Contains raw CSV files for tickers
│   ├── logs/               # Scanner output and logs
│   └── active_tickers.csv  # Optional: restrict scanner to specific tickers
├── src/finance_vibe/
│   └── swing_scanner.py

````

---

## Requirements

- Python 3.10+
- Packages:

```bash
pip install pandas pandas_ta
````

---

## CSV Format

Each raw ticker CSV should include:

| Date | Open | High | Low | Close | Volume |
| ---- | ---- | ---- | --- | ----- | ------ |

* `Date` must be parseable by Pandas (`YYYY-MM-DD`)
* Recommended: **weekly bars** (`1wk`) or daily bars

---

## Usage

1. **Place raw data CSVs** in `/data/raw/`
2. (Optional) Add `active_tickers.csv` with column `Ticker` to restrict scanning
3. Run the scanner:

```bash
python src/finance_vibe/swing_scanner.py
```

---

## Output

* Console output (Markdown table) shows:

| Symbol | Setup Type | Close | EMA20 | EMA50 | RSI | ATR | Notes |
| ------ | ---------- | ----- | ----- | ----- | --- | --- | ----- |

* CSV archive saved automatically to:

```
data/logs/swing_setups_YYYY-MM-DD.csv
```

* **Setup Type**:

  * `SETUP_LONG` → Bullish swing setup
  * `SETUP_SHORT` → Bearish swing setup
* **Notes** → Reason setup triggered (e.g., “Pullback into 20EMA”)

---

## Scanner Logic Overview

1. **Trend Check**: Price relative to EMA20/EMA50
2. **Momentum Check**: RSI and MACD alignment
3. **Volatility Check**: ATR / CCI threshold
4. **Pullback Filter**: Price near short-term moving average
5. **Composite Scoring**:

   * Trend + Momentum + Volatility = Total Score
   * Score ≥ threshold → flagged as trade setup

---

## Logging

* Logs saved to `/data/logs/swing_scanner.log`
* Info includes:

  * Number of raw files found
  * Number of tickers scanned
  * Reason for rejections (insufficient data, inactive tickers, failing filters)

---

## Example

```
| Symbol | Setup Type | Close  | EMA20  | EMA50  | RSI  | ATR  | Notes               |
|--------|------------|--------|--------|--------|------|------|--------------------|
| SPY    | SETUP_LONG | 687.35 | 687.13 | 685.25 | 50.1 | 8.10 | Pullback into 20EMA|
| QQQ    | SETUP_SHORT| 607.87 | 609.81 | 612.99 | 47.4 | 9.95 | Pullback into 20EMA|
```

---

## Tips

* Verify **Notes** and chart visually before entering a trade
* Use **ATR** to set stop-loss levels
* Archive CSVs for historical performance tracking
* Adjust EMA, RSI, or pullback thresholds in code to fine-tune your strategy

---

