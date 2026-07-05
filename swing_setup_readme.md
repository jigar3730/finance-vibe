# Swing Scanner

## Overview

`swing_scanner.py` is the **tactical layer** of Finance Vibe. It detects `SETUP_LONG` and `SETUP_SHORT` opportunities from EMA trend, RSI band, MACD histogram momentum, and ATR-based volatility context.

Macro regime context lives in `analysis_engine.py` (`vibe_report_<date>.csv`).

## Usage

```bash
python src/finance_vibe/swing_scanner.py weekly
python src/finance_vibe/swing_scanner.py daily
```

## Output

`data/logs/{mode}/swing_setups_<YYYY-MM-DD>.csv`

| Symbol | Setup Type | Close | EMA20 | EMA50 | RSI | ATR | Notes |
| ------ | ---------- | ----- | ----- | ----- | --- | --- | ----- |

## Logic Summary

**Long:** EMA20 > EMA50, rising EMA50, price within 2% of EMA20, RSI 45–60 (40–60 daily), MACD histogram rising without overextension.

**Short:** EMA20 < EMA50, falling EMA50, price within 2% of EMA20, RSI 50–65, MACD histogram falling without overextension.
