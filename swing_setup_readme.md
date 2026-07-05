# Swing Scanner (Tactical Layer)

Companion to the macro Vibe Score in `analysis_engine.py`. This module flags
**SETUP_LONG** and **SETUP_SHORT** pullbacks into EMA20 when trend and momentum align.

## Usage

```bash
python src/finance_vibe/swing_scanner.py weekly
python src/finance_vibe/swing_scanner.py daily
```

## Output

`data/logs/{mode}/swing_setups_<YYYY-MM-DD>.csv`

| Symbol | Setup Type | Close | EMA20 | EMA50 | RSI | ATR | Notes |

## Long setup (`SETUP_LONG`)

- `EMA20 > EMA50` and `EMA50` rising vs prior bar
- `EMA20 ≤ Close ≤ EMA20 × 1.02`
- RSI 45–60 (weekly) or 40–60 (daily)
- MACD histogram up two bars in a row and below `2 × std(MACD_Hist, 20)`

## Short setup (`SETUP_SHORT`)

- `EMA20 < EMA50` and `EMA50` falling vs prior bar
- `EMA20 × 0.98 ≤ Close ≤ EMA20`
- RSI 50–65
- MACD histogram down two bars in a row and above `−2 × std(MACD_Hist, 20)`

## Notes

- Only symbols listed in `data/active_tickers.csv` are scanned
- Requires ≥ 60 bars per raw CSV
- Uses `pandas_ta` for EMA, MACD, RSI, and ATR (distinct from SMA-based macro engine)
