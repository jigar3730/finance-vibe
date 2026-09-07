
**`breakout_scanner.py`**  structure.

### What the new scanner adds

* **SMA 20 / 50 / 200**
* **RSI 14**
* **MACD 15/30/9** — matching the direction we've already been tuning
* **Bollinger Bands 20/2**
* **BB Width percentile**
* **ATR 14 + ATR percentile**
* **20-bar range contraction**
* **RVOL 20**
* **Volume dry-up detection**
* **20-bar resistance/support**
* **Distance to resistance in ATR**
* **Price extension in ATR**
* **Monthly → Weekly → Daily alignment**
* **Momentum acceleration**
* **Pre-breakout vs confirmed-breakout classification**
* **Fakeout detection**
* **100-point Breakout Readiness Score**
* Individual factor scores so we can tune the model later
* CSV output suitable for your existing pipeline

Most importantly, it is designed to find the **pre-breakout candidate**, rather than only finding stocks after they have already broken out.

### New file

/src/finance_vibe/breakout_scanner.py

### One important design decision

I deliberately **didn't make the score the primary source of truth**.

The CSV preserves the underlying features:

```text
BB Width Pctl
KC Width Pctl
ATR Pctl
Range20 Pctl
RVOL20
RSI Slope
MACD Hist Slope
Distance Resistance ATR
Extension ATR
Daily Trend Bull
Weekly Trend Bull
Monthly Trend Bull
MTF Alignment
Compression
Breakout Triggered
Breakout Confirmation
Failed Breakout
...
```

That gives us the ability to backtest and determine **which features actually have predictive value** before we start aggressively tuning weights.

### Current architecture

```text
OHLCV
  │
  ├── Daily
  ├── Weekly
  └── Monthly
        │
        ▼
   Feature Engine
        │
        ├── Trend
        ├── Momentum
        ├── Volatility
        ├── Volume
        └── Structure
        │
        ▼
  Breakout Readiness
        │
        ├── PRE_BREAKOUT
        ├── WATCH
        ├── DEVELOPING
        ├── BREAKOUT_CONFIRMED
        └── FAILED_BREAKOUT
        │
        ▼
 breakout_setups_YYYY-MM-DD.csv
```

One caveat: with your current OHLCV data, the script uses **Monthly/Weekly/Daily** MTFA. It does **not pretend that daily data contains 4H/1H information**. We can add that later if your raw dataset contains intraday bars.

The next step I recommend is **not immediately changing your production scanner**. Run this alongside the existing `swing_scanner.py`, inspect the candidates it produces, and then we'll compare **Breakout Readiness vs your existing Swing Score** on the same ticker universe. That will tell us whether this is actually discovering earlier opportunities rather than simply generating another flavor of the same signals.

# Core Breakout Structure

| Pillar         | Primary Tool                                | What it tells us                             |
| -------------- | ------------------------------------------- | -------------------------------------------- |
| **Trend**      | **SMA 20 / 50 / 200**                       | Direction and structural alignment           |
| **Volatility** | **Bollinger Bands + Keltner Channel + ATR** | Compression → squeeze → volatility expansion |
| **Momentum**   | **RSI + MACD Histogram**                    | Whether momentum is building                 |
| **Volume**     | **Relative Volume + OBV**                   | Whether participation is arriving            |
| **Structure**  | **Price levels / swing highs**              | Where the actual breakout occurs             |

# overall script structure 
breakout_scanner.py

        │
        ▼
Load OHLCV
        │
        ▼
Clean / validate data
        │
        ▼
Create timeframes
        │
        ├── Monthly
        ├── Weekly
        └── Daily
        │
        ▼
Calculate indicators
        │
        ├── SMA
        ├── RSI
        ├── MACD
        ├── Bollinger
        ├── ATR
        └── Volume/RVOL
        │
        ▼
Detect market structure
        │
        ├── Support
        ├── Resistance
        ├── Swing highs
        ├── Swing lows
        └── Consolidation
        │
        ▼
Detect compression
        │
        ├── BB contraction
        ├── ATR contraction
        ├── Range contraction
        └── Volume dry-up
        │
        ▼
Momentum analysis
        │
        ├── RSI
        ├── RSI slope
        ├── MACD
        └── MACD histogram slope
        │
        ▼
Breakout proximity
        │
        ├── Distance to resistance
        └── Distance / ATR
        │
        ▼
MTF alignment
        │
        ├── Monthly
        ├── Weekly
        └── Daily
        │
        ▼
BREAKOUT READINESS SCORE
        │
        ▼
Fakeout / risk penalties
        │
        ▼
Final classification

# Script intent 
Make it identify states. not scores 
TREND
    BULLISH

VOLATILITY
    COMPRESSING

VOLUME
    DRYING_UP

MOMENTUM
    ACCELERATING

STRUCTURE
    UNDER_RESISTANCE

BREAKOUT_DISTANCE
    0.7 ATR

MTF
    ALIGNED

STATUS
    PRE_BREAKOUT