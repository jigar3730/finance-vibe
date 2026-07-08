# Coiled Cobra Scanner
## Scoring Rubric & Technical Design

Version: 1.0

---

# Overview

The **Coiled Cobra Scanner** is a systematic technical analysis model designed to identify **high-probability macro reversal opportunities**.

Unlike momentum scanners that search for stocks already breaking out, this scanner attempts to identify securities that are:

- Deeply discounted
- Near major Fibonacci support
- Sitting on institutional volume shelves
- Showing momentum exhaustion
- Beginning a bullish reversal

The scanner produces a maximum score of **100 points**.

Only high-confluence setups are returned.

---

# Overall Scoring Matrix

| Category | Weight | Purpose |
|----------|---------:|---------|
| Fibonacci Confluence | 30 | Long-term structural support |
| Volume Profile Shelf | 25 | Institutional accumulation |
| Deep Markdown | 15 | Oversold discount |
| MACD Compression | 20 | Selling exhaustion |
| Bullish MACD Crossover | 10 | Entry trigger |

Maximum Score = **100**

---

# Grade Classification

| Score | Grade | Meaning |
|--------:|-------|---------|
| 85-100 | A | Institutional Setup |
| 70-84 | B | Valid Reversal |
| Below 70 | Reject | Insufficient confluence |

---

# Indicator Details

## 1. Fibonacci Confluence (30 Points)

### Objective

Determine whether price is sitting near a major long-term Fibonacci retracement.

### Calculation

Rolling Lookback

- Weekly Mode = 52 weeks
- Daily Mode = 252 trading days

High

```
Highest High
```

Low

```
Lowest Low
```

Retracement Levels

```
61.8%

78.6%
```

### Pass Criteria

Current price must be within **2.5%** of either level.

```
abs(price - Fib618) / Fib618 <= 2.5%

OR

abs(price - Fib786) / Fib786 <= 2.5%
```

### Points Awarded

```
30
```

---

## 2. Volume Profile Shelf (25 Points)

### Objective

Identify institutional accumulation zones.

### Method

The scanner:

1. Looks back over the analysis window
2. Divides the entire price range into 10 equal bins
3. Aggregates trading volume inside each price bin
4. Finds the three highest-volume nodes (HVNs)

If today's closing price falls inside one of those nodes:

```
PASS
```

### Points Awarded

```
25
```

### Purpose

Institutions often accumulate positions over long periods.

High-volume price shelves frequently become major support.

---

## 3. Deep Markdown Extension (15 Points)

### Objective

Measure how deeply discounted the stock is relative to trend.

### Daily Mode

```
Price < EMA20 × 0.93
```

### Weekly Mode

```
Price < EMA20 × 0.85
```

AND

```
Price < EMA50
```

### Points Awarded

```
15
```

### Purpose

This prevents chasing stocks already extended upward.

---

## 4. MACD Compression (20 Points)

### Objective

Identify momentum exhaustion.

### Conditions

```
MACD < 0
```

AND

```
abs(MACD - Signal) / EMA20 <= 0.005
```

### Interpretation

Momentum is no longer accelerating downward.

Selling pressure has compressed.

Many reversals begin during this condition.

### Points Awarded

```
20
```

---

## 5. Bullish MACD Crossover (10 Points)

### Objective

Provide an entry trigger.

### Conditions

Previous Bar

```
MACD <= Signal
```

Current Bar

```
MACD > Signal
```

### Points Awarded

```
10
```

This factor intentionally carries the lowest weight.

A MACD crossover alone is not considered a high-quality reversal.

---

# Example Scores

## Example A

| Indicator | Score |
|------------|------:|
| Fibonacci | 30 |
| Volume Shelf | 25 |
| Markdown | 15 |
| MACD Compression | 20 |
| MACD Cross | 10 |

Total

```
100
```

Grade

```
A
```

---

## Example B

| Indicator | Score |
|------------|------:|
| Fibonacci | 30 |
| Volume Shelf | 25 |
| Markdown | 15 |
| MACD Compression | 20 |
| MACD Cross | 0 |

Total

```
90
```

Grade

```
A
```

---

## Example C

| Indicator | Score |
|------------|------:|
| Fibonacci | 30 |
| Volume Shelf | 25 |
| Markdown | 15 |
| MACD Compression | 0 |
| MACD Cross | 0 |

Total

```
70
```

Grade

```
B
```

---

# Scanner Philosophy

The scanner favors:

- Deep value
- Institutional accumulation
- Structural support
- Momentum exhaustion
- Confirmation rather than prediction

It intentionally avoids chasing stocks making new highs.

---

# Strengths

- Multi-factor scoring
- Objective rules
- No discretionary inputs
- Easy to backtest
- Low computational cost
- Suitable for large watchlists
- Supports daily and weekly timeframes

---

# Known Limitations

Current Version 1 does **not** consider:

- Relative Strength vs SPY
- Sector strength
- ATR normalization
- ADX trend strength
- RSI recovery
- Volume expansion
- Multi-timeframe confirmation
- 200 EMA trend direction
- Market regime filters

As a result, some "falling knife" setups may still pass the scanner.

---

# Recommended Version 2 Scoring Matrix

| Factor | Weight |
|---------|--------:|
| Fibonacci Confluence | 20 |
| Volume Profile Shelf | 15 |
| Deep Markdown | 10 |
| MACD Compression | 10 |
| Bullish MACD Cross | 5 |
| Relative Strength vs SPY | 10 |
| Weekly RSI Recovery | 10 |
| ADX Trend Strength | 5 |
| Sector Relative Strength | 5 |
| 200 EMA Trend Filter | 5 |
| Multi-Timeframe Alignment | 10 |
| Volume Expansion | 5 |

Total

```
100
```

---

# Recommended Grade Scale

| Score | Grade | Description |
|--------:|-------|-------------|
| 90-100 | S | Exceptional Institutional Setup |
| 80-89 | A | High Probability Reversal |
| 70-79 | B | Valid Swing Candidate |
| 60-69 | C | Watchlist |
| Below 60 | Reject |

---

# Future Enhancements

Recommended additions include:

- Relative Strength Ranking
- Sector Rotation Analysis
- Anchored VWAP
- ATR-based Position Sizing
- Dynamic Volume Profile
- Market Regime Detection
- Risk/Reward Scoring
- Volatility Compression
- Breadth Confirmation
- Institutional Accumulation Metrics

---

# Summary

The current implementation is a **structural reversal scanner**, not a momentum scanner.

It identifies stocks exhibiting:

- Long-term Fibonacci support
- Institutional volume accumulation
- Significant markdown from trend
- Momentum exhaustion
- Early bullish confirmation

The design emphasizes quality over quantity, returning only setups with a minimum score of **70** while reserving the highest grades for those exhibiting multiple layers of technical confluence.