# Work Log – Swing Trade Planner Development

## Date: 2026-02-26

### Objective

Build a **systematic trade planner** to work with swing scanner outputs and generate **entry, exit, and options strategies** for each trade signal.

---

## Tasks Completed

### 1. Reviewed Swing Scanner Outputs

- Analyzed `swing_scanner.py` outputs for both **before and after MACD logic updates**.
- Observed:
  - Old MACD momentum trigger was too weak (`MACD_Hist.iloc[-1] > h.min()`).
  - New logic improved setup filtering using rising MACD histogram with overbought check.
- Ensured filtered signals now properly reflect **valid long and short setups**.

---

### 2. Addressed Team Feedback for Scanner

- Added **pullback “buy zone”**: price must be near EMA20 but not too far.
- Ensured **EMA50 slope check** to avoid dead-cat bounces.
- Handled **NaN values** from pandas_ta indicators with `df.dropna()`.
- Adjusted **RSI ranges** for short setups to be asymmetric (45–65 instead of 40–60).

---

### 3. Built `trade_planner.py` Skeleton

- Inputs:
  - Scanner CSV with columns: `Symbol, Setup Type, Close, EMA20, EMA50, RSI, ATR`.
- Outputs:
  - Stock entry, stop, target 1 & 2.
  - LEAPS options type, expiry window, suggested delta.
  - Risk notes.
- Logic:
  - ATR-based sizing for stock targets.
  - EMA20/EMA50 based entries and stops.
  - Separate handling for **long (CALL)** vs **short (PUT)** trades.

---

### 4. Parameterized File Handling

- Initially had **hardcoded CSV paths**.
- Updated to:
  - Automatically detect the **latest scanner CSV** in `./data/logs` using filename pattern `swing_setups_YYYY-MM-DD.csv`.
  - Auto-generate output file as `trade_plan_YYYY-MM-DD.csv`.
  - One-command execution: `python trade_planner.py`.

---

### 5. Key Advantages of the New Trade Planner

- Clean separation from scanner logic.
- Deterministic entries, stops, targets, and LEAPS options parameters.
- Ready for integration with **position sizing / risk management** layer.
- Eliminates manual filename management.
- Fully reproducible — always uses **most recent scanner output**.

---

### Next Steps

1. Add **position sizing / risk allocation rules** to each trade:
   - Determine how many shares or contracts to buy based on % of portfolio risk.
2. Integrate **LEAPS strike selection** using delta/ATM guidance.
3. Optionally, add **portfolio-level constraints** (max open trades, sector exposure, etc.).

---

**Summary:**  
By the end of today, we built a **deterministic trade planner** that automatically reads scanner results, calculates stock entries/stops/targets, and prepares LEAPS options suggestions, while fully parameterizing input/output and preparing the system for portfolio-level risk management.

### Sample output

Symbol,Setup Type,Stock Entry,Stock Stop,Target 1,Target 2,LEAPS Type,LEAPS Expiry Min,LEAPS Expiry Max,Suggested Delta,Risk Notes
INTC,SETUP_LONG,46.28,43.09,48.92,51.56,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
IWM,SETUP_LONG,263.32,256.41,268.36,273.4,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
PSTG,SETUP_LONG,73.53,65.43,81.23,88.93,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
SPY,SETUP_LONG,691.16,681.57,699.13,707.1,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
XLC,SETUP_LONG,116.17,115.41,117.68,119.19,CALL,Feb-2027,Feb-2028,0.65 – 0.8,Stop based on EMA50; adjust if invalidated
