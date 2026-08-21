# 🧮 Trade Plan Architecture Documentation

This document outlines the systematic, mathematical pipeline used by the `quant-platform-scanner` ecosystem to transform raw market indicators into actionable, volatility-adjusted trade execution architectures.

---

## 🧭 The Core Workflow Pipeline

The execution architecture operates as a strict multi-step data processing lifecycle:

```mermaid
graph TD
    A([Market Universe])
    A -->|coiled_cobra.py| B[Passing coils]
    B -->|trade_planner.py| C[Close / Coil_Low / 2R-3R]
    C -->|trade_planner.py| D[Ranked plan]

    style A fill:#f9f,stroke:#333,stroke-width:1px
    style B fill:#bbf,stroke:#333,stroke-width:1px
    style C fill:#bfb,stroke:#333,stroke-width:1px
    style D fill:#fbb,stroke:#333,stroke-width:1px
```

## Coiled Cobra expansion (live)

Spec: **`Coiled Cobra Rubric .MD`**. Quality-swing EMA pullback math is offline-only (`config.compute_swing_levels`).

### 1. Entry
$$\text{Stock Entry} = \text{Close}$$

### 2. Stop
Protect the coil floor, then take the tightest of three constraints, remaining below entry:
$$\text{floor} = \text{Coil\_Low (else Swing Low)}$$
$$\text{Stock Stop} = \min(\max(\text{floor} - 0.25\times\text{ATR},\ \text{entry} - 1.5\times\text{ATR},\ \text{entry} - 5\%\times\text{Close}),\ \text{entry} - 0.25\times\text{ATR})$$

### 3. Targets
$$\text{Target 1} = \text{Entry} + 2R,\quad \text{Target 2} = \text{Entry} + 3R$$
where \(R = \text{Entry} - \text{Stop}\).

### 4. Capital Efficiency Metrics
Risk per share and Risk-to-Reward ($R:R$) ratios are computed linearly to evaluate trade viability before capital allocation:
$$\text{Risk Per Share} = \text{Stock Entry} - \text{Stock Stop}$$
$$\text{Risk-to-Reward (R:R)} = \frac{\text{Target Level} - \text{Stock Entry}}{\text{Risk Per Share}}$$

---

## 🔍 Case Study Workbook: Legacy Swing Mode Reference

*Historical reference for offline swing-pullback scanner (`swing_scanner.py`). For live Coiled Cobra expansion, refer to Section "Coiled Cobra expansion (live)" above.*

### Raw Scanner Inputs
* **Close:** `342.89`
* **EMA50:** `332.04`
* **ATR:** `7.46`

### Calculation Output Log
1.  **Stock Entry:** $342.89 - (0.25 \times 7.46) = \mathbf{341.02}$
2.  **Stock Stop:** $332.04 - (0.50 \times 7.46) = \mathbf{328.31}$
3.  **Target 1:** $341.02 + 7.46 = \mathbf{348.48}$
4.  **Target 2:** $341.02 + (2 \times 7.46) = \mathbf{355.94}$
5.  **Risk Per Share:** $341.02 - 328.31 = \mathbf{12.71}$
6.  **Target 1 R:R Ratio:** $\frac{348.48 - 341.02}{12.71} = \mathbf{0.59}$
7.  **Target 2 R:R Ratio:** $\frac{355.94 - 341.02}{12.71} = \mathbf{1.17}$

---

## Options overlay (not live)

Live `trade_plan_*.csv` files are **equity expansion** only (Close / Coil_Low / 2R / 3R). LEAPS expiry and delta bands are not written. Historical swing options notes remain below for the offline lab.

| Parameter | Assigned Value / Range | Operational Strategy Logic |
| :--- | :--- | :--- |
| **Options Type** | `CALL` | Assigned strictly for `SETUP_LONG` structures. |
| **Suggested Delta** | `0.65 – 0.80` | Targets deep In-The-Money (ITM) positions to mimic underlying stock replacement closely while mitigating high theta decay. |
| **Expiration Window** | `30 to 90 Days Out` | Matches market parameters dynamically (`Min: Current Month + 1` to `Max: Current Month + 3`). |

---
*Document Version: 1.0.0 | System Date Context: 2026-07-07*

---

## 🛡️ Operational Safeguards & Exception Handling

### 1. Data Validation Gating
Before any asset enters the calculation engine, it must pass a structural completeness gate inside `swing_scanner.py`. 
* **`IGNORE` Classification:** The asset is healthy but does not meet the explicit entry profile (e.g., RSI is too high, or price is not interacting with the EMA20).
* **`insufficient_data` Exception:** Triggered if a ticker lacks sufficient trading history to cleanly establish the backward-looking `EMA50` baseline. Assets failing this check are instantly aborted to prevent mathematical skew in volatile new listings or low-liquidity pairs.

### 2. Asset Suffix Standardization
To maintain compatibility between internal log systems and downstream web services (such as Yahoo Finance asset tracking templates), the symbol tracking architecture mandates clean, un-suffixed global symbols (e.g., `SPY`, `HLT`). Global exchanges or market flags are handled at the UI rendering layer, keeping core data files pristine.

### 3. Dynamic Stop Invalidations
As codified in the `Risk Notes` column output, the `Stock Stop` is directly dependent on structural moving average behavior. If a systemic market shift pushes the moving average line significantly away from the initial calculation print before order execution, the order parameters are considered structurally invalidated and must be manually or programmatically re-calculated.