# Finance Vibe 📈  
## Project Intent & Engineering Philosophy

The **Finance Vibe** project is engineered as a **Modular Data Pipeline**.  
The intent is to transition from isolated "scripts" to a **Systematic Analysis Engine**.

By decoupling **data ingestion**, **mathematical processing**, and **reporting layers**, we ensure the system is:

- Idempotent  
- Scalable  
- Reproducible  

This project prioritizes:

- **Environmental Parity**
- **Mathematical Robustness**

The system operates within a stable **5-year weekly regime** to eliminate high-frequency noise and ensure reproducible results across any compute node.

---

## Key Architectural Decisions

### Hermetic Environment (Dev Containers)

We utilize **Docker-based Dev Containers** to enforce *Environment-as-Code*.

This approach:

- Locks the Python 3.12 interpreter
- Locks system dependencies
- Stores environment configuration in version control

The environment becomes a **disposable, reproducible artifact**.

---

### Stateless Logic & State Isolation

The repository architecture enforces a strict boundary between:

- `/src/` — Application logic  
- `/data/` — System state  

Within `/data/`:

- `/raw/` — Immutable source data  
- `/logs/` — Analytical output  

This isolation ensures:

- The pipeline can be audited independently  
- Data can be wiped without risking the codebase  
- Code and state remain cleanly separated  

---

### Shadow Math (Architectural Redundancy)

The system implements a **Dual-Engine Pattern** for logic verification.

A **Shadow Engine** (`analysis_engine_local.py`) runs alongside the primary engine to enable:

- Differential testing  
- Safe mathematical experimentation  
- Validation of scoring changes  

This is particularly important for validating our **Manual Mean Absolute Deviation (MAD)** implementation before promoting changes to the main pipeline.

---

### Decoupled Orchestration

`run_vibe.py` functions as a **Stateless Orchestrator**.

It manages the lifecycle:

```
Discovery → Ingestion → Analysis → Comparison
```

This design allows:

- Execution via external schedulers (Cron / GitHub Actions)
- Clear exit codes for failure handling
- Clean separation between orchestration and computation

## Strategic Trend & Momentum Pipeline

A professional-grade Python framework for fetching and analyzing stock data from Yahoo Finance.

This project uses a **Composite Vibe Score** to identify high-conviction trends using weekly data science workflows.

---

# 🏗 Project Structure

```
src/finance_vibe/
```

## Core Python Logic

- **config.py** — Central settings (5y Weekly data, Static ETFs, Paths)  
- **ticker_provider.py** — Merges top active stocks with benchmark ETFs (SPY, QQQ, IWM)  
- **data_ingestor.py** — Pulls 5 years of weekly historical data  
- **analysis_engine.py** — The Math Engine -uses pandas and Pandas ta to calculate (SMA, MACD, RSI, and Robust CCI)  
- **analysis_engine_local.py** - This uses local python code to generate (SMA, MACD, RSI, and Robust CCI)
- **run_vibe.py** — Master script to run the full pipeline  

```
data/
```

- **raw/** — Original CSV files (Ignored by Git)  
- **logs/** — Archive for dated Vibe Reports (CSV format)  

```
notebooks/
```

- Jupyter notebooks for data exploration  

---

# 🚀 How to Use

## 1️⃣ Open in Dev Container

Ensure Docker is running.

Reopen the project in the container to auto-install:
- Python 3.12  
- Pandas  
- Required extensions (Rainbow CSV, Excel Viewer)

---

## 2️⃣ Run the Pipeline

Execute the master command to ingest data and run analysis:

```bash
python src/finance_vibe/run_vibe.py
```

---

## 3️⃣ Reset Data

To clear out old raw files and force a fresh fetch:

```bash
rm data/raw/*.csv
```

---

# 📈 Technical Logic: The Composite Vibe Score

The core **Actionable Logic** is driven by a **Weighted Scoring System (-10 to +10)** to identify trend strength and momentum confluence.

---

## 📊 Scoring Matrix

| Indicator  | Logic | Weight |
|------------|--------|--------|
| **Trend** | Price > SMA(20) > SMA(50) | ±4.0 Points |
| **Momentum** | MACD Histogram & RSI > their 20-EMAs | ±3.0 Points |
| **Volatility** | Robust CCI > 0 and > its 20-EMA | ±3.0 Points |

---

## 🎯 Action Tiers

- 🔥 **GO ALL IN** (Score 8 to 10) — Maximum bullish confluence  
- ✅ **ACCUMULATE** (Score 4 to 7) — Positive trend and momentum  
- ⏳ **WAIT / CASH** (Score -3 to 3) — Neutral zone; no clear edge  
- ⚠️ **DISTRIBUTE** (Score -4 to -7) — Bearish divergence or weakening trend  
- 🚫 **AVOID** (Score -8 to -10) — High-conviction bearish trend  

---

# 🛠 Project Standards

## Robust CCI Calculation

Uses a manual **Mean Absolute Deviation (MAD)** formula to prevent extreme values (-4000+) during low-volatility periods.

## Data Transparency

Reports include the raw indicator values (SMA20, CCI, RSI, etc.) alongside the final score for manual verification.

## Archive Logic

Every run generates a timestamped CSV in:

```
data/logs/
```

for historical tracking.

---

# 📝 Future Roadmap

- [ ] **Automation** — Set cron to execute `run_vibe.py` every Saturday at 09:00  
- [ ] **Alerting** — Integrate Discord/Telegram Webhooks for high-score signals  
- [ ] **Visualization** — Add Matplotlib logic to generate "Vibe Charts" (Price vs. Score)