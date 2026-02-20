# Finance Vibe 📈

A robust Python data science project for fetching and analyzing stock data from Yahoo Finance.  Data Science & Mean Reversion Pipeline

A professional-grade Python framework for algorithmic market analysis, focusing on **Mean Reversion**, **Volatility Confluence**, and **Standardized Data Science workflows**.

## 🏗 Project Structure
- `src/finance_vibe/`: Core Python logic (Ingestion and Processing).
- `data/raw/`: Original CSV files from Yahoo Finance (Ignored by Git).
- `notebooks/`: Jupyter notebooks for data exploration.

## 🚀 How to Use
1. **Open in Dev Container**: Ensure Docker is running and "Reopen in Container" in VS Code.
2. **Fetch Data**: Run the following command in the terminal:
   \`\`\`bash
   python3 src/finance_vibe/ingest.py
   \`\`\`

## 🛠 Tech Stack
- Python 3.12
- yfinance (Data Provider)
- Pandas (Data Manipulation)
- Docker / Dev Containers (Environment)

## 🛠 Project Health Check

| Component | Status | Why it's "Robust" |
| :--- | :--- | :--- |
| **Environment** | Docker Container | Your PC stays clean; the project is portable. |
| **Code** | `src/` Layout | Logical separation of ingestion and analysis. |
| **Data** | `.gitignore` Protected | You aren't bloating your Git history with CSVs. |
| **Analysis** | Jupyter Notebooks | Reproducible research and visualizations. |
| **History** | Git Initialized | You have a "Save Point" for every major change. |



---

## 🛠 Project Standards & Architecture

### 1. Modular Directory Structure
To maintain a clean separation between logic and data, this project follows a strict modular hierarchy:
* **`/data`**: Local storage for raw market data (excluded from Version Control).
* **`/src`**: Source code containing the engine of the project.
    * `indicators.py`: Central library for all mathematical formulas (RSI, MA, BB).
    * `mean_reversion.py`: The primary scanning engine using weighted scoring.
* **`.gitignore`**: Prevents data bloat and environment leaks in GitHub.

### 2. Data Science Principles
* **Normalization (The Apple-to-Apple Rule):** We use **Rolling Standard Deviations** (Z-Scores) to compare volatile assets (Crypto) against stable assets (ETFs) on the same mathematical scale.
* **Idempotent Ingestion:** Data scripts are designed to handle missing values via `.ffill()` and can be re-run without duplicating entries.
* **Provider Agnostic Math:** Functions are designed to process **Pandas DataFrames**, making the logic independent of the data source (Yahoo, ThetaData, etc.).

---

## 📈 Technical Logic: The Triple-Check System

The core "Actionable" logic is driven by a **Weighted Scoring System (0-10)** to eliminate noise and identify high-conviction reversals.

### The Scoring Matrix
| Indicator | Logic | Weight |
| :--- | :--- | :--- |
| **Mean Reversion** | Price distance from 200-MA > 2 StdDev | 4 Points |
| **Volatility** | Price piercing Upper/Lower Bollinger Bands | 3 Points |
| **Momentum** | RSI entering Exhaustion Zones (<30 or >70) | 3 Points |

### Signal Hierarchy
* **🚀 REVERSAL (Score 7-10):** Extreme confluence. Statistical outlier with momentum exhaustion.
* **⚠️ STRETCHED (Score 4-6):** Significant price-to-mean deviation.
* **🔍 WATCH (Score 2-3):** Single indicator trigger; monitor for further extension.
* **WAIT (Score 0-1):** Price is within normal historical ranges.

---

## 🚀 How to Run
1. **Ingest Data**:
   `python src/finance_vibe/bulk_ingest.py`
2. **Run Scanner**:
   `python src/finance_vibe/mean_reversion.py`
   # Finance Vibe Analysis 🚀

A modular Python data pipeline that identifies "market vibes" by merging dynamic market discovery with technical indicators.

## 🛠 Project Structure
- `config.py`: The system "brain" containing shared parameters (5y Weekly data, Static ETFs).
- `ticker_provider.py`: Discovers the top active stocks + benchmark ETFs (SPY, QQQ, IWM).
- `data_ingestor.py`: Pulls and cleans 5 years of weekly historical data.
- `analysis_engine.py`: Calculates RSI/CCI convergence trends.

## ⚙️ Shared Logic
All scripts use the naming convention defined in `config.py`:
`{TICKER}_{PERIOD}_{INTERVAL}.csv`

---

## 📝 Future Roadmap
- [ ] Refactor math into `indicators.py` for OOP compliance.
- [ ] Integrate Options Sentiment (Put/Call Ratio) for 4th-level confluence.
- [ ] Implement automated PDF Reporting for daily signal snapshots.