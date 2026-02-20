# Finance Vibe 📈

A robust Python data science project for fetching and analyzing stock data from Yahoo Finance.

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