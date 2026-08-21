# Learn Finance Vibe

This project is a **lab**: live Coiled Cobra scans, expansion trade plans, and a ranking model — plus offline TA engines so you can see how markets, technical analysis, and machine learning fit together.

The dashboard at **port 5000** (container `finance_vibe`) has **Plans**, **Learn**, and **Docs**. This file is the curriculum index. Specs stay in their own documents; primers (`LearnTA.md`, `LearnML.md`) teach the words.

After you edit markdown on the host, **rebuild the image** so `/app` picks it up (`docker compose up -d --build`). Compose mounts **data** at `/app/data`, not these docs.

Suggested order: **A → B → C**. Then operate: scan, read a plan, train only when you understand Spearman.

---

## Track A — Markets and technical analysis

| Step | Topic | Read | Run (optional) |
| ---- | ----- | ---- | -------------- |
| A1 | OHLCV, adjusted close, why QQQ is the benchmark | [LearnTA.md](/docs/learn-ta) | `docker exec finance_vibe ls /app/data/raw/daily/QQQ_5y_1d.csv` |
| A2 | EMA, ATR, MACD, RSI in Cobra language | [LearnTA.md](/docs/learn-ta) | Open a raw CSV; compare Close vs a 20-bar average |
| A3 | Relative strength vs QQQ | Rubric § RS | Live scan column `RS_Score` / `RS 63d` |
| A4 | Coil vs pullback vs vibe | LearnTA “two labs”; [swing](/docs/swing); [vibe](/docs/vibe) | Offline only — not in `run_vibe` |
| A5 | Risk: R-multiple, Coil_Low, 5% cap | [Trade plan math](/docs/trade-plan) | Open `trade_plan_*.csv` |

**Source of truth for live scoring:** [Coiled Cobra Rubric](/docs/rubric).

---

## Track B — The system

| Step | Topic | Read | Run |
| ---- | ----- | ---- | --- |
| B1 | Universe and survivorship (`active_tickers.csv`) | [Operation manual](/docs/ops) | `docker exec -w /app finance_vibe python src/finance_vibe/ticker_provider.py` |
| B2 | Walk-forward: only bars through `i` | [Backtest guide](/docs/backtest) | Smoke: `coiled_cobra_backtest.py --backtest --tickers SPY,QQQ,IWM` |
| B3 | Hard gates vs Score vs `ML_Rank` | Rubric gates; LearnML | Compare columns on a cleaned plan |
| B4 | Docker: image (`/app/src`, markdown) vs volume (`/app/data`) | [MLOps.md](/docs/mlops) §2 | `docker exec finance_vibe ls /app /app/data/logs/daily` |

Live path: ingest → `coiled_cobra.py` → planner → helper. `run_vibe.py` **does not train**.

---

## Track C — AI / ML under the hood

| Step | Topic | Read | Run |
| ---- | ----- | ---- | --- |
| C1 | Features `X`, target `y`, leakage | [LearnML.md](/docs/learn-ml) | — |
| C2 | `Is_New_Coil`, temporal split, 2-week embargo | LearnML; [CoiledCobraML.md](/docs/cobra-ml) | Need a full `--backtest` CSV |
| C3 | Boosting, MAE vs RMSE vs Spearman | LearnML | Read `coiled_cobra_ml_model_metadata.json` |
| C4 | Fail-soft inference (`ml_ranker.py`) | LearnML; MLOps §4.11 | Scan; check `ML_Rank` null vs filled |
| C5 | Train in the container | [MLOps.md](/docs/mlops) §5 | `--backtest` then `coiled_cobra_ml_training.py` |

---

## Quick Docker card

```bash
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra.py
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra_backtest.py --backtest
docker exec -w /app finance_vibe python src/finance_vibe/coiled_cobra_ml_training.py \
  --csv /app/data/logs/daily/coiled_cobra_backtest_trades_YYYY-MM-DD.csv \
  --artifacts-dir /app/data/logs/daily
```

Replace the date with the file you actually produced. Full train/deploy: **MLOps.md**.
