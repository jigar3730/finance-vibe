import pandas as pd
import sys
import os
import numpy as np

# --- 1. ENVIRONMENT SETUP ---
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_FILE_DIR, "../../"))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from finance_vibe import config
    from finance_vibe.analysis_engine_local import calculate_vibe_score
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


def run_backtest(ticker: str):
    print("\n" + "="*80)
    print(f"📊 BACKTESTING: {ticker} (Weekly/5Y)")
    print("Logic: Extension-Aware / RSI Caps / Weekly Bonus / Component Debugging")
    print("="*80)

    data_path = config.get_raw_path(ticker)
    if not os.path.exists(data_path):
        print(f"❌ ERROR: File not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    date_col = next(
        (c for c in df.columns if 'Date' in c or 'Datetime' in c), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.sort_index()

    initial_capital = config.BACKTEST_INITIAL_CAPITAL
    cash = initial_capital
    holdings = 0
    scores_seen = []

    # Warmup for indicators
    test_df = df.iloc[50:]

    for i in range(len(test_df)):
        current_date = test_df.index[i]
        window = df.loc[:current_date]

        try:
            # Use the new component-aware function
            result = calculate_vibe_score(
                ticker, window, return_components=True)
            score = result.get("Score", 0)
            components = result.get("Components", {})
            scores_seen.append(score)
        except Exception as e:
            print(f"[{current_date.date()}] ❌ Error calculating score: {e}")
            continue

        price = test_df['Close'].iloc[i]

        # Print full component breakdown for debugging
        comp_str = " | ".join(f"{k}:{v}" for k, v in components.items())
        print(
            f"[{current_date.date()}] Score: {score} | {comp_str} | Close: ${price:,.2f}")

        # BUY: Score hits Starter Position or higher
        if score >= config.BACKTEST_BUY_SCORE and cash > 0:
            holdings = cash / price
            cash = 0
            print(f"[{current_date.date()}] 🟢 BUY  @ ${price:,.2f} | Score: {score}")

        # SELL: Score drops to No Edge / Bearish
        elif score <= config.BACKTEST_SELL_SCORE and holdings > 0:
            cash = holdings * price
            holdings = 0
            print(
                f"[{current_date.date()}] 🔴 SELL @ ${price:,.2f} | Val: ${cash:,.2f} | Score: {score}")

    # Results
    final_price = test_df['Close'].iloc[-1]
    final_val = cash if holdings == 0 else holdings * final_price
    total_return = ((final_val - initial_capital) / initial_capital) * 100

    print("-"*80)
    if scores_seen:
        print(
            f"Score Range: Min {min(scores_seen)} | Max {max(scores_seen)} | Avg {np.mean(scores_seen):.1f}")
    print(f"FINAL VALUE:  ${final_val:,.2f}")
    print(f"TOTAL RETURN: {total_return:.2f}%")
    print("-"*80)


if __name__ == "__main__":
    run_backtest("QQQ")
