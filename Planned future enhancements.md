  This is an exceptionally well-thought-out, institutional-grade rubric. You have successfully merged structural market geometry with options-specific Greeks—specifically targeting the "volatility tax" ($Vega$ risk) that ruins most retail LEAPS buyers.

As a senior financial planner looking at capital preservation, and a market practitioner looking at edge, **I fully endorse this rubric.** Here is my institutional critique, followed by a professional, step-by-step execution playbook to operationalize this system.

---

## 🔍 Critical Review & Enhancements

Your rubric is structurally sound, but to make it truly bulletproof for a portfolio, we need to tighten a few loose screws.

### 1. The Volatility Paradox (The $70\%$ IV Problem)

You correctly penalized the $69.96\%$ IV in your HOOD example. However, for high-beta, high-growth tech or momentum stocks, **IV Rank (IVR)** or **IV Percentile (IVP)** is a better metric than raw IV.

* **The Fix:** If a stock's historical *low* IV over 52 weeks is $60\%$, then a $70\%$ IV is actually relatively cheap (low IV Rank). Ensure your rubric explicitly uses **IV Rank < 30**, rather than an arbitrary absolute IV percentage.

### 2. Delta & Liquidity Guardrails (Missing Metrics)

Because LEAPS are illiquid compared to front-month options, you risk losing $2-5\%$ of your position value just crossing the bid-ask spread.

* **The Fix:** Add a strict **Liquidity Gatekeeper**. If the Bid-Ask spread of the LEAPS contract is wider than $5\%$, the trade is a hard pass, regardless of the technical score. Furthermore, ensure you are buying a **minimum of 0.70 Delta** to simulate stock replacement and minimize extrinsic decay ($Theta$).

### 3. The "Scale-In" Trap

For Grade B setups (75–84 pts), you suggest a $50\%$ starter position. If the reason for the lower score is high IV, scaling into a falling volatility environment means your first $50\%$ will lose value rapidly to $Vega$ burn, even if the stock goes sideways or slightly up.

* **The Fix:** For Grade B setups triggered by high IV, the execution protocol should mandate a **diagonally structured entry** or a strict **price-limit order**, rather than market orders.

---

## 🛠️ Step-by-Step Execution Playbook

To implement this systematically without letting emotion or FOMO override the rubric, follow this operational workflow.

### Step 1: The Weekly Scan & Universe Filtering

Do not hunt for trades intraday. Run your filters when the market is closed.

* **Action:** Filter your watch list for underlying equities trading cleanly above their rising Daily 200-SMA or Weekly 200-MA.
* **Tooling:** Use a charting platform to flag assets where the 14-period RSI is oversold or flattening out near key structural horizontal support.

### Step 2: The Volatility Check (The Gatekeeper)

Before you calculate momentum, check the options chain. If the volatility regime is wrong, the trade is dead.

* **Action:** Pull up the IV Rank / IV Percentile.
* If **IVR > 40**, stop. Do not pass go. You are overpaying for premium.
* If **IVR < 30** and Bollinger Bands are compressing, proceed to the scoring matrix.



### Step 3: Run the Scoring Matrix

Manually score the asset using your 100-point system. Treat this like an underwriter assessing a loan.

### Step 4: Order Execution & Order Types

Never use market orders on LEAPS.

* **For Grade A Setups (85-100 pts):** Place a **Limit Order** exactly at the **Mid-Price** of the options chain. If it does not fill within 30 minutes, adjust to Mid + $0.05.
* **For Grade B Setups (75-84 pts):** Place a **GTC (Good 'Til Canceled) Limit Order** at the **Buyer's Bid** or a calculated price targets based on the structural support retest. Let the market come to you to offset the higher IV premium you are paying.

### Step 5: Capital Allocation & Sizing

* **Max Portfolio Risk:** No single LEAPS position should represent more than **$2\% - 3\%$ of your total liquid trading capital** at initiation.
* Because you are buying high-Delta ($>0.70$), treat this as a stock replacement strategy. If your target allocation for owning the actual stock was $10,000, your LEAPS premium allocation should be roughly $3,000–$4,000, allowing the remaining capital to sit in high-yield cash equivalents generating risk-free yield.

---

## 📈 Summary of Execution Protocol

| Setup Rating | Score | Sizing Strategy | Order Entry Protocol |
| --- | --- | --- | --- |
| **Grade A** | 85 – 100 | Full Position ($2-3\%$ Portfolio Cap) | Limit Order at Mid-Price. |
| **Grade B** | 75 – 84 | Starter Position ($1-1.5\%$ Portfolio Cap) | GTC Limit Order at Bid or structural key levels. |
| **Below 75** | < 75 | **0% Allocation** | Sit on hands. Archive the chart for a monthly review. |

This rubric elevates your process from simple retail speculation to systematic portfolio management. Stick to these rules, let the high-IV setups pass you by, and protect that capital.
