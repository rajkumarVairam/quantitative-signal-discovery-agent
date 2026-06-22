# Trading Signal Dashboard — User Guide

A Streamlit web dashboard that runs three quantitative signals daily across all S&P 500 stocks
and tells you which ones to buy, avoid, and how much to risk on each trade.

## Quick Start

```powershell
# 1. Install dependencies (one-time)
uv sync --locked --extra webapp

# 2. Download S&P 500 price data (one-time, ~2-3 minutes)
uv run python -m signal_discovery_workflow.download_data

# 3. Launch the dashboard
uv run streamlit run webapp\app.py
```

Opens at **http://localhost:8501**

> **Note:** No NVIDIA API key is required to run the dashboard. It computes signals locally from the downloaded price data.

---

## How Data Stays Fresh

```
4:00 PM ET   Market closes
4:15 PM ET   Yahoo Finance publishes closing prices
5:00 PM CST  Windows Task Scheduler runs webapp/update_data.py
               → downloads latest prices for all 386 S&P 500 stocks
               → overwrites CSV files in src/signal_discovery_workflow/data/sp500/
               → logs result to webapp/update_data.log
6:00 PM CST  Dashboard cache expires → next page visit shows fresh data
```

To refresh manually at any time, click the **Refresh** button in the top-right corner of the dashboard.

---

## The Three Signals

### 1. Mean Reversion
**Formula:** `Rank( Decay_Linear(Close, 20) / Close )`

Compares a linearly-weighted 20-day average price to today's closing price.
A high rank means the stock has **fallen below its recent average** and is likely to bounce back.

- Best hold period: **2–5 trading days**
- Best entry: stock down 3–5% below its 20-day moving average
- Exit: when price returns to the moving average, or after 5 days

### 2. Volatility
**Formula:** `TS_Std( TS_Return(Close, 1), 10 )`

Standard deviation of daily returns over the past 10 days.
Used primarily for **position sizing** — not for deciding what to buy, but how much.

- High rank = wild mover → use smaller position size
- Low rank = stable mover → can use larger position size

### 3. Momentum
**Formula:** `Mul( TS_Return(Close, 10), TS_Mean(Open, 30) )`

Combines 10-day price return with 30-day average open price.
A high rank means the stock has **strong recent price momentum** and is likely to continue.

- Best hold period: **5–10 trading days**
- Best entry: stock in an uptrend, RSI below 70
- Exit: when RSI exceeds 75, or after 10 days

---

## The Composite Score

```
Composite = 45% × Mean Reversion Rank
          + 45% × Momentum Rank
          + 10% × (1 − Volatility Rank)
```

Volatility rank is **inverted** because low volatility is preferred — it means more
predictable moves and better risk control. The composite ranks every stock from 0 (worst)
to 1 (best) across the combined signal.

---

## Dashboard Tabs

### Overview Tab — Your Daily Starting Point

Shows all 386 stocks ranked by composite score with color-coded columns:

| Column | What it tells you | Decision trigger |
|--------|------------------|-----------------|
| **Composite** | Overall signal strength (0–1) | > 0.70 → strong candidate |
| **Mean Rev** | How oversold vs 20-day average | > 0.75 → bounce trade setup |
| **Momentum** | Strength of recent price trend | > 0.80 → trend-following setup |
| **Volatility** | Daily price swing size | < 0.50 → safer, more stable |
| **RSI(14)** | Overbought/oversold indicator | 30–65 = sweet spot; > 75 = skip |
| **Vol Surge** | Today's volume vs 20-day average | > 1.2× = institutional activity |
| **vs MA20 %** | Distance from 20-day moving average | Negative = below average = bounce opportunity |
| **Action** | Final verdict | Strong Buy / Buy / Neutral / Avoid / Overbought / Oversold |

**Color coding:**
- Green = strong signal / favorable value
- Red = weak signal / unfavorable value
- RSI column is inverted: green = moderate (safe), red = extreme (dangerous)

---

### Mean Reversion Tab — Short Bounce Trades (2–5 days)

Ranks stocks by their mean reversion signal. Best setups combine:

- Mean Rev Rank **> 0.75**
- RSI **< 50** (pullback, not panic)
- vs MA20 % **< −3%** (meaningfully below average)

**How to read the bar chart:** Green bars = strong bounce candidates. Red bars = extended above average (avoid).

---

### Volatility Tab — Position Sizing Tool

Two side-by-side tables:
- **High Volatility** (left): Stocks with big daily swings → use smaller positions
- **Low Volatility** (right): Stable stocks → can use larger positions

**Scatter plot:** Each bubble is a stock. Look for bubbles in the **top-right** (high composite + high volume surge) that are **green** (Buy/Strong Buy). These are active stocks with strong signals.

**Position sizing formula:**
```
Shares = (Account × Risk%) ÷ (1.5 × ATR)
```
Example: $10,000 account, 1% risk, ATR = $5.00 → Shares = $100 ÷ $7.50 = 13 shares

---

### Momentum Tab — Trend-Following Trades (5–10 days)

Ranks stocks by momentum signal. Best setups combine:

- Momentum Rank **> 0.80**
- 50-day trend **positive** (stock above its 50-day average)
- RSI **between 50 and 70** (trending but not overextended)

**Bar chart:** Bar height = momentum strength. Color = 50-day trend (green = uptrend confirmed).

---

### Risk Guide Tab — Before You Place Any Order

**This is the most important tab.** Every trade needs a plan before you enter.

#### Buy Candidates Table
Shows all stocks currently labeled Buy or Strong Buy with pre-calculated levels:

| Column | Meaning |
|--------|---------|
| **Price ($)** | Current closing price = your approximate entry |
| **ATR($)** | Average daily price range over 14 days |
| **Stop Loss** | Exit price if trade goes wrong (Entry − 1.5 × ATR) |
| **Take Profit** | Target exit price (Entry + 2.5 × ATR) |
| **R:R** | Risk/reward ratio — always 1.67× in this setup |

#### Interactive Position Sizing Calculator
Enter your account size and risk percentage, select a stock, and the calculator tells you:
- Exactly how many shares to buy
- Total position value
- Maximum dollar loss if stop-loss is hit

#### Stocks to Avoid
Lists stocks with weak signals or overbought RSI. Skipping these is as important as picking the right ones.

---

## Daily Trading Workflow

Follow this routine every evening after 5:30 PM CST:

```
Step 1 — Overview tab
  Scan the top 20 stocks by Composite Score
  Note any with Action = "Strong Buy"

Step 2 — Filter by RSI
  Skip any with RSI > 70 (overbought — wait for pullback)
  Prefer RSI between 40–65

Step 3 — Confirm volume activity
  Prefer stocks with Volume Surge > 1.2×
  High volume means smart money is moving — increases signal reliability

Step 4 — Choose your strategy
  Want a 2–5 day bounce trade? → Check Mean Reversion tab for top picks
  Want a 5–10 day trend trade? → Check Momentum tab for top picks
  Stock appears in both?       → Strongest possible setup

Step 5 — Risk Guide tab
  Enter your account size and risk % (start with 1%)
  Select the stock from the calculator
  Write down: Entry price, Stop Loss price, Take Profit price, Shares to buy

Step 6 — Place your order next morning
  Buy at market open (or use a limit order near the prior close)
  Set a stop-loss order immediately — before you do anything else
```

---

## Understanding the Action Labels

| Label | Meaning | What to do |
|-------|---------|------------|
| 🟢 **Strong Buy** | Composite ≥ 0.70, volatility manageable | High-conviction trade |
| 🟢 **Buy** | Composite ≥ 0.55 | Standard trade setup |
| 🟡 **Neutral** | Mixed signals | Skip unless you have other reasons |
| 🔴 **Avoid** | Composite ≤ 0.30 | Do not buy; consider exiting if you hold |
| ⚠️ **Overbought** | RSI > 75 | Even good signals fail from here — wait |
| 🔵 **Oversold** | RSI < 25 | Extreme fear — watch for a bounce but wait for confirmation |

---

## 7 Quick Trading Rules

1. **Never risk more than 1–2% of your account on a single trade.** Use the position sizing calculator.
2. **Always set your stop-loss before you enter.** Never move it further away once set.
3. **RSI > 75 = wait.** Even strong signals can pull back from overbought levels first.
4. **Volume confirms price.** Prefer signals with Volume Surge > 1.2×.
5. **Trade with the 50-day trend.** Momentum signals work best when the trend is positive.
6. **Don't concentrate.** Spread buys across different sectors — not 5 tech stocks.
7. **These are signals, not guarantees.** Combine with your own research and current market conditions.

---

## What the Signals Cannot Predict

- Earnings surprises, company news, or analyst downgrades
- Macro events: Fed decisions, economic data releases, geopolitical events
- Intraday moves — signals are based on end-of-day prices only
- Stocks outside the S&P 500 universe

The signals provide a **statistical edge** over many trades. No single trade is guaranteed. The edge compounds when you apply consistent position sizing and stop-loss discipline across many trades over time.

---

## Files

| File | Purpose |
|------|---------|
| `webapp/app.py` | Streamlit dashboard |
| `webapp/signals.py` | Signal computation and trading metrics |
| `webapp/update_data.py` | Daily data refresh script (run by Task Scheduler) |
| `webapp/update_data.log` | Log of each daily refresh (auto-generated, gitignored) |
| `.streamlit/config.toml` | Dashboard dark theme configuration |
