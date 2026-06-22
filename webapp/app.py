from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make webapp/signals importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from signals import (
    build_report,
    compute_position_size,
    latest_trading_date,
    load_data,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="S&P 500 Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Cached data loader ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_report() -> tuple[pd.DataFrame, str]:
    data = load_data()
    return build_report(data), latest_trading_date(data)


# ── Colour helpers ─────────────────────────────────────────────────────────────

ACTION_COLORS = {
    "Strong Buy": "#00FF88",
    "Buy": "#00CC66",
    "Neutral": "#AAAAAA",
    "Avoid": "#FF4444",
    "Overbought": "#FF8800",
    "Oversold": "#6699FF",
}


def _fmt_pct(v):
    return f"{v:+.2f}%" if pd.notna(v) else ""


# ── Header ─────────────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("📈 S&P 500 Quantitative Signal Dashboard")
with col_btn:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── Load report ────────────────────────────────────────────────────────────────

with st.spinner("Fetching live data from Yahoo Finance…"):
    df, data_date = get_report()

st.caption(
    f"Data as of: **{data_date}** · Live from Yahoo Finance · Cache TTL: 1 hour"
)

# ── KPI cards ──────────────────────────────────────────────────────────────────

strong_buy = (df["action"] == "Strong Buy").sum()
buy = (df["action"] == "Buy").sum()
avoid = (df["action"] == "Avoid").sum()
overbought = (df["action"] == "Overbought").sum()
avg_rsi = df["rsi"].mean()
vol_surges = (df["volume_surge"] > 1.5).sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Stocks", len(df))
k2.metric("🟢 Buy Signals", buy + strong_buy, delta=f"{strong_buy} Strong Buy")
k3.metric("🔴 Avoid", avoid)
k4.metric("⚠️ Overbought", overbought)
k5.metric("Avg RSI", f"{avg_rsi:.1f}")
k6.metric("🔊 Volume Surges", vol_surges, help="Stocks with today's volume > 1.5× 20-day average")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_ov, tab_mr, tab_vol, tab_mom, tab_risk = st.tabs([
    "🏆 Overview",
    "🔄 Mean Reversion",
    "⚡ Volatility",
    "🚀 Momentum",
    "🛡️ Risk Guide",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_ov:
    st.subheader("Top Stock Picks — Composite Signal Score")
    st.caption(
        "Composite = **45% Mean Reversion** + **45% Momentum** + **10% Low Volatility**  "
        "· Scores are cross-sectional percentile ranks (0 = weakest, 1 = strongest)"
    )

    n = st.slider("Stocks to display", min_value=10, max_value=min(100, len(df)), value=50, step=10)
    display = df.head(n).copy()

    # Format for display
    display_view = display[[
        "ticker", "close", "ret_1d_pct", "ret_5d_pct",
        "composite_score", "mean_rev_rank", "momentum_rank", "vol_rank",
        "rsi", "volume_surge", "price_vs_ma20_pct", "action",
    ]].rename(columns={
        "ticker": "Ticker", "close": "Price ($)", "ret_1d_pct": "1D %",
        "ret_5d_pct": "5D %", "composite_score": "Composite",
        "mean_rev_rank": "Mean Rev", "momentum_rank": "Momentum",
        "vol_rank": "Volatility", "rsi": "RSI(14)",
        "volume_surge": "Vol Surge", "price_vs_ma20_pct": "vs MA20 %",
        "action": "Action",
    })

    styled = (
        display_view.style
        .background_gradient(subset=["Composite", "Mean Rev", "Momentum"], cmap="RdYlGn", vmin=0, vmax=1)
        .background_gradient(subset=["Volatility"], cmap="RdYlGn_r", vmin=0, vmax=1)
        .background_gradient(subset=["RSI(14)"], cmap="RdYlGn_r", vmin=20, vmax=80)
        .format({
            "Price ($)": "${:.2f}",
            "1D %": "{:+.2f}%",
            "5D %": "{:+.2f}%",
            "Composite": "{:.3f}",
            "Mean Rev": "{:.3f}",
            "Momentum": "{:.3f}",
            "Volatility": "{:.3f}",
            "RSI(14)": "{:.1f}",
            "Vol Surge": "{:.2f}×",
            "vs MA20 %": "{:+.1f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, height=450, hide_index=True)

    # Grouped bar chart — top 20 stocks
    st.subheader("Signal Breakdown — Top 20 Stocks")
    top20 = df.head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean Reversion", x=top20["ticker"], y=top20["mean_rev_rank"],
                         marker_color="#00D4FF"))
    fig.add_trace(go.Bar(name="Momentum", x=top20["ticker"], y=top20["momentum_rank"],
                         marker_color="#00FF88"))
    fig.add_trace(go.Bar(name="Volatility (lower=safer)", x=top20["ticker"], y=top20["vol_rank"],
                         marker_color="#FF8800", opacity=0.7))
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=380,
        legend=dict(orientation="h", y=1.08),
        xaxis_title="Ticker",
        yaxis_title="Percentile Rank (0–1)",
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════════════

with tab_mr:
    st.subheader("🔄 Mean Reversion Signal")
    st.info(
        "**Formula:** `Rank( Decay_Linear(Close, 20) / Close )`\n\n"
        "Compares a linearly-weighted 20-day average price to today's closing price. "
        "The decay function gives the most recent day **20× more weight** than the oldest day, "
        "so recent weakness is amplified.\n\n"
        "**High rank (→ 1.0):** Price has dropped well below its recent average → "
        "historically these stocks tend to bounce back within 2–5 trading days.\n\n"
        "**Low rank (→ 0.0):** Price is extended above its average → mean-reversion risk."
    )

    mr_sorted = df.sort_values("mean_rev_rank", ascending=False).copy()
    n_mr = st.slider("Stocks to show", 10, min(100, len(df)), 30, key="mr_slider")

    mr_view = mr_sorted.head(n_mr)[[
        "ticker", "close", "ret_1d_pct", "ret_5d_pct",
        "mean_rev_rank", "price_vs_ma20_pct", "rsi", "composite_score", "action",
    ]].rename(columns={
        "ticker": "Ticker", "close": "Price ($)", "ret_1d_pct": "1D %", "ret_5d_pct": "5D %",
        "mean_rev_rank": "Mean Rev Rank", "price_vs_ma20_pct": "vs MA20 %",
        "rsi": "RSI(14)", "composite_score": "Composite", "action": "Action",
    })

    st.dataframe(
        mr_view.style
        .background_gradient(subset=["Mean Rev Rank"], cmap="RdYlGn", vmin=0, vmax=1)
        .background_gradient(subset=["RSI(14)"], cmap="RdYlGn_r", vmin=20, vmax=80)
        .format({
            "Price ($)": "${:.2f}", "1D %": "{:+.2f}%", "5D %": "{:+.2f}%",
            "Mean Rev Rank": "{:.3f}", "vs MA20 %": "{:+.1f}%",
            "RSI(14)": "{:.1f}", "Composite": "{:.3f}",
        }),
        use_container_width=True, height=420, hide_index=True,
    )

    top15_mr = mr_sorted.head(15)
    fig_mr = px.bar(
        top15_mr, x="ticker", y="mean_rev_rank",
        color="mean_rev_rank", color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        title="Top 15 Mean Reversion Candidates",
        template="plotly_dark",
        labels={"ticker": "Ticker", "mean_rev_rank": "Mean Rev Rank"},
        hover_data={"price_vs_ma20_pct": ":.1f", "rsi": ":.1f", "close": ":.2f"},
    )
    fig_mr.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_mr, use_container_width=True)

    st.markdown(
        "**Trading tip:** Best setups have Mean Rev Rank **> 0.75**, "
        "RSI **< 50**, and Price vs MA20 **< −3%**. "
        "Consider a 2–5 day long trade. Exit when price returns to the 20-day MA."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

with tab_vol:
    st.subheader("⚡ Volatility Signal")
    st.info(
        "**Formula:** `TS_Std( TS_Return(Close, 1), 10 )`\n\n"
        "Measures the standard deviation of daily returns over the past 10 trading days — "
        "a proxy for how much the stock swings day-to-day.\n\n"
        "In the **composite score**, volatility rank is **inverted** (lower = better) "
        "because excessive volatility means higher risk without guaranteed reward.\n\n"
        "Use this tab to size your positions: **higher ATR → smaller position** to keep dollar risk constant."
    )

    col_hv, col_lv = st.columns(2)

    with col_hv:
        st.markdown("##### 🔥 High Volatility — Big Moves (Higher Risk)")
        high_vol = df.sort_values("vol_rank", ascending=False).head(15)[[
            "ticker", "vol_rank", "atr", "rsi", "composite_score", "action"
        ]].rename(columns={
            "ticker": "Ticker", "vol_rank": "Vol Rank", "atr": "ATR($)",
            "rsi": "RSI(14)", "composite_score": "Composite", "action": "Action",
        })
        st.dataframe(
            high_vol.style
            .background_gradient(subset=["Vol Rank"], cmap="Reds", vmin=0, vmax=1)
            .format({"Vol Rank": "{:.3f}", "ATR($)": "${:.2f}", "RSI(14)": "{:.1f}", "Composite": "{:.3f}"}),
            use_container_width=True, height=360, hide_index=True,
        )

    with col_lv:
        st.markdown("##### 🧘 Low Volatility — Stable (Lower Risk)")
        low_vol = df.sort_values("vol_rank", ascending=True).head(15)[[
            "ticker", "vol_rank", "atr", "rsi", "composite_score", "action"
        ]].rename(columns={
            "ticker": "Ticker", "vol_rank": "Vol Rank", "atr": "ATR($)",
            "rsi": "RSI(14)", "composite_score": "Composite", "action": "Action",
        })
        st.dataframe(
            low_vol.style
            .background_gradient(subset=["Vol Rank"], cmap="Greens_r", vmin=0, vmax=1)
            .format({"Vol Rank": "{:.3f}", "ATR($)": "${:.2f}", "RSI(14)": "{:.1f}", "Composite": "{:.3f}"}),
            use_container_width=True, height=360, hide_index=True,
        )

    st.subheader("Volatility vs Composite Score")
    fig_scatter = px.scatter(
        df,
        x="vol_rank",
        y="composite_score",
        size=df["volume_surge"].clip(upper=5),
        color="action",
        hover_name="ticker",
        hover_data={"rsi": ":.1f", "close": ":.2f", "atr": ":.2f"},
        title="Each bubble = one stock  ·  Size = Volume Surge  ·  Color = Action",
        template="plotly_dark",
        color_discrete_map=ACTION_COLORS,
        labels={"vol_rank": "Volatility Rank (0=low, 1=high)", "composite_score": "Composite Score"},
    )
    fig_scatter.update_layout(height=420, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown(
        "**Position sizing rule:** `Shares = (Account × Risk%) ÷ (1.5 × ATR)` — "
        "a larger ATR means a smaller position to keep your dollar risk per trade constant."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

with tab_mom:
    st.subheader("🚀 Momentum Signal")
    st.info(
        "**Formula:** `Mul( TS_Return(Close, 10), TS_Mean(Open, 30) )`\n\n"
        "Combines the 10-day price return with the 30-day average opening price. "
        "Stocks already moving up with higher price levels score the highest.\n\n"
        "**High rank (→ 1.0):** Strong recent price momentum — trend-following "
        "setups that tend to continue for 5–10 more days.\n\n"
        "**Best with:** 50-day trend positive AND RSI below 70 (not yet overextended)."
    )

    mom_sorted = df.sort_values("momentum_rank", ascending=False).copy()
    n_mom = st.slider("Stocks to show", 10, min(100, len(df)), 30, key="mom_slider")

    mom_view = mom_sorted.head(n_mom)[[
        "ticker", "close", "ret_1d_pct", "ret_5d_pct",
        "momentum_rank", "trend_50d_pct", "rsi", "composite_score", "action",
    ]].rename(columns={
        "ticker": "Ticker", "close": "Price ($)", "ret_1d_pct": "1D %", "ret_5d_pct": "5D %",
        "momentum_rank": "Mom Rank", "trend_50d_pct": "50D Trend %",
        "rsi": "RSI(14)", "composite_score": "Composite", "action": "Action",
    })

    st.dataframe(
        mom_view.style
        .background_gradient(subset=["Mom Rank"], cmap="RdYlGn", vmin=0, vmax=1)
        .background_gradient(subset=["RSI(14)"], cmap="RdYlGn_r", vmin=20, vmax=80)
        .format({
            "Price ($)": "${:.2f}", "1D %": "{:+.2f}%", "5D %": "{:+.2f}%",
            "Mom Rank": "{:.3f}", "50D Trend %": "{:+.1f}%",
            "RSI(14)": "{:.1f}", "Composite": "{:.3f}",
        }),
        use_container_width=True, height=420, hide_index=True,
    )

    top15_mom = mom_sorted.head(15)
    fig_mom = px.bar(
        top15_mom, x="ticker", y="momentum_rank",
        color="trend_50d_pct",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title="Top 15 Momentum Stocks — colored by 50-day trend return",
        template="plotly_dark",
        labels={"ticker": "Ticker", "momentum_rank": "Momentum Rank", "trend_50d_pct": "50D Return %"},
        hover_data={"rsi": ":.1f", "close": ":.2f"},
    )
    fig_mom.update_layout(height=380)
    st.plotly_chart(fig_mom, use_container_width=True)

    st.markdown(
        "**Trading tip:** Best momentum setups have Momentum Rank **> 0.80**, "
        "50D Trend **> 0%**, and RSI **< 70**. "
        "Hold for 5–10 days. Set a stop-loss 1.5× ATR below your entry price."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK GUIDE
# ══════════════════════════════════════════════════════════════════════════════

with tab_risk:
    st.subheader("🛡️ Risk Management Guide")

    st.markdown("""
### ATR-Based Stop Loss & Take Profit

Every stock in this table has pre-calculated entry zones based on its **ATR(14)** —
the average daily price range over the past 14 days.

| Level | Formula | Meaning |
|-------|---------|---------|
| **Stop Loss** | Entry − **1.5 × ATR** | Exit if the trade moves against you |
| **Take Profit** | Entry + **2.5 × ATR** | Exit when your profit target is hit |
| **Risk / Reward** | 2.5 ÷ 1.5 = **1.67×** | For every $1 risked, target $1.67 gain |

A 1.67× R:R means you only need to be **right 38% of the time** to break even —
giving you plenty of margin for imperfect entries.
""")

    # Buy candidates table
    buy_mask = df["action"].isin(["Strong Buy", "Buy"])
    buy_df = df[buy_mask].sort_values("composite_score", ascending=False)

    st.subheader(f"Buy Candidates Today ({len(buy_df)} stocks)")
    if buy_df.empty:
        st.warning("No clear buy candidates today. The market may be in a neutral or overbought phase.")
    else:
        buy_view = buy_df[[
            "ticker", "close", "atr", "stop_loss", "take_profit",
            "risk_reward", "composite_score", "rsi", "volume_surge", "action",
        ]].rename(columns={
            "ticker": "Ticker", "close": "Price ($)", "atr": "ATR($)",
            "stop_loss": "Stop Loss", "take_profit": "Take Profit",
            "risk_reward": "R:R", "composite_score": "Composite",
            "rsi": "RSI(14)", "volume_surge": "Vol Surge", "action": "Action",
        })
        st.dataframe(
            buy_view.style
            .background_gradient(subset=["Composite"], cmap="Greens", vmin=0.5, vmax=1)
            .format({
                "Price ($)": "${:.2f}", "ATR($)": "${:.2f}",
                "Stop Loss": "${:.2f}", "Take Profit": "${:.2f}",
                "R:R": "{:.2f}×", "Composite": "{:.3f}",
                "RSI(14)": "{:.1f}", "Vol Surge": "{:.2f}×",
            }),
            use_container_width=True, height=380, hide_index=True,
        )

    st.divider()

    # Interactive position sizing calculator
    st.subheader("📐 Position Sizing Calculator")
    st.caption("How many shares to buy so one bad trade never hurts too much.")

    col_inputs, col_result = st.columns([1, 1])
    with col_inputs:
        account_size = st.number_input(
            "Account Equity ($)", min_value=1000, max_value=10_000_000,
            value=10_000, step=1000,
        )
        risk_pct_input = st.slider(
            "Risk per Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.25,
        )
        risk_pct = risk_pct_input / 100

        ticker_options = buy_df["ticker"].tolist() if not buy_df.empty else df["ticker"].head(20).tolist()
        selected_ticker = st.selectbox("Select Stock", ticker_options)

    with col_result:
        if selected_ticker:
            row = df[df["ticker"] == selected_ticker].iloc[0]
            sizing = compute_position_size(account_size, risk_pct, row["close"], row["atr"])
            st.markdown(f"""
**Trade Plan for `{selected_ticker}`**

| | Value |
|--|--|
| Entry Price | **${row['close']:.2f}** |
| Stop Loss | **${row['stop_loss']:.2f}** (−${row['atr'] * 1.5:.2f}) |
| Take Profit | **${row['take_profit']:.2f}** (+${row['atr'] * 2.5:.2f}) |
| Risk Amount | **${sizing['risk_amount']:.0f}** ({risk_pct_input:.1f}% of portfolio) |
| **Shares to Buy** | **{sizing['shares']}** |
| Position Value | **${sizing['position_value']:.0f}** ({sizing['position_pct']:.1f}% of portfolio) |
| Composite Score | {row['composite_score']:.3f} · RSI {row['rsi']:.1f} · Action: {row['action']} |
""")

    st.divider()

    # Stocks to avoid
    avoid_df = df[df["action"].isin(["Avoid", "Overbought"])].sort_values("composite_score")
    st.subheader(f"⚠️ Stocks to Avoid Today ({len(avoid_df)})")
    if not avoid_df.empty:
        avoid_view = avoid_df[[
            "ticker", "close", "composite_score", "rsi", "vol_rank", "trend_50d_pct", "action",
        ]].rename(columns={
            "ticker": "Ticker", "close": "Price ($)", "composite_score": "Composite",
            "rsi": "RSI(14)", "vol_rank": "Vol Rank", "trend_50d_pct": "50D Trend %",
            "action": "Reason",
        })
        st.dataframe(
            avoid_view.style
            .background_gradient(subset=["Composite"], cmap="Reds_r", vmin=0, vmax=0.5)
            .format({
                "Price ($)": "${:.2f}", "Composite": "{:.3f}",
                "RSI(14)": "{:.1f}", "Vol Rank": "{:.3f}", "50D Trend %": "{:+.1f}%",
            }),
            use_container_width=True, height=320, hide_index=True,
        )
    else:
        st.success("No major red flags today.")

    st.divider()

    st.subheader("📋 7 Quick Trading Rules")
    st.markdown("""
1. **Never risk more than 1–2% of your account on a single trade.** Use the calculator above.
2. **Always set your stop-loss before you enter.** Never move it further away once set.
3. **RSI > 75 = wait.** Even strong signals can pull back from overbought levels.
4. **Volume confirms price.** Prefer Buy candidates with Volume Surge > 1.2×.
5. **Trade with the 50-day trend.** Momentum signals work best when 50D trend is positive.
6. **Don't concentrate.** Spread your buys across different sectors.
7. **These are signals, not guarantees.** Combine with your own research and market conditions.
""")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This dashboard is for research and educational purposes only. "
    "Not financial advice. Past signal performance does not guarantee future results. "
    "Always do your own due diligence before trading."
)
