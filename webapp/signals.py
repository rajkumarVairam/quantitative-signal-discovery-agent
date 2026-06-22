from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "src" / "signal_discovery_workflow" / "data" / "sp500"

MEAN_REV_WEIGHT = 0.45
MOMENTUM_WEIGHT = 0.45
VOLATILITY_WEIGHT = 0.10  # low vol is desirable, so we use (1 - vol_rank)

ATR_STOP_MULT = 1.5   # stop loss = entry - 1.5 × ATR
ATR_TARGET_MULT = 2.5  # take profit = entry + 2.5 × ATR


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for field in ["Open", "Close", "High", "Low", "Volume"]:
        path = DATA_DIR / f"{field}.csv"
        data[field] = pd.read_csv(path, index_col=0, parse_dates=True)
    return data


# ── Core signals ───────────────────────────────────────────────────────────────

def signal_mean_reversion(close: pd.DataFrame) -> pd.DataFrame:
    """Rank( Div( Decay_Linear(Close, 20), Close ) )

    Linearly-weighted 20-day average price divided by today's close.
    High rank → price fell below its recent weighted average → bounce expected.
    """
    weights = np.arange(1, 21, dtype=float)
    weights /= weights.sum()
    decay = close.rolling(20).apply(lambda x: np.dot(x, weights), raw=True)
    return (decay / close).rank(axis=1, pct=True)


def signal_volatility(close: pd.DataFrame) -> pd.DataFrame:
    """TS_Std( TS_Return(Close, 1), 10 )

    10-day standard deviation of daily returns.
    High rank → more volatile stock (bigger swings, higher risk).
    In the composite score volatility rank is inverted (low vol is preferred).
    """
    return close.pct_change(1).rolling(10).std().rank(axis=1, pct=True)


def signal_momentum(close: pd.DataFrame, open_: pd.DataFrame) -> pd.DataFrame:
    """Mul( TS_Return(Close, 10), TS_Mean(Open, 30) )

    10-day price return × 30-day average open price.
    High rank → strong recent momentum (trend-following signal).
    """
    raw = close.pct_change(10) * open_.rolling(30).mean()
    return raw.rank(axis=1, pct=True)


# ── Additional trading metrics ─────────────────────────────────────────────────

def compute_rsi(close: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return (100 - 100 / (1 + rs)).iloc[-1]


def compute_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.DataFrame(
        np.maximum(
            np.maximum((high - low).values, (high - prev_close).abs().values),
            (low - prev_close).abs().values,
        ),
        index=close.index,
        columns=close.columns,
    )
    return tr.rolling(period).mean().iloc[-1]


def compute_volume_surge(volume: pd.DataFrame, period: int = 20) -> pd.Series:
    """Today's volume divided by N-day average. >1.5 = unusual institutional activity."""
    return (volume / volume.rolling(period).mean()).iloc[-1]


def compute_price_vs_ma20(close: pd.DataFrame) -> pd.Series:
    """Percentage distance from the 20-day moving average. Negative = below average."""
    ma20 = close.rolling(20).mean()
    return ((close - ma20) / ma20 * 100).iloc[-1]


def compute_trend_50d(close: pd.DataFrame) -> pd.Series:
    """50-day price return as a percentage."""
    return (close.pct_change(50) * 100).iloc[-1]


def compute_ret_1d(close: pd.DataFrame) -> pd.Series:
    return (close.pct_change(1) * 100).iloc[-1]


def compute_ret_5d(close: pd.DataFrame) -> pd.Series:
    return (close.pct_change(5) * 100).iloc[-1]


# ── Action label logic ─────────────────────────────────────────────────────────

def _assign_action(row: pd.Series) -> str:
    score = row["composite_score"]
    rsi = row["rsi"]
    vol = row["vol_rank"]

    if rsi > 75:
        return "Overbought"
    if rsi < 25:
        return "Oversold"
    if score >= 0.70 and vol < 0.70:
        return "Strong Buy"
    if score >= 0.55:
        return "Buy"
    if score <= 0.30:
        return "Avoid"
    return "Neutral"


# ── Position sizing helper ─────────────────────────────────────────────────────

def compute_position_size(account: float, risk_pct: float, price: float, atr: float) -> dict:
    """
    Standard ATR-based position sizing.
    Risk amount = account × risk_pct
    Shares = floor(risk_amount / (ATR_STOP_MULT × ATR))
    """
    risk_amount = account * risk_pct
    stop_distance = ATR_STOP_MULT * atr
    shares = int(risk_amount / stop_distance) if stop_distance > 0 else 0
    position_value = shares * price
    return {
        "shares": shares,
        "position_value": position_value,
        "risk_amount": risk_amount,
        "position_pct": (position_value / account * 100) if account > 0 else 0,
    }


# ── Main report builder ────────────────────────────────────────────────────────

def build_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = data["Close"]
    open_ = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # Run signals on full history then extract latest row
    mr_rank = signal_mean_reversion(close).iloc[-1]
    vol_rank = signal_volatility(close).iloc[-1]
    mom_rank = signal_momentum(close, open_).iloc[-1]

    # Additional metrics (each returns a Series of latest values)
    rsi = compute_rsi(close)
    atr = compute_atr(high, low, close)
    vsurge = compute_volume_surge(volume)
    pma20 = compute_price_vs_ma20(close)
    t50d = compute_trend_50d(close)
    ret1d = compute_ret_1d(close)
    ret5d = compute_ret_5d(close)
    latest_close = close.iloc[-1]

    # Align on tickers that have valid data across all signals
    tickers = (
        latest_close.dropna().index
        .intersection(mr_rank.dropna().index)
        .intersection(mom_rank.dropna().index)
        .intersection(rsi.dropna().index)
        .intersection(atr.dropna().index)
    )

    df = pd.DataFrame(index=tickers)
    df["close"] = latest_close[tickers].round(2)
    df["ret_1d_pct"] = ret1d[tickers].round(2)
    df["ret_5d_pct"] = ret5d[tickers].round(2)
    df["mean_rev_rank"] = mr_rank[tickers]
    df["vol_rank"] = vol_rank[tickers]
    df["momentum_rank"] = mom_rank[tickers]

    # Composite: vol rank is inverted because low volatility is preferred
    df["composite_score"] = (
        MEAN_REV_WEIGHT * df["mean_rev_rank"]
        + MOMENTUM_WEIGHT * df["momentum_rank"]
        + VOLATILITY_WEIGHT * (1.0 - df["vol_rank"])
    )

    df["rsi"] = rsi[tickers].round(1)
    df["atr"] = atr[tickers].round(2)
    df["volume_surge"] = vsurge[tickers].round(2)
    df["price_vs_ma20_pct"] = pma20[tickers].round(2)
    df["trend_50d_pct"] = t50d[tickers].round(2)
    df["stop_loss"] = (df["close"] - ATR_STOP_MULT * df["atr"]).round(2)
    df["take_profit"] = (df["close"] + ATR_TARGET_MULT * df["atr"]).round(2)
    df["risk_reward"] = round(ATR_TARGET_MULT / ATR_STOP_MULT, 2)

    df["action"] = df.apply(_assign_action, axis=1)

    df.index.name = "ticker"
    df = df.reset_index()
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
