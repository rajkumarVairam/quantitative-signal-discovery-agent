from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# 120 calendar days ≈ 85 trading days — enough for the 50-day lookback signals
LOOKBACK_DAYS = 120

MEAN_REV_WEIGHT = 0.45
MOMENTUM_WEIGHT = 0.45
VOLATILITY_WEIGHT = 0.10  # low vol is desirable, so we use (1 - vol_rank)

ATR_STOP_MULT = 1.5   # stop loss = entry - 1.5 × ATR
ATR_TARGET_MULT = 2.5  # take profit = entry + 2.5 × ATR

SP500_TICKERS = [
    'A', 'AAPL', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG',
    'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'AMAT', 'AMD', 'AME', 'AMGN', 'AMT', 'AMZN', 'AON', 'AOS', 'APA',
    'APD', 'APH', 'ARE', 'ATO', 'AVB', 'AVY', 'AXON', 'AXP', 'AZO',
    'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BG', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK',
    'BMY', 'BRO', 'BSX', 'BWA', 'BXP',
    'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLX', 'CMA',
    'CMCSA', 'CME', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL',
    'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTRA', 'CTSH', 'CVS', 'CVX',
    'D', 'DD', 'DE', 'DECK', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DPZ', 'DRI', 'DTE', 'DUK',
    'DVA', 'DVN',
    'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES',
    'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXR',
    'F', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FRT',
    'GD', 'GE', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW',
    'HAL', 'HAS', 'HBAN', 'HD', 'HIG', 'HOLX', 'HON', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM',
    'IBM', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
    'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM',
    'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR',
    'L', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LUV', 'LVS',
    'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM',
    'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPWR', 'MRK', 'MS', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
    'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR',
    'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OXY',
    'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PNC', 'PNR',
    'PNW', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PTC', 'PWR', 'QCOM',
    'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY',
    'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
    'STX', 'STZ', 'SWK', 'SWKS', 'SYK', 'SYY',
    'T', 'TAP', 'TDY', 'TECH', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TPR', 'TRMB', 'TROW', 'TRV', 'TSCO',
    'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL',
    'UDR', 'UHS', 'UNH', 'UNP', 'UPS', 'URI', 'USB',
    'VLO', 'VMC', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY', 'WYNN',
    'XEL', 'XOM', 'YUM', 'ZBH', 'ZBRA',
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> dict[str, pd.DataFrame]:
    """Fetch the last LOOKBACK_DAYS of OHLCV data for all S&P 500 tickers live from Yahoo Finance."""
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    raw = yf.download(
        SP500_TICKERS,
        start=start,
        end=end,
        group_by="column",
        auto_adjust=True,
        progress=False,
    )

    data = {}
    for field in ["Open", "Close", "High", "Low", "Volume"]:
        if field in raw.columns.get_level_values(0):
            df = raw[field].copy()
            df.index.name = "Date"
            data[field] = df.dropna(axis=1, how="all")

    return data


def latest_trading_date(data: dict[str, pd.DataFrame]) -> str:
    idx = data.get("Close", pd.DataFrame()).index
    return idx[-1].strftime("%b %d, %Y") if len(idx) > 0 else "Unknown"


# ── Core signals ───────────────────────────────────────────────────────────────

def signal_mean_reversion(close: pd.DataFrame) -> pd.DataFrame:
    weights = np.arange(1, 21, dtype=float)
    weights /= weights.sum()
    decay = close.rolling(20).apply(lambda x: np.dot(x, weights), raw=True)
    return (decay / close).rank(axis=1, pct=True)


def signal_volatility(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(1).rolling(10).std().rank(axis=1, pct=True)


def signal_momentum(close: pd.DataFrame, open_: pd.DataFrame) -> pd.DataFrame:
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
    return (volume / volume.rolling(period).mean()).iloc[-1]


def compute_price_vs_ma20(close: pd.DataFrame) -> pd.Series:
    ma20 = close.rolling(20).mean()
    return ((close - ma20) / ma20 * 100).iloc[-1]


def compute_trend_50d(close: pd.DataFrame) -> pd.Series:
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

    mr_rank = signal_mean_reversion(close).iloc[-1]
    vol_rank = signal_volatility(close).iloc[-1]
    mom_rank = signal_momentum(close, open_).iloc[-1]

    rsi = compute_rsi(close)
    atr = compute_atr(high, low, close)
    vsurge = compute_volume_surge(volume)
    pma20 = compute_price_vs_ma20(close)
    t50d = compute_trend_50d(close)
    ret1d = compute_ret_1d(close)
    ret5d = compute_ret_5d(close)
    latest_close = close.iloc[-1]

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
