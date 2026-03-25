"""
Data utilities for portfolio RL.
Downloads price data via yfinance, computes technical indicators.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from typing import List, Optional, Tuple


# Default universe of tickers (diversified ETF portfolio)
DEFAULT_TICKERS = [
    "SPY",   # US large cap
    "QQQ",   # US tech
    "IWM",   # US small cap
    "EFA",   # International developed
    "EEM",   # Emerging markets
    "TLT",   # Long-term treasuries
    "GLD",   # Gold
    "VNQ",   # REITs
]


def download_price_data(
    tickers: List[str],
    start: str = "2010-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """Download adjusted close prices via yfinance."""
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle MultiIndex columns that yfinance may return for multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        closes = df["Close"]
    else:
        # Single ticker: df itself is the OHLCV frame
        closes = df[["Close"]] if "Close" in df.columns else df

    # Flatten MultiIndex columns if still present after slicing
    if isinstance(closes.columns, pd.MultiIndex):
        closes.columns = closes.columns.get_level_values(-1)

    closes = closes.dropna(how="all")
    closes = closes.ffill()

    # Drop columns that are entirely NaN after forward-fill
    closes = closes.dropna(axis=1, how="all")

    return closes  # (T, n_assets)


def compute_features(
    prices: pd.DataFrame,
    window_short: int = 10,
    window_long: int = 20,
    rsi_window: int = 14,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-asset features including technical indicators.

    Returns:
        log_returns: (T, n_assets) log returns
        features: (T, n_assets, n_features) feature array
          Features per asset: [log_return, rolling_mean, rolling_std, rsi, macd, bb_pct]
    """
    # Compute log returns; replace inf/-inf with 0
    raw_lr = np.log(prices / prices.shift(1))
    raw_lr = raw_lr.replace([np.inf, -np.inf], np.nan).fillna(0)
    log_returns = raw_lr.values.astype(np.float32)

    n_assets = len(prices.columns)
    T = len(prices)
    n_features = 6
    features = np.zeros((T, n_assets, n_features), dtype=np.float32)

    for i, ticker in enumerate(prices.columns):
        close = prices[ticker]

        # Log returns
        lr = np.log(close / close.shift(1))
        lr = lr.replace([np.inf, -np.inf], np.nan).fillna(0)
        features[:, i, 0] = lr.values.astype(np.float32)

        # Rolling mean return
        roll_mean = lr.rolling(window_short).mean().fillna(0)
        features[:, i, 1] = roll_mean.values.astype(np.float32)

        # Rolling std return (volatility)
        roll_std = lr.rolling(window_long).std().fillna(0)
        features[:, i, 2] = roll_std.values.astype(np.float32)

        # RSI (normalized to [-1, 1])
        rsi = ta.momentum.RSIIndicator(close=close, window=rsi_window).rsi().fillna(50)
        features[:, i, 3] = ((rsi - 50) / 50).values.astype(np.float32)

        # MACD signal (normalized by close price)
        macd_line = ta.trend.MACD(close=close).macd().fillna(0)
        macd_norm = (macd_line / (close + 1e-8)).values.astype(np.float32)
        features[:, i, 4] = np.clip(macd_norm, -0.1, 0.1)

        # Bollinger Band %B (0=lower band, 1=upper band, normalized to [-1, 1])
        bb = ta.volatility.BollingerBands(close=close, window=window_long)
        bb_pct = bb.bollinger_pband().fillna(0.5)
        features[:, i, 5] = ((bb_pct - 0.5) * 2).values.astype(np.float32)

    # Final guard: replace any remaining NaN/inf in features
    features = np.where(np.isnan(features) | np.isinf(features), 0.0, features)

    return log_returns, features


def make_train_test_envs(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = "2010-01-01",
    end: str = "2024-01-01",
    train_ratio: float = 0.8,
    episode_length: int = 63,
    transaction_cost: float = 0.001,
    reward_type: str = "log_return",
    **env_kwargs,
):
    """
    Download data and create train/test portfolio environments.

    Returns:
        train_env: PortfolioEnv for training
        test_env: PortfolioEnv for evaluation
        metadata: dict with tickers, dates, etc.
    """
    from src.envs.portfolio_env import PortfolioEnv

    prices = download_price_data(tickers, start=start, end=end)
    # Keep only tickers that downloaded successfully
    tickers = list(prices.columns)

    log_returns, features = compute_features(prices)
    T = len(prices)
    split = int(T * train_ratio)

    # Ensure train and test splits have enough data for at least one episode
    if split < episode_length + 2:
        raise ValueError(
            f"Train split too short ({split} days) for episode_length={episode_length}. "
            "Try a wider date range or shorter episode_length."
        )
    if (T - split) < episode_length + 2:
        raise ValueError(
            f"Test split too short ({T - split} days) for episode_length={episode_length}. "
            "Try a wider date range or smaller train_ratio."
        )

    train_env = PortfolioEnv(
        price_returns=log_returns[:split],
        features=features[:split],
        episode_length=episode_length,
        transaction_cost=transaction_cost,
        reward_type=reward_type,
        **env_kwargs,
    )
    test_env = PortfolioEnv(
        price_returns=log_returns[split:],
        features=features[split:],
        episode_length=episode_length,
        transaction_cost=transaction_cost,
        reward_type=reward_type,
        **env_kwargs,
    )

    metadata = {
        "tickers": tickers,
        "n_assets": len(tickers),
        "train_start": str(prices.index[0].date()),
        "train_end": str(prices.index[split - 1].date()),
        "test_start": str(prices.index[split].date()),
        "test_end": str(prices.index[-1].date()),
        "T_train": split,
        "T_test": T - split,
    }
    return train_env, test_env, metadata
