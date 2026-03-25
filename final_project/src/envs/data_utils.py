"""
Data utilities for portfolio RL.
Downloads price data via yfinance, computes technical indicators.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from typing import List, Optional, Tuple


# Default 8-asset "Orthogonal Risk Premia" ETF universe.
# Each asset represents a distinct macro risk factor, chosen to stress-test
# the O2O hypothesis: 2022 caused SPY/TLT/HYG to crash simultaneously,
# breaking the stock/bond/credit correlations the offline agent learned on.
# Data begins 2008 due to UUP (USD ETF) and DBC (commodity ETF) launch dates.
DEFAULT_TICKERS = [
    "SPY",   # US equity (S&P 500)
    "EEM",   # Emerging market equity
    "TLT",   # Long-duration Treasury (duration risk)
    "HYG",   # High-yield corporate bonds (credit risk)
    "DBC",   # Diversified commodities
    "GLD",   # Gold (inflation/safe-haven)
    "UUP",   # US Dollar Index (currency risk)
    "SHY",   # Short-term Treasury (cash proxy / safe haven)
]

# Mutual fund proxies mapping 1:1 to the 8 DEFAULT_TICKERS above.
# Allows yfinance to pull data back to the 1990s (Dot-Com bubble coverage)
# while maintaining the exact same 8-dimensional action space.
MUTUAL_FUND_TICKERS = [
    "VFINX",  # S&P 500       → SPY proxy  (data from 1976)
    "VEIEX",  # Emerging Mkt  → EEM proxy  (data from 1994)
    "VUSTX",  # Long Treasury → TLT proxy  (data from 1986)
    "VWEHX",  # High-Yield    → HYG proxy  (data from 1978)
    "PCRIX",  # Commodities   → DBC proxy  (data from 1997)
    "USERX",  # US Gold       → GLD proxy  (data from 1974)
    "VMFXX",  # Money Market  → UUP proxy  (data from 1981, USD/cash)
    "VFISX",  # Short Treasury→ SHY proxy  (data from 1991)
]


def download_price_data(
    tickers: List[str],
    start: str = "2008-01-01",
    end: str = "2026-03-31",
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


def make_train_test_envs_finrl(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = "2008-01-01",
    end: str = "2026-03-31",
    train_ratio: float = 0.8,
    time_window: int = 20,
    transaction_cost: float = 0.001,
    accept_portfolio_weights: bool = False,
    **kwargs,
):
    """
    Convenience wrapper: create FinRL-backed train/test environments.
    Delegates to src.envs.finrl_wrapper.make_finrl_envs.
    """
    from src.envs.finrl_wrapper import make_finrl_envs
    return make_finrl_envs(
        tickers=tickers,
        start=start,
        end=end,
        train_ratio=train_ratio,
        time_window=time_window,
        commission_fee_pct=transaction_cost,
        accept_portfolio_weights=accept_portfolio_weights,
        **kwargs,
    )


def make_train_test_envs(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = "2008-01-01",
    end: str = "2026-03-31",
    train_ratio: float = 0.8,
    episode_length: int = 63,
    transaction_cost: float = 0.001,
    reward_type: str = "log_return",
    use_macro: bool = False,
    use_sentiment: bool = False,
    fred_api_key: Optional[str] = None,
    **env_kwargs,
):
    """
    Download data and create train/test portfolio environments.

    Returns:
        train_env: PortfolioEnv for training
        test_env: PortfolioEnv for evaluation
        metadata: dict with tickers, dates, obs_dim, etc.
    """
    from src.envs.portfolio_env import PortfolioEnv

    prices = download_price_data(tickers, start=start, end=end)
    tickers = list(prices.columns)

    log_returns, features = compute_features(prices)
    T = len(prices)
    split = int(T * train_ratio)

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

    macro_arr = None
    if use_macro:
        from src.envs.macro_features import download_fred_macro, align_macro_to_prices
        macro_df = download_fred_macro(start, end, api_key=fred_api_key)
        macro_arr = align_macro_to_prices(macro_df, prices.index)

    sentiment_arr = None
    if use_sentiment:
        from src.envs.sentiment_features import load_sentiment
        sentiment_arr = load_sentiment(prices.index, start, end, tickers=tickers)

    def _make_env(s, e):
        return PortfolioEnv(
            price_returns=log_returns[s:e],
            features=features[s:e],
            episode_length=episode_length,
            transaction_cost=transaction_cost,
            reward_type=reward_type,
            macro_features=macro_arr[s:e] if macro_arr is not None else None,
            sentiment_features=sentiment_arr[s:e] if sentiment_arr is not None else None,
            **env_kwargs,
        )

    train_env = _make_env(0, split)
    test_env  = _make_env(split, T)

    metadata = {
        "tickers": tickers,
        "n_assets": len(tickers),
        "obs_dim": train_env.observation_space.shape[0],
        "use_macro": use_macro,
        "use_sentiment": use_sentiment,
        "train_start": str(prices.index[0].date()),
        "train_end": str(prices.index[split - 1].date()),
        "test_start": str(prices.index[split].date()),
        "test_end": str(prices.index[-1].date()),
        "T_train": split,
        "T_test": T - split,
    }
    return train_env, test_env, metadata


def make_train_val_test_envs(
    use_mutual_funds: bool = False,
    tickers: Optional[List[str]] = None,
    train_start: str = "2008-01-01",
    train_end: str = "2020-12-31",
    val_start: str = "2021-01-01",
    val_end: str = "2021-12-31",
    test_start: str = "2022-01-01",
    test_end: str = "2026-03-31",
    episode_length: int = 63,
    transaction_cost: float = 0.001,
    reward_type: str = "log_return",
    # Multimodal feature flags
    use_macro: bool = False,
    use_sentiment: bool = False,
    use_alpaca_embeddings: bool = False,
    fred_api_key: Optional[str] = None,
    **env_kwargs,
):
    """
    Create three chronologically separated environments for the O2O pipeline.

    Splits are aligned to the proposal's regime-shift experiment:
      - Train  (offline dataset): 2005–2020  — captures GFC (2008) and COVID crash (2020)
      - Val    (hyperparameter):  2021        — buffer year; used strictly for HP tuning
      - Test   (online O2O):      2022+       — stock/bond correlation breaks; pure OOD regime

    Args:
        use_mutual_funds: if True, use MUTUAL_FUND_TICKERS (history to 1990s).
        tickers: explicit ticker list (overrides use_mutual_funds).
        use_macro: if True, append 8 FRED macroeconomic features to each observation.
            Requires FRED_API_KEY env var (free key from https://fred.stlouisfed.org).
            Falls back to zeros if unavailable so training still works.
        use_sentiment: if True, append SF Fed Daily News Sentiment Index (scalar).
            Downloads automatically to data/dnsi.xlsx on first call.
        use_alpaca_embeddings: if True, also append Alpaca News sentence embeddings (384-d).
            Requires precomputed cache; see sentiment_features.AlpacaNewsEmbeddings.
        fred_api_key: override for FRED API key (default: reads FRED_API_KEY env var).

    Returns:
        train_env, val_env, test_env: PortfolioEnv instances
        metadata: dict with dates, tickers, split sizes, obs_dim
    """
    from src.envs.portfolio_env import PortfolioEnv

    if tickers is None:
        tickers = MUTUAL_FUND_TICKERS if use_mutual_funds else DEFAULT_TICKERS

    # Single download across the full date range
    prices = download_price_data(tickers, start=train_start, end=test_end)
    tickers = list(prices.columns)  # some may have failed to download

    log_returns, features = compute_features(prices)

    # ── Optional macro features (FRED) ────────────────────────────────────────
    macro_arr = None
    if use_macro:
        from src.envs.macro_features import download_fred_macro, align_macro_to_prices
        macro_df = download_fred_macro(train_start, test_end, api_key=fred_api_key)
        macro_arr = align_macro_to_prices(macro_df, prices.index)  # (T, 8)

    # ── Optional sentiment features (SF Fed DNSI + Alpaca) ────────────────────
    sentiment_arr = None
    if use_sentiment:
        from src.envs.sentiment_features import load_sentiment
        sentiment_arr = load_sentiment(
            prices.index, train_start, test_end,
            tickers=tickers,
            use_alpaca_embeddings=use_alpaca_embeddings,
        )  # (T, 1) or (T, 1+384)

    def _date_slice(start_str: str, end_str: str):
        mask = (prices.index >= pd.Timestamp(start_str)) & (prices.index <= pd.Timestamp(end_str))
        idx = np.where(mask)[0]
        if len(idx) < episode_length + 2:
            raise ValueError(
                f"Split {start_str}–{end_str} has only {len(idx)} trading days "
                f"(need at least {episode_length + 2})."
            )
        return idx[0], idx[-1] + 1  # [start, end) slice indices

    tr_s, tr_e = _date_slice(train_start, train_end)
    va_s, va_e = _date_slice(val_start, val_end)
    te_s, te_e = _date_slice(test_start, test_end)

    def _make_env(s, e):
        return PortfolioEnv(
            price_returns=log_returns[s:e],
            features=features[s:e],
            episode_length=episode_length,
            transaction_cost=transaction_cost,
            reward_type=reward_type,
            macro_features=macro_arr[s:e] if macro_arr is not None else None,
            sentiment_features=sentiment_arr[s:e] if sentiment_arr is not None else None,
            **env_kwargs,
        )

    train_env = _make_env(tr_s, tr_e)
    val_env   = _make_env(va_s, va_e)
    test_env  = _make_env(te_s, te_e)

    n_macro = macro_arr.shape[1] if macro_arr is not None else 0
    n_sentiment = sentiment_arr.shape[1] if sentiment_arr is not None else 0

    metadata = {
        "tickers": tickers,
        "n_assets": len(tickers),
        "use_mutual_funds": use_mutual_funds,
        "use_macro": use_macro,
        "use_sentiment": use_sentiment,
        "n_macro_features": n_macro,
        "n_sentiment_features": n_sentiment,
        "obs_dim": train_env.observation_space.shape[0],
        "train_start": str(prices.index[tr_s].date()),
        "train_end":   str(prices.index[tr_e - 1].date()),
        "val_start":   str(prices.index[va_s].date()),
        "val_end":     str(prices.index[va_e - 1].date()),
        "test_start":  str(prices.index[te_s].date()),
        "test_end":    str(prices.index[te_e - 1].date()),
        "T_train": tr_e - tr_s,
        "T_val":   va_e - va_s,
        "T_test":  te_e - te_s,
    }
    return train_env, val_env, test_env, metadata
