"""
FinRL integration for online portfolio RL.

FinRL's PortfolioOptimizationEnv provides:
  - obs: 3D array (n_features, n_stocks, time_window) — historical feature matrix
  - action: (n_stocks + 1,) weights including cash, softmaxed internally
  - reward: log portfolio return
  - commission fee models: "trf" (transaction remainder factor) or "wvm"

Two adapter concerns:
  1. Obs shape: flatten (n_features, n_stocks, time_window) → 1D for MLP/LSTM/Transformer
  2. Action format: FinRL always applies softmax internally.
       - PPO (Gaussian logits): pass raw samples directly → FinRL softmax → portfolio weights ✓
       - DirichletActor (portfolio weights w): pass log(w) → FinRL softmax → recovers w ✓
         because softmax(log(w))_i = w_i / Σ_j w_j = w_i  (since Σw = 1)

FinRLPortfolioWrapper handles both cases transparently via `accept_portfolio_weights`.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Tuple, Dict

try:
    # Import specific submodules directly — avoids FinRL's __init__.py which
    # has unconditional top-level imports of optional packages (wrds, alpaca, etc.)
    import importlib
    _env_mod = importlib.import_module(
        "finrl.meta.env_portfolio_optimization.env_portfolio_optimization"
    )
    PortfolioOptimizationEnv = _env_mod.PortfolioOptimizationEnv
    FINRL_AVAILABLE = True
except (ImportError, AttributeError):
    FINRL_AVAILABLE = False
    PortfolioOptimizationEnv = None


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FinRLPortfolioWrapper:
    """
    Adapts FinRL's PortfolioOptimizationEnv (old gym.Env) to the gymnasium
    interface expected by our agents.

    Observation: flattens (n_features, n_stocks, time_window) → 1D float32 vector.
    Action (accept_portfolio_weights=True, for DirichletActor):
        Expects pre-normalized portfolio weights. Converts w → log(w) before
        passing to FinRL so that FinRL's internal softmax recovers w exactly:
            softmax(log(w))_i = w_i / Σ_j w_j = w_i  (since Σw = 1)
    Action (accept_portfolio_weights=False, for PPO with Gaussian logits):
        Passes raw action directly — FinRL applies softmax.

    Attributes:
        observation_space: gymnasium Box (flat 1D)
        action_space: gymnasium Box matching FinRL's action dim
        episode_length: number of trading steps (used for annualised-return calc)
        n_stocks: number of risky assets (excludes cash)
    """

    def __init__(self, env, accept_portfolio_weights: bool = False):
        self._env = env
        self.accept_portfolio_weights = accept_portfolio_weights

        # Flat observation space (gymnasium Box)
        raw_shape = env.observation_space.shape  # (n_features, n_stocks, time_window)
        flat_dim = int(np.prod(raw_shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        # Action space: keep FinRL's shape but use gymnasium Box
        act_shape = env.action_space.shape
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=act_shape, dtype=np.float32
        )

        self.episode_length = getattr(env, "_episode_length", 252)
        self.n_stocks = act_shape[0] - 1  # subtract cash position

    # ------------------------------------------------------------------
    # gymnasium-compatible interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        result = self._env.reset()
        # FinRL (new_gym_api=True) returns (obs, info); old API returns obs
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return self._flatten_obs(obs), info

    def step(self, action: np.ndarray):
        env_action = self._convert_action(action)
        result = self._env.step(env_action)

        # FinRL new_gym_api=True → (obs, reward, term, trunc, info)
        # FinRL new_gym_api=False → (obs, reward, done, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False

        return self._flatten_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).flatten()

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        if self.accept_portfolio_weights:
            return np.log(np.clip(action, 1e-8, 1.0))
        return np.asarray(action, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Data download using FinRL's pipeline
# ──────────────────────────────────────────────────────────────────────────────

FINRL_TECH_INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]


def download_finrl_data(
    tickers: List[str],
    start: str = "2008-01-01",
    end: str = "2026-03-31",
    tech_indicators: Optional[List[str]] = None,
    include_turbulence: bool = True,
    include_vix: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance and compute technical indicators,
    returning a long-format DataFrame compatible with FinRL's
    PortfolioOptimizationEnv.

    Output columns: date, tic, open, high, low, close, volume, [tech indicators...]

    Note: uses yfinance directly (not FinRL's YahooDownloader, which has a
    column-naming bug with yfinance >= 0.2). Technical indicators are computed
    with the `ta` library, mirroring FinRL's FeatureEngineer output names.
    """
    import yfinance as yf
    import ta as ta_lib

    if tech_indicators is None:
        tech_indicators = FINRL_TECH_INDICATORS

    if not FINRL_AVAILABLE:
        raise ImportError("FinRL is not installed. Run: uv add finrl")

    print(f"Downloading data for FinRL env: {tickers} ({start} → {end})...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Build long-format DataFrame (one row per date per ticker)
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(raw.columns, pd.MultiIndex):
        dfs = []
        for tic in tickers:
            try:
                sub = raw.loc[:, (ohlcv, tic)]
                sub.columns = sub.columns.droplevel(1)
            except KeyError:
                continue
            sub = sub.rename(columns=str.lower)
            sub["tic"] = tic
            sub["date"] = sub.index.strftime("%Y-%m-%d")
            dfs.append(sub.reset_index(drop=True))
    else:
        sub = raw[ohlcv].rename(columns=str.lower).copy()
        sub["tic"] = tickers[0]
        sub["date"] = raw.index.strftime("%Y-%m-%d")
        dfs = [sub.reset_index(drop=True)]

    df = pd.concat(dfs, ignore_index=True)
    df = df.ffill().dropna(subset=["close"])
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    print(f"  Raw: {len(df)} rows, {df['tic'].nunique()} tickers")

    # ── Technical indicators per ticker ───────────────────────────────────
    pieces = []
    for tic, grp in df.groupby("tic"):
        grp = grp.sort_values("date").copy()
        close = grp["close"].astype(float)
        high  = grp["high"].astype(float)
        low   = grp["low"].astype(float)

        if "macd" in tech_indicators:
            grp["macd"] = ta_lib.trend.MACD(close).macd().fillna(0)
        if "rsi_30" in tech_indicators:
            grp["rsi_30"] = ta_lib.momentum.RSIIndicator(close, window=30).rsi().fillna(50)
        if "cci_30" in tech_indicators:
            grp["cci_30"] = ta_lib.trend.CCIIndicator(
                high=high, low=low, close=close, window=30).cci().fillna(0)
        if "dx_30" in tech_indicators:
            grp["dx_30"] = ta_lib.trend.ADXIndicator(
                high=high, low=low, close=close, window=30).adx().fillna(0)
        if "close_30_sma" in tech_indicators:
            grp["close_30_sma"] = close.rolling(30).mean().bfill()
        if "close_60_sma" in tech_indicators:
            grp["close_60_sma"] = close.rolling(60).mean().bfill()

        pieces.append(grp)

    df = pd.concat(pieces, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)

    # ── Turbulence index (Mahalanobis distance from historical returns) ───
    if include_turbulence:
        df = _add_turbulence(df)

    # ── VIX proxy (rolling realized vol of the first ticker) ─────────────
    if include_vix:
        df = _add_vix_proxy(df, tickers[0])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    print(f"  Final: {len(df)} rows, columns: {list(df.columns)}")
    return df


def _add_turbulence(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Turbulence = Mahalanobis distance of today's returns from historical mean."""
    pivoted = df.pivot(index="date", columns="tic", values="close")
    ret = np.log(pivoted / pivoted.shift(1)).fillna(0)
    turb = []
    for i in range(len(ret)):
        if i < lookback:
            turb.append(0.0)
        else:
            hist = ret.iloc[i - lookback:i].values
            curr = ret.iloc[i].values
            try:
                cov = np.cov(hist, rowvar=False)
                diff = curr - hist.mean(axis=0)
                t = float(diff @ np.linalg.pinv(cov) @ diff)
                turb.append(max(t, 0.0))
            except Exception:
                turb.append(0.0)
    turb_df = pd.DataFrame({"date": ret.index, "turbulence": turb})
    df = df.merge(turb_df, on="date", how="left").fillna({"turbulence": 0})
    return df


def _add_vix_proxy(df: pd.DataFrame, ref_ticker: str, window: int = 21) -> pd.DataFrame:
    """VIX proxy: annualised 21-day rolling realised vol of ref_ticker."""
    ref = df[df["tic"] == ref_ticker][["date", "close"]].copy().sort_values("date")
    ref["log_ret"] = np.log(ref["close"] / ref["close"].shift(1)).fillna(0)
    ref["vix"] = ref["log_ret"].rolling(window).std().fillna(0) * np.sqrt(252) * 100
    df = df.merge(ref[["date", "vix"]], on="date", how="left").fillna({"vix": 0})
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────────────────────

def make_finrl_envs(
    tickers: List[str],
    start: str = "2008-01-01",
    end: str = "2026-03-31",
    train_ratio: float = 0.8,
    features: Optional[List[str]] = None,
    time_window: int = 20,
    initial_amount: float = 100_000,
    commission_fee_pct: float = 0.001,
    accept_portfolio_weights: bool = False,
    tech_indicators: Optional[List[str]] = None,
    include_turbulence: bool = True,
) -> Tuple["FinRLPortfolioWrapper", "FinRLPortfolioWrapper", Dict]:
    """
    Download data via FinRL and create wrapped train/test environments.

    Args:
        tickers: list of ticker symbols
        features: FinRL feature columns to use as observation (default: close, high, low)
        time_window: historical lookback window in the observation (default: 20 days)
        accept_portfolio_weights: True for DirichletActor, False for PPO (see wrapper docstring)
        tech_indicators: technical indicator names for FeatureEngineer

    Returns:
        train_env, test_env: wrapped FinRL environments
        metadata: dict with dataset info and obs/action dimensions
    """
    if not FINRL_AVAILABLE:
        raise ImportError("FinRL is not installed. Run: uv add finrl")

    if features is None:
        features = ["close", "high", "low"]

    # Extend features with tech indicators if available
    if tech_indicators is None:
        tech_indicators = FINRL_TECH_INDICATORS
    all_features = features + [f for f in tech_indicators if f not in features]
    if include_turbulence:
        all_features = all_features + ["turbulence"]

    # Download and preprocess
    df = download_finrl_data(
        tickers=tickers,
        start=start,
        end=end,
        tech_indicators=tech_indicators,
        include_turbulence=include_turbulence,
    )

    # Filter to features that actually exist in the dataframe
    available = set(df.columns)
    all_features = [f for f in all_features if f in available]
    print(f"  Using features: {all_features}")

    # Actual tickers after download (some may have failed)
    tickers = sorted(df["tic"].unique().tolist())

    # Time-based train/test split
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * train_ratio)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    df_train = df[df["date"].isin(train_dates)].reset_index(drop=True)
    df_test = df[df["date"].isin(test_dates)].reset_index(drop=True)

    def _make_env(df_split: pd.DataFrame) -> FinRLPortfolioWrapper:
        raw_env = PortfolioOptimizationEnv(
            df=df_split,
            initial_amount=initial_amount,
            comission_fee_pct=commission_fee_pct,  # FinRL typo: one 'm'
            features=all_features,
            time_window=time_window,
            new_gym_api=True,
            normalize_df="by_previous_time",
        )
        wrapped = FinRLPortfolioWrapper(
            raw_env, accept_portfolio_weights=accept_portfolio_weights
        )
        # Inject episode_length from the number of trading days
        wrapped.episode_length = len(df_split["date"].unique()) - time_window
        return wrapped

    train_env = _make_env(df_train)
    test_env = _make_env(df_test)

    # Compute obs and action dims from the wrapped env
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]  # n_stocks + 1 (cash)

    metadata = {
        "tickers": tickers,
        "n_stocks": len(tickers),
        "n_assets": len(tickers),         # alias used by our agents
        "action_dim": action_dim,         # n_stocks + 1
        "obs_dim": obs_dim,
        "features": all_features,
        "time_window": time_window,
        "train_start": str(train_dates[0]),
        "train_end": str(train_dates[-1]),
        "test_start": str(test_dates[0]),
        "test_end": str(test_dates[-1]),
        "T_train": len(train_dates),
        "T_test": len(test_dates),
        "env_backend": "finrl",
    }
    return train_env, test_env, metadata
