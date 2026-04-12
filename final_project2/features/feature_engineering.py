"""Causal per-asset feature engineering pipeline.

This module turns a ``(T, N_assets)`` adjusted-close price frame into a
``(T', F)`` feature matrix + ``(T', N_assets)`` forward-return matrix for
RL training. Every feature is computed using only information available at
or before timestep ``t`` — the single non-negotiable invariant for
financial ML. ``tests/test_features.py`` codifies this with a dedicated
causality check that perturbs future prices and asserts historical
features are unchanged.

Feature layout (flattened ``features[t, :]``, asset-major)
----------------------------------------------------------
For ``N_assets`` tickers, ``k = lookback_returns``, ``M`` MA windows, and
``P`` momentum windows, the feature row is the concatenation of:

1. **Lagged log returns** — ``N_assets * k`` entries, laid out as
   ``[asset0_lag0, asset0_lag1, ..., asset0_lag{k-1},
      asset1_lag0, ..., asset{N-1}_lag{k-1}]``.
   ``lag0`` is the most recent log return ``log(P_t / P_{t-1})``.
2. **Rolling realized volatility** — ``N_assets`` entries (population std
   of log returns over ``volatility_window`` days).
3. **MA ratio features** — for each MA window ``w``, ``N_assets`` entries
   equal to ``P_t / MA_w(t) − 1``.
4. **Momentum features** (optional) — for each momentum window ``w``,
   ``N_assets`` entries equal to the sum of log returns over the last
   ``w`` days.

Forward-return semantics
------------------------
``forward_returns[t, i]`` is the **simple** return
``(P_{t+1, i} − P_{t, i}) / P_{t, i}`` — i.e., the asset-level return
*after* the agent observes ``features[t]`` and chooses an action. The
environment multiplies it by the action weights to compute the reward at
step ``t``. This convention aligns with the MDP in ``CLAUDE.md``:
``r_t = log(1 + w_t^T R_{t+1}) − λ‖Δw‖₁``.

Warmup
------
The full-length feature matrix has NaNs in early rows while rolling
windows fill. We automatically trim those rows, and we also drop the final
row (since ``forward_returns[T-1]`` is undefined — there is no ``P_T``).
The returned :class:`FeatureBundle` is thus guaranteed to contain no NaNs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.config import FeatureConfig

__all__ = [
    "FeatureBundle",
    "build_features",
    "build_features_from_config",
]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureBundle:
    """Output of :func:`build_features`.

    Attributes
    ----------
    features:
        ``(T', F)`` float32 matrix of causal features. Guaranteed NaN-free.
    forward_returns:
        ``(T', N_assets)`` float32 matrix where ``[t, i]`` is the one-step
        simple return of asset ``i`` from date ``dates[t]`` to the next
        trading day. Consumed by the environment's reward function.
    dates:
        ``DatetimeIndex`` of length ``T'`` giving the date associated with
        each feature row (i.e., the day on which the agent acts).
    feature_names:
        List of length ``F`` naming each feature column. Handy for
        debugging and for feature-importance post-hoc analysis.
    tickers:
        List of asset tickers in the order they appear in
        ``forward_returns`` and as the per-asset suffixes of feature names.
    """

    features: np.ndarray
    forward_returns: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: list[str]
    tickers: list[str]

    @property
    def num_steps(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_assets(self) -> int:
        return int(self.forward_returns.shape[1])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(
    prices: pd.DataFrame,
    *,
    lookback_returns: int,
    volatility_window: int,
    ma_windows: Sequence[int],
    include_momentum: bool,
    momentum_windows: Sequence[int],
) -> FeatureBundle:
    """Compute causal market features and forward returns.

    Parameters
    ----------
    prices:
        Adjusted close prices, shape ``(T, N_assets)``. Index must be a
        ``DatetimeIndex`` sorted ascending. **Must contain no NaNs** —
        handle missing data (e.g., via inner-join on trading days) before
        calling this function.
    lookback_returns:
        Number of most-recent log returns to include per asset. Must be ≥ 1.
    volatility_window:
        Rolling window for per-asset realized volatility. Must be ≥ 2.
    ma_windows:
        Tuple of moving-average window lengths. For each window ``w`` we
        emit ``(P_t / MA_w(t)) − 1`` per asset.
    include_momentum:
        If True, include cumulative-log-return momentum features.
    momentum_windows:
        Tuple of momentum window lengths (ignored if
        ``include_momentum=False``). For each window ``w`` we emit
        ``sum(log_returns[t-w+1 : t+1])`` per asset.

    Returns
    -------
    :class:`FeatureBundle` with warmup rows and the final row trimmed so
    downstream code can iterate rows without special-casing NaNs or the
    missing ``forward_returns[T-1]``.

    Raises
    ------
    TypeError
        If ``prices`` is not a :class:`pandas.DataFrame`.
    ValueError
        If ``prices`` contains NaN values or is shorter than the minimum
        length required by the warmup windows.
    """
    _validate_inputs(
        prices=prices,
        lookback_returns=lookback_returns,
        volatility_window=volatility_window,
        ma_windows=ma_windows,
        include_momentum=include_momentum,
        momentum_windows=momentum_windows,
    )

    prices_arr = prices.to_numpy(dtype=np.float64)
    T, n_assets = prices_arr.shape
    tickers = list(prices.columns.astype(str))

    # --- Log returns: log(P_t / P_{t-1}) -----------------------------------
    # Row 0 is NaN by definition (no prior price). We keep the full-length
    # matrix here; warmup trimming happens after all feature blocks are
    # assembled so we don't repeatedly recompute indexing offsets.
    log_returns = np.full_like(prices_arr, np.nan)
    log_returns[1:] = np.log(prices_arr[1:] / prices_arr[:-1])

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    # --- Block 1: lagged log returns, asset-major layout -------------------
    # lagged[t, i, j] = log_returns[t-j, i] for j = 0..k-1.
    lagged = _lag_matrix(log_returns, lookback_returns)  # (T, N, k)
    # Reshape to (T, N*k). `reshape` on a C-contiguous (T, N, k) array
    # yields exactly the asset-major layout documented in the module docstring.
    lagged_flat = lagged.reshape(T, n_assets * lookback_returns)
    feature_blocks.append(lagged_flat)
    feature_names.extend(
        f"logret_{tick}_lag{j}"
        for tick in tickers
        for j in range(lookback_returns)
    )

    # --- Block 2: realized volatility (per asset) --------------------------
    vol = _rolling_std(log_returns, volatility_window)  # (T, N)
    feature_blocks.append(vol)
    feature_names.extend(f"vol{volatility_window}_{tick}" for tick in tickers)

    # --- Block 3: moving-average ratios, one sub-block per window ---------
    for w in ma_windows:
        ma = _rolling_mean(prices_arr, w)  # (T, N)
        ratio = prices_arr / ma - 1.0
        feature_blocks.append(ratio)
        feature_names.extend(f"ma{w}_ratio_{tick}" for tick in tickers)

    # --- Block 4: momentum (cumulative log returns) -----------------------
    if include_momentum:
        for w in momentum_windows:
            mom = _rolling_sum(log_returns, w)  # (T, N)
            feature_blocks.append(mom)
            feature_names.extend(f"mom{w}_{tick}" for tick in tickers)

    features = np.concatenate(feature_blocks, axis=1)  # (T, F)

    # --- Forward returns: (P_{t+1} - P_t) / P_t ---------------------------
    # The last row is undefined. We fill it with NaN and trim before return.
    forward_returns = np.full_like(prices_arr, np.nan)
    forward_returns[:-1] = prices_arr[1:] / prices_arr[:-1] - 1.0

    # --- Warmup trim: first row without any NaN in features --------------
    first_valid = _first_all_valid_row(features)
    # Also stop one row before the end so forward_returns has no NaN.
    end_exclusive = T - 1
    if first_valid >= end_exclusive:
        raise ValueError(
            f"Warmup ({first_valid} rows) exhausts the price series "
            f"({T} rows). Lengthen the data window or shorten the "
            f"lookback / MA / momentum windows."
        )

    features_trim = np.ascontiguousarray(
        features[first_valid:end_exclusive], dtype=np.float32
    )
    forward_returns_trim = np.ascontiguousarray(
        forward_returns[first_valid:end_exclusive], dtype=np.float32
    )
    dates_trim = prices.index[first_valid:end_exclusive]

    return FeatureBundle(
        features=features_trim,
        forward_returns=forward_returns_trim,
        dates=dates_trim,
        feature_names=feature_names,
        tickers=tickers,
    )


def build_features_from_config(
    prices: pd.DataFrame, cfg: FeatureConfig
) -> FeatureBundle:
    """Thin adapter that plucks the relevant fields from a :class:`FeatureConfig`.

    ``cfg.normalize`` is intentionally **ignored** here: normalization is
    the responsibility of :class:`features.normalization.CausalZScoreNormalizer`,
    fit on the train split only. Keeping the two concerns separated means
    the same feature bundle can be renormalized under different statistics
    (e.g., for ablations) without recomputing the raw features.
    """
    return build_features(
        prices,
        lookback_returns=cfg.lookback_returns,
        volatility_window=cfg.volatility_window,
        ma_windows=tuple(cfg.ma_windows),
        include_momentum=cfg.include_momentum,
        momentum_windows=tuple(cfg.momentum_windows),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    *,
    prices: pd.DataFrame,
    lookback_returns: int,
    volatility_window: int,
    ma_windows: Sequence[int],
    include_momentum: bool,
    momentum_windows: Sequence[int],
) -> None:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            f"prices must be a pandas DataFrame, got {type(prices).__name__}"
        )
    if prices.isna().any().any():
        raise ValueError(
            "prices contains NaN values — align and inner-join across "
            "tickers before building features."
        )
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(
            f"prices must have a DatetimeIndex, got {type(prices.index).__name__}"
        )
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be sorted ascending by date.")
    if lookback_returns < 1:
        raise ValueError(f"lookback_returns must be ≥ 1, got {lookback_returns}")
    if volatility_window < 2:
        raise ValueError(f"volatility_window must be ≥ 2, got {volatility_window}")
    if any(w < 1 for w in ma_windows):
        raise ValueError(f"all ma_windows must be ≥ 1, got {tuple(ma_windows)}")
    if include_momentum and any(w < 1 for w in momentum_windows):
        raise ValueError(
            f"all momentum_windows must be ≥ 1, got {tuple(momentum_windows)}"
        )

    # Minimum required length: longest feature window plus 2 (one for the
    # NaN first log-return row, one so we can emit at least a single valid
    # (features, forward_return) pair).
    required = 2 + max(
        [lookback_returns, volatility_window, *ma_windows]
        + (list(momentum_windows) if include_momentum else [])
    )
    if len(prices) < required:
        raise ValueError(
            f"prices has {len(prices)} rows but feature pipeline needs at "
            f"least {required} rows given the configured windows."
        )


def _lag_matrix(values: np.ndarray, k: int) -> np.ndarray:
    """Build a ``(T, N, k)`` lag tensor from a ``(T, N)`` values matrix.

    ``out[t, i, j] = values[t - j, i]`` when ``t ≥ j`` and NaN otherwise.
    Written in an explicit Python loop over ``k`` because ``k`` is small
    (typically 20) and a vectorized stride trick would only obscure intent.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")
    T, n = values.shape
    out = np.full((T, n, k), np.nan, dtype=values.dtype)
    for j in range(k):
        # Row t gets values[t - j] for t ≥ j.
        out[j:, :, j] = values[: T - j] if j > 0 else values
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Strict rolling mean (``min_periods=window``) using pandas."""
    return (
        pd.DataFrame(values)
        .rolling(window=window, min_periods=window)
        .mean()
        .to_numpy()
    )


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling population std (``ddof=0``) ignoring warmup periods.

    Pandas ``.std(ddof=0)`` returns NaN while the window is still filling
    (because ``min_periods=window``), which is exactly what the warmup
    trimmer downstream expects.
    """
    return (
        pd.DataFrame(values)
        .rolling(window=window, min_periods=window)
        .std(ddof=0)
        .to_numpy()
    )


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Strict rolling sum."""
    return (
        pd.DataFrame(values)
        .rolling(window=window, min_periods=window)
        .sum()
        .to_numpy()
    )


def _first_all_valid_row(features: np.ndarray) -> int:
    """Index of the earliest row such that every subsequent row is NaN-free.

    We don't assume NaNs only appear at the start: a downstream block
    (e.g., a buggy MA computation) might produce NaNs later, and we want
    the trim to catch that case loudly rather than silently handing back
    a partially-clean matrix.
    """
    T = features.shape[0]
    any_nan_per_row = np.isnan(features).any(axis=1)
    # Scan from the top; stop at the first row where there is no NaN from
    # that row onward.
    for i in range(T):
        if not any_nan_per_row[i:].any():
            return i
    return T  # pragma: no cover — caught by length check upstream
