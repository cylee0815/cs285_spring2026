"""Tests for ``features.feature_engineering.build_features``.

Critical properties being verified:

1. **Shapes.** Feature matrix and forward-return matrix agree on length,
   and the number of features matches the configured window sizes.
2. **Causality.** A feature at time ``t`` must be invariant under changes
   to prices at time ``> t``. This is the single most important invariant
   in financial ML — any violation produces falsely-positive backtests.
3. **Known values.** On hand-crafted price series we can analytically
   predict the content of specific feature blocks. We check those.
4. **Forward-return semantics.** ``forward_returns[t]`` must equal
   ``(P[t+1] - P[t]) / P[t]`` — that is, the return the reward at step
   ``t`` will observe after the agent acts.
5. **No NaNs after warmup.** The builder is responsible for trimming the
   warmup period so downstream RL code can assume clean inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import (
    FeatureBundle,
    build_features,
    build_features_from_config,
)
from utils.config import FeatureConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Deterministic geometric random-walk prices for 3 assets, 300 days."""
    rng = np.random.default_rng(0)
    T, N = 300, 3
    # Daily log returns ~ N(mu, sigma), different per asset.
    mu = np.array([0.0005, 0.0002, 0.0008])
    sigma = np.array([0.01, 0.005, 0.02])
    log_rets = rng.normal(loc=mu, scale=sigma, size=(T, N))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


@pytest.fixture
def default_feature_kwargs() -> dict:
    return {
        "lookback_returns": 5,
        "volatility_window": 10,
        "ma_windows": (5, 20),
        "include_momentum": True,
        "momentum_windows": (10, 20),
    }


# ---------------------------------------------------------------------------
# Basic shape / API
# ---------------------------------------------------------------------------


def test_returns_feature_bundle(synthetic_prices: pd.DataFrame, default_feature_kwargs) -> None:
    bundle = build_features(synthetic_prices, **default_feature_kwargs)
    assert isinstance(bundle, FeatureBundle)
    assert bundle.features.dtype == np.float32
    assert bundle.forward_returns.dtype == np.float32
    assert bundle.features.ndim == 2
    assert bundle.forward_returns.ndim == 2


def test_feature_count_matches_spec(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    """F = N * (k + 1 + |ma_windows| + |momentum_windows|)."""
    bundle = build_features(synthetic_prices, **default_feature_kwargs)
    n_assets = synthetic_prices.shape[1]
    expected = n_assets * (
        default_feature_kwargs["lookback_returns"]
        + 1  # volatility
        + len(default_feature_kwargs["ma_windows"])
        + len(default_feature_kwargs["momentum_windows"])
    )
    assert bundle.feature_dim == expected
    assert len(bundle.feature_names) == expected


def test_feature_names_are_unique_strings(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    bundle = build_features(synthetic_prices, **default_feature_kwargs)
    assert all(isinstance(n, str) for n in bundle.feature_names)
    assert len(set(bundle.feature_names)) == len(bundle.feature_names)


def test_feature_and_forward_return_rows_match(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    """features[t] and forward_returns[t] must describe the same timestep."""
    bundle = build_features(synthetic_prices, **default_feature_kwargs)
    assert bundle.features.shape[0] == bundle.forward_returns.shape[0]
    assert bundle.features.shape[0] == len(bundle.dates)
    assert bundle.num_steps > 0


def test_momentum_disabled_reduces_feature_count(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    kwargs = {**default_feature_kwargs, "include_momentum": False}
    bundle = build_features(synthetic_prices, **kwargs)
    n_assets = synthetic_prices.shape[1]
    expected = n_assets * (
        kwargs["lookback_returns"] + 1 + len(kwargs["ma_windows"])
    )
    assert bundle.feature_dim == expected


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_features_are_causal(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    """Modifying future prices must not change historical features.

    Strategy: build features on the full series, then build features on a
    copy whose prices from index ``t0`` onward have been replaced with
    arbitrary values. For any row ``t < t0 - max_lookahead``, the two
    feature matrices must agree exactly.
    """
    bundle_full = build_features(synthetic_prices, **default_feature_kwargs)

    t0 = 200
    perturbed = synthetic_prices.copy()
    perturbed.iloc[t0:] = perturbed.iloc[t0:] * 1.5 + 42.0

    bundle_perturbed = build_features(perturbed, **default_feature_kwargs)

    # The returned bundle trims the warmup; compute the common date window
    # that both bundles have and check features agree for dates strictly
    # before the perturbation point.
    common_dates = bundle_full.dates.intersection(bundle_perturbed.dates)
    cutoff_date = synthetic_prices.index[t0]
    common_pre = common_dates[common_dates < cutoff_date]
    assert len(common_pre) > 0

    # Find the row index in each bundle corresponding to each common pre
    # date and compare.
    full_idx = bundle_full.dates.get_indexer(common_pre)
    pert_idx = bundle_perturbed.dates.get_indexer(common_pre)
    np.testing.assert_allclose(
        bundle_full.features[full_idx],
        bundle_perturbed.features[pert_idx],
        rtol=0,
        atol=1e-6,
    )


def test_forward_return_uses_next_day_price(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    """forward_returns[t] must equal (P[t+1] - P[t]) / P[t]."""
    bundle = build_features(synthetic_prices, **default_feature_kwargs)
    prices_arr = synthetic_prices.to_numpy()
    # Map each bundle date to its original row index in the price frame.
    price_idx = synthetic_prices.index.get_indexer(bundle.dates)
    expected = (
        prices_arr[price_idx + 1] - prices_arr[price_idx]
    ) / prices_arr[price_idx]
    np.testing.assert_allclose(
        bundle.forward_returns, expected.astype(np.float32), rtol=1e-5, atol=1e-6
    )


def test_no_nan_after_warmup(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    bundle = build_features(synthetic_prices, **default_feature_kwargs)
    assert not np.isnan(bundle.features).any()
    assert not np.isnan(bundle.forward_returns).any()


def test_last_bundle_row_is_not_last_price_row(
    synthetic_prices: pd.DataFrame, default_feature_kwargs
) -> None:
    """The bundle must stop one row short of the last price (no forward ret)."""
    bundle = build_features(synthetic_prices, **default_feature_kwargs)
    last_bundle_date = bundle.dates[-1]
    last_price_date = synthetic_prices.index[-1]
    assert last_bundle_date < last_price_date


# ---------------------------------------------------------------------------
# Known-value sanity checks
# ---------------------------------------------------------------------------


def test_constant_price_gives_zero_features() -> None:
    """If prices never change, log returns are 0 → vol 0 → MA ratio 0."""
    T, N = 200, 2
    prices = pd.DataFrame(
        np.full((T, N), 100.0),
        index=pd.date_range("2021-01-01", periods=T, freq="B"),
        columns=["X", "Y"],
    )
    bundle = build_features(
        prices,
        lookback_returns=5,
        volatility_window=10,
        ma_windows=(5, 20),
        include_momentum=True,
        momentum_windows=(10, 20),
    )
    # Every feature should be exactly 0 and forward returns should be 0.
    np.testing.assert_allclose(bundle.features, 0.0, atol=1e-6)
    np.testing.assert_allclose(bundle.forward_returns, 0.0, atol=1e-6)


def test_exponential_price_gives_constant_log_returns() -> None:
    """Prices ~exp(rt) → log returns are a constant r per step, per asset."""
    T, N = 200, 2
    t = np.arange(T)
    r = np.array([0.001, 0.002])  # per-asset drift
    prices = 100.0 * np.exp(np.outer(t, r))
    df = pd.DataFrame(
        prices,
        index=pd.date_range("2021-01-01", periods=T, freq="B"),
        columns=["X", "Y"],
    )
    k = 5
    bundle = build_features(
        df,
        lookback_returns=k,
        volatility_window=10,
        ma_windows=(5,),
        include_momentum=False,
        momentum_windows=(),
    )
    # First 2*N*k entries of each row are the lagged log returns, which
    # should all be exactly ``r`` repeated per lag per asset.
    lag_block = bundle.features[:, : N * k]
    # Layout is asset-major: [asset0_lag0, asset0_lag1, ..., asset1_lag0, ...].
    for a in range(N):
        asset_lags = lag_block[:, a * k : (a + 1) * k]
        np.testing.assert_allclose(asset_lags, r[a], atol=1e-6)
    # Vol of a constant return series is 0.
    vol_block = bundle.features[:, N * k : N * k + N]
    np.testing.assert_allclose(vol_block, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_nan_prices() -> None:
    df = pd.DataFrame(
        {"A": [1.0, 2.0, float("nan"), 4.0]},
        index=pd.date_range("2022-01-01", periods=4, freq="B"),
    )
    with pytest.raises(ValueError, match="NaN"):
        build_features(
            df,
            lookback_returns=2,
            volatility_window=2,
            ma_windows=(2,),
            include_momentum=False,
            momentum_windows=(),
        )


def test_rejects_too_short_series() -> None:
    df = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0]},
        index=pd.date_range("2022-01-01", periods=3, freq="B"),
    )
    with pytest.raises(ValueError, match="at least"):
        build_features(
            df,
            lookback_returns=20,
            volatility_window=20,
            ma_windows=(60,),
            include_momentum=False,
            momentum_windows=(),
        )


# ---------------------------------------------------------------------------
# Config adapter
# ---------------------------------------------------------------------------


def test_build_features_from_config(synthetic_prices: pd.DataFrame) -> None:
    """The FeatureConfig adapter must produce the same result as the kwarg form."""
    cfg = FeatureConfig(
        lookback_returns=5,
        volatility_window=10,
        ma_windows=[5, 20],
        include_momentum=True,
        momentum_windows=[10, 20],
        normalize=True,
    )
    from_cfg = build_features_from_config(synthetic_prices, cfg)
    from_kwargs = build_features(
        synthetic_prices,
        lookback_returns=5,
        volatility_window=10,
        ma_windows=(5, 20),
        include_momentum=True,
        momentum_windows=(10, 20),
    )
    np.testing.assert_array_equal(from_cfg.features, from_kwargs.features)
    np.testing.assert_array_equal(
        from_cfg.forward_returns, from_kwargs.forward_returns
    )
