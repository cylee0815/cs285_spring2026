"""Reward-formula tests for the logged-reward switch.

These guard the spec-compliant reward signal:

    r_t = log(1 + w_t . R_{t+1} - lambda * ||w_t - w_{t-1}||_1)

with a clamp at -0.9999 to keep the result finite under catastrophic loss.
"""

from __future__ import annotations

import numpy as np

from core.envs.portfolio_env import PortfolioEnv


def _env_with_constant_returns(returns_row: np.ndarray, lam: float = 0.0) -> PortfolioEnv:
    """Build a 3-step env where forward_returns[t] equals ``returns_row`` for all t."""
    n_assets = returns_row.shape[0]
    n_steps = 3
    features = np.zeros((n_steps, 4), dtype=np.float32)
    returns = np.tile(returns_row.astype(np.float32), (n_steps, 1))
    return PortfolioEnv(
        features=features,
        forward_returns=returns,
        transaction_cost_lambda=lam,
    )


def test_logged_not_linear_zero_turnover():
    """w=[1,0,...], R=[0.01,0,...], zero turnover, lam=0.001 → log(1.01), not 0.01."""
    n_assets = 8
    returns_row = np.zeros(n_assets, dtype=np.float32)
    returns_row[0] = 0.01
    env = _env_with_constant_returns(returns_row, lam=0.001)
    env.reset(seed=0)

    # Force prev_weights == action to zero out turnover.
    w = np.zeros(n_assets, dtype=np.float32)
    w[0] = 1.0
    env.prev_weights = w.astype(np.float64).copy()

    _obs, reward, _term, _trunc, _info = env.step(w)

    expected = np.log1p(0.01)  # ≈ 0.00995033
    assert reward != 0.01, "Reward must not be linear (0.01)."
    np.testing.assert_allclose(reward, expected, atol=1e-7)


def test_turnover_strictly_decreases_reward():
    """At identical forward return, an action with turnover < zero-turnover reward."""
    n_assets = 8
    returns_row = np.full(n_assets, 0.005, dtype=np.float32)  # equal-return assets
    lam = 0.01

    # Case A: zero turnover (prev_weights = action = equal-weight).
    env_a = _env_with_constant_returns(returns_row, lam=lam)
    env_a.reset(seed=0)
    w_eq = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
    env_a.prev_weights = w_eq.astype(np.float64).copy()
    _, r_no_turnover, *_ = env_a.step(w_eq)

    # Case B: same forward return on a concentrated portfolio, prev=equal-weight
    # → nonzero turnover.
    env_b = _env_with_constant_returns(returns_row, lam=lam)
    env_b.reset(seed=0)  # prev_weights defaults to equal-weight
    w_concentrated = np.zeros(n_assets, dtype=np.float32)
    w_concentrated[0] = 1.0
    _, r_with_turnover, *_ = env_b.step(w_concentrated)

    # Same gross portfolio return, but Case B pays the L1 turnover cost.
    assert r_with_turnover < r_no_turnover, (
        f"Turnover should reduce reward: {r_with_turnover} >= {r_no_turnover}"
    )


def test_catastrophic_loss_is_finite():
    """port_return ≈ -0.9999 must yield a finite (large negative) reward, not -inf/NaN."""
    n_assets = 4
    returns_row = np.full(n_assets, -0.9999, dtype=np.float32)
    env = _env_with_constant_returns(returns_row, lam=0.0)
    env.reset(seed=0)

    w = np.zeros(n_assets, dtype=np.float32)
    w[0] = 1.0
    env.prev_weights = w.astype(np.float64).copy()  # zero turnover for clarity

    _, reward, *_ = env.step(w)

    assert np.isfinite(reward), f"Reward is not finite: {reward}"
    assert reward < 0, f"Catastrophic loss should give negative reward, got {reward}"


def test_total_loss_clamped():
    """port_return = -1.0 (impossible but adversarial) must clamp to -0.9999 → finite."""
    n_assets = 4
    returns_row = np.full(n_assets, -1.5, dtype=np.float32)  # below -1
    env = _env_with_constant_returns(returns_row, lam=0.0)
    env.reset(seed=0)

    w = np.zeros(n_assets, dtype=np.float32)
    w[0] = 1.0
    env.prev_weights = w.astype(np.float64).copy()

    _, reward, *_ = env.step(w)

    assert np.isfinite(reward)
    # Clamp floor: log1p(-0.9999) ≈ -9.21
    np.testing.assert_allclose(reward, np.log1p(-0.9999), atol=1e-7)


def test_info_simple_return_is_pre_log():
    """info['portfolio_simple_return'] holds the pre-log post-cost value."""
    n_assets = 3
    returns_row = np.array([0.02, -0.01, 0.03], dtype=np.float32)
    lam = 0.005
    env = _env_with_constant_returns(returns_row, lam=lam)
    env.reset(seed=0)  # prev_weights = equal-weight

    action = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    port_ret = 0.5 * 0.02 + 0.3 * (-0.01) + 0.2 * 0.03
    turnover = abs(0.5 - 1 / 3) + abs(0.3 - 1 / 3) + abs(0.2 - 1 / 3)
    expected_simple = port_ret - lam * turnover

    np.testing.assert_allclose(info["portfolio_simple_return"], expected_simple, atol=1e-7)
    np.testing.assert_allclose(info["portfolio_log_return"], reward, atol=1e-7)
