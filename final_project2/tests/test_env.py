"""Tests for env.portfolio_env.PortfolioEnv.

Written BEFORE implementation (test-first). Every test uses tiny,
hand-crafted data so correctness is verifiable by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from env.portfolio_env import PortfolioEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(
    n_assets: int = 3,
    n_steps: int = 5,
    features: np.ndarray | None = None,
    returns: np.ndarray | None = None,
    transaction_cost_lambda: float = 0.0,
) -> PortfolioEnv:
    """Build an env with deterministic toy data."""
    if features is None:
        features = np.arange(n_steps * 4, dtype=np.float32).reshape(n_steps, 4)
    if returns is None:
        returns = np.full((n_steps, n_assets), 0.01, dtype=np.float32)
    return PortfolioEnv(
        features=features,
        forward_returns=returns,
        transaction_cost_lambda=transaction_cost_lambda,
    )


# ---------------------------------------------------------------------------
# 1. Deterministic reset
# ---------------------------------------------------------------------------

class TestDeterministicReset:
    def test_same_seed_same_state(self):
        env = _make_env()
        obs1, info1 = env.reset(seed=42)
        env.step(np.array([0.5, 0.3, 0.2], dtype=np.float32))
        obs2, info2 = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_clears_weights(self):
        """After reset, previous weights should be equal-weight."""
        env = _make_env(n_assets=3)
        env.reset(seed=0)
        # Step with non-uniform weights, then reset
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        env.reset(seed=0)
        # w_{t-1} should be equal-weight again
        np.testing.assert_allclose(env.prev_weights, np.ones(3) / 3.0)


# ---------------------------------------------------------------------------
# 2. Reward correctness
# ---------------------------------------------------------------------------

class TestRewardCorrectness:
    def test_simple_reward_no_cost(self):
        """r_t = w_t . R_{t+1} with zero transaction cost."""
        returns = np.array([
            [0.02, -0.01, 0.03],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ], dtype=np.float32)
        env = _make_env(n_assets=3, n_steps=3, returns=returns,
                        transaction_cost_lambda=0.0)
        env.reset(seed=0)
        action = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        _obs, reward, _term, _trunc, _info = env.step(action)
        # portfolio return = 0.5*0.02 + 0.3*(-0.01) + 0.2*0.03 = 0.013
        expected = 0.5 * 0.02 + 0.3 * (-0.01) + 0.2 * 0.03
        np.testing.assert_allclose(reward, expected, atol=1e-7)

    def test_reward_with_transaction_cost(self):
        """r_t = w_t . R_{t+1} - lambda * ||w_t - w_{t-1}||_1."""
        returns = np.array([
            [0.02, -0.01, 0.03],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ], dtype=np.float32)
        lam = 0.005
        env = _make_env(n_assets=3, n_steps=3, returns=returns,
                        transaction_cost_lambda=lam)
        env.reset(seed=0)  # prev_weights = [1/3, 1/3, 1/3]
        action = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        _obs, reward, _term, _trunc, _info = env.step(action)

        port_ret = 0.5 * 0.02 + 0.3 * (-0.01) + 0.2 * 0.03
        turnover = (abs(0.5 - 1/3) + abs(0.3 - 1/3) + abs(0.2 - 1/3))
        expected = port_ret - lam * turnover
        np.testing.assert_allclose(reward, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. Turnover correctness
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_turnover_l1(self):
        """Explicit L1 distance check across two steps."""
        returns = np.full((4, 2), 0.01, dtype=np.float32)
        lam = 0.01
        env = _make_env(n_assets=2, n_steps=4, returns=returns,
                        transaction_cost_lambda=lam)
        env.reset(seed=0)  # prev_weights = [0.5, 0.5]

        w1 = np.array([0.8, 0.2], dtype=np.float32)
        _, r1, *_ = env.step(w1)
        turnover1 = abs(0.8 - 0.5) + abs(0.2 - 0.5)
        port_ret1 = 0.8 * 0.01 + 0.2 * 0.01
        np.testing.assert_allclose(r1, port_ret1 - lam * turnover1, atol=1e-7)

        w2 = np.array([0.3, 0.7], dtype=np.float32)
        _, r2, *_ = env.step(w2)
        turnover2 = abs(0.3 - 0.8) + abs(0.7 - 0.2)
        port_ret2 = 0.3 * 0.01 + 0.7 * 0.01
        np.testing.assert_allclose(r2, port_ret2 - lam * turnover2, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Simplex projection
# ---------------------------------------------------------------------------

class TestSimplexProjection:
    def test_negative_weights_projected(self):
        """Negative action components must be clipped to simplex."""
        env = _make_env(n_assets=3)
        env.reset(seed=0)
        action = np.array([-1.0, 2.0, 0.5], dtype=np.float32)
        obs, *_ = env.step(action)
        # After projection, weights must be on simplex
        w = env.prev_weights
        assert np.all(w >= -1e-7), f"Negative weight found: {w}"
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_zero_action_projected(self):
        """All-zero action should project to equal weight."""
        env = _make_env(n_assets=4)
        env.reset(seed=0)
        action = np.zeros(4, dtype=np.float32)
        env.step(action)
        np.testing.assert_allclose(env.prev_weights, np.ones(4) / 4.0, atol=1e-6)

    def test_already_valid_unchanged(self):
        """Valid simplex action should pass through unchanged."""
        env = _make_env(n_assets=3)
        env.reset(seed=0)
        action = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        env.step(action)
        np.testing.assert_allclose(env.prev_weights, action, atol=1e-6)

    def test_unnormalized_positive_scaled(self):
        """Positive but unnormalized action should be rescaled."""
        env = _make_env(n_assets=3)
        env.reset(seed=0)
        action = np.array([2.0, 3.0, 5.0], dtype=np.float32)
        env.step(action)
        expected = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(env.prev_weights, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Done condition
# ---------------------------------------------------------------------------

class TestDoneCondition:
    def test_terminates_at_boundary(self):
        """Episode must end after exactly n_steps - 1 steps."""
        n_steps = 5
        env = _make_env(n_assets=2, n_steps=n_steps)
        env.reset(seed=0)
        action = np.array([0.5, 0.5], dtype=np.float32)

        for i in range(n_steps - 2):
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated, f"Terminated too early at step {i}"
            assert not truncated

        # Last valid step
        _, _, terminated, truncated, _ = env.step(action)
        assert terminated or truncated, "Should be done at final step"

    def test_step_after_done_raises(self):
        """Stepping after episode end should raise."""
        env = _make_env(n_assets=2, n_steps=2)
        env.reset(seed=0)
        action = np.array([0.5, 0.5], dtype=np.float32)
        env.step(action)  # this is the only step (n_steps=2, so step 0->done)
        with pytest.raises(RuntimeError):
            env.step(action)


# ---------------------------------------------------------------------------
# 6. w_{t-1} tracking
# ---------------------------------------------------------------------------

class TestWeightTracking:
    def test_prev_weights_update(self):
        """prev_weights must reflect the last action taken."""
        env = _make_env(n_assets=3, n_steps=4)
        env.reset(seed=0)

        w1 = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        env.step(w1)
        np.testing.assert_allclose(env.prev_weights, w1, atol=1e-6)

        w2 = np.array([0.1, 0.1, 0.8], dtype=np.float32)
        env.step(w2)
        np.testing.assert_allclose(env.prev_weights, w2, atol=1e-6)

    def test_prev_weights_after_reset(self):
        """After reset, prev_weights must be equal-weight."""
        env = _make_env(n_assets=4)
        env.reset(seed=0)
        np.testing.assert_allclose(env.prev_weights, np.ones(4) / 4, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Golden trajectory
# ---------------------------------------------------------------------------

class TestGoldenTrajectory:
    def test_fixed_trajectory(self):
        """Full episode with hand-computed reward sequence."""
        features = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ], dtype=np.float32)
        returns = np.array([
            [0.02, -0.01],
            [0.01, 0.03],
            [-0.02, 0.01],
            [0.00, 0.00],  # unused — last step
        ], dtype=np.float32)
        lam = 0.001
        env = PortfolioEnv(
            features=features,
            forward_returns=returns,
            transaction_cost_lambda=lam,
        )
        env.reset(seed=0)
        # prev_weights = [0.5, 0.5]

        # Step 0: action=[0.6, 0.4], returns=[0.02, -0.01]
        w0 = np.array([0.6, 0.4], dtype=np.float32)
        _, r0, term0, *_ = env.step(w0)
        port0 = 0.6 * 0.02 + 0.4 * (-0.01)
        tc0 = lam * (abs(0.6 - 0.5) + abs(0.4 - 0.5))
        np.testing.assert_allclose(r0, port0 - tc0, atol=1e-7)
        assert not term0

        # Step 1: action=[0.3, 0.7], returns=[0.01, 0.03]
        w1 = np.array([0.3, 0.7], dtype=np.float32)
        _, r1, term1, *_ = env.step(w1)
        port1 = 0.3 * 0.01 + 0.7 * 0.03
        tc1 = lam * (abs(0.3 - 0.6) + abs(0.7 - 0.4))
        np.testing.assert_allclose(r1, port1 - tc1, atol=1e-7)
        assert not term1

        # Step 2: action=[0.5, 0.5], returns=[-0.02, 0.01]
        w2 = np.array([0.5, 0.5], dtype=np.float32)
        _, r2, term2, *_ = env.step(w2)
        port2 = 0.5 * (-0.02) + 0.5 * 0.01
        tc2 = lam * (abs(0.5 - 0.3) + abs(0.5 - 0.7))
        np.testing.assert_allclose(r2, port2 - tc2, atol=1e-7)
        assert term2  # 4 rows → 3 steps → done after step 2


# ---------------------------------------------------------------------------
# 8. Observation space shape
# ---------------------------------------------------------------------------

class TestObservationShape:
    def test_obs_shape_matches_features(self):
        """Observation should have same dim as feature row."""
        features = np.random.randn(10, 7).astype(np.float32)
        returns = np.random.randn(10, 3).astype(np.float32)
        env = PortfolioEnv(features=features, forward_returns=returns)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (7,)


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_mismatched_lengths_raises(self):
        features = np.zeros((10, 4), dtype=np.float32)
        returns = np.zeros((8, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="same number"):
            PortfolioEnv(features=features, forward_returns=returns)

    def test_too_short_raises(self):
        features = np.zeros((1, 4), dtype=np.float32)
        returns = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="at least 2"):
            PortfolioEnv(features=features, forward_returns=returns)
