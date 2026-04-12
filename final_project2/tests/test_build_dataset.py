"""Tests for Milestone 3: behavior policies and offline dataset generation.

Written BEFORE implementation (test-first). Tests use small synthetic data
so correctness is verifiable by inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.portfolio_env import PortfolioEnv


# ---------------------------------------------------------------------------
# Helpers: tiny synthetic price / feature data
# ---------------------------------------------------------------------------

N_ASSETS = 8
TICKERS = ["SPY", "EEM", "TLT", "HYG", "DBC", "GLD", "UUP", "SHY"]


def _make_synthetic_prices(n_days: int = 200, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic price DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days, freq="B")
    # Random walk prices starting at 100
    log_returns = rng.randn(n_days, N_ASSETS) * 0.01
    log_returns[0] = 0.0
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


def _make_tiny_features_and_returns(
    n_steps: int = 50, n_assets: int = N_ASSETS, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small feature and forward-return arrays for env tests."""
    rng = np.random.RandomState(seed)
    features = rng.randn(n_steps, 10).astype(np.float32)
    forward_returns = rng.randn(n_steps, n_assets).astype(np.float32) * 0.01
    return features, forward_returns


# ===========================================================================
# TestBehaviorPolicies
# ===========================================================================

class TestBehaviorPolicies:
    """Each behavior policy must produce valid simplex weights."""

    def test_equal_weight_simplex(self):
        from policies.behavior import EqualWeightPolicy

        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        state = np.zeros(10, dtype=np.float32)
        w = policy.get_action(state)
        assert w.shape == (N_ASSETS,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-7)
        assert np.all(w >= 0)

    def test_equal_weight_deterministic(self):
        from policies.behavior import EqualWeightPolicy

        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        state = np.zeros(10, dtype=np.float32)
        w1 = policy.get_action(state)
        w2 = policy.get_action(state)
        np.testing.assert_array_equal(w1, w2)

    def test_equal_weight_values(self):
        from policies.behavior import EqualWeightPolicy

        policy = EqualWeightPolicy(n_assets=4)
        w = policy.get_action(np.zeros(5))
        np.testing.assert_allclose(w, np.array([0.25, 0.25, 0.25, 0.25]))

    def test_dirichlet_simplex(self):
        from policies.behavior import DirichletPolicy

        policy = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=42)
        state = np.zeros(10, dtype=np.float32)
        for _ in range(20):
            w = policy.get_action(state)
            assert w.shape == (N_ASSETS,)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-7)
            assert np.all(w >= 0)

    def test_dirichlet_deterministic_with_seed(self):
        from policies.behavior import DirichletPolicy

        policy1 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=42)
        policy2 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=42)
        state = np.zeros(10, dtype=np.float32)
        for _ in range(10):
            w1 = policy1.get_action(state)
            w2 = policy2.get_action(state)
            np.testing.assert_array_equal(w1, w2)

    def test_dirichlet_different_seeds_differ(self):
        from policies.behavior import DirichletPolicy

        policy1 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=0)
        policy2 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=99)
        state = np.zeros(10, dtype=np.float32)
        w1 = policy1.get_action(state)
        w2 = policy2.get_action(state)
        assert not np.allclose(w1, w2)

    def test_momentum_simplex(self):
        from policies.behavior import MomentumPolicy

        policy = MomentumPolicy(n_assets=N_ASSETS, lookback=5)
        # State doesn't matter — momentum uses forward_returns data
        # We test via rollout in transition tests. Here just check the
        # interface with a returns history.
        rng = np.random.RandomState(0)
        returns_history = rng.randn(10, N_ASSETS) * 0.01
        w = policy.get_action_from_returns(returns_history)
        assert w.shape == (N_ASSETS,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-7)
        assert np.all(w >= 0)

    def test_momentum_deterministic(self):
        from policies.behavior import MomentumPolicy

        policy = MomentumPolicy(n_assets=N_ASSETS, lookback=5)
        rng = np.random.RandomState(0)
        returns_history = rng.randn(10, N_ASSETS) * 0.01
        w1 = policy.get_action_from_returns(returns_history)
        w2 = policy.get_action_from_returns(returns_history)
        np.testing.assert_array_equal(w1, w2)

    def test_risk_parity_simplex(self):
        from policies.behavior import RiskParityPolicy

        policy = RiskParityPolicy(n_assets=N_ASSETS, lookback=5)
        rng = np.random.RandomState(0)
        returns_history = rng.randn(10, N_ASSETS) * 0.01
        w = policy.get_action_from_returns(returns_history)
        assert w.shape == (N_ASSETS,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-7)
        assert np.all(w >= 0)

    def test_risk_parity_deterministic(self):
        from policies.behavior import RiskParityPolicy

        policy = RiskParityPolicy(n_assets=N_ASSETS, lookback=5)
        rng = np.random.RandomState(0)
        returns_history = rng.randn(10, N_ASSETS) * 0.01
        w1 = policy.get_action_from_returns(returns_history)
        w2 = policy.get_action_from_returns(returns_history)
        np.testing.assert_array_equal(w1, w2)

    def test_risk_parity_inverse_vol(self):
        """Higher vol asset should get lower weight."""
        from policies.behavior import RiskParityPolicy

        policy = RiskParityPolicy(n_assets=2, lookback=5)
        # Asset 0: low vol, asset 1: high vol
        returns = np.zeros((10, 2))
        returns[:, 0] = 0.001  # nearly constant
        returns[:, 1] = np.array([0.05, -0.05, 0.04, -0.04, 0.03,
                                  -0.03, 0.02, -0.02, 0.01, -0.01])
        w = policy.get_action_from_returns(returns)
        assert w[0] > w[1], "Low-vol asset should get higher weight"


# ===========================================================================
# TestTransitionConstruction
# ===========================================================================

class TestTransitionConstruction:
    """Transitions produced by the dataset builder must be correct."""

    def test_transitions_contiguous(self):
        """next_states[i] == states[i+1] for consecutive transitions."""
        from data.build_dataset import build_dataset_from_env

        features, forward_returns = _make_tiny_features_and_returns(n_steps=30)
        env = PortfolioEnv(features=features, forward_returns=forward_returns)

        from policies.behavior import EqualWeightPolicy
        policy = EqualWeightPolicy(n_assets=N_ASSETS)

        dataset = build_dataset_from_env(env, policy, seed=0)
        states = dataset["states"]
        next_states = dataset["next_states"]

        # For all non-terminal transitions, next_states[i] == states[i+1]
        dones = dataset["dones"]
        for i in range(len(states) - 1):
            if not dones[i]:
                np.testing.assert_array_equal(
                    next_states[i], states[i + 1],
                    err_msg=f"Transition {i}: next_states != states[i+1]"
                )

    def test_reward_matches_env(self):
        """Rewards in dataset must match what the env produces."""
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import EqualWeightPolicy

        features, forward_returns = _make_tiny_features_and_returns(n_steps=20)
        env = PortfolioEnv(
            features=features,
            forward_returns=forward_returns,
            transaction_cost_lambda=0.001,
        )
        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        dataset = build_dataset_from_env(env, policy, seed=0)

        # Re-roll the trajectory manually and compare rewards
        env2 = PortfolioEnv(
            features=features,
            forward_returns=forward_returns,
            transaction_cost_lambda=0.001,
        )
        obs, _ = env2.reset(seed=0)
        for i in range(len(dataset["rewards"])):
            action = dataset["actions"][i]
            _, reward, terminated, _, _ = env2.step(action)
            np.testing.assert_allclose(
                dataset["rewards"][i], reward, atol=1e-7,
                err_msg=f"Step {i}: reward mismatch"
            )
            if terminated:
                break

    def test_actions_on_simplex(self):
        """All recorded actions must lie on the simplex."""
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import DirichletPolicy

        features, forward_returns = _make_tiny_features_and_returns(n_steps=30)
        env = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=7)
        dataset = build_dataset_from_env(env, policy, seed=0)

        actions = dataset["actions"]
        np.testing.assert_allclose(actions.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(actions >= -1e-7)

    def test_dataset_starts_with_reset(self):
        """First state in dataset must equal env.reset() observation."""
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import EqualWeightPolicy

        features, forward_returns = _make_tiny_features_and_returns(n_steps=20)
        env = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        dataset = build_dataset_from_env(env, policy, seed=0)

        env2 = PortfolioEnv(features=features, forward_returns=forward_returns)
        expected_obs, _ = env2.reset(seed=0)
        np.testing.assert_array_equal(dataset["states"][0], expected_obs)


# ===========================================================================
# TestDatasetDeterminism
# ===========================================================================

class TestDatasetDeterminism:
    """Same seed must produce identical datasets."""

    def test_same_seed_identical(self):
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import DirichletPolicy

        features, forward_returns = _make_tiny_features_and_returns(n_steps=30)

        env1 = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy1 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=42)
        ds1 = build_dataset_from_env(env1, policy1, seed=0)

        env2 = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy2 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=42)
        ds2 = build_dataset_from_env(env2, policy2, seed=0)

        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            np.testing.assert_array_equal(ds1[key], ds2[key], err_msg=f"{key} differs")

    def test_different_seed_differs(self):
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import DirichletPolicy

        features, forward_returns = _make_tiny_features_and_returns(n_steps=30)

        env1 = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy1 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=0)
        ds1 = build_dataset_from_env(env1, policy1, seed=0)

        env2 = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy2 = DirichletPolicy(n_assets=N_ASSETS, alpha=1.0, seed=99)
        ds2 = build_dataset_from_env(env2, policy2, seed=0)

        assert not np.array_equal(ds1["actions"], ds2["actions"])


# ===========================================================================
# TestSplitIntegrity
# ===========================================================================

class TestSplitIntegrity:
    """No transition should cross a dataset split boundary."""

    def test_no_cross_boundary_transition(self):
        """When splitting dataset, last transition in each split must be done=True
        OR the next_state must not appear as states[0] of the next split."""
        from data.build_dataset import split_dataset

        # Build a fake dataset with known dones
        N = 100
        rng = np.random.RandomState(0)
        states = rng.randn(N, 10).astype(np.float32)
        actions = rng.dirichlet(np.ones(N_ASSETS), size=N).astype(np.float32)
        rewards = rng.randn(N).astype(np.float32)
        next_states = rng.randn(N, 10).astype(np.float32)
        # Mark some transitions as episode boundaries
        dones = np.zeros(N, dtype=bool)
        dones[29] = True  # end of first "episode"
        dones[59] = True  # end of second "episode"
        dones[99] = True  # end of third "episode"

        dataset = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

        train, val, test = split_dataset(dataset, train_frac=0.3, val_frac=0.3)

        # Each split must end with done=True (episode boundary)
        assert train["dones"][-1] == True, "Train split must end at episode boundary"
        assert val["dones"][-1] == True, "Val split must end at episode boundary"
        assert test["dones"][-1] == True, "Test split must end at episode boundary"

    def test_split_covers_all_transitions(self):
        """Total transitions across splits must equal original dataset size."""
        from data.build_dataset import split_dataset

        N = 90
        rng = np.random.RandomState(0)
        dones = np.zeros(N, dtype=bool)
        dones[29] = True
        dones[59] = True
        dones[89] = True

        dataset = {
            "states": rng.randn(N, 5).astype(np.float32),
            "actions": rng.randn(N, N_ASSETS).astype(np.float32),
            "rewards": rng.randn(N).astype(np.float32),
            "next_states": rng.randn(N, 5).astype(np.float32),
            "dones": dones,
        }

        train, val, test = split_dataset(dataset, train_frac=0.34, val_frac=0.33)
        total = len(train["states"]) + len(val["states"]) + len(test["states"])
        assert total == N


# ===========================================================================
# TestDatasetShapes
# ===========================================================================

class TestDatasetShapes:
    """All dataset arrays must have consistent shapes."""

    def test_shapes_consistent(self):
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import EqualWeightPolicy

        features, forward_returns = _make_tiny_features_and_returns(
            n_steps=30, n_assets=N_ASSETS
        )
        env = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        dataset = build_dataset_from_env(env, policy, seed=0)

        N = len(dataset["states"])
        assert N > 0
        assert dataset["states"].shape == (N, features.shape[1])
        assert dataset["actions"].shape == (N, N_ASSETS)
        assert dataset["rewards"].shape == (N,)
        assert dataset["next_states"].shape == (N, features.shape[1])
        assert dataset["dones"].shape == (N,)

    def test_correct_number_of_transitions(self):
        """For a dataset with T rows, env produces T-1 transitions (steps 0..T-2)."""
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import EqualWeightPolicy

        n_steps = 20
        features, forward_returns = _make_tiny_features_and_returns(
            n_steps=n_steps, n_assets=N_ASSETS
        )
        env = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        dataset = build_dataset_from_env(env, policy, seed=0)

        # env with n_steps rows produces n_steps - 1 transitions
        # (act on rows 0..T-2, observe rows 1..T-1)
        # But the last step terminates, so we get n_steps - 2 non-terminal + 1 terminal
        assert len(dataset["states"]) == n_steps - 1

    def test_dones_boolean(self):
        from data.build_dataset import build_dataset_from_env
        from policies.behavior import EqualWeightPolicy

        features, forward_returns = _make_tiny_features_and_returns(n_steps=20)
        env = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        dataset = build_dataset_from_env(env, policy, seed=0)

        assert dataset["dones"].dtype == bool
        # Last transition must be done
        assert dataset["dones"][-1] == True
        # All others should be not done (single episode)
        assert not np.any(dataset["dones"][:-1])


# ===========================================================================
# TestNpzSaveLoad
# ===========================================================================

class TestNpzSaveLoad:
    """Dataset can be saved to and loaded from .npz."""

    def test_save_and_load(self, tmp_path):
        from data.build_dataset import build_dataset_from_env, save_dataset, load_dataset
        from policies.behavior import EqualWeightPolicy

        features, forward_returns = _make_tiny_features_and_returns(n_steps=20)
        env = PortfolioEnv(features=features, forward_returns=forward_returns)
        policy = EqualWeightPolicy(n_assets=N_ASSETS)
        dataset = build_dataset_from_env(env, policy, seed=0)

        path = tmp_path / "test_dataset.npz"
        save_dataset(dataset, str(path), metadata={
            "n_assets": N_ASSETS,
            "feature_dim": features.shape[1],
            "seed": 0,
            "policy_name": "equal_weight",
        })
        assert path.exists()

        loaded = load_dataset(str(path))
        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            np.testing.assert_array_equal(dataset[key], loaded[key])
