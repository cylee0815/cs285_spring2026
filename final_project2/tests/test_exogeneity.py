"""Verify the env-level exogeneity assumption that Continuous GRPO depends on.

The method is sound iff the next state s_{t+1} does not depend on the agent's
action a_t — i.e. the market does not respond to a retail allocation. This
allows action-relative advantages at the same state s_t to reduce to reward
differences (the bootstrapped value term cancels), removing the need for a
critic.

If `test_next_state_independent_of_action_default_obs` ever fails, the GRPO
formulation in this repo is unsound and the method must be revisited.

The companion test `test_exogeneity_breaks_with_include_prev_weights`
documents that the optional `include_prev_weights=True` mode injects the
action into the observation, which violates exogeneity. GRPO must not be
trained with that flag enabled.
"""

from __future__ import annotations

import numpy as np

from core.envs.portfolio_env import PortfolioEnv


N_ASSETS = 8
N_FEATURES = 16
T = 50
EPISODE_LENGTH = 20
TC_LAMBDA = 0.001


def _make_env(data_seed: int = 0, include_prev_weights: bool = False) -> PortfolioEnv:
    """Build a synthetic env with deterministic data for reproducibility."""
    rng = np.random.default_rng(data_seed)
    features = rng.standard_normal((T, N_FEATURES)).astype(np.float32)
    returns = (rng.standard_normal((T, N_ASSETS)) * 0.015).astype(np.float32)
    return PortfolioEnv(
        features=features,
        forward_returns=returns,
        transaction_cost_lambda=TC_LAMBDA,
        episode_length=EPISODE_LENGTH,
        include_prev_weights=include_prev_weights,
    )


def _onehot(idx: int) -> np.ndarray:
    a = np.zeros(N_ASSETS, dtype=np.float32)
    a[idx] = 1.0
    return a


def test_next_state_independent_of_action_default_obs():
    """Exogeneity at the env level: s_{t+1} does not depend on a_t.

    Two envs with identical seeds and identical prefix histories diverge at
    step k with two very different one-hot allocations. The resulting
    observations must be bit-identical (modulo float roundoff).
    """
    a1 = _onehot(0)              # all SPY
    a2 = _onehot(N_ASSETS - 1)   # all SHY

    # Mild prefix to put both envs in a non-trivial internal state
    # (non-default ``prev_weights`` and a non-zero step counter) before the
    # diverging action.
    prefix = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    n_prefix_steps = 3

    env1 = _make_env(data_seed=0)
    env1.reset(seed=42)
    for _ in range(n_prefix_steps):
        env1.step(prefix)
    next_obs1, _, _, _, _ = env1.step(a1)

    env2 = _make_env(data_seed=0)
    env2.reset(seed=42)
    for _ in range(n_prefix_steps):
        env2.step(prefix)
    next_obs2, _, _, _, _ = env2.step(a2)

    np.testing.assert_allclose(
        next_obs1,
        next_obs2,
        atol=1e-12,
        err_msg=(
            "Exogeneity violated: next obs depended on the action. "
            "GRPO's critic-free formulation is unsound under this env."
        ),
    )


def test_exogeneity_holds_at_first_step_too():
    """Same property right out of reset (no prefix), as a sanity check."""
    env1 = _make_env(data_seed=1)
    env1.reset(seed=7)
    next_obs1, _, _, _, _ = env1.step(_onehot(2))

    env2 = _make_env(data_seed=1)
    env2.reset(seed=7)
    next_obs2, _, _, _, _ = env2.step(_onehot(5))

    np.testing.assert_allclose(next_obs1, next_obs2, atol=1e-12)


def test_exogeneity_breaks_with_include_prev_weights():
    """``include_prev_weights=True`` injects the action into the next obs.

    The market-feature portion of the obs stays exogenous, but the appended
    ``prev_weights`` portion is the just-taken action — different actions
    give different next observations. This documents the constraint that
    GRPO must run with the default ``include_prev_weights=False``.
    """
    env1 = _make_env(data_seed=2, include_prev_weights=True)
    env1.reset(seed=11)
    next_obs1, _, _, _, _ = env1.step(_onehot(0))

    env2 = _make_env(data_seed=2, include_prev_weights=True)
    env2.reset(seed=11)
    next_obs2, _, _, _, _ = env2.step(_onehot(N_ASSETS - 1))

    # Market-feature portion (first N_FEATURES entries) is still exogenous.
    np.testing.assert_allclose(
        next_obs1[:N_FEATURES],
        next_obs2[:N_FEATURES],
        atol=1e-12,
        err_msg="Market features should still be exogenous regardless of action.",
    )

    # Trailing prev_weights portion encodes the action and must differ.
    prev_w_1 = next_obs1[N_FEATURES:]
    prev_w_2 = next_obs2[N_FEATURES:]
    assert not np.allclose(prev_w_1, prev_w_2), (
        "include_prev_weights=True is supposed to break exogeneity; "
        "got identical prev_weights across distinct actions."
    )
