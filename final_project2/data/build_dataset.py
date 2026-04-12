"""Offline dataset builder for portfolio RL.

Rolls out a behavior policy through a PortfolioEnv and records transitions
as numpy arrays suitable for IQL training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from env.portfolio_env import PortfolioEnv
from policies.behavior import (
    DirichletPolicy,
    EqualWeightPolicy,
    MomentumPolicy,
    RiskParityPolicy,
)

__all__ = [
    "build_dataset_from_env",
    "save_dataset",
    "load_dataset",
    "split_dataset",
]


# ---------------------------------------------------------------------------
# Dataset type
# ---------------------------------------------------------------------------

Dataset = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_dataset_from_env(
    env: PortfolioEnv,
    policy: EqualWeightPolicy | DirichletPolicy | MomentumPolicy | RiskParityPolicy,
    seed: int = 0,
) -> Dataset:
    """Roll out a single episode and collect transitions.

    Parameters
    ----------
    env:
        A PortfolioEnv instance. The full episode is rolled out (reset to done).
    policy:
        A behavior policy that produces simplex weights.
    seed:
        Seed passed to env.reset().

    Returns
    -------
    Dict with keys: states, actions, rewards, next_states, dones.
    """
    states_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []
    next_states_list: list[np.ndarray] = []
    dones_list: list[bool] = []

    obs, _ = env.reset(seed=seed)

    # For momentum / risk-parity policies, we need to track returns history.
    # We access the env's internal forward_returns for this purpose.
    returns_history: list[np.ndarray] = []

    terminated = False
    t = 0
    while not terminated:
        state = obs.copy()

        # Get action based on policy type
        if isinstance(policy, (MomentumPolicy, RiskParityPolicy)):
            if len(returns_history) < 2:
                # Not enough history yet — fall back to equal weight
                action = np.ones(env._n_assets, dtype=np.float64) / env._n_assets
            else:
                history_arr = np.array(returns_history)
                action = policy.get_action_from_returns(history_arr)
        else:
            action = policy.get_action(state)

        obs, reward, terminated, truncated, info = env.step(action)

        # Record the forward return at this timestep for history-based policies
        returns_history.append(env._forward_returns[t].copy())

        states_list.append(state)
        actions_list.append(action.copy())
        rewards_list.append(reward)
        next_states_list.append(obs.copy())
        dones_list.append(terminated or truncated)

        t += 1

    return {
        "states": np.array(states_list, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.float64),
        "rewards": np.array(rewards_list, dtype=np.float64),
        "next_states": np.array(next_states_list, dtype=np.float32),
        "dones": np.array(dones_list, dtype=bool),
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_dataset(
    dataset: Dataset,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save dataset arrays to a .npz file.

    Metadata values are stored as additional arrays (scalars wrapped in 0-d arrays).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = {}
    for key in ["states", "actions", "rewards", "next_states", "dones"]:
        save_dict[key] = dataset[key]

    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(v)

    np.savez(str(path), **save_dict)


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset from a .npz file.

    Returns a dict with keys: states, actions, rewards, next_states, dones.
    Metadata keys (prefixed with meta_) are excluded.
    """
    data = np.load(str(path), allow_pickle=True)
    dataset: Dataset = {}
    for key in ["states", "actions", "rewards", "next_states", "dones"]:
        dataset[key] = data[key]
    return dataset


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split_dataset(
    dataset: Dataset,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[Dataset, Dataset, Dataset]:
    """Split a dataset at episode boundaries.

    Splits respect episode boundaries marked by dones=True so no transition
    crosses a split boundary. Each split ends at a done=True transition.

    Parameters
    ----------
    dataset:
        Full dataset dict.
    train_frac, val_frac:
        Approximate fraction of transitions for train and val.
        Test gets the remainder.

    Returns
    -------
    (train, val, test) dataset dicts.
    """
    dones = dataset["dones"]
    N = len(dones)

    # Find all episode boundary indices (where done=True)
    boundary_indices = np.where(dones)[0]
    if len(boundary_indices) == 0:
        raise ValueError("Dataset has no episode boundaries (no done=True).")

    # Find split points at episode boundaries
    target_train_end = int(N * train_frac)
    target_val_end = int(N * (train_frac + val_frac))

    # Find the boundary closest to (but not exceeding) target
    train_end = _find_boundary(boundary_indices, target_train_end)
    val_end = _find_boundary(boundary_indices, target_val_end, min_idx=train_end + 1)

    # Ensure we have at least one transition in each split
    if val_end >= N:
        val_end = boundary_indices[-2] if len(boundary_indices) >= 2 else train_end
    if train_end >= val_end:
        raise ValueError("Cannot split dataset: not enough episode boundaries.")

    train_slice = slice(0, train_end + 1)
    val_slice = slice(train_end + 1, val_end + 1)
    test_slice = slice(val_end + 1, N)

    return (
        _slice_dataset(dataset, train_slice),
        _slice_dataset(dataset, val_slice),
        _slice_dataset(dataset, test_slice),
    )


def _find_boundary(
    boundary_indices: np.ndarray, target: int, min_idx: int = 0
) -> int:
    """Find the episode boundary index closest to but <= target."""
    candidates = boundary_indices[boundary_indices >= min_idx]
    candidates = candidates[candidates <= target]
    if len(candidates) == 0:
        # Fall back to the first boundary at or after min_idx
        candidates = boundary_indices[boundary_indices >= min_idx]
        if len(candidates) == 0:
            return boundary_indices[-1]
        return int(candidates[0])
    return int(candidates[-1])


def _slice_dataset(dataset: Dataset, s: slice) -> Dataset:
    """Slice all arrays in a dataset dict."""
    return {key: arr[s] for key, arr in dataset.items()}
