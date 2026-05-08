"""Offline dataset builder for portfolio RL.

Rolls out a behavior policy through a PortfolioEnv and records transitions
as numpy arrays suitable for IQL training.

Datasets may optionally carry a ``dates`` array — one timestamp per
transition, corresponding to the date on which ``state`` was observed.
When present, :mod:`data.splits` can partition the dataset into
train/validation/test subsets by calendar date.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.envs.portfolio_env import PortfolioEnv
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
    dates: np.ndarray | pd.DatetimeIndex | None = None,
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
    dates:
        Optional per-row timestamps aligned with ``env._features``. When
        provided, the output dict additionally contains a ``dates`` key
        of shape ``(n_transitions,)`` giving the observation date of each
        transition. Used downstream to split by calendar date.

    Returns
    -------
    Dict with keys: states, actions, rewards, next_states, dones, and
    optionally ``dates`` (int64 ns since epoch).
    """
    states_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []
    next_states_list: list[np.ndarray] = []
    dones_list: list[bool] = []
    dates_list: list[np.int64] = []

    dates_ns: np.ndarray | None = None
    if dates is not None:
        dt = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
        if len(dt) != env._n_steps:
            raise ValueError(
                f"len(dates)={len(dt)} must match env._n_steps={env._n_steps}"
            )
        # Pandas' default datetime resolution is now `us`; force `ns` so the
        # int64 representation is unambiguous on disk and round-trips with
        # ``pd.to_datetime(..., unit="ns")`` downstream.
        dates_ns = dt.values.astype("datetime64[ns]").astype(np.int64)

    obs, _ = env.reset(seed=seed)

    # For momentum / risk-parity policies, we need to track returns history.
    # We access the env's internal forward_returns for this purpose.
    returns_history: list[np.ndarray] = []

    terminated = False
    t = 0
    while not terminated:
        state = obs.copy()
        if dates_ns is not None:
            dates_list.append(np.int64(dates_ns[t]))

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

    out: Dataset = {
        "states": np.array(states_list, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.float64),
        "rewards": np.array(rewards_list, dtype=np.float64),
        "next_states": np.array(next_states_list, dtype=np.float32),
        "dones": np.array(dones_list, dtype=bool),
    }
    if dates_ns is not None:
        out["dates"] = np.array(dates_list, dtype=np.int64)

    # ------------------------------------------------------------------
    # Fail-fast integrity checks. Any violation here invalidates offline
    # RL training, so raise AssertionError rather than a warning. See
    # `analysis/dataset_diagnostics.py` for the read-only version of the
    # same checks (run after save, optionally CI-gated).
    # ------------------------------------------------------------------
    _assert_dataset_integrity(out)
    return out


def _assert_dataset_integrity(ds: Dataset) -> None:
    """Hard invariants for a freshly rolled-out dataset.

    Checks (all must hold):
      1. shapes match: states, actions, rewards, next_states, dones have
         the same first-axis length; states/next_states have the same
         feature dimension.
      2. no NaN / Inf anywhere in states, next_states, actions, rewards.
      3. next_state continuity: for every non-terminal transition t,
         ``next_states[t] == states[t+1]``.
      4. only the final transition may be terminal (single-episode rollout),
         and that last transition IS terminal.
      5. if dates are present: strictly monotonically increasing.
    """
    N = len(ds["states"])
    for key in ("states", "actions", "rewards", "next_states", "dones"):
        if len(ds[key]) != N:
            raise AssertionError(
                f"Dataset shape mismatch: len({key})={len(ds[key])} vs "
                f"len(states)={N}."
            )
    if ds["states"].shape[1] != ds["next_states"].shape[1]:
        raise AssertionError(
            f"state_dim mismatch: states={ds['states'].shape[1]}, "
            f"next_states={ds['next_states'].shape[1]}."
        )
    for key in ("states", "next_states", "actions", "rewards"):
        arr = ds[key]
        if not np.isfinite(arr).all():
            n_bad = int(np.size(arr) - np.isfinite(arr).sum())
            raise AssertionError(
                f"Dataset contains {n_bad} non-finite entries in '{key}'."
            )
    # Continuity: next_states[t] must equal states[t+1] wherever t is not
    # an episode boundary. A strict per-element equality is fine here
    # because both arrays are copied from the same env-produced buffer.
    dones = ds["dones"]
    if N >= 2:
        non_terminal = ~dones[:-1]
        if non_terminal.any():
            lhs = ds["next_states"][:-1][non_terminal]
            rhs = ds["states"][1:][non_terminal]
            if not np.array_equal(lhs, rhs):
                raise AssertionError(
                    "next_states[t] != states[t+1] for at least one "
                    "non-terminal transition — episode continuity broken."
                )
    if not bool(dones[-1]):
        raise AssertionError(
            "Last transition is not done=True. build_dataset_from_env "
            "rolls a single episode to completion and must terminate."
        )
    if N >= 2 and bool(dones[:-1].any()):
        raise AssertionError(
            "Unexpected intermediate done=True transition; builder emits a "
            "single episode."
        )
    if "dates" in ds:
        diffs = np.diff(ds["dates"].astype(np.int64))
        if not bool((diffs > 0).all()):
            raise AssertionError(
                "Dataset 'dates' are not strictly monotonically increasing."
            )


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
    Any of the optional keys (``dates``, ``forward_returns``) present in
    ``dataset`` are persisted alongside the core transition arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = {}
    for key in ["states", "actions", "rewards", "next_states", "dones"]:
        save_dict[key] = dataset[key]
    for key in ["dates", "forward_returns"]:
        if key in dataset:
            save_dict[key] = dataset[key]

    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(v)

    np.savez(str(path), **save_dict)


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset from a .npz file.

    Returns a dict with keys: states, actions, rewards, next_states, dones,
    and optionally ``dates`` / ``forward_returns`` if present in the file.
    Metadata keys (prefixed with ``meta_``) are excluded.
    """
    data = np.load(str(path), allow_pickle=True)
    dataset: Dataset = {}
    for key in ["states", "actions", "rewards", "next_states", "dones"]:
        dataset[key] = data[key]
    for key in ["dates", "forward_returns"]:
        if key in data.files:
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
