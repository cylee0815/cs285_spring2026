"""Tests for the extended per-step training diagnostics.

The training loop must surface more than just raw losses: q_mean, q_std,
advantage_mean, and policy_entropy are needed for debugging and for the
milestone report. These are computed from a separate forward pass on the
current minibatch so that we do not have to touch ``algorithms.iql``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.iql import IQL
from training.train_iql import compute_diagnostics
from utils.replay_buffer import ReplayBuffer

STATE_DIM = 10
ACTION_DIM = 8
N_TRANSITIONS = 128


@pytest.fixture()
def dataset_path(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    states = rng.standard_normal((N_TRANSITIONS, STATE_DIM)).astype(np.float32)
    raw = rng.exponential(size=(N_TRANSITIONS, ACTION_DIM)).astype(np.float32)
    actions = raw / raw.sum(axis=1, keepdims=True)
    rewards = rng.standard_normal(N_TRANSITIONS).astype(np.float32)
    next_states = rng.standard_normal((N_TRANSITIONS, STATE_DIM)).astype(np.float32)
    dones = np.zeros(N_TRANSITIONS, dtype=np.float32)
    np.savez(
        tmp_path / "d.npz",
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
    return str(tmp_path / "d.npz")


def test_compute_diagnostics_returns_expected_keys(dataset_path: str) -> None:
    agent = IQL(state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu")
    buf = ReplayBuffer(dataset_path, device="cpu")
    s, a, r, s_next, done = buf.sample(32)

    diag = compute_diagnostics(agent, s, a)
    for key in ("q_mean", "q_std", "advantage_mean", "policy_entropy"):
        assert key in diag, f"{key} missing from diagnostics"
        assert np.isfinite(diag[key]), f"{key} is not finite: {diag[key]}"


def test_policy_entropy_is_nonnegative(dataset_path: str) -> None:
    """Softmax entropy must be in [0, log(N)]."""
    agent = IQL(state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu")
    buf = ReplayBuffer(dataset_path, device="cpu")
    s, a, *_ = buf.sample(64)
    diag = compute_diagnostics(agent, s, a)
    assert diag["policy_entropy"] >= 0.0
    assert diag["policy_entropy"] <= float(np.log(ACTION_DIM)) + 1e-4


def test_advantage_mean_is_q_minus_v(dataset_path: str) -> None:
    agent = IQL(state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu")
    buf = ReplayBuffer(dataset_path, device="cpu")
    s, a, *_ = buf.sample(32)
    diag = compute_diagnostics(agent, s, a)
    with torch.no_grad():
        q = agent.q_network(s, a).mean().item()
        v = agent.value_network(s).mean().item()
    assert diag["advantage_mean"] == pytest.approx(q - v, abs=1e-5)
