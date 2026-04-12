"""Tests for IQL: replay buffer, single step, loss decrease, device compat."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.iql import IQL
from utils.replay_buffer import ReplayBuffer
from utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATE_DIM = 10
ACTION_DIM = 8
N_TRANSITIONS = 200


@pytest.fixture()
def dataset_path(tmp_path: Path) -> str:
    """Create a small synthetic .npz dataset and return its path."""
    rng = np.random.default_rng(0)
    states = rng.standard_normal((N_TRANSITIONS, STATE_DIM)).astype(np.float32)
    # Actions on the simplex
    raw = rng.exponential(size=(N_TRANSITIONS, ACTION_DIM)).astype(np.float32)
    actions = raw / raw.sum(axis=1, keepdims=True)
    rewards = rng.standard_normal(N_TRANSITIONS).astype(np.float32)
    next_states = rng.standard_normal((N_TRANSITIONS, STATE_DIM)).astype(np.float32)
    dones = rng.choice([0.0, 1.0], size=N_TRANSITIONS, p=[0.95, 0.05]).astype(np.float32)
    np.savez(
        tmp_path / "test_dataset.npz",
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
    return str(tmp_path / "test_dataset.npz")


def _make_agent(device: str = "cpu") -> IQL:
    return IQL(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=device,
    )


# ---------------------------------------------------------------------------
# TestReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_sample_shapes(self, dataset_path: str) -> None:
        buf = ReplayBuffer(dataset_path, device="cpu")
        s, a, r, s_next, done = buf.sample(32)
        assert s.shape == (32, STATE_DIM)
        assert a.shape == (32, ACTION_DIM)
        assert r.shape == (32, 1)
        assert s_next.shape == (32, STATE_DIM)
        assert done.shape == (32, 1)

    def test_deterministic_sampling_with_seed(self, dataset_path: str) -> None:
        buf = ReplayBuffer(dataset_path, device="cpu")
        set_seed(123)
        s1, a1, *_ = buf.sample(16)
        set_seed(123)
        s2, a2, *_ = buf.sample(16)
        assert torch.equal(s1, s2)
        assert torch.equal(a1, a2)

    def test_buffer_size(self, dataset_path: str) -> None:
        buf = ReplayBuffer(dataset_path, device="cpu")
        assert buf.size == N_TRANSITIONS


# ---------------------------------------------------------------------------
# TestIQLStep
# ---------------------------------------------------------------------------


class TestIQLStep:
    def test_one_step_runs(self, dataset_path: str) -> None:
        agent = _make_agent()
        buf = ReplayBuffer(dataset_path, device="cpu")
        s, a, r, s_next, done = buf.sample(32)
        metrics = agent.update(s, a, r, s_next, done)
        assert set(metrics.keys()) == {"v_loss", "q_loss", "policy_loss"}
        for v in metrics.values():
            assert np.isfinite(v), f"Non-finite loss: {v}"

    def test_get_action_shape(self, dataset_path: str) -> None:
        agent = _make_agent()
        buf = ReplayBuffer(dataset_path, device="cpu")
        s, *_ = buf.sample(1)
        action = agent.get_action(s.squeeze(0))
        assert action.shape == (ACTION_DIM,)
        # Should be valid portfolio weights
        assert torch.all(action >= 0)
        assert torch.isclose(action.sum(), torch.tensor(1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# TestLossDecrease
# ---------------------------------------------------------------------------


class TestLossDecrease:
    def test_value_loss_decreases(self, dataset_path: str) -> None:
        """On a fixed batch, repeated V updates should reduce the V loss."""
        set_seed(0)
        agent = _make_agent()
        buf = ReplayBuffer(dataset_path, device="cpu")
        s, a, r, s_next, done = buf.sample(64)

        first_loss = agent.update_value(s, a)
        for _ in range(50):
            loss = agent.update_value(s, a)
        assert loss < first_loss, f"V loss did not decrease: {first_loss:.4f} -> {loss:.4f}"

    def test_q_loss_decreases(self, dataset_path: str) -> None:
        """On a fixed batch, repeated Q updates should reduce the Q loss."""
        set_seed(0)
        agent = _make_agent()
        buf = ReplayBuffer(dataset_path, device="cpu")
        s, a, r, s_next, done = buf.sample(64)

        first_loss = agent.update_q(s, a, r, s_next, done)
        for _ in range(50):
            loss = agent.update_q(s, a, r, s_next, done)
        assert loss < first_loss, f"Q loss did not decrease: {first_loss:.4f} -> {loss:.4f}"


# ---------------------------------------------------------------------------
# TestDeviceCompatibility
# ---------------------------------------------------------------------------


class TestDeviceCompatibility:
    def test_cpu(self, dataset_path: str) -> None:
        agent = _make_agent(device="cpu")
        buf = ReplayBuffer(dataset_path, device="cpu")
        s, a, r, s_next, done = buf.sample(16)
        metrics = agent.update(s, a, r, s_next, done)
        for v in metrics.values():
            assert np.isfinite(v)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self, dataset_path: str) -> None:
        agent = _make_agent(device="cuda")
        buf = ReplayBuffer(dataset_path, device="cuda")
        s, a, r, s_next, done = buf.sample(16)
        metrics = agent.update(s, a, r, s_next, done)
        for v in metrics.values():
            assert np.isfinite(v)

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps(self, dataset_path: str) -> None:
        agent = _make_agent(device="mps")
        buf = ReplayBuffer(dataset_path, device="mps")
        s, a, r, s_next, done = buf.sample(16)
        metrics = agent.update(s, a, r, s_next, done)
        for v in metrics.values():
            assert np.isfinite(v)
