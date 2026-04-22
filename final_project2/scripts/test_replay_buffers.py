"""Minimal correctness test for ReplayBuffer and NStepReplayBuffer.

Pushes a known sequence of transitions, samples a batch, and verifies:
    1. Tensor shapes and dtypes match the expected offline RL contract.
    2. For NStepReplayBuffer with n_step > 1, the stored reward equals the
       discounted n-step return (sum_{i=0..n-1} gamma^i * r_i), and
       next_obs points to s_{t+n}.
    3. Episode termination flushes the pending n-step queue.

This does not train anything and runs in a second or two on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from offline_rl.agents.replay_buffer import ReplayBuffer, NStepReplayBuffer  # noqa: E402

OBS_DIM = 6
ACTION_DIM = 8
CAP = 64
DEVICE = torch.device("cpu")


def _fake_obs(i: int) -> np.ndarray:
    return np.full(OBS_DIM, float(i), dtype=np.float32)


def _fake_action(i: int) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[i % ACTION_DIM] = 1.0
    return a


# ---------------------------------------------------------------------------
# 1. Vanilla ReplayBuffer
# ---------------------------------------------------------------------------
def test_vanilla_buffer() -> None:
    buf = ReplayBuffer(CAP, OBS_DIM, ACTION_DIM, DEVICE, seq_len=4)
    # Push 10 transitions split across 2 episodes.
    for i in range(10):
        done = (i == 4) or (i == 9)
        ep_start = (i == 0) or (i == 5)
        buf.add(
            obs=_fake_obs(i),
            action=_fake_action(i),
            reward=float(i),
            next_obs=_fake_obs(i + 1),
            done=done,
            episode_start=ep_start,
        )

    assert len(buf) == 10, f"expected 10 transitions, got {len(buf)}"

    batch = buf.sample(batch_size=5)
    assert batch["obs"].shape == (5, OBS_DIM)
    assert batch["actions"].shape == (5, ACTION_DIM)
    assert batch["rewards"].shape == (5,)
    assert batch["next_obs"].shape == (5, OBS_DIM)
    assert batch["dones"].shape == (5,)
    assert batch["obs"].dtype == torch.float32

    # Freeze and verify it blocks further writes.
    buf.freeze()
    buf.add(_fake_obs(99), _fake_action(0), 0.0, _fake_obs(99), True)
    assert len(buf) == 10, "freeze() should block writes"

    print("[OK] ReplayBuffer: 10 stored, sample shapes correct, freeze() honored.")


# ---------------------------------------------------------------------------
# 2. NStepReplayBuffer — n_step=3, gamma=0.9
# ---------------------------------------------------------------------------
def test_nstep_buffer() -> None:
    n_step = 3
    gamma = 0.9
    buf = NStepReplayBuffer(
        CAP, OBS_DIM, ACTION_DIM, DEVICE, seq_len=4, n_step=n_step, gamma=gamma,
    )

    # One episode of 6 non-terminal transitions; index the reward as r_i = i.
    # After 3 pushes, the buffer should emit the first compressed transition
    # with reward = r0 + gamma * r1 + gamma^2 * r2 = 0 + 0.9 + 0.81*2 = 2.52.
    for i in range(6):
        buf.add(
            obs=_fake_obs(i),
            action=_fake_action(i),
            reward=float(i),
            next_obs=_fake_obs(i + 1),
            done=False,
            episode_start=(i == 0),
        )

    # 6 pushes with n_step=3 and no terminal -> the first 6-3+1=4 transitions
    # flush to the underlying buffer one-by-one as the deque slides.
    expected_stored = 6 - n_step + 1
    assert len(buf) == expected_stored, (
        f"expected {expected_stored} stored n-step transitions, got {len(buf)}"
    )

    # Inspect the first stored transition directly (pre-shuffle).
    r0_stored = buf.rewards[0]
    expected_r0 = 0.0 + gamma * 1.0 + gamma**2 * 2.0
    assert abs(r0_stored - expected_r0) < 1e-5, (
        f"first n-step reward mismatch: got {r0_stored}, expected {expected_r0}"
    )
    assert np.allclose(buf.obs[0], _fake_obs(0))
    # next_obs of the first stored transition should be s_{t+n} = s_3, i.e.,
    # the next_obs recorded for the LAST of the three pooled transitions,
    # which was _fake_obs(3).
    assert np.allclose(buf.next_obs[0], _fake_obs(3)), (
        f"stored next_obs mismatch: {buf.next_obs[0]} vs expected {_fake_obs(3)}"
    )
    assert buf.dones[0] == 0.0

    # Now terminate the episode and confirm the buffer flushes the tail.
    size_before_done = len(buf)
    buf.add(
        obs=_fake_obs(6),
        action=_fake_action(6),
        reward=6.0,
        next_obs=_fake_obs(7),
        done=True,
    )
    assert len(buf) > size_before_done, (
        "terminal transition should flush the remaining n-step queue"
    )
    # At least one stored transition must now have done=1.
    assert buf.dones[: len(buf)].max() == 1.0

    # Shape sanity on sample().
    batch = buf.sample(batch_size=4)
    assert batch["rewards"].shape == (4,)
    assert batch["next_obs"].shape == (4, OBS_DIM)

    print(
        f"[OK] NStepReplayBuffer(n_step={n_step}, gamma={gamma}): "
        f"discounted reward = {r0_stored:.4f} (expected {expected_r0:.4f}), "
        f"next_obs = s_{{t+n}}, terminal flush works."
    )


# ---------------------------------------------------------------------------
# 3. NStepReplayBuffer — degenerate case n_step=1 equals vanilla behavior.
# ---------------------------------------------------------------------------
def test_nstep_equals_vanilla_when_n_is_one() -> None:
    buf = NStepReplayBuffer(
        CAP, OBS_DIM, ACTION_DIM, DEVICE, seq_len=4, n_step=1, gamma=0.99,
    )
    for i in range(5):
        buf.add(
            obs=_fake_obs(i),
            action=_fake_action(i),
            reward=float(i),
            next_obs=_fake_obs(i + 1),
            done=(i == 4),
            episode_start=(i == 0),
        )
    assert len(buf) == 5, f"n_step=1 should store every transition, got {len(buf)}"
    # Rewards should match what we fed in (no discounting happens for n=1).
    assert list(buf.rewards[:5]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    print("[OK] NStepReplayBuffer(n_step=1): behaves like vanilla ReplayBuffer.")


def main() -> int:
    test_vanilla_buffer()
    test_nstep_buffer()
    test_nstep_equals_vanilla_when_n_is_one()
    print("\nAll replay-buffer tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
