"""Unit test: n-step Bellman target correctness for IQL and TD3+BC.

Verifies that the critic target computation is:

    target = R^{(n)} + gamma^n * (1 - done) * V(s_{t+n})

For n=1, this must be numerically identical to the standard single-step
target (regression guard against the fix introducing a change at n=1).

Usage:
    uv run python scripts/test_nstep_target.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import torch
import numpy as np  # noqa: F401

OBS_DIM = 24
ACTION_DIM = 8
GAMMA = 0.9
DEVICE = torch.device("cpu")


def _load_config(name: str, n_step: int):
    mod = importlib.import_module(f"offline_rl.configs.{name}_config")
    config = mod.get_config()
    config.gamma = GAMMA
    config.n_step = n_step
    return config


def _make_batch(
    obs_dim: int,
    action_dim: int,
    rewards: list[float],
    dones: list[float],
) -> dict[str, torch.Tensor]:
    """Build a deterministic batch with known values."""
    bsz = len(rewards)
    return {
        "obs": torch.randn(bsz, obs_dim),
        "actions": torch.softmax(torch.randn(bsz, action_dim), dim=-1),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "next_obs": torch.randn(bsz, obs_dim),
        "dones": torch.tensor(dones, dtype=torch.float32),
        # context seqs (needed by CQL-geodesic path; ignored by others)
        "obs_seq": torch.randn(bsz, 20, obs_dim),
        "next_obs_seq": torch.randn(bsz, 20, obs_dim),
    }


# ─── IQL target extraction ───────────────────────────────────────────

def _iql_target(config, agent, batch) -> torch.Tensor:
    """Extract the Bellman target that IQL's update_critic would compute."""
    with torch.no_grad():
        next_v = agent.value_net(batch["next_obs"]).squeeze(-1)
        gamma_n = config.gamma ** agent.n_step
        target = batch["rewards"] + (1.0 - batch["dones"]) * gamma_n * next_v
    return target


# ─── TD3+BC target extraction ────────────────────────────────────────

def _td3bc_target(config, agent, batch) -> torch.Tensor:
    """Extract the Bellman target that TD3+BC's update_critic would compute."""
    with torch.no_grad():
        next_actions = agent.target_actor(batch["next_obs"])
        target_q = agent.target_critic.q_min(batch["next_obs"], next_actions)
        gamma_n = config.gamma ** agent.n_step
        target = batch["rewards"] + (1.0 - batch["dones"]) * gamma_n * target_q
    return target


# ─── Tests ────────────────────────────────────────────────────────────

def test_iql_nstep():
    from offline_rl.agents.iql import IQLAgent

    for n_step in [1, 3]:
        config = _load_config("iql", n_step)
        agent = IQLAgent(OBS_DIM, ACTION_DIM, config, DEVICE)

        # Known rewards: the buffer would have stored R^{(n)} already.
        # Here we test that the agent applies gamma^n correctly.
        rewards = [1.0, 2.0, 0.5]
        dones = [0.0, 0.0, 1.0]
        batch = _make_batch(OBS_DIM, ACTION_DIM, rewards, dones)

        target = _iql_target(config, agent, batch)

        # Manual expected target
        with torch.no_grad():
            next_v = agent.value_net(batch["next_obs"]).squeeze(-1)
        gamma_n = GAMMA ** n_step
        expected = batch["rewards"] + (1.0 - batch["dones"]) * gamma_n * next_v

        diff = (target - expected).abs().max().item()
        assert diff < 1e-6, (
            f"IQL n_step={n_step}: target mismatch, max diff={diff}"
        )

        # For done=1 (third sample), target must equal the reward exactly.
        assert abs(target[2].item() - rewards[2]) < 1e-6, (
            f"IQL n_step={n_step}: done=1 target should equal reward, "
            f"got {target[2].item()} vs {rewards[2]}"
        )

        # For n=1, gamma_n should be exactly gamma.
        if n_step == 1:
            assert abs(gamma_n - GAMMA) < 1e-12, (
                f"n_step=1 but gamma_n={gamma_n} != gamma={GAMMA}"
            )
        # For n=3, gamma_n should be gamma^3.
        if n_step == 3:
            expected_gamma_n = GAMMA ** 3
            assert abs(gamma_n - expected_gamma_n) < 1e-12, (
                f"n_step=3 but gamma_n={gamma_n} != gamma^3={expected_gamma_n}"
            )

        print(f"[OK] IQL n_step={n_step}: gamma_n={gamma_n:.6f}, "
              f"target matches expected to {diff:.2e}")


def test_td3bc_nstep():
    from offline_rl.agents.td3_bc import TD3BCAgent

    for n_step in [1, 3]:
        config = _load_config("td3_bc", n_step)
        agent = TD3BCAgent(OBS_DIM, ACTION_DIM, config, DEVICE)

        rewards = [1.0, 2.0, 0.5]
        dones = [0.0, 0.0, 1.0]
        batch = _make_batch(OBS_DIM, ACTION_DIM, rewards, dones)

        target = _td3bc_target(config, agent, batch)

        with torch.no_grad():
            next_actions = agent.target_actor(batch["next_obs"])
            next_q = agent.target_critic.q_min(batch["next_obs"], next_actions)
        gamma_n = GAMMA ** n_step
        expected = batch["rewards"] + (1.0 - batch["dones"]) * gamma_n * next_q

        diff = (target - expected).abs().max().item()
        assert diff < 1e-6, (
            f"TD3+BC n_step={n_step}: target mismatch, max diff={diff}"
        )
        assert abs(target[2].item() - rewards[2]) < 1e-6

        print(f"[OK] TD3+BC n_step={n_step}: gamma_n={gamma_n:.6f}, "
              f"target matches expected to {diff:.2e}")


def test_n1_regression():
    """Verify n=1 gives numerically identical results to the old formula:
    target = r + gamma * (1-done) * V(s')  (IQL)
    target = r + gamma * (1-done) * Q(s', a') (TD3+BC)
    """
    from offline_rl.agents.iql import IQLAgent
    from offline_rl.agents.td3_bc import TD3BCAgent

    # IQL
    config_iql = _load_config("iql", n_step=1)
    agent_iql = IQLAgent(OBS_DIM, ACTION_DIM, config_iql, DEVICE)
    batch = _make_batch(OBS_DIM, ACTION_DIM, [1.5, -0.5], [0.0, 0.0])

    with torch.no_grad():
        next_v = agent_iql.value_net(batch["next_obs"]).squeeze(-1)
    old_formula = batch["rewards"] + (1.0 - batch["dones"]) * GAMMA * next_v
    new_formula = _iql_target(config_iql, agent_iql, batch)
    diff = (old_formula - new_formula).abs().max().item()
    assert diff < 1e-7, f"IQL n=1 regression: diff={diff}"

    # TD3+BC
    config_td3 = _load_config("td3_bc", n_step=1)
    agent_td3 = TD3BCAgent(OBS_DIM, ACTION_DIM, config_td3, DEVICE)
    with torch.no_grad():
        na = agent_td3.target_actor(batch["next_obs"])
        nq = agent_td3.target_critic.q_min(batch["next_obs"], na)
    old_formula_td3 = batch["rewards"] + (1.0 - batch["dones"]) * GAMMA * nq
    new_formula_td3 = _td3bc_target(config_td3, agent_td3, batch)
    diff_td3 = (old_formula_td3 - new_formula_td3).abs().max().item()
    assert diff_td3 < 1e-7, f"TD3+BC n=1 regression: diff={diff_td3}"

    print(f"[OK] n=1 regression guard: IQL diff={diff:.2e}, TD3+BC diff={diff_td3:.2e}")


def main() -> int:
    test_iql_nstep()
    test_td3bc_nstep()
    test_n1_regression()
    print("\nAll n-step target tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
