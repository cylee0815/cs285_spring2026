"""End-to-end integration test: GRPOTrainer × real PortfolioEnv × real actor.

The Stage C unit tests in ``test_grpo_trainer.py`` use synthetic feature /
return arrays (4 assets, 6 features). This test catches integration issues
those unit tests miss: high-dimensional observations (216 features), 8-asset
simplex actions, and the realistic distribution of forward returns from
``datasets/real_dirichlet.npz`` — the same dataset used by every other
trainer in this repo.

We slice the first ~500 rows of the dataset for speed; that's plenty to
exercise multiple ``collect → update`` cycles and at least one episode
boundary (``episode_length=60``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from core.envs.portfolio_env import PortfolioEnv
from core.networks.policies import DirichletMLPPolicy
from online_rl.agents.grpo import GRPOConfig, GRPOTrainer


_DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "real_dirichlet.npz"


def _load_real_env(episode_length: int = 60, n_rows: int = 500) -> PortfolioEnv:
    """Build a PortfolioEnv from the real dataset, sliced for test speed."""
    if not _DATASET_PATH.exists():
        pytest.skip(f"real dataset not found at {_DATASET_PATH}")
    data = np.load(_DATASET_PATH, allow_pickle=True)
    states = data["states"].astype(np.float32)[:n_rows]
    fwd = data["forward_returns"].astype(np.float32)[: states.shape[0]]
    return PortfolioEnv(
        features=states,
        forward_returns=fwd,
        transaction_cost_lambda=0.001,
        episode_length=episode_length,
        include_prev_weights=False,  # required for GRPO exogeneity (Stage A)
    )


def test_grpo_trainer_end_to_end_on_real_env():
    """5 collect/update cycles on the real env. Asserts the headline invariants:
    finite metrics, ref_actor frozen, actor moved.
    """
    env = _load_real_env(episode_length=60, n_rows=500)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    assert obs_dim == 216 and action_dim == 8, (
        f"unexpected env shape: obs={obs_dim} action={action_dim}"
    )

    torch.manual_seed(42)
    actor = DirichletMLPPolicy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64, n_layers=2,
    )

    cfg = GRPOConfig(
        group_size=4,
        advantage_norm="mean_std",
        beta_kl=0.01,
        clip_eps=0.2,
        epochs_per_batch=2,
        minibatch_size=32,
        lr=3e-4,
        grad_clip=1.0,
        entropy_coef=0.0,
    )
    trainer = GRPOTrainer(actor, env, cfg, device="cpu", seed=42)

    # Snapshot ref_actor and actor params to check invariants after training.
    ref_initial = {k: v.clone() for k, v in trainer.ref_actor.state_dict().items()}
    actor_initial = {k: v.clone() for k, v in trainer.actor.state_dict().items()}

    for i in range(5):
        batch = trainer.collect(num_states=64)
        # Buffer shape sanity, on the real env's actual obs/action dims.
        assert batch["states"].shape == (64, obs_dim)
        assert batch["actions"].shape == (64, cfg.group_size, action_dim)
        assert batch["old_logprobs"].shape == (64, cfg.group_size)
        assert batch["rewards"].shape == (64, cfg.group_size)

        metrics = trainer.update(batch)
        for k, v in metrics.items():
            assert np.isfinite(v), f"non-finite metric on iter {i}: {k}={v}"

    # ref_actor must be byte-identical to its init — KL anchor never moves.
    for k, v in trainer.ref_actor.state_dict().items():
        assert torch.equal(v, ref_initial[k]), (
            f"ref_actor.{k} drifted after 5 iterations on real env"
        )

    # actor must have moved on at least one parameter — training isn't a no-op.
    moved = False
    for k, v in trainer.actor.state_dict().items():
        if not torch.equal(v, actor_initial[k]):
            moved = True
            break
    assert moved, "actor did not move after 5 iterations on real env"
