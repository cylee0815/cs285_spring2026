"""Tests for ``online_rl.agents.grpo.GRPOTrainer``.

Critical: the ``test_ref_actor_does_not_move_after_updates`` /
``test_actor_does_move_after_updates`` pair catches the most common
GRPO bug — accidentally letting the KL anchor drift along with the
trainable actor would silently disable the KL constraint.
"""

from __future__ import annotations

import numpy as np
import torch

from core.envs.portfolio_env import PortfolioEnv
from core.networks.policies import DirichletMLPPolicy
from online_rl.agents.grpo import GRPOConfig, GRPOTrainer


N_ASSETS = 4
N_FEATURES = 6
T = 200
EPISODE_LENGTH = 30


def _make_env(data_seed: int = 0) -> PortfolioEnv:
    rng = np.random.default_rng(data_seed)
    features = rng.standard_normal((T, N_FEATURES)).astype(np.float32)
    returns = (rng.standard_normal((T, N_ASSETS)) * 0.015 + 0.0002).astype(np.float32)
    return PortfolioEnv(
        features=features,
        forward_returns=returns,
        transaction_cost_lambda=0.001,
        episode_length=EPISODE_LENGTH,
    )


def _make_actor(seed: int = 0) -> DirichletMLPPolicy:
    torch.manual_seed(seed)
    return DirichletMLPPolicy(
        obs_dim=N_FEATURES, action_dim=N_ASSETS, hidden_dim=16, n_layers=2,
    )


def _make_trainer(*, group_size: int = 4, seed: int = 0) -> GRPOTrainer:
    cfg = GRPOConfig(
        group_size=group_size,
        advantage_norm="mean_std",
        beta_kl=0.01,
        clip_eps=0.2,
        epochs_per_batch=1,
        minibatch_size=8,
        lr=3e-4,
        grad_clip=1.0,
        entropy_coef=0.0,
    )
    actor = _make_actor(seed=seed)
    env = _make_env(data_seed=seed)
    return GRPOTrainer(actor, env, cfg, device="cpu", seed=seed)


def test_init_snapshots_and_freezes_ref_actor():
    trainer = _make_trainer()

    # ref_actor's parameters are frozen.
    for name, p in trainer.ref_actor.named_parameters():
        assert not p.requires_grad, f"ref_actor.{name} still has requires_grad=True"

    # ref_actor and actor start with identical state_dicts (deepcopy).
    ref_sd = trainer.ref_actor.state_dict()
    act_sd = trainer.actor.state_dict()
    assert set(ref_sd.keys()) == set(act_sd.keys())
    for k in ref_sd:
        assert torch.equal(ref_sd[k], act_sd[k]), f"init mismatch on {k}"

    # ref_actor is in eval mode.
    assert not trainer.ref_actor.training


def test_collect_returns_correct_shapes():
    G = 4
    N = 8
    trainer = _make_trainer(group_size=G)
    batch = trainer.collect(num_states=N)

    assert batch["states"].shape == (N, N_FEATURES)
    assert batch["actions"].shape == (N, G, N_ASSETS)
    assert batch["old_logprobs"].shape == (N, G)
    assert batch["rewards"].shape == (N, G)

    # Sampled actions live on the simplex.
    sums = batch["actions"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert (batch["actions"] >= 0).all()

    # No NaNs anywhere.
    for k, v in batch.items():
        assert torch.all(torch.isfinite(v)), f"non-finite in {k}"


def test_update_runs_and_produces_nonzero_grad_norm():
    trainer = _make_trainer(group_size=4)
    batch = trainer.collect(num_states=16)
    metrics = trainer.update(batch)

    assert "grad_norm" in metrics
    assert metrics["grad_norm"] > 0.0
    assert np.isfinite(metrics["loss"])


def test_actor_moves_but_ref_actor_does_not_after_many_updates():
    """The headline correctness test: actor learns; KL anchor stays put."""
    trainer = _make_trainer(group_size=4, seed=0)

    # Snapshot both at t=0.
    ref_initial = {k: v.clone() for k, v in trainer.ref_actor.state_dict().items()}
    actor_initial = {k: v.clone() for k, v in trainer.actor.state_dict().items()}

    # One batch, repeatedly update on it.
    batch = trainer.collect(num_states=16)
    for _ in range(100):
        trainer.update(batch)

    # ref_actor must be byte-identical to its initial state.
    for k, v in trainer.ref_actor.state_dict().items():
        assert torch.equal(v, ref_initial[k]), (
            f"ref_actor.{k} drifted after 100 update steps — KL anchor not anchoring"
        )

    # actor must have moved on at least one parameter.
    moved = False
    for k, v in trainer.actor.state_dict().items():
        if not torch.equal(v, actor_initial[k]):
            moved = True
            break
    assert moved, "actor did not move after 100 update steps — training loop is a no-op"


def test_collect_handles_episode_boundaries():
    """Spans more states than fit in one episode → collect must reset and continue."""
    G = 2
    # episode_length=30 in the env; ask for more than that.
    trainer = _make_trainer(group_size=G)
    batch = trainer.collect(num_states=40)
    assert batch["states"].shape == (40, N_FEATURES)
    assert torch.all(torch.isfinite(batch["rewards"]))


def test_metrics_dict_has_required_loss_keys():
    trainer = _make_trainer(group_size=4)
    batch = trainer.collect(num_states=16)
    metrics = trainer.update(batch)
    expected = {
        "loss", "surrogate", "kl_to_ref", "approx_kl_old", "clipfrac",
        "entropy", "adv_mean", "adv_std", "reward_mean", "ratio_hard_clipped",
        "grad_norm",
    }
    missing = expected - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"


# NOTE: a full end-to-end reproducibility test (same trainer seed → same
# rollout) would require the trainer to also seed the env's random-start
# RNG on first reset. Today the env uses an unseeded
# np.random.default_rng() if reset(seed=None). Keeping that out of scope
# for this stage; flag for cleanup if/when reproducibility-of-rollouts
# becomes a hard requirement.
