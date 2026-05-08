"""Tests for ``online_rl.agents.grpo.grpo_loss``.

Specifically asserts: gradient flow on actor params, zero surrogate under
constant rewards, KL=0 when actor==ref, ratio=1 / clipfrac=0 right after
sync, KL term carries grad on the actor side, and the diagnostics dict
contains every required key.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from core.networks.policies import DirichletMLPPolicy
from online_rl.agents.grpo import GRPOConfig, grpo_loss


def _make_setup(B: int = 4, G: int = 4, obs_dim: int = 8, action_dim: int = 5,
                seed: int = 0):
    """Build (actor, ref_actor, states, actions, old_logprobs, rewards, cfg)."""
    torch.manual_seed(seed)
    actor = DirichletMLPPolicy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=16, n_layers=2,
    )
    ref = deepcopy(actor)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    states = torch.randn(B, obs_dim)

    # Sample G simplex-valid actions per state under the actor itself.
    with torch.no_grad():
        dist0 = actor.dist(states)             # batch_shape=(B,)
        actions = dist0.sample((G,))           # [G, B, N]
        actions = actions.transpose(0, 1).contiguous()  # [B, G, N]
        old_logp = dist0.log_prob(actions.transpose(0, 1)).transpose(0, 1)  # [B, G]

    rewards = torch.randn(B, G) * 0.01
    cfg = GRPOConfig(group_size=G)
    return actor, ref, states, actions, old_logp, rewards, cfg


def test_loss_finite_and_grad_flows_to_actor():
    actor, ref, s, a, lp, r, cfg = _make_setup()
    loss, metrics = grpo_loss(actor, ref, s, a, lp, r, cfg)

    assert torch.isfinite(loss)
    assert loss.requires_grad

    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in actor.parameters()
    )
    assert has_grad, "Expected non-zero gradients on actor parameters"


def test_constant_rewards_zero_surrogate_and_zero_advantage():
    actor, ref, s, a, lp, _, cfg = _make_setup()
    rewards_const = torch.full((s.shape[0], cfg.group_size), 0.5)

    _, metrics = grpo_loss(actor, ref, s, a, lp, rewards_const, cfg)

    assert abs(metrics["surrogate"]) < 1e-7
    assert abs(metrics["adv_mean"]) < 1e-7
    assert abs(metrics["adv_std"]) < 1e-7


def test_actor_equals_ref_implies_kl_to_ref_zero():
    """``ref_actor`` snapshotted from the *current* ``actor`` → KL must be ~0."""
    actor, _, s, a, lp, r, cfg = _make_setup()

    ref = deepcopy(actor)
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg.beta_kl = 0.05
    _, metrics = grpo_loss(actor, ref, s, a, lp, r, cfg)
    assert abs(metrics["kl_to_ref"]) < 1e-6


def test_old_equals_new_logp_implies_ratio_one_clipfrac_zero():
    """Right after sync (old_logp == new_logp): ratio=1 → clipfrac=0."""
    actor, ref, s, a, _, r, cfg = _make_setup()

    # Fresh log-probs from the current actor — these are exactly the
    # log-probs `grpo_loss` will recompute internally.
    with torch.no_grad():
        new_logp_now = actor.dist(s).log_prob(a.transpose(0, 1)).transpose(0, 1)

    _, metrics = grpo_loss(actor, ref, s, a, new_logp_now, r, cfg)
    assert metrics["clipfrac"] == 0.0
    # approx_kl_old = mean(old - new) = 0 (within fp noise).
    assert abs(metrics["approx_kl_old"]) < 1e-6


def test_kl_term_carries_grad_on_actor_side():
    """If the KL anchor's grad doesn't reach the actor, it doesn't anchor."""
    actor, _, s, a, lp, r, cfg = _make_setup(seed=0)

    # Use a *different* ref (different seed) so KL > 0 and we're not at the
    # KL=0 minimum (where ∂KL/∂θ_actor is zero by definition).
    torch.manual_seed(42)
    ref = DirichletMLPPolicy(
        obs_dim=s.shape[1], action_dim=a.shape[-1], hidden_dim=16, n_layers=2,
    )
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    cfg.beta_kl = 1.0
    cfg.entropy_coef = 0.0

    # Zero-reward path → surrogate is exactly zero, so any non-zero
    # gradient on actor params must come from the KL term.
    actor.zero_grad(set_to_none=True)
    loss, metrics = grpo_loss(actor, ref, s, a, lp, torch.zeros_like(r), cfg)
    assert metrics["kl_to_ref"] > 1e-8, "Sanity: perturbed ref should give KL>0"

    assert loss.requires_grad, "Loss tensor must carry grad"
    loss.backward()

    has_grad_from_kl = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in actor.parameters()
    )
    assert has_grad_from_kl, "KL term must produce gradients on actor params"


def test_metrics_contain_all_required_keys():
    actor, ref, s, a, lp, r, cfg = _make_setup()
    _, metrics = grpo_loss(actor, ref, s, a, lp, r, cfg)

    required = {
        "loss", "surrogate", "kl_to_ref", "approx_kl_old", "clipfrac",
        "entropy", "adv_mean", "adv_std", "reward_mean", "ratio_hard_clipped",
    }
    missing = required - set(metrics.keys())
    assert not missing, f"Missing diagnostic keys: {missing}"


def test_ratio_hard_clipped_diagnostic_counts_clamps():
    """When the recorded old_logp is wildly inconsistent, every entry hard-clips."""
    actor, ref, s, a, _, r, cfg = _make_setup()

    # Set old_logp such that log_ratio = +100 everywhere (well above log(1e3)).
    with torch.no_grad():
        new_logp = actor.dist(s).log_prob(a.transpose(0, 1)).transpose(0, 1)
    bogus_old = new_logp - 100.0

    _, metrics = grpo_loss(actor, ref, s, a, bogus_old, r, cfg)

    B, G = r.shape
    assert metrics["ratio_hard_clipped"] == B * G

    # And: with sane old_logp, almost nothing should be hard-clipped.
    _, metrics_sane = grpo_loss(actor, ref, s, a, new_logp, r, cfg)
    assert metrics_sane["ratio_hard_clipped"] == 0


def test_loss_accepts_ml_collections_configdict():
    """``cfg`` is duck-typed; ``ml_collections.ConfigDict`` must work."""
    pytest.importorskip("ml_collections")
    import ml_collections

    actor, ref, s, a, lp, r, _ = _make_setup()
    cfg = ml_collections.ConfigDict()
    cfg.group_size = 4
    cfg.advantage_norm = "mean_std"
    cfg.beta_kl = 0.01
    cfg.clip_eps = 0.2
    cfg.entropy_coef = 0.0

    loss, metrics = grpo_loss(actor, ref, s, a, lp, r, cfg)
    assert torch.isfinite(loss)
    assert "ratio_hard_clipped" in metrics
