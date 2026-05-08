"""
Continuous GRPO (Group Relative Policy Optimization) for portfolio allocation.

Drops the value critic by computing advantages from a group of G actions
sampled at the same state, exploiting the env's exogeneity property
(see ``tests/test_exogeneity.py``). The bootstrapped value term cancels in
action-relative advantages, so within-group reward differences are the
only signal needed.

This module is built in three layers:
  - GRPOConfig          — hyperparameter container (dataclass)
  - grpo_loss           — pure loss function with PPO-style clip + KL anchor
  - GRPOTrainer         — collect/update loop tying actor + env together
                          (added in a later commit)

The actor must implement ``actor.dist(obs) -> torch.distributions.Dirichlet``
(see ``core/networks/policies.py::DirichletMLPPolicy``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import kl_divergence

from online_rl.agents.grpo_advantages import group_relative_advantage


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO.

    Attributes
    ----------
    group_size:
        Number of actions sampled per state (G).
    advantage_norm:
        Normalization mode for within-group rewards. One of
        ``"raw" | "mean_only" | "mean_std" | "rank"``.
    beta_kl:
        Coefficient on the KL-to-reference-policy anchor term.
    clip_eps:
        PPO-style importance-ratio clip threshold.
    epochs_per_batch:
        Number of full passes over the collected batch per ``update()``.
    minibatch_size:
        Minibatch size (in *states*, not state-action pairs).
    lr:
        Adam learning rate.
    grad_clip:
        Global gradient-norm clip.
    entropy_coef:
        Optional entropy bonus coefficient. Default 0 — Dirichlet entropy
        is closed-form but PPO-style entropy bonuses tend to be unnecessary
        for GRPO since the group-sampling already injects exploration.
    """

    group_size: int = 16
    advantage_norm: str = "mean_std"
    beta_kl: float = 0.01
    clip_eps: float = 0.2
    epochs_per_batch: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    grad_clip: float = 1.0
    entropy_coef: float = 0.0


# Bounds on the importance ratio. The PPO clip in the surrogate is the
# soft constraint; this hard clamp is belt-and-suspenders against
# pathological ratios on the first epoch of an update phase (e.g. when lr
# is too high and old/new log-probs diverge by tens of nats). Logged in
# the metrics dict as ``ratio_hard_clipped``.
_LOG_RATIO_LO = math.log(1e-3)
_LOG_RATIO_HI = math.log(1e3)


def grpo_loss(
    actor: torch.nn.Module,
    ref_actor: torch.nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    cfg: GRPOConfig | Any,
) -> tuple[torch.Tensor, dict]:
    """Compute the GRPO loss for one minibatch.

    Parameters
    ----------
    actor:
        Gradient-tracking policy. Must expose ``actor.dist(obs)`` returning
        a ``torch.distributions.Dirichlet`` whose ``concentration`` carries
        gradient on actor parameters.
    ref_actor:
        Frozen reference policy snapshotted at training start (KL anchor).
        Its parameters must have ``requires_grad=False``; this is the
        responsibility of the trainer, not this function.
    states:
        ``[B, d]`` minibatch of states.
    actions:
        ``[B, G, n_assets]`` actions previously sampled at each state.
    old_logprobs:
        ``[B, G]`` log-probs recorded at *sampling time* (no grad). These
        are the importance-ratio denominator; we never reach back through
        a network to recompute them.
    rewards:
        ``[B, G]`` per-(state, sample) rewards from
        ``env.peek_reward_batch``.
    cfg:
        ``GRPOConfig`` (or an attr-compatible object such as
        ``ml_collections.ConfigDict``).

    Returns
    -------
    loss:
        Scalar tensor with ``requires_grad=True``.
    metrics:
        Diagnostics dict with keys: ``loss``, ``surrogate``, ``kl_to_ref``,
        ``approx_kl_old``, ``clipfrac``, ``entropy``, ``adv_mean``,
        ``adv_std``, ``reward_mean``, ``ratio_hard_clipped``.

    Notes
    -----
    Per-state averaging: rewards/log-probs/ratios are averaged within a
    group (mean over G) first, then over states (mean over B). Equivalent
    to a flat mean only when group sizes are uniform; matters once
    prioritized or variable-G sampling is added.
    """
    # New log-probs via path (b) from the C1 investigation: one MLP forward
    # on [B, d], single batched Dirichlet, then log_prob of [G, B, N] with
    # transposes. ~G× cheaper than reshape-and-call-evaluate_actions.
    dist_new = actor.dist(states)
    actions_T = actions.transpose(0, 1).contiguous()                # [G, B, N]
    new_logp = dist_new.log_prob(actions_T).transpose(0, 1)         # [B, G]
    entropy_per_state = dist_new.entropy()                           # [B]

    # Importance ratio in log-space, hard-clamped before exp to prevent
    # NaN propagation under pathological divergence.
    log_ratio = new_logp - old_logprobs                              # [B, G]
    hard_clipped_mask = (log_ratio < _LOG_RATIO_LO) | (log_ratio > _LOG_RATIO_HI)
    ratio_hard_clipped = int(hard_clipped_mask.sum().item())
    log_ratio_clamped = log_ratio.clamp(_LOG_RATIO_LO, _LOG_RATIO_HI)
    ratio = log_ratio_clamped.exp()                                  # [B, G]

    # Group-relative advantage; rewards never propagate gradients.
    adv = group_relative_advantage(rewards.detach(), mode=cfg.advantage_norm)

    # PPO-style clipped surrogate (negate because optimizer minimizes).
    surrogate1 = ratio * adv
    surrogate2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    surrogate_per = -torch.min(surrogate1, surrogate2)                # [B, G]
    # Per-state averaging: mean over G then over B.
    surrogate_loss = surrogate_per.mean(dim=1).mean()

    # KL anchor. ref_actor's params are frozen, so dist_ref carries no
    # grad; kl_divergence's gradient flows through dist_new.concentration
    # only.
    dist_ref = ref_actor.dist(states)
    kl_per_state = kl_divergence(dist_new, dist_ref)                 # [B]
    kl_to_ref = kl_per_state.mean()

    # Optional entropy bonus.
    entropy_loss = -float(cfg.entropy_coef) * entropy_per_state.mean()

    loss = surrogate_loss + float(cfg.beta_kl) * kl_to_ref + entropy_loss

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
        approx_kl_old = (old_logprobs - new_logp).mean()

    metrics = {
        "loss": loss.item(),
        "surrogate": surrogate_loss.item(),
        "kl_to_ref": kl_to_ref.item(),
        "approx_kl_old": approx_kl_old.item(),
        "clipfrac": clip_frac.item(),
        "entropy": entropy_per_state.mean().item(),
        "adv_mean": adv.mean().item(),
        "adv_std": adv.std().item(),
        "reward_mean": rewards.mean().item(),
        "ratio_hard_clipped": ratio_hard_clipped,
    }
    return loss, metrics
