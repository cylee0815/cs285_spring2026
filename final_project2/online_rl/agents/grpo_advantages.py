"""
Group-relative advantage normalization for Continuous GRPO.

GRPO computes advantages by sampling G actions at the same state and
comparing their rewards within the group, instead of bootstrapping a value
function. The bootstrap term in the standard advantage cancels under the
exogeneity assumption (see ``tests/test_exogeneity.py``), so action-relative
advantages reduce to reward differences inside each group.

This module is intentionally separate from the trainer: the helper is a
pure tensor function with no module-level state, easy to unit-test, and
reusable from any algorithm that wants per-group baseline subtraction.
"""
from __future__ import annotations

from typing import Literal

import torch


AdvantageMode = Literal["raw", "mean_only", "mean_std", "rank"]
_VALID_MODES = ("raw", "mean_only", "mean_std", "rank")


def group_relative_advantage(
    rewards: torch.Tensor,
    mode: AdvantageMode = "mean_std",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize rewards within each group of G actions sampled at the same state.

    Parameters
    ----------
    rewards:
        ``[B, G]`` tensor of per-(state, sample) rewards. Must be 2-D.
    mode:
        - ``"raw"``: returns rewards unchanged. The "no-normalization" baseline
          for ablations; constant-reward rows do NOT collapse to zero.
        - ``"mean_only"``: subtract per-row mean. Constant-reward rows give zero.
        - ``"mean_std"``: full per-row z-score: ``(r - mean) / (std + eps)``.
          Default. Constant-reward rows give zero (the std denominator is
          guarded by ``eps``).
        - ``"rank"``: per-row rank-normalized to centered ``[-1, 1]``.
          Constant-reward rows give zero.
    eps:
        Added to the per-row std before division in ``mean_std`` mode (and
        used as the constant-row tolerance in ``rank`` mode). Default
        ``1e-8`` — small enough not to materially distort signal at typical
        reward scales (1e-3 to 1e-1).

    Returns
    -------
    advantages: ``[B, G]`` tensor on the same device/dtype as ``rewards``.
    """
    if rewards.dim() != 2:
        raise ValueError(f"Expected 2-D rewards [B, G], got shape {tuple(rewards.shape)}")
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {_VALID_MODES}")

    if mode == "raw":
        return rewards.clone()

    if mode == "mean_only":
        return rewards - rewards.mean(dim=1, keepdim=True)

    if mode == "mean_std":
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        return (rewards - mean) / (std + eps)

    # mode == "rank"
    return _rank_centered_per_row(rewards, eps)


def _rank_centered_per_row(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row rank-normalize to centered ``[-1, 1]``.

    Constant-reward rows are detected via per-row std and forced to zero;
    otherwise ``argsort.argsort()`` would return an arbitrary permutation
    centered, which is meaningless signal. With G=1 the function returns
    zeros (only one element per group; no relative ordering possible).
    """
    B, G = rewards.shape
    if G == 1:
        return torch.zeros_like(rewards)

    sorted_idx = rewards.argsort(dim=1)
    ranks = torch.empty_like(rewards)
    cols = torch.arange(G, device=rewards.device, dtype=rewards.dtype).expand(B, G)
    ranks.scatter_(1, sorted_idx, cols)

    centered = 2.0 * ranks / (G - 1) - 1.0

    is_const = rewards.std(dim=1, keepdim=True, unbiased=False) < eps
    return torch.where(is_const, torch.zeros_like(centered), centered)
