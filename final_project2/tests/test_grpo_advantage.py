"""Tests for ``online_rl.agents.grpo_advantages.group_relative_advantage``."""

from __future__ import annotations

import pytest
import torch

from online_rl.agents.grpo_advantages import group_relative_advantage


@pytest.mark.parametrize("mode", ["raw", "mean_only", "mean_std", "rank"])
def test_shape_preserved(mode):
    rewards = torch.randn(7, 5)
    out = group_relative_advantage(rewards, mode=mode)
    assert out.shape == rewards.shape


def test_raw_returns_unchanged():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [-0.5, 0.5, 0.5]])
    out = group_relative_advantage(rewards, mode="raw")
    torch.testing.assert_close(out, rewards)
    # Returned tensor must not be the same object (so callers can mutate freely).
    assert out.data_ptr() != rewards.data_ptr()


def test_mean_only_subtracts_per_row_mean():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = group_relative_advantage(rewards, mode="mean_only")
    torch.testing.assert_close(out.mean(dim=1), torch.zeros(2), atol=1e-7, rtol=0.0)
    expected = rewards - rewards.mean(dim=1, keepdim=True)
    torch.testing.assert_close(out, expected)


def test_mean_std_zscore_properties():
    """Per-row mean ≈ 0 and per-row std ≈ 1 (using unbiased=False to match impl)."""
    torch.manual_seed(0)
    rewards = torch.randn(8, 16) * 5.0 + 3.0  # arbitrary scale and shift
    out = group_relative_advantage(rewards, mode="mean_std")
    torch.testing.assert_close(out.mean(dim=1), torch.zeros(8), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(out.std(dim=1, unbiased=False), torch.ones(8), atol=1e-3, rtol=1e-3)


def test_rank_bounded_and_centered():
    torch.manual_seed(0)
    rewards = torch.randn(6, 9)
    out = group_relative_advantage(rewards, mode="rank")
    assert out.min().item() >= -1.0 - 1e-7
    assert out.max().item() <= 1.0 + 1e-7
    # Per-row ranks 0..G-1 centered are symmetric → mean exactly 0
    torch.testing.assert_close(out.mean(dim=1), torch.zeros(6), atol=1e-7, rtol=0.0)


@pytest.mark.parametrize("mode", ["mean_only", "mean_std", "rank"])
def test_constant_rewards_give_zero_advantage(mode):
    """No within-group signal → no learning signal; output must be zeros.

    ``raw`` mode is intentionally excluded — its contract is "return
    rewards unchanged", which is non-zero for constant rewards.
    """
    rewards = torch.full((4, 5), 2.7)
    out = group_relative_advantage(rewards, mode=mode)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-7, rtol=0.0)


def test_raw_does_not_collapse_constant_rows():
    """Documenting the ``raw`` non-property: constant rows pass through untouched."""
    rewards = torch.full((3, 4), 1.5)
    out = group_relative_advantage(rewards, mode="raw")
    torch.testing.assert_close(out, rewards)


def test_mean_std_robust_to_constant_row_no_nan():
    """Constant rows would div-by-zero without the eps guard."""
    rewards = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0],   # constant
         [1.0, 2.0, 3.0, 4.0]],  # varied
    )
    out = group_relative_advantage(rewards, mode="mean_std")
    assert torch.all(torch.isfinite(out))


def test_rank_handles_g_equals_one():
    rewards = torch.tensor([[1.0], [-2.0], [3.0]])
    out = group_relative_advantage(rewards, mode="rank")
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        group_relative_advantage(torch.zeros(4), mode="mean_std")
    with pytest.raises(ValueError):
        group_relative_advantage(torch.zeros(2, 3, 4), mode="mean_std")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        group_relative_advantage(torch.zeros(2, 3), mode="zscore")  # not a valid mode
