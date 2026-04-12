"""Policy network: pi(a|s) -> portfolio weights on the simplex."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import build_mlp


class PolicyNetwork(nn.Module):
    """Stochastic policy that outputs portfolio weights via softmax."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = build_mlp(state_dim, action_dim, hidden_sizes)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute portfolio weights.

        Args:
            state: (B, state_dim)

        Returns:
            Weight tensor of shape (B, action_dim), non-negative and summing to 1.
        """
        logits = self.net(state)
        return F.softmax(logits, dim=-1)
