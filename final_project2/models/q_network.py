"""Q-network: Q(s, a) -> scalar value."""

from typing import Sequence

import torch
import torch.nn as nn

from models.mlp import build_mlp


class QNetwork(nn.Module):
    """State-action value function Q(s, a)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = build_mlp(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for (state, action) pair.

        Args:
            state: (B, state_dim)
            action: (B, action_dim)

        Returns:
            Q-value tensor of shape (B, 1).
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
