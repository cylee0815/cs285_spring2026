"""Value network: V(s) -> scalar value."""

from typing import Sequence

import torch
import torch.nn as nn

from models.mlp import build_mlp


class ValueNetwork(nn.Module):
    """State value function V(s)."""

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = build_mlp(state_dim, 1, hidden_sizes)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            state: (B, state_dim)

        Returns:
            Value tensor of shape (B, 1).
        """
        return self.net(state)
