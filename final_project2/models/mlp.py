"""Generic MLP builder."""

from typing import Sequence

import torch.nn as nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int] = (256, 256),
    activation: type = nn.ReLU,
) -> nn.Sequential:
    """Build a feedforward MLP.

    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output.
        hidden_sizes: Tuple of hidden layer sizes.
        activation: Activation function class.

    Returns:
        An nn.Sequential MLP.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
