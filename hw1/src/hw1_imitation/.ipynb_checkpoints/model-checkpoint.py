"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        out_dim = chunk_size * action_dim
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        state: (B, state_dim)
        action_chunk: (B, chunk_size, action_dim)
        """
        # raise NotImplementedError
        B = state.shape[0]

        pred = self.net(state)  # (B, chunk_size * action_dim)
        target = action_chunk.view(B, -1)

        loss = torch.mean((pred - target) ** 2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Returns: (B, chunk_size, action_dim)
        """
        # raise NotImplementedError
        B = state.shape[0]

        with torch.no_grad():
            pred = self.net(state)  # (B, chunk_size * action_dim)
            actions = pred.view(B, self.chunk_size, self.action_dim)

        return actions


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.flat_action_dim = chunk_size * action_dim

        in_dim = state_dim + self.flat_action_dim + 1  # +1 for tau

        layers = []
        cur_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur_dim, h))
            layers.append(nn.ReLU())
            cur_dim = h

        layers.append(nn.Linear(cur_dim, self.flat_action_dim))

        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements flow matching loss:
        || v_theta(o, A_tau, tau) - (A - A0) ||^2
        """
        # raise NotImplementedError
        B = state.shape[0]
        device = state.device

        # Flatten expert chunk
        A = action_chunk.view(B, -1)  # (B, D)

        # Sample noise
        A0 = torch.randn_like(A)

        # Sample tau ~ Uniform(0, 1)
        tau = torch.rand(B, device=device)  # (B,)
        tau_col = tau.unsqueeze(1)          # (B, 1)

        # Interpolate
        A_tau = tau_col * A + (1.0 - tau_col) * A0  # (B, D)

        # Network input
        net_in = torch.cat([state, A_tau, tau_col], dim=1)

        v_pred = self.net(net_in)  # (B, D)

        v_target = A - A0

        loss = torch.mean((v_pred - v_target) ** 2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Euler integrate from tau=0 to 1.
        """
        # raise NotImplementedError
        B = state.shape[0]
        device = state.device
        D = self.flat_action_dim

        # Initial noise
        A = torch.randn(B, D, device=device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            tau = torch.full((B,), i / num_steps, device=device)
            tau_col = tau.unsqueeze(1)

            net_in = torch.cat([state, A, tau_col], dim=1)
            v = self.net(net_in)

            # Euler step
            A = A + dt * v

        # Reshape to chunk
        actions = A.view(B, self.chunk_size, self.action_dim)
        return actions
        
PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
