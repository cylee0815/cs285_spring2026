"""Implicit Q-Learning (IQL) for offline reinforcement learning.

Reference:
    Kostrikov et al. (2022), "Offline Reinforcement Learning with
    Implicit Q-Learning", ICLR 2022.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from models.q_network import QNetwork
from models.value_network import ValueNetwork
from models.policy_network import PolicyNetwork


class IQL:
    """IQL agent with Q, V, and policy networks.

    Parameters
    ----------
    state_dim:
        Dimensionality of the state vector.
    action_dim:
        Dimensionality of the action vector.
    hidden_sizes:
        Hidden layer sizes for all networks.
    lr:
        Learning rate for all optimisers.
    gamma:
        Discount factor.
    tau:
        Expectile for value function regression.
    beta:
        Inverse temperature for advantage-weighted policy loss.
    polyak:
        Polyak averaging coefficient for the target value network.
    device:
        Torch device string.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.7,
        beta: float = 3.0,
        polyak: float = 0.005,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.polyak = polyak
        self.device = device

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.value_network = ValueNetwork(state_dim, hidden_sizes).to(device)
        self.value_target = copy.deepcopy(self.value_network)
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_sizes).to(device)

        # Freeze target parameters
        for p in self.value_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric squared loss: |tau - 1(diff < 0)| * diff^2."""
        weight = torch.where(diff < 0, 1 - self.tau, self.tau)
        return (weight * diff.pow(2)).mean()

    # ------------------------------------------------------------------
    # Update steps
    # ------------------------------------------------------------------

    def update_value(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Update V via expectile regression on Q(s,a) - V(s)."""
        with torch.no_grad():
            q_values = self.q_network(states, actions)  # (B, 1)
        v_values = self.value_network(states)  # (B, 1)
        diff = q_values - v_values
        loss = self.expectile_loss(diff)

        self.v_optimizer.zero_grad()
        loss.backward()
        self.v_optimizer.step()
        return loss.item()

    def update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Update Q via Bellman target using V_target(s')."""
        with torch.no_grad():
            v_target = self.value_target(next_states)  # (B, 1)
            target = rewards + self.gamma * (1 - dones) * v_target

        q_pred = self.q_network(states, actions)  # (B, 1)
        loss = nn.functional.mse_loss(q_pred, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        return loss.item()

    def update_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:
        """Update policy via advantage-weighted behavioral cloning."""
        with torch.no_grad():
            q_values = self.q_network(states, actions)
            v_values = self.value_network(states)
            advantages = q_values - v_values
            weights = torch.exp(advantages / self.beta).clamp(max=100.0)

        # Policy outputs softmax probabilities; compute log-prob of dataset actions.
        probs = self.policy_network(states)  # (B, action_dim)
        log_probs = torch.log(probs + 1e-8)
        # Log-likelihood of the dataset action under the current policy.
        # actions are portfolio weights on the simplex — treat as categorical-like.
        log_pi = (log_probs * actions).sum(dim=-1, keepdim=True)  # (B, 1)

        loss = -(weights * log_pi).mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.item()

    def update_target_value(self) -> None:
        """Polyak-average the value network into the target."""
        with torch.no_grad():
            for p, p_target in zip(
                self.value_network.parameters(),
                self.value_target.parameters(),
            ):
                p_target.data.mul_(1 - self.polyak).add_(p.data, alpha=self.polyak)

    # ------------------------------------------------------------------
    # Full update step
    # ------------------------------------------------------------------

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, float]:
        """Run one full IQL update and return loss metrics."""
        v_loss = self.update_value(states, actions)
        q_loss = self.update_q(states, actions, rewards, next_states, dones)
        policy_loss = self.update_policy(states, actions)
        self.update_target_value()
        return {
            "v_loss": v_loss,
            "q_loss": q_loss,
            "policy_loss": policy_loss,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return deterministic portfolio weights for a single state.

        Parameters
        ----------
        state:
            Shape ``(state_dim,)`` or ``(1, state_dim)``.

        Returns
        -------
        action:
            Shape ``(action_dim,)``.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        weights = self.policy_network(state)
        return weights.squeeze(0)
