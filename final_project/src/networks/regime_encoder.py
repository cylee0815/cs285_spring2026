"""
Regime Encoder for market regime inference.

Formal POMDP framing:
  The market has a hidden regime z_t (e.g., low/high volatility, trend/mean-reversion).
  The agent maintains a belief state b_t ≈ p(z_t | o_{1:t}).

  We approximate b_t as the hidden state of a GRU processing recent observations:
    h_t = GRU(o_t, h_{t-1})     [regime belief update]
    Q(s_t, a_t) = f(s_t, h_t, a_t)   [regime-conditioned value function]
    π(a_t | s_t) = g(s_t, h_t)        [regime-conditioned policy]

  The Bellman backup propagates h through the transition:
    Q(s, a, h) = r + γ E_{s', h'}[V(s', h')]
  where h' = GRU(s', h) is updated with the new observation.

  During O2O fine-tuning, regime shift is detected via:
    KL_regime = KL(h_offline_dist || h_online_dist)
  and used to adaptively scale conservatism.
"""
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, Tuple


class RegimeEncoder(nn.Module):
    """
    GRU-based regime encoder.
    Processes a sliding window of observations to produce a regime context vector.
    """
    def __init__(
        self,
        obs_dim: int,
        regime_dim: int = 64,
        gru_layers: int = 1,
        window_len: int = 20,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.regime_dim = regime_dim
        self.gru_layers = gru_layers
        self.window_len = window_len

        # Input projection before GRU
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, regime_dim), nn.Tanh()
        )
        self.gru = nn.GRU(regime_dim, regime_dim, num_layers=gru_layers, batch_first=True)

    def forward(self, obs_seq: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq: (batch, seq_len, obs_dim)
            hidden: initial GRU hidden (gru_layers, batch, regime_dim) or None
        Returns:
            regime_context: (batch, regime_dim) — last hidden state
            new_hidden: (gru_layers, batch, regime_dim)
        """
        x = self.input_proj(obs_seq)  # (batch, seq_len, regime_dim)
        out, new_hidden = self.gru(x, hidden)
        regime_context = out[:, -1, :]  # use last step output
        return regime_context, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.regime_dim, device=device)

    def encode_step(self, obs: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step encode during online rollout."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.input_proj(obs).unsqueeze(1)  # (batch, 1, regime_dim)
        out, new_hidden = self.gru(x, hidden)
        return out.squeeze(1), new_hidden  # (batch, regime_dim), new_hidden


class RegimeConditionedActor(nn.Module):
    """Wraps DirichletActor to accept regime context as additional input."""
    def __init__(self, obs_dim: int, action_dim: int, regime_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        from src.networks.dirichlet_policy import DirichletActor
        self.actor = DirichletActor(obs_dim + regime_dim, action_dim, hidden_dim, n_layers)
        self.regime_dim = regime_dim

    def forward(self, obs: torch.Tensor, regime: torch.Tensor, deterministic: bool = False):
        x = torch.cat([obs, regime], dim=-1)
        return self.actor(x, deterministic)


class RegimeConditionedCritic(nn.Module):
    """Wraps DoubleCritic to accept regime context as additional input."""
    def __init__(self, obs_dim: int, action_dim: int, regime_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        from src.networks.dirichlet_policy import DoubleCritic
        self.critic = DoubleCritic(obs_dim + regime_dim, action_dim, hidden_dim, n_layers)
        self.regime_dim = regime_dim

    def forward(self, obs: torch.Tensor, action: torch.Tensor, regime: torch.Tensor):
        x = torch.cat([obs, regime], dim=-1)
        return self.critic(x, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor, regime: torch.Tensor):
        x = torch.cat([obs, regime], dim=-1)
        return self.critic.q_min(x, action)
