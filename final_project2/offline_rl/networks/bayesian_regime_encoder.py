"""
Bayesian Regime Encoder for uncertainty-aware regime detection.

Extends the deterministic RegimeEncoder with a variational posterior:
  q_phi(h_t | o_{t-k:t}) = N(mu_t, sigma_t^2)

The posterior variance sigma_t^2 captures epistemic uncertainty in regime
belief, enabling:
  1. Uncertainty-weighted CQL penalty: high Var[h] -> more conservative
  2. Thompson sampling exploration: sample h ~ q(h) for diverse behavior
  3. Interpretable regime uncertainty for paper figures

Training includes KL(posterior || prior) regularization to prevent
posterior collapse and encourage meaningful uncertainty estimates.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class BayesianRegimeEncoder(nn.Module):
    """
    Bayesian GRU-based regime encoder.

    Architecture:
      input_proj: obs -> projected obs
      gru: processes sequence -> deterministic hidden h_det
      posterior_head: h_det -> (mu, log_sigma)  [approximate posterior]
      prior_head: h_det -> (mu_prior, log_sigma_prior)  [learned prior]

    The prior is conditioned on the previous hidden state so it captures
    temporal structure; the posterior additionally sees the current obs.
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

        # Posterior head: deterministic GRU hidden -> (mu, log_sigma)
        self.posterior_head = nn.Sequential(
            nn.Linear(regime_dim, regime_dim),
            nn.Tanh(),
            nn.Linear(regime_dim, regime_dim * 2),  # mu and log_sigma
        )

        # Learned prior head (conditioned on GRU hidden from previous step)
        self.prior_head = nn.Sequential(
            nn.Linear(regime_dim, regime_dim),
            nn.Tanh(),
            nn.Linear(regime_dim, regime_dim * 2),
        )

    def _split_posterior(self, h_det: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split posterior head output into mean and std."""
        out = self.posterior_head(h_det)
        mu, log_sigma = out.chunk(2, dim=-1)
        # Clamp log_sigma for numerical stability
        log_sigma = log_sigma.clamp(-5.0, 2.0)
        sigma = log_sigma.exp()
        return mu, sigma

    def _split_prior(self, h_det: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split prior head output into mean and std."""
        out = self.prior_head(h_det)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-5.0, 2.0)
        sigma = log_sigma.exp()
        return mu, sigma

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq: (batch, seq_len, obs_dim)
            hidden: initial GRU hidden (gru_layers, batch, regime_dim) or None
            sample: if True, return reparameterized sample; else return mean

        Returns:
            regime_context: (batch, regime_dim) - sampled or mean regime vector
            regime_std: (batch, regime_dim) - posterior std (epistemic uncertainty)
            kl_loss: scalar - KL(posterior || prior) for regularization
            new_hidden: (gru_layers, batch, regime_dim)
        """
        x = self.input_proj(obs_seq)  # (batch, seq_len, regime_dim)
        out, new_hidden = self.gru(x, hidden)
        h_det = out[:, -1, :]  # last step deterministic hidden

        # Posterior q(h | o_{1:t})
        mu, sigma = self._split_posterior(h_det)

        # Prior p(h) conditioned on GRU hidden (learned temporal prior)
        mu_prior, sigma_prior = self._split_prior(h_det.detach())

        # KL divergence: KL(N(mu, sigma^2) || N(mu_prior, sigma_prior^2))
        kl = (
            torch.log(sigma_prior / sigma)
            + (sigma**2 + (mu - mu_prior)**2) / (2 * sigma_prior**2)
            - 0.5
        )
        kl_loss = kl.sum(dim=-1).mean()

        # Sample or use mean
        if sample and self.training:
            eps = torch.randn_like(mu)
            regime_context = mu + sigma * eps  # reparameterized sample
        else:
            regime_context = mu

        return regime_context, sigma, kl_loss, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.regime_dim, device=device)

    def encode_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step encode during online rollout.

        Args:
            obs: (batch, obs_dim)
            hidden: (gru_layers, batch, regime_dim)
            sample: if True, Thompson sampling (sample from posterior)

        Returns:
            regime_context: (batch, regime_dim)
            regime_std: (batch, regime_dim)
            new_hidden: (gru_layers, batch, regime_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.input_proj(obs).unsqueeze(1)  # (batch, 1, regime_dim)
        out, new_hidden = self.gru(x, hidden)
        h_det = out.squeeze(1)  # (batch, regime_dim)

        mu, sigma = self._split_posterior(h_det)

        if sample:
            eps = torch.randn_like(mu)
            regime_context = mu + sigma * eps
        else:
            regime_context = mu

        return regime_context, sigma, new_hidden


class BayesianRegimeConditionedActor(nn.Module):
    """Wraps DirichletActor to accept Bayesian regime context."""

    def __init__(self, obs_dim: int, action_dim: int, regime_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        from offline_rl.networks.dirichlet_policy import DirichletActor
        self.actor = DirichletActor(obs_dim + regime_dim, action_dim, hidden_dim, n_layers)
        self.regime_dim = regime_dim

    def forward(self, obs: torch.Tensor, regime: torch.Tensor, deterministic: bool = False):
        x = torch.cat([obs, regime], dim=-1)
        return self.actor(x, deterministic)


class BayesianRegimeConditionedCritic(nn.Module):
    """Wraps DoubleCritic to accept Bayesian regime context."""

    def __init__(self, obs_dim: int, action_dim: int, regime_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        from offline_rl.networks.dirichlet_policy import DoubleCritic
        self.critic = DoubleCritic(obs_dim + regime_dim, action_dim, hidden_dim, n_layers)
        self.regime_dim = regime_dim

    def forward(self, obs: torch.Tensor, action: torch.Tensor, regime: torch.Tensor):
        x = torch.cat([obs, regime], dim=-1)
        return self.critic(x, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor, regime: torch.Tensor):
        x = torch.cat([obs, regime], dim=-1)
        return self.critic.q_min(x, action)
