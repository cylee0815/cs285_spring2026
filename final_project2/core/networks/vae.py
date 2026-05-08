"""
VAE and Perturbation Network for BCQ (Batch-Constrained Q-Learning).

The VAE learns the behavioral action distribution conditioned on state:
  encoder: (obs, action) -> z ~ N(mu, sigma)
  decoder: (obs, z) -> logits -> softmax -> portfolio weights

The perturbation network shifts VAE-generated candidates:
  output: softmax(logits + phi * delta_logits)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from core.networks.dirichlet_policy import mlp


class VAE(nn.Module):
    """
    Conditional VAE for generating behavioral action candidates.
    Encoder maps (obs, action) to latent z; decoder maps (obs, z) to actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: (obs, action) -> (mu, log_sigma)
        self.encoder = mlp(obs_dim + action_dim, hidden_dim, latent_dim * 2, n_layers)

        # Decoder: (obs, z) -> action logits
        self.decoder = mlp(obs_dim + latent_dim, hidden_dim, action_dim, n_layers)

    def encode(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        h = self.encoder(x)
        mu, log_sigma = h.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-4, 15)
        return mu, log_sigma

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode to portfolio weights (on simplex)."""
        x = torch.cat([obs, z], dim=-1)
        logits = self.decoder(x)
        return torch.softmax(logits, dim=-1)

    def decode_logits(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode to raw logits (before softmax)."""
        x = torch.cat([obs, z], dim=-1)
        return self.decoder(x)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Forward pass with reparameterized sampling.
        Returns: recon_action, mu, log_sigma
        """
        mu, log_sigma = self.encode(obs, action)
        z = mu + torch.randn_like(mu) * log_sigma.exp()
        recon = self.decode(obs, z)
        return recon, mu, log_sigma

    def sample(self, obs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample candidate actions from prior z ~ N(0, I)."""
        bsz = obs.shape[0]
        obs_rep = obs.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.obs_dim)
        z = torch.randn(bsz * n_samples, self.latent_dim, device=obs.device)
        candidates = self.decode(obs_rep, z)
        return candidates.reshape(bsz, n_samples, self.action_dim)

    def sample_logits(self, obs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample candidate logits (before softmax) from prior."""
        bsz = obs.shape[0]
        obs_rep = obs.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.obs_dim)
        z = torch.randn(bsz * n_samples, self.latent_dim, device=obs.device)
        logits = self.decode_logits(obs_rep, z)
        return logits.reshape(bsz, n_samples, self.action_dim)


class PerturbationNetwork(nn.Module):
    """
    Shifts VAE-generated candidate actions to maximize Q-value.
    output = softmax(logits + phi * tanh(net(obs, action)))
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        phi: float = 0.05,
    ):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden_dim, action_dim, n_layers)
        self.phi = phi

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Perturb action (portfolio weights) and re-normalize to simplex."""
        x = torch.cat([obs, action], dim=-1)
        delta = self.phi * torch.tanh(self.net(x))
        # Perturb in weight space and re-project onto simplex
        perturbed = (action + delta).clamp(1e-7, None)
        return perturbed / perturbed.sum(dim=-1, keepdim=True)
