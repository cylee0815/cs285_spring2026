"""
Dirichlet Actor-Critic for portfolio allocation.

Mathematical motivation:
  Standard SAC with softmax-Gaussian policy optimizes:
    J_wrong = E[Q(s, softmax(a))] - τ * H_gaussian(a)
  where H_gaussian is entropy of the raw Gaussian on logits, NOT on the simplex.

  The correct objective on the simplex is:
    J_correct = E[Q(s, w)] - τ * H_dirichlet(w)
  where w ~ Dir(α(s)) and H[Dir(α)] = log B(α) - (α₀-K)ψ(α₀) + Σᵢ(αᵢ-1)ψ(αᵢ)

  PyTorch's Dirichlet distribution gives us exact log_prob and entropy on the simplex.
  Actor parameterizes α_i(s) = softplus(f_θ(s)_i) + 1 to ensure α_i > 1 (unimodal).
"""
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Dirichlet
from typing import Optional, Tuple

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(in_dim, hidden_dim, out_dim, n_layers, activation=nn.Tanh):
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers += [layer_init(nn.Linear(d, hidden_dim)), activation()]
        d = hidden_dim
    layers.append(layer_init(nn.Linear(d, out_dim), std=0.01))
    return nn.Sequential(*layers)


class DirichletActor(nn.Module):
    """
    Actor that parameterizes a Dirichlet distribution over portfolio weights.

    α_i(s) = softplus(f_θ(s)_i) + 1  → ensures α_i > 1 (unimodal Dirichlet)

    Output: portfolio weights w ~ Dir(α(s)) sampled via reparameterized Gamma trick.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, action_dim, n_layers)

    def get_distribution(self, obs: torch.Tensor) -> Dirichlet:
        alpha = torch.nn.functional.softplus(self.net(obs)) + 1.0  # (batch, n_assets), α > 1
        return Dirichlet(alpha)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Returns:
            w: portfolio weights (batch, n_assets) — on simplex, sum to 1
            log_prob: log π(w|s) — exact log-prob on simplex
            entropy: H[Dir(α(s))] — closed-form Dirichlet entropy
            alpha: concentration params (batch, n_assets)
        """
        dist = self.get_distribution(obs)
        if deterministic:
            w = dist.mean  # E[w] = α / α₀, already on simplex
        else:
            w = dist.rsample()  # reparameterized sample via Gamma trick
            w = w.clamp(1e-6, 1.0)  # numerical safety
            w = w / w.sum(dim=-1, keepdim=True)  # re-normalize after clamp
        log_prob = dist.log_prob(w)   # exact log-prob on simplex
        entropy = dist.entropy()      # closed-form: log B(α) - (α₀-K)ψ(α₀) + Σᵢ(αᵢ-1)ψ(αᵢ)
        return w, log_prob, entropy, dist.concentration


class DoubleCritic(nn.Module):
    """
    Two independent Q-networks for double Q-learning.
    Input: concatenation of [obs, action] (and optionally regime hidden state).
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = mlp(in_dim, hidden_dim, 1, n_layers)
        self.q2 = mlp(in_dim, hidden_dim, 1, n_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
