"""
Geodesic-CQL: Conservative Q-Learning with Fisher-Rao simplex geometry.

Standard CQL applies a uniform conservative penalty to OOD actions.
Geodesic-CQL scales this penalty by the Fisher-Rao distance between
the current policy's mean and the behavioral policy's action:

  L_CQL = L_Bellman + β * d_FR(μ_θ(s), a_behavior) * (E_{a~π_θ}[Q(s,a)] - Q(s, a_behavior))

Motivation: the Fisher-Rao metric is the *natural* Riemannian metric on the probability
simplex (the Fisher information metric for categorical distributions). It measures the
true geometric distance between two portfolio allocations on the simplex, unlike Euclidean
distance on logits which has no geometric meaning.

Fisher-Rao distance: d_FR(w, w') = 2 * arccos(Σᵢ √(wᵢ * w'ᵢ))

Properties:
  - d_FR(w, w') = 0 iff w = w' (identity)
  - Invariant to reparameterization of the simplex
  - d_FR(uniform, w) = 2 * arccos(√(1/K)) ≈ the maximum OOD penalty
  - d_FR(w, w') ≈ √(χ²(w||w')) for nearby distributions (connects to chi-squared divergence)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional, Tuple
import gymnasium as gym

from src.networks.dirichlet_policy import DirichletActor, DoubleCritic
from src.networks.regime_encoder import RegimeEncoder, RegimeConditionedActor, RegimeConditionedCritic
from src.agents.replay_buffer import ReplayBuffer


def fisher_rao_distance(w1: torch.Tensor, w2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute the Fisher-Rao geodesic distance between portfolio allocations.

    d_FR(w, w') = 2 * arccos(Σᵢ √(wᵢ * w'ᵢ))

    This is the Bhattacharyya angle, also equal to the geodesic distance on the
    positive orthant of the unit sphere (via the map wᵢ → √wᵢ).

    Args:
        w1, w2: (batch, n_assets) — portfolio weights, sum to 1
    Returns:
        distance: (batch,) — Fisher-Rao distance, in [0, π]
    """
    # Map to positive orthant of unit sphere
    sqrt_w1 = torch.sqrt(w1.clamp(eps, 1.0))
    sqrt_w2 = torch.sqrt(w2.clamp(eps, 1.0))
    # Dot product (Bhattacharyya coefficient)
    dot = (sqrt_w1 * sqrt_w2).sum(dim=-1)
    dot = dot.clamp(-1.0 + eps, 1.0 - eps)  # numerical safety for arccos
    return 2.0 * torch.acos(dot)


class GeodesicCQL:
    """
    Offline Geodesic-CQL agent for offline pre-training phase.

    Uses Dirichlet policy + regime encoder + Fisher-Rao-scaled CQL penalty.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config,
        device: torch.device,
        offline_buffer: Optional[ReplayBuffer] = None,
    ):
        self.config = config
        self.device = device
        self.action_dim = action_dim

        # Regime encoder
        self.regime_encoder = RegimeEncoder(
            obs_dim=obs_dim,
            regime_dim=config.regime_dim,
            window_len=config.regime_window,
        ).to(device)

        regime_dim = config.regime_dim

        # Actor and double critic (regime-conditioned)
        self.actor = RegimeConditionedActor(
            obs_dim=obs_dim, action_dim=action_dim,
            regime_dim=regime_dim, hidden_dim=config.hidden_dim,
        ).to(device)

        self.critic = RegimeConditionedCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            regime_dim=regime_dim, hidden_dim=config.hidden_dim,
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Learnable temperature
        self.target_entropy = np.log(action_dim)
        self.log_temperature = nn.Parameter(torch.zeros(1, device=device))

        # Offline buffer
        self.offline_buffer = offline_buffer

        # Optimizers
        self.actor_opt = optim.Adam(
            list(self.actor.parameters()) + list(self.regime_encoder.parameters()),
            lr=config.lr
        )
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr)
        self.temp_opt = optim.Adam([self.log_temperature], lr=config.lr)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def update_critic_geodesic_cql(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Critic update with Geodesic-CQL penalty.

        L = L_Bellman + β * E_s[d_FR(μ_θ(s), a_behavior) * (E_{a~π_θ}[Q(s,a)] - Q(s, a_behavior))]

        The Fisher-Rao scaling means: if the current policy's mean is close to the
        behavioral action on the simplex (small d_FR), apply minimal penalty; if the
        policy wants to explore far from behavioral data, apply proportional penalty.
        """
        obs = batch['obs']
        actions = batch['actions']  # behavioral policy actions from dataset
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        obs_seq = batch.get('obs_seq')
        next_obs_seq = batch.get('next_obs_seq')

        # Regime context
        if obs_seq is not None:
            regime, _ = self.regime_encoder(obs_seq)
            next_regime, _ = self.regime_encoder(next_obs_seq)
        else:
            bsz = obs.shape[0]
            regime = torch.zeros(bsz, self.config.regime_dim, device=self.device)
            next_regime = torch.zeros(bsz, self.config.regime_dim, device=self.device)

        # --- Bellman target ---
        with torch.no_grad():
            next_w, next_log_prob, _, _ = self.actor(next_obs, next_regime)
            target_q = self.target_critic.q_min(next_obs, next_w, next_regime)
            target_q = rewards + (1.0 - dones) * self.config.gamma * (
                target_q - self.temperature.detach() * next_log_prob
            )

        q1, q2 = self.critic(obs, actions, regime)
        bellman_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        # --- Geodesic-CQL penalty ---
        # Current policy's mean on simplex: E[Dir(α)] = α / α₀
        w_policy, _, _, alpha = self.actor(obs, regime)
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        policy_mean = alpha / alpha_0  # (batch, n_assets) — Dirichlet mean on simplex

        # Behavioral action is already portfolio weights (on simplex) from dataset
        behavioral_action = actions.clamp(1e-7, 1.0)
        behavioral_action = behavioral_action / behavioral_action.sum(dim=-1, keepdim=True)

        # Fisher-Rao distance between policy mean and behavioral action
        d_fr = fisher_rao_distance(policy_mean, behavioral_action)  # (batch,)

        # Q-values under current policy vs behavioral action
        q_policy_min = self.critic.q_min(obs, w_policy.detach(), regime)
        q_behavioral = torch.min(q1, q2).detach()

        # Geodesic-CQL penalty: d_FR-weighted Q gap
        # Only penalize when policy Q > behavioral Q (the OOD overestimation case)
        q_gap = (q_policy_min - q_behavioral).clamp(min=0.0)
        cql_penalty = (d_fr * q_gap).mean()

        total_loss = bellman_loss + self.config.cql_alpha * cql_penalty

        self.critic_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return {
            'critic/bellman_loss': bellman_loss.item(),
            'critic/cql_penalty': cql_penalty.item(),
            'critic/total_loss': total_loss.item(),
            'critic/fisher_rao_dist': d_fr.mean().item(),
            'critic/q1_mean': q1.mean().item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        obs_seq = batch.get('obs_seq')

        if obs_seq is not None:
            regime, _ = self.regime_encoder(obs_seq)
        else:
            regime = torch.zeros(obs.shape[0], self.config.regime_dim, device=self.device)

        for p in self.critic.parameters():
            p.requires_grad = False

        w, log_prob, entropy, alpha = self.actor(obs, regime)
        q_val = self.critic.q_min(obs, w, regime)

        actor_loss = (self.temperature.detach() * log_prob - q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.regime_encoder.parameters()),
            self.config.max_grad_norm
        )
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        return {
            'actor/loss': actor_loss.item(),
            'actor/entropy': entropy.mean().item(),
            'actor/alpha_mean': alpha.mean().item(),
            'actor/temperature': self.temperature.item(),
        }

    def update_temperature(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        obs_seq = batch.get('obs_seq')

        with torch.no_grad():
            if obs_seq is not None:
                regime, _ = self.regime_encoder(obs_seq)
            else:
                regime = torch.zeros(obs.shape[0], self.config.regime_dim, device=self.device)
            _, log_prob, _, _ = self.actor(obs, regime)

        temp_loss = -(self.log_temperature * (log_prob + self.target_entropy).detach()).mean()
        self.temp_opt.zero_grad()
        temp_loss.backward()
        self.temp_opt.step()
        return {'temp_loss': temp_loss.item()}

    def update_target_critic(self):
        tau = self.config.polyak_tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self) -> Dict[str, float]:
        """Sample from offline buffer and update all components."""
        batch = self.offline_buffer.sample_with_context(self.config.batch_size)
        metrics = {}
        metrics.update(self.update_critic_geodesic_cql(batch))
        metrics.update(self.update_actor(batch))
        metrics.update(self.update_temperature(batch))
        self.update_target_critic()
        return metrics

    def get_policy_state(self) -> Dict:
        """Return transferable policy state for O2O initialization."""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'regime_encoder': self.regime_encoder.state_dict(),
            'log_temperature': self.log_temperature.data,
        }
