"""
Vanilla CQL: Conservative Q-Learning without the Fisher-Rao geodesic penalty.

CQL penalty: alpha * (E_{a~pi}[Q(s,a)] - E_{a~D}[Q(s,a)])

This is a direct ablation of GeodesicCQL — same architecture (Dirichlet actor,
regime encoder), but the standard CQL penalty replaces the geodesic-scaled version.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from src.agents.cql_geodesic import GeodesicCQL
from src.agents.replay_buffer import ReplayBuffer


class VanillaCQLAgent(GeodesicCQL):
    """
    Vanilla CQL as a subclass of GeodesicCQL.
    Overrides only the critic update to use standard CQL penalty.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config,
        device: torch.device,
        offline_buffer: Optional[ReplayBuffer] = None,
    ):
        super().__init__(obs_dim, action_dim, config, device, offline_buffer)
        self.cql_n_samples = getattr(config, 'cql_n_samples', 10)

    def update_critic_geodesic_cql(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Standard CQL critic update (no Fisher-Rao scaling).

        L = L_Bellman + alpha * (E_{a~pi}[Q(s,a)] - E_{a~D}[Q(s,a)])
        """
        obs = batch['obs']
        actions = batch['actions']
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

        # --- Standard CQL penalty ---
        # Sample K actions from current policy, compute average Q
        K = self.cql_n_samples
        bsz = obs.shape[0]
        obs_rep = obs.unsqueeze(1).expand(-1, K, -1).reshape(-1, obs.shape[-1])
        regime_rep = regime.unsqueeze(1).expand(-1, K, -1).reshape(-1, regime.shape[-1])

        w_policy, _, _, _ = self.actor(obs_rep, regime_rep)
        q1_policy, q2_policy = self.critic(obs_rep, w_policy.detach(), regime_rep)
        q1_policy = q1_policy.reshape(bsz, K)
        q2_policy = q2_policy.reshape(bsz, K)

        # CQL penalty: E_pi[Q] - E_D[Q]
        cql_q1 = q1_policy.mean(dim=1).mean() - q1.mean()
        cql_q2 = q2_policy.mean(dim=1).mean() - q2.mean()
        cql_penalty = (cql_q1 + cql_q2) / 2.0

        total_loss = bellman_loss + self.config.cql_alpha * cql_penalty

        self.critic_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return {
            'critic/bellman_loss': bellman_loss.item(),
            'critic/cql_penalty': cql_penalty.item(),
            'critic/total_loss': total_loss.item(),
            'critic/q1_mean': q1.mean().item(),
        }

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        regime = torch.zeros(1, self.config.regime_dim, device=self.device)
        w, _, _, _ = self.actor(obs_t, regime, deterministic=True)
        return w.squeeze(0).cpu().numpy()
