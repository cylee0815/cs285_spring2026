"""
Advantage Weighted Actor-Critic (AWAC) for offline portfolio optimization.

Actor loss: -E[exp(A(s,a)/beta) * log pi(a|s)]
  where A(s,a) = Q(s,a) - V(s)
  and V(s) = E_{a~pi}[Q(s,a)] estimated via K Monte Carlo samples.

AWAC constrains the policy to stay close to the behavioral distribution
by weighting the BC objective by exponentiated advantages.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional

from src.networks.dirichlet_policy import DirichletActor, DoubleCritic
from src.agents.replay_buffer import ReplayBuffer


class AWACAgent:
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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.offline_buffer = offline_buffer

        self.actor = DirichletActor(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)

        self.critic = DoubleCritic(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr)

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        with torch.no_grad():
            # V(s') via K samples from current policy
            K = self.config.awac_n_action_samples
            bsz = next_obs.shape[0]
            next_obs_rep = next_obs.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.obs_dim)
            next_w, _, _, _ = self.actor(next_obs_rep)
            next_q = self.target_critic.q_min(next_obs_rep, next_w)
            next_v = next_q.reshape(bsz, K).mean(dim=1)

            target_q = rewards + (1.0 - dones) * self.config.gamma * next_v

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return {
            'awac/critic_loss': critic_loss.item(),
            'awac/q_mean': q1.mean().item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']

        with torch.no_grad():
            # Compute advantage A = Q(s, a_behavior) - V(s)
            q1, q2 = self.target_critic(obs, actions)
            q_behavior = torch.min(q1, q2)

            # V(s) via K samples
            K = self.config.awac_n_action_samples
            bsz = obs.shape[0]
            obs_rep = obs.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.obs_dim)
            w_sampled, _, _, _ = self.actor(obs_rep)
            q_sampled = self.target_critic.q_min(obs_rep, w_sampled)
            v = q_sampled.reshape(bsz, K).mean(dim=1)

            advantage = q_behavior - v
            weights = torch.exp(advantage / self.config.awac_beta)
            weights = weights.clamp(max=100.0)
            weights = weights / weights.sum() * bsz  # normalize

        # Weighted BC
        behavioral_w = actions.clamp(1e-7, 1.0)
        behavioral_w = behavioral_w / behavioral_w.sum(dim=-1, keepdim=True)

        dist = self.actor.get_distribution(obs)
        log_prob = dist.log_prob(behavioral_w)
        actor_loss = -(weights * log_prob).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()

        return {
            'awac/actor_loss': actor_loss.item(),
            'awac/advantage_mean': advantage.mean().item(),
            'awac/weight_mean': weights.mean().item(),
            'awac/entropy': dist.entropy().mean().item(),
        }

    def update_target_critic(self):
        tau = self.config.polyak_tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self) -> Dict[str, float]:
        batch = self.offline_buffer.sample(self.config.batch_size)
        metrics = {}
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        self.update_target_critic()
        return metrics

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        w, _, _, _ = self.actor(obs_t, deterministic=True)
        return w.squeeze(0).cpu().numpy()
