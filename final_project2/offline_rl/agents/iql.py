"""
Implicit Q-Learning (IQL) for offline portfolio optimization.

Key insight: IQL avoids querying Q on out-of-distribution actions by learning
a value function V(s) via expectile regression against Q(s,a), then extracting
a policy via advantage-weighted behavioral cloning.

Three networks:
  Q(s,a) — double critic, standard Bellman with V as target
  V(s)   — expectile regression: L2_tau(Q(s,a) - V(s))
  pi(a|s) — advantage-weighted extraction: exp((Q-V)/beta) * log pi(a|s)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional

from core.networks.dirichlet_policy import DirichletActor, DoubleCritic, mlp
from core.buffers.replay_buffer import ReplayBuffer


class IQLAgent:
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
        self.offline_buffer = offline_buffer
        self.n_step = getattr(config, 'n_step', 1)

        # Actor (Dirichlet for simplex-native policy)
        self.actor = DirichletActor(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)

        # Double Q-function
        self.critic = DoubleCritic(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Value function V(s) — no action input
        self.value_net = nn.Sequential(
            *mlp(obs_dim, config.hidden_dim, 1, config.n_layers).children()
        ).to(device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr)
        self.value_opt = optim.Adam(self.value_net.parameters(), lr=config.lr)

    def _expectile_loss(self, diff: torch.Tensor, tau: float) -> torch.Tensor:
        weight = torch.where(diff >= 0, tau, 1.0 - tau)
        return (weight * diff.pow(2)).mean()

    def update_value(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """V(s) update via expectile regression against Q(s,a)."""
        obs = batch['obs']
        actions = batch['actions']

        with torch.no_grad():
            q1, q2 = self.target_critic(obs, actions)
            q = torch.min(q1, q2)

        v = self.value_net(obs).squeeze(-1)
        value_loss = self._expectile_loss(q - v, self.config.iql_tau)

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
        self.value_opt.step()

        return {
            'iql/value_loss': value_loss.item(),
            'iql/v_mean': v.mean().item(),
        }

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Q(s,a) update using V(s') as Bellman target (no policy sampling)."""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        with torch.no_grad():
            next_v = self.value_net(next_obs).squeeze(-1)
            gamma_n = self.config.gamma ** self.n_step
            target_q = rewards + (1.0 - dones) * gamma_n * next_v

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return {
            'iql/critic_loss': critic_loss.item(),
            'iql/q_mean': q1.mean().item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Advantage-weighted policy extraction."""
        obs = batch['obs']
        actions = batch['actions']

        with torch.no_grad():
            q1, q2 = self.target_critic(obs, actions)
            q = torch.min(q1, q2)
            v = self.value_net(obs).squeeze(-1)
            advantage = q - v
            weights = torch.exp(advantage / self.config.iql_beta)
            weights = weights.clamp(max=self.config.iql_max_weight)

        # Buffer stores the env's executed weights (already on the simplex);
        # this clamp+renormalize is a numerical safety net, not a transform.
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
            'iql/actor_loss': actor_loss.item(),
            'iql/advantage_mean': advantage.mean().item(),
            'iql/weight_mean': weights.mean().item(),
            'iql/entropy': dist.entropy().mean().item(),
        }

    def update_target_critic(self):
        tau = self.config.polyak_tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self) -> Dict[str, float]:
        batch = self.offline_buffer.sample(self.config.batch_size)
        metrics = {}
        metrics.update(self.update_value(batch))
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        self.update_target_critic()
        return metrics

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        w, _, _, _ = self.actor(obs_t, deterministic=True)
        return w.squeeze(0).cpu().numpy()
