"""
Ensemble Diversified Actor-Critic (EDAC) for offline portfolio optimization.

Uses N Q-networks (default 10) for stronger pessimism than DoubleCritic.
Actor update includes a diversity penalty encouraging Q-functions to
disagree on out-of-distribution actions.

Key: Q_min across all N critics provides a more conservative value estimate,
which is particularly important for noisy financial rewards.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional, List

from offline_rl.networks.dirichlet_policy import DirichletActor, mlp, layer_init
from offline_rl.agents.replay_buffer import ReplayBuffer


class EDACAgent:
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

        # Dirichlet actor
        self.actor = DirichletActor(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)

        # N independent Q-networks
        n_critics = config.n_critics
        in_dim = obs_dim + action_dim
        self.critics = nn.ModuleList([
            mlp(in_dim, config.hidden_dim, 1, config.n_layers)
            for _ in range(n_critics)
        ]).to(device)
        self.target_critics = copy.deepcopy(self.critics)
        for p in self.target_critics.parameters():
            p.requires_grad = False

        # Learnable temperature
        self.target_entropy = np.log(action_dim)
        self.log_temperature = nn.Parameter(torch.zeros(1, device=device))

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = optim.Adam(self.critics.parameters(), lr=config.lr)
        self.temp_opt = optim.Adam([self.log_temperature], lr=config.lr)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _all_q(self, critics, obs, action):
        """Compute Q-values from all critics. Returns (N, batch)."""
        x = torch.cat([obs, action], dim=-1)
        return torch.stack([c(x).squeeze(-1) for c in critics], dim=0)

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        with torch.no_grad():
            next_w, next_log_prob, _, _ = self.actor(next_obs)
            target_qs = self._all_q(self.target_critics, next_obs, next_w)  # (N, batch)
            target_q = target_qs.min(dim=0).values  # pessimistic
            gamma_n = self.config.gamma ** self.n_step
            target_q = rewards + (1.0 - dones) * gamma_n * (
                target_q - self.temperature.detach() * next_log_prob
            )

        # Update all N critics
        all_qs = self._all_q(self.critics, obs, actions)  # (N, batch)
        critic_loss = sum(
            nn.functional.mse_loss(all_qs[i], target_q)
            for i in range(len(self.critics))
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return {
            'edac/critic_loss': critic_loss.item(),
            'edac/q_mean': all_qs.mean().item(),
            'edac/q_std': all_qs.std(dim=0).mean().item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']

        for p in self.critics.parameters():
            p.requires_grad = False

        w, log_prob, entropy, alpha = self.actor(obs)
        all_qs = self._all_q(self.critics, obs, w)  # (N, batch)
        q_min = all_qs.min(dim=0).values

        # Standard SAC actor loss
        actor_loss = (self.temperature.detach() * log_prob - q_min).mean()

        # EDAC diversity penalty: encourage ensemble disagreement on policy actions
        # Penalize low std across critics (want them to disagree on OOD)
        q_std = all_qs.std(dim=0).mean()
        diversity_penalty = -self.config.edac_eta * q_std

        total_actor_loss = actor_loss + diversity_penalty

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()

        for p in self.critics.parameters():
            p.requires_grad = True

        return {
            'edac/actor_loss': total_actor_loss.item(),
            'edac/entropy': entropy.mean().item(),
            'edac/q_std_policy': q_std.item(),
            'edac/temperature': self.temperature.item(),
        }

    def update_temperature(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        with torch.no_grad():
            _, log_prob, _, _ = self.actor(obs)
        temp_loss = -(self.log_temperature * (log_prob + self.target_entropy).detach()).mean()
        self.temp_opt.zero_grad()
        temp_loss.backward()
        self.temp_opt.step()
        return {'edac/temp_loss': temp_loss.item()}

    def update_target_critics(self):
        tau = self.config.polyak_tau
        for p, tp in zip(self.critics.parameters(), self.target_critics.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self) -> Dict[str, float]:
        batch = self.offline_buffer.sample(self.config.batch_size)
        metrics = {}
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        metrics.update(self.update_temperature(batch))
        self.update_target_critics()
        return metrics

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        w, _, _, _ = self.actor(obs_t, deterministic=True)
        return w.squeeze(0).cpu().numpy()
