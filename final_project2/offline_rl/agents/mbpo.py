"""
Model-Based Policy Optimization (MBPO) for offline portfolio RL.

Pipeline:
  1. Train ensemble dynamics model on offline data
  2. Generate short synthetic rollouts (rollout_length=1 for portfolio stability)
  3. Train SAC-Dirichlet on mixed real + synthetic data

Uses short model rollouts to reduce model bias while augmenting the dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional

from offline_rl.networks.dirichlet_policy import DirichletActor, DoubleCritic
from offline_rl.networks.ensemble_dynamics import EnsembleDynamics
from offline_rl.agents.replay_buffer import ReplayBuffer


class MBPOAgent:
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
        self.n_step = getattr(config, 'n_step', 1)
        self._update_step = 0

        # Dynamics model
        self.dynamics = EnsembleDynamics(
            obs_dim, action_dim,
            n_ensemble=config.n_ensemble,
            n_elites=config.n_elites,
            hidden_dim=config.dynamics_hidden_dim,
            n_layers=config.dynamics_n_layers,
            lr=config.dynamics_lr,
            device=device,
        )

        # SAC components (actor + critic, no env interaction)
        self.actor = DirichletActor(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)

        self.critic = DoubleCritic(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.target_entropy = np.log(action_dim)
        self.log_temperature = nn.Parameter(torch.zeros(1, device=device))

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr)
        self.temp_opt = optim.Adam([self.log_temperature], lr=config.lr)

        # Model buffer for synthetic data
        self.model_buffer = ReplayBuffer(
            config.model_buffer_size, obs_dim, action_dim, device,
        )

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def _rollout_model(self):
        """Generate synthetic transitions via model rollouts."""
        batch = self.offline_buffer.sample(self.config.rollout_batch_size)
        obs = batch['obs']

        for _ in range(self.config.rollout_length):
            with torch.no_grad():
                w, _, _, _ = self.actor(obs)
                next_obs, reward = self.dynamics.predict(obs, w)

            # Project portfolio weight dimensions back to simplex
            next_weights = next_obs[:, :self.action_dim]
            next_weights = next_weights.clamp(1e-7, None)
            next_weights = next_weights / next_weights.sum(dim=-1, keepdim=True)
            next_obs = torch.cat([next_weights, next_obs[:, self.action_dim:]], dim=-1)

            # Store synthetic transitions
            for i in range(obs.shape[0]):
                self.model_buffer.add(
                    obs[i].cpu().numpy(),
                    w[i].cpu().numpy(),
                    reward[i].item(),
                    next_obs[i].cpu().numpy(),
                    False,  # no terminal in model rollouts
                )
            obs = next_obs

    def _get_reward(self, obs, action, next_obs, reward):
        """Override in MOPO to add uncertainty penalty."""
        return reward

    def update_sac(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """SAC update on mixed real + synthetic data."""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        # Critic update
        with torch.no_grad():
            next_w, next_log_prob, _, _ = self.actor(next_obs)
            target_q = self.target_critic.q_min(next_obs, next_w)
            gamma_n = self.config.gamma ** self.n_step
            target_q = rewards + (1.0 - dones) * gamma_n * (
                target_q - self.temperature.detach() * next_log_prob
            )

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        # Actor update
        for p in self.critic.parameters():
            p.requires_grad = False

        w, log_prob, entropy, _ = self.actor(obs)
        q_val = self.critic.q_min(obs, w)
        actor_loss = (self.temperature.detach() * log_prob - q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # Temperature update
        with torch.no_grad():
            _, log_prob, _, _ = self.actor(obs)
        temp_loss = -(self.log_temperature * (log_prob + self.target_entropy).detach()).mean()
        self.temp_opt.zero_grad()
        temp_loss.backward()
        self.temp_opt.step()

        # Target critic
        tau = self.config.polyak_tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

        return {
            'mbpo/critic_loss': critic_loss.item(),
            'mbpo/actor_loss': actor_loss.item(),
            'mbpo/entropy': entropy.mean().item(),
            'mbpo/temperature': self.temperature.item(),
            'mbpo/q_mean': q1.mean().item(),
        }

    def update(self) -> Dict[str, float]:
        self._update_step += 1
        metrics = {}

        # Train dynamics model
        dyn_batch = self.offline_buffer.sample(self.config.batch_size)
        metrics.update(self.dynamics.train_step(dyn_batch))

        # Periodically generate synthetic rollouts and update elites
        if self._update_step % self.config.model_train_freq == 0:
            val_batch = self.offline_buffer.sample(min(1000, len(self.offline_buffer)))
            self.dynamics.update_elites(val_batch)
            self._rollout_model()

        # SAC update on mixed data
        bsz = self.config.batch_size
        real_bsz = max(1, int(bsz * self.config.real_ratio))
        model_bsz = bsz - real_bsz

        real_batch = self.offline_buffer.sample(real_bsz)
        if len(self.model_buffer) >= model_bsz:
            model_batch = self.model_buffer.sample(model_bsz)
            mixed = {
                k: torch.cat([real_batch[k], model_batch[k]], dim=0)
                for k in real_batch if k in model_batch
            }
        else:
            mixed = self.offline_buffer.sample(bsz)

        metrics.update(self.update_sac(mixed))
        return metrics

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        w, _, _, _ = self.actor(obs_t, deterministic=True)
        return w.squeeze(0).cpu().numpy()
