"""
TD3+BC: Twin Delayed DDPG with Behavior Cloning regularization.

Actor loss: -lambda * Q(s, pi(s)) + ||pi(s) - a||^2
  where lambda = alpha / (1/N * sum|Q(s_i, a_i)|)  normalizes Q scale.

Uses deterministic MLPPortfolioActor (softmax output) for simplex actions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional

from src.networks.dirichlet_policy import MLPPortfolioActor, DoubleCritic
from src.agents.replay_buffer import ReplayBuffer


class TD3BCAgent:
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
        self._update_step = 0

        # Deterministic actor with softmax output
        self.actor = MLPPortfolioActor(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False

        # Double critic
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
            # Target policy smoothing
            next_actions = self.target_actor(next_obs)
            noise = (torch.randn_like(next_actions) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            # Add noise in logit space, then re-normalize to simplex
            next_actions_noisy = next_actions + noise
            next_actions_noisy = next_actions_noisy.clamp(1e-7, None)
            next_actions_noisy = next_actions_noisy / next_actions_noisy.sum(dim=-1, keepdim=True)

            target_q = self.target_critic.q_min(next_obs, next_actions_noisy)
            target_q = rewards + (1.0 - dones) * self.config.gamma * target_q

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return {
            'td3bc/critic_loss': critic_loss.item(),
            'td3bc/q_mean': q1.mean().item(),
        }

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']

        pi = self.actor(obs)  # deterministic softmax weights
        q = self.critic.q_min(obs, pi)

        # Normalize Q scale (TD3+BC paper)
        lam = self.config.td3bc_alpha / (q.abs().mean().detach() + 1e-8)

        # Normalize behavioral actions to simplex
        behavioral_w = actions.clamp(1e-7, 1.0)
        behavioral_w = behavioral_w / behavioral_w.sum(dim=-1, keepdim=True)

        # TD3+BC loss: -lambda * Q + BC penalty
        actor_loss = -lam * q.mean() + nn.functional.mse_loss(pi, behavioral_w)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()

        return {
            'td3bc/actor_loss': actor_loss.item(),
            'td3bc/q_pi_mean': q.mean().item(),
            'td3bc/lambda': lam.item(),
        }

    def update_targets(self):
        tau = self.config.polyak_tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)
        for p, tp in zip(self.actor.parameters(), self.target_actor.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self) -> Dict[str, float]:
        batch = self.offline_buffer.sample(self.config.batch_size)
        self._update_step += 1

        metrics = {}
        metrics.update(self.update_critic(batch))

        # Delayed policy updates
        if self._update_step % self.config.policy_delay == 0:
            metrics.update(self.update_actor(batch))
            self.update_targets()

        return metrics

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        w = self.actor(obs_t)
        return w.squeeze(0).cpu().numpy()
