"""
Behavior Cloning: supervised imitation of behavioral policy on the simplex.

Loss: -E[log pi(a|s)] where pi is a Dirichlet distribution.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from offline_rl.networks.dirichlet_policy import DirichletActor
from offline_rl.agents.replay_buffer import ReplayBuffer


class BCAgent:
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

        self.actor = DirichletActor(
            obs_dim, action_dim, config.hidden_dim, config.n_layers
        ).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr)

    def update(self) -> Dict[str, float]:
        batch = self.offline_buffer.sample(self.config.batch_size)
        obs = batch['obs']
        actions = batch['actions']

        # Normalize behavioral actions to simplex
        behavioral_w = actions.clamp(1e-7, 1.0)
        behavioral_w = behavioral_w / behavioral_w.sum(dim=-1, keepdim=True)

        dist = self.actor.get_distribution(obs)
        log_prob = dist.log_prob(behavioral_w)
        loss = -log_prob.mean()

        self.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()

        return {
            'bc/loss': loss.item(),
            'bc/log_prob': log_prob.mean().item(),
            'bc/entropy': dist.entropy().mean().item(),
        }

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        w, _, _, _ = self.actor(obs_t, deterministic=True)
        return w.squeeze(0).cpu().numpy()
