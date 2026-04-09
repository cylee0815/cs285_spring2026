"""
Fisher Behavior Cloning: BC using Fisher-Rao geodesic distance on the simplex.

Loss: E[d_FR(E[pi(.|s)], a_behavior)]
     = E[2 * arccos(sum_i sqrt(pi_mean_i * a_behavior_i))]

Tests whether Fisher-Rao loss (the natural Riemannian metric on the simplex)
improves imitation quality compared to standard NLL (BC).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from src.networks.dirichlet_policy import DirichletActor
from src.agents.replay_buffer import ReplayBuffer
from src.agents.cql_geodesic import fisher_rao_distance


class FisherBCAgent:
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

        # Policy mean on simplex: E[Dir(alpha)] = alpha / alpha_0
        dist = self.actor.get_distribution(obs)
        alpha = dist.concentration
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        policy_mean = alpha / alpha_0

        # Fisher-Rao distance as loss
        d_fr = fisher_rao_distance(policy_mean, behavioral_w)
        loss = d_fr.mean()

        self.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_opt.step()

        return {
            'fisher_bc/loss': loss.item(),
            'fisher_bc/distance': d_fr.mean().item(),
            'fisher_bc/entropy': dist.entropy().mean().item(),
        }

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        w, _, _, _ = self.actor(obs_t, deterministic=True)
        return w.squeeze(0).cpu().numpy()
