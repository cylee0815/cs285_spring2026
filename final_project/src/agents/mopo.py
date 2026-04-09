"""
Model-Based Offline Policy Optimization (MOPO).

Extends MBPO with an uncertainty penalty during model rollouts:
  r'(s,a) = r(s,a) - lambda * model_uncertainty

The uncertainty penalty prevents the policy from exploiting inaccurate
regions of the learned dynamics model.
"""
import torch
import numpy as np
from typing import Dict, Optional

from src.agents.mbpo import MBPOAgent
from src.agents.replay_buffer import ReplayBuffer


class MOPOAgent(MBPOAgent):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config,
        device: torch.device,
        offline_buffer: Optional[ReplayBuffer] = None,
    ):
        super().__init__(obs_dim, action_dim, config, device, offline_buffer)
        self.mopo_lambda = getattr(config, 'mopo_lambda', 1.0)

    def _rollout_model(self):
        """Generate synthetic transitions with uncertainty-penalized rewards."""
        batch = self.offline_buffer.sample(self.config.rollout_batch_size)
        obs = batch['obs']

        for _ in range(self.config.rollout_length):
            with torch.no_grad():
                w, _, _, _ = self.actor(obs)
                next_obs, reward, uncertainty = self.dynamics.predict_with_uncertainty(obs, w)

            # Uncertainty-penalized reward (MOPO's key contribution)
            penalized_reward = reward - self.mopo_lambda * uncertainty

            # Project portfolio weight dimensions back to simplex
            next_weights = next_obs[:, :self.action_dim]
            next_weights = next_weights.clamp(1e-7, None)
            next_weights = next_weights / next_weights.sum(dim=-1, keepdim=True)
            next_obs = torch.cat([next_weights, next_obs[:, self.action_dim:]], dim=-1)

            for i in range(obs.shape[0]):
                self.model_buffer.add(
                    obs[i].cpu().numpy(),
                    w[i].cpu().numpy(),
                    penalized_reward[i].item(),
                    next_obs[i].cpu().numpy(),
                    False,
                )
            obs = next_obs
