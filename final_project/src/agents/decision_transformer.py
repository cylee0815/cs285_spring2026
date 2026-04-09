"""
Decision Transformer for offline portfolio optimization.

Casts portfolio allocation as sequence modeling: predict actions conditioned
on return-to-go (RTG), enabling offline training via supervised learning
without Bellman backups or policy gradients.

Training: MSE on predicted vs actual actions from behavioral data.
Inference: condition on target RTG to generate desired-return actions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from src.networks.dt_networks import DecisionTransformerNet
from src.agents.replay_buffer import ReplayBuffer


class DecisionTransformerAgent:
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

        self.model = DecisionTransformerNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_len=config.context_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            max_rtg=config.max_rtg,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        # Precompute RTG after buffer is loaded
        self._rtg_ready = False

    def _ensure_rtg(self):
        if not self._rtg_ready and self.offline_buffer is not None:
            self.offline_buffer.compute_rtg(gamma=1.0)
            self._rtg_ready = True

    def update(self) -> Dict[str, float]:
        self._ensure_rtg()

        batch = self.offline_buffer.sample_trajectory_segment(
            self.config.batch_size, self.config.context_len
        )

        obs_seq = batch['obs_seq']       # (B, T, obs_dim)
        action_seq = batch['action_seq']  # (B, T, action_dim)
        rtg_seq = batch['rtg_seq']       # (B, T)

        # Normalize actions to simplex
        action_seq = action_seq.clamp(1e-7, 1.0)
        action_seq = action_seq / action_seq.sum(dim=-1, keepdim=True)

        # Forward pass
        pred_actions = self.model(rtg_seq, obs_seq, action_seq)  # (B, T, action_dim)
        pred_weights = torch.softmax(pred_actions, dim=-1)

        # MSE loss on predicted vs actual actions
        loss = nn.functional.mse_loss(pred_weights, action_seq)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            'dt/loss': loss.item(),
            'dt/pred_entropy': -(pred_weights * pred_weights.clamp(1e-7).log()).sum(-1).mean().item(),
        }

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        """
        Get action for evaluation. Uses a simple context of just the current obs
        with max RTG conditioning (greedy return-seeking).
        """
        # Build a single-step context
        T = 1
        obs_seq = obs_t.unsqueeze(1)  # (1, 1, obs_dim)
        rtg_seq = torch.tensor([[self.config.max_rtg]], device=self.device)
        action_seq = torch.zeros(1, T, self.action_dim, device=self.device)

        w = self.model.predict_action(rtg_seq, obs_seq, action_seq)
        return w.squeeze(0).cpu().numpy()
