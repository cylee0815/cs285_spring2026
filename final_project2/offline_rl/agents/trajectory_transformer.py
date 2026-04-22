"""
Trajectory Transformer for offline portfolio optimization.

Models the joint (obs, action) distribution autoregressively over
discretized tokens. At inference, uses greedy decoding to predict
the next action given the observation context.

Discretization: each action dimension is binned into n_bins uniform
intervals over [0, 1], with simplex re-normalization after decoding.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from offline_rl.networks.dt_networks import TrajectoryTransformerNet, SimplexDiscretizer
from offline_rl.agents.replay_buffer import ReplayBuffer


class TrajectoryTransformerAgent:
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

        self.discretizer = SimplexDiscretizer(action_dim, n_bins=config.n_bins)

        self.model = TrajectoryTransformerNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_len=config.context_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            n_bins=config.n_bins,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

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

        # Normalize actions and discretize
        action_seq_norm = action_seq.clamp(1e-7, 1.0)
        action_seq_norm = action_seq_norm / action_seq_norm.sum(dim=-1, keepdim=True)
        action_bins = self.discretizer.encode(action_seq_norm.reshape(-1, self.action_dim))
        action_bins = action_bins.reshape(action_seq.shape[0], action_seq.shape[1], self.action_dim)

        # Forward: predict action bin logits
        logits = self.model(obs_seq, action_bins)  # (B, T, action_dim, n_bins)

        # Cross-entropy loss for each action dimension
        B, T, D, C = logits.shape
        logits_flat = logits.reshape(B * T * D, C)
        targets_flat = action_bins.reshape(B * T * D)
        loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Accuracy
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            accuracy = (preds == targets_flat).float().mean().item()

        return {
            'tt/loss': loss.item(),
            'tt/accuracy': accuracy,
        }

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        """Greedy decoding: predict action bins from single observation."""
        obs_seq = obs_t.unsqueeze(1)  # (1, 1, obs_dim)
        # Dummy action bins (zeros — will be predicted)
        action_bins = torch.zeros(1, 1, self.action_dim, dtype=torch.long, device=self.device)

        logits = self.model(obs_seq, action_bins)  # (1, 1, action_dim, n_bins)
        pred_bins = logits[0, 0].argmax(dim=-1).unsqueeze(0)  # (1, action_dim)

        # Decode bins to continuous weights
        weights = self.discretizer.decode(pred_bins)
        return weights.squeeze(0).cpu().numpy()
