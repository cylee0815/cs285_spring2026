"""Replay buffer for offline RL: loads a .npz dataset and samples minibatches."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer backed by NumPy arrays from an offline dataset.

    Parameters
    ----------
    dataset_path:
        Path to a ``.npz`` file containing keys:
        ``states``, ``actions``, ``rewards``, ``next_states``, ``dones``.
    device:
        Torch device string for returned tensors.
    """

    def __init__(self, dataset_path: str, device: str = "cpu") -> None:
        data = np.load(dataset_path)
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.rewards = data["rewards"].astype(np.float32)
        self.next_states = data["next_states"].astype(np.float32)
        self.dones = data["dones"].astype(np.float32)
        self.size = len(self.states)
        self.device = device

    def sample(self, batch_size: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample a random minibatch.

        Returns
        -------
        states, actions, rewards, next_states, dones
            Tensors on ``self.device``. Rewards and dones have shape ``(B, 1)``.
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], device=self.device),
            torch.tensor(self.actions[idx], device=self.device),
            torch.tensor(self.rewards[idx], device=self.device).unsqueeze(-1),
            torch.tensor(self.next_states[idx], device=self.device),
            torch.tensor(self.dones[idx], device=self.device).unsqueeze(-1),
        )
