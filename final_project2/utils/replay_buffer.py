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

    def sample(
        self,
        batch_size: int,
        device: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random minibatch.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.
        device:
            If provided, returned tensors are placed on this device instead
            of the buffer's default device.

        Returns
        -------
        states, actions, rewards, next_states, dones
            Tensors on the target device. Rewards and dones have shape ``(B, 1)``.
        """
        dev = device if device is not None else self.device
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], device=dev),
            torch.tensor(self.actions[idx], device=dev),
            torch.tensor(self.rewards[idx], device=dev).unsqueeze(-1),
            torch.tensor(self.next_states[idx], device=dev),
            torch.tensor(self.dones[idx], device=dev).unsqueeze(-1),
        )
