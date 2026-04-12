"""Replay buffer for offline RL: loads a .npz dataset and samples minibatches.

Supports restricting sampling to a subset of transition indices — this is
how we enforce the train / validation / test split: the buffer used during
training is constructed with only the train indices, so no validation or
test transition can ever enter a training minibatch.
"""

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
    indices:
        Optional 1-D integer array of indices to restrict sampling to.
        When provided, :meth:`sample` draws only from these rows of the
        underlying dataset — used to enforce train/val/test splits.
    """

    def __init__(
        self,
        dataset_path: str,
        device: str = "cpu",
        indices: np.ndarray | None = None,
    ) -> None:
        data = np.load(dataset_path)
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.rewards = data["rewards"].astype(np.float32)
        self.next_states = data["next_states"].astype(np.float32)
        self.dones = data["dones"].astype(np.float32)
        self.total_size = len(self.states)
        self.device = device

        if indices is None:
            self.indices = np.arange(self.total_size, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
            if self.indices.size == 0:
                raise ValueError(
                    "ReplayBuffer received an empty `indices` array — nothing to sample."
                )
            if int(self.indices.max()) >= self.total_size or int(self.indices.min()) < 0:
                raise ValueError(
                    f"`indices` out of bounds for dataset of size {self.total_size}."
                )

        self.size = int(self.indices.size)

    def sample(
        self,
        batch_size: int,
        device: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random minibatch from the allowed indices.

        Returns
        -------
        states, actions, rewards, next_states, dones
            Tensors on the target device. Rewards and dones have shape ``(B, 1)``.
        """
        dev = device if device is not None else self.device
        # Randomly sample positions within the allowed-indices array, then look up
        # the underlying dataset rows. This is both fast and guarantees no
        # out-of-split row is ever drawn.
        pos = np.random.randint(0, self.size, size=batch_size)
        idx = self.indices[pos]
        return (
            torch.tensor(self.states[idx], device=dev),
            torch.tensor(self.actions[idx], device=dev),
            torch.tensor(self.rewards[idx], device=dev).unsqueeze(-1),
            torch.tensor(self.next_states[idx], device=dev),
            torch.tensor(self.dones[idx], device=dev).unsqueeze(-1),
        )
