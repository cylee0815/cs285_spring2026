"""
Replay buffer supporting offline and online RL.

Key features:
  - load_from_env(): generate offline trajectories from a behavioral policy
  - sample(): random transition sampling (for standard TD updates)
  - sample_sequences(): sequential sampling with context window (for regime encoder)
"""
import torch
import numpy as np
from typing import Optional, Dict
import gymnasium as gym


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        seq_len: int = 20,  # context window for regime encoder
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.seq_len = seq_len

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.episode_starts = np.zeros(capacity, dtype=bool)  # True at start of episode

        self._ptr = 0
        self._size = 0
        self._frozen = False  # if True, disallow further additions (offline buffer)

    def add(self, obs, action, reward, next_obs, done, episode_start=False):
        if self._frozen:
            return
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.next_obs[self._ptr] = next_obs
        self.dones[self._ptr] = float(done)
        self.episode_starts[self._ptr] = episode_start
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def freeze(self):
        """Mark as offline buffer — no further writes allowed."""
        self._frozen = True

    def __len__(self):
        return self._size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return self._to_tensors(idx)

    def sample_with_context(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample transitions with a preceding observation sequence for regime encoder.
        Returns obs_seq of shape (batch, seq_len, obs_dim) — the seq_len obs leading up to each transition.
        """
        idx = np.random.randint(0, self._size, size=batch_size)
        batch = self._to_tensors(idx)

        # Build observation sequences (pad with zeros at episode boundaries)
        obs_seqs = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        for i, t in enumerate(idx):
            for k in range(self.seq_len):
                src = (t - (self.seq_len - 1 - k)) % self._size
                if self.episode_starts[src] and k < self.seq_len - 1:
                    # Zero-pad before episode start
                    obs_seqs[i, k] = 0.0
                else:
                    obs_seqs[i, k] = self.obs[src]

        batch['obs_seq'] = torch.tensor(obs_seqs, device=self.device)

        # Also build next obs sequences (shifted by 1)
        next_obs_seqs = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        for i, t in enumerate(idx):
            next_t = (t + 1) % self._size
            for k in range(self.seq_len):
                src = (next_t - (self.seq_len - 1 - k)) % self._size
                if self.episode_starts[src] and k < self.seq_len - 1:
                    next_obs_seqs[i, k] = 0.0
                else:
                    next_obs_seqs[i, k] = self.obs[src]
        batch['next_obs_seq'] = torch.tensor(next_obs_seqs, device=self.device)

        return batch

    def _to_tensors(self, idx: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            'obs': torch.tensor(self.obs[idx], device=self.device),
            'actions': torch.tensor(self.actions[idx], device=self.device),
            'rewards': torch.tensor(self.rewards[idx], device=self.device),
            'next_obs': torch.tensor(self.next_obs[idx], device=self.device),
            'dones': torch.tensor(self.dones[idx], device=self.device),
        }

    def load_from_env(
        self,
        env: gym.Env,
        n_steps: int,
        policy=None,  # callable(obs) -> action, or None for random
        verbose: bool = True,
    ):
        """
        Collect offline data by rolling out a behavioral policy in the environment.
        Uses a uniform-random behavioral policy if none provided.
        """
        obs, _ = env.reset()
        episode_start = True
        collected = 0

        if verbose:
            print(f"Collecting {n_steps} offline transitions...")

        while collected < n_steps:
            if policy is not None:
                action = policy(obs)
            else:
                # Default behavioral: momentum + equal-weight mixture (more realistic than pure random)
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.add(obs, action, reward, next_obs, done, episode_start=episode_start)
            obs = next_obs
            episode_start = False
            collected += 1

            if done:
                obs, _ = env.reset()
                episode_start = True

        if verbose:
            print(f"Collected {self._size} transitions in replay buffer.")
