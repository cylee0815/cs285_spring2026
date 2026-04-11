"""
Replay buffer supporting offline and online RL.

Key features:
  - load_from_env(): generate offline trajectories from a behavioral policy
  - sample(): random transition sampling (for standard TD updates)
  - sample_with_context(): sequential sampling with context window (for regime encoder)
  - compute_rtg(): precompute return-to-go for Decision Transformer
  - sample_trajectory_segment(): sample contiguous trajectory windows for sequence models
  - NStepReplayBuffer: n-step return buffer for multi-step offline RL
"""
import torch
import numpy as np
from collections import deque
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

        Uses a uniform-Dirichlet behavioral policy by default (proper simplex
        samples rather than the unbounded Gaussian draws that
        ``env.action_space.sample()`` produces for Box(-inf, inf) spaces).

        Whenever the env reports ``info['executed_weights']`` we store THAT as
        the transition's action. This way the buffer always holds the true
        behavioral policy on the simplex — downstream algorithms that compute a
        behavioral log-prob or BC loss get a consistent, valid target regardless
        of whether the env was in logit or portfolio-weight mode.
        """
        obs, _ = env.reset()
        episode_start = True
        collected = 0

        action_dim = env.action_space.shape[0]

        def _default_policy(_obs):
            # Uniform sample on the (action_dim)-simplex via the exponential /
            # Gamma(1) trick. Equivalent to Dirichlet(alpha=1).
            u = np.random.exponential(scale=1.0, size=action_dim).astype(np.float32)
            return u / (u.sum() + 1e-12)

        if verbose:
            print(f"Collecting {n_steps} offline transitions...")

        while collected < n_steps:
            action = policy(obs) if policy is not None else _default_policy(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Prefer the actually-executed weights if the env exposes them.
            stored_action = info.get("executed_weights", action)
            self.add(obs, stored_action, reward, next_obs, done,
                     episode_start=episode_start)
            obs = next_obs
            episode_start = False
            collected += 1

            if done:
                obs, _ = env.reset()
                episode_start = True

        if verbose:
            print(f"Collected {self._size} transitions in replay buffer.")

    def compute_rtg(self, gamma: float = 1.0):
        """
        Precompute return-to-go for each transition (for Decision Transformer).
        Must be called after freeze(). Stores self.rtg array of shape (capacity,).
        """
        self.rtg = np.zeros(self._size, dtype=np.float32)
        # Process each episode backwards
        i = self._size - 1
        running_rtg = 0.0
        while i >= 0:
            if i == self._size - 1 or self.episode_starts[i + 1]:
                running_rtg = 0.0
            running_rtg = self.rewards[i] + gamma * running_rtg
            self.rtg[i] = running_rtg
            i -= 1

    def sample_trajectory_segment(
        self, batch_size: int, context_len: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample contiguous trajectory windows for sequence models (DT, TT).
        Returns segments that do not cross episode boundaries.
        Requires compute_rtg() to have been called first.
        """
        if not hasattr(self, 'rtg'):
            self.compute_rtg()

        # Find valid episode start/end indices
        episode_boundaries = []
        ep_start = 0
        for i in range(1, self._size):
            if self.episode_starts[i]:
                episode_boundaries.append((ep_start, i))
                ep_start = i
        episode_boundaries.append((ep_start, self._size))

        # Filter episodes long enough for context_len
        valid_episodes = [(s, e) for s, e in episode_boundaries if e - s >= context_len]
        if not valid_episodes:
            # Fallback: use whatever we have, pad with zeros
            valid_episodes = episode_boundaries

        obs_seqs = np.zeros((batch_size, context_len, self.obs_dim), dtype=np.float32)
        action_seqs = np.zeros((batch_size, context_len, self.action_dim), dtype=np.float32)
        reward_seqs = np.zeros((batch_size, context_len), dtype=np.float32)
        rtg_seqs = np.zeros((batch_size, context_len), dtype=np.float32)
        timestep_seqs = np.zeros((batch_size, context_len), dtype=np.int64)

        for b in range(batch_size):
            ep_idx = np.random.randint(len(valid_episodes))
            ep_start, ep_end = valid_episodes[ep_idx]
            ep_len = ep_end - ep_start
            max_start = max(0, ep_len - context_len)
            seg_start = ep_start + np.random.randint(0, max_start + 1)
            seg_end = min(seg_start + context_len, ep_end)
            seg_len = seg_end - seg_start

            obs_seqs[b, :seg_len] = self.obs[seg_start:seg_end]
            action_seqs[b, :seg_len] = self.actions[seg_start:seg_end]
            reward_seqs[b, :seg_len] = self.rewards[seg_start:seg_end]
            rtg_seqs[b, :seg_len] = self.rtg[seg_start:seg_end]
            timestep_seqs[b, :seg_len] = np.arange(seg_start - ep_start,
                                                     seg_start - ep_start + seg_len)

        return {
            'obs_seq': torch.tensor(obs_seqs, device=self.device),
            'action_seq': torch.tensor(action_seqs, device=self.device),
            'reward_seq': torch.tensor(reward_seqs, device=self.device),
            'rtg_seq': torch.tensor(rtg_seqs, device=self.device),
            'timestep_seq': torch.tensor(timestep_seqs, device=self.device),
        }


class NStepReplayBuffer(ReplayBuffer):
    """
    Replay buffer that stores n-step returns for multi-step offline RL.
    Accumulates n transitions and computes the discounted return before storing.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        seq_len: int = 20,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        super().__init__(capacity, obs_dim, action_dim, device, seq_len)
        self.n_step = n_step
        self.gamma = gamma
        self._n_step_buffer: deque = deque(maxlen=n_step)

    def add(self, obs, action, reward, next_obs, done, episode_start=False):
        if self._frozen:
            return

        self._n_step_buffer.append((obs, action, reward, next_obs, done, episode_start))

        # If episode ends, flush all remaining transitions
        if done:
            while self._n_step_buffer:
                self._store_n_step_transition()
            return

        # Store when buffer is full
        if len(self._n_step_buffer) == self.n_step:
            self._store_n_step_transition()

    def _store_n_step_transition(self):
        """Compute n-step return and store the compressed transition."""
        if not self._n_step_buffer:
            return

        obs_0, action_0, _, _, _, ep_start_0 = self._n_step_buffer[0]

        # Compute n-step discounted return
        n_step_return = 0.0
        last_next_obs = self._n_step_buffer[0][3]
        last_done = self._n_step_buffer[0][4]

        for i, (_, _, r, next_obs, d, _) in enumerate(self._n_step_buffer):
            n_step_return += (self.gamma ** i) * r
            last_next_obs = next_obs
            last_done = d
            if d:
                break

        super().add(obs_0, action_0, n_step_return, last_next_obs, last_done, ep_start_0)
        self._n_step_buffer.popleft()
