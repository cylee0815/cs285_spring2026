"""
PPO agent supporting MLP, LSTM, and Transformer policies.

Key differences per architecture:
  - MLP: standard stateless rollout
  - LSTM: carry (h, c) hidden states, reset on done; detach after each step
  - Transformer: maintain obs context buffer, reset on done

Training loop: collect n_steps rollouts, compute GAE, update for n_epochs mini-batches.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym

from src.networks.policies import make_policy, LSTMPolicy, TransformerPolicy


class RolloutBuffer:
    """Stores rollout transitions for PPO update."""

    def __init__(self, n_steps: int, obs_dim: int, action_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.device = device

        self.obs = torch.zeros(n_steps, obs_dim, device=device)
        self.actions = torch.zeros(n_steps, action_dim, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.advantages = torch.zeros(n_steps, device=device)
        self.returns = torch.zeros(n_steps, device=device)

        self._ptr = 0

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.values[self._ptr] = value
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = done
        self._ptr += 1

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ):
        """Compute GAE advantages and discounted returns."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]

            delta = (
                self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values

    def get(self):
        return (
            self.obs,
            self.actions,
            self.log_probs,
            self.values,
            self.advantages,
            self.returns,
            self.dones,
        )

    def reset(self):
        self._ptr = 0


def _detach_hidden(hidden):
    """Detach LSTM hidden state tuple from computation graph."""
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(h.detach() for h in hidden)
    return hidden.detach()


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    Compatible with MLP, LSTM, and Transformer policies.
    """

    def __init__(self, env: gym.Env, config, device: torch.device):
        self.env = env
        self.config = config
        self.device = device

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build policy
        self.policy = make_policy(config.arch, obs_dim, action_dim, config).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(config.n_steps, obs_dim, action_dim, device)

        # Episode state
        self._obs, _ = env.reset()
        self._obs = torch.tensor(self._obs, device=device, dtype=torch.float32)
        self._done = False
        self._hidden = None  # For LSTM/Transformer

        # Stats
        self._episode_return = 0.0
        self._episode_length = 0
        self._completed_episodes = []

        self.is_lstm = isinstance(self.policy, LSTMPolicy)
        self.is_transformer = isinstance(self.policy, TransformerPolicy)

    def _prepare_obs_for_policy(self, obs: torch.Tensor) -> torch.Tensor:
        """Add batch dimension if needed for LSTM/Transformer policies."""
        if (self.is_lstm or self.is_transformer) and obs.dim() == 1:
            return obs.unsqueeze(0)
        return obs

    @torch.no_grad()
    def collect_rollouts(self):
        """Collect n_steps transitions from the environment."""
        self.buffer.reset()
        self.policy.eval()

        for _ in range(self.config.n_steps):
            policy_obs = self._prepare_obs_for_policy(self._obs)
            action, log_prob, _, value, new_hidden = self.policy.get_action_and_value(
                policy_obs,
                hidden=self._hidden,
            )

            # Squeeze batch dim added for LSTM/Transformer
            if self.is_lstm or self.is_transformer:
                action_stored = action.squeeze(0)
                log_prob_scalar = log_prob.squeeze(0)
                value_scalar = value.squeeze(0) if value.dim() > 0 else value
            else:
                action_stored = action
                log_prob_scalar = log_prob
                value_scalar = value

            action_np = action_stored.cpu().numpy()

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            self.buffer.add(
                obs=self._obs,
                action=action_stored,
                log_prob=log_prob_scalar,
                value=value_scalar,
                reward=torch.tensor(reward, device=self.device, dtype=torch.float32),
                done=torch.tensor(float(done), device=self.device, dtype=torch.float32),
            )

            self._episode_return += float(reward)
            self._episode_length += 1

            self._obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)

            if done:
                self._completed_episodes.append(
                    {
                        "episode_return": self._episode_return,
                        "episode_length": self._episode_length,
                        "portfolio_value": info.get("portfolio_value", 1.0),
                    }
                )
                self._episode_return = 0.0
                self._episode_length = 0
                # Reset hidden state at episode end
                self._hidden = None
                obs_reset, _ = self.env.reset()
                self._obs = torch.tensor(obs_reset, device=self.device, dtype=torch.float32)
            else:
                # Detach to prevent accumulating computation graph across steps
                self._hidden = _detach_hidden(new_hidden)

        # Compute last value for bootstrapping
        policy_obs = self._prepare_obs_for_policy(self._obs)
        _, _, _, last_value, _ = self.policy.get_action_and_value(
            policy_obs,
            hidden=self._hidden,
        )
        last_value = last_value.squeeze()

        self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )

    def update(self) -> Dict[str, float]:
        """
        Collect rollouts and update policy.
        Returns metrics dict compatible with hw5-style logging.
        """
        self._completed_episodes = []
        self.collect_rollouts()

        obs, actions, old_log_probs, old_values, advantages, returns, dones = (
            self.buffer.get()
        )

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # PPO update epochs
        metrics: Dict[str, list] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        self.policy.train()
        n_samples = self.config.n_steps

        for epoch in range(self.config.n_epochs):
            indices = torch.randperm(n_samples, device=self.device)

            for start in range(0, n_samples, self.config.batch_size):
                mb_idx = indices[start : start + self.config.batch_size]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_old_values = old_values[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Evaluate actions with current policy
                # For LSTM/Transformer: use zero hidden state — standard approach for
                # random minibatches (not sequences).
                log_probs, entropy, values = self.policy.evaluate_actions(
                    mb_obs, mb_actions
                )

                # Policy loss (clipped PPO objective)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (values - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values,
                    -self.config.clip_eps,
                    self.config.clip_eps,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    metrics["policy_loss"].append(policy_loss.item())
                    metrics["value_loss"].append(value_loss.item())
                    metrics["entropy"].append(entropy_loss.item())
                    metrics["approx_kl"].append(
                        ((ratio - 1) - torch.log(ratio)).mean().item()
                    )
                    metrics["clip_fraction"].append(
                        ((ratio - 1).abs() > self.config.clip_eps).float().mean().item()
                    )

        # Aggregate metrics
        result = {k: float(np.mean(v)) for k, v in metrics.items()}

        # Episode stats
        if self._completed_episodes:
            result["episode_return"] = float(
                np.mean([e["episode_return"] for e in self._completed_episodes])
            )
            result["episode_length"] = float(
                np.mean([e["episode_length"] for e in self._completed_episodes])
            )
            result["portfolio_value"] = float(
                np.mean([e["portfolio_value"] for e in self._completed_episodes])
            )

        return result

    @torch.no_grad()
    def evaluate(self, eval_env: gym.Env, n_episodes: int = 5) -> Dict[str, float]:
        """Run evaluation episodes and return performance metrics."""
        self.policy.eval()

        episode_returns = []
        portfolio_values = []
        turnovers = []

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
            hidden = None
            done = False
            ep_return = 0.0
            ep_turnover = 0.0
            info = {}

            while not done:
                policy_obs = self._prepare_obs_for_policy(obs)
                action, _, _, _, new_hidden = self.policy.get_action_and_value(
                    policy_obs,
                    hidden=hidden,
                )
                action_np = action.squeeze(0).cpu().numpy()
                obs_np, reward, terminated, truncated, info = eval_env.step(action_np)
                done = terminated or truncated
                obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
                hidden = _detach_hidden(new_hidden) if not done else None
                ep_return += float(reward)
                ep_turnover += info.get("turnover", 0.0)

            episode_returns.append(ep_return)
            portfolio_values.append(info.get("portfolio_value", 1.0))
            turnovers.append(ep_turnover)

        annual_returns = [
            (pv - 1.0) * (252 / eval_env.episode_length)
            for pv in portfolio_values
        ]

        return {
            "eval/episode_return": float(np.mean(episode_returns)),
            "eval/portfolio_value": float(np.mean(portfolio_values)),
            "eval/annual_return": float(np.mean(annual_returns)),
            "eval/avg_turnover": float(np.mean(turnovers)),
            "eval/std_episode_return": float(np.std(episode_returns)),
        }
