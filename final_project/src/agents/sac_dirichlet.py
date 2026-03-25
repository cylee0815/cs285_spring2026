"""
SAC-Dirichlet: Soft Actor-Critic with Dirichlet policy on the portfolio simplex.

Key differences from standard SAC:
  1. Actor parameterizes Dir(α(s)) instead of N(μ(s), σ(s)) + softmax
  2. Entropy is the exact Dirichlet entropy (closed-form)
  3. Q-function is conditioned on regime belief state h_t from RegimeEncoder
  4. Policy operates directly on the simplex — no post-hoc normalization
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional
import gymnasium as gym

from src.networks.dirichlet_policy import DirichletActor, DoubleCritic
from src.networks.regime_encoder import RegimeEncoder, RegimeConditionedActor, RegimeConditionedCritic
from src.agents.replay_buffer import ReplayBuffer


class SACDirichlet:
    """
    SAC with Dirichlet policy and optional regime conditioning.

    Actor loss: E[τ * log π(w|s,h) - Q_min(s, w, h)]
      where log π is the exact Dirichlet log-prob on the simplex.

    Critic loss: MSE to Bellman target using min(Q1, Q2) from target networks.

    Temperature loss: -τ * (log π(w|s,h) + H_target)
      where H_target = log(n_assets) is the maximum entropy on the simplex.
    """

    def __init__(self, env: gym.Env, config, device: torch.device):
        self.env = env
        self.config = config
        self.device = device

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]  # = n_assets
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Regime encoder
        self.regime_encoder = RegimeEncoder(
            obs_dim=obs_dim,
            regime_dim=config.regime_dim,
            window_len=config.regime_window,
        ).to(device)

        regime_dim = config.regime_dim

        # Actor (regime-conditioned Dirichlet)
        self.actor = RegimeConditionedActor(
            obs_dim=obs_dim, action_dim=action_dim,
            regime_dim=regime_dim, hidden_dim=config.hidden_dim,
        ).to(device)

        # Double critic (regime-conditioned)
        self.critic = RegimeConditionedCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            regime_dim=regime_dim, hidden_dim=config.hidden_dim,
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # Learnable temperature (log τ)
        # H_target = log(n_assets): maximum entropy of uniform Dir on simplex
        self.target_entropy = np.log(action_dim)
        self.log_temperature = nn.Parameter(torch.zeros(1, device=device))

        # Optimizers
        self.actor_opt = optim.Adam(
            list(self.actor.parameters()) + list(self.regime_encoder.parameters()),
            lr=config.lr
        )
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr)
        self.temp_opt = optim.Adam([self.log_temperature], lr=config.lr)

        # Online replay buffer
        self.buffer = ReplayBuffer(
            config.buffer_size, obs_dim, action_dim, device,
            seq_len=config.regime_window,
        )

        # Episode state
        self._obs, _ = env.reset()
        self._episode_start = True
        self._gru_hidden = self.regime_encoder.init_hidden(1, device)
        self._episode_return = 0.0

    @property
    def temperature(self):
        return self.log_temperature.exp()

    @torch.no_grad()
    def _get_regime(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        regime, self._gru_hidden = self.regime_encoder.encode_step(obs_t, self._gru_hidden)
        return regime  # (1, regime_dim)

    @torch.no_grad()
    def collect_step(self):
        """Collect one transition from the environment."""
        regime = self._get_regime(self._obs)
        obs_t = torch.tensor(self._obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        w, _, _, _ = self.actor(obs_t, regime, deterministic=False)
        action = w.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.buffer.add(self._obs, action, reward, next_obs, done, self._episode_start)
        self._episode_return += reward
        self._obs = next_obs
        self._episode_start = False

        if done:
            self._obs, _ = self.env.reset()
            self._episode_start = True
            self._gru_hidden = self.regime_encoder.init_hidden(1, self.device)
            self._episode_return = 0.0

    def update_critic(self, batch: Dict[str, torch.Tensor], cql_weight: float = 0.0) -> Dict[str, float]:
        """Update Q-networks with Bellman TD error (+ optional CQL penalty for O2O use)."""
        obs, actions, rewards, next_obs = (
            batch['obs'], batch['actions'], batch['rewards'], batch['next_obs']
        )
        dones = batch['dones']
        obs_seq = batch.get('obs_seq', None)
        next_obs_seq = batch.get('next_obs_seq', None)

        # Compute regime context
        if obs_seq is not None:
            regime, _ = self.regime_encoder(obs_seq)
            next_regime, _ = self.regime_encoder(next_obs_seq)
        else:
            # Fallback: zero regime context
            bsz = obs.shape[0]
            regime = torch.zeros(bsz, self.config.regime_dim, device=self.device)
            next_regime = torch.zeros(bsz, self.config.regime_dim, device=self.device)

        with torch.no_grad():
            # Target: r + γ * (1 - done) * (min_Q(s', a') - τ * log π(a'|s'))
            next_w, next_log_prob, _, _ = self.actor(next_obs, next_regime, deterministic=False)
            target_q = self.target_critic.q_min(next_obs, next_w, next_regime)
            target_q = rewards + (1.0 - dones) * self.config.gamma * (
                target_q - self.temperature.detach() * next_log_prob
            )

        q1, q2 = self.critic(obs, actions, regime)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        metrics = {'critic_loss': critic_loss.item()}

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return metrics

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Dirichlet actor. Loss = τ * log π(w|s) - Q_min(s, w)."""
        obs = batch['obs']
        obs_seq = batch.get('obs_seq', None)

        if obs_seq is not None:
            regime, _ = self.regime_encoder(obs_seq)
        else:
            regime = torch.zeros(obs.shape[0], self.config.regime_dim, device=self.device)

        # Freeze critic during actor update
        for p in self.critic.parameters():
            p.requires_grad = False

        w, log_prob, entropy, alpha = self.actor(obs, regime, deterministic=False)
        q_val = self.critic.q_min(obs, w, regime)

        # Actor loss: minimize (τ * log_prob - Q) = maximize (Q - τ * entropy)
        actor_loss = (self.temperature.detach() * log_prob - q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.regime_encoder.parameters()),
            self.config.max_grad_norm
        )
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        return {
            'actor_loss': actor_loss.item(),
            'entropy': entropy.mean().item(),
            'temperature': self.temperature.item(),
            'alpha_mean': alpha.mean().item(),
        }

    def update_temperature(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Dual gradient descent on temperature τ."""
        obs = batch['obs']
        obs_seq = batch.get('obs_seq', None)

        with torch.no_grad():
            if obs_seq is not None:
                regime, _ = self.regime_encoder(obs_seq)
            else:
                regime = torch.zeros(obs.shape[0], self.config.regime_dim, device=self.device)
            _, log_prob, _, _ = self.actor(obs, regime)

        # τ loss: -τ * (log π + H_target)
        temp_loss = -(self.log_temperature * (log_prob + self.target_entropy).detach()).mean()

        self.temp_opt.zero_grad()
        temp_loss.backward()
        self.temp_opt.step()

        return {'temp_loss': temp_loss.item()}

    def update_target_critic(self):
        """Polyak averaging of target critic."""
        tau = self.config.polyak_tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self, external_batch: Optional[Dict] = None) -> Dict[str, float]:
        """
        Collect one step, then update all networks.
        external_batch: if provided (offline data), mixed with online buffer sample.
        """
        self.collect_step()

        if len(self.buffer) < self.config.batch_size:
            return {}

        batch = self.buffer.sample_with_context(self.config.batch_size)

        if external_batch is not None:
            # Mix offline and online batches
            batch = {
                k: torch.cat([batch[k], external_batch[k]], dim=0)
                for k in batch if k in external_batch
            }

        metrics = {}
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))
        metrics.update(self.update_temperature(batch))
        self.update_target_critic()

        return metrics

    @torch.no_grad()
    def evaluate(self, eval_env: gym.Env, n_episodes: int = 5) -> Dict[str, float]:
        episode_returns, portfolio_values, turnovers = [], [], []

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            gru_hidden = self.regime_encoder.init_hidden(1, self.device)
            done = False
            ep_return = 0.0
            ep_turnover = 0.0
            info = {}

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                regime, gru_hidden = self.regime_encoder.encode_step(obs_t, gru_hidden)
                w, _, _, _ = self.actor(obs_t, regime, deterministic=True)
                action = w.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                ep_return += reward
                ep_turnover += info.get('turnover', 0.0)

            episode_returns.append(ep_return)
            portfolio_values.append(info.get('portfolio_value', 1.0))
            turnovers.append(ep_turnover)

        annual_returns = [(pv - 1.0) * (252 / eval_env.episode_length) for pv in portfolio_values]
        return {
            'eval/episode_return': float(np.mean(episode_returns)),
            'eval/portfolio_value': float(np.mean(portfolio_values)),
            'eval/annual_return': float(np.mean(annual_returns)),
            'eval/avg_turnover': float(np.mean(turnovers)),
        }
