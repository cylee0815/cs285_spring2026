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
from src.networks.bayesian_regime_encoder import (
    BayesianRegimeEncoder, BayesianRegimeConditionedActor, BayesianRegimeConditionedCritic,
)
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
        self.bayesian = getattr(config, 'bayesian', False)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]  # = n_assets
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        regime_dim = config.regime_dim

        # Regime encoder — Bayesian or deterministic
        if self.bayesian:
            self.regime_encoder = BayesianRegimeEncoder(
                obs_dim=obs_dim,
                regime_dim=regime_dim,
                window_len=config.regime_window,
            ).to(device)
            self.actor = BayesianRegimeConditionedActor(
                obs_dim=obs_dim, action_dim=action_dim,
                regime_dim=regime_dim, hidden_dim=config.hidden_dim,
            ).to(device)
            self.critic = BayesianRegimeConditionedCritic(
                obs_dim=obs_dim, action_dim=action_dim,
                regime_dim=regime_dim, hidden_dim=config.hidden_dim,
            ).to(device)
        else:
            self.regime_encoder = RegimeEncoder(
                obs_dim=obs_dim,
                regime_dim=regime_dim,
                window_len=config.regime_window,
            ).to(device)
            self.actor = RegimeConditionedActor(
                obs_dim=obs_dim, action_dim=action_dim,
                regime_dim=regime_dim, hidden_dim=config.hidden_dim,
            ).to(device)
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

        # Bayesian hyperparams
        self.kl_beta = getattr(config, 'kl_beta', 0.1)

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
    def _get_regime(self, obs: np.ndarray, thompson_sample: bool = False) -> torch.Tensor:
        """
        Get regime context for current observation.
        With Bayesian encoder and thompson_sample=True, samples from the posterior
        for risk-aware exploration (Thompson sampling).
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.bayesian:
            regime, _, self._gru_hidden = self.regime_encoder.encode_step(
                obs_t, self._gru_hidden, sample=thompson_sample
            )
        else:
            regime, self._gru_hidden = self.regime_encoder.encode_step(obs_t, self._gru_hidden)
        return regime  # (1, regime_dim)

    @torch.no_grad()
    def collect_step(self):
        """
        Collect one transition from the environment.
        With Bayesian encoder, uses Thompson sampling: each episode samples a
        regime hypothesis from the posterior, producing diverse exploration behavior.
        """
        regime = self._get_regime(self._obs, thompson_sample=self.bayesian)
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

    def _encode_regime(self, obs_seq: Optional[torch.Tensor], batch_size: int):
        """Encode regime context, handling both Bayesian and deterministic encoders."""
        if obs_seq is None:
            regime = torch.zeros(batch_size, self.config.regime_dim, device=self.device)
            return regime, None, 0.0
        if self.bayesian:
            regime, regime_std, kl_loss, _ = self.regime_encoder(obs_seq)
            return regime, regime_std, kl_loss
        else:
            regime, _ = self.regime_encoder(obs_seq)
            return regime, None, 0.0

    def update_critic(self, batch: Dict[str, torch.Tensor], cql_weight: float = 0.0) -> Dict[str, float]:
        """Update Q-networks with Bellman TD error (+ optional CQL penalty for O2O use)."""
        obs, actions, rewards, next_obs = (
            batch['obs'], batch['actions'], batch['rewards'], batch['next_obs']
        )
        dones = batch['dones']
        obs_seq = batch.get('obs_seq', None)
        next_obs_seq = batch.get('next_obs_seq', None)

        bsz = obs.shape[0]
        regime, _, kl_loss = self._encode_regime(obs_seq, bsz)
        next_regime, _, _ = self._encode_regime(next_obs_seq, bsz)

        with torch.no_grad():
            # Target: r + γ * (1 - done) * (min_Q(s', a') - τ * log π(a'|s'))
            next_w, next_log_prob, _, _ = self.actor(next_obs, next_regime, deterministic=False)
            target_q = self.target_critic.q_min(next_obs, next_w, next_regime)
            target_q = rewards + (1.0 - dones) * self.config.gamma * (
                target_q - self.temperature.detach() * next_log_prob
            )

        q1, q2 = self.critic(obs, actions, regime)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        total_loss = critic_loss
        if self.bayesian and kl_loss > 0:
            total_loss = total_loss + self.kl_beta * kl_loss

        metrics = {'critic_loss': critic_loss.item()}
        if self.bayesian:
            metrics['critic_kl_loss'] = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss

        self.critic_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return metrics

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Dirichlet actor. Loss = τ * log π(w|s) - Q_min(s, w)."""
        obs = batch['obs']
        obs_seq = batch.get('obs_seq', None)

        regime, _, kl_loss = self._encode_regime(obs_seq, obs.shape[0])

        # Freeze critic during actor update
        for p in self.critic.parameters():
            p.requires_grad = False

        w, log_prob, entropy, alpha = self.actor(obs, regime, deterministic=False)
        q_val = self.critic.q_min(obs, w, regime)

        # Actor loss: minimize (τ * log_prob - Q) = maximize (Q - τ * entropy)
        actor_loss = (self.temperature.detach() * log_prob - q_val).mean()

        total_actor_loss = actor_loss
        if self.bayesian and kl_loss > 0:
            total_actor_loss = total_actor_loss + self.kl_beta * kl_loss

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
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
            regime, _, _ = self._encode_regime(obs_seq, obs.shape[0])
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
        all_sharpes, all_max_drawdowns = [], []

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            gru_hidden = self.regime_encoder.init_hidden(1, self.device)
            done = False
            ep_return = 0.0
            ep_turnover = 0.0
            step_returns = []
            pv_trajectory = [1.0]
            info = {}

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                if self.bayesian:
                    regime, _, gru_hidden = self.regime_encoder.encode_step(obs_t, gru_hidden, sample=False)
                else:
                    regime, gru_hidden = self.regime_encoder.encode_step(obs_t, gru_hidden)
                w, _, _, _ = self.actor(obs_t, regime, deterministic=True)
                action = w.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                ep_return += reward
                ep_turnover += info.get('turnover', 0.0)
                step_returns.append(reward)
                pv_trajectory.append(info.get('portfolio_value', pv_trajectory[-1]))

            episode_returns.append(ep_return)
            portfolio_values.append(info.get('portfolio_value', 1.0))
            turnovers.append(ep_turnover)

            # Sharpe ratio (annualized)
            arr = np.array(step_returns)
            if len(arr) > 1 and arr.std() > 1e-8:
                all_sharpes.append(float(arr.mean() / arr.std() * np.sqrt(252)))
            else:
                all_sharpes.append(0.0)

            # Max drawdown
            pv_arr = np.array(pv_trajectory)
            running_max = np.maximum.accumulate(pv_arr)
            drawdowns = (running_max - pv_arr) / np.maximum(running_max, 1e-8)
            all_max_drawdowns.append(float(drawdowns.max()))

        annual_returns = [(pv - 1.0) * (252 / eval_env.episode_length) for pv in portfolio_values]
        return {
            'eval/episode_return': float(np.mean(episode_returns)),
            'eval/portfolio_value': float(np.mean(portfolio_values)),
            'eval/annual_return': float(np.mean(annual_returns)),
            'eval/avg_turnover': float(np.mean(turnovers)),
            'eval/sharpe_ratio': float(np.mean(all_sharpes)),
            'eval/max_drawdown': float(np.mean(all_max_drawdowns)),
        }
