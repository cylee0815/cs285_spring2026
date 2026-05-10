"""
Offline-to-Online (O2O) Agent.

Pipeline:
  Phase 1 — Offline pre-training with Geodesic-CQL on historical data.
  Phase 2 — Online fine-tuning with SAC-Dirichlet, mixing offline + online data.

O2O conservatism annealing:
  During fine-tuning, monitor the KL divergence between offline and online regime
  distributions. When online regimes closely match offline training regimes, reduce
  the CQL penalty weight (safe to exploit pre-trained policy). When regimes diverge
  (distribution shift), maintain conservatism (don't overfit to OOD Q estimates).

  cql_weight(t) = sigmoid(λ * KL(h_offline || h_online))

  This is the key contribution of the O2O architecture: adaptive conservatism driven
  by regime-level distribution shift, rather than a fixed annealing schedule.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
import gymnasium as gym
import copy

from core.buffers.replay_buffer import ReplayBuffer
from offline_rl.agents.cql_geodesic import GeodesicCQL
from online_rl.agents.sac_dirichlet import SACDirichlet


def regime_kl_divergence(h_offline: torch.Tensor, h_online: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Estimate KL divergence between offline and online regime distributions.
    Uses Gaussian approximation: fit N(μ, σ²) to each batch of regime vectors
    and compute KL(offline || online).

    KL(N₁ || N₂) = Σᵢ [log(σ₂ᵢ/σ₁ᵢ) + (σ₁ᵢ² + (μ₁ᵢ-μ₂ᵢ)²)/(2σ₂ᵢ²) - 0.5]
    """
    mu1, std1 = h_offline.mean(0), h_offline.std(0).clamp(min=eps)
    mu2, std2 = h_online.mean(0), h_online.std(0).clamp(min=eps)

    kl = (torch.log(std2 / std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5)
    return kl.sum().item()


class O2OAgent:
    """
    Full Offline-to-Online pipeline with Geodesic-CQL pre-training
    and SAC-Dirichlet fine-tuning with adaptive regime-based conservatism.
    """

    def __init__(
        self,
        train_env: gym.Env,
        eval_env: gym.Env,
        config,
        device: torch.device,
    ):
        self.train_env = train_env
        self.eval_env = eval_env
        self.config = config
        self.device = device
        self.bayesian = getattr(config, 'bayesian', False)

        obs_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Offline buffer (frozen after loading)
        self.offline_buffer = ReplayBuffer(
            config.offline_buffer_size, obs_dim, action_dim, device,
            seq_len=config.regime_window,
        )

        # Offline pre-trainer (Bayesian mode propagated via config)
        self.cql_agent = GeodesicCQL(
            obs_dim=obs_dim, action_dim=action_dim,
            config=config, device=device,
            offline_buffer=self.offline_buffer,
        )

        # Online fine-tuner (shares architecture with CQL agent)
        self.sac_agent = SACDirichlet(train_env, config, device)

        # Regime statistics for KL monitoring (accumulated during offline training)
        self._offline_regime_samples: List[torch.Tensor] = []
        # Bayesian: also store regime variances for uncertainty-weighted CQL
        self._offline_regime_vars: List[torch.Tensor] = []
        self._phase = 'offline'  # 'offline' or 'online'

    def load_offline_data(
        self,
        behavioral_policy=None,
        policy_mixture=None,
        n_steps: Optional[int] = None,
        mixture_seed: Optional[int] = None,
    ):
        """Populate offline buffer from environment trajectories.

        Pass either ``behavioral_policy`` (single callable) or
        ``policy_mixture`` (list of (callable, weight)). Defaults to the
        legacy uniform-Dirichlet behavior baked into ``load_from_env``.
        """
        n = n_steps or self.config.offline_buffer_size
        self.offline_buffer.load_from_env(
            self.train_env,
            n_steps=n,
            policy=behavioral_policy,
            policy_mixture=policy_mixture,
            mixture_seed=mixture_seed,
            verbose=True,
        )
        self.offline_buffer.freeze()

    def pretrain_offline(self, n_updates: int) -> List[Dict]:
        """Phase 1: offline Geodesic-CQL pre-training."""
        self._phase = 'offline'
        history = []

        print(f"Phase 1: Offline pre-training for {n_updates} updates...")
        for step in range(n_updates):
            metrics = self.cql_agent.update()
            history.append(metrics)

            # Accumulate regime samples for offline distribution
            if step % 100 == 0:
                batch = self.offline_buffer.sample_with_context(min(256, len(self.offline_buffer)))
                with torch.no_grad():
                    obs_seq = batch.get('obs_seq')
                    if obs_seq is not None:
                        if self.bayesian:
                            regime, regime_std, _, _ = self.cql_agent.regime_encoder(obs_seq, sample=False)
                            self._offline_regime_samples.append(regime.cpu())
                            self._offline_regime_vars.append(regime_std.pow(2).cpu())
                        else:
                            regime, _ = self.cql_agent.regime_encoder(obs_seq)
                            self._offline_regime_samples.append(regime.cpu())

        return history

    def transfer_to_online(self):
        """
        Transfer pre-trained weights from CQL agent to SAC agent.
        This initializes the online fine-tuner with the offline-pre-trained policy,
        which is the core of O2O transfer.
        """
        print("Transferring offline policy to online agent...")
        policy_state = self.cql_agent.get_policy_state()
        self.sac_agent.actor.load_state_dict(policy_state['actor'])
        self.sac_agent.critic.load_state_dict(policy_state['critic'])
        self.sac_agent.target_critic.load_state_dict(policy_state['target_critic'])
        self.sac_agent.regime_encoder.load_state_dict(policy_state['regime_encoder'])
        self.sac_agent.log_temperature.data.copy_(policy_state['log_temperature'])
        # The shared train_env was rolled to exhaustion while populating the
        # offline buffer; re-anchor the SAC agent on a fresh reset so its
        # next collect_step() doesn't trip the env's "Episode is done"
        # guard before the first online action is taken.
        self.sac_agent._obs, _ = self.sac_agent.env.reset()
        self.sac_agent._episode_start = True
        self.sac_agent._episode_return = 0.0
        self.sac_agent._gru_hidden = self.sac_agent.regime_encoder.init_hidden(
            1, self.sac_agent.device
        )

    def _compute_adaptive_cql_weight(
        self,
        online_regime_samples: torch.Tensor,
        online_regime_var: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute CQL conservatism weight based on regime KL divergence
        and (optionally) epistemic uncertainty.

        Full formula (Bayesian):
          β_t = β₀ · sigmoid(λ · KL(h_offline || h_online) + γ · Var[h_t])

        Deterministic fallback:
          β_t = β₀ · sigmoid(λ · KL)

        High KL → maintain conservatism (large CQL weight).
        High Var[h_t] → uncertain regime → more conservative.
        """
        if not self._offline_regime_samples:
            return self.config.cql_alpha

        h_offline = torch.cat(self._offline_regime_samples[-10:], dim=0)
        h_online = online_regime_samples

        kl = regime_kl_divergence(h_offline.to(self.device), h_online)

        # Combine KL term with variance term (Bayesian extension)
        score = self.config.regime_kl_scale * kl
        if self.bayesian and online_regime_var is not None:
            var_scale = getattr(self.config, 'var_scale', 1.0)
            mean_var = online_regime_var.mean().item()
            score += var_scale * mean_var

        weight = torch.sigmoid(torch.tensor(score)).item()
        return self.config.cql_alpha * weight

    def finetune_online(self, n_steps: int) -> List[Dict]:
        """
        Phase 2: online fine-tuning with SAC-Dirichlet.

        Mixes offline and online data. CQL penalty weight adapts to regime KL.
        """
        self._phase = 'online'
        self.transfer_to_online()
        history = []

        print(f"Phase 2: Online fine-tuning for {n_steps} steps...")
        recent_regimes = []
        recent_vars = []

        for step in range(n_steps):
            # Collect online step
            self.sac_agent.collect_step()

            if len(self.sac_agent.buffer) < self.config.batch_size:
                continue

            # Sample from both buffers
            online_batch = self.sac_agent.buffer.sample_with_context(self.config.batch_size // 2)
            offline_batch = self.offline_buffer.sample_with_context(self.config.batch_size // 2)

            # Monitor online regime distribution
            if step % 50 == 0 and 'obs_seq' in online_batch:
                with torch.no_grad():
                    if self.bayesian:
                        online_regime, online_std, _, _ = self.sac_agent.regime_encoder(
                            online_batch['obs_seq'], sample=False
                        )
                        recent_regimes.append(online_regime.cpu())
                        recent_vars.append(online_std.pow(2).cpu())
                    else:
                        online_regime, _ = self.sac_agent.regime_encoder(online_batch['obs_seq'])
                        recent_regimes.append(online_regime.cpu())
                    if len(recent_regimes) > 20:
                        recent_regimes = recent_regimes[-20:]
                        recent_vars = recent_vars[-20:]

            # Adaptive CQL weight (with optional Bayesian variance term).
            # When ``getattr(config, 'adaptive_conservatism', True)`` is False,
            # the weight is pinned to ``config.cql_alpha`` for the whole online
            # phase — that's the Phase 2C 'naive fine-tune' condition.
            if not getattr(self.config, 'adaptive_conservatism', True):
                cql_w = self.config.cql_alpha
            elif recent_regimes:
                h_online = torch.cat(recent_regimes[-5:], dim=0).to(self.device)
                var_online = torch.cat(recent_vars[-5:], dim=0) if recent_vars else None
                cql_w = self._compute_adaptive_cql_weight(h_online, var_online)
            else:
                cql_w = self.config.cql_alpha

            # Merge batches
            mixed_batch = {
                k: torch.cat([online_batch[k], offline_batch[k]], dim=0)
                for k in online_batch if k in offline_batch
            }

            # Update with adaptive conservatism
            critic_metrics = self.sac_agent.update_critic(mixed_batch, cql_weight=cql_w)
            actor_metrics = self.sac_agent.update_actor(mixed_batch)
            temp_metrics = self.sac_agent.update_temperature(mixed_batch)
            self.sac_agent.update_target_critic()

            metrics = {**critic_metrics, **actor_metrics, **temp_metrics, 'cql_weight': cql_w}
            history.append(metrics)

        return history

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        if self._phase == 'offline':
            # Evaluate CQL agent
            from offline_rl.agents.cql_geodesic import GeodesicCQL
            # Use CQL actor directly
            return self._evaluate_cql(n_episodes)
        else:
            return self.sac_agent.evaluate(self.eval_env, n_episodes)

    @torch.no_grad()
    def _evaluate_cql(self, n_episodes: int) -> Dict[str, float]:
        episode_returns, portfolio_values = [], []
        all_sharpes, all_max_drawdowns = [], []

        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            gru_hidden = self.cql_agent.regime_encoder.init_hidden(1, self.device)
            done = False
            ep_return = 0.0
            step_returns = []
            pv_trajectory = [1.0]
            info = {}

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                if self.bayesian:
                    regime, _, gru_hidden = self.cql_agent.regime_encoder.encode_step(
                        obs_t, gru_hidden, sample=False
                    )
                else:
                    regime, gru_hidden = self.cql_agent.regime_encoder.encode_step(obs_t, gru_hidden)
                w, _, _, _ = self.cql_agent.actor(obs_t, regime, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(w.squeeze(0).cpu().numpy())
                done = terminated or truncated
                ep_return += reward
                step_returns.append(reward)
                pv_trajectory.append(info.get('portfolio_value', pv_trajectory[-1]))

            episode_returns.append(ep_return)
            portfolio_values.append(info.get('portfolio_value', 1.0))

            # Sharpe ratio
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

        annual_returns = [(pv - 1.0) * (252 / self.eval_env.episode_length) for pv in portfolio_values]
        return {
            'eval/episode_return': float(np.mean(episode_returns)),
            'eval/portfolio_value': float(np.mean(portfolio_values)),
            'eval/annual_return': float(np.mean(annual_returns)),
            'eval/sharpe_ratio': float(np.mean(all_sharpes)),
            'eval/max_drawdown': float(np.mean(all_max_drawdowns)),
        }
