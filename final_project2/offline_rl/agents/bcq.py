"""
Batch-Constrained Q-Learning (BCQ) for offline portfolio optimization.

BCQ constrains the policy to stay near the behavioral distribution by:
1. Training a VAE to model p(a|s) from the offline data
2. Generating K candidate actions from the VAE
3. Perturbing candidates with a learned perturbation network
4. Selecting the candidate with the highest Q-value

This avoids querying Q on OOD actions entirely.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional

from offline_rl.networks.dirichlet_policy import DoubleCritic
from offline_rl.networks.vae import VAE, PerturbationNetwork
from offline_rl.agents.replay_buffer import ReplayBuffer


class BCQAgent:
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
        self.n_step = getattr(config, 'n_step', 1)

        # VAE for behavioral action generation
        self.vae = VAE(
            obs_dim, action_dim, config.latent_dim,
            config.hidden_dim, config.n_layers,
        ).to(device)

        # Perturbation network
        self.perturbation = PerturbationNetwork(
            obs_dim, action_dim, config.hidden_dim,
            config.n_layers, phi=config.bcq_phi,
        ).to(device)

        # Double critic
        self.critic = DoubleCritic(
            obs_dim, action_dim, config.hidden_dim, config.n_layers,
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.vae_opt = optim.Adam(self.vae.parameters(), lr=config.lr)
        self.perturbation_opt = optim.Adam(self.perturbation.parameters(), lr=config.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr)

    def update_vae(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']

        # Normalize to simplex
        behavioral_w = actions.clamp(1e-7, 1.0)
        behavioral_w = behavioral_w / behavioral_w.sum(dim=-1, keepdim=True)

        recon, mu, log_sigma = self.vae(obs, behavioral_w)

        # Reconstruction loss (MSE on simplex)
        recon_loss = nn.functional.mse_loss(recon, behavioral_w)
        # KL divergence
        kl_loss = -0.5 * (1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()).sum(dim=-1).mean()

        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_opt.zero_grad()
        vae_loss.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), self.config.max_grad_norm)
        self.vae_opt.step()

        return {
            'bcq/vae_loss': vae_loss.item(),
            'bcq/recon_loss': recon_loss.item(),
            'bcq/kl_loss': kl_loss.item(),
        }

    def update_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        with torch.no_grad():
            # Generate candidates from VAE + perturbation, select best
            K = self.config.bcq_n_candidates
            candidates = self.vae.sample(next_obs, n_samples=K)  # (batch, K, action_dim)
            bsz = next_obs.shape[0]

            next_obs_rep = next_obs.unsqueeze(1).expand(-1, K, -1).reshape(-1, self.obs_dim)
            candidates_flat = candidates.reshape(-1, self.action_dim)
            perturbed = self.perturbation(next_obs_rep, candidates_flat)

            target_q = self.target_critic.q_min(next_obs_rep, perturbed)
            target_q = target_q.reshape(bsz, K)
            # Select best candidate per sample
            best_q = target_q.max(dim=1).values
            gamma_n = self.config.gamma ** self.n_step
            target_q = rewards + (1.0 - dones) * gamma_n * best_q

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_opt.step()

        return {
            'bcq/critic_loss': critic_loss.item(),
            'bcq/q_mean': q1.mean().item(),
        }

    def update_perturbation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch['obs']

        # Generate VAE candidates and perturb
        with torch.no_grad():
            candidates = self.vae.sample(obs, n_samples=1).squeeze(1)  # (batch, action_dim)

        perturbed = self.perturbation(obs, candidates)
        # Maximize Q
        q = self.critic.q_min(obs, perturbed)
        perturbation_loss = -q.mean()

        self.perturbation_opt.zero_grad()
        perturbation_loss.backward()
        nn.utils.clip_grad_norm_(self.perturbation.parameters(), self.config.max_grad_norm)
        self.perturbation_opt.step()

        return {'bcq/perturbation_loss': perturbation_loss.item()}

    def update_target_critic(self):
        tau = self.config.polyak_tau
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def update(self) -> Dict[str, float]:
        batch = self.offline_buffer.sample(self.config.batch_size)
        metrics = {}
        metrics.update(self.update_vae(batch))
        metrics.update(self.update_critic(batch))
        metrics.update(self.update_perturbation(batch))
        self.update_target_critic()
        return metrics

    @torch.no_grad()
    def get_action(self, obs_t: torch.Tensor) -> np.ndarray:
        """Select best of K VAE candidates by Q-value."""
        K = self.config.bcq_n_candidates
        candidates = self.vae.sample(obs_t, n_samples=K)  # (1, K, action_dim)
        obs_rep = obs_t.expand(K, -1)
        candidates_flat = candidates.squeeze(0)  # (K, action_dim)
        perturbed = self.perturbation(obs_rep, candidates_flat)
        q = self.critic.q_min(obs_rep, perturbed)
        best_idx = q.argmax()
        return perturbed[best_idx].cpu().numpy()
