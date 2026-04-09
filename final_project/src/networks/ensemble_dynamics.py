"""
Probabilistic Ensemble Dynamics Model for MBPO/MOPO.

Each network predicts (delta_obs, reward) with mean and log-variance.
Ensemble disagreement provides epistemic uncertainty for MOPO.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class ProbabilisticDynamicsNet(nn.Module):
    """
    Single probabilistic dynamics network.
    Predicts delta_obs and reward with Gaussian mean and log-variance.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 400, n_layers: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        output_dim = 2 * (obs_dim + 1)  # mean + logvar for (delta_obs, reward)

        layers = []
        d = obs_dim + action_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(d, hidden_dim), nn.SiLU()])
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

        # Separate output log-variance bounds
        self.max_logvar = nn.Parameter(torch.ones(obs_dim + 1) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(obs_dim + 1) * -10.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Returns:
            mean: (batch, obs_dim+1) — predicted [delta_obs, reward]
            logvar: (batch, obs_dim+1) — log variance
        """
        x = torch.cat([obs, action], dim=-1)
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=-1)
        # Soft clamp logvar
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar


class EnsembleDynamics:
    """
    Ensemble of probabilistic dynamics models.
    Trains all members, selects top-k elites for rollouts.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_ensemble: int = 7,
        n_elites: int = 5,
        hidden_dim: int = 400,
        n_layers: int = 4,
        lr: float = 1e-3,
        device: torch.device = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_ensemble = n_ensemble
        self.n_elites = n_elites
        self.device = device or torch.device('cpu')

        self.networks = nn.ModuleList([
            ProbabilisticDynamicsNet(obs_dim, action_dim, hidden_dim, n_layers)
            for _ in range(n_ensemble)
        ]).to(self.device)

        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=lr)
            for net in self.networks
        ]

        self.elite_indices: List[int] = list(range(n_elites))
        self._validation_losses = [float('inf')] * n_ensemble

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train all ensemble members on a batch. Returns mean loss."""
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards'].unsqueeze(-1)
        next_obs = batch['next_obs']

        delta_obs = next_obs - obs
        target = torch.cat([delta_obs, rewards], dim=-1)

        losses = []
        for i, (net, opt) in enumerate(zip(self.networks, self.optimizers)):
            mean, logvar = net(obs, actions)
            # Gaussian NLL
            inv_var = (-logvar).exp()
            mse = ((mean - target) ** 2) * inv_var
            nll = (mse + logvar).sum(dim=-1).mean()
            # Regularize logvar bounds
            nll += 0.01 * (net.max_logvar.sum() - net.min_logvar.sum())

            opt.zero_grad()
            nll.backward()
            opt.step()
            losses.append(nll.item())

        return {
            'dynamics/mean_loss': np.mean(losses),
            'dynamics/std_loss': np.std(losses),
        }

    def update_elites(self, val_batch: Dict[str, torch.Tensor]):
        """Select top-k networks by validation loss."""
        obs = val_batch['obs']
        actions = val_batch['actions']
        rewards = val_batch['rewards'].unsqueeze(-1)
        next_obs = val_batch['next_obs']

        delta_obs = next_obs - obs
        target = torch.cat([delta_obs, rewards], dim=-1)

        val_losses = []
        with torch.no_grad():
            for net in self.networks:
                mean, logvar = net(obs, actions)
                mse = ((mean - target) ** 2).sum(dim=-1).mean()
                val_losses.append(mse.item())

        self._validation_losses = val_losses
        sorted_indices = np.argsort(val_losses)
        self.elite_indices = sorted_indices[:self.n_elites].tolist()

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        network_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next_obs and reward using a single elite network.
        Returns: (next_obs, reward)
        """
        if network_idx is None:
            network_idx = self.elite_indices[np.random.randint(len(self.elite_indices))]

        mean, logvar = self.networks[network_idx](obs, action)
        std = (0.5 * logvar).exp()
        sample = mean + std * torch.randn_like(mean)

        delta_obs = sample[:, :self.obs_dim]
        reward = sample[:, self.obs_dim:].squeeze(-1)
        next_obs = obs + delta_obs

        return next_obs, reward

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict using random elite, also return ensemble uncertainty.
        Uncertainty = mean pairwise L2 between elite predictions.
        """
        # Get predictions from all elites
        predictions = []
        for idx in self.elite_indices:
            mean, _ = self.networks[idx](obs, action)
            predictions.append(mean)
        predictions = torch.stack(predictions, dim=0)  # (n_elites, batch, obs_dim+1)

        # Uncertainty: std across elites
        uncertainty = predictions.std(dim=0).mean(dim=-1)  # (batch,)

        # Use random elite for actual prediction
        chosen = self.elite_indices[np.random.randint(len(self.elite_indices))]
        next_obs, reward = self.predict(obs, action, network_idx=chosen)

        return next_obs, reward, uncertainty
