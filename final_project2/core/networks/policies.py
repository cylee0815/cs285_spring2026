"""
Policy networks for portfolio RL.

Three architectures:
  - MLPPolicy: feedforward MLP (stateless)
  - LSTMPolicy: LSTM-based recurrent policy
  - TransformerPolicy: causal transformer with fixed context window

All share the same interface for PPO training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
from typing import Optional, Tuple


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPPolicy(nn.Module):
    """Feedforward MLP actor-critic for portfolio allocation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_std_init: float = -0.5,
    ):
        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers += [layer_init(nn.Linear(in_dim, hidden_dim)), nn.Tanh()]
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Actor head
        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Critic head
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden=None,
        action: Optional[torch.Tensor] = None,
    ):
        features = self.get_features(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value, None  # None = no hidden state

    def get_value(self, obs: torch.Tensor, hidden=None):
        return self.critic(self.get_features(obs)).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, hidden=None):
        features = self.get_features(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return log_prob, entropy, value


class LSTMPolicy(nn.Module):
    """
    LSTM-based actor-critic.
    Processes observations sequentially; hidden state carries temporal memory.

    Usage during rollout: pass (obs_t, hidden_t) → get action, get hidden_{t+1}
    Usage during update: feed full episodes with zero-initialized hidden state.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lstm_layers: int = 1,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # Input projection
        self.input_proj = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh()
        )

        # LSTM core
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)

        # Actor head
        self.actor_mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Critic head
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize zero hidden state."""
        return (
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device),
        )

    def _forward_lstm(
        self,
        obs: torch.Tensor,
        hidden,
        done: Optional[torch.Tensor] = None,
    ):
        """
        Forward through LSTM.
        obs: (batch, seq_len, obs_dim) or (batch, obs_dim)
        hidden: (h, c) each (lstm_layers, batch, hidden_dim)
        done: (batch, seq_len) or (batch,) or None — resets hidden at episode boundaries
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
            squeeze = True
        else:
            squeeze = False

        x = self.input_proj(obs)  # (batch, seq_len, hidden_dim)

        if done is not None and done.any():
            # Step through sequence, resetting hidden at done steps
            outputs = []
            h, c = hidden
            for t in range(x.shape[1]):
                if done.dim() == 2:
                    mask = (1.0 - done[:, t].float()).unsqueeze(0).unsqueeze(-1)
                else:
                    mask = (1.0 - done.float()).unsqueeze(0).unsqueeze(-1)
                h = h * mask
                c = c * mask
                out, (h, c) = self.lstm(x[:, t : t + 1, :], (h, c))
                outputs.append(out)
            lstm_out = torch.cat(outputs, dim=1)
            new_hidden = (h, c)
        else:
            lstm_out, new_hidden = self.lstm(x, hidden)

        if squeeze:
            lstm_out = lstm_out.squeeze(1)

        return lstm_out, new_hidden

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden=None,
        action: Optional[torch.Tensor] = None,
        done=None,
    ):
        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        lstm_out, new_hidden = self._forward_lstm(obs, hidden, done)

        mean = self.actor_mean(lstm_out)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(lstm_out).squeeze(-1)

        return action, log_prob, entropy, value, new_hidden

    def get_value(self, obs: torch.Tensor, hidden=None, done=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)
        lstm_out, _ = self._forward_lstm(obs, hidden, done)
        return self.critic(lstm_out).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden=None,
        done=None,
    ):
        """Used during PPO update — feed full episode sequences or flat minibatch."""
        # When called with a flat (T, obs_dim) batch, treat each step as seq_len=1
        # with independent zero hidden states.
        if obs.dim() == 2:
            batch = obs.shape[0]
            if hidden is None:
                hidden = self.init_hidden(batch, obs.device)
            # obs is (batch, obs_dim); _forward_lstm will unsqueeze to (batch, 1, obs_dim)
            lstm_out, _ = self._forward_lstm(obs, hidden, done)
        else:
            # obs is (batch, seq_len, obs_dim)
            batch = obs.shape[0]
            if hidden is None:
                hidden = self.init_hidden(batch, obs.device)
            lstm_out, _ = self._forward_lstm(obs, hidden, done)

        mean = self.actor_mean(lstm_out)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)

        # Align action shape with distribution batch shape
        if actions.shape != mean.shape:
            actions = actions.view_as(mean)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(lstm_out).squeeze(-1)
        return log_prob, entropy, value


class TransformerPolicy(nn.Module):
    """
    Causal transformer actor-critic.
    Maintains a fixed-length context window of past observations.
    Uses causal masking so position t only attends to positions <= t.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.context_len = context_len
        self.d_model = d_model
        self.obs_dim = obs_dim

        # Input projection
        self.input_proj = nn.Linear(obs_dim, d_model)

        # Positional encoding (learnable)
        self.pos_embed = nn.Embedding(context_len, d_model)

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Actor head
        self.actor_mean = layer_init(nn.Linear(d_model, action_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Critic head
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)

        # Register causal mask buffer
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _forward_transformer(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: (batch, seq_len, obs_dim) where seq_len <= context_len
        Returns: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = obs_seq.shape
        x = self.input_proj(obs_seq)

        positions = torch.arange(seq_len, device=obs_seq.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Use causal mask for current seq_len
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        out = self.transformer(x, mask=causal_mask, is_causal=True)
        return out

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,  # context buffer: (batch, context_len-1, obs_dim)
        action: Optional[torch.Tensor] = None,
    ):
        """
        obs: (batch, obs_dim) or (obs_dim,) — current observation
        hidden: context buffer (batch, context_len-1, obs_dim) or None
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]

        if hidden is None:
            # Initialize context with zeros
            ctx = torch.zeros(batch, self.context_len - 1, self.obs_dim, device=obs.device)
        else:
            ctx = hidden

        # Append current obs to context
        full_ctx = torch.cat([ctx, obs.unsqueeze(1)], dim=1)  # (batch, context_len, obs_dim)

        transformer_out = self._forward_transformer(full_ctx)  # (batch, context_len, d_model)
        features = transformer_out[:, -1, :]  # Use last token as representation

        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)

        # Update context: shift left and append current obs
        new_ctx = torch.cat([ctx[:, 1:, :], obs.unsqueeze(1)], dim=1)

        return action, log_prob, entropy, value, new_ctx

    def get_value(self, obs: torch.Tensor, hidden=None):
        _, _, _, value, _ = self.get_action_and_value(obs, hidden)
        return value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden=None,
    ):
        """
        For PPO update. Process flat buffer — each step independently with zero context.
        obs: (T, obs_dim) flat
        actions: (T, action_dim) flat
        """
        if obs.dim() == 2:
            # Flat buffer: treat each step independently with zero context
            batch = obs.shape[0]
            ctx = torch.zeros(batch, self.context_len - 1, self.obs_dim, device=obs.device)
            full_ctx = torch.cat([ctx, obs.unsqueeze(1)], dim=1)  # (batch, context_len, obs_dim)
            transformer_out = self._forward_transformer(full_ctx)
            features = transformer_out[:, -1, :]
        else:
            # Already shaped as (batch, seq_len, obs_dim)
            transformer_out = self._forward_transformer(obs)
            features = transformer_out[:, -1, :]

        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Independent(Normal(mean, std), 1)

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return log_prob, entropy, value


def make_policy(arch: str, obs_dim: int, action_dim: int, config) -> nn.Module:
    """Factory function to create a policy network by architecture name."""
    if arch == "mlp":
        return MLPPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        )
    elif arch == "lstm":
        return LSTMPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            lstm_layers=getattr(config, "lstm_layers", 1),
        )
    elif arch == "transformer":
        return TransformerPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_len=getattr(config, "context_len", 20),
            d_model=getattr(config, "d_model", 128),
            n_heads=getattr(config, "n_heads", 4),
            n_layers=getattr(config, "n_layers", 2),
        )
    else:
        raise ValueError(
            f"Unknown architecture: {arch}. Choose from 'mlp', 'lstm', 'transformer'."
        )
