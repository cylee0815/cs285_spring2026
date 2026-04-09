"""
Decision Transformer and Trajectory Transformer networks for portfolio RL.

Decision Transformer: causal GPT-style model over (RTG, obs, action) triples.
Trajectory Transformer: autoregressive model over discretized (obs, action) tokens.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional


class DecisionTransformerNet(nn.Module):
    """
    Decision Transformer: predict actions conditioned on return-to-go.

    Input sequence: [RTG_0, obs_0, act_0, RTG_1, obs_1, act_1, ...]
    Causal attention mask ensures each token only attends to prior tokens.
    Action prediction is extracted from the obs token positions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_rtg: float = 10.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.d_model = d_model
        self.max_rtg = max_rtg

        # Token type embeddings
        self.rtg_embed = nn.Linear(1, d_model)
        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)

        # Positional embedding: 3 tokens per timestep
        seq_len = 3 * context_len
        self.pos_embed = nn.Embedding(seq_len, d_model)

        # Layer norm
        self.embed_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction head (applied at obs token positions)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        )

    def forward(
        self,
        rtg_seq: torch.Tensor,     # (batch, context_len)
        obs_seq: torch.Tensor,     # (batch, context_len, obs_dim)
        action_seq: torch.Tensor,  # (batch, context_len, action_dim)
        timestep_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns predicted actions at each timestep: (batch, context_len, action_dim)
        """
        batch_size, T = obs_seq.shape[:2]

        # Normalize RTG
        rtg = (rtg_seq / self.max_rtg).clamp(-1.0, 1.0).unsqueeze(-1)  # (B, T, 1)

        # Embed each token type
        rtg_tokens = self.rtg_embed(rtg)         # (B, T, d_model)
        obs_tokens = self.obs_embed(obs_seq)      # (B, T, d_model)
        act_tokens = self.action_embed(action_seq)  # (B, T, d_model)

        # Interleave: [RTG_0, obs_0, act_0, RTG_1, obs_1, act_1, ...]
        seq_len = 3 * T
        tokens = torch.zeros(batch_size, seq_len, self.d_model, device=obs_seq.device)
        tokens[:, 0::3] = rtg_tokens
        tokens[:, 1::3] = obs_tokens
        tokens[:, 2::3] = act_tokens

        # Add positional embeddings
        positions = torch.arange(seq_len, device=obs_seq.device)
        tokens = tokens + self.pos_embed(positions)
        tokens = self.embed_ln(self.dropout(tokens))

        # Causal transformer
        mask = self.causal_mask[:seq_len, :seq_len]
        out = self.transformer(tokens, mask=mask)

        # Extract action predictions from obs token positions (index 1::3)
        obs_out = out[:, 1::3]  # (B, T, d_model)
        action_preds = self.action_head(obs_out)  # (B, T, action_dim)

        return action_preds

    def predict_action(
        self,
        rtg_seq: torch.Tensor,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next action (last timestep). Returns softmax weights."""
        preds = self.forward(rtg_seq, obs_seq, action_seq)
        return torch.softmax(preds[:, -1], dim=-1)


class SimplexDiscretizer:
    """
    Discretize continuous portfolio weights into bins for Trajectory Transformer.
    Each dimension is independently binned into n_bins uniform intervals over [0, 1].
    """

    def __init__(self, action_dim: int, n_bins: int = 32):
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Discretize continuous actions to bin indices. (batch, action_dim) -> (batch, action_dim) long."""
        actions_np = actions.clamp(0, 1).cpu().numpy()
        bins = np.digitize(actions_np, self.bin_edges[1:])  # 0-indexed
        bins = np.clip(bins, 0, self.n_bins - 1)
        return torch.tensor(bins, dtype=torch.long, device=actions.device)

    def decode(self, bins: torch.Tensor) -> torch.Tensor:
        """Convert bin indices back to continuous values (bin centers)."""
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        centers_t = torch.tensor(centers, dtype=torch.float32, device=bins.device)
        actions = centers_t[bins]  # lookup
        # Re-normalize to simplex
        actions = actions.clamp(1e-7, None)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        return actions


class TrajectoryTransformerNet(nn.Module):
    """
    Trajectory Transformer: autoregressive model over discretized (obs, action) tokens.

    For portfolio RL, obs tokens use continuous embeddings while action tokens
    use discrete bin embeddings (n_bins per dimension, predicted autoregressively).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_len: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_bins: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.d_model = d_model
        self.n_bins = n_bins

        # Obs: continuous embedding
        self.obs_embed = nn.Linear(obs_dim, d_model)

        # Action: discrete embedding per dimension
        # Each action dimension gets its own embedding table
        self.action_embeds = nn.ModuleList([
            nn.Embedding(n_bins, d_model) for _ in range(action_dim)
        ])

        # Combined action token: sum of per-dimension embeddings
        self.action_proj = nn.Linear(d_model * action_dim, d_model)

        # Positional embedding: 2 tokens per step (obs, action)
        seq_len = 2 * context_len
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.embed_ln = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Action prediction heads: one per dimension, predicting n_bins classes
        self.action_heads = nn.ModuleList([
            nn.Linear(d_model, n_bins) for _ in range(action_dim)
        ])

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        )

    def forward(
        self,
        obs_seq: torch.Tensor,        # (B, T, obs_dim)
        action_bins_seq: torch.Tensor,  # (B, T, action_dim) — discretized action indices
    ) -> torch.Tensor:
        """
        Returns logits for each action dimension: (B, T, action_dim, n_bins)
        """
        B, T = obs_seq.shape[:2]

        # Embed observations
        obs_tokens = self.obs_embed(obs_seq)  # (B, T, d_model)

        # Embed discretized actions: sum per-dimension embeddings
        act_embeds = []
        for d in range(self.action_dim):
            act_embeds.append(self.action_embeds[d](action_bins_seq[:, :, d]))
        act_concat = torch.cat(act_embeds, dim=-1)  # (B, T, d_model * action_dim)
        act_tokens = self.action_proj(act_concat)  # (B, T, d_model)

        # Interleave: [obs_0, act_0, obs_1, act_1, ...]
        seq_len = 2 * T
        tokens = torch.zeros(B, seq_len, self.d_model, device=obs_seq.device)
        tokens[:, 0::2] = obs_tokens
        tokens[:, 1::2] = act_tokens

        positions = torch.arange(seq_len, device=obs_seq.device)
        tokens = tokens + self.pos_embed(positions)
        tokens = self.embed_ln(self.dropout_layer(tokens))

        mask = self.causal_mask[:seq_len, :seq_len]
        out = self.transformer(tokens, mask=mask)

        # Predict action bins from obs token positions
        obs_out = out[:, 0::2]  # (B, T, d_model)
        action_logits = torch.stack(
            [head(obs_out) for head in self.action_heads], dim=2
        )  # (B, T, action_dim, n_bins)

        return action_logits
