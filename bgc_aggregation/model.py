"""Small residual aggregators operating on frozen ESMC chunk embeddings."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .common import POSITION_DIM


@dataclass
class ModelConfig:
    embedding_dim: int = 1152
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1
    architecture: str = "transformer"
    use_positions: bool = True

    def validate(self):
        if min(self.embedding_dim, self.hidden_dim, self.num_layers,
               self.num_heads, self.feedforward_dim) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("Hidden dimension must be divisible by the number of heads")
        if not 0 <= self.dropout < 1 or self.architecture not in {"transformer", "mlp"}:
            raise ValueError("Invalid dropout or architecture")


def weighted_mean(embeddings, weights):
    return (embeddings.float() * weights.float().unsqueeze(-1)).sum(dim=1)


class ChunkAggregator(nn.Module):
    """Predict a low-rank correction to the residue-weighted chunk mean."""

    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.input_norm = nn.LayerNorm(config.embedding_dim)
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        if config.architecture == "transformer":
            self.position_projection = nn.Sequential(
                nn.Linear(POSITION_DIM, config.hidden_dim), nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)) if config.use_positions else None
            # Construct layers separately so their initial weights are independent.
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim, nhead=config.num_heads,
                    dim_feedforward=config.feedforward_dim, dropout=config.dropout,
                    activation="gelu", batch_first=True, norm_first=True)
                for _ in range(config.num_layers)])
        else:
            self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(config.dropout),
                                     nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU())
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, embeddings, positions, weights, padding_mask):
        # Explicitly zero padding, including arbitrary values provided by a caller.
        embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0)
        positions = positions.masked_fill(padding_mask.unsqueeze(-1), 0)
        weights = weights.masked_fill(padding_mask, 0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        baseline = weighted_mean(embeddings, weights)
        if self.config.architecture == "mlp":
            pooled = self.mlp(self.input_projection(self.input_norm(baseline)))
        else:
            hidden = self.input_projection(self.input_norm(embeddings))
            if self.position_projection is not None:
                hidden = hidden + self.position_projection(positions)
            for layer in self.layers:
                hidden = layer(hidden, src_key_padding_mask=padding_mask)
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0)
            pooled = weighted_mean(hidden, weights)
        correction = self.output_projection(self.output_norm(pooled))
        return baseline + correction.float()

    def specification(self):
        return asdict(self.config)


def reconstruction_loss(prediction, target, variance, cosine_weight=0.5):
    """Return one loss per protein; variance must come from training teachers."""
    prediction, target = prediction.float(), target.float()
    mse = (prediction - target).square().mean(dim=-1) / max(float(variance), 1e-8)
    cosine = 1 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    return mse + cosine_weight * cosine
