"""Small residual aggregators operating on frozen ESMC chunk embeddings."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def __post_init__(self):
        # Recurrence supplies order; no explicit position features are used.
        # Preserve the metadata convention of existing Transformer/MLP runs.
        if self.architecture == "lstm":
            self.use_positions = False

    def validate(self):
        if min(self.embedding_dim, self.hidden_dim, self.num_layers,
               self.num_heads, self.feedforward_dim) < 1:
            raise ValueError("All model dimensions must be positive")
        # Attention splits the hidden width across heads; bidirectionality splits
        # the LSTM width equally between forward and backward recurrent states.
        if self.architecture == "transformer" and self.hidden_dim % self.num_heads:
            raise ValueError("Hidden dimension must be divisible by the number of heads")
        if self.architecture == "lstm" and self.hidden_dim % 2:
            raise ValueError("BiLSTM hidden dimension must be even (half per direction)")
        if not 0 <= self.dropout < 1 or self.architecture not in {"transformer", "mlp", "lstm"}:
            raise ValueError("Invalid dropout or architecture")


def weighted_mean(embeddings, weights):
    # embeddings: [batch, chunks, channels]; weights: [batch, chunks].
    # Callers supply normalized core-length weights so short final cores count less.
    return (embeddings.float() * weights.float().unsqueeze(-1)).sum(dim=1)


class ChunkAggregator(nn.Module):
    """Predict a low-rank correction to the residue-weighted chunk mean."""

    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        # Compress frozen ESMC features to a small trainable hidden space.
        # ESMC itself is external to this module and receives no gradient updates.
        self.input_norm = nn.LayerNorm(config.embedding_dim)
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        if config.architecture == "transformer":
            # Learn a projection of continuous chunk geometry rather than a fixed
            # lookup table whose size would restrict the number of input chunks.
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
        elif config.architecture == "lstm":
            # Concatenating both directions recovers the shared hidden_dim width.
            # PyTorch LSTM dropout acts between recurrent layers, not after the last.
            self.lstm = nn.LSTM(
                input_size=config.hidden_dim, hidden_size=config.hidden_dim // 2,
                num_layers=config.num_layers, batch_first=True, bidirectional=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0)
        else:
            # This control sees the pooled vector only and has no chunk-order signal.
            self.mlp = nn.Sequential(nn.GELU(), nn.Dropout(config.dropout),
                                     nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU())
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        # Start exactly at the weighted-mean baseline. Training learns a correction
        # instead of having to reconstruct the entire ESMC vector from scratch.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, embeddings, positions, weights, padding_mask):
        # Shapes: embeddings [B, C, D], positions [B, C, 5], weights/mask [B, C].
        # B = proteins, C = padded chunk count, D = ESMC embedding width;
        # True mask entries denote padding and must never contribute to pooling.
        # Explicitly zero padding, including arbitrary values provided by a caller.
        embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0)
        positions = positions.masked_fill(padding_mask.unsqueeze(-1), 0)
        weights = weights.masked_fill(padding_mask, 0)
        # Renormalize valid core weights after masking to preserve a mean per protein.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        baseline = weighted_mean(embeddings, weights)
        if self.config.architecture == "mlp":
            # Pool before the MLP, making its prediction invariant to chunk ordering.
            pooled = self.mlp(self.input_projection(self.input_norm(baseline)))
        else:
            hidden = self.input_projection(self.input_norm(embeddings))
            if self.config.architecture == "lstm":
                # Packing assumes each protein has a nonempty prefix of valid chunks.
                lengths = (~padding_mask).sum(dim=1)
                valid = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
                if torch.any(lengths == 0) or not torch.equal(~padding_mask, valid):
                    raise ValueError("BiLSTM requires nonempty sequences with right padding")
                # Masking only the pooled output would still let the backward
                # recurrence read padding. Packing excludes it in both directions.
                packed = pack_padded_sequence(hidden, lengths.cpu(), batch_first=True, enforce_sorted=False)
                encoded, _ = self.lstm(packed)
                hidden, _ = pad_packed_sequence(encoded, batch_first=True, total_length=hidden.shape[1])
            else:
                # Attention mixes valid chunks; the position features supply their
                # locations because self-attention alone does not encode sequence order.
                if self.position_projection is not None:
                    hidden = hidden + self.position_projection(positions)
                for layer in self.layers:
                    hidden = layer(hidden, src_key_padding_mask=padding_mask)
            # Padded queries can still have nonzero outputs after attention.
            # Remove them before the same residue-weighted pooling used by the baseline.
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0)
            pooled = weighted_mean(hidden, weights)
        # All architectures return one vector in the original ESMC coordinate space.
        correction = self.output_projection(self.output_norm(pooled))
        return baseline + correction.float()

    def specification(self):
        return asdict(self.config)


def reconstruction_loss(prediction, target, variance, cosine_weight=0.5):
    """Return one loss per protein; variance must come from training teachers."""
    prediction, target = prediction.float(), target.float()
    # Normalize coordinate error by training-teacher variance to set its loss scale.
    # Validation/test statistics must not influence this normalization.
    mse = (prediction - target).square().mean(dim=-1) / max(float(variance), 1e-8)
    # The cosine term also penalizes directional error, relevant to cosine retrieval.
    # Keep per-protein losses so the caller controls batch averaging.
    cosine = 1 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    return mse + cosine_weight * cosine
