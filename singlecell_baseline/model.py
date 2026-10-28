"""Gene identity + expression/mask encoding + a bidirectional Transformer."""

import torch
from torch import nn


def make_mask(shape: tuple[int, int], fraction: float, seed: int) -> torch.Tensor:
    """Mask a fixed number of randomly chosen genes in each cell, including zeros."""
    if len(shape) != 2 or min(shape) < 1 or not 0 < fraction < 1:
        raise ValueError("Expected a nonempty (cells, genes) shape and 0 < fraction < 1.")
    if shape[1] < 2:
        raise ValueError("At least two genes are required to retain visible context.")
    count = min(shape[1] - 1, max(1, round(shape[1] * fraction)))
    random = torch.rand(shape, generator=torch.Generator().manual_seed(seed))
    mask = torch.zeros(shape, dtype=torch.bool)
    return mask.scatter_(1, random.topk(count, dim=1).indices, True)


def masked_input(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove target values at the input boundary; the independent mask distinguishes zeros."""
    return target.masked_fill(mask, 0.0)


class ExpressionTransformer(nn.Module):
    def __init__(self, n_genes: int, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        if n_genes < 2 or d_model < 1 or n_heads < 1 or d_model % n_heads or n_layers < 1:
            raise ValueError("Need >=2 genes, >=1 layer, and d_model divisible by n_heads.")
        self.gene_embedding = nn.Embedding(n_genes, d_model)
        self.value_mask_encoding = nn.Linear(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        # Encoder clones otherwise start with identical parameters when n_layers > 1.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("gene_mean", torch.zeros(n_genes))
        # Start at the baseline, then learn a context-dependent residual correction.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError("values and bool mask must have identical (batch, genes) shapes.")
        if values.shape[1] != self.gene_embedding.num_embeddings:
            raise ValueError("Gene count/order must match the checkpoint vocabulary.")
        values = masked_input(values, mask)  # Defense against accidentally passing full targets.
        gene_ids = torch.arange(values.shape[1], device=values.device)
        value_mask = torch.stack((values, mask.to(values.dtype)), dim=-1)
        tokens = self.gene_embedding(gene_ids)[None] + self.value_mask_encoding(value_mask)
        return self.gene_mean + self.head(self.encoder(tokens)).squeeze(-1)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        raise ValueError("Loss requires at least one masked position.")
    return (prediction[mask] - target[mask]).square().mean()
