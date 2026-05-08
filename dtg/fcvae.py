"""Fully-connected VAE."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from dtg.lsr import NormalLSR


def _mlp(
    dims: Sequence[int],
    activation: nn.Module = nn.ReLU(),
    batch_norm: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Stack Linear + (BN + activation + dropout) blocks. Last layer is bare."""
    layers: list[nn.Module] = []
    n = len(dims) - 1
    for i in range(n):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i != n - 1:
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FCVAE(nn.Module):
    """Fully-connected VAE with a standard N(0, I) prior.

    Args:
        input_dim: flat input dimension (features * seq_len).
        h_dims: list of hidden sizes; the last entry is the encoder output
            size (before the latent heads). The decoder mirrors this list.
        latent_dim: latent space dimensionality.
        dropout: optional dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        h_dims: Sequence[int],
        latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder: input_dim -> h_dims[-1]
        self.encoder = _mlp(
            [input_dim, *h_dims], dropout=dropout
        )
        # latent heads
        self.lsr = NormalLSR(input_dim=h_dims[-1], latent_dim=latent_dim)
        # decoder: latent_dim -> input_dim, mirrors encoder
        rev = list(h_dims)[::-1]
        self.decoder = _mlp(
            [latent_dim, *rev, input_dim], dropout=dropout
        )

        # learned scale of the Gaussian likelihood, lower-bounded for stability
        self.log_scale = nn.Parameter(torch.zeros(()))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.lsr(h)
        return q.base_dist.loc, q.base_dist.scale

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.lsr(h)
        z = q.rsample()
        x_hat = self.decoder(z)
        return x_hat, q.base_dist.loc, q.base_dist.scale

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


def fcvae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    scale: torch.Tensor,
    kl_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Negative ELBO with closed-form KL against N(0, I).

    Returns ``{loss, recon, kl}`` (all scalars).
    """
    # Gaussian NLL: -log p(x|z) = sum 0.5 log(2 pi sigma^2) + 0.5 (x-x_hat)^2/sigma^2
    var = scale.pow(2)
    recon = 0.5 * (
        ((x - x_hat) ** 2) / var + var.log() + torch.log(torch.tensor(6.2831853))
    ).flatten(1).sum(dim=1)

    # closed-form KL(N(mu, sigma) || N(0, 1))
    kl = 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * sigma.log() - 1).sum(dim=1)

    elbo = recon + kl_weight * kl
    return {
        "loss": elbo.mean(),
        "recon": recon.mean(),
        "kl": kl.mean(),
    }
