"""Temporal-convolutional VAE with VampPrior."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from dtg.lsr import VampPrior


class TemporalBlock(nn.Module):
    """Causal, dilated 1D conv + activation + dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        )
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.dropout(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: nn.Module,
        is_last: bool,
    ) -> None:
        super().__init__()
        self.block1 = TemporalBlock(
            in_channels, out_channels, kernel_size, dilation, dropout, activation
        )
        self.block2 = TemporalBlock(
            out_channels,
            out_channels,
            kernel_size,
            dilation,
            dropout,
            None if is_last else activation,
        )
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        if self.skip is not None:
            self.skip.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block2(self.block1(x))
        r = x if self.skip is None else self.skip(x)
        return y + r


class TCN(nn.Module):
    """Temporal Convolutional Network (stack of dilated residual blocks)."""

    def __init__(
        self,
        in_channels: int,
        h_channels: Sequence[int],
        kernel_size: int,
        dilation_base: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        dims = [in_channels, *h_channels]
        blocks: list[nn.Module] = []
        for i in range(len(dims) - 1):
            blocks.append(
                ResidualBlock(
                    dims[i],
                    dims[i + 1],
                    kernel_size,
                    dilation_base ** i,
                    dropout,
                    activation,
                    is_last=(i == len(dims) - 2),
                )
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCDecoder(nn.Module):
    """Inverse of the TCN encoder: linear unprojection -> upsample -> TCN."""

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        h_channels: Sequence[int],
        seq_len: int,
        sampling_factor: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.sampling_factor = sampling_factor
        self.first_channels = h_channels[0]

        self.entry = nn.Linear(
            latent_dim, h_channels[0] * (seq_len // sampling_factor)
        )
        self.upsample = nn.Upsample(scale_factor=sampling_factor)
        self.tcn = TCN(
            in_channels=h_channels[0],
            h_channels=[*h_channels[1:], out_channels],
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            activation=nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.entry(z)
        x = x.view(z.size(0), self.first_channels, self.seq_len // self.sampling_factor)
        x = self.upsample(x)
        return self.tcn(x)


class TCEncoder(nn.Module):
    """TCN -> avg-pool over time -> flatten."""

    def __init__(
        self,
        in_channels: int,
        h_channels: Sequence[int],
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tcn = TCN(
            in_channels=in_channels,
            h_channels=h_channels,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            activation=nn.ReLU(),
        )
        self.pool = nn.AvgPool1d(sampling_factor)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(self.tcn(x)))


class TCVAE(nn.Module):
    """Temporal-convolutional VAE with VampPrior.

    Inputs are (batch, channels, seq_len) tensors.
    """

    def __init__(
        self,
        input_channels: int,
        seq_len: int,
        h_channels: Sequence[int],
        latent_dim: int,
        kernel_size: int = 16,
        dilation_base: int = 2,
        sampling_factor: int = 10,
        n_pseudo_inputs: int = 1000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.encoder = TCEncoder(
            in_channels=input_channels,
            h_channels=h_channels,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            sampling_factor=sampling_factor,
            dropout=dropout,
        )

        h_flat = h_channels[-1] * (seq_len // sampling_factor)

        self.lsr = VampPrior(
            input_dim=h_flat,
            latent_dim=latent_dim,
            encoder=self.encoder,
            n_components=n_pseudo_inputs,
            pseudo_input_shape=(input_channels, seq_len),
        )

        self.decoder = TCDecoder(
            latent_dim=latent_dim,
            out_channels=input_channels,
            h_channels=list(h_channels)[::-1],
            seq_len=seq_len,
            sampling_factor=sampling_factor,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.lsr(h)  # also refreshes prior params
        z = q.rsample()
        x_hat = self.decoder(z)
        # log q(z|x) and log p(z) — used by tcvae_loss
        log_q = q.log_prob(z)
        log_p = self.lsr.get_prior().log_prob(z)
        return x_hat, log_q, log_p, z

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        # ensure prior params are current
        self.lsr._compute_prior_params()
        z = self.lsr.get_prior().sample(torch.Size([n])).to(device)
        return self.decoder(z)


def tcvae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    scale: torch.Tensor,
    kl_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Negative ELBO with Monte-Carlo KL estimate (single z sample)."""
    var = scale.pow(2)
    two_pi = torch.log(torch.tensor(6.2831853))
    recon = 0.5 * (((x - x_hat) ** 2) / var + var.log() + two_pi).flatten(1).sum(dim=1)
    kl = log_q - log_p
    elbo = recon + kl_weight * kl
    return {
        "loss": elbo.mean(),
        "recon": recon.mean(),
        "kl": kl.mean(),
    }
