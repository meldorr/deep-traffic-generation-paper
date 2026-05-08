"""Latent space regularization modules.

Two priors are provided:

- ``NormalLSR``: standard isotropic N(0, I) prior. Drop-in replacement
  if you want a vanilla VAE latent space.
- ``VampPrior``: variational mixture of posteriors prior, learned
  pseudo-inputs fed back through the encoder.

Both modules expose the same minimal interface: ``forward(hidden)`` returns
the posterior distribution, and ``get_prior()`` returns the prior. The model
calls ``rsample()`` on the posterior and uses ``log_prob()`` on both to
compute the KL term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import (
    Categorical,
    Distribution,
    Independent,
    MixtureSameFamily,
    Normal,
)


class NormalLSR(nn.Module):
    """Standard normal prior. Posterior is a diagonal Gaussian."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.z_loc = nn.Linear(input_dim, latent_dim)
        self.z_log_var = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Hardtanh(min_val=-6.0, max_val=2.0),
        )
        self.register_buffer(
            "prior_loc", torch.zeros(1, latent_dim), persistent=False
        )
        self.register_buffer(
            "prior_log_var", torch.zeros(1, latent_dim), persistent=False
        )

    def forward(self, hidden: torch.Tensor) -> Independent:
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        return Independent(Normal(loc, (log_var / 2).exp()), 1)

    def get_prior(self) -> Independent:
        return Independent(
            Normal(self.prior_loc, (self.prior_log_var / 2).exp()), 1
        )


class VampPrior(nn.Module):
    """VampPrior. The prior is a mixture of Gaussians whose component means
    are obtained by feeding learned pseudo-inputs through the encoder.

    Args:
        input_dim: size of the encoder output (post-pool, post-flatten).
        latent_dim: dimensionality of z.
        encoder: shared encoder module (used to embed pseudo-inputs).
        n_components: number of mixture components.
        pseudo_input_shape: shape (without batch dim) the encoder expects,
            e.g. ``(channels, seq_len)``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder: nn.Module,
        n_components: int,
        pseudo_input_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components
        self.encoder = encoder
        self.pseudo_input_shape = pseudo_input_shape

        # posterior heads
        self.z_loc = nn.Linear(input_dim, latent_dim)
        self.z_log_var = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Hardtanh(min_val=-6.0, max_val=2.0),
        )

        # per-component log_var head (decoupled from posterior)
        self.prior_log_var_NN = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Hardtanh(min_val=-6.0, max_val=2.0),
        )

        # pseudo-input generator: idle one-hot rows -> X-shaped tensors
        flat_input = 1
        for d in pseudo_input_shape:
            flat_input *= d
        self.pseudo_inputs_NN = nn.Sequential(
            nn.Linear(n_components, n_components),
            nn.ReLU(),
            nn.Linear(n_components, flat_input),
            nn.Hardtanh(min_val=-1.0, max_val=1.0),
        )

        # one-hot identity input -> non-persistent buffer so it follows .to()
        self.register_buffer(
            "idle_input", torch.eye(n_components), persistent=False
        )

        # learned mixture weights (logits)
        self.prior_weights = nn.Parameter(torch.ones(1, n_components))

        # cached prior parameters (recomputed each forward)
        self._prior_means: torch.Tensor | None = None
        self._prior_log_vars: torch.Tensor | None = None

    def _compute_prior_params(self) -> None:
        X = self.pseudo_inputs_NN(self.idle_input)
        X = X.view((self.n_components,) + tuple(self.pseudo_input_shape))
        h = self.encoder(X)
        self._prior_means = self.z_loc(h)
        self._prior_log_vars = self.prior_log_var_NN(h)

    def forward(self, hidden: torch.Tensor) -> Independent:
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        # refresh cached prior params for the next call to get_prior()
        self._compute_prior_params()
        return Independent(Normal(loc, (log_var / 2).exp()), 1)

    def get_prior(self) -> MixtureSameFamily:
        if self._prior_means is None:
            self._compute_prior_params()
        return MixtureSameFamily(
            Categorical(logits=self.prior_weights.view(self.n_components)),
            Independent(
                Normal(
                    self._prior_means,
                    (self._prior_log_vars / 2).exp(),
                ),
                1,
            ),
        )

    def pseudo_means_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (means, scales) of every pseudo-input component."""
        self._compute_prior_params()
        return self._prior_means, (self._prior_log_vars / 2).exp()
