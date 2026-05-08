"""Reconstruction, sampling and clustering helpers.

These functions take an already-loaded model and a dataset, run inference
(no gradients), and return numpy / pandas / Traffic outputs ready to be
pickled or plotted.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from traffic.core import Traffic

from dtg.traffic_builder import build_traffic


# ---------- low-level inference ---------------------------------------------


@torch.no_grad()
def encode_dataset(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Return the posterior mean of every sample as a numpy array.

    Encodes in chunks (to avoid one huge MPS/CUDA forward pass) and
    replaces any non-finite outputs with 0 so downstream sklearn ops
    don't overflow.
    """
    model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size].to(device)
        mu, _ = model.encode(batch)
        chunks.append(mu.detach().cpu().numpy())
    Z = np.concatenate(chunks, axis=0)

    bad = ~np.isfinite(Z)
    n_bad = int(bad.any(axis=1).sum())
    if n_bad:
        print(
            f"  warning: {n_bad}/{len(Z)} latents had non-finite values "
            f"(replaced with 0). The model may need retraining."
        )
        Z = np.where(bad, 0.0, Z)
    return Z


@torch.no_grad()
def decode_latents(
    model: torch.nn.Module, z: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Decode a batch of latents (returned on CPU)."""
    model.eval()
    return model.decoder(z.to(device)).detach().cpu()


# ---------- reconstruction ---------------------------------------------------


def reconstruct_one(
    model: torch.nn.Module,
    dataset_tensor: torch.Tensor,
    idx: int,
    scaler,
    features: Sequence[str],
    coordinates: dict[str, float],
    device: torch.device,
    original_traffic: Traffic | None = None,
) -> Traffic:
    """Reconstruct a single trajectory and wrap it as a Traffic.

    If ``original_traffic`` is provided, returns a Traffic containing both
    the original flight at ``idx`` and the reconstruction (in that order).
    """
    model.eval()
    x = dataset_tensor[idx].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    x_hat = out[0]  # convention: forward returns (x_hat, ...)
    reconstructed = build_traffic(
        x_hat, scaler=scaler, features=features, coordinates=coordinates,
        forward=False,
    )
    if original_traffic is None:
        return reconstructed
    return original_traffic[idx] + reconstructed


# ---------- clustering -------------------------------------------------------


def cluster_latents(
    Z: np.ndarray, n_clusters: int, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-2D + GaussianMixture. Returns ``(Z_2d, labels)``.

    sklearn matmul on macOS Accelerate raises spurious FP-exception flags
    even on well-conditioned inputs; we suppress those here so the output
    isn't drowned in noise.
    """
    with np.errstate(all="ignore"):
        Z_2d = PCA(n_components=2).fit_transform(Z)
        labels = GaussianMixture(
            n_components=n_clusters, random_state=random_state
        ).fit_predict(Z_2d)
    return Z_2d, labels


def latent_dataframe(Z_2d: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    arr = np.concatenate([Z_2d, labels[:, None]], axis=1)
    return pd.DataFrame(arr, columns=["X1", "X2", "label"])


def cluster_to_traffics(
    model: torch.nn.Module,
    Z: np.ndarray,
    labels: np.ndarray,
    scaler,
    features: Sequence[str],
    coordinates: dict[str, float],
    device: torch.device,
) -> list[Traffic]:
    """For each cluster, decode its members and build a Traffic object."""
    out: list[Traffic] = []
    for c in np.unique(labels):
        members = torch.from_numpy(Z[labels == c]).float()
        decoded = decode_latents(model, members, device)
        traf = build_traffic(
            decoded, scaler=scaler, features=features, coordinates=coordinates,
            forward=False,
        )
        traf = traf.assign(cluster=int(c))
        out.append(traf)
    return out


# ---------- VampPrior generation ---------------------------------------------


@torch.no_grad()
def vamp_generate(
    model: torch.nn.Module,
    component_indexes: Sequence[int],
    n_samples: int,
    device: torch.device,
) -> tuple[np.ndarray, list[torch.Tensor]]:
    """Sample ``n_samples`` latents around two pseudo-input components and
    decode them.

    Returns:
        latents_concat: ``(2 * n_samples + 2, latent_dim)`` numpy array.
            Order: gen1, gen2, pseudo_mean_1, pseudo_mean_2.
        decoded: list of two CPU tensors, each ``(n_samples + 1, ...)``,
            with the pseudo-input mean prepended at index 0.
    """
    model.eval()
    means, scales = model.lsr.pseudo_means_scales()
    decoded: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    for k, idx in enumerate(component_indexes):
        comp = torch.distributions.Independent(
            torch.distributions.Normal(means[idx], scales[idx]), 1
        )
        z = comp.sample(torch.Size([n_samples]))
        latents.append(z)
        z_with_mean = torch.cat([means[idx].unsqueeze(0), z], dim=0)
        decoded.append(model.decoder(z_with_mean).detach().cpu())

    # build the latent concat for downstream PCA: gen1, gen2, pi1, pi2
    pi_means = torch.stack([means[i] for i in component_indexes], dim=0)
    latent_cat = torch.cat([*latents, pi_means], dim=0)
    return latent_cat.detach().cpu().numpy(), decoded


def vamp_latent_pca(
    train_latents: np.ndarray,
    gen_latents: np.ndarray,
    n_per_component: int,
) -> pd.DataFrame:
    """PCA-project (train, gen1, gen2, pi1, pi2) into 2D and tag the rows."""
    concat = np.concatenate([train_latents, gen_latents], axis=0)
    with np.errstate(all="ignore"):
        pca = PCA(n_components=2).fit(concat[: -len(gen_latents)])
        proj = pca.transform(concat)
    df = pd.DataFrame(proj, columns=["X1", "X2"])
    df["type"] = pd.NA
    n_gen = n_per_component
    df.loc[df.index[-(2 * n_gen + 2):], "type"] = "GEN1"
    df.loc[df.index[-(n_gen + 2):], "type"] = "GEN2"
    df.loc[df.index[-2:], "type"] = "PI1"
    df.loc[df.index[-1:], "type"] = "PI2"
    return df
