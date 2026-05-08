#!/usr/bin/env python3
"""Run reconstruction, clustering and VampPrior generation, write pickles.

Usage:
    python generate.py [fcvae_ckpt] [tcvae_ckpt]

Defaults: ``checkpoints/fcvae.pt`` and ``checkpoints/tcvae.pt``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from traffic.core import Traffic

from dtg.data import TrafficDataset
from dtg.fcvae import FCVAE
from dtg.generate import (
    cluster_latents,
    cluster_to_traffics,
    encode_dataset,
    latent_dataframe,
    reconstruct_one,
    vamp_generate,
    vamp_latent_pca,
)
from dtg.tcvae import TCVAE
from dtg.train import pick_device


# anchor coordinates (LSZH FAF, hardcoded — same as the paper)
_FAF = {"latitude": 47.546585, "longitude": 8.447731}

_RESULTS = Path("results")


def _load_fcvae(path: str) -> tuple[FCVAE, dict, MinMaxScaler]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = FCVAE(
        input_dim=ckpt["input_dim"],
        h_dims=cfg["model"]["h_dims"],
        latent_dim=cfg["model"]["latent_dim"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg, ckpt["scaler"]


def _load_tcvae(path: str) -> tuple[TCVAE, dict, MinMaxScaler]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = TCVAE(
        input_channels=ckpt["input_channels"],
        seq_len=ckpt["seq_len"],
        h_channels=cfg["model"]["h_channels"],
        latent_dim=cfg["model"]["latent_dim"],
        kernel_size=cfg["model"]["kernel_size"],
        dilation_base=cfg["model"]["dilation_base"],
        sampling_factor=cfg["model"]["sampling_factor"],
        n_pseudo_inputs=cfg["model"]["n_pseudo_inputs"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg, ckpt["scaler"]


def main(
    fcvae_ckpt: str = "checkpoints/fcvae.pt",
    tcvae_ckpt: str = "checkpoints/tcvae.pt",
) -> None:
    print("Loading checkpoints...")
    fcvae, fcvae_cfg, fcvae_scaler = _load_fcvae(fcvae_ckpt)
    tcvae, tcvae_cfg, tcvae_scaler = _load_tcvae(tcvae_ckpt)

    device = pick_device("auto")
    print(f"Using device: {device}")
    fcvae.to(device).eval()
    tcvae.to(device).eval()

    print("Loading dataset...")
    ds_fcvae_cfg = fcvae_cfg["data"]
    ds_tcvae_cfg = tcvae_cfg["data"]
    fcvae_ds = TrafficDataset.from_file(
        ds_fcvae_cfg["path"],
        features=ds_fcvae_cfg["features"],
        shape="linear",
        scaler=fcvae_scaler,
        info_features=ds_fcvae_cfg["info_features"],
        info_index=ds_fcvae_cfg["info_index"],
    )
    tcvae_ds = TrafficDataset.from_file(
        ds_tcvae_cfg["path"],
        features=ds_tcvae_cfg["features"],
        shape="image",
        scaler=tcvae_scaler,
        info_features=ds_tcvae_cfg["info_features"],
        info_index=ds_tcvae_cfg["info_index"],
    )

    features = ds_fcvae_cfg["features"]

    # ---- reconstruction (single trajectory) -------------------------------
    (_RESULTS / "reconstruction").mkdir(parents=True, exist_ok=True)
    print("Reconstructing one trajectory per model...")
    j = 10795
    original_traffic = Traffic.from_file(ds_fcvae_cfg["path"])
    fcvae_recon = reconstruct_one(
        fcvae, fcvae_ds.data, j, fcvae_scaler, features, _FAF, device,
        original_traffic=original_traffic,
    )
    tcvae_recon = reconstruct_one(
        tcvae, tcvae_ds.data, j, tcvae_scaler, features, _FAF, device,
        original_traffic=original_traffic,
    )
    fcvae_recon.to_pickle(_RESULTS / "reconstruction" / "reconstruction_fcvae.pkl")
    tcvae_recon.to_pickle(_RESULTS / "reconstruction" / "reconstruction_tcvae.pkl")

    # ---- clustering -------------------------------------------------------
    (_RESULTS / "clustering").mkdir(parents=True, exist_ok=True)
    print("Clustering FCVAE latents...")
    Z_fcvae = encode_dataset(fcvae, fcvae_ds.data, device)
    Z2_fcvae, labels_fcvae = cluster_latents(Z_fcvae, n_clusters=4)
    df_fcvae = latent_dataframe(Z2_fcvae, labels_fcvae)
    df_fcvae.to_pickle(_RESULTS / "clustering" / "Z_embedded_fcvae.pkl")
    traffics_fcvae = cluster_to_traffics(
        fcvae, Z_fcvae, labels_fcvae, fcvae_scaler, features, _FAF, device
    )
    with open(_RESULTS / "clustering" / "traffics_clust_fcvae.pkl", "wb") as f:
        pickle.dump(traffics_fcvae, f)

    print("Clustering TCVAE latents...")
    Z_tcvae = encode_dataset(tcvae, tcvae_ds.data, device)
    Z2_tcvae, labels_tcvae = cluster_latents(Z_tcvae, n_clusters=7)
    df_tcvae = latent_dataframe(Z2_tcvae, labels_tcvae)
    df_tcvae.to_pickle(_RESULTS / "clustering" / "Z_embedded_tcvae.pkl")
    traffics_tcvae = cluster_to_traffics(
        tcvae, Z_tcvae, labels_tcvae, tcvae_scaler, features, _FAF, device
    )
    with open(_RESULTS / "clustering" / "traffics_clust_tcvae.pkl", "wb") as f:
        pickle.dump(traffics_tcvae, f)

    # ---- VampPrior generation --------------------------------------------
    (_RESULTS / "generation").mkdir(parents=True, exist_ok=True)
    print("Generating VampPrior trajectories...")
    component_indexes = [262, 787]
    n_per = 100
    gen_latents, decoded = vamp_generate(
        tcvae, component_indexes=component_indexes, n_samples=n_per, device=device
    )
    df_gen = vamp_latent_pca(Z_tcvae, gen_latents, n_per_component=n_per)
    df_gen.to_pickle(_RESULTS / "generation" / "latent_space_vampprior_tcvae.pkl")

    from dtg.traffic_builder import build_traffic
    traf_gen1 = build_traffic(
        decoded[0], scaler=tcvae_scaler, features=features,
        coordinates=_FAF, forward=False,
    ).assign(gen_number=1)
    traf_gen2 = build_traffic(
        decoded[1], scaler=tcvae_scaler, features=features,
        coordinates=_FAF, forward=False,
    ).assign(gen_number=2)
    traf_gen1.to_pickle(_RESULTS / "generation" / "tcvae_traf_gen1.pkl")
    traf_gen2.to_pickle(_RESULTS / "generation" / "tcvae_traf_gen2.pkl")

    print("Done.")


if __name__ == "__main__":
    fcvae_ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fcvae.pt"
    tcvae_ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/tcvae.pt"
    main(fcvae_ckpt, tcvae_ckpt)
