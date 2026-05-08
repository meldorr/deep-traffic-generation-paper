#!/usr/bin/env python3
"""Train the FCVAE.

Usage:
    python train_fcvae.py [path/to/config.yaml]

If no config path is provided, ``configs/fcvae.yaml`` is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from dtg.data import TrafficDataset, get_dataloaders
from dtg.fcvae import FCVAE, fcvae_loss
from dtg.train import pick_device, train


def main(config_path: str = "configs/fcvae.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    torch.manual_seed(cfg["training"]["seed"])

    ds_cfg = cfg["data"]
    dataset = TrafficDataset.from_file(
        ds_cfg["path"],
        features=ds_cfg["features"],
        shape=ds_cfg["shape"],
        info_features=ds_cfg["info_features"],
        info_index=ds_cfg["info_index"],
    )

    train_loader, val_loader = get_dataloaders(
        dataset,
        batch_size=ds_cfg["batch_size"],
        train_ratio=ds_cfg["train_ratio"],
        val_ratio=ds_cfg["val_ratio"],
        num_workers=ds_cfg["num_workers"],
        seed=cfg["training"]["seed"],
    )

    device = pick_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    model = FCVAE(
        input_dim=dataset.input_dim,
        h_dims=cfg["model"]["h_dims"],
        latent_dim=cfg["model"]["latent_dim"],
        dropout=cfg["model"]["dropout"],
    )
    print(model)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=lambda x, x_hat, mu, sigma, *, scale, kl_weight: fcvae_loss(
            x, x_hat, mu, sigma, scale=scale, kl_weight=kl_weight
        ),
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        lr_step=cfg["training"]["lr_step"],
        lr_gamma=cfg["training"]["lr_gamma"],
        grad_clip=cfg["training"]["grad_clip"],
        kl_weight=cfg["training"]["kl_weight"],
        device=device,
        checkpoint_path=cfg["checkpoint"]["out"],
        extra_to_save={
            "config": cfg,
            "scaler": dataset.scaler,
            "input_dim": dataset.input_dim,
            "seq_len": dataset.seq_len,
        },
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/fcvae.yaml")
