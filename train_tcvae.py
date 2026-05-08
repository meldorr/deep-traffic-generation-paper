#!/usr/bin/env python3
"""Train the VampPrior TCVAE.

Usage:
    python train_tcvae.py [path/to/config.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from dtg.data import TrafficDataset, get_dataloaders
from dtg.tcvae import TCVAE, tcvae_loss
from dtg.train import pick_device, train


def main(config_path: str = "configs/tcvae.yaml") -> None:
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

    model = TCVAE(
        input_channels=dataset.input_dim,
        seq_len=dataset.seq_len,
        h_channels=cfg["model"]["h_channels"],
        latent_dim=cfg["model"]["latent_dim"],
        kernel_size=cfg["model"]["kernel_size"],
        dilation_base=cfg["model"]["dilation_base"],
        sampling_factor=cfg["model"]["sampling_factor"],
        n_pseudo_inputs=cfg["model"]["n_pseudo_inputs"],
        dropout=cfg["model"]["dropout"],
    )
    print(model)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=lambda x, x_hat, log_q, log_p, z, *, scale, kl_weight: tcvae_loss(
            x, x_hat, log_q, log_p, scale=scale, kl_weight=kl_weight
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
            "input_channels": dataset.input_dim,
            "seq_len": dataset.seq_len,
        },
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/tcvae.yaml")
