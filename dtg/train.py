"""Plain torch training loop and device selection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader


def pick_device(preference: str = "auto") -> torch.device:
    """Pick a device: cuda > mps > cpu (or honor an explicit choice)."""
    pref = preference.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    loss_fn: Callable[..., dict[str, torch.Tensor]],
    *,
    epochs: int,
    lr: float = 1e-3,
    lr_step: int = 200,
    lr_gamma: float = 0.5,
    grad_clip: float = 0.5,
    kl_weight: float = 1.0,
    device: torch.device | str = "auto",
    checkpoint_path: str | Path | None = None,
    extra_to_save: dict | None = None,
) -> dict[str, list[float]]:
    """Train ``model`` for ``epochs`` epochs.

    ``loss_fn`` is called as ``loss_fn(x, *outputs, scale=model.scale,
    kl_weight=kl_weight)`` and must return a dict with at least ``"loss"``.

    ``model.forward(x)`` is expected to return a tuple of tensors which
    will be expanded as ``*outputs`` to ``loss_fn``.

    The best model (lowest val loss, or last train loss if no val) is saved
    to ``checkpoint_path``. ``extra_to_save`` is merged into the saved dict.
    """
    if not isinstance(device, torch.device):
        device = pick_device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_gamma
    )

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(x)
            losses = loss_fn(x, *outputs, scale=model.scale, kl_weight=kl_weight)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += losses["loss"].item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            running = 0.0
            n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device, non_blocking=True)
                    outputs = model(x)
                    losses = loss_fn(
                        x, *outputs, scale=model.scale, kl_weight=kl_weight
                    )
                    running += losses["loss"].item()
                    n_batches += 1
            val_loss = running / max(n_batches, 1)
            history["val_loss"].append(val_loss)

        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"epoch {epoch + 1:>4d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)"
        )

        # checkpoint best
        score = val_loss if val_loader is not None else train_loss
        if checkpoint_path is not None and score < best_loss:
            best_loss = score
            payload = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            if extra_to_save:
                payload.update(extra_to_save)
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_path)

    return history
