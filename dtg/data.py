"""Traffic dataset wrapper and dataloader factory."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, Dataset, random_split
from traffic.core import Traffic


class TrafficDataset(Dataset):
    """Tensorized view over a Traffic object.

    Args:
        traffic: source Traffic object.
        features: per-timestep features to keep (e.g.
            ``["track", "groundspeed", "altitude", "timedelta"]``).
        shape: ``"linear"`` flattens to ``(features * seq_len)``;
            ``"image"`` keeps ``(features, seq_len)``.
        scaler: a scikit-learn-style scaler. Fitted on first use.
        info_features: extra columns to keep aside (e.g. lat/lon at FAF).
        info_index: row index in each flight's frame to grab info from
            (``-1`` for last point, ``0`` for first).
    """

    def __init__(
        self,
        traffic: Traffic,
        features: Sequence[str],
        shape: str = "image",
        scaler: MinMaxScaler | None = None,
        info_features: Sequence[str] = (),
        info_index: int | None = -1,
    ) -> None:
        assert shape in ("linear", "image"), f"unsupported shape: {shape}"

        self.features = list(features)
        self.shape = shape
        self.info_features = list(info_features)
        self.info_index = info_index

        # stack each flight's features into a flat row
        data = np.stack(
            [f.data[self.features].values.ravel() for f in traffic]
        ).astype(np.float32)

        # fit/transform with scaler (default: MinMax to [-1, 1])
        self.scaler = scaler if scaler is not None else MinMaxScaler(
            feature_range=(-1, 1)
        )
        try:
            check_is_fitted(self.scaler)
            data = self.scaler.transform(data)
        except NotFittedError:
            data = self.scaler.fit_transform(data)

        tensor = torch.from_numpy(data).float()
        if shape == "image":
            # (N, features, seq_len)
            tensor = tensor.view(tensor.size(0), -1, len(self.features))
            tensor = tensor.transpose(1, 2)
        self.data = tensor

        # info columns at a single row index
        if info_index is not None and self.info_features:
            infos = np.asarray(
                [
                    f.data[self.info_features]
                    .iloc[info_index]
                    .values.ravel()
                    for f in traffic
                ],
                dtype=np.float32,
            )
            self.infos = torch.from_numpy(infos)
        else:
            self.infos = torch.empty(len(self.data), 0)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        features: Sequence[str],
        shape: str = "image",
        scaler: MinMaxScaler | None = None,
        info_features: Sequence[str] = (),
        info_index: int | None = -1,
    ) -> "TrafficDataset":
        return cls(
            Traffic.from_file(path),
            features,
            shape,
            scaler,
            info_features,
            info_index,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.infos[idx]

    @property
    def input_dim(self) -> int:
        """Per-sample feature dimension. For ``image`` shape this is the
        number of channels; for ``linear`` it is features * seq_len."""
        return self.data.shape[1] if self.shape == "image" else self.data.shape[-1]

    @property
    def seq_len(self) -> int:
        if self.shape == "image":
            return self.data.shape[2]
        return self.data.shape[-1] // len(self.features)


def get_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    """Split into train/val and return DataLoaders. Test split omitted."""
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_train * val_ratio)
    n_train -= n_val
    n_remainder = n_total - n_train - n_val  # unused (would be test)

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, _ = random_split(
        dataset, [n_train, n_val, n_remainder], generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if n_val > 0
        else None
    )
    return train_loader, val_loader
