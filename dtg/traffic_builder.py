"""Decoded numpy arrays -> traffic.core.Traffic objects.

Reverses the dataset's scaler, integrates timedelta into a timestamp axis,
and reconstructs latitude/longitude by walking from a known anchor point
along the predicted (track, groundspeed) sequence.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pitot.geodesy import destination
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic


_KTS_TO_M_PER_S = 1852.0 / 3600.0


def _walk_latlon(
    track: np.ndarray,
    groundspeed: np.ndarray,
    timedelta: np.ndarray,
    anchor: tuple[float, float],
    forward: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate (track, groundspeed, dt) from an anchor lat/lon."""
    n = len(track)
    lat = np.empty(n)
    lon = np.empty(n)

    if forward:
        lat[0], lon[0] = anchor
        for i in range(1, n):
            dt = timedelta[i] - timedelta[i - 1]
            d = 0.99 * groundspeed[i - 1] * _KTS_TO_M_PER_S * dt
            lat[i], lon[i], _ = destination(
                lat[i - 1], lon[i - 1], track[i - 1], d
            )
        return lat, lon

    # walk backwards from the last point
    track_r = (track[::-1] - 180.0) % 360.0
    gs_r = groundspeed[::-1]
    td_r = timedelta[::-1]
    lat[0], lon[0] = anchor
    for i in range(1, n):
        dt = td_r[i - 1] - td_r[i]
        d = 0.99 * gs_r[i - 1] * _KTS_TO_M_PER_S * dt
        lat[i], lon[i], _ = destination(lat[i - 1], lon[i - 1], track_r[i - 1], d)
    return lat[::-1], lon[::-1]


def build_traffic(
    decoded: torch.Tensor | np.ndarray,
    scaler: MinMaxScaler,
    features: Sequence[str],
    coordinates: dict[str, float],
    forward: bool = False,
    base_ts: pd.Timestamp | None = None,
) -> Traffic:
    """Build a Traffic object from decoded model outputs.

    Args:
        decoded: tensor of shape (N, features, seq_len) or (N, features*seq_len).
        scaler: the dataset's MinMaxScaler (used to invert).
        features: ordered feature names (must include
            ``track``, ``groundspeed``, ``timedelta``).
        coordinates: ``{latitude, longitude}`` of the anchor point.
        forward: ``True`` if anchor is the first sample, ``False`` if it's the last.
        base_ts: timestamp anchor (defaults to "now" rounded to seconds).
    """
    if isinstance(decoded, torch.Tensor):
        arr = decoded.detach().cpu().numpy()
    else:
        arr = np.asarray(decoded)

    n_features = len(features)
    if arr.ndim == 3:  # (N, F, T) -> (N, T*F) by transposing
        arr = arr.transpose(0, 2, 1).reshape(arr.shape[0], -1)
    n_samples = arr.shape[0]

    # invert the scaler
    arr = scaler.inverse_transform(arr)
    # back to (N, T, F)
    arr = arr.reshape(n_samples, -1, n_features)
    n_obs = arr.shape[1]

    if base_ts is None:
        base_ts = pd.Timestamp.utcnow().tz_convert("UTC").round(freq="s")

    rows = []
    track_idx = features.index("track")
    gs_idx = features.index("groundspeed")
    td_idx = features.index("timedelta")
    for i in range(n_samples):
        traj = arr[i]
        lat, lon = _walk_latlon(
            track=traj[:, track_idx],
            groundspeed=traj[:, gs_idx],
            timedelta=traj[:, td_idx],
            anchor=(coordinates["latitude"], coordinates["longitude"]),
            forward=forward,
        )
        df = pd.DataFrame(traj, columns=list(features))
        df["latitude"] = lat
        df["longitude"] = lon
        df["timestamp"] = base_ts + pd.to_timedelta(df["timedelta"], unit="s")
        flight_id = f"TRAJ_{i}"
        df["flight_id"] = flight_id
        df["callsign"] = flight_id
        df["icao24"] = flight_id
        rows.append(df)

    return Traffic(pd.concat(rows, ignore_index=True))
