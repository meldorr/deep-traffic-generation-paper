#!/usr/bin/env python3
"""Read pickles produced by ``generate.py`` and write paper figures."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from traffic.core import Traffic

from dtg.plotting import (
    plot_clustering,
    plot_generation_tcvae,
    plot_reconstruction,
)


_RESULTS = Path("results")
_FIG = _RESULTS / "figures"


def main() -> None:
    _FIG.mkdir(parents=True, exist_ok=True)

    print("Plotting reconstruction (figure 6)...")
    plot_reconstruction(
        Traffic.from_file(_RESULTS / "reconstruction" / "reconstruction_fcvae.pkl"),
        Traffic.from_file(_RESULTS / "reconstruction" / "reconstruction_tcvae.pkl"),
        out=_FIG / "figure_6.png",
    )

    print("Plotting FCVAE clustering (figure 7)...")
    Z_fcvae = pd.read_pickle(_RESULTS / "clustering" / "Z_embedded_fcvae.pkl")
    with open(_RESULTS / "clustering" / "traffics_clust_fcvae.pkl", "rb") as f:
        traffics_fcvae = pickle.load(f)
    plot_clustering(Z_fcvae, traffics_fcvae, "FCVAE", _FIG / "figure_7.png")

    print("Plotting TCVAE clustering (figure 8)...")
    Z_tcvae = pd.read_pickle(_RESULTS / "clustering" / "Z_embedded_tcvae.pkl")
    with open(_RESULTS / "clustering" / "traffics_clust_tcvae.pkl", "rb") as f:
        traffics_tcvae = pickle.load(f)
    plot_clustering(Z_tcvae, traffics_tcvae, "TCVAE", _FIG / "figure_8.png")

    print("Plotting VampPrior generation (figures 12, 13)...")
    Z_gen = pd.read_pickle(
        _RESULTS / "generation" / "latent_space_vampprior_tcvae.pkl"
    )
    traf_gen1 = Traffic.from_file(_RESULTS / "generation" / "tcvae_traf_gen1.pkl")
    traf_gen2 = Traffic.from_file(_RESULTS / "generation" / "tcvae_traf_gen2.pkl")
    plot_generation_tcvae(
        Z_gen, traf_gen1, traf_gen2,
        out_png=_FIG / "figure_12.png",
        out_html=_FIG / "figure_13.html",
    )

    print("Done.")


if __name__ == "__main__":
    main()
