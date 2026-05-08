"""Matplotlib + altair plots for the paper figures."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from cartes.crs import EuroPP, PlateCarree
from traffic.core import Traffic
from traffic.data import airports


_COLOR_CYCLE = (
    "#a6cee3 #1f78b4 #b2df8a #33a02c #fb9a99 #e31a1c "
    "#fdbf6f #ff7f00 #cab2d6 #6a3d9a #ffff99 #b15928"
).split()


def plot_reconstruction(
    reconstruction_fcvae: Traffic, reconstruction_tcvae: Traffic, out: str | Path
) -> None:
    with plt.style.context("traffic"):
        fig, ax = plt.subplots(
            1, 2, figsize=(13, 8), subplot_kw=dict(projection=EuroPP())
        )
        ax[0].set_title("FCVAE reconstruction", pad=20, fontsize=20)
        reconstruction_fcvae[0].plot(ax[0], lw=2)
        reconstruction_fcvae[1].plot(ax[0], lw=2)

        ax[1].set_title("TCVAE reconstruction", pad=20, fontsize=20)
        reconstruction_tcvae[0].plot(ax[1], lw=2, label="original")
        reconstruction_tcvae[1].plot(ax[1], lw=2, label="reconstructed")
        ax[1].set_extent(ax[0].get_extent(crs=PlateCarree()))
        legend = fig.legend(
            loc="lower center", bbox_to_anchor=(0.5, 0.2), ncol=2, fontsize=18
        )
        legend.get_frame().set_edgecolor("none")
    fig.savefig(out, transparent=False, dpi=300)
    plt.close(fig)


def plot_clustering(
    Z: pd.DataFrame, traffics: list[Traffic], title: str, out: str | Path
) -> None:
    colors = [_COLOR_CYCLE[int(i)] for i in Z["label"]]
    with plt.style.context("traffic"):
        fig = plt.figure(figsize=(30, 15))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122, projection=EuroPP())

        ax0.scatter(Z["X1"], Z["X2"], s=4, c=colors)
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_title(f"{title} latent space", fontsize=30, pad=18)
        ax0.grid(False)

        ax1.set_title(f"{title} reconstructed trajectories", fontsize=30, pad=18)
        for i, traf in enumerate(traffics):
            traf.plot(ax1, alpha=0.2, color=_COLOR_CYCLE[i])
    fig.savefig(out, transparent=False, dpi=300)
    plt.close(fig)


def plot_generation_tcvae(
    Z: pd.DataFrame,
    traf_gen1: Traffic,
    traf_gen2: Traffic,
    out_png: str | Path,
    out_html: str | Path,
) -> None:
    with plt.style.context("traffic"):
        fig = plt.figure(figsize=(17, 12))
        ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222, projection=EuroPP())

        ax0.scatter(
            Z.query("type.isnull()").X1, Z.query("type.isnull()").X2,
            c="#bab0ac", s=4, label="Observed",
        )
        ax0.scatter(
            Z.query("type == 'GEN1'").X1, Z.query("type == 'GEN1'").X2,
            c="#9ecae9", s=8, label="Generation pseudo-input 1",
        )
        ax0.scatter(
            Z.query("type == 'GEN2'").X1, Z.query("type == 'GEN2'").X2,
            c="#ffbf79", s=8, label="Generation pseudo-input 2",
        )
        ax0.scatter(
            Z.query("type == 'PI1'").X1, Z.query("type == 'PI1'").X2,
            c="#4c78a8", s=50, label="Pseudo-input 1",
        )
        ax0.scatter(
            Z.query("type == 'PI2'").X1, Z.query("type == 'PI2'").X2,
            c="#f58518", s=50, label="Pseudo-input 2",
        )
        ax0.set_title("Latent Space", fontsize=18)

        legend = ax0.legend(loc="upper left", fontsize=12)
        legend.get_frame().set_edgecolor("none")
        for h in legend.legend_handles[:3]:
            h._sizes = [50]

        ax1.set_title("Generated synthetic trajectories", pad=0, fontsize=18)
        traf_gen1.plot(ax1, alpha=0.2, color="#9ecae9")
        traf_gen1["TRAJ_0"].plot(ax1, color="#4c78a8", lw=2)
        traf_gen2.plot(ax1, alpha=0.2, color="#ffbf79")
        traf_gen2["TRAJ_0"].plot(ax1, color="#f58518", lw=2)
        airports["LSZH"].point.plot(ax1)
        fig.tight_layout()
    fig.savefig(out_png, transparent=False, dpi=300)
    plt.close(fig)

    # altitude / groundspeed altair charts
    chart1 = _alt_chart(traf_gen1, "altitude", "#9ecae9", "#4c78a8")
    chart2 = _alt_chart(traf_gen1, "groundspeed", "#9ecae9", "#4c78a8")
    chart3 = _alt_chart(traf_gen2, "altitude", "#ffbf79", "#f58518")
    chart4 = _alt_chart(traf_gen2, "groundspeed", "#ffbf79", "#f58518")
    plots = (
        alt.vconcat(alt.hconcat(chart1, chart2), alt.hconcat(chart3, chart4))
        .configure_title(fontSize=18)
        .configure_axis(labelFontSize=12, titleFontSize=14)
    )
    plots.save(str(out_html), scale_factor=2.0)


def _alt_chart(
    traf: Traffic, y: str, dim_color: str, focus_color: str
) -> alt.LayerChart:
    """Per-trajectory line chart with the lead trajectory highlighted."""
    pseudo_id = "TRAJ_999"
    traf_with_focus = traf + traf["TRAJ_0"].assign(flight_id=pseudo_id)
    return alt.layer(
        *(
            flight.chart().encode(
                x=alt.X("timedelta", title="timedelta (in s)"),
                y=alt.Y(y, title=None),
                opacity=alt.condition(
                    alt.datum.flight_id == pseudo_id,
                    alt.value(1), alt.value(0.2),
                ),
                color=alt.condition(
                    alt.datum.flight_id == pseudo_id,
                    alt.value(focus_color), alt.value(dim_color),
                ),
            )
            for flight in traf_with_focus
        )
    ).properties(title=f"{y}")
