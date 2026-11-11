"""Overview figure: series, run-length posterior, recent-change probability, rolling NLPD."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import BOCDResult


def plot_overview(
    x_axis: Sequence,
    level: np.ndarray,
    result: BOCDResult,
    baseline_log_pred: np.ndarray | None = None,
    true_changepoints: Sequence | None = None,
    level_label: str = "Series",
    window: int = 5,
    rolling: int = 50,
    max_display_r: int | None = None,
):
    """Four-panel figure. ``level`` is plotted on top (e.g. price); the model was run on its increments."""
    post = result.run_length_posterior
    if max_display_r is not None:
        post = post[:, : max_display_r + 1]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.4, 0.8, 0.8]})

    axes[0].plot(x_axis, level, color="black", lw=0.9)
    axes[0].set_ylabel(level_label)

    mesh = axes[1].pcolormesh(
        x_axis, np.arange(post.shape[1]), post.T, cmap="magma", shading="auto",
        norm=mcolors.LogNorm(vmin=1e-4, vmax=1.0),
    )
    axes[1].plot(x_axis, result.map_run_length, color="cyan", lw=0.7, label="MAP run length")
    axes[1].set_ylabel("Run length r")
    axes[1].legend(loc="upper left", fontsize=8)
    cax = axes[1].inset_axes([1.005, 0.0, 0.012, 1.0])  # outside the axis, so panels stay aligned
    fig.colorbar(mesh, cax=cax, label="P(r_t | x_1:t)")

    axes[2].plot(x_axis, result.prob_recent_change(window), color="crimson", lw=0.8)
    axes[2].set_ylabel(f"P(r_t < {window})")
    axes[2].set_ylim(0, 1.02)

    roll = pd.Series(result.nlpd).rolling(rolling, min_periods=1).mean()
    axes[3].plot(x_axis, roll, color="navy", lw=1.0, label="BOCD (Student-t)")
    if baseline_log_pred is not None:
        roll_b = pd.Series(-baseline_log_pred).rolling(rolling, min_periods=1).mean()
        axes[3].plot(x_axis, roll_b, color="grey", lw=1.0, ls="--", label="Expanding Gaussian")
    axes[3].set_ylabel(f"NLPD ({rolling}-step mean)")
    axes[3].legend(loc="upper left", fontsize=8)

    if true_changepoints is not None:
        for ax in axes:
            for cp in true_changepoints:
                ax.axvline(cp, color="limegreen", lw=1.0, ls=":")

    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
