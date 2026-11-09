"""Consistent chart styling for the notebooks and README figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import config

# Categorical slots 1-3 of a colour-blind-validated palette (fixed order, one per source).
SOURCE_COLORS = {"Bayut": "#2a78d6", "OLX": "#eb6834", "Cairo": "#1baf7a"}
PRIMARY = "#2a78d6"
MUTED = "#8a8984"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "text.color": TEXT,
    })


def _short(value: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= div:
            out = f"{value / div:.1f}".rstrip("0").rstrip(".")
            return out + suffix
    return f"{value:.0f}"


egp_formatter = FuncFormatter(lambda v, _: _short(v))


def save(fig, name: str) -> None:
    """Save a figure into reports/figures for the README."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
