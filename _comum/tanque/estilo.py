# -*- coding: utf-8 -*-
"""
Figure style shared by the two articles (journal print, light surface).

Palette: the validated categorical order of the data-visualisation reference
(blue, orange, aqua, yellow, magenta, green, violet, red), used in fixed
order and never cycled; sequential blue ramp; recessive grid and axes; text
in ink colours, never in series colours. One y-axis per panel.

Widths follow the Elsevier/Springer grids: 90 mm (single column), 140 mm
(1.5 columns), 190 mm (full width).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
MM = 1 / 25.4
WIDTH = {"single": 90 * MM, "onehalf": 140 * MM, "full": 190 * MM}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            fontweight="bold", color=INK)


def save(fig, path, png=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(fig)
