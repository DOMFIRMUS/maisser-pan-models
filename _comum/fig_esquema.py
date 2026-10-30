# -*- coding: utf-8 -*-
"""
Schematic of the pan, the lamps, the wind tunnel and the floating plates with
the energy and mass fluxes of the coupled model (review Fig. 1, Article 1 Fig. 1).

Usage: python fig_esquema.py pt|en out_path
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.patches as mp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tanque import estilo as st   # noqa: E402

L = {
    "pt": dict(lamps="lâmpadas halógenas IR (18 × 250 W)", fan="túnel de vento", U=r"$U$",
               plate=r"placas ($\phi$ = 0,74–0,77)", gap="fresta", water="água", floor="fundo",
               wall="parede", I=r"$I$ (onda curta)", LWl="onda longa\ndos bulbos", LE=r"$\lambda E$",
               H=r"$Q_H$", LW=r"$L_\uparrow$", Gp=r"$G_{pw}$", qw=r"$q_{par}$", qb=r"$q_{fun}$",
               K=r"$K(z,t)$", probes="termistores\n4, 12, 20 cm", level="transdutores\nde nível",
               air=r"$T_a,\ e_a$", beer=r"$I(z)$", absorb="absorção\nespectral"),
    "en": dict(lamps="halogen IR lamps (18 × 250 W)", fan="wind tunnel", U=r"$U$",
               plate=r"plates ($\phi$ = 0.74–0.77)", gap="gap", water="water", floor="floor",
               wall="wall", I=r"$I$ (short wave)", LWl="envelope\nlong wave", LE=r"$\lambda E$",
               H=r"$Q_H$", LW=r"$L_\uparrow$", Gp=r"$G_{pw}$", qw=r"$q_w$", qb=r"$q_b$",
               K=r"$K(z,t)$", probes="thermistors\n4, 12, 20 cm", level="level\ntransducers",
               air=r"$T_a,\ e_a$", beer=r"$I(z)$", absorb="spectral\nabsorption"),
}


def arrow(ax, x0, y0, x1, y1, color, lw=1.4, style="-|>", ms=7, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms, ls=ls,
                                shrinkA=0, shrinkB=0))


def main(lang="pt", out=None):
    st.setup()
    t = L[lang]
    fig, ax = st.plt.subplots(figsize=(st.WIDTH["full"], 95 * st.MM))
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(-1.2, 7.4); ax.axis("off")
    blue, orange, aqua, yellow, magenta, green, violet, red = st.SERIES
    # pan
    pan = dict(x0=1.0, x1=9.0, yb=0.0, yt=2.6, yw=2.2)
    ax.add_patch(mp.Rectangle((pan["x0"], pan["yb"]), pan["x1"] - pan["x0"], pan["yw"],
                              facecolor="#dceaf9", edgecolor="none"))
    # stratification cue: warmer top band
    ax.add_patch(mp.Rectangle((pan["x0"], pan["yw"] - 0.35), pan["x1"] - pan["x0"], 0.35,
                              facecolor="#f6d2c1", edgecolor="none", alpha=0.8))
    for x in (pan["x0"], pan["x1"]):
        ax.plot([x, x], [pan["yb"], pan["yt"]], color=st.INK2, lw=2.2, solid_capstyle="butt")
    ax.plot([pan["x0"], pan["x1"]], [pan["yb"], pan["yb"]], color=st.INK2, lw=2.2)
    # support frame
    for x in (1.4, 8.6):
        ax.plot([x, x], [-1.0, 0.0], color=st.AXIS, lw=1.6)
    ax.plot([0.9, 9.1], [-0.15, -0.15], color=st.AXIS, lw=1.0)
    # plates on the right part
    xs = np.arange(4.55, 8.3, 0.85)
    for x in xs:
        ax.add_patch(mp.FancyBboxPatch((x, pan["yw"] - 0.10), 0.72, 0.30,
                                       boxstyle="round,pad=0.0,rounding_size=0.04",
                                       facecolor="#f4f3ef", edgecolor=st.INK2, lw=0.8))
    ax.text(7.25, 3.55, t["plate"], ha="center", fontsize=7, color=st.INK2)
    ax.annotate(t["gap"], xy=(xs[2] + 0.785, pan["yw"]), xytext=(xs[2] + 0.4, pan["yw"] - 0.75),
                fontsize=6.5, color=st.INK2, arrowprops=dict(arrowstyle="-", color=st.MUTED, lw=0.6))
    # lamps
    for x in np.linspace(1.6, 8.4, 7):
        ax.add_patch(mp.Ellipse((x, 6.1), 0.34, 0.52, facecolor="#ffe6a6", edgecolor=yellow, lw=1.0))
        ax.plot([x, x], [6.37, 6.9], color=st.MUTED, lw=0.8)
    ax.plot([1.2, 8.8], [6.9, 6.9], color=st.AXIS, lw=1.2)
    ax.text(5.0, 7.1, t["lamps"], ha="center", fontsize=7.5, color=st.INK)
    # short wave and envelope long wave
    for x in (2.3, 3.4):
        arrow(ax, x, 5.85, x, pan["yw"] + 0.05, yellow, lw=1.6)
    arrow(ax, 5.75, 5.85, 5.75, pan["yw"] + 0.22, yellow, lw=1.6)
    ax.text(2.45, 4.6, t["I"], fontsize=7, color=st.INK2)
    arrow(ax, 4.6, 5.85, 4.25, pan["yw"] + 0.05, orange, lw=1.0, ls="--")
    ax.text(4.65, 4.95, t["LWl"], fontsize=6.5, color=st.INK2)
    # spectral absorption inset curve in the water
    zz = np.linspace(0, 1, 50)
    ax.plot(9.6 + 0.6 * (0.3 * np.exp(-zz / 0.03) + 0.4 * np.exp(-zz / 0.15) + 0.3 * np.exp(-zz / 1.2)),
            pan["yw"] - 2.1 * zz, color=yellow, lw=1.2)
    ax.plot([9.6, 9.6], [pan["yw"], pan["yb"] + 0.1], color=st.AXIS, lw=0.6)
    ax.text(9.65, pan["yw"] + 0.1, t["beer"], fontsize=7, color=st.INK2)
    ax.text(9.6, -0.55, t["absorb"], fontsize=6.5, color=st.INK2, ha="center")
    # fan / tunnel
    ax.add_patch(mp.Polygon([[-0.5, 2.9], [0.95, 2.75], [0.95, 3.35], [-0.5, 4.4]],
                            facecolor="#efeee9", edgecolor=st.AXIS, lw=0.8))
    ax.text(-0.45, 4.55, t["fan"], fontsize=7, color=st.INK2)
    arrow(ax, 0.6, 3.05, 3.2, 2.62, aqua, lw=1.6)
    ax.text(1.7, 3.05, t["U"], fontsize=8, color=st.INK, style="italic")
    # air-side fluxes from open water
    arrow(ax, 2.0, pan["yw"] + 0.05, 2.0, 3.85, blue, lw=1.6)
    ax.text(1.55, 3.95, t["LE"], fontsize=8, color=st.INK)
    arrow(ax, 2.8, pan["yw"] + 0.05, 2.8, 3.55, violet, lw=1.2)
    ax.text(2.88, 3.6, t["H"], fontsize=8, color=st.INK)
    arrow(ax, 3.7, pan["yw"] + 0.05, 3.7, 3.45, magenta, lw=1.2, ls="--")
    ax.text(3.78, 3.45, t["LW"], fontsize=8, color=st.INK)
    # gap evaporation and plate fluxes
    arrow(ax, xs[3] + 0.785, pan["yw"] + 0.02, xs[3] + 0.785, 3.35, blue, lw=1.0)
    arrow(ax, xs[2] + 0.36, pan["yw"] + 0.2, xs[2] + 0.36, 3.25, violet, lw=1.0)
    arrow(ax, xs[4] + 0.36, pan["yw"] - 0.12, xs[4] + 0.36, pan["yw"] - 0.62, red, lw=1.0)
    ax.text(xs[4] + 0.45, pan["yw"] - 0.62, t["Gp"], fontsize=7, color=st.INK)
    # wall and floor exchange
    arrow(ax, pan["x0"] + 0.05, 1.1, pan["x0"] - 0.55, 1.1, green, lw=1.2)
    ax.text(-0.5, 1.25, t["qw"], fontsize=7, color=st.INK)
    arrow(ax, 5.0, pan["yb"] + 0.05, 5.0, -0.7, green, lw=1.2)
    ax.text(5.1, -0.75, t["qb"], fontsize=7, color=st.INK)
    # mixing
    ax.annotate("", xy=(3.0, 0.5), xytext=(3.0, 1.6),
                arrowprops=dict(arrowstyle="<|-|>", color=st.MUTED, lw=1.0, mutation_scale=7))
    ax.text(3.1, 1.0, t["K"], fontsize=7, color=st.INK2)
    # probes and level sensors
    for h, lab in ((0.4, "4"), (1.2, "12"), (2.0, "20")):
        ax.plot(4.0, pan["yb"] + h * 0.95, "o", ms=4, color=st.INK2)
    ax.plot([4.0, 4.0], [0.0, 2.05], color=st.MUTED, lw=0.6)
    ax.text(4.12, 0.65, t["probes"], fontsize=6.5, color=st.INK2)
    for x in (7.2, 7.6):
        ax.add_patch(mp.Rectangle((x, 0.02), 0.25, 0.22, facecolor=st.INK2, edgecolor="none"))
    ax.text(7.2, 0.35, t["level"], fontsize=6.5, color=st.INK2)
    # air state
    ax.text(7.6, 4.3, t["air"], fontsize=7.5, color=st.INK2)
    ax.plot(7.5, 4.35, "s", ms=4, color=st.INK2)
    st.save(fig, out)


if __name__ == "__main__":
    lang = sys.argv[1] if len(sys.argv) > 1 else "pt"
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[1] /
                                                      "00_REVISAO_EQUACOES" / "figuras" / "fig_esquema_fluxos")
    main(lang, out)
