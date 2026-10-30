# -*- coding: utf-8 -*-
"""
Supporting-information figures of Article 1.

  figS1_espectro     lamp and solar spectra, silicon response, absorption profile in water
  figS2_c1_ajuste    fit of the cover module to campaign C1 (thermistors, QDY30A, micrometer)
  figS3_correlacao   posterior correlation matrix of the C2 parameters
  figS4_recesso      suppression against recess length (phi = 0.74)

Usage: python figuras_si_artigo1.py [names...]
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "_comum"))
from tanque import estilo as st      # noqa: E402
import matplotlib.pyplot as plt      # noqa: E402

CAL = HERE.parent / "saidas" / "calibracao"
SIM = HERE.parent / "saidas" / "simulacoes"
TAB = HERE.parent / "saidas" / "tabelas"
FIG = HERE.parent / "saidas" / "figuras"
st.setup()
C = st.SERIES


def figS1_espectro():
    import espectro as es
    lam = es.LAM
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 65 * st.MM))
    ax = axs[0]
    for T, col in ((2300, C[1]), (2700, C[3])):
        S = es.lamp_spectrum(T)
        ax.plot(lam, S / S.max(), "-", color=col, lw=1.1, label=f"lamp, {T} K")
    sun = es.solar_reference(lam)
    ax.plot(lam, sun / sun.max(), "-", color=C[0], lw=1.1, label="sunlight (approximate clear sky)")
    ax.plot(lam, es.silicon_response(lam), "--", color=st.INK2, lw=1.0, label="silicon sensor response")
    ax.set_xlim(0.3, 3.0); ax.set_ylim(0, 1.05)
    ax.set_xlabel("wavelength (µm)"); ax.set_ylabel("relative spectral irradiance / response")
    ax.legend(fontsize=6.5)
    ax = axs[1]
    for T, col in ((2300, C[1]), (2700, C[3])):
        z, Tz = es.absorption_profile(T)
        ax.plot(Tz, z * 1000, "-", color=col, lw=1.1, label=f"lamp, {T} K")
    S = es.solar_reference(lam); S = S / np.trapezoid(S, lam)
    a = es.water_absorption(lam)
    z = np.concatenate([[0], np.logspace(-5, 0, 300)])
    Tz = np.array([np.trapezoid(S * np.exp(-a * zz), lam) for zz in z]) * 0.94
    ax.plot(Tz, z * 1000, "-", color=C[0], lw=1.1, label="sunlight")
    ax.axhline(220, color=st.AXIS, lw=0.6)
    ax.text(0.98, 200, "pan floor", fontsize=6.5, color=st.MUTED, ha="right", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(250, 0.01)
    ax.set_yticks([0.01, 0.1, 1, 10, 100]); ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("fraction of incident energy below depth"); ax.set_ylabel("depth (mm)")
    ax.legend(fontsize=6.5, loc="upper left")
    st.panel_label(axs[0], "a"); st.panel_label(axs[1], "b")
    fig.tight_layout()
    st.save(fig, FIG / "figS1_espectro")


def figS2_c1_ajuste():
    import calibracao_c1 as c1
    m = json.load(open(CAL / "map_c1.json"))
    prob = c1.C1Problem(m["base"], rho=m["rho_used"], weights=m["weights_used"], processes=1)
    pred = prob.predict(prob.params_from_u(np.array(m["u"])[None]))
    _, parts = prob.loglik(pred, return_parts=True)
    o = prob.obs
    fig, axs = plt.subplots(3, 1, figsize=(st.WIDTH["full"], 120 * st.MM), sharex=True)
    ax = axs[0]
    for k, col in zip(c1.PROBES, (C[0], C[3], C[2])):
        ax.plot(o.index, o[k], ".", ms=1.0, color=col, alpha=0.5)
        ax.plot(o.index, pred[k][:, 0], "-", lw=0.7, color=col, label=k.replace("Tw_p", "thermistor "))
    ax.set_ylabel("water temperature (°C)")
    ax.legend(fontsize=6.5, ncol=3, loc="upper left")
    ax = axs[1]
    ax.plot(o.index, o.level_a_qc - c1.qdy_fitted(prob, parts, pred), ".", ms=1.0, color=C[1],
            label="QDY30A residual")
    mres = parts["micrometer"]["res"][:, 0]
    ax.plot(prob.mic.time, mres, "o", ms=3, color=C[6], label="micrometer residual")
    ax.axhline(0, color=st.AXIS, lw=0.6)
    ax.set_ylabel("level residual (mm)")
    ax.legend(fontsize=6.5, loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
    ax = axs[2]
    ax.plot(o.index, pred["E"][:, 0] * 3600, "-", color=C[0], lw=0.7)
    ax.set_ylabel("E (mm h$^{-1}$)")
    for a in axs:
        for s in prob.segs:
            if s.phi > 0:
                a.axvspan(s.start, s.end, color=C[2], alpha=0.07, lw=0)
    axs[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
    for i, a in enumerate(axs):
        st.panel_label(a, "abc"[i])
    fig.tight_layout()
    st.save(fig, FIG / "figS2_c1_ajuste")


def figS3_correlacao():
    from figuras_artigo1 import SYM
    c = pd.read_csv(TAB / "posterior_corr_c2.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(st.WIDTH["onehalf"], 110 * st.MM))
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list("div", [st.DIVERGING[2], st.DIVERGING[1], st.DIVERGING[0]])
    im = ax.imshow(c.values, cmap=div, vmin=-1, vmax=1)
    labs = [SYM.get(k, k) for k in c.index]
    ax.set_xticks(range(len(c))); ax.set_xticklabels(labs, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(labs, fontsize=6.5)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            if i != j and abs(c.values[i, j]) >= 0.5:
                v = c.values[i, j]
                ax.text(j, i, (f"{v:.2f}" if abs(v) >= 0.95 else f"{v:.1f}").replace("-", "−"), ha="center",
                        va="center", fontsize=5,
                        color="white" if abs(c.values[i, j]) > 0.75 else st.INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="posterior correlation (transformed scale)")
    fig.tight_layout()
    st.save(fig, FIG / "figS3_correlacao")


def figS4_recesso():
    d = pd.read_csv(SIM / "recesso.csv")
    fig, ax = plt.subplots(figsize=(st.WIDTH["single"], 65 * st.MM))
    for (I, U), col, ls in (((820.0, 0.0), C[3], "-"), ((820.0, 3.0), C[3], "--"),
                            ((300.0, 0.0), C[0], "-"), ((300.0, 3.0), C[0], "--")):
        s = d[(d.I == I) & (d.U == U)]
        ax.plot(s.R_rec_mm, s.S, ls, marker="o", ms=3, color=col, lw=1.1,
                label=f"index {I:.0f} W m$^{{-2}}$, fan {U:.0f} m s$^{{-1}}$")
    ax.axhline(0.74, color=st.AXIS, lw=0.6); ax.text(29, 0.745, "$\\phi$", fontsize=7, color=st.MUTED)
    ax.set_xlabel("recess length $R_{rec}$ (mm)"); ax.set_ylabel("suppression S")
    ax.legend(fontsize=6)
    fig.tight_layout()
    st.save(fig, FIG / "figS4_recesso")


ALL = ["figS1_espectro", "figS2_c1_ajuste", "figS3_correlacao", "figS4_recesso"]

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        print(name, flush=True)
        globals()[name]()
