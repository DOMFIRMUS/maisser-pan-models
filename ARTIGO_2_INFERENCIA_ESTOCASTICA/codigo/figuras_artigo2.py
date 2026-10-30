# -*- coding: utf-8 -*-
"""
Figures of Article 2 (vector PDF + 600 dpi PNG in saidas/figuras).

  fig01_sensores      the two C2 transducers and their disagreement; QDY vs micrometer in C1
  fig02_recuperacao   synthetic recovery test of the estimator
  fig03_evaporacao    smoothed evaporation with its uncertainty (C2), best driver
  fig04_drivers       cross-validated comparison of the evaporation equations
  fig05_supressao     direct suppression by window, hierarchical estimates
  fig06_delineamento  expected information gain of the next coverage experiments

Usage: python figuras_artigo2.py [names...]
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "_comum"))
from tanque import estilo as st, config as cfg   # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402
import matplotlib.dates as mdates                # noqa: E402

OUT = HERE.parent / "saidas"
FIG = OUT / "figuras"
st.setup()
C = st.SERIES
DRV_LABEL = {"constante": "constant", "dalton_ar": "Dalton (air VPD)", "dalton_superficie": "Dalton (surface Δe)",
             "penman_indice": "Penman-type (index)", "ryan_harleman": "Ryan–Harleman (free + forced)",
             "livre_forcada_potencia": "free + forced, power law", "churchill_usagi": "Churchill–Usagi",
             "churchill_usagi_fundo": "Churchill–Usagi + background", "penman1948_fixo": "Penman 1948 (fixed)",
             "carrier_fixo": "Carrier/ASHRAE (fixed)"}


def best_driver():
    """Driver with the highest leave-one-segment-out score (AIC if CV is missing)."""
    cv = OUT / "c2_validacao_cruzada_resumo.csv"
    if cv.exists():
        d = pd.read_csv(cv)
        return d.sort_values("loso_score", ascending=False).driver.iloc[0]
    fits = json.load(open(OUT / "c2_ajustes.json"))
    return min(fits, key=lambda k: fits[k]["aic"])


def fig01_sensores():
    import dados_ee as de
    a, _ = de.series_10min("C2")
    fig, axs = plt.subplots(1, 3, figsize=(st.WIDTH["full"], 62 * st.MM),
                            gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    ax = axs[0]
    d = a["level_a_qc"] - a["level_b_qc"]
    ax.plot(a.index, d - d.dropna().iloc[0], ".", ms=1, color=C[6])
    ax.set_ylabel("ODY − CS456 (mm)")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("C2: difference between transducers", fontsize=8)
    ax = axs[1]
    # within-segment changes of the difference vs change of sensor temperature (dark periods)
    seg = a["segment"]
    dd = d.groupby(seg).transform(lambda x: x - x.mean())
    tt = a["Tw_cs"].groupby(seg).transform(lambda x: x - x.mean())
    dark = a["lamp_on"] < 0.5
    ax.plot(tt[~dark], dd[~dark], ".", ms=1, color=C[3], alpha=0.5, label="lamps on")
    ax.plot(tt[dark], dd[dark], ".", ms=1, color=C[0], alpha=0.5, label="lamps off")
    ax.set_xlabel("sensor temperature anomaly (K)")
    ax.set_ylabel("difference anomaly (mm)")
    ax.legend(fontsize=6.5, markerscale=5)
    ax.set_title("thermal response", fontsize=8)
    ax = axs[2]
    import ajuste_c1 as a1
    c = a1.prepare_c1()
    r1 = json.load(open(OUT / "c1_supressao_direta.json"))
    ca, ga = r1["c_a"], r1["theta"]["g_a"]
    seen = set()
    for sid, s in c.groupby("seg_c1"):
        m = s["mic"].dropna()
        if len(m) < 2:
            continue
        # transducer change corrected for its thermal response (bulk temperature regressor)
        q = (s["level_a_qc"] - ca * (s["Tb"] - 30.0)).reindex(m.index)
        cov = s.phi.iloc[0] > 0
        lab = ("covered" if cov else "uncovered") if cov not in seen else None
        seen.add(cov)
        ax.plot(m - m.iloc[0], q - q.iloc[0], "o-", ms=3, lw=0.6, color=C[2] if cov else C[0], label=lab)
    lim = ax.get_xlim()
    xx = np.linspace(min(lim[0], -6), 0.5, 10)
    ax.plot(xx, xx, "--", color=st.MUTED, lw=0.8, label="1:1")
    ax.plot(xx, ga * xx, "-", color=st.INK2, lw=0.8, label=f"gain {ga:.2f}")
    ax.set_xlabel("micrometer change (mm)")
    ax.set_ylabel("QDY30A change, thermally corrected (mm)")
    ax.set_title("C1: transducer vs micrometer", fontsize=8)
    ax.legend(fontsize=6.5)
    for i, x in enumerate(axs):
        st.panel_label(x, "abc"[i])
    fig.tight_layout()
    st.save(fig, FIG / "fig01_sensores")


def fig02_recuperacao():
    df = pd.read_csv(OUT / "recuperacao_sintetica.csv")
    keys = [("g_a", "ODY gain g"), ("c_a", "$c_a$ (mm K$^{-1}$)"), ("c_b", "$c_b$ (mm K$^{-1}$)"),
            ("delta", "drift δ (mm h$^{-1}$)"), ("beta0", "β$_1$"), ("beta1", "β$_2$"),
            ("phi", "$\\varphi$ (AR)"), ("s_eta", "σ$_η$"), ("s_a", "σ$_a$"), ("s_b", "σ$_b$")]
    fig, ax = plt.subplots(figsize=(st.WIDTH["onehalf"], 60 * st.MM))
    for i, (k, lab) in enumerate(keys):
        tr = df[f"{k}_true"].iloc[0]
        rel = df[f"{k}_hat"] / tr
        ax.plot(np.full(len(rel), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(rel)), rel, "o",
                ms=2.5, color=C[0], alpha=0.7)
        ax.plot([i - 0.3, i + 0.3], [rel.mean()] * 2, "-", color=st.INK, lw=1.2)
        if f"{k}_cover" in df:
            ax.text(i, 1.43, f"{100 * df[f'{k}_cover'].mean():.0f}%", ha="center", fontsize=6.5, color=st.INK2)
    ax.axhline(1, color=st.AXIS, lw=0.8)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k[1] for k in keys], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("estimate / true value")
    ax.set_ylim(0.6, 1.52)
    ax.text(-0.3, 1.475, "coverage of the 95 % intervals of the linear coefficients:", fontsize=6.5,
            color=st.INK2, ha="left", va="bottom")
    fig.tight_layout()
    st.save(fig, FIG / "fig02_recuperacao")


def fig03_evaporacao():
    import dados_ee as de
    best = best_driver()
    E = pd.read_csv(OUT / f"c2_{best}_E.csv", index_col=0, parse_dates=True)
    a, _ = de.series_10min("C2")
    fig, axs = plt.subplots(2, 1, figsize=(st.WIDTH["full"], 95 * st.MM), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.4]))
    ax = axs[0]
    ax.plot(a.index, a["I"] * 1000, "-", color=C[3], lw=0.7)
    ax.set_ylabel("irradiance index (W m$^{-2}$)")
    ax = axs[1]
    ax.fill_between(E.index, E.E_mm_h - 1.96 * E.E_sd, E.E_mm_h + 1.96 * E.E_sd, color=C[0], alpha=0.25,
                    lw=0, label="95 % interval")
    ax.plot(E.index, E.E_mm_h, "-", color=C[0], lw=0.8, label="smoothed evaporation")
    ax.plot(E.index, E.E_reg.where(E.E_reg != 0), "-", color=C[1], lw=0.6, alpha=0.8, label=f"driver part ({DRV_LABEL[best]})")
    ax.set_ylabel("evaporation (mm h$^{-1}$)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    st.panel_label(axs[0], "a"); st.panel_label(axs[1], "b")
    fig.tight_layout()
    st.save(fig, FIG / "fig03_evaporacao")


def fig04_drivers():
    cv = pd.read_csv(OUT / "c2_validacao_cruzada_resumo.csv").sort_values("loso_score")
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 65 * st.MM), sharey=True)
    y = np.arange(len(cv))
    ax = axs[0]
    ax.barh(y, cv.diff_to_best, color=C[0], height=0.6)
    ax.errorbar(cv.diff_to_best, y, xerr=1.96 * cv.diff_se, fmt="none", ecolor=st.INK2, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([DRV_LABEL[d] for d in cv.driver])
    ax.set_xlabel("held-out log score − best (nats)")
    ax.set_title("leave-one-segment-out", fontsize=8)
    ax = axs[1]
    ax.barh(y, cv.aic - cv.aic.min(), color=C[1], height=0.6)
    ax.set_xlabel("ΔAIC")
    ax.set_title("in-sample", fontsize=8)
    for i, a in enumerate(axs):
        st.panel_label(a, "ab"[i])
    fig.tight_layout()
    st.save(fig, FIG / "fig04_drivers")


def fig05_supressao():
    w = pd.read_csv(OUT / "hierarquico_janelas.csv")
    cells = pd.read_csv(OUT / "hierarquico_celulas.csv")
    m1 = None
    p1 = HERE.parents[1] / "ARTIGO_1_MODELO_FISICO_ACOPLADO" / "saidas" / "tabelas" / "c1_supressao_janelas_modelo.csv"
    if p1.exists():
        m1 = pd.read_csv(p1).set_index("win")
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 85 * st.MM), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]
    y = np.arange(len(w))[::-1]
    ax.errorbar(w.S_bruto, y + 0.15, xerr=[w.S_bruto - w.S_bruto_lo, w.S_bruto_hi - w.S_bruto], fmt="o", ms=3.5,
                color=C[0], ecolor=C[0], lw=0.8, label="direct suppression (state-space)")
    ax.errorbar(w.S_hier, y - 0.15, xerr=[w.S_hier - w.S_hier_lo, w.S_hier_hi - w.S_hier], fmt="s", ms=3.5,
                color=C[2], ecolor=C[2], lw=0.8, label="hierarchical (shrinkage)")
    if m1 is not None:
        if "S_direct_model" not in m1:
            raise RuntimeError("companion window table lacks S_direct_model: run Article 1's collector first")
        ax.plot(m1.reindex(w.win)["S_direct_model"], y, "D", ms=3.5, color=C[1],
                label="mechanistic model (same surface temperature)")
    ax.plot(w.phi, y, "|", ms=10, mew=1.5, color=st.INK, label="covered fraction $\\phi$")
    ax.set_yticks(y); ax.set_yticklabels([f"{a} {b} {c}" for a, b, c in zip(w.win, w.version, w.regime)], fontsize=6.5)
    ax.set_xlabel("suppression")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax = axs[1]
    yy = np.arange(len(cells))[::-1]
    ax.errorbar(cells.S_media, yy + 0.12, xerr=[cells.S_media - cells.S_lo, cells.S_hi - cells.S_media], fmt="o",
                color=C[0], ms=4, lw=1.0, label="cell median")
    ax.errorbar(cells.S_pred, yy - 0.12, xerr=[cells.S_pred - cells.S_pred_lo, cells.S_pred_hi - cells.S_pred],
                fmt="o", mfc="white", color=C[6], ms=4, lw=0.8, label="new run (predictive)")
    ax.set_yticks(yy); ax.set_yticklabels(cells.celula.str.replace("vento forte", "strong fan").str.replace(
        "vento fraco", "weak fan").str.replace("ventilador desligado", "fan off"), fontsize=6.5)
    ax.set_xlabel("suppression")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    for i, a in enumerate(axs):
        st.panel_label(a, "ab"[i])
    fig.tight_layout()
    st.save(fig, FIG / "fig05_supressao")


def fig06_delineamento():
    e = pd.read_csv(OUT / "delineamento_eig_unico.csv")
    p = pd.read_csv(OUT / "delineamento_previsoes_modelos.csv")
    r = json.load(open(OUT / "delineamento_resumo.json"))
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 65 * st.MM))
    ax = axs[0]
    lab = {"M1": "constant efficiency", "M2": "open-area power law", "M3": "gaps + leak"}
    dashes = {"M1": (0, (6, 2)), "M2": (0, (3, 1.5)), "M3": (0, (1.2, 1.2))}
    for m, col in zip(("M1", "M2", "M3"), (C[0], C[1], C[2])):
        s = p[(p.model == m) & (p.U == 0.0)]
        ax.fill_between(s.phi, s.S_lo, s.S_hi, color=col, alpha=0.18, lw=0)
        ax.plot(s.phi, s.S_med, "-", color=col, lw=1.2, label=lab[m])
        s3 = p[(p.model == m) & (p.U == 3.2)]
        ax.fill_between(s3.phi, s3.S_lo, s3.S_hi, facecolor="none", edgecolor=col, hatch="....", lw=0, alpha=0.5)
        ax.plot(s3.phi, s3.S_med, linestyle=dashes[m], color=col, lw=1.0)
    # current estimates of the V2 plates (phi = 0.741), dodged for legibility
    ax.errorbar([0.725], [r["S_obs"][1]], yerr=1.96 * r["S_obs_sd"][1], fmt="o", mfc="white", color="k",
                ms=3.5, lw=0.8, label="current, fan off (95 %)")
    ax.errorbar([0.757], [r["S_obs"][0]], yerr=1.96 * r["S_obs_sd"][0], fmt="o", color="k", ms=3.5, lw=0.8,
                label="current, 3.2 m s$^{-1}$ (95 %)")
    ax.plot([0, 1], [0, 1], ":", color=st.MUTED, lw=0.8)
    ax.set_xlabel("covered fraction $\\phi$"); ax.set_ylabel("suppression S")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("solid: fan off; dashed and dotted bands: 3.2 m s$^{-1}$", fontsize=7)
    ax = axs[1]
    for U, col in zip(sorted(e.U.unique()), (C[0], C[1], C[2])):
        s = e[e.U == U]
        ax.plot(s.phi, s.eig, "o-", color=col, ms=3.5, lw=1.0, label=f"U = {U:.1f} m s$^{{-1}}$")
    ax.set_xlabel("covered fraction of the next run"); ax.set_ylabel("expected information gain (nats)")
    ax.legend(fontsize=6.5)
    for i, a in enumerate(axs):
        st.panel_label(a, "ab"[i])
    fig.tight_layout()
    st.save(fig, FIG / "fig06_delineamento")


ALL = ["fig01_sensores", "fig02_recuperacao", "fig03_evaporacao", "fig04_drivers", "fig05_supressao",
       "fig06_delineamento"]

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        print(name, flush=True)
        globals()[name]()
