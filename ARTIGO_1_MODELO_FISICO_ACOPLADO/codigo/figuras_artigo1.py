# -*- coding: utf-8 -*-
"""
Figures of Article 1 (vector PDF + 600 dpi PNG in saidas/figuras).

  fig02_c2_ajuste        C2 forcing, thermistors and levels with the model (two segments)
  fig03_perfis           vertical temperature profiles, observed vs model
  fig04_posteriores      posterior vs prior of the physical parameters
  fig05_energia          energy partition of the uncovered pan by run type
  fig06_c1_janelas       C1: modelled suppression by window and covered-water temperatures
  fig07_shapley          Shapley attribution of suppression to the mechanisms
  fig08_mapas            suppression and efficiency maps (phi x U) for three irradiances
  fig09_sobol            Sobol indices of suppression and saved depth

Usage: python figuras_artigo1.py [names...]
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
import calibracao as cb              # noqa: E402
import dados_modelo as dm            # noqa: E402
import modelo_fisico as mf           # noqa: E402

CAL = HERE.parent / "saidas" / "calibracao"
SIM = HERE.parent / "saidas" / "simulacoes"
FIG = HERE.parent / "saidas" / "figuras"
st.setup()
C = st.SERIES
PROBE_LABEL = {"Tw_cs": "floor (CS456 body)", "Tw_04": "4 cm", "Tw_12": "12 cm", "Tw_20": "20 cm"}


def _c2_mode():
    m = json.load(open(CAL / "map_c2.json"))
    return m


def fig02_c2_ajuste():
    m = _c2_mode()
    prob = cb.C2Problem(rho=m["rho_used"], weights=m["weights_used"], processes=1)
    pred = prob.predict(prob.params_from_u(np.array(m["u"])[None]))
    _, parts = prob.loglik(pred, return_parts=True)
    o = prob.obs
    sel = ["C2-S5", "C2-S7"]
    fig, axs = plt.subplots(4, 2, figsize=(st.WIDTH["full"], 150 * st.MM), sharex="col",
                            gridspec_kw=dict(height_ratios=[0.8, 1.4, 1.0, 1.0]))
    for j, sname in enumerate(sel):
        seg = next(s for s in prob.segs if s.name == sname)
        f = seg.forcing
        mk = (prob.seg_of == sname)
        t = o.index[mk]
        ax = axs[0, j]
        ax.plot(f.index, f.I, color=C[3], lw=0.8, label="irradiance index (W m$^{-2}$)")
        ax.set_ylabel("index (W m$^{-2}$)")
        on = (f.I > 50).to_numpy()
        grp = np.cumsum(on != np.roll(on, 1))
        for gid in np.unique(grp[on]):
            mm = (grp == gid) & on
            ax.text(f.index[mm][len(f.index[mm]) // 2], f.I[mm].max() * 1.04, f"{f.U[mm].mean():.1f}",
                    ha="center", va="bottom", fontsize=6, color=st.INK2)
        ax.set_ylim(0, f.I.max() * 1.3)
        ax = axs[1, j]
        for k, col in zip(cb.T_STREAMS, C):
            v = prob.valid[k][mk]
            y = o[k].to_numpy()[mk]
            ax.plot(t[v], y[v], ".", ms=1.2, color=col, alpha=0.6)
            ax.plot(t, pred[k][mk, 0], "-", lw=0.9, color=col, label=PROBE_LABEL[k])
        ax.set_ylabel("water temperature (°C)")
        ax = axs[2, j]
        # sensor records corrected with the fitted observation model
        nseg = len(prob.names); si = prob.names.index(sname)
        sc = parts["level_a_qc"]["dH_scale"][0]
        ba = parts["level_a_qc"]["beta"][0]
        bb = parts["level_b_qc"]["beta"][0]
        g = ba[-1] / sc
        Tc = o.Tcs_c.to_numpy()[mk]; td = o.t_seg_d.to_numpy()[mk]
        ya = (o.level_a_qc.to_numpy()[mk] - ba[si] - ba[nseg] * Tc) / g
        yb = (o.level_b_qc.to_numpy()[mk] - bb[si] - bb[nseg] * Tc - bb[nseg + 1 + si] * td) / cb.K_CS
        dH = pred["dH_mm"][mk, 0]
        ax.plot(t, ya, ".", ms=1.2, color=C[1], alpha=0.6, label=f"ODY ÷ gain {g:.2f}, thermal-corrected")
        ax.plot(t, yb, ".", ms=1.2, color=C[6], alpha=0.6, label="CS456, thermal- and drift-corrected")
        ax.plot(t, dH, "-", color=st.INK, lw=1.0, label="model (mass-equivalent depth)")
        ax.set_ylabel("depth change (mm)")
        ax = axs[3, j]
        ax.plot(t, pred["E"][mk, 0] * 3600, "-", color=C[0], lw=1.0)
        ax.set_ylabel("E (mm h$^{-1}$)")
        ax.set_ylim(bottom=0)
        for a in axs[:, j]:
            a.tick_params(axis="x", labelrotation=0)
        axs[3, j].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
    axs[1, 0].legend(ncol=2, fontsize=6.5, loc="upper right")
    axs[2, 0].legend(ncol=1, fontsize=6.5, loc="upper right")
    for i, lab in enumerate("abcdefgh"):
        st.panel_label(axs.ravel()[i], lab)
    fig.tight_layout()
    st.save(fig, FIG / "fig02_c2_ajuste")


def fig03_perfis():
    m = _c2_mode()
    pr_ = cb.C2Problem(rho=m["rho_used"], weights=m["weights_used"], processes=1)
    P = pr_.params_from_u(np.array(m["u"])[None])
    segs = {s.name: s for s in dm.c2_segments()}
    s_ = mf.sigma_grid(cb.N_LAYERS); zc = np.cumsum(s_) - s_ / 2
    cases = [("C2-S5", "2026-10-01", 0.0), ("C2-S3", "2026-09-25", 3.0)]
    hours = ["10:00", "12:00", "14:00", "16:00", "18:00"]
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 70 * st.MM), sharey=True)
    off = dm.probe_offsets_c2()
    for ax, (sn, day, Unom) in zip(axs, cases):
        seg = segs[sn]
        F = dm.forcing_arrays(seg)
        out = mf.simulate(F, {k: v.copy() for k, v in P.items()}, dm.initial_profile(seg, zc), seg.H0, N=cb.N_LAYERS)
        idx = seg.forcing.index
        for hh, col in zip(hours, st.SEQ[4::2]):
            t = pd.Timestamp(f"{day} {hh}")
            k = idx.get_indexer([t])[0]
            H = out.H[k, 0]
            ax.plot(out.T[k, 0, :], (H * (1 - zc)) * 100, "-", color=col, lw=1.2, label=hh)
            ob = seg.obs.loc[t]
            for pk, h in dm.PROBES_C2.items():
                if pk == "Tw_20" and ob["depth_obs"] - 0.20 < dm.SUBMERGED_MIN:
                    continue
                ax.plot(ob[pk] - off[pk], h * 100, "o", ms=3.5, color=col, mec="white", mew=0.5)
        U = seg.forcing.U[(seg.forcing.index.date == pd.Timestamp(day).date()) & (seg.forcing.I > 50)].mean()
        ax.set_title(f"{pd.Timestamp(day):%d %b}: full array, fan {U:.1f} m s$^{{-1}}$", fontsize=8)
        ax.set_xlabel("temperature (°C)")
    axs[0].set_ylabel("height above floor (cm)")
    axs[1].legend(title="time", fontsize=6.5, title_fontsize=6.5, loc="lower right")
    for i, a in enumerate(axs):
        st.panel_label(a, "ab"[i])
    fig.tight_layout()
    st.save(fig, FIG / "fig03_perfis")


SYM = {"A_s": "$A_s$", "A_m": "$A_m$", "l_m": r"$\ell_m$", "A_d": "$A_d$", "l_d": r"$\ell_d$", "a_b": "$a_b$",
       "U_w": "$U_w$", "G_w": "$G_w$", "U_b": "$U_b$", "c_n": "$c_n$", "c_f": "$c_f$", "n_f": "$n_f$",
       "C_0": "$C_0$", "K_bg": "$K_{bg}$", "N2_0": "$N_0^2$", "k_w": "$k_w$", "d_w": "$d_w$",
       "m_1": "$m_1$", "m_2": "$m_2$", "a_p": "$a_p$", "tau_p": r"$\tau_p$", "G_pw": "$G_{pw}$", "C_p": "$C_p$",
       "r_open": "$r$", "R_rec_V1": "$R_{rec}$ (V1)", "R_rec_V2": "$R_{rec}$ (V2)",
       "C_u_V1": "$C_u$ (V1)", "C_u_V2": "$C_u$ (V2)", "k_g": "$k_g$"}
COVER = ["a_p", "G_pw", "C_p", "C_u_V1", "C_u_V2", "k_g", "r_open", "R_rec_V1", "R_rec_V2"]
# units of the parameters on the scale on which the priors are defined
UNITS = {"l_m": "m", "l_d": "m", "U_w": "W m$^{-2}$ K$^{-1}$", "U_b": "W m$^{-2}$ K$^{-1}$",
         "c_f": "mm s$^{-1}$", "C_0": "mm s$^{-1}$", "K_bg": "m$^2$ s$^{-1}$", "N2_0": "s$^{-2}$",
         "k_w": "m", "d_w": "m", "G_pw": "W m$^{-2}$ K$^{-1}$", "C_p": "J m$^{-2}$ K$^{-1}$",
         "C_u_V1": "m s$^{-1}$", "C_u_V2": "m s$^{-1}$", "r_open": "m", "R_rec_V1": "m", "R_rec_V2": "m"}


def fig04_posteriores():
    from priors import Prior, PRIORS_C2
    import calibracao_c1 as c1
    sets = []
    for tag, spec, names in (("c2", PRIORS_C2, None), ("c1", c1.PRIORS_C1, COVER)):
        ch = cb.load_chain(tag)
        X = ch["chain"][ch["chain"].shape[0] // 2:].reshape(-1, ch["chain"].shape[2])
        pr_ = Prior(spec)
        for i, nm in enumerate(pr_.names):
            if names is None or nm in names:
                sets.append((nm, pr_.tr[i], X[:, i], pr_.mu[i], pr_.sd[i], tag))
    n = len(sets)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(st.WIDTH["full"], 27 * nrow * st.MM))
    rng = np.random.default_rng(0)
    for i, (nm, tr, post, mu, sd, tag) in enumerate(sets):
        ax = axs.ravel()[i]
        ax.grid(False)
        prs = mu + sd * rng.standard_normal(20000)
        lo = min(np.percentile(post, 0.2), np.percentile(prs, 0.5))
        hi = max(np.percentile(post, 99.8), np.percentile(prs, 99.5))
        bins = np.linspace(lo, hi, 40)
        ax.hist(prs, bins=bins, density=True, color=st.GRID, edgecolor="none", label="prior")
        ax.hist(post, bins=bins, density=True, color=C[0] if tag == "c2" else C[2], alpha=0.9,
                edgecolor="none", label="posterior")
        sym, u = SYM.get(nm, nm), UNITS.get(nm)
        if tr == "log":
            ttl = f"ln({sym} / {u})" if u else f"ln({sym})"
        elif tr == "logit":
            ttl = f"logit({sym})"
        else:
            ttl = f"{sym} ({u})" if u else sym
        ax.set_title(ttl, fontsize=6.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axs.ravel()[j].axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=st.GRID, label="prior"), Patch(color=C[0], label="posterior, uncovered pan (C2)"),
                        Patch(color=C[2], label="posterior, plates (C1)")], ncol=3, fontsize=7,
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    st.save(fig, FIG / "fig04_posteriores")


def fig05_energia():
    m = _c2_mode()
    pr_ = cb.C2Problem(rho=m["rho_used"], weights=m["weights_used"], processes=1)
    P = pr_.params_from_u(np.array(m["u"])[None])
    s_ = mf.sigma_grid(cb.N_LAYERS); zc = np.cumsum(s_) - s_ / 2
    rows = []
    for seg in dm.c2_segments():
        F = dm.forcing_arrays(seg)
        out = mf.simulate(F, {k: v.copy() for k, v in P.items()}, dm.initial_profile(seg, zc), seg.H0,
                          N=cb.N_LAYERS, store_fluxes=True)
        f = seg.forcing
        on = (f.I > 50).to_numpy()
        grp = (on != np.roll(on, 1)).cumsum()
        for g in np.unique(grp[on]):
            mk = (grp == g) & on
            if mk.sum() < 24:
                continue
            row = dict(segment=seg.name, start=f.index[mk][0], I=f.I[mk].mean(), U=f.U[mk].mean(),
                       cfg=int(np.bincount(f.cfg[mk]).argmax()))
            for k, v in out.fluxes.items():
                row[k] = float(v[mk, 0].mean())
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["cfg", "U"])
    df.to_csv(CAL.parent / "tabelas" / "balanco_energia_corridas.csv", index=False) if (CAL.parent / "tabelas").exists() else None
    (CAL.parent / "tabelas").mkdir(exist_ok=True)
    df.to_csv(CAL.parent / "tabelas" / "balanco_energia_corridas.csv", index=False)
    # sinks of the absorbed radiation stacked upwards from zero, gains (the wall under the
    # lower rail) downwards; the floor exchange (< 0.2 % of the absorbed radiation) joins the wall
    df["wall_floor"] = df["wall"] + df["floor"]
    terms = [("LE", "latent heat", C[0]), ("H_sens", "sensible heat", C[6]), ("LW_net", "long wave", C[4]),
             ("wall_floor", "wall and floor", C[5]), ("storage", "storage", C[3])]
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 72 * st.MM),
                            gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]
    x = np.arange(len(df))
    absorbed = df.Rabs_water.to_numpy()
    pos = np.zeros(len(df)); neg = np.zeros(len(df))
    for key, lab, col in terms:
        val = -df[key].to_numpy() if key != "storage" else df[key].to_numpy()
        frac = val / absorbed
        up = np.where(frac > 0, frac, 0.0); dn = np.where(frac < 0, frac, 0.0)
        ax.bar(x, up, bottom=pos, color=col, width=0.8, edgecolor="white", linewidth=0.5, label=lab)
        ax.bar(x, dn, bottom=neg, color=col, width=0.8, edgecolor="white", linewidth=0.5, hatch="////")
        pos += up; neg += dn
    cfgname = {0: "full", 1: "upper", 2: "lower"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cfgname[c]}\n{u:.1f}\n{t:%d/%m}" for c, u, t in zip(df.cfg, df.U, df.start)],
                       fontsize=6.5)
    ax.set_ylabel("fraction of absorbed radiation")
    ax.set_xlabel("lamp rail / fan speed (m s$^{-1}$) / date")
    ax.axhline(1, color=st.AXIS, lw=0.6); ax.axhline(0, color=st.AXIS, lw=0.6)
    ax.legend(ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax = axs[1]
    for c, mk, fill in zip(sorted(df.cfg.unique()), ["o", "s", "^"], [True, False, True]):
        s = df[df.cfg == c]
        ax.plot(s.U, (s.H_sens / s.LE), mk, color=C[c], ms=5, mfc=C[c] if fill else "white", mew=1.0,
                label=cfgname[c])
    ax.set_xlabel("fan speed (m s$^{-1}$)")
    ax.set_ylabel("Bowen ratio $Q_H/\\lambda E$")
    ax.legend(fontsize=6.5, title="lamps", title_fontsize=6.5)
    st.panel_label(axs[0], "a"); st.panel_label(axs[1], "b")
    fig.tight_layout()
    st.save(fig, FIG / "fig05_energia")
    return df


def fig06_c1_janelas():
    import diagnostico_c1 as d1
    m = json.load(open(CAL / "map_c1.json"))
    ws, series = d1.window_runs(m)
    ws.to_csv(CAL.parent / "tabelas" / "c1_supressao_janelas_modelo.csv", index=False)
    cov = ws[ws.cover == 1].copy()
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 70 * st.MM),
                            gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]
    ax.grid(axis="x", visible=False)
    x = np.arange(len(cov))
    phi = cov.n_plates * 0.031437 / 1.1029
    ax.plot(x, phi, "_", color=st.INK, ms=13, mew=1.4, label="covered fraction $\\phi$")
    for ver, col, mk in (("V1", C[0], "o"), ("V2", C[2], "s")):
        sel = (cov.version == ver).to_numpy()
        ax.plot(x[sel], cov.S_model[sel], mk, color=col, ms=6, mec="white", mew=0.6, label=f"S, {ver} plates")
    ax.set_xticks(x); ax.set_xticklabels([f"{w}\n{r}" for w, r in zip(cov.win, cov.regime)], fontsize=6)
    ax.set_ylabel("suppression S (model, counterfactual)")
    lo = min(cov.S_model.min(), phi.min())
    ax.set_ylim(np.floor((lo - 0.05) * 20) / 20, 0.85)
    ax.legend(fontsize=6.5, loc="lower left", ncol=3)
    ax = axs[1]
    for w, col in (("W07", C[0]), ("W12", C[2])):
        s = series[w]
        hrs = (s["t"] - s["t"][0]).total_seconds() / 3600
        ax.plot(hrs, s["Ts"], "-", color=col, lw=1.2)
        ax.plot(hrs, s["Ts0"], "--", color=col, lw=1.0)
        ax.plot(hrs, s["Tp"], ":", color=col, lw=1.2)
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color=C[0], lw=1.2, label="W07 (fan on)"), Line2D([], [], color=C[2], lw=1.2, label="W12 (calm)")]
    h2 = [Line2D([], [], color=st.INK2, ls="-", lw=1.2, label="water, covered"),
          Line2D([], [], color=st.INK2, ls="--", lw=1.0, label="water, uncovered"),
          Line2D([], [], color=st.INK2, ls=":", lw=1.2, label="plate")]
    leg = ax.legend(handles=h1, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax.add_artist(leg)
    ax.legend(handles=h2, fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.60), frameon=True,
              facecolor="white", edgecolor="none", framealpha=1.0)
    ax.set_xlabel("hours since window start")
    ax.set_ylabel("temperature (°C)")
    st.panel_label(axs[0], "a"); st.panel_label(axs[1], "b")
    fig.tight_layout()
    st.save(fig, FIG / "fig06_c1_janelas")
    return ws


def fig07_shapley():
    d = pd.read_csv(SIM / "shapley_grade.csv")
    mechs = [("sw_shade", "shading", C[3]), ("sw_vapour", "aerodynamic blocking", C[0]),
             ("sw_insul", "insulation", C[4]), ("sw_cond", "plate-water conduction", C[6])]
    fig, axs = plt.subplots(1, 3, figsize=(st.WIDTH["full"], 65 * st.MM), sharey=True)
    for ax, I in zip(axs, (300.0, 520.0, 820.0)):
        s = d[(d.I == I) & (d.phi == 0.74)].sort_values("U")
        x = np.arange(len(s))
        pos = np.zeros(len(s)); neg = np.zeros(len(s))
        for key, lab, col in mechs:
            v = s[key].to_numpy()
            b = np.where(v >= 0, pos, neg)
            ax.bar(x, v, bottom=b, color=col, width=0.7, edgecolor="white", linewidth=0.5, label=lab)
            pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
        ax.plot(x, s.S_total, "o", color=st.INK, ms=4, label="total suppression")
        ax.set_xticks(x); ax.set_xticklabels([f"{u:.0f}" for u in s.U])
        ax.set_xlabel("fan speed (m s$^{-1}$)")
        ax.set_title(f"index {I:.0f} W m$^{{-2}}$, $\\phi$ = 0.74", fontsize=8)
        ax.axhline(0, color=st.AXIS, lw=0.6)
    axs[0].set_ylabel("contribution to suppression")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    for i, a in enumerate(axs):
        st.panel_label(a, "abc"[i])
    fig.tight_layout()
    st.save(fig, FIG / "fig07_shapley")


def fig08_mapas():
    d = pd.read_csv(SIM / "mapa_supressao.csv")
    fig, axs = plt.subplots(2, 3, figsize=(st.WIDTH["full"], 115 * st.MM), sharex=True, sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", st.SEQ)
    div = LinearSegmentedColormap.from_list("div", [st.DIVERGING[2], st.DIVERGING[1], st.DIVERGING[0]])
    for j, I in enumerate((300.0, 520.0, 820.0)):
        s = d[d.I == I]
        P = s.pivot(index="U", columns="phi", values="S")
        R = s.pivot(index="U", columns="phi", values="S") / np.array(P.columns)[None, :]
        ax = axs[0, j]
        axs[0, j].grid(False); axs[1, j].grid(False)
        cs = ax.contourf(P.columns, P.index, P.values, levels=np.linspace(0, 1, 21), cmap=cmap)
        cl = ax.contour(P.columns, P.index, P.values, levels=[0.4, 0.6, 0.8], colors="white", linewidths=0.6)
        [tx.set_rotation(0) for tx in ax.clabel(cl, fmt="%.1f", fontsize=6.5, inline_spacing=2)]
        ax.set_title(f"index {I:.0f} W m$^{{-2}}$", fontsize=8)
        ax = axs[1, j]
        cs2 = ax.contourf(R.columns, R.index, R.values, levels=np.linspace(0.8, 1.2, 17), cmap=div, extend="both")
        cl2 = ax.contour(R.columns, R.index, R.values, levels=[0.9, 1.0, 1.1], colors=st.INK, linewidths=0.6)
        [tx.set_rotation(0) for tx in ax.clabel(cl2, fmt="%.1f", fontsize=6.5, inline_spacing=2)]
        axs[1, j].set_xlabel("covered fraction $\\phi$")
    axs[0, 0].set_ylabel("fan speed (m s$^{-1}$)"); axs[1, 0].set_ylabel("fan speed (m s$^{-1}$)")
    fig.colorbar(cs, ax=axs[0, :], label="suppression S", fraction=0.025, pad=0.01)
    fig.colorbar(cs2, ax=axs[1, :], label="efficiency S/$\\phi$", fraction=0.025, pad=0.01)
    for i, a in enumerate(axs.ravel()):
        st.panel_label(a, "abcdef"[i])
    st.save(fig, FIG / "fig08_mapas")


def fig09_sobol():
    d = pd.read_csv(SIM / "sobol.csv")
    lab = {"phi": "covered fraction $\\phi$", "a_p": "plate absorptance", "tau_p": "plate transmittance",
           "G_pw": "plate–water conductance", "k_g": "gap transfer enhancement",
           "C_u": "under-plate ventilation", "r_open": "opening size", "R_rec": "recess length", "U": "fan speed", "I": "irradiance index"}
    fig, axs = plt.subplots(1, 2, figsize=(st.WIDTH["full"], 65 * st.MM), sharey=True)
    for ax, (out, title) in zip(axs, (("S", "suppression S"), ("saved_mm_d", "saved depth (mm d$^{-1}$)"))):
        s = d[d.output == out].reset_index(drop=True)
        y = np.arange(len(s))
        ax.barh(y + 0.18, s.ST, height=0.34, color=C[0], label="total")
        ax.errorbar(s.ST, y + 0.18, xerr=[s.ST - s.ST_lo, s.ST_hi - s.ST], fmt="none", ecolor=st.INK2, lw=0.6)
        s1 = s.S1.clip(lower=0)
        ax.barh(y - 0.18, s1, height=0.34, color=C[1], label="first order")
        ax.errorbar(s1, y - 0.18, xerr=[(s1 - s.S1_lo).clip(lower=0), (s.S1_hi - s1).clip(lower=0)],
                    fmt="none", ecolor=st.INK2, lw=0.6)
        ax.set_yticks(y); ax.set_yticklabels([lab[v] for v in s.input])
        ax.set_title(title, fontsize=8); ax.set_xlim(left=0)
        ax.set_xlabel("Sobol index")
    axs[1].legend(fontsize=6.5, loc="lower right")
    st.panel_label(axs[0], "a"); st.panel_label(axs[1], "b")
    fig.tight_layout()
    st.save(fig, FIG / "fig09_sobol")


def fig10_projeto():
    """Design map (opening size x recess length) and spectral transfer of S(phi)."""
    d = pd.read_csv(SIM / "projeto_aberturas.csv")
    m1 = json.load(open(CAL / "map_c1.json"))["params"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", st.SEQ)
    fig = plt.figure(figsize=(st.WIDTH["full"], 64 * st.MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.62, 1.15], wspace=0.2)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    cax = fig.add_subplot(gs[0, 2])
    axs[1].sharey(axs[0])
    for ax, U in zip(axs[:2], (0.0, 3.0)):
        ax.grid(False)
        s = d[d.U == U].pivot(index="R_rec_mm", columns="r_open_mm", values="S")
        y = np.array(s.index) + 0.25           # offset so that R_rec = 0 shows on a log axis
        cs = ax.contourf(s.columns, y, s.values, levels=np.linspace(0.5, 0.95, 19), cmap=cmap)
        cl = ax.contour(s.columns, y, s.values, levels=[0.6, 0.7, 0.8], colors="white", linewidths=0.6)
        [tx.set_rotation(0) for tx in ax.clabel(cl, fmt="%.1f", fontsize=6)]
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot(m1["r_open"] * 1000, m1["R_rec_V2"] * 1000 + 0.25, "o", color=st.INK, ms=5, mec="white", mew=0.8)
        ax.set_xlabel("opening size $r$ (mm)")
        ax.set_title(f"fan {U:.0f} m s$^{{-1}}$, $\\phi$ = 0.74", fontsize=8)
        yt = [0, 1, 2, 5, 10, 20]
        ax.set_yticks(np.array(yt) + 0.25); ax.set_yticklabels([str(v) for v in yt])
        ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
        ax.minorticks_off()
    axs[0].set_ylabel("recess length $R_{rec}$ (mm)")
    plt.setp(axs[1].get_yticklabels(), visible=False)
    fig.colorbar(cs, cax=cax, label="suppression S")
    ax = axs[2]
    es = pd.read_csv(SIM / "transferencia_espectral.csv")
    es = es[es.S_curve.notna() & (es.U == 3.0)]
    unc = SIM / "incerteza_posterior.csv"
    if unc.exists():
        u = pd.read_csv(unc)
        u = u[u.U == 3.0].groupby("phi").S.quantile([0.025, 0.975]).unstack()
        ax.fill_between(u.index, u[0.025], u[0.975], color=C[0], alpha=0.2, lw=0, label="lamps, 95 % posterior")
    for case, lab, col, ls in (("lamp", "lamps", C[0], "-"), ("solar", "sunlight", C[1], "--"),
                               ("solar_white", "sunlight, $a_p$ = 0.25", C[2], ":")):
        s = es[es.case == case].sort_values("phi")
        ax.plot(s.phi, s.S_curve, ls, color=col, lw=1.3, label=lab)
    ax.plot([0, 1], [0, 1], color=st.AXIS, lw=0.6)
    ax.text(0.80, 0.86, "S = $\\phi$", fontsize=6.5, color=st.MUTED, ha="left", va="bottom", rotation=40)
    ax.set_xlim(0.25, 0.95); ax.set_ylim(0.2, 1)
    ax.set_xlabel("covered fraction $\\phi$"); ax.set_ylabel("suppression S")
    ax.set_title("3 m s$^{-1}$, 820 W m$^{-2}$", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    for i, a in enumerate(axs):
        st.panel_label(a, "abc"[i])
    st.save(fig, FIG / "fig10_projeto_espectro")


ALL = ["fig02_c2_ajuste", "fig03_perfis", "fig04_posteriores", "fig05_energia", "fig06_c1_janelas",
       "fig07_shapley", "fig08_mapas", "fig09_sobol", "fig10_projeto"]

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        print(name, flush=True)
        globals()[name]()
