# -*- coding: utf-8 -*-
"""
Campaign C1 diagnostics for a C1 parameter set: thermistors, QDY30A and
micrometer against the model, and the suppression of every covered window
computed against a counterfactual uncovered simulation with the same forcing
and initial state.

Usage: python diagnostico_c1.py [map_c1.json] [out.png]
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibracao_c1 as c1    # noqa: E402
import calibracao as cb       # noqa: E402
import modelo_fisico as mf    # noqa: E402
import dados_modelo as dm     # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "_comum"))
from tanque import config as cfg, estilo as st, propriedades as pr   # noqa: E402


def full_params(m, extra=None):
    P = {k: np.array([v], float) for k, v in m["base"].items()}
    P.update({k: np.array([v], float) for k, v in m["params"].items()})
    if extra:
        P.update({k: np.array([v], float) for k, v in extra.items()})
    return P


def window_runs(m, extra=None, P=None):
    """Per-window simulations (covered and counterfactual). Returns a table and
    the time series of each covered window."""
    segs = c1.micrometer_segments(dm.c1_segments())
    s_ = mf.sigma_grid(c1.N_LAYERS); zc = np.cumsum(s_) - s_ / 2
    P = full_params(m, extra) if P is None else P
    rows, series = [], {}
    for w in cfg.WINDOWS_C1:
        wid, a, b, cov, ver, npl, regime = w
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        best, ov = None, 0
        for s in segs:
            o = (min(b, s.end) - max(a, s.start)).total_seconds()
            if o > ov:
                best, ov = s, o
        if best is None or ov < 0.8 * (b - a).total_seconds():
            rows.append(dict(win=wid, note="window not inside one segment")); continue
        seg = best
        Pv = {k: v.copy() for k, v in P.items()}
        c1.version_params(Pv, seg.version)
        T0 = c1.initial_profiles(seg, Pv, zc)
        F = dm.forcing_arrays(seg)
        oc = mf.simulate(F, Pv, T0, seg.H0, N=c1.N_LAYERS, store_fluxes=True)
        F0 = dm.forcing_arrays(seg); F0.phi = np.zeros_like(F0.phi)
        oo = mf.simulate(F0, Pv, T0, seg.H0, N=c1.N_LAYERS, store_fluxes=True)
        idx = seg.forcing.index
        mk = (idx >= a) & (idx < b)
        hrs = mk.sum() * 300 / 3600
        Ec = oc.E[mk, :].sum(0) * 300 / 997 * 1000
        Eo = oo.E[mk, :].sum(0) * 300 / 997 * 1000
        # evaporation the uncovered pan would have with its skin at the temperature of the
        # covered skin (same air): reference of the 'direct' suppression of the companion paper
        Ts_c = oc.Ts[mk, :]
        Ue = F.U[mk][:, None] * Pv.get("s_U", np.ones(1))[None, :]
        Ta_, ea_ = F.Ta[mk][:, None], F.ea[mk][:, None]
        Ce_h, _, _ = mf.transfer_velocities(Ts_c, Ta_ * np.ones_like(Ts_c), ea_ * np.ones_like(Ts_c), Ue,
                                            {k: v[None, :] for k, v in Pv.items()})
        es_c = pr.esat(Ts_c)
        drv = es_c / (pr.R_V * (Ts_c + pr.K0)) - ea_ / (pr.R_V * (Ta_ + pr.K0))
        Eh = (Ce_h * drv).sum(0) * 300 / 997 * 1000
        rows.append(dict(win=wid, segment=seg.name, cover=cov, version=ver, n_plates=npl, regime=regime,
                         hours=hrs, E_cov_mm_h=float(np.median(Ec / hrs)), E_open_mm_h=float(np.median(Eo / hrs)),
                         S_model=float(np.median(1 - Ec / Eo)) if cov else np.nan,
                         S_direct_model=float(np.median(1 - Ec / Eh)) if cov else np.nan,
                         Ts_cov=float(oc.Ts[mk].mean()), Ts_open=float(oo.Ts[mk].mean()),
                         Tp=float(oc.Tp[mk].mean()), I_mean=float(seg.forcing.I[mk].mean()),
                         U_mean=float(seg.forcing.U[mk].mean())))
        series[wid] = dict(t=idx[mk], Ec=oc.E[mk, 0], Eo=oo.E[mk, 0], Ts=oc.Ts[mk, 0], Ts0=oo.Ts[mk, 0],
                           Tp=oc.Tp[mk, 0], fl_c={k: v[mk, 0] for k, v in oc.fluxes.items()},
                           fl_o={k: v[mk, 0] for k, v in oo.fluxes.items()})
    return pd.DataFrame(rows), series


def main(map_path=None, out_png=None):
    import matplotlib.pyplot as plt
    map_path = Path(map_path or c1.OUT / "map_c1.json")
    out_png = Path(out_png or c1.OUT / "diagnostico_c1.png")
    m = json.load(open(map_path))
    prob = c1.C1Problem(m["base"], rho=m["rho_used"], weights=m["weights_used"], processes=1)
    U = np.array(m["u"])[None]
    pred = prob.predict(prob.params_from_u(U))
    ll, parts = prob.loglik(pred, return_parts=True)
    o = prob.obs
    fig, axs = plt.subplots(4, 1, figsize=(20, 15), sharex=True)
    for k, col in zip(c1.PROBES, st.SERIES):
        axs[0].plot(o.index, o[k], ".", ms=1.5, color=col, alpha=.5)
        axs[0].plot(o.index, pred[k][:, 0], "-", lw=.8, color=col, label=k)
        axs[1].plot(o.index, o[k] - pred[k][:, 0], ".", ms=1.5, color=col, label=k)
    axs[0].legend(); axs[1].axhline(0, color="k", lw=.5); axs[1].legend()
    beta = parts["level_a_qc"]["beta"][0]
    sc = parts["level_a_qc"]["dH_scale"][0]
    X = np.column_stack([prob.D, prob.obs.Tb_c.to_numpy(), pred["dH_mm"][:, 0] / sc])
    axs[2].plot(o.index, o.level_a_qc - X @ beta, ".", ms=1.5,
                label=f"QDY resid (gain {beta[-1] / sc:.3f}, c {beta[-2]:.3f} mm/K)")
    mres = parts["micrometer"]["res"][:, 0]
    axs[2].plot(prob.mic.time, mres, "o", ms=4, color=st.SERIES[1], label="micrometer resid")
    axs[2].axhline(0, color="k", lw=.5); axs[2].legend()
    axs[3].plot(o.index, pred["E"][:, 0] * 3600, "k-", lw=.8); axs[3].set_ylabel("E (mm/h)")
    for ax in axs:
        for s in prob.segs:
            if s.phi > 0:
                ax.axvspan(s.start, s.end, color=st.SERIES[0], alpha=0.06)
    plt.tight_layout(); plt.savefig(out_png, dpi=55); plt.close(fig)
    rms = {k: float(np.sqrt(np.nanmean((o[k] - pred[k][:, 0]) ** 2))) for k in c1.PROBES}
    rms["QDY"] = float(np.sqrt(np.nanmean((o.level_a_qc - X @ beta) ** 2)))
    rms["micrometer"] = float(np.sqrt(np.mean(mres ** 2)))
    print("RMS:", {k: round(v, 3) for k, v in rms.items()})
    ws, _ = window_runs(m)
    pd.set_option("display.width", 220)
    print(ws.round(3).to_string())
    ws.to_csv(c1.OUT / "c1_window_suppression_map.csv", index=False)
    return ws


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
