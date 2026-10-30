# -*- coding: utf-8 -*-
"""
Collects every number quoted in the Article 1 manuscript.

Reads the calibration, posterior, scenario and verification outputs and
writes saidas/resultados_artigo1.json (flat keys used as {{placeholders}} in
the manuscript) plus the manuscript tables in saidas/tabelas/. Fit
statistics are recomputed at the current mode (map_c2.json, map_c1.json).

Usage: python resultados_artigo1.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibracao as cb          # noqa: E402
import dados_modelo as dm        # noqa: E402
import modelo_fisico as mf       # noqa: E402
from priors import Prior, PRIORS_C2   # noqa: E402

SAI = HERE.parent / "saidas"
CAL, SIM, TAB = SAI / "calibracao", SAI / "simulacoes", SAI / "tabelas"
R = {}
TXT = {}          # the same values formatted for the manuscript


def fmt(value, nd=None):
    """Text form: fixed decimals, true minus sign, powers of ten for small numbers."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)) or nd == 0:
        t = f"{int(round(value))}"
    elif nd is None:
        t = f"{value:g}"
    elif value != 0 and abs(value) < 1e-3:
        e = int(np.floor(np.log10(abs(value))))
        t = f"${value / 10 ** e:.1f}\\times10^{{{e}}}$"
    else:
        t = f"{value:.{nd}f}"
    return t.replace("-", "−") if not t.startswith("$") else t


def sigtxt(v, n=2):
    """n significant digits (3 for values between 1 and 10); powers of ten below 1e-3."""
    if v == 0 or not np.isfinite(v):
        return "0"
    if 1 <= abs(v) < 10:
        n = max(n, 3)
    e = int(np.floor(np.log10(abs(v))))
    if e < -3:
        return f"${v / 10 ** e:.{n - 1}f}\\times10^{{{e}}}$"
    d = max(n - 1 - e, 0)
    return f"{v:.{d}f}".replace("-", "−")


def put_sig(key, value, n=2):
    value = float(value)
    R[key] = value
    TXT[key] = sigtxt(value, n)


def put(key, value, nd=None):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    R[key] = round(value, nd) if (nd is not None and isinstance(value, float)) else value
    TXT[key] = fmt(value, nd)


# ---------------------------------------------------------------- numerics
def numerics():
    v = json.load(open(TAB / "verificacao_numerica.json"))
    d = v["dt300_vs_dt20"]
    put("num_rms04", d["Tw_04"], 3); put("num_rms12", d["Tw_12"], 3); put("num_rms20", d["Tw_20"], 3)
    rr = [d["Tw_04"], d["Tw_12"], d["Tw_20"]]
    put("num_rms_min", min(rr), 3); put("num_rms_max", max(rr), 3)
    put("num_rms_ts", d["Ts_rms"], 2); put("num_E", d["E_rel_pct"], 2)
    put("num_grid_max", max(v["N24_vs_N48"][k] for k in ("Tw_04", "Tw_12", "Tw_20")), 3)
    put("num_grid_E", v["N24_vs_N48"]["E_rel_pct"], 3)
    put("num_closure", max(v["closure_uncovered_W_m2"], v["closure_covered_W_m2"]))


# ---------------------------------------------------------------- C2 fit at the mode
def _runs(f, min_steps=24):
    on = (f.I > 50).to_numpy()
    grp = (on != np.roll(on, 1)).cumsum()
    for g in np.unique(grp[on]):
        mk = (grp == g) & on
        if mk.sum() >= min_steps:
            yield f.index[mk][0], f.index[mk][-1] + pd.Timedelta(minutes=5), mk


def fit_c2():
    m = json.load(open(CAL / "map_c2.json"))
    prob = cb.C2Problem(rho=m["rho_used"], weights=m["weights_used"], processes=1)
    pred = prob.predict(prob.params_from_u(np.array(m["u"])[None]))
    _, parts = prob.loglik(pred, return_parts=True)
    o = prob.obs
    rows = []
    lamp = np.zeros(len(o), bool)
    for s in prob.segs:
        mk = (prob.seg_of == s.name)
        Ii = s.forcing.I.reindex(o.index[mk], method="nearest").to_numpy()
        lamp[mk] = Ii > 50
    for k in cb.T_STREAMS:
        v = prob.valid[k]
        r = o[k].to_numpy() - pred[k][:, 0]
        rad, _ = cb.unit_rms(pred["rad_" + k], v)
        X = np.column_stack([rad[:, 0]] if k == "Tw_cs" else [np.ones(len(r)), rad[:, 0]])
        e = r - X @ parts[k]["beta"][0]
        rows.append(dict(stream=k, n=int(v.sum()), rmse=float(np.sqrt(np.mean(e[v] ** 2))),
                         rmse_lamp=float(np.sqrt(np.mean(e[v & lamp] ** 2))),
                         rmse_dark=float(np.sqrt(np.mean(e[v & ~lamp] ** 2))),
                         bias=float(np.mean(e[v])), offset=float(parts[k]["beta"][0][0]) if k != "Tw_cs" else 0.0,
                         rho=m["rho_used"][k], weight=m["weights_used"][k]))
    nseg = len(prob.names)
    Tc = o.Tcs_c.to_numpy(); td = o.t_seg_d.to_numpy()
    sc = parts["level_a_qc"]["dH_scale"][0]
    ba = parts["level_a_qc"]["beta"][0]; bb = parts["level_b_qc"]["beta"][0]
    g = ba[-1] / sc
    dH = pred["dH_mm"][:, 0]
    ya = (o.level_a_qc.to_numpy() - ba[:nseg] @ np.eye(nseg)[[prob.names.index(x) for x in prob.seg_of]].T
          - ba[nseg] * Tc) / g
    Dm = prob.D
    yb = (o.level_b_qc.to_numpy() - Dm @ bb[:nseg] - bb[nseg] * Tc - (Dm * td[:, None]) @ bb[nseg + 1:]) / cb.K_CS
    for k, y in (("level_a_qc", ya), ("level_b_qc", yb)):
        v = prob.valid[k]
        e = y - dH
        rows.append(dict(stream=k, n=int(v.sum()), rmse=float(np.sqrt(np.mean(e[v] ** 2))),
                         rmse_lamp=float(np.sqrt(np.mean(e[v & lamp] ** 2))),
                         rmse_dark=float(np.sqrt(np.mean(e[v & ~lamp] ** 2))), bias=float(np.mean(e[v])),
                         rho=m["rho_used"][k], weight=m["weights_used"][k]))
    fit = pd.DataFrame(rows)
    fit.to_csv(TAB / "ajuste_c2.csv", index=False)
    for _, r in fit.iterrows():
        put(f"c2_rmse_{r.stream}", r.rmse, 2); put(f"c2_rmse_lamp_{r.stream}", r.rmse_lamp, 2)
        put(f"c2_rmse_dark_{r.stream}", r.rmse_dark, 2)
        if r.stream.startswith("Tw") and r.stream != "Tw_cs":
            put(f"c2_offset_{r.stream}", r.offset, 2)
    put("c2_ody_gain", g, 3); put("c2_ody_ca", ba[nseg], 3); put("c2_cs_cb", bb[nseg], 3)
    drift = bb[nseg + 1:]
    put("c2_cs_drift_min", float(drift.min()), 2); put("c2_cs_drift_max", float(drift.max()), 2)
    put("c2_cs_drift_first", float(drift[0]), 2); put("c2_cs_drift_last", float(drift[-1]), 2)
    pd.DataFrame(dict(segment=prob.names, drift_mm_d=drift / cb.K_CS)).to_csv(TAB / "deriva_cs456.csv", index=False)
    # run-scale evaporation: corrected records vs model
    rr = []
    for s in prob.segs:
        mk = (prob.seg_of == s.name)
        t = o.index[mk]
        for a, b, fm in _runs(s.forcing):
            def level(y, when):
                w = (t >= when - pd.Timedelta(minutes=30)) & (t <= when + pd.Timedelta(minutes=30))
                return np.nanmean(y[mk][w]) if w.any() else np.nan
            hrs = (b - a).total_seconds() / 3600
            a2 = a + pd.Timedelta(minutes=30)
            row = dict(segment=s.name, start=a, hours=hrs, I=float(s.forcing.I[fm].mean()),
                       U=float(s.forcing.U[fm].mean()), cfg=int(np.bincount(s.forcing.cfg[fm]).argmax()))
            for nm, y in (("model", dH), ("ody", ya), ("cs456", yb)):
                row[f"E_{nm}"] = float(-(level(y, b) - level(y, a2)) / ((b - a2).total_seconds() / 3600))
            rr.append(row)
    runs = pd.DataFrame(rr)
    runs.to_csv(TAB / "evaporacao_corridas_c2.csv", index=False)
    for nm in ("ody", "cs456"):
        e = runs[f"E_{nm}"] - runs.E_model
        put(f"c2_run_rmse_{nm}", float(np.sqrt(np.mean(e ** 2))), 3)
        put(f"c2_run_bias_{nm}", float(e.mean()), 3)
        put(f"c2_run_r2_{nm}", float(1 - np.sum(e ** 2) / np.sum((runs[f'E_{nm}'] - runs[f'E_{nm}'].mean()) ** 2)), 3)
    put("c2_n_runs", int(len(runs)))
    put("c2_E_full_calm", float(runs[(runs.cfg == 0) & (runs.U < 0.5)].E_model.mean()), 2)
    put("c2_E_full_fan3", float(runs[(runs.cfg == 0) & (runs.U > 2.8)].E_model.mean()), 2)
    put("c2_E_min", float(runs.E_model.min()), 2); put("c2_E_max", float(runs.E_model.max()), 2)
    put("c2_E_ratio_fan3_calm", R["c2_E_full_fan3"] / R["c2_E_full_calm"], 2)
    # observed 4-cm minus 12-cm difference (offset-corrected) in calm full-array runs
    off = dm.probe_offsets_c2()
    d412 = (o.Tw_04 - off["Tw_04"]) - (o.Tw_12 - off["Tw_12"])
    calm = np.zeros(len(o), bool)
    for s in prob.segs:
        mk = (prob.seg_of == s.name)
        Ii = s.forcing.I.reindex(o.index[mk], method="nearest").to_numpy()
        Ui = s.forcing.U.reindex(o.index[mk], method="nearest").to_numpy()
        calm[mk] = (Ii > 700) & (Ui < 0.3)
    put("c2_obs_dT04_12_calm_max", float(d412[calm].max()), 1)
    return fit, runs


def _warm_layer(out, k, T12):
    """Thickness (cm) of the layer warmer than the 12-cm temperature + 1 K at step k."""
    T = out.T[k, 0]; H = out.H[k, 0]
    z = out.zc_frac * H
    warm = T > T12[k, 0] + 1.0
    return float(z[warm].max() * 100) if warm.any() else 0.0


def profiles():
    """Skin-bulk contrast of every C2 lamp run (mean of the last 2 h of the run)."""
    m = json.load(open(CAL / "map_c2.json"))
    P = mf.default_params(1)
    P.update({k: np.array([v]) for k, v in m["params"].items()})
    s_ = mf.sigma_grid(cb.N_LAYERS); zc = np.cumsum(s_) - s_ / 2
    rows = []
    for seg in dm.c2_segments():
        out = mf.simulate(dm.forcing_arrays(seg), {k: v.copy() for k, v in P.items()},
                          dm.initial_profile(seg, zc), seg.H0, N=cb.N_LAYERS)
        T12, _ = mf.temperature_at_height(out, 0.12)
        Tfl, _ = mf.temperature_at_height(out, 0.005)
        f = seg.forcing
        for a, b, mk in _runs(f):
            last = mk & np.asarray(f.index >= b - pd.Timedelta(hours=2))
            rows.append(dict(segment=seg.name, start=a, I=float(f.I[mk].mean()), U=float(f.U[mk].mean()),
                             cfg=int(np.bincount(f.cfg[mk]).argmax()),
                             Ts=float(out.Ts[last, 0].mean()), T12=float(T12[last, 0].mean()),
                             Tfloor=float(Tfl[last, 0].mean()),
                             dT_skin=float((out.Ts[last, 0] - T12[last, 0]).mean()),
                             warm_layer_cm=_warm_layer(out, np.flatnonzero(mk)[-1], T12)))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "perfis_corridas_c2.csv", index=False)
    full = df[df.cfg == 0]
    put("prof_dTskin_calm", float(full[full.U < 0.3].dT_skin.mean()), 1)
    put("prof_dTskin_fan3", float(full[full.U > 2.8].dT_skin.mean()), 1)
    put("prof_warm_calm_cm", float(full[full.U < 0.3].warm_layer_cm.mean()), 0)
    put("prof_warm_fan3_cm", float(full[full.U > 2.8].warm_layer_cm.mean()), 0)
    put("prof_Ts_calm", float(full[full.U < 0.3].Ts.mean()), 1)
    put("prof_Ts_fan3", float(full[full.U > 2.8].Ts.mean()), 1)
    return df


# ---------------------------------------------------------------- posterior
def _nat(u, tr):
    return np.exp(u) if tr == "log" else (1 / (1 + np.exp(-u)) if tr == "logit" else u)


def posterior_c2():
    p = pd.read_csv(TAB / "posterior_c2.csv").set_index("parameter")
    info = json.load(open(CAL / "mode_c2_posterior.json"))
    put("c2_mcmc_iter", info["n_iter"]); put("c2_mcmc_walkers", info["walkers"])
    put("c2_mcmc_draws", info["n_kept"]); put("c2_max_rhat", info["max_rhat"], 3)
    put("c2_max_iat_post", info["max_iat"], 0)
    put("c2_rhat_min", float(pd.read_csv(TAB / "posterior_c2.csv").rhat.min()), 2)
    put("c2_rhat_max", float(pd.read_csv(TAB / "posterior_c2.csv").rhat.max()), 2)
    put("c2_n_iat", info["n_iter"] * (1 - info["burn"]) / info["max_iat"], 0)
    put("c2_min_ess", info["min_ess"], 0); put("c2_acc", info["acceptance"], 2)
    for k, r in p.iterrows():
        put_sig(f"post_{k}", r.post_median); put_sig(f"post_{k}_lo", r.post_lo); put_sig(f"post_{k}_hi", r.post_hi)
    # derived quantities from the draws
    z = cb.load_chain("c2")
    ch = z["chain"]; keep = ch[ch.shape[0] // 2:].reshape(-1, ch.shape[2])
    pr = Prior(PRIORS_C2)
    X = {k: _nat(keep[:, i], pr.tr[i]) for i, k in enumerate(pr.names)}
    Atot = X["A_s"] + X["A_m"] + X["A_d"]
    q = lambda a: np.percentile(a, [50, 2.5, 97.5])
    for nm, arr, nd in (("A_tot", Atot, 2), ("f_s", X["A_s"] / Atot, 3), ("f_m", X["A_m"] / Atot, 3),
                        ("f_d", X["A_d"] / Atot, 3),
                        ("A_tot_upper", Atot * X["m_1"], 2), ("A_tot_lower", Atot * X["m_2"], 2)):
        a, lo, hi = q(arr)
        put(f"post_{nm}", a, nd); put(f"post_{nm}_lo", lo, nd); put(f"post_{nm}_hi", hi, nd)
    # transfer velocity at typical lamp-run conditions
    rng = np.random.default_rng(1)
    idx = rng.choice(len(keep), size=min(2000, len(keep)), replace=False)
    Pd = {k: v[idx] for k, v in X.items()}
    rows = []
    for U in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
        Ce, Ch, _ = mf.transfer_velocities(np.full(len(idx), 36.0), np.full(len(idx), 30.0),
                                           np.full(len(idx), 2450.0), np.full(len(idx), U), Pd)
        a, lo, hi = q(Ce * 1000)
        rows.append(dict(U=U, Ce_mm_s=a, lo=lo, hi=hi))
        put(f"post_Ce_U{U:g}", a, 2); put(f"post_Ce_U{U:g}_lo", lo, 2); put(f"post_Ce_U{U:g}_hi", hi, 2)
    pd.DataFrame(rows).to_csv(TAB / "velocidade_transferencia_c2.csv", index=False)
    put("post_Ce_ratio_U3_U0", R["post_Ce_U3"] / R["post_Ce_U0"], 1)
    for k in ("l_m", "d_w"):
        for s_ in ("", "_lo", "_hi"):
            put_sig(f"post_{k}{s_}_mm", R[f"post_{k}{s_}"] * 1000)
    put("post_K_bg_over_kappa", R["post_K_bg"] / 1.45e-7, 0)
    # diffusive boundary-layer thickness delta = D / C_e (D = 2.6e-5 m2/s at 30-35 degC)
    put("delta_calm_mm", 2.6e-5 / (R["post_Ce_U0"] * 1e-3) * 1000, 0)
    put("delta_U3_mm", 2.6e-5 / (R["post_Ce_U3"] * 1e-3) * 1000, 1)
    contr = p["contraction"]
    put("post_contr_min_param", str(contr.idxmin())); put("post_contr_min", float(contr.min()), 2)
    names = {"G_w": "the wall-heating coefficient $G_w$", "l_d": "the e-folding depth of the deep band $\\ell_d$",
             "C_0": "the background transfer velocity $C_0$", "a_b": "the floor absorptance $a_b$",
             "U_b": "the floor heat-transfer coefficient $U_b$", "d_w": "the wind-mixing depth $d_w$",
             "U_w": "the wall heat-transfer coefficient $U_w$", "K_bg": "the background diffusivity $K_{bg}$"}
    order = contr.sort_values()
    TXT["post_contr_min_txt"] = names.get(order.index[0], order.index[0])
    TXT["post_contr_2nd_txt"] = names.get(order.index[1], order.index[1])
    put("post_contr_2nd", float(order.iloc[1]), 2)
    put("post_n_contr90", int((contr > 0.9).sum())); put("post_n_params", int(len(contr)))
    return p


def posterior_c1():
    p = pd.read_csv(TAB / "posterior_c1.csv").set_index("parameter")
    info = json.load(open(CAL / "mode_c1_posterior.json"))
    put("c1_mcmc_iter", info["n_iter"]); put("c1_mcmc_walkers", info["walkers"])
    put("c1_mcmc_draws", info["n_kept"]); put("c1_max_rhat", info["max_rhat"], 3)
    put("c1_max_iat_post", info["max_iat"], 0)
    put("c1_rhat_min", float(pd.read_csv(TAB / "posterior_c1.csv").rhat.min()), 2)
    put("c1_rhat_max", float(pd.read_csv(TAB / "posterior_c1.csv").rhat.max()), 2)
    put("c1_n_iat", info["n_iter"] * (1 - info["burn"]) / info["max_iat"], 0)
    put("c1_min_ess", info["min_ess"], 0); put("c1_acc", info["acceptance"], 2)
    for k, r in p.iterrows():
        put_sig(f"post_{k}", r.post_median); put_sig(f"post_{k}_lo", r.post_lo); put_sig(f"post_{k}_hi", r.post_hi)
    for k in ("R_rec_V1", "R_rec_V2", "r_open", "h_p1", "h_p2", "h_p3", "C_u_V1", "C_u_V2"):
        for s in ("", "_lo", "_hi"):
            put_sig(f"post_{k}{s}_mm", R[f"post_{k}{s}"] * 1000)
    put("post_C_u_ratio", float(p.loc["C_u_V1", "post_median"] / p.loc["C_u_V2", "post_median"]), 1)
    return p


def convergence_text():
    """Methods and Discussion sentences on the convergence of the chains, written
    from the diagnostics with the criteria of convergir_cadeias.py."""
    ok, lab = {}, {"c2": "C2", "c1": "C1"}
    for t in ("c2", "c1"):
        info = json.load(open(CAL / f"mode_{t}_posterior.json"))
        ok[t] = info["max_rhat"] < cb.CONV_RHAT and info["n_iter"] * (1 - info["burn"]) / info["max_iat"] >= cb.CONV_NIAT
    crit = (f"a maximum $\\hat R$ below {cb.CONV_RHAT:g} and retained draws (the second half of each chain) "
            f"spanning at least {cb.CONV_NIAT} integrated autocorrelation times")
    top = pd.read_csv(TAB / "posterior_c2.csv").set_index("parameter")["iat"].idxmax()
    slow = ("The slowest direction was the partition between the skin and near-surface absorption bands (Figure S2)."
            if top in ("A_s", "A_m") else
            "The slowest directions involved the radiative partition and the transfer coefficients (Figure S2).")
    caveat = ("{} (Table [tab:priors] and the scenario band) are therefore indicative, whereas the posterior modes, "
              "on which the mechanism, map and Sobol analyses rest, are not affected.")
    if all(ok.values()):
        m = (f"The chains were continued from their last state until {crit} in both campaigns, which took "
             "{{c2_mcmc_iter}} iterations for C2 and {{c1_mcmc_iter}} for C1: at the end, $\\hat R$ was at most "
             "{{c2_max_rhat}} for C2 and {{c1_max_rhat}} for C1, and the retained draws spanned {{c2_n_iat}} and "
             "{{c1_n_iat}} times the longest autocorrelation times, {{c2_max_iat_post}} and {{c1_max_iat_post}} "
             "iterations. " + slow)
        d = "so that long chains were needed to meet the convergence criteria; "
    elif any(ok.values()):
        g, b = ("c1", "c2") if ok["c1"] else ("c2", "c1")
        m = (f"The chains were continued from their last state until {crit}, or until the available computing "
             f"time was exhausted. The {lab[g]} chain met these criteria after {{{{{g}_mcmc_iter}}}} iterations "
             f"($\\hat R$ at most {{{{{g}_max_rhat}}}}; retained draws spanning {{{{{g}_n_iat}}}} times its longest "
             f"autocorrelation time of {{{{{g}_max_iat_post}}}} iterations), but the {lab[b]} chain had not after "
             f"{{{{{b}_mcmc_iter}}}} iterations ($\\hat R$ up to {{{{{b}_max_rhat}}}}; retained draws spanning "
             f"{{{{{b}_n_iat}}}} times its longest autocorrelation time of {{{{{b}_max_iat_post}}}} iterations). "
             + slow + " " + caveat.format(f"The {lab[b]} posterior intervals"))
        d = f"and the {lab[b]} chain had not met the convergence criteria, so its posterior intervals are indicative; "
    else:
        m = (f"The chains were continued from their last state until {crit}, or until the available computing "
             "time was exhausted. Neither chain met these criteria: after {{c2_mcmc_iter}} iterations for C2 and "
             "{{c1_mcmc_iter}} for C1, $\\hat R$ was up to {{c2_max_rhat}} and {{c1_max_rhat}}, and the retained "
             "draws spanned only {{c2_n_iat}} and {{c1_n_iat}} times the longest autocorrelation times "
             "({{c2_max_iat_post}} and {{c1_max_iat_post}} iterations). " + slow + " "
             + caveat.format("All posterior intervals"))
        d = "and the chains had not met the convergence criteria, so all posterior intervals are indicative; "
    TXT["mcmc_conv_txt"] = m
    TXT["mcmc_disc_txt"] = ("the ensemble sampler mixed slowly, above all along the ridge between the skin and "
                            "near-surface absorption bands and in the transfer coefficients (integrated "
                            "autocorrelation times of up to {{c2_max_iat_post}} iterations; Figure S2), " + d)
    R["mcmc_converged_c2"], R["mcmc_converged_c1"] = ok["c2"], ok["c1"]


# ---------------------------------------------------------------- C1 fit and windows
def fit_c1():
    import calibracao_c1 as c1
    m = json.load(open(CAL / "map_c1.json"))
    prob = c1.C1Problem(m["base"], rho=m["rho_used"], weights=m["weights_used"], processes=1)
    pred = prob.predict(prob.params_from_u(np.array(m["u"])[None]))
    _, parts = prob.loglik(pred, return_parts=True)
    o = prob.obs
    for k in c1.PROBES:
        v = np.isfinite(o[k].to_numpy()) & np.isfinite(pred[k][:, 0])
        e = o[k].to_numpy() - pred[k][:, 0]
        e = e - np.nanmean(e[v])
        put(f"c1_rmse_{k}", float(np.sqrt(np.mean(e[v] ** 2))), 2)
    sc = parts["level_a_qc"]["dH_scale"][0]; beta = parts["level_a_qc"]["beta"][0]
    g_, c_ = c1.qdy_coefs(parts)
    put("c1_qdy_gain", g_, 3); put("c1_qdy_c", c_, 3)
    put("c1_qdy_gain_fixed", "fixed" if c1.QDY_GAIN is not None else "free")
    eq = prob.obs.level_a_qc.to_numpy() - c1.qdy_fitted(prob, parts, pred)
    put("c1_qdy_rmse", float(np.sqrt(np.nanmean(eq ** 2))), 2)
    mres = parts["micrometer"]["res"][:, 0]
    put("c1_mic_rmse", float(np.sqrt(np.mean(mres ** 2))), 2)
    put("c1_mic_n", int(len(mres)))
    pm = parts["micrometer"]
    put("c1_mic_sd_hat", float(np.sqrt(pm["rss"][0] / (pm["n"] - pm["p"]))), 2)
    put("c1_mic_nseg", int(pm["p"]))
    micrometer_table(prob)
    f = TAB / "c1_supressao_janelas_modelo.csv"
    ws = pd.read_csv(f if f.exists() else CAL / "c1_window_suppression_map.csv")
    cov = ws[ws.cover == 1].copy()
    cov["phi"] = cov.n_plates * 0.031437 / 1.1029
    cov["dTs"] = cov.Ts_cov - cov.Ts_open
    put("c1_S_min", cov.S_model.min(), 2); put("c1_S_max", cov.S_model.max(), 2)
    put("c1_S_lamp_min", cov[cov.I_mean > 250].S_model.min(), 2)
    put("c1_S_lamp_max", cov[cov.I_mean > 250].S_model.max(), 2)
    lw = cov[cov.I_mean > 250]          # the lamp windows, as for c1_S_lamp_*
    put("c1_Sphi_min", (lw.S_model / lw.phi).min(), 2); put("c1_Sphi_max", (lw.S_model / lw.phi).max(), 2)
    put("c1_Tp_min", cov.Tp.min(), 0); put("c1_Tp_max", cov.Tp.max(), 0)
    for reg in ("strong", "calm", "weak"):
        s = cov[cov.regime == reg]
        if len(s):
            put(f"c1_dTs_{reg}_min", s.dTs.min(), 1); put(f"c1_dTs_{reg}_max", s.dTs.max(), 1)
            put(f"c1_S_{reg}_mean", s.S_model.mean(), 2)
    lamp = cov.I_mean > 250
    groups = {"V2fan": lamp & (cov.version == "V2") & (cov.regime == "strong"),
              "V1fan": lamp & (cov.version == "V1") & (cov.regime == "strong"),
              "calm": lamp & (cov.regime == "calm")}
    for k, mk in groups.items():
        if mk.any():
            put(f"c1_S_{k}_min", cov[mk].S_model.min(), 2); put(f"c1_S_{k}_max", cov[mk].S_model.max(), 2)
    fan = cov.regime == "strong"
    put("c1_dTs_strong_maxabs", float(cov[fan].dTs.abs().max()), 1)
    put("c1_dTs_strong_minabs", float(cov[fan].dTs.abs().min()), 1)
    put("c1_Tp_fan_max", float(cov[fan].Tp.max()), 0)
    if "S_direct_model" in cov:
        put("c1_Sdir_model_min", cov.S_direct_model.min(), 2); put("c1_Sdir_model_max", cov.S_direct_model.max(), 2)
        a2 = HERE.parents[1] / "ARTIGO_2_INFERENCIA_ESTOCASTICA" / "saidas" / "tabelas" / "supressao_direta_janelas.csv"
        if a2.exists():
            d2 = pd.read_csv(a2).set_index("grupo")
            col = "S_rep" if "S_rep" in d2 else "S_direct"
            j = cov.set_index("win").join(d2[[col]], how="inner")
            put("cmp_a2_mae", float(np.mean(np.abs(j.S_direct_model - j[col]))), 2)
            put("cmp_a2_mean", float(np.mean(j.S_direct_model - j[col])), 2)
    # first calibration (opaque plates, gaps only, free QDY30A gain): sensitivity case
    v2 = CAL / "v2_c1_placas_opacas"
    if (v2 / "map_c1.json").exists():
        mv = json.load(open(v2 / "map_c1.json"))
        nu = mv["nuisance"]["level_a_qc"]
        put("c1_v2_gain", float(nu[-1] / mv["qdy_dH_scale"]), 2)
        wv = pd.read_csv(v2 / "c1_window_suppression_map.csv")
        wv = wv[(wv.cover == 1) & (wv.I_mean > 250)]
        put("c1_v2_S_min", wv.S_model.min(), 2); put("c1_v2_S_max", wv.S_model.max(), 2)
        keep = c1.QDY_GAIN
        c1.QDY_GAIN = None
        try:
            pv = c1.C1Problem(mv["base"], rho=mv["rho_used"], weights=mv["weights_used"], processes=1)
            Pv = {k: np.array([v], float) for k, v in mv["params"].items()}
            Pv.setdefault("m_6", Pv["m_5"])
            predv = pv.predict(Pv)
            _, partsv = pv.loglik(predv, return_parts=True)
            put("c1_v2_mic_rmse", float(np.sqrt(np.mean(partsv["micrometer"]["res"][:, 0] ** 2))), 2)
        finally:
            c1.QDY_GAIN = keep
    cov.to_csv(TAB / "c1_janelas_cobertas.csv", index=False)
    fv = TAB / "c1_variantes.csv"
    if fv.exists():
        vv = pd.read_csv(fv).set_index("variant")
        for v, r in vv.iterrows():
            put(f"c1_var{v}_mic_rmse", r.mic_rmse, 2); put(f"c1_var{v}_gain", r.qdy_gain, 2)
            if "fall_model" in r and np.isfinite(r.fall_model):
                put(f"c1_var{v}_fall", r.fall_model, 2); put("c1_mic_fall_obs", r.fall_obs, 2)
            for k in ("S_min", "S_max", "S_mean"):
                put(f"c1_var{v}_{k}", r[k], 2)
        if {"F", "G", "H"} <= set(vv.index):
            put("c1_gain_dS", float(max(abs(vv.S_mean["G"] - vv.S_mean["F"]), abs(vv.S_mean["H"] - vv.S_mean["F"]))), 2)
    return cov


def micrometer_table(prob):
    """Every C1 micrometer reading with its use in the calibration (Table S8)."""
    import calibracao_c1 as c1
    m = pd.read_csv(dm.cfg.MICROMETER_C1)
    m["time"] = pd.to_datetime(m["date"] + " " + m["time"])
    used = set(prob.mic.time)
    why = {pd.Timestamp("2026-08-17 09:21"): "installation of the stilling well (1.8 mm above the trend of the day)",
           pd.Timestamp("2026-08-20 17:51"): "pair implying a fall of 2.1 mm h⁻¹ in 27 min",
           pd.Timestamp("2026-08-20 18:18"): "pair implying a fall of 2.1 mm h⁻¹ in 27 min",
           pd.Timestamp("2026-09-01 16:20"): "taken at the reinstallation of the plates"}
    rows = []
    for _, r in m.iterrows():
        note = str(r.note) if isinstance(r.note, str) else ""
        if r.time in used:
            st_ = "used"
        elif r.time in why:
            st_ = why[r.time]
        elif any(w in note.lower() for w in ("non-monotonic", "refill", "buoyancy", "deployed", "removed")):
            st_ = "change of level or cover (logged)"
        else:
            st_ = "outside a model segment or alone in it"
        rows.append(dict(time=r.time, reading_mm=r.reading_mm, n_plates=r.n_plates, note=note, status=st_))
    d = pd.DataFrame(rows)
    d.to_csv(TAB / "c1_micrometro.csv", index=False)
    put("c1_mic_total", len(d)); put("c1_mic_used", int((d.status == "used").sum()))
    put("c1_mic_excl_explicit", int(d.status.isin(set(why.values())).sum()))
    put("c1_mic_excl_logged", int((d.status == "change of level or cover (logged)").sum()))
    put("c1_mic_outside", int((d.status == "outside a model segment or alone in it").sum()))
    return d


# ---------------------------------------------------------------- scenarios
def scenarios():
    sh = pd.read_csv(SIM / "shapley_grade.csv")
    MECH = {"sw_vapour": "block", "sw_shade": "shade", "sw_insul": "insul", "sw_cond": "cond"}
    for I in (300.0, 520.0, 820.0):
        s = sh[(sh.I == I) & (sh.phi == 0.74)]
        for k, lab in MECH.items():
            put(f"sh_{lab}_I{I:.0f}_min", s[k].min(), 2); put(f"sh_{lab}_I{I:.0f}_max", s[k].max(), 2)
        put(f"sh_S_I{I:.0f}_min", s.S_total.min(), 2); put(f"sh_S_I{I:.0f}_max", s.S_total.max(), 2)
        put(f"sh_shade_frac_I{I:.0f}", float((s.sw_shade / s.S_total).mean()), 2)
        put(f"sh_block_frac_I{I:.0f}", float((s.sw_vapour / s.S_total).mean()), 2)
    for I in (300.0, 820.0):
        s = sh[(sh.I == I) & (sh.phi == 0.74) & (sh.U == 3.0)].iloc[0]
        for k, lab in MECH.items():
            put(f"sh_single_{lab}_I{I:.0f}_U3", s["single_" + k], 2)
            put(f"sh_{lab}_I{I:.0f}_U3", s[k], 2)
        put(f"sh_S_I{I:.0f}_U3", s.S_total, 2)
        put(f"sh_single_sum_I{I:.0f}_U3", float(sum(s["single_" + k] for k in MECH)), 2)
    s = sh[sh.phi == 0.25]
    put("sh_block_frac_phi25", float((s.sw_vapour / s.S_total).mean()), 2)
    s = sh[sh.phi == 0.90]
    put("sh_block_frac_phi90", float((s.sw_vapour / s.S_total).mean()), 2)
    mp_ = pd.read_csv(SIM / "mapa_supressao.csv")
    mp_["eff"] = mp_.S / mp_.phi
    for I in (300.0, 820.0):
        s = mp_[mp_.I == I]
        put(f"map_eff_min_I{I:.0f}", s.eff.min(), 2); put(f"map_eff_max_I{I:.0f}", s.eff.max(), 2)
        hi = s[s.phi >= 0.5]
        put(f"map_effhi_min_I{I:.0f}", hi.eff.min(), 2); put(f"map_effhi_max_I{I:.0f}", hi.eff.max(), 2)
        put(f"map_Eopen_calm_I{I:.0f}", float(s[s.U == 0].E_open.mean()), 2)
        put(f"map_Eopen_U4_I{I:.0f}", float(s[s.U == 4].E_open.mean()), 2)
    s = mp_[(mp_.I == 820) & (np.isclose(mp_.phi, 0.75))]
    put("map_S_phi75_U0", float(s[s.U == 0].S.iloc[0]), 2); put("map_S_phi75_U4", float(s[s.U == 4].S.iloc[0]), 2)
    for ph in (0.25, 0.95):
        s = mp_[(mp_.I == 820) & (np.isclose(mp_.phi, ph))]
        put(f"map_eff_phi{int(ph * 100)}_I820", float(s.eff.mean()), 2)
    sv = SIM / "shapley_sem_vias.csv"
    if sv.exists():
        d = pd.read_csv(sv)
        r_ = d[(d.case == "no_pathways") & (d.U == 3.0)].iloc[0]
        put("sh_block_nopath_I820_U3", r_.sw_vapour, 2); put("sh_S_nopath_I820_U3", r_.S_total, 2)
    s = mp_[(mp_.I == 820) & (np.isclose(mp_.phi, 0.30))]
    put("map_S_phi30_U0", float(s[s.U == 0].S.iloc[0]), 2); put("map_S_phi30_U4", float(s[s.U == 4].S.iloc[0]), 2)
    if "Tp" in mp_:
        s = mp_[(mp_.I == 820) & (np.isclose(mp_.phi, 0.75))]
        put("map_Tp_U0", float(s[s.U == 0].Tp.iloc[0]), 0); put("map_Tp_U4", float(s[s.U == 4].Tp.iloc[0]), 0)
        put("map_dTs_U0", float((s.Ts_cov - s.Ts_open)[s.U == 0].iloc[0]), 1)
        put("map_dTs_U4", float((s.Ts_cov - s.Ts_open)[s.U == 4].iloc[0]), 1)
    so = pd.read_csv(SIM / "sobol.csv")
    z = np.load(SIM / "sobol_amostras.npz")
    put("sobol_N", int(len(z["X"]) // (2 + len(z["names"]))))
    put("sobol_runs", int(len(z["X"])))
    for _, r in so.iterrows():
        o = "S" if r.output == "S" else "dE"
        for c in ("S1", "ST", "ST_lo", "ST_hi", "S1_lo", "S1_hi"):
            put(f"sobol_{o}_{r.input}_{c}", r[c], 2)
    es = pd.read_csv(SIM / "transferencia_espectral.csv")
    for case in ("lamp", "solar", "solar_white"):
        s = es[(es.case == case) & es.S_total.notna()]
        put(f"spec_{case}_S_min", s.S_total.min(), 2); put(f"spec_{case}_S_max", s.S_total.max(), 2)
        for k, lab in MECH.items():
            put(f"spec_{case}_{lab}_mean", s[k].mean(), 2)
            u3 = s[np.isclose(s.U, 3.0)]
            if len(u3):
                put(f"spec_{case}_{lab}_U3", float(u3[k].iloc[0]), 2)
        c = es[(es.case == case) & es.S_curve.notna() & (es.U == 3.0)]
        for ph in (0.25, 0.5, 0.75):
            v = c[np.isclose(c.phi, ph)].S_curve
            if len(v):
                put(f"spec_{case}_S_phi{int(ph * 100)}", float(v.iloc[0]), 2)
    pj = pd.read_csv(SIM / "projeto_aberturas.csv")
    for U in (0.0, 3.0):
        s = pj[pj.U == U]
        put(f"proj_S_min_U{U:g}", s.S.min(), 2); put(f"proj_S_max_U{U:g}", s.S.max(), 2)
        g = lambda r, R_: float(s[(s.r_open_mm == r) & (s.R_rec_mm == R_)].S.iloc[0])
        put(f"proj_S_r2_R0_U{U:g}", g(2, 0), 2); put(f"proj_S_r50_R0_U{U:g}", g(50, 0), 2)
        put(f"proj_S_r30_R5_U{U:g}", g(30, 5), 2); put(f"proj_S_r30_R10_U{U:g}", g(30, 10), 2)
        put(f"proj_S_r30_R0_U{U:g}", g(30, 0), 2); put(f"proj_S_r5_R10_U{U:g}", g(5, 10), 2)
    pc = pd.read_csv(SIM / "projeto_calibrado.csv")
    for _, r in pc.iterrows():
        put(f"proj_S_cal_U{r.U:g}", r.S, 2)
    put("proj_r_cal_mm", float(pc.r_open_mm.iloc[0]), 0); put("proj_Rrec_cal_mm", float(pc.R_rec_mm.iloc[0]), 1)
    rc = pd.read_csv(SIM / "recesso.csv")
    for I in (300.0, 820.0):
        for U in (0.0, 3.0):
            s = rc[(rc.I == I) & (rc.U == U)].set_index("R_rec_mm").S
            for R_ in (0.0, 5.0, 10.0, 30.0):
                put(f"rec_S_I{I:.0f}_U{U:g}_R{R_:g}", float(s.loc[R_]), 2)
    # geometric gap width of a regular array of hexagons (across-flats 0.1905 m) at phi = 0.741
    put("gap_width_mm", 1000 * 0.110 * np.sqrt(3) * (0.741 ** -0.5 - 1), 0)
    sj = json.load(open(SIM / "transferencia_espectral.json"))
    for k in ("f_s", "f_m", "f_d", "l_m", "l_d"):
        put(f"solar_{k}", sj["solar_fit"][k], 3)
    if (SIM / "incerteza_posterior.csv").exists():
        up = pd.read_csv(SIM / "incerteza_posterior.csv")
        for U in (0.0, 3.0):
            for ph in (0.3, 0.5, 0.75, 0.9):
                v = up[(up.U == U) & np.isclose(up.phi, ph)].S
                if len(v):
                    a, lo, hi = np.percentile(v, [50, 2.5, 97.5])
                    tag = f"U{U:g}_phi{int(round(ph * 100))}"
                    put(f"unc_S_{tag}", a, 2); put(f"unc_S_{tag}_lo", lo, 2); put(f"unc_S_{tag}_hi", hi, 2)


def energy():
    df = pd.read_csv(TAB / "balanco_energia_corridas.csv")
    for c in (0, 1, 2):
        s = df[df.cfg == c]
        put(f"en_LE_frac_cfg{c}_min", float((-s.LE / s.Rabs_water).min()), 2)
        put(f"en_LE_frac_cfg{c}_max", float((-s.LE / s.Rabs_water).max()), 2)
        put(f"en_store_frac_cfg{c}_min", float((s.storage / s.Rabs_water).min()), 2)
        put(f"en_store_frac_cfg{c}_max", float((s.storage / s.Rabs_water).max()), 2)
        put(f"en_Rabs_cfg{c}", float(s.Rabs_water.mean()), 0)
    bow = df.H_sens / df.LE
    put("en_bowen_min", float(bow.min()), 2); put("en_bowen_max", float(bow.max()), 2)
    calm = df[df.U < 0.5]
    put("en_LW_frac_calm", float((-calm.LW_net / calm.Rabs_water).mean()), 2)
    put("en_wall_frac_max", float((-df.wall / df.Rabs_water).max()), 2)
    put("en_escape_frac", float((df.escape_up / (df.Rabs_water + df.escape_up)).mean()), 3)


def validation():
    f = SAI / "calibracao" / "validacao_split.csv"
    if not f.exists():
        return
    v = pd.read_csv(f)
    for (seg, stream, fit), r in v.set_index(["segment", "stream", "fit"]).iterrows():
        put(f"val_{seg}_{stream}_{fit}", r.rmse, 3)
    t = v[v.stream.str.startswith("Tw")]
    a = t[t.fit == "training_mode"].rmse.mean(); b = t[t.fit == "full_mode"].rmse.mean()
    put("val_T_rmse_heldout", a, 3); put("val_T_rmse_insample", b, 3)
    put("val_T_ratio", a / b, 2)
    lv = v[(v.stream == "level_b_qc")]
    if "E_model_mm" in lv:
        diffs = []
        for seg in lv.segment.unique():
            s = lv[lv.segment == seg].set_index("fit")
            put(f"val_{seg}_E_heldout", s.loc["training_mode", "E_model_mm"], 1)
            put(f"val_{seg}_E_full", s.loc["full_mode", "E_model_mm"], 1)
            diffs.append(100 * abs(s.loc["training_mode", "E_model_mm"] / s.loc["full_mode", "E_model_mm"] - 1))
        put("val_E_diff_pct", max(diffs), 1)


def derived():
    """Quantities quoted in the text that combine collected values."""
    g = lambda k: R.get(k)
    if g("c2_ody_gain"):
        put("c2_ody_over_pct", 100 * (g("c2_ody_gain") - 1), 0)
        put("c2_ody_10K", abs(g("c2_ody_ca")) * 10, 1)
    if g("c1_qdy_gain") and g("c2_ody_gain"):
        lo, hi = sorted([100 * (g("c2_ody_gain") - 1), 100 * (g("c1_qdy_gain") - 1)])
        put("qdy_over_range", f"{lo:.0f}–{hi:.0f}")
    m = json.load(open(CAL / "map_c2.json"))
    put("w_Tw_04", m["weights_used"]["Tw_04"], 2)
    # how far the tempering weights were from a fixed point after the last round
    ch = [abs(m["weights_next"][k] / m["weights_used"][k] - 1) for k in m["weights_used"] if m["weights_used"][k] > 0]
    put("c2_w_change_max_pct", 100 * max(ch), 0)
    if g("num_closure") is not None:
        e = int(np.ceil(np.log10(max(g("num_closure"), 1e-12))))
        put("num_closure_txt", f"$10^{{{e}}}$")
    for k in ("f_s", "f_m", "f_d"):
        if g(f"post_{k}") is not None:
            put(f"post_{k}_pct", 100 * g(f"post_{k}"), 0)
        if g(f"solar_{k}") is not None:
            put(f"solar_{k}_pct", 100 * g(f"solar_{k}"), 0)
    if g("post_f_s") is not None:
        fs = 100 * g("post_f_s")
        put("f_s_vs_prior_txt", "The near-surface partition agrees with the spectral prior" if fs <= 49.5 else
            "The skin share lies above the upper end of the spectral range of 35–49 %, as for a cooler effective "
            "filament than assumed")
    # the companion paper (state-space analysis of the same records)
    a2 = HERE.parents[1] / "ARTIGO_2_INFERENCIA_ESTOCASTICA" / "saidas" / "resultados_artigo2_texto.json"
    if a2.exists():
        t2 = json.load(open(a2, encoding="utf-8"))
        for k_out, k_in in (("a2_g_a", "c2_g_a"), ("a2_c_a", "c2_c_a"), ("a2_c_b", "c2_c_b"), ("a2_c1_g_a", "c1_g_a")):
            if k_in in t2:
                TXT[k_out] = t2[k_in]
    if g("solar_l_m") is not None:
        put("solar_l_m_mm", 1000 * g("solar_l_m"), 0); put("solar_l_d", g("solar_l_d"), 1)
    if g("en_escape_frac") is not None:
        put("en_escape_frac_pct", 100 * g("en_escape_frac"), 1)
    if g("post_C_p") is not None:
        r2 = lambda v: f"{float(f'{v:.2g}'):.0f}"
        put("post_C_p_txt", f"{r2(g('post_C_p'))} ({r2(g('post_C_p_lo'))}–{r2(g('post_C_p_hi'))})")
    if g("c1_dTs_calm_min") is not None:
        put("c1_dTs_calm_min_abs", abs(g("c1_dTs_calm_min")), 1)
    if g("sh_single_block_I820_U3") is not None:
        put("sh_single_block_resid_pct", 100 * (1 - g("sh_single_block_I820_U3")), 0)
    for k in ("sh_block_frac_phi25", "sh_block_frac_phi90"):
        if g(k) is not None:
            put(k + "_pct", 100 * g(k), 0)
    if g("post_c_n") is not None:
        put("post_c_n_dev_pct", 100 * (1 - g("post_c_n")), 0)
    if g("post_A_tot") is not None:
        put("pyr_frac_pct", 100 / g("post_A_tot"), 0)
        put("rail_ratio", g("post_A_tot_upper") / g("post_A_tot_lower"), 1)
    if g("c2_E_ratio_fan3_calm") is not None:
        put("c2_E_ratio_fan3_calm_pct", 100 * (g("c2_E_ratio_fan3_calm") - 1), 0)
    mp_ = pd.read_csv(SIM / "mapa_supressao.csv")
    w = mp_.pivot_table(index=["I", "phi"], columns="U", values="S")
    put("map_dS_wind_max", float((w.max(axis=1) - w.min(axis=1)).max()), 2)
    if g("sh_S_I820_U3") is not None:
        put("design_eff", g("sh_S_I820_U3") / 0.74, 2)
    mi = CAL / "mode_c2_posterior.json"
    if mi.exists():
        put("c2_max_iat", float(json.load(open(mi))["max_iat"]), 0)
    so = pd.read_csv(SIM / "sobol.csv")
    s = so[so.output == "S"].set_index("input")
    put("sobol_S_small_max", float(s.drop(index=[k for k in ("phi", "R_rec", "C_u", "k_g") if k in s.index]).ST.max()), 2)
    put("sobol_S_sumS1", float(s.S1.sum()), 2)


def main():
    steps = [numerics, fit_c2, profiles, posterior_c2, posterior_c1, convergence_text, energy, fit_c1, scenarios,
             validation]
    for fn in steps:
        try:
            fn()
            print(f"{fn.__name__}: ok", flush=True)
        except FileNotFoundError as e:
            print(f"{fn.__name__}: missing {e.filename}", flush=True)
    derived()
    json.dump(R, open(SAI / "resultados_artigo1.json", "w"), indent=1, default=str)
    json.dump(TXT, open(SAI / "resultados_artigo1_texto.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(R)} values -> resultados_artigo1.json")


if __name__ == "__main__":
    main()
