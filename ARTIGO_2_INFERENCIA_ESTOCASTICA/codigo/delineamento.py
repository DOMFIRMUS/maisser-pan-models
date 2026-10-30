# -*- coding: utf-8 -*-
"""
Bayesian optimal design of the next cover experiments (Article 2).

Question: which cover fractions phi and fan settings U should the next runs
use to learn how suppression depends on coverage? Three hypotheses, each with
one parameter for calm air and one for the strong fan (interpolated in between
through the fan index f(U), the position of the mixed-convection transfer
velocity between calm air and the strong fan), calibrated to the current
evidence (hierarchical posterior of the V2 plates at phi = 0.74-0.77 with the
fan on and off):
  M1  constant efficiency:  S = e(U) phi
  M2  open-area power law:  S = 1 - (1 - phi)^n(U)
  M3  gaps of a regular array of the current plates (width 31 mm at
      phi = 0.741, scaled with phi^-1/2 - 1) evaporating with the per-area
      enhancement of Schlunder (1988) / Aminzadeh et al. (2018), plus a leak
      u(U) under and around the plates (the pathways of the companion paper):
      S = 1 - theta psi - phi u(U),  theta = 1 - phi,
      psi = 1 / (theta + (r/delta) sqrt(theta/pi) (sqrt(pi/(4 theta)) - 1)),
      bounded by 1 (gaps wider than the diffusive layer delta) and by 2 delta/r
      (the limit of the relation for isolated openings), delta(U) = D / C_e(U).
A future run at (phi, U) yields y = log(1 - S) + u + e, with the between-day
spread tau and the measurement error sigma of the current analysis.

Utility: mutual information I(M; y | d) between the model indicator and the
data of a design, by nested Monte Carlo, with equal prior probabilities
(all three hypotheses reproduce the current estimates). A greedy sequential
search picks a batch of runs.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "saidas"
D_VAP = 2.6e-5


A1 = HERE.parents[1] / "ARTIGO_1_MODELO_FISICO_ACOPLADO" / "saidas" / "resultados_artigo1.json"
PHI_REF = 0.741
GAP_REF = 0.031          # gap width of a regular array of the plates at PHI_REF (m)
U_ON = 3.2               # strong fan of the current runs (m s-1)
N_MC = 800               # outer and inner Monte Carlo samples of the EIG estimator


def _a1_transfer():
    """Free-convection transfer velocity and forced-convection law of the
    companion paper's C2 posterior (median), with fall-back values."""
    try:
        r = json.load(open(A1))
        return r["post_Ce_U0"] * 1e-3, r["post_c_f"] * 1e-3, r["post_n_f"]
    except (OSError, KeyError):
        return 2.5e-3, 6.1e-3, 0.34


C_N, C_F, N_F = _a1_transfer()


def transfer_velocity(U, c_n_term=None, c_f=None, n_f=None):
    """Mixed-convection transfer velocity (m s-1): free-convection floor plus
    forced term (Churchill-Usagi cube sum), with the C2 calibration values."""
    c_n_term = C_N if c_n_term is None else c_n_term
    c_f = C_F if c_f is None else c_f
    n_f = N_F if n_f is None else n_f
    U = np.asarray(U, float)
    return np.cbrt(c_n_term ** 3 + (c_f * np.power(np.maximum(U, 0), n_f)) ** 3)


def fan_index(U):
    """0 in calm air, 1 with the strong fan, in between through C_e(U)."""
    c0, c1 = transfer_velocity(0.0), transfer_velocity(U_ON)
    return (transfer_velocity(U) - c0) / (c1 - c0)


def S_model(model, phi, U, theta_par):
    """Suppression under each hypothesis (parameters *_off for calm air and
    *_on for the strong fan, interpolated with fan_index)."""
    phi = np.asarray(phi, float)
    f = fan_index(U)

    def lerp(name):
        return theta_par[name + "_off"] + (theta_par[name + "_on"] - theta_par[name + "_off"]) * f
    if model == "M1":
        return lerp("e") * phi
    if model == "M2":
        return 1 - (1 - phi) ** lerp("n")
    th = np.clip(1 - phi, 1e-6, 1)
    delta = D_VAP / transfer_velocity(U)
    r = GAP_REF * (np.clip(phi, 1e-3, 1) ** -0.5 - 1) / (PHI_REF ** -0.5 - 1)
    ratio = r / delta
    g = np.sqrt(np.pi / (4 * th))
    psi = 1.0 / (th + ratio * np.sqrt(th / np.pi) * np.maximum(g - 1, 0))
    psi = np.clip(psi, 1.0, np.maximum(1.0, 2 / ratio))
    return 1 - th * psi - phi * lerp("u")


def calibrate_priors(S_obs_mean, S_obs_sd, phi_obs=0.755, U_obs=(3.2, 0.0), n=4000, seed=0):
    """Parameter draws of each model consistent with the observed suppression,
    and the posterior model probabilities given that evidence.

    Importance resampling from wide priors with a Gaussian likelihood centred
    on the hierarchical estimate at phi_obs (fan on and fan off); the evidence
    of each model is the prior mean of that likelihood."""
    rng = np.random.default_rng(seed)
    draws, logev = {}, {}
    S_on, S_off = S_obs_mean
    sd_on, sd_off = S_obs_sd
    N = 50 * n
    cand = {"M1": {"e_off": rng.uniform(0.3, 1.2, N), "e_on": rng.uniform(0.3, 1.2, N)},
            "M2": {"n_off": np.exp(rng.uniform(np.log(0.2), np.log(3.0), N)),
                   "n_on": np.exp(rng.uniform(np.log(0.2), np.log(3.0), N))},
            "M3": {"u_off": rng.uniform(0.0, 0.6, N), "u_on": rng.uniform(0.0, 0.6, N)}}
    for m, par in cand.items():
        s_on = S_model(m, phi_obs, U_obs[0], par)
        s_off = S_model(m, phi_obs, U_obs[1], par)
        lw = -0.5 * ((s_on - S_on) / sd_on) ** 2 - 0.5 * ((s_off - S_off) / sd_off) ** 2
        logev[m] = float(lw.max() + np.log(np.mean(np.exp(lw - lw.max()))))
        w = np.exp(lw - lw.max()); w /= w.sum()
        k = rng.choice(len(w), size=n, p=w)
        draws[m] = {key: v[k] for key, v in par.items()}
    le = np.array(list(logev.values()))
    pm = np.exp(le - le.max()); pm /= pm.sum()
    return draws, dict(zip(logev, pm.tolist()))


def eig_discrimination(design, draws, tau, sigma, pm=None, n_out=400, n_in=400, seed=1):
    """Mutual information between the model indicator and the data of a design
    (list of (phi, U) runs), by nested Monte Carlo, with prior model
    probabilities pm (equal if None)."""
    rng = np.random.default_rng(seed)
    models = list(draws)
    pm = {m: 1.0 / len(models) for m in models} if pm is None else pm
    lpm = np.array([np.log(pm[m]) for m in models])
    sd = np.sqrt(tau ** 2 + sigma ** 2)
    phis = np.array([d[0] for d in design]); Us = np.array([d[1] for d in design])
    mean_in = {}
    for m in models:
        k = rng.integers(0, len(next(iter(draws[m].values()))), n_in)
        par = {key: v[k][:, None] for key, v in draws[m].items()}
        mean_in[m] = np.log(np.clip(1 - S_model(m, phis[None, :], Us[None, :], par), 1e-6, None))
    total = 0.0
    for m in models:
        k = rng.integers(0, len(next(iter(draws[m].values()))), n_out)
        par = {key: v[k][:, None] for key, v in draws[m].items()}
        mu = np.log(np.clip(1 - S_model(m, phis[None, :], Us[None, :], par), 1e-6, None))
        y = mu + sd * rng.standard_normal(mu.shape)                       # (n_out, K)
        logp = {}
        for mm in models:
            r = (y[:, None, :] - mean_in[mm][None, :, :]) / sd                # (n_out, n_in, K)
            ll = -0.5 * np.sum(r ** 2, axis=2)
            mx = ll.max(axis=1, keepdims=True)
            logp[mm] = (mx[:, 0] + np.log(np.mean(np.exp(ll - mx), axis=1)))
        lse = np.logaddexp.reduce(np.array([logp[mm] for mm in models]) + lpm[:, None], axis=0)
        total += pm[m] * np.mean(logp[m] - lse)
    return float(total)


def main(batch=4):
    hier = json.load(open(OUT / "hierarquico_resumo.json"))
    cells = pd.read_csv(OUT / "hierarquico_celulas.csv")
    on = cells[cells.celula == "V2, vento forte"].iloc[0]
    off = cells[cells.celula == "V2, ventilador desligado"].iloc[0]
    S_mean = (on.S_media, off.S_media)
    S_sd = ((on.S_hi - on.S_lo) / 3.92, (off.S_hi - off.S_lo) / 3.92)
    tau = hier["tau"][0]
    sigma = float(pd.read_csv(OUT / "hierarquico_janelas.csv").se_y.median())
    draws, pm_evidence = calibrate_priors(S_mean, S_sd)
    # every hypothesis reproduces both current estimates, so their evidence ratios
    # reflect only the widths of the parameter priors: equal prior probabilities
    pm = {m: 1.0 / len(draws) for m in draws}
    phis = np.round(np.arange(0.1, 0.91, 0.1), 2)
    Us = [0.0, 1.5, 3.2]
    single = []
    for U in Us:
        for ph in phis:
            # four Monte Carlo replicates per design: mean and standard error of the estimate
            v = [eig_discrimination([(ph, U)], draws, tau, sigma, pm, n_out=N_MC, n_in=N_MC, seed=1 + k)
                 for k in range(4)]
            single.append(dict(phi=ph, U=U, eig=float(np.mean(v)), eig_se=float(np.std(v, ddof=1) / 2)))
    single = pd.DataFrame(single)
    single.to_csv(OUT / "delineamento_eig_unico.csv", index=False)
    chosen, cum_eig = [], []
    for b in range(batch):
        best = None
        for U in Us:
            for ph in phis:
                v = eig_discrimination(chosen + [(ph, U)], draws, tau, sigma, pm, n_out=N_MC, n_in=N_MC,
                                       seed=10 + b)
                if best is None or v > best[0]:
                    best = (v, (float(ph), float(U)))
        chosen.append(best[1]); cum_eig.append(float(best[0]))
        print(f"run {b + 1}: phi = {best[1][0]:.1f}, U = {best[1][1]:.1f} m/s, cumulative EIG = {best[0]:.3f} nats",
              flush=True)
    # model predictions for the figure
    grid = np.linspace(0.02, 0.95, 60)
    pred = []
    for m, par in draws.items():
        for U in Us:
            k = np.random.default_rng(5).integers(0, len(next(iter(par.values()))), 2000)
            pp = {key: v[k][:, None] for key, v in par.items()}
            S = S_model(m, grid[None, :], U, pp)
            for i, g in enumerate(grid):
                pred.append(dict(model=m, U=U, phi=g, S_med=np.median(S[:, i]),
                                 S_lo=np.percentile(S[:, i], 5), S_hi=np.percentile(S[:, i], 95)))
    pd.DataFrame(pred).to_csv(OUT / "delineamento_previsoes_modelos.csv", index=False)
    # references: four runs at the current coverage (two with each fan setting)
    phi0 = 0.755
    kw = dict(n_out=N_MC, n_in=N_MC)
    ref = dict(current_4=eig_discrimination([(phi0, 0.0), (phi0, U_ON), (phi0, 0.0), (phi0, U_ON)],
                                            draws, tau, sigma, pm, seed=99, **kw),
               current_1_off=eig_discrimination([(phi0, 0.0)], draws, tau, sigma, pm, seed=98, **kw),
               current_1_on=eig_discrimination([(phi0, U_ON)], draws, tau, sigma, pm, seed=97, **kw),
               batch_mc_check=[eig_discrimination(chosen, draws, tau, sigma, pm, seed=200 + k, **kw)
                               for k in range(3)])
    fit_now = {m: {"S_on": float(np.median(S_model(m, phi0, U_ON, par))),
                   "S_off": float(np.median(S_model(m, phi0, 0.0, par)))} for m, par in draws.items()}
    json.dump(dict(tau=tau, sigma=sigma, S_obs=S_mean, S_obs_sd=S_sd, batch=chosen, batch_eig=cum_eig,
                   p_model=pm, p_model_evidence=pm_evidence, reference_eig=ref, fit_current=fit_now,
                   draws_summary={m: {k: [float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
                                      for k, v in par.items()} for m, par in draws.items()}),
              open(OUT / "delineamento_resumo.json", "w"), indent=2)
    print(single.pivot(index="phi", columns="U", values="eig").round(3))
    return chosen


if __name__ == "__main__":
    main()
