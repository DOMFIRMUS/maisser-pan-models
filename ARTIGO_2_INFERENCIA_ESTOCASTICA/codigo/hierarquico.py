# -*- coding: utf-8 -*-
"""
Hierarchical Bayesian model of the window suppression (Article 2).

Data: y_j = log(kappa_j / kappa_0) = log(1 - S_j) for the covered rate windows
of campaign C1, with the covariance Sigma_y of the measurement error from the
state-space fit (delta method on the kappa estimates; the common kappa_0 makes
the windows correlated).

Model:  y = X beta + u + e,   u ~ N(0, tau^2 I),   e ~ N(0, Sigma_y)
        beta ~ N(b0, B0),     tau ~ half-normal(s_tau)
X: intercept (V1 plates, strong fan), V2 plates, weak fan (lower lamp setting),
fan off. Given tau the model is conjugate; p(tau | y) is evaluated exactly on a
grid and the joint posterior is sampled by composition.

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
sys.path.insert(0, str(HERE.parents[1] / "_comum"))
from tanque import config as cfg   # noqa: E402

OUT = HERE.parent / "saidas"
COLS = ["intercepto (V1, vento forte)", "V2", "vento fraco", "ventilador desligado"]


def load(path=OUT / "c1_supressao_direta.json"):
    r = json.load(open(path))
    groups = r["groups"]; mu = np.array(r["kappa_mean"]); C = np.array(r["kappa_cov"])
    meta = {w[0]: w for w in cfg.WINDOWS_C1}
    wins = [g for g in groups if g in meta]
    ref = r.get("reference", {})
    idx = [groups.index(w) for w in wins]
    iref = [groups.index(ref.get(meta[w][6], "uncovered")) if ref else groups.index("uncovered") for w in wins]
    y = np.log(mu[idx] / mu[iref])
    J = np.zeros((len(idx), len(mu)))
    for k, (j, i0) in enumerate(zip(idx, iref)):
        J[k, j] = 1 / mu[j]; J[k, i0] = -1 / mu[i0]
    Sy = J @ C @ J.T
    # average over the profile of the anemometer scale: weighted mean of log(1 - S)
    # across s_U, plus the full between-s_U covariance (the s_U shifts of different
    # windows are correlated, e.g. all calm windows move together)
    prof_f = OUT / "c1_supressao_perfil_sU.csv"
    if prof_f.exists():
        prof = pd.read_csv(prof_f)
        sus = sorted(prof.s_U.unique())
        Y = np.full((len(sus), len(wins)), np.nan)
        wts = np.zeros(len(sus))
        for i, s_ in enumerate(sus):
            sub = prof[prof.s_U == s_]
            wts[i] = float(sub.weight.iloc[0])
            for k, w in enumerate(wins):
                v = sub[(sub.grupo == w) & sub.S_mean.notna()].S_mean
                if len(v):
                    Y[i, k] = np.log(max(1 - float(v.iloc[0]), 1e-6))
        ok = np.isfinite(Y).all(axis=1)
        if ok.any():
            wk = wts[ok] / wts[ok].sum()
            ybar = wk @ Y[ok]
            D = Y[ok] - ybar
            Sy = Sy + (D * wk[:, None]).T @ D
            y = ybar
    X = np.array([[1.0, meta[w][4] == "V2", meta[w][6] == "weak", meta[w][6] == "calm"] for w in wins], float)
    info = pd.DataFrame({"win": wins, "version": [meta[w][4] for w in wins],
                         "n_plates": [meta[w][5] for w in wins], "regime": [meta[w][6] for w in wins],
                         "phi": [cfg.coverage_fraction(meta[w][5]) for w in wins],
                         "y": y, "se_y": np.sqrt(np.diag(Sy))})
    return y, Sy, X, info


def posterior(y, Sy, X, b0=None, B0=None, s_tau=0.5, n_grid=600, n_draw=20000, seed=3):
    p = X.shape[1]
    b0 = np.zeros(p) if b0 is None else b0
    B0 = np.eye(p) * 2.0 ** 2 if B0 is None else B0
    taus = np.linspace(1e-4, 4 * s_tau, n_grid)
    logp = np.empty(n_grid)
    for i, t in enumerate(taus):
        V = X @ B0 @ X.T + Sy + t ** 2 * np.eye(len(y))
        sign, ld = np.linalg.slogdet(V)
        r = y - X @ b0
        logp[i] = -0.5 * (ld + r @ np.linalg.solve(V, r)) - 0.5 * (t / s_tau) ** 2
    w = np.exp(logp - logp.max()); w /= w.sum()
    rng = np.random.default_rng(seed)
    tau_s = rng.choice(taus, size=n_draw, p=w) + (taus[1] - taus[0]) * (rng.random(n_draw) - 0.5)
    tau_s = np.abs(tau_s)
    B0i = np.linalg.inv(B0)
    betas = np.empty((n_draw, p)); us = np.empty((n_draw, len(y)))
    for k, t in enumerate(tau_s):
        R = Sy + t ** 2 * np.eye(len(y))
        Ri = np.linalg.inv(R)
        P = np.linalg.inv(X.T @ Ri @ X + B0i)
        m = P @ (X.T @ Ri @ y + B0i @ b0)
        b = rng.multivariate_normal(m, P)
        betas[k] = b
        # window effects given beta, tau: u | y ~ N(t^2 (Sy + t^2 I)^-1 (y - Xb), ...)
        G = t ** 2 * Ri
        mu_u = G @ (y - X @ b)
        Cu = t ** 2 * (np.eye(len(y)) - G)
        us[k] = rng.multivariate_normal(mu_u, (Cu + Cu.T) / 2 + 1e-12 * np.eye(len(y)))
    return dict(tau=tau_s, beta=betas, u=us, tau_grid=taus, tau_post=w)


def summarise(post, X, info, y):
    S = lambda logk: 1 - np.exp(logk)
    q = lambda a: (float(np.median(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    b = post["beta"]
    cells = {"V1, vento forte": [1, 0, 0, 0], "V1, vento fraco": [1, 0, 1, 0],
             "V2, vento forte": [1, 1, 0, 0], "V2, ventilador desligado": [1, 1, 0, 1]}
    rows = []
    for name, x in cells.items():
        lk = b @ np.array(x, float)
        pred = lk + post["tau"] * np.random.default_rng(1).standard_normal(len(lk))
        m, lo, hi = q(S(lk)); pm, plo, phi_ = q(S(pred))
        rows.append(dict(celula=name, S_media=m, S_lo=lo, S_hi=hi, S_pred=pm, S_pred_lo=plo, S_pred_hi=phi_))
    cells_df = pd.DataFrame(rows)
    # window-level shrinkage estimates
    th = post["beta"] @ X.T + post["u"]
    wrow = []
    for j, w in enumerate(info.win):
        m, lo, hi = q(S(th[:, j]))
        wrow.append(dict(win=w, S_bruto=float(S(y[j])), S_bruto_lo=float(S(y[j] + 1.96 * info.se_y[j])),
                         S_bruto_hi=float(S(y[j] - 1.96 * info.se_y[j])), S_hier=m, S_hier_lo=lo, S_hier_hi=hi))
    win_df = info.merge(pd.DataFrame(wrow), on="win")
    eff = {}
    for k, nm in enumerate(COLS):
        eff[nm] = q(b[:, k])
    probs = {"P(S_V2 > S_V1)": float(np.mean(b[:, 1] < 0)),
             "P(S_fan_off > S_strong)": float(np.mean(b[:, 3] < 0)),
             "P(S_weak > S_strong)": float(np.mean(b[:, 2] < 0))}
    tau = q(post["tau"])
    return cells_df, win_df, eff, probs, tau


def main():
    y, Sy, X, info = load()
    post = posterior(y, Sy, X)
    cells, wins, eff, probs, tau = summarise(post, X, info, y)
    pd.set_option("display.width", 200)
    print(wins.round(3).to_string())
    print(cells.round(3).to_string())
    print("effects (log scale):", {k: tuple(round(x, 3) for x in v) for k, v in eff.items()})
    print("probabilities:", probs, " tau:", tuple(round(x, 3) for x in tau))
    cells.to_csv(OUT / "hierarquico_celulas.csv", index=False)
    wins.to_csv(OUT / "hierarquico_janelas.csv", index=False)
    json.dump(dict(effects=eff, probabilities=probs, tau=tau, n_windows=len(y)),
              open(OUT / "hierarquico_resumo.json", "w"), indent=2)
    np.savez_compressed(OUT / "hierarquico_posterior.npz", tau=post["tau"], beta=post["beta"], u=post["u"])
    return post


if __name__ == "__main__":
    main()
