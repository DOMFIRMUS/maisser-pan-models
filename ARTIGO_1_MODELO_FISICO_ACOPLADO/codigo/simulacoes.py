# -*- coding: utf-8 -*-
"""
Scenario simulations with the calibrated pan model (Article 1).

  * standard laboratory day: lamps on 08:00-17:00 at a given irradiance index,
    fan on with the lamps, room air warming 2.6 K under the lamps (C2 mean for
    the full array), constant air vapour pressure 2.45 kPa; three days are
    simulated and the last one analysed. Irradiance index, wind speed and
    cover fraction may differ between walkers (per-walker multipliers), so a
    whole design is integrated at once;
  * suppression S = 1 - E_cover / E_open over the analysed day, for the same
    forcing and the same initial state (includes the thermal feedback);
  * Shapley attribution of S to four cover mechanisms (aerodynamic blocking,
    shading, insulation of the covered water, plate-water conduction), from
    the 2^4 simulations with each subset of mechanisms switched on;
  * variance-based global sensitivity (Sobol first-order and total indices,
    Saltelli et al. 2010 / Jansen 1999 estimators, bootstrap intervals).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import itertools
import json
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modelo_fisico as mf      # noqa: E402

DT = 300.0
I_REF = 820.0                   # SP-LITE index of the full array (C2)
MECHS = ["sw_vapour", "sw_shade", "sw_insul", "sw_cond"]
MECH_LABEL = {"sw_vapour": "aerodynamic blocking", "sw_shade": "shading",
              "sw_insul": "insulation", "sw_cond": "plate-water conduction"}


def base_params(map_c2, map_c1, version="V2"):
    """Physics from the C2 mode/posterior draw and cover from C1."""
    P = {k: float(v[0]) for k, v in mf.default_params(1).items()}
    P.update({k: v for k, v in map_c2["params"].items()})
    for k in ("a_p", "G_pw", "C_p", "r_open", "tau_p", "k_g"):
        if k in map_c1["params"]:
            P[k] = map_c1["params"][k]
    for k in ("R_rec", "C_u"):
        if f"{k}_{version}" in map_c1["params"]:
            P[k] = map_c1["params"][f"{k}_{version}"]
    return P


def standard_day(n_days=3, on=(8.0, 17.0), Ta0=28.8, dTa=2.6, ea=2450.0, covered=True):
    """Shared forcing: lamps at I_REF (cfg 1, per-walker multiplier m_1), fan at
    1 m/s (per-walker scale s_U), cover flag phi = 0.5 (per-walker phi_w)."""
    n = int(n_days * 86400 / DT)
    t = np.arange(n) * DT
    hod = (t / 3600.0) % 24
    lamp = (hod >= on[0]) & (hod < on[1])
    Ta = np.empty(n); x = 0.0
    tau = 1.5 * 3600
    for i in range(n):
        x += ((dTa if lamp[i] else 0.0) - x) * DT / tau
        Ta[i] = Ta0 + x
    return mf.Forcing(t=t, I=np.where(lamp, I_REF, 0.0), U=np.where(lamp, 1.0, 0.0), Ta=Ta,
                      ea=np.full(n, ea), phi=np.full(n, 0.5 if covered else 0.0),
                      cfg=np.where(lamp, 1, 0).astype(int)), lamp


def broadcast(params: dict, W: int):
    return {k: np.broadcast_to(np.asarray(v, float), (W,)).copy() for k, v in params.items()}


PHI_REF = 0.741                 # cover fraction of the calibrated (V2) array


def gap_scale(phi, phi_ref=PHI_REF):
    """Ratio of the gap width at phi to that at phi_ref for a regular array of
    plates of fixed size: w ~ L (phi^-1/2 - 1)."""
    return (phi ** -0.5 - 1.0) / (phi_ref ** -0.5 - 1.0)


def run_design(base, I_idx, U, phi, switches=None, n_days=3, T0=29.0, H0=0.22, extra=None,
               return_fields=False):
    """Suppression of each walker (vectors I_idx, U, phi of equal length W).

    r_open (from base or extra) is the opening size at PHI_REF; it is scaled
    with the gap width of a regular array of the same plates at each phi."""
    I_idx, U, phi = (np.atleast_1d(np.asarray(a, float)) for a in (I_idx, U, phi))
    W = max(len(I_idx), len(U), len(phi))
    P = broadcast(base, W)
    P["m_1"] = np.broadcast_to(I_idx / I_REF, (W,)).copy()
    P["s_U"] = np.broadcast_to(U, (W,)).copy()
    P["phi_w"] = np.broadcast_to(phi, (W,)).copy()
    if extra:
        for k, v in extra.items():
            P[k] = np.broadcast_to(np.asarray(v, float), (W,)).copy()
    gs = gap_scale(np.broadcast_to(phi, (W,)))
    P["r_open"] = P["r_open"] * gs
    # the boundary-layer restart over a gap matters less as the gap widens: mean transfer
    # over a fresh strip scales as (width)^-1/2 in laminar flow
    if "k_g" in P:
        P["k_g"] = 1.0 + (P["k_g"] - 1.0) * np.power(np.maximum(gs, 1e-3), -0.5)
    if switches is not None:
        for k, v in switches.items():
            P[k] = np.broadcast_to(np.asarray(v, float), (W,)).copy()
    Fc, lamp = standard_day(n_days, covered=True)
    Fo, _ = standard_day(n_days, covered=False)
    oc = mf.simulate(Fc, P, T0, H0, dt=DT)
    Po = {k: v for k, v in P.items() if not k.startswith("sw_")}
    oo = mf.simulate(Fo, Po, T0, H0, dt=DT)
    per_day = int(86400 / DT)
    Ec = oc.E[-per_day:].sum(axis=0) * DT / 997.0 * 1000.0
    Eo = oo.E[-per_day:].sum(axis=0) * DT / 997.0 * 1000.0
    out = dict(S=1 - Ec / Eo, E_cov=Ec, E_open=Eo)
    if return_fields:
        lam = lamp[-per_day:]
        out.update(Ts_cov=oc.Ts[-per_day:][lam].mean(0), Ts_open=oo.Ts[-per_day:][lam].mean(0),
                   Tp=oc.Tp[-per_day:][lam].mean(0), Tb_cov=oc.T[-per_day:, :, -1].mean(0),
                   Tb_open=oo.T[-per_day:, :, -1].mean(0))
    return out


def shapley_design(base, I_idx, U, phi, n_days=3, extra=None):
    """Shapley values of the four mechanisms for vectors of scenarios.

    Returns dict mechanism -> (n_scen,) array and S_total (n_scen,)."""
    I_idx, U, phi = (np.atleast_1d(np.asarray(a, float)) for a in (I_idx, U, phi))
    ns = len(phi)
    combos = list(itertools.product([0, 1], repeat=len(MECHS)))
    nc = len(combos)
    rep = lambda a: np.repeat(a, nc)
    sw = {m: np.tile(np.array([c[j] for c in combos], float), ns) for j, m in enumerate(MECHS)}
    ext = {k: np.repeat(np.broadcast_to(np.asarray(v, float), (ns,)), nc) for k, v in (extra or {}).items()}
    r = run_design(base, rep(I_idx), rep(U), rep(phi), switches=sw, n_days=n_days, extra=ext or None)
    Sm = r["S"].reshape(ns, nc)
    v = {c: Sm[:, i] for i, c in enumerate(combos)}
    n = len(MECHS)
    res = {}
    for j, m in enumerate(MECHS):
        tot = np.zeros(ns)
        for c in combos:
            if c[j] == 1:
                continue
            k = sum(c)
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            c1 = list(c); c1[j] = 1
            tot += w * (v[tuple(c1)] - v[c])
        res[m] = tot
    res["S_total"] = v[(1, 1, 1, 1)]
    res["single"] = {m: v[tuple(1 if i == j else 0 for i in range(n))] for j, m in enumerate(MECHS)}
    return res


# ---------------------------------------------------------------- Sobol
SOBOL_INPUTS = [
    ("phi", 0.30, 0.95), ("a_p", 0.05, 0.90), ("tau_p", 0.0, 0.6), ("G_pw", 0.5, 30.0), ("r_open", 0.002, 0.10),
    ("k_g", 1.0, 3.0), ("C_u", 0.0, 2e-3),
    ("R_rec", 0.0, 0.03), ("U", 0.0, 4.0), ("I", 200.0, 1000.0),
]


def saltelli_matrices(n, bounds, rng):
    """A, B and the k matrices AB_i (column i of A replaced by B), uniform on bounds."""
    k = len(bounds)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    A = lo + (hi - lo) * rng.random((n, k))
    B = lo + (hi - lo) * rng.random((n, k))
    AB = np.repeat(A[None], k, axis=0)
    for i in range(k):
        AB[i, :, i] = B[:, i]
    return A, B, AB


def sobol_indices(fA, fB, fAB, n_boot=500, rng=None):
    """First-order (Saltelli 2010) and total (Jansen 1999) indices with bootstrap."""
    rng = rng or np.random.default_rng(0)
    k, n = fAB.shape
    # centring leaves both estimators unbiased and reduces the variance of S1
    c = np.mean(np.concatenate([fA, fB]))
    fA, fB, fAB = fA - c, fB - c, fAB - c

    def est(ix):
        a, b, ab = fA[ix], fB[ix], fAB[:, ix]
        var = np.var(np.concatenate([a, b]), ddof=1)
        S1 = np.mean(b[None] * (ab - a[None]), axis=1) / var
        ST = 0.5 * np.mean((a[None] - ab) ** 2, axis=1) / var
        return S1, ST
    S1, ST = est(np.arange(n))
    boots = [est(rng.integers(0, n, n)) for _ in range(n_boot)]
    B1 = np.array([b[0] for b in boots]); BT = np.array([b[1] for b in boots])
    return dict(S1=S1, ST=ST, S1_lo=np.percentile(B1, 2.5, 0), S1_hi=np.percentile(B1, 97.5, 0),
                ST_lo=np.percentile(BT, 2.5, 0), ST_hi=np.percentile(BT, 97.5, 0))


def sobol_eval(base, X, batch=512, n_days=3):
    """Evaluate S and saved evaporation for rows of X (columns as SOBOL_INPUTS)."""
    names = [s[0] for s in SOBOL_INPUTS]
    S = np.empty(len(X)); saved = np.empty(len(X))
    for i0 in range(0, len(X), batch):
        x = X[i0:i0 + batch]
        col = {nm: x[:, j] for j, nm in enumerate(names)}
        extra = {k: col[k] for k in ("a_p", "tau_p", "G_pw", "r_open", "R_rec", "k_g", "C_u")}
        r = run_design(base, col["I"], col["U"], col["phi"], extra=extra, n_days=n_days)
        S[i0:i0 + batch] = r["S"]
        saved[i0:i0 + batch] = r["E_open"] - r["E_cov"]
    return S, saved
