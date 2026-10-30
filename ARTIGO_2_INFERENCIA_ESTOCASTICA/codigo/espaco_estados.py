# -*- coding: utf-8 -*-
"""
Linear Gaussian state-space model of the pan level records with an exact
diffuse treatment of all time-invariant linear coefficients.

Stochastic state  s_t = (H_t, eta_t, d_t)
   H_t    water depth relative to the start of the current segment (mm)
   eta_t  stochastic deviation of the evaporation rate from its regression
          on physical drivers (mm h-1), AR(1): eta' = phi eta + w
   d_t    drift of the CS456 transducer (mm), random walk with slope delta
Fixed coefficients  b (flat prior, integrated out exactly):
   level of each segment on the CS456 scale, h_seg; offset of the ODY in each
   segment, u_seg; drift slope delta; thermal coefficients c_a (ODY), c_b
   (CS456); regression coefficients beta of the evaporation drivers.

Transition (Delta = step in hours)
   H'   = H - Delta (f_t' beta + eta)
   eta' = phi eta + w_eta,            w_eta ~ N(0, s_eta^2)
   d'   = d + Delta delta + w_d,      w_d   ~ N(0, s_d^2)
Observations (each may be missing)
   y_a = u_seg + g_a H + c_a x_t + v_a     ODY (gain g_a; x_t = T_sensor - 30)
   y_b = h_seg + H + d + c_b x_t + v_b     CS456 (factory pressure scale)

The augmented filter of de Jong (1991) carries the effect of b as a
regression matrix A_t, giving the diffuse log-likelihood
   log L = -1/2 [ sum log|F_t| + q - s' S^-1 s + log|S| ] + const
and the GLS estimate b_hat = S^-1 s with covariance S^-1.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Design:
    ya: np.ndarray          # (n,) ODY level (mm), NaN if missing
    yb: np.ndarray          # (n,) CS456 level (mm), NaN if missing (or absent)
    x: np.ndarray           # (n,) sensor temperature - 30 (K)
    F: np.ndarray           # (n, p) evaporation drivers (mm h-1 per unit beta)
    seg: np.ndarray         # (n,) integer segment index 0..S-1 (segments ordered)
    dt_h: float = 1.0 / 6.0
    has_b: bool = True      # second sensor present
    drift: bool = True      # CS456 drift states present
    xb: np.ndarray = None   # thermal regressor of sensor b (default: x; zeros = none)

    @property
    def S(self):
        return int(self.seg.max()) + 1

    def b_names(self, driver_names=None):
        S = self.S
        p = self.F.shape[1]
        names = [f"h_seg{i}" for i in range(S)] + [f"u_seg{i}" for i in range(S)]
        names += ["delta", "c_a", "c_b"] + [f"beta{j}" if driver_names is None else driver_names[j]
                                             for j in range(p)]
        return names


def _index(des: Design):
    S = des.S
    p = des.F.shape[1]
    ix = dict(h=np.arange(S), u=S + np.arange(S), delta=2 * S, ca=2 * S + 1, cb=2 * S + 2,
              beta=2 * S + 3 + np.arange(p))
    return ix, 2 * S + 3 + p


def kalman(des: Design, theta: dict, smooth=False, return_b=False, mask_segs=None, per_seg=False):
    """Augmented Kalman filter. theta: g_a, phi, s_eta, s_d, s_a, s_b.

    mask_segs: segment indices whose observations are ignored (the state is
    still propagated through them), for cross-validation.
    per_seg: also return the accumulators (S, s, q, log|F|, n) of each
    segment, from which conditional (held-out) scores are computed.
    """
    mask_segs = set() if mask_segs is None else set(mask_segs)
    n = len(des.ya)
    dt = des.dt_h
    ix, nb = _index(des)
    ga, phi = theta["g_a"], theta["phi"]
    se, sd = theta["s_eta"], theta["s_d"]
    Ra, Rb = theta["s_a"] ** 2, theta["s_b"] ** 2
    Tm = np.array([[1.0, -dt, 0.0], [0.0, phi, 0.0], [0.0, 0.0, 1.0]])
    Q = np.diag([1e-8, se ** 2, sd ** 2 if des.drift else 0.0])
    # initial state: H = 0 at the start of segment 0, eta stationary, d = 0
    a = np.zeros(3)
    A = np.zeros((3, nb))
    P = np.diag([0.0, se ** 2 / max(1 - phi ** 2, 1e-6), 0.0])
    Sm = np.zeros((nb, nb)); sv = np.zeros(nb); q = 0.0; ld = 0.0; nobs = 0
    acc = {}
    store = smooth or return_b
    if store:
        a_s = np.empty((n, 3)); A_s = np.empty((n, 3, nb)); P_s = np.empty((n, 3, 3))
        a_f = np.empty((n, 3)); A_f = np.empty((n, 3, nb)); P_f = np.empty((n, 3, 3))
    prev_seg = des.seg[0]
    for t in range(n):
        if des.seg[t] != prev_seg:             # new segment: H restarts at 0 (level in h_seg)
            a[0] = 0.0; A[0, :] = 0.0; P[0, :] = 0.0; P[:, 0] = 0.0
            prev_seg = des.seg[t]
        if store:
            a_s[t] = a; A_s[t] = A; P_s[t] = P
        sg = des.seg[t]
        rows = []
        if sg not in mask_segs:
            if np.isfinite(des.ya[t]):
                rows.append(("a", des.ya[t]))
            if des.has_b and np.isfinite(des.yb[t]):
                rows.append(("b", des.yb[t]))
        if rows:
            m = len(rows)
            Z = np.zeros((m, 3)); X = np.zeros((m, nb)); y = np.zeros(m); R = np.zeros(m)
            for i, (k, val) in enumerate(rows):
                y[i] = val
                if k == "a":
                    Z[i, 0] = ga
                    X[i, ix["u"][sg]] = 1.0
                    X[i, ix["ca"]] = des.x[t]
                    R[i] = Ra
                else:
                    Z[i, 0] = 1.0; Z[i, 2] = 1.0 if des.drift else 0.0
                    X[i, ix["h"][sg]] = 1.0
                    X[i, ix["cb"]] = des.x[t] if des.xb is None else des.xb[t]
                    R[i] = Rb
            v = y - Z @ a
            V = X + Z @ A
            Fm = Z @ P @ Z.T + np.diag(R)
            Fi = np.linalg.inv(Fm)
            dS_ = V.T @ Fi @ V; ds_ = V.T @ Fi @ v; dq_ = v @ Fi @ v; dl_ = np.log(np.linalg.det(Fm))
            Sm += dS_; sv += ds_; q += dq_; ld += dl_; nobs += m
            if per_seg:
                e = acc.setdefault(int(sg), [np.zeros((nb, nb)), np.zeros(nb), 0.0, 0.0, 0])
                e[0] += dS_; e[1] += ds_; e[2] += dq_; e[3] += dl_; e[4] += m
            Kf = P @ Z.T @ Fi                    # filtering gain
            a = a + Kf @ v
            A = A - Kf @ V
            P = P - Kf @ Z @ P
        if store:
            a_f[t] = a; A_f[t] = A; P_f[t] = P
        # time update with the regression on b
        Wb = np.zeros((3, nb))
        Wb[0, ix["beta"]] = -dt * des.F[t]
        if des.drift:
            Wb[2, ix["delta"]] = dt
        a = Tm @ a
        A = Tm @ A + Wb
        P = Tm @ P @ Tm.T + Q
    # diffuse log-likelihood; columns of b never observed are dropped
    used = np.abs(np.diag(Sm)) > 1e-12
    Su = Sm[np.ix_(used, used)]; su = sv[used]
    sign, lds = np.linalg.slogdet(Su)
    bhat_u = np.linalg.solve(Su, su)
    ll = -0.5 * (nobs * np.log(2 * np.pi) + ld + q - su @ bhat_u + lds)
    # ll_prof: likelihood concentrated over b, which, unlike the diffuse one, does not
    # depend on the scale of the regressors (needed to compare designs whose
    # regressors are rescaled by a hyperparameter, e.g. the anemometer scale)
    out = dict(ll=float(ll), ll_prof=float(ll + 0.5 * lds), nobs=nobs, n_diffuse=int(used.sum()))
    if per_seg:
        out["acc"] = acc
    if return_b or smooth:
        bhat = np.zeros(nb); bhat[used] = bhat_u
        cov = np.zeros((nb, nb)); cov[np.ix_(used, used)] = np.linalg.inv(Su)
        out.update(bhat=bhat, cov_b=cov, used=used)
    if smooth:
        out.update(_rts(des, theta, Tm, Q, a_s, A_s, P_s, a_f, A_f, P_f, out["bhat"], ix))
    return out


def _rts(des, theta, Tm, Q, a_p, A_p, P_p, a_f, A_f, P_f, bhat, ix):
    """Rauch-Tung-Striebel smoother conditional on b = b_hat."""
    n = len(a_p)
    xp = a_p + np.einsum("tij,j->ti", A_p, bhat)
    xf = a_f + np.einsum("tij,j->ti", A_f, bhat)
    xs = xf.copy(); Ps = P_f.copy()
    for t in range(n - 2, -1, -1):
        if des.seg[t + 1] != des.seg[t]:
            continue                          # H restarts: segments smoothed separately
        Pp = P_p[t + 1]
        G = P_f[t] @ Tm.T @ np.linalg.pinv(Pp)
        xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])
        Ps[t] = P_f[t] + G @ (Ps[t + 1] - Pp) @ G.T
    beta = bhat[ix["beta"]]
    E_reg = des.F @ beta
    E = E_reg + xs[:, 1]
    return dict(x_smooth=xs, P_smooth=Ps, E_reg=E_reg, E=E, eta_sd=np.sqrt(np.maximum(Ps[:, 1, 1], 0)))


# ---------------------------------------------------------------- estimation
HYPER = ["g_a", "phi", "s_eta", "s_d", "s_a", "s_b"]


def unpack(u, fixed=None):
    th = dict(g_a=u[0], phi=np.tanh(u[1]), s_eta=np.exp(u[2]), s_d=np.exp(u[3]),
              s_a=np.exp(u[4]), s_b=np.exp(u[5]))
    if fixed:
        th.update(fixed)
    return th


def pack(th):
    return np.array([th["g_a"], np.arctanh(th["phi"]), np.log(th["s_eta"]), np.log(th["s_d"]),
                     np.log(th["s_a"]), np.log(th["s_b"])])


def fit(des: Design, u0=None, fixed=None, method="Nelder-Mead", maxiter=4000):
    from scipy.optimize import minimize
    if u0 is None:
        u0 = pack(dict(g_a=1.1, phi=0.9, s_eta=0.05, s_d=0.05, s_a=0.1, s_b=0.4))

    def nll(u):
        try:
            r = kalman(des, unpack(u, fixed))
            return -r["ll"] if np.isfinite(r["ll"]) else 1e12
        except np.linalg.LinAlgError:
            return 1e12
    res = minimize(nll, u0, method=method, options=dict(maxiter=maxiter, xatol=1e-4, fatol=1e-3))
    if method == "Nelder-Mead":
        res2 = minimize(nll, res.x, method="Powell", options=dict(maxiter=maxiter))
        if res2.fun < res.fun:
            res = res2
    th = unpack(res.x, fixed)
    return th, res


def simulate_design(n=2000, seed=0, S=3, p=2):
    """Synthetic records with known parameters, for the recovery test."""
    rng = np.random.default_rng(seed)
    seg = np.repeat(np.arange(S), n // S + 1)[:n]
    t = np.arange(n) / 6.0
    lamp = (np.sin(2 * np.pi * t / 24) > 0.2).astype(float)
    F = np.column_stack([0.5 + 2.0 * lamp + 0.1 * rng.standard_normal(n), lamp * rng.uniform(0, 3, n)])[:, :p]
    beta = np.array([0.3, 0.15])[:p]
    x = 5 * lamp + rng.standard_normal(n) * 0.2
    # close to the C2 estimates; s_a, s_b are the noise of the 10-min means
    th = dict(g_a=1.3, phi=0.75, s_eta=0.05, s_d=0.08, s_a=0.03, s_b=0.08)
    eta = np.zeros(n); d = np.zeros(n); H = np.zeros(n)
    for i in range(1, n):
        eta[i] = th["phi"] * eta[i - 1] + th["s_eta"] * rng.standard_normal()
        d[i] = d[i - 1] + (1 / 6) * (-0.08) + th["s_d"] * rng.standard_normal()
        H[i] = 0.0 if seg[i] != seg[i - 1] else H[i - 1] - (1 / 6) * (F[i - 1] @ beta + eta[i - 1])
    hseg = np.array([220.0, 235.0, 228.0])[:S]; useg = np.array([5.0, -3.0, 2.0])[:S]
    ya = useg[seg] + th["g_a"] * H - 0.2 * x
    yb = hseg[seg] + H + d + 0.15 * x
    # as the logger does: ten one-minute readings per 10-min value, each stored with a
    # 0.1 mm quantum, averaged (the one-minute noise is sqrt(10) times that of the mean)
    def logged(y, s):
        r = y[:, None] + s * np.sqrt(10) * rng.standard_normal((n, 10))
        return np.round(r, 1).mean(axis=1)
    ya = logged(ya, th["s_a"]); yb = logged(yb, th["s_b"])
    return Design(ya=ya, yb=yb, x=x, F=F, seg=seg), th, beta


def conditional_score(des: Design, acc_j, j, b_train, known=("delta", "ca", "cb", "beta")):
    """Held-out log score of segment j given the shared coefficients from training.

    The segment's own offsets (h_j, u_j) stay diffuse and are integrated out.
    acc_j = (S, s, q, logdetF, n) accumulated over segment j's innovations.
    """
    ix, nb = _index(des)
    K = np.concatenate([np.atleast_1d(ix[k]) for k in known]).astype(int)
    Uix = np.array([ix["h"][j], ix["u"][j]], int)
    S, s, q, ld, n = acc_j
    bK = b_train[K]
    qp = q - 2 * bK @ s[K] + bK @ S[np.ix_(K, K)] @ bK
    sU = s[Uix] - S[np.ix_(Uix, K)] @ bK
    SUU = S[np.ix_(Uix, Uix)]
    keep = np.abs(np.diag(SUU)) > 1e-12
    sU, SUU = sU[keep], SUU[np.ix_(keep, keep)]
    sign, lds = np.linalg.slogdet(SUU)
    return float(-0.5 * (n * np.log(2 * np.pi) + ld + qp - sU @ np.linalg.solve(SUU, sU) + lds)), int(n)
