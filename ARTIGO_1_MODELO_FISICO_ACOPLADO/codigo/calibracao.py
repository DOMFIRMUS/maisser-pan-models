# -*- coding: utf-8 -*-
"""
Bayesian calibration of the pan model on campaign C2 (uncovered pan).

Observation streams (10-min bins, see dados_modelo for the time convention)
  thermistors  Tw_cs (CS450 body on the floor), Tw_04, Tw_12, Tw_20 (only while
               submerged by >= 12 mm). Model value: trapezoidal bin mean of the
               probe temperature. Linear nuisance: an offset (not for Tw_cs,
               which anchors the absolute temperature) and a radiative error
               proportional to the deep-band irradiance at the probe.
  ODY30A       y = offset_seg + g_a dH + c_a (T_cs - 30): free gain (single-point
               calibration), thermal coefficient, no drift.
  CS456        y - k_CS dH = offset_seg + c_b (T_cs - 30) + drift_seg t: factory
               pressure scale (k_CS = rho_w g_local / (1000 g_n) = 0.9944 converts
               the logger's mm at 1000 kg m-3 and standard gravity to the model's
               mass-equivalent depth), thermal coefficient, one drift per segment.
  The gain of the CS456 anchors the scale of evaporation (lamp-on drops) and the
  drift-free ODY its night-time level; the energy balance of the thermistor
  streams constrains both.

Likelihood. AR(1) errors within segments (Prais-Winsten), each stream's sigma
integrated out (Jeffreys), nuisance coefficients integrated out (flat priors;
theta-dependent columns are normalised to unit RMS so that they do not add an
implicit prior on theta). The set of observations is fixed by the data alone;
a walker with a non-finite prediction gets log-likelihood -inf.
Residual dependence beyond AR(1) and between thermistors is accounted for by
tempering each stream: w_s = (n_eff,T / 4) / tau_s for thermistors and 1/tau_s
for the level streams, where tau_s is the integrated autocorrelation time of
the prewhitened residuals and n_eff,T the participation ratio of their
correlation matrix (a generalised-Bayes power likelihood). rho and w are
iterated with the posterior mode to a fixed point.

Posterior: differential evolution (current-to-best/1/bin) for the mode, then
an affine-invariant ensemble sampler (Goodman and Weare 2010) started from the
final DE population.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modelo_fisico as mf          # noqa: E402
import dados_modelo as dm           # noqa: E402
from priors import Prior, PRIORS_C2  # noqa: E402

OUT = HERE.parent / "saidas" / "calibracao"
OUT.mkdir(parents=True, exist_ok=True)
N_LAYERS = 24
T_STREAMS = ["Tw_cs", "Tw_04", "Tw_12", "Tw_20"]
L_STREAMS = ["level_a_qc", "level_b_qc"]
K_CS = (mf.RHO_W / 1000.0) * (9.7813 / 9.80665)     # CS456 mm -> model mm
RHO_MAX = 0.999


# ---------------------------------------------------------------- simulation per segment
def segment_predictions(seg, P, zc, probes, nuisance_rad=True, return_out=False):
    """Simulate one segment and return bin-averaged predictions (n_obs, W)."""
    F = dm.forcing_arrays(seg)
    if seg.campaign == "C1":
        T0 = P.pop("_T0")
    else:
        T0 = dm.initial_profile(seg, zc)
    out = mf.simulate(F, P, T0, seg.H0, N=N_LAYERS)
    st = dm.obs_steps(seg)
    ok = st[:, 0] >= 0
    n_obs, W = len(st), out.H.shape[1]
    i0, i1 = st[ok, 0], st[ok, 1]
    res = {}
    Hb = mf.bin_average(out, i0, i1, "H")
    res["H"] = np.full((n_obs, W), np.nan); res["H"][ok] = Hb
    res["E"] = np.full((n_obs, W), np.nan); res["E"][ok] = 0.5 * (out.E[i0] + out.E[i1])
    prev = np.maximum(i0 - 1, 0)
    Tprev = np.where((i0 > 0)[:, None, None], out.T[prev], out.T0[None])
    Hprev = np.where((i0 > 0)[:, None], out.H[prev], out.H0[None])
    I_bin = 0.5 * (out.I_eff[i0] + out.I_eff[i1]) * (1.0 - seg.phi)
    for k, h in probes.items():
        hk = P[h] if isinstance(h, str) else h
        T_a = mf.probe_temperature(Tprev, Hprev, zc, hk)
        T_b = mf.probe_temperature(out.T[i0], out.H[i0], zc, hk)
        T_c = mf.probe_temperature(out.T[i1], out.H[i1], zc, hk)
        a = np.full((n_obs, W), np.nan)
        a[ok] = 0.25 * T_a + 0.5 * T_b + 0.25 * T_c
        res[k] = a
        if nuisance_rad:
            dep = np.maximum(Hb - (hk[None, :] if np.ndim(hk) else hk), 0.0)
            r = np.full((n_obs, W), np.nan)
            r[ok] = I_bin * np.exp(-dep / P["l_d"][None, :])
            res["rad_" + k] = r
    return (res, out) if return_out else res


_W = {}


def _init_worker():
    _W["segs"] = {s.name: s for s in dm.c2_segments()}
    s_ = mf.sigma_grid(N_LAYERS)
    _W["zc"] = np.cumsum(s_) - s_ / 2


def run_segment(args):
    name, P = args
    P = {k: np.array(v, float, copy=True) for k, v in P.items()}
    return name, segment_predictions(_W["segs"][name], P, _W["zc"], dm.PROBES_C2)


# ---------------------------------------------------------------- likelihood machinery
def prewhiten(y, X, valid, seg_of, rho):
    """Prais-Winsten AR(1) transform within segments; blocks restart at gaps."""
    idx = np.where(valid)[0]
    seg = seg_of[idx]
    prev_ok = np.zeros(len(idx), bool)
    prev_ok[1:] = (idx[1:] == idx[:-1] + 1) & (seg[1:] == seg[:-1])
    ys = y[idx]
    yp = np.concatenate([ys[:1], ys[:-1]], axis=0)
    f = np.where(prev_ok, 1.0, np.sqrt(1 - rho ** 2))
    ystar = np.where(prev_ok[:, None], ys - rho * yp, f[:, None] * ys)
    Xstar = None
    if X is not None:
        Xs = X[idx]
        Xp = np.concatenate([Xs[:1], Xs[:-1]], axis=0)
        Xstar = np.where(prev_ok[:, None, None], Xs - rho * Xp, f[:, None, None] * Xs)
    return ystar, Xstar, idx


def marginal(ys, Xs):
    """Marginal log-likelihood with flat priors on beta and Jeffreys on sigma."""
    n, W = ys.shape
    if Xs is None or Xs.shape[2] == 0:
        rss = np.sum(ys ** 2, axis=0)
        return -0.5 * n * np.log(rss), np.zeros((W, 0)), rss, n, 0, ys
    p = Xs.shape[2]
    XtX = np.einsum("nwi,nwj->wij", Xs, Xs)
    Xty = np.einsum("nwi,nw->wi", Xs, ys)
    beta = np.linalg.solve(XtX + 1e-10 * np.eye(p), Xty[..., None])[..., 0]
    res = ys - np.einsum("nwi,wi->nw", Xs, beta)
    rss = np.sum(res ** 2, axis=0)
    sign, logdet = np.linalg.slogdet(XtX + 1e-12 * np.eye(p))
    return -0.5 * (n - p) * np.log(rss) - 0.5 * logdet, beta, rss, n, p, res


def unit_rms(col, valid):
    """Normalise theta-dependent nuisance columns (n, W) to unit RMS over valid rows."""
    s = np.sqrt(np.nanmean(col[valid] ** 2, axis=0))
    return col / np.where(s > 0, s, 1.0)[None, :], s


def iat(x, c=5.0):
    """Integrated autocorrelation time with Sokal's automatic window."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        tau += 2 * acf[k]
        if k >= c * tau:
            break
    return max(float(tau), 1.0)


class Problem:
    """Shared machinery: pool, prediction concatenation, posterior."""
    rho: dict
    weights: dict

    def __enter__(self):
        self.pool = mp.Pool(self.processes, initializer=self._initializer, initargs=self._initargs)
        return self

    def __exit__(self, *a):
        if self.pool:
            self.pool.close(); self.pool.join()

    def predict(self, P):
        # one task per segment and block of walkers, longest segments first
        W = len(next(iter(P.values())))
        nch = max(1, min(getattr(self, "walker_chunks", 1), W))
        cuts = np.linspace(0, W, nch + 1).astype(int)
        order = sorted(self.segs, key=lambda s: -len(s.forcing))
        jobs = [(s.name, {k: np.asarray(v)[cuts[c]:cuts[c + 1]] for k, v in P.items()})
                for s in order for c in range(nch)]
        if self.pool is not None:
            res = self.pool.map(self._runner, jobs, chunksize=1)
        else:
            self._initializer(*self._initargs)
            res = list(map(self._runner, jobs))
        results = {}
        for name, r in res:
            results.setdefault(name, []).append(r)
        results = {n: {k: np.concatenate([r[k] for r in rs], axis=1) for k in rs[0]} for n, rs in results.items()}
        keys = results[self.names[0]].keys()
        out = {k: np.concatenate([results[n][k] for n in self.names], axis=0) for k in keys}
        dH = []
        for n in self.names:
            H = results[n]["H"]
            ref = H[np.argmax(np.isfinite(H[:, 0]))]
            dH.append((H - ref) * 1000.0)
        out["dH_mm"] = np.concatenate(dH, axis=0)
        return out

    def logpost(self, U):
        U = np.atleast_2d(U)
        lp = self.prior.logpdf(U)
        P = self.params_from_u(U)
        bad = ~np.isfinite(lp)
        pred = self.predict(P)
        ll = self.loglik(pred)
        out = lp + ll
        out[bad | ~np.isfinite(out)] = -np.inf
        return out

    def params_from_u(self, U):
        full = mf.default_params(U.shape[0])
        full.update(self.prior.to_params(U))
        return full

    # one stream -----------------------------------------------------------
    def stream_ll(self, name, r, X, valid):
        """r: (n, W) residual before nuisance; X: (n, W, p) or None."""
        W = r.shape[1]
        finite = np.all(np.isfinite(r[valid]), axis=0)
        if X is not None:
            finite &= np.all(np.isfinite(X[valid]), axis=(0, 2))
        r = np.where(np.isfinite(r), r, 0.0)
        if X is not None:
            X = np.where(np.isfinite(X), X, 0.0)
        ys, Xs, idx = prewhiten(r, X, valid, self.seg_of, self.rho[name])
        l, beta, rss, n, p, res = marginal(ys, Xs)
        l = np.where(finite, l, -np.inf)
        return l, dict(n=n, p=p, rss=rss, beta=beta, res=res, idx=idx, ll=l)


class C2Problem(Problem):
    def __init__(self, segs=None, rho=None, weights=None, processes=None, spinup_h=dm.SPINUP_H):
        self.segs = segs or dm.c2_segments()
        self.names = [s.name for s in self.segs]
        self.prior = Prior(PRIORS_C2)
        self.rho = rho or {k: 0.95 for k in T_STREAMS + L_STREAMS}
        self.weights = weights or {k: 1.0 for k in T_STREAMS + L_STREAMS}
        self.walker_chunks = 2      # two blocks of walkers per segment balance the pool
        self.processes = processes or min(self.walker_chunks * len(self.segs), (os.cpu_count() or 4) - 1)
        self.pool = None
        self._initializer, self._initargs, self._runner = _init_worker, (), run_segment
        obs = []
        for s in self.segs:
            o = s.obs.copy()
            o["seg"] = s.name
            o["t_seg_d"] = (o.index - s.start).total_seconds() / 86400.0
            o["valid_time"] = o.index >= s.start + pd.Timedelta(hours=spinup_h)
            obs.append(o)
        self.obs = pd.concat(obs)
        self.obs["Tcs_c"] = self.obs.Tw_cs.interpolate(limit_direction="both") - 30.0
        self.seg_of = self.obs.seg.to_numpy()
        # bins that the model covers (structural, identical for every walker)
        mapped = np.concatenate([dm.obs_steps(s)[:, 0] >= 0 for s in self.segs])
        vt = self.obs.valid_time.to_numpy() & mapped
        self.valid = {k: vt & np.isfinite(self.obs[k].to_numpy()) for k in T_STREAMS + L_STREAMS}
        sub = (self.obs.depth_obs - 0.20 >= dm.SUBMERGED_MIN).to_numpy()
        self.valid["Tw_20"] &= sub
        self.D = pd.get_dummies(self.obs.seg).to_numpy(float)

    def loglik(self, pred, return_parts=False):
        W = pred["H"].shape[1]
        total = np.zeros(W)
        parts = {}
        for k in T_STREAMS:
            v = self.valid[k]
            r = self.obs[k].to_numpy()[:, None] - pred[k]
            rad, _ = unit_rms(pred["rad_" + k], v)
            cols = [rad] if k == "Tw_cs" else [np.ones_like(r), rad]
            X = np.stack(cols, axis=2)
            l, info = self.stream_ll(k, r, X, v)
            total += self.weights[k] * l
            parts[k] = info
        Tc = self.obs.Tcs_c.to_numpy()
        n = len(self.obs)
        # ODY: segment offsets, thermal coefficient, gain on dH (normalised)
        k = "level_a_qc"; v = self.valid[k]
        dHn, sc = unit_rms(pred["dH_mm"], v)
        base = np.column_stack([self.D, Tc])
        X = np.concatenate([np.repeat(base[:, None, :], W, axis=1), dHn[:, :, None]], axis=2)
        y = np.repeat(self.obs[k].to_numpy()[:, None], W, axis=1)
        l, info = self.stream_ll(k, y, X, v)
        info["dH_scale"] = sc
        total += self.weights[k] * l
        parts[k] = info
        # CS456: segment offsets, thermal coefficient, drift per segment
        k = "level_b_qc"; v = self.valid[k]
        base = np.column_stack([self.D, Tc, self.D * self.obs.t_seg_d.to_numpy()[:, None]])
        X = np.repeat(base[:, None, :], W, axis=1)
        r = self.obs[k].to_numpy()[:, None] - K_CS * pred["dH_mm"]
        l, info = self.stream_ll(k, r, X, v)
        total += self.weights[k] * l
        parts[k] = info
        return (total, parts) if return_parts else total


# ---------------------------------------------------------------- dependence diagnostics
def dependence(prob, pred):
    """rho, tau and tempering weights from the residuals of walker 0."""
    _, parts = prob.loglik(pred, return_parts=True)
    names = [k for k in parts]
    rho, tau = {}, {}
    raw = {}
    for k in names:
        info = parts[k]
        # residual before prewhitening = y - model - X beta (re-built from prewhitened? use AR fit)
        e = info["res"][:, 0]
        tau[k] = iat(e)
        raw[k] = (info["idx"], e)
    # AR(1) of the un-whitened residuals: recompute with rho = 0
    old = dict(prob.rho)
    prob.rho = {k: 0.0 for k in names}
    _, parts0 = prob.loglik(pred, return_parts=True)
    prob.rho = old
    for k in names:
        idx, e = parts0[k]["idx"], parts0[k]["res"][:, 0]
        seg = prob.seg_of[idx]
        ok = (idx[1:] == idx[:-1] + 1) & (seg[1:] == seg[:-1])
        a, b = e[1:][ok], e[:-1][ok]
        rho[k] = float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, RHO_MAX))
    # cross-correlation of the thermistor innovations on common rows
    tn = [k for k in names if k.startswith("Tw")]
    common = None
    for k in tn:
        s = set(raw[k][0].tolist())
        common = s if common is None else common & s
    common = np.array(sorted(common))
    M = []
    for k in tn:
        idx, e = raw[k]
        pos = {j: i for i, j in enumerate(idx)}
        M.append([e[pos[j]] for j in common])
    C = np.corrcoef(np.array(M))
    lam = np.linalg.eigvalsh(C)
    n_eff = float(lam.sum() ** 2 / np.sum(lam ** 2))
    w = {}
    for k in names:
        w[k] = (n_eff / len(tn)) / tau[k] if k in tn else 1.0 / tau[k]
    return dict(rho=rho, tau=tau, n_eff_T=n_eff, weights=w, corr_T=C.tolist())


# ---------------------------------------------------------------- optimisers
def differential_evolution(prob, pop=48, gens=120, F=0.6, CR=0.8, seed=1, log=None,
                           x_init=None, k_box=6.0):
    """Vectorised DE, current-to-best/1/bin, reflection at +-k_box prior sd."""
    rng = np.random.default_rng(seed)
    P = len(prob.prior.names)
    X = prob.prior.sample(pop, rng)
    if x_init is not None:
        X[: len(x_init)] = x_init
    lo = prob.prior.mu - k_box * prob.prior.sd
    hi = prob.prior.mu + k_box * prob.prior.sd
    X = np.clip(X, lo, hi)
    fx = prob.logpost(X)
    hist = []
    for g in range(gens):
        best = X[np.argmax(fx)]
        r = np.empty((pop, 2), int)
        for i in range(pop):
            choices = np.delete(np.arange(pop), i)
            r[i] = rng.choice(choices, 2, replace=False)
        V = X + F * (best - X) + F * (X[r[:, 0]] - X[r[:, 1]])
        mask = rng.random((pop, P)) < CR
        mask[np.arange(pop), rng.integers(0, P, pop)] = True
        Y = np.where(mask, V, X)
        Y = np.where(Y > hi, 2 * hi - Y, Y)                    # reflection
        Y = np.where(Y < lo, 2 * lo - Y, Y)
        Y = np.clip(Y, lo, hi)
        fy = prob.logpost(Y)
        better = fy > fx
        X[better], fx[better] = Y[better], fy[better]
        hist.append(float(fx.max()))
        if log and (g % 5 == 0 or g == gens - 1):
            log(f"DE gen {g:4d}  best logpost {fx.max():.2f}  median {np.median(fx):.2f}")
    at_box = np.isclose(X[np.argmax(fx)], lo) | np.isclose(X[np.argmax(fx)], hi)
    if log and at_box.any():
        log(f"WARNING: mode at the DE box for {[prob.prior.names[i] for i in np.where(at_box)[0]]}")
    return X, fx, hist


def stretch_mcmc(prob, X0, n_iter=2000, a=2.0, seed=2, log=None, save=None, thin=1):
    """Affine-invariant ensemble sampler (Goodman and Weare 2010), two halves."""
    rng = np.random.default_rng(seed)
    X = X0.copy()
    Wn, P = X.shape
    assert Wn % 2 == 0 and Wn > 2 * P, "need an even number of walkers > 2 x dimension"
    half = Wn // 2
    lp = prob.logpost(X)
    n_store = -(-n_iter // thin)
    chain = np.empty((n_store, Wn, P)); lpc = np.empty((n_store, Wn))
    acc = np.zeros(Wn)
    t0 = time.time()
    for it in range(n_iter):
        for S in (0, 1):
            s = np.arange(S * half, (S + 1) * half)
            c = np.arange((1 - S) * half, (2 - S) * half)
            z = ((a - 1) * rng.random(half) + 1) ** 2 / a
            partner = X[rng.choice(c, half)]
            Y = partner + z[:, None] * (X[s] - partner)
            lpy = prob.logpost(Y)
            logr = (P - 1) * np.log(z) + lpy - lp[s]
            ok = np.log(rng.random(half)) < logr
            X[s[ok]] = Y[ok]; lp[s[ok]] = lpy[ok]; acc[s[ok]] += 1
        if it % thin == 0:
            chain[it // thin] = X; lpc[it // thin] = lp
        if log and (it % 25 == 0 or it == n_iter - 1):
            log(f"MCMC it {it:5d}  acc {acc.mean() / (it + 1):.3f}  lp max {lp.max():.1f} "
                f"median {np.median(lp):.1f}  {(time.time() - t0) / (it + 1):.2f} s/it")
        if save and (it % 50 == 0 or it == n_iter - 1):
            # write, then rename: an interrupted run never leaves a truncated chain file
            tmp = Path(save).with_name("_tmp_" + Path(save).name)
            np.savez_compressed(tmp, chain=chain[: it // thin + 1], lp=lpc[: it // thin + 1],
                                names=np.array(prob.prior.names), acc=acc / (it + 1))
            os.replace(tmp, save)
    return chain, lpc, acc / n_iter


# ---------------------------------------------------------------- convergence diagnostics
# convergence criteria (second half of the chain retained): max split R-hat below
# CONV_RHAT and retained iterations spanning at least CONV_NIAT times the longest IAT
CONV_RHAT = 1.1
CONV_NIAT = 50


def convergence(tag, burn=0.5):
    """Diagnostics of the concatenated chain of a tag, as in resumo_posterior."""
    z = load_chain(tag)
    chain = z["chain"]
    n = chain.shape[0]
    keep = chain[int(n * burn):]
    rh = np.array([split_rhat(keep[:, :, i]) for i in range(chain.shape[2])])
    tau = np.array([iat(keep[:, :, i].mean(1)) for i in range(chain.shape[2])])
    n_iat = keep.shape[0] / tau.max()
    return dict(n_iter=int(n), walkers=int(chain.shape[1]), max_rhat=float(rh.max()), max_iat=float(tau.max()),
                n_iat=float(n_iat), worst_rhat=str(z["names"][rh.argmax()]), worst_iat=str(z["names"][tau.argmax()]),
                lp_best=float(z["lp"].max()),
                converged=bool(rh.max() < CONV_RHAT and n_iat >= CONV_NIAT),
                n_needed=int(np.ceil(CONV_NIAT * tau.max() / (1 - burn))))


def split_rhat(x):
    """Rank-normalised split R-hat (Vehtari et al. 2021) for one parameter.

    x: (n_draws, n_chains). For an ensemble sampler, walkers are treated as
    chains; this is conservative for the stretch move."""
    from scipy.stats import norm, rankdata
    n, m = x.shape
    h = n // 2
    xs = np.concatenate([x[:h], x[h:2 * h]], axis=1)
    r = rankdata(xs).reshape(xs.shape)
    z = norm.ppf((r - 0.375) / (r.size + 0.25))

    def rhat(y):
        nn = y.shape[0]
        B = nn * y.mean(0).var(ddof=1)
        Wv = y.var(0, ddof=1).mean()
        var = (nn - 1) / nn * Wv + B / nn
        return np.sqrt(var / Wv)
    return float(max(rhat(z), rhat(np.abs(z - np.median(z)))))


def ess(x):
    """ESS from the integrated autocorrelation of the ensemble-mean trace."""
    tau = iat(x.mean(1))
    return float(x.shape[0] * x.shape[1] / tau), float(tau)


# ---------------------------------------------------------------- driver
def save_mode(prob, X, fx, path, extra=None):
    best = X[np.argmax(fx)]
    P = prob.params_from_u(best[None])
    pred = prob.predict(P)
    dep = dependence(prob, pred)
    _, parts = prob.loglik(pred, return_parts=True)
    nuis = {k: (v["beta"][0].tolist() if v["beta"].size else []) for k, v in parts.items()}
    nuis_scale = {k: (v.get("dH_scale")[0].item() if v.get("dH_scale") is not None else None)
                  for k, v in parts.items()}
    m = {"u": best.tolist(), "names": prob.prior.names, "logpost": float(fx.max()),
         "params": {k: float(v[0]) for k, v in prob.prior.to_params(best[None]).items()},
         "rho_used": prob.rho, "weights_used": prob.weights,
         "rho_next": dep["rho"], "tau_next": dep["tau"], "weights_next": dep["weights"],
         "n_eff_T": dep["n_eff_T"], "corr_T": dep["corr_T"], "nuisance": nuis,
         "nuisance_dH_scale": nuis_scale}
    if extra:
        m.update(extra)
    json.dump(m, open(path, "w"), indent=2)
    return m


def main(stage="all", gens=100, n_iter=2000, walkers=48, rounds=2, tag="c2"):
    logf = open(OUT / f"calibracao_{tag}_log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    with C2Problem() as prob:
        log(f"segments: {prob.names}; obs rows {len(prob.obs)}; params {prob.prior.names}")
        log("valid rows: " + ", ".join(f"{k} {int(v.sum())}" for k, v in prob.valid.items()))
        mpath = OUT / f"map_{tag}.json"
        if stage in ("all", "map"):
            x_init = None
            for rd in range(rounds):
                X, fx, hist = differential_evolution(prob, pop=walkers, gens=gens if rd == 0 else gens // 2,
                                                     log=log, x_init=x_init, seed=1 + rd)
                m = save_mode(prob, X, fx, mpath, extra={"round": rd})
                np.savez_compressed(OUT / f"de_population_{tag}.npz", X=X, fx=fx, hist=hist,
                                    names=np.array(prob.prior.names))
                log(f"round {rd}: rho {m['rho_next']}")
                log(f"round {rd}: tau {m['tau_next']}  n_eff_T {m['n_eff_T']:.2f}")
                log(f"round {rd}: weights {m['weights_next']}")
                prob.rho = m["rho_next"]; prob.weights = m["weights_next"]
                x_init = X[np.argsort(fx)[::-1]]
            # final mode under the fixed-point rho and weights
            fx = prob.logpost(X)
            save_mode(prob, X, fx, mpath, extra={"round": "final"})
            np.savez_compressed(OUT / f"de_population_{tag}.npz", X=X, fx=fx,
                                names=np.array(prob.prior.names))
        if stage in ("all", "mcmc"):
            m = json.load(open(mpath))
            prob.rho = m["rho_used"]; prob.weights = m["weights_used"]
            de = np.load(OUT / f"de_population_{tag}.npz")
            X0 = de["X"][np.argsort(de["fx"])[::-1][:walkers]]
            rng = np.random.default_rng(5)
            X0 = X0 + 0.01 * prob.prior.sd * rng.standard_normal(X0.shape)
            chain, lpc, acc = stretch_mcmc(prob, X0, n_iter=n_iter, log=log,
                                           save=OUT / f"mcmc_chain_{tag}.npz")
            log(f"MCMC done, acceptance {acc.mean():.3f}")
        if stage == "continue":
            m = json.load(open(mpath))
            prob.rho = m["rho_used"]; prob.weights = m["weights_used"]
            continue_chain(prob, tag, n_iter, log)
    logf.close()


def chain_files(tag, out=OUT):
    """The chain of a tag: the first run and its continuations, in order."""
    conts = sorted(out.glob(f"mcmc_chain_{tag}_cont*.npz"), key=lambda p: int(p.stem.split("cont")[-1]))
    return [out / f"mcmc_chain_{tag}.npz"] + conts


def load_chain(tag, out=OUT):
    """Concatenate the first run and its continuations: chain (n, W, P), lp (n, W), names, acc."""
    parts = [np.load(f) for f in chain_files(tag, out)]
    chain = np.concatenate([p["chain"] for p in parts])
    lp = np.concatenate([p["lp"] for p in parts])
    n = np.array([len(p["lp"]) for p in parts], float)
    acc = sum(p["acc"] * k for p, k in zip(parts, n)) / n.sum()
    return dict(chain=chain, lp=lp, names=parts[0]["names"], acc=acc)


def continue_chain(prob, tag, n_iter, log, seed=None):
    """Run n_iter more iterations from the last state of the existing chain;
    the continuation is written to the next mcmc_chain_<tag>_cont<k>.npz."""
    files = chain_files(tag, prob_out(prob))
    last = np.load(files[-1])
    X0 = last["chain"][-1]
    k = len(files)
    log(f"continuing {tag} from {files[-1].name} ({len(last['lp'])} iterations) -> cont{k}")
    _, _, acc = stretch_mcmc(prob, X0, n_iter=n_iter, log=log, seed=seed if seed is not None else 100 + k,
                             save=prob_out(prob) / f"mcmc_chain_{tag}_cont{k}.npz")
    log(f"MCMC continuation done, acceptance {acc.mean():.3f}")


def prob_out(prob):
    return OUT


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--iter", type=int, default=2000)
    ap.add_argument("--walkers", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tag", default="c2")
    a = ap.parse_args()
    main(a.stage, a.gens, a.iter, a.walkers, a.rounds, a.tag)
