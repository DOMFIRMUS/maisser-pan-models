# -*- coding: utf-8 -*-
"""
Calibration of the floating-cover module on campaign C1.

The physics of the uncovered pan comes from the C2 posterior (modular, 'cut'
Bayesian approach: C1 data do not update the C2 parameters; C2 uncertainty is
fixed at the C2 posterior mode, a two-stage plug-in approximation). The C1
parameters are
  * the ratio of pan-average irradiance to the SP-LITE float reading for the
    lamp configurations of C1 (m_3: mixed halogen/LED until 19 Aug; m_4: 15
    drying lamps on 20-21 Aug; m_5: 18 drying lamps from 24 Aug; m_6: the days
    of that period with the index below 450 W m-2, when few lamps were on);
  * a scale of the anemometer reading (s_U; different position in C1);
  * the heights of the three C1 thermistors above the floor (undocumented);
  * the cover parameters: absorptance a_p, plate-water conductance G_pw,
    areal heat capacity C_p, effective opening size r_open and the recess
    diffusion length R_rec (one per plate version).
Streams: three thermistors (radiative error as normalised linear nuisance);
the QDY30A level (segment offsets, gain on dH, thermal coefficient on the bulk
water temperature, no drift); and the manual micrometer readings of the
surface height (segment offsets only, gain 1, model height including the
thermal expansion of the column), which anchor the evaporation scale.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modelo_fisico as mf          # noqa: E402
import dados_modelo as dm           # noqa: E402
import calibracao as cb             # noqa: E402
from priors import Prior            # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "_comum"))
from tanque import config as cfg, propriedades as pr   # noqa: E402

OUT = cb.OUT
N_LAYERS = cb.N_LAYERS
PROBES = dm.PROBES_C1
PROBE_H = {"Tw_p1": "h_p1", "Tw_p2": "h_p2", "Tw_p3": "h_p3"}
STREAMS = PROBES + ["level_a_qc", "micrometer"]

PRIORS_C1 = {
    "m_3": ("log", 1.0, 0.50, "pan-average/float-reading ratio, mixed halogen+LED period"),
    "m_4": ("log", 1.0, 0.50, "same, 15 drying lamps (20-21 Aug)"),
    "m_5": ("log", 1.0, 0.50, "same, 18 drying lamps (from 24 Aug)"),
    "m_6": ("log", 1.0, 0.50, "same, days of that period with the index below 450 W m-2 (few lamps on)"),
    "s_U": ("log", 1.0, 0.30, "C1 anemometer position (0.09 m above water)"),
    "h_p1": ("lin", 0.10, 0.06, "thermistor 1 height above floor (m), undocumented"),
    "h_p2": ("lin", 0.19, 0.03, "thermistor 2 (near-surface probe) height (m)"),
    "h_p3": ("lin", 0.10, 0.06, "thermistor 3 height above floor (m), undocumented"),
    "a_p": ("logit", 0.45, 0.80, "plate absorptance of the lamp radiation, white print"),
    "C_u_V1": ("log", 5e-4, 1.00, "transfer velocity of the vented air pockets under the finned V1 plates (m s-1)"),
    "C_u_V2": ("log", 5e-4, 1.00, "same, V2 plates"),
    "k_g": ("log", 1.5, 0.60, "forced-transfer enhancement over the gaps (boundary-layer restart downstream of dry plates)"),
    "G_pw": ("log", 4.0, 0.80, "plate-water conductance (W m-2 K-1), 31 mm shell, 5 % infill"),
    "C_p": ("log", 6000.0, 0.50, "areal heat capacity of plates (J m-2 K-1)"),
    "r_open": ("log", 0.010, 1.00, "effective opening size in Schlunder's relation (m)"),
    "R_rec_V1": ("log", 0.004, 0.90, "recess diffusion length d_rec/zeta, V1 plates (m)"),
    "R_rec_V2": ("log", 0.004, 0.90, "same, V2 plates (lighter: larger freeboard)"),
}

_W = {}


def _init_worker(base_params):
    _W["segs"] = {s.name: s for s in micrometer_segments(dm.c1_segments())}
    s_ = mf.sigma_grid(N_LAYERS)
    _W["zc"] = np.cumsum(s_) - s_ / 2
    _W["base"] = base_params


def initial_profiles(seg, P, zc):
    """(W, N) initial profiles from the three probes at walker-specific heights."""
    W = len(P["h_p1"])
    T0 = np.empty((W, len(zc)))
    temps = np.array([seg.T0_heights["p1"], seg.T0_heights["p2"], seg.T0_heights["p3"]])
    height = seg.H0 * (1 - zc)
    for w in range(W):
        hs = np.array([P["h_p1"][w], P["h_p2"][w], P["h_p3"][w]])
        o = np.argsort(hs)
        T0[w] = np.interp(height, np.clip(hs[o], 0, seg.H0), temps[o])
    return T0


def geometric_height(T, H, s):
    """Surface height (m) of a column of mass-equivalent depth H with profile T."""
    return H * np.sum(s[None, None, :] * (mf.RHO_W / pr.water_density(T)), axis=-1)


def run_segment(args):
    name, Pc1 = args
    seg = _W["segs"][name]
    zc = _W["zc"]
    W = len(next(iter(Pc1.values())))
    P = {k: np.broadcast_to(np.asarray(v, float), (W,)).copy() for k, v in _W["base"].items()}
    P.update({k: np.array(v, float, copy=True) for k, v in Pc1.items()})
    version_params(P, seg.version)
    P["_T0"] = initial_profiles(seg, P, zc)
    res, out = cb.segment_predictions(seg, P, zc, PROBE_H, return_out=True)
    # surface height at each micrometer reading (end of the step containing it)
    mic = seg.extra.get("mic")
    if mic is not None and len(mic):
        s_ = mf.sigma_grid(N_LAYERS)
        k = np.clip(((mic.index - seg.start) / pd.Timedelta(minutes=5)).astype(int) - 1, 0, len(out.H) - 1)
        res["mic_h"] = geometric_height(out.T[k], out.H[k], s_) * 1000.0
    else:
        res["mic_h"] = np.zeros((0, W))
    return name, res


# Micrometer error model. None: the standard error is estimated (Jeffreys prior,
# integrated out like those of the other streams), so that it absorbs the model
# error as well as the reading error (about 0.2 mm: micrometer, meniscus and
# small level oscillations). A number fixes it (sensitivity case, --mic-sd 0.2).
MIC_SD = None
# Gain of the QDY30A relative to the water mass in C1, calibrated on the step of
# 18 Aug when 27 plates were deployed: 3.73 mm on the transducer (10-min medians
# before and after) against 3.95 mm on the micrometer (3.907 mm between readings
# 31 min apart, plus about 0.05 mm of evaporation) -> 0.944 (+-0.07 for reading
# errors of about 0.2 mm). None = free gain (sensitivity case). Set by --qdy-gain.
QDY_GAIN = 0.944
# The first reading of the campaign (installation of the stilling well) lies
# 1.8 mm above the trend of the following ten readings of the same day.
MIC_EXCLUDE = [pd.Timestamp("2026-08-17 09:21")]
# A pair taken 27 min apart at the end of a run (20 Aug 17:51 and 18:18) implies a
# fall of 2.1 mm/h, twice the highest evaporation of the campaign, so one of them
# is wrong; the reading of 1 Sep 16:20 was taken at a reinstallation of the
# plates, i.e. at a change of cover (segment boundary).
MIC_EXCLUDE += [pd.Timestamp(t) for t in ("2026-08-20 17:51", "2026-08-20 18:18", "2026-09-01 16:20")]


def version_params(P, version):
    """Set the parameters that differ between plate versions (recess length and
    under-plate ventilation) for a segment covered by `version`."""
    v = "V1" if version == "V1" else "V2"
    for k in ("R_rec", "C_u"):
        if f"{k}_{v}" in P:
            P[k] = P[f"{k}_{v}"]
    return P


def micrometer_segments(segs):
    """Attach the micrometer readings falling inside each segment."""
    m = pd.read_csv(cfg.MICROMETER_C1)
    m["time"] = pd.to_datetime(m["date"] + " " + m["time"])
    bad = m["note"].fillna("").str.contains("non-monotonic|refill|buoyancy|deployed|removed", case=False)
    bad |= m["time"].isin(MIC_EXCLUDE)
    m = m[~bad].set_index("time")
    for s in segs:
        sub = m[(m.index > s.start + pd.Timedelta(minutes=10)) & (m.index <= s.end)]
        s.extra["mic"] = sub["reading_mm"] if len(sub) >= 2 else None
    return segs


class C1Problem(cb.Problem):
    def __init__(self, base_params, rho=None, weights=None, processes=None, spinup_h=dm.SPINUP_H):
        self.segs = micrometer_segments(dm.c1_segments())
        self.names = [s.name for s in self.segs]
        self.prior = Prior(PRIORS_C1)
        self.base = base_params
        self.rho = rho or {k: 0.95 for k in STREAMS}
        self.rho["micrometer"] = 0.0
        self.weights = weights or {k: 1.0 for k in STREAMS}
        self.processes = processes or min(len(self.segs), (os.cpu_count() or 4) - 1)
        self.pool = None
        self._initializer, self._initargs, self._runner = _init_worker, (base_params,), run_segment
        obs = []
        for s in self.segs:
            o = s.obs.copy()
            o["seg"] = s.name
            o["valid_time"] = o.index >= s.start + pd.Timedelta(hours=min(spinup_h, 2.0))
            obs.append(o)
        self.obs = pd.concat(obs)
        self.obs["Tb_c"] = self.obs[["Tw_p1", "Tw_p3"]].mean(axis=1).interpolate(limit_direction="both") - 30.0
        self.seg_of = self.obs.seg.to_numpy()
        mapped = np.concatenate([dm.obs_steps(s)[:, 0] >= 0 for s in self.segs])
        vt = self.obs.valid_time.to_numpy() & mapped
        self.valid = {k: vt & np.isfinite(self.obs[k].to_numpy()) for k in PROBES + ["level_a_qc"]}
        self.D = pd.get_dummies(self.obs.seg).to_numpy(float)
        mic = [(s.name, t, v) for s in self.segs if s.extra.get("mic") is not None
               for t, v in s.extra["mic"].items()]
        self.mic = pd.DataFrame(mic, columns=["seg", "time", "reading_mm"])
        self.mic_D = pd.get_dummies(self.mic.seg).to_numpy(float) if len(self.mic) else None

    def params_from_u(self, U):
        P = self.prior.to_params(U)
        for k in ("h_p1", "h_p2", "h_p3"):
            P[k] = np.clip(P[k], 0.005, 0.24)
        return P

    def logpost(self, U):
        U = np.atleast_2d(U)
        lp = self.prior.logpdf(U)
        raw = self.prior.to_params(U)
        for k in ("h_p1", "h_p2", "h_p3"):
            lp[(raw[k] < 0.005) | (raw[k] > 0.24)] = -np.inf
        pred = self.predict(self.params_from_u(U))
        out = lp + self.loglik(pred)
        out[~np.isfinite(out)] = -np.inf
        return out

    def predict(self, P):
        out = super().predict(P)
        return out

    def loglik(self, pred, return_parts=False):
        W = pred["H"].shape[1]
        total = np.zeros(W)
        parts = {}
        for k in PROBES:
            v = self.valid[k]
            r = self.obs[k].to_numpy()[:, None] - pred[k]
            rad, _ = cb.unit_rms(pred["rad_" + k], v)
            l, info = self.stream_ll(k, r, rad[:, :, None], v)
            total += self.weights[k] * l
            parts[k] = info
        k = "level_a_qc"; v = self.valid[k]
        dHn, sc = cb.unit_rms(pred["dH_mm"], v)
        base = np.column_stack([self.D, self.obs.Tb_c.to_numpy()])
        if QDY_GAIN is None:
            X = np.concatenate([np.repeat(base[:, None, :], W, axis=1), dHn[:, :, None]], axis=2)
            y = np.repeat(self.obs[k].to_numpy()[:, None], W, axis=1)
        else:
            # gain calibrated independently on the plate-deployment step
            X = np.repeat(base[:, None, :], W, axis=1)
            y = self.obs[k].to_numpy()[:, None] - QDY_GAIN * pred["dH_mm"]
        l, info = self.stream_ll(k, y, X, v)
        info["dH_scale"] = sc
        total += self.weights[k] * l
        parts[k] = info
        if len(self.mic):
            r = self.mic.reading_mm.to_numpy()[:, None] - pred["mic_h"]
            X = np.repeat(self.mic_D[:, None, :], W, axis=1)
            l, beta, rss, n, p, res = cb.marginal(r, X)
            if MIC_SD:
                # known reading error; segment offsets integrated out (their
                # determinant term is the same for every walker)
                l = -0.5 * rss / MIC_SD ** 2
            total += self.weights["micrometer"] * l
            parts["micrometer"] = dict(n=n, p=p, rss=rss, beta=beta, res=res,
                                       idx=np.arange(len(self.mic)), ll=l)
        return (total, parts) if return_parts else total


def qdy_coefs(parts):
    """(gain, thermal coefficient mm/K, fitted values) of the QDY30A stream."""
    info = parts["level_a_qc"]
    beta = info["beta"][0]
    if QDY_GAIN is None:
        return float(beta[-1] / info["dH_scale"][0]), float(beta[-2])
    return float(QDY_GAIN), float(beta[-1])


def qdy_fitted(prob, parts, pred, w=0):
    """Fitted QDY30A values (offsets + thermal term + gain x modelled depth change)."""
    beta = parts["level_a_qc"]["beta"][w]
    g, c = qdy_coefs(parts)
    nseg = prob.D.shape[1]
    return prob.D @ beta[:nseg] + c * prob.obs.Tb_c.to_numpy() + g * pred["dH_mm"][:, w]


def c1_dependence(prob, pred):
    """rho, tau and weights for the C1 streams (micrometer: white, weight 1)."""
    _, parts = prob.loglik(pred, return_parts=True)
    old = dict(prob.rho)
    prob.rho = {k: 0.0 for k in STREAMS}
    _, parts0 = prob.loglik(pred, return_parts=True)
    prob.rho = old
    rho, tau, w = {}, {}, {}
    for k in PROBES + ["level_a_qc"]:
        idx, e = parts0[k]["idx"], parts0[k]["res"][:, 0]
        seg = prob.seg_of[idx]
        ok = (idx[1:] == idx[:-1] + 1) & (seg[1:] == seg[:-1])
        rho[k] = float(np.clip(np.corrcoef(e[1:][ok], e[:-1][ok])[0, 1], 0.0, cb.RHO_MAX))
        tau[k] = cb.iat(parts[k]["res"][:, 0])
    # thermistor cross-correlation
    common = set(parts[PROBES[0]]["idx"].tolist())
    for k in PROBES[1:]:
        common &= set(parts[k]["idx"].tolist())
    common = np.array(sorted(common))
    M = []
    for k in PROBES:
        pos = {j: i for i, j in enumerate(parts[k]["idx"])}
        M.append([parts[k]["res"][pos[j], 0] for j in common])
    C = np.corrcoef(np.array(M)); lam = np.linalg.eigvalsh(C)
    n_eff = float(lam.sum() ** 2 / np.sum(lam ** 2))
    for k in PROBES:
        w[k] = (n_eff / len(PROBES)) / tau[k]
    w["level_a_qc"] = 1.0 / tau["level_a_qc"]
    rho["micrometer"] = 0.0; tau["micrometer"] = 1.0; w["micrometer"] = 1.0
    return dict(rho=rho, tau=tau, weights=w, n_eff_T=n_eff)


def state_from_chain(path, prior, n=None, seed=0):
    """Last state of an earlier chain mapped onto the current parameter list;
    parameters absent from that chain start from prior draws."""
    z = np.load(path)
    names = [str(k) for k in z["names"]]
    X = z["chain"][-1]
    rng = np.random.default_rng(seed)
    out = prior.mu + prior.sd * rng.standard_normal((len(X), len(prior.names)))
    alias = {"m_6": "m_5"}       # a new configuration starts from the one it was split from
    for j, k in enumerate(prior.names):
        src = k if k in names else alias.get(k)
        if src in names:
            out[:, j] = X[:, names.index(src)]
    if n and n > len(out):
        # more walkers than the earlier chain had: jittered copies of its walkers
        extra = out[rng.choice(len(out), n - len(out))] + 0.01 * prior.sd * rng.standard_normal((n - len(out), len(prior.names)))
        out = np.vstack([out, extra])
    return out[:n] if n else out


def base_from_map(path=None):
    m = json.load(open(path or OUT / "map_c2.json"))
    P = mf.default_params(1)
    for k, v in m["params"].items():
        P[k] = np.array([v])
    return {k: float(v[0]) for k, v in P.items()}


def save_mode(prob, X, fx, path, extra=None):
    best = X[np.argmax(fx)]
    pred = prob.predict(prob.params_from_u(best[None]))
    dep = c1_dependence(prob, pred)
    _, parts = prob.loglik(pred, return_parts=True)
    m = {"u": best.tolist(), "names": prob.prior.names, "logpost": float(fx.max()),
         "params": {k: float(v[0]) for k, v in prob.params_from_u(best[None]).items()},
         "rho_used": prob.rho, "weights_used": prob.weights,
         "rho_next": dep["rho"], "tau_next": dep["tau"], "weights_next": dep["weights"],
         "n_eff_T": dep["n_eff_T"],
         "nuisance": {k: (v["beta"][0].tolist() if v["beta"].size else []) for k, v in parts.items()},
         "qdy_dH_scale": float(parts["level_a_qc"]["dH_scale"][0]), "base": prob.base,
         "qdy_gain": qdy_coefs(parts)[0], "qdy_thermal": qdy_coefs(parts)[1],
         "qdy_gain_fixed": QDY_GAIN is not None, "mic_sd": MIC_SD}
    if extra:
        m.update(extra)
    json.dump(m, open(path, "w"), indent=2)
    return m


def main(stage="map", gens=80, n_iter=1200, walkers=32, base_path=None, tag="c1", processes=None,
         rounds=2, init_from=None, weights_from=None):
    logf = open(OUT / f"calibracao_{tag}_log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    base = base_from_map(base_path)
    mpath = OUT / f"map_{tag}.json"
    w0 = json.load(open(weights_from)) if weights_from else None
    with C1Problem(base, processes=processes, rho=w0["rho_used"] if w0 else None,
                   weights=w0["weights_used"] if w0 else None) as prob:
        log(f"C1 segments {prob.names}; obs {len(prob.obs)}; micrometer readings {len(prob.mic)}; "
            f"base from {base_path or 'map_c2.json'}")
        if stage in ("map", "all"):
            x_init = state_from_chain(init_from, prob.prior) if init_from else None
            for rd in range(rounds):
                X, fx, hist = cb.differential_evolution(prob, pop=walkers, gens=gens if rd == 0 else gens // 2,
                                                        log=log, x_init=x_init, seed=3 + rd)
                m = save_mode(prob, X, fx, mpath, extra={"round": rd})
                log(f"round {rd}: rho {m['rho_next']}  weights {m['weights_next']}")
                prob.rho = m["rho_next"]; prob.weights = m["weights_next"]
                x_init = X[np.argsort(fx)[::-1]]
            fx = prob.logpost(X)
            save_mode(prob, X, fx, mpath, extra={"round": "final"})
            np.savez_compressed(OUT / f"de_population_{tag}.npz", X=X, fx=fx, names=np.array(prob.prior.names))
        if stage in ("mcmc", "all"):
            m = json.load(open(mpath))
            prob.rho = m["rho_used"]; prob.weights = m["weights_used"]
            if init_from:
                # start from the last state of an earlier chain (already spread over the posterior)
                # the spread of the earlier chain, re-centred on the current mode (the
                # earlier posterior may sit elsewhere if the model or data changed)
                X0 = state_from_chain(init_from, prob.prior, walkers)
                X0 = X0 - X0.mean(axis=0) + np.array(m["u"])
                log(f"walkers initialised from the spread of the last state of {init_from}, "
                    f"re-centred on the mode in {mpath.name}")
            else:
                de = np.load(OUT / f"de_population_{tag}.npz")
                X0 = de["X"][np.argsort(de["fx"])[::-1][:walkers]]
                rng = np.random.default_rng(11)
                X0 = X0 + 0.01 * prob.prior.sd * rng.standard_normal(X0.shape)
            chain, lpc, acc = cb.stretch_mcmc(prob, X0, n_iter=n_iter, log=log,
                                              save=OUT / f"mcmc_chain_{tag}.npz")
            log(f"MCMC {tag} done, acceptance {acc.mean():.3f}")
        if stage == "continue":
            m = json.load(open(mpath))
            prob.rho = m["rho_used"]; prob.weights = m["weights_used"]
            cb.continue_chain(prob, tag, n_iter, log)
    logf.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="map")
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--iter", type=int, default=1200)
    ap.add_argument("--walkers", type=int, default=32)
    ap.add_argument("--base", default=None)
    ap.add_argument("--tag", default="c1")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--weights-from", default=None)
    ap.add_argument("--qdy-gain", default=None, help="QDY30A gain (float) or 'free'")
    ap.add_argument("--mic-sd", default=None, help="micrometer error (mm) or 'free' (estimated)")
    a = ap.parse_args()
    if a.mic_sd is not None:
        MIC_SD = None if a.mic_sd == "free" else float(a.mic_sd)
    if a.qdy_gain is not None:
        QDY_GAIN = None if a.qdy_gain == "free" else float(a.qdy_gain)
    main(a.stage, a.gens, a.iter, a.walkers, a.base, a.tag, a.processes, a.rounds, a.init_from, a.weights_from)
