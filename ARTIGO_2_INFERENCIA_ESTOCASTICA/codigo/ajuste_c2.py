# -*- coding: utf-8 -*-
"""
Campaign C2: state-space fits for every evaporation-driver formulation.

For each candidate in dados_ee.DRIVERS the hyperparameters are estimated by
maximising the diffuse log-likelihood (augmented Kalman filter), then the
model is smoothed. Output: saidas/c2_ajustes.json (hyperparameters, b_hat,
standard errors, log-likelihood, AIC, BIC) and saidas/c2_<driver>_E.csv
(smoothed 10-min evaporation with its uncertainty).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dados_ee as de        # noqa: E402
import espaco_estados as ee  # noqa: E402

OUT = HERE.parent / "saidas"
OUT.mkdir(parents=True, exist_ok=True)
HYPER_DRIVERS = {"livre_forcada_potencia": {"n": [0.35, 0.5, 0.65, 0.8, 1.0]},
                 "churchill_usagi": {"n": [0.3, 0.5, 0.8], "k": [2.0, 4.0, 8.0]},
                 "churchill_usagi_fundo": {"n": [0.3, 0.5, 0.8], "k": [2.0, 4.0, 8.0]}}
# the grids are extended in any direction in which the maximum lies on their edge
HYPER_LIMITS = {"n": (0.1, 1.5), "k": (0.25, 128.0)}


def design_c2(a, driver, hyper=None):
    seg_codes = pd.factorize(a["segment"])[0]
    F = de.driver_matrix(a, driver, hyper)
    x = (a["Tw_cs"].interpolate(limit_direction="both") - 30.0).to_numpy()
    return ee.Design(ya=a["level_a_qc"].to_numpy(), yb=a["level_b_qc"].to_numpy(), x=x, F=F,
                     seg=seg_codes, dt_h=de.DT_H)


def prepare_c2():
    a, ev = de.series_10min("C2")
    a = a[a["segment"].notna()].copy()
    # drop segments with less than 6 h of level data
    cnt = a.groupby("segment")["level_a_qc"].count()
    keep = cnt[cnt >= 36].index
    a = a[a["segment"].isin(keep)]
    return a


def grid(h):
    keys = list(h)
    vals = np.array(np.meshgrid(*[h[k] for k in keys])).reshape(len(keys), -1).T
    return [dict(zip(keys, v)) for v in vals]


def _extend(v, key, up):
    lo, hi = HYPER_LIMITS[key]
    if key == "k":
        x = v * 2.0 if up else v / 2.0
    else:
        x = round(v + 0.15, 3) if up else round(v - 0.15, 3)
    return x if lo <= x <= hi else None


def fit_driver(a, driver, u0=None, log=print):
    """Maximum-likelihood fit; nonlinear hyperparameters are profiled on a grid
    that is extended, one coordinate at a time, until the maximum is interior or
    a limit of HYPER_LIMITS is reached."""
    if driver not in HYPER_DRIVERS:
        des = design_c2(a, driver, None)
        th, _ = ee.fit(des, u0=u0, maxiter=3000)
        r = ee.kalman(des, th, return_b=True)
        log(f"   {driver}: ll={r['ll']:.2f}  g_a={th['g_a']:.3f} phi={th['phi']:.3f}")
        return None, th, r, des, []
    done = {}

    def run(hy):
        key = tuple(sorted(hy.items()))
        if key in done:
            return
        des = design_c2(a, driver, hy)
        th, _ = ee.fit(des, u0=u0, maxiter=3000)
        r = ee.kalman(des, th, return_b=True)
        done[key] = (dict(hy), th, r, des)
        log(f"   {driver} {hy}: ll={r['ll']:.2f}  g_a={th['g_a']:.3f} phi={th['phi']:.3f} "
            f"s_eta={th['s_eta']:.4f} s_d={th['s_d']:.4f}")

    for hy in grid(HYPER_DRIVERS[driver]):
        run(hy)
    while True:
        best = max(done.values(), key=lambda b: b[2]["ll"])
        new = []
        for key in best[0]:
            vals = sorted({v[0][key] for v in done.values()})
            for up, edge in ((False, vals[0]), (True, vals[-1])):
                if best[0][key] == edge:
                    x = _extend(edge, key, up)
                    if x is not None:
                        hy = dict(best[0]); hy[key] = x
                        if tuple(sorted(hy.items())) not in done:
                            new.append(hy)
        if not new:
            break
        for hy in new:
            run(hy)
    best = max(done.values(), key=lambda b: b[2]["ll"])
    profile = [dict(**v[0], ll=v[2]["ll"]) for v in done.values()]
    return best[0], best[1], best[2], best[3], profile


def _fit_one(drv):
    """Fit one formulation (worker); warm start from an earlier fit if any."""
    a = prepare_c2()
    old = OUT / "c2_ajustes_anterior.json"
    u0 = None
    if old.exists():
        prev = json.load(open(old)).get(drv)
        if prev:
            u0 = ee.pack(prev["theta"])
    lines = []
    t0 = time.time()
    hy, th, r, des, profile = fit_driver(a, drv, u0=u0, log=lines.append)
    res, E = _summarise(a, drv, hy, th, r, des)
    res["hyper_profile"] = profile
    E.to_csv(OUT / f"c2_{drv}_E.csv")
    lines.append(f"{drv}: ll={r['ll']:.2f} AIC={res['aic']:.1f} BIC={res['bic']:.1f} "
                 f"beta={np.round(res['beta'], 4)} delta={res['delta']:.4f} mm/h c_a={res['c_a']:.3f} "
                 f"c_b={res['c_b']:.3f} g_a={th['g_a']:.3f} hyper={hy}  ({time.time() - t0:.0f} s)")
    return drv, res, lines


def main(drivers=None, processes=4):
    from multiprocessing import Pool
    logf = open(OUT / "c2_ajustes_log.txt", "w", encoding="utf-8")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    a = prepare_c2()
    log(f"C2: {len(a)} 10-min steps, {a.segment.nunique()} segments, "
        f"ODY obs {a.level_a_qc.notna().sum()}, CS456 obs {a.level_b_qc.notna().sum()}")
    if (OUT / "c2_ajustes.json").exists() and not (OUT / "c2_ajustes_anterior.json").exists():
        (OUT / "c2_ajustes_anterior.json").write_text((OUT / "c2_ajustes.json").read_text())
    results = {}
    drvs = list(drivers or de.DRIVERS)
    with Pool(min(processes, len(drvs))) as pool:
        for drv, res, lines in pool.imap_unordered(_fit_one, drvs):
            for m in lines:
                log(m)
            results[drv] = res
            json.dump({d: results[d] for d in drvs if d in results}, open(OUT / "c2_ajustes.json", "w"),
                      indent=2, default=float)
    logf.close()
    return results


def _summarise(a, drv, hy, th, r, des):
    """Information criteria, coefficients and smoothed evaporation of a fit."""
    k_hyper = 6 + (len(hy) if hy else 0)
    k_b = r["n_diffuse"]
    aic = -2 * r["ll"] + 2 * (k_hyper + k_b)
    bic = -2 * r["ll"] + np.log(r["nobs"]) * (k_hyper + k_b)
    sm = ee.kalman(des, th, smooth=True)
    ix, _ = ee._index(des)
    se = np.sqrt(np.diag(sm["cov_b"]))
    res = dict(hyper=hy, theta={k: float(v) for k, v in th.items()}, ll=r["ll"], nobs=r["nobs"],
               n_diffuse=k_b, aic=float(aic), bic=float(bic),
               beta=sm["bhat"][ix["beta"]].tolist(), beta_se=se[ix["beta"]].tolist(),
               delta=float(sm["bhat"][ix["delta"]]), delta_se=float(se[ix["delta"]]),
               c_a=float(sm["bhat"][ix["ca"]]), c_a_se=float(se[ix["ca"]]),
               c_b=float(sm["bhat"][ix["cb"]]), c_b_se=float(se[ix["cb"]]))
    # smoothed evaporation; its variance from eta smoothing plus beta uncertainty
    Fb = des.F
    cb = sm["cov_b"][np.ix_(ix["beta"], ix["beta"])]
    var_reg = np.einsum("ti,ij,tj->t", Fb, cb, Fb)
    E = pd.DataFrame({"E_mm_h": sm["E"], "E_reg": sm["E_reg"],
                      "E_sd": np.sqrt(sm["eta_sd"] ** 2 + var_reg),
                      "segment": a["segment"].to_numpy()}, index=a.index)
    return res, E


if __name__ == "__main__":
    main(sys.argv[1:] or None)
