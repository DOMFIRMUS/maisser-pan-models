# -*- coding: utf-8 -*-
"""
Campaign C1: direct suppression of every covered window from the level records.

Model. The evaporation of the C1 pan is written as
    E_t = kappa_g(t) * z_t + eta_t,      z_t = f_t' beta_C2
where z_t is the evaporation an UNCOVERED pan would have at the measured
surface temperature and air state of minute t, from the driver selected and
calibrated on campaign C2, and kappa_g is one coefficient per group g:
uncovered C1 time (kappa_C1, absorbing the apparatus differences between
campaigns), each covered rate window W02...W15, and the remaining covered time.
The direct suppression of window j is S_j = 1 - kappa_j / kappa_C1: the
fraction of evaporation removed at the same surface temperature (the thermal
feedback of the cover on the water is not included; see contrafactual.py).

Observations: the QDY30A (free gain, thermal coefficient on the bulk water
temperature, segment offsets) and the manual micrometer readings of the
surface height (gain 1 by construction, segment offsets, corrected for the
thermal expansion of the column), which anchor the scale.

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
sys.path.insert(0, str(HERE.parents[1] / "_comum"))
from tanque import config as cfg, propriedades as pr   # noqa: E402

OUT = HERE.parent / "saidas"
C1_START, C1_END = "2026-08-17 09:00", "2026-09-03 17:15"
H_NOM_MM = 220.0


def prepare_c1():
    a, ev = de.series_10min("C1")
    a = a[(a.index >= C1_START) & (a.index <= C1_END)].copy()
    a = a[a["n_plates"].notna()]
    # segments break at level events and at every change of cover
    key = a["n_plates"].round().astype(int).astype(str) + "|" + a["version"].fillna("?").astype(str)
    new = (a["segment"] != a["segment"].shift()) | (key != key.shift())
    gap = a.index.to_series().diff() > pd.Timedelta(minutes=10)
    a["seg_c1"] = (new | gap).cumsum().to_numpy()
    cnt = a.groupby("seg_c1")["level_a_qc"].count()
    a = a[a["seg_c1"].isin(cnt[cnt >= 24].index)]
    a["seg_code"] = pd.factorize(a["seg_c1"])[0]
    # micrometer readings -> 10-min bins, corrected for thermal expansion
    m = pd.read_csv(cfg.MICROMETER_C1)
    m["time"] = pd.to_datetime(m["date"] + " " + m["time"])
    bad = m["note"].fillna("").str.contains("non-monotonic|refill|buoyancy|deployed|removed", case=False)
    # first reading of the campaign (installation of the stilling well): 1.8 mm above
    # the trend of the following ten readings of the same day
    bad |= m["time"] == pd.Timestamp("2026-08-17 09:21")
    # a 27-min pair at the end of a run that disagrees with the transducer by 0.4 mm, and a
    # reading referring to a plate reinstallation that the transducer does not show
    bad |= m["time"].isin(pd.to_datetime(["2026-08-20 17:51", "2026-08-20 18:18", "2026-09-01 16:20"]))
    m = m[~bad]
    a["mic"] = np.nan
    for _, r in m.iterrows():
        b = r["time"].floor("10min")
        if b in a.index:
            Tb = a.at[b, "Tb"]
            corr = H_NOM_MM * (997.0 / pr.water_density(Tb) - 1.0)
            a.at[b, "mic"] = r["reading_mm"] - corr
    # groups for kappa: each covered window; the uncovered rate windows pooled by
    # fan regime (the like-for-like references); the remaining time
    a["grupo"] = np.where(a["phi"] > 0, "cov_other", "uncovered")
    for w in cfg.WINDOWS_C1:
        wid, t0, t1, cov, regime = w[0], pd.Timestamp(w[1]), pd.Timestamp(w[2]), w[3], w[6]
        sel = (a.index >= t0) & (a.index <= t1)
        if cov:
            a.loc[sel & (a["phi"] > 0), "grupo"] = wid
        else:
            a.loc[sel & (a["phi"] == 0), "grupo"] = f"unc_{regime}"
    return a


# reference group of each covered window: uncovered windows with the same fan
# regime; C1 has no uncovered calm window, so calm windows are referred to the
# strong-fan windows (the C2 formulation carries the effect of the fan)
REFERENCE = {"weak": "unc_weak", "strong": "unc_strong", "calm": "unc_strong"}


def z_regressor(a, c2fit, driver, s_U=1.0):
    """Uncovered-pan evaporation (mm h-1) from the C2 driver and coefficients."""
    b = a.copy()
    b["Ucl"] = b["Ucl"] * s_U
    F = de.driver_matrix(b, driver, c2fit.get("hyper"))
    return F @ np.asarray(c2fit["beta"])


def design_c1(a, z):
    groups = ["uncovered", "unc_strong", "unc_weak"] +         [w[0] for w in cfg.WINDOWS_C1 if w[3] and (a["grupo"] == w[0]).any()] + ["cov_other"]
    F = np.column_stack([z * (a["grupo"] == g).to_numpy() for g in groups])
    x = (a["Tb"].interpolate(limit_direction="both") - 30.0).to_numpy()
    des = ee.Design(ya=a["level_a_qc"].to_numpy(), yb=a["mic"].to_numpy(), x=x, F=F,
                    seg=a["seg_code"].to_numpy(), dt_h=de.DT_H, has_b=True, drift=False,
                    xb=np.zeros(len(a)))
    return des, groups


S_U_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
# prior of the anemometer scale, as in the companion paper (log-normal, median 1,
# log-sd 0.3: the anemometer sat 0.09 m above the water in C1)
S_U_PRIOR_SD = 0.30


def fit_c1(a, c2fit, driver, s_U_grid=S_U_GRID, log=print):
    """State-space fit for each value of the C1 anemometer scale s_U; the
    suppression is later averaged over s_U with weights proportional to the
    likelihood concentrated over the regression coefficients (invariant to the
    scale of z, which grows with s_U) times the prior of s_U.
    Returns a list of (s_U, theta, filter, design, groups, z)."""
    fits = []
    u0 = ee.pack(dict(g_a=1.1, phi=0.9, s_eta=0.03, s_d=1e-3, s_a=0.1, s_b=0.2))
    for sU in s_U_grid:
        z = z_regressor(a, c2fit, driver, sU)
        des, groups = design_c1(a, z)
        th, res = ee.fit(des, u0=u0, fixed=dict(s_d=1e-6), maxiter=2500)
        r = ee.kalman(des, th, return_b=True)
        log(f"  s_U={sU}: ll={r['ll']:.2f} ll_prof={r['ll_prof']:.2f} g_a={th['g_a']:.3f} phi={th['phi']:.3f} "
            f"s_eta={th['s_eta']:.4f} s_a={th['s_a']:.3f} s_mic={th['s_b']:.3f}")
        fits.append((sU, th, r, des, groups, z))
    return fits


def suppression_table(r, des, groups, n_draw=20000, seed=7):
    """S_j = 1 - kappa_j / kappa_ref(j) with its sampling distribution; the
    reference is the uncovered windows of the same fan regime (REFERENCE), and
    the pooled uncovered time is reported as a sensitivity case (S_pooled)."""
    ix, _ = ee._index(des)
    bi = ix["beta"]
    mu = r["bhat"][bi]; C = r["cov_b"][np.ix_(bi, bi)]
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mu, C, size=n_draw)
    regime = {w[0]: w[6] for w in cfg.WINDOWS_C1}
    i_pool = groups.index("uncovered")
    q = lambda S: dict(S_direct=float(np.median(S)), S_mean=float(S.mean()), S_sd=float(S.std()),
                       S_lo=float(np.percentile(S, 2.5)), S_hi=float(np.percentile(S, 97.5)))
    rows = []
    for j, g in enumerate(groups):
        if g in ("uncovered", "unc_strong", "unc_weak"):
            rows.append(dict(grupo=g, kappa=mu[j], kappa_se=np.sqrt(C[j, j]), referencia=""))
            continue
        ref = REFERENCE.get(regime.get(g, "strong"), "unc_strong")
        i_ref = groups.index(ref)
        row = dict(grupo=g, kappa=mu[j], kappa_se=np.sqrt(C[j, j]), referencia=ref)
        row.update(q(1 - draws[:, j] / draws[:, i_ref]))
        Sp = 1 - draws[:, j] / draws[:, i_pool]
        row.update(S_pooled=float(np.median(Sp)), S_pooled_lo=float(np.percentile(Sp, 2.5)),
                   S_pooled_hi=float(np.percentile(Sp, 97.5)))
        rows.append(row)
    return pd.DataFrame(rows), mu[i_pool], np.sqrt(C[i_pool, i_pool])


def main(driver=None, suffix=""):
    c2 = json.load(open(OUT / "c2_ajustes.json"))
    if driver is None:
        cv = OUT / "c2_validacao_cruzada_resumo.csv"
        if cv.exists():
            driver = pd.read_csv(cv).sort_values("loso_score", ascending=False).driver.iloc[0]
        else:
            driver = min(c2, key=lambda k: c2[k]["aic"])
    logf = open(OUT / f"c1_ajuste_log{suffix}.txt", "w", encoding="utf-8")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    a = prepare_c1()
    log(f"C1: {len(a)} steps, {a.seg_code.nunique()} segments, QDY obs {a.level_a_qc.notna().sum()}, "
        f"micrometer obs {a.mic.notna().sum()}; driver {driver}")
    fits = fit_c1(a, c2[driver], driver, log=log)
    lls = np.array([f[2]["ll_prof"] for f in fits])
    lpr = np.array([-0.5 * (np.log(f[0]) / S_U_PRIOR_SD) ** 2 for f in fits])
    lpost = lls + lpr
    w = np.exp(lpost - lpost.max()); w /= w.sum()
    ib = int(np.argmax(lpost))
    tabs = []
    for (sU, th, r, des, groups, z), wk in zip(fits, w):
        sm_k = ee.kalman(des, th, smooth=True)
        t_k, _, _ = suppression_table(sm_k, des, groups, n_draw=4000)
        t_k["s_U"] = sU; t_k["weight"] = wk
        tabs.append(t_k)
    prof = pd.concat(tabs)
    prof.to_csv(OUT / f"c1_supressao_perfil_sU{suffix}.csv", index=False)
    # suppression averaged over the s_U profile (mixture of the per-s_U sampling distributions)
    rng = np.random.default_rng(11)
    marg = {}
    for g in prof.grupo.unique():
        sub = prof[(prof.grupo == g) & prof.S_mean.notna()]
        if not len(sub):
            continue
        draws = np.concatenate([rng.normal(r_.S_mean, r_.S_sd, int(round(20000 * r_.weight)))
                                for _, r_ in sub.iterrows() if r_.weight > 1e-4])
        marg[g] = dict(S_marg=float(np.median(draws)), S_marg_lo=float(np.percentile(draws, 2.5)),
                       S_marg_hi=float(np.percentile(draws, 97.5)))
    sU, th, r, des, groups, z = fits[ib]
    sm = ee.kalman(des, th, smooth=True)
    tab, k0, k0se = suppression_table(sm, des, groups)
    tab = tab.merge(pd.DataFrame([dict(grupo=g, **v) for g, v in marg.items()]), on="grupo", how="left")
    ix, _ = ee._index(des)
    se = np.sqrt(np.diag(sm["cov_b"]))
    res = dict(driver=driver, s_U=sU, s_U_profile=[[float(f[0]), float(f[2]["ll"])] for f in fits],
               s_U_profile_conc=[[float(f[0]), float(f[2]["ll_prof"])] for f in fits],
               s_U_logprior=lpr.tolist(), s_U_weights=w.tolist(),
               s_U_mean=float(np.sum(w * np.array([f[0] for f in fits]))), theta={k: float(v) for k, v in th.items()}, ll=r["ll"],
               kappa_uncovered=float(k0), kappa_uncovered_se=float(k0se),
               kappa_ref={g: [float(sm["bhat"][ix["beta"]][groups.index(g)]),
                              float(np.sqrt(sm["cov_b"][ix["beta"], ix["beta"]][groups.index(g)]))]
                          for g in ("unc_strong", "unc_weak")},
               reference=REFERENCE,
               c_a=float(sm["bhat"][ix["ca"]]), c_a_se=float(se[ix["ca"]]),
               suppression=tab.to_dict(orient="records"),
               groups=groups, kappa_mean=sm["bhat"][ix["beta"]].tolist(),
               kappa_cov=sm["cov_b"][np.ix_(ix["beta"], ix["beta"])].tolist())
    json.dump(res, open(OUT / f"c1_supressao_direta{suffix}.json", "w"), indent=2, default=float)
    tab.to_csv(OUT / f"c1_supressao_direta{suffix}.csv", index=False)
    pd.DataFrame({"E_mm_h": sm["E"], "z_uncovered_mm_h": z, "grupo": a["grupo"].to_numpy(),
                  "eta_sd": sm["eta_sd"]}, index=a.index).to_csv(OUT / f"c1_E_suavizada{suffix}.csv")
    log(f"best s_U {sU}; profile weights {np.round(w, 3).tolist()}")
    log(f"kappa (dark-period uncovered) = {k0:.3f} +- {k0se:.3f}; QDY gain {th['g_a']:.3f}; c_a {res['c_a']:.3f} mm/K")
    log(tab.round(3).to_string())
    logf.close()
    return res


if __name__ == "__main__":
    drv = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "best" else None
    main(drv, suffix=sys.argv[2] if len(sys.argv) > 2 else "")
