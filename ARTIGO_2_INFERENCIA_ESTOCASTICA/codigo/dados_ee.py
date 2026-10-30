# -*- coding: utf-8 -*-
"""
Data for the state-space analysis (Article 2): 10-min series of both
campaigns with segment ids, surface-temperature proxies and the physical
drivers of evaporation.

Surface temperature proxy T_s
  C2: the 20 cm thermistor (2-5 cm below the surface) whenever it is
      submerged by at least 12 mm (config.SUBMERGED_MIN), otherwise the mean of
      the 12 cm thermistor and the CS450 body (the water is mixed then).
  C1: thermistor 2, the near-surface probe (position undocumented).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_comum"))
from tanque import dados, eventos, config as cfg, propriedades as pr   # noqa: E402

STEP = "10min"
DT_H = 10.0 / 60.0


def series_10min(campaign, guard_min=20):
    """10-min means with segment ids (segments break at every level step)."""
    df = dados.load_all()
    q, ev = eventos.qc(df)
    c = q[q.campaign == campaign].copy()
    if campaign == "C2":
        c = c[(c.index >= cfg.C2_START) & (c.index <= cfg.C2_END)]
    cols = ["Rs_sp", "Rs_li", "U", "Ta", "RH", "ea", "VPD", "phi", "n_plates",
            "level_a_qc", "level_b_qc", "segment", "lamp_on", "fan_on"]
    cols += [k for k in ("Tw_cs", "Tw_04", "Tw_12", "Tw_20", "Tw_p1", "Tw_p2", "Tw_p3") if k in c]
    c = c[[k for k in cols if k in c]]
    c["lamp_on"] = c["lamp_on"].astype(float)
    c["fan_on"] = c["fan_on"].astype(float)
    agg = c.resample(STEP).mean()
    agg["segment"] = c["segment"].resample(STEP).max()
    agg["n_level"] = c["level_a_qc"].resample(STEP).count()
    agg["version"] = q.loc[c.index, "version"].resample(STEP).first() if "version" in q else None
    # steps and their guard windows split segments: a 10-min bin that holds
    # minutes from two segments, or no valid level, is dropped
    seg_min = c["segment"].resample(STEP).min()
    agg.loc[seg_min != agg["segment"], ["level_a_qc", "level_b_qc"]] = np.nan
    agg["campaign"] = campaign
    agg = add_drivers(agg, campaign)
    return agg, ev[ev.campaign == campaign]


ODY_GAIN_SS = 1.30


def surface_temperature(a, campaign):
    if campaign == "C2":
        Tb = a[["Tw_12", "Tw_cs"]].mean(axis=1)
        # submergence of the 20-cm probe from the ODY depth, with the ODY gain
        # estimated in this paper (ODY_GAIN_SS) rather than the nominal value
        sub = cfg.corrected_depth_c2(a["level_a_qc"].ffill(), gain=ODY_GAIN_SS) - 0.20 >= cfg.SUBMERGED_MIN
        Ts = a["Tw_20"].where(sub, Tb)
        return Ts.fillna(Tb), Tb
    Tb = a[["Tw_p1", "Tw_p3"]].mean(axis=1)
    return a["Tw_p2"].fillna(Tb), Tb


def add_drivers(a, campaign):
    a = a.copy()
    Ts, Tb = surface_temperature(a, campaign)
    a["Ts"] = Ts
    a["Tb"] = Tb
    ea = a["ea"]
    es = pr.esat(Ts)
    a["dE_s"] = (es - ea) / 1000.0                                  # kPa
    a["VPD_k"] = a["VPD"] / 1000.0
    rv_s = pr.vapour_density(es, Ts)
    rv_a = pr.vapour_density(ea, a["Ta"])
    a["drho_v"] = (rv_s - rv_a) * 1000.0                           # g m-3
    Tvs = pr.virtual_temperature(Ts, es)
    Tva = pr.virtual_temperature(a["Ta"], ea)
    a["dTv"] = np.maximum(np.asarray(Tvs - Tva, float), 0.0)
    a["I"] = a["Rs_sp"].clip(lower=0).fillna(0) / 1000.0           # kW m-2 (index)
    a["Ucl"] = a["U"].clip(lower=0).fillna(0)
    return a


def driver_matrix(a, name, hyper=None):
    """Regressors f_t (n x p) of the evaporation rate E_t = f_t' beta (mm h-1).

    The candidate set reproduces the families of Section 4 of the review.
    `hyper` holds nonlinear hyperparameters (wind exponent n, Churchill-Usagi
    weight k) for the forms that need them.
    """
    h = hyper or {}
    U = a["Ucl"].to_numpy()
    dr = a["drho_v"].to_numpy()
    dTv = a["dTv"].to_numpy()
    dEs = a["dE_s"].to_numpy()
    vpd = a["VPD_k"].to_numpy()
    I = a["I"].to_numpy()
    n = h.get("n", 0.8)
    if name == "constante":
        F = np.ones((len(a), 1))
    elif name == "dalton_ar":
        F = np.column_stack([vpd, U * vpd])
    elif name == "dalton_superficie":
        F = np.column_stack([dEs, U * dEs])
    elif name == "penman_indice":
        F = np.column_stack([I, vpd, U * vpd])
    elif name == "ryan_harleman":
        F = np.column_stack([dr * np.cbrt(dTv), dr * U])
    elif name == "livre_forcada_potencia":
        F = np.column_stack([dr * np.cbrt(dTv), dr * np.power(U, n), dr])
    elif name == "churchill_usagi":
        k = h.get("k", 1.0)
        F = (dr * np.cbrt(dTv + (k * np.power(U, n)) ** 3))[:, None]
    elif name == "churchill_usagi_fundo":
        k = h.get("k", 1.0)
        F = np.column_stack([dr * np.cbrt(dTv + (k * np.power(U, n)) ** 3), dr])
    elif name == "penman1948_fixo":
        # Penman (1948) wind function in mm d-1 kPa-1 applied to the surface
        # difference, a single free multiplier
        F = ((2.6 * (1 + 0.54 * U)) * dEs / 24.0)[:, None]
    elif name == "carrier_fixo":
        F = (((0.089 + 0.0782 * U) / 2430.0) * dEs * 3600.0)[:, None]
    else:
        raise ValueError(name)
    F = np.nan_to_num(F, nan=0.0)
    return F


DRIVERS = ["constante", "dalton_ar", "dalton_superficie", "penman_indice", "ryan_harleman",
           "livre_forcada_potencia", "churchill_usagi", "churchill_usagi_fundo",
           "penman1948_fixo", "carrier_fixo"]
