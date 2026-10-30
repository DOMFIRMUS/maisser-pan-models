# -*- coding: utf-8 -*-
"""
Leave-one-segment-out cross-validation of the evaporation drivers (C2).

For each driver, with the hyperparameters of its full-data fit, the shared
coefficients (beta, CS456 drift slope, thermal coefficients) are re-estimated
with one segment's observations removed, and the held-out segment is scored
by its exact diffuse one-step-ahead predictive log density, conditional on the
training coefficients and integrating out the segment's own sensor offsets
(espaco_estados.conditional_score). The sum over folds is the out-of-sample
log score; differences between drivers are reported with their fold-wise
standard error.

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
sys.path.insert(0, str(HERE))
import espaco_estados as ee  # noqa: E402
import ajuste_c2 as aj       # noqa: E402

OUT = HERE.parent / "saidas"


def loso_scores(a, driver, fit):
    des = aj.design_c2(a, driver, fit.get("hyper"))
    th = {k: float(v) for k, v in fit["theta"].items()}
    full = ee.kalman(des, th, per_seg=True)
    rows = []
    for j in range(des.S):
        tr = ee.kalman(des, th, return_b=True, mask_segs=[j])
        sc, n = ee.conditional_score(des, full["acc"][j], j, tr["bhat"])
        ix, _ = ee._index(des)
        rows.append(dict(driver=driver, fold=j, score=sc, n=n,
                         beta=tr["bhat"][ix["beta"]].tolist()))
    return pd.DataFrame(rows)


def main():
    fits = json.load(open(OUT / "c2_ajustes.json"))
    a = aj.prepare_c2()
    allrows = []
    for drv, fit in fits.items():
        df = loso_scores(a, drv, fit)
        allrows.append(df)
        print(f"{drv:26s} LOSO log score {df.score.sum():10.2f}  (n={df.n.sum()})", flush=True)
    res = pd.concat(allrows)
    res.to_csv(OUT / "c2_validacao_cruzada_folds.csv", index=False)
    tot = res.groupby("driver").score.sum().sort_values(ascending=False)
    best = tot.index[0]
    piv = res.pivot(index="fold", columns="driver", values="score")
    rows = []
    for d in tot.index:
        diff = piv[d] - piv[best]
        rows.append(dict(driver=d, loso_score=tot[d], diff_to_best=diff.sum(),
                         diff_se=float(np.sqrt(len(diff)) * diff.std(ddof=1)) if d != best else 0.0,
                         aic=fits[d]["aic"], bic=fits[d]["bic"], ll=fits[d]["ll"]))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "c2_validacao_cruzada_resumo.csv", index=False)
    print(summ.round(2).to_string())
    return summ


if __name__ == "__main__":
    main()
