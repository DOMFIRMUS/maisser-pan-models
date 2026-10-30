# -*- coding: utf-8 -*-
"""
Simulation-based check of the state-space estimator (Article 2).

Synthetic records with parameters close to the C2 estimates, built as the logger
builds them (10-min means of ten one-minute readings stored with a 0.1 mm quantum),
and three segments of 800 10-min steps are fitted
with the same code as the data; for every replicate we record the estimates
and whether the 95 % intervals of the linear coefficients cover the truth.

Usage: python recuperacao.py [n_rep]
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import espaco_estados as ee   # noqa: E402

OUT = HERE.parent / "saidas"


def main(n_rep=20):
    rows = []
    t0 = time.time()
    for rep in range(n_rep):
        des, th, beta = ee.simulate_design(n=2400, seed=100 + rep)
        thh, res = ee.fit(des, maxiter=2500)
        r = ee.kalman(des, thh, return_b=True)
        ix, _ = ee._index(des)
        se = np.sqrt(np.diag(r["cov_b"]))
        true_b = {"beta0": beta[0], "beta1": beta[1], "delta": -0.08, "c_a": -0.2, "c_b": 0.15}
        est_b = {"beta0": (r["bhat"][ix["beta"][0]], se[ix["beta"][0]]),
                 "beta1": (r["bhat"][ix["beta"][1]], se[ix["beta"][1]]),
                 "delta": (r["bhat"][ix["delta"]], se[ix["delta"]]),
                 "c_a": (r["bhat"][ix["ca"]], se[ix["ca"]]),
                 "c_b": (r["bhat"][ix["cb"]], se[ix["cb"]])}
        row = dict(rep=rep)
        for k, v in th.items():
            row[f"{k}_true"] = v; row[f"{k}_hat"] = float(thh[k])
        for k, (m, s) in est_b.items():
            row[f"{k}_true"] = float(true_b[k]); row[f"{k}_hat"] = float(m); row[f"{k}_se"] = float(s)
            row[f"{k}_cover"] = bool(abs(m - true_b[k]) <= 1.96 * s)
        rows.append(row)
        print(f"rep {rep}: g_a {thh['g_a']:.3f} c_a {est_b['c_a'][0]:.3f} c_b {est_b['c_b'][0]:.3f} "
              f"delta {est_b['delta'][0]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "recuperacao_sintetica.csv", index=False)
    df = pd.DataFrame(rows)
    summ = {}
    for k in ["g_a", "phi", "s_eta", "s_d", "s_a", "s_b", "beta0", "beta1", "delta", "c_a", "c_b"]:
        tr = df[f"{k}_true"].iloc[0]; h = df[f"{k}_hat"]
        summ[k] = dict(true=float(tr), mean=float(h.mean()), sd=float(h.std(ddof=1)),
                       rel_bias=float((h.mean() - tr) / abs(tr)),
                       coverage=float(df[f"{k}_cover"].mean()) if f"{k}_cover" in df else None)
    json.dump(summ, open(OUT / "recuperacao_resumo.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
