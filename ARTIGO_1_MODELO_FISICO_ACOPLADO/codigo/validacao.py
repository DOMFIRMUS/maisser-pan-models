# -*- coding: utf-8 -*-
"""
Split-sample validation of the uncovered-pan model (campaign C2).

The model is recalibrated (differential evolution, rho and tempering weights
fixed at the full-data values) without two segments, and the held-out
segments are predicted with the recalibrated mode. Skill of the prediction is
compared with the in-sample fit of the same segments: RMSE of each
thermistor, and the evaporation implied by the CS456 (gain 1, drift and
offset fitted to the held-out record itself, i.e. only the shape of the
level change is predicted).

Usage: python validacao.py [S3,S5]
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibracao as cb      # noqa: E402
import dados_modelo as dm    # noqa: E402
import modelo_fisico as mf   # noqa: E402

OUT = cb.OUT


def rmse_by_segment(prob, pred, names):
    rows = []
    o = prob.obs
    for k in cb.T_STREAMS:
        y = o[k].to_numpy()
        for s in names:
            m = prob.valid[k] & (prob.seg_of == s)
            if m.sum() == 0:
                continue
            r = y[m] - pred[k][m, 0]
            rows.append(dict(segment=s, stream=k, rmse=float(np.sqrt(np.mean((r - r.mean()) ** 2))),
                             bias=float(r.mean()), n=int(m.sum())))
    return pd.DataFrame(rows)


def fit(hold=("C2-S3", "C2-S5"), gens=100, walkers=40):
    """Recalibrate on the training segments (differential evolution)."""
    full = json.load(open(OUT / "map_c2.json"))
    segs = dm.c2_segments()
    train = [s for s in segs if s.name not in hold]
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    with cb.C2Problem(segs=train, rho=full["rho_used"], weights=full["weights_used"]) as prob:
        log(f"training on {prob.names}")
        start = OUT / "mode_c2_posterior.json"
        u0 = np.array(json.load(open(start))["u"] if start.exists() else full["u"])
        X, fx, _ = cb.differential_evolution(prob, pop=walkers, gens=gens, log=log, x_init=u0[None])
        best = X[np.argmax(fx)]
    json.dump({"u": best.tolist(), "names": prob.prior.names, "held_out": list(hold),
               "params": {k: float(v[0]) for k, v in prob.prior.to_params(best[None]).items()}},
              open(OUT / "map_c2_validacao.json", "w"), indent=2)


def main(hold=("C2-S3", "C2-S5"), refit=True):
    if refit or not (OUT / "map_c2_validacao.json").exists():
        fit(hold)
    full = json.load(open(OUT / "map_c2.json"))
    best = np.array(json.load(open(OUT / "map_c2_validacao.json"))["u"])
    segs = dm.c2_segments()
    # predictions of all segments with the training mode and with the full mode
    res = []
    with cb.C2Problem(segs=segs, rho=full["rho_used"], weights=full["weights_used"]) as allp:
        for tag, u in (("training_mode", best), ("full_mode", np.array(full["u"]))):
            pred = allp.predict(allp.params_from_u(u[None]))
            df = rmse_by_segment(allp, pred, list(hold))
            df["fit"] = tag
            res.append(df)
            # level: CS456 with its own offset, thermal term and drift fitted on the held-out record
            for s in hold:
                m = allp.valid["level_b_qc"] & (allp.seg_of == s)
                y = allp.obs.level_b_qc.to_numpy()[m]
                dH = cb.K_CS * pred["dH_mm"][m, 0]
                t = allp.obs.t_seg_d.to_numpy()[m]; Tc = allp.obs.Tcs_c.to_numpy()[m]
                Xd = np.column_stack([np.ones(m.sum()), Tc, t])
                beta, *_ = np.linalg.lstsq(Xd, y - dH, rcond=None)
                r = y - dH - Xd @ beta
                res.append(pd.DataFrame([dict(segment=s, stream="level_b_qc", rmse=float(np.sqrt(np.mean(r ** 2))),
                                              bias=0.0, n=int(m.sum()), fit=tag,
                                              E_model_mm=float(-(pred["dH_mm"][m, 0][-1] - pred["dH_mm"][m, 0][0])))]))
    out = pd.concat(res)
    out.to_csv(OUT / "validacao_split.csv", index=False)
    print(out.round(3).to_string())
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    hold = tuple(f"C2-{x}" for x in args[0].split(",")) if args else ("C2-S3", "C2-S5")
    if "--fit-only" in sys.argv:
        fit(hold)
    else:
        main(hold, refit="--score-only" not in sys.argv)
