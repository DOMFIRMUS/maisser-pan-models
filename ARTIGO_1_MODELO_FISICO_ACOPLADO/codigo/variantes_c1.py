# -*- coding: utf-8 -*-
"""
Sensitivity of the C1 calibration to the structure of the plate module and of
the observation model: each archived mode is re-evaluated with the current code
and data (micrometer exclusions included) and summarised by its fit and by the
suppression of the covered lamp windows. Writes saidas/tabelas/c1_variantes.csv.

Usage: python variantes_c1.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibracao_c1 as c1      # noqa: E402
import diagnostico_c1 as d1     # noqa: E402

CAL = HERE.parent / "saidas" / "calibracao"
TAB = HERE.parent / "saidas" / "tabelas"
VARIANTS = [
    ("A", "opaque plates, gaps only, free QDY30A gain, micrometer error estimated",
     CAL / "v2_c1_placas_opacas" / "map_c1.json", None),
    ("B", "as A + plate transmittance, micrometer error 0.2 mm", CAL / "map_c1_v3_semCu.json", None),
    ("C", "as B + under-plate evaporation", CAL / "map_c1_v4_Cu_ganholivre.json", None),
    ("D", "as C with the QDY30A gain fixed by the deployment step", CAL / "map_c1_v5_ganho0944.json", 0.944),
    ("E", "as D + gap enhancement, under-plate evaporation per plate version",
     CAL / "map_c1_v8_E_micfixo.json", 0.944),
    ("F", "final: as E with the micrometer error estimated and a multiplier for the low lamp settings",
     CAL / "map_c1.json", 0.944),
    ("G", "as F with the QDY30A gain at its lower limit", CAL / "map_c1_g0884.json", 0.884),
    ("H", "as F with the QDY30A gain at its upper limit", CAL / "map_c1_g1004.json", 1.004),
]


FALL = (pd.Timestamp("2026-08-26 10:17"), pd.Timestamp("2026-08-28 12:29"))


def evaluate(path, gain):
    m = json.load(open(path))
    keep = c1.QDY_GAIN
    c1.QDY_GAIN = gain
    try:
        prob = c1.C1Problem(m["base"], rho=m["rho_used"], weights=m["weights_used"], processes=1)
        P = {k: np.array([v], float) for k, v in m["params"].items()}
        if "m_6" not in P:            # earlier variants: one multiplier for the whole drying-lamp period
            P["m_6"] = P["m_5"].copy(); m["params"]["m_6"] = m["params"]["m_5"]
        pred = prob.predict(P)
        _, parts = prob.loglik(pred, return_parts=True)
        g, c = c1.qdy_coefs(parts)
        row = dict(qdy_gain=g, qdy_thermal=c,
                   mic_rmse=float(np.sqrt(np.mean(parts["micrometer"]["res"][:, 0] ** 2))))
        # fall of the surface between two readings of the V2 segment of 26-28 August
        # (same segment, so the offset cancels), observed and modelled
        t = list(pd.to_datetime(prob.mic.time))
        i1, i2 = t.index(FALL[0]), t.index(FALL[1])
        row["fall_obs"] = float(prob.mic.reading_mm.iloc[i1] - prob.mic.reading_mm.iloc[i2])
        row["fall_model"] = float(pred["mic_h"][i1, 0] - pred["mic_h"][i2, 0])
        for k in c1.PROBES:
            e = prob.obs[k].to_numpy() - pred[k][:, 0]
            v = np.isfinite(e)
            row[f"rmse_{k}"] = float(np.sqrt(np.mean((e[v] - e[v].mean()) ** 2)))
        mm = {"base": m["base"], "params": m["params"]}
        ws, _ = d1.window_runs(mm)
        cov = ws[(ws.cover == 1) & (ws.I_mean > 250)]
        row.update(S_min=float(cov.S_model.min()), S_max=float(cov.S_model.max()), S_mean=float(cov.S_model.mean()))
        return row
    finally:
        c1.QDY_GAIN = keep


def main():
    rows = []
    for tag, desc, path, gain in VARIANTS:
        if not Path(path).exists():
            continue
        r = evaluate(path, gain)
        r.update(variant=tag, description=desc)
        rows.append(r)
        print(tag, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "c1_variantes.csv", index=False)
    return df


if __name__ == "__main__":
    main()
