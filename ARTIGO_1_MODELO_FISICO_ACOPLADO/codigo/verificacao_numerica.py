# -*- coding: utf-8 -*-
"""
Numerical verification of the pan model (Article 1).

  * energy closure of the water column, step by step:
    R_abs + LW + H + LE + wall + floor + G_plate - storage = 0,
    for an uncovered C2 segment and a covered C1 segment;
  * time-step convergence: the 300-s solution against 60-s and 20-s
    solutions (forcing held constant within each 5-min interval), compared at
    the thermistor heights and in cumulative evaporation;
  * grid convergence: 24 against 48 layers with the same top-layer thickness.

Writes saidas/tabelas/verificacao_numerica.json.

Usage: python verificacao_numerica.py
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import modelo_fisico as mf      # noqa: E402
import dados_modelo as dm       # noqa: E402
import calibracao as cb         # noqa: E402

CAL = cb.OUT
TAB = HERE.parent / "saidas" / "tabelas"
WATER = ["Rabs_water", "LW_net", "H_sens", "LE", "wall", "floor", "G_plate"]


def params_c2():
    m = json.load(open(CAL / "map_c2.json"))
    P = mf.default_params(1)
    P.update({k: np.array([v]) for k, v in m["params"].items()})
    return P


def refine(F, r, dt):
    """Forcing on a step dt/r, piecewise constant within each original step."""
    rep = lambda a: None if a is None else np.repeat(np.asarray(a), r)
    return mf.Forcing(t=np.arange(len(F.t) * r) * dt / r, I=rep(F.I), U=rep(F.U), Ta=rep(F.Ta),
                      ea=rep(F.ea), phi=rep(F.phi), cfg=rep(F.cfg),
                      jumps={k * r: v for k, v in F.jumps.items()})


def closure(out):
    res = sum(out.fluxes[k] for k in WATER) - out.fluxes["storage"]
    scale = np.abs(out.fluxes["Rabs_water"]).max()
    return float(np.abs(res).max()), float(scale)


def run(seg, P, N=24, dt=300.0, r=1, store=False):
    s_ = mf.sigma_grid(N); zc = np.cumsum(s_) - s_ / 2
    F = dm.forcing_arrays(seg)
    if r > 1:
        F = refine(F, r, dt)
    return mf.simulate(F, {k: v.copy() for k, v in P.items()}, dm.initial_profile(seg, zc), seg.H0,
                       N=N, dt=dt / r, store_fluxes=store)


def compare(ref, out, r_ref, r_out=1):
    """RMS differences at the probe heights on the common 5-min grid."""
    res = {}
    for k, h in dm.PROBES_C2.items():
        if k == "Tw_cs":
            continue
        a, da = mf.temperature_at_height(ref, h)
        b, db = mf.temperature_at_height(out, h)
        a = a[r_ref - 1::r_ref, 0]; b = b[r_out - 1::r_out, 0]
        ok = np.isfinite(a) & np.isfinite(b)
        res[k] = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
    Ea = ref.E[:, 0].sum() * (300.0 / r_ref); Eb = out.E[:, 0].sum() * (300.0 / r_out)
    res["E_rel_pct"] = float(100 * (Eb - Ea) / Ea)
    res["Ts_rms"] = float(np.sqrt(np.mean((ref.Ts[r_ref - 1::r_ref, 0] - out.Ts[r_out - 1::r_out, 0]) ** 2)))
    return res


def main():
    P = params_c2()
    segs = {s.name: s for s in dm.c2_segments()}
    seg = segs["C2-S5"]
    out = {"segment": seg.name, "days": float(len(seg.forcing) * 300 / 86400)}
    o300 = run(seg, P, store=True)
    out["closure_uncovered_W_m2"], out["Rabs_max_W_m2"] = closure(o300)
    o60 = run(seg, P, r=5)
    o20 = run(seg, P, r=15)
    out["dt300_vs_dt20"] = compare(o20, o300, 15)
    out["dt60_vs_dt20"] = compare(o20, o60, 15, 5)
    o48 = run(seg, P, N=48)
    out["N24_vs_N48"] = compare(o48, o300, 1)
    # covered C1 segment with the C1 mode
    import calibracao_c1 as c1
    m1 = json.load(open(CAL / "map_c1.json"))
    P1 = {k: np.array([v], float) for k, v in m1["base"].items()}
    P1.update({k: np.array([v], float) for k, v in m1["params"].items()})
    segs1 = [s for s in c1.micrometer_segments(dm.c1_segments()) if s.phi > 0]
    s1 = max(segs1, key=lambda s: len(s.forcing))
    c1.version_params(P1, s1.version)
    s_ = mf.sigma_grid(c1.N_LAYERS); zc = np.cumsum(s_) - s_ / 2
    oc = mf.simulate(dm.forcing_arrays(s1), P1, c1.initial_profiles(s1, P1, zc), s1.H0,
                     N=c1.N_LAYERS, store_fluxes=True)
    out["covered_segment"] = s1.name
    out["closure_covered_W_m2"], _ = closure(oc)
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(TAB / "verificacao_numerica.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
