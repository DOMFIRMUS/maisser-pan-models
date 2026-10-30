# -*- coding: utf-8 -*-
"""
Diagnostic plots of a parameter set against the C2 observations: the four
thermistors, both level transducers (with their fitted nuisance terms) and
the modelled evaporation, segment by segment.

Usage: python diagnostico_ajuste.py [map.json] [out.png]
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibracao as cb      # noqa: E402
import modelo_fisico as mf   # noqa: E402


def main(map_path=None, out_png=None):
    map_path = Path(map_path or cb.OUT / "map.json")
    out_png = Path(out_png or cb.OUT / "diagnostico_map.png")
    m = json.load(open(map_path))
    prob = cb.C2Problem(rho=m.get("rho"))
    P = {**mf.default_params(1), **prob.prior.to_params(np.array(m["u"])[None])}
    pred = prob.predict(P)
    ll, parts = prob.loglik(pred, return_parts=True)
    o = prob.obs
    segs = prob.names
    fig, axs = plt.subplots(4, 1, figsize=(20, 15), sharex=True)
    t = o.index
    colors = dict(Tw_cs="C0", Tw_04="C1", Tw_12="C2", Tw_20="C3")
    for k in cb.T_STREAMS:
        axs[0].plot(t, o[k], ".", ms=1.5, color=colors[k], alpha=0.5)
        axs[0].plot(t, pred[k][:, 0], "-", lw=0.8, color=colors[k], label=k)
        axs[1].plot(t, o[k] - pred[k][:, 0], ".", ms=1.5, color=colors[k], label=k)
    axs[0].legend(ncol=4); axs[0].set_ylabel("T (degC)")
    axs[1].axhline(0, color="k", lw=0.5); axs[1].set_ylabel("obs - model (K)"); axs[1].legend(ncol=4)
    import pandas as pd
    segd = pd.get_dummies(o.seg).to_numpy(float)
    base = np.column_stack([segd, o.Tcs_c, o.t_days])
    ba = parts["level_a_qc"]["beta"][0]
    bb = parts["level_b_qc"]["beta"][0]
    fa = np.column_stack([base, pred["dH_mm"][:, 0]]) @ ba
    fb = pred["dH_mm"][:, 0] + base @ bb
    axs[2].plot(t, o.level_a_qc - fa, ".", ms=1.5, label=f"ODY resid (gain {ba[-1]:.3f}, c {ba[-3]:.3f} mm/K, d {ba[-2]:.3f} mm/d)")
    axs[2].plot(t, o.level_b_qc - fb, ".", ms=1.5, label=f"CS456 resid (c {bb[-2]:.3f} mm/K, d {bb[-1]:.3f} mm/d)")
    axs[2].axhline(0, color="k", lw=0.5); axs[2].legend(); axs[2].set_ylabel("level resid (mm)")
    axs[3].plot(t, pred["E"][:, 0] * 3600, "k-", lw=0.8)
    axs[3].set_ylabel("E model (mm/h)")
    for ax in axs:
        for s in prob.segs:
            ax.axvline(s.start, color="grey", lw=0.5, ls=":")
    plt.tight_layout()
    plt.savefig(out_png, dpi=55)
    rms = {k: float(np.sqrt(np.nanmean((o[k] - pred[k][:, 0]) ** 2))) for k in cb.T_STREAMS}
    rms["level_a"] = float(np.sqrt(np.nanmean((o.level_a_qc - fa) ** 2)))
    rms["level_b"] = float(np.sqrt(np.nanmean((o.level_b_qc - fb) ** 2)))
    print("RMS:", {k: round(v, 3) for k, v in rms.items()})
    print("ODY beta tail (Tcs, t, gain):", ba[-3:], " CS456 tail (Tcs, t):", bb[-2:])
    print("total E (mm) by segment:")
    for s in prob.segs:
        mk = (o.seg == s.name).to_numpy()
        print(f"  {s.name}: {np.nansum(pred['E'][mk, 0]) * 600:.1f} mm")
    return rms


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
