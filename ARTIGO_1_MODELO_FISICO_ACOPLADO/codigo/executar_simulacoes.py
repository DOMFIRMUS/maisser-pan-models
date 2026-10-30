# -*- coding: utf-8 -*-
"""
Runs the scenario analyses of Article 1 and writes their outputs:

  saidas/simulacoes/mapa_supressao.csv     S, E and temperatures on phi x U x I
  saidas/simulacoes/shapley_grade.csv      Shapley attribution on a scenario grid
  saidas/simulacoes/sobol.csv              Sobol indices of S and of saved depth
  saidas/simulacoes/incerteza_posterior.csv  S(phi) bands from posterior draws

Usage: python executar_simulacoes.py [mapa shapley sobol posterior]
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulacoes as sm      # noqa: E402

CAL = HERE.parent / "saidas" / "calibracao"
OUT = HERE.parent / "saidas" / "simulacoes"
OUT.mkdir(parents=True, exist_ok=True)


def load_base():
    c2 = json.load(open(CAL / "map_c2.json")); c1 = json.load(open(CAL / "map_c1.json"))
    return sm.base_params(c2, c1), c2, c1


def mapa(base):
    phis = np.round(np.arange(0.25, 0.951, 0.05), 3)
    Us = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    rows = []
    for I in (300.0, 520.0, 820.0, 1000.0):
        P, U = np.meshgrid(phis, Us, indexing="ij")
        r = sm.run_design(base, np.full(P.size, I), U.ravel(), P.ravel(), return_fields=True)
        for i in range(P.size):
            rows.append(dict(I=I, phi=P.ravel()[i], U=U.ravel()[i], **{k: float(v[i]) for k, v in r.items()}))
        print(f"  mapa I={I} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mapa_supressao.csv", index=False)
    return df


def shapley(base):
    phis = np.array([0.25, 0.50, 0.74, 0.90])
    Us = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    Is = np.array([300.0, 520.0, 820.0])
    G = np.array(np.meshgrid(phis, Us, Is, indexing="ij")).reshape(3, -1)
    rows = []
    for k0 in range(0, G.shape[1], 20):
        g = G[:, k0:k0 + 20]
        res = sm.shapley_design(base, g[2], g[1], g[0])
        for i in range(g.shape[1]):
            row = dict(phi=g[0, i], U=g[1, i], I=g[2, i], S_total=float(res["S_total"][i]))
            for m in sm.MECHS:
                row[m] = float(res[m][i]); row["single_" + m] = float(res["single"][m][i])
            rows.append(row)
        print(f"  shapley {k0 + g.shape[1]}/{G.shape[1]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "shapley_grade.csv", index=False)
    return df


def sobol(base, n=4096, seed=11):
    rng = np.random.default_rng(seed)
    bounds = [(lo, hi) for _, lo, hi in sm.SOBOL_INPUTS]
    A, B, AB = sm.saltelli_matrices(n, bounds, rng)
    X = np.vstack([A, B] + [AB[i] for i in range(len(bounds))])
    t0 = time.time()
    S, saved = sm.sobol_eval(base, X)
    print(f"  sobol {len(X)} runs in {time.time() - t0:.0f} s", flush=True)
    k = len(bounds)
    rows = []
    for name, y in (("S", S), ("saved_mm_d", saved)):
        fA, fB = y[:n], y[n:2 * n]
        fAB = y[2 * n:].reshape(k, n)
        ind = sm.sobol_indices(fA, fB, fAB)
        for j, (nm, lo, hi) in enumerate(sm.SOBOL_INPUTS):
            rows.append(dict(output=name, input=nm, lo=lo, hi=hi, **{key: float(v[j]) for key, v in ind.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sobol.csv", index=False)
    np.savez_compressed(OUT / "sobol_amostras.npz", X=X, S=S, saved=saved,
                        names=np.array([s[0] for s in sm.SOBOL_INPUTS]))
    return df


def sobol_reindex(base=None):
    """Recompute the Sobol indices from the stored samples (no new simulations)."""
    z = np.load(OUT / "sobol_amostras.npz")
    k = len(sm.SOBOL_INPUTS)
    n = len(z["X"]) // (k + 2)
    rows = []
    for name, y in (("S", z["S"]), ("saved_mm_d", z["saved"])):
        ind = sm.sobol_indices(y[:n], y[n:2 * n], y[2 * n:].reshape(k, n))
        for j, (nm, lo, hi) in enumerate(sm.SOBOL_INPUTS):
            rows.append(dict(output=name, input=nm, lo=lo, hi=hi, **{key: float(v[j]) for key, v in ind.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sobol.csv", index=False)
    print(df.round(3).to_string())
    return df


def recesso(base):
    """Suppression against the recess diffusion length at the C1 cover fraction."""
    rows = []
    Rs = np.array([0.0, 0.5, 1, 2, 5, 10, 20, 30]) * 1e-3
    for I in (300.0, 820.0):
        for U in (0.0, 3.0):
            r = sm.run_design(base, np.full(len(Rs), I), np.full(len(Rs), U), np.full(len(Rs), 0.74),
                              extra={"R_rec": Rs}, return_fields=True)
            for i, R_ in enumerate(Rs):
                rows.append(dict(I=I, U=U, phi=0.74, R_rec_mm=R_ * 1000, S=float(r["S"][i]),
                                 E_cov=float(r["E_cov"][i]), E_open=float(r["E_open"][i]),
                                 Ts_cov=float(r["Ts_cov"][i]), Tp=float(r["Tp"][i])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recesso.csv", index=False)
    print(df.round(3).to_string())
    return df


def projeto(base):
    """Design map: suppression against opening size and recess diffusion length
    at the C1 cover fraction (full lamps; calm and fan at 3 m/s)."""
    ro = np.array([2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100]) * 1e-3
    Rr = np.array([0.0, 0.5, 1, 2, 5, 10, 20]) * 1e-3
    G = np.array(np.meshgrid(ro, Rr, [0.0, 3.0], indexing="ij")).reshape(3, -1)
    r = sm.run_design(base, np.full(G.shape[1], 820.0), G[2], np.full(G.shape[1], 0.74),
                      extra={"r_open": G[0], "R_rec": G[1]})
    df = pd.DataFrame(dict(r_open_mm=G[0] * 1000, R_rec_mm=G[1] * 1000, U=G[2], S=r["S"],
                           E_cov=r["E_cov"], E_open=r["E_open"]))
    df.to_csv(OUT / "projeto_aberturas.csv", index=False)
    print(df.pivot_table(index="r_open_mm", columns=["U", "R_rec_mm"], values="S").round(2).to_string())
    # the calibrated plates (opening size and recess length of the C1 mode, V2)
    c = sm.run_design(base, np.full(2, 820.0), np.array([0.0, 3.0]), np.full(2, 0.74))
    pd.DataFrame(dict(U=[0.0, 3.0], r_open_mm=base["r_open"] * 1000, R_rec_mm=base["R_rec"] * 1000,
                      S=c["S"])).to_csv(OUT / "projeto_calibrado.csv", index=False)
    return df


def sem_vias(base):
    """Shapley attribution at phi = 0.74 with the gap enhancement and the under-plate
    ventilation switched off (k_g = 1, C_u = 0): how much the two pathways weaken blocking."""
    rows = []
    Us = np.array([0.0, 3.0])
    for tag, extra in (("calibrated", None), ("no_pathways", {"k_g": 1.0, "C_u": 0.0})):
        res = sm.shapley_design(base, np.full(len(Us), 820.0), Us, np.full(len(Us), 0.74), extra=extra)
        for i, U in enumerate(Us):
            row = dict(case=tag, U=U, S_total=float(res["S_total"][i]))
            for m in sm.MECHS:
                row[m] = float(res[m][i])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "shapley_sem_vias.csv", index=False)
    print(df.round(3).to_string())
    return df


def posterior(base, n_draw=60, seed=5):
    """S(phi) under full lamps for draws of the C2 and C1 posteriors."""
    rng = np.random.default_rng(seed)
    import modelo_fisico as mf
    from priors import Prior, PRIORS_C2
    sys.path.insert(0, str(HERE))
    import calibracao_c1 as c1
    import calibracao as cb
    ch2 = cb.load_chain("c2"); ch1 = cb.load_chain("c1")
    X2 = ch2["chain"][ch2["chain"].shape[0] // 2:].reshape(-1, ch2["chain"].shape[2])
    X1 = ch1["chain"][ch1["chain"].shape[0] // 2:].reshape(-1, ch1["chain"].shape[2])
    p2 = Prior(PRIORS_C2); p1 = Prior(c1.PRIORS_C1)
    phis = np.round(np.arange(0.25, 0.951, 0.05), 3)
    rows = []
    for d in range(n_draw):
        q2 = p2.to_params(X2[rng.integers(len(X2))][None]); q1 = p1.to_params(X1[rng.integers(len(X1))][None])
        b = dict(base)
        b.update({k: float(v[0]) for k, v in q2.items()})
        for k in ("a_p", "G_pw", "C_p", "r_open", "tau_p", "k_g"):
            if k in q1:
                b[k] = float(q1[k][0])
        b["R_rec"] = float(q1["R_rec_V2"][0])
        if "C_u_V2" in q1:
            b["C_u"] = float(q1["C_u_V2"][0])
        for U in (0.0, 3.0):
            r = sm.run_design(b, np.full(len(phis), 820.0), np.full(len(phis), U), phis)
            for i, ph in enumerate(phis):
                rows.append(dict(draw=d, U=U, phi=ph, S=float(r["S"][i]), E_open=float(r["E_open"][i])))
        if d % 10 == 0:
            print(f"  posterior draw {d}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "incerteza_posterior.csv", index=False)
    return df


def espectral(base):
    """Spectral transfer: the same delivered energy (A_tot per unit index)
    redistributed with the solar three-band partition, and a lower plate
    absorptance for sunlight (white print: 0.25); Shapley at phi = 0.74 and S(phi)."""
    import espectro as es
    fit = es.solar_three_band_fit()
    A_tot = base["A_s"] + base["A_m"] + base["A_d"]
    sol = dict(base)
    tot = fit["f_s"] + fit["f_m"] + fit["f_d"]
    sol.update(A_s=A_tot * fit["f_s"] / tot, A_m=A_tot * fit["f_m"] / tot, l_m=fit["l_m"],
               A_d=A_tot * fit["f_d"] / tot, l_d=fit["l_d"])
    rows = []
    for tag, b, ap in (("lamp", base, base["a_p"]), ("solar", sol, base["a_p"]), ("solar_white", sol, 0.25)):
        bb = dict(b); bb["a_p"] = ap
        Us = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        res = sm.shapley_design(bb, np.full(len(Us), 820.0), Us, np.full(len(Us), 0.74))
        for i, U in enumerate(Us):
            row = dict(case=tag, phi=0.74, U=U, S_total=float(res["S_total"][i]))
            for m in sm.MECHS:
                row[m] = float(res[m][i])
            rows.append(row)
        phis = np.round(np.arange(0.25, 0.951, 0.05), 3)
        for U in (0.0, 3.0):
            r = sm.run_design(bb, np.full(len(phis), 820.0), np.full(len(phis), U), phis)
            for i, ph in enumerate(phis):
                rows.append(dict(case=tag, phi=ph, U=U, S_curve=float(r["S"][i]), E_open=float(r["E_open"][i])))
        print(f"  espectral {tag} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "transferencia_espectral.csv", index=False)
    json.dump(dict(solar_fit={k: float(v) for k, v in fit.items()}, A_tot=A_tot), open(OUT / "transferencia_espectral.json", "w"), indent=2)
    return df


if __name__ == "__main__":
    steps = sys.argv[1:] or ["mapa", "shapley", "sobol"]
    base, c2, c1m = load_base()
    json.dump(base, open(OUT / "parametros_base.json", "w"), indent=2)
    for st in steps:
        t0 = time.time()
        print(f"{st} ...", flush=True)
        {"mapa": mapa, "shapley": shapley, "sobol": sobol, "posterior": posterior,
         "espectral": espectral, "sobol_reindex": sobol_reindex, "recesso": recesso,
         "projeto": projeto, "sem_vias": sem_vias}[st](base)
        print(f"{st} done in {time.time() - t0:.0f} s", flush=True)
