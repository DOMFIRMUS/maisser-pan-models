# -*- coding: utf-8 -*-
"""
Posterior summaries and convergence diagnostics of the ensemble MCMC runs.

For each tag (c2, c1) the second half of the chain is kept; for every
parameter it reports prior median and 95 % interval, posterior median and
95 % interval, rank-normalised split R-hat (walkers as chains), effective
sample size and integrated autocorrelation time, and the posterior
correlation matrix. The highest-posterior draw of the whole chain is written
as mode_<tag>_posterior.json.

With --promote, the posterior mode replaces map_<tag>.json (the DE result
is kept as map_<tag>_de.json); the nuisance coefficients and the dependence
diagnostics are recomputed at the new mode, so that every downstream script
(diagnostics, scenarios, figures) uses it.

Usage: python resumo_posterior.py c2 c1 [--promote]
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibracao as cb          # noqa: E402
from priors import Prior, PRIORS_C2   # noqa: E402

CAL = cb.OUT
TAB = HERE.parent / "saidas" / "tabelas"
TAB.mkdir(parents=True, exist_ok=True)


def _prior(tag):
    if tag == "c2":
        return Prior(PRIORS_C2)
    import calibracao_c1 as c1
    return Prior(c1.PRIORS_C1)


def _nat(u, tr):
    return np.exp(u) if tr == "log" else (1 / (1 + np.exp(-u)) if tr == "logit" else u)


def summarize(tag, burn=0.5):
    z = cb.load_chain(tag)
    chain, lp = z["chain"], z["lp"]
    names = [str(s) for s in z["names"]]
    pr = _prior(tag)
    assert names == pr.names
    n = chain.shape[0]
    keep = chain[int(n * burn):]
    rows = []
    for i, k in enumerate(names):
        tr = pr.tr[i]
        u = keep[:, :, i]
        x = _nat(u, tr)
        q = np.percentile(x, [2.5, 50, 97.5])
        plo, pmed, phi = _nat(pr.mu[i] + np.array([-1.96, 0.0, 1.96]) * pr.sd[i], tr)
        ess, tau = cb.ess(u)
        # shrinkage of the posterior relative to the prior on the unconstrained scale
        contraction = 1 - u.var() / pr.sd[i] ** 2
        z_shift = (u.mean() - pr.mu[i]) / pr.sd[i]
        rows.append(dict(parameter=k, transform=tr, prior_median=pmed, prior_lo=plo, prior_hi=phi,
                         post_median=q[1], post_lo=q[0], post_hi=q[2], post_mean=float(x.mean()),
                         rhat=cb.split_rhat(u), ess=ess, iat=tau,
                         contraction=float(contraction), z_shift=float(z_shift)))
    df = pd.DataFrame(rows)
    flat = keep.reshape(-1, keep.shape[2])
    corr = pd.DataFrame(np.corrcoef(flat.T), index=names, columns=names)
    df.to_csv(TAB / f"posterior_{tag}.csv", index=False)
    corr.to_csv(TAB / f"posterior_corr_{tag}.csv")
    ib = np.unravel_index(np.argmax(lp), lp.shape)
    u_best = chain[ib[0], ib[1]]
    info = dict(tag=tag, n_iter=int(n), walkers=int(chain.shape[1]), burn=burn, n_kept=int(flat.shape[0]),
                acceptance=float(np.mean(z["acc"])),
                lp_best=float(lp.max()), lp_median_kept=float(np.median(lp[int(n * burn):])),
                lp_best_iter=int(ib[0]), u=u_best.tolist(), names=names,
                params={k: float(_nat(u_best[i], pr.tr[i])) for i, k in enumerate(names)},
                max_rhat=float(df["rhat"].max()), min_ess=float(df["ess"].min()), max_iat=float(df["iat"].max()))
    json.dump(info, open(CAL / f"mode_{tag}_posterior.json", "w"), indent=2)
    print(f"--- {tag}: {n} iterations x {chain.shape[1]} walkers, kept {flat.shape[0]} draws; "
          f"lp best {info['lp_best']:.1f} (iter {ib[0]}), median kept {info['lp_median_kept']:.1f}; "
          f"max R-hat {info['max_rhat']:.3f}, min ESS {info['min_ess']:.0f}, max IAT {info['max_iat']:.0f}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.drop(columns=["transform"]).to_string(float_format=lambda v: f"{v:.4g}"))
    return df, info


def promote(tag):
    """Replace map_<tag>.json by the posterior mode (recomputing nuisance terms)."""
    info = json.load(open(CAL / f"mode_{tag}_posterior.json"))
    src = CAL / f"map_{tag}.json"
    m_old = json.load(open(src))
    if m_old.get("source") != "posterior_mode":
        shutil.copy(src, CAL / f"map_{tag}_de.json")
    u = np.array(info["u"])[None]
    extra = {"round": "posterior_mode", "source": "posterior_mode", "lp_chain": info["lp_best"]}
    if tag == "c2":
        with cb.C2Problem(rho=m_old["rho_used"], weights=m_old["weights_used"]) as prob:
            fx = prob.logpost(u)
            m = cb.save_mode(prob, u, fx, src, extra=extra)
    else:
        import calibracao_c1 as c1
        base = c1.base_from_map(CAL / "map_c2.json")   # the chain is run on the current C2 mode
        with c1.C1Problem(base, rho=m_old["rho_used"], weights=m_old["weights_used"]) as prob:
            fx = prob.logpost(u)
            m = c1.save_mode(prob, u, fx, src, extra=extra)
    print(f"{tag}: map_{tag}.json <- posterior mode (logpost {m['logpost']:.2f}; chain {info['lp_best']:.2f})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for tag in args or ["c2", "c1"]:
        summarize(tag)
        if "--promote" in sys.argv:
            promote(tag)
