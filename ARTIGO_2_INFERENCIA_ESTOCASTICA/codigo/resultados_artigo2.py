# -*- coding: utf-8 -*-
"""
Collects every number quoted in the Article 2 manuscript into
saidas/resultados_artigo2.json (values) and resultados_artigo2_texto.json
(the same values formatted for the text), and writes the manuscript tables.

Usage: python resultados_artigo2.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "saidas"
TAB = OUT / "tabelas"
TAB.mkdir(parents=True, exist_ok=True)
A1 = HERE.parents[1] / "ARTIGO_1_MODELO_FISICO_ACOPLADO" / "saidas"
R, TXT = {}, {}
MINUS = "−"


def sigtxt(v, n=2):
    if v == 0 or not np.isfinite(v):
        return "0"
    if 1 <= abs(v) < 10:
        n = max(n, 3)
    e = int(np.floor(np.log10(abs(v))))
    if e < -3:
        return f"${v / 10 ** e:.{n - 1}f}\\times10^{{{e}}}$"
    d = max(n - 1 - e, 0)
    return f"{v:.{d}f}".replace("-", MINUS)


def put(key, value, nd=None):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, str):
        R[key] = value; TXT[key] = value; return
    R[key] = value
    if isinstance(value, int) or nd == 0:
        TXT[key] = f"{int(round(value))}".replace("-", MINUS)
    elif nd is None:
        TXT[key] = sigtxt(float(value))
    else:
        TXT[key] = f"{value:.{nd}f}".replace("-", MINUS)


def best_driver():
    cv = OUT / "c2_validacao_cruzada_resumo.csv"
    if cv.exists():
        return pd.read_csv(cv).sort_values("loso_score", ascending=False).driver.iloc[0]
    fits = json.load(open(OUT / "c2_ajustes.json"))
    return min(fits, key=lambda k: fits[k]["aic"])


# ---------------------------------------------------------------- recovery
def recovery():
    s = json.load(open(OUT / "recuperacao_resumo.json"))
    df = pd.read_csv(OUT / "recuperacao_sintetica.csv")
    put("rec_n", int(len(df)))
    for k, v in s.items():
        put(f"rec_{k}_bias_pct", 100 * v["rel_bias"], 1)
        if v.get("coverage") is not None:
            put(f"rec_{k}_cov_pct", 100 * v["coverage"], 0)
    bias = [abs(100 * v["rel_bias"]) for k, v in s.items() if k in ("g_a", "c_a", "c_b", "delta", "beta0", "beta1")]
    put("rec_maxbias_lin_pct", max(bias), 1)
    cov = [100 * v["coverage"] for v in s.values() if v.get("coverage") is not None]
    put("rec_cov_min_pct", min(cov), 0); put("rec_cov_max_pct", max(cov), 0)


# ---------------------------------------------------------------- C2 fits and CV
DRV_LABEL = {"constante": "Constant", "dalton_ar": "Dalton, air VPD", "dalton_superficie": "Dalton, surface Δe",
             "penman_indice": "Penman-type with radiation index", "ryan_harleman": "Ryan–Harleman (free + forced)",
             "livre_forcada_potencia": "Free + forced, power law", "churchill_usagi": "Churchill–Usagi",
             "churchill_usagi_fundo": "Churchill–Usagi + background", "penman1948_fixo": "Penman (1948), fixed",
             "carrier_fixo": "Carrier (ASHRAE), fixed"}


DRV_LC = {"constante": "constant", "dalton_ar": "Dalton (air VPD)", "dalton_superficie": "Dalton (surface Δe)",
          "penman_indice": "Penman-type", "ryan_harleman": "Ryan–Harleman",
          "livre_forcada_potencia": "free + forced power-law", "churchill_usagi": "Churchill–Usagi",
          "churchill_usagi_fundo": "Churchill–Usagi with background", "penman1948_fixo": "Penman (1948)",
          "carrier_fixo": "Carrier (ASHRAE)"}


def c2():
    fits = json.load(open(OUT / "c2_ajustes.json"))
    best = best_driver()
    put("best_driver", best); put("best_driver_label", DRV_LABEL[best])
    f = fits[best]
    th = f["theta"]
    put("c2_g_a", th["g_a"], 3); put("c2_phi", th["phi"], 2)
    put("c2_s_eta", th["s_eta"], 3); put("c2_s_d", th["s_d"], 3)
    put("c2_s_a", th["s_a"], 3); put("c2_s_b", th["s_b"], 2)
    put("c2_delta_mm_h", f["delta"], 3); put("c2_delta_mm_d", 24 * f["delta"], 2)
    put("c2_delta_mm_d_abs", abs(24 * f["delta"]), 1)
    put("c2_delta_se_mm_d", 24 * f["delta_se"], 2)
    put("c2_c_a", f["c_a"], 3); put("c2_c_a_se", f["c_a_se"], 3)
    put("c2_c_b", f["c_b"], 3); put("c2_c_b_se", f["c_b_se"], 3)
    put("c2_nobs", int(f["nobs"]))
    put("q_ody_sd", 0.1 / np.sqrt(12), 3); put("q_cs_sd", 0.1 / np.sqrt(12), 3)
    # gains across drivers (robustness)
    ga = [v["theta"]["g_a"] for v in fits.values()]
    put("c2_g_a_min", min(ga), 3); put("c2_g_a_max", max(ga), 3)
    put("c2_ody_over_pct", 100 * (th["g_a"] - 1), 0)
    if "constante" in fits:
        put("c2_c_a_const", fits["constante"]["c_a"], 2)
    phys = [v["c_a"] for d, v in fits.items() if d not in ("constante", "penman_indice")]
    put("c2_c_a_phys_range", f"{min(phys):.2f} to {max(phys):.2f}".replace("-", MINUS))
    if "penman_indice" in fits:
        put("c2_c_a_penind", fits["penman_indice"]["c_a"], 2)
    # transfer velocity implied by the best driver (mm s-1) for dTv = 10 K, per unit vapour-density difference
    import dados_ee as de
    if best in ("ryan_harleman", "livre_forcada_potencia", "churchill_usagi", "churchill_usagi_fundo"):
        for U in (0.0, 3.0):
            fr = pd.DataFrame(dict(Ucl=[U], drho_v=[1.0], dTv=[10.0], dE_s=[0.0], VPD_k=[0.0], I=[0.0]))
            E = float(de.driver_matrix(fr, best, f.get("hyper"))[0] @ np.asarray(f["beta"]))
            put(f"c2_Ce_U{U:g}", E / 3.6 * 1000, 1)   # mm s-1
    # time scale of the AR(1) deviation
    put("c2_eta_tau_h", -1 / 6 / np.log(th["phi"]) if 0 < th["phi"] < 1 else np.nan, 1)
    rows = []
    cv = pd.read_csv(OUT / "c2_validacao_cruzada_resumo.csv") if (OUT / "c2_validacao_cruzada_resumo.csv").exists() else None
    for d, v in fits.items():
        row = dict(driver=d, label=DRV_LABEL[d], k=len(v["beta"]) + (len(v["hyper"]) if v.get("hyper") else 0),
                   hyper=json.dumps(v.get("hyper")) if v.get("hyper") else "", ll=v["ll"], aic=v["aic"], bic=v["bic"],
                   g_a=v["theta"]["g_a"], phi=v["theta"]["phi"], s_eta=v["theta"]["s_eta"],
                   beta=";".join(f"{b:.4g}" for b in v["beta"]), beta_se=";".join(f"{b:.2g}" for b in v["beta_se"]))
        if cv is not None and d in set(cv.driver):
            c = cv.set_index("driver").loc[d]
            row.update(loso=c.loso_score, dloso=c.diff_to_best, dloso_se=c.diff_se)
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["daic"] = tab.aic - tab.aic.min(); tab["dbic"] = tab.bic - tab.bic.min()
    tab.to_csv(TAB / "comparacao_drivers.csv", index=False)
    if cv is not None:
        c = cv.sort_values("loso_score", ascending=False).reset_index(drop=True)
        put("cv_best_label", DRV_LABEL[c.driver[0]]); put("cv_best_label_lc", DRV_LC[c.driver[0]])
        put("cv_second_label", DRV_LABEL[c.driver[1]]); put("cv_second_label_lc", DRV_LC[c.driver[1]])
        put("cv_second_diff", float(c.diff_to_best[1]), 1); put("cv_second_se", float(c.diff_se[1]), 1)
        da = c[c.driver == "dalton_ar"].diff_to_best.iloc[0]; ds = c[c.driver == "dalton_superficie"].diff_to_best.iloc[0]
        put("cv_dalton_diff_gap", float(ds - da), 0)
        cst = c[c.driver == "constante"].iloc[0]
        put("cv_const_diff", float(cst.diff_to_best), 0)
        for d in ("penman1948_fixo", "carrier_fixo", "dalton_ar", "dalton_superficie", "penman_indice",
                  "ryan_harleman", "livre_forcada_potencia", "churchill_usagi", "churchill_usagi_fundo", "constante"):
            if d in set(c.driver):
                r_ = c[c.driver == d].iloc[0]
                put(f"cv_{d}_diff", float(r_.diff_to_best), 0); put(f"cv_{d}_se", float(r_.diff_se), 0)
                if r_.diff_se > 0:
                    put(f"cv_{d}_nse", float(abs(r_.diff_to_best) / r_.diff_se), 1)
        # hyperparameters selected for the formulations with a profiled grid
        fits = json.load(open(OUT / "c2_ajustes.json"))
        for d in ("livre_forcada_potencia", "churchill_usagi", "churchill_usagi_fundo"):
            for k, v in (fits.get(d, {}).get("hyper") or {}).items():
                put(f"hyper_{d}_{k}", float(v), 2)
        # agreement of rankings
        aic_rank = tab.sort_values("aic").driver.tolist()
        put("aic_best_label", DRV_LABEL[aic_rank[0]]); put("aic_best_label_lc", DRV_LC[aic_rank[0]])
        put("cv_n_folds", int(pd.read_csv(OUT / "c2_validacao_cruzada_folds.csv").fold.nunique()))
    for d in ("penman1948_fixo", "carrier_fixo"):
        if d in fits:
            put(f"mult_{d}", fits[d]["beta"][0], 2); put(f"mult_{d}_se", fits[d]["beta_se"][0], 2)
    return best


def evaporation(best):
    import dados_ee as de
    E = pd.read_csv(OUT / f"c2_{best}_E.csv", index_col=0, parse_dates=True)
    a, _ = de.series_10min("C2")
    a = a.reindex(E.index)
    lamp = (a["lamp_on"] > 0.5).to_numpy()
    put("E_sd_lamp_med", float(np.median(E.E_sd[lamp])), 3)
    put("E_sd_dark_med", float(np.median(E.E_sd[~lamp])), 3)
    put("E_hw_lamp", 1.96 * float(np.median(E.E_sd[lamp])), 2)
    put("E_hw_dark", 1.96 * float(np.median(E.E_sd[~lamp])), 2)
    put("E_dark_med", float(np.median(E.E_mm_h[~lamp])), 3)
    put("E_reg_share", float(1 - np.var(E.E_mm_h - E.E_reg) / np.var(E.E_mm_h)), 2)
    put("E_reg_share_pct", 100 * R["E_reg_share"], 0)
    # run means against the companion mechanistic model
    f1 = A1 / "tabelas" / "evaporacao_corridas_c2.csv"
    if f1.exists():
        r1 = pd.read_csv(f1, parse_dates=["start"])
        rows = []
        for _, r in r1.iterrows():
            a0 = r.start + pd.Timedelta(minutes=30)
            b0 = r.start + pd.Timedelta(hours=r.hours)
            mk = (E.index >= a0) & (E.index < b0)
            rows.append(dict(start=r.start, cfg=r.cfg, U=r.U, E_ss=float(E.E_mm_h[mk].mean()),
                             E_ss_sd=float(E.E_sd[mk].mean() / np.sqrt(max(mk.sum() / 6, 1))),
                             E_model=r.E_model))
        cmp_ = pd.DataFrame(rows)
        cmp_.to_csv(TAB / "corridas_ss_vs_modelo.csv", index=False)
        d = cmp_.E_ss - cmp_.E_model
        put("cmp_rmse", float(np.sqrt(np.mean(d ** 2))), 2); put("cmp_bias", float(d.mean()), 2)
        put("cmp_r", float(np.corrcoef(cmp_.E_ss, cmp_.E_model)[0, 1]), 2)
        put("cmp_n", int(len(cmp_)))
        put("E_run_min", float(cmp_.E_ss.min()), 2); put("E_run_max", float(cmp_.E_ss.max()), 2)


# ---------------------------------------------------------------- C1
def c1():
    r = json.load(open(OUT / "c1_supressao_direta.json"))
    put("c1_driver_label", DRV_LABEL[r["driver"]])
    put("c1_s_U", r["s_U"], 2)
    put("c1_kappa0", r["kappa_uncovered"], 2); put("c1_kappa0_se", r["kappa_uncovered_se"], 2)
    for g, nm in (("unc_strong", "strong"), ("unc_weak", "weak")):
        put(f"c1_kappa_{nm}", r["kappa_ref"][g][0], 2); put(f"c1_kappa_{nm}_se", r["kappa_ref"][g][1], 2)
    th = r["theta"]
    put("c1_g_a", th["g_a"], 3); put("c1_s_mic", th["s_b"], 2); put("c1_s_a", th["s_a"], 3)
    put("c1_c_a", r["c_a"], 3); put("c1_c_a_se", r["c_a_se"], 3)
    if r.get("s_U_profile_conc"):
        ll = np.array([x[1] for x in r["s_U_profile_conc"]])
        sus = np.array([x[0] for x in r["s_U_profile_conc"]])
        wts = np.array(r["s_U_weights"])
        put("c1_sU_dll", float(ll.max() - ll.min()), 1)
        put("c1_sU_llmax", float(sus[np.argmax(ll)]), 2)
        put("c1_sU_mean", float(r["s_U_mean"]), 2)
        keep = sus[wts > 0.01]
        put("c1_sU_lo", float(keep.min()), 2); put("c1_sU_hi", float(keep.max()), 2)
    t = pd.DataFrame(r["suppression"])
    w = t[t.grupo.str.startswith("W")].copy()
    # reported estimate: the s_U-averaged log-scale estimate of the hierarchical input
    # (1 - exp(ybar), interval from ybar +- 1.96 se, which includes the between-s_U spread)
    hj = OUT / "hierarquico_janelas.csv"
    if hj.exists():
        hw = pd.read_csv(hj).set_index("win")
        w["S_rep"] = w.grupo.map(hw.S_bruto); w["S_rep_lo"] = w.grupo.map(hw.S_bruto_lo)
        w["S_rep_hi"] = w.grupo.map(hw.S_bruto_hi)
    else:
        w["S_rep"], w["S_rep_lo"], w["S_rep_hi"] = w.S_marg, w.S_marg_lo, w.S_marg_hi
    put("c1_Spool_shift", float((w.S_pooled - w.S_direct).mean()), 2)
    put("c1_n_windows", int(len(w)))
    put("c1_Sdir_min", w.S_rep.min(), 2); put("c1_Sdir_max", w.S_rep.max(), 2)
    put("c1_Sdir_mean", w.S_rep.mean(), 2)
    put("c1_Sdir_halfwidth_med", float(np.median((w.S_rep_hi - w.S_rep_lo) / 2)), 2)
    import sys as _s
    _s.path.insert(0, str(HERE.parents[1] / "_comum"))
    from tanque import config as _cfg
    meta = {x[0]: x for x in _cfg.WINDOWS_C1}
    phi = w.grupo.map({k: _cfg.coverage_fraction(v[5]) for k, v in meta.items()})
    put("c1_eff_mean", float((w.S_rep / phi).mean()), 2)
    prof_f = OUT / "c1_supressao_perfil_sU.csv"
    if prof_f.exists():
        pr_ = pd.read_csv(prof_f)
        pr_ = pr_[pr_.grupo.str.startswith("W")]
        pr_["regime"] = pr_.grupo.map({k: v[6] for k, v in meta.items()})
        rng_ = pr_.groupby("grupo").S_direct.agg(lambda x: x.max() - x.min())
        pr_ = pr_[pr_.weight > 0.01]           # values of s_U that carry weight
        rng_ = pr_.groupby("grupo").S_direct.agg(lambda x: x.max() - x.min())
        calm = pr_[pr_.regime == "calm"]
        put("c1_Scalm_sU_range", f"{calm.S_direct.min():.2f}–{calm.S_direct.max():.2f}")
        fan = [g for g in rng_.index if meta[g][6] == "strong"]
        put("c1_Sfan_sU_range", float(rng_[fan].max()), 2)
        calm_shift = calm.groupby("grupo").S_direct.agg(lambda x: x.max() - x.min())
        put("c1_Scalm_sU_shift_min", float(calm_shift.min()), 2); put("c1_Scalm_sU_shift_max", float(calm_shift.max()), 2)
        # sensitivity: the estimates at s_U = 0.75, near the companion paper's calibration
        pr0 = pd.read_csv(prof_f)
        pr0 = pr0[pr0.grupo.str.startswith("W") & np.isclose(pr0.s_U, 0.75)]
        if len(pr0):
            reg0 = pr0.grupo.map({k: v[6] for k, v in meta.items()}); ver0 = pr0.grupo.map({k: v[4] for k, v in meta.items()})
            put("c1_sU075_min", pr0.S_direct.min(), 2); put("c1_sU075_max", pr0.S_direct.max(), 2)
            put("c1_sU075_mean", pr0.S_direct.mean(), 2)
            put("c1_sU075_calm", pr0[reg0 == "calm"].S_direct.mean(), 2)
            put("c1_sU075_v2fan", pr0[(reg0 == "strong") & (ver0 == "V2")].S_direct.mean(), 2)
    cu_f = OUT / "c1_supressao_direta_cu.csv"
    if cu_f.exists():
        cu = pd.read_csv(cu_f).set_index("grupo")
        col = "S_marg" if "S_marg" in cu else "S_direct"
        d = cu.loc[w.grupo, col].to_numpy() - w.S_rep.to_numpy()
        put("c1_Scu_shift", float(np.mean(d)), 2)
        reg = cu.index.map(lambda g: meta[g][6] if g in meta else "")
        cuw = cu[[str(g).startswith("W") for g in cu.index]]
        regw = pd.Series([meta[g][6] for g in cuw.index], index=cuw.index)
        verw = pd.Series([meta[g][4] for g in cuw.index], index=cuw.index)
        put("c1_cu_calm_mean", float(cuw.loc[regw == "calm", col].mean()), 2)
        put("c1_cu_fan_mean", float(cuw.loc[(regw == "strong") & (verw == "V2"), col].mean()), 2)
        put("c1_Scu_absmax", float(np.max(np.abs(d))), 2)
    log = (OUT / "c1_ajuste_log.txt").read_text(encoding="utf-8")
    import re as _re
    mm = _re.search(r"micrometer obs (\d+)", log)
    if mm:
        put("c1_n_mic", int(mm.group(1)))
    w.to_csv(TAB / "supressao_direta_janelas.csv", index=False)
    # comparison with the companion mechanistic model: like for like ('direct' suppression of
    # the model, at the covered skin temperature) and against its total (counterfactual) suppression
    f1 = A1 / "tabelas" / "c1_janelas_cobertas.csv"
    if f1.exists():
        m = pd.read_csv(f1).set_index("win")
        j = w.set_index("grupo").join(m[[c for c in ("S_model", "S_direct_model", "regime") if c in m]], how="inner")
        for col, tag in (("S_direct_model", "dir"), ("S_model", "tot")):
            if col in j:
                d = j.S_rep - j[col]
                put(f"cmp_S{tag}_mean_diff", float(d.mean()), 2)
                put(f"cmp_S{tag}_mae", float(np.mean(np.abs(d))), 2)
                inside = ((j[col] >= j.S_rep_lo) & (j[col] <= j.S_rep_hi)).sum()
                put(f"cmp_S{tag}_inside", int(inside))
        put("cmp_S_n", int(len(j)))
        if "U_mean" in m:
            us = m[m.regime == "strong"].U_mean
            put("c1_Ueff_lo", float(us.min() * r["s_U"]), 1); put("c1_Ueff_hi", float(us.max() * r["s_U"]), 1)
        lamp = m.reindex(j.index).I_mean > 250 if "I_mean" in m else np.ones(len(j), bool)
        put("a1_Stot_min", float(j.S_model[lamp].min()), 2); put("a1_Stot_max", float(j.S_model[lamp].max()), 2)
    # companion-paper values quoted in the text
    t1 = A1 / "resultados_artigo1_texto.json"
    if t1.exists():
        T1 = json.load(open(t1, encoding="utf-8"))
        for k_out, k_in in (("a1_s_U", "post_s_U"), ("a1_s_U_lo", "post_s_U_lo"), ("a1_s_U_hi", "post_s_U_hi"),
                            ("a1_h_p2_mm", "post_h_p2_mm"), ("gap_width_mm", "gap_width_mm"),
                            ("a1_c1_mic_used", "c1_mic_used"), ("a1_ody_gain", "c2_ody_gain"),
                            ("a1_ody_ca", "c2_ody_ca"), ("a1_cs_cb", "c2_cs_cb")):
            if k_in in T1:
                TXT[k_out] = T1[k_in]


def hierarchical():
    h = json.load(open(OUT / "hierarquico_resumo.json"))
    put("h_tau", h["tau"][0], 2); put("h_tau_lo", h["tau"][1], 2); put("h_tau_hi", h["tau"][2], 2)
    pr = h["probabilities"]
    put("h_p_V2", pr["P(S_V2 > S_V1)"], 2); put("h_p_off", pr["P(S_fan_off > S_strong)"], 2)
    put("h_p_weak", pr["P(S_weak > S_strong)"], 2)
    put("h_p_off_lower", 1 - pr["P(S_fan_off > S_strong)"], 2)
    eff = h["effects"]
    for k, nm in zip(eff, ("int", "V2", "weak", "off")):
        put(f"h_eff_{nm}", eff[k][0], 2); put(f"h_eff_{nm}_lo", eff[k][1], 2); put(f"h_eff_{nm}_hi", eff[k][2], 2)
    c = pd.read_csv(OUT / "hierarquico_celulas.csv")
    names = {"V1, vento forte": "V1_strong", "V1, vento fraco": "V1_weak", "V2, vento forte": "V2_strong",
             "V2, ventilador desligado": "V2_off", "V2, vento fraco": "V2_weak"}
    for _, r in c.iterrows():
        k = names[r.celula]
        for col, s in (("S_media", ""), ("S_lo", "_lo"), ("S_hi", "_hi"), ("S_pred", "_pred"),
                       ("S_pred_lo", "_pred_lo"), ("S_pred_hi", "_pred_hi")):
            put(f"h_{k}{s}", r[col], 2)
    w = pd.read_csv(OUT / "hierarquico_janelas.csv")
    put("h_shrink_max", float((w.S_bruto - w.S_hier).abs().max()), 2)


def design():
    d = json.load(open(OUT / "delineamento_resumo.json"))
    e = pd.read_csv(OUT / "delineamento_eig_unico.csv")
    b = d["batch"]
    for i, (ph, U) in enumerate(b, 1):
        put(f"des_run{i}_phi", ph, 1); put(f"des_run{i}_U", U, 1)
    top = e.sort_values("eig", ascending=False).iloc[0]
    put("des_best_phi", top.phi, 1); put("des_best_U", top.U, 1); put("des_best_eig", top.eig, 2)
    if "eig_se" in e:
        put("des_eig_se_max", float(e.eig_se.max()), 2)
        hi = e[np.isclose(e.phi, top.phi)]
        put("des_best_eig_min", float(hi.eig.min()), 2); put("des_best_eig_max", float(hi.eig.max()), 2)
        for _, r_ in hi.iterrows():
            put(f"des_eig09_U{r_.U:g}".replace(".", ""), r_.eig, 2)
    cur = e[np.isclose(e.phi, 0.7)]
    put("des_eig_current_max", float(cur.eig.max()), 2)
    ref = d.get("reference_eig", {})
    if ref:
        put("des_eig_cur4", ref["current_4"], 2)
        put("des_eig_cur1_off", ref["current_1_off"], 2); put("des_eig_cur1_on", ref["current_1_on"], 2)
    chk = d.get("reference_eig", {}).get("batch_mc_check")
    if chk:
        put("des_batch_eig_mc", float(np.mean(chk)), 2); put("des_batch_eig_mc_sd", float(np.std(chk, ddof=1)), 2)
    if len(d.get("batch_eig", [])) >= 2:
        inc = np.diff([0.0] + d["batch_eig"])
        put("des_inc_first2", float(inc[:2].sum()), 2); put("des_inc_last2_max", float(inc[2:].max()), 2)
    for m, v in d.get("fit_current", {}).items():
        put(f"des_fit_{m}_on", v["S_on"], 2); put(f"des_fit_{m}_off", v["S_off"], 2)
    put("des_tau", d["tau"], 2); put("des_sigma", d["sigma"], 2)
    if d.get("batch_eig"):
        put("des_batch_eig", d["batch_eig"][-1], 2)
    for m, par in d["draws_summary"].items():
        for k, v in par.items():
            put(f"des_{m}_{k}", v[0], None); put(f"des_{m}_{k}_lo", v[1], None); put(f"des_{m}_{k}_hi", v[2], None)


def main():
    steps = [recovery, c2, c1, hierarchical, design]
    best = None
    for fn in steps:
        try:
            out = fn()
            if fn is c2:
                best = out
            print(f"{fn.__name__}: ok", flush=True)
        except (FileNotFoundError, KeyError) as e:
            print(f"{fn.__name__}: missing {e}", flush=True)
    if best:
        try:
            evaporation(best); print("evaporation: ok")
        except (FileNotFoundError, KeyError) as e:
            print(f"evaporation: missing {e}")
    json.dump(R, open(OUT / "resultados_artigo2.json", "w"), indent=1, default=str)
    json.dump(TXT, open(OUT / "resultados_artigo2_texto.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(R)} values -> resultados_artigo2.json")


if __name__ == "__main__":
    main()
