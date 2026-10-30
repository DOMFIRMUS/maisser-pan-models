# -*- coding: utf-8 -*-
"""
Unattended continuation of the C1 and C2 ensemble chains until convergence,
followed by the post-processing of both articles.

Convergence (calibracao.convergence; second half of the chain retained):
max rank-normalised split R-hat below CONV_RHAT (walkers as chains) and
retained iterations spanning at least CONV_NIAT times the longest integrated
autocorrelation time. The C2 mode (map_c2.json) and the C1 mode (map_c1.json)
are kept fixed: C1 is conditioned on the C2 mode (plug-in), and the scenarios
rest on the modes; only the posterior summaries and the posterior propagation
are updated.

The chains are continued in chunks (mcmc_chain_<tag>_cont<k>.npz, saved every
50 iterations); the run can be interrupted at any time and started again, and
it resumes from the last saved state. Progress (in Portuguese, for the author):
../saidas/convergencia/STATUS.txt; full log: ../saidas/convergencia/convergencia.log.

Usage:
    python convergir_cadeias.py                  C1, then C2, then post-processing
    python convergir_cadeias.py --post-only      post-processing with the chains as they are
    options: --only c1|c2, --max-hours-c1 H, --max-hours-c2 H, --chunk-c1 N, --chunk-c2 N, --no-post
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibracao as cb     # noqa: E402

A1 = HERE.parent
ROOT = A1.parent
A2 = ROOT / "ARTIGO_2_INFERENCIA_ESTOCASTICA"
DIR = A1 / "saidas" / "convergencia"
DIR.mkdir(parents=True, exist_ok=True)
STATUS = DIR / "STATUS.txt"
LOG = DIR / "convergencia.log"
STATE = DIR / "estado.json"
NOME = {"c1": "C1 (campanha com placas)", "c2": "C2 (campanha descoberta)"}


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake():
    """Ask Windows not to sleep while this process runs (released when it exits)."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log("pedido ao Windows para não suspender enquanto o processo roda")
        # a click in a classic console window (QuickEdit) would freeze the run at its next print
        k32 = ctypes.windll.kernel32
        h, mode = k32.GetStdHandle(-10), ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)
    except Exception as e:      # not on Windows
        log(f"keep-awake indisponível: {e}")


def single_instance():
    """Lock file released by the system when the process ends."""
    f = open(DIR / "convergencia.lock", "w")
    try:
        try:
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:     # not on Windows
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Outra execução de convergir_cadeias.py já está rodando. Feche-a antes de iniciar outra.")
        sys.exit(1)
    return f


def check_files(tag):
    """Set aside unreadable chain files (e.g. a power cut while writing with an older version)."""
    import numpy as np
    for f in cb.chain_files(tag):
        try:
            np.load(f)["lp"]
        except Exception as e:
            bad = f.with_suffix(".corrompido")
            f.rename(bad)
            log(f"{f.name} ilegível ({e}); renomeado para {bad.name}")


def load_state():
    return json.load(open(STATE, encoding="utf-8")) if STATE.exists() else {}


def save_state(st):
    json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def fmt_h(h):
    return f"{h:.1f} h" if h >= 1 else f"{60 * h:.0f} min"


def write_status(st, phase, extra=""):
    L = [f"CONVERGÊNCIA DAS CADEIAS MCMC — Artigo 1 (MAISSER-PE)",
         f"Atualizado em {datetime.now():%d/%m/%Y %H:%M}",
         f"Fase atual: {phase}", "",
         f"Critério: R-hat máximo < {cb.CONV_RHAT} e metade retida da cadeia >= {cb.CONV_NIAT} "
         f"tempos de autocorrelação (τ).", ""]
    for tag in ("c1", "c2"):
        d = st.get(tag)
        if not d:
            L.append(f"{NOME[tag]}: ainda não avaliada")
            continue
        s = (f"{NOME[tag]}: {d['n_iter']} iterações x {d['walkers']} walkers; R-hat máx {d['max_rhat']:.3f} "
             f"({d['worst_rhat']}); τ máx {d['max_iat']:.0f} it ({d['worst_iat']}); "
             f"retido = {d['n_iat']:.1f} τ")
        if d["converged"]:
            s += f"  -> CONVERGIU" + (f" em {d['when_converged']}" if d.get("when_converged") else "")
        else:
            if d.get("sec_per_it"):
                rem = max(d["n_needed"] - d["n_iter"], 0)
                if rem > 0:
                    h = rem * d["sec_per_it"] / 3600
                    eta = datetime.now() + timedelta(hours=h)
                    s += (f"\n    faltam ~{rem} iterações para {cb.CONV_NIAT} τ (~{fmt_h(h)}, previsão {eta:%d/%m %H:%M}; "
                          f"τ ainda pode crescer)")
                else:
                    s += "\n    comprimento suficiente; aguardando o R-hat cair abaixo do limite"
            if d.get("budget_left_h") is not None:
                s += f"\n    tempo máximo restante para esta cadeia: {fmt_h(d['budget_left_h'])}"
            if d.get("stopped"):
                s += f"\n    PARADA: {d['stopped']}"
        L.append(s)
    L += ["", "Ao final das cadeias o pós-processamento roda sozinho: resumos do posterior, propagação",
          "do posterior, figuras, coletor de resultados, manuscritos dos dois artigos, PDFs, pacotes de",
          "submissão e pasta do depósito GitHub/Zenodo.",
          "",
          "Para interromper: feche a janela preta (o progresso salvo é mantido). Para retomar, rode",
          "RODAR_CONVERGENCIA.bat de novo. Para finalizar com as cadeias como estão: FINALIZAR_AGORA.bat."]
    if extra:
        L += ["", extra]
    STATUS.write_text("\n".join(L) + "\n", encoding="utf-8")


def make_problem(tag):
    if tag == "c2":
        m = json.load(open(cb.OUT / "map_c2.json"))
        prob = cb.C2Problem(rho=m["rho_used"], weights=m["weights_used"])
    else:
        import calibracao_c1 as c1
        m = json.load(open(cb.OUT / "map_c1.json"))
        prob = c1.C1Problem(c1.base_from_map(cb.OUT / "map_c2.json"), rho=m["rho_used"], weights=m["weights_used"])
    return prob


def run_chain(tag, chunk, max_hours, st):
    check_files(tag)
    d = cb.convergence(tag)
    d.update({k: st.get(tag, {}).get(k) for k in ("sec_per_it", "when_converged") if st.get(tag, {}).get(k)})
    prev = st.get(tag, {}).get("hours_used", 0.0)      # the budget accumulates over restarts
    d["hours_used"] = prev
    st[tag] = d
    save_state(st)
    log(f"{tag}: {d['n_iter']} it, R-hat máx {d['max_rhat']:.3f}, τ máx {d['max_iat']:.0f}, "
        f"{d['n_iat']:.1f} τ retidos, convergiu={d['converged']}, {prev:.1f} h usadas de {max_hours:.0f} h")
    if d["converged"]:
        write_status(st, f"{NOME[tag]} já convergida")
        return
    if prev >= max_hours:
        d["budget_left_h"] = 0.0
        d["stopped"] = f"tempo máximo de {max_hours:.0f} h (somado entre reinícios) já usado"
        log(f"{tag}: {d['stopped']}; segue para a próxima etapa")
        save_state(st)
        write_status(st, f"{NOME[tag]} encerrada")
        return
    t0 = time.time()
    lp_map = json.load(open(cb.OUT / f"map_{tag}.json"))["logpost"]
    chain_log = open(cb.OUT / f"calibracao_{tag}_log.txt", "a", encoding="utf-8")

    def clog(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        chain_log.write(line + "\n"); chain_log.flush()
        w = msg.split()
        if w[:2] == ["MCMC", "it"] and int(w[2]) % 100:
            return          # the console and the main log get every 100th iteration
        log(f"{tag}: {msg}")

    with make_problem(tag) as prob:
        while True:
            used = prev + (time.time() - t0) / 3600
            d["budget_left_h"] = max(max_hours - used, 0)
            write_status(st, f"continuando a cadeia {NOME[tag]}")
            if used >= max_hours:
                d["stopped"] = f"tempo máximo de {max_hours:.0f} h (somado entre reinícios) atingido sem convergir"
                log(f"{tag}: {d['stopped']}")
                break
            n0, t1 = d["n_iter"], time.time()
            cb.continue_chain(prob, tag, chunk, clog)
            new = cb.convergence(tag)
            spi = (time.time() - t1) / max(new["n_iter"] - n0, 1)
            d.update(new)
            d["sec_per_it"] = spi
            d["hours_used"] = prev + (time.time() - t0) / 3600
            d["budget_left_h"] = max(max_hours - d["hours_used"], 0)
            if d["lp_best"] > lp_map + 3:
                log(f"{tag}: a cadeia achou log-posterior {d['lp_best']:.1f} > modo {lp_map:.1f} + 3 "
                    f"(o modo usado nos cenários é mantido; ver texto)")
            log(f"{tag}: {d['n_iter']} it, R-hat máx {d['max_rhat']:.3f} ({d['worst_rhat']}), "
                f"τ máx {d['max_iat']:.0f} ({d['worst_iat']}), {d['n_iat']:.1f} τ retidos, "
                f"{spi:.2f} s/it, convergiu={d['converged']}")
            if d["converged"]:
                d["when_converged"] = f"{datetime.now():%d/%m/%Y %H:%M}"
                d.pop("stopped", None)
                save_state(st)
                break
            save_state(st)
    chain_log.close()
    save_state(st)
    write_status(st, f"{NOME[tag]} encerrada")


# ---------------------------------------------------------------- post-processing
def post(st):
    py = [sys.executable, "-X", "utf8"]
    c1, c2_ = A1 / "codigo", A2 / "codigo"
    steps = [
        ("resumos do posterior (C2 e C1)", c1, py + ["resumo_posterior.py", "c2", "c1"]),
        ("propagação do posterior (cenários)", c1, py + ["executar_simulacoes.py", "posterior"]),
        ("figuras do Artigo 1", c1, py + ["figuras_artigo1.py"]),
        ("figuras do material suplementar", c1, py + ["figuras_si_artigo1.py"]),
        ("coletor de resultados do Artigo 1", c1, py + ["resultados_artigo1.py"]),
        ("manuscrito do Artigo 1", A1 / "manuscrito", py + ["construir_manuscrito.py"]),
        ("coletor de resultados do Artigo 2", c2_, py + ["resultados_artigo2.py"]),
        ("manuscrito do Artigo 2", A2 / "manuscrito", py + ["construir_manuscrito.py"]),
        ("PDFs (Word)", ROOT / "_comum",
         ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "docx_para_pdf.ps1",
          str(A1 / "manuscrito" / "ARTIGO1_manuscrito_AWR.docx"),
          str(A1 / "manuscrito" / "ARTIGO1_material_suplementar.docx"),
          str(A2 / "manuscrito" / "ARTIGO2_manuscrito_JHydrol.docx")]),
        ("pacotes de submissão", ROOT / "_comum", py + ["pacote_submissao.py", "1", "2"]),
        ("pasta do depósito GitHub/Zenodo", ROOT / "_comum", py + ["preparar_deposito.py"]),
    ]
    fails = []
    for i, (name, cwd, cmd) in enumerate(steps, 1):
        write_status(st, f"pós-processamento {i}/{len(steps)}: {name}")
        log(f"pós-processamento {i}/{len(steps)}: {name}")
        t0 = time.time()
        with open(DIR / "pos_processamento.log", "a", encoding="utf-8") as out:
            out.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {name}: {' '.join(cmd)}\n"); out.flush()
            try:
                r = subprocess.run(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, timeout=6 * 3600)
                rc = r.returncode
            except Exception as e:
                out.write(f"{e}\n"); rc = -1
        log(f"   {name}: {'ok' if rc == 0 else f'FALHOU (código {rc})'} em {fmt_h((time.time() - t0) / 3600)}")
        if rc != 0:
            fails.append(name)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["c1", "c2"])
    ap.add_argument("--max-hours-c1", type=float, default=16.0)
    ap.add_argument("--max-hours-c2", type=float, default=60.0)
    ap.add_argument("--chunk-c1", type=int, default=400)
    ap.add_argument("--chunk-c2", type=int, default=300)
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--post-only", action="store_true")
    a = ap.parse_args()
    lock = single_instance()      # noqa: F841 (held until exit)
    keep_awake()
    st = load_state()
    log("=" * 70)
    log(f"início: {' '.join(sys.argv[1:]) or '(padrão: C1, C2 e pós-processamento)'}")
    if not a.post_only:
        tags = [a.only] if a.only else ["c1", "c2"]
        for tag in tags:
            run_chain(tag, a.chunk_c1 if tag == "c1" else a.chunk_c2,
                      a.max_hours_c1 if tag == "c1" else a.max_hours_c2, st)
    if a.no_post:
        write_status(st, "cadeias encerradas (pós-processamento não solicitado)")
        return
    for tag in ("c1", "c2"):
        st.setdefault(tag, {}).update(cb.convergence(tag))
    save_state(st)
    fails = post(st)
    ok = all(st[t]["converged"] for t in ("c1", "c2"))
    msg = ("TUDO PRONTO. " + ("As duas cadeias convergiram. " if ok else
                              "ATENÇÃO: nem todas as cadeias convergiram; o texto do manuscrito descreve isso. ")
           + ("Pós-processamento sem falhas." if not fails else
              "Etapas com falha (ver pos_processamento.log): " + "; ".join(fails)))
    write_status(st, "concluído", msg)
    log(msg)


if __name__ == "__main__":
    main()
