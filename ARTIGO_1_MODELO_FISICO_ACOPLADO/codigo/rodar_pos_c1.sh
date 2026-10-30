#!/usr/bin/env bash
# Article 1 pipeline after the C1 MCMC: promote the posterior mode, refit the
# gain-sensitivity variants, rerun the scenarios, figures, checks and collector,
# and build the manuscript. Each step is logged to ../saidas/pos_c1.log.
set -e
cd "$(dirname "$0")"
PY=/c/Users/jeanf/AppData/Local/Python/pythoncore-3.14-64/python.exe
LOG=../saidas/pos_c1.log
CAL=../saidas/calibracao
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

step "promote C1 posterior mode"
$PY -X utf8 resumo_posterior.py c1 --promote >> "$LOG" 2>&1
cp $CAL/map_c1.json $CAL/map_c1_v9_F.json

for g in 0.884 1.004; do
  tag=c1_g${g/./}
  step "variant with QDY30A gain $g ($tag)"
  $PY -X utf8 calibracao_c1.py --stage map --gens 60 --walkers 36 --rounds 1 --base $CAL/map_c2.json \
      --tag $tag --qdy-gain $g --init-from $CAL/mcmc_chain_c1.npz --weights-from $CAL/map_c1.json >> "$LOG" 2>&1
done

step "variants table"
$PY -X utf8 variantes_c1.py >> "$LOG" 2>&1
step "scenarios"
$PY -X utf8 executar_simulacoes.py mapa shapley sobol espectral projeto recesso sem_vias posterior >> ../saidas/simulacoes_run5.out 2>&1
step "numerical verification"
$PY -X utf8 verificacao_numerica.py >> "$LOG" 2>&1
step "figures"
$PY -X utf8 figuras_artigo1.py >> "$LOG" 2>&1
$PY -X utf8 figuras_si_artigo1.py >> "$LOG" 2>&1
step "collector"
$PY -X utf8 resultados_artigo1.py >> "$LOG" 2>&1
step "build"
$PY -X utf8 ../manuscrito/construir_manuscrito.py >> "$LOG" 2>&1
step "done"
