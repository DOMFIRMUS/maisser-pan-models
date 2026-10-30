# Evaporation pan with floating covers under infrared lamps (MAISSER-PE): data and code for a Bayesian heat–mass transfer model and state-space inference

**DOI:** [10.5281/zenodo.23787708](https://doi.org/10.5281/zenodo.23787708)

Data, code and model outputs of two modelling studies of an instrumented Class A evaporation pan
(1.185 m diameter) heated by halogen infrared lamps, ventilated by a wind tunnel and partly covered by
3-D-printed hexagonal floating plates. Project **MAISSER-PE** (FACEPE, grant APQ-1428-3.01/25),
Departamento de Energia Nuclear, Universidade Federal de Pernambuco, Recife, Brazil.

- **Campaign C1** (runs 17 Aug – 3 Sep 2026; logger file 18 Aug – 10 Sep): covered and uncovered
  runs; raw records published in doi:[10.5281/zenodo.22844940](https://doi.org/10.5281/zenodo.22844940) and copied here unchanged
  (`data/raw/C1`).
- **Campaign C2** (runs 22 Sep – 15 Oct 2026; logger file 17 Sep – 22 Oct): uncovered pan,
  water-temperature profile, two level transducers, two pyranometers, factorial lamp × fan design
  (`data/raw/C2`).

Authors: Jean Firmino Cardoso, Hugo Emanoel Bezerra de Lima, Bruno Felipe de Carvalho, Vitória Regina da Silva Bulhões, Rebeca Carolina de Albuquerque Oliveira Lancha, Hugo Barcelar de Araújo Filho, José Roberto da Silva, Ivan García-Fornaris, Alexandre Carlos Araújo de Santana, Yaicel Ge Proenza, Abel Gámez Rodríguez, Raquel Milani, Daiane Francisca do Nascimento Silva, Yelina González Pérez, Severino Martins dos Santos Neto, Allan Almeida Albuquerque, Clériston Moura Vieira Júnior, Emmanuel Damilano Dutra, Artur Paiva Coutinho, Marcelo Metri Corrêa, Marcus Metri Corrêa, Manoel Henriques de Sá Campos Filho, João Paulo Cerquinho Cajueiro, José Miguel Reichert, Daniel Milian Pérez, Antonio Celso Dantas Antonino. Contact: Jean Firmino Cardoso
(ORCID 0000-0001-6092-713X; jean.firmino@ufpe.br).

## Related manuscripts (under review)

1. *Radiative heating, stratification and evaporation suppression by modular floating covers: a
   Bayesian-calibrated heat–mass transfer model of an evaporation pan under infrared lamps* —
   folder `ARTIGO_1_MODELO_FISICO_ACOPLADO`.
2. Companion paper: state-space inference of evaporation and of its suppression from the level
   records, hierarchical model and Bayesian experimental design — folder `ARTIGO_2_INFERENCIA_ESTOCASTICA`.

## Layout

```
_comum/tanque/            shared package: paths, geometry, campaign metadata, record processing,
                          event detection, water and air properties, figure style
_comum/fig_esquema.py     schematic of the rig
ARTIGO_1_MODELO_FISICO_ACOPLADO/codigo/   Article 1: physical model, priors, calibration (C2, then C1), posterior summaries,
               convergence driver, scenarios (maps, Shapley, Sobol, posterior band), figures,
               results collector
ARTIGO_1_MODELO_FISICO_ACOPLADO/saidas/   calibration (modes map_*.json, MCMC chains mcmc_chain_*.npz, logs), simulations,
               tables (CSV), figures (PDF + PNG), convergence logs, resultados_artigo1.json
ARTIGO_2_INFERENCIA_ESTOCASTICA/codigo/   Article 2: state-space model, driver fits and cross-validation (C2), direct
               suppression (C1), hierarchical model, Bayesian design, figures, results collector
ARTIGO_2_INFERENCIA_ESTOCASTICA/saidas/   fits, cross-validation, suppression estimates, hierarchical posterior, design, tables,
               figures, resultados_artigo2.json
data/raw/C1    CR1000 TOA5 exports of C1 and the manual micrometer readings
data/raw/C2    CR1000 TOA5 export of C2 (one-minute records), gzip-compressed (gunzip gives the
               original file; pandas reads it directly)
```

Folder and file names are in Portuguese (codigo = code, saidas = outputs, tabelas = tables,
figuras = figures, calibracao = calibration, simulacoes = simulations).

## Reproduce

Python 3.12 or later with numpy, scipy, pandas and matplotlib (`pip install -r requirements.txt`).
The first call that reads the records builds `_comum/dados_processados/serie_1min_C1_C2.pkl` from
`data/raw` (other raw locations can be set with the environment variables `MAISSER_RAW_C1` and
`MAISSER_RAW_C2`).

Article 1 (run in `ARTIGO_1_MODELO_FISICO_ACOPLADO/codigo`; the times are for a 12-thread laptop):

```
python calibracao.py --stage map --gens 100 --walkers 40 --rounds 2 --tag c2     # C2 mode, ~10 min
python calibracao.py --stage mcmc --iter 1500 --walkers 40 --tag c2              # C2 chain, ~3 h
python resumo_posterior.py c2 --promote                                          # C2 posterior mode
python calibracao_c1.py --stage all --gens 80 --iter 2000 --walkers 36 --rounds 2 --base ../saidas/calibracao/map_c2.json --tag c1 --qdy-gain 0.944
bash rodar_pos_c1.sh                     # C1 mode, gain variants, scenarios, figures, collector
python convergir_cadeias.py --no-post    # continues both chains until convergence (tens of hours)
python resumo_posterior.py c2 c1         # posterior summaries (second half of each chain)
python executar_simulacoes.py posterior  # posterior band of the scenarios
python figuras_artigo1.py; python figuras_si_artigo1.py; python resultados_artigo1.py
```

The chains are stored as `mcmc_chain_<tag>.npz` followed by its continuations
`mcmc_chain_<tag>_cont<k>.npz` (arrays `chain` [iteration, walker, parameter] on the unconstrained
scale, `lp` log-posterior, `names`, `acc`); `calibracao.load_chain(tag)` concatenates them and
`priors.Prior` maps them to physical units. Convergence criteria: maximum rank-normalised split
R-hat below 1.1 and the retained half spanning at least 50 integrated autocorrelation times
(`calibracao.convergence`; history in `saidas/convergencia/`).

Article 2 (run in `ARTIGO_2_INFERENCIA_ESTOCASTICA/codigo`, after Article 1):

```
python recuperacao.py 20                 # synthetic recovery test
python run_pipeline.py                   # C2 driver fits, cross-validation, C1 direct suppression
python ajuste_c1.py churchill_usagi _cu  # sensitivity to the transfer equation
python hierarquico.py; python delineamento.py
python resultados_artigo2.py; python figuras_artigo2.py
```

## Licences

Code: MIT (`LICENSE`). Data, outputs and documentation: CC BY 4.0 (`LICENSE-data.txt`).
Please cite this deposit (see `CITATION.cff`) and, for the C1 records, doi:10.5281/zenodo.22844940.
