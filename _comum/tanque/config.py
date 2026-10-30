# -*- coding: utf-8 -*-
"""
Paths, geometry and campaign metadata shared by the two articles.

Two measurement campaigns on the same Class A pan (DEN/UFPE, Recife):

  C1  14 Aug - 10 Sep 2026, CR1000 program codigo_sensores_v8.CR1.
      Covered and uncovered runs (V1 and V2 hexagonal plates), fan weak,
      strong or off, lamps at several settings. Raw files: data/raw/C1 (also doi:10.5281/zenodo.22844940).
  C2  17 Sep - 22 Oct 2026, programs v18-v22 (v22_FINAL from 24 Sep).
      Uncovered pan; water-temperature profile (4, 12, 20 cm), two level
      transducers (QDY30A and CS456), two pyranometers, three lamp settings
      crossed with fan settings, dark (lamps off) periods between runs.
      Raw file: data/raw/C2/CR1000_dados_do_tanque.dat.gz.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from pathlib import Path
import os
import numpy as np

# Paths ----------------------------------------------------------------------
COMUM = Path(__file__).resolve().parents[1]
PROJETO = COMUM.parent                                   # 08_MODELAGEM_TANQUE_MAISSER
DOUTORADO = PROJETO.parents[1]
RAW_C1 = Path(os.environ.get("MAISSER_RAW_C1", PROJETO / "data" / "raw" / "C1"))
RAW_C2 = Path(os.environ.get("MAISSER_RAW_C2", PROJETO / "data" / "raw" / "C2" / "CR1000_dados_do_tanque.dat.gz"))
MICROMETER_C1 = RAW_C1 / "micrometer_readings.csv"
PROCESSADOS = COMUM / "dados_processados"
PROCESSADOS.mkdir(parents=True, exist_ok=True)

# Pan geometry ------------------------------------------------------------------
PAN_DIAMETER_M = 1.185                       # internal diameter
PAN_RADIUS_M = PAN_DIAMETER_M / 2
PAN_AREA_M2 = np.pi * PAN_RADIUS_M ** 2      # 1.1029 m2
PAN_DEPTH_M = 0.25
PAN_PERIMETER_M = np.pi * PAN_DIAMETER_M

# Hexagonal plates (regular hexagon, side 110 mm) --------------------------------
HEX_SIDE_M = 0.110
HEX_AREA_M2 = 1.5 * np.sqrt(3) * HEX_SIDE_M ** 2     # 0.031437 m2
HEX_PERIMETER_M = 6 * HEX_SIDE_M


def coverage_fraction(n_plates):
    return n_plates * HEX_AREA_M2 / PAN_AREA_M2


# Level records -------------------------------------------------------------------
LEVEL_QUANTUM_MM = 0.1        # FP2 storage at 100-270 mm
CS456_QUANTUM_PSI = 0.001     # Lvl_psi stored with 3 decimals
PSI_TO_MM_H2O = 703.0696      # 1 psi in mm of water at 4 degC

# Cover timeline of campaign C1 ----------------------------------------------------
# (start, end, n_plates, version). Reconstructed from the buoyancy steps of the
# level transducer (deployment +2.4 to +3.7 mm, removal -2.9 to -4.3 mm; see
# eventos.level_events) together with the micrometer notes. The steps show that
# the plates were removed at the end of the runs of 21, 24, 25 and 28 Aug, so
# those nights were uncovered. Minutes between a step's start and end are
# masked (n_plates = NaN).
COVER_C1 = [
    ("2026-08-14 00:00", "2026-08-18 10:34", 0, "-"),
    ("2026-08-18 10:53", "2026-08-19 09:53", 27, "V1"),
    ("2026-08-19 10:13", "2026-08-21 07:43", 0, "-"),
    ("2026-08-21 08:08", "2026-08-21 15:44", 27, "V1"),
    ("2026-08-21 16:03", "2026-08-24 12:54", 0, "-"),
    ("2026-08-24 13:13", "2026-08-24 18:16", 27, "V1"),
    ("2026-08-24 18:35", "2026-08-25 10:54", 0, "-"),
    ("2026-08-25 11:12", "2026-08-25 17:12", 27, "V2"),
    ("2026-08-25 17:30", "2026-08-26 09:28", 0, "-"),
    ("2026-08-26 09:45", "2026-08-28 16:51", 26, "V2"),
    ("2026-08-28 17:10", "2026-08-31 08:15", 0, "-"),
    ("2026-08-31 08:33", "2026-09-01 13:32", 26, "V2"),
    ("2026-09-01 13:55", "2026-09-01 16:15", 0, "-"),
    ("2026-09-01 16:25", "2026-09-03 17:20", 26, "V2"),
]

# Rate windows of C1 (paper-tank, maisser_core.WINDOWS) ---------------------------
WINDOWS_C1 = [
    ("W01", "2026-08-17 09:20", "2026-08-17 17:26", 0, "-", 0, "weak"),
    ("W02", "2026-08-18 11:20", "2026-08-18 18:20", 1, "V1", 27, "weak"),
    ("W03", "2026-08-19 10:20", "2026-08-19 15:41", 0, "-", 0, "weak"),
    ("W04", "2026-08-20 14:35", "2026-08-20 18:24", 0, "-", 0, "strong"),
    ("W05", "2026-08-21 11:12", "2026-08-21 15:44", 1, "V1", 27, "strong"),
    ("W06", "2026-08-24 07:57", "2026-08-24 13:00", 0, "-", 0, "strong"),
    ("W07", "2026-08-24 13:25", "2026-08-24 18:01", 1, "V1", 27, "strong"),
    ("W08", "2026-08-25 11:25", "2026-08-25 17:18", 1, "V2", 27, "strong"),
    ("W09", "2026-08-26 09:47", "2026-08-26 18:49", 1, "V2", 26, "strong"),
    ("W10", "2026-08-27 09:22", "2026-08-27 17:04", 1, "V2", 26, "strong"),
    ("W11", "2026-08-28 07:57", "2026-08-28 16:55", 1, "V2", 26, "strong"),
    ("W12", "2026-08-31 08:45", "2026-08-31 17:37", 1, "V2", 26, "calm"),
    ("W13", "2026-09-01 08:35", "2026-09-01 13:35", 1, "V2", 26, "calm"),
    ("W14", "2026-09-02 07:56", "2026-09-02 16:30", 1, "V2", 26, "calm"),
    ("W15", "2026-09-03 08:08", "2026-09-03 16:39", 1, "V2", 26, "calm"),
]

# Campaign C2 analysis period: from program v21 (22 Sep 18:24, after the fill,
# the recalibration of the level sensors and the correction of the swapped
# 4/12 cm thermistor channels) to the draining for cleaning at the end of
# 15 Oct (log 'informacoes sobre os experimentos.xlsx').
C2_START = "2026-09-22 18:40"
C2_END = "2026-10-15 17:00"

RECONFIGURATION_C1 = "2026-09-10"

# Heights of the C2 thermistors above the pan bottom (m); the CS456 body lies
# on the bottom. The 20 cm probe is out of the water when depth < 0.20 m.
PROBE_HEIGHT_C2 = {"Tw_04": 0.04, "Tw_12": 0.12, "Tw_20": 0.20, "Tw_cs": 0.01}
HMP45C_HEIGHT_C2 = 0.40   # above the pan bottom

# Absolute depth in C2: the ODY30A was recalibrated to a hand measurement of
# 224.5 mm on 22 Sep 18:17 (program v21 changelog); later depths follow the ODY
# changes divided by its gain relative to water mass (ODY_GAIN, estimated in
# the calibrations; the depth enters only the submergence test of the 20 cm
# probe and the initial depth of each segment, so mm-level errors are harmless).
DEPTH_REF_C2 = ("2026-09-22 18:17", 0.2245)
ODY_GAIN = 1.12
# the 20 cm thermistor is used only when submerged by at least 12 mm
SUBMERGED_MIN = 0.012


def corrected_depth_c2(level_a_mm, gain=None):
    """C2 water depth (m) from the ODY reading (gain relative to water mass:
    ODY_GAIN unless given)."""
    H_ref = DEPTH_REF_C2[1]
    g = ODY_GAIN if gain is None else gain
    return H_ref + (np.asarray(level_a_mm, float) - 1000.0 * H_ref) / 1000.0 / g
