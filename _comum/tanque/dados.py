# -*- coding: utf-8 -*-
"""
Loading and harmonisation of the two CR1000 campaigns into one 1-min frame.

Harmonised columns (SI-like units, 1-min means unless noted):

  campaign            'C1' or 'C2'
  Rs_sp, Rs_li        irradiance index of the Kipp & Zonen SP-LITE and of the
                      LI-COR LI200X silicon pyranometers (W m-2, calibrated to
                      sunlight; an index of lamp output, not the radiant load).
                      C1: SP-LITE on a float in the water-surface plane;
                      C2: LI200X levelled outside the pan, SP-LITE position
                      undocumented. Rs_a is an alias of the SP-LITE.
  U                   air speed over the pan (m s-1)
  Ta, RH              air temperature (degC) and relative humidity (%) as
                      logged (RH is the per-minute minimum in both programs)
  ea                  air vapour pressure (Pa) = RH/100 esat(Ta)
  Tw_*                water temperatures (degC); C2: Tw_04, Tw_12, Tw_20 and
                      Tw_cs (CS45x body); C1: Tw_p1, Tw_p2, Tw_p3
  level_a             QDY30A hydrostatic transducer (mm), both campaigns
  level_b             CS456 transducer (mm), C2 only
  n_plates, version, phi   cover state (C1 from config.COVER_C1; C2 uncovered)

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as cfg
from . import propriedades as pr

NA = ["NAN", "NaN", "-99.999", "-99999", "-100", "-7999", "7999", "-9999"]

MAP_C1 = {
    "irradiancia_Avg": "Rs_sp", "irradiancia_licor_Avg": "Rs_li", "WS_ms_Avg": "U",
    "AirTC_Avg": "Ta", "RH_Min": "RH",
    "T108_C_Avg": "Tw_p1", "T108_C_2_Avg": "Tw_p2", "T108_C_3_Avg": "Tw_p3",
    "nivel_tanque_Avg": "level_a", "nivel_tanque_Std": "level_a_sd",
}
MAP_C2 = {
    "SlrW_Avg": "Rs_li", "SlrW_2_Avg": "Rs_sp", "WS_ms_Avg": "U",
    "AirTC_Avg": "Ta", "RH_Min": "RH",
    "T108_4cm_Avg": "Tw_04", "T108_12cm_Avg": "Tw_12", "T108_20cm_Avg": "Tw_20",
    "Temp_C_CS450_Avg": "Tw_cs",
    "Level_mm_Avg": "level_a", "Level_CS456_mm_Avg": "level_b", "Lvl_psi_Avg": "psi_b",
    "pH_Avg": "pH",
}


def _read_toa5(path):
    d = pd.read_csv(path, skiprows=[0, 2, 3], na_values=NA, low_memory=False,
                    encoding="latin-1")
    d = d[d.TIMESTAMP.astype(str).str.match(r"\d{4}-\d{2}-\d{2}")].copy()
    d["TIMESTAMP"] = pd.to_datetime(d.TIMESTAMP, format="%Y-%m-%d %H:%M:%S")
    return d


def load_c1(raw_dir=cfg.RAW_C1):
    files = sorted(Path(raw_dir).glob("*.dat"))
    d = pd.concat([_read_toa5(f) for f in files], ignore_index=True)
    d = d.sort_values("TIMESTAMP").drop_duplicates("TIMESTAMP", keep="first")
    d = d[d.TIMESTAMP < cfg.RECONFIGURATION_C1]
    out = d.rename(columns=MAP_C1)[["TIMESTAMP"] + list(MAP_C1.values())]
    out = out.set_index("TIMESTAMP")
    out["campaign"] = "C1"
    out["n_plates"] = np.nan
    out["version"] = None
    for a, b, n, v in cfg.COVER_C1:
        m = (out.index >= a) & (out.index <= b)
        out.loc[m, "n_plates"] = n
        out.loc[m, "version"] = v
    return out


def load_c2(path=cfg.RAW_C2):
    d = _read_toa5(path)
    d = d.sort_values("TIMESTAMP").drop_duplicates("TIMESTAMP", keep="last")
    out = d.rename(columns=MAP_C2)[["TIMESTAMP", "Code_Version"] + list(MAP_C2.values())]
    out = out.set_index("TIMESTAMP")
    out["campaign"] = "C2"
    out["n_plates"] = 0.0
    out["version"] = "-"
    return out


def derived(df):
    df = df.copy()
    df["Rs_a"] = df["Rs_sp"]
    df["phi"] = cfg.coverage_fraction(df["n_plates"])
    df["ea"] = pr.esat(df["Ta"]) * df["RH"] / 100.0
    df["VPD"] = pr.esat(df["Ta"]) - df["ea"]
    return df


def load_all(cache=True):
    """Both campaigns on a regular 1-min grid (gaps kept as NaN rows)."""
    f = cfg.PROCESSADOS / "serie_1min_C1_C2.pkl"
    if cache and f.exists():
        # rebuild when the code or the raw files are newer than the cache
        here = Path(__file__).resolve().parent
        deps = [here / "config.py", here / "dados.py", cfg.RAW_C2] + sorted(Path(cfg.RAW_C1).glob("*.dat"))
        newest = max(Path(d).stat().st_mtime for d in deps if Path(d).exists())
        if f.stat().st_mtime >= newest:
            return pd.read_pickle(f)
    c1 = derived(load_c1())
    c2 = derived(load_c2())
    c2 = c2[(c2.index >= cfg.C2_START) & (c2.index <= cfg.C2_END)]
    frames = []
    for c in (c1, c2):
        idx = pd.date_range(c.index.min(), c.index.max(), freq="1min")
        c = c.reindex(idx)
        c["campaign"] = c["campaign"].ffill()
        frames.append(c)
    df = pd.concat(frames)
    df.index.name = "time"
    df.to_pickle(f)
    return df
