# -*- coding: utf-8 -*-
"""
Segments, forcing and observations for the pan model.

A segment is a stretch of record between two level steps (refills, plate
handling, cover changes), during which water can leave the pan only by
evaporation. Each segment is simulated from an initial profile interpolated
between the thermistors, and the first hours (spin-up) are left out of the
likelihood.

Time convention. TOA5 timestamps mark the END of each 1-min record (C2: one
scan at the timestamp; C1: mean of the three 20-s scans ending there).
Forcing and observations are therefore binned closed on the right and
labelled on the left: the 5-min forcing bin labelled t holds the records
(t, t+5], which is the interval integrated by model step t -> t+5; the 10-min
observation bin labelled t holds (t, t+10] and is compared with the
trapezoidal mean of the model states at t, t+5 and t+10. Bins are aligned to
the first minute of each segment, which is also the time of the initial
state.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_comum"))
from tanque import dados, eventos, config as cfg      # noqa: E402

DT_MODEL = 300.0          # s
OBS_MIN = 10
SPINUP_H = 6.0
PROBES_C2 = {"Tw_cs": 0.01, "Tw_04": 0.04, "Tw_12": 0.12, "Tw_20": 0.20}
# absolute depth and the submergence rule of the 20 cm probe: see config.py
DEPTH_REF = cfg.DEPTH_REF_C2
ODY_GAIN = cfg.ODY_GAIN
SUBMERGED_MIN = cfg.SUBMERGED_MIN
LAMP_ON = 50.0


@dataclass
class Segment:
    name: str
    campaign: str
    start: pd.Timestamp
    end: pd.Timestamp
    forcing: pd.DataFrame        # 5-min forcing (bins (t, t+5], labelled t)
    obs: pd.DataFrame            # 10-min observations (bins (t, t+10], labelled t)
    T0_heights: dict             # probe height (or name) -> initial temperature
    H0: float                    # initial depth (m)
    phi: float = 0.0
    version: str = "-"
    extra: dict = field(default_factory=dict)


_CACHE = {}


def series():
    if "q" not in _CACHE:
        df = dados.load_all()
        q, ev = eventos.qc(df)
        _CACHE["q"], _CACHE["ev"] = q, ev
    return _CACHE["q"], _CACHE["ev"]


def lamp_config_minute(I, campaign):
    """Lamp-configuration code per 1-min record from the SP-LITE index level.

    C2: 0 = all 18 lamps (index ~820), 1 = upper rail only (~300),
        2 = lower rail only (~520). C1 codes are assigned by date (C1_LAMP_PERIODS).
    """
    I = np.asarray(I, float)
    c = np.zeros(len(I), dtype=int)
    if campaign == "C2":
        c[(I > LAMP_ON) & (I <= 420)] = 1
        c[(I > 420) & (I <= 700)] = 2
    return c


def _bins(index, start, minutes):
    """Left label of the right-closed bin of each timestamp, aligned to start."""
    k = np.ceil(((index - start) / pd.Timedelta(minutes=1)).to_numpy() / minutes) - 1
    return start + pd.to_timedelta(np.maximum(k, 0) * minutes, unit="min")


def _forcing(s, start, campaign):
    I = s["Rs_sp"].clip(lower=0)
    U = s["U"].clip(lower=0)
    # short gaps (<= 10 min): carry the lamp and fan state across
    I = I.interpolate(limit=10, limit_direction="both").fillna(0.0)
    U = U.interpolate(limit=10, limit_direction="both").fillna(0.0)
    f = pd.DataFrame({
        "I": I, "U": U,
        "Ta": s["Ta"].interpolate(limit_direction="both"),
        "ea": s["ea"].interpolate(limit_direction="both"),
        "phi": s["phi"].ffill().bfill().fillna(0.0),
    }, index=s.index)
    if campaign == "C2":
        code = lamp_config_minute(I.to_numpy(), "C2")
    else:
        code = c1_lamp_config(s.index)
    f["code"] = np.where(I.to_numpy() > LAMP_ON, code, -1)
    f = f[f.index > start]
    lab = _bins(f.index, start, 5)
    g = f.groupby(lab)
    out = g[["I", "U", "Ta", "ea", "phi"]].mean()

    def mode_on(x):
        x = x[x >= 0]
        return int(x.mode().iloc[0]) if len(x) else (0 if campaign == "C2" else int(code[0]))
    out["cfg"] = g["code"].apply(mode_on).astype(int)
    full = pd.date_range(start, out.index.max(), freq="5min")
    out = out.reindex(full)
    out[["Ta", "ea", "phi"]] = out[["Ta", "ea", "phi"]].ffill().bfill()
    out[["I", "U"]] = out[["I", "U"]].fillna(0.0)
    out["cfg"] = out["cfg"].ffill().bfill().astype(int)
    return out


def _obs(s, start, cols):
    o = s[cols]
    o = o[o.index > start]
    lab = _bins(o.index, start, OBS_MIN)
    out = o.groupby(lab).mean()
    full = pd.date_range(start, out.index.max(), freq=f"{OBS_MIN}min")
    return out.reindex(full)


def corrected_depth(level_a_mm, t=None):
    """Depth (m) from the ODY reading.

    The single-point recalibration made the ODY read the hand-measured depth
    (224.5 mm) on 22 Sep 18:17; later depths follow the ODY changes divided by
    its gain relative to water mass."""
    return cfg.corrected_depth_c2(level_a_mm)


def probe_offsets_c2():
    """Inter-probe offsets relative to the CS450 body, from dark, calm periods
    with a well-mixed column (lamps off for > 8 h, |T12 - Tcs| < 0.3 K)."""
    if "off" in _CACHE:
        return _CACHE["off"]
    q, _ = series()
    c = q[(q.campaign == "C2")].copy()
    on = c["lamp_on"].fillna(False).astype(bool)
    last_on = c.index.to_series().where(on).ffill()
    dark = (~on) & ((c.index.to_series() - last_on) > pd.Timedelta(hours=8)) & (c["U"].fillna(0) < 0.3)
    sub = c[dark.to_numpy()]
    sub = sub[(sub.Tw_12 - sub.Tw_cs).abs() < 0.3]
    off = {"Tw_cs": 0.0}
    for k in ("Tw_04", "Tw_12", "Tw_20"):
        dd = (sub[k] - sub.Tw_cs)
        if k == "Tw_20":
            dd = dd[corrected_depth(sub.level_a_qc, sub.index) - 0.20 >= SUBMERGED_MIN]
        off[k] = float(dd.median())
    _CACHE["off"] = off
    return off


def c2_segments(min_hours=6.0, verbose=False):
    q, ev = series()
    c = q[q.campaign == "C2"]
    e = ev[ev.campaign == "C2"]                    # every detected level event cuts
    cuts = [(r.t0, r.t1) for _, r in e.sort_values("t0").iterrows()]
    bounds = []
    t_start = pd.Timestamp(cfg.C2_START)
    for t0, t1 in cuts:
        if t1 <= t_start:
            continue
        bounds.append((t_start, t0 - pd.Timedelta(minutes=5)))
        t_start = t1 + pd.Timedelta(minutes=20)
    bounds.append((t_start, pd.Timestamp(cfg.C2_END)))
    off = probe_offsets_c2()
    segs = []
    for i, (a, b) in enumerate(bounds):
        if (b - a) < pd.Timedelta(hours=min_hours):
            if verbose and b > a:
                print(f"[dados_modelo] C2 bound {a} - {b} dropped ({(b - a)})")
            continue
        s = c[a:b]
        a0 = s.index[0]
        f = _forcing(s, a0, "C2")
        o = _obs(s, a0, ["Tw_cs", "Tw_04", "Tw_12", "Tw_20", "level_a_qc", "level_b_qc"])
        o["depth_obs"] = corrected_depth(o["level_a_qc"].interpolate(limit_direction="both"), o.index)
        first = s.iloc[:3].median(numeric_only=True)       # records at a0..a0+2 min
        H0 = float(corrected_depth(first["level_a_qc"], a0))
        T0 = {h: float(first[k]) - off[k] for k, h in PROBES_C2.items()
              if np.isfinite(first[k]) and (k != "Tw_20" or H0 - 0.20 >= SUBMERGED_MIN)}
        segs.append(Segment(f"C2-S{len(segs) + 1}", "C2", a0, s.index[-1], f, o, T0, H0,
                            extra=dict(offsets=off)))
    return segs


def initial_profile(seg: Segment, zc_frac, H0=None):
    """Interpolate the probe temperatures onto the model layers."""
    H0 = seg.H0 if H0 is None else H0
    hs = np.array(sorted(seg.T0_heights))
    Ts = np.array([seg.T0_heights[h] for h in hs])
    height = H0 * (1 - np.asarray(zc_frac))          # height of layer centres above floor
    return np.interp(height, hs, Ts)


def forcing_arrays(seg: Segment, I_scale=1.0):
    from modelo_fisico import Forcing
    f = seg.forcing
    t = (f.index - f.index[0]).total_seconds().to_numpy()
    return Forcing(t=t, I=f.I.to_numpy() * I_scale, U=f.U.to_numpy(), Ta=f.Ta.to_numpy(),
                   ea=f.ea.to_numpy(), phi=f.phi.to_numpy(), cfg=f.cfg.to_numpy().astype(int))


def obs_steps(seg: Segment):
    """Model steps (i0, i1) spanning each 10-min observation bin; -1 if absent.

    Bin labelled t covers model steps starting at t (i0) and t+5 (i1)."""
    pos = {t: i for i, t in enumerate(seg.forcing.index)}
    rows = []
    for t in seg.obs.index:
        i0 = pos.get(t); i1 = pos.get(t + pd.Timedelta(minutes=5))
        rows.append((-1, -1) if i0 is None or i1 is None else (i0, i1))
    return np.array(rows)


# ---------------------------------------------------------------------------
# Campaign C1 (covered and uncovered runs)
# ---------------------------------------------------------------------------
C1_START = "2026-08-17 09:05"
C1_END = "2026-09-03 17:15"
# lamp periods of C1 (thesis log): mixed halogen + LED reflectors until 19 Aug,
# 15 drying lamps on 20-21 Aug, 18 drying lamps (variable number on) from 24 Aug
C1_LAMP_PERIODS = [("2026-08-14", "2026-08-19 23:59", 3), ("2026-08-20", "2026-08-21 23:59", 4),
                   ("2026-08-22", "2026-09-10", 5)]
PROBES_C1 = ["Tw_p1", "Tw_p2", "Tw_p3"]


# Which of the 18 drying lamps were on was not logged; the index stayed at one level
# on each day (about 900, 860, 730, 630, 285 or 150 W m-2). Days whose median index
# with the lamps on is below C1_LOW_INDEX get their own configuration (code 6),
# because a few lamps see the float and the pan in a different proportion than
# the full array does (in C2 the two rails differ by a factor of 1.7).
C1_LOW_INDEX = 450.0


def c1_low_index_days():
    if "low_days" not in _CACHE:
        q, _ = series()
        c = q[(q.campaign == "C1") & (q.index >= C1_LAMP_PERIODS[-1][0])]
        I = c["Rs_sp"]
        med = I[I > LAMP_ON].groupby(I[I > LAMP_ON].index.normalize()).median()
        _CACHE["low_days"] = set(med.index[med < C1_LOW_INDEX])
    return _CACHE["low_days"]


def c1_lamp_config(index):
    c = np.full(len(index), 5, dtype=int)
    for a, b, code in C1_LAMP_PERIODS:
        c[(index >= a) & (index <= b)] = code
    low = np.asarray(pd.DatetimeIndex(index).normalize().isin(list(c1_low_index_days())))
    c[(c == 5) & low] = 6
    return c


def c1_segments(min_hours=4.0):
    """C1 segments: break at every level step and at every change of cover."""
    q, ev = series()
    c = q[(q.campaign == "C1") & (q.index >= C1_START) & (q.index <= C1_END)].copy()
    e = ev[ev.campaign == "C1"]
    mask = np.zeros(len(c), bool)
    for _, r in e.iterrows():
        mask |= (c.index >= r.t0 - pd.Timedelta(minutes=5)) & (c.index <= r.t1 + pd.Timedelta(minutes=20))
    npl = c["n_plates"]
    mask |= npl.isna().to_numpy()
    key = (npl.ffill().fillna(-1).astype(int).astype(str) + "|" + c["version"].fillna("?").astype(str))
    seg = np.zeros(len(c), int)
    k = 0
    for i in range(1, len(c)):
        if mask[i]:
            seg[i] = -1
            continue
        if mask[i - 1] or key.iloc[i] != key.iloc[i - 1]:
            k += 1
        seg[i] = k
    seg[0] = -1 if mask[0] else 0
    c["segc"] = seg
    out = []
    for sid, s in c[c.segc >= 0].groupby("segc"):
        if (s.index[-1] - s.index[0]) < pd.Timedelta(hours=min_hours):
            continue
        a0 = s.index[0]
        f = _forcing(s, a0, "C1")
        o = _obs(s, a0, PROBES_C1 + ["level_a_qc"])
        first = s.iloc[:3].median(numeric_only=True)
        H0 = float(first["level_a_qc"]) / 1000.0
        segm = Segment(f"C1-S{len(out) + 1:02d}", "C1", a0, s.index[-1], f, o,
                       {"p1": float(first["Tw_p1"]), "p2": float(first["Tw_p2"]),
                        "p3": float(first["Tw_p3"])},
                       H0, phi=float(s.phi.iloc[0]), version=str(s.version.iloc[0]),
                       extra=dict(n_plates=int(s.n_plates.iloc[0])))
        out.append(segm)
    return out
