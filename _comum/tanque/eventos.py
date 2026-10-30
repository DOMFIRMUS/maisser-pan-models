# -*- coding: utf-8 -*-
"""
Quality control and event segmentation of the harmonised 1-min series.

  * level steps: refills (> +5 mm), cover deployments / removals (2-5 mm),
    sensor handling; each step closes a 'segment' of the level record in
    which water mass can only decrease by evaporation;
  * spikes and out-of-water readings are masked;
  * lamp state (Rs_a > 20 W m-2), fan state (U > 0.3 m s-1) and 'runs'
    (contiguous lamp-on periods) with their mean forcing.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
import numpy as np
import pandas as pd

LAMP_ON_WM2 = 20.0
FAN_ON_MS = 0.3
REFILL_MM = 5.0
STEP_MM = 1.8
GUARD_MIN = 20          # minutes masked around each step (sensor settling)


def _robust_steps(x, lag=10, thr=STEP_MM):
    """Indices where the 10-min median changes by more than `thr` (mm).

    Returns a list of (t_start, t_end, step) with step = median after -
    median before. Consecutive exceedances are merged into one event.
    """
    med = x.rolling(5, center=True, min_periods=3).median()
    d = med.shift(-lag) - med
    hit = d.abs() > thr
    events = []
    if not hit.any():
        return events
    t = hit[hit].index
    grp = (t.to_series().diff() > pd.Timedelta(minutes=lag)).cumsum().values
    for g in np.unique(grp):
        tt = t[grp == g]
        t0, t1 = tt[0], tt[-1] + pd.Timedelta(minutes=lag)
        pre = x[t0 - pd.Timedelta(minutes=15): t0].median()
        post = x[t1: t1 + pd.Timedelta(minutes=15)].median()
        events.append((t0, t1, float(post - pre)))
    return events


def level_events(df, col="level_a"):
    rows = []
    for camp, c in df.groupby("campaign"):
        x = c[col].where(c[col] > 50.0)
        for t0, t1, s in _robust_steps(x):
            if not np.isfinite(s):
                kind = "end_or_gap"
            elif s > REFILL_MM:
                kind = "refill"
            elif s > STEP_MM:
                kind = "cover_on_or_handling"
            elif s < -REFILL_MM:
                kind = "drain"
            else:
                kind = "cover_off_or_handling"
            rows.append(dict(campaign=camp, t0=t0, t1=t1, step_mm=s, kind=kind))
    return pd.DataFrame(rows)


def qc(df):
    """Adds masks and segment ids. Returns a copy."""
    df = df.copy()
    for col in ("level_a", "level_b"):
        if col not in df:
            continue
        x = df[col].where(df[col] > 50.0)
        # spikes: deviation from 7-min running median > 1.5 mm
        med = x.rolling(7, center=True, min_periods=4).median()
        x = x.where((x - med).abs() < 1.5)
        df[col + "_qc"] = x
    ev = level_events(df, "level_a")
    seg = np.zeros(len(df), dtype=int)
    mask = np.zeros(len(df), dtype=bool)
    k = 0
    starts = sorted(ev.t0.tolist())
    idx = df.index
    breaks = [idx.searchsorted(t) for t in starts]
    for i, (t0, t1) in enumerate(zip(ev.t0, ev.t1)):
        m = (idx >= t0 - pd.Timedelta(minutes=5)) & (idx <= t1 + pd.Timedelta(minutes=GUARD_MIN))
        mask |= m
    # segments: break at every event and at campaign changes
    # cut also where the campaign changes (excluding the first row)
    camp = df.campaign.to_numpy()
    changes = list(np.flatnonzero(camp[1:] != camp[:-1]) + 1)
    cuts = sorted(set(breaks) | set(changes))
    cuts = sorted(set(cuts + [0, len(df)]))
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg[a:b] = k
        k += 1
    df["segment"] = seg
    df["step_mask"] = mask
    for col in ("level_a_qc", "level_b_qc"):
        if col in df:
            df.loc[mask, col] = np.nan
    df["lamp_on"] = df[["Rs_sp", "Rs_li"]].max(axis=1) > LAMP_ON_WM2
    df["fan_on"] = df["U"] > FAN_ON_MS
    return df, ev


def runs(df, min_minutes=60):
    """Contiguous lamp-on periods with mean forcing (one row per run)."""
    on = df["lamp_on"].fillna(False).astype(bool)
    gid = (on != on.shift()).cumsum()
    rows = []
    for g, s in df.groupby(gid):
        if not on.loc[s.index[0]] or len(s) < min_minutes:
            continue
        rows.append(dict(
            campaign=s.campaign.iloc[0], start=s.index[0], end=s.index[-1],
            minutes=len(s), Rs_sp=s.Rs_sp.mean(), Rs_li=s.Rs_li.mean(),
            U=s.U.mean(), U_sd=s.U.std(), Ta=s.Ta.mean(), RH=s.RH.mean(),
            ea=s.ea.mean(), phi=s.phi.mean(), n_plates=s.n_plates.mean(),
            version=s.version.dropna().iloc[0] if s.version.notna().any() else None))
    r = pd.DataFrame(rows)
    if len(r):
        r["lamp_level"] = pd.cut(r.Rs_li, [0, 250, 500, 800, 2000],
                                 labels=["very_low", "low", "mid", "high"])
    return r
