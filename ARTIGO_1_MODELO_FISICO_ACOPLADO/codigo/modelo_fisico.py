# -*- coding: utf-8 -*-
"""
Coupled heat-mass transfer model of the Class A pan with floating covers.

State: temperature profile T(z, t) of the water column (N layers on a stretched
sigma grid, fine at the surface), plate temperature T_p(t) and water depth H(t).
Every array carries a leading ensemble axis W, so many parameter sets are
integrated at once (ensemble MCMC, Sobol designs).

Governing equations (00_REVISAO_EQUACOES, sections 3-8 and 10):

  rho c dT/dt = d/dz( rho c K dT/dz ) + S_rad(z) + S_wall(z)            (heat)
  dH/dt       = - E / rho_w                                             (mass)
  C_p dT_p/dt = a_p A_tot I - eps sigma (T_p^4 - T_a^4) - rho c_p C_hp (T_p - T_a)
                - G_pw (T_p - T_s)                                      (plate)

  open water (fraction 1 - phi):
     -k dT/dz|0 = - eps sigma (T_s^4 - T_a^4) - rho c_p C_h (T_s - T_a) - lambda E
  covered water (fraction phi): conductive exchange with the plate only.

  E      = (1 - phi) * dRho_v / (1/(psi C_e) + R_rec / D)   (gap evaporation)
  C_e    = (C_n^3 + C_f^3)^(1/3) + C_0                       (mixed convection)
  C_n    = 0.14 c_n D (g dTv / (Tv nu D))^(1/3)              (free convection)
  C_f    = c_f 1e-3 U^n_f                                    (forced convection)
  psi    = max(Schlunder (1988) per-open-area factor, 1)
  S_rad  = I [A_m e^(-z/l_m)/l_m + A_d e^(-z/l_d)/l_d] (+ A_s I at the skin;
           a_b of what reaches the floor absorbed there, the rest reflected up
           through the deep band and partly lost through the surface)
  S_wall = (2/R) [U_w (T_a - T) + G_w I]  (lateral exchange through the wall)

The irradiance I is the silicon-pyranometer index; A_s + A_m + A_d = A_tot
converts it to the energy actually delivered to a horizontal surface (true
short wave incl. kappa_R, plus the long wave of the lamp envelopes). The plate
absorbs a fraction a_p of the same A_tot I.

Time integration (step dt, default 300 s): backward Euler; the surface energy
balance is linearised in T_s and the plate is coupled to the skin layer
implicitly (2x2 elimination, energy-conserving); the eddy diffusivity, its
stratification damping and the convective-adjustment flag are evaluated with
one Picard iteration on the end-of-step profile, which removes the first-order
time-step error of a diffusivity frozen at the start of the step.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_comum"))
from tanque import propriedades as pr          # noqa: E402
from tanque import config as cfg                 # noqa: E402

KAPPA_W = 1.45e-7        # molecular thermal diffusivity of water (m2 s-1)
K_CONV = 2.0e-3          # convective-adjustment diffusivity (m2 s-1)
EPS_W = 0.97             # emissivity of water
EPS_P = 0.92             # emissivity of the printed polymer plates
R_PAN = cfg.PAN_RADIUS_M
RHO_W = 997.0
PICARD = 1               # Picard iterations for the stability-dependent diffusivity

# ---------------------------------------------------------------------------
# Parameters: name -> (default, lower, upper, description)
# ---------------------------------------------------------------------------
PARAMS = {
    # radiation, per unit of the irradiance index (dimensionless)
    "A_s": (0.70, 0.0, 3.0, "skin absorption per unit index (IR > 1.4 um + envelope long wave)"),
    "A_m": (0.70, 0.0, 3.0, "near-surface band per unit index"),
    "l_m": (0.005, 5e-4, 0.05, "e-folding depth of the near-surface band (m)"),
    "A_d": (0.50, 0.0, 3.0, "deep-penetrating band per unit index"),
    "l_d": (0.20, 0.03, 3.0, "e-folding depth of the deep band (m)"),
    "a_b": (0.50, 0.0, 1.0, "absorptance of the pan floor"),
    # walls and floor
    "U_w": (8.0, 0.0, 60.0, "wall heat-transfer coefficient (W m-2 K-1)"),
    "G_w": (0.05, 0.0, 1.0, "wall heating by the lamps per unit index"),
    "U_b": (5.0, 0.0, 60.0, "floor heat-transfer coefficient (W m-2 K-1)"),
    # air side
    "c_n": (1.0, 0.1, 5.0, "multiplier of the free-convection correlation"),
    "c_f": (3.5, 0.0, 30.0, "forced-convection coefficient (mm s-1 at 1 m s-1)"),
    "n_f": (0.6, 0.2, 1.5, "forced-convection velocity exponent"),
    "C_0": (0.5, 0.0, 5.0, "background (room-draught) transfer velocity (mm s-1)"),
    # water side
    "K_bg": (1e-5, 1e-8, 1e-3, "eddy diffusivity of the unstratified column (m2 s-1)"),
    "N2_0": (1e-3, 1e-7, 1.0, "buoyancy-frequency scale of stratification damping (s-2)"),
    "k_w": (1e-4, 0.0, 5e-3, "wind-mixing coefficient (m)"),
    "d_w": (0.03, 0.001, 0.2, "e-folding depth of wind mixing (m)"),
    # cover
    "a_p": (0.45, 0.0, 1.0, "absorptance of the plates for the lamp radiation"),
    "G_pw": (4.0, 0.05, 100.0, "plate-water conductance (W m-2 K-1)"),
    "C_p": (6000.0, 500.0, 5e4, "areal heat capacity of the plates (J m-2 K-1)"),
    "r_open": (0.010, 0.0005, 0.2, "effective size of the openings in Schlunder's relation (m)"),
    "R_rec": (0.004, 0.0, 0.1, "recess diffusion length d_rec/zeta of the gap water (m)"),
    "tau_p": (0.0, 0.0, 1.0, "transmittance of the plates for the near-surface and deep bands"),
    "C_u": (0.0, 0.0, 0.02, "transfer velocity of the vented air pockets under the plates (m s-1)"),
    "k_g": (1.0, 0.5, 10.0, "enhancement of forced transfer over the gaps (boundary-layer restart)"),
}
NAMES = list(PARAMS)


def default_params(W=1, **over):
    p = {k: np.full(W, v[0], float) for k, v in PARAMS.items()}
    for k, v in over.items():
        p[k] = np.broadcast_to(np.asarray(v, float), (W,)).copy()
    return p


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def sigma_grid(N=24, dz0_frac=0.008):
    """Layer thickness fractions (sum 1), geometric stretching from the top."""
    from scipy.optimize import brentq
    if N == 1:
        return np.array([1.0])
    f = lambda r: dz0_frac * (r ** N - 1) / (r - 1) - 1.0
    r = brentq(f, 1.0001, 3.0)
    s = dz0_frac * r ** np.arange(N)
    return s / s.sum()


# ---------------------------------------------------------------------------
# Air-side transfer
# ---------------------------------------------------------------------------
def transfer_velocities(Ts, Ta, ea, U, p, es_surf=None, Tv_surf=None, k_forced=1.0):
    """Mass (C_e) and heat (C_h) transfer velocities (m s-1).

    Free convection uses the virtual-temperature difference between the
    surface and the air (buoyancy from heat and vapour); forced convection a
    power law in U; the two are blended as a Churchill-Usagi cube sum, and a
    background term is added. es_surf: vapour pressure at the surface (default
    saturation at Ts; pass ea for a dry surface such as a plate). Tv_surf:
    override of the surface virtual temperature (area-weighted over a
    partially covered pan). k_forced multiplies the forced term (gaps between
    dry plates, where the vapour and thermal boundary layers restart).
    """
    es = pr.esat(Ts) if es_surf is None else es_surf
    Tf = 0.5 * (Ts + Ta)
    ap = pr.air_props(Tf, 0.5 * (es + ea))
    Tvs = pr.virtual_temperature(Ts, es) if Tv_surf is None else Tv_surf
    Tva = pr.virtual_temperature(Ta, ea)
    dTv = np.maximum(Tvs - Tva, 0.0)
    gfac = pr.G * dTv / (0.5 * (Tvs + Tva))
    Cn_m = 0.14 * p["c_n"] * ap["D"] * np.cbrt(gfac / (ap["nu"] * ap["D"]))
    Cn_h = 0.14 * p["c_n"] * ap["alpha"] * np.cbrt(gfac / (ap["nu"] * ap["alpha"]))
    Cf_m = k_forced * p["c_f"] * 1e-3 * np.power(np.maximum(U, 0.0), p["n_f"])
    lew = (ap["Sc"] / ap["Pr"]) ** (2.0 / 3.0)
    Cf_h = Cf_m * lew
    C0 = p["C_0"] * 1e-3
    Ce = np.cbrt(Cn_m ** 3 + Cf_m ** 3) + C0
    Ch = np.cbrt(Cn_h ** 3 + Cf_h ** 3) + C0 * lew
    return Ce, Ch, ap


def schlunder_psi(theta, r_open, Ce, D):
    """Evaporation per unit open area relative to the open surface (>= 1).

    Schlunder (1988): for wet patches of size r covering a fraction theta of an
    otherwise inert surface under a diffusive layer delta = D/C_e,
    E/E0 = 1 / (1 + (2 r / (pi delta)) f(theta)), f = g (g - 1), g =
    sqrt(pi/(4 theta)); per unit open area the factor is E/(theta E0). The
    relation assumes delta larger than the patch spacing; when it is not,
    Thomson's principle bounds the per-open-area flux from below by that of an
    isolated column (factor 1), which is enforced here. f is clipped at zero
    beyond theta = pi/4, outside the dilute-patch range.
    """
    theta = np.clip(theta, 1e-6, 1.0)
    delta = D / np.maximum(Ce, 1e-6)
    g = np.sqrt(np.pi / (4 * theta))
    f = np.maximum(g * (g - 1.0), 0.0)
    ratio = 1.0 / (1.0 + 2 * r_open / (np.pi * delta) * f)
    return np.maximum(ratio / theta, 1.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class Forcing:
    """Forcing on the model time grid (length n). Arrays of shape (n,)."""
    t: np.ndarray            # seconds since start
    I: np.ndarray            # irradiance index (W m-2)
    U: np.ndarray            # air speed (m s-1)
    Ta: np.ndarray           # air temperature (degC)
    ea: np.ndarray           # air vapour pressure (Pa)
    phi: np.ndarray          # covered fraction (0-1)
    cfg: np.ndarray = None   # lamp configuration code per step (multiplier m_<code>)
    jumps: dict = field(default_factory=dict)   # step index -> added depth (m)


@dataclass
class Output:
    T: np.ndarray            # (n, W, N) layer temperatures at the END of each step
    H: np.ndarray            # (n, W) water depth (m, mass-equivalent at RHO_W)
    E: np.ndarray            # (n, W) evaporation of the pan (kg m-2 s-1, per pan area)
    Ts: np.ndarray           # (n, W) skin temperature
    Tp: np.ndarray           # (n, W) plate temperature
    fluxes: dict             # time series of energy terms (n, W), W m-2 of pan area
    zc_frac: np.ndarray      # (N,) layer-centre depth fractions
    I_eff: np.ndarray = None  # (n, W) irradiance index after the configuration multiplier
    T0: np.ndarray = None     # (W, N) initial profile
    H0: np.ndarray = None     # (W,) initial depth


def _diffusivity(T, dzi, zint, U, phi, sw_v, p, W):
    rho_l = pr.water_density(T)
    N2 = pr.G * (rho_l[:, 1:] - rho_l[:, :-1]) / (0.5 * (rho_l[:, 1:] + rho_l[:, :-1])) / dzi
    damp = 1.0 / (1.0 + np.maximum(N2, 0.0) / p.get("N2_0", np.full(W, np.inf))[:, None])
    K = KAPPA_W + damp * (p["K_bg"][:, None] + (p["k_w"] * U * (1 - phi * sw_v))[:, None] *
                          np.exp(-zint / p["d_w"][:, None]))
    return np.where(N2 < -1e-7, K + K_CONV, K)


def simulate(forcing: Forcing, p: dict, T0, H0, N=24, dt=300.0, store_fluxes=False,
             Tp0=None):
    """Integrate the model. T0: (W, N) or (N,) initial profile; H0: (W,) or scalar."""
    W = len(next(iter(p.values())))
    n = len(forcing.t)
    s = sigma_grid(N)
    edges = np.concatenate([[0.0], np.cumsum(s)])       # depth fractions of interfaces
    zc = 0.5 * (edges[:-1] + edges[1:])
    T = np.broadcast_to(np.asarray(T0, float), (W, N)).copy()
    H = np.broadcast_to(np.asarray(H0, float), (W,)).copy()
    T_init, H_init = T.copy(), H.copy()
    Tp = T[:, 0].copy() if Tp0 is None else np.broadcast_to(np.asarray(Tp0, float), (W,)).copy()

    outT = np.empty((n, W, N)); outH = np.empty((n, W)); outE = np.empty((n, W))
    outTs = np.empty((n, W)); outTp = np.empty((n, W)); outI = np.empty((n, W))
    mult = np.column_stack([np.ones(W)] + [p.get(f"m_{j}", np.ones(W)) for j in range(1, 8)])
    sU = p.get("s_U", np.ones(W))
    cfgs = np.zeros(n, dtype=int) if forcing.cfg is None else np.asarray(forcing.cfg, int)
    keys = ["Rabs_water", "Rabs_plate", "LW_net", "H_sens", "LE", "wall", "floor",
            "G_plate", "escape_up", "storage"]
    fl = {k: np.empty((n, W)) for k in keys} if store_fluxes else {}

    sig = pr.SIGMA
    one = np.ones(W)
    A_tot = p["A_s"] + p["A_m"] + p["A_d"]
    # mechanism switches of the cover (1 = active); all four off reproduces the
    # uncovered pan. vapour: evaporation only through the gaps and sheltering of
    # the water from wind stress (aerodynamic blocking); shade: plates
    # intercept the lamp radiation; insul: no long-wave/sensible exchange of the
    # water under the plates; cond: plate-water conduction.
    sw_v = p.get("sw_vapour", one); sw_s = p.get("sw_shade", one)
    sw_i = p.get("sw_insul", one); sw_c = p.get("sw_cond", one)
    for k in range(n):
        I = forcing.I[k] * mult[:, cfgs[k]]          # (W,)
        U = forcing.U[k] * sU                         # (W,)
        Ta = forcing.Ta[k]; ea = forcing.ea[k]
        if "phi_w" in p:                              # per-walker covered fraction
            phi = np.where(forcing.phi[k] > 0, p["phi_w"], 0.0)
        else:
            phi = forcing.phi[k] * one
        cov = phi > 0
        covered = bool(np.any(cov))
        if k in forcing.jumps:                        # refill
            H = H + forcing.jumps[k]
        dz = H[:, None] * s[None, :]                  # (W, N)
        zi = H[:, None] * edges[None, :]              # interface depths (W, N+1)
        rc = RHO_W * pr.water_cp(T)                   # (W, N) volumetric heat capacity
        cap = rc * dz                                 # J m-2 K-1 per layer
        open_f = 1.0 - phi

        # ---------------- radiation absorbed in the column (W m-2 of pan)
        # the plates absorb the skin band (infrared beyond 1.4 um and envelope long
        # wave) and transmit a fraction tau_p of the visible/near-infrared bands
        Iw_s = I * (1.0 - phi * sw_s)
        Iw = I * (1.0 - phi * sw_s * (1.0 - p.get("tau_p", 0.0 * one)))
        em = np.exp(-zi / p["l_m"][:, None]); ed = np.exp(-zi / p["l_d"][:, None])
        Sm = Iw[:, None] * p["A_m"][:, None] * (em[:, :-1] - em[:, 1:])
        Sd = Iw[:, None] * p["A_d"][:, None] * (ed[:, :-1] - ed[:, 1:])
        Ibot = Iw * (p["A_m"] * em[:, -1] + p["A_d"] * ed[:, -1])
        src = Sm + Sd
        src[:, -1] += p["a_b"] * Ibot
        # reflected by the floor: travels up through the deep band, the rest escapes
        Rf = (1 - p["a_b"]) * Ibot
        hb = H[:, None] - zi                           # height of interfaces above the floor
        eu = np.exp(-hb / p["l_d"][:, None])
        src += Rf[:, None] * (eu[:, 1:] - eu[:, :-1])
        escape = Rf * eu[:, 0]
        src[:, 0] += Iw_s * p["A_s"]
        # ---------------- walls and floor (explicit in T)
        wall = (2.0 / R_PAN) * dz * (p["U_w"][:, None] * (Ta - T) + (p["G_w"] * I)[:, None])
        src += wall
        floor = p["U_b"] * (Ta - T[:, -1])
        src[:, -1] += floor

        # ---------------- surface fluxes (linearised in Ts)
        Ts = T[:, 0]
        es = pr.esat(Ts)
        if covered:
            # buoyancy over the pan from the area-weighted surface virtual temperature
            Tv_s = pr.virtual_temperature(Ts, es)
            Tv_p = pr.virtual_temperature(Tp, ea)
            # (the plate surface replaces the water surface for the air only when
            # the insulation mechanism is active)
            Tv_surf = np.where(cov, (1 - phi * sw_i) * Tv_s + phi * sw_i * Tv_p, Tv_s)
            Ce, Ch, ap = transfer_velocities(Ts, Ta, ea, U, p, Tv_surf=Tv_surf)
            _, Chp, _ = transfer_velocities(Tp, Ta, ea, U, p, es_surf=ea * one)
        else:
            Ce, Ch, ap = transfer_velocities(Ts, Ta, ea, U, p)
            Chp = Ch
        rv_s = es / (pr.R_V * (Ts + pr.K0)); rv_a = ea / (pr.R_V * (Ta + pr.K0))
        drv = rv_s - rv_a
        if covered:
            # the air crosses dry plates and wet gaps alternately: the vapour and
            # thermal boundary layers restart at each gap (unheated starting
            # length), which raises the forced transfer of the gaps by k_g
            Ce_g, Ch_g, _ = transfer_velocities(Ts, Ta, ea, U, p, Tv_surf=Tv_surf,
                                                k_forced=p.get("k_g", one))
            psi = schlunder_psi(open_f, p["r_open"], Ce, ap["D"])
            Cgap = 1.0 / (1.0 / (psi * Ce_g) + p["R_rec"] / ap["D"])
            # under-plate evaporation: the finned underside of the plates traps air
            # pockets over the water that vent at the plate edges (transfer velocity C_u)
            Cu = p.get("C_u", 0.0 * one)
            Ce_eff = np.where(cov, sw_v * (open_f * Cgap + phi * Cu) + (1 - sw_v) * Ce, Ce)
            # the same three-dimensional diffusion field carries the heat (delta_h = alpha/C_h)
            psi_h = schlunder_psi(open_f, p["r_open"], Ch, ap["alpha"])
            Ch_gap = 1.0 / (1.0 / (psi_h * Ch_g) + p["R_rec"] / ap["alpha"])
            Ch_eff = np.where(cov, sw_i * open_f * Ch_gap + (1 - sw_i) * Ch, Ch)
            lw_f = np.where(cov, sw_i * open_f + (1 - sw_i), 1.0)
        else:
            Ce_eff = Ce; Ch_eff = Ch; lw_f = one
        lam = pr.latent_heat(Ts)
        E = Ce_eff * drv                                          # kg m-2 s-1 (pan area)
        drv_dT = pr.desat_dt(Ts) / (pr.R_V * (Ts + pr.K0)) - es / (pr.R_V * (Ts + pr.K0) ** 2)
        LE = lam * E
        dLE = lam * Ce_eff * drv_dT
        rho_a = ap["rho"]
        Hs = rho_a * pr.CP_AIR * Ch_eff * (Ts - Ta)
        dHs = rho_a * pr.CP_AIR * Ch_eff
        LWo = lw_f * EPS_W * sig * ((Ts + pr.K0) ** 4 - (Ta + pr.K0) ** 4)
        dLW = lw_f * 4 * EPS_W * sig * (Ts + pr.K0) ** 3
        Qs = -(LE + Hs + LWo)                                     # net non-radiative gain
        dQ = dLE + dHs + dLW
        # plate, eliminated implicitly against the skin temperature:
        #   A Tp' = B + Gp Ts' ;  flux to the skin phi Gp (Tp' - Ts')
        if covered:
            # absorbed per plate area: a_p of the radiation the plate does not transmit
            Ip = p["a_p"] * (A_tot - p.get("tau_p", 0.0 * one) * (p["A_m"] + p["A_d"])) * I * sw_s
            hr = 4 * EPS_P * sig * (Tp + pr.K0) ** 3
            hc = rho_a * pr.CP_AIR * Chp
            LWp = EPS_P * sig * ((Tp + pr.K0) ** 4 - (Ta + pr.K0) ** 4)
            Gp = p["G_pw"] * sw_c * cov
            Ac = p["C_p"] / dt + hr + hc + Gp
            Bc = p["C_p"] * Tp / dt + Ip - LWp + hr * Tp + hc * Ta
            dQp = phi * Gp * (1.0 - Gp / Ac)                    # exact, not a linearisation
            Qp_const = phi * Gp * Bc / Ac
        else:
            Ip = np.zeros(W); Gp = np.zeros(W); Ac = np.ones(W); Bc = np.zeros(W)
            Qp_const = np.zeros(W); dQp = np.zeros(W)

        # ---------------- implicit solve with a Picard pass on the diffusivity
        zint = zi[:, 1:-1]
        dzi = 0.5 * (dz[:, :-1] + dz[:, 1:])
        Tk = T
        for it in range(PICARD + 1):
            K = _diffusivity(Tk, dzi, zint, U, phi, sw_v, p, W)
            cond = 0.5 * (rc[:, :-1] + rc[:, 1:]) * K / dzi       # W m-2 K-1 between layers
            a = np.zeros((W, N)); c = np.zeros((W, N))
            a[:, 1:] = -cond; c[:, :-1] = -cond
            b = cap / dt
            b = b.copy()
            b[:, 1:] += cond; b[:, :-1] += cond
            d = cap / dt * T + src
            b[:, 0] += dQ + dQp
            d[:, 0] += Qs + dQ * Ts + Qp_const
            Tk = _thomas(a, b, c, d)
        Tn = Tk
        dS = np.sum(cap * (Tn - T), axis=1) / dt
        Tsn = Tn[:, 0]
        if covered:
            Tp = np.where(cov, (Bc + Gp * Tsn) / Ac, Tsn)
            Gplate = phi * Gp * (Tp - Tsn)
        else:
            Tp = Tsn.copy(); Gplate = np.zeros(W)
        T = Tn
        Enew = E + (Ce_eff * drv_dT) * (Tsn - Ts)                  # consistent with implicit Ts
        H = H - np.maximum(Enew, -1e-6) * dt / RHO_W
        outT[k] = T; outH[k] = H; outE[k] = Enew; outTs[k] = Tsn; outTp[k] = Tp
        outI[k] = I
        if store_fluxes:
            fl["Rabs_water"][k] = np.sum(Sm + Sd, axis=1) + Iw_s * p["A_s"] + p["a_b"] * Ibot + Rf - escape
            fl["Rabs_plate"][k] = phi * Ip
            fl["LW_net"][k] = -(LWo + dLW * (Tsn - Ts))
            fl["H_sens"][k] = -(Hs + dHs * (Tsn - Ts))
            fl["LE"][k] = -(LE + dLE * (Tsn - Ts))
            fl["wall"][k] = np.sum(wall, axis=1)
            fl["floor"][k] = floor
            fl["G_plate"][k] = Gplate
            fl["escape_up"][k] = escape
            fl["storage"][k] = dS
    return Output(outT, outH, outE, outTs, outTp, fl, zc, outI, T_init, H_init)


def _thomas(a, b, c, d):
    """Vectorised Thomas algorithm along axis 1 (W, N)."""
    W, N = d.shape
    cp = np.empty((W, N)); dp = np.empty((W, N))
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, N):
        m = b[:, i] - a[:, i] * cp[:, i - 1]
        cp[:, i] = c[:, i] / m
        dp[:, i] = (d[:, i] - a[:, i] * dp[:, i - 1]) / m
    x = np.empty((W, N))
    x[:, -1] = dp[:, -1]
    for i in range(N - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def probe_temperature(T, H, zc_frac, h_above_bottom):
    """Vectorised temperature at a height above the floor.

    T: (n, W, N); H: (n, W); h_above_bottom: scalar or (W,). Returns (n, W).
    A point above the modelled surface takes the skin temperature (clipped), so
    that every walker sees the same data set; the calibration masks such
    probes from the observed depth instead.
    """
    h = np.asarray(h_above_bottom, float)
    frac = np.clip((H - h) / H, 0.0, 1.0)
    zc = np.asarray(zc_frac)
    N = len(zc)
    idx = np.clip(np.searchsorted(zc, frac) - 1, 0, N - 2)
    f = np.clip((frac - zc[idx]) / (zc[idx + 1] - zc[idx]), 0.0, 1.0)
    T0 = np.take_along_axis(T, idx[..., None], axis=2)[..., 0]
    T1 = np.take_along_axis(T, (idx + 1)[..., None], axis=2)[..., 0]
    return T0 * (1 - f) + T1 * f


def bin_average(out: Output, i0, i1, field):
    """Trapezoidal mean of a state over the 10-min bin [t, t+10 min] that spans
    model steps i0 (t..t+5) and i1 (t+5..t+10): 0.25 S(t) + 0.5 S(t+5) +
    0.25 S(t+10), where S(t) is the state at the end of step i0-1 (or the
    initial state)."""
    arr = getattr(out, field)
    if field == "T":
        init = out.T0[None]
    elif field == "H":
        init = out.H0[None]
    else:
        init = arr[:1]
    prev = np.where((i0 > 0)[(...,) + (None,) * (arr.ndim - 1)],
                    arr[np.maximum(i0 - 1, 0)], np.broadcast_to(init, arr[i0].shape))
    return 0.25 * prev + 0.5 * arr[i0] + 0.25 * arr[i1]


def temperature_at_height(out: Output, h_above_bottom):
    """Model temperature at a fixed height above the floor (n, W); NaN when the
    point is above the water surface."""
    Tz = probe_temperature(out.T, out.H, out.zc_frac, h_above_bottom)
    depth = out.H - h_above_bottom
    return np.where(depth < 0, np.nan, Tz), depth
