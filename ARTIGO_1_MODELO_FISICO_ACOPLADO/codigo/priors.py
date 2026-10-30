# -*- coding: utf-8 -*-
"""
Prior distributions of the physical parameters (uncovered pan, campaign C2)
and of the cover parameters (campaign C1).

Each prior is defined on an unconstrained scale u (log for positive
parameters, logit for fractions) as a normal N(mu, sd); the median and the
spread follow from physics, the spectral calculation (espectro.py) or
textbook heat-transfer correlations - see the manuscript, Table 2.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
import numpy as np

# name: (transform, median, sd on the transformed scale, source / reasoning)
PRIORS_C2 = {
    "A_s": ("log", 0.80, 0.50, "spectral calc. (2300-2900 K): top-layer share 0.35-0.49 x kappa_R 1.3-2.1, + envelope long wave"),
    "A_m": ("log", 0.48, 0.50, "spectral calc.: 0.27-0.28 x kappa_R"),
    "l_m": ("log", 0.014, 0.60, "spectral calc.: 13-15 mm"),
    "A_d": ("log", 0.43, 0.50, "spectral calc.: 0.21-0.35 x kappa_R"),
    "l_d": ("log", 0.28, 0.50, "spectral calc.: 0.25-0.36 m (clear water)"),
    "a_b": ("logit", 0.50, 1.00, "absorptance of a steel floor, uninformative"),
    "U_w": ("log", 8.0, 0.60, "outer natural convection + radiation of a thin steel wall"),
    "G_w": ("log", 0.05, 1.00, "lamp heating of rim and wall, small"),
    "U_b": ("log", 5.0, 0.60, "floor over a ventilated frame"),
    "c_n": ("log", 1.0, 0.40, "Sh = 0.14 Ra^(1/3); Bower and Saylor (2009) within x1.5"),
    "c_f": ("log", 3.5, 0.70, "laminar flat plate at the pan scale (Re < 5e5): 3.4 mm s-1 at 1 m s-1"),
    "n_f": ("lin", 0.60, 0.15, "laminar 0.5; fan turbulence and the rim step push it up"),
    "C_0": ("log", 0.50, 0.80, "room draughts, mm s-1"),
    "K_bg": ("log", 1e-5, 1.50, "convective/wall-driven mixing of the unstratified column"),
    "N2_0": ("log", 1e-3, 1.50, "stratification damping scale; N2 ~ 1e-2 to 0.3 s-2 near the surface"),
    "k_w": ("log", 1e-4, 1.50, "wind-driven mixing, uninformative"),
    "d_w": ("log", 0.03, 0.70, "depth of wind mixing, ~ few cm"),
    "m_1": ("log", 1.0, 0.40, "upper-rail lamps: pan-average / index ratio relative to all lamps"),
    "m_2": ("log", 1.0, 0.40, "lower-rail lamps: pan-average / index ratio relative to all lamps"),
}

PRIORS_COVER = {
    "a_p": ("logit", 0.45, 0.80, "white printed PLA/PP under a 2500 K spectrum"),
    "G_pw": ("log", 4.0, 0.80, "31 mm printed shell, 5 % infill: mostly still air"),
    "C_p": ("log", 6000.0, 0.50, "areal heat capacity of 106-176 g plates (+ entrained water)"),
    "r_open": ("log", 0.010, 1.00, "effective opening size in Schlunder's relation (slots, perimeter band)"),
    "R_rec": ("log", 0.004, 0.90, "recess diffusion length d_rec/zeta: freeboard 6-17 mm, ventilation 1-10"),
    "s_I": ("log", 1.0, 0.30, "C1/C2 ratio of the irradiance index (sensor position)"),
}


def _fwd(x, tr):
    if tr == "log":
        return np.log(x)
    if tr == "logit":
        return np.log(x / (1 - x))
    return x


def _inv(u, tr):
    if tr == "log":
        return np.exp(u)
    if tr == "logit":
        return 1 / (1 + np.exp(-u))
    return u


class Prior:
    def __init__(self, spec):
        self.spec = spec
        self.names = list(spec)
        self.tr = [spec[k][0] for k in self.names]
        self.mu = np.array([_fwd(spec[k][1], spec[k][0]) for k in self.names])
        self.sd = np.array([spec[k][2] for k in self.names])

    def to_params(self, U):
        """U: (W, P) unconstrained -> dict of (W,) arrays."""
        U = np.atleast_2d(U)
        return {k: _inv(U[:, i], t) for i, (k, t) in enumerate(zip(self.names, self.tr))}

    def to_u(self, p):
        return np.column_stack([_fwd(np.asarray(p[k], float), t) for k, t in zip(self.names, self.tr)])

    def logpdf(self, U):
        U = np.atleast_2d(U)
        z = (U - self.mu) / self.sd
        return -0.5 * np.sum(z ** 2, axis=1)

    def sample(self, n, rng):
        return self.mu + self.sd * rng.standard_normal((n, len(self.names)))

    def bounds(self, k=3.5):
        return list(zip(self.mu - k * self.sd, self.mu + k * self.sd))
