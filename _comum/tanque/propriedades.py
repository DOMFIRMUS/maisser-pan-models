# -*- coding: utf-8 -*-
"""
Thermophysical properties of water, moist air and water vapour.

Every function accepts scalars or numpy arrays and works in SI units unless the
name says otherwise (temperatures in degC where the argument is called `t`,
in K where it is called `T`). Formulas and their sources are listed in
00_REVISAO_EQUACOES (section 2) and cited in the manuscripts.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
import numpy as np

# Physical constants -------------------------------------------------------
SIGMA = 5.670374419e-8      # Stefan-Boltzmann (W m-2 K-4)
G = 9.80665                 # gravity (m s-2)
R_D = 287.05                # gas constant, dry air (J kg-1 K-1)
R_V = 461.5                 # gas constant, water vapour (J kg-1 K-1)
EPS = R_D / R_V             # 0.622
CP_AIR = 1005.0             # specific heat of dry air (J kg-1 K-1)
CP_V = 1860.0               # specific heat of water vapour (J kg-1 K-1)
P0 = 101325.0               # reference pressure (Pa)
K0 = 273.15
KARMAN = 0.41


# Saturation vapour pressure ----------------------------------------------
def esat(t):
    """Saturation vapour pressure over liquid water (Pa).

    Alduchov & Eskridge (1996) Magnus form, a = 610.94 Pa, b = 17.625,
    c = 243.04 degC; error < 0.4 % from -40 to 50 degC against Wagner & Pruss.
    """
    t = np.asarray(t, float)
    return 610.94 * np.exp(17.625 * t / (t + 243.04))


def desat_dt(t):
    """Slope of the saturation curve, d esat / dT (Pa K-1)."""
    t = np.asarray(t, float)
    return esat(t) * 17.625 * 243.04 / (t + 243.04) ** 2


def esat_fao(t):
    """FAO-56 Tetens form (kPa), kept for comparison with Allen et al. (1998)."""
    t = np.asarray(t, float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


# Latent heat and psychrometric constant -----------------------------------
def latent_heat(t):
    """Latent heat of vaporisation (J kg-1), Henderson-Sellers (1984)."""
    T = np.asarray(t, float) + K0
    return 1.91846e6 * (T / (T - 33.91)) ** 2


def psychrometric(t, p=P0):
    """Psychrometric constant gamma = cp p / (eps lambda) (Pa K-1)."""
    return CP_AIR * p / (EPS * latent_heat(t))


# Moist air ------------------------------------------------------------------
def specific_humidity(e, p=P0):
    """Specific humidity (kg kg-1) from vapour pressure e (Pa)."""
    e = np.asarray(e, float)
    return EPS * e / (p - (1 - EPS) * e)


def vapour_density(e, t):
    """Absolute humidity (kg m-3) from vapour pressure (Pa) and temperature."""
    return np.asarray(e, float) / (R_V * (np.asarray(t, float) + K0))


def virtual_temperature(t, e, p=P0):
    """Virtual temperature (K)."""
    q = specific_humidity(e, p)
    return (np.asarray(t, float) + K0) * (1 + 0.608 * q)


def air_density(t, e=0.0, p=P0):
    """Moist-air density (kg m-3)."""
    return (p - (1 - EPS) * np.asarray(e, float)) / (R_D * (np.asarray(t, float) + K0))


def air_viscosity(t):
    """Dynamic viscosity of air (Pa s), Sutherland."""
    T = np.asarray(t, float) + K0
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


def air_conductivity(t):
    """Thermal conductivity of air (W m-1 K-1), Sutherland-type fit."""
    T = np.asarray(t, float) + K0
    return 0.0241 * (T / 273.15) ** 1.5 * (273.15 + 194.0) / (T + 194.0)


def vapour_diffusivity(t, p=P0):
    """Binary diffusion coefficient of water vapour in air (m2 s-1).

    D = 2.178e-5 (T/273.15)^1.81 (p0/p)  (Massman 1998 recommendation).
    """
    T = np.asarray(t, float) + K0
    return 2.178e-5 * (T / 273.15) ** 1.81 * (P0 / p)


def air_props(t, e=0.0, p=P0):
    """Dictionary of air properties at film temperature t (degC)."""
    rho = air_density(t, e, p)
    mu = air_viscosity(t)
    k = air_conductivity(t)
    nu = mu / rho
    alpha = k / (rho * CP_AIR)
    D = vapour_diffusivity(t, p)
    return dict(rho=rho, mu=mu, k=k, nu=nu, alpha=alpha, D=D,
                Pr=nu / alpha, Sc=nu / D)


# Liquid water -------------------------------------------------------------
def water_density(t):
    """Density of air-free water (kg m-3), Tanaka et al. (2001), 0-40 degC."""
    t = np.asarray(t, float)
    a1, a2, a3, a4, a5 = -3.983035, 301.797, 522528.9, 69.34881, 999.974950
    return a5 * (1 - (t + a1) ** 2 * (t + a2) / (a3 * (t + a4)))


def water_cp(t):
    """Specific heat of liquid water (J kg-1 K-1), polynomial fit 10-60 degC."""
    t = np.asarray(t, float)
    return 4217.4 - 3.720283 * t + 0.1412855 * t ** 2 - 2.654387e-3 * t ** 3 \
        + 2.093236e-5 * t ** 4


def water_conductivity(t):
    """Thermal conductivity of liquid water (W m-1 K-1), 0-80 degC fit."""
    t = np.asarray(t, float)
    return 0.5650 + 1.9e-3 * t - 7.6e-6 * t ** 2


def water_expansion(t):
    """Volumetric thermal expansion coefficient beta (K-1)."""
    t = np.asarray(t, float)
    h = 0.01
    return -(water_density(t + h) - water_density(t - h)) / (2 * h) / water_density(t)


def water_kinematic_viscosity(t):
    """Kinematic viscosity of water (m2 s-1), Vogel equation / density."""
    T = np.asarray(t, float) + K0
    mu = 1e-3 * np.exp(-3.7188 + 578.919 / (-137.546 + T))
    return mu / water_density(t)


def water_diffusivity(t):
    """Thermal diffusivity of water (m2 s-1)."""
    return water_conductivity(t) / (water_density(t) * water_cp(t))


# Convenience ---------------------------------------------------------------
def vpd_and_de(ta, rh, tw, p=P0):
    """Air vapour pressure ea, air deficit VPD and surface-air difference dE (Pa)."""
    ea = esat(ta) * np.asarray(rh, float) / 100.0
    return ea, esat(ta) - ea, esat(tw) - ea


def mm_per_hour_from_flux(E_kg_m2_s, t=25.0):
    """Evaporation flux (kg m-2 s-1) to equivalent depth rate (mm h-1)."""
    return np.asarray(E_kg_m2_s, float) / water_density(t) * 1e3 * 3600.0


def flux_from_mm_per_hour(E_mm_h, t=25.0):
    return np.asarray(E_mm_h, float) * water_density(t) / 1e3 / 3600.0
