# -*- coding: utf-8 -*-
"""
Spectral radiation of the halogen infrared lamps and its absorption in water.

Used only to build physically based PRIORS for the radiative parameters of the
pan model:
  * kappa_R  ratio of the irradiance actually incident on the water to the
             reading of a silicon pyranometer calibrated to sunlight;
  * the fractions of the incident energy absorbed (i) in the first millimetre
    (skin), (ii) within a few centimetres and (iii) deeper than the water
    column (reaching the pan bottom), and their e-folding depths.

Ingredients (all approximate; the calibration estimates the actual values):
  - Planck emission of a tungsten filament with a wavelength-dependent
    emissivity (linear fit to the spectral emissivity of tungsten at
    2500-2800 K, de Vos 1954);
  - transmission of the hard-glass envelope (opaque beyond ~2.8 um: that energy
    heats the bulb and is re-emitted as long-wave radiation, which the water
    absorbs at the surface);
  - absorption coefficient of pure water, a(lambda) (m-1), compiled from Pope &
    Fry (1997) in the visible and Hale & Querry (1973) / Kou et al. (1993) in the
    near infrared, at the resolution needed for band integrals;
  - relative spectral response of a silicon-photodiode pyranometer (typical
    LI200 / SP-LITE curve, 0.4-1.1 um);
  - a reference solar spectrum (5778 K blackbody with the main water-vapour and
    oxygen bands removed), only to reproduce the sunlight calibration of the
    silicon sensors.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - MAISSER-PE
License: MIT
"""
import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23

# wavelength grid (um)
LAM = np.linspace(0.30, 5.00, 2351)

# absorption coefficient of pure water a (m-1) at selected wavelengths (um)
_AW = np.array([
    [0.30, 0.046], [0.35, 0.0115], [0.40, 0.0066], [0.45, 0.0092], [0.50, 0.0204],
    [0.55, 0.0565], [0.60, 0.2224], [0.65, 0.340], [0.70, 0.624], [0.725, 1.49],
    [0.74, 2.60], [0.75, 2.47], [0.77, 2.38], [0.80, 1.96], [0.82, 2.70],
    [0.85, 4.30], [0.875, 5.50], [0.90, 6.80], [0.92, 14.0], [0.94, 27.0],
    [0.96, 41.0], [0.97, 45.0], [0.98, 44.0], [1.00, 36.0], [1.03, 22.0],
    [1.06, 15.0], [1.10, 18.0], [1.13, 50.0], [1.15, 95.0], [1.18, 105.0],
    [1.20, 104.0], [1.25, 110.0], [1.30, 135.0], [1.33, 210.0], [1.36, 600.0],
    [1.38, 1200.0], [1.40, 1800.0], [1.45, 2900.0], [1.50, 2300.0], [1.55, 1000.0],
    [1.60, 650.0], [1.65, 550.0], [1.70, 600.0], [1.75, 800.0], [1.80, 900.0],
    [1.85, 1500.0], [1.90, 8000.0], [1.94, 11500.0], [2.00, 7000.0], [2.05, 3600.0],
    [2.10, 2600.0], [2.15, 1900.0], [2.20, 1600.0], [2.25, 1900.0], [2.30, 2400.0],
    [2.40, 4900.0], [2.50, 8500.0], [2.60, 17000.0], [2.70, 70000.0], [2.80, 5.0e5],
    [2.95, 1.2e6], [3.10, 6.0e5], [3.50, 2.0e5], [4.00, 1.5e5], [4.50, 2.5e5],
    [5.00, 1.2e5]])


def water_absorption(lam=LAM):
    return np.exp(np.interp(lam, _AW[:, 0], np.log(_AW[:, 1])))


def planck(lam_um, T):
    """Spectral exitance (W m-2 um-1) of a blackbody."""
    lam = lam_um * 1e-6
    return 2 * np.pi * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * KB * T)) * 1e-6


# spectral emissivity of tungsten near 2500-2800 K, read from the curves of
# de Vos (1954); used only for the shape of the lamp spectrum
_EW = np.array([[0.30, 0.47], [0.40, 0.46], [0.50, 0.45], [0.60, 0.44], [0.70, 0.42],
                [0.80, 0.40], [1.00, 0.36], [1.20, 0.32], [1.50, 0.28], [2.00, 0.23],
                [2.50, 0.20], [3.00, 0.18], [5.00, 0.15]])


def tungsten_emissivity(lam_um):
    """Spectral emissivity of tungsten near 2500-2800 K (after de Vos 1954)."""
    return np.interp(lam_um, _EW[:, 0], _EW[:, 1])


GLASS_REFLECTANCE = 0.08      # two surfaces of the envelope, near-normal incidence


def glass_absorptance(lam_um, cut=2.8):
    """Hard-glass envelope: transparent from 0.32 to 2.6 um, opaque beyond `cut`."""
    a = np.zeros_like(lam_um)
    a[lam_um < 0.32] = 1.0
    ramp = (lam_um > 2.6) & (lam_um < cut)
    a[ramp] = (lam_um[ramp] - 2.6) / (cut - 2.6)
    a[lam_um >= cut] = 1.0
    return a


def glass_transmittance(lam_um, cut=2.8):
    """Transmittance = 1 - reflectance - absorptance (reflected light is
    returned towards the reflector and mostly re-emitted forward)."""
    return np.clip(1.0 - GLASS_REFLECTANCE - glass_absorptance(lam_um, cut), 0.0, 1.0)


def silicon_response(lam_um):
    """Relative spectral response of a silicon-cell pyranometer (typical)."""
    pts = np.array([[0.35, 0.0], [0.40, 0.18], [0.50, 0.40], [0.60, 0.60], [0.70, 0.78],
                    [0.80, 0.90], [0.90, 0.98], [0.95, 1.00], [1.00, 0.80], [1.05, 0.40],
                    [1.10, 0.10], [1.15, 0.0]])
    return np.interp(lam_um, pts[:, 0], pts[:, 1], left=0.0, right=0.0)


def solar_reference(lam_um):
    """Crude clear-sky solar spectrum: 5778 K blackbody, atmospheric bands removed."""
    s = planck(lam_um, 5778.0)
    for c, w, depth in [(0.76, 0.01, 0.6), (0.94, 0.04, 0.7), (1.13, 0.05, 0.8),
                        (1.38, 0.08, 1.0), (1.87, 0.10, 1.0), (2.7, 0.25, 1.0)]:
        s *= 1 - depth * np.exp(-0.5 * ((lam_um - c) / w) ** 2)
    s[lam_um < 0.30] = 0
    return s


def lamp_spectrum(T_fil=2500.0, cut=2.8):
    """Relative spectral irradiance of a halogen IR lamp after the envelope."""
    return planck(LAM, T_fil) * tungsten_emissivity(LAM) * glass_transmittance(LAM, cut)


def kappa_R(T_fil=2500.0, cut=2.8):
    """Incident lamp irradiance (lambda < cut) per unit silicon-pyranometer reading.

    A silicon sensor calibrated to sunlight reports R = K * integral(r S_lamp),
    with K = integral(S_sun) / integral(r S_sun). The true irradiance is
    integral(S_lamp); kappa_R = true / reported.
    """
    S = lamp_spectrum(T_fil, cut)
    sun = solar_reference(LAM)
    r = silicon_response(LAM)
    K = np.trapezoid(sun, LAM) / np.trapezoid(r * sun, LAM)
    return np.trapezoid(S, LAM) / (K * np.trapezoid(r * S, LAM))


def absorption_profile(T_fil=2500.0, cut=2.8, z=None, albedo=0.03):
    """Fraction of the lamp irradiance transmitted below depth z (m).

    Returns z, I(z)/I0 for the short-wave part transmitted by the envelope.
    """
    if z is None:
        z = np.concatenate([[0], np.logspace(-5, 0, 300)])
    S = lamp_spectrum(T_fil, cut)
    S = S / np.trapezoid(S, LAM)
    a = water_absorption(LAM)
    T = np.array([np.trapezoid(S * np.exp(-a * zz), LAM) for zz in z]) * (1 - albedo)
    return z, T


def three_band_fit(T_fil=2500.0, cut=2.8, depth=0.22, skin=None, albedo=0.03):
    """Fit the model's convention: skin share f_s absorbed in the top model layer
    (thickness `skin`, default 0.008 depth as in sigma_grid) plus two bands that
    start at z = 0, I(z)/I0 = f_m exp(-z/l_m) + f_d exp(-z/l_d) for z >= skin.
    """
    from scipy.optimize import least_squares
    skin = 0.008 * depth if skin is None else skin
    z, T = absorption_profile(T_fil, cut, albedo=albedo)
    Tsk = np.interp(skin, z, T)
    f_s = 1 - albedo - Tsk
    m = (z >= skin) & (z <= depth)
    zz, TT = z[m], T[m]

    def res(p):
        fm, lm, fd, ld = p
        model = fm * np.exp(-zz / lm) + fd * np.exp(-zz / ld)
        return np.log(model) - np.log(TT)
    r = least_squares(res, [0.6 * Tsk, 0.01, 0.4 * Tsk, 0.5],
                      bounds=([0, 1e-4, 0, 0.02], [1, 0.1, 1, 20]))
    fm, lm, fd, ld = r.x
    # bands starting at z = 0 overlap the skin layer: what they deposit there is
    # part of the skin share
    f_s_band = f_s - (fm * (1 - np.exp(-skin / lm)) + fd * (1 - np.exp(-skin / ld)))
    return dict(T_fil=T_fil, skin_m=skin, f_s=f_s, f_s_extra=f_s_band, f_m=fm, l_m=lm, f_d=fd, l_d=ld,
                T_bottom=float(np.interp(depth, z, T)), rmse_log=float(np.sqrt(np.mean(r.fun ** 2))))


def glass_absorbed_fraction(T_fil=2500.0, cut=2.8):
    """Fraction of the filament output absorbed (not reflected) by the envelope."""
    S0 = planck(LAM, T_fil) * tungsten_emissivity(LAM)
    return np.trapezoid(S0 * glass_absorptance(LAM, cut), LAM) / np.trapezoid(S0, LAM)


if __name__ == "__main__":
    for T in (2300, 2500, 2700, 2900):
        fit = three_band_fit(T)
        print(f"T={T} K  kappa_R={kappa_R(T):.2f}  glass_abs={glass_absorbed_fraction(T):.2f}  "
              f"f_s(top {fit['skin_m']*1000:.1f} mm)={fit['f_s']:.3f} [bands add {fit['f_s']-fit['f_s_extra']:.3f}] "
              f"f_m={fit['f_m']:.3f} l_m={fit['l_m']*1000:.2f} mm "
              f"f_d={fit['f_d']:.3f} l_d={fit['l_d']:.2f} m  T(0.22m)={fit['T_bottom']:.3f} "
              f"rmse={fit['rmse_log']:.3f}")


def solar_three_band_fit(depth=0.22, skin=None, albedo=0.06):
    """Same three-band fit for clear-sky sunlight (crude reference spectrum),
    per unit of incident short wave; used for the spectral-transfer scenario."""
    from scipy.optimize import least_squares
    skin = 0.008 * depth if skin is None else skin
    S = solar_reference(LAM)
    S = S / np.trapezoid(S, LAM)
    a = water_absorption(LAM)
    z = np.concatenate([[0], np.logspace(-5, 0, 300)])
    T = np.array([np.trapezoid(S * np.exp(-a * zz), LAM) for zz in z]) * (1 - albedo)
    Tsk = np.interp(skin, z, T)
    f_s = 1 - albedo - Tsk
    m = (z >= skin) & (z <= depth)
    zz, TT = z[m], T[m]

    def res(p):
        fm, lm, fd, ld = p
        return np.log(fm * np.exp(-zz / lm) + fd * np.exp(-zz / ld)) - np.log(TT)
    r = least_squares(res, [0.3, 0.02, 0.4, 2.0], bounds=([0, 1e-4, 0, 0.05], [1, 0.2, 1, 100]))
    fm, lm, fd, ld = r.x
    return dict(f_s=f_s, f_m=fm, l_m=lm, f_d=fd, l_d=ld, T_bottom=float(np.interp(depth, z, T)))
