import numpy as np
from scipy.special import erfc, erfcx

_C = 4 * np.log(2)  # converts FWHM to the Gaussian exponent: exp(-_C * u**2 / fwhm**2)


def gauss(x, x0, fwhm, A):
    """Area-normalized Gaussian (integral = A), parameterized by FWHM."""
    c = _C / fwhm**2
    return A * np.sqrt(c / np.pi) * np.exp(-c * (x - x0) ** 2)


def dgauss(x, x0, fwhm, A):
    """First derivative of gauss with respect to x."""
    c = _C / fwhm**2
    return -2 * c * (x - x0) * gauss(x, x0, A, fwhm)


def d2gauss(x, x0, fwhm, A):
    """Second derivative of gauss with respect to x."""
    c = _C / fwhm**2
    u = x - x0
    return (4 * c**2 * u**2 - 2 * c) * gauss(x, x0, A, fwhm)


def exp_decay(x, x0, A, tau):
    """Exponential decay starting at x0, zero before it."""
    x = np.asarray(x, float)
    z = -(x - x0) / tau
    return A * np.where(x >= x0, np.exp(np.minimum(z, 0.0)), 0.0)


def exp_decay_conv_gauss(x, x0, fwhm, A, tau):
    """
    Analytical convolution of  A*exp(-(x-x0)/tau)*H(x-x0)  with a unit-area
    Gaussian IRF of full width at half maximum `fwhm`.

    Reduces to the bare exponential decay as fwhm -> 0. A is the amplitude the
    un-convolved decay would have at x0.
    """
    z = np.asarray(x, dtype=float) - x0
    # if tau <= 0 or fwhm <= 0:
    #     raise ValueError(f"tau and fwhm must be > 0, got tau={tau}, fwhm={fwhm}")

    s = fwhm / 2.354820045
    u = (s / tau - z / s) / np.sqrt(2)
    out = np.empty_like(z)

    m = u >= 0
    out[m] = np.exp(-(z[m] ** 2) / (2 * s**2)) * erfcx(u[m])
    nm = ~m
    out[nm] = np.exp(s**2 / (2 * tau**2) - z[nm] / tau) * erfc(u[nm])

    return (A / 2) * out


edcg = exp_decay_conv_gauss


def second_order(x, x0, A, k, c0):
    """Second-order decay starting at x0, zero before it: A / (1 + c0*k*(x - x0))."""
    x = np.asarray(x, float)
    z = np.maximum(x - x0, 0.0)  # 0 before x0: avoids a zero/negative denominator there
    return A * np.where(x >= x0, 1.0 / (1.0 + c0 * k * z), 0.0)


FUNCTIONS = {
    "ed": exp_decay,
    "exp_decay": exp_decay,
    "exp_decay_conv_gauss": exp_decay_conv_gauss,
    "edcg": exp_decay_conv_gauss,
    "gauss": gauss,
    "dgauss": dgauss,
    "d2gauss": d2gauss,
    "second_order": second_order,
    "so": second_order,
}
