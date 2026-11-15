"""
Created on Sat Jun 27 10:28:16 2026

@author: morit
"""

import numpy as np
from numpy.linalg import inv
from scipy.special import erf  # TODO: get rid of this dependency


def gauss(x, x0, A, FWHM):  # normalized to 1 for IRF
    s = FWHM / (2 * np.sqrt(2 * np.log(2)))
    y = (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-((x - x0) ** 2) / (2 * s**2))
    return A * y


def normgauss(x, x0, s):
    y = (s * np.sqrt(2 * np.pi)) * np.exp(-((x - x0) ** 2) / (2 * s**2))
    return y / max(y)


def exp_decay(t, t0, A, tau):
    # k=1/tau
    return A * np.exp(-(t - t0) / tau) * np.heaviside(t - t0, 0)


def edcg(t, t0, A, tau, FWHM): #exponential decay convolved with gauss
    t = t - t0
    s = FWHM / (2 * np.sqrt(2 * np.log(2)))
    y1 = np.exp(s**2 / (2 * tau**2)) * np.exp(-t / tau)
    y2 = erf((t - (s**2 / tau)) / (s * np.sqrt(2)))
    y = (A / 2) * y1 * (1 + y2)
    return y


def generate_test_data(noise_level=0.0):
    x = np.concatenate([
            np.arange(-10, 2, 0.05),       # 240 points around x0
            np.geomspace(2, 10000, 260),   # ~70 per decade, plenty for smooth decays
        ])
    y = np.linspace(300, 800, 100)

    DADS1 = gauss(y, 320, -0.3, 80) + gauss(y, 600, 0.4, 300) + gauss(y, 450, 0.3, 80)
    DADS2 = gauss(y, 320, -0.2, 80) + gauss(y, 400, 0.5, 230) + gauss(y, 700, 0.5, 200)
    DADS3 = gauss(y, 320, -0.5, 80) + gauss(y, 650, 0.3, 400) + gauss(y, 520, 0.03, 100)

    D1 = edcg(x, -5.21, 1, 13, 0.242)
    D2 = edcg(x, -5.21, 1, 121, 0.242)
    D3 = edcg(x, -5.21, 1, 3001, 0.242)

    DADS = np.column_stack((DADS1, DADS2, DADS3))
    # DADS = DADS+np.random.uniform(-0.001,0.001,DADS.shape)
    D = np.column_stack((D1, D2, D3))
    D = D + np.random.uniform(-noise_level, noise_level, D.shape)

    Z = DADS @ D.T

    return x, y, Z


def make_DADS(Z, D):
    G = D @ D.T
    if np.isclose(np.linalg.det(G), 0):
        G = G + 1e-12 * np.eye(*G.shape)
        DADS = (Z @ D.T) @ (inv(G))
    else:
        DADS = (Z @ D.T) @ (inv(G))
    return DADS
