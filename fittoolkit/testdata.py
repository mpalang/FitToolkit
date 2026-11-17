"""
Created on Sat Jun 27 10:28:16 2026

@author: morit
"""

import numpy as np
from numpy.linalg import inv

from fittoolkit.models import edcg, exp_decay, gauss


def generate_test_data(
    t0=-1.2, tau1=13, tau2=121, tau3=3001, FWHM=0.2, noise_level=0.0
):
    x = np.concatenate(
        [
            np.arange(-10, 2, 0.05),  # 240 points around x0
            np.geomspace(2, 10000, 260),  # ~70 per decade, plenty for smooth decays
        ]
    )

    true = {
        "exp1": {"x0": t0, "A": 1, "tau": tau1},
        "exp2": {"x0": t0, "A": 1, "tau": tau2},
        "exp3": {"x0": t0, "A": 1, "tau": tau3},
    }

    if FWHM is not None:
        for value in true.values():
            value["FWHM"] = FWHM

    if FWHM is None:
        D1 = exp_decay(x, t0, 1, tau1)
        D2 = exp_decay(x, t0, 1, tau2)
        D3 = exp_decay(x, t0, 1, tau3)
    else:
        D1 = edcg(x, t0, FWHM, 1, tau1)
        D2 = edcg(x, t0, FWHM, 1, tau2)
        D3 = edcg(x, t0, FWHM, 1, tau3)

    D = np.column_stack((D1, D2, D3))

    # DADS = DADS+np.random.uniform(-0.001,0.001,DADS.shape)
    y = np.linspace(300, 800, 100)

    DADS1 = gauss(y, 320, -0.3, 80) + gauss(y, 600, 0.4, 300) + gauss(y, 450, 0.3, 80)
    DADS2 = gauss(y, 320, -0.2, 80) + gauss(y, 400, 0.5, 230) + gauss(y, 700, 0.5, 200)
    DADS3 = gauss(y, 320, -0.5, 80) + gauss(y, 650, 0.3, 400) + gauss(y, 520, 0.03, 100)
    DADS = np.column_stack((DADS1, DADS2, DADS3))

    Z = DADS @ D.T

    rng = np.random.default_rng(69)  # e.g. seed=0 as a function argument
    Z = Z + rng.normal(0, noise_level, Z.shape)

    return x, y, Z, DADS, true


def make_DADS(Z, D):
    G = D @ D.T
    if np.isclose(np.linalg.det(G), 0):
        G = G + 1e-12 * np.eye(*G.shape)
        DADS = (Z @ D.T) @ (inv(G))
    else:
        DADS = (Z @ D.T) @ (inv(G))
    return DADS
