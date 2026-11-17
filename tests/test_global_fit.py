"""
Created on Fri Jul  3 19:12:09 2026

@author: morit
"""

import numpy as np
import pytest

from fittoolkit import GlobalFit
from fittoolkit.models import edcg, exp_decay, gauss
from fittoolkit.testdata import generate_test_data
from fittoolkit.tools import convolve

# Let's assume you can see three in the components with lifetimes in the 10s, 100s, and 1000s.
funs_c = {"exp1": edcg, "exp2": edcg, "exp3": edcg}
parms_c = {
    "exp1": {"x0": 0, "FWHM": 0.1, "A": 1, "tau": 10},
    "exp2": {"x0": 0, "FWHM": 0.1, "A": 1, "tau": 100},
    "exp3": {"x0": 0, "FWHM": 0.1, "A": 1, "tau": 1000},
}
bounds_c = {
    "exp1": {"x0": (-10, 10), "FWHM": (0.03, 10), "A": (1, 1), "tau": (0.1, 100)},
    "exp2": {"x0": (-10, 10), "FWHM": (0.03, 10), "A": (1, 1), "tau": (50, 1000)},
    "exp3": {"x0": (-10, 10), "FWHM": (0.03, 10), "A": (1, 1), "tau": (1000, 10000)},
}
cp_c = ["x0", "FWHM"]

# No IRF:
funs = {"exp1": exp_decay, "exp2": exp_decay, "exp3": exp_decay}
parms = {
    "exp1": {"x0": 0, "A": 1, "tau": 10},
    "exp2": {"x0": 0, "A": 1, "tau": 100},
    "exp3": {"x0": 0, "A": 1, "tau": 1000},
}
bounds = {
    "exp1": {"x0": (-10, 10), "A": (1, 1), "tau": (0.1, 100)},
    "exp2": {"x0": (-10, 10), "A": (1, 1), "tau": (50, 1000)},
    "exp3": {"x0": (-10, 10), "A": (1, 1), "tau": (1000, 10000)},
}
cp = ["x0"]


def test_convolution():
    import numpy as np

    t0 = 0
    tau = 30
    FWHM = 10
    t = np.linspace(-20, 100, 500)
    y = np.exp(-t / tau) * np.heaviside(t - t0, 1)
    irf = lambda x: gauss(x, 0, FWHM, 1)
    y_numeric = convolve(t, y, irf)
    y_analytic = edcg(t, t0, FWHM, 1, tau)

    diff = np.abs(y_numeric - y_analytic)
    max_error = np.max(diff) / np.max(np.abs(y_analytic))
    sum_error = np.sum(diff) / np.sum(np.abs(y_analytic))

    # from matplotlib import pyplot as plt
    # fig,ax = plt.subplots()
    # ax.plot(t,y_numeric)
    # ax.plot(t,y_analytic)
    # plt.show()

    assert max_error < 1e-1, f"convolution error {max_error:.2e}"
    assert sum_error < 1e-2, f"convolution error {sum_error:.2e}"


def test_no_noise():
    x, y, Z, _, true = generate_test_data(noise_level=0.0)
    gf = GlobalFit(
        x,
        y,
        Z,
        funs_c,
        parms_c,
        bounds_c,
        settings={"iterations": 7, "common_parms": cp_c},
        #    callback=print
    )
    lines = [f"{'fun':6} {'parm':5} {'true':>10} {'fit':>10} {'diff':>10} {'rel':>9}"]
    for fun, true_parms in true.items():
        for name, t in true_parms.items():
            f = gf.opt_parms[fun][name]
            d = f - t
            rel = d / t if t != 0 else float("nan")
            lines.append(f"{fun:6} {name:5} {t:10.4g} {f:10.4g} {d:10.3g} {rel:9.3%}")
    result = "\n".join(lines)
    print(result)

    for fun, true_parms in true.items():
        for name, t in true_parms.items():
            tol = {
                "rel": 1e-3
            }  # x0: one grid step, no IRF {"abs": 0.02} if name == "x0" else
            assert gf.opt_parms[fun][name] == pytest.approx(t, **tol), f"{fun}.{name}"


def test_noise():
    x, y, Z, _, true = generate_test_data(noise_level=0.2)
    gf = GlobalFit(
        x,
        y,
        Z,
        funs_c,
        parms_c,
        bounds_c,
        settings={"iterations": 7, "common_parms": cp_c},
        #    callback=print
    )
    lines = [f"{'fun':6} {'parm':5} {'true':>10} {'fit':>10} {'diff':>10} {'rel':>9}"]
    for fun, true_parms in true.items():
        for name, t in true_parms.items():
            f = gf.opt_parms[fun][name]
            d = f - t
            rel = d / t if t != 0 else float("nan")
            lines.append(f"{fun:6} {name:5} {t:10.4g} {f:10.4g} {d:10.3g} {rel:9.3%}")

    result = "\n".join(lines)
    print(result)

    for fun, true_parms in true.items():
        for name, t in true_parms.items():
            assert gf.opt_parms[fun][name] == pytest.approx(t, rel=3e-2), (
                f"{fun}.{name}"
            )


def test_noise_with_irf():
    x, y, Z, _, true = generate_test_data(FWHM=None, noise_level=0.0)

    FWHM = 0.2
    irf = lambda x, fwhm: gauss(x, 0, fwhm, 1)
    Zc = np.zeros(Z.shape)
    for i in range(len(y)):
        Zc[i, :] = convolve(x, Z[i, :], lambda x: irf(x, FWHM))

    # Add irf to functions
    funs["irf"] = irf
    parms["irf"] = {"FWHM": 1}
    bounds["irf"] = {"FWHM": (1e-5, 2)}

    gf = GlobalFit(
        x,
        y,
        Zc,
        funs,
        parms,
        bounds,
        settings={"iterations": 2, "common_parms": cp},
        callback=print,
    )

    lines = [f"{'fun':6} {'parm':5} {'true':>10} {'fit':>10} {'diff':>10} {'rel':>9}"]
    for fun, true_parms in true.items():
        for name, t in true_parms.items():
            f = gf.opt_parms[fun][name]
            d = f - t
            rel = d / t if t != 0 else float("nan")
            lines.append(f"{fun:6} {name:5} {t:10.4g} {f:10.4g} {d:10.3g} {rel:9.3%}")

    result = "\n".join(lines)
    print(result)

    for fun, true_parms in true.items():
        for name, t in true_parms.items():
            assert gf.opt_parms[fun][name] == pytest.approx(t, rel=1e-2), (
                f"{fun}.{name}"
            )


# #Test different methods:
# GF = [1]
# methods = [
#     "migrad",
#     "L-BFGS-B",
#     "TNC",
#     "COBYLA",
#     "SLSQP",
#     "trust-constr",
#     "CG",
#     "Powell",
#     "BFGS",
# ]


if __name__ == "__main__":
    test_no_noise()
    test_noise()
    test_convolution()
    test_noise_with_irf()

