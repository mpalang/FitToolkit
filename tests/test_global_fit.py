"""
Created on Fri Jul  3 19:12:09 2026

@author: morit
"""

import pytest
from generate_test_data import edcg, generate_test_data

from fittoolkit import GlobalFit

# Let's assume you can see three in the components with lifetimes in the 10s, 100s, and 1000s.
funs = {"exp1": edcg, "exp2": edcg, "exp3": edcg}
parms = {
    "exp1": {"x0": 0, "A": 1, "tau": 10, "FWHM": 0.1},
    "exp2": {"x0": 0, "A": 1, "tau": 100, "FWHM": 0.1},
    "exp3": {"x0": 0, "A": 1, "tau": 1000, "FWHM": 0.1},
}
bounds = {
    "exp1": {"x0": (-10, 10), "A": (1, 1), "tau": (0.1, 100), "FWHM": (0.03, 10)},
    "exp2": {"x0": (-10, 10), "A": (1, 1), "tau": (50, 1000), "FWHM": (0.03, 10)},
    "exp3": {"x0": (-10, 10), "A": (1, 1), "tau": (1000, 10000), "FWHM": (0.03, 10)},
}
cp = ["x0", "FWHM"]

# Test data:           t0,    A,  tau
true = {
    "exp1": {"x0": -5.21, "A": 1, "tau": 13, "FWHM": 0.242},
    "exp2": {"x0": -5.21, "A": 1, "tau": 121, "FWHM": 0.242},
    "exp3": {"x0": -5.21, "A": 1, "tau": 3001, "FWHM": 0.242},
}


def test_no_noise_with_bounds():
    x, y, Z = generate_test_data(noise_level=0.0)
    gf = GlobalFit(
        x,
        y,
        Z,
        funs,
        parms,
        bounds,
        settings={"iterations": 7, "common_parms": cp},
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
            assert gf.opt_parms[fun][name] == pytest.approx(t, rel=1e-3), (
                f"{fun}.{name}"
            )


def test_noise_with_bounds():
    x, y, Z = generate_test_data(noise_level=0.02)
    gf = GlobalFit(
        x,
        y,
        Z,
        funs,
        parms,
        bounds,
        settings={"iterations": 7, "common_parms": cp},
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


# def test_no_noise_without_bounds():
#     x, y, Z = generate_test_data(noise_level=0.0)
#     gf = GlobalFit(
#         x,
#         y,
#         Z,
#         funs,
#         parms,
#         settings={"iterations": 7, "common_parms": cp},
#         #    callback=print
#     )
#     lines = [f"{'fun':6} {'parm':5} {'true':>10} {'fit':>10} {'diff':>10} {'rel':>9}"]
#     for fun, true_parms in true.items():
#         for name, t in true_parms.items():
#             f = gf.opt_parms[fun][name]
#             d = f - t
#             rel = d / t if t != 0 else float("nan")
#             lines.append(f"{fun:6} {name:5} {t:10.4g} {f:10.4g} {d:10.3g} {rel:9.3%}")
#     result = "\n".join(lines)
#     print(result)

#     for fun, true_parms in true.items():
#         for name, t in true_parms.items():
#             assert gf.opt_parms[fun][name] == pytest.approx(t, rel=1e-3), (
#                 f"{fun}.{name}"
#             )


# def test_noise_without_bounds():
#     x, y, Z = generate_test_data(noise_level=0.02)
#     gf = GlobalFit(
#         x,
#         y,
#         Z,
#         funs,
#         parms,
#         settings={"iterations": 7, "common_parms": cp},
#         #    callback=print
#     )
#     lines = [f"{'fun':6} {'parm':5} {'true':>10} {'fit':>10} {'diff':>10} {'rel':>9}"]
#     for fun, true_parms in true.items():
#         for name, t in true_parms.items():
#             f = gf.opt_parms[fun][name]
#             d = f - t
#             rel = d / t if t != 0 else float("nan")
#             lines.append(f"{fun:6} {name:5} {t:10.4g} {f:10.4g} {d:10.3g} {rel:9.3%}")

#     result = "\n".join(lines)
#     print(result)

#     for fun, true_parms in true.items():
#         for name, t in true_parms.items():
#             assert gf.opt_parms[fun][name] == pytest.approx(t, rel=3e-2), (
#                 f"{fun}.{name}"
#             )


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
    test_no_noise_with_bounds()
    test_noise_with_bounds()
    # test_no_noise_without_bounds()
    # test_noise_without_bounds()
