"""
Created on Sun Jul 26 11:11:23 2026

@author: moritzpalang
"""

import numpy as np

# from collections.abc import Callable
# from dataclasses import dataclass


# =============================================================================
# Convolute Functions with irf
# =============================================================================
# def convolve(y, irf):
#     """
#     Takes two np.arrays and convolves them and outputs an np.array of length len(y)
#     """
#     irf = np.asarray(irf, dtype=float)
#     irf = irf / np.sum(irf) #normalize convolution function
#     y_conv = np.convolve(y, irf, mode="full")
#     # center kernel
#     k = len(irf) // 2
#     return y_conv[k:k + len(y)]


def convolve(x, y, irf, edge="edge"):
    """
    Convolve y(x) with irf on an arbitrary, possibly non-uniform grid x.

    irf  : callable, irf(d) -> weight at offset d = x_i - x_j
           e.g. lambda d: np.exp(-(d - x0)**2 / (2*s**2))
    edge : 'edge' renormalises each row, compensating kernel truncation at the
           boundaries; 'zero' does not (and then irf must have unit area).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be 1-D and the same length")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")

    w = np.empty_like(x)  # trapezoid weights, non-uniform ok
    w[1:-1] = (x[2:] - x[:-2]) / 2
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2

    K = irf(x[:, None] - x[None, :]) * w
    if edge == "edge":
        s = K.sum(axis=1, keepdims=True)
        if np.any(s == 0):
            raise ValueError(
                "irf has no support on this grid (too narrow for the spacing?)"
            )
        K = K / s
    elif edge != "zero":
        raise ValueError("edge must be 'edge' or 'zero'")
    return K @ y


# Check below. I think none of this is used anymore, but I will keep it for now in case we want to use it later.
# @dataclass
# class FitProgress:
#     iteration: int
#     total_iterations: int
#     chi2: float
#     parameters: dict


# @dataclass
# class InputParms:
#     parms: list[list[str]]
#     p0: list[list[float]]
#     p_lower: list[list[float]]|None=None
#     p_upper: list[list[float]]|None=None

#     common_parms: list[str]|None=None

#     irf_p0: list[float]|None=None
#     irf_p_lower: list[float]|None=None
#     irf_p_upper: list[float]|None=None


# def unpack(Parms, P0, P_lower, P_upper, common_parms: list|None=None,
#            irfParms:list|None=None, irfP0:list|None=None,
#            irfP_lower:list|None=None, irfP_upper:list|None=None):

#     fit_parms = []
#     fit_p0 = []
#     fit_p_lower = []
#     fit_p_upper = []
#     p_map = []*len(Parms)
#     common_indices = {}
#     irf_indices = {}

#     for n in range(len(Parms)):
#         fun_name = 'fun'+str(n+1)+'_'
#         for parm,p0,p_lower,p_upper in zip(Parms[n],P0[n],P_lower[n],P_upper[n]):
#             is_common = (parm in common_parms)
#             index = len(fit_parms)
#             if is_common:
#                 if parm in fit_parms:
#                     p_map[n].append(common_indices[parm])
#                 else:
#                     fit_parms.append(parm)
#                     fit_p0.append(p0)
#                     fit_p_lower.append(p_lower)
#                     fit_p_upper.append(p_upper)
#                     p_map[n].append(index)
#                     common_indices[name] = index #TODO: watch this and make sure index assignment is correct.
#             else:
#                 name = fun_name+name
#                 fit_parms.append(name)
#                 fit_p0.append(p0)
#                 fit_p_lower.append(p_lower)
#                 fit_p_upper.append(p_upper)
#                 p_map[n].append(index)

#     if irfParms:
#         for parm,p0,p_lower,p_upper in zip(irfParms,irfP0,irfP_lower,irfP_upper):
#             is_common = (name in common_parms)
#             if is_common:
#                 if name in fit_parms:
#                     irf_indices.append(common_indices[name])
#             else:
#                 name = 'irf_'+name
#                 fit_parms.append(name)
#                 fit_p0.append(p0)
#                 fit_p_lower.append(p_lower)
#                 fit_p_upper.append(p_upper)

#     return fit_parms, fit_p0, fit_p_lower, fit_p_upper, p_map, common_indices, irf_indices


# @dataclass
# class FitParms:
#     input_parms: InputParms

#     def __post_init__(self):
#         ip = self.input_parms
#         unpacked = unpack(ip.parms, ip.p0, ip.p_lower, ip.p_upper,ip.common_parms)
#         self.parms, self.p0, self.p_lower, self.p_upper, self.p_map, self.common_indices, self.irf_indices = unpacked


# def getelements(L,indices):
#     return [l for n,l in enumerate(L) if n in indices]
