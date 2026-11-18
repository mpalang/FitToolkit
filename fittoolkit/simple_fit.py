"""
Created on Wed Nov 18 2026

@author: Moritz Lang

WL: Wavelength 1D array
dt: Delay times 1D array
Z: Data Values as 2D array with shape (len(WL),len(dt))

"""

from collections.abc import Callable

import numpy as np

# Define default settings here:
default_settings = {
    "common_parms": [],
    "method": "Powell",  # 'migrad','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',
    "scan": False,
    "simplex": False,
    "hesse": False,
    "tol": None,
    "errors": None,
    "strategy": None,  # Minuit strategy: 0,1,2. 0=fast, 1=default, 2=robust
    "iterations": 7,
    "fixed_DADS": None,
    "scale": False,
}


# Main Simple Fit Class:
class SimpleFit:
    """
    Global fit of 2D spectroscopic data (e.g. transient absorption).

    The data are modeled as ``Z ≈ DADS @ D``: each kinetic function in ``funs``
    gives one row of ``D`` (its time dependence), and the decay-associated
    spectra ``DADS`` are solved linearly at every step. Only the kinetic
    parameters are optimized. The fit runs on construction.

    Parameters
    ----------
    x : np.ndarray
        Time axis, shape (n_t,). Does not need to be uniformly spaced.
    y : np.ndarray
        Data array, shape (n_t,).
    funs : dict[str, Callable]
        Kinetic functions by name, each called as ``f(x, *parms)`` with the
        parameters in the order of ``parms[name]``. The reserved name
        ``"irf"`` is not a component: it is the instrument response, called
        as ``irf(offset, *parms)`` and used to convolve all other functions.
    parms : dict[str, dict[str, float]]
        Start values per function, e.g. ``{"exp1": {"x0": 0, "tau": 10}}``.
        Use the key ``"irf"`` for the IRF parameters.
    bounds : dict[str, dict[str, tuple[float, float]]], optional
        Bounds per function and parameter, same structure as ``parms``.
        Equal lower and upper bounds fix a parameter.
    settings : dict, optional
        Overrides for the default settings, e.g.
        ``"method"``: "migrad" (default) or a SciPy method
        ('L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'CG', 'Powell', 'BFGS');
        ``"iterations"``: number of fit iterations;
        ``"common_parms"``: parameter name(s) shared by all functions, e.g. "x0";
        ``"hesse"``: compute parameter uncertainties;
        ``"fixed_DADS"``: use these DADS instead of solving for them.
    name : str, default "Data"
        Name of the data set, used in output.
    callback : Callable, optional
        Called with progress messages (str).

    Attributes
    ----------
    opt_parms : dict[str, dict[str, float]]
        Fitted parameters, same structure as ``parms``.
    DADS : np.ndarray
        Decay-associated spectra, shape (n_wl, n_components). Use this if the spectra are known.
    chi2s, goodnesses : list[float]
        Chi² and goodness of fit after each iteration.
    error_log : list[str]
        Parameter sets where the model returned non-finite values.

    Raises
    ------
    RuntimeError
        If the fit ends at parameters where the model is non-finite.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        funs: dict[str, Callable],
        parms: dict[str, dict[str, float]],  # required
        bounds: dict[str, dict[str, (float, float)]] | None = None,  # Optional
        settings: dict | None = None,
        name: str = "Data",
        callback: Callable | None = None,  # Optional
    ):

        pass
