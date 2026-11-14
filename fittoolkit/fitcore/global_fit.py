"""
Created on Thu Jan 19 11:29:24 2023

@author: work

WL: Wavelength 1D array
dt: Delay times 1D array
Z: Data Values as 2D array with shape (len(WL),len(dt))

"""

from collections.abc import Callable

import numpy as np
from iminuit import Minuit as Mi
from numpy.linalg import inv

from fittoolkit.utils.tools import convolve

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
    "fixed_DADS": False,
    "scale": False,
    "use_irf": False,
}


# Main Global Fit Class:
class GlobalFit:
    """
    Creates global fit Object to fit 2D spectroscopic Data and return DADS.
    Possible Fit Methods:
        'L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',
        #Additional Information required: 'trust-krylov','trust-ncg','dogleg','Newton-CG','trust-exact'

    callback: function that handles progres update as plain text.

    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        Z: np.ndarray,  # required
        funs: dict[str, Callable],
        parms: dict[str, dict[str, float]],  # required
        bounds: dict[str, dict[str, (float, float)]] | None = None,  # Optional
        settings: dict | None = None,
        name: str = "Data",
        callback: Callable | None = None,  # Optional
    ):

        # Declarations:
        self.callback = callback

        self.error_log = []  # writes a message to the error log if a non-finite value is encountered during the fit.
        self.nfcn = 0  # counts the number of function calls during the fit.

        self.x = x
        self.y = y
        self.Z = Z

        self.funs = funs
        self.parms = parms
        self.bounds = bounds
        self.make_keys_lowercase()

        self.initialize()
        if settings:
            self.update_settings(**settings)

        self.validate()
        self.unpack_parameters()

        if self.set["scale"]:
            self.scale()

        self.fit()

    # =============================================================================
    # Settings and data wrangling
    # =============================================================================

    @property
    def no_comps(self):
        return len(self.funs)

    def make_keys_lowercase(self):
        no_comps = self.no_comps
        self.funs = {name.lower(): fun for name, fun in self.funs.items()}
        self.parms = {name.lower(): parm for name, parm in self.parms.items()}
        if self.bounds:
            self.bounds = {name.lower(): bound for name, bound in self.bounds.items()}
        if self.no_comps != no_comps:
            raise ValueError("Duplicate Functionname detected!")

    def validate(self):
        if self.Z.shape != (np.squeeze(self.y.shape), np.squeeze(self.x.shape)):
            raise ValueError("x,y and Z shapes must match")
        if self.bounds:
            pass  # TODO: check bounds
        if self.set["use_irf"] and "irf" not in self.funs:
            raise ValueError(
                "IRF usage is enabled but no IRF function was provided in funs."
            )

    def initialize(self):
        self.set_defaults()
        self.deviation = 1  # this is just to have a value for deviation before the fit is run. It will be overwritten by the fit function.
        self.D = np.zeros((self.no_comps, len(self.x)))
        self.result = "No Fit Available"

    def set_defaults(self):
        """restores default settings"""
        self.set = default_settings

    def update_settings(self, **set_update):
        "update settings. Input settings dict keys must match default keys."
        for key, value in set_update.items():
            if type(value) == dict:
                for k in value:
                    self.set[key][k] = value[k]
            else:
                self.set[key] = value

        if "irf" not in [fun.lower() for fun in self.funs]:
            self.set["use_irf"] = (
                False  # If no irf was passed, no irf can be used. Maybe add a user info here.
            )

    def unpack_parameters(self):
        self.fit_parms = {}
        self.fit_bounds = {}

        for fun, parms in self.parms.items():
            for name, p in parms.items():
                if name in self.set["common_parms"]:
                    if name in self.fit_parms:
                        continue
                    else:
                        self.fit_parms[name] = p
                        if self.bounds:
                            self.fit_bounds[name] = self.bounds[fun][name]
                else:
                    self.fit_parms[f"{fun}_{name}"] = p
                    if self.bounds:
                        self.fit_bounds[f"{fun}_{name}"] = self.bounds[fun][name]

    def get_scaling_factors(self):
        def get_factor(x):
            return 10 ** -round(np.log10(abs(x))) if x != 0 else 1

        self.scaling_factors = {
            name: get_factor(v) for name, v in self.fit_parms.items()
        }

    def scale(self):
        self.fit_parms = {
            name: v * self.scaling_factors[name] for name, v in self.fit_parms.items()
        }
        if self.bounds:
            self.fit_bounds = {
                name: (
                    v[0] * self.scaling_factors[name],
                    v[1] * self.scaling_factors[name],
                )
                for name, v in self.fit_bounds.items()
            }

    def unscale(self):
        self.fit_parms = {
            name: v / self.scaling_factors[name] for name, v in self.fit_parms.items()
        }
        if self.bounds:
            self.fit_bounds = {
                name: (
                    v[0] / self.scaling_factors[name],
                    v[1] * self.scaling_factors[name],
                )
                for name, v in self.fit_bounds.items()
            }

    def get_fun_parms(self, fun_name, fit_parms):
        fun_parms = []
        for name in self.parms[fun_name]:
            if name in self.set["common_parms"]:
                fun_parms.append(fit_parms[name])
            else:
                fun_parms.append(fit_parms[f"{fun_name}_{name}"])
        return fun_parms

    def get_goodness(self, chi2):
        return float(chi2 / np.sum(self.Z**2))

    def pack_parameters(self, fit_parms):
        "Creates optimized parameters (opt_parms) attribute with original structure."
        self.fit_parms = fit_parms
        self.opt_parms = {}

        for fun, parms in self.parms.items():
            self.opt_parms[fun] = {}
            for name in parms:
                if name in self.set["common_parms"]:
                    self.opt_parms[fun][name] = fit_parms[name]
                else:
                    self.opt_parms[fun][name] = fit_parms[f"{fun}_{name}"]

    # =============================================================================
    # Actual Fit Function
    # =============================================================================
    def fit(self, **set_update):

        # =============================================================================
        #  Definitions for fit:
        # =============================================================================
        fit_parms = self.fit_parms
        parm_names = tuple(self.fit_parms.keys())
        D = np.full((self.no_comps, len(self.x)), np.nan)

        def make_D(*p):
            fit_parms = dict(zip(parm_names, p))
            for n, (fun_name, fun) in enumerate(self.funs.items()):
                if fun_name == "irf":
                    continue
                if self.set["use_irf"]:
                    D[n, :] = convolve(
                        self.x,
                        fun(self.x, *self.get_fun_parms(fun_name, fit_parms)),
                        lambda x: self.funs["irf"](
                            x, *self.get_fun_parms("irf", fit_parms)
                        ),
                    )
                else:
                    D[n, :] = fun(self.x, *self.get_fun_parms(fun_name, fit_parms))
            return D

        def make_DADS(Z, D):
            if self.set["fixed_DADS"] is not None:
                return self.set["fixed_DADS"]
            G = D @ D.T
            if np.linalg.cond(G) > 1e12:
                G = G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G))
            return (Z @ D.T) @ (inv(G))
            # Try new approach?:
            # return np.linalg.lstsq(D.T, Z.T, rcond=None)[0].T

        def calc_chi2(*p):
            D = make_D(*p)
            if not np.all(np.isfinite(D)):
                p_str = ", ".join(f"{n}={v:.3g}" for n, v in zip(parm_names, p))
                self.error_log.append(
                    f"Non-finite D matrix encountered for parameters: {p_str}"
                )

            DADS = make_DADS(self.Z, D)
            Zfit = DADS @ D

            mask = np.isfinite(self.Z)
            r = self.Z - Zfit

            if not np.all(np.isfinite(r[mask])):
                self.error_log.append(
                    f"Non-finite chi2 encountered for parameters: {p_str} -> {np.sum(r[mask] ** 2):.3g}"
                )
                return 1e3 * np.sum(self.Z[mask] ** 2)  # Use the last valid chi2 value

            return np.sum(r[mask] ** 2)

        # =============================================================================
        # create Minuit object and set fit settings
        p0 = tuple(fit_parms[name] for name in parm_names)
        m = Mi(calc_chi2, *p0, name=parm_names)
        if self.bounds:
            for name, bound in self.fit_bounds.items():
                m.limits[name] = bound
        if self.set["errors"]:
            m.errors = self.set["errors"]
        if self.set["tol"]:
            m.tol = self.set["tol"]
        if self.set["strategy"]:
            m.strategy = self.set["strategy"]

        if self.set["simplex"]:
            if self.callback:
                self.callback("running simplex")
            m.simplex()

        # =============================================================================
        #  Start Fit:
        # =============================================================================
        if self.callback:
            self.callback("start fit")

        self.goodnesses = []  # create list so you can follow the error throughout the iterations.
        self.chi2s = []  # create list so you can follow the error throughout the iterations.

        for iteration in range(self.set["iterations"]):
            # =============================================================================
            # Run Fit
            if self.set["method"] == "migrad":
                m.migrad()
            elif self.set["method"]:
                m.scipy(self.set["method"])

            if self.set["hesse"]:
                m.hesse()

            # =============================================================================
            # Update Data for next iteration
            fit_parms = dict(zip(parm_names, list(m.values)))
            bad = {
                k: type(v).__name__
                for k, v in fit_parms.items()
                if not isinstance(v, (int, float, np.floating))
            }
            if bad:
                raise TypeError(f"non-numeric fit parameters: {bad}")
            D = make_D(*m.values)
            DADS = make_DADS(self.Z, D)

            goodness = self.get_goodness(calc_chi2(*m.values))
            self.goodnesses.append(goodness)
            self.chi2s.append(calc_chi2(*m.values))

            # User output:
            if self.callback:
                report = f"Iter.{iteration}/{self.set['iterations']} finished with chi2={calc_chi2(*m.values):.3g}  (goodness: {goodness:.3g})"
                self.callback(report)

            self.nfcn = self.nfcn + m.nfcn

        # -------------------
        # End Iterations Loop. Unpack results.
        # -------------------
        self.D = D
        self.DADS = DADS
        self.Z_fit = DADS @ D
        if self.set["scale"]:
            self.unscale()
        self.pack_parameters(fit_parms)
        self.residuum = self.Z_fit - self.Z
        self.m = m

        # # =============================================================================
        # # print Results
        # # =============================================================================
        # if self.set['print_level']>0:
        #     print(m.params)
        #     print(m.fmin)
        #     if self.callback:
        #         self.callback(f'{m.params}')
        #         self.callback(f'{m.fmin}')

    # =============================================================================
    # TODO: Singular Value Decomposition
    # =============================================================================
    # class svd:
    #     def __init__(self,global_fit):
    #         self.u,self.s,self.v=np.linalg.svd(global_fit.Z)

    #     def plot(self):
    #         decays=[]
    #         specs=[]
    #         for n,i in enumerate(s):
    #             if i>(np.nanmax(s)/10):
    #                 decays.append(v[n,:]*i)
    #                 specs.append(u[:,n]*i)

    #         plot(fs[key]['fit'].wl,specs,title=key)
    #         timeplot(fs[key]['fit'].dt,decays,title=key,
    #                  x_break=5,labelpad=(0.1,0.1))
