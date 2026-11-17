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

from fittoolkit.tools import convolve

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


# Main Global Fit Class:
class GlobalFit:
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
        Spectral axis (e.g. wavelength), shape (n_wl,).
    Z : np.ndarray
        Data, shape (n_wl, n_t). NaNs are allowed and excluded from chi².
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
        Z: np.ndarray,  # required
        funs: dict[str, Callable],
        parms: dict[str, dict[str, float]],  # required
        bounds: dict[str, dict[str, (float, float)]] | None = None,  # Optional
        settings: dict | None = None,
        name: str = "Data",
        callback: Callable | None = None,  # Optional
        ):

        # Declarations:
        self.name = name
        self.callback = callback

        self.error_log = []  # writes a message to the error log if a non-finite value is encountered during the fit.
        self.nfcn = 0  # counts the number of function calls during the fit.

        self.x = x
        self.y = y
        self.Z = Z

        self.irf = None
        self.funs = funs
        self.parms = parms
        self.bounds = bounds

        self.fit_parms = {}
        self.fit_bounds = {}
        self.fixed_parms = []

        self.set_defaults()
        if settings:
            self.update_settings(**settings)
        # self.make_keys_lowercase()
        self.validate()
        self.unpack_input()
        if self.set["scale"]:
            self.scale()

        self.initialize()

        self.fit()

    # =============================================================================
    # Settings and data wrangling
    # =============================================================================
 
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


    def make_keys_lowercase(self):
        no_comps = len(self.funs)
        self.funs = {name.lower(): fun for name, fun in self.funs.items()}
        self.parms = {name.lower(): parm for name, parm in self.parms.items()}
        if self.bounds:
            self.bounds = {name.lower(): bound for name, bound in self.bounds.items()}
        if len(self.funs) != no_comps:
            raise ValueError("Duplicate Functionname detected!")


    def validate(self):
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("x and y must be 1-D")
        if self.Z.shape != (self.y.size, self.x.size):
            raise ValueError("Z must have shape (len(y), len(x))")
        if self.parms.keys() != self.funs.keys():
            raise ValueError("funs and parms must have the same keys")
        if self.bounds is not None and self.bounds.keys() != self.parms.keys():
            raise ValueError("bounds and parms must have the same keys")


    @property
    def no_comps(self):
        return len(self.funs)


    def initialize(self):
        """Initialize fit variables."""
        self.D = np.zeros((self.no_comps, len(self.x)))
        self.result = "No Fit Available"

        self.mask = np.isfinite(self.Z)
        self.has_nans = not self.mask.all()
        self.Zsq = np.sum(self.Z[self.mask] ** 2)


    def unpack_input(self):
        """Unpack input functions, parameters and bounds. Also recognize irf by key and separate from the rest."""

        for fun_name, _ in self.funs.items():
            parms = self.parms[fun_name]

            for parm_name, p in parms.items():
                if parm_name in self.set["common_parms"]:
                    if parm_name in self.fit_parms:
                        continue
                    else:
                        self.fit_parms[parm_name] = p
                        if self.bounds:
                            self.fit_bounds[parm_name] = self.bounds[fun_name][parm_name]
                            if self.bounds[fun_name][parm_name][0]==self.bounds[fun_name][parm_name][1]:
                                self.fixed_parms.append(parm_name)
                else:
                    self.fit_parms[f"{fun_name}_{parm_name}"] = p
                    if self.bounds:
                        self.fit_bounds[f"{fun_name}_{parm_name}"] = self.bounds[fun_name][parm_name]
                        if self.bounds[fun_name][parm_name][0]==self.bounds[fun_name][parm_name][1]:
                            self.fixed_parms.append(f"{fun_name}_{parm_name}")

        self.irf = self.funs.pop("irf", None)


    
    def get_fun_parms(self, fun_name, fit_parms):
        fun_parms = []
        for name in self.parms[fun_name]:
            if name in self.set["common_parms"]:
                fun_parms.append(fit_parms[name])
            else:
                fun_parms.append(fit_parms[f"{fun_name}_{name}"])
        return fun_parms


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

        return self.opt_parms


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

    
#
    def get_goodness(self, chi2):
        return float(chi2 / np.sum(self.Z**2))



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
                if self.irf:
                    D[n, :] = convolve(self.x,
                                       fun(self.x, *self.get_fun_parms(fun_name, fit_parms)),
                                       lambda x: self.irf(x, *self.get_fun_parms("irf", fit_parms)
                                    ),
                    )
                else:
                    D[n, :] = fun(self.x, *self.get_fun_parms(fun_name, fit_parms))
            return D

        def lsq_DADS(Z, D):
            """Least-squares DADS; also returns Z @ D.T for the fast chi2."""
            ZDt = Z @ D.T
            G = D @ D.T
            if np.linalg.cond(G) > 1e12:
                G = G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G))
            return ZDt @ inv(G), ZDt

        def make_DADS(Z, D):
            if self.set["fixed_DADS"] is not None:
                return self.set["fixed_DADS"]
            return lsq_DADS(Z, D)[0]
            # Try new approach?:
            # return np.linalg.lstsq(D.T, Z.T, rcond=None)[0].T

        def calc_chi2(*p):
            def p_str():
                return ", ".join(f"{n}={v:.3g}" for n, v in zip(parm_names, p))

            D = make_D(*p)
            # if not np.isfinite(D).all():
            #     self.error_log.append(f"Non-finite D matrix for parameters: {p_str()}")
            #     return 1e3 * self.Zsq  # penalty

            if not np.isfinite(D).all(): # TODO: temporary test
                bad = ~np.isfinite(D)
                print("rows:", np.unique(np.where(bad)[0]))
                print("x:   ", self.x[np.unique(np.where(bad)[1])][:10])
                print("parms per fun:", {f: self.get_fun_parms(f, dict(zip(parm_names, p))) for f in self.funs})
                raise SystemExit

            DADS = make_DADS(self.Z, D)
            r = self.Z - DADS @ D
            if self.has_nans:
                r = r[self.mask]
            r = r.ravel()
            chi2 = r @ r

            if not np.isfinite(chi2):
                self.error_log.append(f"Non-finite chi2 for parameters: {p_str()}")
                return 1e3 * self.Zsq  # penalty

            return chi2

        # =============================================================================
        # create Minuit object and set fit settings
        p0 = tuple(fit_parms[name] for name in parm_names)
        m = Mi(calc_chi2, *p0, name=parm_names)
        if self.bounds:
            for name, bound in self.fit_bounds.items():
                m.limits[name] = bound
        for fp in self.fixed_parms:
            m.fixed[fp] = True
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
            if not np.isfinite(D).all():
                p_str = ", ".join(f"{n}={v:.3g}" for n, v in fit_parms.items())
                raise RuntimeError(
                    f"Fit ended at invalid parameters (non-finite D): {p_str}. "
                    f"{len(self.error_log)} bad evaluations; see error_log."
                )
            DADS = make_DADS(self.Z, D)

            goodness = self.get_goodness(calc_chi2(*m.values))
            self.goodnesses.append(goodness)
            self.chi2s.append(calc_chi2(*m.values))

            # User output:
            if self.callback:
                report = f"Iteration {iteration + 1}/{self.set['iterations']} finished with chi2={calc_chi2(*m.values):.3g}  (goodness: {goodness:.3g})"
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
        self.opt_parms = self.pack_parameters(fit_parms)
        self.residuum = self.Z_fit - self.Z
        self.m = m

        self.result = f"{m.params}\n\n{m.fmin}"

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
