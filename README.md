# fittoolkit

Fitting library: global fitting of kinetics / time-resolved spectroscopy data (e.g. transient absorption), including convolution with an instrument response function (IRF).

## Features

- Global fitting of 2D data sets (`x`, `y`, `Z`) with several fit functions at once
- Parameters shared between functions ("common" parameters), with bounds
- IRF convolution with its own parameter set
  - x-aware numerical convolution for **non-uniformly spaced** x-axes
  - analytical exponentially modified Gaussian (`emg`), parameterized by FWHM and numerically stable (`erfcx`)
- Minimization via iminuit (Minuit)

## Installation

Clone the repository, then run from the root folder (the one containing `pyproject.toml`):

```bash
pip install .        # regular install
pip install -e .     # editable install for development
```

Requires Python 3.9–3.13.

## Package structure

```
fittoolkit/
├── pyproject.toml
└── fittoolkit/
    ├── fitcore/    # Fit class and fit models
    └── utils/      # helper tools
```

## Usage

```python
import numpy as np
from fittoolkit.fitcore import GlobalFit  

fit = Fit(
    x=x,                  # e.g. time axis
    y=y,                  # e.g. wavelength axis
    Z=Z,                  # 2D data, shape (len(y), len(x))  # TODO: check
    p0=[...],             # initial parameter values
    funs=["emg(x, x0, A1, tau1, fwhm)"],
    settings={},          # optional
    name="Data",          # optional
    callback=None,        # optional, called during the fit
)
```

### Built-in functions for expressions

| Name  | Description                                                        |
|-------|--------------------------------------------------------------------|
| `emg` | Exponential decay convolved with a Gaussian IRF: `emg(x, x0, A, tau, fwhm)` |

Arguments are matched **by position**, so keep the order shown above.

## Development

Tests and linting run on GitHub Actions (Ubuntu and Windows, Python 3.9–3.13):

```bash
ruff check .
pytest
```

## License
?


