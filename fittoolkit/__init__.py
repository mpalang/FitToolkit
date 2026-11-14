"""
fittoolkit - Scientific fitting and optimization toolkit
"""

# --- Core fitting API ---
from .fitcore.global_fit import GlobalFit

# --- Version ---
__version__ = "0.1.0"

# --- Public API definition ---
__all__ = [
    "GlobalFit",
]
