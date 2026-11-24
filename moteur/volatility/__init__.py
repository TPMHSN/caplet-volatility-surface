"""
Volatility module - vol surface construction and cap/floor pricing support.
"""

from . import schemas

__all__ = [
    "schemas",
    "VolSurfaceBuilder",
    "vol_store",
    "VolSurfaceSet",
    "VolStore",
]

try:
    from .vol_surface_builder import VolSurfaceBuilder
    from .vol_store import vol_store, VolSurfaceSet, VolStore
except Exception:  # pragma: no cover - allows importing schemas without QuantLib
    VolSurfaceBuilder = None
    vol_store = None
    VolSurfaceSet = None
    VolStore = None
