"""
In-memory store for built volatility surfaces.
"""

import QuantLib as ql
from typing import Dict, Optional, List
from datetime import date as dt_date
from dataclasses import dataclass, field

from market_data.schemas import CapFloorVolData
from .schemas import VolType
from .vol_convention import CapFloorVolConvention


@dataclass
class VolSurfaceSet:
    """Container for a built volatility surface with metadata."""

    currency: str
    valuation_date: dt_date
    surface: ql.CapFloorTermVolSurface
    vol_type: VolType
    raw_data: CapFloorVolData
    convention: Optional[CapFloorVolConvention] = None
    smile_method: str = "bilinear"
    sabr_params: Optional[List[dict]] = None
    atm_curve: Optional[ql.CapFloorTermVolCurve] = None
    # SABR caplet data for direct StrippedOptionlet construction in pricing
    sabr_caplet_dates: Optional[List] = None  # ql.Date per caplet fixing
    sabr_caplet_strikes: Optional[List[float]] = None  # strike grid used
    sabr_forward_rates: Optional[List[float]] = None  # forward rate per caplet
    sabr_expiry_times: Optional[List[float]] = None  # T per caplet
    # Fingerprint of the curve set the surface was stripped on (None: built without curves).
    source_curve_fingerprint: Optional[str] = None
    build_sequence: int = field(default=0, repr=False)

    @property
    def surface_id(self) -> str:
        return (
            f"capfloor_{self.currency}_{self.valuation_date.isoformat()}_"
            f"{self.vol_type.value}_{self.smile_method}"
        )

    @property
    def num_expiries(self) -> int:
        return len(self.raw_data.expiries)

    @property
    def num_strikes(self) -> int:
        return len(self.raw_data.strikes)


class VolStore:
    """In-memory store for built volatility surfaces."""

    def __init__(self):
        self._surfaces: Dict[str, VolSurfaceSet] = {}
        self._active_surface_ids_by_currency: Dict[str, str] = {}
        self._build_sequence = 0

    def store_surface(self, surface_set: VolSurfaceSet):
        """Store a vol surface set."""
        self._build_sequence += 1
        surface_set.build_sequence = self._build_sequence
        self._surfaces[surface_set.surface_id] = surface_set
        self._active_surface_ids_by_currency[surface_set.currency] = surface_set.surface_id

    def get_surface(self, surface_id: str) -> Optional[VolSurfaceSet]:
        """Retrieve a vol surface set by its stable ID."""
        return self._surfaces.get(surface_id)

    def get_surface_by_currency(self, currency: str) -> Optional[VolSurfaceSet]:
        """Retrieve the latest active vol surface for a currency."""
        active_surface_id = self._active_surface_ids_by_currency.get(currency)
        if active_surface_id is not None:
            return self._surfaces.get(active_surface_id)

        currency_surfaces = [
            surface for surface in self._surfaces.values() if surface.currency == currency
        ]
        if not currency_surfaces:
            return None
        return max(currency_surfaces, key=lambda surface: surface.build_sequence)

    def get_active_surface_id(self, currency: str) -> Optional[str]:
        """Stable id of the surface a currency prices with by default, if any."""
        return self._active_surface_ids_by_currency.get(currency)

    def set_active_surface_id(self, currency: str, surface_id: str) -> None:
        """Select an already stored surface as the currency default."""
        if surface_id not in self._surfaces:
            raise KeyError(surface_id)
        self._active_surface_ids_by_currency[currency] = surface_id

    def get_available_surfaces(self) -> List[VolSurfaceSet]:
        """Get all available surfaces."""
        return sorted(
            self._surfaces.values(),
            key=lambda surface: (-surface.build_sequence, surface.currency, surface.surface_id),
        )

    def invalidate_currency(self, currency: str) -> None:
        """Remove every cap/floor surface of a currency whose curves were replaced.

        Cap/floor surfaces consume the curves during optionlet stripping, so keeping
        one after its curves changed reproduces the same class of silent mixing.
        """
        for surface_id in [
            surface_id
            for surface_id, surface in self._surfaces.items()
            if surface.currency == currency
        ]:
            del self._surfaces[surface_id]
        self._active_surface_ids_by_currency.pop(currency, None)

    def clear(self):
        """Clear all stored surfaces."""
        self._surfaces.clear()
        self._active_surface_ids_by_currency.clear()
        self._build_sequence = 0


# Global vol store instance
vol_store = VolStore()
