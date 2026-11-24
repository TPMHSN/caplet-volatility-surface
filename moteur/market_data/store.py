"""
In-memory market data store.
"""

from typing import TYPE_CHECKING, Dict, Optional
from datetime import date

from .schemas import (
    OISMarketData,
    ForwardMarketData,
    CapFloorVolData,
    SwaptionVolData,
    MarketDataStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .market_state import StagedMarketData


class MarketDataStore:
    """In-memory store for market data."""

    def __init__(self):
        self._ois_curves: Dict[str, OISMarketData] = {}
        self._forward_curves: Dict[str, ForwardMarketData] = {}
        self._capfloor_surfaces: Dict[str, CapFloorVolData] = {}
        self._swaption_vols: Dict[str, SwaptionVolData] = {}
        self._swaption_otm_vols: Dict[str, SwaptionVolData] = {}
        self._valuation_date: Optional[date] = None

    @staticmethod
    def _swaption_key(currency: str, index_tenor: str) -> str:
        return f"{currency}_{index_tenor}"

    def store_ois_curve(self, currency: str, data: OISMarketData):
        """Store OIS curve data."""
        key = currency
        self._ois_curves[key] = data
        if self._valuation_date is None:
            self._valuation_date = data.valuation_date

    def store_forward_curve(self, currency: str, index_tenor: str, data: ForwardMarketData):
        """Store forward curve data."""
        key = f"{currency}_{index_tenor}"
        self._forward_curves[key] = data
        if self._valuation_date is None:
            self._valuation_date = data.valuation_date

    def store_capfloor_surface(self, currency: str, data: CapFloorVolData):
        """Store cap/floor volatility surface (one per currency)."""
        self._capfloor_surfaces[currency] = data
        if self._valuation_date is None:
            self._valuation_date = data.valuation_date

    def store_swaption_vol(self, currency: str, data: SwaptionVolData):
        """Store swaption volatility data."""
        key = self._swaption_key(currency, data.index_tenor)
        self._swaption_vols[key] = data
        if self._valuation_date is None:
            self._valuation_date = data.valuation_date

    def get_ois_curve(self, currency: str) -> Optional[OISMarketData]:
        """Retrieve OIS curve data."""
        return self._ois_curves.get(currency)

    def get_forward_curve(self, currency: str, index_tenor: str) -> Optional[ForwardMarketData]:
        """Retrieve forward curve data."""
        key = f"{currency}_{index_tenor}"
        return self._forward_curves.get(key)

    def get_forward_curves(self, currency: str) -> Dict[str, ForwardMarketData]:
        """Retrieve every forwarding dataset of a currency, keyed by index tenor."""
        prefix = f"{currency}_"
        return {
            key[len(prefix) :]: data
            for key, data in self._forward_curves.items()
            if key.startswith(prefix)
        }

    def get_capfloor_surface(self, currency: str) -> Optional[CapFloorVolData]:
        """Retrieve cap/floor volatility surface by currency."""
        return self._capfloor_surfaces.get(currency)

    def get_swaption_vol(
        self, currency: str, index_tenor: Optional[str] = None
    ) -> Optional[SwaptionVolData]:
        """Retrieve swaption volatility data by exact currency/tenor."""
        if index_tenor is None:
            return None
        return self._swaption_vols.get(self._swaption_key(currency, index_tenor))

    def store_swaption_otm_vol(self, currency: str, data: SwaptionVolData):
        """Store swaption OTM volatility data."""
        key = self._swaption_key(currency, data.index_tenor)
        self._swaption_otm_vols[key] = data
        if self._valuation_date is None:
            self._valuation_date = data.valuation_date

    def get_swaption_otm_vol(
        self, currency: str, index_tenor: Optional[str] = None
    ) -> Optional[SwaptionVolData]:
        """Retrieve swaption OTM volatility data by exact currency/tenor."""
        if index_tenor is None:
            return None
        return self._swaption_otm_vols.get(self._swaption_key(currency, index_tenor))

    def get_status(self) -> MarketDataStatus:
        """Get status of all loaded market data."""
        return MarketDataStatus(
            ois_curves=list(self._ois_curves.keys()),
            forward_curves=list(self._forward_curves.keys()),
            cap_floor_surfaces=list(self._capfloor_surfaces.keys()),
            swaption_vols=list(self._swaption_vols.keys()),
            swaption_otm_vols=list(self._swaption_otm_vols.keys()),
            valuation_date=self._valuation_date,
        )

    def replace_all(self, staged: "StagedMarketData") -> None:
        """Replace the whole raw state in one copy-on-write transition.

        Callers stage and validate every source first, so a complete state is
        published or none is. Existing collections are swapped for fresh ones rather
        than mutated, which keeps any reference captured earlier intact.
        """
        self._ois_curves = dict(staged.ois)
        self._forward_curves = dict(staged.forward)
        self._capfloor_surfaces = dict(staged.capfloor)
        self._swaption_vols = dict(staged.swaption)
        self._swaption_otm_vols = dict(staged.swaption_otm)
        self._valuation_date = staged.resolved_valuation_date()

    def clear(self):
        """Clear all stored data, replacing the collections rather than mutating them."""
        self._ois_curves = {}
        self._forward_curves = {}
        self._capfloor_surfaces = {}
        self._swaption_vols = {}
        self._swaption_otm_vols = {}
        self._valuation_date = None


# Global market data store instance
market_data_store = MarketDataStore()
