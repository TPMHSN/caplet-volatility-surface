"""
Cap/Floor volatility convention per currency.

Defines which index tenor is the natural quoting convention for each expiry:
  - Short expiries (<= cutoff) are quoted against the short index (e.g. Euribor 3M)
  - Long expiries (> cutoff) are quoted against the long index (e.g. Euribor 6M)
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class CapFloorVolConvention:
    """Market convention for cap/floor vol quoting."""

    currency: str
    cutoff_years: float = 2.0
    short_index_tenor: str = "3M"
    long_index_tenor: str = "6M"

    def get_quoting_tenor(self, expiry_years: float) -> str:
        """Return the natural quoting index tenor for a given expiry."""
        if expiry_years <= self.cutoff_years:
            return self.short_index_tenor
        return self.long_index_tenor

    def needs_conversion(self, expiry_years: float, target_tenor: str) -> bool:
        """Check if a conversion is needed for a given expiry and target tenor."""
        return self.get_quoting_tenor(expiry_years) != target_tenor


DEFAULT_CONVENTIONS: Dict[str, CapFloorVolConvention] = {
    "EUR": CapFloorVolConvention(
        currency="EUR",
        cutoff_years=2.0,
        short_index_tenor="3M",
        long_index_tenor="6M",
    ),
}


def get_vol_convention(currency: str) -> CapFloorVolConvention:
    """Get the vol convention for a currency, defaulting to EUR-style if unknown."""
    if currency in DEFAULT_CONVENTIONS:
        return DEFAULT_CONVENTIONS[currency]
    # Default: same structure as EUR
    return CapFloorVolConvention(currency=currency)
