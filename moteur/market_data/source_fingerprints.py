"""
Fingerprints of the raw market sources consumed by curve and cube builders.

These fingerprints answer one question only: *are the quotes currently loaded the
same economic inputs as the ones this derived object was built from?* They are
therefore computable at any time from the market data store, without replaying a
bootstrap or a calibration.

``source_label`` is deliberately excluded: renaming ``5 YR`` into ``5Y`` without
touching the instrument, the quote or the convention must not invalidate a curve.
The label stays available in the existing calibration diagnostics.
"""

from __future__ import annotations

from typing import Any

from .canonical_hash import LINEAGE_SCHEMA_VERSION, canonical_fingerprint
from .schemas import (
    ForwardMarketData,
    FuturesDataPoint,
    MarketDataPoint,
    OISMarketData,
    SwaptionVolData,
)

OIS_SOURCE_SCHEMA = "curve-source-ois-v2"
FORWARD_SOURCE_SCHEMA = "curve-source-forward-v3"
SWAPTION_VOL_SOURCE_SCHEMA = "swaption-vol-source-v1"


def _quote_payload(point: MarketDataPoint) -> dict[str, Any]:
    """Encode the economic identity of a quote, excluding its presentation label."""
    metadata = point.quote_metadata
    return {
        "tenor": point.tenor,
        "rate": float(point.rate),
        "instrument_type": metadata.instrument_type.value,
        "convention_id": metadata.convention_id,
    }


def _futures_payload(point: FuturesDataPoint) -> dict[str, Any]:
    metadata = point.quote_metadata
    return {
        "imm_date": point.imm_date,
        "rate_basis": point.rate_basis,
        "rate": float(point.rate),
        "instrument_type": metadata.instrument_type.value,
        "convention_id": metadata.convention_id,
    }


def ois_source_payload(data: OISMarketData) -> dict[str, Any]:
    """Canonical payload of the OIS quotes offered to a curve build."""
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "currency": data.currency,
        "valuation_date": data.valuation_date,
        "fixings": [_quote_payload(point) for point in data.fixings],
        "deposits": [_quote_payload(point) for point in data.deposits],
        "swaps": [_quote_payload(point) for point in data.swaps],
    }


def forward_source_payload(data: ForwardMarketData) -> dict[str, Any]:
    """Canonical payload of the forwarding quotes offered to a curve build."""
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "currency": data.currency,
        "valuation_date": data.valuation_date,
        "index_tenor": data.index_tenor,
        "fixings": [_quote_payload(point) for point in data.fixings],
        "deposits": [_quote_payload(point) for point in data.deposits],
        "fras": [_quote_payload(point) for point in data.fras],
        "futures": [_futures_payload(point) for point in data.futures],
        "swaps": [_quote_payload(point) for point in data.swaps],
    }


def swaption_vol_source_payload(data: SwaptionVolData) -> dict[str, Any]:
    """Canonical payload of a swaption ATM or OTM volatility grid."""
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "currency": data.currency,
        "index_tenor": data.index_tenor,
        "valuation_date": data.valuation_date,
        "data_points": [
            {
                "expiry": point.expiry,
                "tenor": point.tenor,
                "strike": None if point.strike is None else float(point.strike),
                "volatility": float(point.volatility),
                "vol_type": point.vol_type,
            }
            for point in data.data_points
        ],
    }


def ois_source_fingerprint(data: OISMarketData) -> str:
    """Fingerprint the OIS quotes consumed by a curve build."""
    return canonical_fingerprint(OIS_SOURCE_SCHEMA, ois_source_payload(data))


def forward_source_fingerprint(data: ForwardMarketData) -> str:
    """Fingerprint the forwarding quotes consumed by a curve build."""
    return canonical_fingerprint(FORWARD_SOURCE_SCHEMA, forward_source_payload(data))


def swaption_vol_source_fingerprint(data: SwaptionVolData) -> str:
    """Fingerprint a swaption ATM or OTM volatility grid."""
    return canonical_fingerprint(SWAPTION_VOL_SOURCE_SCHEMA, swaption_vol_source_payload(data))
