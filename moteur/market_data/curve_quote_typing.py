"""Single source of the canonical convention of a curve calibration quote (chantier CURVCONV, L3).

Every curve quote carries a convention identifier. This module is the only place that maps an input
(currency, curve, instrument type, tenor) to that identifier, whatever the entry point: the Excel
workbook loader, the CSV and manual entry paths, and the offline migration of stored snapshots.
The mapping is an explicit table: an input that has no row is rejected, never given a default.

Curve keys are ``OIS`` (the overnight curve of the currency) or the EURIBOR index tenor of a
projection curve (``1M``, ``3M``, ``6M``).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import QuantLib as ql

from curves.helpers import parse_tenor, tenor_to_years

from .curve_quote_conventions import (
    EUR_ESTR_DEPOSIT,
    EUR_ESTR_OIS,
    EUR_EURIBOR_1M_DEPOSIT,
    EUR_EURIBOR_1M_FIXING,
    EUR_EURIBOR_1M_IRS,
    EUR_EURIBOR_1M_SHORT_IRS,
    EUR_EURIBOR_3M_DEPOSIT,
    EUR_EURIBOR_3M_IRS,
    EUR_EURIBOR_3M_SERIAL_FUTURE,
    EUR_EURIBOR_6M_DEPOSIT,
    EUR_EURIBOR_6M_FRA,
    EUR_EURIBOR_6M_IRS,
    GBP_SONIA_DEPOSIT,
    GBP_SONIA_OIS_ACT365_ANNUAL,
    USD_SOFR_OIS_ACT360_ANNUAL,
    CurveQuoteConvention,
)
from .schemas import (
    CurveInstrumentType,
    CurveQuoteMetadata,
    ForwardMarketData,
    MarketDataPoint,
    OISMarketData,
)

OVERNIGHT_ANCHOR_TENOR = "1D"
_ONE_WEEK = ql.Period(1, ql.Weeks)
_SHORT_IRS_LIMIT_YEARS = 2.0


@dataclass(frozen=True)
class _Rule:
    """How the table places a quote: the profiles it may return and a readable tenor condition."""

    profiles: tuple[CurveQuoteConvention, ...]
    tenor_rule: str
    resolve: Callable[[str | None], CurveQuoteConvention]

    def __call__(self, tenor: str | None) -> CurveQuoteConvention:
        return self.resolve(tenor)


def _fixed(convention: CurveQuoteConvention, tenor_required: str | None = None) -> _Rule:
    def resolve(tenor: str | None) -> CurveQuoteConvention:
        if tenor_required is not None and tenor != tenor_required:
            raise ValueError(
                f"{convention.convention_id} quotes only the {tenor_required} tenor, got {tenor}"
            )
        return convention

    note = f"only the {tenor_required} tenor" if tenor_required else "any tenor or date"
    return _Rule((convention,), note, resolve)


def _overnight_index_swap(convention: CurveQuoteConvention) -> _Rule:
    def resolve(tenor: str | None) -> CurveQuoteConvention:
        if tenor is None or parse_tenor(tenor) < _ONE_WEEK:
            raise ValueError(
                f"OIS quote {tenor} is shorter than one week: an overnight rate is not a quoted "
                "OIS swap, type it as a deposit anchor"
            )
        return convention

    return _Rule((convention,), "one week or longer", resolve)


def _resolve_euribor_1m_swap(tenor: str | None) -> CurveQuoteConvention:
    if tenor is None:
        raise ValueError("A EURIBOR 1M swap quote needs its tenor")
    if tenor_to_years(tenor) < _SHORT_IRS_LIMIT_YEARS:
        return EUR_EURIBOR_1M_SHORT_IRS
    return EUR_EURIBOR_1M_IRS


_euribor_1m_swap = _Rule(
    (EUR_EURIBOR_1M_SHORT_IRS, EUR_EURIBOR_1M_IRS),
    "the short profile below 2 years, the long profile from 2 years",
    _resolve_euribor_1m_swap,
)


# Identifiers of the first profile generation, kept only so that the offline migration can rewrite
# snapshots written with them. They are not resolvable at run time.
RETIRED_CONVENTION_IDS: dict[str, str] = {
    "EUR-ESTR-DEPOSIT-V1": "EUR-ESTR-DEPOSIT-V2",
    "EUR-ESTR-OIS-V1": "EUR-ESTR-OIS-V2",
    "USD-SOFR-OIS-ACT360-ANNUAL-V1": "USD-SOFR-OIS-ACT360-ANNUAL-V2",
    "USD-SOFR-OIS-ACT360-QUARTERLY-V1": "USD-SOFR-OIS-ACT360-ANNUAL-V2",
    "GBP-SONIA-OIS-ACT365-ANNUAL-V1": "GBP-SONIA-OIS-ACT365-ANNUAL-V2",
    "EUR-EURIBOR-1M-FIXING-V1": "EUR-EURIBOR-1M-FIXING-V2",
    "EUR-EURIBOR-1M-DEPOSIT-V1": "EUR-EURIBOR-1M-DEPOSIT-V2",
    "EUR-EURIBOR-1M-SHORT-IRS-V1": "EUR-EURIBOR-1M-SHORT-IRS-V2",
    "EUR-EURIBOR-1M-IRS-V1": "EUR-EURIBOR-1M-IRS-V2",
    "EUR-EURIBOR-3M-DEPOSIT-V1": "EUR-EURIBOR-3M-DEPOSIT-V2",
    "EUR-EURIBOR-3M-SERIAL-FUTURE-V1": "EUR-EURIBOR-3M-SERIAL-FUTURE-V2",
    "EUR-EURIBOR-3M-IRS-V1": "EUR-EURIBOR-3M-IRS-V2",
    "EUR-EURIBOR-6M-DEPOSIT-V1": "EUR-EURIBOR-6M-DEPOSIT-V2",
    "EUR-EURIBOR-6M-FRA-V1": "EUR-EURIBOR-6M-FRA-V2",
    "EUR-EURIBOR-6M-IRS-V1": "EUR-EURIBOR-6M-IRS-V2",
}

_RULES: dict[tuple[str, str, CurveInstrumentType], _Rule] = {
    ("EUR", "OIS", CurveInstrumentType.DEPOSIT): _fixed(EUR_ESTR_DEPOSIT, OVERNIGHT_ANCHOR_TENOR),
    ("EUR", "OIS", CurveInstrumentType.OIS_SWAP): _overnight_index_swap(EUR_ESTR_OIS),
    ("USD", "OIS", CurveInstrumentType.OIS_SWAP): _overnight_index_swap(USD_SOFR_OIS_ACT360_ANNUAL),
    ("GBP", "OIS", CurveInstrumentType.DEPOSIT): _fixed(GBP_SONIA_DEPOSIT, OVERNIGHT_ANCHOR_TENOR),
    ("GBP", "OIS", CurveInstrumentType.OIS_SWAP): _overnight_index_swap(
        GBP_SONIA_OIS_ACT365_ANNUAL
    ),
    ("EUR", "1M", CurveInstrumentType.FIXING): _fixed(EUR_EURIBOR_1M_FIXING, "1M"),
    ("EUR", "1M", CurveInstrumentType.DEPOSIT): _fixed(EUR_EURIBOR_1M_DEPOSIT, "1M"),
    ("EUR", "1M", CurveInstrumentType.IRS_SWAP): _euribor_1m_swap,
    ("EUR", "3M", CurveInstrumentType.DEPOSIT): _fixed(EUR_EURIBOR_3M_DEPOSIT, "3M"),
    ("EUR", "3M", CurveInstrumentType.FUTURE): _fixed(EUR_EURIBOR_3M_SERIAL_FUTURE),
    ("EUR", "3M", CurveInstrumentType.IRS_SWAP): _fixed(EUR_EURIBOR_3M_IRS),
    ("EUR", "6M", CurveInstrumentType.DEPOSIT): _fixed(EUR_EURIBOR_6M_DEPOSIT, "6M"),
    ("EUR", "6M", CurveInstrumentType.FRA): _fixed(EUR_EURIBOR_6M_FRA),
    ("EUR", "6M", CurveInstrumentType.IRS_SWAP): _fixed(EUR_EURIBOR_6M_IRS),
}


_CALENDAR_TENOR = re.compile(r"^\d+[DWMY]$")
_FRA_TENOR = re.compile(r"^\d+x\d+$")


def canonical_tenor(instrument_type: CurveInstrumentType, tenor: str | None) -> str | None:
    """Return the tenor in its stored spelling, or reject one that is not a tenor.

    A calendar tenor is ``<n><D|W|M|Y>`` in upper case; a FRA is ``<start>x<end>`` in lower case.
    A future has no tenor. Anything else (empty, ``NAN``, ``abc``) is an error, not a quote.
    """
    if instrument_type == CurveInstrumentType.FUTURE or tenor is None:
        return None if instrument_type == CurveInstrumentType.FUTURE else tenor
    text = tenor.strip()
    if instrument_type == CurveInstrumentType.FRA:
        text = text.lower()
        if not _FRA_TENOR.match(text):
            raise ValueError(f"FRA tenor '{tenor}' must look like 1x7 (start x end, in months)")
        return text
    text = text.upper()
    if not _CALENDAR_TENOR.match(text):
        raise ValueError(f"Tenor '{tenor}' must look like 1W, 3M or 10Y")
    return text


def resolve_curve_quote_convention(
    currency: str,
    curve_key: str,
    instrument_type: CurveInstrumentType,
    tenor: str | None = None,
) -> CurveQuoteConvention:
    """Return the canonical profile of a quote, or reject it when the table has no row for it."""
    key = (currency.upper(), curve_key.upper(), CurveInstrumentType(instrument_type))
    rule = _RULES.get(key)
    if rule is None:
        raise ValueError(
            f"No canonical convention for a {key[2].value} quote on the {key[0]} {key[1]} curve"
        )
    return rule(canonical_tenor(key[2], tenor))


def describe_curve_conventions(currency: str, curve_key: str) -> list[dict[str, str]]:
    """List the profiles the table admits on one curve, for entry screens.

    One entry per (instrument type, profile), with the tenor condition under which the table
    assigns it. Raises ``ValueError`` when the table has no row at all for the curve.
    """
    options = [
        {
            "instrument_type": instrument_type.value,
            "convention_id": profile.convention_id,
            "tenor_rule": rule.tenor_rule,
        }
        for (rule_currency, rule_curve, instrument_type), rule in _RULES.items()
        if rule_currency == currency.upper() and rule_curve == curve_key.upper()
        for profile in rule.profiles
    ]
    if not options:
        raise ValueError(f"No canonical convention is defined for the {currency} {curve_key} curve")
    return options


def curve_quote_metadata(
    source_label: str, instrument_type: CurveInstrumentType, convention: CurveQuoteConvention
) -> CurveQuoteMetadata:
    return CurveQuoteMetadata(
        source_label=source_label,
        instrument_type=instrument_type,
        convention_id=convention.convention_id,
    )


def typed_point(
    currency: str,
    curve_key: str,
    instrument_type: CurveInstrumentType,
    *,
    tenor: str,
    rate: float,
    source_label: str,
) -> MarketDataPoint:
    """Build a typed tenor quote whose convention comes from the canonical table."""
    tenor = canonical_tenor(instrument_type, tenor) or tenor
    convention = resolve_curve_quote_convention(currency, curve_key, instrument_type, tenor)
    return MarketDataPoint(
        tenor=tenor,
        rate=rate,
        quote_metadata=curve_quote_metadata(source_label or tenor, instrument_type, convention),
    )


def verify_curve_market_data_typing(
    data: OISMarketData | ForwardMarketData, *, reject_duplicates: bool = False
) -> None:
    """Check that every quote carries the profile the canonical table assigns to it.

    The schema only guarantees that a convention id is present. A hand-typed quote (manual entry, a
    stored snapshot) could still carry a valid id of the wrong profile, such as the short 1M swap
    profile on a 5Y swap. Each point must also sit in the field of its instrument type. With
    ``reject_duplicates`` (entry paths) a repeated tenor or date in one field is an error; the
    builder leaves duplicates to the bootstrap's own pillar check. Raises ``ValueError`` naming
    the offending quote.
    """
    curve_key = "OIS" if isinstance(data, OISMarketData) else data.index_tenor
    swap_type = (
        CurveInstrumentType.OIS_SWAP
        if isinstance(data, OISMarketData)
        else CurveInstrumentType.IRS_SWAP
    )
    field_types = {
        "fixings": CurveInstrumentType.FIXING,
        "deposits": CurveInstrumentType.DEPOSIT,
        "fras": CurveInstrumentType.FRA,
        "swaps": swap_type,
        "futures": CurveInstrumentType.FUTURE,
    }
    for field_name, expected_type in field_types.items():
        seen: set[str] = set()
        for point in getattr(data, field_name, []):
            metadata = point.quote_metadata
            tenor = getattr(point, "tenor", None)
            if metadata.instrument_type != expected_type:
                raise ValueError(
                    f"Quote {metadata.source_label} is typed {metadata.instrument_type.value} "
                    f"but sits in {field_name} of the {data.currency} {curve_key} curve "
                    f"(expected {expected_type.value})"
                )
            key = tenor if tenor is not None else str(getattr(point, "imm_date", ""))
            if reject_duplicates and key.upper() in seen:
                raise ValueError(
                    f"{field_name} of the {data.currency} {curve_key} curve quote {key} twice"
                )
            seen.add(key.upper())
            expected = resolve_curve_quote_convention(
                data.currency, curve_key, metadata.instrument_type, tenor
            )
            if metadata.convention_id != expected.convention_id:
                hint = (
                    " (retired identifier: migrate the snapshot with "
                    "scripts/migrate_market_snapshots.py)"
                    if metadata.convention_id in RETIRED_CONVENTION_IDS
                    else ""
                )
                raise ValueError(
                    f"Quote {metadata.source_label} ({field_name}, {tenor or 'future'}) on the "
                    f"{data.currency} {curve_key} curve carries {metadata.convention_id}, "
                    f"expected {expected.convention_id}{hint}"
                )


def _metadata_payload(
    point: dict[str, Any],
    currency: str,
    curve_key: str,
    instrument_type: CurveInstrumentType,
    tenor: str | None,
    source_label: str,
) -> None:
    """Attach the canonical metadata to one raw JSON point, or verify the one it already carries."""
    convention = resolve_curve_quote_convention(currency, curve_key, instrument_type, tenor)
    existing = point.get("quote_metadata")
    if existing is not None and existing.get("convention_id") == "USD-SOFR-OIS-ACT360-QUARTERLY-V1":
        raise ValueError(
            "USD-SOFR-OIS-ACT360-QUARTERLY-V1 -> USD-SOFR-OIS-ACT360-ANNUAL-V2 "
            "changes instrument economics; source correction is required, migration refused"
        )
    if existing is not None and existing.get("convention_id") in RETIRED_CONVENTION_IDS:
        existing = {**existing, "convention_id": RETIRED_CONVENTION_IDS[existing["convention_id"]]}
        point["quote_metadata"] = existing
    if existing is None:
        point["quote_metadata"] = {
            "source_label": source_label,
            "instrument_type": instrument_type.value,
            "convention_id": convention.convention_id,
        }
    elif (
        existing.get("convention_id") != convention.convention_id
        or existing.get("instrument_type") != instrument_type.value
    ):
        raise ValueError(
            f"{currency} {curve_key} quote {source_label} is typed "
            f"{existing.get('instrument_type')}/{existing.get('convention_id')}, the canonical "
            f"table gives {instrument_type.value}/{convention.convention_id}"
        )


def migrate_ois_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Type a stored OIS quote set that predates mandatory conventions (raw JSON in, raw JSON out).

    A deposit of one day is the overnight anchor; a deposit of one week or more was an OIS quote
    that the legacy loader filed as a deposit and becomes an OIS swap. Anything else is rejected.
    Points already typed are checked against the canonical table and kept. The result validates as
    ``OISMarketData``.
    """
    result = copy.deepcopy(dict(payload))
    currency = result["currency"]
    if result.get("fixings"):
        raise ValueError(f"{currency} OIS fixings have no canonical convention")
    anchors: list[dict[str, Any]] = []
    swaps: list[dict[str, Any]] = []
    for point in result.get("deposits", []):
        if point["tenor"] == OVERNIGHT_ANCHOR_TENOR:
            kind, bucket = CurveInstrumentType.DEPOSIT, anchors
        else:
            kind, bucket = CurveInstrumentType.OIS_SWAP, swaps
        _metadata_payload(point, currency, "OIS", kind, point["tenor"], point["tenor"])
        bucket.append(point)
    for point in result.get("swaps", []):
        if point["tenor"] == OVERNIGHT_ANCHOR_TENOR:
            # A one-day OIS was never a market instrument (E-15): it is the overnight anchor.
            label = (point.get("quote_metadata") or {}).get("source_label", point["tenor"])
            point["quote_metadata"] = None
            _metadata_payload(
                point, currency, "OIS", CurveInstrumentType.DEPOSIT, point["tenor"], label
            )
            anchors.append(point)
            continue
        _metadata_payload(
            point, currency, "OIS", CurveInstrumentType.OIS_SWAP, point["tenor"], point["tenor"]
        )
        swaps.append(point)
    result.update({"fixings": [], "deposits": anchors, "swaps": swaps})
    OISMarketData.model_validate(result)
    return result


def migrate_forward_payload(
    payload: Mapping[str, Any], *, drop_futures_fra_mirror: bool = False
) -> dict[str, Any]:
    """Type a stored projection-curve quote set that predates mandatory conventions.

    ``drop_futures_fra_mirror`` removes the FRA copies of a futures strip that the legacy loader
    stored next to it (the bootstrap ignored them), only on EUR 3M and with explicit caller
    authorization. Rate equality alone cannot prove a mirror: the caller must confirm the legacy
    provenance before selecting this destructive option. Without it every FRA is kept.
    The result validates as ``ForwardMarketData``.
    """
    result = copy.deepcopy(dict(payload))
    currency, curve_key = result["currency"], result["index_tenor"]
    if (
        drop_futures_fra_mirror
        and currency == "EUR"
        and curve_key == "3M"
        and result.get("futures")
    ):
        future_rates = [future["rate"] for future in result["futures"]]
        result["fras"] = [
            fra
            for fra in result.get("fras", [])
            if not any(abs(fra["rate"] - rate) <= 1e-12 for rate in future_rates)
        ]
    for point in result.get("fixings", []):
        _metadata_payload(
            point, currency, curve_key, CurveInstrumentType.FIXING, point["tenor"], point["tenor"]
        )
    for point in result.get("deposits", []):
        _metadata_payload(
            point, currency, curve_key, CurveInstrumentType.DEPOSIT, point["tenor"], point["tenor"]
        )
    for point in result.get("fras", []):
        _metadata_payload(
            point, currency, curve_key, CurveInstrumentType.FRA, point["tenor"], point["tenor"]
        )
    for future in result.get("futures", []):
        _metadata_payload(
            future, currency, curve_key, CurveInstrumentType.FUTURE, None, future["imm_date"]
        )
    for point in result.get("swaps", []):
        _metadata_payload(
            point, currency, curve_key, CurveInstrumentType.IRS_SWAP, point["tenor"], point["tenor"]
        )
    ForwardMarketData.model_validate(result)
    return result
