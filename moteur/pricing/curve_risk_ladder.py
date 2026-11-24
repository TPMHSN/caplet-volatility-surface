"""Shared quote-level curve risk point collection and ladder section assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date as dt_date
from typing import Optional

import QuantLib as ql

from config import get_currency_config, is_rfr_index
from curves.helpers import (
    date_to_ql,
    index_tenor_months,
    parse_fra_tenor,
    parse_tenor,
)
from curves.short_end_selection import (
    is_fra_before_cutoff,
    is_future_before_cutoff,
    short_end_cutoff_date,
)
from market_data.curve_quote_conventions import get_curve_quote_convention
from market_data.schemas import ForwardMarketData, FuturesDataPoint, MarketDataPoint, OISMarketData
from pricing.diagnostics import (
    CurveRiskCell,
    CurveRiskColumn,
    CurveRiskRow,
    CurveRiskSection,
)

FAMILY_ORDER: tuple[str, ...] = ("fixings", "cash_bills", "fras", "futures", "swaps")
FAMILY_LABELS: dict[str, str] = {
    "fixings": "Fixings/anchors",
    "cash_bills": "Cash/bills",
    "fras": "FRAs",
    "futures": "Futures",
    "swaps": "Swaps",
}


@dataclass(frozen=True)
class CurveRiskPoint:
    """Market quote used to construct one curve."""

    curve_key: str
    curve_label: str
    curve_role: str
    field: str
    index: int
    family: str
    point_key: str
    label: str
    tenor: Optional[str]
    maturity_date: Optional[dt_date]
    quote: float
    sort_key: tuple[float, str]


def canonical_discount_curve_key(currency: str, discount_curve_key: Optional[str]) -> str:
    """Normalize a discount curve selector to the internal curve key."""
    if discount_curve_key is None:
        return "OIS"
    key = discount_curve_key.upper().strip()
    if key in ("OIS", "ESTR", "SOFR", "SONIA") or is_rfr_index(currency, key):
        return "OIS"
    return key


def canonical_forward_curve_key(
    currency: str,
    index_tenor: str,
    forward_curve_key: Optional[str],
) -> str:
    """Normalize a forward curve selector to the internal curve key."""
    key = (forward_curve_key or index_tenor).upper().strip()
    if key in ("OIS", "ESTR", "SOFR", "SONIA") or is_rfr_index(currency, key):
        return "OIS"
    return key


def curve_display_label(currency: str, curve_key: str) -> str:
    """Return a user-facing curve label for EUR, USD and GBP ladders."""
    normalized = curve_key.upper().strip()
    if normalized == "OIS":
        overnight = {"EUR": "ESTR", "USD": "SOFR", "GBP": "SONIA"}.get(currency.upper(), "OIS")
        return f"{overnight}/OIS"
    if currency.upper() == "EUR":
        return f"EURIBOR {normalized}"
    return f"{currency.upper()} {normalized}"


def curve_role_for_key(curve_key: str, discount_key: str, forward_key: str) -> str:
    """Return whether a curve is a pricing role or a calibration primitive."""
    is_discount = curve_key == discount_key
    is_forward = curve_key == forward_key
    if is_discount and is_forward:
        return "discount_forward"
    if is_forward:
        return "forward"
    if is_discount:
        return "discount"
    return "calibration"


def _selected_curve_keys(discount_key: str, forward_key: str) -> list[str]:
    keys: list[str] = []
    for key in (forward_key, discount_key):
        if key not in keys:
            keys.append(key)
    if "OIS" not in keys and any(key != "OIS" for key in keys):
        keys.append("OIS")
    return keys


def _collect_selected_points(
    *,
    currency: str,
    selected_curve_keys: list[str],
    discount_key: str,
    forward_key: str,
    ois_data: Optional[OISMarketData],
    forward_market_data: dict[str, ForwardMarketData],
) -> dict[tuple[str, str, str], CurveRiskPoint]:
    points: dict[tuple[str, str, str], CurveRiskPoint] = {}
    for curve_key in selected_curve_keys:
        curve_label = curve_display_label(currency, curve_key)
        curve_role = curve_role_for_key(curve_key, discount_key, forward_key)
        if curve_key == "OIS":
            if ois_data is None:
                continue
            for point in _ois_points(ois_data, curve_key, curve_label, curve_role):
                points[(point.family, point.point_key, point.curve_key)] = point
            continue

        forward_data = forward_market_data_for_key(forward_market_data, curve_key)
        if forward_data is None:
            continue
        for point in _forward_points(forward_data, curve_key, curve_label, curve_role):
            points[(point.family, point.point_key, point.curve_key)] = point
    return points


def _ois_points(
    data: OISMarketData,
    curve_key: str,
    curve_label: str,
    curve_role: str,
) -> list[CurveRiskPoint]:
    points: list[CurveRiskPoint] = []
    for index, point in enumerate(data.deposits):
        if not is_deposit_helper_used(point):
            continue
        source_label = point.quote_metadata.source_label
        points.append(
            _market_point(
                data.valuation_date,
                curve_key,
                curve_label,
                curve_role,
                "deposits",
                index,
                "cash_bills",
                source_label,
                source_label,
                point.tenor,
                _tenor_maturity_date(data.currency, data.valuation_date, point.tenor),
                point.rate,
            )
        )
    for index, point in enumerate(data.swaps):
        points.append(
            _market_point(
                data.valuation_date,
                curve_key,
                curve_label,
                curve_role,
                "swaps",
                index,
                "swaps",
                point.tenor,
                point.tenor,
                point.tenor,
                _tenor_maturity_date(data.currency, data.valuation_date, point.tenor),
                point.rate,
            )
        )
    return points


def _forward_points(
    data: ForwardMarketData,
    curve_key: str,
    curve_label: str,
    curve_role: str,
) -> list[CurveRiskPoint]:
    points: list[CurveRiskPoint] = []
    for index, point in enumerate(data.deposits):
        if not is_deposit_helper_used(point):
            continue
        source_label = point.quote_metadata.source_label
        points.append(
            _market_point(
                data.valuation_date,
                curve_key,
                curve_label,
                curve_role,
                "deposits",
                index,
                "cash_bills",
                source_label,
                source_label,
                point.tenor,
                _tenor_maturity_date(data.currency, data.valuation_date, point.tenor),
                point.rate,
            )
        )

    if data.fras:
        for index, point in enumerate(data.fras):
            if not is_forward_fra_helper_used(data, point):
                continue
            normalized = point.tenor.upper().replace("X", "x")
            points.append(
                _market_point(
                    data.valuation_date,
                    curve_key,
                    curve_label,
                    curve_role,
                    "fras",
                    index,
                    "fras",
                    normalized,
                    normalized,
                    normalized,
                    _fra_maturity_date(data.currency, data.valuation_date, point.tenor),
                    point.rate,
                )
            )

    for index, point in enumerate(data.futures):
        if not is_forward_future_helper_used(data, point):
            continue
        point_key = point.imm_date.isoformat()
        source_label = point.quote_metadata.source_label
        points.append(
            _market_point(
                data.valuation_date,
                curve_key,
                curve_label,
                curve_role,
                "futures",
                index,
                "futures",
                point_key,
                source_label,
                None,
                _future_maturity_date(data.currency, point, data.index_tenor),
                point.rate,
            )
        )

    for index, point in enumerate(data.swaps):
        points.append(
            _market_point(
                data.valuation_date,
                curve_key,
                curve_label,
                curve_role,
                "swaps",
                index,
                "swaps",
                point.tenor,
                point.tenor,
                point.tenor,
                _tenor_maturity_date(data.currency, data.valuation_date, point.tenor),
                point.rate,
            )
        )
    return points


def with_effective_dates(
    points: Mapping[tuple[str, str, str], CurveRiskPoint],
    helper_dates: Mapping[tuple[str, str, int], dt_date],
    reference_date: dt_date,
) -> dict[tuple[str, str, str], CurveRiskPoint]:
    """Replace the estimated maturities by those of the helpers the curves were built with.

    ``helper_dates`` maps ``(curve_key, market-data field, index)`` to the helper's maturity
    date. A quote without a helper (excluded or observation) keeps its estimated date.
    """
    dated: dict[tuple[str, str, str], CurveRiskPoint] = {}
    for cell, point in points.items():
        maturity = helper_dates.get((point.curve_key, point.field, point.index))
        if maturity is None or maturity == point.maturity_date:
            dated[cell] = point
            continue
        dated[cell] = replace(
            point,
            maturity_date=maturity,
            sort_key=(_sort_year(reference_date, maturity, point.tenor, point.label), point.label),
        )
    return dated


def _market_point(
    valuation_date: dt_date,
    curve_key: str,
    curve_label: str,
    curve_role: str,
    field: str,
    index: int,
    family: str,
    point_key: str,
    label: str,
    tenor: Optional[str],
    maturity_date: Optional[dt_date],
    quote: float,
) -> CurveRiskPoint:
    return CurveRiskPoint(
        curve_key=curve_key,
        curve_label=curve_label,
        curve_role=curve_role,
        field=field,
        index=index,
        family=family,
        point_key=point_key,
        label=label,
        tenor=tenor,
        maturity_date=maturity_date,
        quote=quote,
        sort_key=(_sort_year(valuation_date, maturity_date, tenor, label), label),
    )


def _build_sections(
    *,
    columns: list[CurveRiskColumn],
    points_by_cell: dict[tuple[str, str, str], CurveRiskPoint],
    raw_values: dict[tuple[str, str, str], float],
    scaled_values: dict[tuple[str, str, str], float],
) -> list[CurveRiskSection]:
    row_points: dict[tuple[str, str], list[CurveRiskPoint]] = {}
    for (family, point_key, _curve_key), point in points_by_cell.items():
        row_points.setdefault((family, point_key), []).append(point)

    sections: list[CurveRiskSection] = []
    for family in FAMILY_ORDER:
        rows: list[CurveRiskRow] = []
        matching_keys = [key for key in row_points if key[0] == family]
        matching_keys.sort(key=lambda key: min(point.sort_key for point in row_points[key]))
        for _, point_key in matching_keys:
            points = row_points[(family, point_key)]
            representative = min(points, key=lambda point: point.sort_key)
            cells: list[CurveRiskCell] = []
            for column in columns:
                cell_key = (family, point_key, column.curve_key)
                point = points_by_cell.get(cell_key)
                raw_value = raw_values.get(cell_key, 0.0)
                scaled_value = scaled_values.get(cell_key, 0.0)
                cells.append(
                    CurveRiskCell(
                        curve_key=column.curve_key,
                        curve_label=column.curve_label,
                        used_in_bootstrap=point is not None,
                        quote=point.quote if point is not None else None,
                        raw_dv01=raw_value,
                        scaled_dv01=scaled_value,
                        delta=scaled_value,
                    )
                )
            rows.append(
                CurveRiskRow(
                    family=family,
                    family_label=FAMILY_LABELS[family],
                    point_key=point_key,
                    label=representative.label,
                    tenor=representative.tenor,
                    maturity_date=representative.maturity_date,
                    cells=cells,
                    raw_total_delta=sum(cell.raw_dv01 for cell in cells),
                    total_delta=sum(cell.delta for cell in cells),
                )
            )
        if rows:
            sections.append(
                CurveRiskSection(
                    family=family,
                    label=FAMILY_LABELS[family],
                    rows=rows,
                    raw_total_delta=sum(row.raw_total_delta for row in rows),
                    total_delta=sum(row.total_delta for row in rows),
                )
            )
    return sections


def _bump_ois_quote(
    data: OISMarketData,
    field: str,
    index: int,
    bump_size: float,
) -> OISMarketData:
    points = list(getattr(data, field))
    points[index] = points[index].model_copy(update={"rate": points[index].rate + bump_size})
    return data.model_copy(update={field: points})


def _bump_forward_quote(
    data: ForwardMarketData,
    field: str,
    index: int,
    bump_size: float,
) -> ForwardMarketData:
    points = list(getattr(data, field))
    points[index] = points[index].model_copy(update={"rate": points[index].rate + bump_size})
    return data.model_copy(update={field: points})


def forward_market_data_for_key(
    forward_market_data: dict[str, ForwardMarketData],
    key: str,
) -> Optional[ForwardMarketData]:
    return (
        forward_market_data.get(key)
        or forward_market_data.get(key.upper())
        or forward_market_data.get(key.lower())
    )


def is_deposit_helper_used(point: MarketDataPoint) -> bool:
    """A typed deposit is a helper unless its profile keeps it as an observation."""
    return get_curve_quote_convention(point.quote_metadata.convention_id).used_in_bootstrap


def is_forward_future_helper_used(
    market_data: ForwardMarketData,
    point: FuturesDataPoint,
) -> bool:
    return is_future_before_cutoff(market_data, point, short_end_cutoff_date(market_data))


def is_forward_fra_helper_used(
    market_data: ForwardMarketData,
    point: MarketDataPoint,
) -> bool:
    return is_fra_before_cutoff(market_data, point, short_end_cutoff_date(market_data))


def _tenor_maturity_date(currency: str, reference_date: dt_date, tenor: str) -> dt_date:
    calendar = get_currency_config(currency)["calendar"]
    maturity = calendar.advance(date_to_ql(reference_date), parse_tenor(tenor))
    return dt_date(maturity.year(), maturity.month(), maturity.dayOfMonth())


def _fra_maturity_date(currency: str, reference_date: dt_date, fra_tenor: str) -> dt_date:
    _, months_to_end = parse_fra_tenor(fra_tenor)
    calendar = get_currency_config(currency)["calendar"]
    maturity = calendar.advance(date_to_ql(reference_date), ql.Period(months_to_end, ql.Months))
    return dt_date(maturity.year(), maturity.month(), maturity.dayOfMonth())


def _future_maturity_date(currency: str, point: FuturesDataPoint, index_tenor: str) -> dt_date:
    calendar = get_currency_config(currency)["calendar"]
    maturity = calendar.advance(
        date_to_ql(point.imm_date),
        ql.Period(index_tenor_months(index_tenor), ql.Months),
    )
    return dt_date(maturity.year(), maturity.month(), maturity.dayOfMonth())


def _sort_year(
    reference_date: dt_date,
    maturity_date: Optional[dt_date],
    tenor: Optional[str],
    label: str,
) -> float:
    if maturity_date is not None:
        return max((maturity_date - reference_date).days / 365.25, 0.0)
    text = tenor or label
    try:
        period = parse_tenor(text)
    except Exception:
        return 1.0e9
    if period.units() == ql.Days:
        return period.length() / 365.25
    if period.units() == ql.Weeks:
        return period.length() * 7 / 365.25
    if period.units() == ql.Months:
        return period.length() / 12.0
    if period.units() == ql.Years:
        return float(period.length())
    return 1.0e9
