"""Exact-date selection of the futures and FRAs kept next to the first swap of a forward curve.

Decision D-19: a short-end instrument must not end after the first swap matures, so it never
overlaps a swap constraint on the same region. The boundary is a date, not a month count: it is the
earliest swap maturity of the curve (whatever the order of the quotes), and each future or FRA is
compared with it through its own end date. The builder and the quote-risk ladder share these
functions, so the ladder never reports a helper the curve does not contain. Equal dates are kept
(an FRA ending on the swap pillar stays; a duplicated pillar remains an explicit bootstrap error).
"""

from __future__ import annotations

import QuantLib as ql

from config import calendar_label_to_ql
from market_data.curve_quote_conventions import CurveQuoteConvention, get_curve_quote_convention
from market_data.schemas import ForwardMarketData, FuturesDataPoint, MarketDataPoint

from .helpers import date_to_ql, index_tenor_months, parse_fra_tenor, parse_tenor

_BUSINESS_DAY_CONVENTIONS = {
    "Following": ql.Following,
    "ModifiedFollowing": ql.ModifiedFollowing,
}


def _business_day_convention(convention: CurveQuoteConvention) -> int:
    try:
        return _BUSINESS_DAY_CONVENTIONS[convention.business_day_convention]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported curve business-day convention: {convention.business_day_convention}"
        ) from exc


def _spot_date(convention: CurveQuoteConvention, valuation_date: ql.Date) -> ql.Date:
    calendar = calendar_label_to_ql(convention.calendar)
    # QuantLib helpers adjust the evaluation date before counting settlement days.
    return calendar.advance(calendar.adjust(valuation_date), convention.settlement_days, ql.Days)


def swap_maturity_date(point: MarketDataPoint, valuation_date: ql.Date) -> ql.Date:
    """Maturity of the swap helper: spot advanced by the tenor on the convention's calendar."""
    convention = get_curve_quote_convention(point.quote_metadata.convention_id)
    return calendar_label_to_ql(convention.calendar).advance(
        _spot_date(convention, valuation_date),
        parse_tenor(point.tenor),
        _business_day_convention(convention),
        convention.end_of_month,
    )


def fra_end_date(point: MarketDataPoint, valuation_date: ql.Date) -> ql.Date:
    """Maturity of the FRA helper: start from spot, then one index tenor after it."""
    convention = get_curve_quote_convention(point.quote_metadata.convention_id)
    months_to_start, months_to_end = parse_fra_tenor(point.tenor)
    calendar = calendar_label_to_ql(convention.calendar)
    bdc = _business_day_convention(convention)
    start = calendar.advance(
        _spot_date(convention, valuation_date),
        ql.Period(months_to_start, ql.Months),
        bdc,
        convention.end_of_month,
    )
    return calendar.advance(
        start,
        ql.Period(months_to_end - months_to_start, ql.Months),
        bdc,
        convention.end_of_month,
    )


def future_end_date(point: FuturesDataPoint, index_tenor: str) -> ql.Date:
    """Maturity of the future helper: the IMM date advanced by one index tenor."""
    convention = get_curve_quote_convention(point.quote_metadata.convention_id)
    return calendar_label_to_ql(convention.calendar).advance(
        date_to_ql(point.imm_date),
        ql.Period(index_tenor_months(index_tenor), ql.Months),
        _business_day_convention(convention),
        convention.end_of_month,
    )


def short_end_cutoff_date(market_data: ForwardMarketData) -> ql.Date | None:
    """Earliest swap maturity of the curve, or ``None`` when it quotes no swap."""
    if not market_data.swaps:
        return None
    valuation_date = date_to_ql(market_data.valuation_date)
    return min(swap_maturity_date(point, valuation_date) for point in market_data.swaps)


def is_future_before_cutoff(
    market_data: ForwardMarketData,
    point: FuturesDataPoint,
    cutoff: ql.Date | None,
) -> bool:
    """Whether a future ends on or before the first swap maturity."""
    return cutoff is None or future_end_date(point, market_data.index_tenor) <= cutoff


def is_fra_before_cutoff(
    market_data: ForwardMarketData,
    point: MarketDataPoint,
    cutoff: ql.Date | None,
) -> bool:
    """Whether an FRA ends on or before the first swap maturity."""
    if cutoff is None:
        return True
    return fra_end_date(point, date_to_ql(market_data.valuation_date)) <= cutoff
