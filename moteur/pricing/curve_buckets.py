"""
Shared helpers for curve-delta bucketing.
"""

from collections.abc import Iterable
from typing import Final

import QuantLib as ql

DELTA_TIME_BUCKETS: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
    (5.0, 7.0),
    (7.0, 10.0),
    (10.0, 12.0),
    (12.0, 15.0),
    (15.0, 20.0),
    (20.0, 25.0),
    (25.0, 30.0),
    (30.0, 50.0),
)

QUOTE_MATURITY_BUCKETS: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0),
    (9.0, 10.0),
    (10.0, 11.0),
    (11.0, 12.0),
    (12.0, 15.0),
    (15.0, 20.0),
    (20.0, 25.0),
    (25.0, 30.0),
    (30.0, 40.0),
    (40.0, 50.0),
)

CURVE_SAMPLE_YEARS: Final[tuple[float, ...]] = (
    0.083,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
)


def bucket_label(start_year: float, end_year: float) -> str:
    """Format a maturity bucket label for diagnostics and UI."""
    return f"{int(start_year)}Y-{int(end_year)}Y"


def bucket_bounds_for_year(
    maturity_years: float,
    buckets: Iterable[tuple[float, float]] = QUOTE_MATURITY_BUCKETS,
) -> tuple[float, float] | None:
    """Return the configured bucket containing a relative quote maturity."""
    configured_buckets = tuple(buckets)
    for index, (start_year, end_year) in enumerate(configured_buckets):
        is_last_bucket_endpoint = (
            index == len(configured_buckets) - 1 and maturity_years == end_year
        )
        if start_year <= maturity_years < end_year or is_last_bucket_endpoint:
            return start_year, end_year
    return None


def make_bucketed_curve_handle(
    base_curve: ql.YieldTermStructureHandle,
    start_year: float,
    end_year: float,
    bump: float,
    sample_years: Iterable[float] = CURVE_SAMPLE_YEARS,
) -> ql.YieldTermStructureHandle:
    """Apply a zero-rate spread on one maturity bucket over the original curve.

    Internal boundaries belong to the next bucket. The bucket ending at the final
    sample owns that endpoint too, so the spread basis still sums to a parallel
    shift at the long end. QuantLib carries this terminal spread flat beyond the
    final sample; the underlying calibrated curve is never rebuilt here.
    """
    sample_years = tuple(sample_years)
    final_sample_year = max(sample_years)
    reference_date = base_curve.referenceDate()
    dates = ql.DateVector()
    spreads = ql.QuoteHandleVector()

    for year in sample_years:
        bucket_date = reference_date + ql.Period(int(year * 365 + 0.5), ql.Days)
        is_final_endpoint = year == end_year == final_sample_year
        spread_value = bump if start_year <= year < end_year or is_final_endpoint else 0.0
        dates.append(bucket_date)
        spreads.append(ql.QuoteHandle(ql.SimpleQuote(spread_value)))

    bumped_curve = ql.PiecewiseZeroSpreadedTermStructure(base_curve, spreads, dates)
    bumped_curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(bumped_curve)
