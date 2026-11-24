"""
QuantLib helper utilities for curve construction.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date as dt_date
from functools import lru_cache, wraps
from threading import RLock
from typing import Literal, ParamSpec, TypeVar, get_args

import QuantLib as ql

_QL_EVALUATION_DATE_LOCK = RLock()
_P = ParamSpec("_P")
_R = TypeVar("_R")


def adjust_to_business_day(d: dt_date, currency: str) -> dt_date:
    """Adjust date to next modified-following business day if not already one."""
    from config import get_currency_config

    calendar = get_currency_config(currency)["calendar"]
    ql_d = ql.Date(d.day, d.month, d.year)
    if not calendar.isBusinessDay(ql_d):
        return ql_to_date(calendar.adjust(ql_d, ql.ModifiedFollowing))
    return d


def date_to_ql(d: dt_date) -> ql.Date:
    """Convert Python date to QuantLib Date."""
    return ql.Date(d.day, d.month, d.year)


def ql_to_date(ql_date: ql.Date) -> dt_date:
    """Convert QuantLib Date to Python date."""
    return dt_date(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())


@contextmanager
def ql_state_lock() -> Iterator[None]:
    """Serialize access to process-global QuantLib settings and index histories."""
    with _QL_EVALUATION_DATE_LOCK:
        yield


def serialized_ql_state(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Serialize a callable's synchronous access to process-global QuantLib state."""

    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with ql_state_lock():
            settings = ql.Settings.instance()
            original_date = settings.evaluationDate
            try:
                return function(*args, **kwargs)
            finally:
                settings.evaluationDate = original_date

    return wrapper


@contextmanager
def ql_evaluation_date(val_date: dt_date) -> Iterator[None]:
    """
    Context manager to temporarily set QuantLib evaluation date.

    Useful for separating curve date (market data reference) from valuation date
    (pricing/discounting date). Automatically restores the previous evaluation date.

    Args:
        val_date: Valuation date to set

    Example:
        with ql_evaluation_date(pricing_date):
            npv = swap.NPV()  # Evaluated as of pricing_date
    """
    with ql_state_lock():
        settings = ql.Settings.instance()
        old = settings.evaluationDate
        settings.evaluationDate = date_to_ql(val_date)
        try:
            yield
        finally:
            settings.evaluationDate = old


def parse_tenor(tenor_str: str) -> ql.Period:
    """
    Parse tenor string to QuantLib Period.

    Args:
        tenor_str: Tenor string (e.g., 'ON', 'TN', '1D', '1W', '1M', '3M', '1Y', '10Y')

    Returns:
        QuantLib Period object
    """
    tenor_str = tenor_str.upper().strip()

    # Handle special cases
    special_tenors = {
        "ON": ql.Period(1, ql.Days),  # Overnight
        "TN": ql.Period(2, ql.Days),  # Tomorrow-Next
    }

    if tenor_str in special_tenors:
        return special_tenors[tenor_str]

    # Extract number and unit
    number = ""
    unit = ""

    for char in tenor_str:
        if char.isdigit():
            number += char
        else:
            unit += char

    if not number or not unit:
        raise ValueError(
            f"Invalid tenor format: {tenor_str}. Use formats like '1D', '1W', '3M', '1Y', or special tenors 'ON', 'TN'"
        )

    num = int(number)

    # Map unit to QuantLib TimeUnit
    unit_map = {
        "D": ql.Days,
        "W": ql.Weeks,
        "M": ql.Months,
        "Y": ql.Years,
    }

    if unit not in unit_map:
        raise ValueError(f"Unknown tenor unit: {unit}. Use D, W, M, or Y")

    return ql.Period(num, unit_map[unit])


def tenor_to_years(tenor: str) -> float:
    """Convert a tenor string to a year fraction.

    Uses ``parse_tenor`` for robust handling of D/W/M/Y tenors.
    """
    period = parse_tenor(tenor)
    if period.units() == ql.Days:
        return period.length() / 365.0
    if period.units() == ql.Weeks:
        return period.length() / 52.0
    if period.units() == ql.Months:
        return period.length() / 12.0
    if period.units() == ql.Years:
        return float(period.length())
    raise ValueError(f"Unsupported tenor unit in: {tenor}")


def parse_fra_tenor(fra_str: str) -> tuple[int, int]:
    """
    Parse FRA tenor string (e.g., '3x6', '6x9').

    Args:
        fra_str: FRA tenor string

    Returns:
        Tuple of (months_to_start, months_to_end) as integers

    Note:
        FraRateHelper expects integer months, not ql.Period objects.
    """
    fra_str = fra_str.lower().strip()
    parts = fra_str.split("x")

    if len(parts) != 2:
        raise ValueError(f"Invalid FRA tenor format: {fra_str}. Expected format like '3x6'")

    start = int(parts[0])
    end = int(parts[1])

    return start, end


def index_tenor_months(index_tenor: str) -> int:
    """Return the length in months of a EURIBOR index tenor, or reject an unknown one."""
    months = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}.get(index_tenor.upper())
    if months is None:
        raise ValueError(f"Unsupported index tenor: {index_tenor}")
    return months


def create_quote_handle(rate: float) -> ql.QuoteHandle:
    """Create a QuoteHandle from a rate value."""
    return ql.QuoteHandle(ql.SimpleQuote(rate))


def create_relinkable_handle() -> ql.RelinkableYieldTermStructureHandle:
    """Create a relinkable yield term structure handle."""
    return ql.RelinkableYieldTermStructureHandle()


@lru_cache(maxsize=32)
def _daily_date_grid(first_serial: int, last_serial: int) -> tuple[ql.Date, ...]:
    """Return an immutable inclusive daily date grid for the serial-number range."""
    if first_serial > last_serial:
        raise ValueError("Curve end date must not precede its start date")
    return tuple(ql.Date(serial) for serial in range(first_serial, last_serial + 1))


def freeze_curve(
    bootstrapped_curve: ql.YieldTermStructure,
    day_counter: ql.DayCounter | None = None,
) -> ql.DiscountCurve:
    """Freeze a bootstrapped curve into a static DiscountCurve (no re-bootstrap).

    Samples the curve at dense intervals (daily) to preserve the interpolation
    behavior of the original bootstrapped curve.
    """
    if day_counter is None:
        day_counter = ql.Actual365Fixed()

    nodes = bootstrapped_curve.nodes()
    first_date = nodes[0][0]
    last_date = nodes[-1][0]

    # Dense sampling: daily intervals to preserve interpolation behavior.
    dates = _daily_date_grid(first_date.serialNumber(), last_date.serialNumber())
    discount = bootstrapped_curve.discount
    dfs = [discount(sample_date) for sample_date in dates]

    frozen = ql.DiscountCurve(dates, dfs, day_counter)
    frozen.enableExtrapolation()
    return frozen


class CurveInterpolation:
    """Enumeration of available curve interpolation methods."""

    SMOOTH_FORWARD = "SmoothForward"
    LINEAR_FORWARD = "LinearForward"
    NATURAL_LOG_CUBIC = "NaturalLogCubic"


CurveInterpolationMethod = Literal["LinearForward", "SmoothForward", "NaturalLogCubic"]


DEFAULT_BOOTSTRAP_ACCURACY = 1.0e-12
LEGACY_BOOTSTRAP_ACCURACY = 1.0e-4


def resolve_interpolation(interpolation: str) -> str:
    """Return the interpolation method, rejecting a name the builder does not implement."""
    supported = get_args(CurveInterpolationMethod)
    if interpolation not in supported:
        raise ValueError(
            f"Unsupported curve interpolation: {interpolation}. Use one of {', '.join(supported)}"
        )
    return interpolation


def get_piecewise_curve(
    helpers: list,
    valuation_date: ql.Date,
    day_counter: ql.DayCounter,
    interpolation: str = CurveInterpolation.SMOOTH_FORWARD,
    bootstrap_accuracy: float = DEFAULT_BOOTSTRAP_ACCURACY,
) -> ql.YieldTermStructure:
    """
    Create a piecewise yield curve with specified interpolation.

    Args:
        helpers: List of RateHelper objects
        valuation_date: Valuation date
        day_counter: Day count convention
        interpolation: Interpolation method (LinearForward, SmoothForward or NaturalLogCubic)
        bootstrap_accuracy: Iterative bootstrap convergence tolerance

    Returns:
        QuantLib YieldTermStructure
    """
    resolved = resolve_interpolation(interpolation)

    if resolved == CurveInterpolation.SMOOTH_FORWARD:
        bootstrap = ql.IterativeBootstrap(
            accuracy=bootstrap_accuracy,
            maxAttempts=50,
            dontThrow=True,
            dontThrowSteps=10,
            maxEvaluations=1000,
        )
        return ql.PiecewiseMonotonicLogParabolicCubicDiscount(
            valuation_date, helpers, day_counter, bootstrap
        )
    elif resolved == CurveInterpolation.LINEAR_FORWARD:
        return ql.PiecewiseLogLinearDiscount(valuation_date, helpers, day_counter)
    elif resolved == CurveInterpolation.NATURAL_LOG_CUBIC:
        bootstrap = ql.IterativeBootstrap(
            accuracy=bootstrap_accuracy,
            maxAttempts=50,
            dontThrow=False,
            maxEvaluations=1000,
        )
        return ql.PiecewiseNaturalLogCubicDiscount(valuation_date, helpers, day_counter, bootstrap)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation}")
