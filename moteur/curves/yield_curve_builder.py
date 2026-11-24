"""
Multi-curve yield curve builder using QuantLib.
"""

import logging
from datetime import date as dt_date
from types import MappingProxyType
from typing import Dict, Mapping, Optional

import QuantLib as ql

from config import (
    calendar_label_to_ql,
    day_count_label_to_ql,
    frequency_label_to_ql,
    get_currency_config,
    get_index,
    get_index_with_curve,
)
from market_data.curve_quote_conventions import (
    CurveQuoteConvention,
    get_curve_quote_convention,
)
from market_data.curve_quote_typing import verify_curve_market_data_typing
from market_data.schemas import (
    ForwardMarketData,
    FuturesDataPoint,
    FuturesRateBasis,
    MarketDataPoint,
    OISMarketData,
)
from pricing.calibration_fixings import scoped_ibor_calibration_fixing
from services.exceptions import PricingError

from .calibration_context import CurveCalibrationContext, RawQuoteKey
from .calibration_diagnostics import (
    CalibrationHelperBinding,
    CurveBuildResult,
    CurveCalibrationReport,
    CurveFixingObservation,
    build_calibration_report,
)
from .calibration_residuals import CalibrationRateEvaluator
from .curve_config import forward_curve_day_counter, ois_curve_day_counter
from .effective_conventions import (
    CurveLineage,
    EffectiveHelperSpec,
    build_curve_lineage,
    business_day_convention_label,
    calendar_label,
    day_count_label,
    frequency_label,
    schedule_rule_label,
)
from .helpers import (
    DEFAULT_BOOTSTRAP_ACCURACY,
    LEGACY_BOOTSTRAP_ACCURACY,
    CurveInterpolation,
    create_quote_handle,
    date_to_ql,
    freeze_curve,
    get_piecewise_curve,
    index_tenor_months,
    parse_fra_tenor,
    parse_tenor,
    ql_evaluation_date,
    ql_to_date,
    resolve_interpolation,
    serialized_ql_state,
)
from .schemas import CurveQuoteExclusion
from .short_end_selection import (
    future_end_date,
    is_fra_before_cutoff,
    short_end_cutoff_date,
)

logger = logging.getLogger(__name__)

_TIGHT_EUR_FORWARD_TENORS = frozenset({"1M", "3M", "6M"})


def _calendar_from_name(name: str) -> ql.Calendar:
    """Resolve a serialized calendar used by an explicit curve profile."""
    return calendar_label_to_ql(name)


def _business_day_convention_from_name(name: str) -> int:
    """Resolve a serialized business-day convention to QuantLib."""
    conventions = {
        "Following": ql.Following,
        "ModifiedFollowing": ql.ModifiedFollowing,
    }
    try:
        return conventions[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported curve business-day convention: {name}") from exc


def _convention(
    point: MarketDataPoint | FuturesDataPoint,
    expected_instrument_type: str,
    expected_currency: str,
) -> CurveQuoteConvention:
    """Resolve the mandatory quote metadata and validate it before constructing a helper."""
    metadata = point.quote_metadata
    convention = get_curve_quote_convention(metadata.convention_id)
    if metadata.instrument_type.value != expected_instrument_type:
        raise ValueError(
            f"Quote {metadata.source_label} is {metadata.instrument_type.value}, "
            f"expected {expected_instrument_type}"
        )
    if convention.instrument_type != expected_instrument_type:
        raise ValueError(
            f"Convention {convention.convention_id} is for {convention.instrument_type}, "
            f"not {expected_instrument_type}"
        )
    convention_currency = convention.index_name.split("-", maxsplit=1)[0]
    if convention_currency != expected_currency.upper():
        raise ValueError(
            f"Convention {convention.convention_id} belongs to {convention_currency}, "
            f"not {expected_currency.upper()}"
        )
    return convention


def _fixing_observation(
    point: MarketDataPoint, convention: CurveQuoteConvention
) -> CurveFixingObservation:
    """Keep a fixing with the snapshot as an observation that no helper uses."""
    assert point.quote_metadata is not None
    return CurveFixingObservation(
        source_label=point.quote_metadata.source_label,
        tenor=point.tenor,
        convention_id=convention.convention_id,
        market_rate=point.rate,
        quote_provenance=convention.quote_provenance,
        calibration_role=convention.calibration_role,
    )


def _future_convexity_sigma(
    convention: CurveQuoteConvention,
    futures_day_counter: ql.DayCounter,
) -> float:
    """Return the convexity volatility of a future from its versioned profile.

    A profile must carry the value, its source, its date and its time basis, and the time basis
    must be the day counter the builder measures ``T1`` and ``T2`` with.
    """
    if (
        convention.convexity_sigma is None
        or convention.convexity_sigma_source is None
        or convention.convexity_sigma_as_of is None
        or convention.convexity_time_basis is None
    ):
        raise ValueError(f"Future convention {convention.convention_id} lacks versioned convexity")
    time_basis = day_count_label_to_ql(convention.convexity_time_basis)
    if time_basis.name() != futures_day_counter.name():
        raise ValueError(
            f"Convention {convention.convention_id} measures convexity time with "
            f"{time_basis.name()}, the builder uses {futures_day_counter.name()}"
        )
    return convention.convexity_sigma


def _future_convention_day_counter(
    convention: CurveQuoteConvention,
    forward_index: ql.IborIndex,
    index_tenor: str,
) -> ql.DayCounter:
    """Validate a future profile against the QuantLib index used by its helper."""
    if convention.forward_curve_key != index_tenor:
        raise ValueError(
            f"Convention {convention.convention_id} forwards on "
            f"{convention.forward_curve_key}, not {index_tenor}"
        )
    if convention.settlement_days != forward_index.fixingDays():
        raise ValueError(
            f"Convention {convention.convention_id} uses {convention.settlement_days} "
            f"fixing days, index uses {forward_index.fixingDays()}"
        )
    expected_calendar = _calendar_from_name(convention.calendar)
    if expected_calendar.name() != forward_index.fixingCalendar().name():
        raise ValueError(
            f"Convention {convention.convention_id} uses {expected_calendar.name()}, "
            f"index uses {forward_index.fixingCalendar().name()}"
        )
    expected_business_day_convention = _business_day_convention_from_name(
        convention.business_day_convention
    )
    if expected_business_day_convention != forward_index.businessDayConvention():
        raise ValueError(
            f"Convention {convention.convention_id} uses business-day convention "
            f"{convention.business_day_convention}, index differs"
        )
    if convention.end_of_month != forward_index.endOfMonth():
        raise ValueError(
            f"Convention {convention.convention_id} end-of-month flag differs from the index"
        )
    if convention.floating_leg_frequency is None:
        raise ValueError(f"Missing floating-leg frequency for {convention.convention_id}")
    expected_tenor = ql.Period(frequency_label_to_ql(convention.floating_leg_frequency))
    if expected_tenor != forward_index.tenor():
        raise ValueError(
            f"Convention {convention.convention_id} uses {expected_tenor}, "
            f"index uses {forward_index.tenor()}"
        )
    day_counter = day_count_label_to_ql(convention.day_count)
    if day_counter.name() != forward_index.dayCounter().name():
        raise ValueError(
            f"Convention {convention.convention_id} uses {day_counter.name()}, "
            f"index uses {forward_index.dayCounter().name()}"
        )
    return day_counter


def _helper_binding(
    helper: ql.RateHelper,
    point: MarketDataPoint,
    native_quote: ql.SimpleQuote,
    raw_quote_key: RawQuoteKey,
    day_counter: ql.DayCounter,
    projection_index: ql.InterestRateIndex | None = None,
) -> CalibrationHelperBinding:
    """Bind a helper to the identity its quote metadata declares."""
    metadata = point.quote_metadata
    assert metadata is not None
    convention = get_curve_quote_convention(metadata.convention_id)
    return CalibrationHelperBinding(
        helper=helper,
        source_label=metadata.source_label,
        tenor=point.tenor,
        instrument_type=metadata.instrument_type.value,
        convention_id=metadata.convention_id,
        quote_provenance=convention.quote_provenance,
        calibration_role=convention.calibration_role,
        market_rate=point.rate,
        day_counter=day_counter,
        native_quote=native_quote,
        raw_quote_key=raw_quote_key,
        projection_index=projection_index,
    )


def _mutable_quote(value: float) -> tuple[ql.SimpleQuote, ql.QuoteHandle]:
    """Create and retain a mutable quote alongside its QuantLib handle."""
    quote = ql.SimpleQuote(value)
    return quote, ql.QuoteHandle(quote)


def _effective_helper_spec(
    *,
    instrument_type: str,
    curve_key: str,
    pillar_label: str,
    convention: CurveQuoteConvention,
    quote_kind: str = "rate",
    settlement_days: Optional[int] = None,
    calendar: Optional[ql.Calendar] = None,
    business_day_convention: Optional[int] = None,
    day_counter: Optional[ql.DayCounter] = None,
    end_of_month: Optional[bool] = None,
    fixed_leg_frequency: Optional[int] = None,
    fixed_leg_day_counter: Optional[ql.DayCounter] = None,
    floating_leg_frequency: Optional[int] = None,
    schedule_generation_rule: Optional[int] = None,
    payment_lag_days: Optional[int] = None,
    rate_cutoff_days: Optional[int] = None,
    discount_curve_key: Optional[str] = None,
    forward_curve_key: Optional[str] = None,
    index: Optional[ql.Index] = None,
    convexity_adjustment_vol: Optional[float] = None,
    payment_calendar: Optional[ql.Calendar] = None,
    fixed_calendar: Optional[ql.Calendar] = None,
    overnight_calendar: Optional[ql.Calendar] = None,
) -> EffectiveHelperSpec:
    """Describe a helper from the very convention values handed to its constructor.

    Every argument is read off the objects passed to the QuantLib helper, so the
    curve fingerprint can never diverge from a supposed copy of a profile.
    """
    return EffectiveHelperSpec(
        instrument_type=instrument_type,
        curve_key=curve_key,
        quote_kind=quote_kind,
        pillar_label=pillar_label,
        convention_id=convention.convention_id,
        settlement_days=None if settlement_days is None else int(settlement_days),
        calendar=calendar_label(calendar),
        business_day_convention=business_day_convention_label(business_day_convention),
        day_count=day_count_label(day_counter),
        end_of_month=None if end_of_month is None else bool(end_of_month),
        fixed_leg_frequency=frequency_label(fixed_leg_frequency),
        fixed_leg_day_count=day_count_label(fixed_leg_day_counter),
        floating_leg_frequency=frequency_label(floating_leg_frequency),
        schedule_generation_rule=schedule_rule_label(schedule_generation_rule),
        payment_lag_days=payment_lag_days,
        rate_cutoff_days=rate_cutoff_days,
        discount_curve_key=discount_curve_key,
        forward_curve_key=forward_curve_key,
        index_name=None if index is None else str(index.name()),
        convexity_adjustment_vol=convexity_adjustment_vol,
        payment_calendar=calendar_label(payment_calendar),
        fixed_calendar=calendar_label(fixed_calendar),
        overnight_calendar=calendar_label(overnight_calendar),
        quote_provenance=convention.quote_provenance,
        calibration_role=convention.calibration_role,
        convexity_sigma_source=convention.convexity_sigma_source,
        convexity_sigma_as_of=convention.convexity_sigma_as_of,
        convexity_time_basis=convention.convexity_time_basis,
    )


def _ois_calibration_tolerance_bps(currency: str) -> float:
    """Return the quote-fit acceptance threshold for an OIS curve."""
    return {"USD": 0.01, "GBP": 0.05}.get(currency.upper(), 0.2)


def _default_forward_bootstrap_accuracy(currency: str, index_tenor: str) -> float:
    """Return the evidence-backed default accuracy for a forward builder."""
    if currency.upper() == "EUR" and index_tenor.upper() in _TIGHT_EUR_FORWARD_TENORS:
        return DEFAULT_BOOTSTRAP_ACCURACY
    return LEGACY_BOOTSTRAP_ACCURACY


DEPOSIT_FUTURE_PILLAR_COLLISION = "deposit_future_pillar_collision"


def _exclude_futures_sharing_a_deposit_pillar(
    curve_key: str,
    helpers: list[ql.RateHelper],
    helper_bindings: list[CalibrationHelperBinding],
    effective_specs: list[EffectiveHelperSpec],
) -> tuple[
    list[ql.RateHelper],
    list[CalibrationHelperBinding],
    list[EffectiveHelperSpec],
    tuple[CurveQuoteExclusion, ...],
]:
    """Drop every future whose effective pillar equals a deposit pillar (decision D-21).

    The deposit constrains the period starting at spot and anchors the curve; the future
    constrains an IMM period and needs a convexity correction. A same-pillar pair is otherwise a
    QuantLib bootstrap error. The comparison uses the helpers' effective ``pillarDate()``, is
    decided before calibration for every interpolation, and does not depend on quote order. No
    pillar is moved and no other pair of instruments is touched: any other duplicate stays an
    explicit error. The source quote is kept and reported as an exclusion.
    """
    if not (len(helpers) == len(helper_bindings) == len(effective_specs)):
        raise ValueError(f"Helpers, bindings and specs diverged for {curve_key}")
    deposits_by_pillar: dict[int, CalibrationHelperBinding] = {}
    for binding in helper_bindings:
        if binding.instrument_type == "deposit":
            deposits_by_pillar.setdefault(binding.helper.pillarDate().serialNumber(), binding)

    exclusions: list[CurveQuoteExclusion] = []
    kept: list[int] = []
    for position, binding in enumerate(helper_bindings):
        pillar = binding.helper.pillarDate()
        deposit = deposits_by_pillar.get(pillar.serialNumber())
        if binding.instrument_type != "future" or deposit is None or binding.quote_index is None:
            kept.append(position)
            continue
        exclusions.append(
            CurveQuoteExclusion(
                curve_key=curve_key,
                source_label=binding.source_label,
                quote_index=binding.quote_index,
                market_rate=binding.market_rate,
                pillar_date=ql_to_date(pillar),
                conflicting_source_label=deposit.source_label,
                conflicting_pillar_date=ql_to_date(deposit.helper.pillarDate()),
                reason=DEPOSIT_FUTURE_PILLAR_COLLISION,
            )
        )
    if not exclusions:
        return helpers, helper_bindings, effective_specs, ()
    return (
        [helpers[i] for i in kept],
        [helper_bindings[i] for i in kept],
        [effective_specs[i] for i in kept],
        tuple(sorted(exclusions, key=lambda item: item.quote_index)),
    )


def _bounded_natural_future_exclusion(
    curve_key: str,
    helper_bindings: list[CalibrationHelperBinding],
) -> tuple[
    CurveQuoteExclusion,
    list[CalibrationHelperBinding],
]:
    """Select the sole future whose pillar is within one day of a deposit pillar."""
    deposits = [binding for binding in helper_bindings if binding.instrument_type == "deposit"]
    candidates: list[tuple[CalibrationHelperBinding, CalibrationHelperBinding]] = []
    for future in helper_bindings:
        if future.instrument_type != "future" or future.quote_index is None:
            continue
        for deposit in deposits:
            if abs(future.helper.pillarDate() - deposit.helper.pillarDate()) <= 1:
                candidates.append((future, deposit))

    if len(candidates) != 1:
        raise ValueError(
            f"NaturalLogCubic calibration for {curve_key} failed with all quotes; "
            "the bounded fallback requires exactly one future within one day of a deposit "
            f"pillar, found {len(candidates)}."
        )

    future, deposit = candidates[0]
    exclusion = CurveQuoteExclusion(
        curve_key=curve_key,
        source_label=future.source_label,
        quote_index=future.quote_index,
        market_rate=future.market_rate,
        pillar_date=ql_to_date(future.helper.pillarDate()),
        conflicting_source_label=deposit.source_label,
        conflicting_pillar_date=ql_to_date(deposit.helper.pillarDate()),
        reason=(
            "NaturalLogCubic failed with the complete quote set; this future is the sole "
            "future whose pillar is within one day of a deposit pillar."
        ),
    )
    retained_bindings = [
        binding
        for binding in helper_bindings
        if not (binding.instrument_type == "future" and binding.quote_index == future.quote_index)
    ]
    return exclusion, retained_bindings


def _solve_non_converged_bootstrap(
    context: CurveCalibrationContext,
    report: CurveCalibrationReport,
    fixing_observations: list[CurveFixingObservation],
    discount_curve: ql.YieldTermStructureHandle | None = None,
) -> CurveCalibrationReport:
    """Solve jointly a SmoothForward calibration that the iterative bootstrap left unrepriced.

    MonotonicLogParabolicCubic is non-local, so QuantLib bootstraps it with a pillar-by-pillar
    fixed-point loop that can cycle; with dontThrow it then returns an iterate that misprices
    quotes. The same equations are solved by Newton from that iterate. A curve that already
    reprices is returned untouched; a failed solve keeps the failing report, which risk rejects.
    """
    if report.passed or context.interpolation != CurveInterpolation.SMOOTH_FORWARD:
        return report
    try:
        # The solver and its report use the rate evaluator: check it against the helpers first.
        context.verify_evaluator_fidelity(discount_curve)
        pillar_values = context.solve_globally(discount_curve)
    except PricingError as exc:
        logger.warning("Global calibration of %s failed: %s", report.curve_key, exc)
        return report
    context.adopt_global_solution(pillar_values)
    curve = ql.YieldTermStructureHandle(context.pricing_curve())
    implied_rates = CalibrationRateEvaluator(context.helper_bindings).rates(
        curve, curve if discount_curve is None else discount_curve
    )
    solved = build_calibration_report(
        report.curve_key,
        list(context.helper_bindings),
        fixing_observations,
        tolerance_bps=report.tolerance_bps,
        excluded_quotes=report.excluded_quotes,
        implied_rates=implied_rates,
        solver="global_newton",
    )
    logger.info(
        "%s: iterative bootstrap residual %.3g bp, global calibration residual %.3g bp",
        report.curve_key,
        report.max_abs_residual_bps,
        solved.max_abs_residual_bps,
    )
    return solved


class YieldCurveBuilder:
    """Builder for multi-curve yield curve construction."""

    def __init__(self, currency: str = "EUR"):
        self.currency = currency
        self.config = get_currency_config(currency)

    def build_ois_curve(
        self,
        market_data: OISMarketData,
        interpolation: str = CurveInterpolation.SMOOTH_FORWARD,
        bootstrap_accuracy: Optional[float] = None,
    ) -> ql.YieldTermStructureHandle:
        """
        Build OIS discount curve.

        Args:
            market_data: OIS market data (deposits and swaps)
            interpolation: Interpolation method
            bootstrap_accuracy: Iterative bootstrap convergence tolerance. ``None`` applies the
                validated currency default.

        Returns:
            YieldTermStructureHandle for the OIS curve
        """
        return self.build_ois_curve_with_report(
            market_data,
            interpolation=interpolation,
            bootstrap_accuracy=bootstrap_accuracy,
        ).curve

    @serialized_ql_state
    def build_ois_curve_with_report(
        self,
        market_data: OISMarketData,
        interpolation: str = CurveInterpolation.SMOOTH_FORWARD,
        bootstrap_accuracy: Optional[float] = None,
    ) -> CurveBuildResult:
        """Build an OIS curve and retain quote-level calibration evidence."""
        verify_curve_market_data_typing(market_data)
        curve_day_counter = ois_curve_day_counter(self.currency)

        # Convert valuation date
        valuation_date = date_to_ql(market_data.valuation_date)
        ql.Settings.instance().evaluationDate = valuation_date

        # Get OIS index
        ois_index = get_index(self.currency, "OIS")

        # Build rate helpers
        helpers: list[ql.RateHelper] = []
        helper_bindings: list[CalibrationHelperBinding] = []
        fixing_observations: list[CurveFixingObservation] = []
        effective_specs: list[EffectiveHelperSpec] = []

        for point in market_data.fixings:
            convention = _convention(point, "fixing", self.currency)
            if convention.used_in_bootstrap:
                raise ValueError(
                    f"Convention {convention.convention_id} would make an OIS fixing a helper: "
                    "OIS fixings are kept as observations"
                )
            fixing_observations.append(_fixing_observation(point, convention))

        # Add the overnight anchor deposits
        for quote_index, point in enumerate(market_data.deposits):
            tenor = parse_tenor(point.tenor)
            convention = _convention(point, "deposit", self.currency)
            native_quote, quote = _mutable_quote(point.rate)
            if convention.forward_curve_key != "OIS":
                raise ValueError(
                    f"Convention {convention.convention_id} forwards on "
                    f"{convention.forward_curve_key}, not OIS"
                )
            settlement_days = convention.settlement_days
            calendar = _calendar_from_name(convention.calendar)
            business_day_convention = _business_day_convention_from_name(
                convention.business_day_convention
            )
            end_of_month = convention.end_of_month
            day_counter = day_count_label_to_ql(convention.day_count)

            helper = ql.DepositRateHelper(
                quote,
                tenor,
                settlement_days,
                calendar,
                business_day_convention,
                end_of_month,
                day_counter,
            )
            helpers.append(helper)
            helper_bindings.append(
                _helper_binding(
                    helper,
                    point,
                    native_quote,
                    RawQuoteKey("deposit", quote_index),
                    day_counter,
                )
            )
            effective_specs.append(
                _effective_helper_spec(
                    instrument_type="deposit",
                    curve_key="OIS",
                    pillar_label=point.tenor,
                    convention=convention,
                    settlement_days=settlement_days,
                    calendar=calendar,
                    business_day_convention=business_day_convention,
                    day_counter=day_counter,
                    end_of_month=end_of_month,
                    discount_curve_key="OIS",
                    forward_curve_key="OIS",
                )
            )

        # Add OIS swap helpers
        for quote_index, point in enumerate(market_data.swaps):
            tenor = parse_tenor(point.tenor)
            if tenor < ql.Period(1, ql.Weeks):
                # Typed or not: a one-day OIS helper degenerates when the evaluation date is the
                # last business day of a month falling on a Friday (document §16, E-15).
                raise ValueError(
                    f"OIS quote {point.tenor} is shorter than one week: an overnight rate is "
                    "not a quoted OIS swap, type it as a deposit anchor"
                )
            native_quote, quote = _mutable_quote(point.rate)
            convention = _convention(point, "ois_swap", self.currency)
            schedule_calendar = _calendar_from_name(convention.calendar)
            payment_calendar = _calendar_from_name(
                convention.payment_calendar or convention.calendar
            )
            fixed_calendar = _calendar_from_name(convention.fixed_calendar or convention.calendar)
            overnight_calendar = _calendar_from_name(
                convention.overnight_calendar or convention.calendar
            )
            business_day_convention = _business_day_convention_from_name(
                convention.business_day_convention
            )
            if convention.fixed_leg_frequency is None:
                raise ValueError(f"Missing fixed-leg frequency for {convention.convention_id}")
            if convention.fixed_leg_day_count != convention.day_count:
                raise ValueError(
                    "QuantLib OISRateHelper cannot represent a fixed-leg day count "
                    f"different from the curve day count for {convention.convention_id}"
                )
            if (
                day_count_label_to_ql(convention.floating_leg_day_count).name()
                != ois_index.dayCounter().name()
            ):
                raise ValueError(
                    f"Floating-leg day count differs from {ois_index.name()} for "
                    f"{convention.convention_id}"
                )
            if overnight_calendar.name() != ois_index.fixingCalendar().name():
                raise ValueError(
                    f"Overnight calendar differs from {ois_index.name()} for "
                    f"{convention.convention_id}"
                )
            if convention.schedule_generation_rule != "Backward":
                raise ValueError(
                    f"Unsupported OIS schedule rule for {convention.convention_id}: "
                    f"{convention.schedule_generation_rule}"
                )
            payment_frequency = frequency_label_to_ql(convention.fixed_leg_frequency)
            helper_kwargs = {
                "paymentLag": convention.payment_lag_days,
                "paymentConvention": business_day_convention,
                "paymentFrequency": payment_frequency,
                "paymentCalendar": payment_calendar,
                "endOfMonth": convention.end_of_month,
                "fixedPaymentFrequency": payment_frequency,
                "fixedCalendar": fixed_calendar,
                "averagingMethod": ql.RateAveraging.Compound,
                "lookbackDays": 0,
                "lockoutDays": convention.rate_cutoff_days,
                "applyObservationShift": False,
                "rule": ql.DateGeneration.Backward,
                "overnightCalendar": overnight_calendar,
                "convention": business_day_convention,
            }
            effective_pillar_label = point.tenor
            if convention.index_name == "USD-SOFR":
                # Settlement advances T+2 directly from the curve date, even
                # when that date is a holiday, as observed in the external
                # comparison data. QuantLib's relative OIS helper first
                # adjusts a holiday evaluation date and would start one day late.
                settlement_date = schedule_calendar.advance(
                    valuation_date,
                    convention.settlement_days,
                    ql.Days,
                )
                maturity_date = schedule_calendar.advance(
                    settlement_date,
                    tenor,
                    business_day_convention,
                    convention.end_of_month,
                )
                helper = ql.OISRateHelper.forDates(
                    settlement_date,
                    maturity_date,
                    quote,
                    ois_index,
                    **helper_kwargs,
                )
                effective_pillar_label = (
                    f"{point.tenor}|{ql_to_date(settlement_date).isoformat()}"
                    f"|{ql_to_date(maturity_date).isoformat()}"
                )
            else:
                helper = ql.OISRateHelper(
                    convention.settlement_days,
                    tenor,
                    quote,
                    ois_index,
                    **helper_kwargs,
                )
            if schedule_calendar.name() != fixed_calendar.name():
                raise ValueError(
                    f"Schedule and fixed calendars differ for {convention.convention_id}"
                )
            fixed_leg_day_counter = (
                None
                if convention.fixed_leg_day_count is None
                else day_count_label_to_ql(convention.fixed_leg_day_count)
            )
            effective_specs.append(
                _effective_helper_spec(
                    instrument_type="ois_swap",
                    curve_key="OIS",
                    pillar_label=effective_pillar_label,
                    convention=convention,
                    settlement_days=convention.settlement_days,
                    calendar=schedule_calendar,
                    business_day_convention=business_day_convention,
                    day_counter=ois_index.dayCounter(),
                    end_of_month=convention.end_of_month,
                    fixed_leg_frequency=payment_frequency,
                    fixed_leg_day_counter=fixed_leg_day_counter,
                    floating_leg_frequency=payment_frequency,
                    schedule_generation_rule=ql.DateGeneration.Backward,
                    payment_lag_days=convention.payment_lag_days,
                    rate_cutoff_days=convention.rate_cutoff_days,
                    payment_calendar=payment_calendar,
                    fixed_calendar=fixed_calendar,
                    overnight_calendar=overnight_calendar,
                    discount_curve_key=convention.discount_curve_key,
                    forward_curve_key=convention.forward_curve_key,
                    index=ois_index,
                )
            )
            helpers.append(helper)
            helper_bindings.append(
                _helper_binding(
                    helper,
                    point,
                    native_quote,
                    RawQuoteKey("swap", quote_index),
                    ois_index.dayCounter(),
                )
            )

        # Ensure we have enough helpers
        if len(helpers) < 2:
            raise ValueError(
                f"Not enough instruments to build curve. Need at least 2, got {len(helpers)}. "
                f"Tip: Use tenors >= 1W for deposits and at least one swap."
            )

        effective_bootstrap_accuracy = (
            bootstrap_accuracy if bootstrap_accuracy is not None else DEFAULT_BOOTSTRAP_ACCURACY
        )

        # Build piecewise curve
        curve = get_piecewise_curve(
            helpers,
            valuation_date,
            curve_day_counter,
            interpolation,
            effective_bootstrap_accuracy,
        )

        # Enable extrapolation
        curve.enableExtrapolation()
        curve.discount(curve.maxDate())
        report = build_calibration_report(
            f"{self.currency}_OIS",
            helper_bindings,
            fixing_observations,
            tolerance_bps=_ois_calibration_tolerance_bps(self.currency),
        )

        calibration_context = CurveCalibrationContext(
            curve_key=f"{self.currency}_OIS",
            native_curve=curve,
            helper_bindings=tuple(helper_bindings),
            valuation_date=market_data.valuation_date,
            interpolation=interpolation,
            day_counter=curve_day_counter,
            bootstrap_accuracy=effective_bootstrap_accuracy,
            tolerance_bps=report.tolerance_bps,
        )
        report = _solve_non_converged_bootstrap(calibration_context, report, fixing_observations)
        # Freeze into static DiscountCurve to decouple from evaluationDate
        frozen = freeze_curve(calibration_context.pricing_curve(), curve_day_counter)
        return CurveBuildResult(
            curve=ql.YieldTermStructureHandle(frozen),
            report=report,
            effective_helper_specs=tuple(effective_specs),
            effective_bootstrap_accuracy=effective_bootstrap_accuracy,
            curve_day_count=day_count_label(curve_day_counter),
            calibration_context=calibration_context,
        )

    def build_forward_curve(
        self,
        market_data: ForwardMarketData,
        discount_curve: ql.YieldTermStructureHandle,
        interpolation: str = CurveInterpolation.SMOOTH_FORWARD,
        bootstrap_accuracy: Optional[float] = None,
    ) -> ql.YieldTermStructureHandle:
        """
        Build forward curve with OIS discounting.

        Uses FuturesRateHelper with actual IMM dates. Raw futures yields
        receive the convexity adjustment from their versioned convention;
        already adjusted rates receive no second adjustment.
        Swaps complete the long end.

        Args:
            market_data: Forward curve market data (deposits, futures, swaps)
            discount_curve: OIS discount curve handle
            interpolation: Interpolation method
            bootstrap_accuracy: Iterative bootstrap convergence tolerance. ``None`` applies the
                validated currency/index default.

        Returns:
            YieldTermStructureHandle for the forward curve
        """
        return self.build_forward_curve_with_report(
            market_data,
            discount_curve,
            interpolation=interpolation,
            bootstrap_accuracy=bootstrap_accuracy,
        ).curve

    @serialized_ql_state
    def build_forward_curve_with_report(
        self,
        market_data: ForwardMarketData,
        discount_curve: ql.YieldTermStructureHandle,
        interpolation: str = CurveInterpolation.SMOOTH_FORWARD,
        bootstrap_accuracy: Optional[float] = None,
        calibration_fixing: float | None = None,
    ) -> CurveBuildResult:
        """Build a forwarding curve with a captured curve-date fixing, when available."""
        with scoped_ibor_calibration_fixing(
            self.currency,
            market_data.index_tenor,
            market_data.valuation_date,
            calibration_fixing,
        ):
            return self._build_forward_curve_with_report_unscoped(
                market_data,
                discount_curve,
                interpolation=interpolation,
                bootstrap_accuracy=bootstrap_accuracy,
            )

    def _build_forward_curve_with_report_unscoped(
        self,
        market_data: ForwardMarketData,
        discount_curve: ql.YieldTermStructureHandle,
        interpolation: str,
        bootstrap_accuracy: Optional[float],
    ) -> CurveBuildResult:
        """Construct helpers and solve while the calibration fixing scope is active."""
        verify_curve_market_data_typing(market_data)
        curve_day_counter = forward_curve_day_counter(self.currency, market_data.index_tenor)

        # Convert valuation date
        valuation_date = date_to_ql(market_data.valuation_date)
        ql.Settings.instance().evaluationDate = valuation_date

        calibration_discount_curve = ql.RelinkableYieldTermStructureHandle()
        calibration_discount_curve.linkTo(discount_curve.currentLink())

        # Get forward index
        forward_index = get_index(self.currency, market_data.index_tenor)

        idx_months = index_tenor_months(market_data.index_tenor)

        # No future or FRA may end after the first swap matures (decision D-19), so a short-end
        # instrument never overlaps a swap constraint on the same region.
        cutoff_date = short_end_cutoff_date(market_data)

        # Build rate helpers
        helpers: list[ql.RateHelper] = []
        helper_bindings: list[CalibrationHelperBinding] = []
        fixing_observations: list[CurveFixingObservation] = []
        effective_specs: list[EffectiveHelperSpec] = []
        curve_key = market_data.index_tenor
        for point in market_data.fixings:
            convention = _convention(point, "fixing", self.currency)
            if convention.forward_curve_key != market_data.index_tenor:
                raise ValueError(
                    f"Convention {convention.convention_id} forwards on "
                    f"{convention.forward_curve_key}, not {market_data.index_tenor}"
                )
            if convention.used_in_bootstrap:
                raise ValueError(
                    f"Forward fixing {point.tenor} cannot be used without a distinct pillar"
                )
            fixing_observations.append(_fixing_observation(point, convention))

        # Deposit helpers: only the deposit of the curve's own tenor anchors it (S-11)
        for quote_index, point in enumerate(market_data.deposits):
            tenor = parse_tenor(point.tenor)
            if tenor != ql.Period(idx_months, ql.Months):
                raise ValueError(
                    f"Deposit {point.tenor} is not the {market_data.index_tenor} tenor of this "
                    "curve: a projection curve is calibrated only with instruments of its own tenor"
                )
            native_quote, quote = _mutable_quote(point.rate)
            convention = _convention(point, "deposit", self.currency)
            if convention.forward_curve_key != market_data.index_tenor:
                raise ValueError(
                    f"Convention {convention.convention_id} forwards on "
                    f"{convention.forward_curve_key}, not {market_data.index_tenor}"
                )
            settlement_days = convention.settlement_days
            calendar = _calendar_from_name(convention.calendar)
            business_day_convention = _business_day_convention_from_name(
                convention.business_day_convention
            )
            end_of_month = convention.end_of_month
            day_counter = day_count_label_to_ql(convention.day_count)

            helper = ql.DepositRateHelper(
                quote,
                tenor,
                settlement_days,
                calendar,
                business_day_convention,
                end_of_month,
                day_counter,
            )
            helpers.append(helper)
            helper_bindings.append(
                _helper_binding(
                    helper,
                    point,
                    native_quote,
                    RawQuoteKey("deposit", quote_index),
                    day_counter,
                )
            )
            effective_specs.append(
                _effective_helper_spec(
                    instrument_type="deposit",
                    curve_key=curve_key,
                    pillar_label=point.tenor,
                    convention=convention,
                    settlement_days=settlement_days,
                    calendar=calendar,
                    business_day_convention=business_day_convention,
                    day_counter=day_counter,
                    end_of_month=end_of_month,
                    discount_curve_key="OIS",
                    forward_curve_key=curve_key,
                )
            )

        if not any(binding.instrument_type == "deposit" for binding in helper_bindings):
            raise ValueError(
                f"The {self.currency} {market_data.index_tenor} curve needs a deposit of its own "
                "tenor to anchor it at spot"
            )

        # Short-end: FuturesRateHelper with actual IMM dates + convexity
        if market_data.futures:
            for quote_index, point in enumerate(market_data.futures):
                convention = _convention(point, "future", self.currency)
                futures_day_counter = _future_convention_day_counter(
                    convention,
                    forward_index,
                    market_data.index_tenor,
                )
                metadata = point.quote_metadata
                assert metadata is not None
                source_label = metadata.source_label
                convention_id = convention.convention_id
                point_vol = _future_convexity_sigma(convention, futures_day_counter)
                imm_ql = date_to_ql(point.imm_date)
                t_start = futures_day_counter.yearFraction(valuation_date, imm_ql)
                end_ql = future_end_date(point, market_data.index_tenor)
                if cutoff_date is not None and end_ql > cutoff_date:
                    continue
                t_end = futures_day_counter.yearFraction(valuation_date, end_ql)
                futures_price = 100.0 - point.rate * 100.0
                native_quote, price_quote = _mutable_quote(futures_price)
                # The EUR 3M Excel import marks vendor-adjusted rates explicitly.
                # Only raw futures yields require another convexity adjustment.
                already_adjusted = (
                    point.rate_basis == FuturesRateBasis.CONVEXITY_ADJUSTED_RATE
                )
                applied_convexity_vol = 0.0 if already_adjusted else point_vol
                ca = 0.5 * applied_convexity_vol**2 * t_start * t_end
                ca_quote = create_quote_handle(ca)
                helper = ql.FuturesRateHelper(
                    price_quote,
                    imm_ql,
                    forward_index,
                    ca_quote,
                )
                helpers.append(helper)
                helper_bindings.append(
                    CalibrationHelperBinding(
                        helper=helper,
                        source_label=source_label,
                        tenor=point.imm_date.isoformat(),
                        instrument_type="future",
                        convention_id=convention_id,
                        market_rate=point.rate,
                        day_counter=futures_day_counter,
                        quote_kind="futures_price",
                        quote_index=quote_index,
                        native_quote=native_quote,
                        raw_quote_key=RawQuoteKey("future", quote_index),
                        convexity_adjustment=ca,
                        projection_index=forward_index,
                    )
                )
                effective_specs.append(
                    _effective_helper_spec(
                        instrument_type="future",
                        curve_key=curve_key,
                        pillar_label=point.imm_date.isoformat(),
                        convention=convention,
                        quote_kind="futures_price",
                        settlement_days=forward_index.fixingDays(),
                        calendar=forward_index.fixingCalendar(),
                        business_day_convention=forward_index.businessDayConvention(),
                        day_counter=futures_day_counter,
                        end_of_month=forward_index.endOfMonth(),
                        discount_curve_key="OIS",
                        forward_curve_key=curve_key,
                        index=forward_index,
                        convexity_adjustment_vol=applied_convexity_vol,
                    )
                )

        # FRA helpers (the EURIBOR 6M short end).
        if market_data.fras:
            for quote_index, point in enumerate(market_data.fras):
                months_to_start, months_to_end = parse_fra_tenor(point.tenor)
                if not is_fra_before_cutoff(market_data, point, cutoff_date):
                    continue
                convention = _convention(point, "fra", self.currency)
                if convention.forward_curve_key != market_data.index_tenor:
                    raise ValueError(
                        f"Convention {convention.convention_id} forwards on "
                        f"{convention.forward_curve_key}, not {market_data.index_tenor}"
                    )
                if convention.settlement_days != forward_index.fixingDays():
                    raise ValueError(
                        f"Convention {convention.convention_id} uses "
                        f"{convention.settlement_days} fixing days, index uses "
                        f"{forward_index.fixingDays()}"
                    )
                if months_to_end - months_to_start != idx_months:
                    raise ValueError(
                        f"FRA {point.tenor} spans {months_to_end - months_to_start}M, "
                        f"not the {idx_months}M index tenor"
                    )
                if convention.floating_leg_frequency is None:
                    raise ValueError(
                        f"Missing floating-leg frequency for {convention.convention_id}"
                    )
                floating_period = ql.Period(
                    frequency_label_to_ql(convention.floating_leg_frequency)
                )
                if floating_period != ql.Period(idx_months, ql.Months):
                    raise ValueError(
                        f"Convention {convention.convention_id} uses {floating_period}, "
                        f"not the {idx_months}M index tenor"
                    )
                settlement_days = convention.settlement_days
                calendar = _calendar_from_name(convention.calendar)
                business_day_convention = _business_day_convention_from_name(
                    convention.business_day_convention
                )
                end_of_month = convention.end_of_month
                day_counter = day_count_label_to_ql(convention.day_count)
                native_quote, quote = _mutable_quote(point.rate)
                helper = ql.FraRateHelper(
                    quote,
                    months_to_start,
                    months_to_end,
                    settlement_days,
                    calendar,
                    business_day_convention,
                    end_of_month,
                    day_counter,
                    ql.Pillar.LastRelevantDate,
                    ql.Date(),
                    True,
                )
                helpers.append(helper)
                helper_bindings.append(
                    _helper_binding(
                        helper,
                        point,
                        native_quote,
                        RawQuoteKey("fra", quote_index),
                        day_counter,
                        forward_index,
                    )
                )
                effective_specs.append(
                    _effective_helper_spec(
                        instrument_type="fra",
                        curve_key=curve_key,
                        pillar_label=point.tenor,
                        convention=convention,
                        settlement_days=settlement_days,
                        calendar=calendar,
                        business_day_convention=business_day_convention,
                        day_counter=day_counter,
                        end_of_month=end_of_month,
                        discount_curve_key="OIS",
                        forward_curve_key=curve_key,
                    )
                )

        # Add swap helpers with OIS discounting
        for quote_index, point in enumerate(market_data.swaps):
            tenor = parse_tenor(point.tenor)
            native_quote, quote = _mutable_quote(point.rate)
            convention = _convention(point, "irs_swap", self.currency)
            if convention.forward_curve_key != market_data.index_tenor:
                raise ValueError(
                    f"Convention {convention.convention_id} forwards on "
                    f"{convention.forward_curve_key}, not {market_data.index_tenor}"
                )
            if convention.settlement_days != forward_index.fixingDays():
                raise ValueError(
                    f"Convention {convention.convention_id} uses "
                    f"{convention.settlement_days} fixing days, index uses "
                    f"{forward_index.fixingDays()}"
                )
            if convention.fixed_leg_frequency is None or convention.fixed_leg_day_count is None:
                raise ValueError(f"Incomplete fixed-leg convention: {convention.convention_id}")
            calendar = _calendar_from_name(convention.calendar)
            fixed_leg_frequency = frequency_label_to_ql(convention.fixed_leg_frequency)
            business_day_convention = _business_day_convention_from_name(
                convention.business_day_convention
            )
            fixed_leg_day_count = day_count_label_to_ql(convention.fixed_leg_day_count)
            if convention.payment_lag_days != 0:
                raise ValueError(
                    f"SwapRateHelper supports only zero payment lag for "
                    f"{convention.convention_id}"
                )
            if convention.schedule_generation_rule != "Backward":
                raise ValueError(
                    f"Unsupported swap schedule rule for {convention.convention_id}: "
                    f"{convention.schedule_generation_rule}"
                )

            helper_args = (
                quote,
                tenor,
                calendar,
                fixed_leg_frequency,
                business_day_convention,
                fixed_leg_day_count,
                forward_index,
                ql.QuoteHandle(),  # spread
                ql.Period(0, ql.Days),  # forward start
                calibration_discount_curve,  # discounting curve
            )
            helper = ql.SwapRateHelper(
                *helper_args,
                convention.settlement_days,
                ql.Pillar.LastRelevantDate,
                ql.Date(),
                convention.end_of_month,
                None,
                business_day_convention,
            )
            fixed_schedule = helper.swap().fixedSchedule()
            floating_schedule = helper.swap().floatingSchedule()
            if (
                fixed_schedule.rule() != ql.DateGeneration.Backward
                or floating_schedule.rule() != ql.DateGeneration.Backward
            ):
                raise ValueError(
                    f"QuantLib did not apply Backward schedules for {convention.convention_id}"
                )
            helpers.append(helper)
            helper_bindings.append(
                _helper_binding(
                    helper,
                    point,
                    native_quote,
                    RawQuoteKey("swap", quote_index),
                    forward_index.dayCounter(),
                )
            )
            effective_specs.append(
                _effective_helper_spec(
                    instrument_type="irs_swap",
                    curve_key=curve_key,
                    pillar_label=point.tenor,
                    convention=convention,
                    settlement_days=convention.settlement_days,
                    calendar=calendar,
                    business_day_convention=business_day_convention,
                    day_counter=forward_index.dayCounter(),
                    end_of_month=convention.end_of_month,
                    fixed_leg_frequency=fixed_leg_frequency,
                    fixed_leg_day_counter=fixed_leg_day_count,
                    schedule_generation_rule=ql.DateGeneration.Backward,
                    payment_lag_days=convention.payment_lag_days,
                    discount_curve_key="OIS",
                    forward_curve_key=curve_key,
                    index=forward_index,
                )
            )

        effective_bootstrap_accuracy = (
            bootstrap_accuracy
            if bootstrap_accuracy is not None
            else _default_forward_bootstrap_accuracy(self.currency, market_data.index_tenor)
        )

        curve_key = f"{self.currency}_{market_data.index_tenor}"
        helpers, helper_bindings, effective_specs, collision_exclusions = (
            _exclude_futures_sharing_a_deposit_pillar(
                curve_key, helpers, helper_bindings, effective_specs
            )
        )

        def calibrate(
            active_helpers: list[ql.RateHelper],
            active_bindings: list[CalibrationHelperBinding],
            excluded_quotes: tuple[CurveQuoteExclusion, ...] = (),
        ) -> tuple[ql.YieldTermStructure, CurveCalibrationReport]:
            curve = get_piecewise_curve(
                active_helpers,
                valuation_date,
                curve_day_counter,
                interpolation,
                effective_bootstrap_accuracy,
            )
            curve.enableExtrapolation()
            curve.discount(curve.maxDate())
            report = build_calibration_report(
                curve_key,
                active_bindings,
                fixing_observations,
                excluded_quotes=excluded_quotes,
            )
            return curve, report

        natural_interpolation = (
            resolve_interpolation(interpolation) == CurveInterpolation.NATURAL_LOG_CUBIC
        )
        initial_error: RuntimeError | None = None
        try:
            curve, report = calibrate(
                helpers, helper_bindings, excluded_quotes=collision_exclusions
            )
        except RuntimeError as exc:
            if not natural_interpolation:
                raise
            initial_error = exc

        if natural_interpolation and (initial_error is not None or not report.passed):
            try:
                exclusion, retained_bindings = _bounded_natural_future_exclusion(
                    curve_key,
                    helper_bindings,
                )
            except ValueError as exc:
                if initial_error is not None:
                    raise exc from initial_error
                raise
            retained_helpers = [binding.helper for binding in retained_bindings]
            try:
                curve, report = calibrate(
                    retained_helpers,
                    retained_bindings,
                    excluded_quotes=(*collision_exclusions, exclusion),
                )
            except RuntimeError as exc:
                raise ValueError(
                    f"NaturalLogCubic calibration for {curve_key} still failed after the "
                    f"single permitted future exclusion ({exclusion.source_label})."
                ) from exc
            if not report.passed:
                raise ValueError(
                    f"NaturalLogCubic calibration for {curve_key} remained outside the "
                    f"{report.tolerance_bps:g} bp tolerance after the single permitted future "
                    f"exclusion ({exclusion.source_label}); maximum residual is "
                    f"{report.max_abs_residual_bps:g} bp."
                )
            # The fingerprint must describe the helpers the bootstrap actually used.
            if len(effective_specs) != len(helper_bindings):
                raise ValueError(f"Helper specs and bindings diverged for {curve_key}")
            retained_ids = {id(binding) for binding in retained_bindings}
            effective_specs = [
                spec
                for spec, binding in zip(effective_specs, helper_bindings)
                if id(binding) in retained_ids
            ]
            helper_bindings = retained_bindings

        calibration_context = CurveCalibrationContext(
            curve_key=curve_key,
            native_curve=curve,
            helper_bindings=tuple(helper_bindings),
            valuation_date=market_data.valuation_date,
            interpolation=interpolation,
            day_counter=curve_day_counter,
            bootstrap_accuracy=effective_bootstrap_accuracy,
            tolerance_bps=report.tolerance_bps,
            discount_curve_handle=calibration_discount_curve,
        )
        report = _solve_non_converged_bootstrap(
            calibration_context, report, fixing_observations, calibration_discount_curve
        )
        # Freeze into static DiscountCurve to decouple from evaluationDate
        frozen = freeze_curve(calibration_context.pricing_curve(), curve_day_counter)
        return CurveBuildResult(
            curve=ql.YieldTermStructureHandle(frozen),
            report=report,
            effective_helper_specs=tuple(effective_specs),
            effective_bootstrap_accuracy=effective_bootstrap_accuracy,
            curve_day_count=day_count_label(curve_day_counter),
            calibration_context=calibration_context,
        )

    def build_multi_curve_with_lineage(
        self,
        ois_data: OISMarketData,
        forward_data_dict: Dict[str, ForwardMarketData],
        interpolation: str = CurveInterpolation.SMOOTH_FORWARD,
        observed_calibration_fixings: Mapping[str, float] | None = None,
    ) -> tuple[
        Dict[str, ql.YieldTermStructureHandle],
        CurveLineage,
        Dict[str, CurveCalibrationReport],
    ]:
        """Build a complete multi-curve setup with provenance and calibration reports.

        The effective helper specifications used by the fingerprint are the ones the
        bootstrap actually consumed, collected while the helpers were constructed.
        """
        ois_result = self.build_ois_curve_with_report(ois_data, interpolation=interpolation)
        curves: Dict[str, ql.YieldTermStructureHandle] = {"OIS": ois_result.curve}
        calibration_reports: Dict[str, CurveCalibrationReport] = {"OIS": ois_result.report}
        effective_specs: list[EffectiveHelperSpec] = list(ois_result.effective_helper_specs)
        bootstrap_accuracies: Dict[str, float] = {
            "OIS": float(ois_result.effective_bootstrap_accuracy or 0.0),
        }
        curve_day_counts: Dict[str, Optional[str]] = {"OIS": ois_result.curve_day_count}

        observed = dict(observed_calibration_fixings or {})
        chosen_fixings: dict[str, float] = {}
        for tenor, forward_data in forward_data_dict.items():
            forward_result = self.build_forward_curve_with_report(
                forward_data,
                ois_result.curve,
                interpolation=interpolation,
                calibration_fixing=observed.get(tenor),
            )
            if self.currency == "EUR":
                if tenor in observed:
                    chosen_fixings[tenor] = float(observed[tenor])
                else:
                    with ql_evaluation_date(ois_data.valuation_date):
                        index = get_index_with_curve(self.currency, tenor, forward_result.curve)
                        chosen_fixings[tenor] = float(
                            index.fixing(date_to_ql(ois_data.valuation_date), True)
                        )
            curves[tenor] = forward_result.curve
            calibration_reports[tenor] = forward_result.report
            effective_specs.extend(forward_result.effective_helper_specs)
            bootstrap_accuracies[tenor] = float(forward_result.effective_bootstrap_accuracy or 0.0)
            curve_day_counts[tenor] = forward_result.curve_day_count

        lineage = build_curve_lineage(
            currency=self.currency,
            curve_date=ois_data.valuation_date,
            requested_interpolation=interpolation,
            effective_interpolation=resolve_interpolation(interpolation),
            ois_data=ois_data,
            forward_data_by_tenor=forward_data_dict,
            effective_helper_specs=tuple(effective_specs),
            bootstrap_accuracies=bootstrap_accuracies,
            curve_day_counts=curve_day_counts,
            calibration_fixings=chosen_fixings,
            observed_calibration_fixings=observed,
        )
        return curves, lineage, calibration_reports


class CurveSet:
    """Container for a complete set of curves.

    ``implied_forward_tenors`` (``1M``, ``3M``, ``6M``; USD and GBP only) lists the historical
    implied-forward identifiers ``USD_3M``, ``GBP_3M``, ... They are not curves: each key is an
    alias of the OIS projection curve (``get_curve_by_key``, ``get_forward_curve``), kept for
    compatibility. The pricing and the curve-sensitivity engine use them as key aliases and the
    worker market snapshot carries them (``market_snapshot.py``), so this attribute must not be
    removed or renamed. The Courbes tab no longer builds them: it reads the implied forward of an
    overnight curve as a view of the OIS curve, ``{ccy}_OIS`` at a ``forward_tenor`` horizon
    (``curves/forward_views.py``), which returns exactly the same points as the historical
    identifier (plan FWDOIS, decision D-F6; test O3 of ``test_curves_ois_forward_views.py``).
    """

    def __init__(
        self,
        currency: str,
        valuation_date: dt_date,
        ois_curve: ql.YieldTermStructureHandle,
        forward_curves: Dict[str, ql.YieldTermStructureHandle],
        ois_market_data: Optional[OISMarketData] = None,
        forward_market_data: Optional[Dict[str, ForwardMarketData]] = None,
        interpolation: str = CurveInterpolation.SMOOTH_FORWARD,
        lineage: Optional[CurveLineage] = None,
        discount_interpolation: Optional[str] = None,
        forward_interpolation: Optional[str] = None,
        ois_forward_curve: Optional[ql.YieldTermStructureHandle] = None,
        calibration_reports: Optional[Dict[str, CurveCalibrationReport]] = None,
        implied_forward_tenors: list[str] | None = None,
        discount_curves: Optional[Dict[str, ql.YieldTermStructureHandle]] = None,
        calibration_fixings: Optional[Mapping[str, float]] = None,
        observed_calibration_fixings: Optional[Mapping[str, float]] = None,
    ):
        self.currency = currency
        self.valuation_date = valuation_date
        self.ois_curve = ois_curve
        self.ois_forward_curve = ois_forward_curve or ois_curve
        self.forward_curves = forward_curves
        self.discount_curves = discount_curves if discount_curves is not None else forward_curves
        self.ois_market_data = ois_market_data
        self.forward_market_data = forward_market_data or {}
        # Copy into read-only mappings: risk scenarios must retain the exact anchor
        # selected during this build even if the fixing store changes later.
        self.calibration_fixings = MappingProxyType(dict(calibration_fixings or {}))
        self.observed_calibration_fixings = MappingProxyType(
            dict(observed_calibration_fixings or {})
        )
        self.calibration_reports = calibration_reports or {}
        self.implied_forward_tenors = list(implied_forward_tenors or [])
        self.interpolation = interpolation
        self.effective_interpolation = resolve_interpolation(interpolation)
        # Optional by contract for legacy and test curve sets; every production
        # builder fills it, and its absence is reported as unverified, never as a match.
        self.lineage = lineage
        self.discount_interpolation = resolve_interpolation(discount_interpolation or interpolation)
        self.forward_interpolation = resolve_interpolation(forward_interpolation or interpolation)

    @property
    def curve_date(self) -> dt_date:
        """
        Get the curve date (market data reference date).

        This is an alias for valuation_date providing semantic clarity when
        separating curve date from pricing valuation date.
        """
        return self.valuation_date

    @property
    def curve_quote_exclusions(self) -> list[CurveQuoteExclusion]:
        """Return unique, notified quote exclusions across all calibration roles."""
        unique: dict[tuple[object, ...], CurveQuoteExclusion] = {}
        for report in self.calibration_reports.values():
            for exclusion in report.excluded_quotes:
                key = (
                    exclusion.curve_key,
                    exclusion.instrument_type,
                    exclusion.quote_index,
                    exclusion.source_label,
                    exclusion.market_rate,
                    exclusion.pillar_date,
                    exclusion.conflicting_source_label,
                    exclusion.conflicting_pillar_date,
                    exclusion.reason,
                )
                unique.setdefault(key, exclusion)
        return list(unique.values())

    def get_discount_curve(self, key: Optional[str] = None) -> ql.YieldTermStructureHandle:
        """Get the discount curve. Defaults to OIS if no key specified."""
        if key is None:
            return self.ois_curve
        normalized_key = key.upper().strip()
        if normalized_key in self.discount_curves:
            return self.discount_curves[normalized_key]
        return self.get_curve_by_key(key)

    def get_curve_by_key(self, key: str) -> ql.YieldTermStructureHandle:
        """Get a curve by key name (OIS, 3M, 6M, etc.)."""
        k = key.upper().strip()
        if k in ("OIS", "ESTR", "SOFR", "SONIA"):
            return self.ois_curve
        if k in self.implied_forward_tenors:
            return self.ois_curve
        if k in self.forward_curves:
            return self.forward_curves[k]
        available = ["OIS"] + list(self.forward_curves.keys())
        raise ValueError(f"Curve '{key}' not found. Available: {available}")

    def get_forward_curve(self, tenor: str) -> ql.YieldTermStructureHandle:
        """Get a forward curve by tenor. RFR tenors (SOFR, SONIA) map to the OIS curve."""
        from config import is_rfr_index

        normalized_tenor = tenor.upper().strip()
        if normalized_tenor == "OIS":
            return self.ois_forward_curve
        if is_rfr_index(self.currency, normalized_tenor):
            return self.ois_forward_curve
        if normalized_tenor in self.implied_forward_tenors:
            return self.ois_forward_curve
        if normalized_tenor not in self.forward_curves:
            raise ValueError(f"Forward curve for tenor {tenor} not available")
        return self.forward_curves[normalized_tenor]
