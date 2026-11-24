"""Reprice retained calibration helpers against explicit yield curves.

QuantLib's Python bindings do not expose ``RateHelper.setTermStructure``.  This
module therefore evaluates the same economic instruments without rebuilding a
piecewise curve: money-market helpers use their retained dates and day counts,
and swap helpers reuse their retained cashflow schedules with cloned indexes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import QuantLib as ql

from .calibration_diagnostics import CalibrationHelperBinding

YieldCurve: TypeAlias = ql.YieldTermStructure | ql.YieldTermStructureHandle
_OIS_TYPES = frozenset({"ois_swap"})


@dataclass(frozen=True)
class _TelescopicOisPlan:
    """Static OIS helper cashflows: (start, end, nominal, gearing, spread x accrual, payment)
    per overnight coupon, and (nominal x accrual, payment) per fixed coupon."""

    floating: tuple[tuple[ql.Date, ql.Date, float, float, float, ql.Date], ...]
    fixed: tuple[tuple[float, ql.Date], ...]


class CalibrationRateEvaluator:
    """Reprice one fixed set of retained helpers on many explicit curves.

    Schedules are read once. A compounded overnight coupon without lookback, lockout,
    observation shift or daily-compounded spread, whose value dates are its accrual dates,
    telescopes exactly, ``prod_i (1 + f_i tau_i) = P(v_0) / P(v_n)``, which is also how
    QuantLib's compounding pricer projects it. Every other helper uses the generic
    cloned-instrument evaluation.
    """

    def __init__(self, bindings: Sequence[CalibrationHelperBinding]) -> None:
        self._bindings = tuple(bindings)
        # Plans split history from projection at this date; rates() refuses any other.
        self._evaluation_date = ql.Settings.instance().evaluationDate
        self._ois_plans = tuple(
            (
                _telescopic_ois_plan(binding.helper)
                if binding.instrument_type.lower() in _OIS_TYPES
                else None
            )
            for binding in self._bindings
        )

    def rates(
        self,
        projection_curve: YieldCurve,
        discount_curve: YieldCurve | None = None,
    ) -> tuple[float, ...]:
        """Return normalized decimal rates, in binding order, on explicit curves."""
        if ql.Settings.instance().evaluationDate != self._evaluation_date:
            raise ValueError("Calibration evaluator used outside its evaluation date")
        projection = _curve_handle(projection_curve)
        discount = projection if discount_curve is None else _curve_handle(discount_curve)
        return tuple(
            (
                _evaluate_helper_rate(binding, projection, discount)
                if plan is None
                else _telescopic_ois_rate(plan, projection, discount)
            )
            for binding, plan in zip(self._bindings, self._ois_plans)
        )


def evaluate_calibration_rates(
    bindings: Sequence[CalibrationHelperBinding],
    projection_curve: YieldCurve,
    discount_curve: YieldCurve | None = None,
) -> tuple[float, ...]:
    """Return normalized decimal rates for retained helpers on explicit curves."""
    projection_handle = _curve_handle(projection_curve)
    discount_handle = projection_handle if discount_curve is None else _curve_handle(discount_curve)
    return tuple(
        _evaluate_helper_rate(binding, projection_handle, discount_handle) for binding in bindings
    )


def evaluate_helper_rate(
    binding: CalibrationHelperBinding,
    projection_curve: YieldCurve,
    discount_curve: YieldCurve | None = None,
) -> float:
    """Return one helper's implied quote normalized to a decimal rate."""
    projection_handle = _curve_handle(projection_curve)
    discount_handle = projection_handle if discount_curve is None else _curve_handle(discount_curve)
    return _evaluate_helper_rate(binding, projection_handle, discount_handle)


def _evaluate_helper_rate(
    binding: CalibrationHelperBinding,
    projection_curve: ql.YieldTermStructureHandle,
    discount_curve: ql.YieldTermStructureHandle,
) -> float:
    instrument_type = binding.instrument_type.lower()
    if instrument_type in {"deposit", "fixing"}:
        return _simple_forward_rate(binding, projection_curve)
    if instrument_type == "fra":
        return _fra_rate(binding, projection_curve)
    if instrument_type == "future":
        return _simple_forward_rate(binding, projection_curve) + binding.convexity_adjustment
    if instrument_type == "ois_swap":
        return _ois_fair_rate(binding.helper, projection_curve, discount_curve)
    if instrument_type == "irs_swap":
        return _ibor_swap_fair_rate(binding.helper, projection_curve, discount_curve)
    raise ValueError(f"Unsupported calibration helper type: {binding.instrument_type}")


def _curve_handle(curve: YieldCurve) -> ql.YieldTermStructureHandle:
    if isinstance(curve, ql.YieldTermStructureHandle):
        return curve
    return ql.YieldTermStructureHandle(curve)


def _simple_forward_rate(
    binding: CalibrationHelperBinding,
    projection_curve: ql.YieldTermStructureHandle,
) -> float:
    start_date = binding.helper.earliestDate()
    end_date = binding.helper.maturityDate()
    accrual = binding.day_counter.yearFraction(start_date, end_date)
    if accrual <= 0.0:
        raise ValueError(f"Calibration helper {binding.instrument_type} has non-positive accrual")
    return (
        projection_curve.discount(start_date) / projection_curve.discount(end_date) - 1.0
    ) / accrual


def _fra_rate(
    binding: CalibrationHelperBinding,
    projection_curve: ql.YieldTermStructureHandle,
) -> float:
    if binding.projection_index is None:
        raise ValueError("A FRA calibration binding must retain its projection index")
    index = binding.projection_index.clone(projection_curve)
    fixing_date = index.fixingDate(binding.helper.earliestDate())
    return index.fixing(fixing_date)


def _ois_fair_rate(
    helper: ql.RateHelper,
    projection_curve: ql.YieldTermStructureHandle,
    discount_curve: ql.YieldTermStructureHandle,
) -> float:
    if not isinstance(helper, ql.OISRateHelper):
        raise TypeError("An OIS calibration binding must retain an OISRateHelper")
    swap = helper.swap()
    overnight_index = swap.overnightIndex().clone(projection_curve)
    floating_pv = 0.0
    for cashflow in swap.overnightLeg():
        coupon = ql.as_overnight_indexed_coupon(cashflow)
        if coupon is None:
            raise TypeError("OIS helper overnight leg contains a non-overnight coupon")
        cloned_coupon = _clone_overnight_coupon(coupon, overnight_index, swap.lookbackDays())
        floating_pv += cloned_coupon.amount() * discount_curve.discount(cashflow.date())
    return floating_pv / _fixed_leg_annuity(swap.fixedLeg(), discount_curve)


def _clone_overnight_coupon(
    coupon: ql.OvernightIndexedCoupon,
    overnight_index: ql.OvernightIndex,
    lookback_days: int,
) -> ql.OvernightIndexedCoupon:
    """Clone a coupon onto a curve without changing its value-date construction.

    QuantLib exposes the generated value dates but not the ``telescopicValueDates``
    constructor flag. Rebuild the full-grid form first and retain it when its dates
    match; otherwise require the telescopic form to reproduce the retained coupon.
    """

    def build(telescopic_value_dates: bool) -> ql.OvernightIndexedCoupon:
        return ql.OvernightIndexedCoupon(
            coupon.date(),
            coupon.nominal(),
            coupon.accrualStartDate(),
            coupon.accrualEndDate(),
            overnight_index,
            coupon.gearing(),
            coupon.spread(),
            coupon.referencePeriodStart(),
            coupon.referencePeriodEnd(),
            coupon.dayCounter(),
            telescopic_value_dates,
            coupon.averagingMethod(),
            lookback_days,
            coupon.lockoutDays(),
            coupon.applyObservationShift(),
            coupon.compoundSpreadDaily(),
            coupon.rateComputationStartDate(),
            coupon.rateComputationEndDate(),
            coupon.exCouponDate(),
        )

    original_value_dates = tuple(coupon.valueDates())
    full_grid_coupon = build(False)
    if tuple(full_grid_coupon.valueDates()) == original_value_dates:
        return full_grid_coupon

    telescopic_coupon = build(True)
    if tuple(telescopic_coupon.valueDates()) != original_value_dates:
        raise ValueError("Unable to reproduce retained OIS coupon value dates")
    return telescopic_coupon


def _coupon_telescopes(coupon: ql.OvernightIndexedCoupon, today: ql.Date) -> bool:
    """Whether a projected compounded coupon equals ``P(v_0) / P(v_n) - 1`` exactly."""
    value_dates = coupon.valueDates()
    # A null sub-period date means QuantLib computes the rate over the accrual period.
    no_sub_period = (coupon.accrualStartDate(), ql.Date())
    return (
        coupon.fixingDays() == 0
        and coupon.lockoutDays() == 0
        and not coupon.applyObservationShift()
        and coupon.averagingMethod() == ql.RateAveraging.Compound
        and (coupon.spread() == 0.0 or not coupon.compoundSpreadDaily())
        and coupon.rateComputationStartDate() in no_sub_period
        and coupon.rateComputationEndDate() in (coupon.accrualEndDate(), ql.Date())
        # QuantLib normalizes by the interest dates: a non-business accrual boundary
        # shifts the value dates and breaks the identity.
        and value_dates[0] == coupon.accrualStartDate()
        and value_dates[-1] == coupon.accrualEndDate()
        # A fixing on or before today would come from the index history instead.
        and value_dates[0] > today
    )


def _telescopic_ois_plan(helper: ql.RateHelper) -> _TelescopicOisPlan | None:
    """Static cashflows of a projected OIS helper, or None when it does not telescope."""
    if not isinstance(helper, ql.OISRateHelper):
        raise TypeError("An OIS calibration binding must retain an OISRateHelper")
    swap = helper.swap()
    today = ql.Settings.instance().evaluationDate
    floating = []
    for cashflow in swap.overnightLeg():
        coupon = ql.as_overnight_indexed_coupon(cashflow)
        if coupon is None:
            raise TypeError("OIS helper overnight leg contains a non-overnight coupon")
        if not _coupon_telescopes(coupon, today):
            return None
        value_dates = coupon.valueDates()
        floating.append(
            (
                value_dates[0],
                value_dates[-1],
                coupon.nominal(),
                coupon.gearing(),
                coupon.spread() * coupon.accrualPeriod(),
                cashflow.date(),
            )
        )
    fixed = []
    for cashflow in swap.fixedLeg():
        coupon = ql.as_fixed_rate_coupon(cashflow)
        if coupon is None:
            raise TypeError("Calibration swap fixed leg contains a non-fixed coupon")
        fixed.append((coupon.nominal() * coupon.accrualPeriod(), cashflow.date()))
    return _TelescopicOisPlan(tuple(floating), tuple(fixed))


def _telescopic_ois_rate(
    plan: _TelescopicOisPlan,
    projection_curve: ql.YieldTermStructureHandle,
    discount_curve: ql.YieldTermStructureHandle,
) -> float:
    floating_pv = sum(
        nominal
        * (
            gearing * (projection_curve.discount(start) / projection_curve.discount(end) - 1.0)
            + spread
        )
        * discount_curve.discount(payment)
        for start, end, nominal, gearing, spread, payment in plan.floating
    )
    annuity = sum(weight * discount_curve.discount(payment) for weight, payment in plan.fixed)
    if annuity <= 0.0:
        raise ValueError("Calibration swap has a non-positive fixed-leg annuity")
    return floating_pv / annuity


def _ibor_swap_fair_rate(
    helper: ql.RateHelper,
    projection_curve: ql.YieldTermStructureHandle,
    discount_curve: ql.YieldTermStructureHandle,
) -> float:
    if not isinstance(helper, ql.SwapRateHelper):
        raise TypeError("An IRS calibration binding must retain a SwapRateHelper")
    swap = helper.swap()
    ibor_index = swap.iborIndex().clone(projection_curve)
    cloned_swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,
        swap.nominal(),
        swap.fixedSchedule(),
        swap.fixedRate(),
        swap.fixedDayCount(),
        swap.floatingSchedule(),
        ibor_index,
        swap.spread(),
        swap.floatingDayCount(),
        swap.paymentConvention(),
    )
    cloned_swap.setPricingEngine(ql.DiscountingSwapEngine(discount_curve))
    return cloned_swap.fairRate()


def _fixed_leg_annuity(
    fixed_leg: ql.Leg,
    discount_curve: ql.YieldTermStructureHandle,
) -> float:
    annuity = 0.0
    for cashflow in fixed_leg:
        coupon = ql.as_fixed_rate_coupon(cashflow)
        if coupon is None:
            raise TypeError("Calibration swap fixed leg contains a non-fixed coupon")
        annuity += (
            coupon.nominal() * coupon.accrualPeriod() * discount_curve.discount(cashflow.date())
        )
    if annuity <= 0.0:
        raise ValueError("Calibration swap has a non-positive fixed-leg annuity")
    return annuity
