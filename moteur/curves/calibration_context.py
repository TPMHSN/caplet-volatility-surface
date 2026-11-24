"""Reusable QuantLib calibration state for isolated curve-risk calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import QuantLib as ql

from services.exceptions import PricingError

from .calibration_diagnostics import CalibrationHelperBinding
from .calibration_jacobian import FloatArray, solve_calibration_system
from .calibration_residuals import CalibrationRateEvaluator, evaluate_calibration_rates
from .helpers import (
    CurveInterpolation,
    freeze_curve,
    get_piecewise_curve,
    ql_evaluation_date,
    resolve_interpolation,
)

RawQuoteCategory = Literal["fixing", "deposit", "fra", "future", "swap"]
# Largest helper residual accepted from a SmoothForward bootstrap in a risk scenario.
SCENARIO_CONVERGENCE_BPS = 1.0e-5
# Largest residual of a global solution on QuantLib's cloned instruments.
GLOBAL_SOLUTION_CHECK_BPS = 1.0e-6
# Largest gap between the rate evaluator and QuantLib's helpers on the same curve (decimal).
EVALUATOR_FIDELITY_TOLERANCE = 1.0e-8


@dataclass(frozen=True, order=True)
class RawQuoteKey:
    """Stable coordinate of a quote in its source market-data collection."""

    category: RawQuoteCategory
    index: int


@dataclass
class CurveCalibrationContext:
    """Native bootstrap objects retained for quote shocks and implicit calibration risk."""

    curve_key: str
    native_curve: ql.YieldTermStructure
    helper_bindings: tuple[CalibrationHelperBinding, ...]
    valuation_date: date
    interpolation: str
    day_counter: ql.DayCounter
    bootstrap_accuracy: float
    tolerance_bps: float
    discount_curve_handle: ql.RelinkableYieldTermStructureHandle | None = None
    pillar_dates: tuple[ql.Date, ...] = field(init=False)
    pillar_values: tuple[float, ...] = field(init=False)
    # Pillars of the initial native bootstrap. pillar_values differ only after
    # adopt_global_solution; see native_state_values for what the helpers reprice now.
    native_pillar_values: tuple[float, ...] = field(init=False)
    globally_solved: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Snapshot the calibrated native pillars after the initial bootstrap."""
        self.interpolation = resolve_interpolation(self.interpolation)
        self.pillar_dates = tuple(self.native_curve.dates())
        self.pillar_values = tuple(float(value) for value in self.native_curve.data())
        self.native_pillar_values = self.pillar_values

    def target_rates(self) -> tuple[float, ...]:
        """Current quote of every active helper as a decimal rate, in binding order."""
        rates = []
        for binding in self.helper_bindings:
            quote = binding.native_quote
            if quote is None:
                raise PricingError(
                    f"Active helper {binding.source_label} for {self.curve_key} has no quote"
                )
            value = quote.value()
            rates.append(
                (100.0 - value) / 100.0 if binding.quote_kind == "futures_price" else value
            )
        return tuple(rates)

    def solve_globally(
        self,
        discount_curve: ql.YieldTermStructureHandle | None = None,
        start: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """Pillar discount factors repricing every active quote, solved jointly by Newton.

        Same helpers, interpolator and pillars as the bootstrap; only the solver differs.
        Starts from ``start`` (default: the current pillars). Raises PricingError on failure.
        """
        with ql_evaluation_date(self.valuation_date):
            evaluator = CalibrationRateEvaluator(self.helper_bindings)
            targets = np.asarray(self.target_rates())
            anchor = self.pillar_values[0]
            initial = self.pillar_values if start is None else tuple(start)

            def residuals(log_values: FloatArray) -> FloatArray:
                curve = ql.YieldTermStructureHandle(
                    self.explicit_curve((anchor, *np.exp(log_values)))
                )
                discount = curve if discount_curve is None else discount_curve
                return np.asarray(evaluator.rates(curve, discount)) - targets

            solution = solve_calibration_system(residuals, np.log(np.asarray(initial[1:])))
        return (anchor, *(float(value) for value in np.exp(solution)))

    def adopt_global_solution(self, pillar_values: Sequence[float]) -> None:
        """Make a global solution the calibrated pillars used for pricing and risk."""
        values = tuple(float(value) for value in pillar_values)
        if len(values) != len(self.pillar_dates) or values[0] != self.pillar_values[0]:
            raise ValueError(f"Global solution does not match the {self.curve_key} pillars")
        self.pillar_values = values
        self.globally_solved = True

    def pricing_curve(self) -> ql.YieldTermStructure:
        """The calibrated curve: the native bootstrap, or the explicit global solution."""
        if self.globally_solved:
            return self.explicit_curve(self.pillar_values)
        return self.native_curve

    def explicit_curve(self, pillar_values: Sequence[float]) -> ql.YieldTermStructure:
        """Create an explicit curve at the retained pillars with the original interpolation."""
        values = [float(value) for value in pillar_values]
        if len(values) != len(self.pillar_dates):
            raise ValueError(
                f"Expected {len(self.pillar_dates)} pillar values for {self.curve_key}, "
                f"got {len(values)}"
            )
        dates = list(self.pillar_dates)
        if self.interpolation == CurveInterpolation.LINEAR_FORWARD:
            curve = ql.DiscountCurve(dates, values, self.day_counter)
        elif self.interpolation == CurveInterpolation.SMOOTH_FORWARD:
            curve = ql.MonotonicLogParabolicCubicDiscountCurve(
                dates,
                values,
                self.day_counter,
            )
        elif self.interpolation == CurveInterpolation.NATURAL_LOG_CUBIC:
            curve = ql.NaturalLogCubicDiscountCurve(dates, values, self.day_counter)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")
        curve.enableExtrapolation()
        return curve

    def rebootstrap_and_freeze(
        self,
        shocks: Mapping[RawQuoteKey, float],
        discount_curve: ql.YieldTermStructureHandle | None = None,
    ) -> ql.YieldTermStructureHandle:
        """Apply decimal-rate shocks, rebootstrap, freeze, and restore native base state."""
        bindings_by_key: dict[RawQuoteKey, list[CalibrationHelperBinding]] = {}
        for binding in self.helper_bindings:
            if binding.raw_quote_key is not None:
                bindings_by_key.setdefault(binding.raw_quote_key, []).append(binding)

        unknown_keys = set(shocks).difference(bindings_by_key)
        if unknown_keys:
            unknown = ", ".join(str(key) for key in sorted(unknown_keys))
            raise KeyError(f"Unknown quote coordinates for {self.curve_key}: {unknown}")
        if discount_curve is not None and self.discount_curve_handle is None:
            raise ValueError(f"Curve {self.curve_key} has no external discount-curve dependency")

        mutable_bindings = [
            binding for binding in self.helper_bindings if binding.native_quote is not None
        ]
        base_quote_values = {
            id(binding.native_quote): binding.native_quote.value()
            for binding in mutable_bindings
            if binding.native_quote is not None
        }
        base_discount_curve = (
            self.discount_curve_handle.currentLink()
            if self.discount_curve_handle is not None
            else None
        )

        rebind_helpers = False
        with ql_evaluation_date(self.valuation_date):
            try:
                if discount_curve is not None and self.discount_curve_handle is not None:
                    self.discount_curve_handle.linkTo(discount_curve.currentLink())
                for key, shock in shocks.items():
                    for binding in bindings_by_key[key]:
                        quote = binding.native_quote
                        if quote is None:
                            raise ValueError(f"Quote {key} for {self.curve_key} is not mutable")
                        base_value = base_quote_values[id(quote)]
                        native_shock = (
                            -100.0 * shock if binding.quote_kind == "futures_price" else shock
                        )
                        quote.setValue(base_value + native_shock)
                if self.globally_solved:
                    # QuantLib's bootstrap already failed on the base quotes: solve the shocked
                    # system from the base solution, without rebinding any helper.
                    frozen = freeze_curve(self._solve_scenario_globally(), self.day_counter)
                    return ql.YieldTermStructureHandle(frozen)
                # A fresh bootstrap starts from QuantLib's canonical guesses. Recalculating the
                # retained curve would start from its last state, making every scenario depend
                # on the ones computed before it (within the bootstrap accuracy).
                rebind_helpers = True
                scenario_curve = get_piecewise_curve(
                    [binding.helper for binding in self.helper_bindings],
                    self.pillar_dates[0],
                    self.day_counter,
                    self.interpolation,
                    self.bootstrap_accuracy,
                )
                scenario_curve.enableExtrapolation()
                scenario_curve.discount(scenario_curve.maxDate())
                smooth = self.interpolation == CurveInterpolation.SMOOTH_FORWARD
                # A SmoothForward bootstrap either converges (~1e-7 bp) or cycles (bp-sized): a
                # risk scenario must be an exact calibration, not one within publication tolerance.
                threshold = SCENARIO_CONVERGENCE_BPS if smooth else self.tolerance_bps
                failure = self._active_helper_residual_failure(threshold)
                if failure is None:
                    frozen = freeze_curve(scenario_curve, self.day_counter)
                elif smooth:
                    frozen = freeze_curve(self._solve_scenario_globally(), self.day_counter)
                else:
                    raise PricingError(failure)
                return ql.YieldTermStructureHandle(frozen)
            finally:
                for binding in mutable_bindings:
                    quote = binding.native_quote
                    if quote is not None:
                        quote.setValue(base_quote_values[id(quote)])
                if self.discount_curve_handle is not None and base_discount_curve is not None:
                    self.discount_curve_handle.linkTo(base_discount_curve)
                if rebind_helpers:
                    # Helpers hold a raw pointer to the last curve that bootstrapped them:
                    # rebind them to the retained curve before the scenario curve is released.
                    self.native_curve.recalculate()
                    self.native_curve.discount(self.native_curve.maxDate())

    def native_state_values(self) -> tuple[float, ...]:
        """Current pillars of the retained native curve, which its helpers reprice now.

        An unconverged SmoothForward iterate moves on each recalculation, so this can differ
        from the construction-time ``native_pillar_values``.
        """
        with ql_evaluation_date(self.valuation_date):
            self.native_curve.discount(self.native_curve.maxDate())
            return tuple(float(value) for value in self.native_curve.data())

    def verify_evaluator_fidelity(
        self, discount_curve: ql.YieldTermStructureHandle | None = None
    ) -> None:
        """Raise unless the rate evaluator reproduces the native helpers on their own curve.

        The global solver and its report rely on the evaluator, so it must first match
        QuantLib's helpers where both can be evaluated.
        """
        with ql_evaluation_date(self.valuation_date):
            native = ql.YieldTermStructureHandle(self.explicit_curve(self.native_state_values()))
            rates = CalibrationRateEvaluator(self.helper_bindings).rates(
                native, native if discount_curve is None else discount_curve
            )
            worst = 0.0
            for binding, rate in zip(self.helper_bindings, rates):
                implied = binding.helper.impliedQuote()
                if binding.quote_kind == "futures_price":
                    implied = (100.0 - implied) / 100.0
                worst = max(worst, abs(rate - implied))
        if not worst <= EVALUATOR_FIDELITY_TOLERANCE:
            raise PricingError(
                f"Calibration evaluator does not reproduce the {self.curve_key} helpers "
                f"(error {worst:.3g})"
            )

    def _solve_scenario_globally(self) -> ql.YieldTermStructure:
        """Solve the current (shocked) quotes from the base solution and check the result."""
        values = self.solve_globally(self.discount_curve_handle, self.pillar_values)
        curve = self.explicit_curve(values)
        handle = ql.YieldTermStructureHandle(curve)
        discount = handle if self.discount_curve_handle is None else self.discount_curve_handle
        # Checked on cloned QuantLib instruments, independently of the telescopic OIS plans.
        rates = evaluate_calibration_rates(self.helper_bindings, handle, discount)
        worst_bps = 1.0e4 * max(
            abs(rate - target) for rate, target in zip(rates, self.target_rates())
        )
        if not worst_bps <= GLOBAL_SOLUTION_CHECK_BPS:
            raise PricingError(
                f"Global scenario calibration for {self.curve_key} leaves a {worst_bps:.3g} bp "
                "residual on the QuantLib instruments"
            )
        return curve

    def _active_helper_residual_failure(self, threshold_bps: float) -> str | None:
        """Why a non-throwing bootstrap's active helpers do not reprice, or None if they do."""
        for binding in self.helper_bindings:
            quote = binding.native_quote
            if quote is None:
                raise PricingError(
                    f"Active helper {binding.source_label} for {self.curve_key} has no quote"
                )
            try:
                implied_quote = binding.helper.impliedQuote()
            except RuntimeError as exc:
                return (
                    f"Shocked calibration for {self.curve_key} could not reprice "
                    f"{binding.source_label}: {exc}"
                )
            if binding.quote_kind == "futures_price":
                implied_rate = (100.0 - implied_quote) / 100.0
                market_rate = (100.0 - quote.value()) / 100.0
            else:
                implied_rate = implied_quote
                market_rate = quote.value()
            residual_bps = (implied_rate - market_rate) * 10_000.0
            if abs(residual_bps) > threshold_bps:
                return (
                    f"Shocked calibration for {self.curve_key} failed at "
                    f"{binding.source_label}: {residual_bps:g} bp residual exceeds "
                    f"{threshold_bps:g} bp"
                )
        return None
