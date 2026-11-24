"""Quote-level diagnostics produced while a QuantLib curve is still bootstrapped."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

import QuantLib as ql

from .effective_conventions import EffectiveHelperSpec
from .helpers import ql_to_date
from .schemas import CurveQuoteExclusion

if TYPE_CHECKING:
    from .calibration_context import CurveCalibrationContext, RawQuoteKey

MARKET_QUOTE_TOLERANCE_BPS = 0.2


@dataclass(frozen=True)
class CalibrationHelperBinding:
    """Bind one QuantLib helper to the market quote that created it."""

    helper: ql.RateHelper
    source_label: str
    tenor: str
    instrument_type: str
    convention_id: str
    market_rate: float
    day_counter: ql.DayCounter
    quote_kind: Literal["rate", "futures_price"] = "rate"
    quote_index: int | None = None
    native_quote: ql.SimpleQuote | None = None
    raw_quote_key: RawQuoteKey | None = None
    convexity_adjustment: float = 0.0
    projection_index: ql.InterestRateIndex | None = None
    quote_provenance: str = "market_quote"
    calibration_role: str = "helper"


@dataclass(frozen=True)
class CurveCalibrationResidual:
    """Implied-quote residual for one helper used by the bootstrap."""

    source_label: str
    tenor: str
    instrument_type: str
    convention_id: str
    market_rate: float
    implied_rate: float
    residual_bps: float
    pillar_date: date
    quote_provenance: str = "market_quote"
    calibration_role: str = "helper"


@dataclass(frozen=True)
class CurveFixingObservation:
    """Fixing retained with the snapshot but deliberately excluded from helpers."""

    source_label: str
    tenor: str
    convention_id: str
    market_rate: float
    used_in_bootstrap: bool = False
    quote_provenance: str = "fixing_observed"
    calibration_role: str = "observation"


@dataclass(frozen=True)
class CurveCalibrationReport:
    """Calibration evidence for one built curve."""

    curve_key: str
    residuals: tuple[CurveCalibrationResidual, ...]
    fixing_observations: tuple[CurveFixingObservation, ...]
    tolerance_bps: float = MARKET_QUOTE_TOLERANCE_BPS
    excluded_quotes: tuple[CurveQuoteExclusion, ...] = ()
    # "global_newton" when QuantLib's iterative bootstrap did not reprice the quotes.
    solver: Literal["quantlib_iterative", "global_newton"] = "quantlib_iterative"

    @property
    def max_abs_residual_bps(self) -> float:
        """Return the largest absolute helper residual, or zero for an empty report."""
        absolute_residuals = [abs(point.residual_bps) for point in self.residuals]
        if any(not math.isfinite(residual) for residual in absolute_residuals):
            return math.inf
        return max(absolute_residuals, default=0.0)

    @property
    def passed(self) -> bool:
        """Whether every helper quote is inside the market acceptance tolerance."""
        return self.max_abs_residual_bps <= self.tolerance_bps


@dataclass(frozen=True)
class CurveBuildResult:
    """Frozen curve handle and the diagnostics computed before freezing."""

    curve: ql.YieldTermStructureHandle
    report: CurveCalibrationReport
    effective_helper_specs: tuple[EffectiveHelperSpec, ...] = ()
    effective_bootstrap_accuracy: float | None = None
    curve_day_count: str | None = None
    calibration_context: CurveCalibrationContext | None = None


def build_calibration_report(
    curve_key: str,
    helper_bindings: list[CalibrationHelperBinding],
    fixing_observations: list[CurveFixingObservation],
    tolerance_bps: float = MARKET_QUOTE_TOLERANCE_BPS,
    excluded_quotes: tuple[CurveQuoteExclusion, ...] = (),
    implied_rates: Sequence[float] | None = None,
    solver: Literal["quantlib_iterative", "global_newton"] = "quantlib_iterative",
) -> CurveCalibrationReport:
    """Calculate quote residuals after calibration.

    Residuals come from the bootstrapped helpers, or from ``implied_rates`` (decimal, in
    binding order) when the calibrated curve is not the helpers' own piecewise curve.
    """
    if implied_rates is not None and len(implied_rates) != len(helper_bindings):
        raise ValueError("One implied rate is required per calibration helper")
    residuals = tuple(
        _calibration_residual(binding, None if implied_rates is None else implied_rates[index])
        for index, binding in enumerate(helper_bindings)
    )
    return CurveCalibrationReport(
        curve_key=curve_key,
        residuals=residuals,
        fixing_observations=tuple(fixing_observations),
        tolerance_bps=tolerance_bps,
        excluded_quotes=excluded_quotes,
        solver=solver,
    )


def _calibration_residual(
    binding: CalibrationHelperBinding,
    implied_rate: float | None = None,
) -> CurveCalibrationResidual:
    """Normalize QuantLib helper quote units to decimal rates."""
    if implied_rate is None:
        implied_quote = binding.helper.impliedQuote()
        implied_rate = (
            (100.0 - implied_quote) / 100.0
            if binding.quote_kind == "futures_price"
            else implied_quote
        )
    return CurveCalibrationResidual(
        source_label=binding.source_label,
        tenor=binding.tenor,
        instrument_type=binding.instrument_type,
        convention_id=binding.convention_id,
        market_rate=binding.market_rate,
        implied_rate=implied_rate,
        residual_bps=(implied_rate - binding.market_rate) * 10_000.0,
        pillar_date=ql_to_date(binding.helper.pillarDate()),
        quote_provenance=binding.quote_provenance,
        calibration_role=binding.calibration_role,
    )
