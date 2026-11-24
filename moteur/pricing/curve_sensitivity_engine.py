"""
Shared curve sensitivity engine for pricing diagnostics.

The engine owns curve bump construction, quote-level rebootstrap caching, maturity
buckets, quote buckets, and curve risk ladder assembly. Instrument pricers only
provide a typed repricing callback.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import date as dt_date
from functools import wraps
from typing import Callable, Optional, ParamSpec, TypeVar

import numpy as np
import QuantLib as ql

from curves.calibration_context import RawQuoteKey
from curves.calibration_diagnostics import CalibrationHelperBinding, CurveCalibrationReport
from curves.helpers import (
    date_to_ql,
    parse_tenor,
    ql_evaluation_date,
    ql_to_date,
    serialized_ql_state,
)
from curves.yield_curve_builder import CurveSet, YieldCurveBuilder
from market_data.schemas import ForwardMarketData, OISMarketData
from pricing.calibration_fixings import constant_ibor_fixings
from pricing.calibration_risk_context import RiskCalibrationContext, build_risk_calibration_context
from pricing.curve_buckets import (
    DELTA_TIME_BUCKETS,
    bucket_bounds_for_year,
    bucket_label,
    make_bucketed_curve_handle,
)
from pricing.curve_risk_ladder import (
    FAMILY_LABELS,
    FAMILY_ORDER,
    CurveRiskPoint,
    _build_sections,
    _bump_forward_quote,
    _bump_ois_quote,
    _collect_selected_points,
    _selected_curve_keys,
    curve_display_label,
    curve_role_for_key,
    forward_market_data_for_key,
    with_effective_dates,
)
from pricing.diagnostics import (
    CurveInstrumentDelta,
    CurveQuoteBucketDelta,
    CurveRiskColumn,
    CurveRiskLadder,
    KeyRateRiskCurveContribution,
    KeyRateRiskMatrix,
    KeyRateRiskRow,
    parallel_check_warning,
)
from pricing.quote_jacobian_engine import QuoteJacobianArtifact, build_quote_jacobian
from pricing.risk_cache import (
    IdentityLruCache,
    clear_risk_caches,
    create_risk_cache,
    risk_context_fingerprint,
)
from services.exceptions import PricingError

DeltaConvention = str
RepriceFn = Callable[[ql.YieldTermStructureHandle, ql.YieldTermStructureHandle], "RepriceOutput"]
LegDeltaAllocator = Callable[["RepriceOutput", "RepriceOutput", float], tuple[float, float]]
QUOTE_BOOTSTRAP_ACCURACY = 1.0e-12
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _discount_interpolation(curve_set: CurveSet) -> str:
    return getattr(curve_set, "discount_interpolation", curve_set.interpolation)


def _forward_interpolation(curve_set: CurveSet) -> str:
    return getattr(curve_set, "forward_interpolation", curve_set.interpolation)


def _filter_excluded_quote_points(
    curve_set: CurveSet,
    points: dict[tuple[str, str, str], CurveRiskPoint],
    *,
    discount_key: str,
    forward_key: str,
) -> dict[tuple[str, str, str], CurveRiskPoint]:
    """Remove a quote only when every selected curve role excludes it."""
    prefix = f"{curve_set.currency.upper()}_"

    def report_excludes(report_key: str, point: CurveRiskPoint) -> bool:
        report = curve_set.calibration_reports.get(report_key)
        if report is None:
            return False
        return any(
            exclusion.curve_key.upper().strip().removeprefix(prefix) == point.curve_key
            and exclusion.quote_index == point.index
            for exclusion in report.excluded_quotes
        )

    def all_consuming_roles_exclude(point: CurveRiskPoint) -> bool:
        if point.field != "futures":
            return False
        role_decisions: list[bool] = []
        if point.curve_key == forward_key:
            role_decisions.append(report_excludes(point.curve_key, point))
        if point.curve_key == discount_key:
            discount_report_key = (
                f"discount_{point.curve_key}"
                if f"discount_{point.curve_key}" in curve_set.calibration_reports
                else point.curve_key
            )
            role_decisions.append(report_excludes(discount_report_key, point))
        return bool(role_decisions) and all(role_decisions)

    return {key: point for key, point in points.items() if not all_consuming_roles_exclude(point)}


def _exclusion_signature(report: CurveCalibrationReport) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted((exclusion.curve_key, exclusion.quote_index) for exclusion in report.excluded_quotes)
    )


def _build_forward_curve_for_risk(
    *,
    builder: YieldCurveBuilder,
    curve_set: CurveSet,
    market_data: ForwardMarketData,
    discount_curve: ql.YieldTermStructureHandle,
    interpolation: str,
    bootstrap_accuracy: float,
    expected_report_key: str | None,
) -> ql.YieldTermStructureHandle:
    """Build one risk curve and reject a finite difference that changes its quote set."""
    result = builder.build_forward_curve_with_report(
        market_data,
        discount_curve,
        bootstrap_accuracy=bootstrap_accuracy,
        interpolation=interpolation,
    )
    expected_report = (
        curve_set.calibration_reports.get(expected_report_key)
        if expected_report_key is not None
        else None
    )
    if expected_report is not None and _exclusion_signature(result.report) != _exclusion_signature(
        expected_report
    ):
        raise ValueError(
            f"Curve-risk scenario for {result.report.curve_key} changed the calibrated quote set; "
            "the finite difference is rejected."
        )
    return result.curve


def _calibration_reports_with_shifted_exclusion_rates(
    curve_set: CurveSet,
    forward_market_data: dict[str, ForwardMarketData],
) -> dict[str, CurveCalibrationReport]:
    """Copy reports while aligning notified excluded rates with shifted raw quotes."""
    prefix = f"{curve_set.currency.upper()}_"
    reports: dict[str, CurveCalibrationReport] = {}
    for report_key, report in curve_set.calibration_reports.items():
        exclusions = []
        for exclusion in report.excluded_quotes:
            curve_key = exclusion.curve_key.upper().strip().removeprefix(prefix)
            market_data = forward_market_data_for_key(forward_market_data, curve_key)
            if market_data is None or exclusion.quote_index >= len(market_data.futures):
                exclusions.append(exclusion)
                continue
            exclusions.append(
                exclusion.model_copy(
                    update={"market_rate": market_data.futures[exclusion.quote_index].rate}
                )
            )
        reports[report_key] = replace(report, excluded_quotes=tuple(exclusions))
    return reports


@dataclass(frozen=True)
class RepriceOutput:
    """Instrument repricing result under a curve scenario."""

    npv: float
    fixed_leg_npv: Optional[float] = None
    float_leg_npv: Optional[float] = None


@dataclass(frozen=True)
class CurveScenario:
    """Forward and discount curve handles for one bumped scenario."""

    forward_curve: ql.YieldTermStructureHandle
    discount_curve: ql.YieldTermStructureHandle


@dataclass(frozen=True)
class QuoteCalibrationBase:
    """Tightly calibrated base curves used only for quote-risk finite differences."""

    scenario: CurveScenario
    ois_curve: ql.YieldTermStructureHandle


@dataclass(frozen=True)
class QuoteSensitivity:
    """Local market-quote DV01 (-1e-4 dV/dq), with signed leg contributions.

    The independent finite-bump reference also uses this container in regression tests.
    """

    point: CurveRiskPoint
    dv01: float
    pay_leg_dv01: float
    receive_leg_dv01: float
    discount_role_dv01: float = 0.0
    forward_role_dv01: float = 0.0
    pay_leg_discount_dv01: float = 0.0
    receive_leg_discount_dv01: float = 0.0
    pay_leg_forward_dv01: float = 0.0
    receive_leg_forward_dv01: float = 0.0


@dataclass(frozen=True)
class QuoteRiskTotals:
    """Sums over all active quotes of the ``QuoteSensitivity`` values, in the same fields."""

    dv01: float = 0.0
    pay_leg_dv01: float = 0.0
    receive_leg_dv01: float = 0.0
    discount_role_dv01: float = 0.0
    forward_role_dv01: float = 0.0
    pay_leg_discount_dv01: float = 0.0
    receive_leg_discount_dv01: float = 0.0
    pay_leg_forward_dv01: float = 0.0
    receive_leg_forward_dv01: float = 0.0


DV01_SOURCE_MARKET_QUOTES = "market_quote_jacobian"
DV01_SOURCE_ZERO_SPREAD_FALLBACK = "parallel_zero_spread_fallback"
PARALLEL_CHECK_ABS_TOLERANCE = 0.01
PARALLEL_CHECK_REL_TOLERANCE = 5.0e-3


@dataclass(frozen=True)
class CurveSensitivityResult:
    """Full curve diagnostics produced by the shared sensitivity engine.

    ``quote_dv01_total`` is the plain sum of local market-quote DV01s. The parallel
    market-quote shock is an independent check and never rescales any other value.
    ``delta_parallel`` keeps each caller's legacy convention for its historical ``delta``.
    ``quote_sensitivities`` is filled only with diagnostics; ``quote_totals`` always is.
    """

    delta_parallel: float
    parallel_output: RepriceOutput
    bucketed_curve_deltas: dict[str, float]
    raw_bucketed_curve_deltas: dict[str, float]
    bucket_scaling_factor: Optional[float]
    curve_bucketed_deltas: dict[str, float]
    raw_curve_bucketed_deltas: dict[str, float]
    curve_bucket_scaling_factor: Optional[float]
    quote_sensitivities: list[QuoteSensitivity]
    quote_scaling_factor: Optional[float]
    curve_instrument_deltas: list[CurveInstrumentDelta]
    curve_quote_bucket_deltas: list[CurveQuoteBucketDelta]
    curve_risk_ladder: Optional[CurveRiskLadder]
    key_rate_risk: Optional[KeyRateRiskMatrix]
    unbucketed_quote_dv01: float = 0.0
    unbucketed_quote_count: int = 0
    quote_dv01_total: float = 0.0
    parallel_dv01: float = 0.0
    parallel_check_residual: float = 0.0
    parallel_check_tolerance: float = PARALLEL_CHECK_ABS_TOLERANCE
    parallel_check_passed: bool = True
    dv01_source: str = DV01_SOURCE_MARKET_QUOTES
    warnings: tuple[str, ...] = ()
    quote_totals: QuoteRiskTotals = QuoteRiskTotals()


_CURVE_BUMP_CACHE: IdentityLruCache[CurveScenario] = create_risk_cache(max_entries=512)
_MARKET_SCENARIO_CACHE: IdentityLruCache[CurveSet] = create_risk_cache(max_entries=64)
_QUOTE_BASE_CACHE: IdentityLruCache[QuoteCalibrationBase] = create_risk_cache(max_entries=32)
# Artifacts hold explicit curves and small matrices only; room for one per (market, roles).
_QUOTE_JACOBIAN_CACHE: IdentityLruCache[QuoteJacobianArtifact] = create_risk_cache(max_entries=32)
_QUOTE_HELPER_DATES_CACHE: IdentityLruCache[dict[tuple[str, str, int], dt_date]] = (
    create_risk_cache(max_entries=32)
)
# Native calibration graphs live only inside a risk_calibration_scope (see below).
_SCOPED_RISK_CONTEXTS: ContextVar[IdentityLruCache[RiskCalibrationContext] | None] = ContextVar(
    "SCOPED_RISK_CONTEXTS", default=None
)


@contextmanager
def risk_calibration_scope() -> Iterator[None]:
    """Share native calibration graphs within one risk computation, then release them.

    Their relative-date QuantLib helpers re-initialize on every change of the process-global
    evaluation date (about 85 us per OIS helper). A process-wide cache of these graphs made
    each later date change pay for every cached market. Frozen curves, scenarios and Jacobian
    artifacts, which hold no helper, stay cached across computations. Nested scopes share
    the outermost one.
    """
    if _SCOPED_RISK_CONTEXTS.get() is not None:
        yield
        return
    # Deliberately not registered with the process-wide invalidation set: the scope is short
    # lived, and registering would mutate that set while another thread may iterate it.
    token = _SCOPED_RISK_CONTEXTS.set(IdentityLruCache(max_entries=8))
    try:
        yield
    finally:
        _SCOPED_RISK_CONTEXTS.reset(token)


def within_risk_calibration_scope(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run a risk computation inside one shared risk_calibration_scope."""

    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with risk_calibration_scope():
            return function(*args, **kwargs)

    return wrapper


def clear_curve_sensitivity_cache() -> None:
    """Clear cached quote-bumped curve handles after market curves are rebuilt or cleared."""
    clear_risk_caches()


def delta_from_outputs(
    base_output: RepriceOutput,
    bumped_output: RepriceOutput,
    delta_convention: DeltaConvention,
) -> float:
    """Return a legacy finite-difference delta under the requested sign convention.

    Only historical ``delta`` fields use this; local quote DV01 uses the calibration Jacobian.
    """
    if delta_convention == "bumped_minus_base":
        return bumped_output.npv - base_output.npv
    return base_output.npv - bumped_output.npv


def central_dv01(
    down_output: RepriceOutput,
    up_output: RepriceOutput,
    bump_size: float,
) -> float:
    """Signed DV01 in currency per bp: ``[V(q - h) - V(q + h)] / (2 * h_bp)``."""
    bump_bp = bump_size * 10_000.0
    if bump_bp <= 0.0:
        raise ValueError("DV01 bump size must be strictly positive")
    return (down_output.npv - up_output.npv) / (2.0 * bump_bp)


def _central_leg_dv01(
    down_output: RepriceOutput,
    up_output: RepriceOutput,
    bump_size: float,
    leg_delta_allocator: Optional[LegDeltaAllocator],
    dv01: float,
) -> tuple[float, float]:
    """Split a central DV01 into (pay, receive) legs with the same denominator."""
    if leg_delta_allocator is None:
        return 0.0, dv01
    bump_bp = bump_size * 10_000.0
    pay, receive = leg_delta_allocator(down_output, up_output, dv01 * 2.0 * bump_bp)
    return pay / (2.0 * bump_bp), receive / (2.0 * bump_bp)


def _parallel_check(
    parallel_dv01: float,
    quote_dv01_total: float,
) -> tuple[float, float, bool]:
    """Residual, tolerance and status of the independent parallel control."""
    residual = parallel_dv01 - quote_dv01_total
    tolerance = max(
        PARALLEL_CHECK_ABS_TOLERANCE,
        PARALLEL_CHECK_REL_TOLERANCE * max(abs(parallel_dv01), abs(quote_dv01_total)),
    )
    return float(residual), float(tolerance), bool(abs(residual) <= tolerance)


@serialized_ql_state
@within_risk_calibration_scope
def compute_curve_sensitivities(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    bump_size: float,
    base_output: RepriceOutput,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    discount_curve_key: str,
    forward_curve_key: str,
    reprice_with_curves: RepriceFn,
    delta_convention: DeltaConvention,
    parallel_scenario_override: Optional[CurveScenario] = None,
    parallel_down_scenario_override: Optional[CurveScenario] = None,
    central_difference: bool = False,
    constant_ibor_fixing: bool = False,
    bucket_grid: tuple[tuple[float, float], ...] = DELTA_TIME_BUCKETS,
    curve_bucket_grid: Optional[tuple[tuple[float, float], ...]] = None,
    leg_delta_allocator: Optional[LegDeltaAllocator] = None,
    net_only_key_rate_risk: bool = False,
    include_diagnostics: bool = True,
    use_tangent_for_summary: bool = True,
) -> CurveSensitivityResult:
    """
    Compute the product DV01 and curve diagnostics for any instrument repriceable from two curves.

    DV01 is the plain sum of local market-quote DV01s from implicit calibration derivatives,
    taken quote by quote with diagnostics and as one tangent direction without them.
    Grid-based model callers set use_tangent_for_summary=False to preserve the same
    pillar finite differences with and without details; their report assembly is still skipped.
    A central parallel market-quote shock is computed independently as
    a check and never rescales any value. ``delta_parallel`` keeps the caller's legacy delta
    convention (``delta_convention``, ``central_difference`` and the parallel overrides).
    Calibration Jacobians and explicit parameter scenarios are shared across compatible trades.
    """
    if curve_set.ois_market_data is None or not _quote_level_sensitivities_enabled(curve_set):
        raise PricingError("Local market-quote DV01 requires retained calibration market data")
    legacy_up = parallel_scenario_override or _parallel_shift_scenario(
        base_forward_curve,
        base_discount_curve,
        bump_size,
    )
    parallel_output = reprice_with_curves(legacy_up.forward_curve, legacy_up.discount_curve)
    delta_parallel = delta_from_outputs(base_output, parallel_output, delta_convention)
    legacy_down_output: RepriceOutput | None = None
    if central_difference:
        legacy_down = parallel_down_scenario_override or _parallel_shift_scenario(
            base_forward_curve, base_discount_curve, -bump_size
        )
        legacy_down_output = reprice_with_curves(
            legacy_down.forward_curve,
            legacy_down.discount_curve,
        )
        delta_parallel = (
            delta_from_outputs(legacy_down_output, parallel_output, delta_convention) / 2
        )

    if (
        parallel_scenario_override is not None
        and parallel_down_scenario_override is not None
        and legacy_down_output is not None
    ):
        check_up_output, check_down_output = parallel_output, legacy_down_output
    else:
        check_up = _independent_parallel_scenario(
            curve_set=curve_set,
            currency=currency,
            curve_date=curve_date,
            bump_size=bump_size,
            discount_curve_key=discount_curve_key,
            forward_curve_key=forward_curve_key,
            base_discount_curve=base_discount_curve,
            base_forward_curve=base_forward_curve,
            constant_ibor_fixing=constant_ibor_fixing,
        )
        check_down = _independent_parallel_scenario(
            curve_set=curve_set,
            currency=currency,
            curve_date=curve_date,
            bump_size=-bump_size,
            discount_curve_key=discount_curve_key,
            forward_curve_key=forward_curve_key,
            base_discount_curve=base_discount_curve,
            base_forward_curve=base_forward_curve,
            constant_ibor_fixing=constant_ibor_fixing,
        )
        check_up_output = reprice_with_curves(check_up.forward_curve, check_up.discount_curve)
        check_down_output = reprice_with_curves(
            check_down.forward_curve,
            check_down.discount_curve,
        )
    parallel_dv01 = central_dv01(check_down_output, check_up_output, bump_size)

    quote_level_enabled = _quote_level_sensitivities_enabled(curve_set)
    quote_risk_arguments = {
        "curve_set": curve_set,
        "currency": currency,
        "curve_date": curve_date,
        "discount_curve_key": discount_curve_key,
        "forward_curve_key": forward_curve_key,
        "reprice_with_curves": reprice_with_curves,
        "leg_delta_allocator": leg_delta_allocator,
        "constant_ibor_fixing": constant_ibor_fixing,
    }
    # Smooth prices need only one tangent direction for the aggregate DV01. Grid-based
    # model prices retain the pillar differences: crossing interpolation cells in a single
    # direction can differ from summing their derivatives (Griewank and Walther, 2008, ch. 3).
    quote_sensitivities: list[QuoteSensitivity] = []
    if include_diagnostics or not use_tangent_for_summary:
        quote_sensitivities = _compute_quote_sensitivities_jacobian(**quote_risk_arguments)
        if not quote_sensitivities:
            raise PricingError("Local market-quote DV01 has no active calibration quotes")
        quote_totals = _sum_quote_risk(quote_sensitivities)
        if not include_diagnostics:
            quote_sensitivities = []
    else:
        quote_totals = _compute_quote_risk_totals_tangent(**quote_risk_arguments)
    warnings: tuple[str, ...] = ()
    quote_dv01_total = quote_totals.dv01
    dv01_source = DV01_SOURCE_MARKET_QUOTES
    residual, tolerance, passed = _parallel_check(parallel_dv01, quote_dv01_total)
    if not passed:
        warnings = (parallel_check_warning(parallel_dv01, quote_dv01_total, residual, tolerance),)

    raw_bucketed: dict[str, float] = {}
    raw_curve_bucketed: dict[str, float] = {}
    curve_instrument_deltas: list[CurveInstrumentDelta] = []
    curve_quote_bucket_deltas: list[CurveQuoteBucketDelta] = []
    curve_risk_ladder: CurveRiskLadder | None = None
    key_rate_risk: KeyRateRiskMatrix | None = None
    unbucketed_quote_dv01 = 0.0
    unbucketed_quote_count = 0
    if include_diagnostics:
        raw_bucketed = _compute_bucketed_deltas(
            base_discount_curve=base_discount_curve,
            base_forward_curve=base_forward_curve,
            bump_size=bump_size,
            bucket_grid=bucket_grid,
            reprice_with_curves=reprice_with_curves,
        )
        if curve_bucket_grid is not None:
            raw_curve_bucketed = _compute_bucketed_deltas(
                base_discount_curve=base_discount_curve,
                base_forward_curve=base_forward_curve,
                bump_size=bump_size,
                bucket_grid=curve_bucket_grid,
                reprice_with_curves=reprice_with_curves,
            )
        curve_instrument_deltas = _to_curve_instrument_deltas(quote_sensitivities)
        curve_quote_bucket_deltas = bucket_curve_instrument_deltas(
            curve_instrument_deltas,
            curve_date,
        )
        unbucketed_quote_count = len(curve_instrument_deltas) - sum(
            bucket.quote_count for bucket in curve_quote_bucket_deltas
        )
        if unbucketed_quote_count:
            unbucketed_quote_dv01 = quote_dv01_total - sum(
                bucket.quote_dv01 for bucket in curve_quote_bucket_deltas
            )
        # Without raw quotes there is nothing to lay out: no ladder or KRR is fabricated.
        if quote_level_enabled and quote_sensitivities:
            curve_risk_ladder = _assemble_curve_risk_ladder_from_quote_sensitivities(
                currency=currency,
                bump_size=bump_size,
                quote_dv01_total=quote_dv01_total,
                discount_curve_key=discount_curve_key,
                forward_curve_key=forward_curve_key,
                quote_sensitivities=quote_sensitivities,
            )
            if leg_delta_allocator is not None or net_only_key_rate_risk:
                key_rate_risk = _assemble_key_rate_risk(
                    currency=currency,
                    bump_size=bump_size,
                    discount_curve_key=discount_curve_key,
                    forward_curve_key=forward_curve_key,
                    quote_sensitivities=quote_sensitivities,
                    net_only=net_only_key_rate_risk,
                )

    return CurveSensitivityResult(
        delta_parallel=delta_parallel,
        parallel_output=parallel_output,
        bucketed_curve_deltas=dict(raw_bucketed),
        raw_bucketed_curve_deltas=raw_bucketed,
        bucket_scaling_factor=None,
        curve_bucketed_deltas=dict(raw_curve_bucketed),
        raw_curve_bucketed_deltas=raw_curve_bucketed,
        curve_bucket_scaling_factor=None,
        quote_sensitivities=quote_sensitivities,
        quote_scaling_factor=None,
        curve_instrument_deltas=curve_instrument_deltas,
        curve_quote_bucket_deltas=curve_quote_bucket_deltas,
        curve_risk_ladder=curve_risk_ladder,
        key_rate_risk=key_rate_risk,
        unbucketed_quote_dv01=unbucketed_quote_dv01,
        unbucketed_quote_count=unbucketed_quote_count,
        quote_dv01_total=quote_dv01_total,
        parallel_dv01=parallel_dv01,
        parallel_check_residual=residual,
        parallel_check_tolerance=tolerance,
        parallel_check_passed=passed,
        dv01_source=dv01_source,
        warnings=warnings,
        quote_totals=quote_totals,
    )


def curve_diagnostics_fields(result: CurveSensitivityResult) -> dict[str, object]:
    """Common ``BucketedCurveDeltaDiagnostics`` fields for every product.

    Quote values follow the local DV01 sign convention. ``delta_parallel`` and the
    ``parallel_minus_*`` differences are legacy wire fields that now refer to the independent
    parallel check. Scaling factors are always ``None`` and ``scaled_*`` equal their exact value.
    """
    has_quotes = bool(result.curve_instrument_deltas)
    quote_sum = result.quote_dv01_total if has_quotes else None
    bucket_sum = sum(result.bucketed_curve_deltas.values())
    curve_bucket_sum = sum(result.curve_bucketed_deltas.values())
    has_curve_buckets = bool(result.curve_bucketed_deltas)
    parallel = result.parallel_dv01
    return {
        "bump_bp": 1.0,
        "delta_parallel": parallel,
        "delta_bucket_sum": bucket_sum,
        "delta_parallel_minus_bucket_sum": parallel - bucket_sum,
        "bucketed_curve_deltas": result.bucketed_curve_deltas,
        "raw_bucketed_dv01": result.raw_bucketed_curve_deltas,
        "raw_bucket_sum": bucket_sum,
        "scaled_bucket_sum": bucket_sum,
        "bucket_scaling_factor": None,
        "parallel_minus_raw_bucket_sum": parallel - bucket_sum,
        "curve_bucketed_deltas": result.curve_bucketed_deltas,
        "raw_curve_bucketed_dv01": result.raw_curve_bucketed_deltas,
        "curve_bucket_sum": curve_bucket_sum if has_curve_buckets else None,
        "raw_curve_bucket_sum": curve_bucket_sum if has_curve_buckets else None,
        "curve_bucket_scaling_factor": None,
        "delta_parallel_minus_curve_bucket_sum": (
            parallel - curve_bucket_sum if has_curve_buckets else None
        ),
        "parallel_minus_raw_curve_bucket_sum": (
            parallel - curve_bucket_sum if has_curve_buckets else None
        ),
        "delta_instrument_sum": quote_sum,
        "delta_parallel_minus_instrument_sum": (
            parallel - quote_sum if quote_sum is not None else None
        ),
        "curve_instrument_deltas": result.curve_instrument_deltas,
        "curve_quote_bucket_deltas": result.curve_quote_bucket_deltas,
        "curve_risk_ladder": result.curve_risk_ladder,
        "key_rate_risk": result.key_rate_risk,
        "raw_quote_sum": quote_sum,
        "scaled_quote_sum": quote_sum,
        "quote_scaling_factor": None,
        "parallel_minus_raw_quote_sum": parallel - quote_sum if quote_sum is not None else None,
        "unbucketed_quote_dv01": result.unbucketed_quote_dv01,
        "unbucketed_quote_count": result.unbucketed_quote_count,
        "quote_dv01_total": result.quote_dv01_total,
        "parallel_dv01": parallel,
        "parallel_check_residual": result.parallel_check_residual,
        "parallel_check_tolerance": result.parallel_check_tolerance,
        "parallel_check_passed": result.parallel_check_passed,
        "dv01_source": result.dv01_source,
    }


def _independent_parallel_scenario(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    bump_size: float,
    discount_curve_key: str,
    forward_curve_key: str,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    constant_ibor_fixing: bool,
) -> CurveScenario:
    """Parallel market-quote shock with full rebootstrap, or zero spread without raw quotes."""
    scenario = parallel_market_quote_shift_scenario(
        curve_set=curve_set,
        currency=currency,
        curve_date=curve_date,
        bump_size=bump_size,
        discount_curve_key=discount_curve_key,
        forward_curve_key=forward_curve_key,
        base_discount_curve=base_discount_curve,
        base_forward_curve=base_forward_curve,
        constant_ibor_fixing=constant_ibor_fixing,
    )
    if scenario is not None:
        return scenario
    return _parallel_shift_scenario(base_forward_curve, base_discount_curve, bump_size)


def bucket_curve_instrument_deltas(
    instrument_deltas: list[CurveInstrumentDelta],
    reference_date: dt_date,
) -> list[CurveQuoteBucketDelta]:
    """Aggregate quote-level deltas into standard quote maturity buckets."""
    day_counter = ql.Actual365Fixed()
    reference_ql = date_to_ql(reference_date)
    buckets: dict[tuple[str, str, str, float, float], CurveQuoteBucketDelta] = {}

    for entry in instrument_deltas:
        if entry.maturity_date is None:
            continue

        maturity_years = _quote_maturity_years(entry, reference_ql, day_counter)
        bounds = bucket_bounds_for_year(maturity_years)
        if bounds is None:
            continue

        start_year, end_year = bounds
        label = bucket_label(start_year, end_year)
        key = (entry.curve_key, entry.curve_role, label, start_year, end_year)
        existing = buckets.get(key)
        if existing is None:
            buckets[key] = CurveQuoteBucketDelta(
                curve_key=entry.curve_key,
                curve_role=entry.curve_role,
                bucket_label=label,
                start_year=start_year,
                end_year=end_year,
                pay_leg_delta=entry.pay_leg_delta,
                receive_leg_delta=entry.receive_leg_delta,
                quote_dv01=entry.quote_dv01,
                raw_quote_dv01=entry.raw_quote_dv01,
                scaled_quote_dv01=entry.scaled_quote_dv01,
                delta=entry.delta,
                quote_count=1,
            )
            continue

        buckets[key] = existing.model_copy(
            update={
                "pay_leg_delta": existing.pay_leg_delta + entry.pay_leg_delta,
                "receive_leg_delta": existing.receive_leg_delta + entry.receive_leg_delta,
                "quote_dv01": existing.quote_dv01 + entry.quote_dv01,
                "raw_quote_dv01": existing.raw_quote_dv01 + entry.raw_quote_dv01,
                "scaled_quote_dv01": existing.scaled_quote_dv01 + entry.scaled_quote_dv01,
                "delta": existing.delta + entry.delta,
                "quote_count": existing.quote_count + 1,
            }
        )

    return sorted(
        buckets.values(),
        key=lambda entry: (entry.curve_key, entry.curve_role, entry.start_year),
    )


def _quote_maturity_years(
    entry: CurveInstrumentDelta,
    reference_date: ql.Date,
    day_counter: ql.DayCounter,
) -> float:
    """
    Prefer the market tenor for regular quotes so a 50Y quote remains in the 40Y-50Y bucket.

    Calendar-adjusted maturity dates can exceed an exact tenor by a few days because of leap
    years or business-day adjustment. FRA and future labels are not simple periods, so they keep
    using their explicit maturity dates.
    """
    if entry.tenor:
        try:
            tenor = parse_tenor(entry.tenor)
        except ValueError:
            tenor = None
        if tenor is not None:
            if tenor.units() == ql.Days:
                return max(tenor.length() / 365.0, 0.0)
            if tenor.units() == ql.Weeks:
                return max(tenor.length() * 7.0 / 365.0, 0.0)
            if tenor.units() == ql.Months:
                return max(tenor.length() / 12.0, 0.0)
            if tenor.units() == ql.Years:
                return max(float(tenor.length()), 0.0)

    return max(
        0.0,
        day_counter.yearFraction(reference_date, date_to_ql(entry.maturity_date)),
    )


def _parallel_shift_scenario(
    base_forward_curve: ql.YieldTermStructureHandle,
    base_discount_curve: ql.YieldTermStructureHandle,
    bump_size: float,
) -> CurveScenario:
    spread = ql.QuoteHandle(ql.SimpleQuote(bump_size))
    bumped_disc_ts = ql.ZeroSpreadedTermStructure(base_discount_curve, spread)
    bumped_disc_ts.enableExtrapolation()
    bumped_fwd_ts = ql.ZeroSpreadedTermStructure(base_forward_curve, spread)
    bumped_fwd_ts.enableExtrapolation()
    return CurveScenario(
        forward_curve=ql.YieldTermStructureHandle(bumped_fwd_ts),
        discount_curve=ql.YieldTermStructureHandle(bumped_disc_ts),
    )


@serialized_ql_state
@within_risk_calibration_scope
def parallel_market_quote_shift_scenario(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    bump_size: float,
    discount_curve_key: str,
    forward_curve_key: str,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    bump_discount_role: bool = True,
    bump_forward_role: bool = True,
    constant_ibor_fixing: bool = False,
) -> CurveScenario | None:
    """Build a parallel market-quote bump with full curve rebootstrap.

    The selected discount and projection market curves are shifted at every
    bootstrap quote. Role flags allow discount-only and projection-only
    decompositions. ``None`` reports that retained raw calibration quotes are unavailable.
    """
    if not _quote_level_sensitivities_enabled(curve_set):
        return None

    if not bump_discount_role and not bump_forward_role:
        return CurveScenario(base_forward_curve, base_discount_curve)
    if bump_discount_role and bump_forward_role:
        selected_curve_keys = _selected_curve_keys(discount_curve_key, forward_curve_key)
    elif bump_discount_role:
        selected_curve_keys = _selected_curve_keys(discount_curve_key, discount_curve_key)
    else:
        selected_curve_keys = _selected_curve_keys(forward_curve_key, forward_curve_key)

    ois_data = getattr(curve_set, "ois_market_data", None)
    forward_market_data = getattr(curve_set, "forward_market_data", {}) or {}
    points_by_cell = _collect_selected_points(
        currency=currency,
        selected_curve_keys=selected_curve_keys,
        discount_key=discount_curve_key,
        forward_key=forward_curve_key,
        ois_data=ois_data,
        forward_market_data=forward_market_data,
    )
    points_by_cell = _filter_excluded_quote_points(
        curve_set,
        points_by_cell,
        discount_key=discount_curve_key,
        forward_key=forward_curve_key,
    )
    if not points_by_cell:
        return None

    cache_key = (
        _cache_root_curve_set(curve_set),
        *risk_context_fingerprint(curve_set, tuple(selected_curve_keys)),
        constant_ibor_fixing,
        "parallel_market_quotes",
        currency,
        curve_date.isoformat(),
        round(bump_size, 12),
        discount_curve_key,
        forward_curve_key,
        bump_discount_role,
        bump_forward_role,
    )
    cached = _CURVE_BUMP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    shocks = {_point_quote_key(point): bump_size for point in points_by_cell.values()}
    with (
        ql_evaluation_date(curve_date),
        constant_ibor_fixings(curve_set, selected_curve_keys, enabled=constant_ibor_fixing),
    ):
        context = _risk_calibration_context(
            curve_set, discount_curve_key, forward_curve_key, constant_ibor_fixing
        )
        bumped_forward_curve, bumped_discount_curve = context.quote_scenario(shocks)
    scenario = CurveScenario(
        forward_curve=bumped_forward_curve if bump_forward_role else base_forward_curve,
        discount_curve=bumped_discount_curve if bump_discount_role else base_discount_curve,
    )
    _CURVE_BUMP_CACHE[cache_key] = scenario
    return scenario


def _scenario_compatibility_key(
    curve_set: CurveSet,
    shift_bps: float,
    discount_key: str,
    forward_key: str,
) -> tuple[object, ...]:
    """Every input that changes a rate-shocked curve set; equal keys share one frozen object."""
    exclusion_signature = tuple(
        sorted(
            (report_key, _exclusion_signature(report))
            for report_key, report in (getattr(curve_set, "calibration_reports", {}) or {}).items()
        )
    )
    return (
        curve_set,
        *risk_context_fingerprint(
            curve_set, tuple(_selected_curve_keys(discount_key, forward_key))
        ),
        getattr(curve_set, "calibration_anchor", curve_set),
        curve_set.currency,
        curve_set.curve_date,
        _discount_interpolation(curve_set),
        _forward_interpolation(curve_set),
        exclusion_signature,
        discount_key,
        forward_key,
        round(float(shift_bps), 12),
        "constant_calibration_fixings",
    )


@serialized_ql_state
@within_risk_calibration_scope
def market_quote_shift_curve_set(
    curve_set: CurveSet,
    shift_bps: float,
    discount_key: str,
    forward_key: str,
) -> CurveSet:
    """Shift every selected market quote by ``shift_bps``, rebootstrap, keep the shifted quotes.

    The result is a frozen ``CurveSet`` shared by every compatible caller (same source curves,
    interpolations, exclusion set, canonical role keys, signed shift and calibration fixing
    mode). Its market data is the shifted quote set, so local quote DV01 is differentiated
    at ``q + shift`` with the same calibration conventions.
    """
    import math

    if not math.isfinite(shift_bps):
        raise ValueError("Rate shock must be finite")
    if shift_bps == 0:
        return curve_set
    aliases = {"SOFR", "SONIA", "ESTR", "OIS", *curve_set.implied_forward_tenors}
    discount_key = "OIS" if discount_key in aliases else discount_key
    forward_key = "OIS" if forward_key in aliases else forward_key
    selected_curve_keys = _selected_curve_keys(discount_key, forward_key)
    key = (
        *_scenario_compatibility_key(curve_set, shift_bps, discount_key, forward_key),
        *risk_context_fingerprint(curve_set, tuple(selected_curve_keys)),
    )
    if key in _MARKET_SCENARIO_CACHE:
        return _MARKET_SCENARIO_CACHE[key]
    scenario = parallel_market_quote_shift_scenario(
        curve_set=curve_set,
        currency=curve_set.currency,
        curve_date=curve_set.curve_date,
        bump_size=shift_bps / 10000,
        discount_curve_key=discount_key,
        forward_curve_key=forward_key,
        base_discount_curve=curve_set.get_discount_curve(discount_key),
        base_forward_curve=curve_set.get_forward_curve(forward_key),
        constant_ibor_fixing=True,
    )
    if scenario is None:
        raise PricingError("Market-quote shift requires retained calibration quotes")
    ois_data = curve_set.ois_market_data
    forward_data = dict(curve_set.forward_market_data)
    points = _collect_selected_points(
        currency=curve_set.currency,
        selected_curve_keys=selected_curve_keys,
        discount_key=discount_key,
        forward_key=forward_key,
        ois_data=ois_data,
        forward_market_data=forward_data,
    )
    points = _filter_excluded_quote_points(
        curve_set,
        points,
        discount_key=discount_key,
        forward_key=forward_key,
    )
    for point in points.values():
        if point.curve_key == "OIS":
            ois_data = _bump_ois_quote(ois_data, point.field, point.index, shift_bps / 10000)
        else:
            forward_data[point.curve_key] = _bump_forward_quote(
                forward_data[point.curve_key], point.field, point.index, shift_bps / 10000
            )
    ois_scenario = parallel_market_quote_shift_scenario(
        curve_set=curve_set,
        currency=curve_set.currency,
        curve_date=curve_set.curve_date,
        bump_size=shift_bps / 10000,
        discount_curve_key="OIS",
        forward_curve_key="OIS",
        base_discount_curve=curve_set.get_discount_curve("OIS"),
        base_forward_curve=curve_set.get_forward_curve("OIS"),
        constant_ibor_fixing=True,
    )
    if ois_scenario is None:
        raise PricingError("Market-quote shift requires retained OIS calibration quotes")
    handles = dict(curve_set.forward_curves)
    discount_handles = dict(curve_set.discount_curves)
    ois_curve = ois_scenario.discount_curve
    ois_forward_curve = ois_scenario.forward_curve
    if discount_key == "OIS":
        ois_curve = scenario.discount_curve
    else:
        discount_handles[discount_key] = scenario.discount_curve
    if forward_key == "OIS":
        ois_forward_curve = scenario.forward_curve
    else:
        handles[forward_key] = scenario.forward_curve
    calibration_reports = _calibration_reports_with_shifted_exclusion_rates(
        curve_set,
        forward_data,
    )
    shifted = CurveSet(
        currency=curve_set.currency,
        valuation_date=curve_set.curve_date,
        ois_curve=ois_curve,
        forward_curves=handles,
        discount_curves=discount_handles,
        ois_market_data=ois_data,
        forward_market_data=forward_data,
        interpolation=curve_set.interpolation,
        discount_interpolation=_discount_interpolation(curve_set),
        forward_interpolation=_forward_interpolation(curve_set),
        ois_forward_curve=ois_forward_curve,
        calibration_reports=calibration_reports,
        calibration_fixings=curve_set.calibration_fixings,
        observed_calibration_fixings=curve_set.observed_calibration_fixings,
        implied_forward_tenors=curve_set.implied_forward_tenors,
    )
    shifted.calibration_anchor = getattr(curve_set, "calibration_anchor", curve_set)
    _MARKET_SCENARIO_CACHE[key] = shifted
    return shifted


def _compute_bucketed_deltas(
    *,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    bump_size: float,
    bucket_grid: tuple[tuple[float, float], ...],
    reprice_with_curves: RepriceFn,
) -> dict[str, float]:
    """Central zero-rate-spread DV01 per maturity bucket, in the ``central_dv01`` convention.

    These buckets shock the zero curves, not the market quotes: they are a separate diagnostic
    whose sum approximates the zero-spread parallel risk, never a split of the product DV01.
    """
    values: dict[str, float] = {}
    for start_year, end_year in bucket_grid:
        outputs = []
        for direction in (1.0, -1.0):
            bumped_disc = make_bucketed_curve_handle(
                base_discount_curve,
                start_year,
                end_year,
                direction * bump_size,
            )
            bumped_fwd = make_bucketed_curve_handle(
                base_forward_curve,
                start_year,
                end_year,
                direction * bump_size,
            )
            outputs.append(reprice_with_curves(bumped_fwd, bumped_disc))
        up_output, down_output = outputs
        values[bucket_label(start_year, end_year)] = central_dv01(
            down_output,
            up_output,
            bump_size,
        )
    return values


@within_risk_calibration_scope
def _compute_quote_sensitivities(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    bump_size: float,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    discount_curve_key: str,
    forward_curve_key: str,
    reprice_with_curves: RepriceFn,
    leg_delta_allocator: Optional[LegDeltaAllocator],
    constant_ibor_fixing: bool = False,
) -> list[QuoteSensitivity]:
    """Central DV01 of every selected bootstrap quote, with full dependent-curve rebootstrap.

    Both ``q - h`` and ``q + h`` scenarios are built from the same unshifted raw quote set; no
    value is rescaled.
    """
    if not _quote_level_sensitivities_enabled(curve_set):
        return []

    ois_data = getattr(curve_set, "ois_market_data", None)
    forward_market_data = getattr(curve_set, "forward_market_data", {}) or {}
    points_by_cell = _collect_selected_points(
        currency=currency,
        selected_curve_keys=_selected_curve_keys(discount_curve_key, forward_curve_key),
        discount_key=discount_curve_key,
        forward_key=forward_curve_key,
        ois_data=ois_data,
        forward_market_data=forward_market_data,
    )
    points_by_cell = _filter_excluded_quote_points(
        curve_set,
        points_by_cell,
        discount_key=discount_curve_key,
        forward_key=forward_curve_key,
    )
    if not points_by_cell:
        return []
    points_by_cell = _effective_dated_points(
        curve_set,
        points_by_cell,
        curve_date=curve_date,
        discount_key=discount_curve_key,
        forward_key=forward_curve_key,
        constant_ibor_fixing=constant_ibor_fixing,
    )

    quote_base = _quote_calibration_base(
        curve_set=curve_set,
        constant_ibor_fixing=constant_ibor_fixing,
        currency=currency,
        curve_date=curve_date,
        ois_data=ois_data,
        forward_market_data=forward_market_data,
        discount_key=discount_curve_key,
        forward_key=forward_curve_key,
        base_discount_curve=base_discount_curve,
        base_forward_curve=base_forward_curve,
    )

    sensitivities: list[QuoteSensitivity] = []
    for point in points_by_cell.values():
        outputs = []
        for direction in (1.0, -1.0):
            scenario = _quote_bumped_scenario(
                curve_set=curve_set,
                constant_ibor_fixing=constant_ibor_fixing,
                currency=currency,
                curve_date=curve_date,
                point=point,
                bump_size=direction * bump_size,
                ois_data=ois_data,
                forward_market_data=forward_market_data,
                discount_key=discount_curve_key,
                forward_key=forward_curve_key,
                base_discount_curve=quote_base.scenario.discount_curve,
                base_forward_curve=quote_base.scenario.forward_curve,
                base_bootstrap_discount_curve=quote_base.ois_curve,
            )
            outputs.append(reprice_with_curves(scenario.forward_curve, scenario.discount_curve))
        up_output, down_output = outputs
        dv01 = central_dv01(down_output, up_output, bump_size)
        pay_dv01, receive_dv01 = _central_leg_dv01(
            down_output,
            up_output,
            bump_size,
            leg_delta_allocator,
            dv01,
        )
        sensitivities.append(
            QuoteSensitivity(
                point=point,
                dv01=dv01,
                pay_leg_dv01=pay_dv01,
                receive_leg_dv01=receive_dv01,
            )
        )
    return sensitivities


def _quote_jacobian_artifact(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    discount_curve_key: str,
    forward_curve_key: str,
    constant_ibor_fixing: bool,
) -> tuple[list[CurveRiskPoint], QuoteJacobianArtifact]:
    """Active risk-ladder quotes and their shared implicit calibration artifact."""
    if not _quote_level_sensitivities_enabled(curve_set):
        raise PricingError("Local market-quote DV01 requires retained calibration market data")
    points = _collect_selected_points(
        currency=currency,
        selected_curve_keys=_selected_curve_keys(discount_curve_key, forward_curve_key),
        discount_key=discount_curve_key,
        forward_key=forward_curve_key,
        ois_data=curve_set.ois_market_data,
        forward_market_data=curve_set.forward_market_data,
    )
    selected = list(
        _filter_excluded_quote_points(
            curve_set, points, discount_key=discount_curve_key, forward_key=forward_curve_key
        ).values()
    )
    if not selected:
        raise PricingError("Local market-quote DV01 has no active calibration quotes")
    quote_keys = [_point_quote_key(point) for point in selected]
    key = (
        _cache_root_curve_set(curve_set),
        *risk_context_fingerprint(
            curve_set, tuple(_selected_curve_keys(discount_curve_key, forward_curve_key))
        ),
        constant_ibor_fixing,
        discount_curve_key,
        forward_curve_key,
        _discount_interpolation(curve_set),
        _forward_interpolation(curve_set),
        tuple(quote_keys),
    )
    artifact = _QUOTE_JACOBIAN_CACHE.get(key)
    if artifact is None:
        with (
            ql_evaluation_date(curve_date),
            constant_ibor_fixings(
                curve_set, (discount_curve_key, forward_curve_key), enabled=constant_ibor_fixing
            ),
        ):
            context = _risk_calibration_context(
                curve_set, discount_curve_key, forward_curve_key, constant_ibor_fixing
            )
            active_keys = {
                (key[0], binding.raw_quote_key)
                for key, block in context.blocks.items()
                for binding in block.helper_bindings
            }
            if active_keys - set(quote_keys):
                raise PricingError("Active calibration quote is missing from the risk ladder")
            artifact = build_quote_jacobian(context, quote_keys)
            _QUOTE_HELPER_DATES_CACHE[key] = _helper_maturity_dates(context)
        _QUOTE_JACOBIAN_CACHE[key] = artifact
    helper_dates = _QUOTE_HELPER_DATES_CACHE.get(key)
    if helper_dates is not None:
        dated = with_effective_dates(
            {(point.family, point.point_key, point.curve_key): point for point in selected},
            helper_dates,
            curve_date,
        )
        selected = list(dated.values())
    return selected, artifact


def _parameter_direction_risk(
    artifact: QuoteJacobianArtifact,
    up_curves: tuple[ql.YieldTermStructureHandle, ql.YieldTermStructureHandle],
    down_curves: tuple[ql.YieldTermStructureHandle, ql.YieldTermStructureHandle],
    reprice_with_curves: RepriceFn,
    leg_delta_allocator: LegDeltaAllocator | None,
) -> tuple[float, ...]:
    """Central V(x - h d) - V(x + h d) in the QuoteSensitivity field order after ``point``.

    Pricing partials hold the other role fixed. Calibration coupling is carried by the
    direction ``d`` itself, including shared-curve roles.
    """

    def role_risk(
        role_up: tuple[ql.YieldTermStructureHandle, ql.YieldTermStructureHandle],
        role_down: tuple[ql.YieldTermStructureHandle, ql.YieldTermStructureHandle],
    ) -> tuple[float, float, float]:
        if all(up is down for up, down in zip(role_up, role_down)):
            return (0.0, 0.0, 0.0)
        up = reprice_with_curves(*role_up)
        down = reprice_with_curves(*role_down)
        difference = down.npv - up.npv
        pay, receive = (
            (0.0, difference)
            if leg_delta_allocator is None
            else leg_delta_allocator(down, up, difference)
        )
        return difference, pay, receive

    base_forward, base_discount = artifact.base_curves
    discount = role_risk((base_forward, up_curves[1]), (base_forward, down_curves[1]))
    forward = role_risk((up_curves[0], base_discount), (down_curves[0], base_discount))
    total = tuple(d + f for d, f in zip(discount, forward))
    return (*total, discount[0], forward[0], *discount[1:], *forward[1:])


def _compute_quote_sensitivities_jacobian(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    discount_curve_key: str,
    forward_curve_key: str,
    reprice_with_curves: RepriceFn,
    leg_delta_allocator: LegDeltaAllocator | None,
    constant_ibor_fixing: bool = False,
) -> list[QuoteSensitivity]:
    """Local quote DV01 per quote (adjoint layout, one repricing pair per calibration pillar).

    Never falls back to quote bumps. Only diagnostics need this per-quote layout.
    """
    selected, artifact = _quote_jacobian_artifact(
        curve_set=curve_set,
        currency=currency,
        curve_date=curve_date,
        discount_curve_key=discount_curve_key,
        forward_curve_key=forward_curve_key,
        constant_ibor_fixing=constant_ibor_fixing,
    )
    node_risk = [
        _parameter_direction_risk(
            artifact, up_curves, down_curves, reprice_with_curves, leg_delta_allocator
        )
        for up_curves, down_curves in artifact.parameter_scenarios
    ]
    gradients = np.asarray(node_risk) / (2.0 * artifact.parameter_step)
    values = 1.0e-4 * artifact.jacobian.parameter_quote_derivatives.T @ gradients
    if not np.all(np.isfinite(values)):
        raise PricingError("Local market-quote DV01 produced nonfinite instrument sensitivities")
    return [
        QuoteSensitivity(point, *(float(item) for item in value))
        for point, value in zip(selected, values)
    ]


def _compute_quote_risk_totals_tangent(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    discount_curve_key: str,
    forward_curve_key: str,
    reprice_with_curves: RepriceFn,
    leg_delta_allocator: LegDeltaAllocator | None,
    constant_ibor_fixing: bool = False,
) -> QuoteRiskTotals:
    """Sum over quotes of the local quote DV01, in forward (tangent) mode.

    ``sum_q -1e-4 dV/dq = -1e-4 (dV/dx) . s`` with ``s = C_x^-1 B 1``: one directional
    derivative (two repricings per pricing role) replaces one per calibration pillar.
    It equals the sum of the per-quote adjoint values up to finite-difference rounding.
    """
    _, artifact = _quote_jacobian_artifact(
        curve_set=curve_set,
        currency=currency,
        curve_date=curve_date,
        discount_curve_key=discount_curve_key,
        forward_curve_key=forward_curve_key,
        constant_ibor_fixing=constant_ibor_fixing,
    )
    up_curves, down_curves = artifact.tangent_scenarios
    risk = _parameter_direction_risk(
        artifact, up_curves, down_curves, reprice_with_curves, leg_delta_allocator
    )
    scale = 1.0e-4 * artifact.tangent_scale / (2.0 * artifact.parameter_step)
    values = [scale * item for item in risk]
    if not all(np.isfinite(values)):
        raise PricingError("Local market-quote DV01 produced a nonfinite total")
    return QuoteRiskTotals(*values)


def _sum_quote_risk(quote_sensitivities: list[QuoteSensitivity]) -> QuoteRiskTotals:
    """Plain sums of the per-quote values, field by field."""
    return QuoteRiskTotals(
        *(
            sum(getattr(item, name) for item in quote_sensitivities)
            for name in QuoteRiskTotals.__dataclass_fields__
        )
    )


def _quote_level_sensitivities_enabled(curve_set: CurveSet) -> bool:
    return abs(float(getattr(curve_set, "_shift_bps", 0.0) or 0.0)) <= 1.0e-12


def _quote_calibration_base(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    ois_data: Optional[OISMarketData],
    forward_market_data: dict[str, ForwardMarketData],
    discount_key: str,
    forward_key: str,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    constant_ibor_fixing: bool = False,
) -> QuoteCalibrationBase:
    """Rebuild the unbumped quote curves with the same tight tolerance as bumped curves."""
    cache_key = (
        _cache_root_curve_set(curve_set),
        *risk_context_fingerprint(
            curve_set, tuple(_selected_curve_keys(discount_key, forward_key))
        ),
        constant_ibor_fixing,
        "quote_calibration_base",
        currency,
        curve_date.isoformat(),
        discount_key,
        forward_key,
        QUOTE_BOOTSTRAP_ACCURACY,
    )
    cached = _QUOTE_BASE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if ois_data is None:
        fallback = QuoteCalibrationBase(
            scenario=CurveScenario(base_forward_curve, base_discount_curve),
            ois_curve=curve_set.get_discount_curve("OIS"),
        )
        _QUOTE_BASE_CACHE[cache_key] = fallback
        return fallback

    with constant_ibor_fixings(
        curve_set, (discount_key, forward_key), enabled=constant_ibor_fixing
    ):
        builder = YieldCurveBuilder(currency)
        rebuilt_ois = builder.build_ois_curve(
            ois_data,
            bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
            interpolation=_discount_interpolation(curve_set),
        )
        rebuilt_forward_ois = (
            rebuilt_ois
            if _forward_interpolation(curve_set) == _discount_interpolation(curve_set)
            else builder.build_ois_curve(
                ois_data,
                bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                interpolation=_forward_interpolation(curve_set),
            )
        )
        scenario = CurveScenario(
            forward_curve=_curve_after_ois_bump(
                curve_set=curve_set,
                key=forward_key,
                bumped_ois=(rebuilt_forward_ois if forward_key == "OIS" else rebuilt_ois),
                base_curve=base_forward_curve,
                forward_market_data=forward_market_data,
                builder=builder,
                bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                interpolation=_forward_interpolation(curve_set),
                expected_report_key=forward_key if forward_key != "OIS" else None,
            ),
            discount_curve=_curve_after_ois_bump(
                curve_set=curve_set,
                key=discount_key,
                bumped_ois=rebuilt_ois,
                base_curve=base_discount_curve,
                forward_market_data=forward_market_data,
                builder=builder,
                bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                interpolation=_discount_interpolation(curve_set),
                expected_report_key=(
                    (
                        f"discount_{discount_key}"
                        if _discount_interpolation(curve_set) != _forward_interpolation(curve_set)
                        else discount_key
                    )
                    if discount_key != "OIS"
                    else None
                ),
            ),
        )
        rebuilt = QuoteCalibrationBase(scenario=scenario, ois_curve=rebuilt_ois)
    _QUOTE_BASE_CACHE[cache_key] = rebuilt
    return rebuilt


def _risk_calibration_context(
    curve_set: CurveSet,
    discount_key: str,
    forward_key: str,
    constant_ibor_fixing: bool,
) -> RiskCalibrationContext:
    """Reuse isolated native calibration graphs for one market and fixing context."""
    key = (
        _cache_root_curve_set(curve_set),
        *risk_context_fingerprint(
            curve_set, tuple(_selected_curve_keys(discount_key, forward_key))
        ),
        discount_key,
        forward_key,
        constant_ibor_fixing,
        _discount_interpolation(curve_set),
        _forward_interpolation(curve_set),
    )
    scoped = _SCOPED_RISK_CONTEXTS.get()
    if scoped is None:
        # Outside a risk computation the graph is transient: nothing keeps its helpers alive.
        return build_risk_calibration_context(curve_set, discount_key, forward_key)
    cached = scoped.get(key)
    if cached is None:
        cached = build_risk_calibration_context(curve_set, discount_key, forward_key)
        scoped[key] = cached
    return cached


_FIELD_BY_QUOTE_CATEGORY = {
    "fixing": "fixings",
    "deposit": "deposits",
    "fra": "fras",
    "future": "futures",
    "swap": "swaps",
}


def _displayed_helper_date(binding: CalibrationHelperBinding) -> ql.Date:
    """End of the instrument period: an FRA ends one index tenor after its start, not at spot+N."""
    if binding.instrument_type == "fra":
        return binding.helper.pillarDate()
    return binding.helper.maturityDate()


def _helper_maturity_dates(
    context: RiskCalibrationContext,
) -> dict[tuple[str, str, int], dt_date]:
    """Maturity of every active helper, keyed by curve, market-data field and quote index."""
    return {
        (
            block_key[0],
            _FIELD_BY_QUOTE_CATEGORY[binding.raw_quote_key.category],
            binding.raw_quote_key.index,
        ): ql_to_date(_displayed_helper_date(binding))
        for block_key, block in context.blocks.items()
        for binding in block.helper_bindings
        if binding.raw_quote_key is not None
    }


def _effective_dated_points(
    curve_set: CurveSet,
    points: dict[tuple[str, str, str], CurveRiskPoint],
    *,
    curve_date: dt_date,
    discount_key: str,
    forward_key: str,
    constant_ibor_fixing: bool,
) -> dict[tuple[str, str, str], CurveRiskPoint]:
    """Give each quote the maturity of the helper the calibrated curves really use."""
    with (
        ql_evaluation_date(curve_date),
        constant_ibor_fixings(curve_set, (discount_key, forward_key), enabled=constant_ibor_fixing),
    ):
        context = _risk_calibration_context(
            curve_set, discount_key, forward_key, constant_ibor_fixing
        )
        helper_dates = _helper_maturity_dates(context)
    return with_effective_dates(points, helper_dates, curve_date)


def _point_quote_key(point: CurveRiskPoint) -> tuple[str, RawQuoteKey]:
    """Map retained source collections to native calibration quote coordinates."""
    categories = {
        "fixings": "fixing",
        "deposits": "deposit",
        "fras": "fra",
        "futures": "future",
        "swaps": "swap",
    }
    return point.curve_key, RawQuoteKey(categories[point.field], point.index)


def _quote_bumped_scenario(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    point: CurveRiskPoint,
    bump_size: float,
    ois_data: Optional[OISMarketData],
    forward_market_data: dict[str, ForwardMarketData],
    discount_key: str,
    forward_key: str,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    base_bootstrap_discount_curve: ql.YieldTermStructureHandle,
    constant_ibor_fixing: bool = False,
) -> CurveScenario:
    """Finite scenario using reusable native calibration objects; frozen result shared by trades."""
    key = (
        _cache_root_curve_set(curve_set),
        *risk_context_fingerprint(
            curve_set, tuple(_selected_curve_keys(discount_key, forward_key))
        ),
        "reusable_quote",
        constant_ibor_fixing,
        currency,
        curve_date,
        discount_key,
        forward_key,
        point.curve_key,
        point.field,
        point.index,
        bump_size,
    )
    cached = _CURVE_BUMP_CACHE.get(key)
    if cached is not None:
        return cached
    with (
        ql_evaluation_date(curve_date),
        constant_ibor_fixings(curve_set, (discount_key, forward_key), enabled=constant_ibor_fixing),
    ):
        context = _risk_calibration_context(
            curve_set, discount_key, forward_key, constant_ibor_fixing
        )
        forward, discount = context.quote_scenario({_point_quote_key(point): bump_size})
        scenario = CurveScenario(forward, discount)
    _CURVE_BUMP_CACHE[key] = scenario
    return scenario


def _quote_bumped_scenario_reference(
    *,
    curve_set: CurveSet,
    currency: str,
    curve_date: dt_date,
    point: CurveRiskPoint,
    bump_size: float,
    ois_data: Optional[OISMarketData],
    forward_market_data: dict[str, ForwardMarketData],
    discount_key: str,
    forward_key: str,
    base_discount_curve: ql.YieldTermStructureHandle,
    base_forward_curve: ql.YieldTermStructureHandle,
    base_bootstrap_discount_curve: ql.YieldTermStructureHandle,
    constant_ibor_fixing: bool = False,
) -> CurveScenario:
    cache_key = (
        _cache_root_curve_set(curve_set),
        *risk_context_fingerprint(
            curve_set, tuple(_selected_curve_keys(discount_key, forward_key))
        ),
        "reference_quote",
        constant_ibor_fixing,
        currency,
        curve_date.isoformat(),
        round(bump_size, 12),
        discount_key,
        forward_key,
        point.curve_key,
        point.field,
        point.index,
    )
    cached = _CURVE_BUMP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with constant_ibor_fixings(
        curve_set, (discount_key, forward_key), enabled=constant_ibor_fixing
    ):
        builder = YieldCurveBuilder(currency)
        if point.curve_key == "OIS":
            if ois_data is None:
                raise ValueError("OIS market data is unavailable")
            bumped_ois = builder.build_ois_curve(
                _bump_ois_quote(ois_data, point.field, point.index, bump_size),
                bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                interpolation=_discount_interpolation(curve_set),
            )
            bumped_forward_ois = (
                bumped_ois
                if _forward_interpolation(curve_set) == _discount_interpolation(curve_set)
                else builder.build_ois_curve(
                    _bump_ois_quote(ois_data, point.field, point.index, bump_size),
                    bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                    interpolation=_forward_interpolation(curve_set),
                )
            )
            bumped_discount_curve = _curve_after_ois_bump(
                curve_set=curve_set,
                key=discount_key,
                bumped_ois=bumped_ois,
                base_curve=base_discount_curve,
                forward_market_data=forward_market_data,
                builder=builder,
                bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                interpolation=_discount_interpolation(curve_set),
                expected_report_key=(
                    (
                        f"discount_{discount_key}"
                        if _discount_interpolation(curve_set) != _forward_interpolation(curve_set)
                        else discount_key
                    )
                    if discount_key != "OIS"
                    else None
                ),
            )
            bumped_forward_curve = _curve_after_ois_bump(
                curve_set=curve_set,
                key=forward_key,
                bumped_ois=(bumped_forward_ois if forward_key == "OIS" else bumped_ois),
                base_curve=base_forward_curve,
                forward_market_data=forward_market_data,
                builder=builder,
                bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                interpolation=_forward_interpolation(curve_set),
                expected_report_key=forward_key if forward_key != "OIS" else None,
            )
            scenario = CurveScenario(bumped_forward_curve, bumped_discount_curve)
        else:
            forward_data = forward_market_data_for_key(forward_market_data, point.curve_key)
            if forward_data is None:
                raise ValueError(f"Forward market data is unavailable for {point.curve_key}")
            bumped_data = _bump_forward_quote(forward_data, point.field, point.index, bump_size)
            bumped_curve = _build_forward_curve_for_risk(
                builder=builder,
                curve_set=curve_set,
                market_data=bumped_data,
                discount_curve=base_bootstrap_discount_curve,
                bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                interpolation=_forward_interpolation(curve_set),
                expected_report_key=(point.curve_key if forward_key == point.curve_key else None),
            )
            bumped_discount_curve = bumped_curve
            if discount_key == point.curve_key and _discount_interpolation(
                curve_set
            ) != _forward_interpolation(curve_set):
                bumped_discount_curve = _build_forward_curve_for_risk(
                    builder=builder,
                    curve_set=curve_set,
                    market_data=bumped_data,
                    discount_curve=base_bootstrap_discount_curve,
                    bootstrap_accuracy=QUOTE_BOOTSTRAP_ACCURACY,
                    interpolation=_discount_interpolation(curve_set),
                    expected_report_key=f"discount_{point.curve_key}",
                )
            scenario = CurveScenario(
                forward_curve=(
                    bumped_curve if forward_key == point.curve_key else base_forward_curve
                ),
                discount_curve=(
                    bumped_discount_curve
                    if discount_key == point.curve_key
                    else base_discount_curve
                ),
            )

    _CURVE_BUMP_CACHE[cache_key] = scenario
    return scenario


def _curve_after_ois_bump(
    *,
    curve_set: CurveSet,
    key: str,
    bumped_ois: ql.YieldTermStructureHandle,
    base_curve: ql.YieldTermStructureHandle,
    forward_market_data: dict[str, ForwardMarketData],
    builder: YieldCurveBuilder,
    bootstrap_accuracy: float,
    interpolation: str,
    expected_report_key: str | None,
) -> ql.YieldTermStructureHandle:
    if key == "OIS":
        return bumped_ois
    forward_data = forward_market_data_for_key(forward_market_data, key)
    if forward_data is None:
        return base_curve
    return _build_forward_curve_for_risk(
        builder=builder,
        curve_set=curve_set,
        market_data=forward_data,
        discount_curve=bumped_ois,
        bootstrap_accuracy=bootstrap_accuracy,
        interpolation=interpolation,
        expected_report_key=expected_report_key,
    )


def _cache_root_curve_set(curve_set: CurveSet) -> object:
    return getattr(curve_set, "_original", curve_set)


def _to_curve_instrument_deltas(
    quote_sensitivities: list[QuoteSensitivity],
) -> list[CurveInstrumentDelta]:
    return [
        CurveInstrumentDelta(
            curve_key=item.point.curve_key,
            curve_role=item.point.curve_role,
            instrument_type=_instrument_type_for_field(item.point.field),
            label=item.point.label,
            tenor=item.point.tenor,
            maturity_date=item.point.maturity_date,
            quote=item.point.quote,
            pay_leg_delta=item.pay_leg_dv01,
            receive_leg_delta=item.receive_leg_dv01,
            quote_dv01=item.dv01,
            raw_quote_dv01=item.dv01,
            scaled_quote_dv01=item.dv01,
            delta=item.dv01,
        )
        for item in quote_sensitivities
    ]


def _instrument_type_for_field(field: str) -> str:
    return {
        "deposits": "deposit",
        "fras": "fra",
        "futures": "future",
        "swaps": "swap",
    }.get(field, field)


def _assemble_curve_risk_ladder_from_quote_sensitivities(
    *,
    currency: str,
    bump_size: float,
    quote_dv01_total: float,
    discount_curve_key: str,
    forward_curve_key: str,
    quote_sensitivities: list[QuoteSensitivity],
) -> Optional[CurveRiskLadder]:
    """Ladder of exact quote DV01s: every total is a plain sum, nothing is rescaled."""
    selected_curve_keys = _selected_curve_keys(discount_curve_key, forward_curve_key)
    if not selected_curve_keys:
        return None

    columns = [
        CurveRiskColumn(
            curve_key=key,
            curve_label=curve_display_label(currency, key),
            curve_role=curve_role_for_key(key, discount_curve_key, forward_curve_key),
        )
        for key in selected_curve_keys
    ]

    points_by_cell = {
        (item.point.family, item.point.point_key, item.point.curve_key): item.point
        for item in quote_sensitivities
    }
    values = {
        (item.point.family, item.point.point_key, item.point.curve_key): item.dv01
        for item in quote_sensitivities
    }
    sections = _build_sections(
        columns=columns,
        points_by_cell=points_by_cell,
        raw_values=values,
        scaled_values=values,
    )
    total_delta = sum(section.total_delta for section in sections)
    return CurveRiskLadder(
        currency=currency,
        bump_bp=bump_size * 10000.0,
        curve_columns=columns,
        sections=sections,
        raw_total_delta=total_delta,
        total_delta=total_delta,
        target_delta=quote_dv01_total,
        scaling_factor=None,
        residual=quote_dv01_total - total_delta,
        raw_residual=quote_dv01_total - total_delta,
        warnings=[],
    )


def _assemble_key_rate_risk(
    *,
    currency: str,
    bump_size: float,
    discount_curve_key: str,
    forward_curve_key: str,
    quote_sensitivities: list[QuoteSensitivity],
    net_only: bool = False,
) -> KeyRateRiskMatrix:
    """Aggregate exact ESTR/OIS and index quote DV01s by point; legs are plain sums."""
    curve_keys = _selected_curve_keys(discount_curve_key, forward_curve_key)
    curve_labels = [curve_display_label(currency, key) for key in curve_keys]

    row_entries: dict[tuple[str, str], list[QuoteSensitivity]] = {}
    for item in quote_sensitivities:
        row_entries.setdefault((item.point.family, item.point.point_key), []).append(item)

    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    ordered_entries = sorted(
        row_entries.items(),
        key=lambda entry: (
            family_rank.get(entry[0][0], len(FAMILY_ORDER)),
            min(item.point.sort_key for item in entry[1]),
        ),
    )
    rows = [
        _key_rate_risk_row(family=family, point_key=point_key, entries=entries, net_only=net_only)
        for (family, point_key), entries in ordered_entries
    ]

    pay_total = sum(row.pay_leg_dv01 for row in rows)
    receive_total = sum(row.receive_leg_dv01 for row in rows)
    net_total = sum(row.net_dv01 for row in rows)
    quote_pay_total = 0.0 if net_only else sum(item.pay_leg_dv01 for item in quote_sensitivities)
    quote_receive_total = 0.0 if net_only else sum(item.receive_leg_dv01 for item in quote_sensitivities)
    quote_net_total = sum(item.dv01 for item in quote_sensitivities) if net_only else quote_pay_total + quote_receive_total
    return KeyRateRiskMatrix(
        display_mode="net_only" if net_only else "legs",
        currency=currency,
        bump_bp=bump_size * 10000.0,
        discount_curve_key=discount_curve_key,
        forward_curve_key=forward_curve_key,
        curve_labels=curve_labels,
        rows=rows,
        raw_pay_leg_total=pay_total,
        raw_receive_leg_total=receive_total,
        raw_net_total=net_total,
        pay_leg_total=pay_total,
        receive_leg_total=receive_total,
        net_total=net_total,
        target_pay_leg_dv01=quote_pay_total,
        target_receive_leg_dv01=quote_receive_total,
        target_net_dv01=quote_net_total,
        pay_scaling_factor=None,
        receive_scaling_factor=None,
        pay_residual=quote_pay_total - pay_total,
        receive_residual=quote_receive_total - receive_total,
        net_residual=quote_net_total - net_total,
        warnings=[],
    )


def _key_rate_risk_row(
    *,
    family: str,
    point_key: str,
    entries: list[QuoteSensitivity],
    net_only: bool = False,
) -> KeyRateRiskRow:
    representative = min(entries, key=lambda item: item.point.sort_key).point
    contributions = [_key_rate_curve_contribution(item, net_only) for item in entries]
    pay = sum(item.pay_leg_dv01 for item in contributions)
    receive = sum(item.receive_leg_dv01 for item in contributions)
    net = sum(item.net_dv01 for item in contributions)
    return KeyRateRiskRow(
        family=family,
        family_label=FAMILY_LABELS.get(family, family),
        point_key=point_key,
        label=representative.label,
        tenor=representative.tenor,
        maturity_date=representative.maturity_date,
        curve_keys=[item.curve_key for item in contributions],
        curve_contributions=contributions,
        raw_pay_leg_dv01=pay,
        raw_receive_leg_dv01=receive,
        raw_net_dv01=net,
        pay_leg_dv01=pay,
        receive_leg_dv01=receive,
        net_dv01=net,
    )


def _key_rate_curve_contribution(item: QuoteSensitivity, net_only: bool = False) -> KeyRateRiskCurveContribution:
    pay = 0.0 if net_only else item.pay_leg_dv01
    receive = 0.0 if net_only else item.receive_leg_dv01
    net = item.dv01 if net_only else pay + receive
    return KeyRateRiskCurveContribution(
        curve_key=item.point.curve_key,
        curve_label=item.point.curve_label,
        raw_pay_leg_dv01=pay,
        raw_receive_leg_dv01=receive,
        raw_net_dv01=net,
        pay_leg_dv01=pay,
        receive_leg_dv01=receive,
        net_dv01=net,
    )
