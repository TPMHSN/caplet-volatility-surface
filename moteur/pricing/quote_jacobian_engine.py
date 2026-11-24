"""Implicit market-quote risk built once per market calibration context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import QuantLib as ql

from curves.calibration_jacobian import CalibrationJacobian, FloatArray, calibration_jacobian
from curves.calibration_residuals import CalibrationRateEvaluator
from pricing.calibration_risk_context import BlockKey, CurvePair, QuoteKey, RiskCalibrationContext
from services.exceptions import PricingError

PARAMETER_STEP = 1.0e-5


@dataclass(frozen=True)
class QuoteJacobianArtifact:
    """Quote transformation and independent parameter perturbations shared by all trades.

    ``parameter_scenarios`` perturb one pillar each (adjoint layout, one column per quote).
    ``tangent_scenarios`` perturb every pillar along ``s = dx/dq . 1``, the calibrated
    response to a parallel move of all quotes, scaled by ``1 / tangent_scale``.
    """

    jacobian: CalibrationJacobian
    parameter_scenarios: tuple[tuple[CurvePair, CurvePair], ...]
    parameter_step: float
    quote_keys: tuple[QuoteKey, ...]
    base_curves: CurvePair
    tangent_scenarios: tuple[CurvePair, CurvePair]
    tangent_scale: float


def build_quote_jacobian(
    context: RiskCalibrationContext,
    quote_keys: list[QuoteKey],
    *,
    parameter_step: float = PARAMETER_STEP,
) -> QuoteJacobianArtifact:
    """Build d(log pillar DF)/dq without rebuilding or solving any shocked bootstrap.

    Calibration instruments and trades see explicit curves on the native pillars with the
    native interpolator. They are not daily frozen: every instrument reads discount factors
    on dates, where a daily-frozen copy returns the same values, and freezing each perturbed
    curve dominated the cost of the Jacobian.
    """
    blocks = list(context.blocks.items())
    offsets: dict[BlockKey, slice] = {}
    parameters: list[float] = []
    incidence_rows: list[list[float]] = []
    for key, block in blocks:
        count = len(block.pillar_values) - 1
        if count != len(block.helper_bindings):
            raise PricingError("Calibration pillars and retained helpers have different dimensions")
        offsets[key] = slice(len(parameters), len(parameters) + count)
        parameters.extend(np.log(block.pillar_values[1:]).tolist())
        for binding in block.helper_bindings:
            incidence_rows.append(
                [float((key[0], binding.raw_quote_key) == quote) for quote in quote_keys]
            )
    x = np.asarray(parameters)
    incidence = np.asarray(incidence_rows)
    # A parameter perturbation changes only one block. Reuse unchanged block curves.
    states: dict[tuple[BlockKey, tuple[float, ...]], ql.YieldTermStructureHandle] = {}

    def curves_at(values: FloatArray) -> dict[BlockKey, ql.YieldTermStructureHandle]:
        curves = {}
        for key, block in blocks:
            node_values = tuple(values[offsets[key]])
            cache_key = (key, node_values)
            if cache_key not in states:
                states[cache_key] = ql.YieldTermStructureHandle(
                    block.explicit_curve((block.pillar_values[0], *np.exp(node_values)))
                )
            curves[key] = states[cache_key]
        return curves

    # Schedules are read once under the calibration date, then reused by every evaluation.
    evaluators = {key: CalibrationRateEvaluator(block.helper_bindings) for key, block in blocks}

    def implied_quotes(values: FloatArray) -> FloatArray:
        curves = curves_at(values)
        rates: list[float] = []
        for key, _ in blocks:
            rates.extend(
                evaluators[key].rates(
                    curves[key],
                    curves[context.ois_key] if key[0] != "OIS" else curves[key],
                )
            )
        return np.asarray(rates)

    # Independently validate that the residual evaluator reproduces native helper semantics.
    native_rates = np.array(
        [
            (
                (100.0 - binding.helper.impliedQuote()) / 100.0
                if binding.quote_kind == "futures_price"
                else binding.helper.impliedQuote()
            )
            for _, block in blocks
            for binding in block.helper_bindings
        ]
    )
    # Helpers reprice their live native curve, not its construction-time snapshot: an
    # unconverged SmoothForward iterate moves whenever it is recalculated. Dependent curves
    # were calibrated on the calibrated OIS.
    calibrated = curves_at(x)
    evaluated: list[float] = []
    for key, block in blocks:
        native = ql.YieldTermStructureHandle(block.explicit_curve(block.native_state_values()))
        discount = native if key[0] == "OIS" else calibrated[context.ois_key]
        evaluated.extend(evaluators[key].rates(native, discount))
    residual_error = np.max(np.abs(np.asarray(evaluated) - native_rates))
    if not np.isfinite(residual_error) or residual_error > 1.0e-8:
        raise PricingError(
            f"Calibration evaluator does not reproduce native helpers (error {residual_error:.3g})"
        )
    jacobian = calibration_jacobian(implied_quotes, x, incidence, step=parameter_step)

    def pair_at(values: FloatArray) -> CurvePair:
        curves = curves_at(values)
        return curves[context.forward_key], curves[context.discount_key]

    scenarios = []
    for column in range(x.size):
        up, down = x.copy(), x.copy()
        up[column] += parameter_step
        down[column] -= parameter_step
        scenarios.append((pair_at(up), pair_at(down)))
    # Forward mode (tangent) for aggregates: one direction instead of one per pillar.
    tangent = jacobian.parameter_quote_derivatives.sum(axis=1)
    tangent_scale = float(np.max(np.abs(tangent))) if tangent.size else 0.0
    if not np.isfinite(tangent_scale) or tangent_scale <= 0.0:
        raise PricingError("Parallel market-quote tangent of the calibration is zero or nonfinite")
    direction = tangent / tangent_scale
    return QuoteJacobianArtifact(
        jacobian,
        tuple(scenarios),
        parameter_step,
        tuple(quote_keys),
        pair_at(x),
        (pair_at(x + parameter_step * direction), pair_at(x - parameter_step * direction)),
        tangent_scale,
    )
