"""Implicit calibration derivatives (Henrard, JCF 17(4), 2014).

For normalized implied quotes C(x) and helper-to-market incidence B, solve
C_x dx/dq = B. The Jacobian never differentiates or reruns a bootstrap solver.
``solve_calibration_system`` solves C(x) = q itself, only where QuantLib's iterative
bootstrap does not converge for a non-local interpolator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from services.exceptions import PricingError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationJacobian:
    """Immutable parameter-to-quote transform and conditioning evidence."""

    parameter_quote_derivatives: FloatArray
    condition_number: float
    residual_jacobian: FloatArray


def calibration_jacobian(
    implied_quotes: Callable[[FloatArray], FloatArray],
    parameters: FloatArray,
    quote_incidence: FloatArray,
    *,
    step: float = 1.0e-5,
    condition_limit: float = 1.0e12,
) -> CalibrationJacobian:
    """Differentiate normalized helper rates at fixed active calibration pillars.

    A symmetric parameter perturbation evaluates instruments, without bootstrapping.
    Parameters are normally log discount factors, giving a dimensionless step.
    Singular or nonfinite systems raise explicitly; no pseudoinverse or risk fallback.
    """
    x = np.asarray(parameters, dtype=float)
    incidence = np.asarray(quote_incidence, dtype=float)
    if x.ndim != 1 or not x.size or incidence.ndim != 2 or incidence.shape[0] != x.size:
        raise PricingError("Calibration Jacobian requires a square helper/parameter system")
    if not np.isfinite(step) or step <= 0.0:
        raise PricingError("Calibration Jacobian step must be finite and positive")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(incidence)):
        raise PricingError("Calibration Jacobian inputs must be finite")
    if np.any(~np.any(incidence != 0.0, axis=0)):
        raise PricingError("A requested market quote has no active helper in the calibration")
    matrix = _residual_jacobian(implied_quotes, x, step)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > condition_limit:
        raise PricingError("Calibration Jacobian is singular or ill-conditioned")
    try:
        derivatives = lu_solve(lu_factor(matrix), incidence)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise PricingError("Calibration Jacobian factorization failed") from exc
    if not np.all(np.isfinite(derivatives)):
        raise PricingError("Calibration Jacobian solution is not finite")
    derivatives.setflags(write=False)
    matrix.setflags(write=False)
    return CalibrationJacobian(derivatives, condition, matrix)


def solve_calibration_system(
    residuals: Callable[[FloatArray], FloatArray],
    parameters: FloatArray,
    *,
    tolerance: float = 1.0e-12,
    step: float = 1.0e-6,
    max_iterations: int = 20,
    condition_limit: float = 1.0e12,
) -> FloatArray:
    """Solve the square calibration system ``residuals(x) = 0`` by Newton-Raphson.

    Every pillar is solved jointly, as curve construction with a non-local interpolator
    requires (Hagan and West, Appl. Math. Finance 13(2), 2006; Andersen and Piterbarg,
    Interest Rate Modeling, vol. I, ch. 6, 2010). Backtracking halves the step until the
    maximum residual decreases. Non-convergence or a singular system raises; no fallback.
    """
    x = np.asarray(parameters, dtype=float).copy()
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise PricingError("Global calibration requires finite pillar parameters")
    current = np.asarray(residuals(x), dtype=float)
    if current.shape != x.shape or not np.all(np.isfinite(current)):
        raise PricingError("Global calibration residuals are nonfinite or not square")
    for _ in range(max_iterations):
        error = float(np.max(np.abs(current)))
        if error <= tolerance:
            return x
        matrix = _residual_jacobian(residuals, x, step)
        condition = float(np.linalg.cond(matrix))
        if not np.isfinite(condition) or condition > condition_limit:
            raise PricingError("Global calibration Jacobian is singular or ill-conditioned")
        try:
            direction = lu_solve(lu_factor(matrix), -current)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise PricingError("Global calibration Jacobian factorization failed") from exc
        damping = 1.0
        while True:
            candidate = x + damping * direction
            trial = np.asarray(residuals(candidate), dtype=float)
            if np.all(np.isfinite(trial)) and float(np.max(np.abs(trial))) < error:
                x, current = candidate, trial
                break
            damping *= 0.5
            if damping < 1.0e-6:
                raise PricingError("Global calibration line search found no descent step")
    if float(np.max(np.abs(current))) <= tolerance:
        return x
    raise PricingError(
        f"Global calibration did not converge in {max_iterations} Newton iterations "
        f"(max residual {float(np.max(np.abs(current))):.3g})"
    )


def _residual_jacobian(
    residuals: Callable[[FloatArray], FloatArray],
    x: FloatArray,
    step: float,
) -> FloatArray:
    """Central-difference Jacobian of square helper residuals in pillar parameters."""
    matrix = np.empty((x.size, x.size), dtype=float)
    for column in range(x.size):
        up, down = x.copy(), x.copy()
        up[column] += step
        down[column] -= step
        high = np.asarray(residuals(up), dtype=float)
        low = np.asarray(residuals(down), dtype=float)
        if high.shape != x.shape or low.shape != x.shape:
            raise PricingError("Calibration helper count differs from parameter count")
        matrix[:, column] = (high - low) / (2.0 * step)
    if not np.all(np.isfinite(matrix)):
        raise PricingError("Calibration Jacobian contains nonfinite derivatives")
    return matrix
