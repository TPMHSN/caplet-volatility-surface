"""Frozen N_variance_wing 3M construction selected on 23 October 2026.

The six positive 1Y cap quotes determine a common smile in absolute moneyness
for the first three optionlets. The 18M and 2Y residual prices determine the
four later optionlets. D.4 supplies only the negative wing. No ICE reference
or historical prediction enters calibration.

The native construction is the numerical port of the sealed research
``nonflat_block_20261023`` and its residual-allocation helper. The separately
named coupon evaluator adds bounded fixing-time interpolation at unchanged
absolute moneyness for contractual coupons between the seven native fixings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import numpy.typing as npt
import QuantLib as ql
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

from services.caplet_normal_curves import _calibrate, _payment_time, _time_value, _vol_at
from services.exceptions import PricingError
from services.latest_normal_model_math import TIME_SMOOTHING, d4_normal_vol, fit_latest_smile
from services.thomas_stripping_schemas import OptionletNode, StrippingFamily

FloatArray = npt.NDArray[np.float64]
QuoteRows = Sequence[tuple[int, Sequence[float]]]
POSITIVE_STRIKES = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
VALIDATED_STRIKE_DOMAIN = (-0.015, 0.05)
MODEL_NAME = "N_variance_wing"


class CandidatePricingError(PricingError, ValueError):
    """Accepted candidate inputs could not produce an admissible numerical result."""


def _undiscounted(forward: Any, strike: Any, expiry: Any, vol: Any) -> FloatArray:
    std = vol * np.sqrt(expiry)
    z = (forward - strike) / std
    return (forward - strike) * ndtr(z) + std * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)


def _readonly(values: Any) -> FloatArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _finite_query(strikes: Any) -> FloatArray:
    query = np.asarray(strikes, dtype=float)
    if query.size == 0 or np.any(~np.isfinite(query)):
        raise ValueError("Les strikes du candidat doivent être finis et non vides.")
    return query


def _validate_inputs(
    three: StrippingFamily, rows: QuoteRows
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    strikes = np.asarray(three.strikes, dtype=float)
    positive = np.flatnonzero(strikes >= 0)
    if (
        three.index_tenor != "3M"
        or three.status != "ready"
        or three.atm_caps
        or len(three.optionlets) != 7
        or len(three.blocks) != 3
        or [n.cap_maturity_months for n in three.optionlets] != list(range(6, 25, 3))
        or [c.maturity_months for c in three.caps] != [12, 18, 24]
        or [month for month, _ in rows] != [12, 18, 24]
        or np.any(~np.isfinite(strikes))
        or np.any(np.diff(strikes) <= 0)
        or not np.array_equal(strikes[positive], np.asarray(POSITIVE_STRIKES))
    ):
        raise ValueError(
            "Le candidat 3M exige les caps natifs 1Y, 18M et 2Y, sept caplets "
            "et les strikes positifs 0 ; 0,5 ; 1 ; 2 ; 3 ; 5 %."
        )
    for cap, (_, values) in zip(three.caps, rows, strict=True):
        flat = np.asarray(values, dtype=float)
        if (
            flat.shape != strikes.shape
            or np.any(~np.isfinite(flat))
            or np.any(flat <= 0)
            or not np.array_equal(flat, np.asarray(cap.flat_vols))
        ):
            raise ValueError("Les quotes du candidat doivent correspondre à la famille 3M native.")
    previous_expiry = 0.0
    for node in three.optionlets:
        if (
            not all(
                isfinite(v)
                for v in (
                    node.forward_rate,
                    node.expiry_years,
                    node.accrual_fraction,
                    node.discount_factor,
                )
            )
            or node.expiry_years <= previous_expiry
            or node.accrual_fraction <= 0
            or node.discount_factor <= 0
            or len(node.volatilities) != len(strikes)
            or any(v is None or not isfinite(v) or v <= 0 for v in node.volatilities)
        ):
            raise ValueError("Calendrier ou volatilités natives du candidat invalides.")
        previous_expiry = node.expiry_years
    return strikes, positive


def _fit_common(
    rows: QuoteRows,
    strikes: FloatArray,
    positive: npt.NDArray[np.int64],
    forwards: FloatArray,
    expiries: FloatArray,
    annuities: FloatArray,
) -> tuple[CubicSpline, dict[str, Any]]:
    """Exact six-coefficient variance fit, with the frozen three starts."""
    forward, expiry, annuity = forwards[:3], expiries[:3], annuities[:3]
    k = strikes[positive]
    quotes = np.array(rows[0][1])[positive]
    reference_forward = float(np.average(forward, weights=annuity))
    knots = k - reference_forward
    knots[0] = -max(forward)
    knots[-1] = 0.05 - min(forward)
    target = np.sum(
        annuity[:, None]
        * _undiscounted(forward[:, None], k[None, :], expiry[:, None], quotes[None, :]),
        axis=0,
    )
    std = quotes[None, :] * np.sqrt(expiry[:, None])
    z = (forward[:, None] - k[None, :]) / std
    vega = (
        np.sum(
            annuity[:, None] * np.sqrt(expiry[:, None]) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
            axis=0,
        )
        * 1e-4
    )
    if np.any(vega <= 0) or np.any(~np.isfinite(vega)):
        raise ValueError("Vega de calibration du premier bloc candidat nul ou non fini.")

    def transform(value: FloatArray) -> FloatArray:
        return (100 * value) ** 2

    def curve(parameters: FloatArray, x: FloatArray) -> FloatArray:
        spline = CubicSpline(knots, parameters, bc_type="natural", extrapolate=False)
        query = np.asarray(x)
        clipped = np.clip(query, knots[0], knots[-1])
        transformed = spline(clipped) + spline(clipped, 1) * (query - clipped)
        return np.sqrt(np.maximum(transformed, 1e-14)) / 100

    def residual(parameters: FloatArray) -> FloatArray:
        vols = curve(parameters, k[None, :] - forward[:, None])
        predicted = np.sum(
            annuity[:, None] * _undiscounted(forward[:, None], k[None, :], expiry[:, None], vols),
            axis=0,
        )
        return (predicted - target) / vega

    initial = transform(quotes)
    bounds = (transform(np.full(6, 0.0005)), transform(np.full(6, 0.06)))
    fits = [
        least_squares(
            residual,
            np.clip(initial + tilt, bounds[0] + 1e-8, bounds[1] - 1e-8),
            bounds=bounds,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-11,
            max_nfev=1500,
        )
        for tilt in (np.zeros(6), np.linspace(-0.03, 0.03, 6), np.linspace(0.03, -0.03, 6))
    ]
    best = min(fits, key=lambda result: np.dot(result.fun, result.fun))
    singular_values = np.linalg.svd(best.jac, compute_uv=False)
    if not best.success:
        raise ValueError("Calibration du candidat non convergée : " + best.message)
    raw_spline = CubicSpline(knots, best.x, bc_type="natural", extrapolate=False)
    stationary = raw_spline.derivative().roots(extrapolate=False)
    raw_positive_min = float(np.min(raw_spline(np.r_[knots, stationary])))
    raw_full_min = min(
        raw_positive_min, float(raw_spline(knots[0]) - 0.015 * raw_spline(knots[0], 1))
    )
    if raw_full_min <= 0:
        raise ValueError("Variance cubique non positive avant la racine carrée du candidat.")
    metadata = {
        "kind": "variance",
        "moneyness_knots": knots.tolist(),
        "parameters": best.x.tolist(),
        "weighted_residual_bp": best.fun.tolist(),
        "max_residual_eur_1m": float(np.max(abs(best.fun * vega)) * 1e6),
        "success": bool(best.success),
        "jacobian_condition": float(singular_values[0] / singular_values[-1]),
        "multistart_costs": [float(np.dot(fit.fun, fit.fun)) for fit in fits],
        "multistart_parameter_spread": float(np.max(np.ptp([fit.x for fit in fits], axis=0))),
        "common_atm_bp": float(curve(best.x, 0)) * 1e4,
        "bound_hit": bool(
            np.any(np.isclose(best.x, bounds[0], atol=1e-6))
            or np.any(np.isclose(best.x, bounds[1], atol=1e-6))
        ),
        "min_raw_transformed_positive": raw_positive_min,
        "min_raw_transformed_full": raw_full_min,
    }
    if metadata["max_residual_eur_1m"] > 1e-4:
        raise ValueError("Le premier bloc candidat ne reproduit pas exactement les prix natifs.")
    return raw_spline, metadata


def _common_value(spline: CubicSpline, query: FloatArray) -> FloatArray:
    clipped = np.clip(query, spline.x[0], spline.x[-1])
    transformed = spline(clipped) + spline(clipped, 1) * (query - clipped)
    return np.sqrt(np.maximum(transformed, 1e-14)) / 100


def _allocate_tail(
    three: StrippingFamily,
    strikes: FloatArray,
    prefix: FloatArray,
    valuation: ql.Date,
) -> FloatArray:
    """Retain the original residual prices, smoothing and third-caplet anchor."""
    nodes = three.optionlets
    payments = np.array([_payment_time(valuation, node) for node in nodes])
    matrix = np.empty((7, len(strikes)), dtype=float)
    matrix[:3] = prefix
    for j, strike in enumerate(strikes):
        first_tv = sum(
            _time_value(node, float(strike), float(vol))
            for node, vol in zip(nodes[:3], prefix[:, j])
        )
        residual_18m = (
            three.blocks[1].residual_time_values[j]
            + three.blocks[0].residual_time_values[j]
            - first_tv
        )
        residual_24m = three.blocks[2].residual_time_values[j]
        intervals = []
        for start, end, residual in ((3, 5, residual_18m), (5, 7, residual_24m)):
            block = nodes[start:end]
            constant = brentq(
                lambda vol, block=block, strike=strike, residual=residual: sum(
                    _time_value(node, float(strike), vol) for node in block
                )
                - residual,
                1e-10,
                1.0,
                xtol=1e-14,
            )
            intervals.append(
                {
                    "valuation": valuation,
                    "start_time": payments[start - 1],
                    "end_time": payments[end - 1],
                    "block": block,
                    "residual_time_value": residual,
                    "constant": constant,
                }
            )
        times = [payments[2], payments[4], payments[6]]
        fit = _calibrate(
            intervals,
            times,
            [interval["constant"] for interval in intervals],
            float(strike),
            TIME_SMOOTHING,
            True,
            left_boundary_vol=float(prefix[2, j]),
        )
        matrix[3:, j] = [
            _vol_at(fit["parameters"], times, payment, True) for payment in payments[3:]
        ]
    return matrix


@dataclass(frozen=True)
class FrozenThreeMonthCandidate:
    """Calibrated state owned by one server snapshot, with no module globals."""

    nodes: tuple[OptionletNode, ...]
    strikes: FloatArray
    positive_strikes: FloatArray
    F: FloatArray
    T: FloatArray
    A: FloatArray
    matrix: FloatArray
    fit: dict[str, Any]
    wing_fit: tuple[dict[str, Any], ...]
    target_prices: FloatArray
    _common_spline: CubicSpline
    _tail_splines: tuple[CubicSpline, ...]
    name: str = MODEL_NAME
    domain: tuple[float, float] = VALIDATED_STRIKE_DOMAIN

    def components(self, index: int, strikes: Any) -> dict[str, FloatArray]:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < 7:
            raise ValueError("Indice de caplet candidat hors des sept fixings natifs.")
        query = _finite_query(strikes)
        try:
            return self._components_validated(index, query)
        except CandidatePricingError:
            raise
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            raise CandidatePricingError("Échec numérique du smile candidat : " + str(exc)) from exc

    def _components_validated(self, index: int, query: FloatArray) -> dict[str, FloatArray]:
        fit = self.wing_fit[index]
        d4 = d4_normal_vol(
            self.F[index],
            query,
            self.T[index],
            *(fit[key] for key in ("alpha", "beta", "rho", "nu")),
        )
        central = (
            _common_value(self._common_spline, query - self.F[index])
            if index < 3
            else self._tail_splines[index - 3](query)
        )
        x = np.clip((query + 0.015) / 0.015, 0, 1)
        weight = x**3 * (10 - 15 * x + 6 * x * x)
        normal = (1 - weight) * d4 + weight * central
        if np.any(~np.isfinite(normal)) or np.any(normal < 1e-10) or np.any(normal > 1.0):
            raise CandidatePricingError("Volatilité normale du candidat non finie ou hors bornes.")
        return {
            "normal_vol": normal,
            "d4_normal_vol": d4,
            "cubic_normal_vol": central,
            "blend_weight": weight,
        }

    def vol(self, index: int, strikes: Any) -> FloatArray:
        return self.components(index, strikes)["normal_vol"]

    def vol_for_coupon(self, forward: float, expiry: float, strikes: Any) -> FloatArray:
        """Bounded normal-vol interpolation in fixing time at the same K-F.

        This contractual-coupon policy is explicit and preserves every native
        value. It does not extrapolate before or after the native fixing range.
        Strike-domain validation belongs to the public pricing service; source
        bridge queries retain the frozen formula and their separate provenance.
        """
        query = _finite_query(strikes)
        if not isfinite(forward) or not isfinite(expiry):
            raise ValueError("Forward ou expiration du coupon candidat non fini.")
        if expiry < self.T[0] - 1e-12 or expiry > self.T[-1] + 1e-12:
            raise ValueError(
                "Fixing du coupon 3M hors des fixings natifs du candidat ; aucune extrapolation."
            )
        exact = np.flatnonzero(np.abs(self.T - expiry) <= 1e-12)
        if len(exact):
            index = int(exact[0])
            source_query = (
                query if abs(forward - self.F[index]) <= 1e-12 else query - forward + self.F[index]
            )
            return self.vol(index, source_query)
        right = int(np.searchsorted(self.T, expiry))
        left = right - 1
        weight = (expiry - self.T[left]) / (self.T[right] - self.T[left])
        lower = self.vol(left, query - forward + self.F[left])
        upper = self.vol(right, query - forward + self.F[right])
        return (1 - weight) * lower + weight * upper

    def shape_check(self, index: int) -> dict[str, float | bool]:
        """Same dense price-shape thresholds as the sealed comparison."""
        strikes = np.round(np.linspace(*self.domain, 2601), 12)
        vols = self.vol(index, strikes)
        calls = _undiscounted(self.F[index], strikes, self.T[index], vols)
        slopes = np.diff(calls) / np.diff(strikes)
        density = 2 * np.diff(slopes) / (strikes[2:] - strikes[:-2])
        return {
            "passed": bool(
                slopes.min() >= -1 - 1e-6 and slopes.max() <= 1e-6 and density.min() >= -1e-4
            ),
            "min_vol_bp": float(vols.min() * 1e4),
            "min_call_slope": float(slopes.min()),
            "max_call_slope": float(slopes.max()),
            "min_density": float(density.min()),
        }

    def native_repricing(self) -> dict[str, Any]:
        native_prices = np.asarray(
            [
                1e6
                * self.A[index]
                * _undiscounted(
                    self.F[index], self.strikes, self.T[index], self.vol(index, self.strikes)
                )
                for index in range(7)
            ]
        )
        points = [
            {
                "maturity_months": (12, 18, 24)[block],
                "strike": float(strike),
                "target_price_eur_1m": float(self.target_prices[block, j] * 1e6),
                "model_price_eur_1m": float(native_prices[:count, j].sum()),
                "residual_eur_1m": float(
                    native_prices[:count, j].sum() - self.target_prices[block, j] * 1e6
                ),
            }
            for block, count in enumerate((3, 5, 7))
            for j, strike in enumerate(self.strikes)
        ]
        return {
            "points": points,
            "max_abs_positive_eur_1m": max(
                abs(p["residual_eur_1m"]) for p in points if p["strike"] >= 0
            ),
            "max_abs_negative_eur_1m": max(
                (abs(p["residual_eur_1m"]) for p in points if p["strike"] < 0), default=None
            ),
        }


def _build_validated_candidate(
    three: StrippingFamily,
    rows3m: QuoteRows,
    valuation: ql.Date,
    strikes: FloatArray,
    positive: npt.NDArray[np.int64],
) -> FrozenThreeMonthCandidate:
    """Numerical construction after the native data contract has been checked."""
    nodes = tuple(node.model_copy(deep=True) for node in three.optionlets)
    forwards = np.array([node.forward_rate for node in nodes])
    expiries = np.array([node.expiry_years for node in nodes])
    annuities = np.array([node.accrual_fraction * node.discount_factor for node in nodes])
    target_prices = np.array(
        [
            [
                float(
                    np.sum(
                        annuities[:count]
                        * _undiscounted(
                            forwards[:count],
                            strike,
                            expiries[:count],
                            np.full(count, rows3m[block][1][j]),
                        )
                    )
                )
                for j, strike in enumerate(strikes)
            ]
            for block, count in enumerate((3, 5, 7))
        ]
    )
    spline, metadata = _fit_common(rows3m, strikes, positive, forwards, expiries, annuities)
    prefix = np.array([node.volatilities for node in nodes[:3]], dtype=float)
    prefix[:, positive] = np.array(
        [_common_value(spline, strikes[positive] - forwards[index]) for index in range(3)]
    )
    matrix = _allocate_tail(three, strikes, prefix, valuation)
    fits = tuple(
        fit_latest_smile(
            forwards[index], expiries[index], strikes[positive], matrix[index, positive], "3M"
        )
        for index in range(7)
    )
    return FrozenThreeMonthCandidate(
        nodes=nodes,
        strikes=_readonly(strikes),
        positive_strikes=_readonly(strikes[positive]),
        F=_readonly(forwards),
        T=_readonly(expiries),
        A=_readonly(annuities),
        matrix=_readonly(matrix),
        fit=metadata,
        wing_fit=fits,
        target_prices=_readonly(target_prices),
        _common_spline=spline,
        _tail_splines=tuple(
            CubicSpline(strikes[positive], matrix[index, positive], bc_type="natural")
            for index in range(3, 7)
        ),
    )


def build_candidate(
    three: StrippingFamily, rows3m: QuoteRows, valuation: ql.Date
) -> FrozenThreeMonthCandidate:
    """Validate the native layout, then classify calibration failures as pricing errors."""
    strikes, positive = _validate_inputs(three, rows3m)
    try:
        return _build_validated_candidate(three, rows3m, valuation, strikes, positive)
    except CandidatePricingError:
        raise
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise CandidatePricingError(
            "Échec numérique de la calibration candidate : " + str(exc)
        ) from exc
