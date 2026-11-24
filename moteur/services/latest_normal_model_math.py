"""Versioned latest-normal smile policies with unshifted Bachelier pricing.

The 3% displacement belongs only to the D.4 component. Pricing always uses
the original forward and strike. The 3M policy is financially unchanged from
v2; v3 adds an explicit native-preserving 6M policy.
"""

from collections.abc import Mapping
from math import isfinite, pi, sqrt
from typing import NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.optimize import OptimizeResult, least_squares, minimize
from scipy.special import ndtr

from services.latest_normal_local_price_repair import LocalPriceRepair, apply_local_price_repair
from services.normal_cev_smiles import _targets, normal_strike_vol

MODEL_SHIFT = 0.03
STRIKE_DOMAIN = (-0.015, 0.05)
TIME_SMOOTHING = 1e-8
THREE_MONTH_SMILE_POLICY = "d4_to_native_cubic_positive"
SIX_MONTH_SMILE_POLICY = "d4_below_native_boundary_native_cubic"
THREE_MONTH_BLEND = (-0.005, 0.02)
SIX_MONTH_TRANSITION_WIDTH = 0.0075
D4_BOUNDARY_TIE_TOLERANCE_BP = 1e-6
_D4_LOWER_BOUNDS = np.asarray([0.0, -0.995, 0.0])
_D4_UPPER_BOUNDS = np.asarray([1.0, 0.995, 3.0])

FloatArray = npt.NDArray[np.float64]
NumericVector = list[float] | FloatArray


class SmileFitResult(TypedDict):
    price_repair: NotRequired[LocalPriceRepair]
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_normal_vol: float
    fit_rmse_bp: float
    fit_cost: float
    beta_penalty: float
    bound_hit: bool
    pricing_policy: str
    native_boundary_strike: float | None
    boundary_tie_error_bp: float | None
    optimizer_success: bool
    optimizer_status: int


class SmileComponents(TypedDict):
    normal_vol: float | FloatArray
    d4_normal_vol: float | FloatArray
    cubic_normal_vol: float | FloatArray
    blend_weight: float | FloatArray


def smile_policy(tenor: str) -> str:
    if tenor == "3M":
        return THREE_MONTH_SMILE_POLICY
    if tenor == "6M":
        return SIX_MONTH_SMILE_POLICY
    raise ValueError("Indice inconnu.")


def smile_transition(strikes: list[float] | np.ndarray, tenor: str) -> tuple[float, float]:
    if tenor == "3M":
        return THREE_MONTH_BLEND
    if tenor != "6M":
        raise ValueError("Indice inconnu.")
    native_boundary = float(np.min(np.asarray(strikes, dtype=float)))
    transition_start = max(STRIKE_DOMAIN[0], native_boundary - SIX_MONTH_TRANSITION_WIDTH)
    return transition_start, native_boundary


def d4_normal_vol(forward, strike, expiry, alpha, beta, rho, nu):
    inputs = (forward, expiry, alpha, beta, rho, nu)
    k = np.asarray(strike, dtype=float) + MODEL_SHIFT
    f = float(forward) + MODEL_SHIFT
    if (
        not all(isfinite(float(v)) for v in inputs)
        or not np.all(np.isfinite(k))
        or f <= 0
        or np.any(k <= 0)
        or expiry <= 0
        or alpha <= 0
        or not 0 <= beta <= 1
        or abs(rho) >= 1
        or nu < 0
    ):
        raise ValueError("Paramètres D.4 hors du domaine du modèle.")
    power = 1 - beta
    log_fk = np.log(f / k)
    integral = log_fk if power < 1e-10 else k**power * np.expm1(power * log_fk) / power
    prefactor = np.empty_like(k)
    near = np.abs(log_fk) < 1e-10
    prefactor[near] = f**beta
    prefactor[~near] = k[~near] * np.expm1(log_fk[~near]) / integral[~near]
    z = nu * integral / alpha
    ratio = np.empty_like(z)
    near_z = np.abs(z) < 1e-5
    ratio[near_z] = 1 - rho * z[near_z] / 2 + (2 - 3 * rho * rho) * z[near_z] ** 2 / 12
    zo = z[~near_z]
    chi = np.arcsinh((zo - rho) / sqrt(1 - rho * rho)) + np.arctanh(rho)
    ratio[~near_z] = zo / chi
    geom = f * k
    numerator = 1 + expiry * (
        rho * beta * alpha * nu / (4 * geom ** (power / 2)) + (2 - 3 * rho * rho) * nu * nu / 24
    )
    denominator = 1 + expiry * beta * (2 - beta) * alpha * alpha / (24 * geom**power)
    out = alpha * prefactor * ratio * numerator / denominator
    if np.any(~np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("D.4 produit une volatilité non positive ou non finie.")
    return float(out) if out.ndim == 0 else out


def alpha_d4(forward, expiry, beta, rho, nu, target):
    if (
        not all(isfinite(float(v)) for v in (forward, expiry, beta, rho, nu, target))
        or target <= 0
        or forward + MODEL_SHIFT <= 0
    ):
        raise ValueError("Cible ATM D.4 invalide.")
    f = forward + MODEL_SHIFT
    fb, scale = f**beta, f ** (1 - beta)
    a = expiry * beta * (2 - beta) / (24 * scale * scale)
    b = expiry * rho * beta * nu / (4 * scale)
    c = expiry * (2 - 3 * rho * rho) * nu * nu / 24
    q, d = target * a - fb * b, fb * (1 + c)
    if abs(q) < 1e-14:
        if d <= 0:
            raise ValueError("Aucune racine ATM positive.")
        alpha = target / d
    else:
        discriminant = d * d - 4 * q * target
        if discriminant < 0 or d + sqrt(discriminant) <= 0:
            raise ValueError("Aucune petite racine ATM positive.")
        alpha = 2 * target / (d + sqrt(discriminant))
    if (
        not isfinite(alpha)
        or alpha <= 0
        or abs(d4_normal_vol(forward, forward, expiry, alpha, beta, rho, nu) / target - 1) > 1e-9
    ):
        raise ValueError("Ancrage ATM D.4 insuffisamment précis.")
    return alpha


def _fit_three_month_smile(
    forward: float,
    expiry: float,
    k: FloatArray,
    v: FloatArray,
    seed: Mapping[str, float] | None = None,
) -> SmileFitResult:
    grid = np.unique(np.r_[np.linspace(k[0], k[-1], 81), k, forward])
    cubic = normal_strike_vol(k, v, grid)
    atm = normal_strike_vol(k, v, forward)
    widths = np.diff(grid)
    quadrature = np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]
    std = cubic * sqrt(expiry)
    distance = -np.abs(forward - grid)
    z = distance / std
    density = np.exp(-z * z / 2) / sqrt(2 * pi)
    otm = distance * ndtr(z) + std * density
    vega = sqrt(expiry) * density
    relative = np.minimum(vega / np.maximum(otm, 1e-12), 1e4) ** 2
    penalty = 8.0
    weights = quadrature * relative
    weights /= weights.sum()

    def unpack(x: FloatArray) -> tuple[float, float, float, float]:
        beta, rho, nu = map(float, x)
        return alpha_d4(forward, expiry, beta, rho, nu, atm), beta, rho, nu

    def residual(x: FloatArray) -> FloatArray:
        try:
            error = (
                np.sqrt(weights) * (d4_normal_vol(forward, grid, expiry, *unpack(x)) - cubic) * 1e4
            )
            return np.r_[error, sqrt(penalty) * (x[0] - 0.5)]
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(len(grid) + 1, 1e6)

    starts = [(0.5, -0.3, 0.5), (0.5, 0.0, 1.5), (0.0, 0.0, 0.5), (1.0, 0.3, 0.5)]
    if seed is not None:
        starts.insert(0, tuple(seed[q] for q in ("beta", "rho", "nu")))
    candidates: list[OptimizeResult] = []
    for initial in starts:
        solved = least_squares(
            residual,
            initial,
            bounds=(_D4_LOWER_BOUNDS, _D4_UPPER_BOUNDS),
            max_nfev=300,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        if solved.success and np.max(np.abs(solved.fun)) < 1e5:
            candidates.append(solved)
    if not candidates:
        raise ValueError("Calibration D.4 non convergée.")
    best = min(candidates, key=lambda solution: float(solution.fun @ solution.fun))
    alpha, beta, rho, nu = unpack(best.x)
    errors = 1e4 * (d4_normal_vol(forward, grid, expiry, alpha, beta, rho, nu) - cubic)
    return {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "nu": nu,
        "atm_normal_vol": atm,
        "fit_rmse_bp": float(np.sqrt(np.mean(errors**2))),
        "fit_cost": float(best.fun @ best.fun),
        "beta_penalty": penalty,
        "bound_hit": bool(
            beta < 1e-5 or beta > 1 - 1e-5 or abs(rho) > 0.9949 or nu < 1e-5 or nu > 2.9999
        ),
        "pricing_policy": THREE_MONTH_SMILE_POLICY,
        "native_boundary_strike": None,
        "boundary_tie_error_bp": None,
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
    }


def _fit_six_month_smile(
    forward: float,
    expiry: float,
    k: FloatArray,
    v: FloatArray,
    seed: Mapping[str, float] | None = None,
) -> SmileFitResult:
    native_boundary = float(k[0])
    if native_boundary < STRIKE_DOMAIN[0] - 1e-12:
        raise ValueError("La frontière native 6M est hors du domaine validé.")
    grid = np.unique(np.r_[np.linspace(k[0], k[-1], 81), k, forward])
    cubic = normal_strike_vol(k, v, grid)
    atm = normal_strike_vol(k, v, forward)
    widths = np.diff(grid)
    quadrature = np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]
    weights = quadrature / np.sqrt(1 + (grid - forward) ** 2)
    weights /= weights.sum()
    penalty = 10.0

    def unpack(x: FloatArray) -> tuple[float, float, float, float]:
        beta, rho, nu = map(float, x)
        return alpha_d4(forward, expiry, beta, rho, nu, atm), beta, rho, nu

    def objective(x: FloatArray) -> float:
        try:
            error = (
                np.sqrt(weights) * (d4_normal_vol(forward, grid, expiry, *unpack(x)) - cubic) * 1e4
            )
            value = float(error @ error + penalty * (float(x[0]) - 0.5) ** 2)
            return value if isfinite(value) else 1e12
        except (ValueError, FloatingPointError, OverflowError):
            return 1e12

    def boundary_error_bp(x: FloatArray) -> float:
        try:
            error = 1e4 * (d4_normal_vol(forward, native_boundary, expiry, *unpack(x)) - v[0])
            return float(error) if isfinite(float(error)) else 1e6
        except (ValueError, FloatingPointError, OverflowError):
            return 1e6

    starts = [
        (1.0, -0.995, 0.1),
        (0.5, -0.3, 0.5),
        (0.5, 0.0, 1.5),
        (0.0, 0.0, 0.5),
        (1.0, 0.3, 0.5),
    ]
    if seed is not None:
        starts.insert(0, tuple(seed[q] for q in ("beta", "rho", "nu")))
    solved_candidates: list[OptimizeResult] = []
    observed_boundary_errors: list[float] = []
    for initial in starts:
        solved = minimize(
            objective,
            np.clip(np.asarray(initial, dtype=float), _D4_LOWER_BOUNDS, _D4_UPPER_BOUNDS),
            method="SLSQP",
            bounds=tuple(zip(_D4_LOWER_BOUNDS, _D4_UPPER_BOUNDS, strict=True)),
            constraints=({"type": "eq", "fun": boundary_error_bp},),
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if np.all(np.isfinite(solved.x)):
            observed_error = abs(boundary_error_bp(solved.x))
            observed_boundary_errors.append(observed_error)
            within_bounds = np.all(solved.x >= _D4_LOWER_BOUNDS - 1e-12) and np.all(
                solved.x <= _D4_UPPER_BOUNDS + 1e-12
            )
            if (
                within_bounds
                and observed_error <= D4_BOUNDARY_TIE_TOLERANCE_BP
                and objective(solved.x) < 1e11
            ):
                solved_candidates.append(solved)
    if not solved_candidates:
        observed = min(observed_boundary_errors, default=float("inf"))
        raise ValueError(
            "Ancrage frontière D.4 6M infaisable dans les bornes : "
            f"écart observé {observed:.9g} bp, tolérance "
            f"{D4_BOUNDARY_TIE_TOLERANCE_BP:.9g} bp."
        )
    best = min(
        solved_candidates,
        key=lambda solution: (objective(solution.x), *(float(value) for value in solution.x)),
    )
    alpha, beta, rho, nu = unpack(best.x)
    boundary_tie_error_bp = boundary_error_bp(best.x)
    if abs(boundary_tie_error_bp) > D4_BOUNDARY_TIE_TOLERANCE_BP:
        raise ValueError("Ancrage frontière D.4 6M insuffisamment précis.")
    errors = 1e4 * (d4_normal_vol(forward, grid, expiry, alpha, beta, rho, nu) - cubic)
    return {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "nu": nu,
        "atm_normal_vol": atm,
        "fit_rmse_bp": float(np.sqrt(np.mean(errors**2))),
        "fit_cost": objective(best.x),
        "beta_penalty": penalty,
        "bound_hit": bool(
            beta < 1e-5 or beta > 1 - 1e-5 or abs(rho) > 0.9949 or nu < 1e-5 or nu > 2.9999
        ),
        "pricing_policy": SIX_MONTH_SMILE_POLICY,
        "native_boundary_strike": native_boundary,
        "boundary_tie_error_bp": boundary_tie_error_bp,
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
    }


def fit_latest_smile(
    forward: float,
    expiry: float,
    strikes: NumericVector,
    vols: NumericVector,
    tenor: str,
    seed: Mapping[str, float] | None = None,
) -> SmileFitResult:
    k, v = _targets(strikes, vols)
    if tenor not in ("3M", "6M"):
        raise ValueError("Indice inconnu.")
    if tenor == "3M" and np.any(k < 0):
        raise ValueError("Le smile 3M se calibre sur les strikes nuls et positifs.")
    if not k[0] <= forward <= k[-1]:
        raise ValueError("L’ATM doit être couvert par les strikes de calibration.")
    if tenor == "3M":
        return _fit_three_month_smile(forward, expiry, k, v, seed)
    return _fit_six_month_smile(forward, expiry, k, v, seed)


def latest_smile_components(
    forward: float,
    expiry: float,
    strikes: NumericVector,
    targets: NumericVector,
    parameters: SmileFitResult,
    tenor: str,
    strike: float | FloatArray,
) -> SmileComponents:
    query = np.asarray(strike, dtype=float)
    if (
        not np.all(np.isfinite(query))
        or np.any(query < STRIKE_DOMAIN[0])
        or np.any(query > STRIKE_DOMAIN[1])
    ):
        raise ValueError("Strike hors du domaine validé [−1,5 % ; 5 %].")
    k, v = _targets(strikes, targets)
    d4 = np.asarray(
        d4_normal_vol(
            forward,
            query,
            expiry,
            *(parameters[q] for q in ("alpha", "beta", "rho", "nu")),
        )
    )
    cubic = CubicSpline(k, v, bc_type="natural", extrapolate=True)(query)
    transition_start, transition_end = smile_transition(k, tenor)
    if transition_end <= transition_start + 1e-14:
        weight = np.ones_like(query)
    else:
        x = np.clip((query - transition_start) / (transition_end - transition_start), 0, 1)
        weight = x * x * x * (10 - 15 * x + 6 * x * x)
    vol = (1 - weight) * d4 + weight * cubic
    repair = parameters.get("price_repair")
    if repair is not None:
        vol = apply_local_price_repair(forward, expiry, query, vol, repair)
    if np.any(~np.isfinite(vol)) or np.any(vol <= 0):
        raise ValueError("Le smile final devient non positif ou non fini.")
    scalar = query.ndim == 0
    return {
        "normal_vol": float(vol) if scalar else vol,
        "d4_normal_vol": float(d4) if scalar else d4,
        "cubic_normal_vol": float(cubic) if scalar else cubic,
        "blend_weight": float(weight) if scalar else weight,
    }


def latest_normal_vol(forward, expiry, strikes, targets, parameters, tenor, strike):
    return latest_smile_components(forward, expiry, strikes, targets, parameters, tenor, strike)[
        "normal_vol"
    ]
