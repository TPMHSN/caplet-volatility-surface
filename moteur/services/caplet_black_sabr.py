"""CEV-seeded Hagan SABR smiles for shifted-Black caplet curves.

The Bloomberg construction paper fits in normal-volatility space.  Its
blackToNormal/sabrToNormal conversions refer to a separate technical report:
here the market conversion is exact by price equivalence, and the model uses
the published Hagan normal approximation (Managing Smile Risk, B.69).
The output Black surface is the price-equivalent conversion of the fitted
normal formula. CEV and SABR use positive forward rates and strikes without displacement;
calibration targets and fitted Black quotes also use no quotation shift.
Both shifts are serialized with the calibrated parameters.
"""

from itertools import pairwise
from math import isfinite, log, sqrt

import numpy as np
import QuantLib as ql
from scipy.optimize import brentq, least_squares

from services.thomas_black import (
    black_otm_time_value, normal_otm_time_value, normal_to_black_vol,
)

DEFAULT_SABR_SHIFT = 0.0
DEFAULT_BLACK_QUOTE_SHIFT = 0.0


def _hagan_terms(forward, strikes, expiry, alpha, beta, rho, nu, sabr_shift=0.0):
    if not isfinite(sabr_shift) or sabr_shift < 0:
        raise ValueError("Le décalage SABR doit être fini et positif ou nul.")
    forward = forward + sabr_shift
    strikes = np.asarray(strikes, dtype=float) + sabr_shift
    if (
        not all(isfinite(float(v)) for v in (forward, expiry, alpha, beta, rho, nu))
        or not np.all(np.isfinite(strikes))
        or forward <= 0 or np.any(strikes <= 0) or expiry <= 0 or alpha <= 0
        or not 0 <= beta <= 1 or abs(rho) >= 1 or nu < 0
    ):
        raise ValueError("SABR exige F+s, K+s, T, α > 0 ; 0 ≤ β ≤ 1 ; |ρ| < 1 ; ν ≥ 0.")
    log_fk = np.log(forward / strikes)
    power = 1 - beta
    geometric = forward * strikes
    scale = geometric ** (power / 2)
    denominator = 1 + power * power * log_fk**2 / 24 + power**4 * log_fk**4 / 1920
    numerator = 1 + log_fk**2 / 24 + log_fk**4 / 1920
    z = (nu / alpha) * scale * log_fk
    ratio = np.empty_like(z)
    near = np.abs(z) < 1e-5
    ratio[near] = 1 - rho * z[near] / 2 + (2 - 3 * rho * rho) * z[near] ** 2 / 12
    other = z[~near]
    x = np.arcsinh((other - rho) / sqrt(1 - rho * rho)) + np.arctanh(rho)
    ratio[~near] = other / x
    correction_common = (
        rho * beta * nu * alpha / (4 * scale)
        + (2 - 3 * rho * rho) * nu * nu / 24
    )
    return strikes, geometric, scale, denominator, numerator, ratio, correction_common


def hagan_normal_vol(forward, strikes, expiry, alpha, beta, rho, nu, sabr_shift=0.0):
    """Hagan normal implied vol (B.69), optionally on displaced rates F+s, K+s."""
    _k, geometric, scale, denominator, numerator, ratio, common = _hagan_terms(
        forward, strikes, expiry, alpha, beta, rho, nu, sabr_shift
    )
    correction = 1 + expiry * (
        -beta * (2 - beta) * alpha * alpha / (24 * scale * scale) + common
    )
    result = alpha * geometric ** (beta / 2) * numerator / denominator * ratio * correction
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Volatilité normale Hagan non positive ou non finie.")
    return float(result) if result.ndim == 0 else result


def hagan_black_vol(forward, strikes, expiry, alpha, beta, rho, nu, sabr_shift=0.0):
    """Hagan Black vol (2.17), quoted on F+s and K+s when a shift is supplied."""
    _k, _geometric, scale, denominator, _numerator, ratio, common = _hagan_terms(
        forward, strikes, expiry, alpha, beta, rho, nu, sabr_shift
    )
    correction = 1 + expiry * (
        (1 - beta) ** 2 * alpha * alpha / (24 * scale * scale) + common
    )
    result = alpha / (scale * denominator) * ratio * correction
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Volatilité Black Hagan non positive ou non finie.")
    return float(result) if result.ndim == 0 else result


def _black_to_normal(forward, strike, expiry, black_vol, quote_shift=0.0):
    """Convert a shifted-Black quote by inverting its OTM time value."""
    if not isfinite(black_vol) or black_vol <= 0:
        raise ValueError("Volatilité Black cible absente ou non positive.")
    if (not all(isfinite(value) for value in (forward, strike, expiry, quote_shift))
            or quote_shift < 0 or expiry <= 0
            or forward + quote_shift <= 0 or strike + quote_shift <= 0):
        raise ValueError("La cotation Black exige F+q, K+q, T > 0 et un décalage q positif ou nul.")
    root_time = sqrt(expiry)
    price = black_otm_time_value(
        forward + quote_shift, strike + quote_shift, black_vol * root_time
    )
    if not isfinite(price) or price <= 0:
        raise ValueError("Valeur temps Black trop petite pour une conversion normale fiable.")

    def price_at(stddev):
        return normal_otm_time_value(forward, strike, stddev)

    def validated(stddev):
        return (isfinite(stddev) and stddev > 0
                and abs(price_at(stddev) / price - 1) <= 1e-8)

    # QL provides a fast initial inversion, but its tail price subtraction can
    # lose relative precision for low strikes at short expiries.
    option = ql.Option.Put if forward > strike else ql.Option.Call
    try:
        initial = ql.bachelierBlackFormulaImpliedVol(option, strike, forward, expiry, price)
        if validated(initial * root_time):
            return initial
    except RuntimeError:
        pass
    upper = max(abs(forward - strike), black_vol * root_time * sqrt(
        (forward + quote_shift) * (strike + quote_shift)), 1e-6)
    for _ in range(64):
        if price_at(upper) >= price:
            break
        upper *= 2
    else:
        raise ValueError("Impossible d'encadrer la volatilité normale équivalente.")
    stddev = brentq(
        lambda value: price_at(value) / price - 1, 0.0, upper,
        xtol=1e-15, maxiter=250,
    )
    if not validated(stddev):
        raise ValueError("Conversion Black→normale insuffisamment précise.")
    return stddev / root_time


def equivalent_black_sabr_vol(
    forward, strike, expiry, alpha, beta, rho, nu, sabr_shift=0.0, quote_shift=0.0,
):
    """Express the SABR price in a requested Black quotation convention.

    The model displacement and quote displacement are independent. Keeping the
    low-level quote default at zero preserves explicit undisplaced comparisons.
    """
    if not isfinite(quote_shift) or quote_shift < 0:
        raise ValueError("Le décalage de cotation Black doit être fini et positif ou nul.")
    normal = hagan_normal_vol(forward, strike, expiry, alpha, beta, rho, nu, sabr_shift)
    black, reason, _ = normal_to_black_vol(
        forward, strike, expiry, normal, shift=quote_shift
    )
    if black is None:
        raise ValueError(f"SABR normal→Black, K={strike:.6g} : {reason}")
    return black


def _calibrate_cevs(forward, expiry, strikes, target, weights, previous):
    level = float(np.median(target))
    starts = [0.0, 0.5, 1.0]
    if previous is not None:
        starts.insert(0, previous["beta"])
    candidates = []

    def residual(parameters):
        alpha, beta = np.exp(parameters[0]), parameters[1]
        try:
            return np.sqrt(weights) * (
                hagan_normal_vol(forward, strikes, expiry, alpha, beta, 0.0, 0.0) - target
            ) * 1e4
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(len(strikes), 1e6)

    for beta in starts:
        alpha = np.clip(level / forward**beta, 1e-8, 100.0)
        solved = least_squares(
            residual, [log(alpha), float(np.clip(beta, 0, 1))],
            bounds=([log(1e-9), 0.0], [log(1000.0), 1.0]),
            max_nfev=180, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        if solved.success and np.all(np.isfinite(solved.fun)) and np.max(np.abs(solved.fun)) < 1e6:
            candidates.append(solved)
    if not candidates:
        raise ValueError("Calibration CEV non convergée.")
    best = min(candidates, key=lambda item: float(np.dot(item.fun, item.fun)))
    alpha, beta = float(np.exp(best.x[0])), float(best.x[1])
    modeled = hagan_normal_vol(forward, strikes, expiry, alpha, beta, 0.0, 0.0)
    return {"alpha": alpha, "beta": beta,
            "rmse_normal_bp": float(np.sqrt(np.mean(((modeled - target) * 1e4) ** 2)))}, best


def _calibrate_sabr(forward, expiry, strikes, target, weights, cev, previous):
    nu_max = min(5.0, sqrt(12.0 / expiry))
    starts = [(cev["alpha"], cev["beta"], -0.5, 0.2),
              (cev["alpha"], cev["beta"], 0.0, 0.05),
              (cev["alpha"], cev["beta"], 0.5, 0.5)]
    # A constant/CEV target is a valid SABR limit. Starting only at positive
    # vol-of-vol can exhaust the optimizer on this nearly flat objective.
    if cev["rmse_normal_bp"] < 1e-6:
        starts.insert(0, (cev["alpha"], cev["beta"], 0.0, 0.0))
    if previous is not None:
        starts.insert(0, (previous["alpha"], previous["beta"], previous["rho"], previous["nu"]))
    candidates = []

    def residual(parameters):
        alpha, beta, rho, nu = np.exp(parameters[0]), *parameters[1:]
        try:
            return np.sqrt(weights) * (
                hagan_normal_vol(forward, strikes, expiry, alpha, beta, rho, nu) - target
            ) * 1e4
        except (ValueError, FloatingPointError, OverflowError):
            return np.full(len(strikes), 1e6)

    for alpha, beta, rho, nu in starts:
        alpha = float(np.clip(alpha, 1e-8, 100))
        guess = [log(alpha), float(np.clip(beta, 0, 1)),
                 float(np.clip(rho, -0.998, 0.998)), float(np.clip(nu, 1e-8, nu_max * 0.95))]
        solved = least_squares(
            residual, guess,
            bounds=([log(1e-9), 0.0, -0.999, 0.0],
                    [log(1000.0), 1.0, 0.999, nu_max]),
            max_nfev=350, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        if solved.success and np.all(np.isfinite(solved.fun)) and np.max(np.abs(solved.fun)) < 1e6:
            candidates.append(solved)
    if not candidates:
        raise ValueError("Calibration SABR non convergée.")
    best = min(candidates, key=lambda item: float(np.dot(item.fun, item.fun)))
    alpha, beta, rho, nu = map(float, (np.exp(best.x[0]), *best.x[1:]))
    modeled = hagan_normal_vol(forward, strikes, expiry, alpha, beta, rho, nu)
    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu,
            "rmse_normal_bp": float(np.sqrt(np.mean(((modeled - target) * 1e4) ** 2)))}, best


def calibrate_black_smile(
    forward, expiry, strikes, black_vols, previous=None, sabr_shift=DEFAULT_SABR_SHIFT,
    quote_shift=DEFAULT_BLACK_QUOTE_SHIFT,
):
    """Fit CEV then four free SABR parameters; return transparent diagnostics.

    ``previous`` is a calibrated SABR dict used only as an optimizer start.
    CEV/SABR fit F+s and K+s; input/output Black quotes use F+q and K+q.
    Missing target strikes leave the slice unavailable; no synthetic quotes are made.
    """
    record = {"status": "unavailable", "sabr_shift": sabr_shift,
              "quote_shift": quote_shift, "input_strikes": list(strikes),
              "input_black_vols": list(black_vols), "cev": None, "sabr": None,
              "warnings": []}
    try:
        if not isfinite(sabr_shift) or sabr_shift < 0:
            raise ValueError("Le décalage SABR doit être fini et positif ou nul.")
        if not isfinite(quote_shift) or quote_shift < 0:
            raise ValueError("Le décalage de cotation Black doit être fini et positif ou nul.")
        if len(strikes) < 4 or len(black_vols) != len(strikes):
            raise ValueError("Au moins quatre cibles Black, une par strike, sont nécessaires pour SABR.")
        if any(not isfinite(k) for k in strikes):
            raise ValueError("Les strikes doivent être finis.")
        if any(b <= a for a, b in pairwise(strikes)):
            raise ValueError("Les strikes doivent être strictement croissants.")
        if any(vol is None or not isfinite(vol) or vol <= 0 for vol in black_vols):
            raise ValueError("Un des points Black manque : smile non calibré.")
        if not isfinite(forward) or not isfinite(expiry) or expiry <= 0:
            raise ValueError("Forward fini et temps jusqu'au fixing strictement positif requis.")
        if forward + sabr_shift <= 0 or any(k + sabr_shift <= 0 for k in strikes):
            raise ValueError("SABR exige F+s et K+s strictement positifs.")
        if forward + quote_shift <= 0 or any(k + quote_shift <= 0 for k in strikes):
            raise ValueError("La cotation Black exige F+q et K+q strictement positifs.")
        grid = np.linspace(strikes[0], strikes[-1], 81)
        strikes = np.asarray(strikes, dtype=float)
        black_vols = np.asarray(black_vols, dtype=float)
        target = np.asarray([
            _black_to_normal(forward, float(k), expiry, float(v), quote_shift)
            for k, v in zip(strikes, black_vols)
        ])
        # Equation (26) of Hagan/Konikov.  With decimal rates these weights
        # are nearly uniform; that unit convention is kept explicit.
        weights = 1 / np.sqrt(1 + (strikes - forward) ** 2)
        shifted_forward, shifted_strikes = forward + sabr_shift, strikes + sabr_shift
        cev, _cev_fit = _calibrate_cevs(
            shifted_forward, expiry, shifted_strikes, target, weights, previous
        )
        cev["sabr_shift"] = sabr_shift
        cev["quote_shift"] = quote_shift
        record["cev"] = cev
        sabr, fit = _calibrate_sabr(
            shifted_forward, expiry, shifted_strikes, target, weights, cev, previous
        )
        sabr["sabr_shift"] = sabr_shift
        sabr["quote_shift"] = quote_shift
        params = {key: sabr[key] for key in ("alpha", "beta", "rho", "nu", "sabr_shift", "quote_shift")}
        black_model = np.asarray([
            equivalent_black_sabr_vol(forward, float(k), expiry, **params)
            for k in strikes
        ])
        dense = np.asarray([
            equivalent_black_sabr_vol(forward, float(k), expiry, **params)
            for k in grid
        ])
        errors = (black_model - black_vols) * 1e4
        sabr.update({
            "model_black_vols": black_model.tolist(),
            "errors_black_bp": errors.tolist(),
            "rmse_black_bp": float(np.sqrt(np.mean(errors * errors))),
            "max_abs_error_black_bp": float(np.max(np.abs(errors))),
            "smile_strikes": grid.tolist(), "smile_black_vols": dense.tolist(),
        })
        record["sabr"] = sabr
        record["status"] = "calibrated"
        if sabr["beta"] < 1e-4 or sabr["beta"] > 1 - 1e-4:
            record["warnings"].append("β proche d'une borne : son identification est fragile.")
        if abs(sabr["rho"]) > 0.998 or sabr["nu"] > 0.999 * min(5, sqrt(12 / expiry)):
            record["warnings"].append("ρ ou ν proche d'une borne : smile à examiner.")
        if sabr["nu"] ** 2 * expiry > 1:
            record["warnings"].append("ν²T > 1 : approximation asymptotique à examiner.")
        if np.any(fit.active_mask):
            record["warnings"].append("Au moins un paramètre SABR est sur une borne.")
        if sabr["rmse_normal_bp"] > 1:
            record["warnings"].append("RMSE du smile supérieur à 1 bp de volatilité normale : prix des caps à contrôler.")
        if sabr["max_abs_error_black_bp"] > 10:
            record["warnings"].append("Écart nodal supérieur à 10 bp de volatilité Black : interpolation à contrôler.")
        calls = np.array([
            ql.blackFormula(ql.Option.Call, float(k) + quote_shift, forward + quote_shift,
                            float(v) * sqrt(expiry))
            for k, v in zip(grid, dense)
        ])
        slopes = np.diff(calls) / np.diff(grid)
        if np.any(slopes > 1e-8) or np.any(slopes < -1 - 1e-8) or np.any(np.diff(slopes) < -1e-7):
            raise ValueError("Prix call SABR non monotones ou non convexes en strike sur la grille : smile non utilisable.")
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        record["status"] = "unavailable"
        record["sabr"] = None
        record["warnings"].append(str(exc))
    return record
