"""Steps 1–5: Bachelier blocks and normal linear/quadratic time curves.

Volatilities are absolute rates (100 bp = .01); prices and vegas are per
unit notional. No Black conversion or CEV/SABR fitting enters this pipeline.
"""
from bisect import bisect_left
from datetime import date
from math import exp, fsum, pi, sqrt
import numpy as np
import QuantLib as ql
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from scipy.optimize import least_squares
from curves.helpers import date_to_ql, ql_evaluation_date
from services.thomas_optionlet import loaded_context
from services.thomas_stripping import _intrinsic, _time_value, _validated_quotes
from services.thomas_stripping_blocks import build_block_family

MIN_NORMAL_VOL = 1e-10
MAX_NORMAL_VOL = 1.0
METHODS = (("linear", False), ("quadratic", True))

class NormalCurvesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valuation_date: date
    source_token: str = Field(min_length=1)
    smoothing: FiniteFloat = Field(default=1e-8, ge=0, le=1)

def _normal_price(node, strike, vol):
    return _intrinsic(node, strike) + _time_value(node, strike, vol)

def _normal_vega_bp(node, strike, vol):
    std = vol * sqrt(node.expiry_years)
    d = (node.forward_rate - strike) / std
    return node.accrual_fraction * node.discount_factor * sqrt(node.expiry_years) * exp(-d*d/2) / sqrt(2*pi) * 1e-4

def _payment_time(valuation, node):
    return ql.Actual365Fixed().yearFraction(valuation, date_to_ql(node.payment_date))


def _initial_nodes(constants, times):
    count = len(constants)
    if count == 1:
        return np.array([constants[0], constants[0]], dtype=float)
    result = np.empty(count + 1, dtype=float)
    for k in range(1, count):
        left, right = times[k] - times[k - 1], times[k + 1] - times[k]
        result[k] = (constants[k - 1] * left + constants[k] * right) / (left + right)
    # Equation (16) and the endpoint rules in section 3.5 of the paper.
    result[0] = 1.5 * constants[0] - 0.5 * result[1]
    result[-1] = 1.5 * constants[-1] - 0.5 * result[-2]
    return np.clip(result, MIN_NORMAL_VOL, MAX_NORMAL_VOL)


def _vol_at(parameters, times, term, quadratic):
    i = min(max(bisect_left(times, term) - 1, 0), len(times) - 2)
    width = times[i + 1] - times[i]
    u = min(max((term - times[i]) / width, 0.0), 1.0)
    if quadratic:
        left, middle, right = parameters[2 * i : 2 * i + 3]
        return left * (2 * u * u - 3 * u + 1) + middle * (4 * u - 4 * u * u) + right * (2 * u * u - u)
    return parameters[i] * (1 - u) + parameters[i + 1] * u


def _slopes(parameters, times, quadratic):
    if not quadratic:
        return np.diff(parameters) / np.diff(times)
    slopes = []
    for i in range(len(times) - 1):
        half_width = (times[i + 1] - times[i]) / 2
        slopes.extend((
            (parameters[2 * i + 1] - parameters[2 * i]) / half_width,
            (parameters[2 * i + 2] - parameters[2 * i + 1]) / half_width,
        ))
    return np.asarray(slopes)


def _calibrate(intervals, times, constants, strike, smoothing, quadratic, left_boundary_vol=None):
    initial_nodes = _initial_nodes(constants, times)
    initial = (
        np.array([value for i in range(len(constants)) for value in
                  (initial_nodes[i], (initial_nodes[i] + initial_nodes[i + 1]) / 2)]
                 + [initial_nodes[-1]], dtype=float)
        if quadratic else initial_nodes
    )

    if left_boundary_vol is not None:
        initial[0] = left_boundary_vol

    def fitted_parameters(values):
        # The transferred prefix and fitted curve share this payment date.
        # Fix its value exactly; equal least_squares bounds are not permitted.
        return values if left_boundary_vol is None else np.r_[left_boundary_vol, values]

    def residuals(parameters):
        prices = []
        for interval in intervals:
            model = fsum(_time_value(node, strike, _vol_at(
                parameters, times, _payment_time(interval["valuation"], node), quadratic
            )) for node in interval["block"])
            prices.append(sqrt(1 / interval["end_time"]) * (model - interval["residual_time_value"]))
        slopes = _slopes(parameters, times, quadratic)
        # Scale both terms equally for numerical tolerances; λ and the minimizer are unchanged.
        return 10000 * np.r_[prices, sqrt(smoothing) * np.diff(slopes)]

    solved = least_squares(
        lambda values: residuals(fitted_parameters(values)),
        initial if left_boundary_vol is None else initial[1:],
        bounds=(MIN_NORMAL_VOL, MAX_NORMAL_VOL),
        max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-12,
    )
    if not solved.success or not np.all(np.isfinite(solved.x)):
        raise ValueError("Optimisation non convergée : " + solved.message)
    parameters = fitted_parameters(solved.x)
    # Positive nodal values alone do not guarantee a positive quadratic between nodes.
    for interval in intervals:
        for fraction in np.linspace(0, 1, 11):
            t = interval["start_time"] + fraction * (
                interval["end_time"] - interval["start_time"]
            )
            if _vol_at(parameters, times, t, quadratic) <= 0:
                raise ValueError("La courbe quadratique devient non positive entre deux nœuds.")
    final_residuals = residuals(parameters) / 10000
    price_count = len(intervals)
    return {
        "parameters": parameters,
        "initial_parameters": initial.tolist(),
        "initial_times": list(times) if not quadratic else [t for i in range(len(times)-1) for t in (times[i], (times[i]+times[i+1])/2)] + [times[-1]],
        "evaluations": solved.nfev,
        "weighted_price_residual_norm": float(np.linalg.norm(final_residuals[:price_count])),
        "smoothing_residual_norm": float(np.linalg.norm(final_residuals[price_count:])),
    }


def _curve_samples(prefix, prefix_vols, intervals, times, constants, methods, valuation):
    samples = []
    for node in prefix:
        term = _payment_time(valuation, node)
        vol = prefix_vols[node.number]
        samples.append({"maturity_years": term, "constant": vol,
                        "linear": vol, "quadratic": vol})
    for i, interval in enumerate(intervals):
        start, end = interval["start_time"], interval["end_time"]
        for j in range(16):
            term = start + (end - start) * j / 16
            samples.append({
                "maturity_years": term,
                "constant": constants[i],
                "linear": float(_vol_at(methods["linear"]["parameters"], times, term, False))
                if methods["linear"] is not None else None,
                "quadratic": float(_vol_at(methods["quadratic"]["parameters"], times, term, True))
                if methods["quadratic"] is not None else None,
            })
    end = intervals[-1]["end_time"]
    samples.append({
        "maturity_years": end,
        "constant": constants[-1],
        "linear": float(_vol_at(methods["linear"]["parameters"], times, end, False))
        if methods["linear"] is not None else None,
        "quadratic": float(_vol_at(methods["quadratic"]["parameters"], times, end, True))
        if methods["quadratic"] is not None else None,
    })
    return samples


def _strike_curve(valuation, family, rows, strike, column, smoothing):
    options = family.optionlets
    prefix = [node for node in options if node.source == "converted"]
    prefix_vols = {node.number: node.volatilities[column] for node in prefix}
    issues = []
    intervals = []
    count = len(prefix)
    previous_time = _payment_time(valuation, prefix[-1]) if prefix else 0.0
    for (month, flat), bootstrap in zip(rows, family.blocks):
        constant = bootstrap.volatilities[column]
        included = [node for node in options if node.cap_maturity_months <= month]
        block = included[count:]
        if constant is None or any(v is None for v in prefix_vols.values()):
            issues.extend(issue.message for issue in family.issues if issue.strike == strike)
            break
        market_price = fsum(_normal_price(node, strike, flat[column]) for node in included)
        end_time = _payment_time(valuation, included[-1])
        intrinsic = fsum(_intrinsic(node, strike) for node in block)
        residual_time = bootstrap.residual_time_values[column]
        intervals.append({"valuation": valuation, "month": month,
            "start_time": previous_time, "end_time": end_time,
            "included": included, "block": block, "constant": constant,
            "market_price": market_price, "normal_quote": flat[column],
            "intrinsic": intrinsic, "residual_time_value": residual_time,
            "residual_price": intrinsic + residual_time})
        previous_time, count = end_time, len(included)
    empty = {"strike": strike, "status": "unavailable", "issues": issues,
             "intervals": [], "points": [], "samples": [], "calibration": {},
             "max_curve_gap": None, "max_repricing_error": {}}
    if not intervals:
        return empty
    times = [intervals[0]["start_time"]] + [row["end_time"] for row in intervals]
    constants = [row["constant"] for row in intervals]
    methods = {}
    for name, quadratic in METHODS:
        try:
            methods[name] = _calibrate(
                intervals, times, constants, strike, smoothing, quadratic,
                left_boundary_vol=prefix_vols[prefix[-1].number] if prefix else None,
            )
        except (ValueError, RuntimeError) as exc:
            methods[name] = None
            issues.append(f"{name} : {exc}")
    points = []
    for node in options[:count]:
        term = _payment_time(valuation, node)
        values = {"constant": node.volatilities[column]}
        for name, quadratic in METHODS:
            values[name] = prefix_vols[node.number] if node.number in prefix_vols else (
                float(_vol_at(methods[name]["parameters"], times, term, quadratic))
                if methods[name] is not None else None)
        points.append({"fixing_date": str(node.fixing_date), "payment_date": str(node.payment_date),
            "maturity_years": term, "source": "raccord 3M→6M" if node.number in prefix_vols else "stripping", **values})
    by_number = {node.number: point for node, point in zip(options, points)}
    max_errors = {"constant": 0.0, "linear": None, "quadratic": None}
    result_rows = []
    for row in intervals:
        flows = []
        for node in row["included"]:
            point = by_number[node.number]
            vols = {name: point[name] for name in max_errors}
            prices = {name: _normal_price(node, strike, vol) if vol is not None else None for name, vol in vols.items()}
            vegas = {name: _normal_vega_bp(node, strike, vol) if vol is not None else None for name, vol in vols.items()}
            flows.append({**{key: str(getattr(node, key)) for key in ("fixing_date", "start_date", "end_date", "payment_date")},
                **{key: getattr(node, key) for key in ("accrual_fraction", "expiry_years", "forward_rate", "discount_factor")},
                "source": point["source"], "normal_quote_price": _normal_price(node, strike, row["normal_quote"]),
                "normal_vols": vols, "normal_prices": prices, "normal_vegas_bp": vegas})
        prices = {name: fsum(flow["normal_prices"][name] for flow in flows)
            if all(flow["normal_prices"][name] is not None for flow in flows) else None for name in max_errors}
        errors = {name: price-row["market_price"] if price is not None else None for name, price in prices.items()}
        for name, error in errors.items():
            if error is not None:
                max_errors[name] = max(max_errors[name] or 0, abs(error))
        result_rows.append({"expiry_months": row["month"], "start_years": row["start_time"], "end_years": row["end_time"],
            "normal_quote_bp": row["normal_quote"]*10000, "market_price": row["market_price"],
            "forward_price": row["residual_price"], "forward_time_value": row["residual_time_value"],
            "intrinsic": row["intrinsic"], "constant_vol": row["constant"], "model_prices": prices,
            "repricing_errors": errors, "cashflows": flows})
    samples = _curve_samples(prefix, prefix_vols, intervals, times, constants, methods, valuation)
    gaps = [abs(sample["linear"]-sample["quadratic"]) for sample in samples if sample["linear"] is not None and sample["quadratic"] is not None]
    return {"strike": strike, "status": "partial" if issues else "ready", "issues": issues,
        "intervals": result_rows, "points": points, "samples": samples,
        "calibration": {name: {key: value for key, value in fit.items() if key != "parameters"} if fit is not None else None for name, fit in methods.items()},
        "time_interpolation": {"times":times,"constant_vols":constants,
            "linear_parameters":methods["linear"]["parameters"].tolist() if methods["linear"] is not None else None,
            "quadratic_parameters":methods["quadratic"]["parameters"].tolist() if methods["quadratic"] is not None else None,
            "prefix":[{"time":_payment_time(valuation,node),"vol":prefix_vols[node.number]} for node in prefix]},
        "max_curve_gap": max(gaps) if gaps else None, "max_repricing_error": max_errors}

def prepare_normal_curves(request, market_store, curve_store):
    context = loaded_context(market_store, curve_store)
    if request.source_token != context.source_token:
        raise ValueError("Les données ou courbes ont changé : actualiser les sources.")
    if context.curve_date != request.valuation_date or context.cap_quotes_date != request.valuation_date:
        raise ValueError("La date des caps et des courbes doit correspondre à la valorisation.")
    if not all(any(curve.key == key and curve.current for curve in context.curves) for key in ("OIS", "3M", "6M")):
        raise ValueError("Construire les courbes €STR, Euribor 3M et Euribor 6M à jour.")
    data = market_store.get_capfloor_surface("EUR")
    if data is None:
        raise ValueError("Importer les cotations VOL CAP dans Market Data.")
    original, groups = _validated_quotes(data)
    if "3M" not in groups or "6M" not in groups:
        raise ValueError("Il faut les deux familles natives 3M et 6M.")
    selected = [j for j, strike in enumerate(original) if strike <= .05+1e-10]
    if not selected:
        raise ValueError("Importer au moins un strike fini jusqu’à 5 %.")
    strikes = [original[j] for j in selected]
    groups = {tenor: [(month, [flat[j] for j in selected]) for month, flat in rows] for tenor, rows in groups.items()}
    curve_set = curve_store.get_curve_set("EUR")
    with ql_evaluation_date(request.valuation_date):
        valuation = date_to_ql(request.valuation_date)
        spot = ql.TARGET().advance(valuation, 2, ql.Days)
        three = build_block_family(valuation, spot, "3M", groups["3M"], strikes, 1.0, curve_set, {}, False, build_black=False)
        six = build_block_family(valuation, spot, "6M", groups["6M"], strikes, 1.0, curve_set, {}, False, source_3m=three, basis_moneyness=True, build_black=False)
        families = [{"index_tenor": tenor, "strikes": strikes,
            "first_quoted_expiry": family.first_quoted_expiry, "last_quoted_expiry": family.last_quoted_expiry,
            "curves": [_strike_curve(valuation, family, groups[tenor], strike, j, request.smoothing) for j, strike in enumerate(strikes)]}
            for tenor, family in (("3M", three), ("6M", six))]
    return {"volatility_type": "normal", "valuation_date": str(request.valuation_date),
        "source_token": request.source_token, "smoothing": request.smoothing,
        "calibration_strikes": strikes,
        "excluded_strikes": [{"strike": strike, "reason": "Au-delà de 5 %."} for strike in original if strike not in strikes],
        "families": families,
        "assumptions": [
            "Vols flat normales des caps → prix Bachelier → déduction des caplets déjà strippés → vol normale commune de chaque bloc.",
            "Initialisation linéaire depuis les blocs constants (règles du PDF §3.5) ; initialisation quadratique avec les mêmes extrémités et leurs moyennes aux milieux.",
            "Ajustement des prix des blocs avec poids 1/T et pénalité λ sur les différences de pente en volatilité normale décimale ; prix par unité de notionnel.",
            "Interpolation selon la date de paiement ; le prix utilise toujours le temps jusqu’au fixing.",
            "Raccord court 3M→6M en volatilité normale à même moneyness : hypothèse de modèle, pas une cotation directe.",
            "Vega normale : dérivée à volatilités des autres caplets figées, par hausse de 1 bp ; aucune recalibration.",
            "Les cotations de caps ne déterminent pas seules les prix individuels des caplets. Le repricing des caps ne prouve pas l’accord avec ICE.",
        ]}
