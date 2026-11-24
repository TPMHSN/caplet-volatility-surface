"""Bounded experimental normal-vol extensions around the unchanged native model.

No market quote is added. Native successful branches return no override. The
long 3M hypothesis equates both quarterly normal-vol amplitudes, freezes tenor
basis, and derives their scale from first-order six-month compounding.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import copy
from datetime import date
from math import isfinite, pi, sqrt
from types import SimpleNamespace

import numpy as np
import QuantLib as ql
from scipy.special import ndtr

from curves.helpers import date_to_ql, ql_to_date
from services.latest_normal_coupon import coupon_forward_contract
from services.latest_normal_extension_schemas import ExtensionConfig, ExtensionCoverage
from services.latest_normal_model_math import STRIKE_DOMAIN

EXTENSION_VERSION = "experimental_31y_v1"
VALIDATION_STATUS = "experimental_not_independently_validated"
_TOL = 1e-10


def _native_rows(snapshot, tenor):
    # Risk snapshots omit display diagnostics; the calibrated cashflows remain.
    for curve in snapshot.families[tenor]:
        if curve.get("intervals"):
            return curve["intervals"][-1]["cashflows"]
    for family in snapshot.response.get("families", []):
        if family["index_tenor"] == tenor:
            return [row for row in family["slices"] if row.get("status", "ready") == "ready"]
    raise ValueError("Calendrier natif indisponible pour l'extension expérimentale.")


def _time_bounds(snapshot, tenor):
    bounds = []
    for curve in snapshot.families[tenor]:
        fitted = curve.get("time_interpolation")
        if fitted:
            prefix = fitted.get("prefix", [])
            bounds.append(
                (
                    prefix[0]["time"] if prefix else fitted["times"][0],
                    fitted["times"][-1],
                )
            )
        elif curve.get("samples"):
            bounds.append(
                (
                    curve["samples"][0]["maturity_years"],
                    curve["samples"][-1]["maturity_years"],
                )
            )
    if not bounds:
        raise ValueError("Domaine temporel natif indisponible pour l'extension.")
    return max(float(lo) for lo, _ in bounds), min(float(hi) for _, hi in bounds)


def _checked_vols(evaluate, forward, expiry, strike):
    # Check the actual coupon, including the requested strike and ATM, so that
    # transport or a change in expiry cannot inherit an invalid native shape.
    grid = np.unique(np.round(np.r_[np.linspace(*STRIKE_DOMAIN, 2601), forward, strike], 12))
    grid = grid[(grid >= STRIKE_DOMAIN[0]) & (grid <= STRIKE_DOMAIN[1])]
    vols = np.asarray(evaluate(grid), dtype=float)
    if vols.shape != grid.shape:
        vols = np.broadcast_to(vols, grid.shape)
    if np.any(~np.isfinite(vols)) or np.any(vols <= 0.0):
        raise ValueError("Smile expérimental : volatilité non positive ou non finie.")
    std = vols * sqrt(expiry)
    z = (forward - grid) / std
    calls = (forward - grid) * ndtr(z) + std * np.exp(-z * z / 2) / sqrt(2 * pi)
    slopes = np.diff(calls) / np.diff(grid)
    density = 2 * np.diff(slopes) / (grid[2:] - grid[:-2])
    if (
        np.any(~np.isfinite(calls))
        or slopes.min() < -1.0 - 1e-6
        or slopes.max() > 1e-6
        or density.min() < -1e-4
    ):
        raise ValueError("Smile expérimental non monotone ou non convexe ; pricing indisponible.")
    sigma = float(evaluate(float(strike)))
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("Volatilité expérimentale non positive ou non finie.")
    return sigma


def make_extension_provider(config: ExtensionConfig | dict | None = None):
    """Return the private coupon provider; None preserves a native branch.

    The provider owns a bounded fit cache. It never mutates source snapshots,
    candidates, fixing histories, rate curves, or native quote matrices.
    """
    config = (
        config
        if isinstance(config, ExtensionConfig)
        else ExtensionConfig.model_validate(config or {})
    )
    fits = OrderedDict()

    def native_smile(snapshot, request, forward, expiry, start, end, fixing, payment):
        from services import latest_normal_model as native

        targets, _, notes = native._pricing_targets(
            snapshot, request, forward, expiry, start, end, fixing, payment
        )
        local = copy(snapshot)
        local.fits = fits
        params = native._fit(local, request.index_tenor, forward, expiry, targets)
        strikes = [curve["strike"] for curve in snapshot.families[request.index_tenor]]

        def evaluate(query):
            query = np.asarray(query, dtype=float)
            inside = (query >= STRIKE_DOMAIN[0]) & (query <= STRIKE_DOMAIN[1])
            if np.all(inside):
                return native.latest_normal_vol(
                    forward, expiry, strikes, targets, params, request.index_tenor, query
                )
            # The public native evaluator correctly rejects source wings. Here
            # evaluate its same frozen D.4/cubic formula explicitly, reporting
            # extrapolation separately and checking the transported price shape.
            from scipy.interpolate import CubicSpline

            from services.latest_normal_model_math import d4_normal_vol, smile_transition

            result = np.empty_like(query)
            if np.any(inside):
                result[inside] = native.latest_normal_vol(
                    forward, expiry, strikes, targets, params, request.index_tenor, query[inside]
                )
            wing = query[~inside]
            d4 = d4_normal_vol(
                forward, wing, expiry, *(params[key] for key in ("alpha", "beta", "rho", "nu"))
            )
            cubic = CubicSpline(strikes, targets, bc_type="natural", extrapolate=True)(wing)
            lo, hi = smile_transition(strikes, request.index_tenor)
            if hi <= lo + 1e-14:
                weight = np.ones_like(wing)
            else:
                x = np.clip((wing - lo) / (hi - lo), 0, 1)
                weight = x * x * x * (10 - 15 * x + 6 * x * x)
            result[~inside] = (1 - weight) * d4 + weight * cubic
            return float(result) if result.ndim == 0 else result

        return evaluate, notes

    def six_month_smile(snapshot, request, forward, expiry, start, end, fixing, payment):
        from services import latest_normal_model as native

        six_request = SimpleNamespace(
            index_tenor="6M", valuation_date=request.valuation_date, day_count=request.day_count
        )
        term = ql.Actual365Fixed().yearFraction(date_to_ql(request.valuation_date), payment)
        lower, upper = _time_bounds(snapshot, "6M")
        try:
            evaluate, notes = native_smile(
                snapshot, six_request, forward, expiry, start, end, fixing, payment
            )
            return (
                evaluate,
                {
                    "source_fixing_date": str(ql_to_date(fixing)),
                    "source_payment_date": str(ql_to_date(payment)),
                    "source_fixing_extrapolated": False,
                    "source_forward": forward,
                    "source_time_policy": "native",
                },
                notes,
            )
        except ValueError:
            if lower - _TOL <= term <= upper + _TOL:
                raise
        # A successful native _pricing_targets query whose fit is invalid must
        # never be silently converted to an extrapolation.
        try:
            native._pricing_targets(
                snapshot, six_request, forward, expiry, start, end, fixing, payment
            )
        except ValueError:
            pass
        else:
            raise ValueError("Smile 6M natif invalide ; aucune substitution par l'extension.")

        rows = _native_rows(snapshot, "6M")
        row = rows[0] if term < lower else rows[-1]
        sf, sp = (
            date_to_ql(date.fromisoformat(str(row["fixing_date"]))),
            date_to_ql(date.fromisoformat(str(row["payment_date"]))),
        )
        ss, se = (
            date_to_ql(date.fromisoformat(str(row["start_date"]))),
            date_to_ql(date.fromisoformat(str(row["end_date"]))),
        )
        source_forward, source_expiry = float(row["forward_rate"]), float(row["expiry_years"])
        evaluate, notes = native_smile(
            snapshot, six_request, source_forward, source_expiry, ss, se, sf, sp
        )

        def transported(query):
            return evaluate(np.asarray(query) - forward + source_forward)

        return (
            transported,
            {
                "source_fixing_date": str(ql_to_date(sf)),
                "source_payment_date": str(ql_to_date(sp)),
                "source_fixing_extrapolated": True,
                "source_forward": source_forward,
                "source_time_policy": "early" if term < lower else "terminal",
            },
            notes,
        )

    def provider(snapshot, request, forward, expiry, start, end, fixing, payment):
        from services import latest_normal_model as native

        if request.index_tenor not in ("3M", "6M"):
            raise ValueError("Extension disponible uniquement pour Euribor 3M/6M.")
        if fixing <= date_to_ql(request.valuation_date):
            return None
        horizon = date_to_ql(request.valuation_date) + ql.Period(config.max_horizon_years, ql.Years)
        if max(start, end, fixing, payment) > horizon:
            raise ValueError("Coupon au-delà de l'horizon contractuel de 31 ans.")
        if not isfinite(expiry) or expiry <= 0 or not isfinite(forward):
            raise ValueError("Forward ou expiration du coupon expérimental invalide.")
        horizon_string = str(ql_to_date(horizon))
        assumptions = ["Experimental extension; no independent ICE validation or new market quote."]
        notes = []
        weights, scale, basis = None, None, None
        rho, blend = None, 1.0
        candidate = snapshot.candidate

        if request.index_tenor == "3M":
            if candidate is None:
                raise ValueError("L'extension 3M exige le candidat figé calibré.")
            if candidate.T[0] - 1e-12 <= expiry <= candidate.T[-1] + 1e-12:
                return None
            if expiry < candidate.T[0]:
                source_forward = float(candidate.F[0])

                def evaluate(query):
                    return candidate.vol(0, np.asarray(query) - forward + source_forward)

                node = candidate.nodes[0]
                metadata = {
                    "source_fixing_date": str(node.fixing_date),
                    "source_payment_date": str(node.payment_date),
                    "source_fixing_extrapolated": True,
                    "source_forward": source_forward,
                    "source_time_policy": "early",
                }
                source_tenor, classification = "3M", "early_flat_normal"
                requested_fixing, requested_payment = fixing, payment
                source_domain = candidate.domain

                def source_query(query):
                    return np.asarray(query) - forward + source_forward

                assumptions.append(
                    "First native normal smile held flat in expiry at equal absolute K-F."
                )
            else:
                three = coupon_forward_contract("3M", fixing)
                six = coupon_forward_contract("6M", fixing)
                projection = snapshot.curves.get_forward_curve("3M")
                six_projection = snapshot.curves.get_forward_curve("6M")
                source_forward = six.forward_rate(six_projection)
                # Split the standard 6M index at the first standard 3M maturity.
                # The remaining subperiod can be a business-day stub: project
                # its exact dates, without inventing a second fixing quote.
                middle, finish = three.index_maturity_date, six.index_maturity_date
                if (
                    middle >= finish
                    or middle < projection.referenceDate()
                    or finish > projection.maxDate()
                ):
                    raise ValueError("Sous-période de composition hors du domaine de la courbe.")
                d1 = three.projection_accrual_fraction
                d2 = ql.Actual360().yearFraction(middle, finish)
                d6 = six.projection_accrual_fraction
                f1 = three.forward_rate(projection)
                f2 = (projection.discount(middle) / projection.discount(finish) - 1.0) / d2
                w1 = d1 * (1.0 + d2 * f2) / d6
                w2 = d2 * (1.0 + d1 * f1) / d6
                rho = float(config.correlation_rho)
                scale = sqrt(w1 * w1 + w2 * w2 + 2.0 * rho * w1 * w2)
                if not all(isfinite(v) for v in (w1, w2, scale, f2)) or min(w1, w2, scale) <= 0:
                    raise ValueError("Composition 3M/6M invalide.")
                weights = (w1, w2)
                composed = ((1 + d1 * f1) * (1 + d2 * f2) - 1) / d6
                basis = source_forward - composed
                source_eval, metadata, notes = six_month_smile(
                    snapshot,
                    request,
                    source_forward,
                    expiry,
                    six.index_value_date,
                    six.index_maturity_date,
                    fixing,
                    six.index_maturity_date,
                )
                blend = min(1.0, (expiry - float(candidate.T[-1])) / config.blend_years)
                last = len(candidate.T) - 1
                terminal_forward = float(candidate.F[-1])

                def evaluate(query):
                    query = np.asarray(query)
                    mapped = source_forward + scale * (query - forward)
                    inferred = source_eval(mapped) / scale
                    if blend >= 1.0:
                        return inferred
                    terminal = candidate.vol(last, query - forward + terminal_forward)
                    return (1.0 - blend) * terminal + blend * inferred

                source_tenor, classification = "6M", "long_3m_from_6m"
                requested_fixing, requested_payment = fixing, six.index_maturity_date
                source_domain = STRIKE_DOMAIN

                def source_query(query):
                    return metadata["source_forward"] + scale * (np.asarray(query) - forward)

                assumptions.extend(
                    [
                        "Standard EUR6 source at the target fixing; source payment is its standard index maturity.",
                        "First-order compounding of two quarterly subperiods with equal normal-vol amplitudes.",
                        "Deterministic tenor basis; correlation is a scenario hypothesis, not calibrated to ICE.",
                        "K6=F6+A*(K3-F3); sigma3=sigma6/A, with A=sqrt(w1^2+w2^2+2*rho*w1*w2).",
                        "Linear normal-vol blend over one year after the last native 3M fixing.",
                    ]
                )
        else:
            # Preserve every successful existing target branch, including its
            # exact-fixing shortcut and existing short 3M-to-6M bridge.
            try:
                native._pricing_targets(
                    snapshot, request, forward, expiry, start, end, fixing, payment
                )
            except ValueError:
                lower, upper = _time_bounds(snapshot, "6M")
                term = ql.Actual365Fixed().yearFraction(date_to_ql(request.valuation_date), payment)
                if lower - _TOL <= term <= upper + _TOL:
                    raise
            else:
                return None
            evaluate, metadata, notes = six_month_smile(
                snapshot, request, forward, expiry, start, end, fixing, payment
            )
            classification = (
                "early_flat_normal"
                if metadata["source_time_policy"] == "early"
                else "terminal_flat_normal"
            )
            source_tenor = "6M"
            requested_fixing, requested_payment = fixing, payment
            source_domain = STRIKE_DOMAIN

            def source_query(query):
                return metadata["source_forward"] + np.asarray(query) - forward

            assumptions.append(
                "Boundary native normal smile held flat in expiry at equal absolute K-F."
            )

        sigma = _checked_vols(evaluate, forward, expiry, request.strike)
        source_strike = float(source_query(request.strike))
        source_wing = not source_domain[0] <= source_strike <= source_domain[1]
        source_grid = source_query(np.asarray(STRIKE_DOMAIN))
        grid_wing = bool(
            np.min(source_grid) < source_domain[0] or np.max(source_grid) > source_domain[1]
        )
        if blend < 1.0 and request.index_tenor == "3M" and classification == "long_3m_from_6m":
            terminal_query = request.strike - forward + float(candidate.F[-1])
            source_wing = (
                source_wing or not candidate.domain[0] <= terminal_query <= candidate.domain[1]
            )
            terminal_grid = np.asarray(STRIKE_DOMAIN) - forward + float(candidate.F[-1])
            grid_wing = grid_wing or bool(
                terminal_grid.min() < candidate.domain[0]
                or terminal_grid.max() > candidate.domain[1]
            )
        coverage = ExtensionCoverage(
            classification=classification,
            source_tenor=source_tenor,
            source_fixing_date=metadata["source_fixing_date"],
            source_payment_date=metadata["source_payment_date"],
            source_strike=source_strike,
            source_strike_domain=source_domain,
            source_strike_extrapolated=source_wing,
            shape_grid_source_strike_extrapolated=grid_wing,
            source_fixing_extrapolated=metadata["source_fixing_extrapolated"],
            requested_source_fixing_date=ql_to_date(requested_fixing),
            requested_source_payment_date=ql_to_date(requested_payment),
            horizon_date=horizon_string,
            blend_weight=blend,
            correlation_rho=rho,
            composition_weights=weights,
            six_month_to_three_month_scale=scale,
            deterministic_basis=basis,
            assumptions=assumptions,
        ).model_dump(mode="json")
        warnings = [
            "Experimental volatility extension: inferred/extrapolated coupon; independent ICE validation unavailable.",
            *notes,
        ]
        if metadata["source_fixing_extrapolated"]:
            warnings.append(
                "Source fixing extrapolated: boundary normal smile held flat; rate curves remain bounded."
            )
        if source_wing or grid_wing:
            warnings.append(
                "Source smile wing outside its validated strike domain; formula evaluated without strike clamping."
            )
        return {
            "normal_vol": sigma,
            "vol_source": "experimental_" + classification,
            "warnings": list(dict.fromkeys(warnings)),
            "coverage": coverage,
        }

    return provider


def _isolated_snapshot(snapshot):
    local = copy(snapshot)
    local.fits = OrderedDict(snapshot.fits)
    local.checked_candidate_coupons = set(snapshot.checked_candidate_coupons)
    return local


def _price_extended_model(
    request, market_store=None, curve_store=None, *, config=None, snapshot=None
):
    """Price explicit experimental coverage; public calls validate source freshness."""
    from services import latest_normal_model as native
    from services.latest_normal_candidate_schemas import CANDIDATE_MODEL
    from services.latest_normal_extension_schemas import ExtendedCandidatePriceResponse
    from services.latest_normal_schemas import LatestNormalPriceRequest

    config = config if config is not None else getattr(request, "extension", None)
    config = (
        config
        if isinstance(config, ExtensionConfig)
        else ExtensionConfig.model_validate(config or {})
    )
    base_request = LatestNormalPriceRequest.model_validate(
        request.model_dump(exclude={"extension", "coupon_schedule"})
    )
    public = snapshot is None
    if public:
        snapshot = native._snapshot_for_pricing(
            base_request, market_store, curve_store, CANDIDATE_MODEL
        )
    result = native._price_latest_normal(
        base_request,
        market_store,
        curve_store,
        expected_model=CANDIDATE_MODEL,
        snapshot_override=_isolated_snapshot(snapshot),
        coupon_vol_provider=make_extension_provider(config),
        coupon_schedule=(
            [term.model_dump() for term in request.coupon_schedule]
            if getattr(request, "coupon_schedule", None) is not None
            else None
        ),
    )
    if public:
        native._snapshot_for_pricing(base_request, market_store, curve_store, CANDIDATE_MODEL)
    for row in result["caplets"]:
        row.setdefault("notional", base_request.notional)
        row.setdefault("strike", base_request.strike)
    future = [row for row in result["caplets"] if row["status"] == "future"]
    extended = sum("coverage" in row for row in future)
    result.update(
        extension_version=EXTENSION_VERSION,
        extension_config=config.model_dump(),
        validation_status=VALIDATION_STATUS,
        native_coupon_count=len(future) - extended,
        extended_coupon_count=extended,
    )
    ExtendedCandidatePriceResponse.model_validate(result)
    return result


def _extended_surface(request, market_store=None, curve_store=None, *, snapshot=None):
    """Evaluate identical contractual standard coupons in native and extended channels.

    Every failed slice remains explicitly unavailable. This comparison never
    substitutes zero for a missing native or extended model value.
    """
    from curves.helpers import ql_evaluation_date
    from services import latest_normal_model as native
    from services.latest_normal_candidate_schemas import CANDIDATE_MODEL
    from services.latest_normal_extension_schemas import ExtendedCandidateSurfaceResponse
    from services.latest_normal_schemas import LatestNormalPriceRequest

    public = snapshot is None
    if public:
        snapshot = native._snapshot_for_pricing(request, market_store, curve_store, CANDIDATE_MODEL)
    local = _isolated_snapshot(snapshot)
    provider = make_extension_provider(request.extension)
    slices = []
    with ql_evaluation_date(request.valuation_date):
        valuation = date_to_ql(request.valuation_date)
        nominal_starts = {}
        fixing_dates = request.fixing_dates
        if request.grid_to_30y:
            calendar = ql.TARGET()
            spot = calendar.advance(valuation, 2, ql.Days)
            months = 3 if request.index_tenor == "3M" else 6
            fixing_dates = []
            for month in range(months, 360, months):
                start = calendar.advance(spot, ql.Period(month, ql.Months), ql.ModifiedFollowing)
                fixing = ql_to_date(calendar.advance(start, -2, ql.Days))
                fixing_dates.append(fixing)
                nominal_starts[fixing] = month / 12
        for fixing_date in fixing_dates:
            row = {
                "fixing_date": str(fixing_date),
                "nominal_start_years": nominal_starts.get(fixing_date),
                "status": "unavailable",
                "curve": [],
                "warnings": [],
                "issues": [],
                "native_issues": [],
            }
            try:
                fixing = date_to_ql(fixing_date)
                if fixing <= valuation:
                    raise ValueError(
                        "Une surface future exige un fixing postérieur à la valorisation."
                    )
                horizon = valuation + ql.Period(request.extension.max_horizon_years, ql.Years)
                contract = coupon_forward_contract(request.index_tenor, fixing)
                start, end = contract.index_value_date, contract.index_maturity_date
                if max(start, end, fixing) > horizon:
                    raise ValueError("Coupon au-delà de l'horizon contractuel de 31 ans.")
                forward = contract.forward_rate(local.curves.get_forward_curve(request.index_tenor))
                if end > local.curves.get_discount_curve("OIS").maxDate():
                    raise ValueError("Paiement hors du domaine de la courbe €STR.")
                expiry = ql.Actual365Fixed().yearFraction(valuation, fixing)
                row.update(
                    start_date=str(ql_to_date(start)),
                    end_date=str(ql_to_date(end)),
                    payment_date=str(ql_to_date(end)),
                    expiry_years=expiry,
                    forward_rate=forward,
                )
                price_request = LatestNormalPriceRequest(
                    model_id=request.model_id,
                    valuation_date=request.valuation_date,
                    product_type="cap",
                    index_tenor=request.index_tenor,
                    notional=1.0,
                    effective_date=ql_to_date(start),
                    maturity_date=ql_to_date(end),
                    strike=request.strikes[0],
                )
                native_evaluate = None
                try:
                    if request.index_tenor == "3M":

                        def native_evaluate(query, forward=forward, expiry=expiry):
                            return local.candidate.vol_for_coupon(forward, expiry, query)

                    else:
                        targets, _, _ = native._pricing_targets(
                            local, price_request, forward, expiry, start, end, fixing, end
                        )
                        params = native._fit(local, "6M", forward, expiry, targets)
                        native_strikes = [curve["strike"] for curve in local.families["6M"]]

                        def native_evaluate(
                            query,
                            forward=forward,
                            expiry=expiry,
                            native_strikes=native_strikes,
                            targets=targets,
                            params=params,
                        ):
                            return native.latest_normal_vol(
                                forward, expiry, native_strikes, targets, params, "6M", query
                            )

                    _checked_vols(native_evaluate, forward, expiry, request.strikes[0])
                except (ValueError, RuntimeError, ArithmeticError) as exc:
                    native_evaluate = None
                    row["native_issues"].append(str(exc))
                for strike in request.strikes:
                    point = {
                        "strike": strike,
                        "native_normal_vol": None,
                        "extended_normal_vol": None,
                        "coverage": None,
                    }
                    if native_evaluate is not None:
                        point["native_normal_vol"] = float(native_evaluate(strike))
                    try:
                        coupon_request = price_request.model_copy(update={"strike": strike})
                        override = provider(
                            local, coupon_request, forward, expiry, start, end, fixing, end
                        )
                        if override is None:
                            if point["native_normal_vol"] is None:
                                raise ValueError(
                                    "Smile natif indisponible ; aucune substitution par l'extension."
                                )
                            point["extended_normal_vol"] = point["native_normal_vol"]
                        else:
                            point["extended_normal_vol"] = override["normal_vol"]
                            point["coverage"] = override["coverage"]
                            row["warnings"].extend(override["warnings"])
                    except (ValueError, RuntimeError, ArithmeticError) as exc:
                        row["issues"].append(str(exc))
                    row["curve"].append(point)
                row["status"] = (
                    "ready"
                    if all(point["extended_normal_vol"] is not None for point in row["curve"])
                    else "unavailable"
                )
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                row["issues"].append(str(exc))
            for key in ("issues", "warnings", "native_issues"):
                row[key] = list(dict.fromkeys(row[key]))
            slices.append(row)
    if public:
        native._snapshot_for_pricing(request, market_store, curve_store, CANDIDATE_MODEL)
    result = {
        "model_id": request.model_id,
        "valuation_date": str(request.valuation_date),
        "index_tenor": request.index_tenor,
        "extension_version": EXTENSION_VERSION,
        "extension_config": request.extension.model_dump(),
        "validation_status": VALIDATION_STATUS,
        "slices": slices,
    }
    ExtendedCandidateSurfaceResponse.model_validate(result)
    return result


def price_extended_model(
    request, market_store=None, curve_store=None, *, config=None, snapshot=None
):
    """Keep explicit extension failures in the native API error contract."""
    from pydantic import ValidationError as PydanticValidationError

    from services.exceptions import ServiceError
    from services.latest_normal_candidate import CandidatePricingError
    from services.latest_normal_model import LatestNormalValidationError

    try:
        return _price_extended_model(
            request, market_store, curve_store, config=config, snapshot=snapshot
        )
    except ServiceError:
        raise
    except PydanticValidationError as exc:
        raise CandidatePricingError(str(exc)) from exc
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, ArithmeticError, KeyError, TypeError) as exc:
        raise CandidatePricingError(str(exc)) from exc


def extended_surface(request, market_store=None, curve_store=None, *, snapshot=None):
    """Return native API validation errors for an expired or invalid surface source."""
    from pydantic import ValidationError as PydanticValidationError

    from services.exceptions import ServiceError
    from services.latest_normal_candidate import CandidatePricingError
    from services.latest_normal_model import LatestNormalValidationError

    try:
        return _extended_surface(request, market_store, curve_store, snapshot=snapshot)
    except ServiceError:
        raise
    except PydanticValidationError as exc:
        raise CandidatePricingError(str(exc)) from exc
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, ArithmeticError, KeyError, TypeError) as exc:
        raise CandidatePricingError(str(exc)) from exc


def preview_extended_schedule(request):
    """Use the pricing calendar itself to preview every contractual period."""
    import itertools

    from curves.helpers import ql_evaluation_date
    from services.latest_normal_model import LatestNormalValidationError, cap_schedule

    try:
        calendar = ql.TARGET()
        convention = {
            "ModifiedFollowing": ql.ModifiedFollowing,
            "Following": ql.Following,
            "Preceding": ql.Preceding,
            "Unadjusted": ql.Unadjusted,
        }[request.business_day_convention]
        with ql_evaluation_date(request.valuation_date):
            schedule = cap_schedule(request, calendar, convention)
            return [
                {
                    "start_date": str(ql_to_date(start)),
                    "end_date": str(ql_to_date(end)),
                    "notional": request.notional,
                    "strike": request.strike,
                }
                for start, end in itertools.pairwise(schedule)
            ]
    except (ValueError, RuntimeError) as exc:
        raise LatestNormalValidationError(str(exc)) from exc
