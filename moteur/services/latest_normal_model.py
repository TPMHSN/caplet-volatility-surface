"""Production entry point for the selected normal caplet model, version v3.

An immutable server snapshot holds isolated rate curves, fixing histories and
quadratic targets. Client supplied smile parameters never enter pricing.
"""

import itertools
import json
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from io import BytesIO
from math import exp, fsum, isfinite, pi, sqrt
from threading import RLock
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import numpy as np
import QuantLib as ql
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from pydantic import ValidationError as PydanticValidationError
from scipy.special import ndtr

from config import get_index
from curves.helpers import date_to_ql, ql_evaluation_date, ql_to_date
from curves.yield_curve_builder import CurveSet, YieldCurveBuilder
from market_data.fixing_store import fixing_store
from market_data.store import MarketDataStore
from market_data.template_loader import ExcelTemplateLoader
from pricing.curve_risk_ladder import canonical_discount_curve_key, canonical_forward_curve_key
from pricing.curve_sensitivity_engine import (
    RepriceOutput,
    _quote_calibration_base,
    compute_curve_sensitivities,
    curve_diagnostics_fields,
)
from services.caplet_normal_curves import _strike_curve
from services.exceptions import PricingError, ServiceError, ValidationError
from services.latest_normal_candidate import FrozenThreeMonthCandidate, build_candidate
from services.latest_normal_candidate_adapter import (
    bridge_provider,
    candidate_slice,
    candidate_source,
    check_coupon_shape,
)
from services.latest_normal_candidate_schemas import (
    CANDIDATE_MODEL,
    CANDIDATE_POLICY,
    CANDIDATE_VERSION,
    CandidatePreparationResponse,
)
from services.latest_normal_coupon import coupon_forward_contract
from services.latest_normal_diagnostics import (
    build_atm_usage,
    build_source_provenance,
    flat_equivalent_normal_vol,
    quote_fit_point,
    quote_fit_product_pair,
    summarize_quote_fit,
)
from services.latest_normal_input_validation import validate_native_cap_quotes
from services.latest_normal_local_price_repair import build_local_price_repair, price_repair_warning
from services.latest_normal_model_math import (
    MODEL_SHIFT,
    SIX_MONTH_SMILE_POLICY,
    STRIKE_DOMAIN,
    THREE_MONTH_SMILE_POLICY,
    TIME_SMOOTHING,
    fit_latest_smile,
    latest_normal_vol,
    latest_smile_components,
    smile_policy,
    smile_transition,
)
from services.latest_normal_schemas import (
    LatestNormalModelRequest,
    LatestNormalPreparationResponse,
    LatestNormalPriceRequest,
    LatestNormalPriceResponse,
    NativeQuoteFitDiagnostics,
    NativeQuoteFitPoint,
    RiskReferenceCheck,
)
from services.latest_normal_short_end import (
    couple_first_six_month_pillar,
    prepare_short_end_source,
    supports_short_end_policy,
)
from services.normal_cap_pricer import cap_schedule, target_at
from services.thomas_caps_market_data import MAX_EXPANDED_BYTES, MAX_WORKBOOK_BYTES
from services.thomas_optionlet import loaded_context
from services.thomas_stripping import _schedule as native_cap_schedule
from services.thomas_stripping import _validated_quotes
from services.thomas_stripping_blocks import _convert_short_node, build_block_family
from services.thomas_stripping_schemas import OptionletNode, StrippingFamily

MODEL = "latest_normal_v3"
VERSION = "v3"
MAX_SNAPSHOTS = 8
MAX_CACHED_FITS = 1024
RISK_REFERENCE_TOLERANCE_EUR = 1e-4
_snapshots = OrderedDict()
_cache_lock = RLock()


class LatestNormalValidationError(ValidationError, ValueError):
    """Latest-normal input or state does not satisfy the pricing contract."""


class LatestNormalPricingError(PricingError, ValueError):
    """Latest-normal numerical work failed after valid inputs were accepted."""


@dataclass
class _Snapshot:
    valuation_date: date
    source_token: str
    source: str
    curves: CurveSet
    histories: dict
    fixing_fingerprint: str
    families: dict
    response: dict
    fits: OrderedDict = field(default_factory=OrderedDict)
    source_3m_family: StrippingFamily | None = None
    candidate: FrozenThreeMonthCandidate | None = None
    model: str = MODEL
    checked_candidate_coupons: set[tuple[float, float]] = field(default_factory=set)
    cap_vol_data: object | None = None


def _fixing_histories(valuation_date):
    return {
        tenor: {
            d: float(r)
            for d, r in fixing_store.get_all_fixings("EUR", tenor).items()
            if d <= valuation_date
        }
        for tenor in ("3M", "6M")
    }


def _fixing_fingerprint(histories):
    payload = {
        tenor: {d.isoformat(): r for d, r in sorted(history.items())}
        for tenor, history in histories.items()
    }
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def _observed_fixings(valuation_date, histories):
    """Own the QL lock through bootstrap and restoration of every touched history."""
    with ql_evaluation_date(valuation_date):
        if not ql.TARGET().isBusinessDay(date_to_ql(valuation_date)):
            raise ValueError("La date de valorisation doit être un jour ouvré TARGET.")
        saved = []
        settings = ql.Settings.instance()
        original_enforce = settings.enforcesTodaysHistoricFixings
        try:
            for tenor in ("3M", "6M"):
                index = get_index("EUR", tenor)
                history = index.timeSeries()
                saved.append((index, list(history.dates()), list(history.values())))
                index.clearFixings()
                valid = [
                    (date_to_ql(d), r)
                    for d, r in histories[tenor].items()
                    if index.isValidFixingDate(date_to_ql(d))
                ]
                if valid:
                    index.addFixings([d for d, _ in valid], [r for _, r in valid], True)
            settings.enforcesTodaysHistoricFixings = True
            yield
        finally:
            settings.enforcesTodaysHistoricFixings = original_enforce
            for index, dates, rates in reversed(saved):
                index.clearFixings()
                if dates:
                    index.addFixings(dates, rates, True)


def _validate_histories(valuation_date, histories):
    for tenor, history in histories.items():
        if valuation_date not in history:
            raise ValueError(
                f"Fixing Euribor {tenor} observé du {valuation_date} absent. Importer les fixings avant de calibrer le dernier modèle."
            )
        if any(not isfinite(rate) for rate in history.values()):
            raise ValueError("Un fixing observé est non fini.")


def _copy_loaded_market(request, market_store, curve_store):
    context = loaded_context(market_store, curve_store)
    if context.source_token != request.source_token:
        raise ValueError("Sources modifiées : actualiser puis recalculer le dernier modèle.")
    market = MarketDataStore()
    ois = market_store.get_ois_curve("EUR")
    caps = market_store.get_capfloor_surface("EUR")
    quotes = {t: market_store.get_forward_curve("EUR", t) for t in ("3M", "6M")}
    if ois is None or caps is None or any(q is None for q in quotes.values()):
        raise ValueError("Importer €STR, Euribor 3M, Euribor 6M et VOL CAP.")
    if any(
        source.valuation_date != request.valuation_date for source in (ois, caps, *quotes.values())
    ):
        raise ValueError("Les quatre sources doivent avoir la date de valorisation.")
    market.store_ois_curve("EUR", ois.model_copy(deep=True))
    market.store_capfloor_surface("EUR", caps.model_copy(deep=True))
    for tenor, quote in quotes.items():
        market.store_forward_curve("EUR", tenor, quote.model_copy(deep=True))
    if loaded_context(market_store, curve_store).source_token != request.source_token:
        raise ValueError("Sources modifiées pendant leur lecture : recalculer.")
    return market


def _workbook_market(content, valuation_date):
    if len(content) > MAX_WORKBOOK_BYTES:
        raise ValueError("Le classeur dépasse 10 Mo.")
    try:
        with ZipFile(BytesIO(content)) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise ValueError("Le classeur décompressé dépasse 40 Mo.")
        book = load_workbook(BytesIO(content), data_only=True)
    except (BadZipFile, InvalidFileException) as exc:
        raise ValueError("Fichier .xlsx invalide.") from exc
    try:
        sheets = {name.strip().upper() for name in book.sheetnames}
        required = {"ESTR", "EUR3", "EUR6", "VOL CAP"}
        if required - sheets:
            raise ValueError("Feuilles manquantes : " + ", ".join(sorted(required - sheets)))
        for name in list(book.sheetnames):
            if name.strip().upper() not in required:
                del book[name]
        selected = BytesIO()
        book.save(selected)
    finally:
        book.close()
    parsed = ExcelTemplateLoader().load_template(selected.getvalue(), valuation_date)
    market = MarketDataStore()
    market.store_ois_curve("EUR", parsed["ois"]["EUR"])
    market.store_capfloor_surface("EUR", parsed["capfloor"]["EUR"])
    for tenor in ("3M", "6M"):
        market.store_forward_curve("EUR", tenor, parsed["forward"][f"EUR_{tenor}"])
    return market


def _clean_grid(values):
    # Explicit ATM/blend knots can differ from linspace knots by one floating
    # point bit. Do not differentiate prices across an effectively zero width.
    values = np.unique(np.round(values, 12))
    return values[np.r_[True, np.diff(values) > 1e-10]]


def _shape_check(forward, expiry, strikes, targets, params, tenor):
    transition_start, transition_end = smile_transition(strikes, tenor)
    grid = _clean_grid(
        np.r_[np.linspace(*STRIKE_DOMAIN, 651), forward, transition_start, transition_end]
    )
    grid = grid[(grid >= STRIKE_DOMAIN[0]) & (grid <= STRIKE_DOMAIN[1])]
    vol = latest_normal_vol(forward, expiry, strikes, targets, params, tenor, grid)
    std = vol * sqrt(expiry)
    z = (forward - grid) / std
    calls = (forward - grid) * ndtr(z) + std * np.exp(-z * z / 2) / sqrt(2 * pi)
    slopes = np.diff(calls) / np.diff(grid)
    density = 2 * np.diff(slopes) / (grid[2:] - grid[:-2])
    if np.min(slopes) < -1 - 1e-6 or np.max(slopes) > 1e-6 or np.min(density) < -1e-4:
        raise ValueError(
            "La grille de prix du smile présente une violation de monotonie ou de convexité dans le domaine du modèle."
        )


def _fit(snapshot, tenor, forward, expiry, targets):
    curves = snapshot.families[tenor]
    strikes = [c["strike"] for c in curves]
    key = (tenor, float(forward), float(expiry), tuple(float(v) for v in targets))
    if key in snapshot.fits:
        snapshot.fits.move_to_end(key)
        return snapshot.fits[key]
    params = fit_latest_smile(forward, expiry, strikes, targets, tenor)
    try:
        _shape_check(forward, expiry, strikes, targets, params, tenor)
    except ValueError:
        repair = (
            build_local_price_repair(forward, expiry, strikes, targets, STRIKE_DOMAIN)
            if tenor == "6M"
            else None
        )
        if repair is None:
            raise
        params["price_repair"] = repair
        _shape_check(forward, expiry, strikes, targets, params, tenor)
    snapshot.fits[key] = params
    while len(snapshot.fits) > MAX_CACHED_FITS:
        snapshot.fits.popitem(last=False)
    return params


def _native_quote_fit_diagnostics(
    snapshot: _Snapshot,
    tenor: str,
    diagnostic_curves: list[dict],
    expected_native_quotes: Mapping[float, Mapping[int, float]],
) -> NativeQuoteFitDiagnostics:
    """Reprice every native quote with quadratic and actual final-smile cap/floor prices."""
    model_curves = snapshot.families[tenor]
    model_strikes = [curve["strike"] for curve in model_curves]
    points: list[NativeQuoteFitPoint] = []
    curves_by_strike = {float(curve["strike"]): curve for curve in diagnostic_curves}
    for strike, quotes_by_expiry in expected_native_quotes.items():
        curve = curves_by_strike.get(float(strike))
        intervals = curve.get("intervals", []) if curve is not None else []
        intervals_by_expiry = {int(interval["expiry_months"]): interval for interval in intervals}
        issues = [str(issue) for issue in curve.get("issues", [])] if curve is not None else []
        for expiry_months, normal_quote in quotes_by_expiry.items():
            interval = intervals_by_expiry.get(expiry_months)
            if interval is None:
                reason = "; ".join(issues) or (
                    f"Native quote interval {tenor} {expiry_months}M at strike {strike:.12g} "
                    "is unavailable."
                )
                points.append(
                    NativeQuoteFitPoint(
                        expiry_months=expiry_months,
                        strike=strike,
                        market_quote_normal_vol_bp=normal_quote * 10_000.0,
                        status="unavailable",
                        block=None,
                        quadratic=None,
                        final=None,
                        reason=reason,
                    )
                )
                continue
            market_price = float(interval["market_price"])
            block_price = interval["model_prices"].get("constant")
            quadratic_price = interval["model_prices"].get("quadratic")
            cashflows = interval["cashflows"]
            parity = fsum(
                float(flow["accrual_fraction"])
                * float(flow["discount_factor"])
                * (float(flow["forward_rate"]) - strike)
                for flow in cashflows
            )
            quadratic_pair = (
                quote_fit_product_pair(market_price, float(quadratic_price), parity)
                if quadratic_price is not None
                else None
            )
            if strike < STRIKE_DOMAIN[0] or strike > STRIKE_DOMAIN[1]:
                points.append(
                    NativeQuoteFitPoint(
                        expiry_months=interval["expiry_months"],
                        strike=strike,
                        market_quote_normal_vol_bp=normal_quote * 10_000.0,
                        status="non_covered",
                        block=(
                            quote_fit_product_pair(market_price, float(block_price), parity)
                            if block_price is not None
                            else None
                        ),
                        quadratic=quadratic_pair,
                        final=None,
                        reason="Strike is outside the validated final-smile domain.",
                    )
                )
                continue
            try:
                final_prices = []
                for flow in cashflows:
                    term = (
                        date.fromisoformat(flow["payment_date"]) - snapshot.valuation_date
                    ).days / 365.0
                    if snapshot.candidate is not None and tenor == "3M":
                        sigma = float(
                            snapshot.candidate.vol_for_coupon(
                                flow["forward_rate"], flow["expiry_years"], strike
                            )
                        )
                    else:
                        targets = [
                            target_at(
                                model_curve,
                                term,
                                "quadratic",
                                flow["payment_date"],
                                flow["fixing_date"],
                            )[0]
                            for model_curve in model_curves
                        ]
                        params = _fit(
                            snapshot,
                            tenor,
                            flow["forward_rate"],
                            flow["expiry_years"],
                            targets,
                        )
                        sigma = latest_normal_vol(
                            flow["forward_rate"],
                            flow["expiry_years"],
                            model_strikes,
                            targets,
                            params,
                            tenor,
                            strike,
                        )
                    std = sigma * sqrt(flow["expiry_years"])
                    distance = flow["forward_rate"] - strike
                    z = distance / std
                    final_prices.append(
                        flow["accrual_fraction"]
                        * flow["discount_factor"]
                        * (distance * ndtr(z) + std * exp(-z * z / 2.0) / sqrt(2.0 * pi))
                    )
                point = quote_fit_point(
                    expiry_months=interval["expiry_months"],
                    strike=strike,
                    market_quote_normal_vol=normal_quote,
                    market_cap_price=market_price,
                    block_cap_price=float(block_price),
                    quadratic_cap_price=float(quadratic_price),
                    final_cap_price=fsum(final_prices),
                    cap_floor_parity=parity,
                )
            except (ValueError, RuntimeError, OverflowError, KeyError, TypeError) as exc:
                point = NativeQuoteFitPoint(
                    expiry_months=interval["expiry_months"],
                    strike=strike,
                    market_quote_normal_vol_bp=normal_quote * 10_000.0,
                    status="unavailable",
                    block=(
                        quote_fit_product_pair(market_price, float(block_price), parity)
                        if block_price is not None
                        else None
                    ),
                    quadratic=quadratic_pair,
                    final=None,
                    reason=str(exc),
                )
            points.append(point)
    block = summarize_quote_fit(points, "block")
    quadratic = summarize_quote_fit(points, "quadratic")
    final = summarize_quote_fit(points, "final")
    warnings = []
    if final.passed is not True:
        if snapshot.candidate is not None and tenor == "3M":
            warnings.append(
                "Le contrôle de repricing de toutes les quotes 3M est hors tolérance ou incomplet. "
                "Les strikes négatifs restent dans ce diagnostic, bien qu’ils ne soient pas des "
                "contraintes exactes du candidat. Ce statut est distinct de la convergence du solveur."
            )
        else:
            warnings.append(
                "Internal final-smile native quote validation failed or is incomplete; "
                "the model remains inspectable but is not production-approved."
            )
    return NativeQuoteFitDiagnostics(
        expected_point_count=sum(len(quotes) for quotes in expected_native_quotes.values()),
        emitted_point_count=len(points),
        points=points,
        block=block,
        quadratic=quadratic,
        final=final,
        warnings=warnings,
    )


def _validate_preparation_response(payload: Mapping[str, object]) -> None:
    """Classify only server-produced preparation-contract failures as HTTP 500."""
    try:
        contract = (
            CandidatePreparationResponse
            if payload.get("model") == CANDIDATE_MODEL
            else LatestNormalPreparationResponse
        )
        contract.model_validate(payload)
    except PydanticValidationError as exc:
        raise LatestNormalPricingError(str(exc)) from exc


def _prepare(
    market: MarketDataStore,
    valuation_date: date,
    source_token: str,
    source: str,
    histories: dict[str, dict[date, float]],
    *,
    use_candidate: bool = False,
    build_diagnostics: bool = True,
) -> _Snapshot:
    _validate_histories(valuation_date, histories)
    ois = market.get_ois_curve("EUR")
    quotes = {t: market.get_forward_curve("EUR", t) for t in ("3M", "6M")}
    cap_data = market.get_capfloor_surface("EUR")
    original, groups = _validated_quotes(cap_data)
    # Keep imported negative block knots in the 3M source bridge. The validated
    # pricing domain limits queries, rather than altering the source bootstrap.
    selected = [j for j, k in enumerate(original) if k <= STRIKE_DOMAIN[1] + 1e-12]
    strikes = [original[j] for j in selected]
    positive = [k for k in strikes if k >= 0]
    if len(positive) < 2 or positive[0] > 1e-12 or positive[-1] < 0.05 - 1e-12:
        raise ValueError(
            "Le dernier modèle requiert des strikes de calibration couvrant [0 % ; 5 %]."
        )
    if "3M" not in groups or "6M" not in groups:
        raise ValueError("Il faut les familles natives de caps 3M et 6M.")
    groups = {
        tenor: [(month, [flat[j] for j in selected]) for month, flat in rows]
        for tenor, rows in groups.items()
    }
    with _observed_fixings(valuation_date, histories):
        builder = YieldCurveBuilder("EUR")
        observed_calibration_fixings = {tenor: histories[tenor][valuation_date] for tenor in quotes}
        built_curves, lineage, calibration_reports = builder.build_multi_curve_with_lineage(
            ois,
            quotes,
            observed_calibration_fixings=observed_calibration_fixings,
        )
        curves = CurveSet(
            currency="EUR",
            valuation_date=valuation_date,
            ois_curve=built_curves["OIS"],
            forward_curves={
                tenor: curve for tenor, curve in built_curves.items() if tenor != "OIS"
            },
            ois_market_data=ois,
            forward_market_data=quotes,
            lineage=lineage,
            calibration_fixings=lineage.calibration_fixings,
            observed_calibration_fixings=lineage.observed_calibration_fixings,
            calibration_reports=calibration_reports,
        )
        valuation = date_to_ql(valuation_date)
        spot = ql.TARGET().advance(valuation, 2, ql.Days)
        if use_candidate:
            for native_tenor, native_rows in groups.items():
                native_nodes = native_cap_schedule(
                    valuation,
                    spot,
                    native_tenor,
                    native_rows[-1][0],
                    native_rows[0][0],
                    curves,
                    strikes,
                )
                validate_native_cap_quotes(native_tenor, native_rows, strikes, native_nodes)
        three = build_block_family(
            valuation, spot, "3M", groups["3M"], strikes, 1.0, curves, {}, False, build_black=False
        )
        use_linear_short_end = supports_short_end_policy(three, groups["6M"])
        candidate = build_candidate(three, groups["3M"], valuation) if use_candidate else None
        outside_bridge: list[float] = []
        if candidate is not None and not use_linear_short_end:
            raise ValueError("Le candidat exige le calendrier natif du raccord conjoint 3M→6M.")
        bridge_source = (
            candidate_source(candidate, three)
            if candidate is not None
            else prepare_short_end_source(three, groups["3M"]) if use_linear_short_end else three
        )
        provider_options = (
            {"source_vol_provider": bridge_provider(candidate, outside_bridge)}
            if candidate is not None
            else {}
        )
        six = build_block_family(
            valuation,
            spot,
            "6M",
            groups["6M"],
            strikes,
            1.0,
            curves,
            {},
            False,
            source_3m=bridge_source,
            basis_moneyness=True,
            build_black=False,
            source_smile_interpolation="natural_cubic",
            **provider_options,
        )
        if use_linear_short_end:
            six = couple_first_six_month_pillar(six, bridge_source, **provider_options)
        diagnostic_families = {
            t: [
                _strike_curve(valuation, family, groups[t], k, j, TIME_SMOOTHING)
                for j, k in enumerate(strikes)
            ]
            for t, family in (("3M", three), ("6M", six))
        }
        families = {
            "3M": [curve for curve in diagnostic_families["3M"] if curve["strike"] >= 0],
            "6M": list(diagnostic_families["6M"]),
        }
    snapshot = _Snapshot(
        valuation_date,
        source_token,
        source,
        curves,
        histories,
        _fixing_fingerprint(histories),
        families,
        {},
        source_3m_family=bridge_source,
        candidate=candidate,
        model=CANDIDATE_MODEL if candidate is not None else MODEL,
        cap_vol_data=cap_data.model_copy(deep=True),
    )
    if not build_diagnostics:
        return snapshot
    display_families = []
    native_families = {"3M": three, "6M": six}
    fit_warnings: list[str] = []
    for tenor, curves_by_strike in families.items():
        model_strikes = [float(curve["strike"]) for curve in curves_by_strike]
        policy = (
            CANDIDATE_POLICY if candidate is not None and tenor == "3M" else smile_policy(tenor)
        )
        ready = [c for c in curves_by_strike if c.get("intervals")]
        if not ready:
            raise ValueError(f"Aucun calendrier normal {tenor} disponible.")
        flows = ready[0]["intervals"][-1]["cashflows"]
        family_issues = list(
            dict.fromkeys(issue for c in curves_by_strike for issue in c["issues"])
        )
        slices = []
        for flow in flows:
            term = (date.fromisoformat(flow["payment_date"]) - valuation_date).days / 365
            row = {
                key: flow[key]
                for key in (
                    "fixing_date",
                    "payment_date",
                    "start_date",
                    "end_date",
                    "expiry_years",
                    "forward_rate",
                    "discount_factor",
                    "accrual_fraction",
                )
            }
            row.update(
                maturity_years=term,
                pricing_policy=policy,
                status="unavailable",
                alpha=None,
                beta=None,
                rho=None,
                nu=None,
                atm_normal_vol=None,
                fit_rmse_bp=None,
                targets=[],
                curve=[],
                issues=[],
                d4_component=None,
            )
            try:
                if candidate is not None and tenor == "3M":
                    row.update(candidate_slice(candidate, flow["fixing_date"]))
                    slices.append(row)
                    continue
                targets = [
                    target_at(c, term, "quadratic", flow["payment_date"], flow["fixing_date"])[0]
                    for c in curves_by_strike
                ]
                row["targets"] = [
                    {"strike": c["strike"], "normal_vol": v}
                    for c, v in zip(curves_by_strike, targets)
                ]
                params = _fit(snapshot, tenor, flow["forward_rate"], flow["expiry_years"], targets)
                row.update(
                    {
                        key: params[key]
                        for key in ("alpha", "beta", "rho", "nu", "atm_normal_vol", "fit_rmse_bp")
                    }
                )
                row["d4_component"] = {
                    key: params[key]
                    for key in (
                        "pricing_policy",
                        "alpha",
                        "beta",
                        "rho",
                        "nu",
                        "atm_normal_vol",
                        "fit_rmse_bp",
                        "fit_cost",
                        "bound_hit",
                        "native_boundary_strike",
                        "boundary_tie_error_bp",
                        "optimizer_success",
                        "optimizer_status",
                    )
                }
                transition_start, transition_end = smile_transition(model_strikes, tenor)
                grid = _clean_grid(
                    np.r_[
                        np.linspace(*STRIKE_DOMAIN, 131),
                        flow["forward_rate"],
                        transition_start,
                        transition_end,
                    ]
                )
                grid = grid[(grid >= STRIKE_DOMAIN[0]) & (grid <= STRIKE_DOMAIN[1])]
                components = latest_smile_components(
                    flow["forward_rate"],
                    flow["expiry_years"],
                    model_strikes,
                    targets,
                    params,
                    tenor,
                    grid,
                )
                row["curve"] = [
                    {
                        "strike": float(k),
                        **{name: float(values[i]) for name, values in components.items()},
                    }
                    for i, k in enumerate(grid)
                ]
                row["status"] = "ready"
                if "price_repair" in params:
                    row["issues"].append(price_repair_warning(params["price_repair"]))
                if params["bound_hit"]:
                    row["issues"].append("Un paramètre D.4 atteint une borne de calibration.")
            except (ValueError, RuntimeError, OverflowError, KeyError, TypeError) as exc:
                row.update(
                    alpha=None,
                    beta=None,
                    rho=None,
                    nu=None,
                    atm_normal_vol=None,
                    fit_rmse_bp=None,
                    curve=[],
                    d4_component=None,
                )
                row["issues"].append(str(exc))
            slices.append(row)
        expected_native_quotes = {
            strike: {month: flat[column] for month, flat in groups[tenor]}
            for column, strike in enumerate(strikes)
        }
        quote_fit = _native_quote_fit_diagnostics(
            snapshot,
            tenor,
            diagnostic_families[tenor],
            expected_native_quotes,
        )
        fit_warnings.extend(quote_fit.warnings)
        native_family = native_families[tenor]
        final_negative = [point for point in quote_fit.points if point.strike < 0.0]
        if not final_negative:
            negative_coverage = "none"
        elif all(point.status == "ready" and point.final is not None for point in final_negative):
            negative_coverage = "complete"
        elif any(point.status == "ready" and point.final is not None for point in final_negative):
            negative_coverage = "partial"
        else:
            negative_coverage = "none"
        bridge = None
        if tenor == "6M":
            bridge = {
                "source_tenor": "3M",
                "target_tenor": "6M",
                "method": "same_absolute_moneyness",
                "strike_interpolation": (
                    "continuous_frozen_candidate" if candidate is not None else "natural_cubic"
                ),
                "expiry_extrapolation": "forbidden",
                "assumptions": [
                    "Only early 6M coupons whose fixing is covered by native 3M smiles use the bridge.",
                    "The bridge transports normal volatility at the same absolute moneyness.",
                ],
            }
        display_families.append(
            {
                "pricing_policy": policy,
                "index_tenor": tenor,
                "status": "ready" if all(s["status"] == "ready" for s in slices) else "partial",
                "calibration_strikes": model_strikes,
                "slices": slices,
                "issues": family_issues,
                "coverage": {
                    "index_tenor": tenor,
                    "last_covered_fixing_date": native_family.optionlets[-1].fixing_date,
                    "last_covered_payment_date": native_family.optionlets[-1].payment_date,
                    "strike_domain": list(STRIKE_DOMAIN),
                    "native_quote_expiries": [
                        f"{month // 12}Y" if month % 12 == 0 else f"{month}M"
                        for month, _quotes in groups[tenor]
                    ],
                    "native_quote_count": len(groups[tenor]),
                    "input_quote_count": len(groups[tenor]) * len(strikes),
                    "input_strikes": strikes,
                    "input_strike_count": len(strikes),
                    "input_strike_domain": [min(strikes), max(strikes)],
                    "negative_input_quote_count": (
                        sum(strike < 0.0 for strike in strikes) * len(groups[tenor])
                    ),
                    "negative_quote_final_fit_coverage": negative_coverage,
                    "bridge": bridge,
                },
                "quote_fit": quote_fit.model_dump(),
            }
        )
    provenance = build_source_provenance(cap_data, quotes)
    atm_usage = build_atm_usage(cap_data)
    internal_passed = all(
        family["status"] == "ready"
        and family["quote_fit"]["expected_point_count"]
        == family["quote_fit"]["emitted_point_count"]
        == family["coverage"]["input_quote_count"]
        == len(family["quote_fit"]["points"])
        and all(
            point["status"] == "ready" and point["final"] is not None
            for point in family["quote_fit"]["points"]
        )
        and family["quote_fit"]["final"]["passed"] is True
        for family in display_families
    )
    warnings = list(dict.fromkeys([*provenance.import_warnings, *fit_warnings]))
    six_month_strikes = [float(curve["strike"]) for curve in families["6M"]]
    six_month_transition = smile_transition(six_month_strikes, "6M")
    result = {
        "model": MODEL,
        "version": VERSION,
        "model_id": uuid4().hex,
        "valuation_date": str(valuation_date),
        "source_token": source_token,
        "source": source,
        "model_shift": MODEL_SHIFT,
        "time_method": "quadratic",
        "smoothing": TIME_SMOOTHING,
        "strike_domain": list(STRIKE_DOMAIN),
        "families": display_families,
        "fixings": {
            "fingerprint": snapshot.fixing_fingerprint,
            "counts": {t: len(h) for t, h in histories.items()},
            "valuation_rates": {t: h.get(valuation_date) for t, h in histories.items()},
        },
        "configuration": {
            "three_month": THREE_MONTH_SMILE_POLICY,
            "six_month": SIX_MONTH_SMILE_POLICY,
            "beta_penalty_3m": 8,
            "beta_penalty_6m": 10,
            "blend_start": -0.005,
            "blend_end": 0.02,
            "six_month_native_boundary": six_month_strikes[0],
            "six_month_transition_start": six_month_transition[0],
            "six_month_transition_end": six_month_transition[1],
            "six_month_native_cubic_domain": [six_month_strikes[0], STRIKE_DOMAIN[1]],
            "bridge_source": (
                "three_month_linear_pillars_joint_first_six_month_pillar"
                if use_linear_short_end
                else "three_month_blocks_same_absolute_moneyness_natural_cubic"
            ),
        },
        "assumptions": [
            "Courbes €STR et Euribor reconstruites avec les fixings observés avant le bootstrap.",
            "Stripping par blocs puis courbes quadratiques selon le paiement, λ = 10⁻⁸.",
            (
                "Raccord 6M à même moneyness depuis une source auxiliaire 3M linéaire en temps de fixing, "
                "calibrée aux prix natifs ; frontière 2Y et premier pilier 6M calibrés conjointement. "
                "La surface finale de pricing 3M conserve sa construction propre."
                if use_linear_short_end
                else "Raccord historique 6M depuis les blocs constants 3M jusqu’au début contractuel 2Y ; "
                "calendrier de piliers non éligible au raccord linéaire conjoint."
            ),
            "Décalage de 300 bp dans D.4 ; Bachelier utilise les forwards et strikes d’origine.",
            "En 6M, la cubique naturelle conserve tous les strikes natifs retenus ; D.4 est ancré à la frontière native basse et à l’ATM, puis raccordé sous cette frontière.",
            "Une même surface normale sert aux caps et aux floors.",
            "Un coupon 6M antérieur au premier paiement natif utilise le raccord 3M→6M à même moneyness, uniquement si son fixing est couvert par les smiles 3M natives.",
            "Contrôles numériques de forme dans [−1,5 % ; 5 %] ; aucune extrapolation de pricing hors de ce domaine.",
        ],
        "provenance": provenance.model_dump(),
        "atm_usage": atm_usage.model_dump(),
        "validation": {
            "internal_validation": "passed" if internal_passed else "failed",
            "external_validation": "not_run",
            "warnings": fit_warnings,
        },
        "warnings": warnings,
    }
    if candidate is not None:
        result.update(
            model=CANDIDATE_MODEL,
            version=CANDIDATE_VERSION,
            time_method="candidate_variance_3m_quadratic_6m",
        )
        result["configuration"].update(
            three_month=CANDIDATE_POLICY,
            bridge_source="frozen_three_month_variance_wing",
            candidate_name="N_variance_wing",
            candidate_version="frozen_20261023",
            blend_start=-0.015,
            blend_end=0.0,
        )
        result["assumptions"][1:3] = [
            "3M : les trois premiers caplets partagent une courbe de variance en K−F calibrée aux prix du cap 1Y ; les quatre suivants allouent les prix résiduels 18M et 2Y.",
            "La fonction continue du candidat 3M alimente directement le raccord 6M à même K−F ; le solveur conjoint 2Y/2,5Y et les règles 6M de V3 sont conservés.",
            "Aile 3M D.4 raccordée au cœur positif sur [−1,5 % ; 0 %]. Les paramètres D.4 décrivent seulement l’aile, pas la surface entière.",
            "Coupons 3M hors grille : interpolation linéaire de volatilité normale en temps de fixing à même K−F, bornée aux sept fixings natifs, avec contrôle numérique des prix. Cette extension n’est pas couverte par la comparaison ICE des caplets natifs.",
        ]
        if outside_bridge:
            note = (
                f"Raccord 6M : {len(outside_bridge)} évaluations de la source 3M hors du domaine affiché "
                f"[{min(outside_bridge):.6%} ; {max(outside_bridge):.6%}]. La formule figée est conservée ; "
                "ces lectures ne sont pas une validation de cette aile étendue."
            )
            result["warnings"].append(note)
            display_families[1]["coverage"]["bridge"]["assumptions"].append(note)
    if source == "workbook":
        result["source_sheets"] = ["ESTR", "EUR3", "EUR6", "VOL CAP"]
    _validate_preparation_response(result)
    snapshot.response = result
    return snapshot


def _publish(snapshot: _Snapshot) -> dict[str, object]:
    _validate_preparation_response(snapshot.response)
    with _cache_lock:
        _snapshots[snapshot.response["model_id"]] = snapshot
        while len(_snapshots) > MAX_SNAPSHOTS:
            _snapshots.popitem(last=False)
    return deepcopy(snapshot.response)


def _prepare_latest_normal_model(
    request: LatestNormalModelRequest,
    market_store: MarketDataStore,
    curve_store: object,
    *,
    use_candidate: bool = False,
) -> dict:
    market = _copy_loaded_market(request, market_store, curve_store)
    histories = _fixing_histories(request.valuation_date)
    options = {"use_candidate": True} if use_candidate else {}
    snapshot = _prepare(
        market, request.valuation_date, request.source_token, "loaded", histories, **options
    )
    if (
        loaded_context(market_store, curve_store).source_token != request.source_token
        or _fixing_fingerprint(_fixing_histories(request.valuation_date))
        != snapshot.fixing_fingerprint
    ):
        raise ValueError("Sources ou fixings modifiés pendant la calibration : recalculer.")
    return _publish(snapshot)


def _prepare_latest_normal_workbook(
    content: bytes, valuation_date: date, *, use_candidate: bool = False
) -> dict:
    market = _workbook_market(content, valuation_date)
    histories = _fixing_histories(valuation_date)
    token = sha256(content + str(valuation_date).encode()).hexdigest()
    options = {"use_candidate": True} if use_candidate else {}
    snapshot = _prepare(market, valuation_date, token, "workbook", histories, **options)
    if _fixing_fingerprint(_fixing_histories(valuation_date)) != snapshot.fixing_fingerprint:
        raise ValueError("Fixings modifiés pendant la calibration : recalculer.")
    return _publish(snapshot)


def prepare_latest_normal_model(
    request: LatestNormalModelRequest,
    market_store: MarketDataStore,
    curve_store: object,
) -> dict:
    """Prepare a loaded-market snapshot with typed domain failures."""
    try:
        return _prepare_latest_normal_model(request, market_store, curve_store)
    except ServiceError:
        raise
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, OverflowError, KeyError, TypeError) as exc:
        raise LatestNormalPricingError(str(exc)) from exc


def prepare_latest_normal_workbook(content: bytes, valuation_date: date) -> dict:
    """Prepare a workbook snapshot with typed validation and numerical failures."""
    try:
        return _prepare_latest_normal_workbook(content, valuation_date)
    except ServiceError:
        raise
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, OverflowError, KeyError, TypeError) as exc:
        raise LatestNormalPricingError(str(exc)) from exc


def _snapshot_for_pricing(request, market_store, curve_store, expected_model: str = MODEL):
    with _cache_lock:
        snapshot = _snapshots.get(request.model_id)
        if snapshot is None:
            raise ValueError(
                "Calibration expirée ou serveur redémarré : recalculer le dernier modèle."
            )
        _snapshots.move_to_end(request.model_id)
    if snapshot.model != expected_model:
        raise ValueError(
            "La calibration appartient à un autre modèle : recalculer dans le bon onglet."
        )
    if snapshot.valuation_date != request.valuation_date:
        raise ValueError("Date différente de la calibration : recalculer le modèle.")
    if (
        snapshot.source == "loaded"
        and loaded_context(market_store, curve_store).source_token != snapshot.source_token
    ):
        raise ValueError("Sources modifiées : recalculer le dernier modèle.")
    if (
        _fixing_fingerprint(_fixing_histories(request.valuation_date))
        != snapshot.fixing_fingerprint
    ):
        raise ValueError("Fixings modifiés : recalculer le dernier modèle.")
    return snapshot


def _pricing_targets(snapshot, request, forward, expiry, start, end, fixing, payment):
    """Use existing time targets or the existing 3M/6M bridge for early 6M coupons."""
    curves = snapshot.families[request.index_tenor]
    strikes = [curve["strike"] for curve in curves]
    valuation = date_to_ql(request.valuation_date)
    term = ql.Actual365Fixed().yearFraction(valuation, payment)

    def lower_time(curve):
        fitted = curve.get("time_interpolation")
        if not fitted:
            return float("-inf")
        prefix = fitted["prefix"]
        return prefix[0]["time"] if prefix else fitted["times"][0]

    needs_bridge = request.index_tenor == "6M" and any(
        term < lower_time(curve) - 1e-10 for curve in curves
    )
    if not needs_bridge:
        targets = [
            target_at(curve, term, "quadratic", str(ql_to_date(payment)), str(ql_to_date(fixing)))[
                0
            ]
            for curve in curves
        ]
        return targets, "quadratic", []

    source = snapshot.source_3m_family
    if source is None or not source.optionlets:
        raise ValueError(
            "Coupon 6M avant le domaine natif : recalibrer le modèle avec la source du raccord 3M→6M."
        )
    source_times = [node.expiry_years for node in source.optionlets]
    calendar = ql.TARGET()
    spot = calendar.advance(valuation, 2, ql.Days)
    if (
        expiry < min(source_times) - 1e-10
        or expiry > max(source_times) + 1e-10
        or start > calendar.advance(spot, ql.Period(2, ql.Years), ql.ModifiedFollowing)
    ):
        raise ValueError(
            "Fixing du coupon 6M hors du domaine du raccord 3M→6M ; aucune extrapolation."
        )

    coupon_contract = coupon_forward_contract(request.index_tenor, fixing)
    contractual_day_count = {
        "ACT/360": ql.Actual360(),
        "ACT/365F": ql.Actual365Fixed(),
        "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
    }[request.day_count]
    node = OptionletNode(
        number=0,
        cap_maturity_months=0,
        fixing_date=ql_to_date(fixing),
        start_date=ql_to_date(start),
        end_date=ql_to_date(end),
        payment_date=ql_to_date(payment),
        index_maturity_date=ql_to_date(coupon_contract.index_maturity_date),
        accrual_days=int(end - start),
        accrual_fraction=contractual_day_count.yearFraction(start, end),
        projection_accrual_fraction=coupon_contract.projection_accrual_fraction,
        expiry_years=expiry,
        forward_rate=forward,
        discount_factor=snapshot.curves.get_discount_curve("OIS").discount(payment),
        source="initialized",
        volatilities=[None] * len(strikes),
        prices=[None] * len(strikes),
    )
    outside_bridge: list[float] = []
    _convert_short_node(
        node,
        source,
        strikes,
        snapshot.curves,
        True,
        1.0,
        source_smile_interpolation="natural_cubic",
        **(
            {"source_vol_provider": bridge_provider(snapshot.candidate, outside_bridge)}
            if snapshot.candidate is not None
            else {}
        ),
    )
    if any(value is None or not isfinite(value) or value <= 0 for value in node.volatilities):
        raise ValueError("Smile du raccord court 3M→6M indisponible.")
    notes = [
        f"Flux 6M du {node.start_date} au {node.end_date} : raccord normal 3M→6M à même moneyness, sur fixing 3M couvert."
    ]
    if outside_bridge:
        notes.append(
            f"Raccord du coupon 6M : {len(outside_bridge)} lectures de la source candidate hors du domaine affiché "
            f"[{min(outside_bridge):.6%} ; {max(outside_bridge):.6%}]. "
            "Formule figée conservée ; aile étendue non validée."
        )
    return [float(value) for value in node.volatilities], "short_3m_to_6m_bridge", notes


def _price_latest_normal(
    request: LatestNormalPriceRequest,
    market_store: MarketDataStore | None,
    curve_store: object,
    *,
    expected_model: str = MODEL,
    snapshot_override: _Snapshot | None = None,
    coupon_vol_provider=None,
    coupon_schedule=None,
) -> dict:
    # Internal risk scenarios reuse the exact coupon lifecycle without publishing
    # temporary models into the bounded user snapshot cache.
    snapshot = snapshot_override
    if snapshot is None:
        snapshot = _snapshot_for_pricing(request, market_store, curve_store, expected_model)
    elif snapshot.valuation_date != request.valuation_date:
        raise ValueError("Date différente de la calibration du scénario.")
    elif snapshot.model != expected_model:
        raise ValueError("La calibration du scénario appartient à un autre modèle.")
    calendar = ql.TARGET()
    convention = {
        "ModifiedFollowing": ql.ModifiedFollowing,
        "Following": ql.Following,
        "Preceding": ql.Preceding,
        "Unadjusted": ql.Unadjusted,
    }[request.business_day_convention]
    dc = {
        "ACT/360": ql.Actual360(),
        "ACT/365F": ql.Actual365Fixed(),
        "30/360": ql.Thirty360(ql.Thirty360.BondBasis),
    }[request.day_count]
    warnings, rows, annuities, forward_annuity, parity = [], [], [], [], []
    with ql_evaluation_date(request.valuation_date):
        valuation = date_to_ql(request.valuation_date)
        schedule = cap_schedule(request, calendar, convention)
        schedule_dates = [ql_to_date(schedule_date) for schedule_date in schedule]
        if len(schedule_dates) < 2:
            raise ValueError("Le schedule ajusté ne contient aucun flux.")
        if coupon_schedule is not None and len(coupon_schedule) != len(schedule_dates) - 1:
            raise ValueError(f"Échéancier amortissable : {len(schedule_dates) - 1} coupons attendus, {len(coupon_schedule)} reçus.")
        if coupon_schedule is not None and request.include_risk:
            raise ValueError("Risque amortissable non pris en charge.")
        schedule_descriptor = {
            "expected_period_count": len(schedule_dates) - 1,
            "adjusted_start_date": str(schedule_dates[0]),
            "adjusted_end_date": str(schedule_dates[-1]),
        }
        projection = snapshot.curves.get_forward_curve(request.index_tenor)
        discount = snapshot.curves.get_discount_curve("OIS")
        curves = snapshot.families[request.index_tenor]
        strikes = [c["strike"] for c in curves]
        for coupon_index, (start, end) in enumerate(itertools.pairwise(schedule)):
            terms = coupon_schedule[coupon_index] if coupon_schedule is not None else None
            coupon_request = request.model_copy(update=terms) if terms is not None else request
            coupon_notional, coupon_strike = coupon_request.notional, coupon_request.strike
            fixing = calendar.advance(start, -request.fixing_days, ql.Days)
            payment = calendar.advance(end, request.payment_lag, ql.Days)
            fixing_date, payment_date = ql_to_date(fixing), ql_to_date(payment)
            observed = snapshot.histories[request.index_tenor].get(fixing_date)
            accrual = dc.yearFraction(start, end)
            coupon_contract = coupon_forward_contract(request.index_tenor, fixing)
            row = {
                "start_date": str(ql_to_date(start)),
                "end_date": str(ql_to_date(end)),
                "fixing_date": str(fixing_date),
                "payment_date": str(payment_date),
                "accrual_fraction": accrual,
                "index_value_date": str(ql_to_date(coupon_contract.index_value_date)),
                "index_maturity_date": str(ql_to_date(coupon_contract.index_maturity_date)),
                "projection_accrual_fraction": (coupon_contract.projection_accrual_fraction),
                "observed_fixing": None,
                "normal_vol": None,
                "parameters": None,
                "vol_source": None,
                "expiry_years": max(0.0, ql.Actual365Fixed().yearFraction(valuation, fixing)),
                "price": 0.0,
                "normal_vega_eur_bp": 0.0,
            }
            if terms is not None:
                row.update(notional=coupon_notional, strike=coupon_strike)
            if payment <= valuation:
                row.update(
                    status="paid",
                    forward_rate=observed if observed is not None else 0.0,
                    observed_fixing=observed,
                    discount_factor=1.0,
                )
                rows.append(row)
                continue
            if payment > discount.maxDate():
                raise ValueError("Paiement hors du domaine de la courbe €STR.")
            df = discount.discount(payment)
            annuity = accrual * df
            if fixing <= valuation:
                if observed is None:
                    raise ValueError(
                        f"Fixing Euribor {request.index_tenor} du {fixing_date} absent : importer le taux observé."
                    )
                forward = observed
                intrinsic = (
                    max(forward - coupon_strike, 0.0)
                    if request.product_type == "cap"
                    else max(coupon_strike - forward, 0.0)
                )
                row.update(
                    status="fixed",
                    observed_fixing=observed,
                    price=coupon_notional * annuity * intrinsic,
                )
            else:
                forward = coupon_contract.forward_rate(projection)
                expiry = row["expiry_years"]
                ql.Actual365Fixed().yearFraction(valuation, payment)
                override = (
                    coupon_vol_provider(
                        snapshot, coupon_request, forward, expiry, start, end, fixing, payment
                    )
                    if coupon_vol_provider is not None
                    else None
                )
                if override is not None:
                    sigma = float(override["normal_vol"])
                    if not isfinite(sigma) or sigma <= 0:
                        raise ValueError("Volatilité de l'extension non positive ou non finie.")
                    params = None
                    vol_source = override["vol_source"]
                    row["coverage"] = override["coverage"]
                    warnings.extend(override["warnings"])
                elif snapshot.candidate is not None and request.index_tenor == "3M":
                    candidate_key = (float(forward), float(expiry))
                    with _cache_lock:
                        checked = candidate_key in snapshot.checked_candidate_coupons
                    if not checked:
                        check_coupon_shape(snapshot.candidate, forward, expiry)
                        with _cache_lock:
                            if len(snapshot.checked_candidate_coupons) >= MAX_CACHED_FITS:
                                snapshot.checked_candidate_coupons.clear()
                            snapshot.checked_candidate_coupons.add(candidate_key)
                    sigma = float(
                        snapshot.candidate.vol_for_coupon(forward, expiry, coupon_strike)
                    )
                    params = None
                    vol_source = "candidate_fixing_time_moneyness"
                else:
                    targets, vol_source, source_notes = _pricing_targets(
                        snapshot, coupon_request, forward, expiry, start, end, fixing, payment
                    )
                    warnings.extend(source_notes)
                    params = _fit(snapshot, request.index_tenor, forward, expiry, targets)
                    if "price_repair" in params:
                        warnings.append(price_repair_warning(params["price_repair"]))
                    sigma = latest_normal_vol(
                        forward,
                        expiry,
                        strikes,
                        targets,
                        params,
                        request.index_tenor,
                        coupon_strike,
                    )
                std = sigma * sqrt(expiry)
                sign = 1 if request.product_type == "cap" else -1
                distance = sign * (forward - coupon_strike)
                z = distance / std
                density = exp(-z * z / 2) / sqrt(2 * pi)
                row.update(
                    status="future",
                    normal_vol=sigma,
                    vol_source=vol_source,
                    price=coupon_notional * annuity * (distance * ndtr(z) + std * density),
                    normal_vega_eur_bp=coupon_notional * annuity * sqrt(expiry) * density * 1e-4,
                    parameters=(
                        {q: params[q] for q in ("alpha", "beta", "rho", "nu")}
                        if params is not None
                        else None
                    ),
                )
                if params is not None and params["bound_hit"]:
                    warnings.append(
                        "Un paramètre D.4 atteint une borne de calibration sur certains flux."
                    )
            row.update(forward_rate=forward, discount_factor=df)
            if (
                not all(
                    isfinite(row[k])
                    for k in ("price", "forward_rate", "discount_factor", "normal_vega_eur_bp")
                )
                or row["price"] < -1e-8
            ):
                raise ValueError("Prix ou données de flux non finis.")
            annuities.append(annuity * (coupon_notional / request.notional))
            forward_annuity.append(annuity * (coupon_notional / request.notional) * forward)
            parity.append(coupon_notional * annuity * (forward - coupon_strike))
            rows.append(row)
    if snapshot_override is None:
        _snapshot_for_pricing(request, market_store, curve_store, expected_model)
    if not annuities:
        warnings.append("Tous les flux sont déjà payés.")
    price = fsum(r["price"] for r in rows)
    conventions = {
        key: value
        for key, value in request.model_dump(mode="json").items()
        if key
        not in (
            "model_id",
            "product_type",
            "index_tenor",
            "notional",
            "strike",
            "valuation_date",
            "include_risk",
        )
    }
    conventions.update(
        currency="EUR",
        discounting="ESTR",
        calendar="TARGET",
        time_method="quadratic",
        model_version=VERSION,
        smile_policy=smile_policy(request.index_tenor),
        smile_transition=list(smile_transition(strikes, request.index_tenor)),
        native_cubic_domain=[
            0.02 if request.index_tenor == "3M" else min(strikes),
            STRIKE_DOMAIN[1],
        ],
        schedule=schedule_descriptor,
        normal_vega_eur_bp_definition=(
            "parallel_1bp_all_future_coupon_normal_vols_forwards_and_discount_factors_frozen"
        ),
    )
    if snapshot.candidate is not None:
        conventions["model_version"] = CANDIDATE_VERSION
        if request.index_tenor == "3M":
            conventions.update(
                time_method="candidate_fixing_time_moneyness",
                smile_policy=CANDIDATE_POLICY,
                smile_transition=[-0.015, 0.0],
                native_cubic_domain=[0.0, STRIKE_DOMAIN[1]],
            )
            warnings.append(
                "Coupons 3M hors grille : interpolation en temps de fixing à même K−F ; comparaison ICE limitée aux fixings natifs."
            )
    flat_vol = flat_equivalent_normal_vol(
        product_type=request.product_type,
        strike=request.strike,
        notional=request.notional,
        rows=rows,
        total_price=price,
    )
    result = {
        "model": snapshot.model,
        "version": CANDIDATE_VERSION if snapshot.candidate is not None else VERSION,
        "model_id": request.model_id,
        "valuation_date": str(request.valuation_date),
        "product_type": request.product_type,
        "index_tenor": request.index_tenor,
        "strike": request.strike,
        "notional": request.notional,
        "price": price,
        "unit_price": price / request.notional,
        "normal_vega_eur_bp": fsum(r["normal_vega_eur_bp"] for r in rows),
        "atm_rate": fsum(forward_annuity) / fsum(annuities) if fsum(annuities) else 0.0,
        "cap_floor_parity_eur": fsum(parity),
        "caplets": rows,
        "warnings": list(dict.fromkeys(warnings)),
        "conventions": conventions,
        "flat_equivalent_normal_vol": flat_vol.model_dump(),
    }
    if request.include_risk:
        result["risk"] = _price_risk(request, snapshot, rows, price)
        if snapshot_override is None:
            _snapshot_for_pricing(request, market_store, curve_store, expected_model)
    return result


def price_latest_normal(
    request: LatestNormalPriceRequest,
    market_store: MarketDataStore | None,
    curve_store: object,
) -> dict:
    """Price from an immutable snapshot and validate the public response contract."""
    try:
        result = _price_latest_normal(request, market_store, curve_store)
        LatestNormalPriceResponse.model_validate(result)
        return result
    except ServiceError:
        raise
    except PydanticValidationError as exc:
        raise LatestNormalPricingError(str(exc)) from exc
    except ValueError as exc:
        raise LatestNormalValidationError(str(exc)) from exc
    except (RuntimeError, OverflowError, KeyError, TypeError) as exc:
        raise LatestNormalPricingError(str(exc)) from exc


def _frozen_option_value(request, rows, forward_curve, discount_curve):
    """Reprice fixed cashflows and future Bachelier options with base normal vols."""
    cashflow_values = []
    sign = 1 if request.product_type == "cap" else -1
    for row in rows:
        if row["status"] == "paid":
            continue
        payment = date_to_ql(date.fromisoformat(row["payment_date"]))
        df = discount_curve.discount(payment)
        if row["status"] == "fixed":
            forward = row["observed_fixing"]
            payoff = max(sign * (forward - request.strike), 0.0)
        else:
            fixing = date_to_ql(date.fromisoformat(row["fixing_date"]))
            coupon_contract = coupon_forward_contract(request.index_tenor, fixing)
            if (
                row["index_value_date"] != str(ql_to_date(coupon_contract.index_value_date))
                or row["index_maturity_date"]
                != str(ql_to_date(coupon_contract.index_maturity_date))
                or abs(
                    row["projection_accrual_fraction"] - coupon_contract.projection_accrual_fraction
                )
                > 1e-15
            ):
                raise ValueError("Contrat de projection Euribor incohérent.")
            forward = coupon_contract.forward_rate(forward_curve)
            std = row["normal_vol"] * sqrt(row["expiry_years"])
            distance = sign * (forward - request.strike)
            z = distance / std
            payoff = distance * ndtr(z) + std * exp(-z * z / 2) / sqrt(2 * pi)
        cashflow_values.append(request.notional * row["accrual_fraction"] * df * payoff)
    return fsum(cashflow_values)


def _price_risk(
    request: LatestNormalPriceRequest,
    snapshot: _Snapshot,
    rows: list[dict[str, object]],
    base_price: float,
) -> dict[str, object]:
    """Market-quote DV01 and KRR with the calibrated smile held fixed."""
    with _observed_fixings(request.valuation_date, snapshot.histories):
        discount = snapshot.curves.get_discount_curve("OIS")
        forward = snapshot.curves.get_forward_curve(request.index_tenor)
        discount_key = canonical_discount_curve_key("EUR", "OIS")
        forward_key = canonical_forward_curve_key("EUR", request.index_tenor, request.index_tenor)

        def reprice(forward_curve, discount_curve):
            return RepriceOutput(
                npv=_frozen_option_value(request, rows, forward_curve, discount_curve)
            )

        quote_base = _quote_calibration_base(
            curve_set=snapshot.curves,
            currency="EUR",
            curve_date=snapshot.curves.curve_date,
            ois_data=snapshot.curves.ois_market_data,
            forward_market_data=dict(snapshot.curves.forward_market_data),
            discount_key=discount_key,
            forward_key=forward_key,
            base_discount_curve=discount,
            base_forward_curve=forward,
            constant_ibor_fixing=True,
        )
        reference_price = reprice(
            quote_base.scenario.forward_curve,
            quote_base.scenario.discount_curve,
        ).npv
        residual = reference_price - base_price
        reference_check = RiskReferenceCheck(
            base_price_eur=base_price,
            reference_price_eur=reference_price,
            residual_eur=residual,
            tolerance_eur=RISK_REFERENCE_TOLERANCE_EUR,
            passed=abs(residual) <= RISK_REFERENCE_TOLERANCE_EUR,
        )
        if not reference_check.passed:
            raise ValueError(
                "Risque indisponible : la référence nulle diverge du prix initial "
                f"de {residual:.12g} EUR (tolérance "
                f"{RISK_REFERENCE_TOLERANCE_EUR:.12g} EUR)."
            )

        sensitivity = compute_curve_sensitivities(
            curve_set=snapshot.curves,
            currency="EUR",
            curve_date=snapshot.curves.curve_date,
            bump_size=1e-4,
            base_output=RepriceOutput(npv=base_price),
            base_discount_curve=discount,
            base_forward_curve=forward,
            discount_curve_key=discount_key,
            forward_curve_key=forward_key,
            reprice_with_curves=reprice,
            delta_convention="base_minus_bumped",
            constant_ibor_fixing=True,
            net_only_key_rate_risk=True,
        )
    diagnostics = curve_diagnostics_fields(sensitivity)
    return {
        "dv01_eur_bp": sensitivity.quote_dv01_total,
        "diagnostics": diagnostics,
        "volatility_policy": "frozen_normal_vol",
        "reference_check": reference_check.model_dump(),
        "conventions": {
            "source": "market_quote_jacobian",
            "derivative_sign": "-1e-4*dPV/dq",
            "unit": "EUR/bp",
            "quote_shift_bp": 1.0,
            "parallel_diagnostic": "CENTRAL",
            "calibration_fixing_policy": "constant_observed_ibor",
            "volatility_policy": "frozen_per_coupon_normal_vol",
        },
    }
