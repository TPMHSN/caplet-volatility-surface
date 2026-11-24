"""Reuse loaded curves to explain one future optionlet; no stripping or pricing."""

import hashlib
import json
from math import isfinite

import QuantLib as ql

from config import get_index
from curves.helpers import date_to_ql, ql_evaluation_date, ql_to_date
from services.thomas_optionlet_schemas import (
    LoadedContext,
    LoadedCurve,
    OptionletPreparation,
    OptionletRequest,
)


def _data(model):
    return model.model_dump(mode="json") if model is not None else None


def loaded_context(market_store, curve_store) -> LoadedContext:
    """Inspect actual component dates, not the store's first-import date."""
    ois = market_store.get_ois_curve("EUR")
    forwards = {t: market_store.get_forward_curve("EUR", t) for t in ("3M", "6M")}
    caps = market_store.get_capfloor_surface("EUR")
    curves = curve_store.get_curve_set("EUR")
    sources = {"OIS": ois, **forwards}
    curve_rows = []
    signatures = {}
    for key, source in sources.items():
        built = curves is not None and (key == "OIS" or key in curves.forward_curves)
        original = None
        if curves is not None:
            original = (
                curves.ois_market_data if key == "OIS" else curves.forward_market_data.get(key)
            )
        current = bool(
            built
            and source is not None
            and original is not None
            and source.currency == "EUR"
            and source.valuation_date == curves.curve_date
            and (key == "OIS" or source.index_tenor == key)
            and _data(source) == _data(original)
        )
        curve_rows.append(
            LoadedCurve(
                key=key,
                label="€STR / OIS" if key == "OIS" else f"Euribor {key}",
                market_date=source.valuation_date if source is not None else None,
                loaded=source is not None,
                built=built,
                current=current,
            )
        )
        signatures[key] = {"source": _data(source), "built_from": _data(original), "built": built}
    payload = {
        "curves": signatures,
        "cap_quotes": _data(caps),
        "curve_date": str(curves.curve_date) if curves else None,
        "curve_instance": id(curves) if curves else None,
        "handles": {k: id(v) for k, v in curves.forward_curves.items()} if curves else {},
        "interpolation": curves.interpolation if curves else None,
    }
    token = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return LoadedContext(
        market_date=ois.valuation_date if ois else None,
        curve_date=curves.curve_date if curves else None,
        curves=curve_rows,
        cap_quotes_loaded=caps is not None,
        cap_quotes_date=caps.valuation_date if caps else None,
        cap_expiry_count=len(caps.expiries) if caps else 0,
        source_token=token,
    )


def prepare_optionlet(request: OptionletRequest, market_store, curve_store) -> OptionletPreparation:
    context = loaded_context(market_store, curve_store)
    if request.source_token != context.source_token:
        raise ValueError(
            "Les données ou les courbes ont changé : actualiser les sources avant de recalculer."
        )
    relevant = [c for c in context.curves if c.key in ("OIS", request.index_tenor)]
    if not all(c.loaded for c in relevant):
        raise ValueError("Importer les données €STR et Euribor sélectionnées dans Market Data.")
    if not all(c.built for c in relevant):
        raise ValueError(
            "Construire les courbes €STR et Euribor sélectionnées dans le menu Courbes."
        )
    if not all(c.current for c in relevant):
        raise ValueError(
            "Les cotations ont changé ou leurs dates diffèrent : reconstruire les courbes dans Courbes."
        )
    if context.curve_date != request.valuation_date:
        raise ValueError(
            "La date de valorisation doit être égale à la date des courbes et des données pour cette étape."
        )
    if request.fixing_date <= request.valuation_date:
        raise ValueError(
            "Choisir un fixing strictement futur. Un fixing passé ou du jour nécessite un taux constaté, traité dans une étape ultérieure."
        )

    index = get_index("EUR", request.index_tenor)
    fixing = date_to_ql(request.fixing_date)
    if not index.isValidFixingDate(fixing):
        raise ValueError(
            "La date de fixing doit être un jour ouvré TARGET ; elle n'est pas déplacée automatiquement."
        )
    start = index.valueDate(fixing)
    end = index.maturityDate(start)
    payment = end  # Explicit standard product scope: no payment delay.
    valuation = date_to_ql(request.valuation_date)
    curve_set = curve_store.get_curve_set("EUR")
    projection = curve_set.get_forward_curve(request.index_tenor)
    discount = curve_set.get_discount_curve("OIS")
    try:
        # Cached bootstrap helpers can depend on global evaluationDate. Restore it
        # after the synchronous calculation; never let pricing adjust the input date.
        with ql_evaluation_date(request.valuation_date):
            if projection.referenceDate() != valuation or discount.referenceDate() != valuation:
                raise ValueError(
                    "La date de référence des courbes ne correspond pas à la valorisation ; reconstruire les courbes."
                )
            if (
                start < projection.referenceDate()
                or end > projection.maxDate()
                or payment > discount.maxDate()
            ):
                raise ValueError(
                    "La période sort du domaine des courbes. Aucune extrapolation n'est autorisée à cette étape."
                )
            accrual = index.dayCounter().yearFraction(start, end)
            forward = projection.forwardRate(start, end, index.dayCounter(), ql.Simple).rate()
            p_start, p_end = projection.discount(start), projection.discount(end)
            df = discount.discount(payment)
    except RuntimeError as exc:
        raise ValueError(f"Les courbes ne peuvent pas être interrogées : {exc}") from exc
    if (
        not all(isfinite(x) for x in (accrual, forward, p_start, p_end, df))
        or min(accrual, p_start, p_end, df) <= 0
    ):
        raise ValueError("Les courbes produisent des facteurs ou un forward invalides.")
    return OptionletPreparation(
        request=request,
        source_token=context.source_token,
        curve_date=context.curve_date,
        projection_curve=f"EUR_{request.index_tenor}",
        discount_curve="EUR_OIS",
        interpolation=curve_set.effective_interpolation,
        start_date=ql_to_date(start),
        end_date=ql_to_date(end),
        payment_date=ql_to_date(payment),
        accrual_days=end - start,
        accrual_fraction=accrual,
        expiry_days=fixing - valuation,
        expiry_years=ql.Actual365Fixed().yearFraction(valuation, fixing),
        projection_factor_start=p_start,
        projection_factor_end=p_end,
        forward_rate=forward,
        discount_factor=df,
        atm_strike=forward,
        forward_minus_strike=forward - request.strike,
        discounted_accrual_notional=request.notional * accrual * df,
    )
