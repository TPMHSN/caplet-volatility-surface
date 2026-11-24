"""Explicit Bachelier bootstrap, keeping ICAP's native 3M and 6M families separate.

Only Thomas owns this calculation: original market/curve/volatility stores are read,
never modified. All grids, prices, assumptions and rejected inversions are returned.
"""

from bisect import bisect_left
from math import fsum, isfinite, sqrt

import QuantLib as ql
from scipy.interpolate import PchipInterpolator

from config import get_index
from curves.helpers import date_to_ql, parse_tenor, ql_evaluation_date, ql_to_date
from services.thomas_black import build_black_surface
from services.thomas_optionlet import loaded_context, prepare_optionlet
from services.thomas_optionlet_schemas import OptionletRequest
from services.thomas_stripping_schemas import (
    CapNode,
    OptionletNode,
    StrippingFamily,
    StrippingIssue,
    StrippingResult,
    TargetVolatility,
)


def _months(label: str) -> int:
    period = parse_tenor(label)
    if period.units() == ql.Years:
        return 12 * period.length()
    if period.units() == ql.Months:
        return period.length()
    raise ValueError("Les maturités des caps doivent être exprimées en mois ou en années.")


def _label(months: int) -> str:
    return f"{months // 12}Y" if months % 12 == 0 else f"{months}M"


def _bracket(xs: list[float], x: float) -> tuple[int, int, float]:
    """Interpolation only; callers deliberately handle the initial flat region."""
    if x < xs[0] - 1e-12 or x > xs[-1] + 1e-12:
        raise ValueError("Point hors du domaine : aucune extrapolation automatique.")
    pos = bisect_left(xs, x)
    if pos == len(xs):
        return pos - 1, pos - 1, 0.0
    if abs(xs[pos] - x) < 1e-12:
        return pos, pos, 0.0
    lo = max(0, pos - 1)
    return lo, pos, (x - xs[lo]) / (xs[pos] - xs[lo])


def _validated_quotes(data):
    if not data.strikes or len(set(data.strikes)) != len(data.strikes):
        raise ValueError("Les strikes absolus doivent être présents et distincts.")
    if any(not isfinite(k) for k in data.strikes):
        raise ValueError("Strike non fini dans les cotations.")
    if not data.expiries or len(data.expiries) != len(data.volatilities):
        raise ValueError("Dimensions des cotations de caps incohérentes.")
    columns = sorted(range(len(data.strikes)), key=lambda j: data.strikes[j])
    strikes = [data.strikes[j] for j in columns]
    native = data.quote_tenors
    if native is not None and len(native) != len(data.expiries):
        raise ValueError("Un indice de cotation est requis pour chaque maturité.")
    families = {}
    for i, (label, vols) in enumerate(zip(data.expiries, data.volatilities)):
        months = _months(label)
        tenor = (native[i] if native is not None else data.index_tenor) or (
            "3M" if months <= 24 else "6M"
        )
        tenor = tenor.strip().upper()
        if tenor not in ("3M", "6M"):
            raise ValueError(f"Indice {tenor} non pris en charge dans le stripping Thomas.")
        step = 3 if tenor == "3M" else 6
        if months < 2 * step or months % step:
            raise ValueError(f"Maturité {label} incompatible avec la grille régulière {tenor}.")
        if len(vols) != len(strikes) or any(not isfinite(v) or v <= 0 for v in vols):
            raise ValueError(
                f"{label} : volatilité manquante, nulle ou invalide ; aucune cotation n’est remplacée automatiquement."
            )
        families.setdefault(tenor, []).append((months, [vols[j] for j in columns]))
    for rows in families.values():
        rows.sort(key=lambda row: row[0])
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError("Maturités dupliquées dans une famille de cotations.")
    return strikes, families


def _schedule(valuation, spot, tenor, maximum, first_quote, curve_set, strikes):
    index = get_index("EUR", tenor)
    cal = index.fixingCalendar()
    step = 3 if tenor == "3M" else 6
    eom = cal.isEndOfMonth(spot)
    projection, discount = curve_set.get_forward_curve(tenor), curve_set.get_discount_curve("OIS")
    if projection.referenceDate() != valuation or discount.referenceDate() != valuation:
        raise ValueError(
            "Les dates de référence des courbes doivent correspondre à la valorisation."
        )
    # Shared contractual roll anchor. Advancing each date from the ORIGINAL spot
    # prevents backwards schedules from drifting after a terminal holiday adjustment.
    boundaries = {}
    for month in range(0, maximum + 1, step):
        raw = spot + ql.Period(month, ql.Months)
        if eom:
            raw = ql.Date.endOfMonth(raw)
        boundaries[month] = cal.adjust(raw, ql.ModifiedFollowing)
    result = []
    # Spot's first coupon is already fixed on valuation and excluded from market caps.
    for month in range(2 * step, maximum + 1, step):
        start, end = boundaries[month - step], boundaries[month]
        fixing = index.fixingDate(start)
        index_end = index.maturityDate(start)
        if fixing <= valuation or start >= end:
            raise ValueError(
                "Échéancier de cap invalide : les caplets inclus doivent avoir un fixing futur."
            )
        if (
            start < projection.referenceDate()
            or index_end > projection.maxDate()
            or end > discount.maxDate()
        ):
            raise ValueError(
                f"Le cap {tenor} {_label(month)} dépasse le domaine des courbes ; extrapolation refusée."
            )
        delta = index.dayCounter().yearFraction(start, end)
        projection_delta = index.dayCounter().yearFraction(start, index_end)
        forward = (
            projection.discount(start) / projection.discount(index_end) - 1
        ) / projection_delta
        df = discount.discount(end)
        expiry = ql.Actual365Fixed().yearFraction(valuation, fixing)
        if (
            not all(isfinite(x) for x in (delta, projection_delta, forward, df, expiry))
            or min(delta, projection_delta, df, expiry) <= 0
        ):
            raise ValueError("Les courbes donnent un forward ou un facteur invalide.")
        result.append(
            OptionletNode(
                number=len(result) + 1,
                cap_maturity_months=month,
                fixing_date=ql_to_date(fixing),
                start_date=ql_to_date(start),
                end_date=ql_to_date(end),
                payment_date=ql_to_date(end),
                index_maturity_date=ql_to_date(index_end),
                accrual_days=end - start,
                accrual_fraction=delta,
                projection_accrual_fraction=projection_delta,
                expiry_years=expiry,
                forward_rate=forward,
                discount_factor=df,
                source="initialized" if month <= first_quote else "stripped",
                volatilities=[None] * len(strikes),
                prices=[0.0] * len(strikes),
            )
        )
    return result


def _time_value(node: OptionletNode, strike: float, vol: float) -> float:
    """OTM formula avoids subtracting a large ITM intrinsic value in the wings."""
    option = ql.Option.Put if node.forward_rate > strike else ql.Option.Call
    return ql.bachelierBlackFormula(
        option,
        strike,
        node.forward_rate,
        vol * sqrt(node.expiry_years),
        node.accrual_fraction * node.discount_factor,
    )


def _intrinsic(node, strike):
    return node.accrual_fraction * node.discount_factor * max(node.forward_rate - strike, 0.0)


def _implied_vol(node, strike, time_value, guess):
    """Use relative price accuracy, including positive prices deep in the wings."""

    def accurate(sigma):
        return (
            isfinite(sigma)
            and sigma > 0
            and abs(_time_value(node, strike, sigma) / time_value - 1) < 1e-8
        )

    option = ql.Option.Put if node.forward_rate > strike else ql.Option.Call
    try:
        sigma = ql.bachelierBlackFormulaImpliedVol(
            option,
            strike,
            node.forward_rate,
            node.expiry_years,
            time_value,
            node.accrual_fraction * node.discount_factor,
        )
        if accurate(sigma):
            return sigma
    except RuntimeError:
        pass
    # QuantLib's approximation can round a small positive OTM price to zero vol.
    # A monotone bracketed inversion avoids any absolute-price floor in that case.
    low, high = 0.0, max(guess, 1e-8)
    for _ in range(64):
        if _time_value(node, strike, high) >= time_value:
            break
        high *= 2
    else:
        raise ValueError("Impossible d’encadrer la volatilité normale de ce caplet.")
    for _ in range(120):
        sigma = (low + high) / 2
        price = _time_value(node, strike, sigma)
        if accurate(sigma):
            return sigma
        if price < time_value:
            low = sigma
        else:
            high = sigma
    raise ValueError("Valeur temps trop petite pour une inversion suffisamment précise.")


def _build_family(valuation, spot, tenor, rows, strikes, notional, curve_set):
    first, last = rows[0][0], rows[-1][0]
    options = _schedule(valuation, spot, tenor, last, first, curve_set, strikes)
    dates = {o.cap_maturity_months: date_to_ql(o.payment_date) for o in options}
    known_months = [r[0] for r in rows]
    times = [ql.Actual365Fixed().yearFraction(valuation, dates[m]) for m in known_months]
    # Only original market pillars determine the shape. Each strike is a separate
    # PCHIP column; interpolated caps never become additional calibration nodes.
    market_flats = dict(rows)
    flat_interpolator = (
        PchipInterpolator(times, [row[1] for row in rows], axis=0, extrapolate=False)
        if len(rows) > 1
        else None
    )
    caps, issues = [], []
    previous_tv = [0.0] * len(strikes)
    # Accumulators are per unit notional, keeping numerical thresholds size-independent.
    for i, optionlet in enumerate(options):
        month = optionlet.cap_maturity_months
        source = (
            "market"
            if month in known_months
            else "initialized" if month < first else "interpolated"
        )
        if month <= first:
            flat = list(rows[0][1])
        elif month in market_flats:
            flat = list(market_flats[month])  # Preserve original quotes exactly.
        else:
            time = ql.Actual365Fixed().yearFraction(valuation, dates[month])
            if flat_interpolator is None or not times[0] <= time <= times[-1]:
                raise ValueError("Vol flat hors du domaine coté : aucune extrapolation PCHIP.")
            flat = flat_interpolator(time).tolist()
            if any(not isfinite(vol) or vol <= 0 for vol in flat):
                raise ValueError("L’interpolation PCHIP donne une volatilité flat invalide.")
        prices, repriced, errors = [], [], []
        for j, strike in enumerate(strikes):
            included = options[: i + 1]
            cap_tv = fsum(_time_value(o, strike, flat[j]) for o in included)
            intrinsic = fsum(_intrinsic(o, strike) for o in included)
            cap_price = intrinsic + cap_tv
            delta_tv = cap_tv - previous_tv[j]
            previous_tv[j] = cap_tv
            optionlet.prices[j] = notional * (_intrinsic(optionlet, strike) + delta_tv)
            if month <= first:
                optionlet.volatilities[j] = flat[j]
                optionlet.prices[j] = notional * (
                    _intrinsic(optionlet, strike) + _time_value(optionlet, strike, flat[j])
                )
            else:
                try:
                    if delta_tv < 0:
                        raise ValueError(
                            "La différence des prix donne une valeur temps négative : interpolation incompatible avec ce caplet."
                        )
                    if delta_tv == 0:
                        raise ValueError(
                            "Valeur temps numériquement nulle : volatilité non identifiable à la précision disponible."
                        )
                    optionlet.volatilities[j] = _implied_vol(optionlet, strike, delta_tv, flat[j])
                except (RuntimeError, ValueError) as exc:
                    issues.append(
                        StrippingIssue(
                            strike=strike,
                            cap_maturity_months=month,
                            code=(
                                "invalid_caplet_price"
                                if delta_tv < 0
                                else "unidentified_volatility"
                            ),
                            message=str(exc),
                        )
                    )
            prices.append(notional * cap_price)
            if any(o.volatilities[j] is None for o in included):
                repriced.append(None)
                errors.append(None)
            else:
                rebuilt = intrinsic + fsum(
                    _time_value(o, strike, o.volatilities[j]) for o in included
                )
                repriced.append(notional * rebuilt)
                errors.append(notional * (rebuilt - cap_price))
        if not all(isfinite(p) for p in prices + optionlet.prices):
            raise ValueError("Notionnel trop grand ou calcul de prix non fini.")
        caps.append(
            CapNode(
                maturity_months=month,
                expiry=_label(month),
                maturity_date=optionlet.payment_date,
                source=source,
                optionlets_count=i + 1,
                flat_vols=flat,
                prices=prices,
                repriced_prices=repriced,
                repricing_errors=errors,
            )
        )
    residuals = [
        abs(e) for c in caps if c.source == "market" for e in c.repricing_errors if e is not None
    ]
    for c in caps:
        if c.source == "market":
            for j, err in enumerate(c.repricing_errors):
                if err is not None and abs(err) > notional * 1e-10:
                    issues.append(
                        StrippingIssue(
                            strike=strikes[j],
                            cap_maturity_months=c.maturity_months,
                            code="repricing_error",
                            message="Le prix reconstruit du cap dépasse la tolérance de repricing.",
                        )
                    )
    return StrippingFamily(
        index_tenor=tenor,
        status="partial" if issues else "ready",
        strikes=strikes,
        first_quoted_expiry=_label(first),
        last_quoted_expiry=_label(last),
        caps=caps,
        optionlets=options,
        issues=issues,
        market_count=sum(c.source == "market" for c in caps),
        interpolated_count=sum(c.source == "interpolated" for c in caps),
        initialized_count=sum(c.source == "initialized" for c in caps),
        max_abs_repricing_error=max(residuals) if residuals else None,
        black_surface=build_black_surface(
            strikes, [o.fixing_date for o in options], [o.expiry_years for o in options],
            [o.forward_rate for o in options], [o.volatilities for o in options],
        ),
    )


def _query_target(request, preparation, families):
    result = {
        "index_tenor": request.index_tenor,
        "strike": request.strike,
        "fixing_date": request.fixing_date,
        "normal_vol": None,
    }
    family = next((f for f in families if f.index_tenor == request.index_tenor), None)
    if family is None:
        return TargetVolatility(
            **result, status="unavailable", message="Aucune cotation native pour cet indice."
        )
    if request.stripping_method == "block_bootstrap":
        from services.thomas_stripping_blocks import query_block_vol

        try:
            _bracket(family.strikes, request.strike)
            _bracket([o.expiry_years for o in family.optionlets], preparation.expiry_years)
        except ValueError:
            return TargetVolatility(**result, status="out_of_domain",
                message="Contrat hors des strikes absolus ou des fixings de la surface ; aucune extrapolation du contrat demandé.")
        try:
            result["normal_vol"] = query_block_vol(
                family, preparation.expiry_years, request.strike, None,
            )
        except (ValueError, RuntimeError) as exc:
            return TargetVolatility(**result, status="unavailable", message=str(exc))
        return TargetVolatility(**result, status="available", message=(
            "Smile normale PWL enrichie des points ATM de caps activés ; interpolation entre fixings à strike constant. "
            "Les premiers caplets 6M viennent du modèle de conversion 3M→6M, pas de cotations 6M individuelles."
        ))
    try:
        tlo, thi, tw = _bracket(
            [o.expiry_years for o in family.optionlets], preparation.expiry_years
        )
        klo, khi, kw = _bracket(family.strikes, request.strike)
    except ValueError:
        return TargetVolatility(
            **result,
            status="out_of_domain",
            message=(
                f"Contrat hors du domaine {family.index_tenor} : fixings du {family.optionlets[0].fixing_date} "
                f"au {family.optionlets[-1].fixing_date}, strikes de {family.strikes[0]*100:g} % "
                f"à {family.strikes[-1]*100:g} %. Aucune extrapolation ni conversion d’indice."
            ),
        )
    values = [
        family.optionlets[t].volatilities[k]
        for t, k in [(tlo, klo), (tlo, khi), (thi, klo), (thi, khi)]
    ]
    if any(v is None for v in values):
        return TargetVolatility(
            **result,
            status="unavailable",
            message="Un point nécessaire à l’interpolation a échoué au stripping ; aucune volatilité de remplacement.",
        )
    result["normal_vol"] = ((1 - kw) * values[0] + kw * values[1]) * (1 - tw) + (
        (1 - kw) * values[2] + kw * values[3]
    ) * tw
    initial = any(family.optionlets[t].source == "initialized" for t in (tlo, thi))
    return TargetVolatility(
        **result,
        status="available",
        message=(
            "Volatilité normale interpolée linéairement en strike et en temps de fixing, sans SABR. "
            + (
                "Au moins un point utilisé provient de l’initialisation par la vol flat du premier cap."
                if initial
                else "Les points utilisés proviennent du stripping."
            )
        ),
    )


def prepare_stripping(request: OptionletRequest, market_store, curve_store) -> StrippingResult:
    preparation = prepare_optionlet(request, market_store, curve_store)
    context = loaded_context(market_store, curve_store)
    data = market_store.get_capfloor_surface("EUR")
    if data is None:
        raise ValueError("Importer les volatilités normales de caps EUR dans Market Data.")
    if data.currency != "EUR" or data.valuation_date != request.valuation_date:
        raise ValueError(
            "La date et la devise des cotations de caps doivent correspondre aux courbes."
        )
    strikes, groups = _validated_quotes(data)
    if (request.stripping_method == "block_bootstrap" and request.include_atm_caps
            and data.atm_vols is not None
            and (len(data.atm_vols) != len(data.expiries)
                 or any(not isfinite(v) or v < 0 for v in data.atm_vols))):
        raise ValueError("Dimensions ou valeurs des cotations ATM invalides ; zéro signifie uniquement une cotation absente.")
    if request.stripping_method == "pchip" and request.include_atm_caps:
        raise ValueError("Le mode historique PCHIP n’incorpore pas les caps ATM : désactiver include_atm_caps ou choisir le mode par blocs.")
    for tenor in groups:
        if not any(c.key == tenor and c.current for c in context.curves):
            raise ValueError(f"Construire une courbe EURIBOR {tenor} à jour avant le stripping.")
    curve_set = curve_store.get_curve_set("EUR")
    with ql_evaluation_date(request.valuation_date):
        valuation = date_to_ql(request.valuation_date)
        spot = ql.TARGET().advance(valuation, 2, ql.Days)
        if request.stripping_method == "pchip":
            families = [
                _build_family(valuation, spot, tenor, groups[tenor], strikes, request.notional, curve_set)
                for tenor in ("3M", "6M") if tenor in groups
            ]
        else:
            from services.thomas_stripping_blocks import build_block_family

            atm_by_tenor = {tenor: {} for tenor in groups}
            for i, label in enumerate(data.expiries):
                month = _months(label)
                tenor = (data.quote_tenors[i] if data.quote_tenors is not None else data.index_tenor) or ("3M" if month <= 24 else "6M")
                atm_by_tenor[tenor.strip().upper()][month] = data.atm_vols[i] if data.atm_vols else None
            families = []
            for tenor in ("3M", "6M"):
                if tenor in groups:
                    families.append(build_block_family(
                        valuation, spot, tenor, groups[tenor], strikes, request.notional, curve_set,
                        atm_by_tenor[tenor], request.include_atm_caps,
                        source_3m=next((f for f in families if f.index_tenor == "3M"), None),
                        basis_moneyness=request.basis_moneyness,
                    ))
    warnings = [
        "La colonne ATM n’est pas utilisée : son strike varie avec la maturité. Le stripping porte sur les strikes absolus."
    ]
    if request.stripping_method == "block_bootstrap":
        warnings = [
            "Bootstrap par blocs d’échéances sur caps cotés (vols normales constantes par bloc) ; ni les données ni les paramètres d’un terminal externe ne sont reproduits automatiquement.",
            "La conversion 3M→6M transfère la smile normale selon une hypothèse de basis ; les caplets courts 6M restent des valeurs modélisées.",
        ]
        if request.include_atm_caps:
            warnings.append("Les strikes ATM des caps sont calculés avec les courbes et les flux retenus ; aucun strike ATM publié par le contributeur n’est fourni par cet import.")
            if any(a.status == "missing" for f in families for a in f.atm_caps):
                warnings.append("Certaines cotations ATM sont absentes : leurs diagnostics sont affichés et aucun point ATM fictif n’est ajouté.")
        else:
            warnings.append("Incorporation des caps ATM désactivée ; seules les cotations aux strikes absolus sont calibrées.")
    if any(f.issues for f in families):
        warnings.append(
            "Certains points ne sont pas strippables : consulter les diagnostics. Les trous de surface ne sont pas comblés automatiquement."
        )
    return StrippingResult(
        stripping_method=request.stripping_method,
        flat_vol_interpolation="pchip" if request.stripping_method == "pchip" else "none",
        basis_moneyness=request.basis_moneyness,
        include_atm_caps=request.include_atm_caps,
        source_token=context.source_token,
        valuation_date=request.valuation_date,
        notional=request.notional,
        spot_date=ql_to_date(spot),
        families=families,
        target=_query_target(request, preparation, families),
        warnings=warnings,
        conventions=[
            "EURIBOR 3M et 6M : familles natives séparées, aucune conversion de cotation.",
            "Spot = valorisation +2 jours ouvrés TARGET. Premier coupon spot déjà fixé exclu des caps de marché.",
            "Échéanciers emboîtés sur une ancre contractuelle commune ; recul par tenor, règle de fin de mois du spot, ajustement Modified Following appliqué une seule fois.",
            "Fixing = début −2 jours ouvrés TARGET ; paiement en fin de période, sans délai ; intérêts ACT/360, temps optionnel ACT/365F.",
            "Forward calculé sur la période native de l’indice EURIBOR. Son échéance peut différer de la fin du coupon après ajustement calendaire ; les deux dates sont affichées.",
            "Initialisation : même volatilité normale que le premier cap coté, pour tous ses caplets, à strike fixé.",
            "Interpolation PCHIP des vols flat à partir des seuls caps cotés, en temps réel ACT/365F de paiement ajusté, séparément pour chaque strike et sans extrapolation. Avec deux cotations, PCHIP se réduit à une droite.",
            "Différences de prix, inversion Bachelier par valeur temps et contrôle des prix reconstruits ; aucun point invalide remplacé par zéro.",
        ] if request.stripping_method == "pchip" else [
            "Spot = valorisation +2 jours ouvrés TARGET. Premier coupon spot déjà fixé exclu des caps de marché.",
            "Échéanciers emboîtés sur une ancre commune ; Modified Following, règle de fin de mois. Fixing début−2 jours ouvrés TARGET, paiement en fin, ACT/360 ; temps optionnel ACT/365F.",
            "Stripping aux seules maturités de caps cotées : une volatilité normale commune par bloc résiduel et par strike, obtenue par inversion du prix agrégé Bachelier.",
            "Premier cap 3M : volatilité flat de ce cap pour ses caplets ; aucune initialisation indépendante par la flat du premier cap 6M.",
            "Raccord 3M→6M jusqu’au début contractuel 2Y inclus : smiles normales transportées ; le fixing 2Y prolonge explicitement à plat la dernière smile 3M native.",
            "Les forwards 3M et 6M de conversion sont calculés aux dates exactes ; moneyness et transport temporel sont indiqués dans la provenance de chaque caplet.",
            "Smiles PWL en volatilité normale. Ailes : linéaire sur un intervalle de strike adjacent puis plate, choix de complément Thomas explicite ; aucune volatilité invalide remplacée par zéro.",
            "ATM activé : prix du cap ATM à son taux par, anciens caplets évalués dans leurs smiles enrichies, volatilité du bloc résiduel calibrée puis ajoutée à ce strike. Les strikes absolus restent inchangés.",
            "Tous les caps absolus et ATM incorporés sont repricés avec les smiles finales ; SABR constitue une approximation ultérieure séparée.",
        ],
    )
