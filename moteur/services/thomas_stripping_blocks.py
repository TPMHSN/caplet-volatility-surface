"""Thomas' documented block bootstrap and explicit normal 3M/6M basis assumption.

This is a block bootstrap over quoted-cap expiry blocks with constant per-block
normal vols and an explicit 3M/6M basis transfer, not a claim to reproduce any
vendor's proprietary terminal configuration. Original cap prices are
calibration targets; no fictitious flat-cap quotes are introduced.
"""

from bisect import bisect_left
from collections.abc import Callable
from math import fsum, isfinite

from scipy.optimize import brentq

from config import get_index
from curves.helpers import date_to_ql
from services.thomas_black import build_black_surface
from services.thomas_stripping import _bracket, _intrinsic, _label, _schedule, _time_value
from services.thomas_stripping_schemas import (
    ATMCapDiagnostic,
    CapNode,
    OptionletNode,
    StrippingBlock,
    StrippingFamily,
    StrippingIssue,
)


def _smile_value(strikes, vols, strike):
    """PWL normal smile; extend one adjacent interval, then keep the wing flat.

    The wing cutoff is OUR explicit completion convention: piecewise-linear/flat
    wing extrapolation is a common convention that does not by itself fix a
    numerical cutoff.
    Missing points and nonpositive extrapolated vols are never replaced by zero.
    """
    if not strikes or len(strikes) != len(vols):
        raise ValueError("Smile absente ou dimensions incohérentes.")
    note = None
    query = strike
    if len(strikes) == 1:
        if abs(strike - strikes[0]) > 1e-12:
            raise ValueError("Une seule cotation ne définit pas les ailes de la smile.")
        lo = hi = 0
        weight = 0.0
    elif strike < strikes[0]:
        query = max(strike, 2 * strikes[0] - strikes[1])
        lo, hi = 0, 1
        weight = (query - strikes[lo]) / (strikes[hi] - strikes[lo])
        note = "Aile basse : extension linéaire sur un intervalle de strike, puis plate."
    elif strike > strikes[-1]:
        query = min(strike, 2 * strikes[-1] - strikes[-2])
        lo, hi = len(strikes) - 2, len(strikes) - 1
        weight = (query - strikes[lo]) / (strikes[hi] - strikes[lo])
        note = "Aile haute : extension linéaire sur un intervalle de strike, puis plate."
    else:
        lo, hi, weight = _bracket(strikes, strike)
    if vols[lo] is None or vols[hi] is None:
        raise ValueError("Un point nécessaire de la smile est indisponible.")
    value = (1 - weight) * vols[lo] + weight * vols[hi]
    if not isfinite(value) or value <= 0:
        raise ValueError("La smile donne une volatilité non positive ou non finie ; aucune correction automatique.")
    return value, note


def _smile_vol(node, strike):
    return _smile_value(node.smile_strikes, node.smile_volatilities, strike)[0]


def _black_surface_from_smiles(nodes):
    """Retain every enriched ATM/converted knot in the displayed Black mesh.

    The rectangular cap-strike matrix is only a diagnostic of the original
    absolute quotes; it is not the enriched smile used for pricing. A shared
    union grid retains the additional kinks. Do not create new wing values just
    for display: each row stops at its own explicitly constructed smile domain.
    """
    grid = sorted({strike for node in nodes for strike in node.smile_strikes})
    normal_rows = []
    for node in nodes:
        values = []
        for strike in grid:
            if not node.smile_strikes or strike < node.smile_strikes[0] or strike > node.smile_strikes[-1]:
                values.append(None)
                continue
            try:
                values.append(_smile_vol(node, strike))
            except (ValueError, RuntimeError):
                values.append(None)
        normal_rows.append(values)
    return build_black_surface(grid, [o.fixing_date for o in nodes],
        [o.expiry_years for o in nodes], [o.forward_rate for o in nodes], normal_rows)


def _block_vol(nodes, strike, target_time_value, guess):
    if not nodes:
        raise ValueError("Aucun caplet résiduel à calibrer.")
    if not isfinite(target_time_value) or target_time_value <= 0:
        raise ValueError(
            "Valeur temps résiduelle négative ou numériquement nulle : bloc non identifiable ; aucune volatilité nulle imposée."
        )

    def relative_price(vol):
        return fsum(_time_value(o, strike, vol) for o in nodes) / target_time_value - 1

    high = max(guess, 1e-8)
    for _ in range(64):
        if relative_price(high) >= 0:
            break
        high *= 2
    else:
        raise ValueError("Impossible d’encadrer la volatilité normale du bloc.")
    sigma = brentq(relative_price, 0.0, high, xtol=1e-15, rtol=1e-13)
    if sigma <= 0 or not isfinite(sigma) or abs(relative_price(sigma)) > 1e-8:
        raise ValueError("Inversion du bloc insuffisamment précise, notamment dans les ailes.")
    return sigma


def _forward_3m_at_fixing(node, curve_set):
    index = get_index("EUR", "3M")
    start = index.valueDate(date_to_ql(node.fixing_date))
    end = index.maturityDate(start)
    projection = curve_set.get_forward_curve("3M")
    if start < projection.referenceDate() or end > projection.maxDate():
        raise ValueError("Forward 3M de conversion hors du domaine de la courbe.")
    delta = index.dayCounter().yearFraction(start, end)
    return (projection.discount(start) / projection.discount(end) - 1) / delta


def _convert_short_node(node, source_family, strikes, curve_set, moneyness, notional,
                        source_smile_interpolation="linear",
                        source_vol_provider: Callable[[OptionletNode, float], float] | None = None):
    source = source_family.optionlets
    times = [o.expiry_years for o in source]
    extrapolated = node.expiry_years > times[-1] + 1e-12
    if extrapolated:
        lo = hi = len(source) - 1
        weight = 0.0
    else:
        lo, hi, weight = _bracket(times, node.expiry_years)
    left, right = source[lo], source[hi]
    forward_3m = _forward_3m_at_fixing(node, curve_set)
    node.source = "converted"
    node.conversion_source_fixings = list(dict.fromkeys([left.fixing_date, right.fixing_date]))
    node.conversion_source_forward = forward_3m
    node.conversion_forward_spread = node.forward_rate - forward_3m
    node.conversion_expiry_extrapolated = extrapolated
    node.conversion_notes = [
        "Transfert de la smile NORMALE 3M vers 6M : hypothèse de modèle, aucune cotation 6M directe.",
        (
            "Même moneyness absolue : K6−F6 = K3−F3. En temps, chaque smile source est aussi interrogée à la même moneyness."
            if moneyness else "Même strike absolu : transfert sans ajustement de moneyness."
        ),
    ]
    if extrapolated:
        node.conversion_notes.append(
            "Fixing 2Y complété par extrapolation plate de la dernière smile 3M native (~1.75Y), jusqu’à la frontière contractuelle 2Y seulement."
        )

    def transported(source_node, strike):
        # At an interpolated/extrapolated expiry, first transport at fixed
        # moneyness to F3(target), then apply F6(target)-F3(target). Algebraically
        # the combined strike query is K6-F6(target)+F3(source).
        source_strike = (
            strike - node.forward_rate + source_node.forward_rate if moneyness else strike
        )
        if source_vol_provider is not None:
            value = float(source_vol_provider(source_node, float(source_strike)))
            if not isfinite(value) or not 1e-10 <= value <= 1.0:
                raise ValueError("Smile source continue hors des bornes de volatilité.")
            return value, None
        if source_smile_interpolation == "natural_cubic" and source_node.smile_strikes[0] <= source_strike <= source_node.smile_strikes[-1]:
            from services.normal_cev_smiles import normal_strike_vol
            return normal_strike_vol(source_node.smile_strikes, source_node.smile_volatilities, source_strike), None
        return _smile_value(source_node.smile_strikes, source_node.smile_volatilities, source_strike)

    def converted(strike):
        a, note_a = transported(left, strike)
        b, note_b = transported(right, strike)
        for note in (note_a, note_b):
            if note and note not in node.conversion_notes:
                node.conversion_notes.append(note)
        return (1 - weight) * a + weight * b

    shifted_knots = sorted({
        k + node.forward_rate - o.forward_rate if moneyness else k
        for o in (left, right) for k in o.smile_strikes
    })
    # Include absolute cap strikes as knots too: this records any wing completion
    # used by the subsequent 6M bootstrap and retains it for all future queries.
    node.smile_strikes = sorted(set(shifted_knots + strikes))
    node.smile_volatilities = []
    for strike in node.smile_strikes:
        try:
            node.smile_volatilities.append(converted(strike))
        except ValueError as exc:
            node.smile_volatilities.append(None)
            if str(exc) not in node.conversion_notes:
                node.conversion_notes.append(str(exc))
    node.smile_sources = ["basis_converted"] * len(node.smile_strikes)
    for j, strike in enumerate(strikes):
        try:
            vol = _smile_vol(node, strike)
            node.volatilities[j] = vol
            node.prices[j] = notional * (_intrinsic(node, strike) + _time_value(node, strike, vol))
        except ValueError:
            node.volatilities[j] = None


def _insert_atm(nodes, strike, vol):
    # Check every node before mutating any, preserving the absolute cap quotes.
    positions = []
    for node in nodes:
        pos = bisect_left(node.smile_strikes, strike)
        if pos > 0 and abs(node.smile_strikes[pos - 1] - strike) <= 1e-12:
            pos -= 1
        existing = pos < len(node.smile_strikes) and abs(node.smile_strikes[pos] - strike) <= 1e-12
        if existing:
            old = node.smile_volatilities[pos]
            if old is None or abs(old - vol) > 1e-9:
                raise ValueError("La cotation ATM contredit une cotation au même strike absolu ; le nœud absolu est conservé.")
        positions.append((node, pos, existing))
    for node, pos, existing in positions:
        if not existing:
            node.smile_strikes.insert(pos, strike)
            node.smile_volatilities.insert(pos, vol)
            node.smile_sources.insert(pos, "atm_cap")


def _atm_step(month, included, prior, block, flat_vol, notional):
    annuity = fsum(o.accrual_fraction * o.discount_factor for o in included)
    strike = fsum(o.accrual_fraction * o.discount_factor * o.forward_rate for o in included) / annuity
    result = ATMCapDiagnostic(
        expiry=_label(month), maturity_months=month, strike=strike,
        flat_vol=flat_vol if flat_vol and flat_vol > 0 else None,
        market_price=None, repriced_price=None, repricing_error=None, block_vol=None,
        block_optionlets_count=len(block), status="missing", message="Cotation ATM absente ; aucun point ATM ajouté.",
    )
    if flat_vol is None or flat_vol == 0:
        return result
    try:
        if not isfinite(flat_vol) or flat_vol < 0:
            raise ValueError("Cotation ATM invalide.")
        intrinsic = fsum(_intrinsic(o, strike) for o in included)
        market_tv = fsum(_time_value(o, strike, flat_vol) for o in included)
        result.market_price = notional * (intrinsic + market_tv)
        prior_tv = fsum(_time_value(o, strike, _smile_vol(o, strike)) for o in prior)
        sigma = _block_vol(block, strike, market_tv - prior_tv, flat_vol)
        _insert_atm(block, strike, sigma)
        result.block_vol = sigma
        result.status = "included"
        result.message = "Strike ATM du cap calculé comme son taux par ; seul le bloc résiduel reçoit le point calibré à ce strike, qui n’est pas l’ATM de chaque caplet."
    except (ValueError, RuntimeError) as exc:
        result.status = "unavailable"
        result.message = str(exc)
    return result


def build_block_family(valuation, spot, tenor, rows, strikes, notional, curve_set,
                       atm_quotes, include_atm, source_3m=None, basis_moneyness=True, build_black=True,
                       source_smile_interpolation="linear",
                       source_vol_provider: Callable[[OptionletNode, float], float] | None = None):
    if source_smile_interpolation not in ("linear", "natural_cubic"):
        raise ValueError("Interpolation de smile source inconnue.")
    first, last = rows[0][0], rows[-1][0]
    options = _schedule(valuation, spot, tenor, last, first, curve_set, strikes)
    for node in options:
        node.prices = [None] * len(strikes)
    issues, caps, blocks, atm_caps = [], [], [], []
    converted = []
    if tenor == "6M":
        if source_3m is None or source_3m.caps[-1].maturity_months < 24:
            raise ValueError("Le raccord 3M→6M exige des cotations natives 3M couvrant le cap 2Y.")
        if first <= 24:
            raise ValueError("Cette méthode de raccord EUR attend le premier cap natif 6M après 2Y.")
        # Contractual start-month boundary, not an ACT/365F float comparison.
        converted = [o for o in options if o.cap_maturity_months - 6 <= 24]
        for node in converted:
            if source_vol_provider is None:
                _convert_short_node(node, source_3m, strikes, curve_set, basis_moneyness, notional, source_smile_interpolation)
            else:
                _convert_short_node(node, source_3m, strikes, curve_set, basis_moneyness, notional,
                                    source_smile_interpolation, source_vol_provider=source_vol_provider)
            for strike, vol in zip(strikes, node.volatilities):
                if vol is None:
                    issues.append(StrippingIssue(strike=strike, cap_maturity_months=node.cap_maturity_months,
                        code="basis_conversion_unavailable", message="Conversion 3M→6M impossible à ce strike ; voir la provenance du caplet."))

    previous_count = len(converted)
    for month, flat in rows:
        included = [o for o in options if o.cap_maturity_months <= month]
        prior, block = included[:previous_count], included[previous_count:]
        if not block:
            raise ValueError("Le bloc coté ne contient aucun nouveau caplet après le raccord.")
        block_vols, residuals, prices = [], [], []
        for j, strike in enumerate(strikes):
            market_tv = fsum(_time_value(o, strike, flat[j]) for o in included)
            prices.append(notional * (fsum(_intrinsic(o, strike) for o in included) + market_tv))
            residual = None
            sigma = None
            try:
                prior_tv = fsum(_time_value(o, strike, _smile_vol(o, strike)) for o in prior)
                residual = market_tv - prior_tv
                sigma = _block_vol(block, strike, residual, flat[j])
            except (ValueError, RuntimeError) as exc:
                issues.append(StrippingIssue(strike=strike, cap_maturity_months=month,
                    code="invalid_block_price", message=str(exc)))
            block_vols.append(sigma)
            residuals.append(notional * residual if residual is not None else None)
            for node in block:
                node.volatilities[j] = sigma
                if sigma is not None:
                    node.prices[j] = notional * (_intrinsic(node, strike) + _time_value(node, strike, sigma))
        for node in block:
            node.source = "initialized" if tenor == "3M" and not prior else "stripped"
            node.block_expiry = _label(month)
            node.smile_strikes = list(strikes)
            node.smile_volatilities = list(node.volatilities)
            node.smile_sources = ["absolute_cap"] * len(strikes)
        if include_atm:
            diagnostic = _atm_step(month, included, prior, block, atm_quotes.get(month), notional)
            atm_caps.append(diagnostic)
            if diagnostic.status == "unavailable":
                issues.append(StrippingIssue(strike=diagnostic.strike, cap_maturity_months=month,
                    code="atm_cap_unavailable", message=diagnostic.message))
        blocks.append(StrippingBlock(expiry=_label(month), maturity_months=month,
            first_fixing_date=block[0].fixing_date, last_fixing_date=block[-1].fixing_date,
            optionlets_count=len(block), prior_optionlets_count=len(prior), strikes=strikes,
            volatilities=block_vols, residual_time_values=residuals))
        caps.append(CapNode(maturity_months=month, expiry=_label(month), maturity_date=included[-1].payment_date,
            source="market", optionlets_count=len(included), flat_vols=flat, prices=prices,
            repriced_prices=[None] * len(strikes), repricing_errors=[None] * len(strikes)))
        previous_count = len(included)

    # Reprice from the final enriched smiles, not from intermediate bootstrap sums.
    for cap in caps:
        included = options[:cap.optionlets_count]
        for j, strike in enumerate(strikes):
            try:
                price = notional * fsum(_intrinsic(o, strike) + _time_value(o, strike, _smile_vol(o, strike)) for o in included)
                cap.repriced_prices[j] = price
                cap.repricing_errors[j] = price - cap.prices[j]
                if abs(cap.repricing_errors[j]) > notional * 1e-10:
                    issues.append(StrippingIssue(strike=strike, cap_maturity_months=cap.maturity_months,
                        code="repricing_error", message="Le repricing final du cap absolu dépasse la tolérance."))
            except (ValueError, RuntimeError):
                pass
    for atm in atm_caps:
        if atm.status != "included":
            continue
        included = [o for o in options if o.cap_maturity_months <= atm.maturity_months]
        try:
            price = notional * fsum(_intrinsic(o, atm.strike) + _time_value(o, atm.strike, _smile_vol(o, atm.strike)) for o in included)
            atm.repriced_price = price
            atm.repricing_error = price - atm.market_price
            if abs(atm.repricing_error) > notional * 1e-10:
                raise ValueError("Le repricing final du cap ATM dépasse la tolérance.")
        except (ValueError, RuntimeError) as exc:
            atm.status = "unavailable"
            atm.message = str(exc)
            issues.append(StrippingIssue(strike=atm.strike, cap_maturity_months=atm.maturity_months,
                code="atm_repricing_error", message=str(exc)))
    errors = [abs(e) for cap in caps for e in cap.repricing_errors if e is not None]
    errors += [abs(a.repricing_error) for a in atm_caps if a.repricing_error is not None]
    reported_prices = [p for cap in caps for p in cap.prices + cap.repriced_prices]
    reported_prices += [p for node in options for p in node.prices]
    reported_prices += [p for b in blocks for p in b.residual_time_values]
    reported_prices += [p for a in atm_caps for p in (a.market_price, a.repriced_price)]
    if any(p is not None and not isfinite(p) for p in reported_prices + errors):
        raise ValueError("Notionnel trop grand ou calcul de prix non fini.")
    notes = []
    if tenor == "6M":
        notes = [
            "Raccord jusqu’au début contractuel 2Y inclus : quatre premiers caplets 6M issus du modèle 3M ; le fixing 2Y utilise une extrapolation plate explicite de la dernière smile 3M.",
            "Le transfert de toute la smile normale (même moneyness ou même strike) est une hypothèse implémentée ; il ne prouve pas les réglages actuels d’un terminal de marché externe.",
            "Hors strikes disponibles : extension PWL sur un intervalle adjacent, puis plate ; cette borne est un choix explicite Thomas, non spécifié numériquement par une source externe.",
        ]
    return StrippingFamily(index_tenor=tenor, status="partial" if issues else "ready",
        strikes=strikes, first_quoted_expiry=_label(first), last_quoted_expiry=_label(last),
        caps=caps, optionlets=options, issues=issues, market_count=len(caps),
        interpolated_count=0, initialized_count=0,
        max_abs_repricing_error=max(errors) if errors else None,
        blocks=blocks, atm_caps=atm_caps, conversion_notes=notes,
        conversion_coverage="two_year_expiry_flat_extension" if tenor == "6M" else None,
        black_surface=_black_surface_from_smiles(options) if build_black else None)


def query_block_vol(family, expiry, strike, forward):
    """Query between block nodes at constant absolute moneyness (or fixed K).

    The caller chooses the forward: pass None for the fixed-strike alternative.
    At native grid fixings this exactly retains the calibrated PWL smile.
    """
    lo, hi, weight = _bracket([o.expiry_years for o in family.optionlets], expiry)
    values = []
    for i in (lo, hi):
        node = family.optionlets[i]
        k = strike if forward is None else strike - forward + node.forward_rate
        values.append(_smile_vol(node, k))
    return (1 - weight) * values[0] + weight * values[1]
