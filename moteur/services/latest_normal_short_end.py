"""Price-preserving short-end source and joint EUR 3M/6M boundary policy.

Normal volatilities and rates are decimal rates; all prices include the supplied
notional. The source is flat in its first quoted block, then linear in fixing
expiry between the last optionlets of consecutive native cap pillars. The 2Y
6M fixing and 2.5Y endpoint are solved together against the first native 3Y cap.
No quote or original family is mutated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from math import fsum, isfinite

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, least_squares

from services.normal_cev_smiles import normal_strike_vol
from services.thomas_stripping import _intrinsic, _time_value
from services.thomas_stripping_blocks import _smile_value
from services.thomas_stripping_schemas import OptionletNode, StrippingFamily

FloatArray = npt.NDArray[np.float64]
QuoteRows = Sequence[tuple[int, Sequence[float]]]
_MIN_VOL = 1e-10
_MAX_VOL = 1.0
_PRICE_TOLERANCE_PER_NOTIONAL = 1e-10


def _notional(value: float) -> float:
    if not isfinite(value) or value <= 0:
        raise ValueError("Le notionnel du raccord doit être fini et strictement positif.")
    return float(value)


def _vols(values: Sequence[float | None], size: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (size,)
        or np.any(~np.isfinite(array))
        or np.any(array < _MIN_VOL)
        or np.any(array > _MAX_VOL)
    ):
        raise ValueError(f"{name} : volatilités normales absentes ou hors bornes.")
    return array


def _validate_family(family: StrippingFamily, tenor: str) -> FloatArray:
    k = np.asarray(family.strikes, dtype=float)
    if (
        family.index_tenor != tenor
        or family.status != "ready"
        or family.atm_caps
        or len(k) < 2
        or np.any(~np.isfinite(k))
        or np.any(np.diff(k) <= 0)
        or not family.optionlets
        or not family.caps
        or len(family.blocks) != len(family.caps)
    ):
        raise ValueError("Famille native incompatible avec la politique de raccord court.")
    previous_expiry = 0.0
    previous_month = 0
    for node in family.optionlets:
        if (
            not all(
                isfinite(x)
                for x in (
                    node.expiry_years,
                    node.forward_rate,
                    node.discount_factor,
                    node.accrual_fraction,
                )
            )
            or node.expiry_years <= previous_expiry
            or node.cap_maturity_months <= previous_month
            or node.discount_factor <= 0
            or node.accrual_fraction <= 0
            or node.start_date >= node.end_date
        ):
            raise ValueError("Calendrier ou données de projection du raccord invalides.")
        _vols(node.volatilities, len(k), "Caplet natif")
        previous_expiry, previous_month = node.expiry_years, node.cap_maturity_months
    previous_month = 0
    for cap, block in zip(family.caps, family.blocks, strict=True):
        included = [n for n in family.optionlets if n.cap_maturity_months <= cap.maturity_months]
        if (
            cap.maturity_months <= previous_month
            or block.maturity_months != cap.maturity_months
            or len(included) != cap.optionlets_count
            or not included
            or included[-1].cap_maturity_months != cap.maturity_months
            or len(cap.prices) != len(k)
            or not all(isfinite(p) for p in cap.prices)
        ):
            raise ValueError("Piliers de caps incomplets ou incohérents.")
        _vols(cap.flat_vols, len(k), "Quote native")
        previous_month = cap.maturity_months
    if family.caps[-1].optionlets_count != len(family.optionlets):
        raise ValueError("Caplets hors de la couverture des caps natifs.")
    return k


def _price(node: OptionletNode, strike: float, vol: float) -> float:
    return float(_intrinsic(node, strike) + _time_value(node, strike, vol))


def _market_prices_match(family: StrippingFamily, notional: float) -> None:
    tolerance = notional * _PRICE_TOLERANCE_PER_NOTIONAL
    for cap in family.caps:
        included = family.optionlets[: cap.optionlets_count]
        expected = [
            notional * fsum(_price(n, k, cap.flat_vols[j]) for n in included)
            for j, k in enumerate(family.strikes)
        ]
        if any(abs(a - b) > tolerance for a, b in zip(expected, cap.prices, strict=True)):
            raise ValueError("Prix, quotes natives et notionnel du raccord incohérents.")


def _positive_root(residual: Callable[[float], float], initial_upper: float, name: str) -> float:
    """Closed positive-volatility bracket with finite expansion and iteration limits."""
    lower = _MIN_VOL
    low_value = float(residual(lower))
    if not isfinite(low_value) or low_value >= 0:
        raise ValueError(f"{name} : aucun endpoint strictement positif admissible.")
    upper = min(_MAX_VOL, max(initial_upper, 2 * lower))
    for _ in range(64):
        upper_value = float(residual(upper))
        if not isfinite(upper_value):
            raise ValueError(f"{name} : résidu de prix non fini.")
        if upper_value >= 0:
            break
        if upper >= _MAX_VOL:
            raise ValueError(f"{name} : racine non encadrée dans les bornes de volatilité.")
        upper = min(2 * upper, _MAX_VOL)
    else:
        raise ValueError(f"{name} : recherche de borne non convergée.")
    result = float(brentq(residual, lower, upper, xtol=1e-15, rtol=1e-13, maxiter=128))
    if not isfinite(result) or abs(float(residual(result))) > 1e-8:
        raise ValueError(f"{name} : inversion de prix insuffisamment précise.")
    return result


def _set_node_vols(
    node: OptionletNode, strikes: Sequence[float], values: Sequence[float], notional: float
) -> None:
    vols = _vols(values, len(strikes), "Smile recalibrée")
    node.volatilities = [float(v) for v in vols]
    node.smile_strikes = list(strikes)
    node.smile_volatilities = list(node.volatilities)
    node.smile_sources = [
        "basis_converted" if node.source == "converted" else "absolute_cap"
    ] * len(strikes)
    node.prices = [notional * _price(node, k, float(v)) for k, v in zip(strikes, vols, strict=True)]


def _refresh_caps(family: StrippingFamily, notional: float) -> None:
    errors: list[float] = []
    for cap in family.caps:
        included = family.optionlets[: cap.optionlets_count]
        cap.repriced_prices = [
            notional * fsum(_price(n, k, float(n.volatilities[j])) for n in included)
            for j, k in enumerate(family.strikes)
        ]
        cap.repricing_errors = [a - b for a, b in zip(cap.repriced_prices, cap.prices, strict=True)]
        errors.extend(abs(e) for e in cap.repricing_errors)
    if not errors or not all(isfinite(e) for e in errors):
        raise ValueError("Repricing natif du raccord non fini ou incomplet.")
    if max(errors) > notional * _PRICE_TOLERANCE_PER_NOTIONAL:
        raise ValueError("Le raccord ne préserve pas les prix des caps natifs.")
    family.max_abs_repricing_error = max(errors)


def supports_short_end_policy(three: StrippingFamily, rows6m: QuoteRows) -> bool:
    """Structural gate only; unsupported layouts keep their explicitly separate policy."""
    return bool(
        three.index_tenor == "3M"
        and three.status == "ready"
        and not three.atm_caps
        and three.caps
        and three.caps[-1].maturity_months == 24
        and [n.cap_maturity_months for n in three.optionlets] == list(range(6, 25, 3))
        and rows6m
        and rows6m[0][0] == 36
        and all(m >= 36 and m % 6 == 0 for m, _ in rows6m)
        and all(a[0] < b[0] for a, b in pairwise(rows6m))
        and all(len(v) == len(three.strikes) for _, v in rows6m)
    )


def prepare_short_end_source(
    three: StrippingFamily, rows3m: QuoteRows, notional: float = 1.0
) -> StrippingFamily:
    """Recalibrate fixing-linear endpoint vols while preserving all native 3M cap prices."""
    notional = _notional(notional)
    strikes = _validate_family(three, "3M")
    if [month for month, _ in rows3m] != [cap.maturity_months for cap in three.caps]:
        raise ValueError("Les quotes 3M doivent couvrir exactement les piliers natifs.")
    for cap, (_, values) in zip(three.caps, rows3m, strict=True):
        flat = _vols(values, len(strikes), "Quote 3M")
        if np.max(abs(flat - np.asarray(cap.flat_vols))) > 1e-12:
            raise ValueError("Les quotes 3M ne correspondent pas à la famille native.")
    _market_prices_match(three, notional)
    result = three.model_copy(deep=True)
    count = 0
    for (month, values), block_report in zip(rows3m, result.blocks, strict=True):
        flat = _vols(values, len(strikes), "Quote 3M")
        included = [n for n in result.optionlets if n.cap_maturity_months <= month]
        prior, block = included[:count], included[count:]
        if not block:
            raise ValueError("Le pilier 3M ne contient aucun nouveau caplet.")
        if not prior:
            for node in block:
                _set_node_vols(node, result.strikes, flat, notional)
        else:
            left_time, right_time = prior[-1].expiry_years, block[-1].expiry_years
            if right_time <= left_time:
                raise ValueError("Expirations de fixing 3M non croissantes.")
            weights = [(n.expiry_years - left_time) / (right_time - left_time) for n in block]
            if not all(0 < w <= 1 for w in weights):
                raise ValueError("Interpolation 3M hors du domaine des piliers.")
            for j, strike in enumerate(strikes):
                target = fsum(
                    _time_value(n, float(strike), float(flat[j])) for n in included
                ) - fsum(_time_value(n, float(strike), float(n.volatilities[j])) for n in prior)
                if not isfinite(target) or target <= 0:
                    raise ValueError("Résidu de valeur temps 3M incompatible avec le préfixe.")
                left_vol = float(prior[-1].volatilities[j])

                def residual(
                    right_vol: float,
                    *,
                    strike: float = float(strike),
                    left_vol: float = left_vol,
                    block: Sequence[OptionletNode] = tuple(block),
                    weights: Sequence[float] = tuple(weights),
                    target: float = target,
                ) -> float:
                    return (
                        fsum(
                            _time_value(n, float(strike), (1 - w) * left_vol + w * right_vol)
                            for n, w in zip(block, weights, strict=True)
                        )
                        / target
                        - 1
                    )

                endpoint = _positive_root(residual, max(left_vol, float(flat[j])), "Pilier 3M")
                for node, weight in zip(block, weights, strict=True):
                    node.volatilities[j] = (1 - weight) * left_vol + weight * endpoint
            for node in block:
                _set_node_vols(
                    node, result.strikes, [float(v) for v in node.volatilities], notional
                )
        # The report retains a price-equivalent constant for the entire block,
        # not the endpoint of the new linear trajectory.
        block_report.residual_time_values = [
            notional * fsum(_time_value(n, k, float(n.volatilities[j])) for n in block)
            for j, k in enumerate(result.strikes)
        ]
        count = len(included)
    _refresh_caps(result, notional)
    result.conversion_notes.append(
        "Source du raccord court : premier bloc 3M plat puis volatilité normale linéaire "
        "en expiration du fixing entre les derniers caplets des caps natifs ; prix natifs conservés."
    )
    return result


def _smile(strikes: Sequence[float], vols: Sequence[float], query: float) -> float:
    """Retain the existing natural-cubic interior and explicit bounded PWL/flat wings."""
    if strikes[0] <= query <= strikes[-1]:
        result = float(normal_strike_vol(strikes, vols, query))
    else:
        result = float(_smile_value(strikes, vols, query)[0])
    if not isfinite(result) or not _MIN_VOL <= result <= _MAX_VOL:
        raise ValueError("Smile du raccord hors des bornes de volatilité.")
    return result


def _without_flat_boundary_notes(notes: Sequence[str]) -> list[str]:
    return [
        note
        for note in notes
        if not ("2Y" in note and ("plat" in note.lower() or "extrapol" in note.lower()))
    ]


def couple_first_six_month_pillar(
    six: StrippingFamily, source: StrippingFamily, notional: float = 1.0,
    source_vol_provider: Callable[[OptionletNode, float], float] | None = None,
) -> StrippingFamily:
    """Jointly solve the 2Y bridge vol and 2.5Y native vol under the 3Y cap constraint."""
    notional = _notional(notional)
    strikes = _validate_family(six, "6M")
    source_strikes = _validate_family(source, "3M")
    rows6m = [(cap.maturity_months, cap.flat_vols) for cap in six.caps]
    if (
        not supports_short_end_policy(source, rows6m)
        or not np.array_equal(strikes, source_strikes)
        or len(six.optionlets) < 5
        or [n.cap_maturity_months for n in six.optionlets[:5]] != [12, 18, 24, 30, 36]
        or [n.source for n in six.optionlets[:5]] != ["converted"] * 4 + ["stripped"]
        or six.caps[0].optionlets_count != 5
        or six.blocks[0].prior_optionlets_count != 4
        or six.blocks[0].optionlets_count != 1
    ):
        raise ValueError("Le raccord conjoint exige la structure native 3M 2Y / 6M 3Y.")
    _market_prices_match(six, notional)
    result = six.model_copy(deep=True)
    left = source.optionlets[-1]
    middle, endpoint = result.optionlets[3:5]
    if not left.expiry_years < middle.expiry_years < endpoint.expiry_years:
        raise ValueError("Le fixing de raccord n'est pas encadré par les deux ancres.")
    weight = (middle.expiry_years - left.expiry_years) / (endpoint.expiry_years - left.expiry_years)
    source_vols = _vols(left.volatilities, len(strikes), "Dernière ancre 3M")
    anchors = np.asarray(
        [
            (float(source_vol_provider(left, float(k - middle.forward_rate + left.forward_rate)))
             if source_vol_provider is not None else
             _smile(source.strikes, source_vols, float(k - middle.forward_rate + left.forward_rate)))
            for k in strikes
        ]
    )
    if source_vol_provider is not None:
        _vols(anchors, len(strikes), "Ancre source continue")
    first_cap = result.caps[0]
    included = result.optionlets[:5]
    prior = included[:3]
    residual_tv = np.asarray(
        [
            fsum(_time_value(n, float(k), first_cap.flat_vols[j]) for n in included)
            - fsum(_time_value(n, float(k), float(n.volatilities[j])) for n in prior)
            for j, k in enumerate(strikes)
        ]
    )
    if np.any(~np.isfinite(residual_tv)) or np.any(residual_tv <= 0):
        raise ValueError("Valeur temps du cap 3Y incompatible avec le préfixe 6M.")

    def midpoint_vols(values: FloatArray) -> FloatArray:
        transported = np.asarray(
            [
                _smile(
                    result.strikes, values, float(k - middle.forward_rate + endpoint.forward_rate)
                )
                for k in strikes
            ]
        )
        return (1 - weight) * anchors + weight * transported

    def errors(values: FloatArray) -> FloatArray:
        try:
            midpoint = midpoint_vols(values)
            return np.asarray(
                [
                    (
                        _time_value(middle, float(k), float(vm))
                        + _time_value(endpoint, float(k), float(ve))
                    )
                    / target
                    - 1
                    for k, vm, ve, target in zip(
                        strikes, midpoint, values, residual_tv, strict=True
                    )
                ]
            )
        except (ValueError, OverflowError, FloatingPointError):
            return np.full(len(strikes), 1e6)

    initial = _vols(endpoint.volatilities, len(strikes), "Ancre initiale 6M")
    solved = least_squares(
        errors,
        np.clip(initial, 2 * _MIN_VOL, _MAX_VOL * (1 - 1e-8)),
        bounds=(_MIN_VOL, _MAX_VOL),
        max_nfev=2000,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    if not solved.success or np.any(~np.isfinite(solved.x)) or np.max(abs(errors(solved.x))) > 1e-9:
        raise ValueError("Calibration conjointe 2Y / 2,5Y impossible dans les bornes.")
    _set_node_vols(middle, result.strikes, midpoint_vols(solved.x), notional)
    _set_node_vols(endpoint, result.strikes, solved.x, notional)
    middle.conversion_expiry_extrapolated = False
    middle.conversion_source_fixings = [left.fixing_date, endpoint.fixing_date]
    middle.conversion_source_forward = left.forward_rate
    middle.conversion_forward_spread = middle.forward_rate - left.forward_rate
    middle.conversion_notes = _without_flat_boundary_notes(middle.conversion_notes) + [
        (
            "Fixing 2Y résolu conjointement avec le caplet 2,5Y : interpolation linéaire "
            "en expiration à même moneyness absolue sous contrainte du prix du cap natif 3Y."
        ),
        f"Ancres de fixing : 3M {left.fixing_date} et 6M {endpoint.fixing_date}.",
    ]
    first_block = result.blocks[0]
    first_block.volatilities = list(endpoint.volatilities)
    first_block.residual_time_values = [
        notional * _time_value(endpoint, k, float(v))
        for k, v in zip(result.strikes, endpoint.volatilities, strict=True)
    ]
    result.conversion_coverage = None
    result.conversion_notes = _without_flat_boundary_notes(result.conversion_notes) + [
        (
            "Raccord court conjoint : la source 3M linéaire en fixing et l'ancre native 6M 2,5Y "
            "encadrent le fixing 2Y ; le cap natif 3Y est repricé sans extrapolation plate."
        )
    ]
    _refresh_caps(result, notional)
    return result
