"""Numerical and source diagnostics for the latest normal model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import exp, fsum, isfinite, pi, sqrt
from typing import Literal

import QuantLib as ql
from scipy.special import ndtr

from curves.helpers import parse_tenor
from market_data.schemas import CapFloorVolData, ForwardMarketData
from services.latest_normal_schemas import (
    FlatEquivalentNormalVol,
    LatestNormalAtmUsage,
    LatestNormalSourceProvenance,
    NativeQuoteFitPoint,
    QuoteFitPremium,
    QuoteFitProductPair,
    QuoteFitStageSummary,
    QuoteTenorProvenance,
)

REFERENCE_NOTIONAL = 10_000_000.0
QUOTE_FIT_ABSOLUTE_TOLERANCE_EUR = 25.0
QUOTE_FIT_RELATIVE_TOLERANCE = 0.005


def _premium_fit(market_price: float, model_price: float) -> QuoteFitPremium:
    market_price = max(0.0, market_price)
    model_price = max(0.0, model_price)
    residual = model_price - market_price
    market_reference = market_price * REFERENCE_NOTIONAL
    model_reference = model_price * REFERENCE_NOTIONAL
    tolerance = max(
        QUOTE_FIT_ABSOLUTE_TOLERANCE_EUR,
        QUOTE_FIT_RELATIVE_TOLERANCE * market_reference,
    )
    return QuoteFitPremium(
        market_price_unit=market_price,
        model_price_unit=model_price,
        residual_unit=residual,
        market_price_eur_10m=market_reference,
        model_price_eur_10m=model_reference,
        residual_eur_10m=residual * REFERENCE_NOTIONAL,
        tolerance_eur_10m=tolerance,
        passed=abs(residual * REFERENCE_NOTIONAL) <= tolerance,
    )


def quote_fit_product_pair(
    market_cap_price: float,
    model_cap_price: float,
    cap_floor_parity: float,
) -> QuoteFitProductPair:
    market_floor_price = market_cap_price - cap_floor_parity
    model_floor_price = model_cap_price - cap_floor_parity
    if market_floor_price < -1e-12 or model_floor_price < -1e-12:
        raise ValueError("Cap-floor parity produced a materially negative premium.")
    return QuoteFitProductPair(
        cap=_premium_fit(market_cap_price, model_cap_price),
        floor=_premium_fit(market_floor_price, model_floor_price),
    )


def quote_fit_point(
    *,
    expiry_months: int,
    strike: float,
    market_quote_normal_vol: float,
    market_cap_price: float,
    block_cap_price: float,
    quadratic_cap_price: float,
    final_cap_price: float,
    cap_floor_parity: float,
) -> NativeQuoteFitPoint:
    """Build separate cap and floor quote gates from one parity-consistent cap price."""
    values = (
        strike,
        market_quote_normal_vol,
        market_cap_price,
        block_cap_price,
        quadratic_cap_price,
        final_cap_price,
        cap_floor_parity,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("Native quote-fit inputs must be finite.")
    return NativeQuoteFitPoint(
        expiry_months=expiry_months,
        strike=strike,
        market_quote_normal_vol_bp=market_quote_normal_vol * 10_000.0,
        status="ready",
        block=quote_fit_product_pair(market_cap_price, block_cap_price, cap_floor_parity),
        quadratic=quote_fit_product_pair(market_cap_price, quadratic_cap_price, cap_floor_parity),
        final=quote_fit_product_pair(market_cap_price, final_cap_price, cap_floor_parity),
    )


def summarize_quote_fit(
    points: Sequence[NativeQuoteFitPoint],
    stage: Literal["block", "quadratic", "final"],
) -> QuoteFitStageSummary:
    pairs = [getattr(point, stage) for point in points if getattr(point, stage) is not None]
    if not pairs:
        return QuoteFitStageSummary(
            cap_max_abs_residual_eur_10m=None,
            floor_max_abs_residual_eur_10m=None,
            cap_passed=None,
            floor_passed=None,
            passed=None,
        )
    complete = len(pairs) == len(points)
    cap_passed = complete and all(pair.cap.passed for pair in pairs)
    floor_passed = complete and all(pair.floor.passed for pair in pairs)
    return QuoteFitStageSummary(
        cap_max_abs_residual_eur_10m=max(abs(pair.cap.residual_eur_10m) for pair in pairs),
        floor_max_abs_residual_eur_10m=max(abs(pair.floor.residual_eur_10m) for pair in pairs),
        cap_passed=cap_passed,
        floor_passed=floor_passed,
        passed=cap_passed and floor_passed,
    )


def _flat_price(
    product_type: Literal["cap", "floor"],
    strike: float,
    notional: float,
    rows: Sequence[Mapping[str, object]],
    sigma: float,
) -> float:
    sign = 1.0 if product_type == "cap" else -1.0
    values: list[float] = []
    for row in rows:
        expiry = float(row["expiry_years"])
        forward = float(row["forward_rate"])
        annuity = float(row["accrual_fraction"]) * float(row["discount_factor"])
        distance = sign * (forward - float(row.get("strike", strike)))
        std = sigma * sqrt(expiry)
        if std <= 0.0:
            option_value = max(distance, 0.0)
        else:
            z = distance / std
            option_value = distance * ndtr(z) + std * exp(-z * z / 2.0) / sqrt(2.0 * pi)
        values.append(float(row.get("notional", notional)) * annuity * option_value)
    return fsum(values)


def flat_equivalent_normal_vol(
    *,
    product_type: Literal["cap", "floor"],
    strike: float,
    notional: float,
    rows: Sequence[Mapping[str, object]],
    total_price: float,
) -> FlatEquivalentNormalVol:
    """Invert the trade price to one cap/floor normal vol at the actual strike."""
    future = [row for row in rows if row.get("status") == "future"]
    if not future:
        return FlatEquivalentNormalVol(
            status="unavailable",
            value_bp=None,
            repricing_residual_eur=None,
            reason="No future stochastic coupons remain for a flat-vol inversion.",
        )
    if any(float(row["expiry_years"]) <= 0.0 for row in future):
        return FlatEquivalentNormalVol(
            status="unavailable",
            value_bp=None,
            repricing_residual_eur=None,
            reason="A future coupon has no positive option expiry.",
        )
    fixed_price = fsum(float(row.get("price", 0.0)) for row in rows if row.get("status") == "fixed")
    target = total_price - fixed_price
    tolerance = max(1e-8, 1e-12 * max(abs(total_price), 1.0))
    intrinsic = _flat_price(product_type, strike, notional, future, 0.0)
    if target < intrinsic - tolerance:
        return FlatEquivalentNormalVol(
            status="unavailable",
            value_bp=None,
            repricing_residual_eur=None,
            reason="Future premium is below intrinsic value.",
        )
    if abs(target - intrinsic) <= tolerance:
        return FlatEquivalentNormalVol(
            status="available",
            value_bp=0.0,
            repricing_residual_eur=intrinsic - target,
            reason=None,
        )
    low, high = 0.0, 0.01
    for _ in range(80):
        if _flat_price(product_type, strike, notional, future, high) >= target:
            break
        high *= 2.0
    else:
        return FlatEquivalentNormalVol(
            status="unavailable",
            value_bp=None,
            repricing_residual_eur=None,
            reason="Unable to bracket a finite flat equivalent normal volatility.",
        )
    for _ in range(160):
        sigma = (low + high) / 2.0
        if _flat_price(product_type, strike, notional, future, sigma) < target:
            low = sigma
        else:
            high = sigma
    sigma = (low + high) / 2.0
    residual = _flat_price(product_type, strike, notional, future, sigma) - target
    return FlatEquivalentNormalVol(
        status="available",
        value_bp=sigma * 10_000.0,
        repricing_residual_eur=residual,
        reason=None,
    )


def _tenor_months(label: str) -> int:
    period = parse_tenor(label)
    if period.units() == ql.Years:
        return period.length() * 12
    if period.units() == ql.Months:
        return period.length()
    raise ValueError("Cap/floor expiry must be expressed in months or years.")


def build_source_provenance(
    cap_data: CapFloorVolData,
    forwards: Mapping[str, ForwardMarketData],
) -> LatestNormalSourceProvenance:
    """Retain declared/inferred tenor lineage without claiming provider authentication."""
    quote_tenors: list[QuoteTenorProvenance] = []
    for position, expiry in enumerate(cap_data.expiries):
        if cap_data.quote_tenors is not None:
            tenor = cap_data.quote_tenors[position]
            declaration: Literal["declared", "inferred"] = "declared"
        elif cap_data.index_tenor is not None:
            tenor = cap_data.index_tenor
            declaration = "declared"
        else:
            tenor = "3M" if _tenor_months(expiry) <= 24 else "6M"
            declaration = "inferred"
        quote_tenors.append(
            QuoteTenorProvenance(
                expiry=expiry,
                index_tenor=tenor,
                declaration=declaration,
            )
        )
    warnings = list(cap_data.import_warnings)
    for tenor in ("3M", "6M"):
        forward = forwards.get(tenor)
        if forward is not None:
            warnings.extend(forward.import_warnings)
    return LatestNormalSourceProvenance(
        quote_tenors=quote_tenors,
        external_origin="unverified",
        import_warnings=list(dict.fromkeys(warnings)),
    )


def build_atm_usage(cap_data: CapFloorVolData) -> LatestNormalAtmUsage:
    """Describe the current explicit non-use of imported ATM caps in stripping."""
    count = len(cap_data.atm_vols or [])
    return LatestNormalAtmUsage(
        market_atm_input_present=count > 0,
        market_atm_quote_count=count,
        used_as_direct_strip_target=False,
        warning=(
            "Market ATM input quotes are not direct stripping targets; fitted ATM values are "
            "interpolated outputs of the common smile."
        ),
    )
