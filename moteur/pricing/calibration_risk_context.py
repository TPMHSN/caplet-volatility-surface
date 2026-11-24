"""Coupled native calibration graph for Pricer risk and explicit market scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import QuantLib as ql

from curves.calibration_context import CurveCalibrationContext, RawQuoteKey
from curves.calibration_diagnostics import CurveBuildResult
from curves.yield_curve_builder import CurveSet, YieldCurveBuilder
from services.exceptions import PricingError

BlockKey = tuple[str, str]
QuoteKey = tuple[str, RawQuoteKey]
CurvePair = tuple[ql.YieldTermStructureHandle, ql.YieldTermStructureHandle]


@dataclass
class RiskCalibrationContext:
    """Private mutable bootstrap objects; only immutable curve copies escape this graph."""

    blocks: dict[BlockKey, CurveCalibrationContext]
    base_curves: dict[BlockKey, ql.YieldTermStructureHandle]
    ois_key: BlockKey
    discount_key: BlockKey
    forward_key: BlockKey

    def quote_scenario(self, shocks: Mapping[QuoteKey, float]) -> CurvePair:
        """Recalibrate a finite quote scenario using the retained objects and active helpers."""
        curves = dict(self.base_curves)
        ois_shocks = {key: value for (curve, key), value in shocks.items() if curve == "OIS"}
        for block_key, context in self.blocks.items():
            curve_key, _ = block_key
            local_shocks = {
                key: value for (curve, key), value in shocks.items() if curve == curve_key
            }
            # A quote excluded from this role can remain active in the other role.
            active = {binding.raw_quote_key for binding in context.helper_bindings}
            local_shocks = {key: value for key, value in local_shocks.items() if key in active}
            if not local_shocks and not (ois_shocks and curve_key != "OIS"):
                continue
            curves[block_key] = context.rebootstrap_and_freeze(
                local_shocks,
                discount_curve=curves[self.ois_key] if curve_key != "OIS" else None,
            )
        return curves[self.forward_key], curves[self.discount_key]


def build_risk_calibration_context(
    curve_set: CurveSet,
    discount_key: str,
    forward_key: str,
    *,
    accuracy: float = 1.0e-12,
) -> RiskCalibrationContext:
    """Build the selected role graph under the caller's calibration date/fixing scope."""
    if curve_set.ois_market_data is None:
        raise PricingError("Market-quote risk requires retained OIS calibration quotes")
    builder = YieldCurveBuilder(curve_set.currency)
    discount_method = curve_set.discount_interpolation
    forward_method = curve_set.forward_interpolation
    ois_key = ("OIS", discount_method)
    blocks: dict[BlockKey, CurveCalibrationContext] = {}
    curves: dict[BlockKey, ql.YieldTermStructureHandle] = {}

    def retain(key: BlockKey, result: CurveBuildResult, report_key: str) -> None:
        context = result.calibration_context
        if context is None:
            raise PricingError(f"Calibration state is unavailable for {key[0]}")
        expected = curve_set.calibration_reports.get(report_key)
        if expected is not None:
            old = {(item.curve_key, item.quote_index) for item in expected.excluded_quotes}
            new = {(item.curve_key, item.quote_index) for item in result.report.excluded_quotes}
            if old != new:
                raise PricingError("Curve risk changed the calibrated quote set")
        if not result.report.passed:
            raise PricingError(
                f"Calibration residuals are invalid for {key[0]} ({key[1]}): "
                f"max error {result.report.max_abs_residual_bps:.6g} bp exceeds "
                f"{result.report.tolerance_bps:.6g} bp"
            )
        blocks[key] = context
        curves[key] = result.curve

    retain(
        ois_key,
        builder.build_ois_curve_with_report(
            curve_set.ois_market_data, interpolation=discount_method, bootstrap_accuracy=accuracy
        ),
        "OIS",
    )
    roles = ((discount_key, discount_method, "discount"), (forward_key, forward_method, "forward"))
    for key, method, role in roles:
        block_key = (key, method)
        if block_key in blocks:
            continue
        if key == "OIS":
            result = builder.build_ois_curve_with_report(
                curve_set.ois_market_data, interpolation=method, bootstrap_accuracy=accuracy
            )
        else:
            market_data = curve_set.forward_market_data.get(key)
            if market_data is None:
                raise PricingError(f"Market-quote risk requires retained {key} calibration quotes")
            result = builder.build_forward_curve_with_report(
                market_data, curves[ois_key], interpolation=method, bootstrap_accuracy=accuracy
            )
        report_key = (
            f"discount_{key}" if role == "discount" and discount_method != forward_method else key
        )
        retain(block_key, result, report_key)
    return RiskCalibrationContext(
        blocks, curves, ois_key, (discount_key, discount_method), (forward_key, forward_method)
    )
