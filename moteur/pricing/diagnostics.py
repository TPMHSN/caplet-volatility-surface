"""
Typed pricing diagnostics shared across instrument responses.
"""

from collections.abc import Mapping
from datetime import date as dt_date
from typing import Dict, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator


class CurveInstrumentDelta(BaseModel):
    """Local DV01 of one market quote used in curve construction.

    Every DV01 field is ``-1e-4 * dV/dq`` from the implicit calibration Jacobian.
    ``raw_quote_dv01`` and ``scaled_quote_dv01`` are deprecated aliases kept for older clients;
    both equal ``quote_dv01``.
    """

    curve_key: str = Field(..., description="Curve key affected by the quote bump")
    curve_role: str = Field(..., description="Pricing role affected by the bumped curve quote")
    instrument_type: str = Field(..., description="Market instrument family")
    label: str = Field(..., description="Display label for the bumped market quote")
    tenor: Optional[str] = Field(default=None, description="Instrument tenor when available")
    maturity_date: Optional[dt_date] = Field(
        default=None,
        description="Approximate quote maturity date when available",
    )
    quote: float = Field(..., description="Original quote in decimal rate terms")
    pay_leg_delta: float = Field(..., description="Pay-leg local quote DV01 per bp")
    receive_leg_delta: float = Field(..., description="Receive-leg local quote DV01 per bp")
    quote_dv01: float = Field(
        default=0.0,
        description="Local quote DV01 per bp from the implicit calibration Jacobian",
    )
    raw_quote_dv01: float = Field(
        default=0.0,
        description="Deprecated alias of quote_dv01 (no allocation scaling is applied)",
    )
    scaled_quote_dv01: float = Field(
        default=0.0,
        description="Deprecated alias of quote_dv01 (no allocation scaling is applied)",
    )
    delta: float = Field(..., description="Local quote DV01 per bp, equal to quote_dv01")

    @model_validator(mode="after")
    def _default_quote_dv01_aliases(self) -> "CurveInstrumentDelta":
        if self.quote_dv01 == 0.0 and self.delta != 0.0:
            self.quote_dv01 = self.delta
        if self.raw_quote_dv01 == 0.0 and self.delta != 0.0:
            self.raw_quote_dv01 = self.delta
        if self.scaled_quote_dv01 == 0.0 and self.delta != 0.0:
            self.scaled_quote_dv01 = self.delta
        return self


class CurveQuoteBucketDelta(BaseModel):
    """Curve quote-level deltas aggregated into a standard maturity bucket."""

    curve_key: str = Field(..., description="Curve key affected by the bucketed quote deltas")
    curve_role: str = Field(..., description="Pricing role affected by the curve key")
    bucket_label: str = Field(..., description="Configured quote maturity bucket label")
    start_year: float = Field(..., description="Bucket start in years from curve date")
    end_year: float = Field(..., description="Bucket end in years from curve date")
    pay_leg_delta: float = Field(..., description="Aggregated pay leg quote DV01")
    receive_leg_delta: float = Field(..., description="Aggregated receive leg quote DV01")
    quote_dv01: float = Field(
        default=0.0,
        description="Plain sum of the local quote DV01s in the bucket",
    )
    raw_quote_dv01: float = Field(
        default=0.0,
        description="Deprecated alias of quote_dv01 (no allocation scaling is applied)",
    )
    scaled_quote_dv01: float = Field(
        default=0.0,
        description="Deprecated alias of quote_dv01 (no allocation scaling is applied)",
    )
    delta: float = Field(..., description="Aggregated bucket DV01, equal to quote_dv01")
    quote_count: int = Field(..., ge=1, description="Number of market quotes in the bucket")

    @model_validator(mode="after")
    def _default_quote_dv01(self) -> "CurveQuoteBucketDelta":
        if self.quote_dv01 == 0.0 and self.delta != 0.0:
            self.quote_dv01 = self.delta
        return self


class CurveRiskColumn(BaseModel):
    """Curve column in the quote-level risk ladder."""

    curve_key: str = Field(..., description="Canonical curve key used by pricing")
    curve_label: str = Field(..., description="Display label for the curve column")
    curve_role: str = Field(..., description="Pricing role for this curve")


class CurveRiskCell(BaseModel):
    """One curve contribution for one bootstrap instrument row."""

    curve_key: str = Field(..., description="Canonical curve key for the cell")
    curve_label: str = Field(..., description="Display label for the curve")
    used_in_bootstrap: bool = Field(
        default=False,
        description="Whether this curve uses the row's market instrument in its bootstrap",
    )
    quote: Optional[float] = Field(default=None, description="Original market quote in decimal")
    raw_dv01: float = Field(
        default=0.0,
        description="Deprecated equality alias of delta; no scaling is applied",
    )
    scaled_dv01: float = Field(
        default=0.0,
        description="Deprecated equality alias of delta; no scaling is applied",
    )
    delta: float = Field(default=0.0, description="Plain local quote DV01 contribution")


class CurveRiskRow(BaseModel):
    """One bootstrap instrument row across all curve columns."""

    family: str = Field(..., description="Instrument family key")
    family_label: str = Field(..., description="Instrument family display label")
    point_key: str = Field(..., description="Stable point key within the family")
    label: str = Field(..., description="Display label for the instrument point")
    tenor: Optional[str] = Field(default=None, description="Instrument tenor when available")
    maturity_date: Optional[dt_date] = Field(
        default=None,
        description="Approximate maturity date for sorting and audit",
    )
    cells: list[CurveRiskCell] = Field(default_factory=list, description="Curve cells")
    raw_total_delta: float = Field(
        default=0.0,
        description="Deprecated equality alias of total_delta; no scaling is applied",
    )
    total_delta: float = Field(default=0.0, description="Displayed row DV01 total")


class CurveRiskSection(BaseModel):
    """Risk ladder section for one instrument family."""

    family: str = Field(..., description="Instrument family key")
    label: str = Field(..., description="Instrument family display label")
    rows: list[CurveRiskRow] = Field(default_factory=list, description="Rows in this family")
    raw_total_delta: float = Field(
        default=0.0,
        description="Deprecated equality alias of total_delta; no scaling is applied",
    )
    total_delta: float = Field(default=0.0, description="Displayed section DV01 total")


class CurveRiskLadder(BaseModel):
    """Quote-level curve risk ladder aggregated by instrument family and curve."""

    currency: str = Field(..., description="Currency of the risk ladder")
    bump_bp: float = Field(default=1.0, description="Quote bump size in bp")
    curve_columns: list[CurveRiskColumn] = Field(
        default_factory=list,
        description="Curve columns displayed in the ladder",
    )
    sections: list[CurveRiskSection] = Field(
        default_factory=list,
        description="Instrument-family sections",
    )
    raw_total_delta: float = Field(
        default=0.0,
        description="Deprecated equality alias of total_delta; no scaling is applied",
    )
    total_delta: float = Field(default=0.0, description="Plain sum of ladder quote DV01")
    target_delta: Optional[float] = Field(
        default=None,
        description="Compatibility alias of the plain quote DV01 total",
    )
    scaling_factor: Optional[float] = Field(
        default=None,
        description="Deprecated scaling field; always null because no scaling is applied",
    )
    residual: float = Field(
        default=0.0,
        description="Comparison-only difference: target_delta minus total_delta; never allocated",
    )
    raw_residual: Optional[float] = Field(
        default=None,
        description="Deprecated equality alias of residual; comparison only",
    )
    warnings: list[str] = Field(default_factory=list, description="Ladder construction warnings")


class KeyRateRiskCurveContribution(BaseModel):
    """One curve's contribution to a combined key-rate row."""

    curve_key: str = Field(..., description="Canonical curve key used by pricing")
    curve_label: str = Field(..., description="User-facing curve label")
    raw_pay_leg_dv01: float = Field(
        ...,
        description="Deprecated equality alias of pay_leg_dv01; no scaling is applied",
    )
    raw_receive_leg_dv01: float = Field(
        ...,
        description="Deprecated equality alias of receive_leg_dv01; no scaling is applied",
    )
    raw_net_dv01: float = Field(
        ...,
        description="Deprecated equality alias of net_dv01; no scaling is applied",
    )
    pay_leg_dv01: float = Field(..., description="Plain pay-leg quote DV01")
    receive_leg_dv01: float = Field(..., description="Plain receive-leg quote DV01")
    net_dv01: float = Field(..., description="Plain pay plus receive quote DV01")


class KeyRateRiskRow(BaseModel):
    """Combined discount-plus-forward KRR for one market maturity point."""

    family: str = Field(..., description="Bootstrap instrument family key")
    family_label: str = Field(..., description="Bootstrap instrument family display label")
    point_key: str = Field(..., description="Stable point key within the family")
    label: str = Field(..., description="Display label for the key-rate point")
    tenor: Optional[str] = Field(default=None, description="Instrument tenor when available")
    maturity_date: Optional[dt_date] = Field(
        default=None,
        description="Approximate point maturity date for sorting and audit",
    )
    curve_keys: list[str] = Field(
        default_factory=list,
        description="Curves contributing to this combined row",
    )
    curve_contributions: list[KeyRateRiskCurveContribution] = Field(
        default_factory=list,
        description="Auditable per-curve contributions before row aggregation",
    )
    raw_pay_leg_dv01: float = Field(
        ...,
        description="Deprecated equality alias of pay_leg_dv01; no scaling is applied",
    )
    raw_receive_leg_dv01: float = Field(
        ...,
        description="Deprecated equality alias of receive_leg_dv01; no scaling is applied",
    )
    raw_net_dv01: float = Field(
        ...,
        description="Deprecated equality alias of net_dv01; no scaling is applied",
    )
    pay_leg_dv01: float = Field(..., description="Plain pay-leg quote DV01")
    receive_leg_dv01: float = Field(..., description="Plain receive-leg quote DV01")
    net_dv01: float = Field(..., description="Plain pay plus receive quote DV01")


class KeyRateRiskMatrix(BaseModel):
    """Combined discount-plus-forward quote KRR; options expose net risk only."""

    display_mode: str = Field(default="legs", description="legs for swaps, net_only for options")

    currency: str = Field(..., description="Currency of the risk matrix")
    bump_bp: float = Field(default=1.0, description="Market quote bump size in bp")
    discount_curve_key: str = Field(..., description="Discount curve used by pricing")
    forward_curve_key: str = Field(..., description="Forward curve used by pricing")
    curve_labels: list[str] = Field(
        default_factory=list,
        description="User-facing curves combined in every key-rate row",
    )
    rows: list[KeyRateRiskRow] = Field(default_factory=list, description="Ordered KRR rows")
    raw_pay_leg_total: float = Field(
        default=0.0,
        description="Deprecated equality alias of pay_leg_total; no scaling is applied",
    )
    raw_receive_leg_total: float = Field(
        default=0.0,
        description="Deprecated equality alias of receive_leg_total; no scaling is applied",
    )
    raw_net_total: float = Field(
        default=0.0,
        description="Deprecated equality alias of net_total; no scaling is applied",
    )
    pay_leg_total: float = Field(default=0.0, description="Plain sum of pay-leg row DV01")
    receive_leg_total: float = Field(
        default=0.0,
        description="Plain sum of receive-leg row DV01",
    )
    net_total: float = Field(default=0.0, description="Plain sum of net row DV01")
    target_pay_leg_dv01: float = Field(
        ...,
        description="Compatibility alias of the plain pay-leg quote DV01 total",
    )
    target_receive_leg_dv01: float = Field(
        ...,
        description="Compatibility alias of the plain receive-leg quote DV01 total",
    )
    target_net_dv01: float = Field(
        ...,
        description="Compatibility alias of the plain net quote DV01 total",
    )
    pay_scaling_factor: Optional[float] = Field(
        default=None,
        description="Deprecated scaling field; always null because no scaling is applied",
    )
    receive_scaling_factor: Optional[float] = Field(
        default=None,
        description="Deprecated scaling field; always null because no scaling is applied",
    )
    pay_residual: float = Field(
        default=0.0,
        description="Comparison-only difference: target pay-leg total minus pay_leg_total",
    )
    receive_residual: float = Field(
        default=0.0,
        description="Comparison-only difference: target receive-leg total minus receive_leg_total",
    )
    net_residual: float = Field(
        default=0.0,
        description="Comparison-only difference: target net total minus net_total",
    )
    warnings: list[str] = Field(default_factory=list, description="KRR construction warnings")


class BucketedCurveDeltaDiagnostics(BaseModel):
    """Market-quote DV01 attribution, independent parallel check and zero-rate buckets.

    The product DV01 is ``quote_dv01_total``: ``-1e-4 * sum(dV/dq)`` from the implicit
    calibration Jacobian. ``parallel_dv01`` is an independent check that never rescales anything;
    every ``*_scaling_factor`` is ``None`` and every ``scaled_*`` value equals its exact
    counterpart. Zero-rate buckets shock the zero curves, not the quotes, so their sum is not
    expected to equal the DV01.
    """

    bump_bp: float = Field(default=1.0, description="Finite-difference bump size in bp")
    quote_dv01_total: float | None = Field(
        default=None,
        description="Product local market-quote DV01 from the calibration Jacobian",
    )
    unbucketed_quote_dv01: float = Field(
        default=0.0, description="Signed quote DV01 omitted from standard maturity buckets"
    )
    unbucketed_quote_count: int = Field(
        default=0, ge=0, description="Number of quotes omitted from standard maturity buckets"
    )
    parallel_dv01: float | None = Field(
        default=None,
        description="Independent central parallel market-quote DV01 check (currency per bp)",
    )
    parallel_check_residual: float | None = Field(
        default=None,
        description="parallel_dv01 - quote_dv01_total",
    )
    parallel_check_tolerance: float | None = Field(
        default=None,
        description="max(0.01, 0.5% * max(|parallel|, |quote total|)) in currency per bp",
    )
    parallel_check_passed: bool | None = Field(
        default=None,
        description="Whether the parallel check residual is within tolerance (warning otherwise)",
    )
    dv01_source: str | None = Field(
        default=None,
        description="market_quote_jacobian for the local DV01 contract",
    )
    delta_parallel: float = Field(
        ...,
        description="Deprecated alias of parallel_dv01 (independent parallel check)",
    )
    delta_bucket_sum: float = Field(..., description="Sum of bucketed curve deltas")
    delta_parallel_minus_bucket_sum: float = Field(
        ...,
        description="Comparison-only difference: parallel DV01 minus zero-rate bucket sum",
    )
    bucketed_curve_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Per-bucket curve sensitivities keyed by maturity bucket label",
    )
    raw_bucketed_dv01: dict[str, float] = Field(
        default_factory=dict,
        description="Deprecated equality alias of bucketed_curve_deltas; no scaling is applied",
    )
    raw_bucket_sum: Optional[float] = Field(
        default=None,
        description="Deprecated equality alias of delta_bucket_sum; no scaling is applied",
    )
    scaled_bucket_sum: Optional[float] = Field(
        default=None,
        description="Deprecated equality alias of delta_bucket_sum; no scaling is applied",
    )
    bucket_scaling_factor: Optional[float] = Field(
        default=None,
        description="Deprecated scaling field; always null because no scaling is applied",
    )
    parallel_minus_raw_bucket_sum: Optional[float] = Field(
        default=None,
        description="Comparison-only difference: parallel DV01 minus zero-rate bucket sum",
    )
    curve_bucketed_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Per-standard-bucket curve sensitivities using the fine curve bucket grid",
    )
    raw_curve_bucketed_dv01: dict[str, float] = Field(
        default_factory=dict,
        description="Deprecated equality alias of curve_bucketed_deltas; no scaling is applied",
    )
    curve_bucket_sum: Optional[float] = Field(
        default=None,
        description="Plain sum of fine zero-rate curve buckets",
    )
    raw_curve_bucket_sum: Optional[float] = Field(
        default=None,
        description="Deprecated equality alias of curve_bucket_sum; no scaling is applied",
    )
    curve_bucket_scaling_factor: Optional[float] = Field(
        default=None,
        description="Deprecated scaling field; always null because no scaling is applied",
    )
    delta_parallel_minus_curve_bucket_sum: Optional[float] = Field(
        default=None,
        description="Comparison-only difference: parallel DV01 minus fine zero-rate bucket sum",
    )
    parallel_minus_raw_curve_bucket_sum: Optional[float] = Field(
        default=None,
        description="Comparison-only difference: parallel DV01 minus fine zero-rate bucket sum",
    )
    delta_instrument_sum: Optional[float] = Field(
        default=None,
        description="Plain sum of per-instrument quote DV01 when raw quotes are available",
    )
    delta_parallel_minus_instrument_sum: Optional[float] = Field(
        default=None,
        description="Comparison-only difference: parallel DV01 minus per-instrument quote sum",
    )
    curve_instrument_deltas: list[CurveInstrumentDelta] = Field(
        default_factory=list,
        description="Per-market-instrument sensitivities from the calibration Jacobian",
    )
    curve_quote_bucket_deltas: list[CurveQuoteBucketDelta] = Field(
        default_factory=list,
        description="Quote-level curve sensitivities aggregated into standard maturity buckets",
    )
    curve_risk_ladder: Optional[CurveRiskLadder] = Field(
        default=None,
        description="Quote-level curve risk ladder by instrument family and curve",
    )
    key_rate_risk: Optional[KeyRateRiskMatrix] = Field(
        default=None,
        description="Combined discount-plus-forward quote KRR summed independently by leg",
    )
    raw_quote_sum: Optional[float] = Field(
        default=None,
        description="Deprecated equality alias of quote_dv01_total; no scaling is applied",
    )
    scaled_quote_sum: Optional[float] = Field(
        default=None,
        description="Deprecated equality alias of quote_dv01_total; no scaling is applied",
    )
    quote_scaling_factor: Optional[float] = Field(
        default=None,
        description="Deprecated scaling field; always null because no scaling is applied",
    )
    parallel_minus_raw_quote_sum: Optional[float] = Field(
        default=None,
        description="Comparison-only difference: parallel DV01 minus quote_dv01_total",
    )


class SwaptionGreeksDiagnostics(BucketedCurveDeltaDiagnostics):
    """Swaption-specific diagnostics for curve and vol sensitivities."""

    rate_volatility_policy: Optional[
        Literal["normal_moneyness", "frozen_effective_vol", "frozen_model_parameters"]
    ] = None
    frozen_effective_vol_dv01: Optional[float] = Field(
        default=None,
        description="Separate signed quote-DV01 control with effective deal volatility frozen",
    )
    frozen_effective_vol_curve_instrument_deltas: list[CurveInstrumentDelta] = Field(
        default_factory=list,
        description="Separate position-signed quote buckets with native effective deal vol frozen",
    )
    frozen_effective_vol_curve_risk_ladder: Optional[CurveRiskLadder] = Field(
        default=None,
        description="Quote-risk ladder of the native frozen-effective-vol control",
    )
    curve_sensitivity: dict[str, float] = Field(
        default_factory=dict,
        description="Optional split between OIS and forward parallel curve bumps",
    )
    vega_parallel_vol: Optional[float] = Field(
        default=None,
        description="NPV change for a +1bp parallel volatility bump",
    )
    vega_source: Optional[str] = Field(
        default=None,
        description="Diagnostic provenance for the vega calculation path",
    )
    vega_note: Optional[str] = Field(
        default=None,
        description="Optional note when the vega diagnostic falls back or degrades",
    )
    reversion: Optional[float] = Field(
        default=None, description="Declared (not fitted) GSR mean reversion of the deal model"
    )
    reversion_bump: Optional[float] = Field(
        default=None, description="Absolute mean reversion bump of the sensitivity below"
    )
    reversion_npv_change: Optional[float] = Field(
        default=None,
        description="NPV change when the deal basket is recalibrated at reversion + bump",
    )
    structure_sensitivity: Optional[Dict[str, Dict[str, Optional[float]]]] = Field(
        default=None,
        description=(
            "Two-factor price: NPV change when each structure number (a, b, rho, vol_ratio) is bumped "
            "and the levels recalibrated on the same basket"
        ),
    )


# Every numeric field below measures a DV01-type sensitivity and flips with the position.
# Market quotes, bump sizes, tolerances, counts, maturities and statuses never do.
DV01_SIGNED_FIELDS = frozenset(
    {
        # Diagnostics totals and parallel check
        "delta_parallel",
        "delta_bucket_sum",
        "delta_parallel_minus_bucket_sum",
        "raw_bucket_sum",
        "scaled_bucket_sum",
        "parallel_minus_raw_bucket_sum",
        "curve_bucket_sum",
        "raw_curve_bucket_sum",
        "delta_parallel_minus_curve_bucket_sum",
        "parallel_minus_raw_curve_bucket_sum",
        "delta_instrument_sum",
        "delta_parallel_minus_instrument_sum",
        "raw_quote_sum",
        "scaled_quote_sum",
        "parallel_minus_raw_quote_sum",
        "quote_dv01_total",
        "unbucketed_quote_dv01",
        "parallel_dv01",
        "parallel_check_residual",
        "vega_parallel_vol",
        "frozen_effective_vol_dv01",
        "reversion_npv_change",
        "npv_change",
        # Quote entries and quote buckets
        "pay_leg_delta",
        "receive_leg_delta",
        "quote_dv01",
        "raw_quote_dv01",
        "scaled_quote_dv01",
        "delta",
        # Curve-risk ladder
        "raw_total_delta",
        "total_delta",
        "target_delta",
        "residual",
        "raw_residual",
        "raw_dv01",
        "scaled_dv01",
        # Key-rate risk
        "raw_pay_leg_total",
        "raw_receive_leg_total",
        "raw_net_total",
        "pay_leg_total",
        "receive_leg_total",
        "net_total",
        "target_pay_leg_dv01",
        "target_receive_leg_dv01",
        "target_net_dv01",
        "pay_residual",
        "receive_residual",
        "net_residual",
        "raw_pay_leg_dv01",
        "raw_receive_leg_dv01",
        "raw_net_dv01",
        "pay_leg_dv01",
        "receive_leg_dv01",
        "net_dv01",
    }
)
DV01_SIGNED_MAPS = frozenset(
    {
        "bucketed_curve_deltas",
        "raw_bucketed_dv01",
        "curve_bucketed_deltas",
        "raw_curve_bucketed_dv01",
        "curve_sensitivity",
    }
)

DiagnosticsT = TypeVar("DiagnosticsT", bound=BaseModel)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def sign_dv01_payload(value: object, sign: float) -> object:
    """Apply a position sign once to every DV01-type field of a JSON-like diagnostics payload."""
    if isinstance(value, dict):
        signed: dict[object, object] = {}
        for key, item in value.items():
            if key in DV01_SIGNED_MAPS and isinstance(item, dict):
                signed[key] = {
                    name: entry * sign if _is_number(entry) else entry
                    for name, entry in item.items()
                }
            elif key in DV01_SIGNED_FIELDS and _is_number(item):
                signed[key] = item * sign
            else:
                signed[key] = sign_dv01_payload(item, sign)
        return signed
    if isinstance(value, list):
        return [sign_dv01_payload(item, sign) for item in value]
    return value


def signed_curve_diagnostics(diagnostics: DiagnosticsT, sign: float) -> DiagnosticsT:
    """Return typed diagnostics with the position sign applied to every DV01-type field."""
    if sign == 1.0:
        return diagnostics
    payload = sign_dv01_payload(diagnostics.model_dump(), sign)
    return type(diagnostics).model_validate(payload)


def parallel_check_warning(
    parallel_dv01: float,
    quote_dv01_total: float,
    residual: float,
    tolerance: float,
) -> str:
    """Warning text for a parallel DV01 check outside tolerance, in currency per bp."""
    return (
        "Parallel DV01 check outside tolerance: parallel "
        f"{parallel_dv01:.4f} vs quote sum {quote_dv01_total:.4f} "
        f"(residual {residual:.4f}, tolerance {tolerance:.4f}, currency per bp)."
    )


def dv01_check_warnings(
    diagnostics: BucketedCurveDeltaDiagnostics | Mapping[str, object] | None,
) -> list[str]:
    """Warnings for a failed independent parallel check, using the diagnostics' signed values."""
    if diagnostics is None:
        return []
    payload = diagnostics if isinstance(diagnostics, Mapping) else diagnostics.model_dump()
    if payload.get("parallel_check_passed") is not False:
        return []
    return [
        parallel_check_warning(
            float(payload.get("parallel_dv01") or 0.0),
            float(payload.get("quote_dv01_total") or 0.0),
            float(payload.get("parallel_check_residual") or 0.0),
            float(payload.get("parallel_check_tolerance") or 0.0),
        )
    ]
