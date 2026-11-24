"""Strict public contracts for the latest normal cap/floor model."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from pricing.diagnostics import BucketedCurveDeltaDiagnostics
from services.latest_normal_model_math import STRIKE_DOMAIN

LatestNormalSmilePolicy = Literal[
    "d4_to_native_cubic_positive", "d4_below_native_boundary_native_cubic"
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LatestNormalModelRequest(_StrictModel):
    valuation_date: date
    source_token: str = Field(min_length=1, max_length=256)


class LatestNormalPriceRequest(_StrictModel):
    model_id: str = Field(min_length=1, max_length=64)
    valuation_date: date
    product_type: Literal["cap", "floor"]
    index_tenor: Literal["3M", "6M"]
    notional: FiniteFloat = Field(gt=0, le=1e12)
    effective_date: date
    maturity_date: date
    strike: FiniteFloat = Field(ge=STRIKE_DOMAIN[0], le=STRIKE_DOMAIN[1])
    business_day_convention: Literal[
        "ModifiedFollowing", "Following", "Preceding", "Unadjusted"
    ] = "ModifiedFollowing"
    day_count: Literal["ACT/360", "ACT/365F", "30/360"] = "ACT/360"
    date_generation: Literal["Forward", "Backward"] = "Forward"
    fixing_days: int = Field(default=2, ge=0, le=10)
    payment_lag: int = Field(default=0, ge=0, le=10)
    roll_day: int | None = Field(default=None, ge=1, le=31)
    end_of_month: bool = False
    include_risk: bool = False


class RiskReferenceCheck(_StrictModel):
    base_price_eur: FiniteFloat
    reference_price_eur: FiniteFloat
    residual_eur: FiniteFloat
    tolerance_eur: FiniteFloat = Field(ge=0)
    passed: bool


class LatestNormalTarget(_StrictModel):
    strike: FiniteFloat
    normal_vol: FiniteFloat = Field(gt=0)


class LatestNormalD4ComponentDiagnostics(_StrictModel):
    pricing_policy: LatestNormalSmilePolicy
    alpha: FiniteFloat = Field(gt=0)
    beta: FiniteFloat = Field(ge=0, le=1)
    rho: FiniteFloat = Field(gt=-1, lt=1)
    nu: FiniteFloat = Field(ge=0)
    atm_normal_vol: FiniteFloat = Field(gt=0)
    fit_rmse_bp: FiniteFloat = Field(ge=0)
    fit_cost: FiniteFloat = Field(ge=0)
    bound_hit: bool
    native_boundary_strike: FiniteFloat | None
    boundary_tie_error_bp: FiniteFloat | None
    optimizer_success: bool
    optimizer_status: int


class LatestNormalCurvePoint(_StrictModel):
    strike: FiniteFloat
    normal_vol: FiniteFloat = Field(gt=0)
    d4_normal_vol: FiniteFloat = Field(gt=0)
    cubic_normal_vol: FiniteFloat
    blend_weight: FiniteFloat = Field(ge=0, le=1)


class LatestNormalSlice(_StrictModel):
    fixing_date: date
    payment_date: date
    start_date: date
    end_date: date
    expiry_years: FiniteFloat = Field(ge=0)
    maturity_years: FiniteFloat = Field(ge=0)
    forward_rate: FiniteFloat
    discount_factor: FiniteFloat = Field(gt=0)
    accrual_fraction: FiniteFloat = Field(gt=0)
    pricing_policy: LatestNormalSmilePolicy
    status: Literal["ready", "unavailable"]
    alpha: FiniteFloat | None
    beta: FiniteFloat | None
    rho: FiniteFloat | None
    nu: FiniteFloat | None
    atm_normal_vol: FiniteFloat | None
    fit_rmse_bp: FiniteFloat | None
    targets: list[LatestNormalTarget]
    curve: list[LatestNormalCurvePoint]
    issues: list[str]
    d4_component: LatestNormalD4ComponentDiagnostics | None


class LatestNormalBridgeCoverage(_StrictModel):
    source_tenor: Literal["3M"] = "3M"
    target_tenor: Literal["6M"] = "6M"
    method: Literal["same_absolute_moneyness"] = "same_absolute_moneyness"
    strike_interpolation: Literal["natural_cubic"] = "natural_cubic"
    expiry_extrapolation: Literal["forbidden"] = "forbidden"
    assumptions: list[str]


class LatestNormalCoverage(_StrictModel):
    index_tenor: Literal["3M", "6M"]
    last_covered_fixing_date: date
    last_covered_payment_date: date
    strike_domain: tuple[FiniteFloat, FiniteFloat]
    native_quote_expiries: list[str] = Field(min_length=1)
    native_quote_count: int = Field(ge=1)
    input_quote_count: int = Field(ge=1)
    input_strikes: list[FiniteFloat] = Field(min_length=1)
    input_strike_count: int = Field(ge=1)
    input_strike_domain: tuple[FiniteFloat, FiniteFloat]
    negative_input_quote_count: int = Field(ge=0)
    negative_quote_final_fit_coverage: Literal["complete", "partial", "none"]
    bridge: LatestNormalBridgeCoverage | None = None

    @model_validator(mode="after")
    def require_raw_input_strike_identity(self) -> LatestNormalCoverage:
        if len(set(self.input_strikes)) != len(self.input_strikes):
            raise ValueError("Raw input strikes must be unique")
        if self.input_strike_count != len(self.input_strikes):
            raise ValueError("Raw input strike count must match the explicit strike list")
        if self.input_strike_domain != (min(self.input_strikes), max(self.input_strikes)):
            raise ValueError("Raw input strike domain must match the explicit strike list")
        if self.input_quote_count != self.native_quote_count * len(self.input_strikes):
            raise ValueError("Raw input quote count must equal expiries times strikes")
        negative_count = sum(strike < 0.0 for strike in self.input_strikes)
        if self.negative_input_quote_count != self.native_quote_count * negative_count:
            raise ValueError("Negative raw input quote count must match expiries times strikes")
        return self


class QuoteFitPremium(_StrictModel):
    market_price_unit: FiniteFloat = Field(ge=0)
    model_price_unit: FiniteFloat = Field(ge=0)
    residual_unit: FiniteFloat
    market_price_eur_10m: FiniteFloat = Field(ge=0)
    model_price_eur_10m: FiniteFloat = Field(ge=0)
    residual_eur_10m: FiniteFloat
    tolerance_eur_10m: FiniteFloat = Field(ge=0)
    passed: bool


class QuoteFitProductPair(_StrictModel):
    cap: QuoteFitPremium
    floor: QuoteFitPremium


class NativeQuoteFitPoint(_StrictModel):
    expiry_months: int = Field(gt=0)
    strike: FiniteFloat
    market_quote_normal_vol_bp: FiniteFloat = Field(gt=0)
    status: Literal["ready", "unavailable", "non_covered"]
    block: QuoteFitProductPair | None
    quadratic: QuoteFitProductPair | None
    final: QuoteFitProductPair | None
    reason: str | None = None


class QuoteFitStageSummary(_StrictModel):
    cap_max_abs_residual_eur_10m: FiniteFloat | None
    floor_max_abs_residual_eur_10m: FiniteFloat | None
    cap_passed: bool | None
    floor_passed: bool | None
    passed: bool | None


class NativeQuoteFitDiagnostics(_StrictModel):
    expected_point_count: int = Field(ge=0)
    emitted_point_count: int = Field(ge=0)
    points: list[NativeQuoteFitPoint]
    block: QuoteFitStageSummary
    quadratic: QuoteFitStageSummary
    final: QuoteFitStageSummary
    warnings: list[str]

    @model_validator(mode="after")
    def require_exact_expected_matrix(self) -> NativeQuoteFitDiagnostics:
        if self.emitted_point_count != len(self.points):
            raise ValueError("Emitted native quote-fit count must match the point list")
        if self.expected_point_count != self.emitted_point_count:
            raise ValueError("Every expected native quote-fit point must be emitted")
        return self


class LatestNormalFamily(_StrictModel):
    pricing_policy: LatestNormalSmilePolicy
    index_tenor: Literal["3M", "6M"]
    status: Literal["ready", "partial"]
    calibration_strikes: list[FiniteFloat]
    slices: list[LatestNormalSlice]
    issues: list[str]
    coverage: LatestNormalCoverage
    quote_fit: NativeQuoteFitDiagnostics


class LatestNormalFixings(_StrictModel):
    fingerprint: str = Field(min_length=1)
    counts: dict[Literal["3M", "6M"], int]
    valuation_rates: dict[Literal["3M", "6M"], FiniteFloat | None]


class LatestNormalConfiguration(_StrictModel):
    three_month: Literal["d4_to_native_cubic_positive"]
    six_month: Literal["d4_below_native_boundary_native_cubic"]
    beta_penalty_3m: FiniteFloat
    beta_penalty_6m: FiniteFloat
    blend_start: FiniteFloat
    blend_end: FiniteFloat
    six_month_native_boundary: FiniteFloat
    six_month_transition_start: FiniteFloat
    six_month_transition_end: FiniteFloat
    six_month_native_cubic_domain: tuple[FiniteFloat, FiniteFloat]
    bridge_source: Literal[
        "three_month_blocks_same_absolute_moneyness_natural_cubic",
        "three_month_linear_pillars_joint_first_six_month_pillar",
    ]


class QuoteTenorProvenance(_StrictModel):
    expiry: str
    index_tenor: Literal["3M", "6M"]
    declaration: Literal["declared", "inferred"]


class LatestNormalSourceProvenance(_StrictModel):
    quote_tenors: list[QuoteTenorProvenance]
    external_origin: Literal["unverified"] = "unverified"
    import_warnings: list[str]


class LatestNormalAtmUsage(_StrictModel):
    market_atm_input_present: bool
    market_atm_quote_count: int = Field(ge=0)
    used_as_direct_strip_target: Literal[False] = False
    market_atm_role: Literal["ignored_as_direct_strip_target"] = "ignored_as_direct_strip_target"
    fitted_atm_role: Literal["interpolated_common_smile_output"] = (
        "interpolated_common_smile_output"
    )
    warning: str


class LatestNormalValidation(_StrictModel):
    internal_validation: Literal["passed", "failed", "unavailable"]
    external_validation: Literal["not_run"] = "not_run"
    warnings: list[str]


class LatestNormalPreparationResponse(_StrictModel):
    model: Literal["latest_normal_v3"]
    version: Literal["v3"]
    model_id: str = Field(min_length=1)
    valuation_date: date
    source_token: str = Field(min_length=1)
    source: Literal["loaded", "workbook"]
    model_shift: FiniteFloat
    time_method: Literal["quadratic"]
    smoothing: FiniteFloat = Field(ge=0)
    strike_domain: tuple[FiniteFloat, FiniteFloat]
    families: list[LatestNormalFamily]
    fixings: LatestNormalFixings
    configuration: LatestNormalConfiguration
    assumptions: list[str]
    provenance: LatestNormalSourceProvenance
    atm_usage: LatestNormalAtmUsage
    validation: LatestNormalValidation
    warnings: list[str]
    source_sheets: list[str] | None = None

    @model_validator(mode="after")
    def require_native_families(self) -> LatestNormalPreparationResponse:
        tenors = [family.index_tenor for family in self.families]
        if sorted(tenors) != ["3M", "6M"]:
            raise ValueError("Latest-normal preparation requires exactly one 3M and one 6M family")
        if any(family.coverage.index_tenor != family.index_tenor for family in self.families):
            raise ValueError("Family coverage tenor must match its family")
        expected_policies = {
            "3M": self.configuration.three_month,
            "6M": self.configuration.six_month,
        }
        for family in self.families:
            expected_policy = expected_policies[family.index_tenor]
            if family.pricing_policy != expected_policy:
                raise ValueError("Family pricing policy must match its configured tenor policy")
            if any(slice_.pricing_policy != expected_policy for slice_ in family.slices):
                raise ValueError("Slice pricing policy must match its family policy")
            if any(
                slice_.d4_component is not None
                and slice_.d4_component.pricing_policy != expected_policy
                for slice_ in family.slices
            ):
                raise ValueError("D.4 component policy must match its family policy")
        return self


class FlatEquivalentNormalVol(_StrictModel):
    status: Literal["available", "unavailable"]
    value_bp: FiniteFloat | None
    repricing_residual_eur: FiniteFloat | None
    reason: str | None


class LatestNormalCaplet(_StrictModel):
    start_date: date
    end_date: date
    fixing_date: date
    payment_date: date
    accrual_fraction: FiniteFloat = Field(gt=0)
    index_value_date: date
    index_maturity_date: date
    projection_accrual_fraction: FiniteFloat = Field(gt=0)
    observed_fixing: FiniteFloat | None
    normal_vol: FiniteFloat | None
    parameters: dict[Literal["alpha", "beta", "rho", "nu"], FiniteFloat] | None
    vol_source: Literal["quadratic", "short_3m_to_6m_bridge"] | None
    expiry_years: FiniteFloat = Field(ge=0)
    price: FiniteFloat = Field(ge=0)
    normal_vega_eur_bp: FiniteFloat = Field(ge=0)
    status: Literal["paid", "fixed", "future"]
    forward_rate: FiniteFloat
    discount_factor: FiniteFloat = Field(gt=0)


class LatestNormalScheduleDescriptor(_StrictModel):
    expected_period_count: int = Field(ge=1)
    adjusted_start_date: date
    adjusted_end_date: date


class LatestNormalPriceConventions(_StrictModel):
    effective_date: date
    maturity_date: date
    business_day_convention: Literal["ModifiedFollowing", "Following", "Preceding", "Unadjusted"]
    day_count: Literal["ACT/360", "ACT/365F", "30/360"]
    date_generation: Literal["Forward", "Backward"]
    fixing_days: int
    payment_lag: int
    roll_day: int | None
    end_of_month: bool
    currency: Literal["EUR"]
    discounting: Literal["ESTR"]
    calendar: Literal["TARGET"]
    time_method: Literal["quadratic"]
    model_version: Literal["v3"]
    smile_policy: LatestNormalSmilePolicy
    smile_transition: tuple[FiniteFloat, FiniteFloat]
    native_cubic_domain: tuple[FiniteFloat, FiniteFloat]
    schedule: LatestNormalScheduleDescriptor
    normal_vega_eur_bp_definition: Literal[
        "parallel_1bp_all_future_coupon_normal_vols_forwards_and_discount_factors_frozen"
    ]


class LatestNormalRiskConventions(_StrictModel):
    source: Literal["market_quote_jacobian"]
    derivative_sign: Literal["-1e-4*dPV/dq"]
    unit: Literal["EUR/bp"]
    quote_shift_bp: Literal[1.0]
    parallel_diagnostic: Literal["CENTRAL"]
    calibration_fixing_policy: Literal["constant_observed_ibor"]
    volatility_policy: Literal["frozen_per_coupon_normal_vol"]


class LatestNormalRisk(_StrictModel):
    dv01_eur_bp: FiniteFloat
    diagnostics: BucketedCurveDeltaDiagnostics
    volatility_policy: Literal["frozen_normal_vol"]
    reference_check: RiskReferenceCheck
    conventions: LatestNormalRiskConventions

    @model_validator(mode="after")
    def require_finite_diagnostics(self) -> LatestNormalRisk:
        def visit(value: object) -> bool:
            if isinstance(value, float):
                return isfinite(value)
            if isinstance(value, dict):
                return all(visit(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(visit(item) for item in value)
            return True

        if not visit(self.diagnostics.model_dump()):
            raise ValueError("Curve-risk diagnostics must contain only finite numbers")
        return self


class LatestNormalPriceResponse(_StrictModel):
    model: Literal["latest_normal_v3"]
    version: Literal["v3"]
    model_id: str = Field(min_length=1)
    valuation_date: date
    product_type: Literal["cap", "floor"]
    index_tenor: Literal["3M", "6M"]
    strike: FiniteFloat
    notional: FiniteFloat = Field(gt=0)
    price: FiniteFloat = Field(ge=0)
    unit_price: FiniteFloat = Field(ge=0)
    normal_vega_eur_bp: FiniteFloat = Field(ge=0)
    atm_rate: FiniteFloat
    cap_floor_parity_eur: FiniteFloat
    caplets: list[LatestNormalCaplet]
    warnings: list[str]
    conventions: LatestNormalPriceConventions
    flat_equivalent_normal_vol: FlatEquivalentNormalVol
    risk: LatestNormalRisk | None = None

    @model_validator(mode="after")
    def require_tenor_pricing_policy(self) -> LatestNormalPriceResponse:
        expected = {
            "3M": "d4_to_native_cubic_positive",
            "6M": "d4_below_native_boundary_native_cubic",
        }[self.index_tenor]
        if self.conventions.smile_policy != expected:
            raise ValueError("Price-convention smile policy must match the priced tenor")
        return self
