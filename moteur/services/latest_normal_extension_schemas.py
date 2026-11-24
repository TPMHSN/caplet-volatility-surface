"""Explicit, bounded assumptions and provenance for experimental extensions."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from services.latest_normal_candidate_schemas import CandidatePriceConventions
from services.latest_normal_schemas import (
    LatestNormalCaplet,
    LatestNormalPriceRequest,
    LatestNormalPriceResponse,
)


class ExtensionConfig(BaseModel):
    """Model hypotheses, never market calibration parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    correlation_rho: FiniteFloat = Field(default=1.0, ge=0.0, le=1.0)
    max_horizon_years: Literal[31] = 31
    blend_years: Literal[1.0] = 1.0


class ExtensionCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classification: Literal["early_flat_normal", "long_3m_from_6m", "terminal_flat_normal"]
    experimental: Literal[True] = True
    validation_status: Literal["experimental_not_independently_validated"] = (
        "experimental_not_independently_validated"
    )
    source_tenor: Literal["3M", "6M"]
    source_fixing_date: date
    source_payment_date: date
    source_strike: FiniteFloat
    source_strike_domain: tuple[FiniteFloat, FiniteFloat]
    source_strike_extrapolated: bool
    shape_grid_source_strike_extrapolated: bool
    source_fixing_extrapolated: bool
    requested_source_fixing_date: date
    requested_source_payment_date: date
    horizon_date: date
    blend_weight: FiniteFloat = Field(ge=0, le=1)
    correlation_rho: FiniteFloat | None = Field(default=None, ge=0, le=1)
    composition_weights: tuple[FiniteFloat, FiniteFloat] | None = None
    six_month_to_three_month_scale: FiniteFloat | None = Field(default=None, gt=0)
    deterministic_basis: FiniteFloat | None = None
    shape_checked: Literal[True] = True
    assumptions: list[str]


class ExtendedCouponTerms(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notional: FiniteFloat = Field(ge=0, le=1e12)
    strike: FiniteFloat = Field(ge=-0.015, le=0.05)


class ExtendedCandidatePriceRequest(LatestNormalPriceRequest):
    extension: ExtensionConfig = Field(default_factory=ExtensionConfig)
    coupon_schedule: list[ExtendedCouponTerms] | None = Field(
        default=None, min_length=1, max_length=480
    )

    @model_validator(mode="after")
    def reject_unsupported_amortizing_risk(self):
        if self.coupon_schedule is not None and self.include_risk:
            raise ValueError("Le risque amortissable n'est pas disponible dans cet endpoint.")
        return self


class ExtendedCandidateCaplet(LatestNormalCaplet):
    notional: FiniteFloat | None = Field(default=None, ge=0, le=1e12)
    strike: FiniteFloat | None = Field(default=None, ge=-0.015, le=0.05)
    vol_source: (
        Literal[
            "quadratic",
            "short_3m_to_6m_bridge",
            "candidate_fixing_time_moneyness",
            "experimental_early_flat_normal",
            "experimental_long_3m_from_6m",
            "experimental_terminal_flat_normal",
        ]
        | None
    )
    coverage: ExtensionCoverage | None = None


class ExtendedCandidatePriceResponse(LatestNormalPriceResponse):
    model: Literal["latest_normal_v3_candidate_3m"]
    version: Literal["v3_candidate_3m_v1"]
    caplets: list[ExtendedCandidateCaplet]
    conventions: CandidatePriceConventions
    extension_version: Literal["experimental_31y_v1"]
    extension_config: ExtensionConfig
    validation_status: Literal["experimental_not_independently_validated"]
    native_coupon_count: int = Field(ge=0)
    extended_coupon_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_tenor_pricing_policy(self):
        expected = (
            "nonflat_variance_wing"
            if self.index_tenor == "3M"
            else "d4_below_native_boundary_native_cubic"
        )
        if self.conventions.smile_policy != expected:
            raise ValueError("Extension base smile policy must match the native candidate tenor")
        future = [row for row in self.caplets if row.status == "future"]
        extended = sum(row.coverage is not None for row in future)
        if (
            self.extended_coupon_count != extended
            or self.native_coupon_count != len(future) - extended
        ):
            raise ValueError(
                "Native and extended coupon counts must match future coupon provenance"
            )
        return self


class ExtendedCandidateSurfaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=64)
    valuation_date: date
    index_tenor: Literal["3M", "6M"] = "3M"
    fixing_dates: list[date] = Field(default_factory=list, max_length=240)
    grid_to_30y: bool = False
    strikes: list[FiniteFloat] = Field(
        default_factory=lambda: [-0.015, -0.01, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05],
        min_length=1,
        max_length=131,
    )
    extension: ExtensionConfig = Field(default_factory=ExtensionConfig)

    @model_validator(mode="after")
    def require_strike_domain(self):
        if self.grid_to_30y == bool(self.fixing_dates):
            raise ValueError("Choisir la grille 30 ans ou des fixings explicites, exclusivement.")
        if len(set(self.fixing_dates)) != len(self.fixing_dates):
            raise ValueError("Surface fixing dates must be unique")
        if any(strike < -0.015 or strike > 0.05 for strike in self.strikes):
            raise ValueError("Surface strikes must lie in [-1.5%,5%]")
        if len(set(self.strikes)) != len(self.strikes):
            raise ValueError("Surface strikes must be unique")
        return self


class ExtendedCandidateSurfacePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strike: FiniteFloat
    native_normal_vol: FiniteFloat | None = Field(ge=0)
    extended_normal_vol: FiniteFloat | None = Field(ge=0)
    coverage: ExtensionCoverage | None = None


class ExtendedCandidateSurfaceSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixing_date: date
    nominal_start_years: FiniteFloat | None = None
    start_date: date | None = None
    end_date: date | None = None
    payment_date: date | None = None
    expiry_years: FiniteFloat | None = None
    forward_rate: FiniteFloat | None = None
    status: Literal["ready", "unavailable"]
    curve: list[ExtendedCandidateSurfacePoint]
    warnings: list[str]
    issues: list[str]
    native_issues: list[str]


class ExtendedCandidateSurfaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    valuation_date: date
    index_tenor: Literal["3M", "6M"]
    extension_version: Literal["experimental_31y_v1"]
    extension_config: ExtensionConfig
    validation_status: Literal["experimental_not_independently_validated"]
    slices: list[ExtendedCandidateSurfaceSlice]
