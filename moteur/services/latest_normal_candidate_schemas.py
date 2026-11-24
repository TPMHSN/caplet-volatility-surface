"""Separate public contracts for the frozen V3 plus 3M candidate."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, FiniteFloat, model_validator

from services.latest_normal_schemas import (
    LatestNormalBridgeCoverage,
    LatestNormalCaplet,
    LatestNormalConfiguration,
    LatestNormalCoverage,
    LatestNormalCurvePoint,
    LatestNormalFamily,
    LatestNormalPreparationResponse,
    LatestNormalPriceConventions,
    LatestNormalPriceResponse,
    LatestNormalSlice,
)

CANDIDATE_MODEL = "latest_normal_v3_candidate_3m"
CANDIDATE_VERSION = "v3_candidate_3m_v1"
CANDIDATE_POLICY = "nonflat_variance_wing"
CandidatePolicy = Literal["nonflat_variance_wing", "d4_below_native_boundary_native_cubic"]


class CandidateCurvePoint(LatestNormalCurvePoint):
    d4_normal_vol: FiniteFloat | None = Field(gt=0)
    cubic_normal_vol: FiniteFloat | None
    blend_weight: FiniteFloat | None = Field(ge=0, le=1)


class CandidateSlice(LatestNormalSlice):
    pricing_policy: CandidatePolicy
    curve: list[CandidateCurvePoint]


class CandidateBridgeCoverage(LatestNormalBridgeCoverage):
    strike_interpolation: Literal["continuous_frozen_candidate"] = "continuous_frozen_candidate"


class CandidateCoverage(LatestNormalCoverage):
    bridge: CandidateBridgeCoverage | None = None


class CandidateFamily(LatestNormalFamily):
    pricing_policy: CandidatePolicy
    slices: list[CandidateSlice]
    coverage: CandidateCoverage


class CandidateConfiguration(LatestNormalConfiguration):
    three_month: Literal["nonflat_variance_wing"]
    bridge_source: Literal["frozen_three_month_variance_wing"]
    candidate_name: Literal["N_variance_wing"]
    candidate_version: Literal["frozen_20261023"]


class CandidatePreparationResponse(LatestNormalPreparationResponse):
    model: Literal["latest_normal_v3_candidate_3m"]
    version: Literal["v3_candidate_3m_v1"]
    time_method: Literal["candidate_variance_3m_quadratic_6m"]
    families: list[CandidateFamily]
    configuration: CandidateConfiguration


class CandidateCaplet(LatestNormalCaplet):
    vol_source: (
        Literal["quadratic", "short_3m_to_6m_bridge", "candidate_fixing_time_moneyness"] | None
    )


class CandidatePriceConventions(LatestNormalPriceConventions):
    time_method: Literal["quadratic", "candidate_fixing_time_moneyness"]
    model_version: Literal["v3_candidate_3m_v1"]
    smile_policy: CandidatePolicy


class CandidatePriceResponse(LatestNormalPriceResponse):
    model: Literal["latest_normal_v3_candidate_3m"]
    version: Literal["v3_candidate_3m_v1"]
    caplets: list[CandidateCaplet]
    conventions: CandidatePriceConventions

    @model_validator(mode="after")
    def require_tenor_pricing_policy(self) -> CandidatePriceResponse:
        expected = (
            "nonflat_variance_wing"
            if self.index_tenor == "3M"
            else "d4_below_native_boundary_native_cubic"
        )
        expected_time = (
            "candidate_fixing_time_moneyness" if self.index_tenor == "3M" else "quadratic"
        )
        if (
            self.conventions.smile_policy != expected
            or self.conventions.time_method != expected_time
        ):
            raise ValueError("Candidate price conventions must match the priced tenor")
        for coupon in self.caplets:
            if (
                self.index_tenor == "3M"
                and coupon.status == "future"
                and (
                    coupon.vol_source != "candidate_fixing_time_moneyness"
                    or coupon.parameters is not None
                )
            ):
                raise ValueError(
                    "Candidate 3M coupons require their actual evaluator, without fictitious SABR parameters"
                )
        return self
