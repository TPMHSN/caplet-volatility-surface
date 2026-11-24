"""
Pydantic schemas for swap instruments and pricing requests/responses.
"""

import math
from typing import Optional, Literal, List
from datetime import date as dt_date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from curves.helpers import CurveInterpolation, CurveInterpolationMethod
from curves.schemas import CurveQuoteExclusion
from pricing.diagnostics import BucketedCurveDeltaDiagnostics


class AmortizationType(str, Enum):
    """Amortization schedule types."""

    BULLET = "bullet"
    LINEAR = "linear"
    CUSTOM = "custom"


class SwapType(str, Enum):
    """Swap type from fixed leg perspective."""

    PAYER = "payer"  # Pay fixed, receive float
    RECEIVER = "receiver"  # Receive fixed, pay float


class BusinessDayConvention(str, Enum):
    """Schedule business-day adjustment convention."""

    EXACT = "exact"
    FOLLOWING = "following"
    PRECEDING = "preceding"
    MODIFIED_FOLLOWING = "modified_following"


class RollConvention(str, Enum):
    """Schedule roll convention."""

    AUTO = "auto"
    END_OF_MONTH = "end_of_month"
    DAY_30 = "day_30"
    DAY_29 = "day_29"


class ScheduleConvention(BaseModel):
    """Schedule generation and date-adjustment convention."""

    business_day_convention: BusinessDayConvention = Field(
        default=BusinessDayConvention.MODIFIED_FOLLOWING,
        description="Business-day adjustment convention for schedule dates.",
    )
    roll_convention: RollConvention = Field(
        default=RollConvention.AUTO,
        description="Roll rule for generated schedule dates.",
    )


class NotionalStep(BaseModel):
    """A single notional step in a custom amortization schedule."""

    date: dt_date = Field(..., description="Date when notional changes")
    notional: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Notional amount from this date",
    )


class AmortizationSchedule(BaseModel):
    """Amortization schedule definition."""

    amortization_type: AmortizationType = Field(
        default=AmortizationType.BULLET, description="Type of amortization"
    )
    custom_notionals: Optional[List[NotionalStep]] = Field(
        default=None, description="Custom notional schedule (required if type=custom)"
    )
    fixed_leg_notionals: Optional[List[float]] = Field(
        default=None, description="Per-period notionals for fixed leg (overrides type)"
    )
    float_leg_notionals: Optional[List[float]] = Field(
        default=None, description="Per-period notionals for float leg (overrides type)"
    )

    @model_validator(mode="after")
    def _validate_leg_notionals(self) -> "AmortizationSchedule":
        for leg_name, values in (
            ("fixed", self.fixed_leg_notionals),
            ("floating", self.float_leg_notionals),
        ):
            if values is not None and any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                raise ValueError(f"{leg_name} leg notionals must be finite and non-negative")
        return self


class RFRParameters(BaseModel):
    """Optional overrides for RFR (overnight index) floating leg parameters."""

    payment_lag: Optional[int] = Field(
        default=None, ge=0, description="Payment lag in business days (e.g., 2 for SOFR)"
    )
    lookback_days: Optional[int] = Field(
        default=None, ge=0, description="Lookback/observation lag in days"
    )
    lockout_days: Optional[int] = Field(
        default=None, ge=0, description="Lockout days (no rate changes near payment)"
    )
    observation_shift: Optional[bool] = Field(
        default=None, description="Apply observation shift (True) vs lockout (False)"
    )
    averaging_method: Optional[Literal["compound", "simple"]] = Field(
        default=None, description="Rate averaging: 'compound' (standard) or 'simple'"
    )


class SwapDefinition(BaseModel):
    """Complete swap instrument definition."""

    currency: str = Field(..., description="Currency (EUR, USD, etc.)")
    swap_type: SwapType = Field(..., description="Payer or receiver swap")

    effective_date: dt_date = Field(..., description="Swap effective date")
    maturity_date: dt_date = Field(..., description="Swap maturity date")

    notional: float = Field(..., gt=0, description="Initial notional amount")
    fixed_rate: float = Field(..., description="Fixed rate (as decimal, e.g., 0.025 for 2.5%)")

    float_index_tenor: str = Field(
        ...,
        description="Floating index tenor (e.g., 'SOFR', 'SONIA', '3M', '6M')",
        pattern=r"^(\d+(M|Y)|SOFR|SONIA|OIS)$",
    )
    float_spread: float = Field(default=0.0, description="Floating leg spread (as decimal)")
    convention_preset_id: Optional[str] = Field(
        default=None,
        description="Versioned convention preset applied before explicit trade overrides.",
    )
    fixing_calendar: Optional[str] = Field(
        default=None,
        description="Calendar identifier used by the overnight fixing index.",
    )
    schedule_calendar: Optional[str] = Field(
        default=None,
        description="Calendar identifier used to generate accrual schedules.",
    )
    payment_calendar: Optional[str] = Field(
        default=None,
        description="Calendar identifier used to adjust payment dates.",
    )

    rfr_params: Optional[RFRParameters] = Field(
        default=None,
        description="Override RFR parameters (payment lag, lookback, etc.). Auto-detected from config if None.",
    )

    # Day count conventions
    fixed_day_count: str = Field(
        default="ACT/360",
        description="Fixed leg day count convention (ACT/360, ACT/365, 30/360, etc.)",
    )
    float_day_count: str = Field(
        default="ACT/360",
        description="Float leg day count convention (ACT/360, ACT/365, 30/360, etc.)",
    )

    # Payment frequencies
    fixed_frequency: Optional[str] = Field(
        default=None,
        description="Fixed leg payment frequency. If not specified, matches float_index_tenor.",
    )
    float_frequency: Optional[str] = Field(
        default=None,
        description="Float leg payment frequency (defaults to float_index_tenor if not specified)",
    )

    # Roll convention
    first_roll_date: Optional[dt_date] = Field(
        default=None, description="First roll date for stub period (optional)"
    )
    last_roll_date: Optional[dt_date] = Field(
        default=None, description="Last roll date for back stub period (optional)"
    )
    schedule_convention: ScheduleConvention = Field(
        default_factory=ScheduleConvention,
        description="Business-day and roll conventions for generated schedules.",
    )

    custom_fixed_rates: Optional[List[float]] = Field(
        default=None,
        description="Per-period fixed rates (overrides fixed_rate). Length must match fixed leg periods.",
    )

    amortization: AmortizationSchedule = Field(
        default_factory=AmortizationSchedule, description="Amortization schedule"
    )


class CashflowItem(BaseModel):
    """Individual cashflow details."""

    date: dt_date = Field(..., description="Payment date")
    notional: float = Field(..., description="Notional for this period")

    accrual_start: dt_date = Field(..., description="Accrual start date")
    accrual_end: dt_date = Field(..., description="Accrual end date")

    rate: float = Field(..., description="Rate for this period (as decimal)")
    amount: float = Field(..., description="Cashflow amount")

    discount_factor: float = Field(..., description="Discount factor at payment date")
    present_value: float = Field(..., description="Present value of this cashflow")
    pv01: float = Field(
        ..., description="PV01 for this period (notional x year_fraction x df x 1bp)"
    )
    is_past: bool = Field(
        default=False,
        description="Whether this cashflow has already been paid (payment_date <= valuation_date)",
    )


class LegCashflows(BaseModel):
    """Cashflows for one leg of the swap."""

    leg_type: Literal["fixed", "float"] = Field(..., description="Type of leg")
    cashflows: List[CashflowItem] = Field(default_factory=list, description="List of cashflows")
    total_npv: float = Field(..., description="Total NPV of this leg")
    total_pv01: float = Field(..., description="Total PV01 of this leg")


class SwapGreeks(BaseModel):
    """Greeks/sensitivities for the swap."""

    dv01: float = Field(
        ...,
        description=(
            "Signed local market-quote DV01 = -1e-4 * sum(dV/dq), computed from the "
            "implicit calibration Jacobian with calibration fixings held constant"
        ),
    )
    pv01: float = Field(..., description="Signed fixed-leg coupon PV01")
    fixed_leg_dv01: float = Field(..., description="Signed fixed leg share of dv01")
    float_leg_dv01: float = Field(..., description="Signed float leg share of dv01")
    all_curves_dv01: float = Field(
        default=0.0,
        description="Alias of dv01: discount and forward market curves bumped together",
    )
    discount_dv01: float = Field(
        default=0.0,
        description="Signed local DV01 contribution from discount-role market quotes",
    )
    forward_dv01: float = Field(
        default=0.0,
        description="Signed local DV01 contribution from projection-role market quotes",
    )
    cross_dv01: float = Field(
        default=0.0,
        description="Residual all_curves_dv01 - discount_dv01 - forward_dv01",
    )
    fixed_leg_discount_dv01: float = Field(
        default=0.0,
        description="Signed fixed-leg local DV01 from discount-role market quotes",
    )
    fixed_leg_forward_dv01: float = Field(
        default=0.0,
        description="Signed fixed-leg local DV01 from projection-role market quotes",
    )
    float_leg_discount_dv01: float = Field(
        default=0.0,
        description="Signed float-leg local DV01 from discount-role market quotes",
    )
    float_leg_forward_dv01: float = Field(
        default=0.0,
        description="Signed float-leg local DV01 from projection-role market quotes",
    )
    fixed_leg_pv01: float = Field(..., description="Signed fixed leg coupon PV01")
    float_leg_pv01: float = Field(..., description="Signed float leg coupon PV01")

    @model_validator(mode="after")
    def _default_all_curves_alias(self) -> "SwapGreeks":
        if self.all_curves_dv01 == 0.0 and self.dv01 != 0.0:
            self.all_curves_dv01 = self.dv01
        return self


class SwapAccruedAmounts(BaseModel):
    """Accrued coupon and clean price for the swap."""

    fixed_leg_accrued: float = Field(..., description="Signed fixed leg accrued coupon")
    float_leg_accrued: float = Field(..., description="Signed float leg accrued coupon")
    net_accrued: float = Field(..., description="Signed net accrued coupon")
    clean_price: float = Field(..., description="NPV minus signed net accrued coupon")


class SwapPricingRequest(BaseModel):
    """Request to price a swap."""

    swap: SwapDefinition = Field(..., description="Swap to price")
    valuation_date: dt_date = Field(..., description="Date for pricing/discounting")

    compute_greeks: bool = Field(default=True, description="Whether to compute Greeks (DV01)")
    compute_greeks_diagnostics: bool = Field(
        default=False,
        description="Whether to compute detailed DV01 diagnostics by bucket and curve quote",
    )
    compute_cashflows: bool = Field(default=True, description="Whether to extract cashflows")

    discount_curve_key: Optional[str] = Field(
        default=None,
        description="Curve key for discounting ('OIS', '3M', '6M'...). Defaults to OIS.",
    )
    forward_curve_key: Optional[str] = Field(
        default=None,
        description="Forward curve key for forecasting ('3M', '6M'...). Defaults to float_index_tenor.",
    )
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD,
        description="Interpolation used to rebuild the discount curve for this calculation.",
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD,
        description="Interpolation used to rebuild projection curves for this calculation.",
    )
    rate_shock_bps: Optional[float] = Field(
        default=None,
        description="Parallel market quote shock in bps, rebuilding discount and forward curves; EUR calibration IBOR fixings constant. Zero-rate fallback without raw quotes.",
    )


class SwapPricingResult(BaseModel):
    """Complete swap pricing result."""

    curve_quote_exclusions: list[CurveQuoteExclusion] = Field(default_factory=list)

    currency: str = Field(..., description="Currency of all monetary result amounts")

    # Core pricing
    npv: float = Field(..., description="Net Present Value")
    fair_rate: float = Field(..., description="Par/fair fixed rate")

    fixed_leg_npv: float = Field(..., description="Fixed leg NPV")
    float_leg_npv: float = Field(..., description="Float leg NPV")
    accrued: SwapAccruedAmounts = Field(..., description="Accrued coupon and clean price")

    # Dates
    curve_date: dt_date = Field(..., description="Curve reference date (market data date)")
    valuation_date: dt_date = Field(..., description="Pricing/discounting date")
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )

    # Optional details
    greeks: Optional[SwapGreeks] = Field(default=None, description="Greeks if requested")
    greeks_diagnostics: Optional[BucketedCurveDeltaDiagnostics] = Field(
        default=None,
        description="Optional bucketed delta diagnostics if Greeks are requested",
    )
    cashflows: Optional[List[LegCashflows]] = Field(
        default=None, description="Detailed cashflows if requested"
    )
    warnings: Optional[List[str]] = Field(
        default=None, description="Pricing warnings (e.g., missing fixings)"
    )


class FairRateRequest(BaseModel):
    """Request to compute only the fair rate."""

    swap: SwapDefinition = Field(..., description="Swap definition")
    valuation_date: dt_date = Field(..., description="Valuation date")

    discount_curve_key: Optional[str] = Field(
        default=None,
        description="Curve key for discounting ('OIS', '3M', '6M'...). Defaults to OIS.",
    )
    forward_curve_key: Optional[str] = Field(
        default=None,
        description="Forward curve key for forecasting ('3M', '6M'...). Defaults to float_index_tenor.",
    )
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )


class FairRateResult(BaseModel):
    """Fair rate result."""

    curve_quote_exclusions: list[CurveQuoteExclusion] = Field(default_factory=list)

    fair_rate: float = Field(..., description="Par/fair fixed rate")
    curve_date: dt_date = Field(..., description="Curve reference date")
    valuation_date: dt_date = Field(..., description="Valuation date")
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )


class SchedulePeriod(BaseModel):
    """A single schedule period."""

    period_number: int = Field(..., description="Period index (1-based)")
    start_date: dt_date = Field(..., description="Period start date")
    end_date: dt_date = Field(..., description="Period end date")


class SchedulePreviewRequest(BaseModel):
    """Request to preview swap schedules without pricing."""

    currency: str = Field(..., description="Currency (EUR, USD, etc.)")
    effective_date: dt_date = Field(..., description="Swap effective date")
    maturity_date: dt_date = Field(..., description="Swap maturity date")
    float_index_tenor: str = Field(
        ...,
        description="Floating index tenor (e.g., 'SOFR', 'SONIA', '3M', '6M')",
        pattern=r"^(\d+(M|Y)|SOFR|SONIA|OIS)$",
    )
    fixed_frequency: Optional[str] = Field(
        default=None,
        description="Optional fixed leg schedule frequency for preview.",
    )
    float_frequency: Optional[str] = Field(
        default=None,
        description="Optional floating leg schedule frequency for preview.",
    )
    first_roll_date: Optional[dt_date] = Field(
        default=None, description="First roll date for stub period (optional)"
    )
    last_roll_date: Optional[dt_date] = Field(
        default=None, description="Last roll date for back stub period (optional)"
    )
    schedule_convention: ScheduleConvention = Field(
        default_factory=ScheduleConvention,
        description="Business-day and roll conventions for generated schedules.",
    )


class SchedulePreviewResponse(BaseModel):
    """Response containing schedule periods for both legs."""

    fixed_leg_periods: List[SchedulePeriod] = Field(..., description="Fixed leg periods")
    float_leg_periods: List[SchedulePeriod] = Field(..., description="Float leg periods")


class CurveStatusDetail(BaseModel):
    """Available curve metadata for one currency."""

    curve_date: dt_date = Field(..., description="Curve reference date")
    forward_tenors: List[str] = Field(default_factory=list, description="Available forward tenors")


class CurvesStatusResponse(BaseModel):
    """Status of available pricing curves across currencies."""

    available_currencies: List[str] = Field(default_factory=list)
    details: dict[str, CurveStatusDetail] = Field(default_factory=dict)
