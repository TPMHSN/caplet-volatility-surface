"""
Pydantic schemas for curve-related API requests and responses.
"""

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .helpers import CurveInterpolationMethod


class CurveQuoteExclusion(BaseModel):
    """One future left out of a calibration: a pillar collision with a deposit, or the fallback."""

    curve_key: str
    source_label: str
    instrument_type: Literal["future"] = "future"
    quote_index: int
    market_rate: float
    pillar_date: date
    conflicting_source_label: str
    conflicting_pillar_date: date
    reason: str


class CurveBuildRequest(BaseModel):
    """Request to build curves."""

    currency: str = Field(default="EUR", description="Currency code")
    forward_tenors: List[str] = Field(
        default=["3M", "6M"], description="Forward curve tenors to build"
    )
    interpolation: CurveInterpolationMethod = Field(
        default="SmoothForward",
        description="Interpolation method: LinearForward, SmoothForward or NaturalLogCubic",
    )
    enforce_quality: bool = Field(
        default=True,
        description="If true, blocks curve build when market data QC reports blocking issues.",
    )


class CurveBuildResponse(BaseModel):
    """Response after building curves."""

    success: bool
    message: str
    currency: str
    valuation_date: date
    curves_built: List[str]


class CurvePoint(BaseModel):
    """Single point on a curve."""

    date: date
    days: int
    year_fraction: float
    zero_rate: Optional[float] = None
    discount_factor: Optional[float] = None
    forward_rate: Optional[float] = None


CurveViewRole = Literal["discount", "projection", "implied_ois_forward"]


class CurveViewContext(BaseModel):
    """Identity of the curve context a view was actually read from.

    A view is either the published reference or an isolated variant rebuilt from the same
    quotes; a variant is never published and never replaces the reference.
    """

    source: Literal["reference", "variant"] = Field(
        description="'reference' when the published curve set was read, 'variant' otherwise"
    )
    role: CurveViewRole = Field(
        description=(
            "'discount' for the OIS discount curve, 'projection' for a term forward curve "
            "bootstrapped against the discount curve, 'implied_ois_forward' for a forward "
            "implied by the OIS projection curve: the schedule of an OIS curve, the historical "
            "USD/GBP implied identifiers (USD_3M, GBP_3M) and the RFR aliases (USD_SOFR, "
            "GBP_SONIA)"
        )
    )
    forward_role: Optional[CurveViewRole] = Field(
        default=None,
        description=(
            "Role of the curve the forward series was read on, which can differ from 'role': an "
            "OIS view read at a horizon keeps 'discount' for its discount factors and zero rates "
            "and reads its forwards as 'implied_ois_forward' on the OIS projection curve; "
            "'projection' for a term forward curve; null when the view carries no forward series"
        ),
    )
    discount_interpolation: CurveInterpolationMethod
    forward_interpolation: CurveInterpolationMethod
    curve_date: date = Field(description="Market data date the curves were built on")
    publication_generation: int = Field(
        description="Publication generation of the reference curve set this view derives from"
    )
    reference_fingerprint: Optional[str] = Field(
        default=None, description="Curve fingerprint of the published reference, when known"
    )
    context_fingerprint: Optional[str] = Field(
        default=None, description="Curve fingerprint of the context actually read, when known"
    )
    rate_definition: str = Field(description="Definition of the rates returned by the view")
    quote_exclusions: List[CurveQuoteExclusion] = Field(
        default_factory=list,
        description=(
            "Calibration exclusions recorded by the context read: the tenor curves, the "
            "discount OIS curve and, on a mixed method pair, the OIS projection curve (report "
            "key forward_OIS), whose exclusions are those of the forwards the view reads"
        ),
    )


class CurveData(BaseModel):
    """Complete curve data for visualization."""

    curve_id: str
    currency: str
    curve_type: str  # 'OIS', 'Forward'
    tenor: Optional[str] = None
    forward_tenor: Optional[str] = Field(
        default=None,
        description=(
            "Horizon of the implied simple-forward series: 'ON' for the expected overnight "
            "fixing, '1M' to '12M' for the forward over that horizon; null without a series"
        ),
    )
    valuation_date: date
    interpolation: Optional[CurveInterpolationMethod] = Field(
        default=None,
        description="Interpolation method of the curve read (the stored one for the reference)",
    )
    context: Optional[CurveViewContext] = Field(
        default=None, description="Identity of the context the points were read from"
    )
    points: List[CurvePoint]


class CurveCalibrationResidualData(BaseModel):
    """One market quote and the quote implied by the calibrated curve."""

    source_label: str
    tenor: str
    instrument_type: str
    convention_id: str
    quote_provenance: str
    calibration_role: str
    market_rate: float
    implied_rate: float
    residual_bps: float
    pillar_date: date


class CurveFixingObservationData(BaseModel):
    """A fixing retained for audit but excluded from bootstrap helpers."""

    source_label: str
    tenor: str
    convention_id: str
    quote_provenance: str
    calibration_role: str
    market_rate: float
    used_in_bootstrap: bool


class CurveCalibrationResponse(BaseModel):
    """Serializable quote-level calibration evidence for one built curve."""

    curve_id: str
    tolerance_bps: float
    max_abs_residual_bps: float
    passed: bool
    solver: Literal["quantlib_iterative", "global_newton"] = Field(
        default="quantlib_iterative",
        description=(
            "Calibration solver: QuantLib's iterative bootstrap, or a joint Newton solve of the "
            "same helpers when that bootstrap did not reprice the quotes (SmoothForward)"
        ),
    )
    residuals: List[CurveCalibrationResidualData]
    fixing_observations: List[CurveFixingObservationData]


class CurveInfo(BaseModel):
    """Information about a single curve."""

    id: str = Field(description="Curve identifier (e.g., 'EUR_OIS', 'EUR_3M')")
    type: str = Field(description="Curve type: 'OIS' or 'Forward'")
    currency: str = Field(description="Currency code")
    tenor: Optional[str] = Field(default=None, description="Tenor for forward curves (e.g., '3M')")
    interpolation: Optional[CurveInterpolationMethod] = Field(
        default=None, description="Interpolation method the stored curve set was built with"
    )
    discount_interpolation: Optional[CurveInterpolationMethod] = Field(
        default=None, description="Discount interpolation of the published reference"
    )
    forward_interpolation: Optional[CurveInterpolationMethod] = Field(
        default=None, description="Forward interpolation of the published reference"
    )
    publication_generation: Optional[int] = Field(
        default=None,
        description="Publication generation of the reference; it changes on every publication",
    )
    curve_fingerprint: Optional[str] = Field(
        default=None, description="Curve fingerprint of the published reference, when known"
    )
    forward_horizons: List[str] = Field(
        default_factory=list,
        description=(
            "Horizons the curve can be read at: 'ON', '1M', '3M', '6M' and '12M' for an OIS "
            "curve, its own tenor for a term or implied forward curve"
        ),
    )


class CurveListResponse(BaseModel):
    """List of available curves."""

    curves: List[CurveInfo]  # Fixed m7: Use typed model instead of Dict[str, str]


class ForwardScheduleRequest(BaseModel):
    """Request to generate a forward rate schedule."""

    curve_id: str = Field(..., description="Curve identifier (e.g., 'EUR_6M')")
    start_date: date = Field(..., description="Start date of the schedule")
    end_date: date = Field(..., description="End date of the schedule")
    roll_period: str = Field(default="6M", description="Roll period (e.g., '3M', '6M')")
    first_roll_date: Optional[date] = Field(
        default=None,
        description="Optional first coupon roll date; legacy schedule generation is kept when omitted",
    )
    discount_interpolation: Optional[CurveInterpolationMethod] = Field(
        default=None,
        description=(
            "Explore an isolated variant discounted with this method; omitted reads the "
            "published reference. The variant is never published."
        ),
    )
    forward_interpolation: Optional[CurveInterpolationMethod] = Field(
        default=None,
        description=(
            "Explore an isolated variant projected with this method; omitted reads the "
            "published reference. The variant is never published."
        ),
    )


class ForwardSchedulePoint(BaseModel):
    """Single point in a forward rate schedule."""

    period_number: int
    start_date: date
    end_date: date
    year_fraction: float
    forward_rate: float = Field(..., description="Simple forward rate in decimal")
    rate_source: Literal["historical_fixing", "partially_observed", "projected_forward"] = Field(
        ...,
        description=(
            "Whether the rate is realized (historical_fixing: the stored fixing, or every "
            "overnight fixing of the period compounded), partly realized (partially_observed: "
            "published overnight fixings compounded, then the forward of the curve for the rest "
            "of an overnight period) or projected (projected_forward)"
        ),
    )
    fixing_date: Optional[date] = Field(
        default=None,
        description=(
            "Historical index fixing date when the rate is the realized fixing of an IBOR "
            "period; None for a compounded overnight period, which has one fixing per business "
            "day"
        ),
    )


class ForwardScheduleResponse(BaseModel):
    """Response with forward rate schedule."""

    curve_id: str
    currency: str
    roll_period: str
    first_roll_date: Optional[date] = None
    valuation_date: date
    interpolation: Optional[CurveInterpolationMethod] = Field(
        default=None, description="Interpolation method of the curve the schedule was read from"
    )
    context: Optional[CurveViewContext] = Field(
        default=None, description="Identity of the context the schedule was read from"
    )
    points: List[ForwardSchedulePoint]
