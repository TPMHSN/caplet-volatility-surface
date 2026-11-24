"""
Pydantic schemas for volatility surfaces, cap/floor pricing, and swaption vol cube.
"""

from datetime import date as dt_date
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from curves.helpers import CurveInterpolation, CurveInterpolationMethod
from curves.schemas import CurveQuoteExclusion
from instruments.schemas import (
    AmortizationType,
    AmortizationSchedule,
    LegCashflows,
    RFRParameters,
    ScheduleConvention,
)
from pricing.diagnostics import BucketedCurveDeltaDiagnostics, SwaptionGreeksDiagnostics
from type_defs import JsonObject

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CapFloorType(str, Enum):
    """Cap, Floor, or Collar."""

    CAP = "cap"
    FLOOR = "floor"
    COLLAR = "collar"


# Input domain of the Gaussian short-rate models (MODTAUX L2): a reversion of 500 % a year or a
# normal volatility of 50 % a year is already far outside any market; the bounds only exclude
# input errors (negative or absurd values), not market choices.
MAX_MODEL_REVERSION = 5.0
MAX_MODEL_VOLATILITY = 0.5


class OptionletStructureMode(str, Enum):
    """Economic interpretation of a two-leg cap/floor optionlet structure."""

    STANDARD_TUNNEL = "standard_tunnel"
    CUSTOM = "custom"


class VolType(str, Enum):
    """Volatility quoting convention."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"


class ModelType(str, Enum):
    """Short-rate model type."""

    HW1F = "hw1f"
    G2PP = "g2pp"
    GSR = "gsr"


class DiagonalStrategy(str, Enum):
    """Diagonal calibration point generation strategy."""

    GRID_ONLY = "grid_only"
    FULL_ANNUAL = "full_annual"


class SmileMethod(str, Enum):
    """Smile construction method."""

    INTERPOLATED = "interpolated"
    SABR_NORMAL = "sabr_normal"
    CONVEX_NORMAL = "convex_normal"
    ATM_ONLY = "atm_only"


class StrikeInterpolation(str, Enum):
    """Strike interpolation of the interpolated swaption smile (PCHIP by default)."""

    PCHIP = "pchip"
    LINEAR = "linear"


class CapFloorSmileMethod(str, Enum):
    """Cap/Floor vol surface smile method."""

    BILINEAR = "bilinear"
    SABR_NORMAL = "sabr_normal"


# ---------------------------------------------------------------------------
# Vol Surface Build
# ---------------------------------------------------------------------------


class VolSurfaceBuildRequest(BaseModel):
    """Request to build a vol surface from stored market data."""

    currency: str = Field(..., description="Currency code (e.g. EUR)")
    vol_type: VolType = Field(default=VolType.NORMAL, description="Vol quoting convention")
    enforce_quality: bool = Field(
        default=True,
        description="If true, blocks build when market data QC reports blocking issues.",
    )
    smile_method: CapFloorSmileMethod = Field(
        default=CapFloorSmileMethod.BILINEAR,
        description="Surface construction method: bilinear interpolation or SABR Normal smile",
    )


class VolSurfaceBuildResponse(BaseModel):
    """Response after building a vol surface."""

    success: bool
    message: str
    surface_id: str
    currency: str
    vol_type: VolType
    num_expiries: int
    num_strikes: int
    smile_method: Optional[CapFloorSmileMethod] = None


class VolSurfaceInfo(BaseModel):
    """Summary info for a built vol surface."""

    surface_id: str
    currency: str
    vol_type: VolType
    valuation_date: dt_date
    num_expiries: int
    num_strikes: int
    smile_method: Optional[CapFloorSmileMethod] = None


class VolSurfaceListResponse(BaseModel):
    """Response listing all built surfaces."""

    surfaces: List[VolSurfaceInfo]


class SabrCalibrationParam(BaseModel):
    """SABR calibration parameters for a single expiry."""

    expiry: str
    alpha: float
    beta: float
    rho: float
    nu: float
    calibration_error: float
    frozen: bool = False
    frozen_pre_first: bool = False
    shift: Optional[float] = None
    atm_input: Optional[float] = None
    atm_model: Optional[float] = None
    atm_gap_bp: Optional[float] = None
    objective_value: Optional[float] = None


class VolSurfaceData(BaseModel):
    """Full surface data for visualization."""

    surface_id: str
    currency: str
    vol_type: VolType
    valuation_date: dt_date
    expiries: List[str]
    strikes: List[float]
    volatilities: List[List[float]]
    atm_vols: Optional[List[float]] = None
    smile_method: Optional[CapFloorSmileMethod] = None
    sabr_params: Optional[List[SabrCalibrationParam]] = None


# ---------------------------------------------------------------------------
# Cap/Floor Instrument
# ---------------------------------------------------------------------------


class CapFloorDefinition(BaseModel):
    """Cap or floor instrument definition."""

    currency: str = Field(..., description="Currency code")
    cap_floor_type: CapFloorType = Field(..., description="Cap or floor")
    effective_date: dt_date = Field(..., description="Start date")
    maturity_date: dt_date = Field(..., description="End date")
    notional: float = Field(..., gt=0, description="Notional amount")
    strike: float = Field(..., description="Strike rate (decimal, e.g. 0.02 for 2%)")
    custom_strikes: Optional[List[float]] = Field(
        default=None,
        description="Per-period strikes (overrides strike). Length must match number of caplet periods.",
    )
    floor_strike: Optional[float] = Field(
        default=None,
        description="Floor strike for collars (decimal). Required when cap_floor_type=collar.",
    )
    custom_floor_strikes: Optional[List[float]] = Field(
        default=None, description="Per-period floor strikes for collars (overrides floor_strike)."
    )
    optionlet_structure_mode: OptionletStructureMode = Field(
        default=OptionletStructureMode.STANDARD_TUNNEL,
        description=(
            "Standard tunnels require cap strike > floor strike on every period; custom "
            "structures permit any explicitly requested cap/floor strike relationship."
        ),
    )
    index_tenor: str = Field(..., description="Underlying index tenor (e.g. 'SOFR', 'SONIA', '6M')")
    day_count: str = Field(default="ACT/360", description="Day count convention")
    position: Literal["long", "short"] = Field(
        default="long", description="Long = buyer, Short = seller"
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
    amortization_type: AmortizationType = Field(
        default=AmortizationType.BULLET, description="Amortization type"
    )
    custom_notionals: Optional[List[float]] = Field(
        default=None, description="Custom notional profile per period"
    )
    rfr_params: Optional["RFRParameters"] = Field(
        default=None,
        description="Override RFR parameters (payment lag, lookback, etc.). Auto-detected from config if None.",
    )


# ---------------------------------------------------------------------------
# Cap/Floor Pricing
# ---------------------------------------------------------------------------


class CapFloorGreeks(BaseModel):
    """Greeks for a cap/floor."""

    dv01: float = Field(
        ...,
        description=(
            "Local market-quote DV01 = -1e-4 * sum(dV/dq), computed from the implicit "
            "calibration Jacobian with calibration fixings and optionlet volatility frozen"
        ),
    )
    # Historical product-specific sensitivity, retained for compatibility.
    # This is distinct from market-quote DV01. Use the position-signed dv01
    # field for every DV01 display, export, and aggregation.
    delta: float = Field(
        ...,
        description="Legacy delta, not DV01: NPV(base) - NPV(+1bp zero-spread curve shift)",
    )
    vega: float = Field(..., description="Vega - NPV change for +1bp vol shift")
    gamma: float = Field(default=0.0, description="Gamma - reserved for future use")


class CapletFloorletItem(BaseModel):
    """Detail for a single caplet/floorlet."""

    period_number: int
    accrual_start: dt_date
    accrual_end: dt_date
    payment_date: dt_date
    notional: float
    forward_rate: float
    strike: float
    floor_strike: Optional[float] = Field(default=None, description="Floor strike (collars only)")
    volatility: float
    npv: float
    discount_factor: float
    year_fraction: float = Field(default=0.0, description="Accrual period year fraction (tau)")
    intrinsic_value: float = Field(
        default=0.0, description="Intrinsic payoff component (discounted)"
    )
    time_value: float = Field(default=0.0, description="Time/optionality premium (NPV - intrinsic)")
    is_past: bool = Field(
        default=False, description="Whether this caplet/floorlet has already been paid"
    )


class CapFloorPricingRequest(BaseModel):
    """Request to price a cap or floor."""

    cap_floor: CapFloorDefinition = Field(..., description="Instrument definition")
    valuation_date: dt_date = Field(..., description="Pricing date")
    compute_greeks: bool = Field(default=True, description="Compute delta/vega")
    compute_greeks_diagnostics: bool = Field(
        default=False, description="Compute detailed DV01 diagnostics by bucket and curve quote"
    )
    compute_caplets: bool = Field(default=True, description="Compute caplet breakdown")
    surface_id: Optional[str] = Field(
        default=None,
        description="Optional explicit cap/floor vol surface ID. Defaults to the latest built surface for the currency.",
    )
    discount_curve_key: Optional[str] = Field(
        default=None,
        description="Curve key for discounting ('OIS', '3M', '6M'...). Defaults to OIS.",
    )
    forward_curve_key: Optional[str] = Field(
        default=None,
        description="Forward curve key for forecasting ('3M', '6M'...). Defaults to index_tenor.",
    )
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    rate_shock_bps: Optional[FiniteFloat] = Field(
        default=None,
        description=(
            "Parallel market-quote rate shock in bps: selected rate-equivalent raw quotes move "
            "and the curves are rebootstrapped with volatility frozen; zero-rate fallback "
            "without raw quotes"
        ),
    )
    vol_shock_bps: Optional[FiniteFloat] = Field(
        default=None,
        description="Parallel vol shock in bps, applied to raw vol quotes on unshocked curves",
    )


CapFloorImpliedVolatilityDiagnostic = Literal[
    # A collar premium (cap minus floor) is not a monotone function of one common vol:
    # the market quotes one flat vol per cap or floor, so each leg is inverted separately.
    "implied_volatility_not_defined_for_collar",
    # The cap or floor flat-vol inversion found no root inside the supported vol range.
    "implied_volatility_solver_failed",
]


class CapFloorPricingResult(BaseModel):
    """Complete cap/floor pricing result."""

    curve_quote_exclusions: list[CurveQuoteExclusion] = Field(default_factory=list)

    npv: float = Field(..., description="Net present value")
    atm_rate: float = Field(..., description="ATM par swap rate (true ATM strike)")
    premium_bps: float = Field(..., description="Premium en bp running (NPV / PV01)")
    premium_pct_notional: float = Field(
        default=0.0, description="Premium en % du notionnel CRD (NPV / Notional CRD × 100)"
    )
    notional_crd: float = Field(
        default=0.0, description="Notionnel en cours à la date de valorisation"
    )
    pv01: float = Field(
        default=0.0, description="PV01 of the underlying swap used for premium normalisation"
    )
    position: str = Field(default="long", description="Position: long or short")
    curve_date: dt_date
    valuation_date: dt_date
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    vol_type: VolType
    surface_id: Optional[str] = Field(
        default=None,
        description="Cap/floor vol surface ID used for pricing.",
    )
    surface_smile_method: Optional[CapFloorSmileMethod] = Field(
        default=None,
        description="Cap/floor surface construction method used for pricing.",
    )
    implied_volatility: Optional[float] = Field(
        default=None,
        description=(
            "Cap or floor only: implied flat volatility, the single vol reproducing the NPV "
            "(decimal, surface vol type). None for a collar, whose premium has no single "
            "flat vol."
        ),
    )
    cap_leg_implied_volatility: Optional[float] = Field(
        default=None,
        description=(
            "Collar only: flat volatility (decimal, surface vol type) reproducing the price "
            "of the long cap leg alone on the collar's floating leg, notionals and cap strikes."
        ),
    )
    floor_leg_implied_volatility: Optional[float] = Field(
        default=None,
        description=(
            "Collar only: flat volatility (decimal, surface vol type) reproducing the price "
            "of the floor leg alone on the collar's floating leg, notionals and floor strikes."
        ),
    )
    implied_volatility_diagnostic: Optional[CapFloorImpliedVolatilityDiagnostic] = Field(
        default=None,
        description="Reason implied_volatility is unavailable.",
    )
    greeks: Optional[CapFloorGreeks] = None
    greeks_diagnostics: Optional[BucketedCurveDeltaDiagnostics] = Field(
        default=None,
        description="Optional bucketed delta diagnostics if Greeks are requested",
    )
    caplets: Optional[List[CapletFloorletItem]] = None
    warnings: Optional[List[str]] = Field(
        default=None, description="Pricing warnings (e.g., missing fixings)"
    )


class CapFloorZeroCostRequest(BaseModel):
    """Request to solve a zero-cost collar strike."""

    cap_floor: CapFloorDefinition = Field(..., description="Collar definition")
    valuation_date: dt_date = Field(..., description="Pricing date")
    solve_for: Literal["floor_strike", "cap_strike"] = Field(
        ...,
        description="Leg to solve: floor_strike keeps cap fixed, cap_strike keeps floor fixed.",
    )
    profile_mode: Literal["constant", "parallel_shift"] = Field(
        default="constant",
        description="Constant strike solve or one parallel shift applied to the solved profile.",
    )
    compute_caplets: bool = Field(
        default=True,
        description="Compute caplet breakdown in the final zero-cost pricing result.",
    )
    surface_id: Optional[str] = Field(
        default=None,
        description="Optional explicit cap/floor vol surface ID. Defaults to latest built surface.",
    )
    discount_curve_key: Optional[str] = Field(
        default=None,
        description="Curve key for discounting ('OIS', '3M', '6M'...). Defaults to OIS.",
    )
    forward_curve_key: Optional[str] = Field(
        default=None,
        description="Forward curve key for forecasting ('3M', '6M'...). Defaults to index_tenor.",
    )
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    rate_shock_bps: Optional[FiniteFloat] = Field(
        default=None,
        description=(
            "Parallel market-quote rate shock in bps: selected rate-equivalent raw quotes move "
            "and the curves are rebootstrapped with volatility frozen; zero-rate fallback "
            "without raw quotes"
        ),
    )
    vol_shock_bps: Optional[FiniteFloat] = Field(
        default=None,
        description="Parallel vol shock in bps, applied to raw vol quotes on unshocked curves",
    )


class CapFloorTargetSolveRequest(BaseModel):
    """Request to solve a collar strike for a target signed NPV."""

    cap_floor: CapFloorDefinition = Field(..., description="Collar definition")
    valuation_date: dt_date = Field(..., description="Pricing date")
    target_npv: FiniteFloat = Field(
        default=0.0,
        description="Target signed collar NPV in deal currency for the displayed position.",
    )
    solve_for: Literal["floor_strike", "cap_strike"] = Field(
        ...,
        description="Leg to solve: floor_strike keeps cap fixed, cap_strike keeps floor fixed.",
    )
    profile_mode: Literal["constant", "parallel_shift"] = Field(
        default="constant",
        description="Constant strike solve or one parallel shift applied to the solved profile.",
    )
    compute_caplets: bool = Field(
        default=True,
        description="Compute caplet breakdown in the final target pricing result.",
    )
    surface_id: Optional[str] = Field(
        default=None,
        description="Optional explicit cap/floor vol surface ID. Defaults to latest built surface.",
    )
    discount_curve_key: Optional[str] = Field(
        default=None,
        description="Curve key for discounting ('OIS', '3M', '6M'...). Defaults to OIS.",
    )
    forward_curve_key: Optional[str] = Field(
        default=None,
        description="Forward curve key for forecasting ('3M', '6M'...). Defaults to index_tenor.",
    )
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    rate_shock_bps: Optional[FiniteFloat] = Field(
        default=None,
        description=(
            "Parallel market-quote rate shock in bps: selected rate-equivalent raw quotes move "
            "and the curves are rebootstrapped with volatility frozen; zero-rate fallback "
            "without raw quotes"
        ),
    )
    vol_shock_bps: Optional[FiniteFloat] = Field(
        default=None,
        description="Parallel vol shock in bps, applied to raw vol quotes on unshocked curves",
    )


class CapFloorTargetSolveResult(BaseModel):
    """Result of a target collar solve."""

    target_npv: float = Field(..., description="Target signed NPV requested for the solve.")
    solve_for: Literal["floor_strike", "cap_strike"]
    profile_mode: Literal["constant", "parallel_shift"]
    solved_cap_floor: CapFloorDefinition = Field(
        ...,
        description="Final collar definition with solved strike/profile.",
    )
    solved_strike: Optional[float] = Field(
        default=None,
        description="Solved constant strike in decimal rate terms.",
    )
    solved_strikes: Optional[List[float]] = Field(
        default=None,
        description="Solved per-period strike profile in decimal rate terms.",
    )
    shift_bps: Optional[float] = Field(
        default=None,
        description="Parallel shift applied to the solved profile, in basis points.",
    )
    residual_npv: float = Field(..., description="Final residual NPV after the solve.")
    iterations: int = Field(..., ge=0, description="Number of solver iterations.")
    converged: bool = Field(..., description="Whether the solver met the residual tolerance.")
    pricing_result: CapFloorPricingResult = Field(
        ...,
        description="Final collar pricing result using the solved strike/profile.",
    )


class CapFloorZeroCostResult(CapFloorTargetSolveResult):
    """Result of a zero-cost collar solve."""


class CapFloorSmileSlice(BaseModel):
    """SABR smile data for a specific cap/floor expiry."""

    expiry: str
    strikes: List[float]
    volatilities: List[float]
    forward_rate: float
    sabr_params: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Swaption Vol Cube
# ---------------------------------------------------------------------------


class SwaptionVolCubeBuildRequest(BaseModel):
    """Request to build a swaption vol cube from stored market data."""

    currency: str = Field(..., description="Currency code (e.g. EUR)")
    index_tenor: str = Field(default="6M", description="Underlying index tenor (e.g. 1M, 3M, 6M)")
    vol_type: Literal[VolType.NORMAL] = Field(
        default=VolType.NORMAL, description="The swaption cube is quoted in normal vols only"
    )
    enforce_quality: bool = Field(
        default=True,
        description="If true, blocks build when market data QC reports blocking issues.",
    )
    smile_method: SmileMethod = Field(
        default=SmileMethod.INTERPOLATED,
        description="Smile method: interpolated, sabr_normal, convex_normal, or explicit atm_only without OTM",
    )
    strike_interpolation: StrikeInterpolation = Field(
        default=StrikeInterpolation.PCHIP,
        description="Strike scheme of the interpolated smile (ignored by sabr_normal)",
    )


class SwaptionVolCubeBuildResponse(BaseModel):
    """Response after building a swaption vol cube."""

    success: bool
    message: str
    cube_id: str
    currency: str
    index_tenor: str
    vol_type: str
    num_expiries: int
    num_tenors: int
    has_smile: bool
    smile_method: Optional[SmileMethod] = None
    smile_vol_type: Optional[str] = None
    strike_interpolation: Optional[StrikeInterpolation] = Field(
        default=None, description="Strike scheme of an interpolated smile; None otherwise"
    )
    conventions_profile: Optional[str] = Field(
        default=None, description="Conventions profile the cube was built under (e.g. EUR-SWPT-V1)"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Smile nodes to review: poor fit, parameter at a bound, or filled quotes.",
    )


class SwaptionVolCubeInfo(BaseModel):
    """Summary info for a built swaption vol cube."""

    cube_id: str
    currency: str
    index_tenor: str
    vol_type: str
    valuation_date: dt_date
    num_expiries: int
    num_tenors: int
    has_smile: bool


class SwaptionVolCubeListResponse(BaseModel):
    """Response listing built swaption volatility cubes."""

    cubes: List[SwaptionVolCubeInfo] = Field(default_factory=list)


class SwaptionVolCubeData(BaseModel):
    """ATM heatmap data for a swaption vol cube."""

    cube_id: str
    currency: str
    index_tenor: str
    vol_type: str
    valuation_date: dt_date
    expiries: List[str]
    tenors: List[str]
    atm_volatilities: List[List[float]]
    has_smile: bool = False
    smile_expiries: Optional[List[str]] = None
    smile_tenors: Optional[List[str]] = None
    smile_method: Optional[SmileMethod] = None
    smile_vol_type: Optional[str] = None
    strike_interpolation: Optional[StrikeInterpolation] = None
    conventions_profile: Optional[str] = None


class SwaptionSmileQuoteComparison(BaseModel):
    """An unchanged source quote and the model read at exactly the same strike."""

    source_kind: Literal["atm", "otm"]
    strike: float
    strike_spread_bp: float
    source_volatility: float
    model_volatility: float
    error_bp: float = Field(..., description="Model minus source normal volatility, in bp")


class SwaptionSmileSlice(BaseModel):
    """Smile data for a specific expiry/tenor point."""

    expiry: str
    tenor: str
    forward_rate: float
    strikes: List[float]
    volatilities: List[float]
    quote_comparison: List[SwaptionSmileQuoteComparison] = Field(default_factory=list)
    sabr_params: Optional[Dict[str, float]] = None
    smile_method: Optional[SmileMethod] = None
    smile_vol_type: Optional[str] = None


# Exact normal SABR node fit status (volatility.swaption_smile.NormalSabrSmileCube.diagnostics).
SwaptionSabrNodeStatus = Literal["ok", "poor_fit", "rho_at_bound", "nu_at_bound", "quote_excluded"]


class SwaptionSabrCalibrationNode(BaseModel):
    """SABR calibration diagnostics for one expiry/tenor node."""

    expiry: str
    tenor: str
    forward_rate: float
    alpha: Optional[float] = None
    beta: Optional[float] = None
    rho: Optional[float] = None
    nu: Optional[float] = None
    num_market_points: int = 0
    rmse_bp: Optional[float] = None
    max_abs_error_bp: Optional[float] = None
    atm_market_bp: Optional[float] = None
    atm_model_bp: Optional[float] = None
    atm_gap_bp: Optional[float] = None
    status: Optional[SwaptionSabrNodeStatus] = Field(
        default=None,
        description=(
            "ok, poor_fit (RMSE > 1.5 bp or max > 5 bp), rho_at_bound, nu_at_bound or "
            "quote_excluded (a quote with a negligible price left out of the fit)."
        ),
    )


class SwaptionSabrCalibrationResponse(BaseModel):
    """Full SABR calibration diagnostics for a swaption cube."""

    cube_id: str
    currency: str
    index_tenor: str
    valuation_date: dt_date
    smile_method: Optional[SmileMethod] = None
    smile_vol_type: Optional[str] = None
    implementation: Optional[str] = None
    nodes: List[SwaptionSabrCalibrationNode]


# ---------------------------------------------------------------------------
# Swaption Instrument
# ---------------------------------------------------------------------------


class SwaptionType(str, Enum):
    """Swaption type (underlying swap direction)."""

    PAYER = "payer"
    RECEIVER = "receiver"


class ExerciseType(str, Enum):
    """Exercise style."""

    EUROPEAN = "european"
    BERMUDAN = "bermudan"


class SettlementType(str, Enum):
    """Swaption settlement type."""

    PHYSICAL = "physical"
    CASH = "cash"


class SwaptionDefinition(BaseModel):
    """Complete swaption instrument definition."""

    currency: str = Field(..., description="Currency code")
    swaption_type: SwaptionType = Field(..., description="Payer or receiver swaption")
    exercise_type: ExerciseType = Field(default=ExerciseType.EUROPEAN, description="Exercise style")

    expiry_date: dt_date = Field(..., description="European exercise date (or first bermudan date)")
    bermudan_dates: Optional[List[dt_date]] = Field(
        default=None, description="Bermudan exercise dates (required if exercise_type=bermudan)"
    )

    swap_effective_date: dt_date = Field(..., description="Underlying swap effective date")
    swap_maturity_date: dt_date = Field(..., description="Underlying swap maturity date")
    notional: float = Field(..., gt=0, description="Notional amount")
    strike: float = Field(..., description="Fixed rate strike (decimal, e.g. 0.025 for 2.5%)")
    fixed_leg_rates: Optional[List[FiniteFloat]] = Field(
        default=None,
        description=(
            "Explicit fixed coupon for each underlying fixed-leg period. "
            "When supplied, the exact nonstandard swaption engine is selected automatically."
        ),
    )

    normal_volatility_mapping: Literal["payoff_equivalent"] = Field(
        default="payoff_equivalent",
        description=(
            "Normal market route: read the market-equivalent strike and rescale by the "
            "annuity ratio. Former normal_moneyness inputs are migrated to this convention."
        ),
    )
    float_index_tenor: str = Field(default="6M", description="Floating index tenor (e.g. '6M')")
    settlement_type: SettlementType = Field(
        default=SettlementType.PHYSICAL, description="Settlement type"
    )

    day_count: str = Field(default="ACT/360", description="Day count convention")
    fixed_day_count: Optional[str] = Field(
        default=None, description="Fixed leg day count. Si absent, utilise day_count."
    )
    float_day_count: Optional[str] = Field(
        default=None, description="Float leg day count. Si absent, utilise le défaut devise."
    )
    fixed_frequency: Optional[str] = Field(
        default=None,
        description="Fixed leg frequency (Annual, Semiannual, Quarterly, Monthly). Si absent, utilise le défaut devise.",
    )
    float_frequency: Optional[str] = Field(
        default=None,
        description="Float leg schedule frequency. Si absent, déduit de float_index_tenor.",
    )
    float_spread: float = Field(
        default=0.0, description="Floating leg spread (decimal, ex: 0.001 = 10bps)"
    )
    position: Literal["long", "short"] = Field(default="long", description="Long = buyer")

    # Roll convention (underlying swap)
    first_roll_date: Optional[dt_date] = Field(
        default=None, description="First roll date for stub period on underlying swap (optional)"
    )
    last_roll_date: Optional[dt_date] = Field(
        default=None,
        description="Last roll date for back stub period on underlying swap (optional)",
    )
    schedule_convention: ScheduleConvention = Field(
        default_factory=ScheduleConvention,
        description="Business-day and roll conventions for the underlying swap schedule.",
    )

    amortization: AmortizationSchedule = Field(
        default_factory=AmortizationSchedule,
        description="Amortization schedule (preferred format)",
    )
    amortization_type: Optional[AmortizationType] = Field(
        default=None,
        description="Deprecated input alias: use amortization.amortization_type",
    )
    custom_notionals: Optional[List[float]] = Field(
        default=None,
        description="Deprecated input alias: use amortization.fixed_leg_notionals / float_leg_notionals",
    )

    @field_validator("normal_volatility_mapping", mode="before")
    @classmethod
    def _migrate_normal_volatility_mapping(cls, value: object) -> object:
        """Reprice saved inputs under the adopted market-equivalent convention."""
        return "payoff_equivalent" if value == "normal_moneyness" else value

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_amortization(cls, values: object) -> object:
        """Accept legacy amortization fields and map them to AmortizationSchedule."""
        if not isinstance(values, dict):
            return values

        if values.get("amortization") is None:
            legacy_type = values.get("amortization_type", AmortizationType.BULLET)
            legacy_notionals = values.get("custom_notionals")
            values["amortization"] = {
                "amortization_type": legacy_type,
                "fixed_leg_notionals": legacy_notionals,
                "float_leg_notionals": legacy_notionals,
            }
        return values


# ---------------------------------------------------------------------------
# Swaption Pricing
# ---------------------------------------------------------------------------


class SwaptionGreeks(BaseModel):
    """Greeks for a swaption."""

    dv01_source: str | None = Field(
        default=None,
        description="Source of scalar DV01, retained when detailed diagnostics are omitted",
    )
    dv01: float = Field(
        ...,
        description=(
            "Local market-quote DV01 = -1e-4 * sum(dV/dq), computed from the implicit "
            "calibration Jacobian with calibration fixings frozen; normal_moneyness re-reads the smile "
            "under rate shocks, while legacy effective vol and model parameters stay frozen; "
            "position sign applied"
        ),
    )
    # Historical product-specific sensitivity, retained for compatibility.
    # This is distinct from market-quote DV01. Use the position-signed dv01
    # field for every DV01 display, export, and aggregation.
    delta: float = Field(
        ...,
        description="Legacy delta, not DV01: NPV(+1bp zero-spread curve shift) - NPV(base)",
    )
    vega: float = Field(..., description="Vega - NPV change for +1bp vol shift")
    gamma: float = Field(
        default=0.0, description="Gamma - second-order NPV change for ±1bp curve shift"
    )


MAX_VOL_RATIO = 10.0
# Smallest ratio of the two volatilities the two-factor route computes (a state with no noise has no grid);
# the one-factor limit itself is the GSR route.
MIN_VOL_RATIO = 1.0e-4


class SwaptionTwoFactorInput(BaseModel):
    """Two-factor structure imposed on the price of an amortizing or Bermudan swaption (MODTAUX L6B, D-M14).

    An absent number takes its declared default: the reversions and the correlation the fixed-structure G2++
    of the XVA uses, the ratio of the two volatilities the one the G2++ calibration of the cube gives for that
    structure. The volatility level (normal vol of the first factor) is calibrated date by date on the deal's
    basket unless imposed.
    """

    a: Optional[float] = Field(default=None, ge=0.0, le=MAX_MODEL_REVERSION)
    b: Optional[float] = Field(default=None, ge=0.0, le=MAX_MODEL_REVERSION)
    rho: Optional[float] = Field(default=None, gt=-1.0, lt=1.0)
    vol_ratio: Optional[float] = Field(
        default=None,
        ge=MIN_VOL_RATIO,
        le=MAX_VOL_RATIO,
        description=(
            "eta / sigma, volatility of factor 2 over factor 1; at least 1e-4 (the one-factor limit is "
            "the gsr model)"
        ),
    )
    level: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=MAX_MODEL_VOLATILITY,
        description="Volatility level imposed on every exercise date (decimal normal vol); calibrated if absent",
    )


class SwaptionPricingRequest(BaseModel):
    """Request to price a swaption."""

    swaption: SwaptionDefinition = Field(..., description="Swaption definition")
    valuation_date: dt_date = Field(..., description="Pricing date")
    compute_greeks: bool = Field(default=True, description="Compute DV01, delta, gamma and vega")
    compute_greeks_diagnostics: bool = Field(
        default=True, description="Include detailed curve and market-quote risk diagnostics"
    )
    discount_curve_key: Optional[str] = Field(
        default=None, description="Override discount curve (default: OIS)"
    )
    forward_curve_key: Optional[str] = Field(
        default=None, description="Override forward curve (default: float_index_tenor)"
    )
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    rate_shock_bps: FiniteFloat | None = Field(
        default=None,
        description=(
            "Parallel market-quote rate shock in bps: selected rate-equivalent raw quotes move "
            "and the curves are rebootstrapped; the cube and GSR parameters stay frozen"
        ),
    )
    vol_shock_bps: FiniteFloat | None = Field(
        default=None,
        description="Parallel vol shock in bps, applied to raw vol quotes on unshocked curves",
    )
    model_type: ModelType = Field(
        default=ModelType.GSR,
        description=(
            "Amortizing and Bermudan swaptions: gsr (default, one factor) or g2pp (two factors, "
            "see two_factor). A swaption priced on the market route has no model."
        ),
    )
    two_factor: Optional[SwaptionTwoFactorInput] = Field(
        default=None, description="Structure of the two-factor model (model_type=g2pp only)"
    )
    initial_reversion: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=MAX_MODEL_REVERSION,
        description=(
            "Amortizing and Bermudan swaptions: the GSR mean reversion imposed by the caller, for this "
            "price only (the stored model is untouched). Absent, the reversion implied by the "
            "volatility ratios of the unshocked cube at its market date (a stored GSR never lends "
            "its own). Refused on the market route of a European bullet swaption, which has no model."
        ),
    )


class SwaptionPeriodDetail(BaseModel):
    """Per-period breakdown of swaption value decomposition."""

    accrual_start: dt_date
    accrual_end: dt_date
    payment_date: dt_date
    notional: float = Field(default=0.0)
    fixed_rate: float = Field(default=0.0)
    float_rate: float = Field(default=0.0)
    fixed_pv: float = Field(default=0.0)
    float_pv: float = Field(default=0.0)
    net_pv: float = Field(default=0.0, description="float_pv + fixed_pv (swap perspective)")
    discount_factor: float = Field(default=0.0)


class SwaptionMarketLineageStatus(str, Enum):
    """Outcome of the swaption market-identity check."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNVERIFIED = "unverified"


class MarketLineageReasonCode(str, Enum):
    """Why a swaption pricing state is not a proven market-identity match."""

    CURVE_INPUTS_STALE = "curve_inputs_stale"
    CUBE_CURVE_STALE = "cube_curve_stale"
    CUBE_ATM_STALE = "cube_atm_stale"
    CUBE_OTM_STALE = "cube_otm_stale"
    MARKET_DATE_MISMATCH = "market_date_mismatch"
    SOURCE_MISSING = "source_missing"
    LEGACY_UNVERIFIED = "legacy_unverified"


class SwaptionMarketLineageDiagnostic(BaseModel):
    """Provenance check between the curves, the cube and the live market sources.

    Diagnostic only: a mismatch never refuses a price in this phase. Raw quotes,
    source labels and full conventions are deliberately absent from this payload.
    """

    status: SwaptionMarketLineageStatus = Field(..., description="match, mismatch or unverified")
    curve_fingerprint: Optional[str] = Field(
        default=None, description="Fingerprint of the curve set used for pricing"
    )
    cube_fingerprint: Optional[str] = Field(
        default=None, description="Fingerprint of the swaption vol cube used for pricing"
    )
    source_curve_fingerprint: Optional[str] = Field(
        default=None, description="Curve fingerprint the cube was actually built on"
    )
    reason_codes: List[MarketLineageReasonCode] = Field(
        default_factory=list,
        description="Deterministically ordered reasons, empty when status is match",
    )
    enforcement_mode: Literal["diagnostic"] = Field(
        default="diagnostic",
        description="Phase 0A reports provenance without refusing a price",
    )
    shock_context: Optional[JsonObject] = Field(
        default=None,
        description="Ephemeral rate/vol shock context applied on top of the validated base",
    )


SwaptionVolatilityMapping = Literal["identity", "market_equivalent", "normal_moneyness"]


# Status of a smile node (build diagnostics and pricing read): the one list the cube, the audit and
# the interface share. synthetic_cells and arbitrage belong to the interpolated smile; the SABR
# fit flags (poor_fit, bounds, quote_excluded) to the exact normal SABR smile.
SwaptionSmileNodeStatus = Literal[
    "ok",
    "quotes_adjusted",
    "synthetic_cells",
    "arbitrage",
    "poor_fit",
    "rho_at_bound",
    "nu_at_bound",
    "quote_excluded",
]
SmileExtrapolatedAxis = Literal["expiry", "tenor", "strike"]


class SwaptionSmileReadDiagnostic(BaseModel):
    """What the market smile read for a swaption price stands on (CUBESWPT section 4.6)."""

    smile_method: SmileMethod = Field(..., description="Method of the cube that was read")
    strike_interpolation: Optional[StrikeInterpolation] = Field(
        default=None, description="Strike scheme of an interpolated cube; None for SABR normal"
    )
    extrapolated_axes: List[SmileExtrapolatedAxis] = Field(
        default_factory=list,
        description="Axes on which the read lies outside the quoted data (flat or tail beyond)",
    )
    synthetic: bool = Field(
        default=False, description="The read rests on a quote interpolated along its node"
    )
    unidentified: bool = Field(
        default=False,
        description="SABR: the model price is negligible there, a neighbouring vol is held",
    )
    adjusted_market_points: int = Field(
        default=0, ge=0,
        description="Source quotes whose model price was adjusted at contributing smile nodes",
    )
    node_status: SwaptionSmileNodeStatus = Field(
        default="ok", description="Worst status among the smile nodes the read blends"
    )


SwaptionOptionStatus = Literal["live", "expired"]


class SwaptionPricingResult(BaseModel):
    """Complete swaption pricing result."""

    curve_quote_exclusions: list[CurveQuoteExclusion] = Field(default_factory=list)

    npv: float = Field(..., description="Net present value")
    option_status: SwaptionOptionStatus = Field(
        default="live",
        description=(
            "'expired' when every exercise date is before the valuation date: npv, intrinsic and "
            "time value are zero, whether or not the option was exercised (SYN-10)"
        ),
    )
    atm_rate: float = Field(..., description="ATM swap rate (fair rate)")
    premium_bps: float = Field(
        ..., description="Running premium in bp = NPV / underlying swap PV01"
    )
    underlying_swap_pv01: float = Field(
        default=0.0, description="PV01 of underlying swap used for premium normalisation"
    )
    intrinsic_value: float = Field(
        default=0.0,
        description=(
            "Intrinsic value: the value at zero volatility on the published cashflows, max(0, best "
            "PV over the open exercise dates of the swap each delivers, its coupons starting on or "
            "after the date); 0 once expired"
        ),
    )
    time_value: float = Field(default=0.0, description="Time value: swaption NPV - intrinsic value")
    position: str = Field(default="long", description="Position")

    curve_date: dt_date
    valuation_date: dt_date
    discount_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    forward_interpolation: CurveInterpolationMethod = Field(
        default=CurveInterpolation.SMOOTH_FORWARD
    )
    vol_type: str
    exercise_type: str
    settlement_type: str
    model_type_used: Optional[str] = Field(
        default=None,
        description="Calibrated model type used by pricing engine when applicable.",
    )
    pricing_method: str = Field(
        default="unspecified",
        description=(
            "Automatic pricing route used by the engine "
            "(e.g. european_market, bermudan_gaussian1d)."
        ),
    )
    is_amortizing: bool = Field(
        default=False, description="Whether the swaption uses a non-bullet notional profile"
    )
    initial_notional: float = Field(default=0.0, description="Initial notional amount")
    average_notional: float = Field(
        default=0.0,
        description=(
            "Average notional of the underlying swap, weighted by the fixed-leg accrual "
            "fractions: sum(N_i * tau_i) / sum(tau_i)"
        ),
    )
    weighted_avg_notional: float = Field(
        default=0.0,
        deprecated=True,
        description="Deprecated alias of average_notional, same value (SWPT-MIN-04)",
    )
    decomposition_details: Optional[JsonObject] = Field(
        default=None,
        description="Optional exact-payoff, routing, market-cube and model methodology details",
    )
    convention_diagnostics: Optional[JsonObject] = Field(
        default=None,
        description="Resolved convention diagnostics, including market comparability for OTC non-standard trades.",
    )

    implied_volatility: Optional[float] = Field(
        default=None,
        description=(
            "Implied volatility on the deal's own swap rate: the single vol reproducing the NPV "
            "with the deal's conventions (decimal, cube vol type)."
        ),
    )
    implied_volatility_diagnostic: Optional[str] = Field(
        default=None,
        description="Reason an implied volatility is unavailable or sits on a numerical boundary.",
    )
    market_equivalent_volatility: Optional[float] = Field(
        default=None,
        description=(
            "European market route: market-cube vol read at the market-equivalent strike (the "
            "strike of the market-convention swaption with the same dates). Its option carries "
            "the quantity 1 - slope when the deal's fixed frequency or index differs from the "
            "market (decomposition_details.annuity_ratio_correction)."
        ),
    )
    market_equivalent_strike: Optional[float] = Field(
        default=None,
        description=(
            "Strike at which the market cube is read (decimal). For a deal on market conventions "
            "it is the equivalent strike (K * A_deal - (PV_float_deal - PV_float_market)) / A_market; "
            "otherwise it is the first-order exercise boundary K' = (K_eq - slope * S) / (1 - slope) "
            "(decomposition_details.annuity_ratio_correction gives K_eq, the slope and the quantity)."
        ),
    )
    market_forward_rate: Optional[float] = Field(
        default=None,
        description="Forward rate of the market-convention swap on the deal's dates (decimal).",
    )
    volatility_mapping: Optional[SwaptionVolatilityMapping] = Field(
        default=None,
        description=(
            "identity: deal on market conventions, the cube is read as is; market_equivalent: "
            "the cube is read at the first-order exercise boundary and rescaled by the annuity "
            "ratio and the quantity 1 - slope."
        ),
    )
    option_maturity_years: Optional[float] = Field(
        default=None, description="Time to swaption expiry in years"
    )
    swap_tenor_years: Optional[float] = Field(
        default=None, description="Underlying swap tenor in years"
    )
    dv01_weighted_maturity_years: Optional[float] = Field(
        default=None,
        description="DV01-weighted average maturity (from bucketed curve deltas)",
    )
    macaulay_duration_years: Optional[float] = Field(
        default=None,
        description="Macaulay duration of the underlying swap fixed leg",
    )

    greeks: Optional[SwaptionGreeks] = None
    greeks_diagnostics: Optional[SwaptionGreeksDiagnostics] = Field(
        default=None,
        description="Optional consistency diagnostics (bucketed checks, finite-difference sanity metrics).",
    )
    cashflows: Optional[List[LegCashflows]] = Field(
        default=None, description="Underlying swap cashflows (fixed + float legs)"
    )
    period_details: Optional[List[SwaptionPeriodDetail]] = Field(
        default=None,
        description=(
            "Per-period breakdown of the whole underlying swap with its net PV; its sum is the "
            "swap's value, not the intrinsic value (which keeps the coupons the best open "
            "exercise date delivers)"
        ),
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Pricing warnings (e.g., missing fixings estimated from forward curve)",
    )
    market_lineage: Optional[SwaptionMarketLineageDiagnostic] = Field(
        default=None,
        description="Market-identity provenance of the curves, cube and raw sources used",
    )
    smile_read: Optional[SwaptionSmileReadDiagnostic] = Field(
        default=None,
        description="European market route: extrapolation, synthetic quote and node status of the "
        "smile read behind the vol (None when the price does not read the smile)",
    )


class SwaptionSupportCombination(BaseModel):
    """Compatibility row for swaption pricing path."""

    exercise_type: str
    settlement_type: str
    structure_kind: Literal["vanilla", "nonstandard"]
    pricing_route: str
    supported: bool
    requires_model: bool
    reason: Optional[str] = None


class SwaptionSupportMatrixResponse(BaseModel):
    """Matrix of supported/unsupported swaption combinations."""

    combinations: List[SwaptionSupportCombination]


# ---------------------------------------------------------------------------
# Hull-White Calibration
# ---------------------------------------------------------------------------


class HWCalibrationPoint(BaseModel):
    """A single calibration point (expiry x tenor)."""

    expiry: str = Field(..., description="Option expiry tenor (e.g. '5Y')")
    tenor: str = Field(..., description="Swap tenor (e.g. '10Y')")


_CORRELATION_MARGIN = 1.0e-6


def clip_correlation(rho: Optional[float]) -> Optional[float]:
    """Keep a stored correlation strictly inside (-1, 1) when it seeds a new request."""
    if rho is None:
        return None
    return min(max(rho, -1.0 + _CORRELATION_MARGIN), 1.0 - _CORRELATION_MARGIN)


class G2Structure(BaseModel):
    """G2++ parameters imposed on the fixed-structure calibration; an absent one takes its declared default."""

    a: Optional[float] = Field(default=None, ge=0.0, le=MAX_MODEL_REVERSION)
    b: Optional[float] = Field(default=None, ge=0.0, le=MAX_MODEL_REVERSION)
    rho: Optional[float] = Field(default=None, gt=-1.0, lt=1.0)


class ModelParameterOverrides(BaseModel):
    """Short-rate parameters imposed on one XVA or simulation request (MODTAUX L7, D-M12).

    The stored model is recalibrated for the request only, the imposed numbers held fixed. HW1F: ``a`` and
    ``sigma`` (the other one is calibrated, both imposed means no calibration); GSR: ``a`` (sigma(t) is calibrated
    date by date); G2++: any of ``a, b, rho, sigma, eta`` (reversions and correlation otherwise those of the stored
    model). Nothing is stored.
    """

    model_config = ConfigDict(extra="forbid")

    a: Optional[float] = Field(default=None, ge=0.0, le=MAX_MODEL_REVERSION)
    b: Optional[float] = Field(default=None, ge=0.0, le=MAX_MODEL_REVERSION)
    rho: Optional[float] = Field(default=None, gt=-1.0, lt=1.0)
    sigma: Optional[float] = Field(default=None, gt=0.0, le=MAX_MODEL_VOLATILITY)
    eta: Optional[float] = Field(default=None, gt=0.0, le=MAX_MODEL_VOLATILITY)


class HWCalibrationRequest(BaseModel):
    """Request to calibrate HW1F, G2++, or GSR."""

    currency: str = Field(..., description="Currency code")
    index_tenor: str = Field(default="6M", description="Forward index tenor used for calibration")
    valuation_date: dt_date = Field(..., description="Valuation date")
    model_type: ModelType = Field(default=ModelType.HW1F, description="Model type")
    calibration_points: Optional[List[HWCalibrationPoint]] = Field(
        default=None, description="Specific calibration points. If None, uses diagonal."
    )
    initial_a: float = Field(
        default=0.05,
        ge=0.0,
        le=MAX_MODEL_REVERSION,
        description="Starting value of the reversion (HW1F, G2++); not used by the GSR, see initial_reversion",
    )
    initial_sigma: float = Field(
        default=0.01, gt=0.0, le=MAX_MODEL_VOLATILITY, description="Initial volatility"
    )
    initial_b: Optional[float] = Field(
        default=0.30,
        ge=0.0,
        le=MAX_MODEL_REVERSION,
        description="G2++ mean reversion factor 2",
    )
    initial_eta: Optional[float] = Field(
        default=0.005, gt=0.0, le=MAX_MODEL_VOLATILITY, description="G2++ volatility factor 2"
    )
    initial_rho: Optional[float] = Field(
        default=-0.40, gt=-1.0, lt=1.0, description="G2++ correlation, strictly inside (-1, 1)"
    )
    initial_reversion: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=MAX_MODEL_REVERSION,
        description=(
            "GSR: the reversion imposed by the caller. Absent, the reversion is calibrated on the "
            "volatility ratios of swaptions of the same expiry (declared default when not identified)."
        ),
    )
    initial_vol: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=MAX_MODEL_VOLATILITY,
        description="Initial volatility for Gaussian1d/GSR-style models",
    )
    coterminal_maturity_years: Optional[float] = Field(
        default=None,
        description="If provided, uses co-terminal diagonal calibration targeting this swap maturity.",
    )
    diagonal_strategy: Optional[DiagonalStrategy] = Field(
        default=None,
        description=(
            "GRID_ONLY=on-grid points only; FULL_ANNUAL=all template-year pairs (interpolated). "
            "None=infer from model_type (GSR→FULL_ANNUAL, else→GRID_ONLY)."
        ),
    )
    max_iterations: int = Field(default=1000, ge=1, description="Max optimizer iterations")
    g2_structure: Optional[G2Structure] = Field(
        default=None,
        description=(
            "G2++ fixed_structure mode only: reversions and correlation imposed by the caller; an "
            "absent value takes the declared default (a 0.01, b 0.5, rho -0.7). initial_a, initial_b "
            "and initial_rho are starting values of the free parameters, never imposed values."
        ),
    )
    g2_calibration_mode: Literal["fixed_structure", "free"] = Field(
        default="fixed_structure",
        description=(
            "G2++ only. fixed_structure: the two reversions and the correlation are fixed (imposed "
            "through g2_structure, else the declared defaults 0.01, 0.5 and -0.7) and the two "
            "volatilities are calibrated; free: the five parameters are calibrated."
        ),
    )


class ModelCalibrationRequest(HWCalibrationRequest):
    """Unified model calibration request (HW1F/G2++/GSR)."""

    pass


class HWHelperDetail(BaseModel):
    """Detail for a single calibration helper."""

    expiry: str
    tenor: str
    market_vol: float
    model_vol: float
    error: float
    option_date: Optional[dt_date] = None
    swap_start_date: Optional[dt_date] = None
    swap_maturity_date: Optional[dt_date] = None
    strike: Optional[float] = None
    atm_forward: Optional[float] = None
    annuity: Optional[float] = None
    vol_type: Optional[str] = None
    shift: Optional[float] = None
    market_price: Optional[float] = None
    model_price: Optional[float] = None
    price_error: Optional[float] = None
    relative_price_error: Optional[float] = None
    implied_vol_error_bp: Optional[float] = None
    vega: Optional[float] = None
    weight: Optional[float] = None
    helper_status: Optional[str] = None
    construction: Optional[str] = None
    payment_lag: Optional[int] = None
    lookback_days: Optional[int] = None
    lockout_days: Optional[int] = None
    observation_shift: Optional[bool] = None
    averaging_method: Optional[str] = None
    telescopic_value_dates: Optional[bool] = None
    convention_source: Optional[str] = None


class HWCalibrationSkippedPoint(BaseModel):
    """Calibration point that was requested but not used to build a helper."""

    expiry: str
    tenor: str
    reason: str


class HWCalibrationResponse(BaseModel):
    """Result of Hull-White or G2++ calibration."""

    success: bool = Field(
        ..., description="True only when the optimizer stopped on a convergence criterion"
    )
    termination: Optional[str] = Field(
        default=None,
        description="Stopping criterion of the optimizer (QuantLib EndCriteria or SciPy status)",
    )
    currency: str
    index_tenor: str = Field(default="6M", description="Forward index tenor used in calibration")
    model_type: ModelType = Field(default=ModelType.HW1F, description="Model type")
    a: float = Field(..., description="Calibrated mean reversion speed")
    sigma: float = Field(..., description="Calibrated volatility")
    b: Optional[float] = Field(default=None, description="G2++ mean reversion factor 2")
    eta: Optional[float] = Field(default=None, description="G2++ volatility factor 2")
    rho: Optional[float] = Field(default=None, description="G2++ correlation")
    effective_sigma: Optional[float] = Field(
        default=None,
        description="G2++ effective short rate volatility: sqrt(sigma^2 + eta^2 + 2*rho*sigma*eta)",
    )
    num_helpers: int
    rmse: float = Field(..., description="Root mean square error")
    max_error: float
    implementation: Optional[str] = Field(
        default=None,
        description="Implementation identifier for model calibration path.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional explanatory notes (e.g. binding limitations and methodology details).",
    )
    sigma_step_dates: Optional[List[dt_date]] = Field(
        default=None,
        description="Optional step dates for piecewise-constant sigma(t) models.",
    )
    sigma_values: Optional[List[float]] = Field(
        default=None,
        description="Optional piecewise-constant sigma values aligned with sigma_step_dates.",
    )
    skipped_points: Optional[List[HWCalibrationSkippedPoint]] = Field(
        default=None,
        description="Requested calibration points excluded before calibration.",
    )
    calibration_mode: Optional[str] = Field(
        default=None,
        description="G2++ calibration mode used (fixed_structure or free); absent for other models.",
    )
    calibration_flags: Optional[List[str]] = Field(
        default=None,
        description=(
            "Structured statuses of what degraded (empty when nothing did): not_converged, "
            "parameters_not_identified, parameter_at_bound:<parameter>, quadrature_not_converged, "
            "degenerate_two_factor, default_reversion, implied_reversion_comparison_failed. The "
            "notes say the same in words."
        ),
    )
    parameter_sources: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Source of each model parameter: calibrated, imposed or default. For a GSR the reversion "
            "is imposed, calibrated_vol_ratios, default_not_identified or not_attempted (a cube that "
            "cannot say: not normal vol, or no usable node)."
        ),
    )
    market_fingerprint: Optional[str] = Field(
        default=None,
        description="Fingerprint of the curves and swaption cube the model was calibrated on (L8)",
    )
    method_version: Optional[str] = Field(
        default=None, description="Version of the calibration method that produced the model (L8)"
    )
    implied_reversion: Optional[float] = Field(
        default=None,
        description=(
            "GSR: the constant reversion that reproduces the volatility ratios of swaptions of the "
            "same expiry, whether or not it was the one used (an imposed reversion is still compared)."
        ),
    )
    diagonal_strategy_effective: Optional[str] = Field(
        default=None,
        description="Effective calibration point selection strategy after fallback.",
    )
    helper_details: List[HWHelperDetail] = Field(default_factory=list)


class ModelCalibrationResponse(HWCalibrationResponse):
    """Unified model calibration response."""

    calibration_snapshot_id: Optional[str] = Field(
        default=None,
        description="Immutable audit-ledger snapshot created for this persisted calibration.",
    )


class StaleShortRateModel(BaseModel):
    """A stored model that is refused for pricing, with the reason (MODTAUX L8B)."""

    model_type: str
    index_tenor: str
    valuation_date: dt_date
    method_version: Optional[str] = None
    reason: str


class ShortRateModelStatusResponse(BaseModel):
    """Status of a calibrated short-rate model for a currency/tenor."""

    available: bool
    currency: str
    index_tenor: str
    model_type: Optional[str] = None
    a: Optional[float] = None
    sigma: Optional[float] = None
    b: Optional[float] = None
    eta: Optional[float] = None
    rho: Optional[float] = None
    implementation: Optional[str] = None
    notes: Optional[str] = None
    valuation_date: Optional[dt_date] = None
    sigma_step_dates: Optional[List[dt_date]] = None
    sigma_values: Optional[List[float]] = None
    # Provenance (MODTAUX L8): a stale model is not available for pricing and says why.
    market_fingerprint: Optional[str] = None
    method_version: Optional[str] = None
    parameter_sources: Optional[Dict[str, str]] = None
    stopping_criterion: Optional[str] = None
    stale: bool = False
    stale_reason: Optional[str] = None


class ShortRateModelPricingOption(BaseModel):
    """A calibrated model that can be used for swaption pricing on the requested tenor."""

    model_type: ModelType
    requested_index_tenor: str
    calibration_index_tenor: str
    exact_tenor_match: bool
    valuation_date: dt_date
    note: Optional[str] = None
    valuation_date_mismatch: bool = False


class ShortRateModelPricingOptionsResponse(BaseModel):
    """Pricing-eligible calibrated short-rate models for a currency/requested tenor."""

    currency: str
    requested_index_tenor: str
    valuation_date: Optional[dt_date] = None
    models: List[ShortRateModelPricingOption] = Field(default_factory=list)
