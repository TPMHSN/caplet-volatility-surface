"""
Pydantic models for market data input and representation.
"""

from datetime import date
from enum import Enum
from math import isfinite
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from curves.helpers import parse_tenor
from type_defs import JsonObject


class MarketDataPoint(BaseModel):
    """Base class for a market data point."""

    tenor: str = Field(..., description="Tenor string (e.g., '1M', '3M', '1Y', '5Y')")
    rate: float = Field(..., allow_inf_nan=False, description="Rate in decimal (e.g., 0.03 for 3%)")
    quote_metadata: "CurveQuoteMetadata" = Field(
        ...,
        description="Source instrument and canonical convention of this quote (mandatory)",
    )


class CurveInstrumentType(str, Enum):
    """Instrument families accepted by the curve market-data contract."""

    FIXING = "fixing"
    DEPOSIT = "deposit"
    FRA = "fra"
    FUTURE = "future"
    OIS_SWAP = "ois_swap"
    IRS_SWAP = "irs_swap"


class CurveQuoteMetadata(BaseModel):
    """Provenance required to reconstruct a calibration instrument deterministically."""

    source_label: str = Field(..., min_length=1)
    instrument_type: CurveInstrumentType
    convention_id: str = Field(..., min_length=1)


class FuturesRateBasis(str, Enum):
    """Economic meaning of a futures rate; adjusted rates need no further convexity correction."""

    FUTURES_RATE = "futures_rate"
    CONVEXITY_ADJUSTED_RATE = "convexity_adjusted_rate"


class FuturesDataPoint(BaseModel):
    """Futures instrument with actual IMM date."""

    imm_date: date = Field(..., description="IMM start date of the future")
    rate: float = Field(
        ..., allow_inf_nan=False, description="Rate in decimal (e.g., 0.0204 for 2.04%)"
    )
    rate_basis: FuturesRateBasis = Field(
        default=FuturesRateBasis.FUTURES_RATE,
        description=(
            "Whether rate is a raw futures yield or already convexity-adjusted. "
            "Raw futures are the default; adjusted Excel rates declare their basis explicitly."
        ),
    )
    quote_metadata: CurveQuoteMetadata = Field(
        ...,
        description="Source instrument and canonical convention of this future (mandatory)",
    )


class OISMarketData(BaseModel):
    """OIS curve market data."""

    currency: str = Field(default="EUR", description="Currency code")
    valuation_date: date = Field(..., description="Valuation date")
    fixings: List[MarketDataPoint] = Field(
        default_factory=list,
        description="Overnight fixing observations and curve anchors",
    )
    deposits: List[MarketDataPoint] = Field(default_factory=list, description="OIS deposits")
    swaps: List[MarketDataPoint] = Field(..., description="OIS swaps")


class ForwardMarketData(BaseModel):
    """Forward curve market data."""

    currency: str = Field(default="EUR", description="Currency code")
    valuation_date: date = Field(..., description="Valuation date")
    index_tenor: str = Field(..., description="Index tenor (e.g., '3M', '6M')")
    fixings: List[MarketDataPoint] = Field(
        default_factory=list,
        description="Index fixing observations kept separate from calibration helpers",
    )
    deposits: List[MarketDataPoint] = Field(default_factory=list, description="Deposits")
    fras: List[MarketDataPoint] = Field(default_factory=list, description="FRAs")
    futures: List[FuturesDataPoint] = Field(
        default_factory=list, description="Futures with IMM dates"
    )
    swaps: List[MarketDataPoint] = Field(..., description="Interest rate swaps")
    import_warnings: List[str] = Field(
        default_factory=list,
        description="Import assumptions retained with the market data for downstream diagnostics",
    )


class CapFloorVolData(BaseModel):
    """Cap/Floor volatility surface data."""

    currency: str = Field(default="EUR", description="Currency code")
    valuation_date: date = Field(..., description="Valuation date")
    index_tenor: Optional[str] = Field(
        default=None,
        description=("Legacy homogeneous quote tenor, used only when quote_tenors is absent"),
    )
    quote_tenors: Optional[List[str]] = Field(
        default=None,
        description="Native cap/floor quote index tenor for each expiry",
    )
    rebasing_method: Literal["shifted_moneyness_extrinsic_cap"] = Field(
        default="shifted_moneyness_extrinsic_cap",
        description="Method used to rebase native cap quotes to the product index tenor",
    )
    expiries: List[str] = Field(..., description="List of expiry tenors")
    strikes: List[float] = Field(..., description="List of strikes (in decimal)")
    volatilities: List[List[float]] = Field(
        ..., description="2D array of volatilities [expiry][strike]"
    )
    atm_vols: Optional[List[float]] = Field(
        default=None, description="ATM normal vols per expiry (decimal). Separate from strike grid."
    )
    import_warnings: List[str] = Field(
        default_factory=list,
        description="Import assumptions retained with the market data for downstream diagnostics",
    )

    @model_validator(mode="after")
    def validate_atm_volatilities(self) -> "CapFloorVolData":
        """Validate optional per-expiry ATM and native quote-tenor metadata."""
        if self.atm_vols is not None:
            if len(self.atm_vols) != len(self.expiries):
                raise ValueError("ATM cap/floor volatility count must match the expiry count")
            if any(not isfinite(volatility) or volatility < 0.0 for volatility in self.atm_vols):
                raise ValueError("ATM cap/floor volatilities must be finite and non-negative")

        if self.quote_tenors is not None:
            if len(self.quote_tenors) != len(self.expiries):
                raise ValueError("Cap/floor quote tenor count must match the expiry count")
            if any(not tenor.strip() for tenor in self.quote_tenors):
                raise ValueError("Cap/floor quote tenors must be non-blank")
            normalized_quote_tenors = [tenor.strip().upper() for tenor in self.quote_tenors]
            for tenor in normalized_quote_tenors:
                try:
                    parse_tenor(tenor)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid cap/floor quote tenor: {tenor}") from exc
            self.quote_tenors = normalized_quote_tenors

        if self.index_tenor is not None:
            normalized_index_tenor = self.index_tenor.strip().upper()
            if not normalized_index_tenor:
                raise ValueError("Cap/floor index tenor must be non-blank")
            try:
                parse_tenor(normalized_index_tenor)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid cap/floor index tenor: {normalized_index_tenor}"
                ) from exc
            self.index_tenor = normalized_index_tenor
        return self


class SwaptionVolDataPoint(BaseModel):
    """Single swaption volatility quote, typed (CUBESWPT section 4.1).

    ``strike`` is the strike spread to the node's ATM forward, in decimal (``None`` for the ATM
    quote itself); ``volatility`` is the internal value, a decimal normal vol per year. The raw
    value and unit, the market-convention profile and the source cell say where it came from.
    """

    expiry: str = Field(..., description="Option expiry (e.g., '1Y', '5Y')")
    tenor: str = Field(..., description="Swap tenor (e.g., '5Y', '10Y')")
    strike: Optional[float] = Field(None, description="Strike spread to ATM, decimal (ATM if None)")
    volatility: float = Field(..., description="Normal volatility, decimal per year (0.0060)")
    vol_type: Literal["normal", "lognormal"] = Field(
        default="normal", description="Volatility type (the cube accepts normal only)"
    )
    raw_value: Optional[float] = Field(None, description="Quoted value, in its raw unit")
    raw_unit: Optional[Literal["bp", "decimal"]] = Field(
        None, description="Unit of the raw value: basis points or decimal"
    )
    convention_id: Optional[str] = Field(
        None, description="Swaption market-convention profile of the quote (e.g. EUR-SWPT-V1)"
    )
    source_cell: Optional[str] = Field(None, description="Source cell, e.g. 'VOL SWAPTION OTM!D2'")

    @property
    def role(self) -> Literal["atm", "smile"]:
        return "atm" if self.strike is None else "smile"

    @property
    def strike_kind(self) -> Literal["atm", "spread_to_atm"]:
        return "atm" if self.strike is None else "spread_to_atm"


class SwaptionVolData(BaseModel):
    """Swaption volatility matrix/cube data."""

    currency: str = Field(default="EUR", description="Currency code")
    index_tenor: str = Field(
        default="6M", description="Underlying index tenor for this swaption surface"
    )
    valuation_date: date = Field(..., description="Valuation date")
    data_points: List[SwaptionVolDataPoint] = Field(..., description="Volatility data points")


class CurveConventionOption(BaseModel):
    """One canonical profile the typing table admits on a curve."""

    instrument_type: CurveInstrumentType
    convention_id: str
    tenor_rule: str


class CurveConventionsResponse(BaseModel):
    """Profiles admitted on one curve, to propose the profile of a quote in entry screens."""

    currency: str
    curve_key: str
    options: List[CurveConventionOption]


class CurveQuoteResolveItem(BaseModel):
    """A quote to type: its instrument and, for tenor quotes, its tenor."""

    instrument_type: CurveInstrumentType
    tenor: Optional[str] = None

    @model_validator(mode="after")
    def require_instrument_tenor(self) -> "CurveQuoteResolveItem":
        if self.instrument_type == CurveInstrumentType.FUTURE:
            if self.tenor is not None:
                raise ValueError("A future has no tenor; identify its contract separately")
        elif self.tenor is None or not self.tenor.strip():
            raise ValueError("A non-future quote requires a nonempty tenor")
        return self


class CurveQuoteResolveRequest(BaseModel):
    currency: str
    curve_key: str
    quotes: List[CurveQuoteResolveItem] = Field(min_length=1)


class CurveQuoteResolveResult(BaseModel):
    instrument_type: CurveInstrumentType
    tenor: Optional[str] = None
    convention_id: str


class CurveQuoteResolveResponse(BaseModel):
    currency: str
    curve_key: str
    quotes: List[CurveQuoteResolveResult]


class MarketDataUploadResponse(BaseModel):
    """Response after uploading market data."""

    success: bool
    message: str
    data_type: str
    num_points: int
    warnings: List[str] = Field(
        default_factory=list, description="Notes on what the dataset lacks (e.g. missing quotes)"
    )


class MarketDataStatus(BaseModel):
    """Status of loaded market data."""

    ois_curves: List[str] = Field(default_factory=list)
    forward_curves: List[str] = Field(default_factory=list)
    cap_floor_surfaces: List[str] = Field(default_factory=list)
    swaption_vols: List[str] = Field(default_factory=list)
    swaption_otm_vols: List[str] = Field(default_factory=list)
    valuation_date: Optional[date] = None


class QualitySeverity(str, Enum):
    """Severity level for market data quality checks."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MarketDataQualityIssue(BaseModel):
    """A single quality issue detected in market data."""

    severity: QualitySeverity
    code: str = Field(..., description="Machine-readable issue code")
    scope: str = Field(..., description="Dataset scope (e.g. EUR_6M, swaption_otm_EUR_6M)")
    message: str = Field(..., description="Human-readable issue summary")
    details: Optional[JsonObject] = Field(
        default=None, description="Additional diagnostics payload"
    )


class MarketDataQualityReport(BaseModel):
    """Aggregate market data quality report."""

    conventions_version: str
    valuation_date: Optional[date] = None
    num_issues: int
    num_errors: int
    num_warnings: int
    issues: List[MarketDataQualityIssue] = Field(default_factory=list)
    checked_objects: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of checked datasets by type",
    )


# ── Fixings ──────────────────────────────────────────────────────────


class FixingDataPoint(BaseModel):
    """A single historical rate fixing."""

    date: date
    rate: float = Field(..., description="Rate in decimal (e.g., 0.025 = 2.5%)")


class FixingTimeSeries(BaseModel):
    """Historical fixings for one index."""

    currency: str
    index_tenor: str = Field(..., description="Index tenor: 'OIS', '1M', '3M', '6M'")
    fixings: List[FixingDataPoint] = Field(default_factory=list)


class FixingUploadResponse(BaseModel):
    """Response after uploading fixings."""

    success: bool
    message: str
    indices_loaded: List[str]
    total_fixings: int
    new_fixings: int
    file_fixings: int = Field(
        default=0,
        description="Number of fixing rows parsed from the uploaded file.",
    )
    stored_total_fixings: int = Field(
        default=0,
        description="Total number of fixings stored after the upload.",
    )
    duplicate_fixings: int = Field(
        default=0,
        description="Existing same-date fixings with the same value.",
    )
    conflict_fixings: int = Field(
        default=0,
        description="Existing same-date fixings with a different value.",
    )
    overwritten_fixings: int = Field(
        default=0,
        description="Existing same-date fixings replaced by the uploaded file.",
    )


class PublicFixingSourceInfo(BaseModel):
    """Public fixing source metadata exposed to the frontend."""

    index: str
    label: str
    currency: str
    index_tenor: str
    provider: str
    source_url: str
    source_start_date: date = Field(
        ..., description="Earliest observation date available from the public source."
    )
    stale_after_days: int


class FixingStatus(BaseModel):
    """Status of stored fixings."""

    indices: List[str] = Field(default_factory=list)
    total_count: int = 0
    date_ranges: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None, description="Per-index min/max dates"
    )
    public_sources: List[PublicFixingSourceInfo] = Field(default_factory=list)


class PublicFixingUpdateRequest(BaseModel):
    """Request to complete stored fixings from configured public sources."""

    indices: List[str] = Field(
        default_factory=list,
        description="Fixing indices to update. Empty means all configured public indices.",
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Optional forced start date. Defaults to last stored fixing + 1 day.",
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Optional end date. Defaults to today's date.",
    )
    overwrite_conflicts: bool = Field(
        default=False,
        description="If false, public source values do not overwrite different stored values.",
    )


PublicFixingErrorCode = Literal["network_access_denied", "network_error"]


class PublicFixingSeriesUpdateResult(BaseModel):
    """Update result for one public fixing source."""

    index: str
    label: str
    currency: str
    index_tenor: str
    provider: str
    source_url: str
    source_start_date: date
    existing_last_date: Optional[date] = None
    requested_fetch_start_date: Optional[date] = None
    fetch_start_date: Optional[date] = None
    fetch_end_date: Optional[date] = None
    start_date_adjusted: bool = False
    source_latest_date: Optional[date] = None
    stored_latest_date: Optional[date] = None
    fetched_count: int = 0
    new_count: int = 0
    duplicate_count: int = 0
    conflict_count: int = 0
    overwritten_count: int = 0
    skipped_count: int = 0
    error_code: Optional[PublicFixingErrorCode] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class PublicFixingUpdateResponse(BaseModel):
    """Response after completing public fixings."""

    success: bool
    message: str
    total_fetched: int
    total_new_fixings: int
    total_conflicts: int
    results: List[PublicFixingSeriesUpdateResult]


FixingObservationStatus = Literal["stored", "missing"]


class FixingObservation(BaseModel):
    """A stored or expected-but-missing fixing observation."""

    index: str
    date: date
    rate: Optional[float] = Field(default=None, description="Stored decimal rate, if available.")
    status: FixingObservationStatus


class FixingQueryResponse(BaseModel):
    """Paginated fixing observations for one index."""

    index: str
    label: str
    source_start_date: date
    requested_start_date: Optional[date] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    include_missing: bool
    total_count: int
    returned_count: int
    stored_count: int
    missing_count: int
    limit: int
    offset: int
    rows: List[FixingObservation]
    warnings: List[str] = Field(default_factory=list)


class FixingAuditSeries(BaseModel):
    """Coverage audit for one fixing series."""

    index: str
    label: str
    currency: str
    index_tenor: str
    provider: str
    source_url: str
    source_start_date: date
    total_count: int
    min_date: Optional[date] = None
    max_date: Optional[date] = None
    audit_start_date: Optional[date] = None
    audit_end_date: Optional[date] = None
    expected_count: int = 0
    stored_in_window_count: int = 0
    missing_count: int = 0
    missing_dates: List[date] = Field(default_factory=list)
    missing_dates_truncated: bool = False
    days_since_latest: Optional[int] = None
    stale: bool = False
    notes: List[str] = Field(default_factory=list)


class FixingAuditResponse(BaseModel):
    """Audit response for configured fixing series."""

    as_of_date: date
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    series: List[FixingAuditSeries]
