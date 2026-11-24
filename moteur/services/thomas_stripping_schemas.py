"""Auditable native-tenor cap/optionlet stripping results for Caps Thomas."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from services.thomas_black_schemas import BlackVolSurface


class CapNode(BaseModel):
    maturity_months: int
    expiry: str
    maturity_date: date
    source: Literal["market", "interpolated", "initialized"]
    optionlets_count: int
    flat_vols: list[float]
    prices: list[float]
    repriced_prices: list[float | None]
    repricing_errors: list[float | None]


class OptionletNode(BaseModel):
    number: int
    cap_maturity_months: int
    fixing_date: date
    start_date: date
    end_date: date
    payment_date: date
    index_maturity_date: date
    accrual_days: int
    accrual_fraction: float
    projection_accrual_fraction: float
    expiry_years: float
    forward_rate: float
    discount_factor: float
    source: Literal["initialized", "stripped", "converted"]
    volatilities: list[float | None]
    prices: list[float | None]
    smile_strikes: list[float] = Field(default_factory=list)
    smile_volatilities: list[float | None] = Field(default_factory=list)
    smile_sources: list[Literal["absolute_cap", "atm_cap", "basis_converted"]] = Field(default_factory=list)
    block_expiry: str | None = None
    conversion_source_fixings: list[date] = Field(default_factory=list)
    conversion_source_forward: float | None = None
    conversion_forward_spread: float | None = None
    conversion_expiry_extrapolated: bool = False
    conversion_notes: list[str] = Field(default_factory=list)


class StrippingBlock(BaseModel):
    expiry: str
    maturity_months: int
    first_fixing_date: date
    last_fixing_date: date
    optionlets_count: int
    prior_optionlets_count: int
    strikes: list[float]
    volatilities: list[float | None]
    residual_time_values: list[float | None]


class ATMCapDiagnostic(BaseModel):
    expiry: str
    maturity_months: int
    strike: float
    strike_source: Literal["par_from_curves"] = "par_from_curves"
    flat_vol: float | None
    market_price: float | None
    repriced_price: float | None
    repricing_error: float | None
    block_vol: float | None
    block_optionlets_count: int
    status: Literal["included", "unavailable", "missing"]
    message: str


class StrippingIssue(BaseModel):
    strike: float
    cap_maturity_months: int
    code: str
    message: str


class StrippingFamily(BaseModel):
    index_tenor: Literal["3M", "6M"]
    status: Literal["ready", "partial", "failed"]
    strikes: list[float]
    first_quoted_expiry: str
    last_quoted_expiry: str
    caps: list[CapNode]
    optionlets: list[OptionletNode]
    issues: list[StrippingIssue]
    market_count: int
    interpolated_count: int
    initialized_count: int
    max_abs_repricing_error: float | None
    black_surface: BlackVolSurface | None = None
    blocks: list[StrippingBlock] = Field(default_factory=list)
    atm_caps: list[ATMCapDiagnostic] = Field(default_factory=list)
    conversion_coverage: Literal["two_year_expiry_flat_extension"] | None = None
    conversion_notes: list[str] = Field(default_factory=list)


class TargetVolatility(BaseModel):
    index_tenor: Literal["3M", "6M"]
    strike: float
    fixing_date: date
    status: Literal["available", "out_of_domain", "unavailable"]
    normal_vol: float | None
    message: str


class StrippingResult(BaseModel):
    flat_vol_interpolation: Literal["pchip", "none"] = "pchip"
    stripping_method: Literal["block_bootstrap", "pchip"] = "pchip"
    basis_moneyness: bool = True
    include_atm_caps: bool = False
    currency: Literal["EUR"] = "EUR"
    source_token: str
    valuation_date: date
    notional: float
    spot_date: date
    conventions: list[str]
    warnings: list[str]
    families: list[StrippingFamily]
    target: TargetVolatility
