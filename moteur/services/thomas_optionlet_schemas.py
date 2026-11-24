"""Explicit contract for the first single-period Caps Thomas calculation."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class OptionletRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: Literal["EUR"] = "EUR"
    instrument: Literal["caplet", "floorlet"]
    index_tenor: Literal["3M", "6M"]
    notional: FiniteFloat = Field(gt=0)
    valuation_date: date
    fixing_date: date
    strike: FiniteFloat
    source_token: str = Field(min_length=1)
    stripping_method: Literal["block_bootstrap", "pchip"] = "block_bootstrap"
    basis_moneyness: bool = True
    include_atm_caps: bool = True


class LoadedCurve(BaseModel):
    key: str
    label: str
    market_date: date | None
    loaded: bool
    built: bool
    current: bool


class LoadedContext(BaseModel):
    currency: Literal["EUR"] = "EUR"
    market_date: date | None
    curve_date: date | None
    curves: list[LoadedCurve]
    cap_quotes_loaded: bool
    cap_quotes_date: date | None
    cap_expiry_count: int
    source_token: str


class OptionletPreparation(BaseModel):
    status: Literal["ready_for_volatility"] = "ready_for_volatility"
    request: OptionletRequest
    source_token: str
    curve_date: date
    projection_curve: str
    discount_curve: str
    interpolation: str
    start_date: date
    end_date: date
    payment_date: date
    accrual_days: int
    accrual_fraction: float
    expiry_days: int
    expiry_years: float
    projection_factor_start: float
    projection_factor_end: float
    forward_rate: float
    discount_factor: float
    atm_strike: float
    forward_minus_strike: float
    discounted_accrual_notional: float
    calendar: Literal["TARGET"] = "TARGET"
    fixing_lag_business_days: Literal[2] = 2
    payment_lag_business_days: Literal[0] = 0
    accrual_day_count: Literal["ACT/360"] = "ACT/360"
    option_time_day_count: Literal["ACT/365F"] = "ACT/365F"
    business_day_convention: Literal["Modified Following"] = "Modified Following"
    end_of_month: bool = True
