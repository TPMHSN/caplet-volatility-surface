"""Read-only data contract for the first Caps Thomas stage."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

TenorPolicy = Literal["market", "3M", "6M"]


class DataIssue(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    sheet: str | None = None
    cell: str | None = None


class SourceSheet(BaseModel):
    name: str
    role: str
    used: bool


class CurveQuote(BaseModel):
    term: str
    term_cell: str
    value_cell: str
    raw_value: float | None
    rate_decimal: float | None
    instrument_hint: str
    raw_unit: str


class CurveInput(BaseModel):
    sheet: str
    purpose: str
    index_tenor: str | None
    quotes: list[CurveQuote] = Field(default_factory=list)


class CapStrike(BaseModel):
    label: str
    header_cell: str
    strike_decimal: float


class CapQuote(BaseModel):
    cell: str
    raw_value: float | None
    normal_vol: float | None


class CapRow(BaseModel):
    expiry: str
    expiry_cell: str
    expiry_months: int
    quote_index_tenor: str
    atm: CapQuote | None
    quotes: list[CapQuote]


class CapSurfaceInput(BaseModel):
    strikes: list[CapStrike] = Field(default_factory=list)
    rows: list[CapRow] = Field(default_factory=list)


class DataCounts(BaseModel):
    curve_quotes: int
    expiries: int
    fixed_strikes: int
    cap_quotes: int
    atm_quotes: int


class ThomasCapsMarketData(BaseModel):
    source_filename: str
    valuation_date: date
    date_source: Literal["filename", "user"]
    currency: Literal["EUR"] = "EUR"
    quote_convention: Literal["normal_bp"] = "normal_bp"
    quote_tenor_policy: TenorPolicy
    status: Literal["ready_for_curves", "needs_attention"]
    sheets: list[SourceSheet]
    curves: list[CurveInput]
    cap_surface: CapSurfaceInput
    counts: DataCounts
    issues: list[DataIssue]
