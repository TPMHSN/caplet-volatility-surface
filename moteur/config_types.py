"""Typed configuration shapes for currency and convention settings."""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

import QuantLib as ql

FrequencyLabel = Literal["Annual", "Semiannual", "Quarterly", "Monthly"]
ConventionMode = Literal["index_linked", "market_standard"]

IndexFactory = Callable[[], ql.IborIndex]
IndexWithCurveFactory = Callable[[ql.YieldTermStructureHandle], ql.IborIndex]


class SwaptionIndexTenorConfig(TypedDict):
    """Swaption tenor split between short and long underlyings."""

    short: str
    long: str
    short_swap_max_years: float


class RFRIndexConfig(TypedDict, total=False):
    """Default conventions for compounded overnight indices."""

    is_default: bool
    payment_lag: int
    lookback_days: int
    lockout_days: int
    observation_shift: bool
    averaging_method: str


OvernightIndexWithCurveFactory = Callable[[ql.YieldTermStructureHandle], ql.OvernightIndex]


class OvernightSwapIndexConfig(TypedDict):
    """Overnight index of a swap that is not exposed as an RFR tenor (XVAEUR X6-OIS, Q-B1 (C)).

    Read only by the swap builder, the convention resolution and the XVA: the conventions endpoint,
    the forward tenor lists and the presets do not see it.
    """

    index_with_curve: OvernightIndexWithCurveFactory
    defaults: RFRIndexConfig


class SwapConventionPresetConfig(TypedDict):
    """Versioned trade convention recipe for a swap family."""

    preset_id: str
    currency: str
    index_tenor: str
    fixed_day_count: str
    float_day_count: str
    fixed_frequency: FrequencyLabel
    float_frequency: FrequencyLabel
    discount_curve_key: str
    forward_curve_key: str
    settlement_days: int
    payment_lag: int
    lookback_days: int
    lockout_days: int
    observation_shift: bool
    averaging_method: Literal["compound", "simple"]
    fixing_calendar: str
    schedule_calendar: str
    payment_calendar: str
    business_day_convention: str
    roll_convention: str


class ConventionDefaultsConfig(TypedDict):
    """Base convention block shared by trading and market defaults."""

    mode: ConventionMode
    fixed_day_count: str
    float_day_count: str
    fixed_frequency: str
    float_frequency: str
    discount_curve_key: str
    forward_curve_key: str
    frequency_by_index_tenor: dict[str, FrequencyLabel]


class TradingDefaultsConfig(ConventionDefaultsConfig):
    """Trading-oriented convention defaults."""


class MarketDefaultsConfig(ConventionDefaultsConfig):
    """Market-standard convention defaults."""

    swaption_index_tenor: SwaptionIndexTenorConfig


class CurrencyConfig(TypedDict, total=False):
    """Full configuration payload used across pricing and market-data modules."""

    calendar: ql.Calendar
    settlement_days: int
    business_day_convention: int
    ois_dc: ql.DayCounter
    ql_currency: ql.Currency
    swap_index_name: str | None
    rfr_indices: dict[str, RFRIndexConfig]
    overnight_swap_indices: dict[str, OvernightSwapIndexConfig]
    swap_convention_presets: dict[str, SwapConventionPresetConfig]
    default_swap_convention_preset_id: str | None
    trading_defaults: TradingDefaultsConfig
    market_defaults: MarketDefaultsConfig
    index_family: dict[str, IndexFactory]
    index_with_curve_family: dict[str, IndexWithCurveFactory]
    supported_day_counts: list[str]
    supported_frequencies: list[str]
    supported_forward_tenors: list[str]
    default_swaption_index_tenor: str
    fixed_leg_dc: ql.DayCounter
    float_leg_dc: ql.DayCounter
    fixed_leg_freq: int
