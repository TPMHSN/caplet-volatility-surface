"""
Configuration module for currency-specific conventions and settings.

The codebase now distinguishes two convention families per currency:
- trading_defaults: index-linked OTC-friendly defaults used by swap/XVA/simulation;
- market_defaults: market-standard defaults used by swaption/vol/calibration.

The legacy flat keys (fixed_leg_dc, fixed_leg_freq, etc.) are preserved for
curve-building and older call sites, and are hydrated from market_defaults.
"""

import QuantLib as ql
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from services.exceptions import ValidationError

from config_swaption import SWAPTION_MARKET_CONVENTIONS
from config_types import (
    CurrencyConfig,
    RFRIndexConfig,
    SwapConventionPresetConfig,
    SwaptionIndexTenorConfig,
)


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env")

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    default_currency: str = "EUR"
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        ]
    )


SUPPORTED_DAY_COUNTS = ["ACT/360", "ACT/365", "30/360", "ACT/ACT"]
SUPPORTED_FREQUENCIES = ["Annual", "Semiannual", "Quarterly", "Monthly"]


class UnknownDayCountError(ValidationError, ValueError):
    """A day-count label that no supported convention goes by (never silently ACT/360).

    A client input error (400) for the service layer, and a ``ValueError`` for the low-level
    callers that already catch one.
    """


_DAY_COUNT_ALIASES = {
    "ACT/360": "ACT/360",
    "ACTUAL/360": "ACT/360",
    "ACT/365": "ACT/365",
    "ACTUAL/365": "ACT/365",
    "ACT/365F": "ACT/365",
    "ACT/365FIXED": "ACT/365",
    "ACT/365(FIXED)": "ACT/365",
    "ACTUAL/365F": "ACT/365",
    "ACTUAL/365FIXED": "ACT/365",
    "ACTUAL/365(FIXED)": "ACT/365",
    "30/360": "30/360",
    "30/360(BONDBASIS)": "30/360",
    "ACT/ACT": "ACT/ACT",
    "ACTUAL/ACTUAL": "ACT/ACT",
}


def _normalize_day_count_label(label: str) -> str:
    """Canonical label of a day-count alias; an unknown label is an error, not ACT/360."""
    key = label.upper().replace(" ", "")
    try:
        return _DAY_COUNT_ALIASES[key]
    except KeyError:
        raise UnknownDayCountError(
            f"Unknown day count '{label}'; supported: ACT/360, ACT/365 (ACT/365F), 30/360, ACT/ACT"
        ) from None


def _normalize_frequency_label(label: str) -> str:
    freq_upper = label.upper().replace(" ", "")
    mapping = {
        "ANNUAL": "Annual",
        "SEMIANNUAL": "Semiannual",
        "QUARTERLY": "Quarterly",
        "MONTHLY": "Monthly",
    }
    return mapping.get(freq_upper, "Annual")


def day_count_label_to_ql(label: str) -> ql.DayCounter:
    normalized = _normalize_day_count_label(label)
    if normalized == "ACT/360":
        return ql.Actual360()
    if normalized == "ACT/365":
        return ql.Actual365Fixed()
    if normalized == "30/360":
        return ql.Thirty360(ql.Thirty360.BondBasis)
    return ql.ActualActual(ql.ActualActual.ISDA)


def calendar_label_to_ql(label: str) -> ql.Calendar:
    """Resolve a stable serialized calendar identifier to QuantLib."""
    calendars = {
        "TARGET": ql.TARGET,
        "SOFR": lambda: ql.Sofr().fixingCalendar(),
        "US-GOVERNMENT-BOND": lambda: ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "US-NY": lambda: ql.UnitedStates(ql.UnitedStates.FederalReserve),
        "UK-EXCHANGE": lambda: ql.UnitedKingdom(ql.UnitedKingdom.Exchange),
        "GBLO": lambda: ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
    }
    try:
        return calendars[label.upper()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported calendar identifier: {label}") from exc


def frequency_label_to_ql(label: str) -> int:
    normalized = _normalize_frequency_label(label)
    if normalized == "Annual":
        return ql.Annual
    if normalized == "Semiannual":
        return ql.Semiannual
    if normalized == "Quarterly":
        return ql.Quarterly
    if normalized == "Monthly":
        return ql.Monthly
    return ql.Annual


def _sort_forward_tenors(tenors: list[str]) -> list[str]:
    unit_order = {"D": 1, "W": 2, "M": 3, "Y": 4}

    def key(tenor: str) -> tuple[int, int]:
        tenor = tenor.upper().strip()
        digits = "".join(ch for ch in tenor if ch.isdigit())
        unit = "".join(ch for ch in tenor if ch.isalpha())
        if not digits:
            return (0, 0)  # RFR names (SOFR, SONIA) sort first
        return unit_order.get(unit, 99), int(digits or 0)

    return sorted(tenors, key=key)


USD_SOFR_TRADE_LEGACY_V0: SwapConventionPresetConfig = {
    "preset_id": "USD-SOFR-TRADE-LEGACY-V0",
    "currency": "USD",
    "index_tenor": "SOFR",
    "fixed_day_count": "ACT/360",
    "float_day_count": "ACT/360",
    "fixed_frequency": "Quarterly",
    "float_frequency": "Quarterly",
    "discount_curve_key": "OIS",
    "forward_curve_key": "OIS",
    "settlement_days": 2,
    "payment_lag": 2,
    "lookback_days": 0,
    "lockout_days": 0,
    "observation_shift": False,
    "averaging_method": "compound",
    "fixing_calendar": "SOFR",
    "schedule_calendar": "US-GOVERNMENT-BOND",
    "payment_calendar": "US-GOVERNMENT-BOND",
    "business_day_convention": "modified_following",
    "roll_convention": "auto",
}


USD_SOFR_TRADE_ACT360_ANNUAL_V1: SwapConventionPresetConfig = {
    "preset_id": "USD-SOFR-TRADE-ACT360-ANNUAL-V1",
    "currency": "USD",
    "index_tenor": "SOFR",
    "fixed_day_count": "ACT/360",
    "float_day_count": "ACT/360",
    "fixed_frequency": "Annual",
    "float_frequency": "Annual",
    "discount_curve_key": "OIS",
    "forward_curve_key": "OIS",
    "settlement_days": 2,
    "payment_lag": 2,
    "lookback_days": 0,
    "lockout_days": 0,
    "observation_shift": False,
    "averaging_method": "compound",
    "fixing_calendar": "SOFR",
    "schedule_calendar": "US-GOVERNMENT-BOND",
    "payment_calendar": "US-GOVERNMENT-BOND",
    "business_day_convention": "modified_following",
    "roll_convention": "auto",
}


USD_SOFR_TRADE_ACT360_QUARTERLY_V1: SwapConventionPresetConfig = {
    "preset_id": "USD-SOFR-TRADE-ACT360-QUARTERLY-V1",
    "currency": "USD",
    "index_tenor": "SOFR",
    "fixed_day_count": "ACT/360",
    "float_day_count": "ACT/360",
    "fixed_frequency": "Quarterly",
    "float_frequency": "Quarterly",
    "discount_curve_key": "OIS",
    "forward_curve_key": "OIS",
    "settlement_days": 2,
    "payment_lag": 2,
    "lookback_days": 0,
    "lockout_days": 0,
    "observation_shift": False,
    "averaging_method": "compound",
    "fixing_calendar": "SOFR",
    "schedule_calendar": "US-GOVERNMENT-BOND",
    "payment_calendar": "US-GOVERNMENT-BOND",
    "business_day_convention": "modified_following",
    "roll_convention": "auto",
}


USD_SOFR_TRADE_30_360_SEMIANNUAL_V1: SwapConventionPresetConfig = {
    "preset_id": "USD-SOFR-TRADE-30_360-SEMIANNUAL-V1",
    "currency": "USD",
    "index_tenor": "SOFR",
    "fixed_day_count": "30/360",
    "float_day_count": "ACT/360",
    "fixed_frequency": "Semiannual",
    "float_frequency": "Quarterly",
    "discount_curve_key": "OIS",
    "forward_curve_key": "OIS",
    "settlement_days": 2,
    "payment_lag": 2,
    "lookback_days": 0,
    "lockout_days": 0,
    "observation_shift": False,
    "averaging_method": "compound",
    "fixing_calendar": "SOFR",
    "schedule_calendar": "US-GOVERNMENT-BOND",
    "payment_calendar": "US-GOVERNMENT-BOND",
    "business_day_convention": "modified_following",
    "roll_convention": "auto",
}


def _swaption_index_block(currency: str) -> SwaptionIndexTenorConfig:
    """Floating index of the market swaption underlying, from the one convention table."""
    profile = SWAPTION_MARKET_CONVENTIONS[currency]
    return {
        "short": profile.short_index,
        "long": profile.long_index,
        "short_swap_max_years": profile.short_max_months / 12.0,
    }


# Currency-specific configuration
CURRENCY_CONFIG: dict[str, CurrencyConfig] = {
    "EUR": {
        "calendar": ql.TARGET(),
        "settlement_days": 2,
        "business_day_convention": ql.ModifiedFollowing,
        "ois_dc": ql.Actual360(),
        "ql_currency": ql.EURCurrency(),
        "swap_index_name": "EuriborSwapIsdaFixA",
        "rfr_indices": {},
        "trading_defaults": {
            "mode": "index_linked",
            "fixed_day_count": "ACT/360",
            "float_day_count": "ACT/360",
            "fixed_frequency": "index_tenor",
            "float_frequency": "index_tenor",
            "discount_curve_key": "OIS",
            "forward_curve_key": "index_tenor",
            "frequency_by_index_tenor": {
                "1M": "Monthly",
                "3M": "Quarterly",
                "6M": "Semiannual",
                "12M": "Annual",
            },
        },
        "market_defaults": {
            "mode": "market_standard",
            "fixed_day_count": SWAPTION_MARKET_CONVENTIONS["EUR"].fixed_day_count,
            "float_day_count": SWAPTION_MARKET_CONVENTIONS["EUR"].float_day_count,
            "fixed_frequency": SWAPTION_MARKET_CONVENTIONS["EUR"].fixed_frequency,
            "float_frequency": "index_tenor",
            "discount_curve_key": "OIS",
            "forward_curve_key": "index_tenor",
            "frequency_by_index_tenor": {
                "1M": "Monthly",
                "3M": "Quarterly",
                "6M": "Semiannual",
                "12M": "Annual",
            },
            "swaption_index_tenor": _swaption_index_block("EUR"),
        },
        "index_family": {
            "OIS": lambda: ql.Estr(),
            "1M": lambda: ql.Euribor1M(),
            "3M": lambda: ql.Euribor3M(),
            "6M": lambda: ql.Euribor6M(),
            "12M": lambda: ql.Euribor1Y(),
        },
        "index_with_curve_family": {
            "1M": lambda h: ql.Euribor1M(h),
            "3M": lambda h: ql.Euribor3M(h),
            "6M": lambda h: ql.Euribor6M(h),
            "12M": lambda h: ql.Euribor1Y(h),
        },
        # XVAEUR X6-OIS (Q-B1 (C)): the EUR overnight swap index, read only by the swap builder,
        # the convention resolution and the XVA; not an RFR tenor (``rfr_indices`` stays empty, so
        # the swap form keeps 3M/6M) and not a forward tenor. Defaults (Q-B2): payment lag of one
        # TARGET business day as the EUR-ESTR-OIS curve quotes, no lookback, lockout or shift.
        "overnight_swap_indices": {
            "OIS": {
                "index_with_curve": lambda h: ql.Estr(h),
                "defaults": {
                    "payment_lag": 1,
                    "lookback_days": 0,
                    "lockout_days": 0,
                    "observation_shift": False,
                    "averaging_method": "compound",
                },
            },
        },
    },
    "GBP": {
        "calendar": ql.UnitedKingdom(ql.UnitedKingdom.Exchange),
        "settlement_days": 0,  # SONIA T+0
        "business_day_convention": ql.ModifiedFollowing,
        "ois_dc": ql.Actual365Fixed(),
        "ql_currency": ql.GBPCurrency(),
        "swap_index_name": None,
        "rfr_indices": {
            "SONIA": {
                "is_default": True,
                "payment_lag": 0,
                "lookback_days": 0,
                "lockout_days": 0,
                "observation_shift": False,
                "averaging_method": "compound",
            },
        },
        "trading_defaults": {
            "mode": "index_linked",
            "fixed_day_count": "ACT/365",
            "float_day_count": "ACT/365",
            "fixed_frequency": "index_tenor",
            "float_frequency": "index_tenor",
            "discount_curve_key": "OIS",
            "forward_curve_key": "index_tenor",
            "frequency_by_index_tenor": {
                "SONIA": "Quarterly",
                "1M": "Monthly",
                "3M": "Quarterly",
                "6M": "Semiannual",
                "12M": "Annual",
            },
        },
        "market_defaults": {
            "mode": "market_standard",
            "fixed_day_count": SWAPTION_MARKET_CONVENTIONS["GBP"].fixed_day_count,
            "float_day_count": SWAPTION_MARKET_CONVENTIONS["GBP"].float_day_count,
            "fixed_frequency": SWAPTION_MARKET_CONVENTIONS["GBP"].fixed_frequency,
            "float_frequency": "index_tenor",
            "discount_curve_key": "OIS",
            "forward_curve_key": "index_tenor",
            "frequency_by_index_tenor": {
                "SONIA": "Quarterly",
                "1M": "Monthly",
                "3M": "Quarterly",
                "6M": "Semiannual",
                "12M": "Annual",
            },
            "swaption_index_tenor": _swaption_index_block("GBP"),
        },
        "index_family": {
            "OIS": lambda: ql.Sonia(),
            "SONIA": lambda: ql.Sonia(),
        },
        "index_with_curve_family": {
            "SONIA": lambda h: ql.Sonia(h),
        },
    },
    "USD": {
        "calendar": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "settlement_days": 2,
        "business_day_convention": ql.ModifiedFollowing,
        "ois_dc": ql.Actual360(),
        "ql_currency": ql.USDCurrency(),
        "swap_index_name": None,  # SOFR OIS swaps only: no LIBOR swap index
        "rfr_indices": {
            "SOFR": {
                "is_default": True,
                "payment_lag": 2,
                "lookback_days": 0,
                "lockout_days": 0,
                "observation_shift": False,
                "averaging_method": "compound",
            },
        },
        "swap_convention_presets": {
            USD_SOFR_TRADE_LEGACY_V0["preset_id"]: USD_SOFR_TRADE_LEGACY_V0,
            USD_SOFR_TRADE_ACT360_ANNUAL_V1["preset_id"]: USD_SOFR_TRADE_ACT360_ANNUAL_V1,
            USD_SOFR_TRADE_ACT360_QUARTERLY_V1["preset_id"]: (USD_SOFR_TRADE_ACT360_QUARTERLY_V1),
            USD_SOFR_TRADE_30_360_SEMIANNUAL_V1["preset_id"]: (USD_SOFR_TRADE_30_360_SEMIANNUAL_V1),
        },
        "default_swap_convention_preset_id": USD_SOFR_TRADE_ACT360_ANNUAL_V1["preset_id"],
        "trading_defaults": {
            "mode": "index_linked",
            "fixed_day_count": "ACT/360",
            "float_day_count": "ACT/360",
            "fixed_frequency": "index_tenor",
            "float_frequency": "index_tenor",
            "discount_curve_key": "OIS",
            "forward_curve_key": "index_tenor",
            "frequency_by_index_tenor": {
                "SOFR": "Annual",
                "1M": "Monthly",
                "3M": "Quarterly",
                "6M": "Semiannual",
                "12M": "Annual",
            },
        },
        "market_defaults": {
            "mode": "market_standard",
            "fixed_day_count": SWAPTION_MARKET_CONVENTIONS["USD"].fixed_day_count,
            "float_day_count": SWAPTION_MARKET_CONVENTIONS["USD"].float_day_count,
            "fixed_frequency": SWAPTION_MARKET_CONVENTIONS["USD"].fixed_frequency,
            "float_frequency": "index_tenor",
            "discount_curve_key": "OIS",
            "forward_curve_key": "index_tenor",
            "frequency_by_index_tenor": {
                "SOFR": "Annual",
                "1M": "Monthly",
                "3M": "Quarterly",
                "6M": "Semiannual",
                "12M": "Annual",
            },
            "swaption_index_tenor": _swaption_index_block("USD"),
        },
        "index_family": {
            "OIS": lambda: ql.Sofr(),
            "SOFR": lambda: ql.Sofr(),
            "1M": lambda: ql.USDLibor(ql.Period(1, ql.Months)),
            "3M": lambda: ql.USDLibor(ql.Period(3, ql.Months)),
            "6M": lambda: ql.USDLibor(ql.Period(6, ql.Months)),
            "12M": lambda: ql.USDLibor(ql.Period(12, ql.Months)),
        },
        "index_with_curve_family": {
            "SOFR": lambda h: ql.Sofr(h),
            "1M": lambda h: ql.USDLibor(ql.Period(1, ql.Months), h),
            "3M": lambda h: ql.USDLibor(ql.Period(3, ql.Months), h),
            "6M": lambda h: ql.USDLibor(ql.Period(6, ql.Months), h),
            "12M": lambda h: ql.USDLibor(ql.Period(12, ql.Months), h),
        },
    },
}


def _hydrate_legacy_fields(currency: str, config: CurrencyConfig) -> None:
    market_defaults = config["market_defaults"]
    supported_forward_tenors = _sort_forward_tenors(list(config["index_with_curve_family"].keys()))
    default_swaption_index_tenor = market_defaults["swaption_index_tenor"]["long"]

    config["supported_day_counts"] = list(SUPPORTED_DAY_COUNTS)
    config["supported_frequencies"] = list(SUPPORTED_FREQUENCIES)
    config["supported_forward_tenors"] = supported_forward_tenors
    config["default_swaption_index_tenor"] = default_swaption_index_tenor

    config["fixed_leg_dc"] = day_count_label_to_ql(market_defaults["fixed_day_count"])
    config["float_leg_dc"] = day_count_label_to_ql(market_defaults["float_day_count"])
    config["fixed_leg_freq"] = frequency_label_to_ql(market_defaults["fixed_frequency"])


for _currency, _config in CURRENCY_CONFIG.items():
    _hydrate_legacy_fields(_currency, _config)


def get_currency_config(currency: str) -> CurrencyConfig:
    """
    Get configuration for a specific currency.

    Args:
        currency: Currency code (e.g., 'EUR', 'USD')

    Returns:
        Dictionary with currency-specific configuration

    Raises:
        ValueError: If currency is not supported
    """
    if currency not in CURRENCY_CONFIG:
        raise ValueError(
            f"Currency {currency} not supported. Available: {list(CURRENCY_CONFIG.keys())}"
        )
    return CURRENCY_CONFIG[currency]


def get_swap_convention_preset(
    currency: str,
    preset_id: str,
) -> SwapConventionPresetConfig:
    """Resolve a versioned swap convention preset for one currency."""
    config = get_currency_config(currency)
    presets = config.get("swap_convention_presets", {})
    try:
        return presets[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown swap convention preset for {currency}: {preset_id}") from exc


def get_supported_forward_tenors(currency: str) -> list[str]:
    """Return supported forwarding tenors for a currency."""
    config = get_currency_config(currency)
    return list(config["supported_forward_tenors"])


def get_index(currency: str, tenor: str) -> ql.IborIndex:
    """
    Get the appropriate index for a currency and tenor.

    Args:
        currency: Currency code (e.g., 'EUR', 'USD')
        tenor: Index tenor (e.g., '3M', '6M', 'OIS')

    Returns:
        QuantLib IborIndex or OvernightIndex
    """
    config = get_currency_config(currency)
    index_family = config["index_family"]

    if tenor not in index_family:
        raise ValueError(
            f"Tenor {tenor} not supported for {currency}. Available: {list(index_family.keys())}"
        )

    return index_family[tenor]()


def get_index_with_curve(
    currency: str, tenor: str, curve_handle: ql.YieldTermStructureHandle
) -> ql.IborIndex:
    """
    Get an index linked to a specific forward curve.

    This is essential for pricing instruments where the forwarding curve
    is explicitly specified (e.g., in multi-curve setups).

    Args:
        currency: Currency code (e.g., 'EUR', 'USD')
        tenor: Index tenor (e.g., '3M', '6M')
        curve_handle: YieldTermStructureHandle to link to the index

    Returns:
        QuantLib IborIndex linked to the provided curve

    Raises:
        ValueError: If currency/tenor combination is not supported
    """
    config = get_currency_config(currency)
    family = config["index_with_curve_family"]
    if tenor not in family:
        raise ValueError(
            f"Tenor {tenor} not supported for {currency}. "
            f"Available: {list(family.keys()) or ['OIS only']}"
        )
    return family[tenor](curve_handle)


def get_swap_index(
    currency: str,
    tenor: str,
    forward_curve: ql.YieldTermStructureHandle,
    discount_curve: ql.YieldTermStructureHandle,
    index_tenor: str = "6M",
) -> ql.SwapIndex:
    """
    Get a SwapIndex for swaption vol cube construction and pricing.

    Args:
        currency: Currency code (e.g., 'EUR')
        tenor: Swap tenor (e.g., '10Y', '1Y')
        forward_curve: Forward curve handle
        discount_curve: Discount curve handle
        index_tenor: Floating index tenor used for forwarding (e.g. '1M', '3M', '6M')

    Returns:
        QuantLib SwapIndex
    """
    config = get_currency_config(currency)
    settlement_days = config["settlement_days"]

    if is_rfr_index(currency, index_tenor):
        overnight_index = get_index_with_curve(currency, index_tenor, forward_curve)
        return ql.OvernightIndexedSwapIndex(
            f"{currency}-{index_tenor.upper()}-OIS",
            ql.Period(tenor),
            settlement_days,
            config["ql_currency"],
            overnight_index,
        )

    swap_index_name = config["swap_index_name"]
    if not swap_index_name:
        raise ValueError(f"SwapIndex not configured for {currency}")

    calendar = config["calendar"]
    fixed_leg_dc = config["fixed_leg_dc"]
    fixed_leg_freq = config["fixed_leg_freq"]
    bdc = config["business_day_convention"]

    freq_to_period = {
        ql.Annual: ql.Period(1, ql.Years),
        ql.Semiannual: ql.Period(6, ql.Months),
        ql.Quarterly: ql.Period(3, ql.Months),
        ql.Monthly: ql.Period(1, ql.Months),
    }
    fixed_leg_period = freq_to_period.get(fixed_leg_freq, ql.Period(1, ql.Years))

    ibor_index = get_index_with_curve(currency, index_tenor, forward_curve)
    return ql.SwapIndex(
        swap_index_name,
        ql.Period(tenor),
        settlement_days,
        config["ql_currency"],
        calendar,
        fixed_leg_period,
        bdc,
        fixed_leg_dc,
        ibor_index,
        discount_curve,
    )


def is_rfr_index(currency: str, tenor: str) -> bool:
    """Check if a given index tenor is an RFR (overnight compounded) index."""
    config = get_currency_config(currency)
    return tenor.upper() in config["rfr_indices"]


def get_rfr_defaults(currency: str, tenor: str) -> RFRIndexConfig:
    """Get default RFR parameters for a currency/tenor pair.

    Returns:
        Dict with keys: payment_lag, lookback_days, lockout_days,
        observation_shift, averaging_method
    """
    config = get_currency_config(currency)
    rfr = config["rfr_indices"]
    key = tenor.upper()
    if key not in rfr:
        raise ValueError(f"{tenor} is not an RFR index for {currency}")
    return dict(rfr[key])


def is_overnight_swap_index(currency: str, tenor: str) -> bool:
    """Whether a swap on ``tenor`` has an overnight (compounded) floating leg.

    The RFR tenors of the currency, or its dedicated overnight swap index (EUR ``OIS``, XVAEUR
    X6-OIS), which is not exposed as an RFR tenor.
    """
    config = get_currency_config(currency)
    key = tenor.upper().strip()
    return key in config["rfr_indices"] or key in config.get("overnight_swap_indices", {})


def is_dedicated_overnight_swap_index(currency: str, tenor: str) -> bool:
    """Whether ``tenor`` is the currency's dedicated (non-RFR) overnight swap index (EUR ``OIS``)."""
    config = get_currency_config(currency)
    key = tenor.upper().strip()
    return key not in config["rfr_indices"] and key in config.get("overnight_swap_indices", {})


def get_overnight_swap_index_with_curve(
    currency: str, tenor: str, curve_handle: ql.YieldTermStructureHandle
) -> ql.OvernightIndex:
    """The overnight index of a swap on ``tenor``, linked to ``curve_handle``."""
    if is_dedicated_overnight_swap_index(currency, tenor):
        entry = get_currency_config(currency)["overnight_swap_indices"][tenor.upper().strip()]
        return entry["index_with_curve"](curve_handle)
    return get_index_with_curve(currency, tenor, curve_handle)


def get_overnight_swap_defaults(currency: str, tenor: str) -> RFRIndexConfig:
    """Default overnight-leg parameters of a swap on ``tenor`` (RFR tenor or dedicated index)."""
    if is_dedicated_overnight_swap_index(currency, tenor):
        entry = get_currency_config(currency)["overnight_swap_indices"][tenor.upper().strip()]
        return dict(entry["defaults"])
    return get_rfr_defaults(currency, tenor)


def get_default_index_tenor(currency: str) -> str:
    """Return the default floating index tenor for a currency.

    Returns the default RFR tenor if one exists, otherwise the first IBOR tenor.
    """
    config = get_currency_config(currency)
    rfr = config["rfr_indices"]
    for name, params in rfr.items():
        if params.get("is_default"):
            return name
    tenors = list(config["index_with_curve_family"].keys())
    return tenors[0] if tenors else "OIS"


settings = Settings()
