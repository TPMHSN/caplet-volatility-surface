"""Versioned swaption market-convention table (CUBESWPT L2): the one source.

One profile per currency (``EUR-SWPT-V1``, ``USD-SWPT-V1``, ``GBP-SWPT-V1``) says which swap a
quoted swaption is on and how the option is read: the underlying (fixed leg, floating index and
its 1Y boundary), the expiry rule, the spot lag, the discounting curve, the settlement methods and
the time basis. ``config``, ``market_data.conventions``, the template loader, the swaption
convention helpers and, through the generated ``frontend/src/generated/swaptionConventions.ts``,
the frontend all read this table; none of them keeps its own copy of the index.

Only the EUR cube is built (``cube_supported``): the USD SOFR and GBP SONIA profiles are correct
data, waiting for a later chantier to build their cubes.

Readers: the index (``short_index``, ``long_index``, ``short_max_months``), the keys and options of
the data sets, the underlying legs' defaults, the expiry calendar of the ATM matrix, the cube flag
and the convention id. ``spot_lag_days``, ``expiry_business_day_convention``, ``time_basis`` and
``discount_curve_key`` are documented here and checked against ``config`` by a test; they are not
read by the pricing code yet (D-23, 'Par Yield' refusal, and the hard-coded ``"OIS"`` key of the
cube lineage belong to later lots).

This module is a leaf (no import of ``config``) so that ``config`` can read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 'Par Yield Curve - Unadjusted' settlement is refused rather than priced as physical (D-23).
SETTLEMENT_METHODS = ("physical", "collateralized_cash")
TIME_BASIS = "ACT/365F"
GENERATED_TS_PATH = "frontend/src/generated/swaptionConventions.ts"


@dataclass(frozen=True)
class SwaptionMarketConvention:
    """Market conventions of one currency's quoted swaptions."""

    profile_id: str
    currency: str
    # Key of the market data sets and of the cube (the forwarding curve family it is quoted on).
    default_index_tenor: str
    # Forwarding keys the market-data screens offer for this currency's swaption sheets.
    index_tenor_options: tuple[str, ...]
    # Floating index of the underlying: up to ``short_max_months`` months, then the long one.
    short_index: str
    long_index: str
    short_max_months: int
    fixed_day_count: str
    fixed_frequency: str
    float_day_count: str
    # Expiry = market date + tenor on this calendar and convention; the swap starts ``spot_lag_days``
    # business days later.
    expiry_calendar: str
    expiry_business_day_convention: str
    spot_lag_days: int
    discount_curve_key: str
    settlement_methods: tuple[str, ...]
    time_basis: str
    cube_supported: bool


SWAPTION_MARKET_CONVENTIONS: dict[str, SwaptionMarketConvention] = {
    "EUR": SwaptionMarketConvention(
        profile_id="EUR-SWPT-V1",
        currency="EUR",
        default_index_tenor="6M",
        index_tenor_options=("6M", "3M", "1M", "12M"),
        short_index="3M",
        long_index="6M",
        short_max_months=12,
        fixed_day_count="30/360",
        fixed_frequency="Annual",
        float_day_count="ACT/360",
        expiry_calendar="TARGET",
        expiry_business_day_convention="ModifiedFollowing",
        spot_lag_days=2,
        discount_curve_key="OIS",
        settlement_methods=SETTLEMENT_METHODS,
        time_basis=TIME_BASIS,
        cube_supported=True,
    ),
    "USD": SwaptionMarketConvention(
        profile_id="USD-SWPT-V1",
        currency="USD",
        default_index_tenor="SOFR",
        index_tenor_options=("SOFR",),
        short_index="SOFR",
        long_index="SOFR",
        short_max_months=12,
        fixed_day_count="ACT/360",
        fixed_frequency="Annual",
        float_day_count="ACT/360",
        # The reference (D-24) retains the swap calendar USNY, to confirm when a USD cube is built;
        # the existing USD calendar of ``config`` is kept meanwhile (57 days differ in 2015-2060).
        expiry_calendar="US-GOVERNMENT-BOND",
        expiry_business_day_convention="ModifiedFollowing",
        spot_lag_days=2,
        discount_curve_key="OIS",
        settlement_methods=SETTLEMENT_METHODS,
        time_basis=TIME_BASIS,
        cube_supported=False,
    ),
    "GBP": SwaptionMarketConvention(
        profile_id="GBP-SWPT-V1",
        currency="GBP",
        default_index_tenor="SONIA",
        index_tenor_options=("SONIA",),
        short_index="SONIA",
        long_index="SONIA",
        short_max_months=12,
        fixed_day_count="ACT/365",
        fixed_frequency="Annual",
        float_day_count="ACT/365",
        # Equal to GBLO (UnitedKingdom.Settlement) on every day of 2015-2060.
        expiry_calendar="UK-EXCHANGE",
        expiry_business_day_convention="ModifiedFollowing",
        spot_lag_days=0,
        discount_curve_key="OIS",
        settlement_methods=SETTLEMENT_METHODS,
        time_basis=TIME_BASIS,
        cube_supported=False,
    ),
}


def get_swaption_market_convention(currency: str) -> Optional[SwaptionMarketConvention]:
    """The currency's profile, or None for a currency the table does not carry."""
    return SWAPTION_MARKET_CONVENTIONS.get(currency.upper())


def render_typescript_table() -> str:
    """The frontend's copy of the table, generated (never edited): see the export script."""
    lines = [
        "// Generated from backend/config_swaption.py (SwaptionMarketConvention).",
        "// Do not edit: run scripts/export_swaption_conventions.py after changing the table.",
        "",
        "export interface SwaptionMarketConventionRow {",
        "  profile_id: string",
        "  currency: string",
        "  default_index_tenor: string",
        "  index_tenor_options: string[]",
        "  short_index: string",
        "  long_index: string",
        "  short_max_months: number",
        "  cube_supported: boolean",
        "}",
        "",
        "export const SWAPTION_MARKET_CONVENTIONS: Record<string, SwaptionMarketConventionRow> = {",
    ]
    for currency, profile in SWAPTION_MARKET_CONVENTIONS.items():
        options = ", ".join(f"'{option}'" for option in profile.index_tenor_options)
        lines += [
            f"  {currency}: {{",
            f"    profile_id: '{profile.profile_id}',",
            f"    currency: '{currency}',",
            f"    default_index_tenor: '{profile.default_index_tenor}',",
            f"    index_tenor_options: [{options}],",
            f"    short_index: '{profile.short_index}',",
            f"    long_index: '{profile.long_index}',",
            f"    short_max_months: {profile.short_max_months},",
            f"    cube_supported: {'true' if profile.cube_supported else 'false'},",
            "  },",
        ]
    lines += ["}", ""]
    return "\n".join(lines)
