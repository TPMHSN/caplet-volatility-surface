"""Canonical, serializable convention profiles for curve calibration quotes."""

from dataclasses import dataclass
from typing import Literal

QuoteProvenance = Literal["market_quote", "fixing_observed"]
CalibrationRole = Literal["helper", "curve_anchor_1d", "observation"]

CurveInstrumentTypeValue = Literal[
    "fixing",
    "deposit",
    "fra",
    "future",
    "ois_swap",
    "irs_swap",
]


@dataclass(frozen=True)
class CurveQuoteConvention:
    """Market convention resolved by the identifier stored on each quote."""

    convention_id: str
    instrument_type: CurveInstrumentTypeValue
    index_name: str
    settlement_days: int
    calendar: str
    business_day_convention: str
    day_count: str
    end_of_month: bool
    fixed_leg_frequency: str | None = None
    fixed_leg_day_count: str | None = None
    floating_leg_frequency: str | None = None
    floating_leg_day_count: str | None = None
    schedule_generation_rule: str | None = None
    payment_lag_days: int = 0
    rate_cutoff_days: int = 0
    discount_curve_key: str | None = None
    forward_curve_key: str | None = None
    payment_calendar: str | None = None
    fixed_calendar: str | None = None
    overnight_calendar: str | None = None
    quote_provenance: QuoteProvenance = "market_quote"
    calibration_role: CalibrationRole = "helper"
    quote_unit: Literal["rate", "price"] | None = None
    contract_resolution: str | None = None
    convexity_sigma: float | None = None
    convexity_sigma_source: str | None = None
    convexity_sigma_as_of: str | None = None
    convexity_time_basis: str | None = None
    pending_source_confirmation: str | None = None

    @property
    def used_in_bootstrap(self) -> bool:
        """A quote becomes a calibration helper unless it is kept as an observation."""
        return self.calibration_role != "observation"


EUR_ESTR_DEPOSIT = CurveQuoteConvention(
    convention_id="EUR-ESTR-DEPOSIT-V2",
    instrument_type="deposit",
    index_name="EUR-ESTR",
    settlement_days=0,
    calendar="TARGET",
    business_day_convention="Following",
    day_count="ACT/360",
    end_of_month=False,
    floating_leg_frequency="Daily",
    floating_leg_day_count="ACT/360",
    discount_curve_key="OIS",
    forward_curve_key="OIS",
    quote_provenance="fixing_observed",
    calibration_role="curve_anchor_1d",
)

EUR_ESTR_OIS = CurveQuoteConvention(
    convention_id="EUR-ESTR-OIS-V2",
    instrument_type="ois_swap",
    index_name="EUR-ESTR",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    fixed_leg_frequency="Annual",
    fixed_leg_day_count="ACT/360",
    floating_leg_frequency="Daily",
    floating_leg_day_count="ACT/360",
    schedule_generation_rule="Backward",
    payment_lag_days=1,
    rate_cutoff_days=0,
    discount_curve_key="OIS",
    forward_curve_key="OIS",
)

USD_SOFR_OIS_ACT360_ANNUAL = CurveQuoteConvention(
    convention_id="USD-SOFR-OIS-ACT360-ANNUAL-V2",
    instrument_type="ois_swap",
    index_name="USD-SOFR",
    settlement_days=2,
    calendar="US-NY",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    fixed_leg_frequency="Annual",
    fixed_leg_day_count="ACT/360",
    floating_leg_frequency="Daily",
    floating_leg_day_count="ACT/360",
    schedule_generation_rule="Backward",
    payment_lag_days=2,
    rate_cutoff_days=0,
    discount_curve_key="OIS",
    forward_curve_key="OIS",
    payment_calendar="US-NY",
    fixed_calendar="US-NY",
    overnight_calendar="SOFR",
)

GBP_SONIA_OIS_ACT365_ANNUAL = CurveQuoteConvention(
    convention_id="GBP-SONIA-OIS-ACT365-ANNUAL-V2",
    instrument_type="ois_swap",
    index_name="GBP-SONIA",
    settlement_days=0,
    calendar="GBLO",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/365",
    end_of_month=True,
    fixed_leg_frequency="Annual",
    fixed_leg_day_count="ACT/365",
    floating_leg_frequency="Daily",
    floating_leg_day_count="ACT/365",
    schedule_generation_rule="Backward",
    payment_lag_days=0,
    rate_cutoff_days=0,
    discount_curve_key="OIS",
    forward_curve_key="OIS",
    payment_calendar="GBLO",
    fixed_calendar="GBLO",
    overnight_calendar="UK-EXCHANGE",
)

# One-day SONIA anchor, quoted as a deposit like the €STR overnight (``EUR_ESTR_DEPOSIT``). An OIS
# helper of one day at T+0 degenerates when the evaluation date is the last business day of a
# month falling on a Friday (its Saturday end rolls back to the start under Modified Following);
# a deposit rolls forward (Following) over the weekend, as the market overnight does.
GBP_SONIA_DEPOSIT = CurveQuoteConvention(
    convention_id="GBP-SONIA-DEPOSIT-ACT365-V2",
    instrument_type="deposit",
    index_name="GBP-SONIA",
    settlement_days=0,
    calendar="GBLO",
    business_day_convention="Following",
    day_count="ACT/365",
    end_of_month=False,
    floating_leg_frequency="Daily",
    floating_leg_day_count="ACT/365",
    discount_curve_key="OIS",
    forward_curve_key="OIS",
    quote_provenance="fixing_observed",
    calibration_role="curve_anchor_1d",
)


EUR_EURIBOR_1M_FIXING = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-1M-FIXING-V2",
    instrument_type="fixing",
    index_name="EUR-EURIBOR-1M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    floating_leg_frequency="Monthly",
    floating_leg_day_count="ACT/360",
    forward_curve_key="1M",
    quote_provenance="fixing_observed",
    calibration_role="observation",
)

EUR_EURIBOR_1M_DEPOSIT = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-1M-DEPOSIT-V2",
    instrument_type="deposit",
    index_name="EUR-EURIBOR-1M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    floating_leg_frequency="Monthly",
    floating_leg_day_count="ACT/360",
    forward_curve_key="1M",
)

EUR_EURIBOR_1M_SHORT_IRS = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-1M-SHORT-IRS-V2",
    instrument_type="irs_swap",
    index_name="EUR-EURIBOR-1M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    fixed_leg_frequency="Annual",
    fixed_leg_day_count="ACT/360",
    floating_leg_frequency="Monthly",
    floating_leg_day_count="ACT/360",
    schedule_generation_rule="Backward",
    payment_lag_days=0,
    discount_curve_key="OIS",
    forward_curve_key="1M",
    pending_source_confirmation=(
        "D-17: fixed-leg day count and switch tenor below 2Y await the quote source; "
        "ACT/360 is retained until then"
    ),
)

EUR_EURIBOR_1M_IRS = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-1M-IRS-V2",
    instrument_type="irs_swap",
    index_name="EUR-EURIBOR-1M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    fixed_leg_frequency="Annual",
    fixed_leg_day_count="30/360",
    floating_leg_frequency="Monthly",
    floating_leg_day_count="ACT/360",
    schedule_generation_rule="Backward",
    payment_lag_days=0,
    discount_curve_key="OIS",
    forward_curve_key="1M",
)

EUR_EURIBOR_3M_DEPOSIT = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-3M-DEPOSIT-V2",
    instrument_type="deposit",
    index_name="EUR-EURIBOR-3M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    floating_leg_frequency="Quarterly",
    floating_leg_day_count="ACT/360",
    forward_curve_key="3M",
)

# FuturesRateHelper delegates the underlying 3M deposit roll to Euribor3M, whose end-of-month
# flag must match this profile (the builder checks it). The convexity volatility, its source,
# date and time basis are versioned here (decision D-19), not read from a global default.
EUR_EURIBOR_3M_SERIAL_FUTURE = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-3M-SERIAL-FUTURE-V2",
    instrument_type="future",
    index_name="EUR-EURIBOR-3M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    floating_leg_frequency="Quarterly",
    floating_leg_day_count="ACT/360",
    forward_curve_key="3M",
    quote_unit="rate",
    contract_resolution="sf_rank_rule_v1",
    convexity_sigma=0.005,
    convexity_sigma_source=(
        "pricer default for the Hull-White style normal short-rate volatility; "
        "not calibrated to market quotes"
    ),
    convexity_sigma_as_of="2026-09-29",
    convexity_time_basis="ACT/360",
)

EUR_EURIBOR_3M_IRS = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-3M-IRS-V2",
    instrument_type="irs_swap",
    index_name="EUR-EURIBOR-3M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    fixed_leg_frequency="Annual",
    fixed_leg_day_count="30/360",
    floating_leg_frequency="Quarterly",
    floating_leg_day_count="ACT/360",
    schedule_generation_rule="Backward",
    payment_lag_days=0,
    discount_curve_key="OIS",
    forward_curve_key="3M",
)

EUR_EURIBOR_6M_DEPOSIT = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-6M-DEPOSIT-V2",
    instrument_type="deposit",
    index_name="EUR-EURIBOR-6M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    floating_leg_frequency="Semiannual",
    floating_leg_day_count="ACT/360",
    forward_curve_key="6M",
)

EUR_EURIBOR_6M_FRA = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-6M-FRA-V2",
    instrument_type="fra",
    index_name="EUR-EURIBOR-6M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    floating_leg_frequency="Semiannual",
    floating_leg_day_count="ACT/360",
    forward_curve_key="6M",
)

EUR_EURIBOR_6M_IRS = CurveQuoteConvention(
    convention_id="EUR-EURIBOR-6M-IRS-V2",
    instrument_type="irs_swap",
    index_name="EUR-EURIBOR-6M",
    settlement_days=2,
    calendar="TARGET",
    business_day_convention="ModifiedFollowing",
    day_count="ACT/360",
    end_of_month=True,
    fixed_leg_frequency="Annual",
    fixed_leg_day_count="30/360",
    floating_leg_frequency="Semiannual",
    floating_leg_day_count="ACT/360",
    schedule_generation_rule="Backward",
    payment_lag_days=0,
    discount_curve_key="OIS",
    forward_curve_key="6M",
)

_CURVE_QUOTE_CONVENTIONS = {
    convention.convention_id: convention
    for convention in (
        EUR_ESTR_DEPOSIT,
        EUR_ESTR_OIS,
        USD_SOFR_OIS_ACT360_ANNUAL,
        GBP_SONIA_OIS_ACT365_ANNUAL,
        GBP_SONIA_DEPOSIT,
        EUR_EURIBOR_1M_FIXING,
        EUR_EURIBOR_1M_DEPOSIT,
        EUR_EURIBOR_1M_SHORT_IRS,
        EUR_EURIBOR_1M_IRS,
        EUR_EURIBOR_3M_DEPOSIT,
        EUR_EURIBOR_3M_SERIAL_FUTURE,
        EUR_EURIBOR_3M_IRS,
        EUR_EURIBOR_6M_DEPOSIT,
        EUR_EURIBOR_6M_FRA,
        EUR_EURIBOR_6M_IRS,
    )
}


def get_curve_quote_convention(convention_id: str) -> CurveQuoteConvention:
    """Resolve one canonical convention identifier or reject an unknown identifier."""
    try:
        return _CURVE_QUOTE_CONVENTIONS[convention_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown curve quote convention: {convention_id}. Snapshots written with a retired "
            "identifier must be migrated with scripts/migrate_market_snapshots.py"
        ) from exc
