"""
Effective helper conventions and curve provenance.

``EffectiveHelperSpec`` records the convention values a rate helper was actually
built with, read from the very QuantLib objects handed to its constructor. The
curve fingerprint serialises those same specifications, so provenance can never
drift from a supposed copy of a convention profile.

Any new helper branch must either enrich this specification or bump
``CURVE_BUILDER_METHOD_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date as dt_date
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import QuantLib as ql

from market_data.canonical_hash import LINEAGE_SCHEMA_VERSION, canonical_fingerprint
from market_data.schemas import ForwardMarketData, OISMarketData
from market_data.source_fingerprints import forward_source_fingerprint, ois_source_fingerprint

CURVE_BUILDER_METHOD_VERSION = "curve-builder-v4"
CURVE_INPUTS_SCHEMA = "curve-inputs-v3"
CURVE_FINGERPRINT_SCHEMA = "curve-fingerprint-v3"

_BUSINESS_DAY_CONVENTION_LABELS: dict[int, str] = {
    ql.Following: "Following",
    ql.ModifiedFollowing: "ModifiedFollowing",
    ql.Preceding: "Preceding",
    ql.ModifiedPreceding: "ModifiedPreceding",
    ql.Unadjusted: "Unadjusted",
    ql.HalfMonthModifiedFollowing: "HalfMonthModifiedFollowing",
    ql.Nearest: "Nearest",
}

_FREQUENCY_LABELS: dict[int, str] = {
    ql.NoFrequency: "NoFrequency",
    ql.Once: "Once",
    ql.Annual: "Annual",
    ql.Semiannual: "Semiannual",
    ql.EveryFourthMonth: "EveryFourthMonth",
    ql.Quarterly: "Quarterly",
    ql.Bimonthly: "Bimonthly",
    ql.Monthly: "Monthly",
    ql.EveryFourthWeek: "EveryFourthWeek",
    ql.Biweekly: "Biweekly",
    ql.Weekly: "Weekly",
    ql.Daily: "Daily",
}

_SCHEDULE_RULE_LABELS: dict[int, str] = {
    ql.DateGeneration.Backward: "Backward",
    ql.DateGeneration.Forward: "Forward",
    ql.DateGeneration.Zero: "Zero",
    ql.DateGeneration.ThirdWednesday: "ThirdWednesday",
    ql.DateGeneration.Twentieth: "Twentieth",
    ql.DateGeneration.TwentiethIMM: "TwentiethIMM",
}


def calendar_label(calendar: Optional[ql.Calendar]) -> Optional[str]:
    """Return the canonical name of the calendar handed to a helper."""
    return None if calendar is None else str(calendar.name())


def day_count_label(day_counter: Optional[ql.DayCounter]) -> Optional[str]:
    """Return the canonical name of the day counter handed to a helper."""
    return None if day_counter is None else str(day_counter.name())


def _enum_label(labels: dict[int, str], value: Optional[int]) -> Optional[str]:
    """Name a QuantLib enum value, falling back to a discriminating placeholder.

    An unmapped value must never break a curve build: the placeholder keeps the
    fingerprint discriminating, and the missing name is a provenance-readability
    issue, not a pricing one.
    """
    if value is None:
        return None
    return labels.get(int(value), f"unmapped:{int(value)}")


def business_day_convention_label(convention: Optional[int]) -> Optional[str]:
    """Return the canonical name of the business-day convention handed to a helper."""
    return _enum_label(_BUSINESS_DAY_CONVENTION_LABELS, convention)


def frequency_label(frequency: Optional[int]) -> Optional[str]:
    """Return the canonical name of the frequency handed to a helper."""
    return _enum_label(_FREQUENCY_LABELS, frequency)


def schedule_rule_label(rule: Optional[int]) -> Optional[str]:
    """Return the canonical name of the schedule generation rule handed to a helper."""
    return _enum_label(_SCHEDULE_RULE_LABELS, rule)


@dataclass(frozen=True)
class EffectiveHelperSpec:
    """Convention values a single calibration helper was actually constructed with."""

    instrument_type: str
    curve_key: str
    quote_kind: str
    pillar_label: str
    convention_id: str
    settlement_days: Optional[int] = None
    calendar: Optional[str] = None
    business_day_convention: Optional[str] = None
    day_count: Optional[str] = None
    end_of_month: Optional[bool] = None
    fixed_leg_frequency: Optional[str] = None
    fixed_leg_day_count: Optional[str] = None
    floating_leg_frequency: Optional[str] = None
    floating_leg_day_count: Optional[str] = None
    schedule_generation_rule: Optional[str] = None
    payment_lag_days: Optional[int] = None
    rate_cutoff_days: Optional[int] = None
    discount_curve_key: Optional[str] = None
    forward_curve_key: Optional[str] = None
    index_name: Optional[str] = None
    convexity_adjustment_vol: Optional[float] = None
    payment_calendar: Optional[str] = None
    fixed_calendar: Optional[str] = None
    overnight_calendar: Optional[str] = None
    quote_provenance: Optional[str] = None
    calibration_role: Optional[str] = None
    convexity_sigma_source: Optional[str] = None
    convexity_sigma_as_of: Optional[str] = None
    convexity_time_basis: Optional[str] = None

    def to_canonical_payload(self) -> dict[str, Any]:
        """Return the serialisable form consumed by the curve fingerprint."""
        return {
            "instrument_type": self.instrument_type,
            "curve_key": self.curve_key,
            "quote_kind": self.quote_kind,
            "pillar_label": self.pillar_label,
            "convention_id": self.convention_id,
            "settlement_days": self.settlement_days,
            "calendar": self.calendar,
            "business_day_convention": self.business_day_convention,
            "day_count": self.day_count,
            "end_of_month": self.end_of_month,
            "fixed_leg_frequency": self.fixed_leg_frequency,
            "fixed_leg_day_count": self.fixed_leg_day_count,
            "floating_leg_frequency": self.floating_leg_frequency,
            "floating_leg_day_count": self.floating_leg_day_count,
            "schedule_generation_rule": self.schedule_generation_rule,
            "payment_lag_days": self.payment_lag_days,
            "rate_cutoff_days": self.rate_cutoff_days,
            "discount_curve_key": self.discount_curve_key,
            "forward_curve_key": self.forward_curve_key,
            "index_name": self.index_name,
            "convexity_adjustment_vol": self.convexity_adjustment_vol,
            "payment_calendar": self.payment_calendar,
            "fixed_calendar": self.fixed_calendar,
            "overnight_calendar": self.overnight_calendar,
            "quote_provenance": self.quote_provenance,
            "calibration_role": self.calibration_role,
            "convexity_sigma_source": self.convexity_sigma_source,
            "convexity_sigma_as_of": self.convexity_sigma_as_of,
            "convexity_time_basis": self.convexity_time_basis,
        }


@dataclass(frozen=True)
class CurveLineage:
    """Deterministic provenance of a published curve set."""

    schema_version: str
    builder_method_version: str
    currency: str
    curve_date: dt_date
    requested_interpolation: str
    effective_interpolation: str
    curve_inputs_fingerprint: str
    curve_fingerprint: str
    ois_source_fingerprint: str
    forward_source_fingerprints: Mapping[str, str] = field(default_factory=dict)
    calibration_fixings: Mapping[str, float] = field(default_factory=dict)
    observed_calibration_fixings: Mapping[str, float] = field(default_factory=dict)
    shock_context: Optional[Mapping[str, Any]] = None

    def with_shock_context(self, shock_context: Mapping[str, Any]) -> "CurveLineage":
        """Return the same provenance carrying an ephemeral shock context.

        Contexts accumulate rather than replace one another, so a composed shock
        (a stress scenario on top of a bump) keeps the whole chain visible.
        """
        merged = {**(self.shock_context or {}), **shock_context}
        return replace(self, shock_context=merged)


def _forward_fingerprints(
    forward_data_by_tenor: Mapping[str, ForwardMarketData],
) -> dict[str, str]:
    return {
        tenor: forward_source_fingerprint(data) for tenor, data in forward_data_by_tenor.items()
    }


def curve_inputs_fingerprint(
    *,
    currency: str,
    curve_date: dt_date,
    requested_interpolation: str,
    ois_data: OISMarketData,
    forward_data_by_tenor: Mapping[str, ForwardMarketData],
    observed_calibration_fixings: Mapping[str, float] | None = None,
) -> str:
    """Fingerprint the economic inputs a curve set depends on.

    Recomputable from the market data store at any time, which is what makes the
    ``curve_inputs_stale`` diagnostic possible without replaying a bootstrap.
    """
    return canonical_fingerprint(
        CURVE_INPUTS_SCHEMA,
        {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "currency": currency,
            "curve_date": curve_date,
            "requested_interpolation": requested_interpolation,
            "ois_source_fingerprint": ois_source_fingerprint(ois_data),
            "forward_source_fingerprints": _forward_fingerprints(forward_data_by_tenor),
            "observed_calibration_fixings": dict(
                sorted((observed_calibration_fixings or {}).items())
            ),
        },
    )


def build_curve_lineage(
    *,
    currency: str,
    curve_date: dt_date,
    requested_interpolation: str,
    effective_interpolation: str,
    ois_data: OISMarketData,
    forward_data_by_tenor: Mapping[str, ForwardMarketData],
    effective_helper_specs: Sequence[EffectiveHelperSpec],
    bootstrap_accuracies: Mapping[str, float],
    curve_day_counts: Optional[Mapping[str, str]] = None,
    calibration_fixings: Mapping[str, float] | None = None,
    observed_calibration_fixings: Mapping[str, float] | None = None,
) -> CurveLineage:
    """Compute the provenance of a curve set from the objects actually consumed."""
    inputs_fingerprint = curve_inputs_fingerprint(
        currency=currency,
        curve_date=curve_date,
        requested_interpolation=requested_interpolation,
        ois_data=ois_data,
        forward_data_by_tenor=forward_data_by_tenor,
        observed_calibration_fixings=observed_calibration_fixings,
    )
    fingerprint = canonical_fingerprint(
        CURVE_FINGERPRINT_SCHEMA,
        {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "builder_method_version": CURVE_BUILDER_METHOD_VERSION,
            "currency": currency,
            "curve_date": curve_date,
            "curve_inputs_fingerprint": inputs_fingerprint,
            "calibration_fixings": dict(sorted((calibration_fixings or {}).items())),
            "requested_interpolation": requested_interpolation,
            "effective_interpolation": effective_interpolation,
            "bootstrap_accuracies": {
                key: float(value) for key, value in bootstrap_accuracies.items()
            },
            # The curve's own day counter is the time metric of the interpolation and
            # of the frozen discount curve: a change moves every discount factor.
            "curve_day_counts": dict(curve_day_counts or {}),
            "effective_helper_specs": [
                spec.to_canonical_payload() for spec in effective_helper_specs
            ],
        },
    )
    return CurveLineage(
        schema_version=LINEAGE_SCHEMA_VERSION,
        builder_method_version=CURVE_BUILDER_METHOD_VERSION,
        currency=currency,
        curve_date=curve_date,
        requested_interpolation=requested_interpolation,
        effective_interpolation=effective_interpolation,
        curve_inputs_fingerprint=inputs_fingerprint,
        curve_fingerprint=fingerprint,
        ois_source_fingerprint=ois_source_fingerprint(ois_data),
        forward_source_fingerprints=_forward_fingerprints(forward_data_by_tenor),
        calibration_fixings=MappingProxyType(dict(calibration_fixings or {})),
        observed_calibration_fixings=MappingProxyType(
            dict(observed_calibration_fixings or {})
        ),
    )
