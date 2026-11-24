"""Shared typing helpers for backend modules."""

from __future__ import annotations

from typing import Protocol, TypeAlias

import QuantLib as ql

JsonScalar: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, object]

ShortRateModel: TypeAlias = ql.HullWhite | ql.G2 | ql.Gsr


class CreditCurveLike(Protocol):
    """Credit-curve operations consumed by stochastic-credit calibration."""

    def hazard_rate_at(self, time: float) -> float:
        """Return the hazard rate at the given year fraction."""

    def survival(self, time: float) -> float:
        """Return survival from the valuation date to the given year fraction."""


class UnderlyingSwapLike(Protocol):
    """Common swap interface used by standard and nonstandard swaptions."""

    def leg(self, leg_idx: int) -> ql.Leg:
        """Return one leg of the underlying swap."""

    def fairRate(self) -> float:
        """Return the par rate when exposed by the QuantLib binding."""

    def legBPS(self, leg_idx: int) -> float:
        """Return the PV01/BPS of the requested leg."""

    def legNPV(self, leg_idx: int) -> float:
        """Return the NPV of the requested leg."""

    def NPV(self) -> float:
        """Return the total swap NPV."""


class PricedSwaptionLike(Protocol):
    """Minimal protocol for swaption-like instruments that can be priced."""

    def setPricingEngine(self, engine: object) -> None:
        """Attach a QuantLib pricing engine."""

    def NPV(self) -> float:
        """Return the instrument NPV."""
