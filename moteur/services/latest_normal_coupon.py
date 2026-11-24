"""Euribor coupon projection contract shared by base and frozen repricing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import QuantLib as ql

from config import get_index

IndexTenor = Literal["3M", "6M"]


@dataclass(frozen=True)
class CouponForwardContract:
    """Index dates and accrual implied by one actual fixing date."""

    index_value_date: ql.Date
    index_maturity_date: ql.Date
    projection_accrual_fraction: float

    def forward_rate(self, forward_curve: ql.YieldTermStructureHandle) -> float:
        """Project the simple index fixing without curve extrapolation."""
        if (
            self.index_value_date < forward_curve.referenceDate()
            or self.index_maturity_date > forward_curve.maxDate()
        ):
            raise ValueError("Sous-jacent Euribor hors du domaine de la courbe.")
        start_discount = forward_curve.discount(self.index_value_date)
        end_discount = forward_curve.discount(self.index_maturity_date)
        forward = (start_discount / end_discount - 1.0) / self.projection_accrual_fraction
        if (
            not all(isfinite(value) for value in (start_discount, end_discount, forward))
            or min(
                start_discount,
                end_discount,
                self.projection_accrual_fraction,
            )
            <= 0.0
        ):
            raise ValueError("La courbe Euribor donne un forward invalide.")
        return forward


def coupon_forward_contract(index_tenor: IndexTenor, fixing_date: ql.Date) -> CouponForwardContract:
    """Resolve the standard index period from the coupon's actual fixing date."""
    index = get_index("EUR", index_tenor)
    index_value_date = index.valueDate(fixing_date)
    index_maturity_date = index.maturityDate(index_value_date)
    projection_accrual_fraction = index.dayCounter().yearFraction(
        index_value_date, index_maturity_date
    )
    if (
        index_maturity_date <= index_value_date
        or not isfinite(projection_accrual_fraction)
        or projection_accrual_fraction <= 0.0
    ):
        raise ValueError("Période sous-jacente Euribor invalide.")
    return CouponForwardContract(
        index_value_date=index_value_date,
        index_maturity_date=index_maturity_date,
        projection_accrual_fraction=projection_accrual_fraction,
    )
