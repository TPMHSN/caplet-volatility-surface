"""Price-equivalent Black caplet surfaces, independent of normal calibration."""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class BlackVolSurface(BaseModel):
    model: Literal["black76"] = "black76"
    shift: float = 0.0
    strikes: list[float]
    fixing_dates: list[date]
    expiry_years: list[float]
    # Annualized Black vol in decimals; reasons are French explanations per cell.
    volatilities: list[list[float | None]]
    unavailable_reasons: list[list[str | None]]
    # Absolute price error per unit of discounted accrual notional (N * delta * D).
    max_abs_price_error: float | None
