"""Parameters of the curve itself, as opposed to the conventions of its instruments.

The instruments of a curve take every convention from their canonical profile
(``market_data/curve_quote_conventions.py``). What remains here is the time axis of the bootstrapped
curve: the day counter that measures its pillars.
"""

import QuantLib as ql

from config import get_currency_config

from .helpers import index_tenor_months


def ois_curve_day_counter(currency: str) -> ql.DayCounter:
    """Day counter of the pillars of an overnight (OIS) curve."""
    return get_currency_config(currency)["ois_dc"]


def forward_curve_day_counter(currency: str, index_tenor: str) -> ql.DayCounter:
    """Day counter of the pillars of a projection curve of a EURIBOR index tenor."""
    index_tenor_months(index_tenor)
    return get_currency_config(currency)["float_leg_dc"]
