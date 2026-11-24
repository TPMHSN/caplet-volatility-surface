"""Scoped IBOR calibration fixings for market-quote risk scenarios."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import QuantLib as ql

from config import get_index, get_index_with_curve
from curves.helpers import date_to_ql, ql_state_lock

if TYPE_CHECKING:
    from curves.yield_curve_builder import CurveSet


@contextmanager
def scoped_ibor_calibration_fixing(
    currency: str,
    tenor: str,
    fixing_date,
    rate: float | None,
) -> Iterator[None]:
    """Install one captured observed reset for a forward bootstrap, then restore history."""
    if currency != "EUR" or rate is None:
        yield
        return
    with ql_state_lock():
        settings = ql.Settings.instance()
        saved_eval = settings.evaluationDate
        index = get_index(currency, tenor)
        history = index.timeSeries()
        dates, values = list(history.dates()), list(history.values())
        try:
            ql_date = date_to_ql(fixing_date)
            settings.evaluationDate = ql_date
            index.addFixing(ql_date, float(rate), True)
            yield
        finally:
            index.clearFixings()
            if dates:
                index.addFixings(dates, values, True)
            settings.evaluationDate = saved_eval


@contextmanager
def constant_ibor_fixings(
    curve_set: CurveSet,
    curve_keys: Iterable[str],
    *,
    enabled: bool = False,
) -> Iterator[None]:
    """Freeze EUR calibration swaps' first fixing, then restore QuantLib state.

    The anchor always comes from the original market curves, including when
    calculating local risk inside an already shifted MtM scenario. This scope
    ends before pricing the trade: it must not create a historical trade fixing.
    """
    with ql_state_lock():
        saved_eval = ql.Settings.instance().evaluationDate
        snapshots: list[tuple[ql.IborIndex, list[ql.Date], list[float]]] = []
        try:
            if enabled and curve_set.currency == "EUR":
                anchor: CurveSet = getattr(curve_set, "calibration_anchor", curve_set)
                fixing_date = date_to_ql(anchor.curve_date)
                ql.Settings.instance().evaluationDate = fixing_date
                for key in sorted(set(curve_keys) - {"OIS"}):
                    if key not in anchor.forward_market_data:
                        continue
                    index = get_index_with_curve("EUR", key, anchor.get_forward_curve(key))
                    history = index.timeSeries()
                    dates, values = list(history.dates()), list(history.values())
                    snapshots.append((index, dates, values))
                    # Production curve sets pin the value selected at construction.
                    # Legacy sets without a captured value retain the forecast fallback.
                    chosen = getattr(anchor, "calibration_fixings", {})
                    rate = chosen.get(key)
                    if rate is None:
                        rate = index.fixing(fixing_date, True)
                    index.addFixing(fixing_date, rate, True)
            yield
        finally:
            for index, dates, values in reversed(snapshots):
                index.clearFixings()
                if dates:
                    index.addFixings(dates, values, True)
            ql.Settings.instance().evaluationDate = saved_eval
