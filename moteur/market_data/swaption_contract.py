"""Source contract of swaption volatility quotes (CUBESWPT section 4.1).

One set of checks for every entry path (Excel template, CSV/Excel upload, manual entry). A quote
is a normal vol per year; the smile is quoted as strike spreads to the node's ATM forward. The
checks refuse silently-wrong inputs, citing the offending cell, and report what is merely absent:

* strike headers strictly increasing and unique (lot M1: old workbooks had -100 and -150 bp swapped);
* cells that are numbers, and strictly positive vols;
* normal vols only;
* a rectangular grid of OTM nodes, three quotes at least per node, missing cells counted and listed.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
from openpyxl.utils import get_column_letter

from config_swaption import get_swaption_market_convention

from .schemas import SwaptionVolData, SwaptionVolDataPoint

BASIS_POINT = 1.0e-4
MIN_QUOTES_PER_NODE = 3
# A normal vol outside [0.01 bp, 500 bp] per year is a unit or type error (a decimal read as bp, a
# lognormal level or a percentage read as a normal vol), not a market level; a strike spread of
# 10 % or more is a bp value read as a decimal.
MIN_PLAUSIBLE_VOL = 1.0e-6
MAX_PLAUSIBLE_VOL = 0.05
MAX_PLAUSIBLE_SPREAD = 0.10

_HEADER_PATTERN = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:bps|bp)?\s*$", re.IGNORECASE)
_PANDAS_DUPLICATE_SUFFIX = re.compile(r"^(.*?)\.\d+$")

SmileRole = Literal["atm", "smile"]


def swaption_convention_id(currency: str) -> Optional[str]:
    """Market-convention profile of the currency's swaption quotes, None if not tabulated."""
    profile = get_swaption_market_convention(currency)
    return None if profile is None else profile.profile_id


def cell_address(sheet: str, column_position: int, row: int) -> str:
    """'SHEET!F3' for a 1-based column position and row."""
    return f"{sheet}!{get_column_letter(column_position)}{row}"


def _header_spread(text: str) -> Optional[float]:
    match = _HEADER_PATTERN.match(text)
    return None if match is None else float(match.group(1)) * BASIS_POINT


def parse_strike_headers(
    columns: Sequence[tuple[object, int]], sheet: str
) -> list[tuple[object, int, float]]:
    """Strike columns as (column key, 1-based position, spread in decimal), strictly increasing.

    ``columns`` pairs each header cell with its real 1-based sheet position (blank filler columns
    may sit between strikes). A header that pandas de-duplicated ('-100bps.1') counts as the
    duplicate it is, but only when its base was already seen: '12.5' is a header, not '12' + '.5'.
    """
    parsed: list[tuple[object, int, float, str]] = []
    for column, position in columns:
        text = str(column).strip()
        spread = _header_spread(text)
        if spread is None:
            suffix = _PANDAS_DUPLICATE_SUFFIX.match(text)
            base = None if suffix is None else _header_spread(suffix.group(1))
            if base is not None and any(
                math.isclose(base, seen[2], abs_tol=1e-12) for seen in parsed
            ):
                text, spread = suffix.group(1), base  # type: ignore[union-attr]
        address = cell_address(sheet, position, 1)
        if spread is None:
            raise ValueError(
                f"{address} ('{text}') is not a strike header (expected a spread in bp such as "
                "'-100bps')"
            )
        if abs(spread) >= MAX_PLAUSIBLE_SPREAD:
            raise ValueError(
                f"{address} ('{text}'): strike spread of {spread / BASIS_POINT:g} bp is implausible"
                " (headers are spreads to ATM in bp, below 1000 bp)"
            )
        if parsed:
            previous = parsed[-1]
            previous_address = cell_address(sheet, previous[1], 1)
            if math.isclose(spread, previous[2], abs_tol=1.0e-12):
                raise ValueError(
                    f"{address} ('{text}') duplicates {previous_address} ('{previous[3]}'): "
                    "strike headers must be unique"
                )
            if spread < previous[2]:
                raise ValueError(
                    f"{address} ('{text}') is not above {previous_address} ('{previous[3]}'): "
                    "strike headers must be strictly increasing"
                )
        parsed.append((column, position, spread, text))
    return [(column, position, spread) for column, position, spread, _ in parsed]


def parse_vol_cell(value: object, address: str) -> Optional[float]:
    """A vol cell in bp as a float; None for a blank cell, ValueError for anything else wrong.

    A numeric string ('60') is read as the number it spells; booleans, other text, zero, negative
    and implausible magnitudes are refused with the cell address.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{address}: '{value}' is not a number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{address}: '{value}' is not a number") from None
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{address}: volatility {value} is not strictly positive")
    if not MIN_PLAUSIBLE_VOL <= number * BASIS_POINT <= MAX_PLAUSIBLE_VOL:
        raise ValueError(
            f"{address}: volatility {value} bp is implausible for a normal vol "
            f"({MIN_PLAUSIBLE_VOL / BASIS_POINT:g} to {MAX_PLAUSIBLE_VOL / BASIS_POINT:g} bp; "
            "a decimal, a percentage or a lognormal level?)"
        )
    return number


def _node_label(expiry: str, tenor: str) -> str:
    return f"{expiry}x{tenor}"


def check_decimal_vol(volatility: float, where: str) -> None:
    """Refuse a decimal normal vol that is not a positive, plausible market level."""
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValueError(f"{where}: volatility {volatility} must be strictly positive")
    if not MIN_PLAUSIBLE_VOL <= volatility <= MAX_PLAUSIBLE_VOL:
        raise ValueError(
            f"{where}: volatility {volatility} (decimal) is implausible for a normal vol "
            f"({MIN_PLAUSIBLE_VOL} to {MAX_PLAUSIBLE_VOL}; a percentage or a lognormal level?)"
        )


def check_strike_spread(spread: float, where: str) -> None:
    """Refuse a strike spread that cannot be a decimal spread to the ATM forward."""
    if not math.isfinite(spread) or abs(spread) >= MAX_PLAUSIBLE_SPREAD:
        raise ValueError(
            f"{where}: strike spread {spread} is implausible (a decimal below 0.10; "
            "a value in bp read as a decimal?)"
        )


def _check_raw_value(point: SwaptionVolDataPoint, where: str) -> None:
    """The raw quote, when carried, must be the one the internal vol was converted from."""
    if point.raw_value is None or point.raw_unit is None:
        return
    expected = point.raw_value * BASIS_POINT if point.raw_unit == "bp" else point.raw_value
    if not math.isclose(point.volatility, expected, rel_tol=1.0e-6, abs_tol=1.0e-12):
        raise ValueError(
            f"{where}: raw value {point.raw_value} {point.raw_unit} disagrees with the vol "
            f"{point.volatility}"
        )


def validate_swaption_quotes(data: SwaptionVolData, role: SmileRole) -> list[str]:
    """Check a whole ATM or smile dataset; return the notes on what is absent.

    Raises ``ValueError`` on a contract breach (the dataset must not be stored).
    """
    points = data.data_points
    if not points:
        raise ValueError(f"The {role} swaption dataset has no quote")
    lognormal = sorted({p.vol_type for p in points if p.vol_type != "normal"})
    if lognormal:
        raise ValueError(
            f"Swaption cube vols must be normal; found {', '.join(lognormal)} (re-quote in bp)"
        )
    expected_convention = swaption_convention_id(data.currency)
    seen_atm: set[tuple[str, str]] = set()
    for point in points:
        where = f"Swaption vol at {_node_label(point.expiry, point.tenor)}"
        if not math.isfinite(point.volatility) or point.volatility <= 0.0:
            raise ValueError(f"{where} must be strictly positive, got {point.volatility}")
        check_decimal_vol(point.volatility, where)
        if point.strike is not None:
            check_strike_spread(float(point.strike), where)
        _check_raw_value(point, where)
        if point.convention_id is not None and point.convention_id != expected_convention:
            raise ValueError(
                f"{where}: convention '{point.convention_id}' contradicts the {data.currency} "
                f"swaption profile '{expected_convention}'"
            )
        if role == "atm":
            if point.strike is not None:
                raise ValueError(f"An ATM dataset carries no strike spread; found one at {where}")
            key = (point.expiry, point.tenor)
            if key in seen_atm:
                raise ValueError(f"Duplicate ATM quote at {_node_label(*key)}")
            seen_atm.add(key)
        elif point.strike is None:
            raise ValueError(
                f"A smile dataset needs a strike spread on every quote; {where} has none"
            )

    notes = [] if role == "atm" else _validate_smile_grid(data)
    if expected_convention is None:
        notes.append(
            f"No swaption convention profile for {data.currency}: its quotes carry no convention"
        )
    return notes


def apply_swaption_contract(
    data: SwaptionVolData, role: SmileRole
) -> tuple[SwaptionVolData, list[str]]:
    """Validate a dataset and type it: a quote without a profile gets the currency's one."""
    notes = validate_swaption_quotes(data, role)
    profile = swaption_convention_id(data.currency)
    if profile is None or all(p.convention_id == profile for p in data.data_points):
        return data, notes
    typed = [
        (
            point
            if point.convention_id is not None
            else point.model_copy(update={"convention_id": profile})
        )
        for point in data.data_points
    ]
    return data.model_copy(update={"data_points": typed}), notes


def _validate_smile_grid(data: SwaptionVolData) -> list[str]:
    quotes: dict[tuple[str, str], dict[float, int]] = defaultdict(dict)
    for point in data.data_points:
        if point.strike is None:
            raise ValueError("A smile quote needs a strike spread")
        spread = round(float(point.strike), 10)
        node = quotes[(point.expiry, point.tenor)]
        node[spread] = node.get(spread, 0) + 1
        if node[spread] > 1:
            raise ValueError(
                f"Duplicate quote at {_node_label(point.expiry, point.tenor)}, "
                f"{spread / BASIS_POINT:g} bp"
            )

    expiries = sorted({expiry for expiry, _ in quotes})
    tenors = sorted({tenor for _, tenor in quotes})
    absent = [
        _node_label(expiry, tenor)
        for expiry in expiries
        for tenor in tenors
        if (expiry, tenor) not in quotes
    ]
    if absent:
        raise ValueError(
            f"The smile node grid must be rectangular; {len(absent)} node(s) absent: "
            + ", ".join(absent[:12])
            + (" ..." if len(absent) > 12 else "")
        )
    thin = [
        f"{_node_label(*node)} has {len(spreads)} quote(s)"
        for node, spreads in quotes.items()
        if len(spreads) < MIN_QUOTES_PER_NODE
    ]
    if thin:
        raise ValueError(
            f"A smile node needs at least three quotes (OTM node {'; '.join(thin[:8])})"
        )

    all_spreads = sorted({spread for node in quotes.values() for spread in node})
    missing: list[str] = []
    count = 0
    for (expiry, tenor), node in sorted(quotes.items()):
        absent_spreads = [s for s in all_spreads if s not in node]
        if absent_spreads:
            count += len(absent_spreads)
            wings = ", ".join(f"{s / BASIS_POINT:g}" for s in absent_spreads)
            missing.append(f"{_node_label(expiry, tenor)}({wings}bp)")
    if not count:
        return []
    return [f"{count} missing OTM quote(s) out of the rectangular grid: " + ", ".join(missing)]
