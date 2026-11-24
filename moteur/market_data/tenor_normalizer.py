"""
Tenor normalization and instrument classification for Excel template parsing.
"""

import re

# Known typos in the Excel template
TYPO_CORRECTIONS = {
    "MC": "MO",  # "18 MC" -> "18 MO", "9 MC" -> "9 MO"
}

# Unit mapping: template format -> internal format
UNIT_MAP = {
    "DY": "D",
    "WK": "W",
    "MO": "M",
    "YR": "Y",
}

# Bloomberg FRA ticker -> FRA tenor mapping (EUR 6M)
BLOOMBERG_FRA_MAP = {
    "EUFR0AG": "1x7",
    "EUFR0BH": "2x8",
    "EUFR0CI": "3x9",
    "EUFR0DJ": "4x10",
    "EUFR0EK": "5x11",
    "EUFR0F1": "6x12",
    "EUFR0I1C": "9x15",
    "EUFR011F": "12x18",
}

# Futures month codes -> month number
FUTURES_MONTH_CODES = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}

# Regex for Euribor 3M futures codes (e.g., ERF26, ERZ27)
FUTURES_REGEX = re.compile(r"^ER([FGHJKMNQUVXZ])(\d{2})$")

# Regex for FRA format "X MO x Y MO" (EUR6M template)
FRA_MO_REGEX = re.compile(r"^(\d+)\s*MO\s*x\s*(\d+)\s*MO$", re.IGNORECASE)

# Regex for Serial Futures format "N SF" (EUR3M template)
SF_REGEX = re.compile(r"^(\d+)\s*SF$", re.IGNORECASE)

# Index tenor string -> number of months
INDEX_TENOR_MONTHS = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}


def normalize_tenor(raw: str) -> str:
    """
    Normalize a tenor string from the Excel template to internal format.

    Examples:
        "1 WK" -> "1W"
        "18 MO" -> "18M"
        "2 YR" -> "2Y"
        "1 DY" -> "1D"
        "18 MC" -> "18M" (typo correction)
        "9 MC" -> "9M" (typo correction)
        "1Yr" -> "1Y" (swaption format)
        "1Mo" -> "1M" (swaption format)
    """
    raw = raw.strip()

    # Handle swaption/vol format like "1Yr", "18Mo", "4Y"
    # Try to match number + unit pattern without space
    match = re.match(r"^(\d+)\s*([A-Za-z]+)$", raw)
    if not match:
        return raw  # Return as-is if doesn't match

    number = match.group(1)
    unit = match.group(2).upper()

    # Apply typo corrections
    for typo, correction in TYPO_CORRECTIONS.items():
        if unit == typo:
            unit = correction
            break

    # Map to internal unit
    if unit in UNIT_MAP:
        return f"{number}{UNIT_MAP[unit]}"

    # Handle already-internal formats (D, W, M, Y)
    if unit in ("D", "W", "M", "Y"):
        return f"{number}{unit}"

    # Handle "Yr" -> "Y", "Mo" -> "M" variants
    unit_lower = unit.lower()
    if unit_lower in ("yr",):
        return f"{number}Y"
    if unit_lower in ("mo",):
        return f"{number}M"

    return f"{number}{unit}"


def classify_forward_instrument(term: str, index_tenor: str) -> dict:
    """
    Classify a forward curve instrument from the Excel template.

    Args:
        term: Raw tenor string from the Excel template.
        index_tenor: Index tenor of the curve (e.g., "1M", "3M", "6M").
            Used to determine deposit/swap boundary for monthly tenors
            and to compute FRA periods for serial futures.

    Returns a dict with:
        - type: "future", "fra", "deposit", or "swap"
        - tenor: normalized tenor string (for deposit/swap)
        - fra_tenor: FRA tenor string like "1x7" (for fra)
        - code: original futures code (for future)

    Detects 6 types:
        1. Euribor 3M futures: regex ER[FGHJKMNQUVXZ]\\d{2}
        2. Bloomberg FRA tickers: EUFR0AG, etc.
        3. FRA "X MO x Y MO" format (EUR6M template)
        4. Serial Futures "N SF" format (EUR3M template)
        5. Deposits: tenors <= index_tenor (or D/W units)
        6. Swaps: tenors > index_tenor (or Y unit)
    """
    term = term.strip()

    # Check for Euribor 3M futures code
    futures_match = FUTURES_REGEX.match(term)
    if futures_match:
        return {"type": "future", "code": term}

    # Check for Bloomberg FRA ticker
    if term.upper() in BLOOMBERG_FRA_MAP:
        fra_tenor = BLOOMBERG_FRA_MAP[term.upper()]
        return {"type": "fra", "fra_tenor": fra_tenor}

    # Check for FRA format "X MO x Y MO"
    fra_mo_match = FRA_MO_REGEX.match(term)
    if fra_mo_match:
        start = fra_mo_match.group(1)
        end = fra_mo_match.group(2)
        return {"type": "fra", "fra_tenor": f"{start}x{end}"}

    # Check for Serial Futures "N SF": rank n of the strip, resolved to an IMM date from the
    # valuation date in template_loader.
    sf_match = SF_REGEX.match(term)
    if sf_match:
        n = int(sf_match.group(1))
        return {"type": "sf", "n": n}

    # Normalize the tenor
    normalized = normalize_tenor(term)

    match = re.match(r"^(\d+)([DWMY])$", normalized)
    if not match:
        return {"type": "swap", "tenor": normalized}

    number = int(match.group(1))
    unit = match.group(2)

    if unit in ("D", "W"):
        return {"type": "deposit", "tenor": normalized}

    if unit == "M":
        # index_tenor decides the deposit/swap boundary
        index_months = INDEX_TENOR_MONTHS[index_tenor]
        if number <= index_months:
            return {"type": "deposit", "tenor": normalized}
        return {"type": "swap", "tenor": normalized}

    # Y unit -> swap
    return {"type": "swap", "tenor": normalized}
