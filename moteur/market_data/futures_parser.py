"""
Futures code parsing and IMM date calculation.

Converts Euribor 3M futures codes (ERF26, ERG26...) and serial ranks (1 SF..12 SF) to the IMM
dates of the futures strip bootstrapped with FuturesRateHelper.

Serial futures convention (1 SF – 12 SF):
  - n=1..6: monthly serial futures (3rd Wednesday of consecutive months)
  - n=7..12: quarterly IMM futures (Mar/Jun/Sep/Dec cycle)
"""

import calendar
from datetime import date

import QuantLib as ql

from .tenor_normalizer import FUTURES_MONTH_CODES, FUTURES_REGEX


def _business_day_settle(valuation_date: date, days: int = 2) -> date:
    """
    Compute the Euribor settlement date by advancing TARGET business days.
    """
    result = ql.TARGET().advance(
        ql.Date(valuation_date.day, valuation_date.month, valuation_date.year),
        days,
        ql.Days,
    )
    return date(result.year(), result.month(), result.dayOfMonth())


def third_wednesday(year: int, month: int) -> date:
    """
    Calculate the 3rd Wednesday of a given month (IMM date).

    Args:
        year: Full year (e.g., 2026)
        month: Month number (1-12)

    Returns:
        Date of the 3rd Wednesday
    """
    # Get the calendar for the month
    cal = calendar.monthcalendar(year, month)

    # Find all Wednesdays (index 2 in the week)
    wednesdays = []
    for week in cal:
        if week[2] != 0:  # Wednesday is index 2
            wednesdays.append(week[2])

    return date(year, month, wednesdays[2])  # 3rd Wednesday (0-indexed: [2])


def futures_code_to_imm_date(code: str) -> date:
    """
    Return the IMM start date for a Euribor futures code (e.g., 'ERF26').

    Args:
        code: Futures code like "ERF26" (ER + month_letter + 2-digit year)

    Returns:
        The 3rd Wednesday of the futures contract's month.
    """
    match = FUTURES_REGEX.match(code.strip())
    if not match:
        raise ValueError(f"Invalid futures code: {code}. Expected format like ERF26.")
    month_code = match.group(1)
    year_suffix = int(match.group(2))
    full_year = 2000 + year_suffix
    month = FUTURES_MONTH_CODES[month_code]
    return third_wednesday(full_year, month)


def sf_to_imm_date(n: int, valuation_date: date, index_tenor_months: int = 3) -> date:
    """
    Return the IMM start date for the Nth serial future.

    Convention:
      n=1..6  – monthly serial: 3rd Wednesday of consecutive months starting
                from the valuation month, skipping dates <= settlement.
      n=7..12 – quarterly: first quarterly IMM (Mar/Jun/Sep/Dec) strictly after
                the last serial, then +3 months per step.

    Args:
        n: Serial future index (1-based, 1..12).
        valuation_date: Pricing/valuation date.
        index_tenor_months: Underlying index tenor in months (default 3).

    Returns:
        The IMM date (3rd Wednesday) for the Nth serial future.
    """
    if n < 1 or n > 12:
        raise ValueError(f"Serial future index must be 1-12, got {n}")

    settlement = _business_day_settle(valuation_date)

    if n <= 6:
        # Monthly serial: 3rd Wednesday of consecutive months, skipping <= settlement
        candidates = []
        year, month = valuation_date.year, valuation_date.month
        while len(candidates) < 6:
            imm = third_wednesday(year, month)
            if imm > settlement:
                candidates.append(imm)
            month += 1
            if month > 12:
                month = 1
                year += 1
        return candidates[n - 1]
    else:
        # Quarterly: first quarterly IMM strictly after the last serial
        last_serial = sf_to_imm_date(6, valuation_date, index_tenor_months)
        quarterly_months = {3, 6, 9, 12}

        year, month = last_serial.year, last_serial.month
        first_quarterly = None
        for delta in range(15):
            check_month = ((month - 1 + delta) % 12) + 1
            check_year = year + (month - 1 + delta) // 12
            if check_month in quarterly_months:
                imm = third_wednesday(check_year, check_month)
                if imm > last_serial:
                    first_quarterly = imm
                    break

        if first_quarterly is None:
            raise ValueError(f"Could not find quarterly IMM date after last serial {last_serial}")

        # Advance (n - 7) quarters from first_quarterly
        steps = n - 7
        total = (first_quarterly.year * 12 + first_quarterly.month - 1) + steps * 3
        target_year = total // 12
        target_month = (total % 12) + 1
        return third_wednesday(target_year, target_month)
