"""Exact price conversion from Bachelier normal vol to Black-76 implied vol.

This module does not calibrate SABR, alter normal prices, or invent a rate shift.
It inverts OTM time value, so caplets and floorlets share the converted vol by parity.
"""

from math import exp, expm1, isfinite, log, pi, sqrt

import QuantLib as ql
from scipy.optimize import brentq
from scipy.special import log_ndtr

from services.thomas_black_schemas import BlackVolSurface




def normal_otm_time_value(forward, strike, stddev):
    """Undiscounted normal OTM price with stable Gaussian tail subtraction."""
    if stddev < 0:
        raise ValueError("Écart type normal négatif.")
    if stddev == 0:
        return 0.0
    z = abs(forward - strike) / stddev
    log_density = -0.5 * z * z - 0.5 * log(2.0 * pi)
    if z == 0:
        return stddev * exp(log_density)
    log_ratio = log(z) + float(log_ndtr(-z)) - log_density
    return stddev * exp(log_density) * -expm1(min(log_ratio, 0.0))


def black_otm_time_value(forward, strike, stddev):
    """Undiscounted OTM Black price, using log-CDFs to preserve tail precision.

    Forward and strike are already displaced by the caller. Evaluating an OTM
    put/call avoids cancellation against intrinsic; log subtraction also avoids
    subtracting nearly equal CDF terms in short-expiry wings.
    """
    if forward <= 0 or strike <= 0 or stddev < 0:
        raise ValueError("Black exige un forward et un strike décalés positifs.")
    if stddev == 0:
        return 0.0
    d1 = log(forward / strike) / stddev + 0.5 * stddev
    d2 = d1 - stddev
    if forward <= strike:
        leading = log(forward) + float(log_ndtr(d1))
        trailing = log(strike) + float(log_ndtr(d2))
    else:
        leading = log(strike) + float(log_ndtr(-d2))
        trailing = log(forward) + float(log_ndtr(-d1))
    return exp(leading) * -expm1(min(trailing - leading, 0.0))


def normal_to_black_vol(forward, strike, expiry, normal_vol, shift=0.0):
    """Return (annual Black vol, unavailable reason, price error per unit annuity).

    Strict Black positivity and its finite-vol price bound are checked first.
    A positive normal vol with underflowed time value is not silently converted to 0.
    """
    if normal_vol is None:
        return None, "Volatilité normale indisponible à ce point.", None
    if not all(isfinite(x) for x in (forward, strike, expiry, normal_vol, shift)):
        return None, "Donnée non finie : conversion Black impossible.", None
    if expiry <= 0:
        return None, "Fixing non futur : volatilité Black non identifiable.", None
    if normal_vol < 0:
        return None, "Volatilité normale négative : conversion impossible.", None
    shifted_forward, shifted_strike = forward + shift, strike + shift
    if not all(isfinite(x) and x > 0 for x in (shifted_forward, shifted_strike)):
        return None, "Black exige F + décalage > 0 et K + décalage > 0.", None
    if normal_vol == 0:
        return 0.0, None, 0.0
    option = ql.Option.Put if forward > strike else ql.Option.Call
    root_time = sqrt(expiry)
    try:
        target = normal_otm_time_value(forward, strike, normal_vol * root_time)
        if not isfinite(target) or target <= 0:
            return None, "Valeur temps trop petite : volatilité Black non identifiable numériquement.", None
        if target >= min(shifted_forward, shifted_strike):
            return None, "Prix normal hors de la borne Black : aucune volatilité Black finie.", None

        def price(stddev):
            return black_otm_time_value(shifted_forward, shifted_strike, stddev)

        def validated(stddev):
            if not isfinite(stddev) or stddev <= 0:
                return None
            repriced = price(stddev)
            if not isfinite(repriced) or abs(repriced / target - 1) > 1e-8:
                return None
            vol = stddev / root_time
            if not isfinite(vol) or vol <= 0:
                return None
            return vol, None, abs(repriced - target)

        try:
            guess = ql.blackFormulaImpliedStdDev(
                option, shifted_strike, shifted_forward, target,
                1.0, 0.0, ql.nullDouble(), 1e-12, 100,
            )
            converted = validated(guess)
            if converted is not None:
                return converted
        except RuntimeError:
            pass
        # Verify the price, not only the solver's stopping tolerance, especially in tails.
        high = 0.5
        while high < 64 and price(high) < target:
            high *= 2
        if price(high) < target:
            return None, "Impossible d’encadrer une volatilité Black finie.", None
        stddev = brentq(lambda value: price(value) / target - 1.0, 0.0, high, xtol=1e-14, maxiter=250)
        converted = validated(stddev)
        if converted is not None:
            return converted
        return None, "Inversion Black insuffisamment précise ; aucun remplacement automatique.", None
    except (RuntimeError, ValueError, OverflowError):
        return None, "Inversion Black impossible à la précision numérique disponible.", None


def build_black_surface(strikes, fixing_dates, expiry_years, forwards, normal_volatilities, shift=0.0):
    """Convert a grid while preserving its shape and all unavailable cells."""
    rows = len(fixing_dates)
    if not (len(expiry_years) == len(forwards) == len(normal_volatilities) == rows):
        raise ValueError("Dimensions temporelles incompatibles pour la surface Black.")
    if any(len(row) != len(strikes) for row in normal_volatilities):
        raise ValueError("Dimensions de strikes incompatibles pour la surface Black.")
    values, reasons, errors = [], [], []
    for forward, expiry, normal_row in zip(forwards, expiry_years, normal_volatilities):
        converted_row, reason_row = [], []
        for strike, normal_vol in zip(strikes, normal_row):
            vol, reason, error = normal_to_black_vol(forward, strike, expiry, normal_vol, shift)
            converted_row.append(vol)
            reason_row.append(reason)
            if error is not None:
                errors.append(error)
        values.append(converted_row)
        reasons.append(reason_row)
    return BlackVolSurface(
        shift=shift, strikes=list(strikes), fixing_dates=list(fixing_dates),
        expiry_years=list(expiry_years), volatilities=values, unavailable_reasons=reasons,
        max_abs_price_error=max(errors, default=None),
    )
