"""Bounded C1 call-price repair for a failing native 6M volatility cubic.

Only a two-segment repair around one existing native knot, away from ATM, is supported. All
native prices, exterior prices and exterior first derivatives stay fixed.
Incompatible nodes or an infeasible local construction return no repair.
The untouched cubic exterior is checked by exact-rational polynomial bounds.
"""

from fractions import Fraction
from itertools import pairwise
from math import comb, pi, sqrt
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import brentq
from scipy.special import ndtr

FloatArray = npt.NDArray[np.float64]
ExactPolynomial = list[Fraction]


class LocalPriceRepair(TypedDict):
    strikes: list[float]
    call_prices: list[float]
    call_slopes: list[float]


def _calls(
    forward: float, expiry: float, strikes: npt.ArrayLike, vols: npt.ArrayLike
) -> FloatArray:
    std = np.asarray(vols) * sqrt(expiry)
    distance = forward - np.asarray(strikes)
    z = -np.abs(distance) / std
    time_value = std * np.exp(-z * z / 2) / sqrt(2 * pi) - np.abs(distance) * ndtr(z)
    return np.asarray(np.maximum(distance, 0) + time_value)


def _poly_add(left: ExactPolynomial, right: ExactPolynomial) -> ExactPolynomial:
    out = [Fraction(0)] * max(len(left), len(right))
    for values in (left, right):
        for i, value in enumerate(values):
            out[i] += value
    return out


def _poly_mul(left: ExactPolynomial, right: ExactPolynomial) -> ExactPolynomial:
    out = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[i + j] += a * b
    return out


def _bernstein_nonnegative(power: ExactPolynomial) -> bool:
    """Sufficient continuous sign certificate with exact binary-float inputs.

    De Casteljau subdivision preserves convex-hull bounds. An unresolved
    interval is rejected rather than accepted from samples or a tolerance.
    """
    n = len(power) - 1
    bernstein = [
        sum((power[j] * Fraction(comb(i, j), comb(n, j)) for j in range(i + 1)), Fraction(0))
        for i in range(n + 1)
    ]

    def certify(values: ExactPolynomial, depth: int) -> bool:
        if min(values) >= 0:
            return True
        if max(values) < 0 or values[0] < 0 or values[-1] < 0 or depth == 16:
            return False
        row, left, right = values, [values[0]], [values[-1]]
        while len(row) > 1:
            row = [(a + b) / 2 for a, b in pairwise(row)]
            left.append(row[0])
            right.append(row[-1])
        return certify(left, depth + 1) and certify(right[::-1], depth + 1)

    return certify(bernstein, 0)


def _cubic_exterior_is_convex(
    cubic: CubicSpline,
    forward: float,
    expiry: float,
    domain: tuple[float, float],
    left: float,
    right: float,
) -> bool:
    """C'' has the sign of T*v^3*v'' + (v + (F-K)*v')^2.

    Work on unit intervals and represent all polynomial arithmetic exactly.
    This certifies the stored cubic coefficients, avoiding root-finder/grid
    blind spots. Floating-point price evaluation still uses the existing guard.
    """
    for i in range(len(cubic.x) - 1):
        low = max(float(cubic.x[i]), domain[0])
        high = min(float(cubic.x[i + 1]), domain[1])
        for a, b in ((low, min(high, left)), (max(low, right), high)):
            if b <= a:
                continue
            shift = Fraction(a) - Fraction(float(cubic.x[i]))
            width = Fraction(b) - Fraction(a)
            base = [Fraction(float(c)) for c in cubic.c[::-1, i]]

            def compose(
                coefficients: ExactPolynomial, shift: Fraction = shift, width: Fraction = width
            ) -> ExactPolynomial:
                return [
                    sum(
                        (
                            coefficients[j] * comb(j, m) * shift ** (j - m) * width**m
                            for j in range(m, len(coefficients))
                        ),
                        Fraction(0),
                    )
                    for m in range(len(coefficients))
                ]

            v = compose(base)
            vp = compose([j * base[j] for j in range(1, 4)])
            vpp = compose([j * (j - 1) * base[j] for j in range(2, 4)])
            positive_v = v.copy()
            positive_v[0] -= Fraction(1e-14)
            if not _bernstein_nonnegative(positive_v):
                return False
            q = _poly_add(v, _poly_mul([Fraction(forward) - Fraction(a), -width], vp))
            v3 = _poly_mul(_poly_mul(v, v), v)
            p = _poly_add(_poly_mul(q, q), [Fraction(expiry) * x for x in _poly_mul(v3, vpp)])
            if not _bernstein_nonnegative(p):
                return False
    return True


def build_local_price_repair(
    forward: float,
    expiry: float,
    strikes: npt.ArrayLike,
    targets: npt.ArrayLike,
    domain: tuple[float, float],
) -> LocalPriceRepair | None:
    """Find a convex Hermite patch; never move targets to make it feasible."""
    k, v = np.asarray(strikes, dtype=float), np.asarray(targets, dtype=float)
    if k[0] > domain[0] or k[-1] < domain[1]:
        return None  # A D.4/blend wing needs its own analysis.
    native_prices = _calls(forward, expiry, k, v)
    secants = np.diff(native_prices) / np.diff(k)
    if np.any(secants < -1) or np.any(secants > 0) or np.any(np.diff(secants) < 0):
        return None
    cubic = CubicSpline(k, v, bc_type="natural")
    # Deterministic 0.1 bp search grid, independent of requested pricing strikes.
    grid = np.unique(np.round(np.r_[np.linspace(*domain, 6501), k], 12))
    vol = cubic(grid)
    if np.any(~np.isfinite(vol)) or np.any(vol <= 0):
        return None
    std = vol * sqrt(expiry)
    std_first, std_second = cubic(grid, 1) * sqrt(expiry), cubic(grid, 2) * sqrt(expiry)
    z = (forward - grid) / std
    phi = np.exp(-z * z / 2) / sqrt(2 * pi)
    first = -ndtr(z) + std_first * phi
    density = phi * (std_second + (1 + z * std_first) ** 2 / std)
    bad = np.flatnonzero(density < -1e-10)
    if not len(bad) or np.any(first < -1) or np.any(first > 0):
        return None
    center = int(np.argmin(abs(k - grid[int(np.argmin(density))])))
    if not 0 < center < len(k) - 1:
        return None
    middle = float(k[center])
    mi = int(np.argmin(abs(grid - middle)))
    li_bound = int(np.searchsorted(grid, k[center - 1], side="left"))
    ri_bound = int(np.searchsorted(grid, k[center + 1], side="right") - 1)
    if bad[0] <= li_bound or bad[-1] >= ri_bound:
        return None  # Do not replace several unrelated native intervals.
    price = _calls(forward, expiry, grid, vol)
    middle_price = float(native_prices[center])
    for width in range(int(bad[-1] - bad[0]) + 2, ri_bound - li_bound + 1):
        first_left = max(li_bound, int(bad[-1]) + 1 - width)
        last_left = min(int(bad[0]) - 1, mi - 1, ri_bound - width)
        for li in range(first_left, last_left + 1):
            ri = li + width
            if ri <= mi:
                continue
            left, right = float(grid[li]), float(grid[ri])
            if left < forward < right:
                continue  # Preserve the existing final/D.4 ATM reporting contract.
            dl, dr = float(first[li]), float(first[ri])
            sl = (middle_price - price[li]) / (middle - left)
            sr = (price[ri] - middle_price) / (right - middle)
            # C'' is linear on each Hermite segment. These four inequalities
            # make its values at both segment ends nonnegative, hence C'' >= 0.
            low = max(-1.0, dl, (3 * sl - dl) / 2, 3 * sr - 2 * dr)
            high = min(0.0, dr, 3 * sl - 2 * dl, (3 * sr - dr) / 2)
            if low > high - 1e-12:
                continue
            # Use the interior of the feasible interval to avoid a zero-density
            # segment endpoint becoming negative under coefficient rounding.
            dm = float((low + high) / 2)
            patch = LocalPriceRepair(
                strikes=[left, middle, right],
                call_prices=[float(price[li]), middle_price, float(price[ri])],
                call_slopes=[dl, dm, dr],
            )
            hermite = CubicHermiteSpline(
                patch["strikes"], patch["call_prices"], patch["call_slopes"]
            )
            # C'' is affine on each segment: exact-rational nonnegative
            # endpoint values certify every interior point of the stored cubic.
            densities = []
            for j in range(2):
                h = Fraction(patch["strikes"][j + 1]) - Fraction(patch["strikes"][j])
                c0, c1 = Fraction(float(hermite.c[0, j])), Fraction(float(hermite.c[1, j]))
                densities.extend([2 * c1, 6 * c0 * h + 2 * c1])
            if min(densities) < 0:
                continue
            if (
                np.min(
                    np.asarray(patch["call_prices"])
                    - np.maximum(forward - np.asarray(patch["strikes"]), 0)
                )
                <= 1e-14
            ):
                continue
            if not _cubic_exterior_is_convex(cubic, forward, expiry, domain, left, right):
                continue
            return patch
    return None


def apply_local_price_repair(
    forward: float,
    expiry: float,
    strike: npt.ArrayLike,
    original_vol: npt.ArrayLike,
    repair: LocalPriceRepair,
) -> FloatArray:
    """Invert only interior repaired prices; preserve exterior and native values."""
    query = np.asarray(strike, dtype=float)
    result = np.asarray(original_vol, dtype=float).copy()
    flat_k, flat_v = query.reshape(-1), result.reshape(-1)
    left, middle, right = repair["strikes"]
    indices = np.flatnonzero((flat_k > left) & (flat_k < right) & (flat_k != middle))
    if not len(indices):
        return result
    curve = CubicHermiteSpline(
        repair["strikes"], repair["call_prices"], repair["call_slopes"], extrapolate=False
    )
    for i in indices:
        strike_i = float(flat_k[i])
        price = float(curve(strike_i))
        intrinsic = max(forward - strike_i, 0.0)
        if not np.isfinite(price) or price - intrinsic <= 1e-14:
            raise ValueError("Le raccord local ne possède pas de valeur temps fiable.")
        upper = max(0.01, 2 * float(flat_v[i]))
        for _ in range(30):
            if _calls(forward, expiry, strike_i, upper) >= price:
                break
            upper *= 2
        else:
            raise ValueError("Inversion normale du raccord local non encadrée.")
        flat_v[i] = brentq(
            lambda sigma, strike_i=strike_i, price=price: (
                float(_calls(forward, expiry, strike_i, sigma)) - price
            ),
            1e-14,
            upper,
            xtol=1e-15,
        )
    return result


def price_repair_warning(repair: LocalPriceRepair) -> str:
    left, middle, right = repair["strikes"]
    return (
        f"Raccord local convexe en prix sur [{left:.6%} ; {right:.6%}], "
        f"nœud natif {middle:.6%} conservé. Prix et pente continus (C1) ; "
        "la densité peut sauter aux raccords."
    )
