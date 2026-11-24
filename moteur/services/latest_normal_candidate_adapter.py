"""Snapshot-local bridge and display adapters; never re-fit the 3M candidate."""

from __future__ import annotations

from collections.abc import Callable
from math import fsum, pi

import numpy as np
from scipy.special import ndtr

from services.latest_normal_candidate import CandidatePricingError, FrozenThreeMonthCandidate
from services.latest_normal_short_end import _set_node_vols
from services.thomas_stripping_schemas import OptionletNode, StrippingFamily


def candidate_source(
    candidate: FrozenThreeMonthCandidate, three: StrippingFamily
) -> StrippingFamily:
    source = three.model_copy(deep=True)
    for index, node in enumerate(source.optionlets):
        _set_node_vols(node, source.strikes, candidate.vol(index, source.strikes), 1.0)
    # Negative-wing residuals are diagnostics, not exact constraints of this candidate.
    errors: list[float] = []
    for cap in source.caps:
        cap.repriced_prices = [
            fsum(float(node.prices[j]) for node in source.optionlets[: cap.optionlets_count])
            for j in range(len(source.strikes))
        ]
        cap.repricing_errors = [a - b for a, b in zip(cap.repriced_prices, cap.prices, strict=True)]
        errors.extend(abs(error) for error in cap.repricing_errors)
    source.max_abs_repricing_error = max(errors)
    return source


def bridge_provider(
    candidate: FrozenThreeMonthCandidate,
    outside: list[float] | None = None,
) -> Callable[[OptionletNode, float], float]:
    by_date = {node.fixing_date: i for i, node in enumerate(candidate.nodes)}

    def evaluate(node: OptionletNode, strike: float) -> float:
        if outside is not None and not candidate.domain[0] <= strike <= candidate.domain[1]:
            outside.append(strike)
        return float(candidate.vol(by_date[node.fixing_date], strike))

    return evaluate


def candidate_slice(candidate: FrozenThreeMonthCandidate, fixing_date: str) -> dict[str, object]:
    index = next(
        i for i, node in enumerate(candidate.nodes) if str(node.fixing_date) == fixing_date
    )
    shape = candidate.shape_check(index)
    if not shape["passed"]:
        raise CandidatePricingError(
            "Le smile candidat échoue au contrôle numérique des prix par strike."
        )
    forward = float(candidate.F[index])
    grid = np.unique(np.r_[np.linspace(*candidate.domain, 131), -0.015, 0.0, forward])
    grid = grid[(grid >= candidate.domain[0]) & (grid <= candidate.domain[1])]
    components = candidate.components(index, grid)
    return {
        "status": "ready",
        "atm_normal_vol": float(candidate.vol(index, forward)),
        "targets": [
            {"strike": float(k), "normal_vol": float(v)}
            for k, v in zip(
                candidate.positive_strikes,
                candidate.vol(index, candidate.positive_strikes),
                strict=True,
            )
        ],
        "curve": [
            {"strike": float(k), **{name: float(values[i]) for name, values in components.items()}}
            for i, k in enumerate(grid)
        ],
    }


def check_coupon_shape(candidate: FrozenThreeMonthCandidate, forward: float, expiry: float) -> None:
    """Check the actual off-grid coupon, not only its neighbouring native slices."""
    strikes = np.round(np.linspace(*candidate.domain, 2601), 12)
    vols = candidate.vol_for_coupon(forward, expiry, strikes)
    std = vols * np.sqrt(expiry)
    z = (forward - strikes) / std
    calls = (forward - strikes) * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2 * pi)
    slopes = np.diff(calls) / np.diff(strikes)
    density = 2 * np.diff(slopes) / (strikes[2:] - strikes[:-2])
    if (
        np.any(~np.isfinite(calls))
        or slopes.min() < -1 - 1e-6
        or slopes.max() > 1e-6
        or density.min() < -1e-4
    ):
        raise CandidatePricingError(
            "Smile du coupon candidat hors grille non monotone ou non convexe ; pricing indisponible."
        )
