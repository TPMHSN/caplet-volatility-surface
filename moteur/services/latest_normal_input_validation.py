"""Necessary price checks on raw native quotes before candidate calibration.

This gate uses the loaded curves and native coupon conventions. It identifies
material contradictions; passing it is not a full arbitrage certificate. Raw
quotes, their strike labels and the candidate's calibration rules are unchanged.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import fsum

from services.exceptions import ValidationError
from services.latest_normal_diagnostics import (
    QUOTE_FIT_ABSOLUTE_TOLERANCE_EUR,
    QUOTE_FIT_RELATIVE_TOLERANCE,
    REFERENCE_NOTIONAL,
)
from services.thomas_stripping import _intrinsic, _time_value
from services.thomas_stripping_schemas import OptionletNode


@dataclass(frozen=True)
class NativeCapQuoteViolation:
    expiry_months: int
    strikes: tuple[float, ...]
    normal_quotes_bp: tuple[float, ...]
    criterion: str
    defect_eur_10m: float
    tolerance_eur_10m: float
    description: str


class CapQuoteConsistencyError(ValidationError, ValueError):
    def __init__(self, tenor: str, violations: list[NativeCapQuoteViolation]):
        self.violations = tuple(violations)
        details = " ".join(issue.description for issue in violations[:3])
        super().__init__(
            f"Cotations natives {tenor} incohérentes avec les courbes et conventions chargées "
            f"({len(violations)} contrôles hors tolérance). {details} "
            "Vérifier les en-têtes de strikes, la colonne ATM et l’ordre des colonnes dans la source. "
            "Calibration arrêtée avant le raccord 3M/6M ; aucune quote n’a été modifiée."
        )


def validate_native_cap_quotes(
    tenor: str,
    rows: Sequence[tuple[int, Sequence[float]]],
    strikes: Sequence[float],
    nodes: Sequence[OptionletNode],
) -> None:
    violations = []
    for month, quotes in rows:
        included = [node for node in nodes if node.cap_maturity_months <= month]
        if not included or included[-1].cap_maturity_months != month:
            raise ValidationError(f"Calendrier natif {tenor} incomplet au pilier {month}M.")
        if len(strikes) != len(quotes) or any(a >= b for a, b in pairwise(strikes)):
            raise ValidationError("Dimensions ou ordre des strikes natifs incohérents.")
        caps, floors = [], []
        for strike, sigma in zip(strikes, quotes, strict=True):
            tv = fsum(_time_value(node, strike, sigma) for node in included)
            caps.append(
                REFERENCE_NOTIONAL * (tv + fsum(_intrinsic(node, strike) for node in included))
            )
            floors.append(
                REFERENCE_NOTIONAL
                * (
                    tv
                    + fsum(
                        node.accrual_fraction
                        * node.discount_factor
                        * max(strike - node.forward_rate, 0.0)
                        for node in included
                    )
                )
            )

        def tolerance(columns, cap_prices=caps, floor_prices=floors):
            # Use the OTM side so a large intrinsic amount cannot mask a defect.
            scale = max(min(cap_prices[j], floor_prices[j]) for j in columns)
            return max(QUOTE_FIT_ABSOLUTE_TOLERANCE_EUR, QUOTE_FIT_RELATIVE_TOLERANCE * scale)

        def record(
            criterion,
            columns,
            defect,
            description,
            native_month=month,
            native_quotes=quotes,
            tolerance_fn=tolerance,
        ):
            allowed = tolerance_fn(columns)
            if defect > allowed:
                expiry = f"{native_month//12}Y" if native_month % 12 == 0 else f"{native_month}M"
                violations.append(
                    NativeCapQuoteViolation(
                        native_month,
                        tuple(strikes[j] for j in columns),
                        tuple(native_quotes[j] * 10000 for j in columns),
                        criterion,
                        defect,
                        allowed,
                        f"{tenor} {expiry} : {description} "
                        f"(écart {defect:.2f} EUR, tolérance {allowed:.2f} EUR pour 10 M EUR).",
                    )
                )

        for j in range(len(strikes) - 1):
            left, right = strikes[j : j + 2]
            record(
                "floor_monotonicity",
                (j, j + 1),
                floors[j] - floors[j + 1],
                f"floor {left:.2%} = {floors[j]:.2f} EUR supérieur au floor {right:.2%} = {floors[j+1]:.2f} EUR",
            )
            record(
                "cap_monotonicity",
                (j, j + 1),
                caps[j + 1] - caps[j],
                f"cap {right:.2%} = {caps[j+1]:.2f} EUR supérieur au cap {left:.2%} = {caps[j]:.2f} EUR",
            )
        for j in range(1, len(strikes) - 1):
            weight = (strikes[j] - strikes[j - 1]) / (strikes[j + 1] - strikes[j - 1])
            chord = (1 - weight) * floors[j - 1] + weight * floors[j + 1]
            record(
                "convexity",
                (j - 1, j, j + 1),
                floors[j] - chord,
                f"prix non convexe entre {strikes[j-1]:.2%}, {strikes[j]:.2%} et {strikes[j+1]:.2%}",
            )
    if violations:
        raise CapQuoteConsistencyError(tenor, violations)
