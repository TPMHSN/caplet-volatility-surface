"""Charge les observations locales et complète une série publique si nécessaire."""

from datetime import date, timedelta
import json
from math import isfinite
from pathlib import Path

from market_data.fixing_store import fixing_store
from market_data.public_fixing_fetcher import fetch_public_fixing_series, get_public_fixing_source
from market_data.schemas import FixingDataPoint


def _read_history(path: Path, tenor: str, as_of: date) -> dict[date, float]:
    if not path.is_file():
        raise ValueError(f"Historique manquant pour EUR {tenor} ({path}).")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if (data["currency"], data["index_tenor"]) != ("EUR", tenor):
        raise ValueError(f"{path.name} ne correspond pas à EUR {tenor}.")
    history: dict[date, float] = {}
    for point in data["fixings"]:
        fixing = FixingDataPoint.model_validate(point)
        if not isfinite(fixing.rate):
            raise ValueError(f"Taux non fini dans {path.name} au {fixing.date}.")
        if fixing.date > as_of:
            continue
        if fixing.date in history and history[fixing.date] != fixing.rate:
            raise ValueError(f"Deux taux différents dans {path.name} au {fixing.date}.")
        history[fixing.date] = fixing.rate
    return history


def load_fixings(as_of: date, directory: Path, *, offline: bool) -> dict:
    histories = {}
    audit = {}
    for tenor in ("3M", "6M", "OIS"):
        name = f"EUR_{tenor}"
        path = directory / f"{name}.json"
        if tenor == "OIS" and not path.is_file():
            continue
        history = _read_history(path, tenor, as_of)
        source = get_public_fixing_source(name)
        new_count = 0
        note = None
        if as_of not in history and not offline:
            last = max(history, default=as_of - timedelta(days=1))
            try:
                observations = fetch_public_fixing_series(source, last + timedelta(days=1), as_of)
                for point in observations:
                    if last < point.date <= as_of:
                        history[point.date] = point.rate
                        new_count += 1
            except Exception as exc:
                note = str(exc)
        if as_of not in history and tenor != "OIS":
            detail = f" ({note})" if note else ""
            raise ValueError(
                f"Fixing Euribor {tenor} observé du {as_of} absent{detail}. "
                f"Vérifier sa publication ou compléter {path.name} avec le taux observé."
            )
        histories[tenor] = history
        audit[tenor] = {
            "date": as_of.isoformat(),
            "taux_decimal": history.get(as_of),
            "observations": len(history),
            "derniere_date": str(max(history)) if history else None,
            "observations_recuperees": new_count,
            "serie": source.series_id,
            "source": source.source_url,
            "note": note,
        }

    fixing_store.clear()
    for tenor, history in histories.items():
        fixing_store.store_fixings(
            "EUR", tenor, [FixingDataPoint(date=d, rate=r) for d, r in sorted(history.items())]
        )
    return audit
