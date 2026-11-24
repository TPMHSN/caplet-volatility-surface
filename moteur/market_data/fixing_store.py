"""
In-memory store for historical rate fixings.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional

from .schemas import FixingDataPoint, FixingStatus

logger = logging.getLogger(__name__)


class FixingSnapshot:
    """Read-only copy of every fixing history of a store at one instant (XVAEUR X9b).

    ``get_fixing`` answers as the store's did when the copy was taken; later upserts to the store
    are not seen.
    """

    __slots__ = ("_fixings",)

    def __init__(self, fixings: Mapping[str, Mapping[date, float]]) -> None:
        self._fixings = fixings

    def get_fixing(self, currency: str, index_tenor: str, fixing_date: date) -> Optional[float]:
        """Get a single fixing rate of the copy. Returns None if it was not stored."""
        history = self._fixings.get(FixingStore._key(currency, index_tenor))
        return None if history is None else history.get(fixing_date)


@dataclass(frozen=True)
class FixingStoreWriteResult:
    """Counters returned by non-destructive fixing upserts."""

    new_count: int = 0
    duplicate_count: int = 0
    conflict_count: int = 0
    overwritten_count: int = 0


class FixingStore:
    """In-memory store for historical rate fixings.

    Keyed by "{currency}_{index_tenor}" (e.g. "EUR_3M", "EUR_OIS").
    Inner dict maps date → rate (decimal).
    """

    def __init__(self) -> None:
        self._fixings: Dict[str, Dict[date, float]] = {}
        # upserts, clear and snapshot exclude each other: a snapshot never sees half an upsert
        self._lock = threading.Lock()

    @staticmethod
    def _key(currency: str, index_tenor: str) -> str:
        return f"{currency}_{index_tenor}"

    def store_fixings(self, currency: str, index_tenor: str, fixings: List[FixingDataPoint]) -> int:
        """Store fixings, returning count of newly added (non-duplicate) entries."""
        result = self.upsert_fixings(
            currency,
            index_tenor,
            fixings,
            overwrite_conflicts=True,
        )
        return result.new_count

    def upsert_fixings(
        self,
        currency: str,
        index_tenor: str,
        fixings: List[FixingDataPoint],
        *,
        overwrite_conflicts: bool = False,
        conflict_tolerance: float = 1.0e-12,
    ) -> FixingStoreWriteResult:
        """Upsert fixings and report duplicates/conflicts without forcing overwrites."""
        key = self._key(currency, index_tenor)
        new_count = 0
        duplicate_count = 0
        conflict_count = 0
        overwritten_count = 0

        with self._lock:
            if key not in self._fixings:
                self._fixings[key] = {}

            for fp in fixings:
                existing = self._fixings[key].get(fp.date)
                if existing is None:
                    new_count += 1
                    self._fixings[key][fp.date] = fp.rate
                    continue

                if abs(existing - fp.rate) <= conflict_tolerance:
                    duplicate_count += 1
                    continue

                conflict_count += 1
                if overwrite_conflicts:
                    overwritten_count += 1
                    self._fixings[key][fp.date] = fp.rate

        return FixingStoreWriteResult(
            new_count=new_count,
            duplicate_count=duplicate_count,
            conflict_count=conflict_count,
            overwritten_count=overwritten_count,
        )

    def get_fixing(self, currency: str, index_tenor: str, fixing_date: date) -> Optional[float]:
        """Get a single fixing rate. Returns None if not stored."""
        key = self._key(currency, index_tenor)
        return self._fixings.get(key, {}).get(fixing_date)

    def get_all_fixings(self, currency: str, index_tenor: str) -> Dict[date, float]:
        """Get all fixings for an index."""
        key = self._key(currency, index_tenor)
        return dict(self._fixings.get(key, {}))

    def export_currency(self, currency: str) -> Dict[str, Dict[date, float]]:
        """Copy of every fixing history of a currency, keyed like the store."""
        return self.export_currencies((currency,))

    def export_currencies(self, currencies: Iterable[str]) -> Dict[str, Dict[date, float]]:
        """Copy selected currency histories atomically between whole fixing upserts."""
        prefixes = tuple(f"{currency}_" for currency in sorted(set(currencies)))
        if not prefixes:
            return {}
        with self._lock:
            return {
                key: dict(history)
                for key, history in self._fixings.items()
                if key.startswith(prefixes)
            }

    def get_last_fixing_date(self, currency: str, index_tenor: str) -> Optional[date]:
        """Return the latest stored fixing date for an index."""
        fixings = self.get_all_fixings(currency, index_tenor)
        if not fixings:
            return None
        return max(fixings.keys())

    def get_index_count(self, currency: str, index_tenor: str) -> int:
        """Return number of stored fixings for an index."""
        key = self._key(currency, index_tenor)
        return len(self._fixings.get(key, {}))

    def get_status(self) -> FixingStatus:
        """Get summary of stored fixings."""
        indices = sorted(self._fixings.keys())
        total = sum(len(v) for v in self._fixings.values())
        date_ranges: Dict[str, Dict[str, str]] = {}
        for key, dates_map in self._fixings.items():
            if dates_map:
                min_d = min(dates_map.keys())
                max_d = max(dates_map.keys())
                date_ranges[key] = {
                    "min_date": min_d.isoformat(),
                    "max_date": max_d.isoformat(),
                }
        return FixingStatus(
            indices=indices,
            total_count=total,
            date_ranges=date_ranges if date_ranges else None,
        )

    def snapshot(self) -> FixingSnapshot:
        """Read-only copy of every fixing history now, taken between two whole upserts (X9b)."""
        with self._lock:
            return FixingSnapshot({key: dict(history) for key, history in self._fixings.items()})

    def clear(self) -> None:
        """Clear all fixings."""
        with self._lock:
            self._fixings.clear()


# Global singleton
fixing_store = FixingStore()
