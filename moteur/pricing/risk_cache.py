"""Bounded, source-identity caches for reusable pricing risk scenarios."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from typing import Generic, TypeVar, cast
from weakref import WeakSet

ValueT = TypeVar("ValueT")
CacheKey = tuple[object, ...]


@dataclass(frozen=True, eq=False)
class _IdentityKey:
    context: object
    dimensions: tuple[object, ...]

    def __hash__(self) -> int:
        return hash((id(self.context), self.dimensions))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _IdentityKey)
            and self.context is other.context
            and self.dimensions == other.dimensions
        )


class IdentityLruCache(Generic[ValueT]):
    """Mapping-like cache whose first key item identifies an immutable market context."""

    def __init__(self, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[_IdentityKey, ValueT] = OrderedDict()
        self._lock = RLock()

    def __getitem__(self, key: CacheKey) -> ValueT:
        with self._lock:
            identity_key = self._identity_key(key)
            value = self._entries[identity_key]
            self._entries.move_to_end(identity_key)
            return value

    def __setitem__(self, key: CacheKey, value: ValueT) -> None:
        if not key:
            raise ValueError("Risk cache keys require a market context")
        with self._lock:
            identity_key = self._identity_key(key)
            self._entries[identity_key] = value
            self._entries.move_to_end(identity_key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple) or not key:
            return False
        with self._lock:
            return self._identity_key(key) in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: CacheKey, default: ValueT | None = None) -> ValueT | None:
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def invalidate_context(self, context: object) -> tuple[ValueT, ...]:
        """Remove entries rooted in exactly this source object."""
        with self._lock:
            stale = [key for key in self._entries if key.context is context]
            removed = tuple(self._entries[key] for key in stale)
            for key in stale:
                del self._entries[key]
            return removed

    @staticmethod
    def _identity_key(key: CacheKey) -> _IdentityKey:
        if not key:
            raise ValueError("Risk cache keys require a market context")
        return _IdentityKey(context=key[0], dimensions=key[1:])


_RISK_CACHES: WeakSet[IdentityLruCache[object]] = WeakSet()


def create_risk_cache(max_entries: int) -> IdentityLruCache[ValueT]:
    """Create a cache included in curve-publication invalidation."""
    cache: IdentityLruCache[ValueT] = IdentityLruCache(max_entries)
    _RISK_CACHES.add(cast(IdentityLruCache[object], cache))
    return cache


def invalidate_risk_context(context: object) -> None:
    """Invalidate one replaced market context across registered risk caches."""
    for cache in tuple(_RISK_CACHES):
        cache.invalidate_context(context)


def clear_risk_caches() -> None:
    """Drop all risk scenarios, such as when the entire curve store is cleared."""
    for cache in tuple(_RISK_CACHES):
        cache.clear()


def curve_market_signature(curve_set: object) -> str | None:
    """Fingerprint current raw curve inputs when the context retains them."""
    from curves.effective_conventions import curve_inputs_fingerprint

    ois_data = getattr(curve_set, "ois_market_data", None)
    forward_data = getattr(curve_set, "forward_market_data", None)
    curve_date = getattr(curve_set, "curve_date", None)
    currency = getattr(curve_set, "currency", None)
    if ois_data is None or forward_data is None or curve_date is None or currency is None:
        return None
    discount_interpolation = getattr(
        curve_set,
        "discount_interpolation",
        getattr(curve_set, "interpolation", "unknown"),
    )
    forward_interpolation = getattr(
        curve_set,
        "forward_interpolation",
        getattr(curve_set, "interpolation", "unknown"),
    )
    interpolation_pair = f"discount={discount_interpolation}|forward={forward_interpolation}"
    return curve_inputs_fingerprint(
        currency=currency,
        curve_date=curve_date,
        requested_interpolation=interpolation_pair,
        ois_data=ois_data,
        forward_data_by_tenor=forward_data,
        observed_calibration_fixings=getattr(
            curve_set, "observed_calibration_fixings", {}
        ),
    )


def stamp_curve_market_signature(curve_set: object, signature: str | None = None) -> str | None:
    """Record the market version used by later risk-cache keys."""
    resolved = curve_market_signature(curve_set) if signature is None else signature
    try:
        vars(curve_set)["_risk_market_signature"] = resolved
    except TypeError:
        pass
    return resolved


def fixing_history_fingerprint(curve_set: object, curve_keys: tuple[str, ...]) -> str:
    """Hash observed QuantLib fixings for the indices used by a risk scenario."""
    from curves.helpers import ql_state_lock

    with ql_state_lock():
        return _fixing_history_fingerprint_locked(curve_set, curve_keys)


def _fixing_history_fingerprint_locked(curve_set: object, curve_keys: tuple[str, ...]) -> str:
    """Read a consistent set of native index histories under the process state lock."""
    from config import get_index_with_curve

    digest = sha256()
    currency = str(getattr(curve_set, "currency", ""))
    get_forward_curve = getattr(curve_set, "get_forward_curve", None)
    for curve_key in sorted(set(curve_keys)):
        try:
            if get_forward_curve is None:
                raise ValueError("Market context has no forward-curve resolver")
            forward_curve = get_forward_curve(curve_key)
            index = get_index_with_curve(currency, curve_key, forward_curve)
        except (KeyError, ValueError):
            digest.update(f"{curve_key}:unavailable;".encode())
            continue
        history = index.timeSeries()
        digest.update(f"{curve_key}:{index.name()};".encode())
        for fixing_date, value in zip(history.dates(), history.values()):
            digest.update(f"{fixing_date.serialNumber()}:{float(value).hex()};".encode())
    return digest.hexdigest()


def risk_context_fingerprint(
    curve_set: object,
    curve_keys: tuple[str, ...],
) -> tuple[object, ...]:
    """Return compact mutable-state dimensions required by a risk-scenario cache key."""
    market_signature = getattr(curve_set, "_risk_market_signature", None)
    if market_signature is None:
        lineage = getattr(curve_set, "lineage", None)
        market_signature = getattr(lineage, "curve_inputs_fingerprint", None)
    if market_signature is None:
        market_signature = stamp_curve_market_signature(curve_set)

    exclusion_signature = tuple(
        sorted(
            (
                report_key,
                tuple(
                    sorted(
                        [
                            (
                                exclusion.curve_key,
                                exclusion.quote_index,
                                exclusion.instrument_type,
                                exclusion.source_label,
                                exclusion.pillar_date,
                            )
                            for exclusion in report.excluded_quotes
                        ],
                        key=repr,
                    )
                ),
            )
            for report_key, report in (getattr(curve_set, "calibration_reports", {}) or {}).items()
        )
    )
    fixing_signature = fixing_history_fingerprint(curve_set, curve_keys)
    calibration_signature = tuple(
        sorted(
            (key, float(value))
            for key, value in getattr(curve_set, "calibration_fixings", {}).items()
        )
    )
    return market_signature, exclusion_signature, fixing_signature, calibration_signature
