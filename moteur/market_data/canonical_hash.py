"""
Versioned canonicalisation primitive shared by every market-lineage fingerprint.

The fingerprint of a market object is always::

    sha256(utf8(schema_prefix + "\\n" + canonical_json(payload)))

Canonicalisation is intentionally strict so that the same payload always produces the
same hash, and two different payloads can never collide through a permissive encoder.
It normalises *encoding*, not meaning: if two payloads carry the same economics under
different spellings — an interpolation alias, say — they still hash differently, and
that is deliberate, staleness prefers a false positive to a missed mismatch. The rules:

- mapping keys must be strings and are sorted; mapping order is irrelevant;
- sequences keep the order in which they were received and consumed;
- dates are encoded as ``YYYY-MM-DD``; ``datetime`` is rejected;
- enums are encoded by their explicit value;
- numbers must be finite and are encoded without any locale-dependent format;
- every other type is rejected, so no ``repr``, pointer or QuantLib memory
  identity can ever reach a fingerprint.

Any change to these rules, or to the meaning of a payload, requires a new schema
version rather than a silent redefinition of existing hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from enum import Enum
from typing import Any

LINEAGE_SCHEMA_VERSION = "market-lineage-v1"

_JSON_SEPARATORS = (",", ":")


class CanonicalizationError(ValueError):
    """A payload cannot be canonicalised deterministically."""


def _canonicalize(value: Any, path: str) -> Any:
    """Return a JSON-ready value built only from deterministic primitives."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _canonicalize(value.value, path)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"Non-finite number at {path}")
        return value
    if isinstance(value, dt_datetime):
        raise CanonicalizationError(
            f"Datetime at {path} is not part of the market identity; use a date"
        )
    if isinstance(value, dt_date):
        return value.isoformat()
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"Non-string mapping key at {path}: {key!r}")
            canonical[key] = _canonicalize(item, f"{path}.{key}")
        return {key: canonical[key] for key in sorted(canonical)}
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        return [_canonicalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonicalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise CanonicalizationError(f"Unsupported payload type at {path}: {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    """Serialise a payload into its single canonical JSON representation."""
    return json.dumps(
        _canonicalize(payload, "$"),
        ensure_ascii=True,
        allow_nan=False,
        separators=_JSON_SEPARATORS,
        sort_keys=True,
    )


def canonical_fingerprint(schema_prefix: str, payload: Any) -> str:
    """Return the full SHA-256 fingerprint of a payload under a schema prefix."""
    if not schema_prefix:
        raise CanonicalizationError("A schema prefix is mandatory")
    encoded = f"{schema_prefix}\n{canonical_json(payload)}".encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
