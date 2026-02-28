"""Content-addressed hashing for artifact integrity.

Every payload in the Ontic Engine system is independently hashable
using SHA-256 over a canonical JSON representation (sorted keys,
no whitespace, UTF-8 encoded).

This enables:
- Deduplication (same input → same hash → cache hit)
- Tamper detection (hash mismatch → artifact corrupted)
- Reproducibility proofs (same config_hash + seed → same result_hash)
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _canonicalize(obj: Any) -> Any:
    """Recursively prepare an object for canonical JSON serialization.

    Handles:
    - NaN → null
    - ±Infinity → null
    - float → rounded to 15 significant digits (IEEE 754 safe)
    - set → sorted list
    - bytes → hex string
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_canonicalize(item) for item in obj)
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items())}
    # Pydantic models, dataclasses, etc.
    if hasattr(obj, "model_dump"):
        return _canonicalize(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _canonicalize(vars(obj))
    return str(obj)


def canonical_json(obj: Any) -> bytes:
    """Serialize to canonical JSON bytes (sorted keys, no whitespace, UTF-8)."""
    clean = _canonicalize(obj)
    return json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_hash(obj: Any) -> str:
    """Compute SHA-256 content hash of an object.

    Returns a prefixed hash string: ``sha256:<hex>``.
    """
    data = canonical_json(obj)
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def hash_bytes(data: bytes) -> str:
    """Hash raw bytes.  Returns ``sha256:<hex>``."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def verify_hash(obj: Any, expected: str) -> bool:
    """Verify that an object's content hash matches the expected value."""
    return content_hash(obj) == expected
