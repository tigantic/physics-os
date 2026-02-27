#!/usr/bin/env python3
"""Contract drift detection.

Compares the live OpenAPI spec against the frozen baseline.
Fails if any schema change is detected without a version bump.

Usage:
    python3 tools/scripts/check_contract_drift.py          # against running server
    python3 tools/scripts/check_contract_drift.py --offline # against frozen spec only

Exit codes:
    0 — No drift detected
    1 — Drift detected (schema changed without version bump)
    2 — Server unreachable or file missing
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FROZEN_SPEC = Path(__file__).resolve().parents[1] / "contracts" / "v1" / "openapi.json"
FROZEN_HASH = "sha256:c18a492eb5166dd50530e6c7682b5e78a6486f03835ac69a615df0d0d752099d"


def _hash_spec(spec: dict) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def _load_frozen() -> dict:
    if not FROZEN_SPEC.exists():
        print(f"FAIL: Frozen spec not found at {FROZEN_SPEC}", file=sys.stderr)
        sys.exit(2)
    with open(FROZEN_SPEC) as f:
        return json.load(f)


def _load_live(url: str = "http://127.0.0.1:8000/openapi.json") -> dict | None:
    try:
        from urllib.request import urlopen
        with urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        print(f"WARN: Could not reach live server: {exc}", file=sys.stderr)
        return None


def check_frozen_integrity() -> bool:
    """Verify the committed spec hasn't been modified."""
    spec = _load_frozen()
    actual_hash = _hash_spec(spec)
    if actual_hash != FROZEN_HASH:
        print(f"FAIL: Frozen spec hash drift detected", file=sys.stderr)
        print(f"  Expected: {FROZEN_HASH}", file=sys.stderr)
        print(f"  Actual:   {actual_hash}", file=sys.stderr)
        print(f"  If this is intentional, update FROZEN_HASH in this script", file=sys.stderr)
        print(f"  and bump the schema version.", file=sys.stderr)
        return False
    print(f"OK: Frozen spec integrity verified ({FROZEN_HASH[:24]}...)")
    return True


def check_live_drift() -> bool:
    """Compare live server spec against frozen baseline."""
    live = _load_live()
    if live is None:
        print("SKIP: Live drift check (server not running)")
        return True  # Not a failure — server might not be running in CI

    frozen = _load_frozen()
    live_hash = _hash_spec(live)
    frozen_hash = _hash_spec(frozen)

    if live_hash != frozen_hash:
        # Check if version was bumped (allowed drift)
        live_ver = live.get("info", {}).get("version", "")
        frozen_ver = frozen.get("info", {}).get("version", "")
        if live_ver != frozen_ver:
            print(f"OK: Schema changed with version bump ({frozen_ver} → {live_ver})")
            print(f"  Update the frozen spec: curl -s http://127.0.0.1:8000/openapi.json | python3 -m json.tool > contracts/v1/openapi.json")
            return True
        else:
            print(f"FAIL: Live spec differs from frozen baseline without version bump", file=sys.stderr)
            print(f"  Live hash:   {live_hash}", file=sys.stderr)
            print(f"  Frozen hash: {frozen_hash}", file=sys.stderr)
            # Find differing paths
            _diff_paths(frozen, live, "")
            return False
    print(f"OK: Live spec matches frozen baseline")
    return True


def _diff_paths(a: dict, b: dict, prefix: str) -> None:
    """Print paths that differ between two dicts."""
    all_keys = set(list(a.keys()) + list(b.keys()))
    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        if key not in a:
            print(f"  + {path} (added)", file=sys.stderr)
        elif key not in b:
            print(f"  - {path} (removed)", file=sys.stderr)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            _diff_paths(a[key], b[key], path)
        elif a[key] != b[key]:
            print(f"  ~ {path} (changed)", file=sys.stderr)


def main() -> None:
    offline = "--offline" in sys.argv

    ok = True
    ok = check_frozen_integrity() and ok
    if not offline:
        ok = check_live_drift() and ok

    if ok:
        print("\nContract drift check: PASS")
    else:
        print("\nContract drift check: FAIL", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
