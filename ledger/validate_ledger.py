#!/usr/bin/env python3
"""
Validate all capability ledger node files against the schema.

Exit codes:
  0 — all nodes valid
  1 — validation errors found

Usage:
  python3 ledger/validate_ledger.py
  python3 ledger/validate_ledger.py --strict  # also checks for unassigned owners
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "ledger"
SCHEMA_PATH = LEDGER / "schema.yaml"
NODES_DIR = LEDGER / "nodes"
INDEX_PATH = LEDGER / "index.yaml"

VALID_STATES = {"V0.0", "V0.1", "V0.2", "V0.3", "V0.4", "V0.5", "V0.6", "V1.0"}
VALID_TIERS = {"A", "B", "C"}
REQUIRED_FIELDS = {"id", "name", "pack", "pack_name", "owner", "tier", "state"}


def validate_node(path: pathlib.Path, strict: bool = False) -> list[str]:
    """Validate a single node YAML file. Returns list of error strings."""
    errors: list[str] = []
    filename = path.name

    try:
        data: dict[str, Any] = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        return [f"{filename}: YAML parse error: {e}"]

    if not isinstance(data, dict):
        return [f"{filename}: top-level value must be a mapping, got {type(data).__name__}"]

    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"{filename}: missing required field '{field}'")
        elif not data[field]:
            errors.append(f"{filename}: field '{field}' is empty")

    # State validation
    if "state" in data and data["state"] not in VALID_STATES:
        errors.append(f"{filename}: invalid state '{data['state']}', must be one of {VALID_STATES}")

    # Tier validation
    if "tier" in data and data["tier"] not in VALID_TIERS:
        errors.append(f"{filename}: invalid tier '{data['tier']}', must be one of {VALID_TIERS}")

    # ID matches filename
    if "id" in data:
        expected_filename = f"{data['id']}.yaml"
        if path.name != expected_filename:
            errors.append(f"{filename}: id '{data['id']}' does not match filename (expected {expected_filename})")

    # Type checks
    if "source_files" in data and data["source_files"] is not None:
        if not isinstance(data["source_files"], list):
            errors.append(f"{filename}: source_files must be a list")

    if "benchmarks" in data and data["benchmarks"] is not None:
        if not isinstance(data["benchmarks"], list):
            errors.append(f"{filename}: benchmarks must be a list")

    if "tests" in data and data["tests"] is not None:
        if not isinstance(data["tests"], dict):
            errors.append(f"{filename}: tests must be a mapping")

    # Strict mode: check for unassigned owners
    if strict and data.get("owner") == "unassigned":
        errors.append(f"{filename}: owner is 'unassigned' (strict mode)")

    return errors


def validate_index(node_ids: set[str]) -> list[str]:
    """Validate the index.yaml file."""
    errors: list[str] = []

    if not INDEX_PATH.exists():
        return ["index.yaml: file does not exist"]

    try:
        data = yaml.safe_load(INDEX_PATH.read_text())
    except yaml.YAMLError as e:
        return [f"index.yaml: YAML parse error: {e}"]

    if not isinstance(data, dict):
        return ["index.yaml: top-level value must be a mapping"]

    # Check total count
    if data.get("total_nodes") != len(node_ids):
        errors.append(
            f"index.yaml: total_nodes ({data.get('total_nodes')}) "
            f"does not match actual node count ({len(node_ids)})"
        )

    # Check all nodes are listed
    if "nodes" in data:
        index_ids = {n["id"] for n in data["nodes"] if isinstance(n, dict) and "id" in n}
        missing = node_ids - index_ids
        extra = index_ids - node_ids
        if missing:
            errors.append(f"index.yaml: missing nodes: {sorted(missing)}")
        if extra:
            errors.append(f"index.yaml: extra nodes not in nodes/: {sorted(extra)}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate capability ledger")
    parser.add_argument("--strict", action="store_true", help="Enable strict checks (e.g., no unassigned owners)")
    args = parser.parse_args()

    if not NODES_DIR.exists():
        print(f"ERROR: {NODES_DIR} does not exist", file=sys.stderr)
        sys.exit(2)

    node_files = sorted(NODES_DIR.glob("*.yaml"))
    if not node_files:
        print(f"ERROR: no .yaml files found in {NODES_DIR}", file=sys.stderr)
        sys.exit(2)

    all_errors: list[str] = []
    node_ids: set[str] = set()

    # Validate each node
    for path in node_files:
        errors = validate_node(path, strict=args.strict)
        all_errors.extend(errors)
        try:
            data = yaml.safe_load(path.read_text())
            if isinstance(data, dict) and "id" in data:
                nid = data["id"]
                if nid in node_ids:
                    all_errors.append(f"{path.name}: duplicate id '{nid}'")
                node_ids.add(nid)
        except Exception:
            pass

    # Validate index
    index_errors = validate_index(node_ids)
    all_errors.extend(index_errors)

    # Report
    print(f"Validated {len(node_files)} node files + index.yaml")
    if all_errors:
        print(f"\n{len(all_errors)} error(s) found:\n")
        for err in all_errors:
            print(f"  ✗ {err}")
        sys.exit(1)
    else:
        print("✓ All valid.")
        # Print summary
        states: dict[str, int] = {}
        tiers: dict[str, int] = {}
        for path in node_files:
            data = yaml.safe_load(path.read_text())
            states[data["state"]] = states.get(data["state"], 0) + 1
            tiers[data["tier"]] = tiers.get(data["tier"], 0) + 1
        print(f"  States: {dict(sorted(states.items()))}")
        print(f"  Tiers:  {dict(sorted(tiers.items()))}")
        sys.exit(0)


if __name__ == "__main__":
    main()
