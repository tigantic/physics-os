#!/usr/bin/env python3
"""Synchronize version numbers across all project manifests.

Reads the canonical VERSION file at the repository root and updates:
  - pyproject.toml         (version field — PACKAGE)
  - CITATION.cff           (version field — RELEASE)
  - tensornet/__init__.py  (__version__ — PACKAGE major.minor)
  - hypertensor/__init__.py (__version__, RUNTIME_VERSION — PACKAGE, RUNTIME)
  - Cargo.toml             (commented header version — RELEASE)

Usage:
  python tools/sync_versions.py          # Dry-run (show diffs)
  python tools/sync_versions.py --apply  # Write changes

Exit codes:
  0 - All versions in sync (or --apply succeeded)
  1 - Drift detected (dry-run mode)
  2 - VERSION file missing or malformed
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "VERSION"


def parse_version_file() -> dict[str, str]:
    """Parse VERSION file into a key->value dict."""
    if not VERSION_FILE.exists():
        print(f"ERROR: {VERSION_FILE} not found", file=sys.stderr)
        sys.exit(2)

    versions: dict[str, str] = {}
    for line in VERSION_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            print(f"ERROR: Malformed line in VERSION: {line!r}", file=sys.stderr)
            sys.exit(2)
        key, _, value = line.partition("=")
        versions[key.strip()] = value.strip()

    required_keys = {"RELEASE", "PLATFORM", "PACKAGE", "RUNTIME", "API_CONTRACT"}
    missing = required_keys - versions.keys()
    if missing:
        print(f"ERROR: VERSION missing keys: {missing}", file=sys.stderr)
        sys.exit(2)

    return versions


def _check_regex(
    path: Path,
    pattern: re.Pattern[str],
    expected: str,
    label: str,
    apply: bool,
    replacement_template: str,
) -> bool:
    """Generic check-and-replace for a regex pattern in a file."""
    if not path.exists():
        print(f"  SKIP: {path.name} not found")
        return True

    content = path.read_text()
    match = pattern.search(content)
    if not match:
        print(f"  WARN: No {label} field found in {path.name}")
        return True

    current = match.group(2)
    if current == expected:
        print(f"  OK: {path.name} {label}={current}")
        return True

    print(f"  DRIFT: {path.name} {label}={current} -> {expected}")
    if apply:
        new_content = pattern.sub(replacement_template.format(expected=expected), content, count=1)
        path.write_text(new_content)
        print(f"  FIXED: {path.name}")
        return True
    return False


def check_pyproject(versions: dict[str, str], apply: bool) -> bool:
    """Check/update pyproject.toml version field."""
    path = REPO_ROOT / "pyproject.toml"
    pattern = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)
    return _check_regex(
        path, pattern, versions["PACKAGE"], "version", apply,
        r'\g<1>{expected}\g<3>',
    )


def check_citation(versions: dict[str, str], apply: bool) -> bool:
    """Check/update CITATION.cff version field."""
    path = REPO_ROOT / "CITATION.cff"
    pattern = re.compile(r'^(version:\s*)(.+)$', re.MULTILINE)
    expected = versions["RELEASE"]

    if not path.exists():
        print(f"  SKIP: {path.name} not found")
        return True

    content = path.read_text()
    match = pattern.search(content)
    if not match:
        print(f"  WARN: No version field found in {path.name}")
        return True

    current = match.group(2).strip().strip('"').strip("'")
    if current == expected:
        print(f"  OK: {path.name} version={current}")
        return True

    print(f"  DRIFT: {path.name} version={current} -> {expected}")
    if apply:
        new_content = pattern.sub(rf'\g<1>"{expected}"', content, count=1)
        path.write_text(new_content)
        print(f"  FIXED: {path.name}")
        return True
    return False


def check_tensornet_init(versions: dict[str, str], apply: bool) -> bool:
    """Check/update tensornet/__init__.py __version__."""
    path = REPO_ROOT / "tensornet" / "__init__.py"
    # PACKAGE is e.g. "40.0.1", tensornet __version__ uses major.minor "40.0.0" or full
    pattern = re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.MULTILINE)
    pkg = versions["PACKAGE"]
    # Use major.minor.0 for the module (patch comes from pyproject)
    parts = pkg.split(".")
    expected = f"{parts[0]}.{parts[1]}.{parts[2] if len(parts) > 2 else '0'}"
    return _check_regex(
        path, pattern, expected, "__version__", apply,
        r'\g<1>{expected}\g<3>',
    )


def check_hypertensor_init(versions: dict[str, str], apply: bool) -> bool:
    """Check/update hypertensor/__init__.py __version__ and RUNTIME_VERSION."""
    path = REPO_ROOT / "hypertensor" / "__init__.py"
    if not path.exists():
        print(f"  SKIP: {path.name} not found")
        return True

    results: list[bool] = []
    content = path.read_text()
    original = content

    # __version__
    ver_pattern = re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.MULTILINE)
    expected_ver = versions["PACKAGE"]
    match = ver_pattern.search(content)
    if match:
        current = match.group(2)
        if current == expected_ver:
            print(f"  OK: {path.name} __version__={current}")
            results.append(True)
        else:
            print(f"  DRIFT: {path.name} __version__={current} -> {expected_ver}")
            if apply:
                content = ver_pattern.sub(rf'\g<1>{expected_ver}\g<3>', content, count=1)
                results.append(True)
            else:
                results.append(False)

    # RUNTIME_VERSION
    rt_pattern = re.compile(r'^(RUNTIME_VERSION\s*=\s*")([^"]+)(")', re.MULTILINE)
    expected_rt = versions["RUNTIME"]
    match = rt_pattern.search(content)
    if match:
        current = match.group(2)
        if current == expected_rt:
            print(f"  OK: {path.name} RUNTIME_VERSION={current}")
            results.append(True)
        else:
            print(f"  DRIFT: {path.name} RUNTIME_VERSION={current} -> {expected_rt}")
            if apply:
                content = rt_pattern.sub(rf'\g<1>{expected_rt}\g<3>', content, count=1)
                results.append(True)
            else:
                results.append(False)

    # API_VERSION
    api_pattern = re.compile(r'^(API_VERSION\s*=\s*")([^"]+)(")', re.MULTILINE)
    expected_api = versions.get("SUBSTRATE_API", versions.get("API_CONTRACT", "1"))
    match = api_pattern.search(content)
    if match:
        current = match.group(2)
        if current == expected_api:
            print(f"  OK: {path.name} API_VERSION={current}")
            results.append(True)
        else:
            print(f"  DRIFT: {path.name} API_VERSION={current} -> {expected_api}")
            if apply:
                content = api_pattern.sub(rf'\g<1>{expected_api}\g<3>', content, count=1)
                results.append(True)
            else:
                results.append(False)

    if apply and content != original:
        path.write_text(content)
        print(f"  FIXED: {path.name}")

    return all(results) if results else True


def check_cargo_toml(versions: dict[str, str], apply: bool) -> bool:
    """Check/update the commented version header in workspace Cargo.toml.

    The workspace Cargo.toml does not have a [package] version field (workspaces
    don't), but we maintain a version comment in the header block for tracking.
    """
    path = REPO_ROOT / "Cargo.toml"
    if not path.exists():
        print(f"  SKIP: {path.name} not found")
        return True

    content = path.read_text()
    expected = versions["RELEASE"]

    # Look for a version comment like: #  Version: 4.0.0
    pattern = re.compile(r'^(#\s*Version:\s*)(.+)$', re.MULTILINE)
    match = pattern.search(content)
    if not match:
        # No version comment exists yet — add one after the first header line
        title_pattern = re.compile(
            r'^(#\s+HYPERTENSOR WORKSPACE\s*\n)',
            re.MULTILINE,
        )
        title_match = title_pattern.search(content)
        if title_match and apply:
            insert_pos = title_match.end()
            version_line = f"#                    Version: {expected}\n"
            content = content[:insert_pos] + version_line + content[insert_pos:]
            path.write_text(content)
            print(f"  ADDED: {path.name} Version: {expected}")
            return True
        elif title_match:
            print(f"  DRIFT: {path.name} no Version comment (expected {expected})")
            return False
        else:
            print(f"  WARN: Cannot locate header in {path.name}")
            return True

    current = match.group(2).strip()
    if current == expected:
        print(f"  OK: {path.name} Version={current}")
        return True

    print(f"  DRIFT: {path.name} Version={current} -> {expected}")
    if apply:
        new_content = pattern.sub(rf'\g<1>{expected}', content, count=1)
        path.write_text(new_content)
        print(f"  FIXED: {path.name}")
        return True
    return False


def main() -> None:
    apply = "--apply" in sys.argv

    print(f"Reading {VERSION_FILE.relative_to(REPO_ROOT)}...")
    versions = parse_version_file()
    for key, val in sorted(versions.items()):
        print(f"  {key}={val}")
    print()

    print("Checking manifests:")
    results = [
        check_pyproject(versions, apply),
        check_citation(versions, apply),
        check_tensornet_init(versions, apply),
        check_hypertensor_init(versions, apply),
        check_cargo_toml(versions, apply),
    ]

    if all(results):
        print("\nAll versions in sync." if not apply else "\nAll versions synchronized.")
        sys.exit(0)
    else:
        print("\nVersion drift detected. Run with --apply to fix.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
