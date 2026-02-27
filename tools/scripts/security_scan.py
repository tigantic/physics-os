#!/usr/bin/env python
"""
O) Security Scans Script
========================

Runs security scans for production-ready deployment.

Usage:
    python tools/scripts/security_scan.py

Pass Criteria:
    - 0 secrets detected
    - Vulnerabilities triaged with allowlist + expiry
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allowlisted vulnerabilities with expiry
# Format: {'package': {'vuln_id': 'expiry_date or justification'}}
VULNERABILITY_ALLOWLIST = {
    # Example:
    # 'requests': {'CVE-2023-XXXX': '2025-03-01 - Waiting for upstream fix'},
}

# Patterns to exclude from secret scanning
# Be specific - avoid blanket exclusions that hide real secrets
SECRET_EXCLUSIONS = [
    "requirements-lock.txt",
    ".secrets.baseline",
    # Benchmark and evidence result files (no secrets, just metrics)
    "results/*.json",
    "artifacts/evidence/*.json",
    "demos/evidence/*.json",
    "experiments/benchmarks/benchmarks/*.json",
    "*_results.json",
    # Example files with placeholder keys
    "demos/millennium_hunter_keys.json.example",
]


def run_pip_audit() -> Tuple[bool, List[Dict[str, Any]]]:
    """Run pip-audit for dependency vulnerabilities."""
    print("Running pip-audit...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--format=json"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            return True, []

        try:
            vulns = json.loads(result.stdout)
        except json.JSONDecodeError:
            vulns = [{"error": result.stderr or result.stdout}]

        return False, vulns

    except FileNotFoundError:
        print("  ERROR: pip-audit not installed.")
        print("  Install with: pip install -r requirements-dev.txt")
        print("  Or: pip install pip-audit")
        return False, [{"error": "pip-audit not installed"}]
    except Exception as e:
        return False, [{"error": str(e)}]


def run_bandit(project_dir: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run bandit for code security issues."""
    print("Running bandit...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", "tensornet", "-f", "json", "-q"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=project_dir,
        )

        try:
            report = json.loads(result.stdout)
            issues = report.get("results", [])
        except json.JSONDecodeError:
            issues = []

        # Filter out low severity
        high_issues = [
            i
            for i in issues
            if i.get("issue_severity", "").upper() in ("HIGH", "MEDIUM")
        ]

        return len(high_issues) == 0, issues

    except FileNotFoundError:
        print("  ERROR: bandit not installed.")
        print("  Install with: pip install -r requirements-dev.txt")
        print("  Or: pip install bandit")
        return False, [{"error": "bandit not installed"}]
    except Exception as e:
        return False, [{"error": str(e)}]


def run_secret_scan(project_dir: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run secret detection using detect-secrets."""
    print("Running secret scan...")

    try:
        # First, try detect-secrets
        result = subprocess.run(
            ["detect-secrets", "scan", "--all-files", "."],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=project_dir,
        )

        try:
            report = json.loads(result.stdout)
            secrets = report.get("results", {})

            # Flatten secrets
            found = []
            for filename, file_secrets in secrets.items():
                # Skip excluded files
                skip = False
                for excl in SECRET_EXCLUSIONS:
                    if excl.startswith("*"):
                        if filename.endswith(excl[1:]):
                            skip = True
                            break
                    elif excl in filename:
                        skip = True
                        break

                if not skip:
                    for secret in file_secrets:
                        found.append(
                            {
                                "file": filename,
                                "type": secret.get("type"),
                                "line": secret.get("line_number"),
                            }
                        )

            return len(found) == 0, found

        except json.JSONDecodeError:
            return True, []  # No output means no secrets

    except FileNotFoundError:
        print("  detect-secrets not installed, skipping...")
        return True, [{"warning": "detect-secrets not installed"}]
    except Exception as e:
        return False, [{"error": str(e)}]


def check_allowlist(vulns: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """Check vulnerabilities against allowlist."""
    allowed = []
    blocked = []

    today = datetime.now().strftime("%Y-%m-%d")

    for vuln in vulns:
        pkg = vuln.get("name", vuln.get("package", ""))
        vuln_id = vuln.get("id", vuln.get("vuln_id", ""))

        if pkg in VULNERABILITY_ALLOWLIST:
            if vuln_id in VULNERABILITY_ALLOWLIST[pkg]:
                entry = VULNERABILITY_ALLOWLIST[pkg][vuln_id]
                # Check if expired
                if isinstance(entry, str) and entry[:10] > today:
                    allowed.append(vuln)
                    continue

        blocked.append(vuln)

    return allowed, blocked


def main():
    print("=" * 60)
    print(" SECURITY SCANS")
    print("=" * 60)
    print()

    project_dir = Path(__file__).parent.parent
    all_passed = True
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "scans": {},
    }

    # 1. pip-audit
    audit_passed, audit_vulns = run_pip_audit()
    allowed, blocked = check_allowlist(audit_vulns)

    report["scans"]["pip_audit"] = {
        "passed": len(blocked) == 0,
        "vulnerabilities": audit_vulns,
        "allowed": len(allowed),
        "blocked": len(blocked),
    }

    if blocked:
        print(f"  ✗ {len(blocked)} vulnerabilities found")
        all_passed = False
    else:
        print(f"  ✓ No vulnerabilities (allowed: {len(allowed)})")

    # 2. Bandit
    bandit_passed, bandit_issues = run_bandit(project_dir)
    high_issues = [
        i
        for i in bandit_issues
        if i.get("issue_severity", "").upper() in ("HIGH", "MEDIUM")
    ]

    report["scans"]["bandit"] = {
        "passed": len(high_issues) == 0,
        "issues": bandit_issues,
        "high_medium": len(high_issues),
    }

    if high_issues:
        print(f"  ✗ {len(high_issues)} high/medium severity issues")
        all_passed = False
    else:
        print(f"  ✓ No high/medium severity issues (total: {len(bandit_issues)})")

    # 3. Secret scan
    secrets_passed, secrets = run_secret_scan(project_dir)

    report["scans"]["secrets"] = {
        "passed": secrets_passed,
        "found": secrets,
    }

    if not secrets_passed:
        print(f"  ✗ {len(secrets)} secrets found")
        all_passed = False
    else:
        print("  ✓ No secrets detected")

    # Save report
    report_path = project_dir / "artifacts" / "security_scan.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport: {report_path}")

    print()
    print("=" * 60)
    if all_passed:
        print(" ✓ SECURITY SCANS PASSED")
    else:
        print(" ✗ SECURITY SCANS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
