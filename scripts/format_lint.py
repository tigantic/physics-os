#!/usr/bin/env python
"""
C) Format & Lint Gates
======================

Runs black, isort, ruff checks on the codebase.

Usage:
    python scripts/format_lint.py [--fix]

Pass Criteria:
    - black: code formatted correctly
    - isort: imports sorted correctly
    - ruff: no linting errors (or only allowed)
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


TARGETS = ['tensornet', 'tests', 'benchmarks', 'scripts', 'proofs']


def run_black(fix: bool = False) -> Tuple[bool, List[str]]:
    """Run black formatter."""
    cmd = [sys.executable, '-m', 'black']
    
    if not fix:
        cmd.append('--check')
    
    cmd.extend(TARGETS)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        issues = []
        if result.returncode != 0:
            for line in result.stdout.split('\n'):
                if 'would reformat' in line or 'reformatted' in line:
                    issues.append(line.strip())
        
        return result.returncode == 0, issues
        
    except FileNotFoundError:
        return False, ["black not installed"]
    except Exception as e:
        return False, [str(e)]


def run_isort(fix: bool = False) -> Tuple[bool, List[str]]:
    """Run isort import sorter."""
    cmd = [sys.executable, '-m', 'isort']
    
    if not fix:
        cmd.append('--check-only')
    
    cmd.extend(TARGETS)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        issues = []
        if result.returncode != 0:
            for line in result.stderr.split('\n'):
                if line.strip():
                    issues.append(line.strip())
        
        return result.returncode == 0, issues
        
    except FileNotFoundError:
        return False, ["isort not installed"]
    except Exception as e:
        return False, [str(e)]


def run_ruff(fix: bool = False) -> Tuple[bool, List[str]]:
    """Run ruff linter."""
    cmd = [sys.executable, '-m', 'ruff', 'check']
    
    if fix:
        cmd.append('--fix')
    
    cmd.extend(TARGETS)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        issues = []
        for line in result.stdout.split('\n'):
            if line.strip() and ':' in line:
                issues.append(line.strip())
        
        return result.returncode == 0, issues
        
    except FileNotFoundError:
        return False, ["ruff not installed"]
    except Exception as e:
        return False, [str(e)]


def main():
    parser = argparse.ArgumentParser(description="Format & Lint Gates")
    parser.add_argument('--fix', action='store_true',
                       help="Apply fixes (format and auto-fix lint issues)")
    parser.add_argument('--json', type=str,
                       help="Output JSON report path")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" FORMAT & LINT GATES")
    print("=" * 60)
    print()
    
    mode = "fix" if args.fix else "check"
    print(f"Mode: {mode}")
    print()
    
    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'mode': mode,
        'checks': {},
    }
    
    all_passed = True
    
    # Black
    print("Running black...", end=' ')
    passed, issues = run_black(args.fix)
    report['checks']['black'] = {'passed': passed, 'issues': issues}
    
    if passed:
        print("✓ PASS")
    else:
        print(f"✗ FAIL ({len(issues)} issues)")
        for issue in issues[:5]:
            print(f"  - {issue}")
        all_passed = False
    
    # isort
    print("Running isort...", end=' ')
    passed, issues = run_isort(args.fix)
    report['checks']['isort'] = {'passed': passed, 'issues': issues}
    
    if passed:
        print("✓ PASS")
    else:
        print(f"✗ FAIL ({len(issues)} issues)")
        for issue in issues[:5]:
            print(f"  - {issue}")
        all_passed = False
    
    # ruff
    print("Running ruff...", end=' ')
    passed, issues = run_ruff(args.fix)
    report['checks']['ruff'] = {'passed': passed, 'issues': issues}
    
    if passed:
        print("✓ PASS")
    else:
        print(f"✗ FAIL ({len(issues)} issues)")
        for issue in issues[:5]:
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")
        all_passed = False
    
    # Summary
    print()
    print("=" * 60)
    if all_passed:
        print(" ✓ ALL FORMAT/LINT CHECKS PASSED")
    else:
        print(" ✗ SOME CHECKS FAILED")
        if not args.fix:
            print()
            print(" Run with --fix to auto-apply fixes:")
            print("   python scripts/format_lint.py --fix")
    print("=" * 60)
    
    # Save report
    if args.json:
        output_path = Path(args.json)
    else:
        output_path = Path(__file__).parent.parent / 'artifacts' / 'format_lint.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
