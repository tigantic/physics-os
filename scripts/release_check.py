#!/usr/bin/env python
"""
Q) Release Check Script
=======================

Validates release readiness before tagging.

Usage:
    python scripts/release_check.py

Pass Criteria:
    - CHANGELOG updated with new version
    - Version bumped in pyproject.toml
    - CITATION.cff updated
    - Zenodo metadata valid
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def check_pyproject(project_root: Path) -> Tuple[bool, str, List[str]]:
    """Check pyproject.toml for version."""
    pyproject_path = project_root / 'pyproject.toml'
    issues = []
    version = None
    
    if not pyproject_path.exists():
        return False, '', ['pyproject.toml not found']
    
    try:
        content = pyproject_path.read_text()
        
        # Extract version
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            version = match.group(1)
        else:
            issues.append('Version not found in pyproject.toml')
        
        # Check required fields
        required = ['name', 'version', 'description', 'authors']
        for field in required:
            if f'{field}' not in content:
                issues.append(f'Missing required field: {field}')
        
    except Exception as e:
        issues.append(f'Error reading pyproject.toml: {e}')
    
    return len(issues) == 0, version or '', issues


def check_changelog(project_root: Path, version: str) -> Tuple[bool, List[str]]:
    """Check CHANGELOG.md for version entry."""
    changelog_path = project_root / 'CHANGELOG.md'
    issues = []
    
    if not changelog_path.exists():
        return False, ['CHANGELOG.md not found']
    
    try:
        content = changelog_path.read_text()
        
        # Check for version entry
        if version and version not in content:
            issues.append(f'Version {version} not found in CHANGELOG.md')
        
        # Check for Unreleased section
        if '[Unreleased]' in content:
            # Check if there are changes under Unreleased
            unreleased_section = content.split('[Unreleased]')[1].split('\n## ')[0]
            if unreleased_section.strip() and len(unreleased_section.strip()) > 50:
                issues.append('Unreleased section has content - should be moved to version section')
        
        # Check date format
        today = datetime.now().strftime('%Y-%m-%d')
        if version:
            # Look for version with recent date
            pattern = rf'\[{re.escape(version)}\].*{today[:7]}'
            if not re.search(pattern, content):
                issues.append(f'Version {version} may not have recent date')
        
    except Exception as e:
        issues.append(f'Error reading CHANGELOG.md: {e}')
    
    return len(issues) == 0, issues


def check_citation(project_root: Path, version: str) -> Tuple[bool, List[str]]:
    """Check CITATION.cff for valid format and version."""
    citation_path = project_root / 'CITATION.cff'
    issues = []
    
    if not citation_path.exists():
        return False, ['CITATION.cff not found']
    
    try:
        content = citation_path.read_text()
        
        # Check required fields
        required = ['cff-version', 'title', 'authors', 'version', 'date-released']
        for field in required:
            if f'{field}:' not in content:
                issues.append(f'Missing required field: {field}')
        
        # Check version matches
        match = re.search(r'version:\s*["\']?([^\s"\']+)', content)
        if match:
            cff_version = match.group(1)
            if version and cff_version != version:
                issues.append(f'Version mismatch: CITATION.cff has {cff_version}, expected {version}')
        
        # Check date
        match = re.search(r'date-released:\s*["\']?(\d{4}-\d{2}-\d{2})', content)
        if match:
            date_str = match.group(1)
            release_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_old = (datetime.now() - release_date).days
            if days_old > 30:
                issues.append(f'Release date is {days_old} days old - may need update')
        
    except Exception as e:
        issues.append(f'Error reading CITATION.cff: {e}')
    
    return len(issues) == 0, issues


def check_zenodo(project_root: Path) -> Tuple[bool, List[str]]:
    """Check .zenodo.json for valid metadata."""
    zenodo_path = project_root / '.zenodo.json'
    issues = []
    
    if not zenodo_path.exists():
        # Not required, but recommended
        return True, ['Warning: .zenodo.json not found (optional)']
    
    try:
        with open(zenodo_path) as f:
            data = json.load(f)
        
        # Check required fields
        required = ['title', 'creators', 'description', 'license']
        for field in required:
            if field not in data:
                issues.append(f'Missing required field: {field}')
        
        # Check creators format
        if 'creators' in data:
            for i, creator in enumerate(data['creators']):
                if 'name' not in creator:
                    issues.append(f'Creator {i} missing name')
        
    except json.JSONDecodeError as e:
        issues.append(f'Invalid JSON: {e}')
    except Exception as e:
        issues.append(f'Error reading .zenodo.json: {e}')
    
    return len(issues) == 0, issues


def check_git_status(project_root: Path) -> Tuple[bool, List[str]]:
    """Check git status for uncommitted changes."""
    import subprocess
    issues = []
    
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=project_root
        )
        
        if result.stdout.strip():
            issues.append('Uncommitted changes detected')
            # Show first few
            changes = result.stdout.strip().split('\n')[:5]
            for change in changes:
                issues.append(f'  {change}')
        
    except Exception as e:
        issues.append(f'Error checking git status: {e}')
    
    return len(issues) == 0, issues


def check_tests_pass(project_root: Path) -> Tuple[bool, List[str]]:
    """Check if tests pass."""
    import subprocess
    issues = []
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=no'],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=project_root
        )
        
        if result.returncode != 0:
            issues.append('Tests failed')
            # Extract summary
            for line in result.stdout.split('\n'):
                if 'passed' in line or 'failed' in line or 'error' in line:
                    issues.append(f'  {line.strip()}')
        
    except Exception as e:
        issues.append(f'Error running tests: {e}')
    
    return len(issues) == 0, issues


def main():
    print("=" * 60)
    print(" RELEASE CHECK")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent.parent
    all_passed = True
    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'checks': {},
    }
    
    # 1. Check pyproject.toml
    print("Checking pyproject.toml...")
    passed, version, issues = check_pyproject(project_root)
    report['checks']['pyproject'] = {'passed': passed, 'version': version, 'issues': issues}
    print(f"  Version: {version}")
    if not passed:
        all_passed = False
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ Valid")
    
    # 2. Check CHANGELOG
    print("\nChecking CHANGELOG.md...")
    passed, issues = check_changelog(project_root, version)
    report['checks']['changelog'] = {'passed': passed, 'issues': issues}
    if not passed:
        all_passed = False
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ Updated")
    
    # 3. Check CITATION.cff
    print("\nChecking CITATION.cff...")
    passed, issues = check_citation(project_root, version)
    report['checks']['citation'] = {'passed': passed, 'issues': issues}
    if not passed:
        all_passed = False
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ Valid")
    
    # 4. Check Zenodo
    print("\nChecking .zenodo.json...")
    passed, issues = check_zenodo(project_root)
    report['checks']['zenodo'] = {'passed': passed, 'issues': issues}
    if not passed:
        for issue in issues:
            if 'Warning' in issue:
                print(f"  ⚠ {issue}")
            else:
                all_passed = False
                print(f"  ✗ {issue}")
    else:
        print("  ✓ Valid")
    
    # 5. Check git status
    print("\nChecking git status...")
    passed, issues = check_git_status(project_root)
    report['checks']['git'] = {'passed': passed, 'issues': issues}
    if not passed:
        all_passed = False
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ Clean")
    
    # 6. Check tests (optional, slow)
    print("\nChecking tests...")
    passed, issues = check_tests_pass(project_root)
    report['checks']['tests'] = {'passed': passed, 'issues': issues}
    if not passed:
        all_passed = False
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ Passing")
    
    # Save report
    report_path = project_root / 'artifacts' / 'release_check.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print()
    print("=" * 60)
    if all_passed:
        print(f" ✓ READY TO TAG: v{version}")
        print()
        print(f"  git tag -a v{version} -m 'Release {version}'")
        print(f"  git push origin v{version}")
    else:
        print(" ✗ NOT READY FOR RELEASE")
        print()
        print("  Fix the issues above before tagging.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
