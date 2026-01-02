#!/usr/bin/env python3
"""
V&V Regression Detection Script
================================

Compares current V&V test results against a baseline to detect regressions.

Usage:
    python detect_vv_regression.py --baseline .vv_baseline.json --current artifacts/ --output report.json

Constitution Compliance: Article IV.1 (Verification), Phase 3 Automation
Tags: [V&V] [CI/CD] [REGRESSION]
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    classname: str
    status: str  # pass, fail, skip, error
    time: float
    message: Optional[str] = None


@dataclass
class CategoryResults:
    """Results for a test category."""
    passed: int
    failed: int
    skipped: int
    errors: int
    tests: list


@dataclass
class RegressionReport:
    """Full regression report."""
    timestamp: str
    regression_detected: bool
    new_failures: list
    fixed_tests: list
    performance_regressions: list
    summary: dict


def parse_junit_xml(xml_path: Path) -> CategoryResults:
    """Parse JUnit XML results."""
    results = CategoryResults(
        passed=0,
        failed=0,
        skipped=0,
        errors=0,
        tests=[]
    )
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for testsuite in root.iter('testsuite'):
            for testcase in testsuite.iter('testcase'):
                test = TestResult(
                    name=testcase.get('name', ''),
                    classname=testcase.get('classname', ''),
                    status='pass',
                    time=float(testcase.get('time', 0)),
                )
                
                failure = testcase.find('failure')
                error = testcase.find('error')
                skipped = testcase.find('skipped')
                
                if failure is not None:
                    test.status = 'fail'
                    test.message = failure.get('message', '')
                    results.failed += 1
                elif error is not None:
                    test.status = 'error'
                    test.message = error.get('message', '')
                    results.errors += 1
                elif skipped is not None:
                    test.status = 'skip'
                    results.skipped += 1
                else:
                    results.passed += 1
                
                results.tests.append(asdict(test))
        
    except Exception as e:
        print(f"Warning: Could not parse {xml_path}: {e}")
    
    return results


def load_baseline(baseline_path: Path) -> dict:
    """Load baseline results."""
    if not baseline_path.exists():
        print(f"No baseline found at {baseline_path}, creating new baseline")
        return {}
    
    try:
        return json.loads(baseline_path.read_text())
    except Exception as e:
        print(f"Warning: Could not load baseline: {e}")
        return {}


def collect_current_results(artifacts_dir: Path) -> dict:
    """Collect all current test results from artifacts."""
    results = {
        'mms': CategoryResults(0, 0, 0, 0, []),
        'benchmark': CategoryResults(0, 0, 0, 0, []),
        'conservation': CategoryResults(0, 0, 0, 0, []),
    }
    
    for subdir in artifacts_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.name == 'mms_results.xml':
                    results['mms'] = parse_junit_xml(file)
                elif file.name == 'benchmark_results.xml':
                    results['benchmark'] = parse_junit_xml(file)
                elif file.name == 'conservation_results.xml':
                    results['conservation'] = parse_junit_xml(file)
    
    return {k: asdict(v) for k, v in results.items()}


def detect_regressions(baseline: dict, current: dict) -> RegressionReport:
    """Compare current results against baseline to detect regressions."""
    new_failures = []
    fixed_tests = []
    performance_regressions = []
    
    # Build lookup of baseline test results
    baseline_tests = {}
    for category, cat_results in baseline.items():
        if isinstance(cat_results, dict) and 'tests' in cat_results:
            for test in cat_results['tests']:
                key = f"{test['classname']}::{test['name']}"
                baseline_tests[key] = test
    
    # Compare current tests
    for category, cat_results in current.items():
        if isinstance(cat_results, dict) and 'tests' in cat_results:
            for test in cat_results['tests']:
                key = f"{test['classname']}::{test['name']}"
                
                if key in baseline_tests:
                    baseline_test = baseline_tests[key]
                    
                    # New failure (was passing, now failing)
                    if baseline_test['status'] == 'pass' and test['status'] in ('fail', 'error'):
                        new_failures.append({
                            'test': key,
                            'category': category,
                            'previous_status': baseline_test['status'],
                            'current_status': test['status'],
                            'message': test.get('message', ''),
                        })
                    
                    # Fixed test (was failing, now passing)
                    if baseline_test['status'] in ('fail', 'error') and test['status'] == 'pass':
                        fixed_tests.append({
                            'test': key,
                            'category': category,
                        })
                    
                    # Performance regression (>50% slower)
                    if baseline_test['time'] > 0 and test['time'] > baseline_test['time'] * 1.5:
                        performance_regressions.append({
                            'test': key,
                            'category': category,
                            'baseline_time': baseline_test['time'],
                            'current_time': test['time'],
                            'slowdown': test['time'] / baseline_test['time'],
                        })
    
    # Calculate summary
    total_current_passed = sum(c.get('passed', 0) for c in current.values() if isinstance(c, dict))
    total_current_failed = sum(c.get('failed', 0) for c in current.values() if isinstance(c, dict))
    total_baseline_passed = sum(c.get('passed', 0) for c in baseline.values() if isinstance(c, dict))
    total_baseline_failed = sum(c.get('failed', 0) for c in baseline.values() if isinstance(c, dict))
    
    regression_detected = len(new_failures) > 0
    
    return RegressionReport(
        timestamp=datetime.utcnow().isoformat() + 'Z',
        regression_detected=regression_detected,
        new_failures=new_failures,
        fixed_tests=fixed_tests,
        performance_regressions=performance_regressions,
        summary={
            'current_passed': total_current_passed,
            'current_failed': total_current_failed,
            'baseline_passed': total_baseline_passed,
            'baseline_failed': total_baseline_failed,
            'new_failures_count': len(new_failures),
            'fixed_tests_count': len(fixed_tests),
            'performance_regressions_count': len(performance_regressions),
        }
    )


def main():
    parser = argparse.ArgumentParser(description='Detect V&V Regressions')
    parser.add_argument('--baseline', type=Path, default=Path('.vv_baseline.json'), help='Baseline file')
    parser.add_argument('--current', type=Path, default=Path('artifacts'), help='Current artifacts directory')
    parser.add_argument('--output', type=Path, default=Path('regression_report.json'), help='Output report file')
    parser.add_argument('--update-baseline', action='store_true', help='Update baseline with current results')
    args = parser.parse_args()
    
    print("V&V Regression Detection")
    print("=" * 40)
    
    # Load baseline
    baseline = load_baseline(args.baseline)
    print(f"Baseline: {len(baseline)} categories")
    
    # Collect current results
    current = collect_current_results(args.current)
    print(f"Current: {len(current)} categories")
    
    # Detect regressions
    report = detect_regressions(baseline, current)
    
    # Print summary
    print("\n" + "=" * 40)
    print("REGRESSION REPORT")
    print("=" * 40)
    
    if report.regression_detected:
        print("⚠️  REGRESSION DETECTED!")
        print(f"\nNew Failures ({len(report.new_failures)}):")
        for failure in report.new_failures:
            print(f"  ❌ {failure['test']}")
            print(f"     {failure['previous_status']} → {failure['current_status']}")
    else:
        print("✅ No regressions detected")
    
    if report.fixed_tests:
        print(f"\nFixed Tests ({len(report.fixed_tests)}):")
        for fix in report.fixed_tests:
            print(f"  ✅ {fix['test']}")
    
    if report.performance_regressions:
        print(f"\nPerformance Regressions ({len(report.performance_regressions)}):")
        for perf in report.performance_regressions:
            print(f"  ⏱️  {perf['test']}: {perf['slowdown']:.2f}x slower")
    
    print(f"\nSummary:")
    print(f"  Current: {report.summary['current_passed']} passed, {report.summary['current_failed']} failed")
    if baseline:
        print(f"  Baseline: {report.summary['baseline_passed']} passed, {report.summary['baseline_failed']} failed")
    
    # Write report
    args.output.write_text(json.dumps(asdict(report), indent=2))
    print(f"\nReport written to: {args.output}")
    
    # Optionally update baseline
    if args.update_baseline:
        args.baseline.write_text(json.dumps(current, indent=2))
        print(f"Baseline updated: {args.baseline}")
    
    # Exit with error code if regression detected
    if report.regression_detected:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
