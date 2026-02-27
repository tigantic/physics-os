#!/usr/bin/env python3
"""DOMINION Validation Suite Runner

Execute the complete Validation Doctrine across all deployments.

Usage:
    python run_validation.py              # Run all tests
    python run_validation.py --deployment 1    # Run Deployment 1 only
    python run_validation.py --crucible        # Run Crucible only
    python run_validation.py --report          # Generate JSON report
    
Author: HyperTensor Physics Laboratory
Copyright (c) 2025 TiganticLabz. All Rights Reserved.
"""

import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

TESTS_DIR = Path(__file__).parent
REPORT_DIR = TESTS_DIR.parent / "validation_reports"

DEPLOYMENTS = {
    1: {
        "name": "The Sovereign Core",
        "file": "test_deployment_1.py",
        "tests": ["Latency Audit", "Frame Budget", "Cold Start"]
    },
    2: {
        "name": "The Comfort Engine", 
        "file": "test_deployment_2.py",
        "tests": ["ASHRAE 55", "Inverse Design", "BC Integrity"]
    },
    3: {
        "name": "Critical Systems Suite",
        "file": "test_deployment_3.py",
        "tests": ["Alpert Correlation", "ASET/RSET", "Kill Switch"]
    },
}

CRUCIBLE = {
    "name": "The Crucible",
    "file": "test_crucible.py",
    "tests": ["Dirty Geometry", "Long Run", "Network Cut"]
}


# ============================================================================
# BANNER
# ============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗  ██████╗ ███╗   ███╗██╗███╗   ██╗██╗ ██████╗ ███╗   ██╗          ║
║     ██╔══██╗██╔═══██╗████╗ ████║██║████╗  ██║██║██╔═══██╗████╗  ██║          ║
║     ██║  ██║██║   ██║██╔████╔██║██║██╔██╗ ██║██║██║   ██║██╔██╗ ██║          ║
║     ██║  ██║██║   ██║██║╚██╔╝██║██║██║╚██╗██║██║██║   ██║██║╚██╗██║          ║
║     ██████╔╝╚██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██║╚██████╔╝██║ ╚████║          ║
║     ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝          ║
║                                                                              ║
║                    V A L I D A T I O N   D O C T R I N E                     ║
║                                                                              ║
║     "The engineering integrity of a Type I Civilization Engine               ║
║      is not assumed—it is PROVEN."                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# RUNNER
# ============================================================================

def run_pytest(test_file: str, verbose: bool = True) -> tuple[int, str, str]:
    """Run pytest on a test file and capture results."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(TESTS_DIR / test_file),
        "-v" if verbose else "-q",
        "--tb=short",
        "--no-header",
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(TESTS_DIR.parent)
    )
    
    return result.returncode, result.stdout, result.stderr


def run_deployment(deployment_num: int, verbose: bool = True) -> dict:
    """Run validation for a specific deployment."""
    if deployment_num not in DEPLOYMENTS:
        print(f"Unknown deployment: {deployment_num}")
        return {"passed": False, "error": "Unknown deployment"}
    
    config = DEPLOYMENTS[deployment_num]
    print(f"\n{'='*60}")
    print(f"DEPLOYMENT {deployment_num}: {config['name'].upper()}")
    print(f"{'='*60}")
    print(f"Tests: {', '.join(config['tests'])}")
    print()
    
    t0 = time.time()
    returncode, stdout, stderr = run_pytest(config['file'], verbose)
    duration = time.time() - t0
    
    print(stdout)
    if stderr and returncode != 0:
        print("STDERR:", stderr)
    
    passed = returncode == 0
    
    return {
        "deployment": deployment_num,
        "name": config['name'],
        "passed": passed,
        "duration_s": round(duration, 2),
        "returncode": returncode,
    }


def run_crucible(verbose: bool = True) -> dict:
    """Run The Crucible final stress tests."""
    print(f"\n{'='*60}")
    print(f"THE CRUCIBLE: FINAL CERTIFICATION")
    print(f"{'='*60}")
    print(f"Tests: {', '.join(CRUCIBLE['tests'])}")
    print()
    
    t0 = time.time()
    returncode, stdout, stderr = run_pytest(CRUCIBLE['file'], verbose)
    duration = time.time() - t0
    
    print(stdout)
    if stderr and returncode != 0:
        print("STDERR:", stderr)
    
    passed = returncode == 0
    
    return {
        "name": "The Crucible",
        "passed": passed,
        "duration_s": round(duration, 2),
        "returncode": returncode,
    }


def run_full_suite(verbose: bool = True) -> dict:
    """Run the complete validation doctrine."""
    print(BANNER)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "deployments": {},
        "crucible": None,
        "certified": False,
    }
    
    # Run all deployments
    for dep_num in sorted(DEPLOYMENTS.keys()):
        dep_result = run_deployment(dep_num, verbose)
        results["deployments"][dep_num] = dep_result
    
    # Run crucible
    results["crucible"] = run_crucible(verbose)
    
    # Determine overall certification
    all_deployments_passed = all(
        d["passed"] for d in results["deployments"].values()
    )
    crucible_passed = results["crucible"]["passed"]
    results["certified"] = all_deployments_passed and crucible_passed
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: dict):
    """Print validation summary."""
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for dep_num, dep_result in results["deployments"].items():
        status = "✓ PASS" if dep_result["passed"] else "✗ FAIL"
        print(f"  Deployment {dep_num}: {status} ({dep_result['duration_s']}s)")
    
    if results["crucible"]:
        status = "✓ PASS" if results["crucible"]["passed"] else "✗ FAIL"
        print(f"  The Crucible:  {status} ({results['crucible']['duration_s']}s)")
    
    print(f"\n{'='*60}")
    
    if results["certified"]:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║               ██████╗███████╗██████╗ ████████╗               ║
║              ██╔════╝██╔════╝██╔══██╗╚══██╔══╝               ║
║              ██║     █████╗  ██████╔╝   ██║                  ║
║              ██║     ██╔══╝  ██╔══██╗   ██║                  ║
║              ╚██████╗███████╗██║  ██║   ██║                  ║
║               ╚═════╝╚══════╝╚═╝  ╚═╝   ╚═╝                  ║
║                                                              ║
║                    C E R T I F I E D                         ║
║                                                              ║
║     "If it survives the Crucible, it ships."                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ██╗    ██╗ █████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗║
║     ██║    ██║██╔══██╗██╔══██╗████╗  ██║██║████╗  ██║██╔════╝║
║     ██║ █╗ ██║███████║██████╔╝██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
║     ██║███╗██║██╔══██║██╔══██╗██║╚██╗██║██║██║╚██╗██║██║   ██║
║     ╚███╔███╔╝██║  ██║██║  ██║██║ ╚████║██║██║ ╚████║╚██████╔╝
║      ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝║
║                                                              ║
║              VALIDATION FAILED - DO NOT SHIP                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)


def save_report(results: dict, output_path: Path = None):
    """Save validation report to JSON."""
    if output_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORT_DIR / f"validation_report_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")
    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DOMINION Validation Doctrine Runner"
    )
    parser.add_argument(
        "--deployment", "-d",
        type=int,
        choices=[1, 2, 3],
        help="Run specific deployment tests only"
    )
    parser.add_argument(
        "--crucible", "-c",
        action="store_true",
        help="Run The Crucible tests only"
    )
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Save JSON report"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (less verbose output)"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.deployment:
        results = run_deployment(args.deployment, verbose)
        results = {"deployments": {args.deployment: results}, "certified": results["passed"]}
    elif args.crucible:
        results = {"crucible": run_crucible(verbose)}
        results["certified"] = results["crucible"]["passed"]
    else:
        results = run_full_suite(verbose)
    
    if args.report:
        save_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if results.get("certified", False) else 1)


if __name__ == "__main__":
    main()
