#!/usr/bin/env python3
"""
ORACLE Command Line Interface.

Usage:
    oracle hunt <source_file> [--output <dir>] [--verbose]
    oracle hunt --address <addr> [--chain <chain>] [--fork] [--output <dir>]
    oracle scan <source_file>  # Quick scan without full verification
    
Examples:
    oracle hunt ./contracts/Vault.sol --output ./reports
    oracle hunt --address 0xc3d688B66703497DAA19211EEdff47f25384cdc3 --chain ethereum --fork
    oracle scan ./contracts/Vault.sol
    
Environment Variables:
    ETH_RPC_URL       - Ethereum RPC endpoint (required for --fork)
    ETHERSCAN_API_KEY - Etherscan API key (required for --address)
    ANTHROPIC_API_KEY - Claude API key (optional, enables LLM analysis)
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ORACLE: Offensive Reasoning and Assumption-Challenging Logic Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hunt from source file
  oracle hunt ./contracts/Vault.sol
  oracle hunt ./contracts/Vault.sol --output ./reports --verbose
  
  # Hunt live contract (requires ETHERSCAN_API_KEY)
  oracle hunt --address 0xc3d688B66703497DAA19211EEdff47f25384cdc3 --chain ethereum
  
  # Hunt with mainnet fork verification (requires ETH_RPC_URL)
  oracle hunt --address 0xc3d688B66703497DAA19211EEdff47f25384cdc3 --fork --min-profit 0.1
  
  # Quick scan
  oracle scan ./contracts/Vault.sol
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Hunt command
    hunt_parser = subparsers.add_parser("hunt", help="Hunt for vulnerabilities")
    hunt_parser.add_argument("source", nargs="?", help="Path to Solidity source file")
    hunt_parser.add_argument("--address", "-a", help="Contract address to analyze")
    hunt_parser.add_argument("--chain", "-c", default="ethereum", 
                            help="Chain (ethereum, arbitrum, optimism, base, polygon)")
    hunt_parser.add_argument("--output", "-o", help="Output directory for reports")
    hunt_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    hunt_parser.add_argument("--fork", "-f", action="store_true", 
                            help="Verify exploits on mainnet fork (requires ETH_RPC_URL)")
    hunt_parser.add_argument("--min-confidence", type=float, default=0.5, 
                            help="Minimum confidence threshold (default: 0.5)")
    hunt_parser.add_argument("--min-profit", type=float, default=0.01,
                            help="Minimum profit in ETH to report (default: 0.01)")
    
    # Scan command (quick)
    scan_parser = subparsers.add_parser("scan", help="Quick scan without verification")
    scan_parser.add_argument("source", help="Path to Solidity source file")
    
    # Circom hunt command (NEW)
    circom_parser = subparsers.add_parser("circom", help="Hunt Circom ZK circuits for vulnerabilities")
    circom_parser.add_argument("source", help="Path to .circom file or directory")
    circom_parser.add_argument("--focus", "-f", default="under-constrained",
                               choices=["under-constrained", "all"],
                               help="Vulnerability focus (default: under-constrained)")
    circom_parser.add_argument("--output", "-o", help="Output file for report")
    circom_parser.add_argument("--json", action="store_true", help="Output as JSON")
    circom_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Import ORACLE
    try:
        from tensornet.infra.oracle import ORACLE
    except ImportError:
        # Try relative import for development
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from tensornet.infra.oracle import ORACLE
    
    if args.command == "hunt":
        run_hunt(args)
    elif args.command == "scan":
        run_scan(args)
    elif args.command == "circom":
        run_circom_hunt(args)


def run_hunt(args):
    """Run full vulnerability hunt."""
    from tensornet.infra.oracle import ORACLE
    
    # Check inputs
    if not args.source and not args.address:
        print("Error: Must provide source file or --address")
        sys.exit(1)
    
    # Check for mainnet fork requirements
    if args.fork:
        if not os.environ.get("ETH_RPC_URL"):
            print("Error: --fork requires ETH_RPC_URL environment variable")
            print("  Example: export ETH_RPC_URL='https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY'")
            sys.exit(1)
    
    if args.address and not os.environ.get("ETHERSCAN_API_KEY"):
        print("Warning: ETHERSCAN_API_KEY not set - may fail to fetch source")
    
    # Initialize ORACLE
    oracle = ORACLE()
    
    # Run hunt
    try:
        if args.address:
            # Mainnet hunt
            if args.fork:
                print("\n🎯 ORACLE: Mainnet Fork Hunt Mode")
                print("=" * 50)
                result = oracle.hunt_address(
                    address=args.address,
                    chain=args.chain,
                    fork_verify=True,
                    min_profit_eth=args.min_profit,
                    verbose=args.verbose if hasattr(args, 'verbose') else True,
                )
            else:
                result = oracle.hunt(
                    address=args.address,
                    chain=args.chain,
                    min_confidence=args.min_confidence,
                    verbose=args.verbose if hasattr(args, 'verbose') else True,
                )
        else:
            result = oracle.hunt(
                file_path=args.source,
                min_confidence=args.min_confidence,
                fork_verify=args.fork,
                min_profit_eth=args.min_profit if hasattr(args, 'min_profit') else 0.0,
                verbose=args.verbose if hasattr(args, 'verbose') else True,
            )
    except Exception as e:
        print(f"Error during hunt: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate reports if exploits found
    if result.verified_exploits and args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating reports in {output_dir}/")
        
        for i, exploit in enumerate(result.verified_exploits):
            report = oracle.generate_report(exploit)
            
            # Save markdown report
            report_path = output_dir / f"report_{i+1}_{exploit.scenario.name.replace(' ', '_')[:30]}.md"
            with open(report_path, "w") as f:
                f.write(report.to_immunefi_markdown())
            print(f"  Saved: {report_path}")
            
            # Save Foundry test
            if exploit.foundry_test:
                test_path = output_dir / f"Exploit{i+1}Test.t.sol"
                with open(test_path, "w") as f:
                    f.write(exploit.foundry_test)
                print(f"  Saved: {test_path}")
    
    # Summary
    print("\n" + "=" * 40)
    if result.verified_exploits:
        total_profit = sum(e.scenario.expected_profit for e in result.verified_exploits)
        print(f"🎯 Found {len(result.verified_exploits)} verified exploit(s)")
        print(f"💰 Total potential profit: {total_profit / 10**18:.2f} ETH")
    else:
        print("✓ No verified exploits found")
        if result.challenges:
            print(f"  ({len(result.challenges)} potential issues identified)")


def run_scan(args):
    """Run quick scan."""
    from tensornet.infra.oracle import ORACLE
    
    # Read source
    with open(args.source, "r") as f:
        source = f.read()
    
    # Quick scan
    oracle = ORACLE()
    result = oracle.quick_scan(source)
    
    print("=" * 40)
    print("ORACLE Quick Scan Results")
    print("=" * 40)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print(f"Contract: {result['contract']}")
    print(f"Type: {result['type']}")
    print(f"Functions: {result['functions']}")
    print(f"Assumptions Extracted: {result['assumptions']}")
    
    if result.get("high_risk"):
        print("\n⚠️  High-Risk Patterns:")
        for risk in result["high_risk"]:
            print(f"  - {risk}")
    else:
        print("\n✓ No obvious high-risk patterns")


def run_circom_hunt(args):
    """Hunt Circom circuits for vulnerabilities."""
    from tensornet.infra.oracle.parsing.circom_parser import hunt_circom
    import json
    
    print("\n" + "=" * 60)
    print("🔮 ORACLE: Circom Circuit Vulnerability Hunter")
    print("=" * 60)
    print(f"Target: {args.source}")
    print(f"Focus: {args.focus}")
    print()
    
    # Run the hunt
    findings = hunt_circom(args.source, focus=args.focus)
    
    # Output results
    if not findings:
        print("✅ No vulnerabilities found!")
        print("\nCircuits appear properly constrained.")
        return
    
    # Group by severity
    critical = [f for f in findings if f["severity"] == "CRITICAL"]
    high = [f for f in findings if f["severity"] == "HIGH"]
    medium = [f for f in findings if f["severity"] == "MEDIUM"]
    low = [f for f in findings if f["severity"] == "LOW"]
    info = [f for f in findings if f["severity"] == "INFO"]
    
    print(f"🔍 Found {len(findings)} potential issues:\n")
    print(f"   🚨 CRITICAL: {len(critical)}")
    print(f"   ⚠️  HIGH:     {len(high)}")
    print(f"   ⚡ MEDIUM:   {len(medium)}")
    print(f"   ℹ️  LOW:      {len(low)}")
    print(f"   📝 INFO:     {len(info)}")
    print()
    
    # Print exploitable findings
    exploitable = [f for f in findings if f.get("exploitable")]
    if exploitable:
        print("=" * 60)
        print("⚡ EXPLOITABLE VULNERABILITIES")
        print("=" * 60)
        
        for finding in exploitable:
            severity_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️"}.get(finding["severity"], "⚡")
            print(f"\n{severity_emoji} [{finding['severity']}] {finding['type'].upper()}")
            print(f"   File: {finding.get('file', 'unknown')}")
            print(f"   Template: {finding['template']}")
            print(f"   Signal: {finding['signal']} ({finding.get('signal_type', 'unknown')})")
            print(f"   Line: {finding['line']}")
            print(f"   Description: {finding['description']}")
            
            # Exploitation guidance
            if finding["type"] == "under_constrained":
                print("\n   💀 EXPLOITATION:")
                print("      The prover can set this signal to ANY value that satisfies")
                print("      the other constraints. This allows proof forgery if the signal")
                print("      affects the circuit's output or nullifier computation.")
            elif finding["type"] == "unconstrained_output":
                print("\n   💀 EXPLOITATION:")
                print("      OUTPUT signal has no constraints! The prover can claim ANY output")
                print("      value. This is a COMPLETE circuit break - proofs are meaningless.")
        
        print()
    
    # Print all findings if verbose
    if args.verbose:
        print("=" * 60)
        print("ALL FINDINGS")
        print("=" * 60)
        
        for finding in findings:
            severity_emoji = {
                "CRITICAL": "🚨",
                "HIGH": "⚠️",
                "MEDIUM": "⚡",
                "LOW": "ℹ️",
                "INFO": "📝"
            }.get(finding["severity"], "❓")
            
            print(f"\n{severity_emoji} [{finding['severity']}] {finding['type']}")
            print(f"   Template: {finding['template']}")
            print(f"   Signal: {finding['signal']}")
            print(f"   Line: {finding['line']}")
            print(f"   {finding['description']}")
    
    # Output to file if requested
    if args.output:
        output_path = Path(args.output)
        
        if args.json:
            with open(output_path, "w") as f:
                json.dump(findings, f, indent=2)
            print(f"\n📄 JSON report saved to: {output_path}")
        else:
            # Generate markdown report
            report = _generate_circom_report(findings, args.source)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"\n📄 Report saved to: {output_path}")
    
    # JSON to stdout if requested
    if args.json and not args.output:
        print("\n" + json.dumps(findings, indent=2))
    
    # Summary
    print("\n" + "=" * 60)
    if exploitable:
        print(f"🎯 {len(exploitable)} EXPLOITABLE vulnerabilities found!")
        print("   These allow proof forgery - submit to bug bounty!")
    else:
        print("✓ No directly exploitable issues found")
        print(f"  ({len(findings)} lower-severity items to review)")


def _generate_circom_report(findings: list, source: str) -> str:
    """Generate a markdown report for Circom findings."""
    exploitable = [f for f in findings if f.get("exploitable")]
    
    report = f"""# ORACLE Circom Circuit Vulnerability Report

**Target:** `{source}`
**Date:** {__import__('datetime').datetime.now().isoformat()[:10]}
**Focus:** Under-constrained signals

---

## Executive Summary

ORACLE analyzed the Circom circuits and found **{len(findings)}** potential issues,
of which **{len(exploitable)}** are exploitable.

| Severity | Count |
|----------|-------|
| 🚨 CRITICAL | {len([f for f in findings if f["severity"] == "CRITICAL"])} |
| ⚠️ HIGH | {len([f for f in findings if f["severity"] == "HIGH"])} |
| ⚡ MEDIUM | {len([f for f in findings if f["severity"] == "MEDIUM"])} |
| ℹ️ LOW | {len([f for f in findings if f["severity"] == "LOW"])} |
| 📝 INFO | {len([f for f in findings if f["severity"] == "INFO"])} |

---

## Exploitable Vulnerabilities

"""
    
    for i, finding in enumerate(exploitable, 1):
        report += f"""
### {i}. {finding['type'].replace('_', ' ').title()} in `{finding['template']}`

**Severity:** {finding['severity']}
**Signal:** `{finding['signal']}` ({finding.get('signal_type', 'unknown')})
**Line:** {finding['line']}
**File:** `{finding.get('file', 'unknown')}`

**Description:**
{finding['description']}

**Exploitation:**
The prover can set `{finding['signal']}` to an arbitrary value because it is not 
constrained by any `===` statement. This allows:
- Proof forgery: Generate valid proofs for false statements
- Double-spending: If used in nullifier computation
- Identity spoofing: If used in commitment verification

**Remediation:**
Add explicit constraint: `{finding['signal']} === <expected_computation>;`

---
"""
    
    if not exploitable:
        report += "\n*No directly exploitable vulnerabilities found.*\n"
    
    report += """
## All Findings

"""
    
    for finding in findings:
        report += f"- **[{finding['severity']}]** `{finding['signal']}` in `{finding['template']}`: {finding['description']}\n"
    
    report += """

---

## About ORACLE

ORACLE (Offensive Reasoning and Assumption-Challenging Logic Engine) is an automated
vulnerability hunter that analyzes smart contracts and ZK circuits for security issues.

For Circom circuits, ORACLE focuses on **under-constrained signals** - the most common
and dangerous class of ZK circuit vulnerabilities. An under-constrained signal can be
set to any value by the prover, breaking the soundness of the proof system.

*Report generated by ORACLE v1.0*
"""
    
    return report
