#!/usr/bin/env python3
"""
ORACLE Command Line Interface.

Usage:
    oracle hunt <source_file> [--output <dir>] [--verbose]
    oracle hunt --address <addr> [--chain <chain>] [--output <dir>]
    oracle scan <source_file>  # Quick scan without full verification
    
Examples:
    oracle hunt ./contracts/Vault.sol --output ./reports
    oracle hunt --address 0x1234...abcd --chain ethereum
    oracle scan ./contracts/Vault.sol
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
  oracle hunt ./contracts/Vault.sol
  oracle hunt ./contracts/Vault.sol --output ./reports --verbose
  oracle scan ./contracts/Vault.sol
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Hunt command
    hunt_parser = subparsers.add_parser("hunt", help="Hunt for vulnerabilities")
    hunt_parser.add_argument("source", nargs="?", help="Path to Solidity source file")
    hunt_parser.add_argument("--address", "-a", help="Contract address to analyze")
    hunt_parser.add_argument("--chain", "-c", default="ethereum", help="Chain (default: ethereum)")
    hunt_parser.add_argument("--output", "-o", help="Output directory for reports")
    hunt_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    hunt_parser.add_argument("--min-confidence", type=float, default=0.5, 
                            help="Minimum confidence threshold (default: 0.5)")
    
    # Scan command (quick)
    scan_parser = subparsers.add_parser("scan", help="Quick scan without verification")
    scan_parser.add_argument("source", help="Path to Solidity source file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Import ORACLE
    try:
        from tensornet.oracle import ORACLE
    except ImportError:
        # Try relative import for development
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from tensornet.oracle import ORACLE
    
    if args.command == "hunt":
        run_hunt(args)
    elif args.command == "scan":
        run_scan(args)


def run_hunt(args):
    """Run full vulnerability hunt."""
    # Check inputs
    if not args.source and not args.address:
        print("Error: Must provide source file or --address")
        sys.exit(1)
    
    # Initialize ORACLE
    oracle = ORACLE()
    
    # Run hunt
    try:
        if args.source:
            result = oracle.hunt(
                file_path=args.source,
                min_confidence=args.min_confidence,
                verbose=args.verbose,
            )
        else:
            result = oracle.hunt(
                address=args.address,
                chain=args.chain,
                min_confidence=args.min_confidence,
                verbose=args.verbose,
            )
    except Exception as e:
        print(f"Error during hunt: {e}")
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
    from tensornet.oracle import ORACLE
    
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


if __name__ == "__main__":
    main()
