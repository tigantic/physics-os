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
    from tensornet.oracle import ORACLE
    
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
