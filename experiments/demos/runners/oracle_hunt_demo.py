#!/usr/bin/env python3
"""
ORACLE Demo: Hunt vulnerabilities in a sample contract.

This demonstrates the full ORACLE pipeline on a vulnerable vault contract.
"""

import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensornet.infra.oracle import ORACLE


# Sample vulnerable vault contract
VULNERABLE_VAULT = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title SimpleVault
 * @notice A simple vault with MULTIPLE vulnerabilities for ORACLE to find
 */
contract SimpleVault {
    IERC20 public asset;
    
    mapping(address => uint256) public shares;
    uint256 public totalShares;
    
    address public owner;
    
    event Deposit(address indexed user, uint256 assets, uint256 shares);
    event Withdraw(address indexed user, uint256 assets, uint256 shares);
    
    constructor(address _asset) {
        asset = IERC20(_asset);
        owner = msg.sender;
    }
    
    // VULNERABILITY 1: First depositor inflation attack
    // No protection against share price manipulation
    function deposit(uint256 assets) external returns (uint256 sharesMinted) {
        require(assets > 0, "Cannot deposit 0");
        
        if (totalShares == 0) {
            sharesMinted = assets;
        } else {
            // shares = assets * totalShares / totalAssets
            sharesMinted = assets * totalShares / totalAssets();
        }
        
        // VULNERABILITY 2: No minimum shares check (could get 0 shares)
        
        asset.transferFrom(msg.sender, address(this), assets);
        
        shares[msg.sender] += sharesMinted;
        totalShares += sharesMinted;
        
        emit Deposit(msg.sender, assets, sharesMinted);
    }
    
    // VULNERABILITY 3: Reentrancy on withdraw
    // Transfers before state update
    function withdraw(uint256 shareAmount) external returns (uint256 assets) {
        require(shareAmount > 0, "Cannot withdraw 0");
        require(shares[msg.sender] >= shareAmount, "Insufficient shares");
        
        // Calculate assets to return
        assets = shareAmount * totalAssets() / totalShares;
        
        // VULNERABLE: Transfer before state update
        asset.transfer(msg.sender, assets);
        
        shares[msg.sender] -= shareAmount;
        totalShares -= shareAmount;
        
        emit Withdraw(msg.sender, assets, shareAmount);
    }
    
    function totalAssets() public view returns (uint256) {
        return asset.balanceOf(address(this));
    }
    
    // VULNERABILITY 4: No access control on emergency function
    function emergencyWithdraw(address to) external {
        // Missing: require(msg.sender == owner, "Only owner");
        uint256 balance = asset.balanceOf(address(this));
        asset.transfer(to, balance);
    }
    
    // VULNERABILITY 5: Precision loss in share calculation
    function previewDeposit(uint256 assets) public view returns (uint256) {
        if (totalShares == 0) return assets;
        // Integer division can cause rounding errors
        return assets * totalShares / totalAssets();
    }
}
'''


def main():
    """Run ORACLE demo."""
    print("=" * 70)
    print("ORACLE DEMO: Hunting Vulnerabilities in SimpleVault")
    print("=" * 70)
    print()
    print("This contract has 5 intentional vulnerabilities:")
    print("  1. First depositor inflation attack")
    print("  2. Zero shares on deposit (rounding)")
    print("  3. Reentrancy on withdraw")
    print("  4. Missing access control on emergencyWithdraw")
    print("  5. Precision loss in share calculation")
    print()
    print("Let's see what ORACLE finds...")
    print()
    
    # Initialize ORACLE
    oracle = ORACLE()
    
    # Run the hunt
    result = oracle.hunt(source=VULNERABLE_VAULT, verbose=True)
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED FINDINGS")
    print("=" * 70)
    
    for i, exploit in enumerate(result.verified_exploits):
        print(f"\n[Exploit {i+1}] {exploit.scenario.name}")
        print("-" * 50)
        print(f"Severity: {oracle.report_generator._classify_severity(exploit)}")
        print(f"Confidence: {exploit.confidence * 100:.0f}%")
        print(f"Expected Profit: {exploit.scenario.expected_profit / 10**18:.2f} ETH")
        print(f"Required Capital: {exploit.scenario.required_capital / 10**18:.4f} ETH")
        print(f"Complexity: {exploit.scenario.complexity}")
        print()
        print("Attack Steps:")
        for j, step in enumerate(exploit.scenario.steps):
            print(f"  {j+1}. {step.action}")
        
        # Generate report
        report = oracle.generate_report(exploit)
        print()
        print("Generated Report Title:", report.title)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Contract: {result.contract.name}")
    print(f"Hunt Time: {result.hunt_time_seconds:.1f} seconds")
    print(f"Assumptions Analyzed: {len(result.assumptions)}")
    print(f"Potential Issues: {len(result.challenges)}")
    print(f"Attack Scenarios: {len(result.scenarios)}")
    print(f"Verified Exploits: {len(result.verified_exploits)}")
    
    if result.verified_exploits:
        total_profit = result.total_potential_profit / 10**18
        print(f"\n💰 Total Potential Profit: {total_profit:.2f} ETH")
        print("\n⚠️  This contract should NOT be deployed!")
    
    return result


if __name__ == "__main__":
    main()
