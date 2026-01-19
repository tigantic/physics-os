// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title RoundingVault - Vulnerable ERC4626-style vault
 * @notice Demonstrates rounding attack surface similar to Ethena sUSDe
 * 
 * VULNERABILITY: First depositor can manipulate share price to steal
 * from subsequent depositors via donation attack.
 */
contract RoundingVault {
    string public name = "Rounding Vault";
    string public symbol = "rVAULT";
    uint8 public decimals = 18;
    
    mapping(address => uint256) public balanceOf;  // shares
    uint256 public totalSupply;  // total shares
    uint256 public totalAssets;  // total underlying
    
    address public owner;
    
    event Deposit(address indexed depositor, uint256 assets, uint256 shares);
    event Withdraw(address indexed withdrawer, uint256 assets, uint256 shares);
    
    constructor() payable {
        owner = msg.sender;
        // Accept any initial funding
        if (msg.value > 0) {
            totalAssets = msg.value;
        }
    }
    
    receive() external payable {
        totalAssets += msg.value;
    }
    
    /**
     * @notice Deposit ETH and receive shares
     * @dev VULNERABLE: Uses floor division, first depositor can manipulate
     */
    function deposit() external payable returns (uint256 shares) {
        require(msg.value > 0, "Must deposit > 0");
        
        if (totalSupply == 0) {
            // First depositor gets 1:1
            shares = msg.value;
        } else {
            // shares = (assets * totalSupply) / totalAssets
            // VULNERABLE: Floor division can round to 0!
            shares = (msg.value * totalSupply) / totalAssets;
        }
        
        // CRITICAL BUG: No check for shares > 0!
        // Subsequent depositors can get 0 shares while vault takes their ETH
        
        balanceOf[msg.sender] += shares;
        totalSupply += shares;
        totalAssets += msg.value;
        
        emit Deposit(msg.sender, msg.value, shares);
        return shares;
    }
    
    /**
     * @notice Withdraw underlying by burning shares
     */
    function withdraw(uint256 shares) external returns (uint256 assets) {
        require(shares > 0, "Must withdraw > 0");
        require(balanceOf[msg.sender] >= shares, "Insufficient shares");
        
        // assets = (shares * totalAssets) / totalSupply
        assets = (shares * totalAssets) / totalSupply;
        
        balanceOf[msg.sender] -= shares;
        totalSupply -= shares;
        totalAssets -= assets;
        
        payable(msg.sender).transfer(assets);
        emit Withdraw(msg.sender, assets, shares);
        return assets;
    }
    
    /**
     * @notice Donate to vault without receiving shares (for attack setup)
     */
    function donate() external payable {
        require(msg.value > 0, "Must donate > 0");
        totalAssets += msg.value;
        // No shares minted - pure donation
    }
    
    /**
     * @notice Preview shares for a deposit amount
     */
    function previewDeposit(uint256 assets) external view returns (uint256) {
        if (totalSupply == 0) return assets;
        return (assets * totalSupply) / totalAssets;
    }
    
    /**
     * @notice Get current exchange rate (assets per share)
     */
    function exchangeRate() external view returns (uint256) {
        if (totalSupply == 0) return 1e18;
        return (totalAssets * 1e18) / totalSupply;
    }
    
    /**
     * @notice Emergency withdraw all (owner only) - INTENTIONAL BACKDOOR
     */
    function emergencyWithdraw() external {
        require(msg.sender == owner, "Not owner");
        uint256 balance = address(this).balance;
        payable(owner).transfer(balance);
        totalAssets = 0;
        totalSupply = 0;
    }
}
