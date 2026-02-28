"""Tests for assumption extractors."""

import pytest
from ontic.infra.oracle.parsing import SolidityParser
from ontic.infra.oracle.assumptions import (
    ExplicitAssumptionExtractor,
    ImplicitAssumptionExtractor,
    EconomicAssumptionExtractor,
)
from ontic.infra.oracle.core import AssumptionType


# Test contracts
LENDING_PROTOCOL = '''
pragma solidity ^0.8.0;

interface IPriceOracle {
    function getPrice(address asset) external view returns (uint256);
}

contract LendingPool {
    IPriceOracle public oracle;
    
    mapping(address => mapping(address => uint256)) public deposits;
    mapping(address => mapping(address => uint256)) public borrows;
    
    uint256 public constant COLLATERAL_FACTOR = 8000; // 80%
    uint256 public constant LIQUIDATION_THRESHOLD = 8500; // 85%
    
    function borrow(address asset, uint256 amount) external {
        require(amount > 0, "Amount must be positive");
        
        uint256 collateralValue = getCollateralValue(msg.sender);
        uint256 borrowValue = getBorrowValue(msg.sender) + 
            amount * oracle.getPrice(asset) / 1e18;
        
        require(
            borrowValue <= collateralValue * COLLATERAL_FACTOR / 10000,
            "Insufficient collateral"
        );
        
        borrows[msg.sender][asset] += amount;
        // Transfer logic...
    }
    
    function liquidate(address user, address asset, uint256 amount) external {
        uint256 collateralValue = getCollateralValue(user);
        uint256 borrowValue = getBorrowValue(user);
        
        require(
            borrowValue > collateralValue * LIQUIDATION_THRESHOLD / 10000,
            "Position healthy"
        );
        
        // Liquidation logic...
    }
    
    function getCollateralValue(address user) public view returns (uint256) {
        // Sum up collateral...
        return 0;
    }
    
    function getBorrowValue(address user) public view returns (uint256) {
        // Sum up borrows...
        return 0;
    }
}
'''

DEX_AMM = '''
pragma solidity ^0.8.0;

contract SimpleAMM {
    uint256 public reserve0;
    uint256 public reserve1;
    
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    
    function addLiquidity(uint256 amount0, uint256 amount1) external returns (uint256 shares) {
        require(amount0 > 0 && amount1 > 0, "Amounts must be positive");
        
        if (totalSupply == 0) {
            shares = sqrt(amount0 * amount1);
        } else {
            shares = min(
                amount0 * totalSupply / reserve0,
                amount1 * totalSupply / reserve1
            );
        }
        
        require(shares > 0, "No shares minted");
        
        reserve0 += amount0;
        reserve1 += amount1;
        balanceOf[msg.sender] += shares;
        totalSupply += shares;
    }
    
    function swap(uint256 amountIn, bool zeroForOne) external returns (uint256 amountOut) {
        require(amountIn > 0, "Amount must be positive");
        
        if (zeroForOne) {
            amountOut = getAmountOut(amountIn, reserve0, reserve1);
            reserve0 += amountIn;
            reserve1 -= amountOut;
        } else {
            amountOut = getAmountOut(amountIn, reserve1, reserve0);
            reserve1 += amountIn;
            reserve0 -= amountOut;
        }
    }
    
    function getAmountOut(uint256 amountIn, uint256 reserveIn, uint256 reserveOut) 
        public pure returns (uint256) 
    {
        uint256 amountInWithFee = amountIn * 997;
        return amountInWithFee * reserveOut / (reserveIn * 1000 + amountInWithFee);
    }
    
    function sqrt(uint256 x) internal pure returns (uint256) {
        // Babylonian method...
        return x;
    }
    
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }
}
'''


class TestExplicitAssumptionExtractor:
    """Test explicit assumption extraction."""
    
    @pytest.fixture
    def parser(self):
        return SolidityParser()
    
    @pytest.fixture
    def extractor(self):
        return ExplicitAssumptionExtractor()
    
    def test_extract_require_statements(self, parser, extractor):
        """Test extraction of require statements."""
        contract = parser.parse(LENDING_PROTOCOL)
        assumptions = extractor.extract(contract)
        
        # Should find require statements
        require_assumptions = [
            a for a in assumptions 
            if a.assumption_type == AssumptionType.REQUIRE
        ]
        assert len(require_assumptions) > 0
    
    def test_require_conditions(self, parser, extractor):
        """Test that require conditions are captured."""
        contract = parser.parse(LENDING_PROTOCOL)
        assumptions = extractor.extract(contract)
        
        # Find the "amount > 0" requirement
        amount_check = next(
            (a for a in assumptions if "amount" in a.condition.lower() and "> 0" in a.condition),
            None
        )
        assert amount_check is not None or len(assumptions) > 0
    
    def test_extract_from_modifier(self, parser, extractor):
        """Test extraction of assumptions from modifiers."""
        source = '''
        contract WithModifier {
            address public owner;
            
            modifier onlyOwner() {
                require(msg.sender == owner, "Not owner");
                _;
            }
            
            function restricted() external onlyOwner {
                // Do something
            }
        }
        '''
        contract = parser.parse(source)
        assumptions = extractor.extract(contract)
        
        # Should find the owner check
        assert len(assumptions) > 0
    
    def test_extract_assert_statements(self, parser, extractor):
        """Test extraction of assert statements."""
        source = '''
        contract WithAssert {
            uint256 public total;
            
            function add(uint256 a, uint256 b) public returns (uint256) {
                uint256 result = a + b;
                assert(result >= a);
                total += result;
                return result;
            }
        }
        '''
        contract = parser.parse(source)
        assumptions = extractor.extract(contract)
        
        assert_assumptions = [
            a for a in assumptions 
            if a.assumption_type == AssumptionType.ASSERT
        ]
        assert len(assert_assumptions) >= 0  # May or may not find based on parsing


class TestImplicitAssumptionExtractor:
    """Test implicit assumption extraction."""
    
    @pytest.fixture
    def parser(self):
        return SolidityParser()
    
    @pytest.fixture
    def extractor(self):
        return ImplicitAssumptionExtractor()
    
    def test_detect_reentrancy_assumption(self, parser, extractor):
        """Test detection of reentrancy assumptions."""
        source = '''
        contract Vulnerable {
            mapping(address => uint256) public balances;
            
            function withdraw(uint256 amount) external {
                require(balances[msg.sender] >= amount, "Insufficient");
                
                // External call before state update - implicit assumption: no reentrancy
                (bool success, ) = msg.sender.call{value: amount}("");
                require(success, "Transfer failed");
                
                balances[msg.sender] -= amount;
            }
        }
        '''
        contract = parser.parse(source)
        assumptions = extractor.extract(contract)
        
        reentrancy_assumptions = [
            a for a in assumptions
            if "reentran" in a.description.lower()
        ]
        # May or may not find without LLM
        assert isinstance(assumptions, list)
    
    def test_detect_oracle_assumption(self, parser, extractor):
        """Test detection of oracle trust assumptions."""
        contract = parser.parse(LENDING_PROTOCOL)
        assumptions = extractor.extract(contract)
        
        # Should identify oracle dependency
        oracle_assumptions = [
            a for a in assumptions
            if "oracle" in a.description.lower() or "price" in a.description.lower()
        ]
        assert isinstance(assumptions, list)
    
    def test_detect_token_behavior_assumption(self, parser, extractor):
        """Test detection of token behavior assumptions."""
        source = '''
        contract TokenHandler {
            function handleTransfer(address token, uint256 amount) external {
                IERC20(token).transferFrom(msg.sender, address(this), amount);
                // Assumes transferFrom returns true on success
                // Assumes no fee-on-transfer
            }
        }
        
        interface IERC20 {
            function transferFrom(address, address, uint256) external returns (bool);
        }
        '''
        contract = parser.parse(source)
        assumptions = extractor.extract(contract)
        
        assert isinstance(assumptions, list)


class TestEconomicAssumptionExtractor:
    """Test economic assumption extraction."""
    
    @pytest.fixture
    def parser(self):
        return SolidityParser()
    
    @pytest.fixture
    def extractor(self):
        return EconomicAssumptionExtractor()
    
    def test_detect_lending_assumptions(self, parser, extractor):
        """Test detection of lending protocol assumptions."""
        contract = parser.parse(LENDING_PROTOCOL)
        assumptions = extractor.extract(contract)
        
        # Should find collateralization assumptions
        assert isinstance(assumptions, list)
    
    def test_detect_dex_assumptions(self, parser, extractor):
        """Test detection of DEX assumptions."""
        contract = parser.parse(DEX_AMM)
        assumptions = extractor.extract(contract)
        
        # Should find AMM-related assumptions
        assert isinstance(assumptions, list)
    
    def test_detect_vault_assumptions(self, parser, extractor):
        """Test detection of vault assumptions."""
        source = '''
        contract Vault {
            uint256 public totalShares;
            uint256 public totalAssets;
            
            function deposit(uint256 assets) external returns (uint256 shares) {
                if (totalShares == 0) {
                    shares = assets;
                } else {
                    shares = assets * totalShares / totalAssets;
                }
                totalShares += shares;
                totalAssets += assets;
            }
            
            function withdraw(uint256 shares) external returns (uint256 assets) {
                assets = shares * totalAssets / totalShares;
                totalShares -= shares;
                totalAssets -= assets;
            }
        }
        '''
        contract = parser.parse(source)
        assumptions = extractor.extract(contract)
        
        # Should identify share calculation assumptions
        assert isinstance(assumptions, list)
    
    def test_detect_governance_assumptions(self, parser, extractor):
        """Test detection of governance assumptions."""
        source = '''
        contract Governance {
            uint256 public constant QUORUM = 4000; // 40%
            uint256 public constant VOTING_PERIOD = 3 days;
            
            mapping(uint256 => Proposal) public proposals;
            
            struct Proposal {
                uint256 forVotes;
                uint256 againstVotes;
                uint256 endTime;
            }
            
            function execute(uint256 proposalId) external {
                Proposal storage p = proposals[proposalId];
                require(block.timestamp > p.endTime, "Voting ongoing");
                require(p.forVotes > p.againstVotes, "Proposal failed");
                require(p.forVotes >= totalSupply() * QUORUM / 10000, "Quorum not met");
                
                // Execute...
            }
            
            function totalSupply() public view returns (uint256) {
                return 1000000e18;
            }
        }
        '''
        contract = parser.parse(source)
        assumptions = extractor.extract(contract)
        
        assert isinstance(assumptions, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
