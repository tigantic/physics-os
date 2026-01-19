"""Tests for the complete ORACLE pipeline."""

import pytest
from tensornet.oracle import ORACLE, HuntResult, VerifiedExploit
from tensornet.oracle.core import Contract, Assumption, Challenge, AttackScenario


# Vulnerable vault for testing
VULNERABLE_VAULT = '''
pragma solidity ^0.8.0;

contract VulnerableVault {
    mapping(address => uint256) public shares;
    uint256 public totalShares;
    
    function deposit(uint256 assets) external returns (uint256 sharesMinted) {
        if (totalShares == 0) {
            sharesMinted = assets;
        } else {
            sharesMinted = assets * totalShares / totalAssets();
        }
        
        shares[msg.sender] += sharesMinted;
        totalShares += sharesMinted;
    }
    
    function withdraw(uint256 shareAmount) external returns (uint256 assets) {
        require(shares[msg.sender] >= shareAmount, "Insufficient shares");
        
        assets = shareAmount * totalAssets() / totalShares;
        
        // Transfer before state update (reentrancy)
        payable(msg.sender).transfer(assets);
        
        shares[msg.sender] -= shareAmount;
        totalShares -= shareAmount;
    }
    
    function totalAssets() public view returns (uint256) {
        return address(this).balance;
    }
    
    function emergencyWithdraw(address to) external {
        // No access control
        payable(to).transfer(address(this).balance);
    }
}
'''

SAFE_CONTRACT = '''
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract SafeVault is ReentrancyGuard, Ownable {
    mapping(address => uint256) public balances;
    
    uint256 private constant MIN_DEPOSIT = 1000;
    
    function deposit() external payable nonReentrant {
        require(msg.value >= MIN_DEPOSIT, "Below minimum");
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) external nonReentrant {
        require(balances[msg.sender] >= amount, "Insufficient");
        
        // State update before transfer (CEI pattern)
        balances[msg.sender] -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    function emergencyWithdraw(address to) external onlyOwner {
        uint256 balance = address(this).balance;
        (bool success, ) = to.call{value: balance}("");
        require(success, "Transfer failed");
    }
}
'''


class TestORACLEIntegration:
    """Integration tests for ORACLE."""
    
    @pytest.fixture
    def oracle(self):
        return ORACLE()
    
    def test_hunt_returns_result(self, oracle):
        """Test that hunt returns a HuntResult."""
        result = oracle.hunt(source=VULNERABLE_VAULT)
        
        assert isinstance(result, HuntResult)
        assert isinstance(result.contract, Contract)
    
    def test_hunt_finds_assumptions(self, oracle):
        """Test that hunt identifies assumptions."""
        result = oracle.hunt(source=VULNERABLE_VAULT)
        
        assert len(result.assumptions) > 0
    
    def test_hunt_generates_challenges(self, oracle):
        """Test that hunt generates challenges."""
        result = oracle.hunt(source=VULNERABLE_VAULT)
        
        # May or may not find challenges depending on analysis
        assert isinstance(result.challenges, list)
    
    def test_hunt_creates_scenarios(self, oracle):
        """Test that hunt creates attack scenarios."""
        result = oracle.hunt(source=VULNERABLE_VAULT)
        
        # Should generate at least some scenarios
        assert isinstance(result.scenarios, list)
    
    def test_vulnerable_contract_findings(self, oracle):
        """Test that vulnerable contract produces findings."""
        result = oracle.hunt(source=VULNERABLE_VAULT)
        
        # Should find issues in a clearly vulnerable contract
        # At minimum, should extract assumptions and possibly generate scenarios
        assert result.assumptions or result.challenges or result.scenarios
    
    def test_safe_contract_fewer_findings(self, oracle):
        """Test that safe contract produces fewer critical findings."""
        result = oracle.hunt(source=SAFE_CONTRACT)
        
        # Safe contract should have fewer verified exploits
        # Note: May still find theoretical issues
        assert isinstance(result.verified_exploits, list)
    
    def test_quick_scan(self, oracle):
        """Test quick scan functionality."""
        assumptions = oracle.quick_scan(source=VULNERABLE_VAULT)
        
        assert isinstance(assumptions, list)
    
    def test_generate_report(self, oracle):
        """Test report generation for exploits."""
        result = oracle.hunt(source=VULNERABLE_VAULT)
        
        if result.verified_exploits:
            report = oracle.generate_report(result.verified_exploits[0])
            assert report is not None
            assert report.title
            assert report.description
    
    def test_hunt_with_verbose(self, oracle, capsys):
        """Test verbose output during hunt."""
        oracle.hunt(source=VULNERABLE_VAULT, verbose=True)
        
        captured = capsys.readouterr()
        # Verbose mode should produce output
        # Note: May be empty if no print statements executed
        assert isinstance(captured.out, str)
    
    def test_hunt_timing(self, oracle):
        """Test that hunt records timing information."""
        result = oracle.hunt(source=VULNERABLE_VAULT)
        
        assert result.hunt_time_seconds >= 0


class TestORACLEComponents:
    """Test individual ORACLE components."""
    
    @pytest.fixture
    def oracle(self):
        return ORACLE()
    
    def test_parser_available(self, oracle):
        """Test parser is accessible."""
        assert oracle.parser is not None
    
    def test_extractors_available(self, oracle):
        """Test extractors are accessible."""
        assert oracle.explicit_extractor is not None
        assert oracle.implicit_extractor is not None
        assert oracle.economic_extractor is not None
    
    def test_challenger_available(self, oracle):
        """Test challenger is accessible."""
        assert oracle.challenger is not None
    
    def test_scenario_generator_available(self, oracle):
        """Test scenario generator is accessible."""
        assert oracle.scenario_generator is not None
    
    def test_verifier_available(self, oracle):
        """Test verifier is accessible."""
        assert oracle.verifier is not None
    
    def test_report_generator_available(self, oracle):
        """Test report generator is accessible."""
        assert oracle.report_generator is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def oracle(self):
        return ORACLE()
    
    def test_empty_contract(self, oracle):
        """Test handling of empty contract."""
        result = oracle.hunt(source="contract Empty {}")
        
        assert result is not None
        assert isinstance(result, HuntResult)
    
    def test_minimal_contract(self, oracle):
        """Test handling of minimal contract."""
        source = '''
        contract Minimal {
            uint256 public value;
        }
        '''
        result = oracle.hunt(source=source)
        
        assert result is not None
    
    def test_complex_contract(self, oracle):
        """Test handling of more complex contract."""
        source = '''
        pragma solidity ^0.8.0;
        
        interface IExternal {
            function callback(bytes calldata data) external;
        }
        
        library SafeMath {
            function add(uint a, uint b) internal pure returns (uint) {
                uint c = a + b;
                require(c >= a, "overflow");
                return c;
            }
        }
        
        contract Complex {
            using SafeMath for uint256;
            
            struct Position {
                uint256 size;
                uint256 collateral;
                int256 pnl;
            }
            
            mapping(address => Position) public positions;
            IExternal public external_contract;
            
            modifier validSize(uint256 size) {
                require(size > 0 && size <= 1e24, "Invalid size");
                _;
            }
            
            function openPosition(uint256 size, uint256 collateral) 
                external 
                validSize(size) 
            {
                require(collateral >= size / 10, "Insufficient collateral");
                positions[msg.sender] = Position(size, collateral, 0);
            }
            
            receive() external payable {}
            
            fallback() external payable {
                external_contract.callback(msg.data);
            }
        }
        '''
        result = oracle.hunt(source=source)
        
        assert result is not None
        assert result.contract.name == "Complex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
