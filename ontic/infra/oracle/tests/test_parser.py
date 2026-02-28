"""Tests for Solidity parser."""

import pytest
from ontic.infra.oracle.parsing import SolidityParser
from ontic.infra.oracle.core import Contract


# Sample contracts for testing
SIMPLE_CONTRACT = '''
pragma solidity ^0.8.0;

contract Simple {
    uint256 public value;
    
    function setValue(uint256 _value) public {
        require(_value > 0, "Value must be positive");
        value = _value;
    }
}
'''

ERC20_CONTRACT = '''
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MyToken is ERC20 {
    address public owner;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    constructor() ERC20("MyToken", "MTK") {
        owner = msg.sender;
    }
    
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }
    
    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
    }
}
'''

MULTI_CONTRACT = '''
pragma solidity ^0.8.0;

interface IOracle {
    function getPrice() external view returns (uint256);
}

library MathLib {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        return a + b;
    }
}

contract Vault {
    using MathLib for uint256;
    
    IOracle public oracle;
    mapping(address => uint256) public balances;
    
    event Deposit(address indexed user, uint256 amount);
    
    constructor(address _oracle) {
        oracle = IOracle(_oracle);
    }
    
    function deposit() external payable {
        require(msg.value > 0, "Must send ETH");
        balances[msg.sender] = balances[msg.sender].add(msg.value);
        emit Deposit(msg.sender, msg.value);
    }
    
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
'''


class TestSolidityParser:
    """Test suite for Solidity parser."""
    
    def test_parse_simple_contract(self):
        """Test parsing a simple contract."""
        parser = SolidityParser()
        contract = parser.parse(SIMPLE_CONTRACT)
        
        assert isinstance(contract, Contract)
        assert contract.name == "Simple"
        assert len(contract.functions) >= 1
        assert len(contract.state_variables) >= 1
    
    def test_extract_functions(self):
        """Test function extraction."""
        parser = SolidityParser()
        contract = parser.parse(SIMPLE_CONTRACT)
        
        # Find setValue function
        set_value_fn = next(
            (f for f in contract.functions if f.name == "setValue"),
            None
        )
        assert set_value_fn is not None
        assert set_value_fn.visibility == "public"
        assert len(set_value_fn.parameters) == 1
        assert set_value_fn.parameters[0][0] == "_value"
    
    def test_extract_state_variables(self):
        """Test state variable extraction."""
        parser = SolidityParser()
        contract = parser.parse(SIMPLE_CONTRACT)
        
        value_var = next(
            (v for v in contract.state_variables if v.name == "value"),
            None
        )
        assert value_var is not None
        assert value_var.var_type == "uint256"
        assert value_var.visibility == "public"
    
    def test_extract_modifiers(self):
        """Test modifier extraction."""
        parser = SolidityParser()
        contract = parser.parse(ERC20_CONTRACT)
        
        assert len(contract.modifiers) >= 1
        only_owner = next(
            (m for m in contract.modifiers if m.name == "onlyOwner"),
            None
        )
        assert only_owner is not None
    
    def test_extract_events(self):
        """Test event extraction."""
        parser = SolidityParser()
        contract = parser.parse(MULTI_CONTRACT)
        
        assert len(contract.events) >= 1
        deposit_event = next(
            (e for e in contract.events if e.name == "Deposit"),
            None
        )
        assert deposit_event is not None
    
    def test_extract_imports(self):
        """Test import extraction."""
        parser = SolidityParser()
        contract = parser.parse(ERC20_CONTRACT)
        
        assert len(contract.imports) >= 1
        assert any("ERC20" in imp for imp in contract.imports)
    
    def test_extract_inheritance(self):
        """Test inheritance extraction."""
        parser = SolidityParser()
        contract = parser.parse(ERC20_CONTRACT)
        
        assert len(contract.inheritance) >= 1
        assert "ERC20" in contract.inheritance
    
    def test_cfg_extraction(self):
        """Test control flow graph extraction."""
        parser = SolidityParser()
        contract = parser.parse(MULTI_CONTRACT)
        
        cfg = parser.extract_cfg(contract)
        
        assert isinstance(cfg, dict)
        # Should have entries for functions
        assert len(cfg) > 0
    
    def test_dfg_extraction(self):
        """Test data flow graph extraction."""
        parser = SolidityParser()
        contract = parser.parse(MULTI_CONTRACT)
        
        dfg = parser.extract_dfg(contract)
        
        assert isinstance(dfg, dict)
    
    def test_call_graph_extraction(self):
        """Test call graph extraction."""
        parser = SolidityParser()
        contract = parser.parse(MULTI_CONTRACT)
        
        call_graph = parser.extract_call_graph(contract)
        
        assert isinstance(call_graph, dict)
    
    def test_parse_empty_contract(self):
        """Test parsing minimal contract."""
        parser = SolidityParser()
        contract = parser.parse("contract Empty {}")
        
        assert contract.name == "Empty"
        assert len(contract.functions) == 0
    
    def test_parse_multiple_contracts(self):
        """Test parsing file with multiple contracts."""
        parser = SolidityParser()
        # Should extract the main contract (Vault)
        contract = parser.parse(MULTI_CONTRACT)
        
        assert contract is not None
        # Parser extracts last contract by default
        assert contract.name in ["Vault", "MathLib", "IOracle"]


class TestParserEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_solidity(self):
        """Test handling of invalid Solidity code."""
        parser = SolidityParser()
        # Should not crash, should return empty contract
        contract = parser.parse("this is not valid solidity")
        
        assert contract is not None
    
    def test_unicode_in_comments(self):
        """Test handling of unicode in comments."""
        source = '''
        // Contract for 日本語 support
        contract Unicode {
            string public message = "Hello 世界";
        }
        '''
        parser = SolidityParser()
        contract = parser.parse(source)
        
        assert contract.name == "Unicode"
    
    def test_assembly_block(self):
        """Test handling of inline assembly."""
        source = '''
        contract WithAssembly {
            function getBalance() public view returns (uint256 bal) {
                assembly {
                    bal := selfbalance()
                }
            }
        }
        '''
        parser = SolidityParser()
        contract = parser.parse(source)
        
        assert contract.name == "WithAssembly"
        assert any(f.has_assembly for f in contract.functions if f.name == "getBalance")
    
    def test_receive_and_fallback(self):
        """Test extraction of receive and fallback functions."""
        source = '''
        contract Receiver {
            receive() external payable {}
            fallback() external payable {}
        }
        '''
        parser = SolidityParser()
        contract = parser.parse(source)
        
        function_names = [f.name for f in contract.functions]
        assert "receive" in function_names or "fallback" in function_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
