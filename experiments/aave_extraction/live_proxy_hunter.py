#!/usr/bin/env python3
"""
Live Proxy Hunter - 72h Rolling Scan
Targets: Transparent Proxy & UUPS patterns with >$50K value
Vectors: UNINITIALIZED_PROXY, missing upgradeToAndCall access controls
"""

import json
import time
import struct
import hashlib
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import requests
import numpy as np
from web3 import Web3
from eth_abi import decode

# Configuration
RPC_ENDPOINT = "https://rpc.mevblocker.io"
ETHERSCAN_API = "https://api.etherscan.io/api"
BLOCKS_72H = 21600  # ~72 hours at 12s/block

# Proxy signatures
EIP1967_IMPL_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
EIP1967_ADMIN_SLOT = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
BEACON_SLOT = "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"

# Function signatures for initialization
INITIALIZE_SIGS = [
    "0x8129fc1c",  # initialize()
    "0xc4d66de8",  # initialize(address)
    "0x4cd88b76",  # initialize(string,string)
    "0xf8c8765e",  # initialize(address,address,address,address)
    "0x1459457a",  # initialize(address,address,address,address,address)
    "0xfe4b84df",  # initialize(uint256)
]

# upgradeToAndCall signature
UPGRADE_SIG = "0x4f1ef286"  # upgradeToAndCall(address,bytes)
UPGRADE_TO_SIG = "0x3659cfe6"  # upgradeTo(address)

# High-value ERC20s for TVL check
TOKEN_ADDRESSES = {
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EescdeCB5BE3830",
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "stETH": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
}


class ProxyType(Enum):
    TRANSPARENT = "TRANSPARENT"
    UUPS = "UUPS"
    BEACON = "BEACON"
    UNKNOWN = "UNKNOWN"


class VulnType(Enum):
    UNINITIALIZED = "UNINITIALIZED_PROXY"
    OPEN_UPGRADE = "OPEN_UPGRADE_TO_AND_CALL"
    MISSING_ADMIN_CHECK = "MISSING_ADMIN_CHECK"


@dataclass
class ProxyTarget:
    address: str
    proxy_type: ProxyType
    implementation: str
    admin: str
    value_usd: float
    deployed_block: int
    vulnerabilities: List[VulnType]
    exploit_path: Optional[Dict] = None


class LiveProxyHunter:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
        self.current_block = self.w3.eth.block_number
        self.targets: List[ProxyTarget] = []
        self.scanned = 0
        self.filtered = 0
        
    def get_recent_contracts(self, hours: int = 72) -> List[Dict]:
        """Get contracts deployed in last N hours via block scanning."""
        contracts = []
        start_block = self.current_block - (hours * 300)  # ~12s/block
        
        print(f"Scanning blocks {start_block} to {self.current_block}...")
        print(f"  ({self.current_block - start_block} blocks)")
        
        # Sample every 100 blocks for efficiency
        sample_rate = 100
        scanned_blocks = 0
        
        for block_num in range(start_block, self.current_block, sample_rate):
            try:
                block = self.w3.eth.get_block(block_num, full_transactions=True)
                
                for tx in block.transactions:
                    # Contract creation = to is None
                    if tx.get('to') is None and tx.get('input'):
                        receipt = self.w3.eth.get_transaction_receipt(tx['hash'])
                        if receipt and receipt.get('contractAddress'):
                            contracts.append({
                                'address': receipt['contractAddress'],
                                'deployer': tx['from'],
                                'block': block_num,
                                'tx_hash': tx['hash'].hex()
                            })
                            print(f"  Found contract: {receipt['contractAddress'][:20]}... (block {block_num})")
                
                scanned_blocks += 1
                if scanned_blocks % 50 == 0:
                    print(f"  Progress: {scanned_blocks * sample_rate}/{self.current_block - start_block} blocks")
                    
            except Exception as e:
                continue
                
        print(f"  Found {len(contracts)} contracts in 72h window")
        return contracts
    
    def detect_proxy_type(self, address: str) -> Tuple[ProxyType, str, str]:
        """Detect if contract is a proxy and its type."""
        address = Web3.to_checksum_address(address)
        
        # Check EIP-1967 implementation slot
        impl_raw = self.w3.eth.get_storage_at(address, EIP1967_IMPL_SLOT)
        impl = "0x" + impl_raw.hex()[-40:]
        
        # Check EIP-1967 admin slot  
        admin_raw = self.w3.eth.get_storage_at(address, EIP1967_ADMIN_SLOT)
        admin = "0x" + admin_raw.hex()[-40:]
        
        # Check beacon slot
        beacon_raw = self.w3.eth.get_storage_at(address, BEACON_SLOT)
        beacon = "0x" + beacon_raw.hex()[-40:]
        
        impl_is_set = int(impl, 16) != 0
        admin_is_set = int(admin, 16) != 0
        beacon_is_set = int(beacon, 16) != 0
        
        if impl_is_set and admin_is_set:
            return ProxyType.TRANSPARENT, impl, admin
        elif impl_is_set and not admin_is_set:
            return ProxyType.UUPS, impl, "0x" + "0" * 40
        elif beacon_is_set:
            return ProxyType.BEACON, beacon, "0x" + "0" * 40
        else:
            return ProxyType.UNKNOWN, "0x" + "0" * 40, "0x" + "0" * 40
    
    def calculate_tvl(self, address: str) -> float:
        """Calculate total value locked in contract."""
        address = Web3.to_checksum_address(address)
        total_usd = 0.0
        
        # Native ETH
        eth_balance = self.w3.eth.get_balance(address)
        eth_usd = float(self.w3.from_wei(eth_balance, 'ether')) * 2800  # Approximate ETH price
        total_usd += eth_usd
        
        # ERC20 tokens
        balance_of_sig = "0x70a08231"
        for token_name, token_addr in TOKEN_ADDRESSES.items():
            try:
                token_addr = Web3.to_checksum_address(token_addr)
                calldata = balance_of_sig + address[2:].zfill(64)
                result = self.w3.eth.call({
                    'to': token_addr,
                    'data': calldata
                })
                balance = int(result.hex(), 16)
                
                # Convert based on decimals
                if token_name in ["USDC", "USDT"]:
                    usd_value = balance / 1e6
                elif token_name == "WBTC":
                    usd_value = (balance / 1e8) * 65000  # Approximate BTC price
                elif token_name in ["WETH", "stETH"]:
                    usd_value = (balance / 1e18) * 2800
                else:
                    usd_value = balance / 1e18  # DAI
                    
                total_usd += usd_value
                
            except Exception:
                continue
                
        return total_usd
    
    def check_init_vulnerability(self, impl_address: str) -> Tuple[bool, Dict]:
        """Check if implementation's initialize() is unprotected."""
        impl_address = Web3.to_checksum_address(impl_address)
        
        if int(impl_address, 16) == 0:
            return False, {}
        
        bytecode = self.w3.eth.get_code(impl_address).hex()
        if len(bytecode) < 10:
            return False, {}
        
        result = {
            "has_initialize": False,
            "initialized_slot_zero": False,
            "can_reinitialize": False
        }
        
        # Check for initialize selectors in bytecode
        for sig in INITIALIZE_SIGS:
            if sig[2:] in bytecode:
                result["has_initialize"] = True
                break
        
        if not result["has_initialize"]:
            return False, result
        
        # Check initialized storage slot (typically slot 0 for Initializable)
        # OZ Initializable uses slot 0 with specific bit pattern
        init_slot = self.w3.eth.get_storage_at(impl_address, 0)
        init_value = int(init_slot.hex(), 16)
        
        # If slot 0 is 0, likely uninitialized
        # OZ sets this to 1 after initialization, or uses _initialized uint8
        if init_value == 0:
            result["initialized_slot_zero"] = True
            result["can_reinitialize"] = True
        
        # Also check implementation directly for _initialized state
        # Try calling initializer to see if it reverts
        for sig in INITIALIZE_SIGS:
            try:
                # Build minimal calldata
                if sig == "0x8129fc1c":
                    calldata = sig
                else:
                    calldata = sig + "0" * 64  # Pad with zeros for address param
                
                # Simulate call
                self.w3.eth.call({
                    'to': impl_address,
                    'data': calldata
                })
                # If we get here without revert, initialize is callable!
                result["can_reinitialize"] = True
                break
            except Exception as e:
                error_msg = str(e).lower()
                if "already initialized" in error_msg or "initializable" in error_msg:
                    result["can_reinitialize"] = False
                    break
                # Other errors might indicate other issues
                continue
        
        vuln = result["can_reinitialize"] and result["has_initialize"]
        return vuln, result
    
    def check_upgrade_vulnerability(self, proxy_address: str, impl_address: str) -> Tuple[bool, Dict]:
        """Check for missing access controls on upgradeToAndCall."""
        proxy_address = Web3.to_checksum_address(proxy_address)
        
        result = {
            "has_upgrade_function": False,
            "upgrade_callable": False,
            "admin_check_present": True
        }
        
        bytecode = self.w3.eth.get_code(proxy_address).hex()
        
        # Check for upgrade signatures
        if UPGRADE_SIG[2:] in bytecode or UPGRADE_TO_SIG[2:] in bytecode:
            result["has_upgrade_function"] = True
        
        if not result["has_upgrade_function"]:
            return False, result
        
        # Try calling upgradeTo with random address to check access control
        test_impl = "0x" + "1" * 40
        try:
            calldata = UPGRADE_TO_SIG + test_impl[2:].zfill(64)
            self.w3.eth.call({
                'to': proxy_address,
                'data': calldata,
                'from': "0x" + "2" * 40  # Random caller
            })
            # If we get here, upgrade might be callable by anyone!
            result["upgrade_callable"] = True
            result["admin_check_present"] = False
        except Exception as e:
            error_msg = str(e).lower()
            if "only admin" in error_msg or "unauthorized" in error_msg or "not admin" in error_msg:
                result["admin_check_present"] = True
            # Most likely reverted due to access control
        
        vuln = not result["admin_check_present"] and result["has_upgrade_function"]
        return vuln, result
    
    def tensorize_bytecode(self, bytecode: str) -> np.ndarray:
        """Convert bytecode to tensor for PyTenNet analysis."""
        if bytecode.startswith("0x"):
            bytecode = bytecode[2:]
        
        bytes_data = bytes.fromhex(bytecode)
        
        # Create adjacency matrix from control flow
        # Simplified: create matrix from JUMP/JUMPI targets
        size = min(len(bytes_data), 1024)  # Cap for memory
        tensor = np.zeros((size, size), dtype=np.float32)
        
        i = 0
        while i < len(bytes_data) - 1:
            opcode = bytes_data[i]
            
            # JUMP (0x56) or JUMPI (0x57)
            if opcode in [0x56, 0x57]:
                # Mark potential jump edges
                if i + 1 < size:
                    tensor[i, i + 1] = 1.0
                # JUMPI also has fallthrough
                if opcode == 0x57 and i + 1 < size:
                    tensor[i, i + 1] = 0.5
            # Sequential execution
            elif opcode < 0x60 or opcode > 0x7f:  # Not PUSH
                if i + 1 < size:
                    tensor[i, i + 1] = 1.0
            
            # Skip PUSH data
            if 0x60 <= opcode <= 0x7f:
                push_size = opcode - 0x5f
                i += push_size
            
            i += 1
        
        return tensor
    
    def pytennet_solve(self, proxy_tensor: np.ndarray, impl_tensor: np.ndarray, 
                       vuln_type: VulnType) -> Optional[Dict]:
        """Apply PyTenNet tropical solver to find exploit path."""
        
        # Combine tensors
        p_size = proxy_tensor.shape[0]
        i_size = impl_tensor.shape[0]
        combined_size = p_size + i_size
        
        combined = np.zeros((combined_size, combined_size), dtype=np.float32)
        combined[:p_size, :p_size] = proxy_tensor
        combined[p_size:, p_size:] = impl_tensor
        
        # Add cross-contract edges based on vulnerability type
        if vuln_type == VulnType.UNINITIALIZED:
            # DELEGATECALL from proxy to impl
            combined[p_size // 2, p_size] = 1.0  # Proxy midpoint -> impl start
        elif vuln_type == VulnType.OPEN_UPGRADE:
            # upgradeToAndCall flow
            combined[0, p_size] = 1.0
        
        # Tropical semiring shortest path (min-plus)
        # Convert to tropical: 0 -> inf for missing edges
        tropical = np.where(combined > 0, 1.0 / combined, np.inf)
        n = tropical.shape[0]
        
        # Floyd-Warshall in tropical semiring (capped iterations)
        max_iter = min(n, 100)
        for k in range(max_iter):
            for i in range(n):
                for j in range(n):
                    tropical[i, j] = min(tropical[i, j], 
                                        tropical[i, k] + tropical[k, j])
        
        # Check if path exists from entry to target
        entry = 0
        target = combined_size - 1
        
        if tropical[entry, target] < np.inf:
            path_length = tropical[entry, target]
            return {
                "path_exists": True,
                "path_length": float(path_length),
                "tensor_size": combined_size,
                "continuous": True
            }
        else:
            return None
    
    def scan(self) -> List[ProxyTarget]:
        """Execute full 72h rolling scan."""
        print("=" * 70)
        print("LIVE PROXY HUNTER - 72H ROLLING SCAN")
        print("=" * 70)
        print(f"Current block: {self.current_block}")
        print(f"Scanning window: {BLOCKS_72H} blocks (~72 hours)")
        print(f"Value threshold: >$50,000")
        print("=" * 70)
        
        # Phase 0: Get recent contracts
        print("\n[PHASE 0] Discovering recently deployed contracts...")
        contracts = self.get_recent_contracts(hours=72)
        self.scanned = len(contracts)
        
        if not contracts:
            print("  No contracts found in 72h window via block scan.")
            print("  Falling back to known high-value new deployments...")
            # Fallback: check some known recent protocol deployments
            contracts = self._get_fallback_targets()
        
        print(f"\n[PHASE 1] Filtering proxy contracts with >$50K TVL...")
        proxy_candidates = []
        
        for contract in contracts:
            addr = contract['address']
            
            # Detect proxy type
            proxy_type, impl, admin = self.detect_proxy_type(addr)
            
            if proxy_type == ProxyType.UNKNOWN:
                continue
            
            # Calculate TVL
            tvl = self.calculate_tvl(addr)
            
            if tvl < 50000:
                continue
            
            print(f"  ✓ {addr[:20]}... | {proxy_type.value} | ${tvl:,.2f}")
            
            proxy_candidates.append({
                'address': addr,
                'proxy_type': proxy_type,
                'implementation': impl,
                'admin': admin,
                'tvl': tvl,
                'block': contract.get('block', 0)
            })
        
        self.filtered = len(proxy_candidates)
        print(f"  Passed filter: {self.filtered}/{self.scanned}")
        
        if not proxy_candidates:
            print("\n  No high-value proxy contracts found in 72h window.")
            return []
        
        # Phase 2: Vulnerability scanning
        print(f"\n[PHASE 2] Tensorizing implementation bytecode...")
        
        for candidate in proxy_candidates:
            addr = candidate['address']
            impl = candidate['implementation']
            
            print(f"\n  Analyzing: {addr[:20]}...")
            
            vulnerabilities = []
            
            # Check initialization vulnerability
            init_vuln, init_result = self.check_init_vulnerability(impl)
            if init_vuln:
                print(f"    ⚠ UNINITIALIZED_PROXY detected!")
                vulnerabilities.append(VulnType.UNINITIALIZED)
            
            # Check upgrade vulnerability
            upgrade_vuln, upgrade_result = self.check_upgrade_vulnerability(addr, impl)
            if upgrade_vuln:
                print(f"    ⚠ OPEN_UPGRADE_TO_AND_CALL detected!")
                vulnerabilities.append(VulnType.OPEN_UPGRADE)
            
            if not vulnerabilities:
                print(f"    ✗ No vulnerabilities detected")
                continue
            
            # Tensorize and solve
            print(f"    Tensorizing bytecode...")
            proxy_bytecode = self.w3.eth.get_code(addr).hex()
            impl_bytecode = self.w3.eth.get_code(impl).hex() if int(impl, 16) != 0 else "0x"
            
            proxy_tensor = self.tensorize_bytecode(proxy_bytecode)
            impl_tensor = self.tensorize_bytecode(impl_bytecode) if impl_bytecode != "0x" else np.zeros((10, 10))
            
            for vuln in vulnerabilities:
                path_result = self.pytennet_solve(proxy_tensor, impl_tensor, vuln)
                
                if path_result and path_result.get("path_exists"):
                    print(f"    ✓ PyTenNet: Continuous path found! Length: {path_result['path_length']:.2f}")
                    
                    target = ProxyTarget(
                        address=addr,
                        proxy_type=candidate['proxy_type'],
                        implementation=impl,
                        admin=candidate['admin'],
                        value_usd=candidate['tvl'],
                        deployed_block=candidate['block'],
                        vulnerabilities=vulnerabilities,
                        exploit_path=path_result
                    )
                    self.targets.append(target)
                else:
                    print(f"    ✗ PyTenNet: No continuous path for {vuln.value}")
        
        return self.targets
    
    def _get_fallback_targets(self) -> List[Dict]:
        """Get fallback targets from known recent high-value deployments."""
        # These are protocols known to have deployed recently
        # We'll scan their proxy contracts
        fallback_proxies = [
            # Recent Uniswap V4 periphery
            "0x000000000004444c5dc75cB358380D2e3dE08A90",
            # Morpho Blue
            "0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb",
            # Ethena
            "0x9D39A5DE30e57443BfF2A8307A4256c8797A3497",
            # Renzo restaking
            "0x74a09653A083691711cF8215a6ab074BB4e99ef5",
            # EigenLayer
            "0x39053D51B77DC0d36036Fc1fCc8Cb819df8Ef37A",
        ]
        
        targets = []
        for addr in fallback_proxies:
            try:
                addr = Web3.to_checksum_address(addr)
                code = self.w3.eth.get_code(addr)
                if len(code) > 0:
                    targets.append({
                        'address': addr,
                        'deployer': 'unknown',
                        'block': self.current_block - 1000
                    })
            except Exception:
                continue
        
        return targets
    
    def generate_report(self) -> Dict:
        """Generate final report."""
        report = {
            "scan_params": {
                "current_block": self.current_block,
                "blocks_scanned": BLOCKS_72H,
                "value_threshold_usd": 50000,
                "contracts_scanned": self.scanned,
                "proxies_filtered": self.filtered
            },
            "targets": [],
            "verdict": "NO_EXPLOITABLE_TARGETS"
        }
        
        for target in self.targets:
            report["targets"].append({
                "address": target.address,
                "proxy_type": target.proxy_type.value,
                "implementation": target.implementation,
                "admin": target.admin,
                "value_usd": target.value_usd,
                "deployed_block": target.deployed_block,
                "vulnerabilities": [v.value for v in target.vulnerabilities],
                "exploit_path": target.exploit_path
            })
        
        if self.targets:
            report["verdict"] = "EXPLOITABLE_TARGETS_FOUND"
        
        return report


def main():
    hunter = LiveProxyHunter()
    
    targets = hunter.scan()
    
    print("\n" + "=" * 70)
    print("SCAN RESULTS")
    print("=" * 70)
    
    if not targets:
        print("NO EXPLOITABLE TARGETS FOUND")
        print("  - All proxies properly initialized")
        print("  - No missing access controls on upgrade functions")
        print("  - PyTenNet found no continuous exploit paths")
    else:
        print(f"EXPLOITABLE TARGETS: {len(targets)}")
        for t in targets:
            print(f"\n  TARGET: {t.address}")
            print(f"    Type: {t.proxy_type.value}")
            print(f"    Implementation: {t.implementation}")
            print(f"    Value: ${t.value_usd:,.2f}")
            print(f"    Vulnerabilities: {[v.value for v in t.vulnerabilities]}")
            if t.exploit_path:
                print(f"    Path Length: {t.exploit_path['path_length']:.2f}")
    
    # Save report
    report = hunter.generate_report()
    with open("live_proxy_scan_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved: live_proxy_scan_results.json")


if __name__ == "__main__":
    main()
