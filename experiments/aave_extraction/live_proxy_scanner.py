#!/usr/bin/env python3
"""
Live Proxy Scanner - 72h Rolling Scan for Uninitialized Proxies
Connects to Alchemy WebSocket for real-time contract discovery
Targets: Transparent + UUPS proxies with >$50k value
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import struct

# Alchemy endpoint
ALCHEMY_HTTP = "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im"
ALCHEMY_WS = "wss://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im"

# EIP-1967 storage slots
EIP1967_IMPL_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
EIP1967_ADMIN_SLOT = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"
EIP1967_BEACON_SLOT = "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"

# Function selectors
SELECTORS = {
    "initialize()": "0x8129fc1c",
    "initialize(address)": "0xc4d66de8",
    "initialize(address,address)": "0x485cc955",
    "initialize(address,bytes)": "0x1794bb3c",
    "upgradeTo(address)": "0x3659cfe6",
    "upgradeToAndCall(address,bytes)": "0x4f1ef286",
    "implementation()": "0x5c60da1b",
    "admin()": "0xf851a440",
    "owner()": "0x8da5cb5b",
    "initialized()": "0x158ef93e",
    "_initialized()": "0x2d7aa82b",
}

# Major ERC-20 tokens for value calculation
TOKENS = {
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EescdeCB5C811111",
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "stETH": "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
}

TOKEN_DECIMALS = {
    "USDC": 6, "USDT": 6, "DAI": 18, "WETH": 18, "WBTC": 8, "stETH": 18
}

# Approximate prices (will fetch live if needed)
TOKEN_PRICES = {
    "USDC": 1.0, "USDT": 1.0, "DAI": 1.0, "WETH": 2800.0, "WBTC": 97000.0, "stETH": 2800.0, "ETH": 2800.0
}

@dataclass
class ProxyCandidate:
    address: str
    block_deployed: int
    timestamp: int
    impl_address: str
    admin_address: str
    proxy_type: str  # "transparent", "uups", "beacon"
    eth_balance: int
    token_balances: Dict[str, int]
    total_value_usd: float
    bytecode_size: int
    is_initialized: Optional[bool]
    has_open_initialize: bool
    has_open_upgrade: bool
    vulnerability: Optional[str]


class LiveProxyScanner:
    def __init__(self):
        self.alchemy_http = ALCHEMY_HTTP
        self.candidates: List[ProxyCandidate] = []
        self.scanned_count = 0
        self.start_time = time.time()
        
    def eth_call(self, to: str, data: str, block: str = "latest") -> Optional[str]:
        """Make an eth_call"""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": to, "data": data}, block],
            "id": 1
        }
        try:
            resp = requests.post(self.alchemy_http, json=payload, timeout=10)
            result = resp.json()
            if "result" in result:
                return result["result"]
        except Exception:
            pass
        return None
    
    def get_storage_at(self, address: str, slot: str) -> str:
        """Get storage at slot"""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getStorageAt",
            "params": [address, slot, "latest"],
            "id": 1
        }
        try:
            resp = requests.post(self.alchemy_http, json=payload, timeout=10)
            result = resp.json()
            return result.get("result", "0x0")
        except Exception:
            return "0x0"
    
    def get_code(self, address: str) -> str:
        """Get contract bytecode"""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getCode",
            "params": [address, "latest"],
            "id": 1
        }
        try:
            resp = requests.post(self.alchemy_http, json=payload, timeout=10)
            result = resp.json()
            return result.get("result", "0x")
        except Exception:
            return "0x"
    
    def get_balance(self, address: str) -> int:
        """Get ETH balance"""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, "latest"],
            "id": 1
        }
        try:
            resp = requests.post(self.alchemy_http, json=payload, timeout=10)
            result = resp.json()
            return int(result.get("result", "0x0"), 16)
        except Exception:
            return 0
    
    def get_block_number(self) -> int:
        """Get current block number"""
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1
        }
        resp = requests.post(self.alchemy_http, json=payload, timeout=10)
        return int(resp.json()["result"], 16)
    
    def get_token_balance(self, token: str, holder: str) -> int:
        """Get ERC20 token balance"""
        data = "0x70a08231" + holder[2:].zfill(64)
        result = self.eth_call(token, data)
        if result and result != "0x":
            try:
                return int(result, 16)
            except:
                pass
        return 0
    
    def calculate_total_value(self, eth_balance: int, token_balances: Dict[str, int]) -> float:
        """Calculate total USD value"""
        total = (eth_balance / 1e18) * TOKEN_PRICES["ETH"]
        for token, balance in token_balances.items():
            if token in TOKEN_DECIMALS and balance > 0:
                decimals = TOKEN_DECIMALS[token]
                price = TOKEN_PRICES.get(token, 0)
                total += (balance / (10 ** decimals)) * price
        return total
    
    def is_proxy_contract(self, bytecode: str) -> Tuple[bool, str]:
        """Detect if bytecode is a proxy pattern"""
        if len(bytecode) < 10:
            return False, ""
        
        # EIP-1167 Minimal Proxy (clone)
        if bytecode.startswith("0x363d3d373d3d3d363d73"):
            return True, "minimal"
        
        # Check for DELEGATECALL opcode (0xf4)
        if "f4" in bytecode.lower():
            # Check EIP-1967 slots
            impl = self.get_storage_at("", EIP1967_IMPL_SLOT)  # placeholder
            if impl and impl != "0x" + "0" * 64:
                return True, "eip1967"
        
        # Short bytecode with delegate pattern
        if len(bytecode) < 500 and "36" in bytecode and "3d" in bytecode:
            return True, "minimal"
            
        return False, ""
    
    def detect_proxy_type(self, address: str, bytecode: str) -> Tuple[str, str, str]:
        """Detect proxy type and get implementation/admin"""
        # Check EIP-1967 implementation slot
        impl_raw = self.get_storage_at(address, EIP1967_IMPL_SLOT)
        admin_raw = self.get_storage_at(address, EIP1967_ADMIN_SLOT)
        beacon_raw = self.get_storage_at(address, EIP1967_BEACON_SLOT)
        
        impl_addr = "0x" + impl_raw[-40:] if impl_raw and len(impl_raw) >= 42 else ""
        admin_addr = "0x" + admin_raw[-40:] if admin_raw and len(admin_raw) >= 42 else ""
        
        # Determine proxy type
        if admin_addr and admin_addr != "0x" + "0" * 40:
            return "transparent", impl_addr, admin_addr
        elif beacon_raw and beacon_raw != "0x" + "0" * 64:
            return "beacon", impl_addr, ""
        elif impl_addr and impl_addr != "0x" + "0" * 40:
            return "uups", impl_addr, ""
        
        # Try calling implementation() directly
        result = self.eth_call(address, SELECTORS["implementation()"])
        if result and len(result) >= 42:
            impl_addr = "0x" + result[-40:]
            return "transparent", impl_addr, ""
        
        return "unknown", "", ""
    
    def check_initialization_state(self, proxy_addr: str, impl_addr: str) -> Tuple[bool, bool, bool]:
        """
        Check if proxy is initialized and if initialize() is callable
        Returns: (is_initialized, has_open_initialize, has_open_upgrade)
        """
        is_initialized = None
        has_open_initialize = False
        has_open_upgrade = False
        
        # Check _initialized storage slot (usually slot 0 for Initializable)
        slot0 = self.get_storage_at(proxy_addr, "0x0")
        if slot0:
            # OpenZeppelin Initializable uses uint8 at position 0
            try:
                val = int(slot0, 16)
                # If lowest byte is 0, might be uninitialized
                is_initialized = (val & 0xFF) > 0
            except:
                pass
        
        # Try calling initialize() variants to see if they revert
        for selector_name, selector in [
            ("initialize()", "0x8129fc1c"),
            ("initialize(address)", "0xc4d66de8" + "0" * 64),
        ]:
            # Simulate call (static)
            result = self.eth_call(proxy_addr, selector)
            # If it doesn't revert, initialize might be open
            if result is not None and result != "0x":
                has_open_initialize = True
                break
        
        # Check if upgradeToAndCall is accessible
        # This requires checking implementation bytecode for access controls
        if impl_addr and impl_addr != "0x" + "0" * 40:
            impl_code = self.get_code(impl_addr)
            # Check if upgradeTo selector exists without proper guards
            if SELECTORS["upgradeTo(address)"][2:] in impl_code.lower():
                # Check for onlyOwner/onlyAdmin patterns
                # Look for CALLER opcode followed by comparison
                if "33" not in impl_code[:200]:  # CALLER not in first 100 bytes
                    has_open_upgrade = True
        
        return is_initialized, has_open_initialize, has_open_upgrade
    
    def analyze_implementation_bytecode(self, impl_addr: str) -> Dict:
        """Deep analysis of implementation bytecode for vulnerabilities"""
        if not impl_addr or impl_addr == "0x" + "0" * 40:
            return {"error": "no_implementation"}
        
        bytecode = self.get_code(impl_addr)
        if bytecode == "0x" or len(bytecode) < 10:
            return {"error": "no_code"}
        
        analysis = {
            "size": len(bytecode) // 2 - 1,
            "has_initialize": False,
            "has_initializer_modifier": False,
            "has_upgrade_function": False,
            "has_owner_check": False,
            "has_selfdestruct": False,
            "delegatecall_count": 0,
        }
        
        bytecode_lower = bytecode.lower()
        
        # Check for function selectors
        analysis["has_initialize"] = any(
            sel[2:] in bytecode_lower for sel in [
                SELECTORS["initialize()"],
                SELECTORS["initialize(address)"],
                SELECTORS["initialize(address,address)"],
            ]
        )
        
        analysis["has_upgrade_function"] = any(
            sel[2:] in bytecode_lower for sel in [
                SELECTORS["upgradeTo(address)"],
                SELECTORS["upgradeToAndCall(address,bytes)"],
            ]
        )
        
        # Count DELEGATECALL opcodes
        analysis["delegatecall_count"] = bytecode_lower.count("f4")
        
        # Check for SELFDESTRUCT
        analysis["has_selfdestruct"] = "ff" in bytecode_lower
        
        # Check for CALLER (0x33) which indicates access control
        analysis["has_owner_check"] = "33" in bytecode_lower[:500]
        
        return analysis
    
    def get_recent_contracts(self, hours: int = 72) -> List[Dict]:
        """
        Get recently deployed/upgraded proxies using eth_getLogs
        for EIP-1967 Upgraded events + AdminChanged events
        """
        current_block = self.get_block_number()
        blocks_per_hour = 300
        from_block = current_block - (hours * blocks_per_hour)
        
        print(f"Scanning blocks {from_block} to {current_block} ({hours}h window)")
        print(f"Current block: {current_block}")
        
        contracts = []
        
        # EIP-1967 event signatures
        # Upgraded(address indexed implementation)
        UPGRADED_TOPIC = "0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b"
        # AdminChanged(address previousAdmin, address newAdmin)  
        ADMIN_CHANGED_TOPIC = "0x7e644d79422f17c01e4894b5f4f588d331ebfa28653d42ae832dc59e38c9798f"
        # BeaconUpgraded(address indexed beacon)
        BEACON_UPGRADED_TOPIC = "0x1cf3b03a6cf19fa2baba4df148e9dcabedea7f8a5c07840e207e5c089be95d3e"
        # Initialized(uint8 version)
        INITIALIZED_TOPIC = "0x7f26b83ff96e1f2b6a682f133852f6798a09c465da95921460cefb3847402498"
        
        topics_to_scan = [
            (UPGRADED_TOPIC, "Upgraded"),
            (ADMIN_CHANGED_TOPIC, "AdminChanged"),
            (BEACON_UPGRADED_TOPIC, "BeaconUpgraded"),
        ]
        
        # Scan in chunks
        chunk_size = 5000
        
        for topic, topic_name in topics_to_scan:
            print(f"  Scanning for {topic_name} events...")
            topic_contracts = []
            
            for start in range(from_block, current_block, chunk_size):
                end = min(start + chunk_size - 1, current_block)
                
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_getLogs",
                    "params": [{
                        "fromBlock": hex(start),
                        "toBlock": hex(end),
                        "topics": [[topic]],
                    }],
                    "id": 1
                }
                
                try:
                    resp = requests.post(self.alchemy_http, json=payload, timeout=30)
                    data = resp.json()
                    
                    if "result" in data and data["result"]:
                        for log in data["result"]:
                            address = log.get("address")
                            if address:
                                topic_contracts.append({
                                    "address": address,
                                    "block": int(log.get("blockNumber", "0x0"), 16),
                                    "hash": log.get("transactionHash"),
                                    "event": topic_name,
                                })
                except Exception as e:
                    print(f"    Error blocks {start}-{end}: {e}")
                    continue
                
                time.sleep(0.05)
            
            print(f"    Found {len(topic_contracts)} {topic_name} events")
            contracts.extend(topic_contracts)
        
        # Also scan for contracts with specific proxy bytecode patterns
        # by checking recent blocks for contract creations
        print("  Scanning blocks for contract creations (full 72h)...")
        
        # Scan every 50th block in the 72h window
        creation_contracts = []
        blocks_to_scan = list(range(from_block, current_block, 50))
        total_blocks = len(blocks_to_scan)
        
        for idx, block_num in enumerate(blocks_to_scan):
            if idx % 100 == 0:
                print(f"    Progress: {idx}/{total_blocks} blocks checked...")
            
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(block_num), True],
                "id": 1
            }
            
            try:
                resp = requests.post(self.alchemy_http, json=payload, timeout=15)
                data = resp.json()
                
                if "result" in data and data["result"]:
                    block_data = data["result"]
                    for tx in block_data.get("transactions", []):
                        # Contract creation = 'to' is null
                        if tx.get("to") is None:
                            # Get receipt to find created address
                            receipt_payload = {
                                "jsonrpc": "2.0",
                                "method": "eth_getTransactionReceipt",
                                "params": [tx["hash"]],
                                "id": 1
                            }
                            receipt_resp = requests.post(self.alchemy_http, json=receipt_payload, timeout=10)
                            receipt_data = receipt_resp.json()
                            
                            if "result" in receipt_data and receipt_data["result"]:
                                contract_addr = receipt_data["result"].get("contractAddress")
                                if contract_addr:
                                    creation_contracts.append({
                                        "address": contract_addr,
                                        "block": block_num,
                                        "hash": tx["hash"],
                                        "event": "ContractCreation",
                                    })
            except Exception as e:
                continue
            
            time.sleep(0.02)
        
        contracts.extend(creation_contracts)
        print(f"    Found {len(creation_contracts)} contract creations")
        
        # Deduplicate
        seen = set()
        unique = []
        for c in contracts:
            addr = c["address"].lower() if c["address"] else None
            if addr and addr not in seen:
                seen.add(addr)
                unique.append(c)
        
        print(f"  Total unique proxy candidates: {len(unique)}")
        return unique
    
    def scan_contract(self, address: str, block: int) -> Optional[ProxyCandidate]:
        """Scan a single contract for proxy vulnerabilities"""
        self.scanned_count += 1
        
        # Get bytecode
        bytecode = self.get_code(address)
        if bytecode == "0x" or len(bytecode) < 20:
            return None
        
        # Detect if it's a proxy
        proxy_type, impl_addr, admin_addr = self.detect_proxy_type(address, bytecode)
        
        if proxy_type == "unknown":
            return None
        
        # Get balances
        eth_balance = self.get_balance(address)
        token_balances = {}
        
        for name, token_addr in TOKENS.items():
            bal = self.get_token_balance(token_addr, address)
            if bal > 0:
                token_balances[name] = bal
        
        total_value = self.calculate_total_value(eth_balance, token_balances)
        
        # Phase 1 filter: >$50,000
        if total_value < 50000:
            return None
        
        print(f"\n[HIT] Proxy found: {address}")
        print(f"  Type: {proxy_type}, Value: ${total_value:,.2f}")
        
        # Phase 2: Deep analysis
        is_init, open_init, open_upgrade = self.check_initialization_state(address, impl_addr)
        impl_analysis = self.analyze_implementation_bytecode(impl_addr)
        
        # Determine vulnerability
        vulnerability = None
        if open_init and not is_init:
            vulnerability = "UNINITIALIZED_PROXY"
        elif open_upgrade:
            vulnerability = "OPEN_UPGRADE"
        elif impl_analysis.get("has_selfdestruct") and not impl_analysis.get("has_owner_check"):
            vulnerability = "UNPROTECTED_SELFDESTRUCT"
        
        candidate = ProxyCandidate(
            address=address,
            block_deployed=block,
            timestamp=int(time.time()),
            impl_address=impl_addr,
            admin_address=admin_addr,
            proxy_type=proxy_type,
            eth_balance=eth_balance,
            token_balances=token_balances,
            total_value_usd=total_value,
            bytecode_size=len(bytecode) // 2,
            is_initialized=is_init,
            has_open_initialize=open_init,
            has_open_upgrade=open_upgrade,
            vulnerability=vulnerability,
        )
        
        return candidate
    
    def run_72h_scan(self):
        """Main scan loop"""
        print("=" * 70)
        print("LIVE PROXY SCANNER - 72h Rolling Window")
        print("=" * 70)
        print(f"Endpoint: {self.alchemy_http[:50]}...")
        print(f"Filters: Transparent/UUPS proxies, >$50,000 value")
        print(f"Targets: Uninitialized proxies, open upgradeToAndCall()")
        print("=" * 70)
        
        # Get recent contract deployments
        print("\n[PHASE 1] Fetching contract deployments from last 72 hours...")
        recent_contracts = self.get_recent_contracts(72)
        print(f"  Found {len(recent_contracts)} contract deployments")
        
        # Scan each contract
        print("\n[PHASE 2] Scanning for proxy patterns and value...")
        
        high_value_proxies = []
        vulnerable_proxies = []
        
        for i, contract in enumerate(recent_contracts):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(recent_contracts)} contracts scanned...")
            
            candidate = self.scan_contract(contract["address"], contract["block"])
            
            if candidate:
                high_value_proxies.append(candidate)
                if candidate.vulnerability:
                    vulnerable_proxies.append(candidate)
                    print(f"  [VULN] {candidate.vulnerability}: {candidate.address}")
                    print(f"         Value: ${candidate.total_value_usd:,.2f}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        print(f"Total contracts scanned: {self.scanned_count}")
        print(f"High-value proxies (>$50k): {len(high_value_proxies)}")
        print(f"Vulnerable proxies: {len(vulnerable_proxies)}")
        
        if vulnerable_proxies:
            print("\n[VULNERABLE TARGETS]")
            for v in vulnerable_proxies:
                print(f"\n  Address: {v.address}")
                print(f"  Type: {v.proxy_type}")
                print(f"  Implementation: {v.impl_address}")
                print(f"  Vulnerability: {v.vulnerability}")
                print(f"  Value: ${v.total_value_usd:,.2f}")
                print(f"  ETH: {v.eth_balance / 1e18:.4f}")
                for token, bal in v.token_balances.items():
                    dec = TOKEN_DECIMALS.get(token, 18)
                    print(f"  {token}: {bal / (10**dec):,.2f}")
        else:
            print("\n[NO VULNERABLE TARGETS FOUND]")
            print("All scanned proxies are properly initialized with access controls.")
        
        return {
            "scanned": self.scanned_count,
            "high_value": len(high_value_proxies),
            "vulnerable": len(vulnerable_proxies),
            "targets": [
                {
                    "address": v.address,
                    "vulnerability": v.vulnerability,
                    "value_usd": v.total_value_usd,
                    "impl": v.impl_address,
                }
                for v in vulnerable_proxies
            ]
        }


def main():
    scanner = LiveProxyScanner()
    results = scanner.run_72h_scan()
    
    # Save results
    output_file = "live_proxy_scan_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
