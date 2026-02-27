#!/usr/bin/env python3
"""
High-Value Proxy Hunter - Scans for high-value contracts with proxy patterns
Looks for:
1. Recent large token transfers to proxy-pattern contracts
2. Proxies with uninitialized state
3. UUPS with missing upgrade access controls
"""

import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

ALCHEMY_HTTP = "https://eth-mainnet.g.alchemy.com/v2/XiHwe3EP0GeLvzFi-e-im"

# EIP-1967 slots
EIP1967_IMPL_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
EIP1967_ADMIN_SLOT = "0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103"

# Major tokens
TOKENS = {
    "USDC": ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6),
    "USDT": ("0xdAC17F958D2ee523a2206206994597C13D831ec7", 6),
    "WETH": ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
    "DAI": ("0x6B175474E89094C44Da98b954EescdeCB5C811111", 18),
}

# Function selectors
INITIALIZE_SELECTORS = [
    "0x8129fc1c",  # initialize()
    "0xc4d66de8",  # initialize(address)
    "0x485cc955",  # initialize(address,address)
]

@dataclass
class ProxyAnalysis:
    address: str
    is_proxy: bool
    proxy_type: str
    impl_address: str
    admin_address: str
    eth_balance_wei: int
    eth_balance_eth: float
    usdc_balance: float
    total_value_usd: float
    initialized_slot0: int
    has_initialize_selector: bool
    initialize_reverts: bool
    upgrade_protected: bool
    vulnerability: Optional[str]
    
class HighValueProxyHunter:
    def __init__(self):
        self.http = ALCHEMY_HTTP
        self.results = []
        
    def rpc_call(self, method: str, params: list) -> Optional[dict]:
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        try:
            resp = requests.post(self.http, json=payload, timeout=15)
            return resp.json()
        except:
            return None
    
    def get_storage(self, addr: str, slot: str) -> str:
        result = self.rpc_call("eth_getStorageAt", [addr, slot, "latest"])
        return result.get("result", "0x0") if result else "0x0"
    
    def get_code(self, addr: str) -> str:
        result = self.rpc_call("eth_getCode", [addr, "latest"])
        return result.get("result", "0x") if result else "0x"
    
    def get_balance(self, addr: str) -> int:
        result = self.rpc_call("eth_getBalance", [addr, "latest"])
        return int(result.get("result", "0x0"), 16) if result else 0
    
    def eth_call(self, to: str, data: str) -> Optional[str]:
        result = self.rpc_call("eth_call", [{"to": to, "data": data}, "latest"])
        if result and "result" in result:
            return result["result"]
        return None
    
    def get_token_balance(self, token: str, holder: str) -> int:
        data = "0x70a08231" + holder[2:].lower().zfill(64)
        result = self.eth_call(token, data)
        if result and result != "0x":
            try:
                return int(result, 16)
            except:
                pass
        return 0
    
    def is_proxy(self, bytecode: str, address: str) -> Tuple[bool, str, str, str]:
        """Check if contract is a proxy and get implementation details"""
        if len(bytecode) < 20:
            return False, "", "", ""
        
        # Check EIP-1967 implementation slot
        impl_raw = self.get_storage(address, EIP1967_IMPL_SLOT)
        admin_raw = self.get_storage(address, EIP1967_ADMIN_SLOT)
        
        impl_addr = ""
        admin_addr = ""
        proxy_type = ""
        
        if impl_raw and impl_raw != "0x" + "0"*64:
            impl_addr = "0x" + impl_raw[-40:]
            if impl_addr != "0x" + "0"*40:
                proxy_type = "eip1967"
        
        if admin_raw and admin_raw != "0x" + "0"*64:
            admin_addr = "0x" + admin_raw[-40:]
            if admin_addr != "0x" + "0"*40:
                proxy_type = "transparent"
        
        # Check for minimal proxy pattern
        if bytecode.lower().startswith("0x363d3d373d3d3d363d73"):
            proxy_type = "minimal"
            # Extract implementation from bytecode
            impl_addr = "0x" + bytecode[22:62]
        
        # Check for DELEGATECALL
        if "f4" in bytecode.lower() and proxy_type == "":
            proxy_type = "custom_delegate"
        
        is_proxy = proxy_type != ""
        
        if not proxy_type and impl_addr == "":
            return False, "", "", ""
        
        return is_proxy, proxy_type, impl_addr, admin_addr
    
    def check_initialization(self, proxy_addr: str, impl_addr: str, proxy_bytecode: str) -> Tuple[int, bool, bool]:
        """
        Check initialization state
        Returns: (slot0_value, has_initialize, initialize_reverts)
        """
        # Read slot 0 (common initializable storage)
        slot0 = self.get_storage(proxy_addr, "0x0")
        slot0_val = int(slot0, 16) if slot0 else 0
        
        # Check if initialize selector exists in implementation
        has_initialize = False
        impl_code = self.get_code(impl_addr) if impl_addr else proxy_bytecode
        impl_code_lower = impl_code.lower()
        
        for sel in INITIALIZE_SELECTORS:
            if sel[2:] in impl_code_lower:
                has_initialize = True
                break
        
        # Try calling initialize() to see if it reverts
        initialize_reverts = True
        for sel in INITIALIZE_SELECTORS:
            padded = sel if sel == "0x8129fc1c" else sel + "0"*64
            result = self.eth_call(proxy_addr, padded)
            if result is not None:
                # Didn't revert - potential vulnerability
                initialize_reverts = False
                break
        
        return slot0_val, has_initialize, initialize_reverts
    
    def check_upgrade_protection(self, impl_addr: str) -> bool:
        """Check if upgrade functions have access control"""
        if not impl_addr or impl_addr == "0x" + "0"*40:
            return True  # No implementation to check
        
        impl_code = self.get_code(impl_addr)
        if len(impl_code) < 20:
            return True
        
        impl_lower = impl_code.lower()
        
        # Check for upgradeTo/upgradeToAndCall selectors
        has_upgrade = "3659cfe6" in impl_lower or "4f1ef286" in impl_lower
        if not has_upgrade:
            return True  # No upgrade function
        
        # Check for CALLER opcode (0x33) near the selector - indicates access control
        # This is a heuristic - real analysis would need full CFG
        upgrade_idx = impl_lower.find("3659cfe6")
        if upgrade_idx == -1:
            upgrade_idx = impl_lower.find("4f1ef286")
        
        if upgrade_idx != -1:
            # Check surrounding bytes for CALLER (0x33) which suggests msg.sender check
            surrounding = impl_lower[max(0, upgrade_idx-100):upgrade_idx+100]
            if "33" in surrounding:
                return True  # Has CALLER check
        
        return False
    
    def analyze_contract(self, address: str) -> Optional[ProxyAnalysis]:
        """Full analysis of a contract"""
        bytecode = self.get_code(address)
        if bytecode == "0x" or len(bytecode) < 20:
            return None
        
        is_proxy, proxy_type, impl_addr, admin_addr = self.is_proxy(bytecode, address)
        
        if not is_proxy:
            return None
        
        # Get balances
        eth_balance = self.get_balance(address)
        eth_balance_eth = eth_balance / 1e18
        
        usdc_addr, usdc_dec = TOKENS["USDC"]
        usdc_balance = self.get_token_balance(usdc_addr, address) / (10**usdc_dec)
        
        # Calculate total value (~$2800/ETH estimate)
        total_value = (eth_balance_eth * 2800) + usdc_balance
        
        # Get other major tokens
        for name, (token_addr, dec) in TOKENS.items():
            if name != "USDC":
                bal = self.get_token_balance(token_addr, address)
                if bal > 0:
                    if name == "WETH":
                        total_value += (bal / 1e18) * 2800
                    elif name == "USDT":
                        total_value += bal / 1e6
                    elif name == "DAI":
                        total_value += bal / 1e18
        
        # Check initialization
        slot0_val, has_init, init_reverts = self.check_initialization(address, impl_addr, bytecode)
        
        # Check upgrade protection
        upgrade_protected = self.check_upgrade_protection(impl_addr)
        
        # Determine vulnerability
        vulnerability = None
        
        # Uninitialized: has initialize, slot0 is 0, and initialize doesn't revert
        if has_init and slot0_val == 0 and not init_reverts:
            vulnerability = "UNINITIALIZED_PROXY"
        # Open upgrade: upgrade function exists without access control
        elif not upgrade_protected:
            vulnerability = "UNPROTECTED_UPGRADE"
        
        return ProxyAnalysis(
            address=address,
            is_proxy=True,
            proxy_type=proxy_type,
            impl_address=impl_addr,
            admin_address=admin_addr,
            eth_balance_wei=eth_balance,
            eth_balance_eth=eth_balance_eth,
            usdc_balance=usdc_balance,
            total_value_usd=total_value,
            initialized_slot0=slot0_val,
            has_initialize_selector=has_init,
            initialize_reverts=init_reverts,
            upgrade_protected=upgrade_protected,
            vulnerability=vulnerability,
        )
    
    def find_high_value_transfers(self, hours: int = 72, min_value_usd: float = 50000) -> List[str]:
        """Find addresses that received large token transfers recently"""
        current_block = self.rpc_call("eth_blockNumber", [])
        if not current_block:
            return []
        
        current = int(current_block["result"], 16)
        from_block = current - (hours * 300)
        
        print(f"Scanning for high-value transfers in blocks {from_block}-{current}")
        
        recipients = set()
        
        # Scan USDC transfers > $50k (50,000 * 1e6)
        usdc_addr = TOKENS["USDC"][0]
        transfer_topic = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        
        chunk_size = 10000
        for start in range(from_block, current, chunk_size):
            end = min(start + chunk_size - 1, current)
            
            result = self.rpc_call("eth_getLogs", [{
                "fromBlock": hex(start),
                "toBlock": hex(end),
                "address": usdc_addr,
                "topics": [transfer_topic],
            }])
            
            if result and "result" in result:
                for log in result["result"]:
                    # Decode transfer: topic[1]=from, topic[2]=to, data=amount
                    if len(log.get("topics", [])) >= 3 and log.get("data"):
                        to_addr = "0x" + log["topics"][2][-40:]
                        try:
                            amount = int(log["data"], 16)
                            if amount >= min_value_usd * 1e6:  # USDC has 6 decimals
                                recipients.add(to_addr)
                        except:
                            pass
            
            time.sleep(0.05)
        
        print(f"  Found {len(recipients)} addresses receiving >${min_value_usd:,.0f} USDC")
        
        # Also check for large ETH transfers (not available via logs, skip for now)
        
        return list(recipients)
    
    def run_hunt(self):
        """Main hunting routine"""
        print("=" * 70)
        print("HIGH-VALUE PROXY HUNTER")
        print("=" * 70)
        print("Mode: Scanning for high-value transfers to proxy contracts")
        print("Threshold: >$50,000")
        print("Targets: Uninitialized proxies, unprotected upgradeToAndCall()")
        print("=" * 70)
        
        # Phase 1: Find high-value transfer recipients
        print("\n[PHASE 1] Finding high-value transfer recipients...")
        candidates = self.find_high_value_transfers(72, 50000)
        
        if not candidates:
            print("  No high-value transfer recipients found")
            # Fall back to scanning known high-value protocols
            print("  Falling back to known protocol scan...")
            candidates = self.get_known_protocol_proxies()
        
        # Phase 2: Analyze each candidate
        print(f"\n[PHASE 2] Analyzing {len(candidates)} candidates...")
        
        vulnerable = []
        high_value_proxies = []
        
        for i, addr in enumerate(candidates):
            if i % 10 == 0 and i > 0:
                print(f"  Progress: {i}/{len(candidates)}")
            
            analysis = self.analyze_contract(addr)
            
            if analysis:
                if analysis.total_value_usd >= 50000:
                    high_value_proxies.append(analysis)
                    print(f"\n  [PROXY] {addr}")
                    print(f"    Type: {analysis.proxy_type}")
                    print(f"    Value: ${analysis.total_value_usd:,.2f}")
                    print(f"    Initialized: {analysis.initialized_slot0 > 0}")
                    print(f"    Upgrade protected: {analysis.upgrade_protected}")
                    
                    if analysis.vulnerability:
                        vulnerable.append(analysis)
                        print(f"    [VULN] {analysis.vulnerability}")
            
            time.sleep(0.02)
        
        # Summary
        print("\n" + "=" * 70)
        print("HUNT COMPLETE")
        print("=" * 70)
        print(f"Candidates scanned: {len(candidates)}")
        print(f"High-value proxies: {len(high_value_proxies)}")
        print(f"Vulnerable proxies: {len(vulnerable)}")
        
        if vulnerable:
            print("\n" + "=" * 70)
            print("VULNERABLE TARGETS")
            print("=" * 70)
            for v in vulnerable:
                print(f"\nAddress: {v.address}")
                print(f"Type: {v.proxy_type}")
                print(f"Implementation: {v.impl_address}")
                print(f"Value: ${v.total_value_usd:,.2f}")
                print(f"Vulnerability: {v.vulnerability}")
                print(f"ETH: {v.eth_balance_eth:.4f}")
                print(f"USDC: ${v.usdc_balance:,.2f}")
                
                if v.vulnerability == "UNINITIALIZED_PROXY":
                    print(f"\n[EXPLOIT PATH]")
                    print(f"1. Call initialize() on proxy {v.address}")
                    print(f"2. Set self as owner/admin")
                    print(f"3. Execute privileged drain function")
                
                elif v.vulnerability == "UNPROTECTED_UPGRADE":
                    print(f"\n[EXPLOIT PATH]")
                    print(f"1. Deploy malicious implementation with drain()")
                    print(f"2. Call upgradeToAndCall(malicious, drain_calldata)")
                    print(f"3. Funds transferred in same transaction")
        else:
            print("\n[NO VULNERABLE TARGETS]")
            print("All analyzed proxies are properly secured.")
        
        # Save results
        results = {
            "scanned": len(candidates),
            "high_value": len(high_value_proxies),
            "vulnerable": len(vulnerable),
            "targets": [asdict(v) for v in vulnerable],
            "all_proxies": [asdict(p) for p in high_value_proxies],
        }
        
        with open("proxy_hunt_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\nResults saved to proxy_hunt_results.json")
        return results
    
    def get_known_protocol_proxies(self) -> List[str]:
        """Fallback list of known protocol proxies to scan"""
        # Recently deployed significant proxies from major protocols
        return [
            # Aave V3 L1
            "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",  # Pool
            "0x64b761D848206f447Fe2dd461b0c635Ec39EbB27",  # PoolConfigurator
            # Compound III
            "0xc3d688B66703497DAA19211EEdff47f25384cdc3",  # cUSDCv3
            # Uniswap Universal Router
            "0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD",
            # 1inch AggregationRouter
            "0x1111111254EEB25477B68fb85Ed929f73A960582",
            # Safe Proxy Factory recent deployments
            "0xa6B71E26C5e0845f74c812102Ca7114b6a896AB2",
            # Maker DSProxy
            "0x5ef30b9986345249bc32d8928B7ee64DE9435E39",
            # Lido stETH
            "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84",
            # Rocket Pool
            "0xae78736Cd615f374D3085123A210448E74Fc6393",
        ]


def main():
    hunter = HighValueProxyHunter()
    hunter.run_hunt()


if __name__ == "__main__":
    main()
