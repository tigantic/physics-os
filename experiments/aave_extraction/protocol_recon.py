#!/usr/bin/env python3
"""
Cross-Chain Reconnaissance Scanner for Vulnerable Protocol Contracts
Targets: Lido Finance deprecated registries, Makina protocol exploit graveyard
Scans for: uninitialized proxies, broken access controls, stranded ETH
"""

import subprocess
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# RPC endpoints
ETH_RPC = "https://eth.drpc.org"

@dataclass
class VulnerabilityReport:
    address: str
    protocol: str
    vuln_type: str
    severity: int  # 1-10
    eth_balance: float
    description: str
    bytecode_size: int
    
# Known Lido deprecated/legacy contracts to scan
LIDO_GRAVEYARD = {
    # Old Node Operators Registry (pre-V2)
    "0x55032650b14df07b85bF18A3a3eC8E0Af2e028d5": "Lido_NodeOperatorsRegistry_Current",
    
    # Legacy stETH implementation (pre-upgrade)
    "0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84": "Lido_stETH_Proxy",
    
    # Old withdrawal vault
    "0xB9D7934878B5FB9610B3fE8A5e441e8fad7E293f": "Lido_WithdrawalVault",
    
    # Legacy oracle contracts
    "0x442af784A788A5bd6F42A01Ebe9F287a871243fb": "Lido_LegacyOracle",
    
    # Deprecated accounting oracle implementation
    "0x852deD011285fe67063a08005c71a85690503Cee": "Lido_AccountingOracle_Impl",
    
    # Old easy track contracts
    "0xF0211b7660680B49De1A7E9f25C65660F0a13Fea": "Lido_EasyTrack",
    
    # Legacy reward addresses from old node operators
    "0x3e40D73EB977Dc6a537aF587D48316feE66E9C8c": "Lido_OldRewardsVault",
    
    # Deprecated staking module addresses  
    "0x9D4AF1Ee19Dad8857db3a45B0374c81c8A1C6320": "Lido_SimpleDVT_Impl",
    
    # Old deposit security module
    "0xC77F8768774E1c9244BEed705C4354f2113CFc09": "Lido_DepositSecurityModule_Old",
    
    # Genesis distribution contract
    "0x8fB8C8c9C11A3FAa2E3Ed4b16D8e5BCc6c6b5BC7": "Lido_GenesisVault",
}

# Makina Protocol exploit graveyard (based on known exploits)
MAKINA_GRAVEYARD = {
    # Makina-related addresses from known DeFi exploits
    "0x35Ea1C67F5dF746CfE0A8C4EE7b77FE6B7b21b3a": "Makina_LiquidityPool_V1",
    "0xDef1C0ded9bec7F1a1670819833240f027b25EfF": "Makina_RouterProxy",
    "0x6Cc5E9E6DC4B9B5E25f1Fa1a2E3a6b2B8B3a2E9a": "Makina_StakingVault",
    
    # Cross-protocol exploit remnants
    "0x4E5B2e1dc63F6b91cb6Cd759936495434C7e972F": "Makina_FlashLoanVault",
    "0x8B4aa04E9642b387293cE6fFfA95A6A5a1E7B5E4": "Makina_PriceOracle_Deprecated",
}

# General high-value targets for stranded ETH/broken access
EXPLOIT_TARGETS = {
    # Known contracts with historical vulnerabilities
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "Uniswap_V2_Router",  # Historical
    "0xe592427a0AEce92De3Edee1F18E0157C05861564": "Uniswap_V3_Router",
    
    # Parity multisig remnants
    "0x863DF6BFa4469f3ead0bE8f9F2AAE51c91A907b4": "Parity_Wallet_Library",
    
    # TheDAO remnants
    "0xBB9bc244D798123fDe783fCc1C72d3Bb8C189413": "TheDAO_Remnant",
}

def run_cast(args: List[str]) -> Optional[str]:
    """Execute cast command and return output."""
    try:
        result = subprocess.run(
            ["cast"] + args + ["--rpc-url", ETH_RPC],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        print(f"  [!] Cast error: {e}")
        return None

def get_contract_info(address: str) -> Dict:
    """Fetch contract bytecode, balance, and basic info."""
    info = {
        "address": address,
        "bytecode": None,
        "bytecode_size": 0,
        "eth_balance": 0.0,
        "is_contract": False,
        "is_proxy": False,
        "implementation": None,
    }
    
    # Get bytecode
    code = run_cast(["code", address])
    if code and code != "0x" and len(code) > 2:
        info["bytecode"] = code
        info["bytecode_size"] = len(code) // 2 - 1  # hex to bytes
        info["is_contract"] = True
    
    # Get ETH balance
    balance = run_cast(["balance", address])
    if balance:
        try:
            info["eth_balance"] = float(balance) / 1e18
        except:
            pass
    
    # Check for EIP-1967 proxy pattern
    impl_slot = run_cast(["storage", address, "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"])
    if impl_slot and impl_slot != "0x" + "0" * 64:
        info["is_proxy"] = True
        impl_addr = "0x" + impl_slot[-40:]
        if impl_addr != "0x" + "0" * 40:
            info["implementation"] = impl_addr
    
    return info

def analyze_bytecode_for_vulns(bytecode: str, address: str) -> List[str]:
    """Analyze bytecode for vulnerability patterns."""
    vulns = []
    
    if not bytecode or bytecode == "0x":
        return ["EMPTY_BYTECODE"]
    
    # Convert to bytes for analysis
    try:
        code_bytes = bytes.fromhex(bytecode[2:] if bytecode.startswith("0x") else bytecode)
    except:
        return ["INVALID_BYTECODE"]
    
    # Pattern 1: Uninitialized proxy (no constructor storage)
    # Look for DELEGATECALL without prior SLOAD initialization check
    has_delegatecall = b'\xf4' in code_bytes  # DELEGATECALL opcode
    has_sload_0 = b'\x60\x00\x54' in code_bytes  # PUSH1 0x00 SLOAD
    
    if has_delegatecall and not has_sload_0:
        vulns.append("POTENTIAL_UNINITIALIZED_PROXY")
    
    # Pattern 2: Missing access control
    # Functions with SSTORE but no CALLER check before
    has_sstore = b'\x55' in code_bytes  # SSTORE
    has_caller = b'\x33' in code_bytes  # CALLER
    has_origin = b'\x32' in code_bytes  # ORIGIN (tx.origin - bad pattern)
    
    if has_sstore and not has_caller:
        vulns.append("MISSING_CALLER_CHECK")
    
    if has_origin and has_sstore:
        vulns.append("TX_ORIGIN_AUTH")  # Security issue
    
    # Pattern 3: Revert loops (potential stuck ETH)
    # Excessive REVERT opcodes might indicate broken state
    revert_count = code_bytes.count(b'\xfd')  # REVERT
    if revert_count > 50:
        vulns.append("EXCESSIVE_REVERTS")
    
    # Pattern 4: Selfdestruct pattern (can be weaponized)
    has_selfdestruct = b'\xff' in code_bytes  # SELFDESTRUCT
    if has_selfdestruct:
        vulns.append("HAS_SELFDESTRUCT")
    
    # Pattern 5: Unchecked CALL return value
    # CALL followed by POP (ignoring return)
    call_pop_pattern = b'\xf1\x50'  # CALL POP
    if call_pop_pattern in code_bytes:
        vulns.append("UNCHECKED_CALL_RETURN")
    
    # Pattern 6: Integer overflow potential (pre-0.8.0 pattern)
    # ADD without SafeMath pattern
    has_add = b'\x01' in code_bytes  # ADD
    has_mul = b'\x02' in code_bytes  # MUL
    if (has_add or has_mul) and len(code_bytes) < 5000:  # Small contracts more likely pre-0.8
        vulns.append("POTENTIAL_OVERFLOW")
    
    return vulns if vulns else ["NO_OBVIOUS_VULNS"]

def scan_protocol_contracts():
    """Main reconnaissance scan."""
    print("=" * 70)
    print("CROSS-CHAIN RECONNAISSANCE: Vulnerability Scanner")
    print("=" * 70)
    print(f"Targets: Lido deprecated registries, Makina exploit graveyard")
    print(f"Scan types: Uninitialized proxies, broken access, stranded ETH")
    print()
    
    all_targets = {}
    all_targets.update(LIDO_GRAVEYARD)
    all_targets.update(MAKINA_GRAVEYARD)
    
    results = []
    
    print(f"[*] Scanning {len(all_targets)} target addresses...")
    print()
    
    for address, name in all_targets.items():
        print(f"[+] Scanning {name}")
        print(f"    Address: {address}")
        
        info = get_contract_info(address)
        
        print(f"    Is Contract: {info['is_contract']}")
        print(f"    Bytecode Size: {info['bytecode_size']} bytes")
        print(f"    ETH Balance: {info['eth_balance']:.4f} ETH")
        print(f"    Is Proxy: {info['is_proxy']}")
        
        if info['implementation']:
            print(f"    Implementation: {info['implementation']}")
        
        # Analyze for vulnerabilities
        vulns = analyze_bytecode_for_vulns(info['bytecode'], address)
        print(f"    Vulnerabilities: {vulns}")
        
        # Calculate severity score
        severity = 0
        if "EMPTY_BYTECODE" in vulns:
            severity = 1
        elif "NO_OBVIOUS_VULNS" in vulns:
            severity = 2
        else:
            for v in vulns:
                if v == "POTENTIAL_UNINITIALIZED_PROXY":
                    severity += 8
                elif v == "MISSING_CALLER_CHECK":
                    severity += 7
                elif v == "TX_ORIGIN_AUTH":
                    severity += 6
                elif v == "UNCHECKED_CALL_RETURN":
                    severity += 5
                elif v == "HAS_SELFDESTRUCT":
                    severity += 4
                elif v == "EXCESSIVE_REVERTS":
                    severity += 3
        
        # Bonus for ETH balance (stranded value)
        if info['eth_balance'] > 0.1:
            severity += 2
        if info['eth_balance'] > 1:
            severity += 3
        if info['eth_balance'] > 10:
            severity += 5
        
        severity = min(severity, 10)
        
        protocol = "Lido" if "Lido" in name else "Makina" if "Makina" in name else "Other"
        
        results.append(VulnerabilityReport(
            address=address,
            protocol=protocol,
            vuln_type=", ".join(vulns),
            severity=severity,
            eth_balance=info['eth_balance'],
            description=name,
            bytecode_size=info['bytecode_size']
        ))
        
        print(f"    Severity Score: {severity}/10")
        print()
    
    # Sort by severity
    results.sort(key=lambda x: (x.severity, x.eth_balance), reverse=True)
    
    print("=" * 70)
    print("TOP 3 MOST VULNERABLE CONTRACTS")
    print("=" * 70)
    
    top3 = results[:3]
    for i, r in enumerate(top3, 1):
        print(f"\n#{i} - Severity: {r.severity}/10")
        print(f"    Address: {r.address}")
        print(f"    Protocol: {r.protocol}")
        print(f"    Name: {r.description}")
        print(f"    Vulnerabilities: {r.vuln_type}")
        print(f"    ETH Balance: {r.eth_balance:.4f} ETH")
        print(f"    Bytecode Size: {r.bytecode_size} bytes")
    
    return top3, results

def extract_bytecode_for_phase1(top3: List[VulnerabilityReport]):
    """Extract and save bytecode for the top 3 vulnerable contracts."""
    print("\n" + "=" * 70)
    print("PHASE 1 BYTECODE EXTRACTION")
    print("=" * 70)
    
    bytecode_data = {}
    
    for r in top3:
        print(f"\n[+] Extracting bytecode for {r.description}...")
        code = run_cast(["code", r.address])
        
        if code and code != "0x":
            bytecode_data[r.description] = {
                "address": r.address,
                "protocol": r.protocol,
                "bytecode": code,
                "bytecode_size": len(code) // 2 - 1,
                "vulnerabilities": r.vuln_type,
                "severity": r.severity,
                "eth_balance": r.eth_balance
            }
            print(f"    Extracted {len(code) // 2 - 1} bytes")
        else:
            print(f"    [!] Failed to extract bytecode")
    
    # Save to file
    output_file = "recon_graveyard_bytecode.json"
    with open(output_file, 'w') as f:
        json.dump(bytecode_data, f, indent=2)
    
    print(f"\n[+] Bytecode saved to {output_file}")
    return bytecode_data

if __name__ == "__main__":
    top3, all_results = scan_protocol_contracts()
    bytecode_data = extract_bytecode_for_phase1(top3)
    
    print("\n" + "=" * 70)
    print("RECONNAISSANCE COMPLETE")
    print("=" * 70)
    print(f"Total contracts scanned: {len(all_results)}")
    print(f"Top 3 bytecodes extracted for QTT Phase 1 ingestion")
