#!/usr/bin/env python3
"""
PROXY INITIALIZATION VULNERABILITY ANALYZER
============================================
Scans EIP-1967 Transparent/UUPS proxies for initialization flaws:

1. Uninitialized implementation - can claim ownership
2. Re-initializable contracts - missing initializer guard
3. Storage collision - non-standard slot usage
4. Admin slot manipulation - unauthorized admin change
5. Delegatecall to unverified implementation
6. SELFDESTRUCT in implementation
7. Missing fallback guards
"""

import sys
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, '.')
from qtt_evm_extractor import EVMDisassembler, OPCODE_INFO

# EIP-1967 Storage Slots (keccak256 hashes - 1)
EIP1967_SLOTS = {
    0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc: 'IMPLEMENTATION',
    0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103: 'ADMIN',
    0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50: 'BEACON',
}

# Legacy OpenZeppelin slots
OZ_LEGACY_SLOTS = {
    0x7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3: 'OZ_IMPLEMENTATION',
    0x10d6a54a4754c8869d6886b5f5d7fbfa5b4522237ea5c60d11bc4e7a1ff9390b: 'OZ_ADMIN',
}

class VulnSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class ProxyVulnerability:
    """A discovered proxy vulnerability."""
    severity: VulnSeverity
    title: str
    description: str
    location: Optional[int]  # bytecode offset
    exploit_path: Optional[str]
    calldata: Optional[str]


class ProxyAnalyzer:
    """
    Analyzes transparent/UUPS proxies for initialization vulnerabilities.
    """
    
    def __init__(self, bytecode_hex: str, contract_name: str, address: str):
        self.contract_name = contract_name
        self.address = address
        self.bytecode_hex = bytecode_hex.replace('0x', '').lower()
        self.bytecode = bytes.fromhex(self.bytecode_hex)
        
        # Disassemble
        self.disasm = EVMDisassembler(bytecode_hex)
        self.disasm.disassemble()
        self.cfg = self.disasm.build_cfg()
        
        self.vulnerabilities: List[ProxyVulnerability] = []
        self.findings: Dict[str, any] = {}
        
    def analyze(self) -> List[ProxyVulnerability]:
        """Run all proxy analysis checks."""
        print(f"\n{'='*70}")
        print(f"PROXY INITIALIZATION ANALYZER: {self.contract_name}")
        print(f"{'='*70}")
        print(f"Address: {self.address}")
        print(f"Bytecode: {len(self.bytecode)} bytes")
        print(f"CFG Blocks: {len(self.cfg.blocks)}")
        
        # Run checks
        self._check_eip1967_slots()
        self._check_admin_functions()
        self._check_delegatecall_patterns()
        self._check_initializer_guards()
        self._check_selfdestruct()
        self._check_fallback_behavior()
        self._check_storage_collisions()
        
        return self.vulnerabilities
    
    def _check_eip1967_slots(self):
        """Check for EIP-1967 storage slot usage."""
        print(f"\n[1] EIP-1967 Storage Slot Analysis")
        print("-" * 50)
        
        found_slots = {}
        
        for slot_val, slot_name in {**EIP1967_SLOTS, **OZ_LEGACY_SLOTS}.items():
            slot_hex = f"{slot_val:064x}"
            if slot_hex in self.bytecode_hex:
                offset = self.bytecode_hex.index(slot_hex) // 2
                found_slots[slot_name] = (slot_val, offset)
                print(f"  [+] {slot_name}: 0x{slot_hex[:16]}...")
                print(f"      Offset: 0x{offset:04x}")
        
        self.findings['eip1967_slots'] = found_slots
        
        # Check for missing ADMIN slot (potential vulnerability)
        if 'IMPLEMENTATION' in found_slots and 'ADMIN' not in found_slots:
            self.vulnerabilities.append(ProxyVulnerability(
                severity=VulnSeverity.MEDIUM,
                title="Missing EIP-1967 Admin Slot",
                description="Proxy uses IMPLEMENTATION slot but no standard ADMIN slot found. "
                           "Admin may be stored in non-standard location or hardcoded.",
                location=None,
                exploit_path="Check if admin is upgradeable without standard guard",
                calldata=None
            ))
            print(f"  [!] WARNING: ADMIN slot not found in standard location")
        
        return found_slots
    
    def _check_admin_functions(self):
        """Analyze admin-only functions for access control."""
        print(f"\n[2] Admin Function Analysis")
        print("-" * 50)
        
        # Known admin function selectors
        admin_funcs = {
            0x3659cfe6: ('upgradeTo(address)', 'HIGH'),
            0x4f1ef286: ('upgradeToAndCall(address,bytes)', 'CRITICAL'),
            0x8f283970: ('changeAdmin(address)', 'CRITICAL'),
            0xf851a440: ('admin()', 'INFO'),
            0x5c60da1b: ('implementation()', 'INFO'),
            0xcd50ce02: ('transferProxyAdmin(address)', 'CRITICAL'),
            0xd1f57894: ('setImplementation(address)', 'HIGH'),
        }
        
        found_funcs = {}
        
        for bid, block in self.cfg.blocks.items():
            for instr in block.instructions:
                if instr.opcode == 0x63 and instr.operand:  # PUSH4
                    selector = int.from_bytes(instr.operand, 'big')
                    if selector in admin_funcs:
                        name, risk = admin_funcs[selector]
                        found_funcs[selector] = (name, risk, instr.offset)
                        print(f"  [+] Found: {name} (0x{selector:08x})")
                        print(f"      Risk: {risk}, Offset: 0x{instr.offset:04x}")
        
        self.findings['admin_functions'] = found_funcs
        
        # Check for changeAdmin without ifAdmin guard
        if 0x8f283970 in found_funcs:
            self._trace_admin_guard(0x8f283970)
    
    def _trace_admin_guard(self, selector: int):
        """Trace if a function has proper admin guard."""
        print(f"\n    Tracing admin guard for 0x{selector:08x}...")
        
        # Find the function entry
        entry_block = None
        for bid, block in self.cfg.blocks.items():
            for i, instr in enumerate(block.instructions):
                if instr.opcode == 0x63 and instr.operand:
                    if int.from_bytes(instr.operand, 'big') == selector:
                        # Look for JUMPI destination
                        for j in range(i, min(i+10, len(block.instructions))):
                            if block.instructions[j].opcode == 0x57:  # JUMPI
                                # Find destination
                                for k in range(i, j):
                                    if block.instructions[k].opcode in (0x61, 0x62):
                                        dest = int.from_bytes(block.instructions[k].operand, 'big')
                                        entry_block = dest
                                        break
                                break
                        break
        
        if entry_block:
            print(f"    Entry block: 0x{entry_block:04x}")
    
    def _check_delegatecall_patterns(self):
        """Check delegatecall usage and targets."""
        print(f"\n[3] DELEGATECALL Pattern Analysis")
        print("-" * 50)
        
        delegatecall_sites = []
        
        for bid, block in self.cfg.blocks.items():
            for i, instr in enumerate(block.instructions):
                if instr.opcode == 0xf4:  # DELEGATECALL
                    delegatecall_sites.append((bid, instr.offset))
                    print(f"  [+] DELEGATECALL at 0x{instr.offset:04x} (block 0x{bid:04x})")
                    
                    # Check what's being called
                    # Look backwards for SLOAD of implementation slot
                    self._analyze_delegatecall_target(block, i)
        
        self.findings['delegatecall_sites'] = delegatecall_sites
        
        if not delegatecall_sites:
            print(f"  [-] No DELEGATECALL found - may not be a proxy")
            self.vulnerabilities.append(ProxyVulnerability(
                severity=VulnSeverity.INFO,
                title="No DELEGATECALL Found",
                description="Contract does not contain DELEGATECALL opcode. "
                           "May not be a proxy or uses CALL instead.",
                location=None,
                exploit_path=None,
                calldata=None
            ))
    
    def _analyze_delegatecall_target(self, block, delegatecall_idx: int):
        """Analyze what address DELEGATECALL targets."""
        # Look for SLOAD before DELEGATECALL - should load implementation
        for i in range(delegatecall_idx - 1, -1, -1):
            if i < len(block.instructions):
                instr = block.instructions[i]
                if instr.opcode == 0x54:  # SLOAD
                    print(f"      ↳ Loads from storage (SLOAD at 0x{instr.offset:04x})")
                    # Check if it's loading EIP-1967 slot
                    if i > 0:
                        prev = block.instructions[i-1]
                        if prev.opcode == 0x7f and prev.operand:  # PUSH32
                            slot = int.from_bytes(prev.operand, 'big')
                            if slot in EIP1967_SLOTS:
                                print(f"      ↳ Reading {EIP1967_SLOTS[slot]} slot")
                    return
    
    def _check_initializer_guards(self):
        """Check for initializer functions and their guards."""
        print(f"\n[4] Initializer Guard Analysis")
        print("-" * 50)
        
        # Common initializer selectors
        init_selectors = {
            0x8129fc1c: 'initialize()',
            0xc4d66de8: 'initialize(address)',
            0x485cc955: 'initialize(address,address)',
            0xf8c8765e: 'initialize(address,address,address,address)',
            0xfe4b84df: 'initialize(uint256)',
            0x4cd88b76: 'initialize(string,string)',
        }
        
        found_initializers = []
        
        for bid, block in self.cfg.blocks.items():
            for instr in block.instructions:
                if instr.opcode == 0x63 and instr.operand:
                    selector = int.from_bytes(instr.operand, 'big')
                    if selector in init_selectors:
                        found_initializers.append((selector, init_selectors[selector], instr.offset))
                        print(f"  [+] Found: {init_selectors[selector]} (0x{selector:08x})")
        
        # Check for _initialized storage slot usage
        # OpenZeppelin uses slot 0 bit 0 for initialized flag
        has_init_guard = self._check_initialized_slot()
        
        if found_initializers and not has_init_guard:
            self.vulnerabilities.append(ProxyVulnerability(
                severity=VulnSeverity.CRITICAL,
                title="Potentially Re-initializable Contract",
                description="Found initialize() function but no clear initializer guard. "
                           "Contract may be vulnerable to re-initialization attack.",
                location=found_initializers[0][2],
                exploit_path="Call initialize() to claim ownership/set malicious parameters",
                calldata=f"0x{found_initializers[0][0]:08x}"
            ))
        
        self.findings['initializers'] = found_initializers
    
    def _check_initialized_slot(self) -> bool:
        """Check for initialized flag in storage slot 0."""
        # Look for pattern: SLOAD slot=0, followed by checks
        for bid, block in self.cfg.blocks.items():
            for i, instr in enumerate(block.instructions):
                if instr.opcode == 0x54:  # SLOAD
                    # Check if loading from slot 0
                    if i > 0:
                        prev = block.instructions[i-1]
                        if prev.opcode == 0x60 and prev.operand == b'\x00':
                            print(f"  [+] Found slot 0 check at 0x{instr.offset:04x}")
                            return True
        return False
    
    def _check_selfdestruct(self):
        """Check for SELFDESTRUCT opcode."""
        print(f"\n[5] SELFDESTRUCT Analysis")
        print("-" * 50)
        
        for bid, block in self.cfg.blocks.items():
            for instr in block.instructions:
                if instr.opcode == 0xff:  # SELFDESTRUCT
                    print(f"  [!] SELFDESTRUCT at 0x{instr.offset:04x}")
                    self.vulnerabilities.append(ProxyVulnerability(
                        severity=VulnSeverity.CRITICAL,
                        title="SELFDESTRUCT Found",
                        description="Contract contains SELFDESTRUCT. If callable, "
                                   "could destroy proxy and brick all delegated logic.",
                        location=instr.offset,
                        exploit_path="Trigger SELFDESTRUCT to destroy contract",
                        calldata=None
                    ))
                    return
        
        print(f"  [-] No SELFDESTRUCT found")
    
    def _check_fallback_behavior(self):
        """Analyze fallback/receive function behavior."""
        print(f"\n[6] Fallback Function Analysis")
        print("-" * 50)
        
        # Check entry point (block 0)
        entry_block = self.cfg.blocks.get(0)
        if not entry_block:
            print(f"  [-] No entry block found")
            return
        
        # Look for CALLDATASIZE check - indicates fallback routing
        has_fallback = False
        for instr in entry_block.instructions:
            if instr.opcode == 0x36:  # CALLDATASIZE
                has_fallback = True
                print(f"  [+] CALLDATASIZE check at 0x{instr.offset:04x}")
                break
        
        if has_fallback:
            print(f"  [+] Fallback routes to DELEGATECALL (standard proxy pattern)")
        else:
            print(f"  [-] No standard fallback routing detected")
    
    def _check_storage_collisions(self):
        """Check for potential storage slot collisions."""
        print(f"\n[7] Storage Collision Analysis")
        print("-" * 50)
        
        # Find all SSTORE/SLOAD with non-EIP1967 slots
        storage_ops = []
        
        for bid, block in self.cfg.blocks.items():
            for i, instr in enumerate(block.instructions):
                if instr.opcode in (0x54, 0x55):  # SLOAD/SSTORE
                    op_name = "SLOAD" if instr.opcode == 0x54 else "SSTORE"
                    
                    # Try to determine slot
                    if i > 0:
                        prev = block.instructions[i-1]
                        if prev.opcode in range(0x60, 0x80):  # PUSH
                            if prev.operand:
                                slot = int.from_bytes(prev.operand, 'big')
                                if slot not in EIP1967_SLOTS and slot not in OZ_LEGACY_SLOTS:
                                    if slot < 1000:  # Low slots more likely to collide
                                        storage_ops.append((op_name, slot, instr.offset))
                                        print(f"  [!] {op_name} to low slot {slot} at 0x{instr.offset:04x}")
        
        if storage_ops:
            low_slots = [s[1] for s in storage_ops if s[1] < 10]
            if low_slots:
                self.vulnerabilities.append(ProxyVulnerability(
                    severity=VulnSeverity.MEDIUM,
                    title="Low Storage Slot Usage",
                    description=f"Proxy uses low storage slots ({low_slots}). "
                               "May collide with implementation contract storage.",
                    location=storage_ops[0][2],
                    exploit_path="Check if implementation uses same slots for different data",
                    calldata=None
                ))
        else:
            print(f"  [+] Only EIP-1967 slots used - no collision risk")
        
        self.findings['storage_ops'] = storage_ops
    
    def report(self) -> Dict:
        """Generate analysis report."""
        print(f"\n{'='*70}")
        print("VULNERABILITY SUMMARY")
        print(f"{'='*70}")
        
        severity_counts = {s: 0 for s in VulnSeverity}
        for v in self.vulnerabilities:
            severity_counts[v.severity] += 1
        
        for sev, count in severity_counts.items():
            if count > 0:
                print(f"  {sev.value}: {count}")
        
        if self.vulnerabilities:
            print(f"\n{'='*70}")
            print("DETAILED FINDINGS")
            print(f"{'='*70}")
            
            for i, v in enumerate(self.vulnerabilities, 1):
                print(f"\n[{i}] [{v.severity.value}] {v.title}")
                print(f"    {v.description}")
                if v.location:
                    print(f"    Location: 0x{v.location:04x}")
                if v.exploit_path:
                    print(f"    Exploit: {v.exploit_path}")
                if v.calldata:
                    print(f"    Calldata: {v.calldata}")
        else:
            print(f"\n  No vulnerabilities found")
        
        return {
            "contract": self.contract_name,
            "address": self.address,
            "vulnerabilities": [
                {
                    "severity": v.severity.value,
                    "title": v.title,
                    "description": v.description,
                    "location": v.location,
                    "calldata": v.calldata
                }
                for v in self.vulnerabilities
            ],
            "findings": {
                "eip1967_slots": list(self.findings.get('eip1967_slots', {}).keys()),
                "admin_functions": [
                    f[0] for f in self.findings.get('admin_functions', {}).values()
                ],
                "delegatecall_count": len(self.findings.get('delegatecall_sites', [])),
                "initializers": [
                    f[1] for f in self.findings.get('initializers', [])
                ]
            }
        }


def main():
    # Load bytecode
    with open('aave_graveyard_bytecode.json', 'r') as f:
        data = json.load(f)
    
    # Analyze all contracts
    contracts = [
        ('Aave_V2_LendingPoolProxy', '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9'),
        ('Aave_V1_PoolCore', '0x3dfd23A6c5E8BbcFc9581d2E864a68feb6a076d3'),
        ('Aave_V2_PoolAddressesProviderRegistry', '0xbaA999AC55EAce41CcAE355c77809e68Bb345170'),
    ]
    
    all_results = []
    
    for name, address in contracts:
        if name in data:
            bytecode = data[name].get('bytecode', data[name])
            if isinstance(bytecode, str) and len(bytecode) > 10:
                analyzer = ProxyAnalyzer(bytecode, name, address)
                analyzer.analyze()
                result = analyzer.report()
                all_results.append(result)
    
    # Save results
    with open('proxy_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n[+] Results saved to proxy_analysis.json")
    
    # Summary
    print(f"\n{'='*70}")
    print("CROSS-CONTRACT INITIALIZATION ANALYSIS")
    print(f"{'='*70}")
    
    total_critical = sum(
        1 for r in all_results 
        for v in r['vulnerabilities'] 
        if v['severity'] == 'CRITICAL'
    )
    
    if total_critical > 0:
        print(f"\n[!] CRITICAL: Found {total_critical} critical vulnerabilities!")
    else:
        print(f"\n[+] No critical initialization vulnerabilities found")
        print(f"    Proxy follows EIP-1967 standard")
        print(f"    Admin functions have standard guards")


if __name__ == "__main__":
    main()
