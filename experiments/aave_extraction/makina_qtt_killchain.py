#!/usr/bin/env python3
"""
QTT Cross-Contract Kill-Chain Analyzer
Targets: Makina Caliber Proxy + DUSD/USDC Pool
Constraint: ReentrancyGuard bypass via cross-contract state desynchronization
Safe Harbor: 0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

OPCODES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV",
    0x10: "LT", 0x11: "GT", 0x14: "EQ", 0x15: "ISZERO", 0x16: "AND",
    0x32: "ORIGIN", 0x33: "CALLER", 0x34: "CALLVALUE", 0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE", 0x37: "CALLDATACOPY",
    0x47: "SELFBALANCE",
    0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE",
    0x54: "SLOAD", 0x55: "SSTORE", 0x56: "JUMP", 0x57: "JUMPI",
    0x5b: "JUMPDEST",
    0x60: "PUSH1", 0x61: "PUSH2", 0x63: "PUSH4", 0x73: "PUSH20",
    0x80: "DUP1", 0x81: "DUP2", 0x90: "SWAP1",
    0xf1: "CALL", 0xf2: "CALLCODE", 0xf3: "RETURN",
    0xf4: "DELEGATECALL", 0xfa: "STATICCALL",
    0xfd: "REVERT", 0xff: "SELFDESTRUCT"
}


@dataclass
class BasicBlock:
    start: int
    end: int
    instructions: list = field(default_factory=list)
    has_call: bool = False
    has_delegatecall: bool = False
    has_staticcall: bool = False
    has_sstore: bool = False
    has_sload: bool = False
    has_caller: bool = False
    has_revert: bool = False
    has_selfdestruct: bool = False
    has_reentrancy_check: bool = False  # SLOAD followed by SSTORE pattern


class CrossContractQTT:
    """Cross-contract QTT analyzer for reentrancy and state desync."""
    
    CALIBER = "0x06147e073B854521c7B778280E7d7dBAfB2D4898"
    DUSD_POOL = "0x32E616F4f17d43f9A5cd9Be0e294727187064cb3"
    SAFE_HARBOR = "0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c"
    
    def __init__(self, caliber_bytecode: str, pool_bytecode: str):
        self.caliber_bytes = bytes.fromhex(caliber_bytecode.replace("0x", ""))
        self.pool_bytes = bytes.fromhex(pool_bytecode.replace("0x", ""))
        
        self.caliber_blocks: dict[int, BasicBlock] = {}
        self.pool_blocks: dict[int, BasicBlock] = {}
        
        self.caliber_edges: list = []
        self.pool_edges: list = []
        
        self.reentrancy_patterns: list = []
        self.cross_call_points: list = []
        
    def disassemble_contract(self, bytecode: bytes, name: str) -> tuple:
        """Phase 1: Disassemble a single contract."""
        print(f"[PHASE 1] Disassembling {name}...")
        print(f"  Bytecode size: {len(bytecode)} bytes")
        
        blocks = {}
        edges = []
        jumpdests = set()
        
        # First pass - find JUMPDESTs
        i = 0
        while i < len(bytecode):
            op = bytecode[i]
            if op == 0x5b:
                jumpdests.add(i)
            if 0x60 <= op <= 0x7f:
                i += (op - 0x5f)
            i += 1
        
        # Second pass - build blocks
        i = 0
        current = BasicBlock(start=0, end=0)
        
        while i < len(bytecode):
            op = bytecode[i]
            
            if op == 0x33:
                current.has_caller = True
            elif op == 0x54:
                current.has_sload = True
            elif op == 0x55:
                current.has_sstore = True
                # Check for reentrancy guard pattern (SLOAD then SSTORE same block)
                if current.has_sload:
                    current.has_reentrancy_check = True
            elif op == 0xf1:
                current.has_call = True
            elif op == 0xf4:
                current.has_delegatecall = True
            elif op == 0xfa:
                current.has_staticcall = True
            elif op == 0xfd:
                current.has_revert = True
            elif op == 0xff:
                current.has_selfdestruct = True
            
            push_val = None
            if 0x60 <= op <= 0x7f:
                push_size = op - 0x5f
                if i + push_size < len(bytecode):
                    push_val = int.from_bytes(bytecode[i+1:i+1+push_size], 'big')
                current.instructions.append((i, OPCODES.get(op, f"PUSH{push_size}"), push_val))
                i += push_size
            else:
                current.instructions.append((i, OPCODES.get(op, f"0x{op:02x}"), None))
            
            if op in (0x00, 0x56, 0x57, 0xf3, 0xfd, 0xfe, 0xff):
                current.end = i
                blocks[current.start] = current
                
                if op == 0x56:  # JUMP
                    for j in range(len(current.instructions)-2, -1, -1):
                        if current.instructions[j][2] is not None:
                            target = current.instructions[j][2]
                            if target in jumpdests:
                                edges.append((current.start, target))
                            break
                elif op == 0x57:  # JUMPI
                    for j in range(len(current.instructions)-2, -1, -1):
                        if current.instructions[j][2] is not None:
                            target = current.instructions[j][2]
                            if target in jumpdests:
                                edges.append((current.start, target))
                            break
                    if i + 1 < len(bytecode):
                        edges.append((current.start, i + 1))
                
                if i + 1 < len(bytecode):
                    current = BasicBlock(start=i + 1, end=i + 1)
            elif i + 1 in jumpdests:
                current.end = i
                blocks[current.start] = current
                edges.append((current.start, i + 1))
                current = BasicBlock(start=i + 1, end=i + 1)
            
            i += 1
        
        if current.instructions:
            current.end = len(bytecode) - 1
            blocks[current.start] = current
        
        print(f"  Blocks: {len(blocks)}")
        print(f"  Edges: {len(edges)}")
        
        return blocks, edges, jumpdests
    
    def find_reentrancy_patterns(self, blocks: dict, name: str) -> list:
        """Find potential reentrancy vulnerabilities."""
        patterns = []
        
        for addr, block in blocks.items():
            # Pattern 1: CALL after SLOAD without SSTORE (check-effect-interaction violation)
            if block.has_call and block.has_sload and not block.has_sstore:
                patterns.append({
                    "type": "CEI_VIOLATION",
                    "block": hex(addr),
                    "contract": name,
                    "description": "CALL after SLOAD without prior SSTORE - possible reentrancy"
                })
            
            # Pattern 2: DELEGATECALL in block (proxy pattern - state can desync)
            if block.has_delegatecall:
                patterns.append({
                    "type": "DELEGATECALL_STATE_DESYNC",
                    "block": hex(addr),
                    "contract": name,
                    "description": "DELEGATECALL can cause state desynchronization"
                })
            
            # Pattern 3: SSTORE after CALL (vulnerable to reentrancy)
            if block.has_call and block.has_sstore:
                # Check order in instructions
                call_idx = None
                sstore_idx = None
                for i, (off, op, val) in enumerate(block.instructions):
                    if op == "CALL":
                        call_idx = i
                    if op == "SSTORE" and call_idx is not None:
                        sstore_idx = i
                        break
                
                if call_idx is not None and sstore_idx is not None and sstore_idx > call_idx:
                    patterns.append({
                        "type": "SSTORE_AFTER_CALL",
                        "block": hex(addr),
                        "contract": name,
                        "description": "SSTORE after CALL - classic reentrancy pattern"
                    })
            
            # Pattern 4: ReentrancyGuard check (SLOAD + check + SSTORE)
            if block.has_reentrancy_check:
                patterns.append({
                    "type": "REENTRANCY_GUARD",
                    "block": hex(addr),
                    "contract": name,
                    "description": "Potential ReentrancyGuard pattern detected"
                })
        
        return patterns
    
    def find_cross_call_points(self, blocks: dict, name: str) -> list:
        """Find external call points that could be exploited cross-contract."""
        calls = []
        
        for addr, block in blocks.items():
            if block.has_call or block.has_delegatecall or block.has_staticcall:
                # Look for target address in preceding pushes
                target_addr = None
                for i in range(len(block.instructions)-1, -1, -1):
                    op = block.instructions[i][1]
                    val = block.instructions[i][2]
                    if op == "PUSH20" and val is not None:
                        target_addr = hex(val)
                        break
                
                call_type = "CALL" if block.has_call else ("DELEGATECALL" if block.has_delegatecall else "STATICCALL")
                calls.append({
                    "type": call_type,
                    "block": hex(addr),
                    "contract": name,
                    "target": target_addr,
                    "has_sload_before": block.has_sload,
                    "has_sstore_after": block.has_sstore
                })
        
        return calls
    
    def build_cross_tensor(self) -> np.ndarray:
        """Phase 2: Build cross-contract state tensor."""
        print("\n[PHASE 2] Building Cross-Contract Tensor...")
        
        n_caliber = len(self.caliber_blocks)
        n_pool = len(self.pool_blocks)
        n_total = n_caliber + n_pool
        
        # Combined adjacency matrix
        adj = np.zeros((n_total, n_total), dtype=np.float32)
        
        caliber_list = list(self.caliber_blocks.keys())
        pool_list = list(self.pool_blocks.keys())
        
        caliber_idx = {addr: i for i, addr in enumerate(caliber_list)}
        pool_idx = {addr: i + n_caliber for i, addr in enumerate(pool_list)}
        
        # Add caliber internal edges
        for src, dst in self.caliber_edges:
            if src in caliber_idx and dst in caliber_idx:
                adj[caliber_idx[src], caliber_idx[dst]] = 1.0
        
        # Add pool internal edges
        for src, dst in self.pool_edges:
            if src in pool_idx and dst in pool_idx:
                adj[pool_idx[src], pool_idx[dst]] = 1.0
        
        # Add cross-contract edges (CALL between contracts)
        for call in self.cross_call_points:
            if call["contract"] == "Caliber" and call["target"]:
                # Caliber calling Pool
                if call["target"].lower() == self.DUSD_POOL.lower():
                    src_idx = caliber_idx.get(int(call["block"], 16))
                    if src_idx is not None:
                        # Connect to pool entry points
                        for dst in pool_list[:5]:  # First few blocks as entry
                            adj[src_idx, pool_idx[dst]] = 2.0  # Cross-contract weight
        
        self.cross_tensor = adj
        self.total_blocks = n_total
        self.caliber_idx = caliber_idx
        self.pool_idx = pool_idx
        
        print(f"  Combined tensor: {n_total}x{n_total}")
        print(f"  Caliber blocks: {n_caliber}")
        print(f"  Pool blocks: {n_pool}")
        print(f"  Cross-contract edges: {np.sum(adj == 2.0)}")
        
        return adj
    
    def apply_constraints(self) -> dict:
        """Phase 3: Apply reentrancy and state desync constraints."""
        print("\n[PHASE 3] Applying Constraints...")
        
        constraints = {
            "reentrancy_bypass": {"found": False, "path": None, "reason": ""},
            "state_desync": {"found": False, "path": None, "reason": ""},
            "emergency_access": {"found": False, "path": None, "reason": ""},
            "routing": {"found": False, "path": None, "reason": ""}
        }
        
        # Constraint 1: ReentrancyGuard bypass via cross-contract desync
        print("  Constraint 1: ReentrancyGuard bypass...")
        
        caliber_guards = [p for p in self.reentrancy_patterns if p["contract"] == "Caliber" and p["type"] == "REENTRANCY_GUARD"]
        caliber_cei_violations = [p for p in self.reentrancy_patterns if p["contract"] == "Caliber" and p["type"] == "CEI_VIOLATION"]
        
        print(f"    Caliber ReentrancyGuard blocks: {len(caliber_guards)}")
        print(f"    Caliber CEI violations: {len(caliber_cei_violations)}")
        
        if caliber_cei_violations:
            constraints["reentrancy_bypass"]["found"] = True
            constraints["reentrancy_bypass"]["path"] = [p["block"] for p in caliber_cei_violations[:3]]
            constraints["reentrancy_bypass"]["reason"] = "CEI violations found - potential reentrancy entry"
        else:
            constraints["reentrancy_bypass"]["reason"] = "No CEI violations - ReentrancyGuard appears intact"
        
        # Constraint 2: Cross-contract state desynchronization
        print("  Constraint 2: State desynchronization...")
        
        delegatecall_patterns = [p for p in self.reentrancy_patterns if p["type"] == "DELEGATECALL_STATE_DESYNC"]
        sstore_after_call = [p for p in self.reentrancy_patterns if p["type"] == "SSTORE_AFTER_CALL"]
        
        print(f"    DELEGATECALL state desync points: {len(delegatecall_patterns)}")
        print(f"    SSTORE after CALL patterns: {len(sstore_after_call)}")
        
        if delegatecall_patterns or sstore_after_call:
            constraints["state_desync"]["found"] = True
            all_desync = delegatecall_patterns + sstore_after_call
            constraints["state_desync"]["path"] = [p["block"] for p in all_desync[:3]]
            constraints["state_desync"]["reason"] = "State desync vectors identified"
        else:
            constraints["state_desync"]["reason"] = "No obvious state desync vectors"
        
        # Constraint 3: Emergency withdraw access control
        print("  Constraint 3: Emergency access control...")
        
        # Look for blocks with CALLER check followed by CALL (access control pattern)
        emergency_candidates = []
        for addr, block in self.caliber_blocks.items():
            if block.has_caller and (block.has_call or block.has_sstore):
                emergency_candidates.append(hex(addr))
        
        print(f"    Emergency function candidates: {len(emergency_candidates)}")
        
        if emergency_candidates:
            constraints["emergency_access"]["found"] = True
            constraints["emergency_access"]["path"] = emergency_candidates[:5]
            constraints["emergency_access"]["reason"] = "Access-controlled functions with state mutation found"
        else:
            constraints["emergency_access"]["reason"] = "No emergency functions identified"
        
        # Constraint 4: Routing to Safe Harbor
        print("  Constraint 4: Routing to Safe Harbor...")
        
        if constraints["state_desync"]["found"] or constraints["reentrancy_bypass"]["found"]:
            constraints["routing"]["found"] = True
            constraints["routing"]["reason"] = f"Routing possible via state desync to {self.SAFE_HARBOR}"
        else:
            constraints["routing"]["reason"] = "No viable routing path without breach"
        
        return constraints
    
    def solve_tropical(self, constraints: dict) -> Optional[list]:
        """Phase 4: Tropical geometry path solving."""
        print("\n[PHASE 4] Tropical Path Solving...")
        
        if not any(c["found"] for c in constraints.values()):
            print("  [X] No viable constraints satisfied")
            return None
        
        # Build tropical distance matrix
        n = self.total_blocks
        tropical = np.full((n, n), np.inf, dtype=np.float32)
        
        for i in range(n):
            tropical[i, i] = 0
        
        # Set edge weights
        for i in range(n):
            for j in range(n):
                if self.cross_tensor[i, j] > 0:
                    weight = 1.0 if self.cross_tensor[i, j] == 1.0 else 0.5  # Cross-contract edges are cheaper
                    tropical[i, j] = weight
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    tropical[i, j] = min(tropical[i, j], tropical[i, k] + tropical[k, j])
        
        # Find path from entry to CALL blocks
        caliber_list = list(self.caliber_blocks.keys())
        pool_list = list(self.pool_blocks.keys())
        
        entry = 0  # First block
        call_blocks = []
        
        for addr, block in self.caliber_blocks.items():
            if block.has_call:
                idx = self.caliber_idx.get(addr)
                if idx is not None:
                    call_blocks.append(idx)
        
        if call_blocks:
            min_dist = min(tropical[entry, cb] for cb in call_blocks)
            print(f"  Tropical distance to CALL: {min_dist}")
            
            if min_dist < np.inf:
                return [hex(caliber_list[entry])] + [hex(caliber_list[cb]) if cb < len(caliber_list) else hex(pool_list[cb - len(caliber_list)]) for cb in call_blocks[:3]]
        
        return None
    
    def generate_payload(self, path: Optional[list], constraints: dict) -> dict:
        """Phase 5: Generate exploitation payload."""
        print("\n[PHASE 5] Payload Generation...")
        
        result = {
            "kill_chain": [],
            "exploitable": False,
            "requires": []
        }
        
        if not path:
            print("  [X] No path found - no payload generated")
            return result
        
        # Check for actual exploitation conditions
        if constraints["state_desync"]["found"]:
            print("  [!] State desync vector exists")
            result["requires"].append("Flash loan to trigger state desync")
        
        if constraints["reentrancy_bypass"]["found"]:
            print("  [!] Reentrancy entry point exists")
            result["requires"].append("Reentrant callback during external call")
        
        if constraints["emergency_access"]["found"]:
            print("  [!] Emergency function access possible")
            result["requires"].append("Proper state permutation for access")
        
        # This would be the actual exploit calldata if viable
        # For now, we document the theoretical path
        result["theoretical_path"] = path
        result["exploitable"] = len(result["requires"]) > 0
        
        return result
    
    def run_full_analysis(self) -> dict:
        """Run complete cross-contract QTT analysis."""
        print("=" * 70)
        print("QTT CROSS-CONTRACT KILL-CHAIN ANALYZER")
        print("=" * 70)
        print(f"Target 1: Caliber Proxy - {self.CALIBER}")
        print(f"Target 2: DUSD/USDC Pool - {self.DUSD_POOL}")
        print(f"Safe Harbor: {self.SAFE_HARBOR}")
        print(f"Value at risk: ~$10,541 USDC in Pool")
        print("=" * 70)
        
        # Phase 1: Disassemble both contracts
        self.caliber_blocks, self.caliber_edges, _ = self.disassemble_contract(
            self.caliber_bytes, "Caliber"
        )
        self.pool_blocks, self.pool_edges, _ = self.disassemble_contract(
            self.pool_bytes, "DUSD_Pool"
        )
        
        # Find vulnerability patterns
        self.reentrancy_patterns = (
            self.find_reentrancy_patterns(self.caliber_blocks, "Caliber") +
            self.find_reentrancy_patterns(self.pool_blocks, "DUSD_Pool")
        )
        
        self.cross_call_points = (
            self.find_cross_call_points(self.caliber_blocks, "Caliber") +
            self.find_cross_call_points(self.pool_blocks, "DUSD_Pool")
        )
        
        print(f"\n  Total reentrancy patterns: {len(self.reentrancy_patterns)}")
        print(f"  Total cross-call points: {len(self.cross_call_points)}")
        
        # Phase 2: Build cross-contract tensor
        self.build_cross_tensor()
        
        # Phase 3: Apply constraints
        constraints = self.apply_constraints()
        
        # Phase 4: Tropical path solving
        path = self.solve_tropical(constraints)
        
        # Phase 5: Generate payload
        payload = self.generate_payload(path, constraints)
        
        # Final assessment
        print("\n" + "=" * 70)
        print("FINAL ASSESSMENT")
        print("=" * 70)
        
        result = {
            "targets": {
                "caliber": self.CALIBER,
                "pool": self.DUSD_POOL
            },
            "safe_harbor": self.SAFE_HARBOR,
            "value_at_risk": {
                "usdc": 10541.45,
                "eth": 0
            },
            "phase1": {
                "caliber_blocks": len(self.caliber_blocks),
                "pool_blocks": len(self.pool_blocks),
                "reentrancy_patterns": len(self.reentrancy_patterns),
                "cross_calls": len(self.cross_call_points)
            },
            "phase3_constraints": constraints,
            "phase4_path": path,
            "phase5_payload": payload,
            "verdict": "ANALYSIS_COMPLETE"
        }
        
        # Determine exploitability
        if payload["exploitable"]:
            print("[!] POTENTIAL EXPLOITATION VECTORS IDENTIFIED")
            for req in payload["requires"]:
                print(f"    - {req}")
            result["verdict"] = "REQUIRES_FURTHER_ANALYSIS"
        else:
            print("[X] No viable exploitation path found")
            result["verdict"] = "NOT_EXPLOITABLE"
        
        return result


def main():
    # Load bytecode from files
    print("Loading bytecode...")
    
    with open("/tmp/caliber_bytecode.txt") as f:
        caliber_code = f.read().strip()
    
    with open("/tmp/dusd_pool_bytecode.txt") as f:
        pool_code = f.read().strip()
    
    print(f"Caliber: {len(caliber_code)} chars")
    print(f"Pool: {len(pool_code)} chars")
    
    analyzer = CrossContractQTT(caliber_code, pool_code)
    result = analyzer.run_full_analysis()
    
    # Save results
    output_path = Path(__file__).parent / "makina_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
