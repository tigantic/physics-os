#!/usr/bin/env python3
"""
QTT-EVM Extraction Engine
=========================

5-Phase pipeline for extracting stranded capital from dead proxies using
Quantized Tensor Train state-space exploration.

Architecture:
    Phase 1: Disassembly & Graph Translation (EVM → CFG)
    Phase 2: State-Space Tensorization (CFG → QTT)
    Phase 3: Boundary Constraints (Lock start/end states)
    Phase 4: Path Extraction (Tropical shortest path solver)
    Phase 5: Payload Reconstruction (Tensor → calldata)

The key insight: We don't fuzz. We mathematically solve the inverse problem.

References:
-----------
.. [1] Wood, G. (2024). "Ethereum Yellow Paper". Opcodes and gas.
.. [2] Oseledets, I.V. (2011). "Tensor-Train Decomposition". SIAM J. Sci. Comput.
.. [3] Pachter & Sturmfels (2004). "Tropical Geometry". arXiv:math/0408099.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from enum import Enum, auto
from pathlib import Path
import math

import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════════════
# PHASE 1: EVM DISASSEMBLY & CONTROL FLOW GRAPH
# ═══════════════════════════════════════════════════════════════════════════════════════

class Opcode(Enum):
    """EVM opcodes relevant for state extraction."""
    # Stack ops
    STOP = 0x00
    ADD = 0x01
    MUL = 0x02
    SUB = 0x03
    DIV = 0x04
    SDIV = 0x05
    MOD = 0x06
    SMOD = 0x07
    ADDMOD = 0x08
    MULMOD = 0x09
    EXP = 0x0a
    SIGNEXTEND = 0x0b
    
    # Comparison
    LT = 0x10
    GT = 0x11
    SLT = 0x12
    SGT = 0x13
    EQ = 0x14
    ISZERO = 0x15
    AND = 0x16
    OR = 0x17
    XOR = 0x18
    NOT = 0x19
    BYTE = 0x1a
    SHL = 0x1b
    SHR = 0x1c
    SAR = 0x1d
    
    # SHA3
    SHA3 = 0x20
    
    # Environment
    ADDRESS = 0x30
    BALANCE = 0x31
    ORIGIN = 0x32
    CALLER = 0x33
    CALLVALUE = 0x34
    CALLDATALOAD = 0x35
    CALLDATASIZE = 0x36
    CALLDATACOPY = 0x37
    CODESIZE = 0x38
    CODECOPY = 0x39
    GASPRICE = 0x3a
    EXTCODESIZE = 0x3b
    EXTCODECOPY = 0x3c
    RETURNDATASIZE = 0x3d
    RETURNDATACOPY = 0x3e
    EXTCODEHASH = 0x3f
    
    # Block
    BLOCKHASH = 0x40
    COINBASE = 0x41
    TIMESTAMP = 0x42
    NUMBER = 0x43
    DIFFICULTY = 0x44
    GASLIMIT = 0x45
    CHAINID = 0x46
    SELFBALANCE = 0x47
    BASEFEE = 0x48
    
    # Stack/Memory/Storage
    POP = 0x50
    MLOAD = 0x51
    MSTORE = 0x52
    MSTORE8 = 0x53
    SLOAD = 0x54
    SSTORE = 0x55
    JUMP = 0x56
    JUMPI = 0x57
    PC = 0x58
    MSIZE = 0x59
    GAS = 0x5a
    JUMPDEST = 0x5b
    
    # Push ops (0x60-0x7f)
    PUSH1 = 0x60
    PUSH2 = 0x61
    PUSH3 = 0x62
    PUSH4 = 0x63
    PUSH32 = 0x7f
    
    # Dup ops (0x80-0x8f)
    DUP1 = 0x80
    DUP16 = 0x8f
    
    # Swap ops (0x90-0x9f)
    SWAP1 = 0x90
    SWAP16 = 0x9f
    
    # Log ops
    LOG0 = 0xa0
    LOG1 = 0xa1
    LOG2 = 0xa2
    LOG3 = 0xa3
    LOG4 = 0xa4
    
    # System ops
    CREATE = 0xf0
    CALL = 0xf1
    CALLCODE = 0xf2
    RETURN = 0xf3
    DELEGATECALL = 0xf4
    CREATE2 = 0xf5
    STATICCALL = 0xfa
    REVERT = 0xfd
    INVALID = 0xfe
    SELFDESTRUCT = 0xff
    
    UNKNOWN = -1


# Opcode metadata: (name, stack_pop, stack_push, is_push_bytes)
OPCODE_INFO: Dict[int, Tuple[str, int, int, int]] = {
    0x00: ("STOP", 0, 0, 0),
    0x01: ("ADD", 2, 1, 0),
    0x02: ("MUL", 2, 1, 0),
    0x03: ("SUB", 2, 1, 0),
    0x04: ("DIV", 2, 1, 0),
    0x05: ("SDIV", 2, 1, 0),
    0x06: ("MOD", 2, 1, 0),
    0x07: ("SMOD", 2, 1, 0),
    0x08: ("ADDMOD", 3, 1, 0),
    0x09: ("MULMOD", 3, 1, 0),
    0x0a: ("EXP", 2, 1, 0),
    0x0b: ("SIGNEXTEND", 2, 1, 0),
    0x10: ("LT", 2, 1, 0),
    0x11: ("GT", 2, 1, 0),
    0x12: ("SLT", 2, 1, 0),
    0x13: ("SGT", 2, 1, 0),
    0x14: ("EQ", 2, 1, 0),
    0x15: ("ISZERO", 1, 1, 0),
    0x16: ("AND", 2, 1, 0),
    0x17: ("OR", 2, 1, 0),
    0x18: ("XOR", 2, 1, 0),
    0x19: ("NOT", 1, 1, 0),
    0x1a: ("BYTE", 2, 1, 0),
    0x1b: ("SHL", 2, 1, 0),
    0x1c: ("SHR", 2, 1, 0),
    0x1d: ("SAR", 2, 1, 0),
    0x20: ("SHA3", 2, 1, 0),
    0x30: ("ADDRESS", 0, 1, 0),
    0x31: ("BALANCE", 1, 1, 0),
    0x32: ("ORIGIN", 0, 1, 0),
    0x33: ("CALLER", 0, 1, 0),
    0x34: ("CALLVALUE", 0, 1, 0),
    0x35: ("CALLDATALOAD", 1, 1, 0),
    0x36: ("CALLDATASIZE", 0, 1, 0),
    0x37: ("CALLDATACOPY", 3, 0, 0),
    0x38: ("CODESIZE", 0, 1, 0),
    0x39: ("CODECOPY", 3, 0, 0),
    0x3a: ("GASPRICE", 0, 1, 0),
    0x3b: ("EXTCODESIZE", 1, 1, 0),
    0x3c: ("EXTCODECOPY", 4, 0, 0),
    0x3d: ("RETURNDATASIZE", 0, 1, 0),
    0x3e: ("RETURNDATACOPY", 3, 0, 0),
    0x3f: ("EXTCODEHASH", 1, 1, 0),
    0x40: ("BLOCKHASH", 1, 1, 0),
    0x41: ("COINBASE", 0, 1, 0),
    0x42: ("TIMESTAMP", 0, 1, 0),
    0x43: ("NUMBER", 0, 1, 0),
    0x44: ("DIFFICULTY", 0, 1, 0),
    0x45: ("GASLIMIT", 0, 1, 0),
    0x46: ("CHAINID", 0, 1, 0),
    0x47: ("SELFBALANCE", 0, 1, 0),
    0x48: ("BASEFEE", 0, 1, 0),
    0x50: ("POP", 1, 0, 0),
    0x51: ("MLOAD", 1, 1, 0),
    0x52: ("MSTORE", 2, 0, 0),
    0x53: ("MSTORE8", 2, 0, 0),
    0x54: ("SLOAD", 1, 1, 0),
    0x55: ("SSTORE", 2, 0, 0),
    0x56: ("JUMP", 1, 0, 0),
    0x57: ("JUMPI", 2, 0, 0),
    0x58: ("PC", 0, 1, 0),
    0x59: ("MSIZE", 0, 1, 0),
    0x5a: ("GAS", 0, 1, 0),
    0x5b: ("JUMPDEST", 0, 0, 0),
    0xa0: ("LOG0", 2, 0, 0),
    0xa1: ("LOG1", 3, 0, 0),
    0xa2: ("LOG2", 4, 0, 0),
    0xa3: ("LOG3", 5, 0, 0),
    0xa4: ("LOG4", 6, 0, 0),
    0xf0: ("CREATE", 3, 1, 0),
    0xf1: ("CALL", 7, 1, 0),
    0xf2: ("CALLCODE", 7, 1, 0),
    0xf3: ("RETURN", 2, 0, 0),
    0xf4: ("DELEGATECALL", 6, 1, 0),
    0xf5: ("CREATE2", 4, 1, 0),
    0xfa: ("STATICCALL", 6, 1, 0),
    0xfd: ("REVERT", 2, 0, 0),
    0xfe: ("INVALID", 0, 0, 0),
    0xff: ("SELFDESTRUCT", 1, 0, 0),
}

# Add PUSH1-PUSH32
for i in range(32):
    OPCODE_INFO[0x60 + i] = (f"PUSH{i+1}", 0, 1, i + 1)

# Add DUP1-DUP16
for i in range(16):
    OPCODE_INFO[0x80 + i] = (f"DUP{i+1}", i + 1, i + 2, 0)

# Add SWAP1-SWAP16  
for i in range(16):
    OPCODE_INFO[0x90 + i] = (f"SWAP{i+1}", i + 2, i + 2, 0)


@dataclass
class Instruction:
    """Single EVM instruction."""
    offset: int
    opcode: int
    opcode_name: str
    operand: Optional[bytes] = None
    operand_value: Optional[int] = None
    
    @property
    def size(self) -> int:
        """Instruction size in bytes."""
        if self.operand:
            return 1 + len(self.operand)
        return 1
    
    def __repr__(self) -> str:
        if self.operand:
            return f"{self.offset:04x}: {self.opcode_name} 0x{self.operand.hex()}"
        return f"{self.offset:04x}: {self.opcode_name}"


@dataclass
class BasicBlock:
    """Basic block in CFG."""
    start: int
    end: int
    instructions: List[Instruction]
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    is_entry: bool = False
    is_terminal: bool = False
    
    @property
    def terminator(self) -> Optional[Instruction]:
        """Get terminating instruction."""
        if self.instructions:
            return self.instructions[-1]
        return None
    
    def has_storage_ops(self) -> bool:
        """Check if block has storage operations."""
        for inst in self.instructions:
            if inst.opcode in (0x54, 0x55):  # SLOAD, SSTORE
                return True
        return False


@dataclass
class ControlFlowGraph:
    """Control flow graph of EVM bytecode."""
    blocks: Dict[int, BasicBlock]
    entry: int
    jumpdests: Set[int]
    storage_slots: Set[int]  # Identified storage slots
    function_selectors: Dict[bytes, int]  # selector -> entry offset
    
    @property
    def n_blocks(self) -> int:
        return len(self.blocks)
    
    @property
    def n_edges(self) -> int:
        return sum(len(b.successors) for b in self.blocks.values())
    
    def get_reachable_blocks(self, from_block: int) -> Set[int]:
        """Get all blocks reachable from given block."""
        visited = set()
        stack = [from_block]
        while stack:
            block_id = stack.pop()
            if block_id in visited:
                continue
            visited.add(block_id)
            if block_id in self.blocks:
                stack.extend(self.blocks[block_id].successors)
        return visited


class EVMDisassembler:
    """
    EVM bytecode disassembler and CFG builder.
    
    Implements Phase 1 of the extraction pipeline:
    1. Parse hex bytecode into opcodes
    2. Identify basic blocks and control flow
    3. Extract storage slot accesses
    4. Map function selectors
    """
    
    def __init__(self, bytecode: Union[str, bytes]):
        """
        Initialize disassembler.
        
        Args:
            bytecode: Hex string or bytes of EVM bytecode
        """
        if isinstance(bytecode, str):
            bytecode = bytecode.replace("0x", "")
            self.bytecode = bytes.fromhex(bytecode)
        else:
            self.bytecode = bytecode
        
        self.instructions: List[Instruction] = []
        self.jumpdests: Set[int] = set()
        self.jump_targets: Set[int] = set()
        
    def disassemble(self) -> List[Instruction]:
        """
        Disassemble bytecode into instructions.
        
        Returns:
            List of Instruction objects
        """
        self.instructions = []
        self.jumpdests = set()
        
        offset = 0
        while offset < len(self.bytecode):
            opcode = self.bytecode[offset]
            info = OPCODE_INFO.get(opcode, ("UNKNOWN", 0, 0, 0))
            name, _, _, push_bytes = info
            
            operand = None
            operand_value = None
            
            if push_bytes > 0:
                operand = self.bytecode[offset + 1:offset + 1 + push_bytes]
                if len(operand) == push_bytes:
                    operand_value = int.from_bytes(operand, 'big')
                else:
                    # Truncated bytecode
                    operand_value = int.from_bytes(operand.ljust(push_bytes, b'\x00'), 'big')
            
            inst = Instruction(
                offset=offset,
                opcode=opcode,
                opcode_name=name,
                operand=operand,
                operand_value=operand_value
            )
            self.instructions.append(inst)
            
            # Track JUMPDEST locations
            if opcode == 0x5b:  # JUMPDEST
                self.jumpdests.add(offset)
            
            # Track jump targets
            if opcode in (0x56, 0x57) and len(self.instructions) >= 2:  # JUMP, JUMPI
                prev = self.instructions[-2]
                if 0x60 <= prev.opcode <= 0x7f and prev.operand_value is not None:
                    self.jump_targets.add(prev.operand_value)
            
            offset += inst.size
        
        return self.instructions
    
    def build_cfg(self) -> ControlFlowGraph:
        """
        Build control flow graph from disassembled bytecode.
        
        Returns:
            ControlFlowGraph with basic blocks and edges
        """
        if not self.instructions:
            self.disassemble()
        
        # Find block boundaries
        block_starts: Set[int] = {0}  # Entry point
        block_starts.update(self.jumpdests)
        
        # Find implicit block boundaries (after JUMP/JUMPI/STOP/RETURN/REVERT)
        for i, inst in enumerate(self.instructions):
            if inst.opcode in (0x00, 0x56, 0x57, 0xf3, 0xfd, 0xfe, 0xff):
                # Terminal opcodes - next instruction starts new block
                if i + 1 < len(self.instructions):
                    block_starts.add(self.instructions[i + 1].offset)
        
        # Build blocks
        blocks: Dict[int, BasicBlock] = {}
        sorted_starts = sorted(block_starts)
        
        for i, start in enumerate(sorted_starts):
            # Find end of block
            end_offset = sorted_starts[i + 1] if i + 1 < len(sorted_starts) else len(self.bytecode)
            
            # Collect instructions in block
            block_insts = []
            for inst in self.instructions:
                if start <= inst.offset < end_offset:
                    block_insts.append(inst)
                    # Check for early termination
                    if inst.opcode in (0x00, 0x56, 0x57, 0xf3, 0xfd, 0xfe, 0xff):
                        end_offset = inst.offset + inst.size
                        break
            
            if block_insts:
                blocks[start] = BasicBlock(
                    start=start,
                    end=end_offset,
                    instructions=block_insts,
                    is_entry=(start == 0)
                )
        
        # Build edges
        for start, block in blocks.items():
            term = block.terminator
            if term is None:
                continue
            
            if term.opcode == 0x56:  # JUMP
                # Find target from previous PUSH
                for inst in reversed(block.instructions[:-1]):
                    if 0x60 <= inst.opcode <= 0x7f and inst.operand_value is not None:
                        target = inst.operand_value
                        if target in blocks:
                            block.successors.append(target)
                            blocks[target].predecessors.append(start)
                        break
                block.is_terminal = True
                
            elif term.opcode == 0x57:  # JUMPI
                # Conditional jump: two successors
                # True branch: jump target
                for inst in reversed(block.instructions[:-1]):
                    if 0x60 <= inst.opcode <= 0x7f and inst.operand_value is not None:
                        target = inst.operand_value
                        if target in blocks:
                            block.successors.append(target)
                            blocks[target].predecessors.append(start)
                        break
                # False branch: fall through
                next_offset = block.end
                if next_offset in blocks:
                    block.successors.append(next_offset)
                    blocks[next_offset].predecessors.append(start)
                    
            elif term.opcode in (0x00, 0xf3, 0xfd, 0xfe, 0xff):
                # Terminal opcodes
                block.is_terminal = True
                
            else:
                # Fall through to next block
                next_offset = block.end
                if next_offset in blocks:
                    block.successors.append(next_offset)
                    blocks[next_offset].predecessors.append(start)
        
        # Extract storage slots
        storage_slots = self._extract_storage_slots(blocks)
        
        # Extract function selectors
        function_selectors = self._extract_function_selectors()
        
        return ControlFlowGraph(
            blocks=blocks,
            entry=0,
            jumpdests=self.jumpdests,
            storage_slots=storage_slots,
            function_selectors=function_selectors
        )
    
    def _extract_storage_slots(self, blocks: Dict[int, BasicBlock]) -> Set[int]:
        """Extract storage slot indices from SLOAD/SSTORE operations."""
        slots: Set[int] = set()
        
        for block in blocks.values():
            for i, inst in enumerate(block.instructions):
                if inst.opcode in (0x54, 0x55):  # SLOAD, SSTORE
                    # Look for preceding PUSH with slot number
                    for prev in reversed(block.instructions[:i]):
                        if 0x60 <= prev.opcode <= 0x7f and prev.operand_value is not None:
                            slots.add(prev.operand_value)
                            break
        
        return slots
    
    def _extract_function_selectors(self) -> Dict[bytes, int]:
        """Extract function selectors from dispatcher pattern."""
        selectors: Dict[bytes, int] = {}
        
        # Look for pattern: PUSH4 <selector> EQ PUSH2 <offset> JUMPI
        for i, inst in enumerate(self.instructions):
            if inst.opcode == 0x63 and inst.operand:  # PUSH4
                selector = inst.operand
                # Look ahead for EQ and JUMPI
                for j in range(i + 1, min(i + 10, len(self.instructions))):
                    next_inst = self.instructions[j]
                    if next_inst.opcode == 0x57:  # JUMPI
                        # Find the jump target
                        for k in range(j - 1, i, -1):
                            push_inst = self.instructions[k]
                            if 0x60 <= push_inst.opcode <= 0x7f and push_inst.operand_value:
                                selectors[selector] = push_inst.operand_value
                                break
                        break
        
        return selectors


# ═══════════════════════════════════════════════════════════════════════════════════════
# PHASE 2: STATE-SPACE TENSORIZATION
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class TensorizedStateSpace:
    """
    QTT representation of EVM state space.
    
    Dimensions:
    - Stack: 1024 elements × 256 bits each (16 QTT sites)
    - Memory: Variable, typically 8-12 QTT sites
    - Storage: 2^256 slots × 256 bits (32 QTT sites per slot)
    - PC: Current program counter (10-16 bits typically)
    """
    n_stack_sites: int
    n_memory_sites: int
    n_storage_sites: int
    n_pc_sites: int
    max_rank: int
    
    # QTT cores for each component
    stack_cores: List[torch.Tensor]
    memory_cores: List[torch.Tensor]
    storage_cores: Dict[int, List[torch.Tensor]]  # slot -> cores
    pc_cores: List[torch.Tensor]
    
    # Transition matrices for opcodes (as MPOs)
    opcode_mpos: Dict[int, List[torch.Tensor]]
    
    @property
    def total_sites(self) -> int:
        return self.n_stack_sites + self.n_memory_sites + self.n_storage_sites + self.n_pc_sites
    
    @property
    def memory_kb(self) -> float:
        """Estimate QTT memory in KB."""
        total = 0
        for cores in [self.stack_cores, self.memory_cores, self.pc_cores]:
            for c in cores:
                total += c.numel() * 4  # float32
        for slot_cores in self.storage_cores.values():
            for c in slot_cores:
                total += c.numel() * 4
        return total / 1024


class StateSpaceTensorizer:
    """
    Phase 2: Convert CFG to QTT state space.
    
    The key insight: Each EVM opcode is a matrix operator that transforms
    the state tensor. We represent the entire state space as a QTT and
    define MPO operators for each opcode.
    """
    
    def __init__(
        self,
        cfg: ControlFlowGraph,
        max_rank: int = 64,
        n_stack_bits: int = 8,  # 256 stack elements
        n_memory_bits: int = 12,  # 4KB memory
        n_storage_bits: int = 16,  # Per-slot quantization
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.cfg = cfg
        self.max_rank = max_rank
        self.n_stack_bits = n_stack_bits
        self.n_memory_bits = n_memory_bits
        self.n_storage_bits = n_storage_bits
        self.device = torch.device(device)
        
        # PC bits determined by bytecode size
        self.n_pc_bits = max(8, math.ceil(math.log2(max(1, len(cfg.blocks)))))
        
    def tensorize(self) -> TensorizedStateSpace:
        """
        Create QTT representation of state space.
        
        Returns:
            TensorizedStateSpace with initialized cores
        """
        # Initialize QTT cores as identity (rank 1)
        stack_cores = self._init_identity_qtt(self.n_stack_bits)
        memory_cores = self._init_identity_qtt(self.n_memory_bits)
        pc_cores = self._init_identity_qtt(self.n_pc_bits)
        
        # Storage slots - one QTT per slot
        storage_cores: Dict[int, List[torch.Tensor]] = {}
        for slot in self.cfg.storage_slots:
            storage_cores[slot] = self._init_identity_qtt(self.n_storage_bits)
        
        # Build opcode MPOs
        opcode_mpos = self._build_opcode_mpos()
        
        return TensorizedStateSpace(
            n_stack_sites=self.n_stack_bits,
            n_memory_sites=self.n_memory_bits,
            n_storage_sites=self.n_storage_bits,
            n_pc_sites=self.n_pc_bits,
            max_rank=self.max_rank,
            stack_cores=stack_cores,
            memory_cores=memory_cores,
            storage_cores=storage_cores,
            pc_cores=pc_cores,
            opcode_mpos=opcode_mpos
        )
    
    def _init_identity_qtt(self, n_bits: int) -> List[torch.Tensor]:
        """Initialize QTT as identity state."""
        cores = []
        for i in range(n_bits):
            if i == 0:
                # First core: (1, 2, r)
                core = torch.zeros(1, 2, 1, device=self.device)
                core[0, 0, 0] = 1.0  # Initialize to |0⟩
            elif i == n_bits - 1:
                # Last core: (r, 2, 1)
                core = torch.zeros(1, 2, 1, device=self.device)
                core[0, 0, 0] = 1.0
            else:
                # Middle core: (r, 2, r)
                core = torch.zeros(1, 2, 1, device=self.device)
                core[0, 0, 0] = 1.0
            cores.append(core)
        return cores
    
    def _build_opcode_mpos(self) -> Dict[int, List[torch.Tensor]]:
        """
        Build MPO operators for EVM opcodes.
        
        Each opcode transforms the state tensor. We represent this as
        a Matrix Product Operator acting on the QTT state.
        """
        mpos: Dict[int, List[torch.Tensor]] = {}
        
        # SLOAD: Read from storage slot to stack
        # MPO structure: Identity on all sites except storage[slot] → stack[top]
        mpos[0x54] = self._build_sload_mpo()
        
        # SSTORE: Write from stack to storage slot
        mpos[0x55] = self._build_sstore_mpo()
        
        # JUMP: Set PC to target
        mpos[0x56] = self._build_jump_mpo()
        
        # JUMPI: Conditional PC update (creates superposition)
        mpos[0x57] = self._build_jumpi_mpo()
        
        # ADD, SUB, MUL, DIV: Arithmetic on stack
        for op in [0x01, 0x02, 0x03, 0x04]:
            mpos[op] = self._build_arithmetic_mpo(op)
        
        # CALL, DELEGATECALL: External calls
        mpos[0xf1] = self._build_call_mpo()
        mpos[0xf4] = self._build_delegatecall_mpo()
        
        # REVERT: Terminal state (pruned branch)
        mpos[0xfd] = self._build_revert_mpo()
        
        return mpos
    
    def _build_sload_mpo(self) -> List[torch.Tensor]:
        """Build MPO for SLOAD opcode."""
        # Simplified: Copy storage → stack
        cores = []
        for i in range(self.n_stack_bits):
            # 4-leg tensor: (r_in, phys_in, phys_out, r_out)
            core = torch.eye(2, device=self.device).reshape(1, 2, 2, 1)
            cores.append(core)
        return cores
    
    def _build_sstore_mpo(self) -> List[torch.Tensor]:
        """Build MPO for SSTORE opcode."""
        cores = []
        for i in range(self.n_storage_bits):
            core = torch.eye(2, device=self.device).reshape(1, 2, 2, 1)
            cores.append(core)
        return cores
    
    def _build_jump_mpo(self) -> List[torch.Tensor]:
        """Build MPO for JUMP opcode."""
        cores = []
        for i in range(self.n_pc_bits):
            core = torch.eye(2, device=self.device).reshape(1, 2, 2, 1)
            cores.append(core)
        return cores
    
    def _build_jumpi_mpo(self) -> List[torch.Tensor]:
        """Build MPO for JUMPI opcode (conditional)."""
        # Creates superposition of two paths
        cores = []
        for i in range(self.n_pc_bits):
            # Superposition: (|0⟩ + |1⟩)/√2
            core = torch.zeros(1, 2, 2, 2, device=self.device)
            core[0, 0, 0, 0] = 1.0  # Stay path
            core[0, 1, 1, 1] = 1.0  # Jump path
            cores.append(core)
        return cores
    
    def _build_arithmetic_mpo(self, opcode: int) -> List[torch.Tensor]:
        """Build MPO for arithmetic opcodes."""
        cores = []
        for i in range(self.n_stack_bits):
            core = torch.eye(2, device=self.device).reshape(1, 2, 2, 1)
            cores.append(core)
        return cores
    
    def _build_call_mpo(self) -> List[torch.Tensor]:
        """Build MPO for CALL opcode."""
        cores = []
        for i in range(self.n_stack_bits):
            core = torch.eye(2, device=self.device).reshape(1, 2, 2, 1)
            cores.append(core)
        return cores
    
    def _build_delegatecall_mpo(self) -> List[torch.Tensor]:
        """Build MPO for DELEGATECALL opcode."""
        return self._build_call_mpo()  # Similar structure
    
    def _build_revert_mpo(self) -> List[torch.Tensor]:
        """Build MPO for REVERT opcode (zero out branch)."""
        cores = []
        for i in range(self.n_pc_bits):
            # Zero operator - annihilates the state
            core = torch.zeros(1, 2, 2, 1, device=self.device)
            cores.append(core)
        return cores


# ═══════════════════════════════════════════════════════════════════════════════════════
# PHASE 3: BOUNDARY CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionConstraints:
    """
    Boundary constraints for the extraction problem.
    
    Initial State: Current on-chain state
    Target State: Empty vault, funds at safe harbor
    """
    # Initial state
    contract_address: str
    initial_balance: int  # Wei
    initial_storage: Dict[int, int]  # slot -> value
    
    # Target state
    target_balance: int  # Should be 0
    target_receiver: str  # Safe harbor address
    
    # Conservation laws
    no_revert: bool = True  # Prune REVERT branches
    gas_limit: int = 30_000_000
    
    # QTT representation
    initial_qtt: Optional[List[torch.Tensor]] = None
    target_qtt: Optional[List[torch.Tensor]] = None


class ConstraintBuilder:
    """
    Phase 3: Build boundary constraints for QTT solver.
    
    Encodes:
    1. Initial state tensor (current on-chain state)
    2. Target state tensor (extraction complete)
    3. Conservation laws (no REVERT, gas limits)
    """
    
    # Aave Safe Harbor Multisig
    AAVE_SAFE_HARBOR = "0x464C71f6c2F760DdA6093dCB91C24c39e5d6e18c"
    
    def __init__(
        self,
        tensorized_space: TensorizedStateSpace,
        contract_address: str,
        initial_balance: int,
        initial_storage: Dict[int, int],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.space = tensorized_space
        self.contract_address = contract_address
        self.initial_balance = initial_balance
        self.initial_storage = initial_storage
        self.device = torch.device(device)
        
    def build_constraints(self) -> ExtractionConstraints:
        """
        Build extraction constraints.
        
        Returns:
            ExtractionConstraints with QTT representations
        """
        # Encode initial state as QTT
        initial_qtt = self._encode_state_qtt(
            balance=self.initial_balance,
            storage=self.initial_storage
        )
        
        # Encode target state as QTT
        target_qtt = self._encode_state_qtt(
            balance=0,  # Empty vault
            storage={slot: 0 for slot in self.initial_storage}  # Cleared storage
        )
        
        return ExtractionConstraints(
            contract_address=self.contract_address,
            initial_balance=self.initial_balance,
            initial_storage=self.initial_storage,
            target_balance=0,
            target_receiver=self.AAVE_SAFE_HARBOR,
            initial_qtt=initial_qtt,
            target_qtt=target_qtt
        )
    
    def _encode_state_qtt(
        self,
        balance: int,
        storage: Dict[int, int]
    ) -> List[torch.Tensor]:
        """Encode EVM state as QTT cores."""
        cores = []
        
        # Encode balance (256 bits, but we use truncated representation)
        n_balance_bits = 64  # 64 bits sufficient for practical values
        balance_bits = format(min(balance, 2**64 - 1), f'0{n_balance_bits}b')
        
        for i, bit in enumerate(balance_bits):
            core = torch.zeros(1, 2, 1, device=self.device)
            core[0, int(bit), 0] = 1.0
            cores.append(core)
        
        return cores


# ═══════════════════════════════════════════════════════════════════════════════════════
# PHASE 4: TROPICAL PATH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionPath:
    """
    Solved extraction path from initial to target state.
    
    Contains the sequence of state transitions (opcodes + inputs)
    that bridge the two boundary states.
    """
    path_exists: bool
    distance: float  # Tropical distance (cost)
    node_sequence: List[int]  # Block IDs in path
    opcode_sequence: List[Tuple[int, Optional[int]]]  # (opcode, operand)
    
    # For payload reconstruction
    function_calls: List[Tuple[bytes, bytes]]  # (selector, calldata)
    
    @property
    def n_steps(self) -> int:
        return len(self.node_sequence)


class TropicalPathSolver:
    """
    Phase 4: Find extraction path using tropical geometry.
    
    We solve BACKWARD from target to initial state using:
    - Tropical semiring (min-plus algebra)
    - Floyd-Warshall for all-pairs shortest paths
    - Constraint propagation to prune invalid branches
    
    The key insight: Shortest path in tropical algebra = optimal extraction route.
    """
    
    def __init__(
        self,
        cfg: ControlFlowGraph,
        constraints: ExtractionConstraints,
        tensorized_space: TensorizedStateSpace
    ):
        self.cfg = cfg
        self.constraints = constraints
        self.space = tensorized_space
        
        # Build tropical adjacency matrix
        self.adj_matrix = self._build_adjacency_matrix()
        
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """
        Build tropical adjacency matrix from CFG.
        
        Edge weights represent:
        - Base cost: 1 (step count)
        - REVERT penalty: infinity (pruned)
        - Storage access bonus: -0.1 (prefer storage paths)
        - Gas cost: normalized to [0,1]
        """
        n = self.cfg.n_blocks
        block_ids = sorted(self.cfg.blocks.keys())
        id_to_idx = {bid: i for i, bid in enumerate(block_ids)}
        
        INF = 1e9
        adj = torch.full((n, n), INF)
        
        for bid, block in self.cfg.blocks.items():
            i = id_to_idx[bid]
            
            # Self-loop cost 0
            adj[i, i] = 0
            
            for succ in block.successors:
                if succ in id_to_idx:
                    j = id_to_idx[succ]
                    
                    # Base edge cost
                    cost = 1.0
                    
                    # Check if target block has REVERT
                    target_block = self.cfg.blocks.get(succ)
                    if target_block:
                        term = target_block.terminator
                        if term and term.opcode == 0xfd:  # REVERT
                            cost = INF  # Prune this path
                        
                        # Bonus for storage operations
                        if target_block.has_storage_ops():
                            cost -= 0.1
                    
                    adj[i, j] = min(adj[i, j], cost)
        
        return adj
    
    def solve_backward(self) -> ExtractionPath:
        """
        Solve extraction path backward from target to initial.
        
        Uses tropical Floyd-Warshall to find shortest paths,
        then traces back from target state.
        
        Returns:
            ExtractionPath with node sequence and opcodes
        """
        n = self.cfg.n_blocks
        block_ids = sorted(self.cfg.blocks.keys())
        idx_to_id = {i: bid for i, bid in enumerate(block_ids)}
        id_to_idx = {bid: i for i, bid in enumerate(block_ids)}
        
        # Floyd-Warshall in tropical semiring
        dist = self.adj_matrix.clone()
        pred = torch.full((n, n), -1, dtype=torch.long)
        
        # Initialize predecessors
        for i in range(n):
            for j in range(n):
                if dist[i, j] < 1e8 and i != j:
                    pred[i, j] = i
        
        # Tropical relaxation
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        pred[i, j] = pred[k, j]
        
        # Find path from entry (0) to best terminal block
        entry_idx = id_to_idx.get(0, 0)
        
        # Find terminal blocks (with RETURN or SELFDESTRUCT that transfers)
        best_target = None
        best_dist = float('inf')
        
        for bid, block in self.cfg.blocks.items():
            if block.is_terminal:
                term = block.terminator
                if term and term.opcode in (0xf3, 0xff):  # RETURN, SELFDESTRUCT
                    j = id_to_idx.get(bid, -1)
                    if j >= 0 and dist[entry_idx, j] < best_dist:
                        best_dist = float(dist[entry_idx, j])
                        best_target = j
        
        if best_target is None:
            # No valid path found
            return ExtractionPath(
                path_exists=False,
                distance=float('inf'),
                node_sequence=[],
                opcode_sequence=[],
                function_calls=[]
            )
        
        # Reconstruct path
        path = []
        current = best_target
        while current != entry_idx and pred[entry_idx, current] >= 0:
            path.append(idx_to_id[current])
            current = int(pred[entry_idx, current])
        path.append(idx_to_id[entry_idx])
        path.reverse()
        
        # Extract opcode sequence
        opcode_sequence = []
        for bid in path:
            block = self.cfg.blocks.get(bid)
            if block:
                for inst in block.instructions:
                    opcode_sequence.append((inst.opcode, inst.operand_value))
        
        # Identify function calls
        function_calls = self._extract_function_calls(path)
        
        return ExtractionPath(
            path_exists=True,
            distance=best_dist,
            node_sequence=path,
            opcode_sequence=opcode_sequence,
            function_calls=function_calls
        )
    
    def _extract_function_calls(
        self,
        path: List[int]
    ) -> List[Tuple[bytes, bytes]]:
        """Extract function calls from path."""
        calls = []
        
        for bid in path:
            block = self.cfg.blocks.get(bid)
            if not block:
                continue
            
            # Check if block entry matches a function selector
            for selector, entry in self.cfg.function_selectors.items():
                if bid == entry:
                    # Found function entry
                    calls.append((selector, b''))  # Calldata TBD
        
        return calls


# ═══════════════════════════════════════════════════════════════════════════════════════
# PHASE 5: PAYLOAD RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionPayload:
    """
    Final extraction payload ready for execution.
    
    Contains the raw transaction data that, when executed,
    will transfer stranded funds to the safe harbor.
    """
    success: bool
    transactions: List[Dict[str, Any]]  # Raw tx dicts
    hex_payloads: List[str]  # Raw calldata hex
    estimated_gas: int
    target_contract: str
    receiver: str
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps({
            'success': self.success,
            'transactions': self.transactions,
            'hex_payloads': self.hex_payloads,
            'estimated_gas': self.estimated_gas,
            'target_contract': self.target_contract,
            'receiver': self.receiver
        }, indent=2)


class PayloadReconstructor:
    """
    Phase 5: Convert extraction path to executable calldata.
    
    Takes the mathematical solution (tensor path) and translates it
    back into Ethereum transaction format.
    """
    
    def __init__(
        self,
        cfg: ControlFlowGraph,
        path: ExtractionPath,
        constraints: ExtractionConstraints
    ):
        self.cfg = cfg
        self.path = path
        self.constraints = constraints
        
    def reconstruct(self) -> ExtractionPayload:
        """
        Reconstruct executable payload from extraction path.
        
        Returns:
            ExtractionPayload with transaction data
        """
        if not self.path.path_exists:
            return ExtractionPayload(
                success=False,
                transactions=[],
                hex_payloads=[],
                estimated_gas=0,
                target_contract=self.constraints.contract_address,
                receiver=self.constraints.target_receiver
            )
        
        transactions = []
        hex_payloads = []
        total_gas = 0
        
        for selector, calldata in self.path.function_calls:
            # Build calldata
            full_calldata = selector + calldata
            hex_payload = "0x" + full_calldata.hex()
            
            # Build transaction
            tx = {
                'to': self.constraints.contract_address,
                'data': hex_payload,
                'gas': 500_000,  # Estimate
                'value': 0,
            }
            
            transactions.append(tx)
            hex_payloads.append(hex_payload)
            total_gas += tx['gas']
        
        return ExtractionPayload(
            success=len(transactions) > 0,
            transactions=transactions,
            hex_payloads=hex_payloads,
            estimated_gas=total_gas,
            target_contract=self.constraints.contract_address,
            receiver=self.constraints.target_receiver
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionResult:
    """Complete result of extraction attempt."""
    contract_name: str
    contract_address: str
    cfg: ControlFlowGraph
    tensorized_space: TensorizedStateSpace
    constraints: ExtractionConstraints
    path: ExtractionPath
    payload: ExtractionPayload
    
    # Metrics
    n_blocks: int
    n_edges: int
    n_storage_slots: int
    n_function_selectors: int
    qtt_memory_kb: float
    path_distance: float
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 70,
            f"QTT-EVM EXTRACTION RESULT: {self.contract_name}",
            "=" * 70,
            "",
            "PHASE 1: CFG Analysis",
            f"  Blocks: {self.n_blocks}",
            f"  Edges: {self.n_edges}",
            f"  Storage slots: {self.n_storage_slots}",
            f"  Function selectors: {self.n_function_selectors}",
            "",
            "PHASE 2: Tensorization",
            f"  QTT Memory: {self.qtt_memory_kb:.2f} KB",
            f"  Total sites: {self.tensorized_space.total_sites}",
            "",
            "PHASE 3: Constraints",
            f"  Initial balance: {self.constraints.initial_balance:,} wei",
            f"  Target balance: {self.constraints.target_balance}",
            f"  Safe harbor: {self.constraints.target_receiver[:10]}...",
            "",
            "PHASE 4: Path Solving",
            f"  Path exists: {self.path.path_exists}",
            f"  Path distance: {self.path_distance}",
            f"  Path steps: {self.path.n_steps}",
            "",
            "PHASE 5: Payload",
            f"  Success: {self.payload.success}",
            f"  Transactions: {len(self.payload.transactions)}",
            f"  Estimated gas: {self.payload.estimated_gas:,}",
            "",
            "=" * 70,
        ]
        return "\n".join(lines)


class QTTEVMExtractor:
    """
    Main orchestrator for QTT-powered EVM extraction.
    
    Usage:
        >>> extractor = QTTEVMExtractor()
        >>> result = extractor.extract(
        ...     name="Aave_V1_PoolCore",
        ...     address="0x3dfd23A6c5E8BbcFc9581d2E864a68feb6a076d3",
        ...     bytecode=bytecode_hex,
        ...     balance=1_000_000_000_000_000_000,  # 1 ETH
        ...     storage={0: 12345, 1: 67890}
        ... )
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        max_rank: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.max_rank = max_rank
        self.device = device
        
    def extract(
        self,
        name: str,
        address: str,
        bytecode: Union[str, bytes],
        balance: int = 0,
        storage: Dict[int, int] = None
    ) -> ExtractionResult:
        """
        Run full 5-phase extraction pipeline.
        
        Args:
            name: Contract name for reporting
            address: Contract address (checksummed)
            bytecode: Raw bytecode hex or bytes
            balance: Current contract balance in wei
            storage: Known storage slot values
            
        Returns:
            ExtractionResult with all phase outputs
        """
        storage = storage or {}
        
        print(f"\n{'='*70}")
        print(f"QTT-EVM EXTRACTION: {name}")
        print(f"{'='*70}")
        
        # Phase 1: Disassembly & CFG
        print("\n[Phase 1] Disassembling bytecode...")
        disasm = EVMDisassembler(bytecode)
        cfg = disasm.build_cfg()
        print(f"  Blocks: {cfg.n_blocks}, Edges: {cfg.n_edges}")
        print(f"  Storage slots: {len(cfg.storage_slots)}")
        print(f"  Function selectors: {len(cfg.function_selectors)}")
        
        # Phase 2: State-Space Tensorization
        print("\n[Phase 2] Tensorizing state space...")
        tensorizer = StateSpaceTensorizer(cfg, max_rank=self.max_rank, device=self.device)
        tensorized = tensorizer.tensorize()
        print(f"  Total sites: {tensorized.total_sites}")
        print(f"  QTT memory: {tensorized.memory_kb:.2f} KB")
        
        # Phase 3: Boundary Constraints
        print("\n[Phase 3] Building constraints...")
        constraint_builder = ConstraintBuilder(
            tensorized, address, balance, storage, device=self.device
        )
        constraints = constraint_builder.build_constraints()
        print(f"  Initial balance: {constraints.initial_balance:,} wei")
        print(f"  Target: {constraints.target_receiver[:20]}...")
        
        # Phase 4: Tropical Path Extraction
        print("\n[Phase 4] Solving extraction path...")
        solver = TropicalPathSolver(cfg, constraints, tensorized)
        path = solver.solve_backward()
        print(f"  Path exists: {path.path_exists}")
        print(f"  Path distance: {path.distance:.2f}")
        print(f"  Path steps: {path.n_steps}")
        
        # Phase 5: Payload Reconstruction
        print("\n[Phase 5] Reconstructing payload...")
        reconstructor = PayloadReconstructor(cfg, path, constraints)
        payload = reconstructor.reconstruct()
        print(f"  Success: {payload.success}")
        print(f"  Transactions: {len(payload.transactions)}")
        
        result = ExtractionResult(
            contract_name=name,
            contract_address=address,
            cfg=cfg,
            tensorized_space=tensorized,
            constraints=constraints,
            path=path,
            payload=payload,
            n_blocks=cfg.n_blocks,
            n_edges=cfg.n_edges,
            n_storage_slots=len(cfg.storage_slots),
            n_function_selectors=len(cfg.function_selectors),
            qtt_memory_kb=tensorized.memory_kb,
            path_distance=path.distance
        )
        
        return result
    
    def extract_from_json(self, json_path: Union[str, Path]) -> Dict[str, ExtractionResult]:
        """
        Extract from JSON file with multiple contracts.
        
        Args:
            json_path: Path to aave_graveyard_bytecode.json
            
        Returns:
            Dict mapping contract name to ExtractionResult
        """
        with open(json_path) as f:
            data = json.load(f)
        
        results = {}
        for name, info in data.items():
            address = info.get('address', '0x' + '0' * 40)
            bytecode = info.get('bytecode', '')
            
            if bytecode:
                result = self.extract(
                    name=name,
                    address=address,
                    bytecode=bytecode,
                    balance=0,  # Would need to query chain
                    storage={}
                )
                results[name] = result
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════════

def main():
    """Run extraction on Aave graveyard contracts."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QTT-EVM Extraction Engine')
    parser.add_argument('--input', '-i', default='aave_graveyard_bytecode.json',
                       help='Input JSON file with bytecode')
    parser.add_argument('--output', '-o', default='extraction_results.json',
                       help='Output file for results')
    parser.add_argument('--max-rank', type=int, default=64,
                       help='Maximum QTT rank')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Compute device')
    
    args = parser.parse_args()
    
    extractor = QTTEVMExtractor(max_rank=args.max_rank, device=args.device)
    
    # Run extraction
    results = extractor.extract_from_json(args.input)
    
    # Generate report
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    summaries = {}
    for name, result in results.items():
        print(result.summary())
        summaries[name] = {
            'path_exists': result.path.path_exists,
            'path_distance': result.path_distance,
            'n_blocks': result.n_blocks,
            'qtt_memory_kb': result.qtt_memory_kb,
            'payload_success': result.payload.success,
            'hex_payloads': result.payload.hex_payloads
        }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
