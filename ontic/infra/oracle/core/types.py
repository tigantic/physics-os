"""
Core type definitions for ORACLE.

These dataclasses define the fundamental data structures passed between
all phases of the vulnerability hunting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import torch


# =============================================================================
# Phase 1: Parsing Types
# =============================================================================


@dataclass
class StateVariable:
    """A contract state variable."""
    
    name: str
    type_name: str
    visibility: str  # public, private, internal
    slot: Optional[int] = None  # storage slot if known
    initial_value: Optional[str] = None
    line: int = 0


@dataclass
class Parameter:
    """A function parameter."""
    
    name: str
    type_name: str
    indexed: bool = False  # for events


@dataclass
class Function:
    """A contract function with its metadata."""
    
    name: str
    visibility: str  # public, external, internal, private
    mutability: str  # pure, view, payable, nonpayable
    parameters: list[Parameter] = field(default_factory=list)
    returns: list[Parameter] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    source: str = ""
    start_line: int = 0
    end_line: int = 0
    
    # Analysis results (populated later)
    reads_state: list[str] = field(default_factory=list)
    writes_state: list[str] = field(default_factory=list)
    external_calls: list[str] = field(default_factory=list)
    
    @property
    def is_public(self) -> bool:
        return self.visibility in ("public", "external")
    
    @property
    def can_receive_eth(self) -> bool:
        return self.mutability == "payable"


@dataclass
class Event:
    """A contract event."""
    
    name: str
    parameters: list[Parameter] = field(default_factory=list)
    line: int = 0


@dataclass
class Modifier:
    """A function modifier."""
    
    name: str
    parameters: list[Parameter] = field(default_factory=list)
    source: str = ""
    line: int = 0


@dataclass
class Contract:
    """A parsed Solidity contract."""
    
    name: str
    kind: str = "contract"  # contract, interface, library, abstract
    is_abstract: bool = False
    inherits: list[str] = field(default_factory=list)
    
    state_variables: list[StateVariable] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    modifiers: list[Modifier] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    
    source: str = ""
    file_path: Optional[str] = None
    
    def get_function(self, name: str) -> Optional[Function]:
        """Get function by name."""
        for f in self.functions:
            if f.name == name:
                return f
        return None
    
    @property
    def public_functions(self) -> list[Function]:
        """All externally callable functions."""
        return [f for f in self.functions if f.is_public]
    
    @property
    def payable_functions(self) -> list[Function]:
        """All functions that can receive ETH."""
        return [f for f in self.functions if f.can_receive_eth]


# =============================================================================
# Control Flow & Data Flow Graphs
# =============================================================================


@dataclass
class CFGNode:
    """A node in the control flow graph."""
    
    id: int
    type: str  # entry, exit, block, branch, loop
    source: str
    line_start: int
    line_end: int


@dataclass
class CFGEdge:
    """An edge in the control flow graph."""
    
    from_node: int
    to_node: int
    condition: Optional[str] = None  # for branch edges


@dataclass
class ControlFlowGraph:
    """Control flow graph for a function."""
    
    function_name: str
    nodes: list[CFGNode] = field(default_factory=list)
    edges: list[CFGEdge] = field(default_factory=list)
    entry: Optional[int] = None
    exit: Optional[int] = None


@dataclass
class DFGNode:
    """A node in the data flow graph."""
    
    id: int
    variable: str
    definition_line: int
    is_parameter: bool = False
    is_state: bool = False


@dataclass
class DFGEdge:
    """A data dependency edge."""
    
    from_node: int  # definition
    to_node: int    # use
    type: str = "use"  # use, def-use, phi


@dataclass
class DataFlowGraph:
    """Data flow graph for a function."""
    
    function_name: str
    nodes: list[DFGNode] = field(default_factory=list)
    edges: list[DFGEdge] = field(default_factory=list)


@dataclass
class CallEdge:
    """An edge in the call graph."""
    
    caller: str
    callee: str
    call_type: str = "internal"  # internal, external, delegatecall
    line: int = 0


@dataclass
class CallGraph:
    """Inter-function call graph."""
    
    contract_name: str
    edges: list[CallEdge] = field(default_factory=list)
    
    def callees(self, function: str) -> list[str]:
        """Get all functions called by a function."""
        return [e.callee for e in self.edges if e.caller == function]
    
    def callers(self, function: str) -> list[str]:
        """Get all functions that call a function."""
        return [e.caller for e in self.edges if e.callee == function]


# =============================================================================
# Phase 1: Semantic Analysis Types
# =============================================================================


@dataclass
class Actor:
    """An actor who interacts with the contract."""
    
    name: str
    role: str  # depositor, borrower, liquidator, admin, oracle, etc.
    capabilities: list[str] = field(default_factory=list)
    trust_level: str = "untrusted"  # trusted, semi-trusted, untrusted


@dataclass
class ValueFlow:
    """How value/tokens flow through the system."""
    
    from_actor: str
    to_actor: str
    asset: str
    condition: str
    function: str


@dataclass
class IntentAnalysis:
    """Semantic understanding of what the contract does."""
    
    protocol_type: str  # lending, dex, vault, bridge, staking, etc.
    description: str
    actors: list[Actor] = field(default_factory=list)
    value_flows: list[ValueFlow] = field(default_factory=list)
    trust_assumptions: list[str] = field(default_factory=list)
    key_invariants: list[str] = field(default_factory=list)


# =============================================================================
# Phase 2: Assumption Types
# =============================================================================


class AssumptionType(Enum):
    """Categories of assumptions."""
    
    EXPLICIT = "explicit"      # require(), assert()
    IMPLICIT = "implicit"      # not checked but assumed
    ECONOMIC = "economic"      # about rational actors
    TEMPORAL = "temporal"      # about time/ordering
    EXTERNAL = "external"      # about external contracts/oracles
    ARITHMETIC = "arithmetic"  # about overflow/precision


@dataclass
class Assumption:
    """An assumption the code makes."""
    
    id: str  # A001, A002, etc.
    type: AssumptionType
    source: str  # function name or "global"
    statement: str  # human-readable assumption
    formal: Optional[str] = None  # formal representation
    confidence: float = 1.0  # how confident we are this is real
    line: Optional[int] = None
    
    # For explicit assumptions
    condition_ast: Optional[Any] = None
    revert_message: Optional[str] = None


# =============================================================================
# Phase 3: Challenge Types
# =============================================================================


class ImpactLevel(Enum):
    """Severity levels matching Immunefi standards."""
    
    CRITICAL = "critical"        # Direct fund theft
    HIGH = "high"                # Significant fund loss
    MEDIUM = "medium"            # Limited fund loss or DoS
    LOW = "low"                  # Minor issues
    INFORMATIONAL = "info"       # No direct impact
    
    def __lt__(self, other: ImpactLevel) -> bool:
        order = [self.INFORMATIONAL, self.LOW, self.MEDIUM, self.HIGH, self.CRITICAL]
        return order.index(self) < order.index(other)


@dataclass
class ExecutionPath:
    """A sequence of function calls reaching a state."""
    
    calls: list[tuple[str, dict[str, Any]]]  # [(function, params), ...]
    constraints: list[str] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class Challenge:
    """Result of challenging an assumption."""
    
    assumption: Assumption
    negation: str  # "What if this is FALSE?"
    reachable: bool  # Can we actually violate it?
    reachability_proof: Optional[ExecutionPath] = None
    impact: ImpactLevel = ImpactLevel.INFORMATIONAL
    impact_description: str = ""
    exploit_sketch: Optional[str] = None


# =============================================================================
# Phase 4: Attack Scenario Types
# =============================================================================


@dataclass
class AttackStep:
    """A single step in an attack."""
    
    action: str  # "flash_loan", "swap", "call", etc.
    target: str  # contract/function
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    description: str = ""  # Alias for expected_effect


@dataclass
class AttackScenario:
    """A complete attack scenario."""
    
    name: str
    description: str
    steps: list[AttackStep] = field(default_factory=list)
    required_capital: int = 0  # in wei
    expected_profit: int = 0   # in wei
    prerequisites: list[str] = field(default_factory=list)
    complexity: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    challenges_addressed: list[str] = field(default_factory=list)


# =============================================================================
# Phase 5: Verification Types
# =============================================================================


@dataclass
class ExecutionTrace:
    """Detailed trace of transaction execution."""
    
    opcodes: list[dict[str, Any]] = field(default_factory=list)
    storage_reads: list[tuple[str, int, int]] = field(default_factory=list)   # (contract, slot, value)
    storage_writes: list[tuple[str, int, int, int]] = field(default_factory=list)  # (contract, slot, old, new)
    balance_changes: list[tuple[str, int, int]] = field(default_factory=list)  # (address, old, new)
    external_calls: list[tuple[str, str, bytes]] = field(default_factory=list)  # (target, func, data)
    events: list[tuple[str, list[Any]]] = field(default_factory=list)
    gas_used: int = 0
    reverted: bool = False
    revert_reason: Optional[str] = None


@dataclass
class ChiMetrics:
    """Exploit proximity metrics."""
    
    storage_writes: int = 0
    unique_slots: int = 0
    call_depth: int = 0
    balance_delta: int = 0
    revert_distance: int = 0
    
    @property
    def chi(self) -> float:
        """Compute Chi value - higher = closer to exploit."""
        return (
            self.storage_writes * 1.0 +
            self.unique_slots * 2.0 +
            self.call_depth * 5.0 +
            abs(self.balance_delta) / 1e18 * 10.0 +
            (1.0 / (self.revert_distance + 1)) * 20.0
        )


@dataclass
class IntervalProof:
    """Proof via interval arithmetic."""
    
    profit_bounds: tuple[float, float]  # (lo, hi)
    guaranteed: bool
    state_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class ConcreteProof:
    """Proof via concrete execution."""
    
    traces: list[ExecutionTrace]
    profit: int
    chi: float
    block_number: int


@dataclass
class KantorovichProof:
    """Proof via Kantorovich theorem."""
    
    discriminant: float
    residual_bound: float
    stability_bound: float
    certified: bool


@dataclass 
class VerifiedExploit:
    """A verified, exploitable vulnerability."""
    
    scenario: AttackScenario
    verification_method: str  # "interval", "concrete", "kantorovich", "mainnet_fork"
    proof: Optional[Union[IntervalProof, ConcreteProof, KantorovichProof]] = None
    confidence: float = 0.0  # 0.0 - 1.0
    foundry_test: str = ""
    fork_profit_wei: int = 0  # Actual profit measured on fork
    fork_block: int = 0  # Block number of fork test


# =============================================================================
# Phase 6: Report Types
# =============================================================================


@dataclass
class Report:
    """A submission-ready vulnerability report."""
    
    title: str
    severity: str
    summary: str
    vulnerability_details: str
    impact: str
    proof_of_concept: str
    tools_used: str
    recommendation: str
    
    def to_immunefi_markdown(self) -> str:
        """Export as Immunefi markdown format."""
        return f"""# {self.title}

## Summary
{self.summary}

## Vulnerability Details
{self.vulnerability_details}

## Impact
{self.impact}

## Proof of Concept

```solidity
{self.proof_of_concept}
```

## Recommendation
{self.recommendation}

## Tools Used
{self.tools_used}
"""
    
    def save(self, path: str) -> None:
        """Save report to file."""
        import os
        os.makedirs(path, exist_ok=True)
        filename = self.title.lower().replace(" ", "_") + ".md"
        with open(os.path.join(path, filename), "w") as f:
            f.write(self.to_immunefi_markdown())


# =============================================================================
# Pipeline Types
# =============================================================================


@dataclass
class HuntResult:
    """Complete result of an ORACLE hunt."""
    
    contract: Contract
    intent: IntentAnalysis
    assumptions: list[Assumption]
    challenges: list[Challenge]
    scenarios: list[AttackScenario]
    verified_exploits: list[VerifiedExploit]
    
    # Metadata
    hunt_time_seconds: float = 0.0
    contract_address: Optional[str] = None
    chain: str = "ethereum"
    block_number: Optional[int] = None
    
    @property
    def has_critical(self) -> bool:
        """Check if any critical vulnerabilities found."""
        return any(e.scenario.expected_profit > 100_000 * 10**18 for e in self.verified_exploits)
    
    @property
    def total_potential_profit(self) -> int:
        """Sum of expected profits from all verified exploits."""
        return sum(e.scenario.expected_profit for e in self.verified_exploits)
