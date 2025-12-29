# Copyright (c) 2025 Tigantic
# Phase 18: Kernel Fusion Optimization
"""
Kernel fusion for reducing overhead in tensor network computations.

Implements automatic detection and fusion of compatible operations
to minimize kernel launch overhead and memory bandwidth usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn


class FusionType(Enum):
    """Types of kernel fusion patterns."""
    
    ELEMENT_WISE = auto()     # Element-wise operations (add, mul, etc.)
    REDUCTION = auto()        # Reduction operations (sum, mean, etc.)
    MATMUL_BIAS = auto()      # Matrix multiply + bias add
    MATMUL_ACTIVATION = auto() # Matrix multiply + activation
    CONTRACTION = auto()      # Tensor contractions
    NORMALIZE = auto()        # Normalization patterns
    CUSTOM = auto()           # Custom fusion pattern


@dataclass
class FusionPattern:
    """A fusion pattern definition.
    
    Attributes:
        name: Pattern name
        fusion_type: Type of fusion
        ops: List of operation types in pattern
        fused_op: Fused operation callable
        speedup_estimate: Estimated speedup factor
        memory_reduction: Estimated memory reduction factor
    """
    
    name: str
    fusion_type: FusionType
    ops: List[str]
    fused_op: Optional[Callable] = None
    speedup_estimate: float = 1.0
    memory_reduction: float = 1.0
    
    def matches(self, op_sequence: List[str]) -> bool:
        """Check if this pattern matches an operation sequence.
        
        Args:
            op_sequence: List of operation type strings
            
        Returns:
            True if pattern matches
        """
        if len(op_sequence) < len(self.ops):
            return False
        
        for i in range(len(op_sequence) - len(self.ops) + 1):
            if op_sequence[i:i + len(self.ops)] == self.ops:
                return True
        
        return False


@dataclass
class OperatorNode:
    """A node in the operator graph.
    
    Attributes:
        id: Unique node identifier
        op_type: Type of operation
        inputs: Input node IDs
        outputs: Output node IDs
        op: Actual operation callable
        attributes: Operation attributes/parameters
        can_fuse: Whether this node can be fused
    """
    
    id: int
    op_type: str
    inputs: List[int]
    outputs: List[int]
    op: Optional[Callable] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    can_fuse: bool = True
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OperatorNode):
            return False
        return self.id == other.id


class OperatorGraph:
    """Graph representation of computation.
    
    Represents a computation as a directed acyclic graph (DAG)
    of operator nodes, enabling optimization through fusion.
    
    Attributes:
        nodes: Dictionary of nodes by ID
        inputs: Input node IDs
        outputs: Output node IDs
    """
    
    def __init__(self) -> None:
        """Initialize empty operator graph."""
        self.nodes: Dict[int, OperatorNode] = {}
        self.inputs: List[int] = []
        self.outputs: List[int] = []
        self._next_id = 0
    
    def add_node(
        self,
        op_type: str,
        inputs: Optional[List[int]] = None,
        op: Optional[Callable] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a node to the graph.
        
        Args:
            op_type: Type of operation
            inputs: Input node IDs
            op: Operation callable
            attributes: Operation attributes
            
        Returns:
            New node ID
        """
        node_id = self._next_id
        self._next_id += 1
        
        node = OperatorNode(
            id=node_id,
            op_type=op_type,
            inputs=inputs or [],
            outputs=[],
            op=op,
            attributes=attributes or {},
        )
        
        self.nodes[node_id] = node
        
        # Update output lists of input nodes
        for input_id in node.inputs:
            if input_id in self.nodes:
                self.nodes[input_id].outputs.append(node_id)
        
        return node_id
    
    def add_input(self, name: str = "input") -> int:
        """Add an input node.
        
        Args:
            name: Input name
            
        Returns:
            Input node ID
        """
        node_id = self.add_node("Input", attributes={"name": name})
        self.inputs.append(node_id)
        return node_id
    
    def mark_output(self, node_id: int) -> None:
        """Mark a node as an output.
        
        Args:
            node_id: Node to mark as output
        """
        if node_id not in self.outputs:
            self.outputs.append(node_id)
    
    def get_node(self, node_id: int) -> Optional[OperatorNode]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            OperatorNode or None
        """
        return self.nodes.get(node_id)
    
    def get_predecessors(self, node_id: int) -> List[OperatorNode]:
        """Get predecessor nodes.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of predecessor nodes
        """
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [self.nodes[i] for i in node.inputs if i in self.nodes]
    
    def get_successors(self, node_id: int) -> List[OperatorNode]:
        """Get successor nodes.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of successor nodes
        """
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [self.nodes[i] for i in node.outputs if i in self.nodes]
    
    def topological_sort(self) -> List[int]:
        """Get topologically sorted node IDs.
        
        Returns:
            List of node IDs in topological order
        """
        visited: Set[int] = set()
        result: List[int] = []
        
        def dfs(node_id: int) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if node:
                for input_id in node.inputs:
                    dfs(input_id)
                result.append(node_id)
        
        for node_id in self.nodes:
            dfs(node_id)
        
        return result
    
    def find_fusible_sequences(self) -> List[List[int]]:
        """Find sequences of nodes that can be fused.
        
        Returns:
            List of fusible node ID sequences
        """
        sequences: List[List[int]] = []
        sorted_nodes = self.topological_sort()
        used: Set[int] = set()
        
        for node_id in sorted_nodes:
            if node_id in used:
                continue
            
            node = self.nodes[node_id]
            if not node.can_fuse:
                continue
            
            # Start a new sequence
            sequence = [node_id]
            current_id = node_id
            
            # Extend forward
            while True:
                successors = self.get_successors(current_id)
                if len(successors) != 1:
                    break
                
                succ = successors[0]
                if not succ.can_fuse or succ.id in used:
                    break
                
                # Check that successor only has this predecessor
                preds = self.get_predecessors(succ.id)
                if len(preds) != 1:
                    break
                
                sequence.append(succ.id)
                current_id = succ.id
            
            if len(sequence) > 1:
                sequences.append(sequence)
                used.update(sequence)
        
        return sequences
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "nodes": {
                node_id: {
                    "op_type": node.op_type,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "attributes": node.attributes,
                }
                for node_id, node in self.nodes.items()
            },
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


class FusedOperator(nn.Module):
    """A fused operator combining multiple operations.
    
    Attributes:
        name: Fused operator name
        ops: List of component operators
        fusion_type: Type of fusion applied
    """
    
    def __init__(
        self,
        name: str,
        ops: List[Callable],
        fusion_type: FusionType = FusionType.CUSTOM,
    ) -> None:
        """Initialize fused operator.
        
        Args:
            name: Operator name
            ops: List of operations to fuse
            fusion_type: Type of fusion
        """
        super().__init__()
        self.name = name
        self.ops = ops
        self.fusion_type = fusion_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute fused operations.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        result = x
        for op in self.ops:
            result = op(result)
        return result


class KernelFuser:
    """Automatic kernel fusion optimizer.
    
    Analyzes computation graphs and applies fusion optimizations
    to reduce kernel launch overhead and memory transfers.
    
    Attributes:
        patterns: Registered fusion patterns
        enable_profiling: Whether to profile fused operations
    """
    
    def __init__(self, enable_profiling: bool = False) -> None:
        """Initialize kernel fuser.
        
        Args:
            enable_profiling: Enable profiling of fused operations
        """
        self.patterns: List[FusionPattern] = []
        self.enable_profiling = enable_profiling
        self._register_default_patterns()
    
    def _register_default_patterns(self) -> None:
        """Register default fusion patterns."""
        # Element-wise fusion
        self.patterns.append(FusionPattern(
            name="elementwise_chain",
            fusion_type=FusionType.ELEMENT_WISE,
            ops=["add", "mul"],
            speedup_estimate=1.5,
            memory_reduction=0.5,
        ))
        
        self.patterns.append(FusionPattern(
            name="add_relu",
            fusion_type=FusionType.ELEMENT_WISE,
            ops=["add", "relu"],
            speedup_estimate=1.8,
            memory_reduction=0.5,
        ))
        
        # Matmul + activation
        self.patterns.append(FusionPattern(
            name="matmul_relu",
            fusion_type=FusionType.MATMUL_ACTIVATION,
            ops=["matmul", "relu"],
            speedup_estimate=1.3,
            memory_reduction=0.7,
        ))
        
        self.patterns.append(FusionPattern(
            name="matmul_gelu",
            fusion_type=FusionType.MATMUL_ACTIVATION,
            ops=["matmul", "gelu"],
            speedup_estimate=1.4,
            memory_reduction=0.7,
        ))
        
        # Matmul + bias
        self.patterns.append(FusionPattern(
            name="matmul_bias",
            fusion_type=FusionType.MATMUL_BIAS,
            ops=["matmul", "add"],
            speedup_estimate=1.2,
            memory_reduction=0.8,
        ))
        
        # Normalization
        self.patterns.append(FusionPattern(
            name="normalize_scale",
            fusion_type=FusionType.NORMALIZE,
            ops=["div", "mul"],
            speedup_estimate=1.4,
            memory_reduction=0.6,
        ))
    
    def register_pattern(self, pattern: FusionPattern) -> None:
        """Register a custom fusion pattern.
        
        Args:
            pattern: Fusion pattern to register
        """
        self.patterns.append(pattern)
    
    def analyze_graph(self, graph: OperatorGraph) -> List[Tuple[List[int], FusionPattern]]:
        """Analyze graph for fusion opportunities.
        
        Args:
            graph: Operator graph to analyze
            
        Returns:
            List of (node_ids, pattern) tuples
        """
        opportunities: List[Tuple[List[int], FusionPattern]] = []
        sequences = graph.find_fusible_sequences()
        
        for sequence in sequences:
            # Get operation types for sequence
            op_types = [graph.nodes[node_id].op_type for node_id in sequence]
            
            # Check against patterns
            for pattern in self.patterns:
                if pattern.matches(op_types):
                    opportunities.append((sequence, pattern))
                    break
        
        return opportunities
    
    def fuse_sequence(
        self,
        graph: OperatorGraph,
        node_ids: List[int],
        pattern: FusionPattern,
    ) -> int:
        """Fuse a sequence of nodes.
        
        Args:
            graph: Operator graph
            node_ids: Node IDs to fuse
            pattern: Fusion pattern to apply
            
        Returns:
            ID of new fused node
        """
        if not node_ids:
            raise ValueError("No nodes to fuse")
        
        # Collect operations
        ops = []
        for node_id in node_ids:
            node = graph.nodes[node_id]
            if node.op:
                ops.append(node.op)
        
        # Create fused operator
        fused = FusedOperator(pattern.name, ops, pattern.fusion_type)
        if pattern.fused_op:
            fused_callable = pattern.fused_op
        else:
            fused_callable = fused.forward
        
        # Get inputs from first node
        first_node = graph.nodes[node_ids[0]]
        
        # Get outputs from last node
        last_node = graph.nodes[node_ids[-1]]
        
        # Create new fused node
        fused_id = graph.add_node(
            op_type=f"Fused_{pattern.name}",
            inputs=first_node.inputs.copy(),
            op=fused_callable,
            attributes={
                "pattern": pattern.name,
                "original_nodes": node_ids,
                "speedup_estimate": pattern.speedup_estimate,
            },
        )
        
        # Update connections
        graph.nodes[fused_id].outputs = last_node.outputs.copy()
        
        # Update successor inputs
        for succ_id in last_node.outputs:
            if succ_id in graph.nodes:
                succ = graph.nodes[succ_id]
                succ.inputs = [fused_id if i == node_ids[-1] else i for i in succ.inputs]
        
        # Mark fused nodes for removal
        for node_id in node_ids:
            graph.nodes[node_id].can_fuse = False
        
        return fused_id
    
    def optimize(self, graph: OperatorGraph) -> OperatorGraph:
        """Apply fusion optimizations to graph.
        
        Args:
            graph: Input operator graph
            
        Returns:
            Optimized graph
        """
        opportunities = self.analyze_graph(graph)
        
        for node_ids, pattern in opportunities:
            try:
                self.fuse_sequence(graph, node_ids, pattern)
            except Exception:
                # Skip failed fusions
                continue
        
        return graph
    
    def estimate_speedup(self, graph: OperatorGraph) -> float:
        """Estimate potential speedup from fusion.
        
        Args:
            graph: Operator graph
            
        Returns:
            Estimated speedup factor
        """
        opportunities = self.analyze_graph(graph)
        
        if not opportunities:
            return 1.0
        
        # Product of individual speedups (conservative)
        total_speedup = 1.0
        for _, pattern in opportunities:
            total_speedup *= pattern.speedup_estimate ** 0.5
        
        return total_speedup


def fuse_operators(
    ops: List[Callable],
    fusion_type: FusionType = FusionType.CUSTOM,
    name: str = "fused",
) -> Callable:
    """Create a fused operator from a list of operations.
    
    Args:
        ops: List of operations to fuse
        fusion_type: Type of fusion
        name: Name for the fused operator
        
    Returns:
        Fused operator callable
    """
    fused = FusedOperator(name, ops, fusion_type)
    return fused.forward


def optimize_graph(graph: OperatorGraph) -> OperatorGraph:
    """Optimize an operator graph with fusion.
    
    Args:
        graph: Input operator graph
        
    Returns:
        Optimized graph
    """
    fuser = KernelFuser()
    return fuser.optimize(graph)
