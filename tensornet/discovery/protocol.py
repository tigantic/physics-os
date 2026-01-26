"""
PROTOCOL — Genesis Primitive Interface Specification

Defines the GenesisPrimitive protocol that all seven primitives implement.
This enables:
    - Uniform API across OT, SGW, RMT, RKHS, PH, GA
    - Chainable pipelines: OT → SGW → RKHS → PH → GA
    - Type-safe composition
    - Automatic profiling and attestation

Constitutional Reference: CONSTITUTION.md, Article II (Code Architecture)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar, Union

import torch


class PrimitiveType(Enum):
    """Genesis primitive identifiers."""
    
    OT = auto()      # Optimal Transport (Layer 20)
    SGW = auto()     # Spectral Graph Wavelets (Layer 21)
    RMT = auto()     # Random Matrix Theory (Layer 22)
    RKHS = auto()    # Kernel Methods (Layer 24)
    PH = auto()      # Persistent Homology (Layer 25)
    GA = auto()      # Geometric Algebra (Layer 26)
    TG = auto()      # Tropical Geometry (Layer 23) - reserved


@dataclass
class PrimitiveConfig:
    """
    Configuration for a Genesis primitive.
    
    All primitives share common configuration options plus
    primitive-specific parameters in the `params` dict.
    
    Attributes:
        primitive_type: Which Genesis layer this configures
        rank_budget: Maximum QTT rank allowed
        tolerance: Numerical tolerance for operations
        dtype: Torch data type (default: float64)
        device: Torch device (default: cpu)
        seed: Random seed for reproducibility
        params: Primitive-specific parameters
    """
    
    primitive_type: PrimitiveType
    rank_budget: int = 64
    tolerance: float = 1e-10
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 42
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Set manual seed for reproducibility (Constitutional requirement)."""
        torch.manual_seed(self.seed)
    
    def with_params(self, **kwargs) -> "PrimitiveConfig":
        """Return a new config with updated params."""
        new_params = {**self.params, **kwargs}
        return PrimitiveConfig(
            primitive_type=self.primitive_type,
            rank_budget=self.rank_budget,
            tolerance=self.tolerance,
            dtype=self.dtype,
            device=self.device,
            seed=self.seed,
            params=new_params,
        )


@dataclass
class PrimitiveResult:
    """
    Result from a Genesis primitive operation.
    
    Every primitive returns a PrimitiveResult that can be passed
    to the next primitive in a chain.
    
    Attributes:
        primitive_type: Which primitive produced this result
        data: Primary output data (type depends on primitive)
        metadata: Additional information about the computation
        elapsed_time: Wall-clock time for computation (seconds)
        memory_peak: Peak memory usage (bytes)
        qtt_rank: Maximum QTT rank used
        findings: Any findings detected during processing
    """
    
    primitive_type: PrimitiveType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_time: float = 0.0
    memory_peak: int = 0
    qtt_rank: int = 0
    findings: List[Any] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if result contains valid data."""
        return self.data is not None
    
    def as_tensor(self) -> Optional[torch.Tensor]:
        """Attempt to extract result as torch tensor."""
        if isinstance(self.data, torch.Tensor):
            return self.data
        if hasattr(self.data, 'to_tensor'):
            return self.data.to_tensor()
        if hasattr(self.data, 'dense'):
            return self.data.dense()
        return None
    
    def chain_to(self, next_primitive: "GenesisPrimitive") -> "PrimitiveResult":
        """Pass this result to the next primitive in a chain."""
        return next_primitive.process(self)


# Type variable for generic primitive inputs/outputs
T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")


class ChainableInterface(Protocol):
    """Protocol for chainable operations."""
    
    def process(self, input_data: Any) -> PrimitiveResult:
        """Process input and return result."""
        ...
    
    def chain(self, next_step: "ChainableInterface") -> "ChainableInterface":
        """Chain with another step."""
        ...


class GenesisPrimitive(ABC):
    """
    Abstract base class for all Genesis primitives.
    
    Every Genesis primitive MUST implement:
        - process(): Main computation method
        - detect_anomalies(): Find unusual patterns
        - detect_invariants(): Find conservation laws
        - detect_bottlenecks(): Find computational/physical bottlenecks
        - predict(): Make predictions based on patterns
    
    Example:
        >>> class OptimalTransportPrimitive(GenesisPrimitive):
        ...     def process(self, input_data: Any) -> PrimitiveResult:
        ...         # Compute optimal transport
        ...         return PrimitiveResult(
        ...             primitive_type=PrimitiveType.OT,
        ...             data=transport_plan,
        ...         )
    """
    
    def __init__(self, config: Optional[PrimitiveConfig] = None) -> None:
        """
        Initialize primitive with configuration.
        
        Args:
            config: Primitive configuration. If None, uses defaults.
        """
        if config is None:
            config = PrimitiveConfig(primitive_type=self.primitive_type)
        self.config = config
        self._setup()
    
    @property
    @abstractmethod
    def primitive_type(self) -> PrimitiveType:
        """Return the primitive type for this class."""
        ...
    
    @property
    def name(self) -> str:
        """Human-readable primitive name."""
        return self.primitive_type.name
    
    @abstractmethod
    def _setup(self) -> None:
        """Initialize primitive-specific resources."""
        ...
    
    @abstractmethod
    def process(self, input_data: Any) -> PrimitiveResult:
        """
        Main processing method.
        
        Args:
            input_data: Input data (raw or PrimitiveResult from previous stage)
            
        Returns:
            PrimitiveResult containing processed data and metadata
        """
        ...
    
    @abstractmethod
    def detect_anomalies(self, data: Any) -> List[Any]:
        """
        Detect anomalous patterns in data.
        
        Args:
            data: Data to analyze (raw or from process())
            
        Returns:
            List of AnomalyFinding objects
        """
        ...
    
    @abstractmethod
    def detect_invariants(self, data: Any) -> List[Any]:
        """
        Detect conservation laws and invariants.
        
        Args:
            data: Data to analyze
            
        Returns:
            List of InvariantFinding objects
        """
        ...
    
    @abstractmethod
    def detect_bottlenecks(self, data: Any) -> List[Any]:
        """
        Detect computational or physical bottlenecks.
        
        Args:
            data: Data to analyze
            
        Returns:
            List of BottleneckFinding objects
        """
        ...
    
    @abstractmethod
    def predict(self, data: Any) -> List[Any]:
        """
        Make predictions based on patterns in data.
        
        Args:
            data: Data to analyze
            
        Returns:
            List of PredictionFinding objects
        """
        ...
    
    def discover(self, input_data: Any) -> PrimitiveResult:
        """
        Full discovery pipeline: process + all detection methods.
        
        Args:
            input_data: Input data to analyze
            
        Returns:
            PrimitiveResult with all findings attached
        """
        start_time = time.perf_counter()
        
        # Process data
        result = self.process(input_data)
        
        # Run all detection methods
        all_findings = []
        all_findings.extend(self.detect_anomalies(result.data))
        all_findings.extend(self.detect_invariants(result.data))
        all_findings.extend(self.detect_bottlenecks(result.data))
        all_findings.extend(self.predict(result.data))
        
        # Attach findings to result
        result.findings = all_findings
        result.elapsed_time = time.perf_counter() - start_time
        
        return result
    
    def chain(self, next_primitive: "GenesisPrimitive") -> "PrimitiveChain":
        """
        Chain this primitive with another.
        
        Args:
            next_primitive: Primitive to chain after this one
            
        Returns:
            PrimitiveChain that executes both in sequence
        """
        return PrimitiveChain([self, next_primitive])
    
    def __rshift__(self, other: "GenesisPrimitive") -> "PrimitiveChain":
        """
        Chain operator: primitive1 >> primitive2.
        
        Example:
            >>> pipeline = ot_primitive >> sgw_primitive >> ph_primitive
        """
        return self.chain(other)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


class PrimitiveChain:
    """
    Chain of Genesis primitives executed in sequence.
    
    Implements the cross-primitive pipeline pattern:
    OT → SGW → RKHS → PH → GA
    
    Example:
        >>> chain = PrimitiveChain([ot, sgw, ph])
        >>> results = chain.execute(input_data)
        >>> final_result = results[-1]
        >>> all_findings = chain.collect_findings()
    """
    
    def __init__(self, primitives: List[GenesisPrimitive]) -> None:
        """
        Initialize chain with primitives.
        
        Args:
            primitives: List of primitives in execution order
        """
        if not primitives:
            raise ValueError("PrimitiveChain requires at least one primitive")
        self.primitives = primitives
    
    def execute(self, input_data: Any) -> List[PrimitiveResult]:
        """
        Execute all primitives in sequence.
        
        Args:
            input_data: Initial input data
            
        Returns:
            List of PrimitiveResult from each stage
        """
        results = []
        current_input = input_data
        
        for primitive in self.primitives:
            result = primitive.discover(current_input)
            results.append(result)
            current_input = result
        
        return results
    
    def collect_findings(self, results: List[PrimitiveResult]) -> List[Any]:
        """
        Collect all findings from all stages.
        
        Args:
            results: List of results from execute()
            
        Returns:
            Flat list of all findings
        """
        all_findings = []
        for result in results:
            all_findings.extend(result.findings)
        return all_findings
    
    def add(self, primitive: GenesisPrimitive) -> "PrimitiveChain":
        """Add a primitive to the chain."""
        return PrimitiveChain(self.primitives + [primitive])
    
    def __rshift__(self, other: GenesisPrimitive) -> "PrimitiveChain":
        """Chain operator: chain >> primitive."""
        return self.add(other)
    
    def __len__(self) -> int:
        return len(self.primitives)
    
    def __iter__(self):
        return iter(self.primitives)
    
    def __repr__(self) -> str:
        names = " >> ".join(p.name for p in self.primitives)
        return f"PrimitiveChain({names})"
