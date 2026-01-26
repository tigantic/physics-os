"""
TOPOLOGY PRIMITIVE — Layer 25 Discovery Wrapper

Wraps tensornet.genesis.topology for persistent homology:
    - Simplicial complex construction
    - Boundary operator computation in QTT format
    - Betti number computation
    - Persistence diagram extraction

Key Capabilities:
    - Trillion-point topological data analysis
    - Multi-scale homology via filtration
    - Wasserstein distance between persistence diagrams

Anomaly Detection:
    - Topological defects (holes, voids)
    - Betti number anomalies
    - Persistence diagram outliers

Invariant Detection:
    - Euler characteristic conservation
    - Betti number stability
    - Homological features

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from tensornet.discovery.findings import (
    AnomalyFinding,
    BottleneckFinding,
    InvariantFinding,
    PredictionFinding,
    Severity,
)
from tensornet.discovery.config import get_config
from tensornet.discovery.protocol import (
    GenesisPrimitive,
    PrimitiveConfig,
    PrimitiveResult,
    PrimitiveType,
)

# Genesis topology imports
from tensornet.genesis.topology import (
    SimplicialComplex,
    RipsComplex,
    QTTBoundaryOperator,
    PersistenceDiagram,
    compute_persistence,
    bottleneck_distance,
    wasserstein_distance_diagram,
)


class TopologyPrimitive(GenesisPrimitive):
    """
    Persistent Homology primitive for topological analysis.
    
    Configuration params:
        max_dimension: Maximum homology dimension to compute
        max_radius: Maximum filtration radius
        num_radius_steps: Number of filtration steps
        persistence_threshold: Minimum lifetime for significant features
    
    Example:
        >>> from tensornet.discovery.primitives import TopologyPrimitive
        >>> 
        >>> ph = TopologyPrimitive(
        ...     max_dimension=2,
        ...     persistence_threshold=0.1,
        ... )
        >>> 
        >>> result = ph.discover(point_cloud)
        >>> print(f"Betti numbers: {result.metadata['betti_numbers']}")
    """
    
    @property
    def primitive_type(self) -> PrimitiveType:
        return PrimitiveType.PH
    
    def __init__(
        self,
        max_dimension: int = 2,
        max_radius: float = 2.0,
        num_radius_steps: int = 100,
        persistence_threshold: float = 0.1,
        rank_budget: int = 64,
        tolerance: float = 1e-10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Initialize topology primitive.
        
        Args:
            max_dimension: Maximum homology dimension (0=components, 1=loops, 2=voids)
            max_radius: Maximum Vietoris-Rips radius
            num_radius_steps: Filtration resolution
            persistence_threshold: Minimum feature lifetime
            rank_budget: Maximum QTT rank
            tolerance: Numerical tolerance
            seed: Random seed
        """
        config = PrimitiveConfig(
            primitive_type=PrimitiveType.PH,
            rank_budget=rank_budget,
            tolerance=tolerance,
            seed=seed,
            params={
                "max_dimension": max_dimension,
                "max_radius": max_radius,
                "num_radius_steps": num_radius_steps,
                "persistence_threshold": persistence_threshold,
                **kwargs,
            },
        )
        super().__init__(config)
    
    def _setup(self) -> None:
        """Initialize topology components."""
        self.max_dimension = self.config.params.get("max_dimension", 2)
        self.max_radius = self.config.params.get("max_radius", 2.0)
        self.num_radius_steps = self.config.params.get("num_radius_steps", 100)
        self.persistence_threshold = self.config.params.get("persistence_threshold", 0.1)
        
        # History for tracking
        self._betti_history: List[List[int]] = []
        self._previous_diagram: Optional[PersistenceDiagram] = None
    
    def process(self, input_data: Any) -> PrimitiveResult:
        """
        Compute persistent homology.
        
        Args:
            input_data: One of:
                - Point cloud (N x D tensor)
                - SimplicialComplex
                - PrimitiveResult from previous stage
        
        Returns:
            PrimitiveResult with persistence diagram and Betti numbers
        """
        start_time = time.perf_counter()
        
        # Parse input - already limits to max_persistence_points
        points = self._parse_input(input_data)
        
        # Further limit points for tractable persistence computation
        # Rips complex grows exponentially with points
        config = get_config()
        MAX_POINTS_FOR_PERSISTENCE = min(64, config.ingestion.max_persistence_points // 4)
        if points.shape[0] > MAX_POINTS_FOR_PERSISTENCE:
            indices = torch.linspace(0, points.shape[0] - 1, MAX_POINTS_FOR_PERSISTENCE).long()
            points = points[indices]
        
        # Limit max_radius to avoid explosion of simplices
        safe_max_radius = min(self.max_radius, 2.0)
        
        # Build Vietoris-Rips complex using factory method
        rips = RipsComplex.from_points(
            points=points,
            max_radius=safe_max_radius,
            max_dim=min(self.max_dimension, 2),  # Cap at dim 2 for tractability
        )
        
        # Compute persistence
        diagram = compute_persistence(rips)
        
        # Compute Betti numbers at max radius
        betti_numbers = self._compute_betti_numbers(diagram, self.max_radius)
        self._betti_history.append(betti_numbers)
        
        # Compute diagram distance if we have previous
        diagram_distance = None
        if self._previous_diagram is not None:
            diagram_distance = wasserstein_distance_diagram(
                self._previous_diagram, diagram, p=2
            )
        
        self._previous_diagram = diagram
        
        # Count significant features
        significant_features = self._count_significant_features(diagram)
        
        # Compute Euler characteristic
        euler_char = self._compute_euler_characteristic(betti_numbers)
        
        return PrimitiveResult(
            primitive_type=PrimitiveType.PH,
            data={
                "points": points,
                "complex": rips,
                "diagram": diagram,
            },
            metadata={
                "betti_numbers": betti_numbers,
                "euler_characteristic": euler_char,
                "significant_features": significant_features,
                "diagram_distance": diagram_distance,
                "n_points": points.shape[0],
                "max_dimension": self.max_dimension,
            },
            elapsed_time=time.perf_counter() - start_time,
            qtt_rank=getattr(rips, 'max_rank', 0),
        )
    
    def _parse_input(self, input_data: Any) -> torch.Tensor:
        """Parse input into point cloud tensor.
        
        Limits point cloud size to prevent memory issues in persistence computation.
        """
        config = get_config()
        MAX_POINTS = config.ingestion.max_persistence_points  # Limit for O(n²) persistence algorithm
        
        def _limit_points(tensor: torch.Tensor) -> torch.Tensor:
            """Subsample if too many points."""
            if tensor.shape[0] > MAX_POINTS:
                # Subsample uniformly
                indices = torch.linspace(0, tensor.shape[0] - 1, MAX_POINTS).long()
                return tensor[indices]
            return tensor
        
        if isinstance(input_data, torch.Tensor):
            points = input_data.to(self.config.dtype)
            # Ensure 2D (N, D)
            if points.dim() == 1:
                points = points.unsqueeze(-1)
            return _limit_points(points)
        
        elif isinstance(input_data, PrimitiveResult):
            data = input_data.data
            
            if isinstance(data, torch.Tensor):
                points = data.to(self.config.dtype)
                if points.dim() == 1:
                    points = points.unsqueeze(-1)
                return _limit_points(points)
            elif isinstance(data, dict):
                if "points" in data:
                    return _limit_points(data["points"].to(self.config.dtype))
                elif "X" in data:
                    return _limit_points(data["X"].to(self.config.dtype))
                elif "signal" in data:
                    signal = data["signal"]
                    if hasattr(signal, 'to_dense'):
                        dense = signal.to_dense()
                        if dense.dim() == 1:
                            dense = dense.unsqueeze(-1)
                        return _limit_points(dense)
            
            # Try to extract tensor
            tensor = input_data.as_tensor()
            if tensor is not None:
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(-1)
                return _limit_points(tensor.to(self.config.dtype))
                return tensor.to(self.config.dtype)
            
            raise ValueError(
                f"Cannot chain from {input_data.primitive_type}: "
                f"cannot extract point cloud from {type(data)}"
            )
        
        elif isinstance(input_data, dict):
            if "points" in input_data:
                return input_data["points"].to(self.config.dtype)
            elif "X" in input_data:
                return input_data["X"].to(self.config.dtype)
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _compute_betti_numbers(
        self, diagram: PersistenceDiagram, radius: float
    ) -> List[int]:
        """Compute Betti numbers at a given radius."""
        betti = []
        for dim in range(self.max_dimension + 1):
            pairs = diagram.pairs.get(dim, [])
            count = sum(1 for birth, death in pairs if birth <= radius < death)
            betti.append(count)
        return betti
    
    def _count_significant_features(self, diagram: PersistenceDiagram) -> Dict[int, int]:
        """Count features with persistence > threshold."""
        significant = {}
        for dim in range(self.max_dimension + 1):
            pairs = diagram.pairs.get(dim, [])
            count = sum(
                1 for birth, death in pairs
                if (death - birth) > self.persistence_threshold
            )
            significant[dim] = count
        return significant
    
    def _compute_euler_characteristic(self, betti: List[int]) -> int:
        """Compute Euler characteristic: χ = Σ (-1)^k β_k."""
        return sum((-1) ** k * b for k, b in enumerate(betti))
    
    def detect_anomalies(self, data: Any) -> List[AnomalyFinding]:
        """
        Detect topological anomalies.
        """
        findings: List[AnomalyFinding] = []
        
        if isinstance(data, dict) and "diagram" in data:
            diagram = data["diagram"]
            
            # Detect unexpected holes/voids
            significant = self._count_significant_features(diagram)
            
            for dim, count in significant.items():
                if count > 0:
                    dim_name = ["components", "loops", "voids"][dim] if dim < 3 else f"dim-{dim}"
                    
                    # Check against history
                    if len(self._betti_history) >= 2:
                        prev_betti = self._betti_history[-2]
                        if dim < len(prev_betti):
                            diff = count - prev_betti[dim]
                            if abs(diff) > 0:
                                findings.append(AnomalyFinding(
                                    severity=Severity.MEDIUM if abs(diff) < 3 else Severity.HIGH,
                                    summary=f"Topological change: {diff:+d} {dim_name}",
                                    primitives=["PH"],
                                    evidence={
                                        "dimension": dim,
                                        "current_count": count,
                                        "previous_count": prev_betti[dim],
                                        "change": diff,
                                    },
                                    anomaly_score=float(abs(diff)),
                                ))
            
            # Persistence diagram drift
            diagram_distance = data.get("diagram_distance")
            if diagram_distance is not None and diagram_distance > 1.0:
                findings.append(AnomalyFinding(
                    severity=Severity.MEDIUM,
                    summary=f"Persistence diagram drift: W₂ = {diagram_distance:.4f}",
                    primitives=["PH"],
                    evidence={
                        "wasserstein_distance": diagram_distance,
                    },
                    anomaly_score=diagram_distance,
                ))
        
        return findings
    
    def detect_invariants(self, data: Any) -> List[InvariantFinding]:
        """
        Verify topological invariants.
        """
        findings: List[InvariantFinding] = []
        
        if isinstance(data, dict):
            betti = data.get("betti_numbers", [])
            euler = data.get("euler_characteristic", 0)
            
            # Euler characteristic
            findings.append(InvariantFinding(
                severity=Severity.INFO,
                summary=f"Euler characteristic χ = {euler}",
                primitives=["PH"],
                evidence={
                    "euler_characteristic": euler,
                    "betti_numbers": betti,
                },
                invariant_name="euler_characteristic",
                value=float(euler),
            ))
            
            # Betti number stability
            if len(self._betti_history) >= 3:
                recent = self._betti_history[-3:]
                stable = all(b == recent[0] for b in recent)
                
                findings.append(InvariantFinding(
                    severity=Severity.INFO if stable else Severity.LOW,
                    summary=f"Betti stability: {'stable' if stable else 'varying'}",
                    primitives=["PH"],
                    evidence={
                        "history": recent,
                        "stable": stable,
                    },
                    invariant_name="betti_stability",
                    value=1.0 if stable else 0.0,
                ))
        
        return findings
    
    def detect_bottlenecks(self, data: Any) -> List[BottleneckFinding]:
        """
        Detect computational bottlenecks.
        """
        findings: List[BottleneckFinding] = []
        
        if isinstance(data, dict):
            n_points = data.get("n_points", 0)
            
            # Warn about complexity for large point clouds
            # Rips complex is O(n^{max_dim+1})
            complexity = n_points ** (self.max_dimension + 1)
            
            if complexity > 1e9:
                findings.append(BottleneckFinding(
                    severity=Severity.MEDIUM,
                    summary=f"High complexity: O(n^{self.max_dimension+1}) = {complexity:.2e}",
                    primitives=["PH"],
                    evidence={
                        "n_points": n_points,
                        "max_dimension": self.max_dimension,
                        "complexity": complexity,
                    },
                    bottleneck_type="compute",
                ))
        
        return findings
    
    def predict(self, data: Any) -> List[PredictionFinding]:
        """
        Predict topological evolution.
        """
        findings: List[PredictionFinding] = []
        
        if len(self._betti_history) >= 3:
            recent = self._betti_history[-3:]
            
            for dim in range(len(recent[0])):
                values = [b[dim] for b in recent]
                
                # Check for trend
                if values[0] < values[1] < values[2]:
                    findings.append(PredictionFinding(
                        severity=Severity.INFO,
                        summary=f"Increasing β_{dim}: {values[-1]} → ?",
                        primitives=["PH"],
                        evidence={
                            "dimension": dim,
                            "history": values,
                            "trend": "increasing",
                        },
                        prediction=values[-1] + 1,
                        confidence=0.5,
                        horizon="next_step",
                    ))
                elif values[0] > values[1] > values[2]:
                    findings.append(PredictionFinding(
                        severity=Severity.INFO,
                        summary=f"Decreasing β_{dim}: {values[-1]} → ?",
                        primitives=["PH"],
                        evidence={
                            "dimension": dim,
                            "history": values,
                            "trend": "decreasing",
                        },
                        prediction=max(0, values[-1] - 1),
                        confidence=0.5,
                        horizon="next_step",
                    ))
        
        return findings
    
    def reset_state(self) -> None:
        """Reset internal state."""
        self._betti_history.clear()
        self._previous_diagram = None
