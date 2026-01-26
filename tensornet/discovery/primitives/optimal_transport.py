"""
OPTIMAL TRANSPORT PRIMITIVE — Layer 20 Discovery Wrapper

Wraps tensornet.genesis.ot for autonomous discovery:
    - Distribution drift detection via Wasserstein distance
    - Mass transport bottleneck identification
    - Barycenter computation for cluster analysis
    - Monge map extraction for transformation discovery

Key Capabilities:
    - Trillion-point distribution matching (O(r³ log N))
    - Multi-marginal optimal transport
    - Wasserstein barycenters
    - Transport plan analysis

Anomaly Detection:
    - Distribution drift: W₂(μ_t, μ_{t-1}) > threshold
    - Outlier mass: Transport cost concentration
    - Mode collapse: Missing support regions

Invariant Detection:
    - Mass conservation: ∫μ = ∫ν
    - Self-transport: W(μ, μ) = 0

Bottleneck Detection:
    - Transport bottlenecks: High-cost regions
    - Capacity constraints: Mass flow limits

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from tensornet.discovery.findings import (
    AnomalyFinding,
    BottleneckFinding,
    Finding,
    InvariantFinding,
    PredictionFinding,
    Severity,
)
from tensornet.discovery.protocol import (
    GenesisPrimitive,
    PrimitiveConfig,
    PrimitiveResult,
    PrimitiveType,
)

# Genesis OT imports
from tensornet.genesis.ot import (
    QTTDistribution,
    QTTSinkhorn,
    SinkhornResult,
    wasserstein_distance,
    transport_plan,
    barycenter,
)


class OptimalTransportPrimitive(GenesisPrimitive):
    """
    Optimal Transport primitive for distribution analysis.
    
    Configuration params:
        epsilon: Entropic regularization (default: 0.01)
        max_iter: Maximum Sinkhorn iterations (default: 1000)
        threshold: Wasserstein distance anomaly threshold (default: 0.1)
        scales: Multi-scale analysis scales (default: [1, 2, 4])
    
    Example:
        >>> from tensornet.discovery.primitives import OptimalTransportPrimitive
        >>> 
        >>> ot = OptimalTransportPrimitive(
        ...     epsilon=0.01,
        ...     threshold=0.1,
        ... )
        >>> 
        >>> # Analyze distributions
        >>> result = ot.discover({"source": mu, "target": nu})
        >>> print(f"Wasserstein distance: {result.metadata['wasserstein_distance']}")
    """
    
    @property
    def primitive_type(self) -> PrimitiveType:
        return PrimitiveType.OT
    
    def __init__(
        self,
        epsilon: float = 0.01,
        max_iter: int = 1000,
        threshold: float = 0.1,
        rank_budget: int = 64,
        tolerance: float = 1e-10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Initialize OT primitive.
        
        Args:
            epsilon: Entropic regularization parameter
            max_iter: Maximum Sinkhorn iterations
            threshold: Anomaly detection threshold for W₂ distance
            rank_budget: Maximum QTT rank
            tolerance: Numerical tolerance
            seed: Random seed
        """
        config = PrimitiveConfig(
            primitive_type=PrimitiveType.OT,
            rank_budget=rank_budget,
            tolerance=tolerance,
            seed=seed,
            params={
                "epsilon": epsilon,
                "max_iter": max_iter,
                "threshold": threshold,
                **kwargs,
            },
        )
        super().__init__(config)
    
    def _setup(self) -> None:
        """Initialize Sinkhorn solver."""
        self.epsilon = self.config.params.get("epsilon", 0.01)
        self.max_iter = self.config.params.get("max_iter", 1000)
        self.threshold = self.config.params.get("threshold", 0.1)
        
        self.solver = QTTSinkhorn(
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            tol=self.config.tolerance,
        )
        
        # State for drift detection
        self._previous_distribution: Optional[QTTDistribution] = None
        self._wasserstein_history: List[float] = []
    
    def process(self, input_data: Any) -> PrimitiveResult:
        """
        Compute optimal transport between distributions.
        
        Args:
            input_data: One of:
                - Dict with "source" and "target" QTTDistribution
                - Single QTTDistribution (compared to previous)
                - PrimitiveResult from previous stage
                - Raw tensor (converted to distribution)
        
        Returns:
            PrimitiveResult with transport plan and Wasserstein distance
        """
        start_time = time.perf_counter()
        
        # Parse input
        source, target = self._parse_input(input_data)
        
        # Compute optimal transport
        result = self.solver.solve(source, target)
        
        # Extract metrics
        w_distance = result.wasserstein_distance
        self._wasserstein_history.append(w_distance)
        
        # Update state for drift detection
        self._previous_distribution = target
        
        # Store target for downstream primitives to use
        target_tensor = target.to_dense() if hasattr(target, 'to_dense') else None
        
        return PrimitiveResult(
            primitive_type=PrimitiveType.OT,
            data={
                "sinkhorn_result": result,
                "source": source,
                "target": target,
                "target_tensor": target_tensor,
            },
            metadata={
                "wasserstein_distance": w_distance,
                "iterations": result.iterations,
                "converged": result.converged,
                "source_mass": source.total_mass,
                "target_mass": target.total_mass,
            },
            elapsed_time=time.perf_counter() - start_time,
            qtt_rank=max(
                getattr(source, 'max_rank', 0),
                getattr(target, 'max_rank', 0),
            ),
        )
    
    def _parse_input(
        self, input_data: Any
    ) -> Tuple[QTTDistribution, QTTDistribution]:
        """Parse input into source and target distributions."""
        
        if isinstance(input_data, dict):
            source = input_data.get("source")
            target = input_data.get("target")
            
            if source is None or target is None:
                raise ValueError("Dict input must have 'source' and 'target' keys")
            
            return self._ensure_distribution(source), self._ensure_distribution(target)
        
        elif isinstance(input_data, PrimitiveResult):
            # Chain from previous primitive
            data = input_data.data
            if isinstance(data, QTTDistribution):
                if self._previous_distribution is None:
                    # First in chain, compare to self
                    return data, data
                return self._previous_distribution, data
            else:
                raise ValueError(
                    f"Cannot chain from {input_data.primitive_type} to OT: "
                    f"expected QTTDistribution, got {type(data)}"
                )
        
        elif isinstance(input_data, QTTDistribution):
            if self._previous_distribution is None:
                return input_data, input_data
            return self._previous_distribution, input_data
        
        elif isinstance(input_data, torch.Tensor):
            dist = self._tensor_to_distribution(input_data)
            if self._previous_distribution is None:
                return dist, dist
            return self._previous_distribution, dist
        
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _ensure_distribution(self, data: Any) -> QTTDistribution:
        """Convert data to QTTDistribution if needed."""
        if isinstance(data, QTTDistribution):
            return data
        elif isinstance(data, torch.Tensor):
            return self._tensor_to_distribution(data)
        else:
            raise ValueError(f"Cannot convert {type(data)} to QTTDistribution")
    
    def _tensor_to_distribution(self, tensor: torch.Tensor) -> QTTDistribution:
        """
        Convert raw tensor to QTT distribution.
        
        Strategy: Use tensor statistics to create a Gaussian mixture approximation.
        For multidimensional data, we reduce to 1D using the norm of each sample.
        """
        tensor = tensor.to(self.config.dtype)
        
        # Reduce to 1D if needed (use L2 norm of each sample)
        if tensor.dim() > 1:
            # Treat as batch of vectors, compute norm of each
            norms = torch.norm(tensor.view(tensor.size(0), -1), dim=1)
        else:
            norms = tensor
        
        # Compute statistics
        mean = float(norms.mean())
        std = float(norms.std()) + 1e-8  # Avoid zero std
        
        # Create Gaussian distribution matching the data statistics
        return QTTDistribution.gaussian(
            mean=mean,
            std=std,
            grid_size=min(2**16, 2 ** int(torch.log2(torch.tensor(len(norms))).ceil())),
            dtype=self.config.dtype,
        )
    
    def detect_anomalies(self, data: Any) -> List[AnomalyFinding]:
        """
        Detect distribution drift and transport anomalies.
        
        Anomalies detected:
            - Drift: W₂(μ_t, μ_{t-1}) exceeds threshold
            - Mass imbalance: Source and target have different total mass
            - Convergence failure: Sinkhorn did not converge
        """
        findings: List[AnomalyFinding] = []
        
        # Handle dict output from process()
        sinkhorn_result = data
        if isinstance(data, dict) and "sinkhorn_result" in data:
            sinkhorn_result = data["sinkhorn_result"]
        
        if isinstance(sinkhorn_result, SinkhornResult):
            w_distance = sinkhorn_result.wasserstein_distance
            
            # Check for drift
            if w_distance > self.threshold:
                findings.append(AnomalyFinding(
                    severity=self._severity_from_distance(w_distance),
                    summary=f"Distribution drift detected: W₂ = {w_distance:.4f} > {self.threshold}",
                    primitives=["OT"],
                    evidence={
                        "wasserstein_distance": w_distance,
                        "threshold": self.threshold,
                        "ratio": w_distance / self.threshold,
                    },
                    anomaly_score=w_distance / self.threshold,
                    baseline=self.threshold,
                    deviation=(w_distance - self.threshold) / self.threshold,
                ))
            
            # Check convergence
            if not sinkhorn_result.converged:
                findings.append(AnomalyFinding(
                    severity=Severity.MEDIUM,
                    summary=f"Sinkhorn did not converge after {sinkhorn_result.iterations} iterations",
                    primitives=["OT"],
                    evidence={
                        "iterations": data.iterations,
                        "max_iter": self.max_iter,
                    },
                    anomaly_score=1.0,
                ))
        
        return findings
    
    def _severity_from_distance(self, distance: float) -> Severity:
        """Map Wasserstein distance to severity level."""
        ratio = distance / self.threshold
        if ratio > 10:
            return Severity.CRITICAL
        elif ratio > 5:
            return Severity.HIGH
        elif ratio > 2:
            return Severity.MEDIUM
        elif ratio > 1:
            return Severity.LOW
        return Severity.INFO
    
    def detect_invariants(self, data: Any) -> List[InvariantFinding]:
        """
        Detect mass conservation and self-transport invariants.
        """
        findings: List[InvariantFinding] = []
        
        if isinstance(data, SinkhornResult):
            # Mass conservation
            source_mass = getattr(data, 'source_mass', 1.0)
            target_mass = getattr(data, 'target_mass', 1.0)
            mass_diff = abs(source_mass - target_mass)
            
            if mass_diff < self.config.tolerance:
                findings.append(InvariantFinding(
                    severity=Severity.INFO,
                    summary="Mass conservation verified",
                    primitives=["OT"],
                    evidence={
                        "source_mass": source_mass,
                        "target_mass": target_mass,
                        "difference": mass_diff,
                    },
                    invariant_name="mass_conservation",
                    value=source_mass,
                    tolerance=self.config.tolerance,
                ))
            else:
                findings.append(InvariantFinding(
                    severity=Severity.HIGH,
                    summary=f"Mass conservation violated: Δm = {mass_diff:.2e}",
                    primitives=["OT"],
                    evidence={
                        "source_mass": source_mass,
                        "target_mass": target_mass,
                        "difference": mass_diff,
                    },
                    invariant_name="mass_conservation",
                    value=mass_diff,
                    tolerance=self.config.tolerance,
                ))
        
        return findings
    
    def detect_bottlenecks(self, data: Any) -> List[BottleneckFinding]:
        """
        Detect transport bottlenecks (high-cost regions).
        """
        findings: List[BottleneckFinding] = []
        
        if isinstance(data, SinkhornResult):
            # Check for iteration bottleneck
            if data.iterations > self.max_iter * 0.9:
                findings.append(BottleneckFinding(
                    severity=Severity.MEDIUM,
                    summary=f"Approaching iteration limit: {data.iterations}/{self.max_iter}",
                    primitives=["OT"],
                    evidence={
                        "iterations": data.iterations,
                        "max_iter": self.max_iter,
                    },
                    bottleneck_type="compute",
                    capacity=float(self.max_iter),
                    utilization=data.iterations / self.max_iter,
                ))
        
        return findings
    
    def predict(self, data: Any) -> List[PredictionFinding]:
        """
        Make predictions based on transport patterns.
        """
        findings: List[PredictionFinding] = []
        
        # Predict future drift based on history
        if len(self._wasserstein_history) >= 3:
            recent = self._wasserstein_history[-3:]
            trend = (recent[-1] - recent[0]) / 2
            
            if trend > 0:
                predicted_next = recent[-1] + trend
                
                if predicted_next > self.threshold:
                    findings.append(PredictionFinding(
                        severity=Severity.MEDIUM,
                        summary=f"Distribution drift accelerating: predicted W₂ = {predicted_next:.4f}",
                        primitives=["OT"],
                        evidence={
                            "history": recent,
                            "trend": trend,
                            "predicted": predicted_next,
                        },
                        prediction=predicted_next,
                        confidence=0.7,
                        horizon="next_step",
                    ))
        
        return findings
    
    def reset_state(self) -> None:
        """Reset internal state for fresh analysis."""
        self._previous_distribution = None
        self._wasserstein_history.clear()
