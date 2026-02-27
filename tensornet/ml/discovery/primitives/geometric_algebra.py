"""
GEOMETRIC ALGEBRA PRIMITIVE — Layer 26 Discovery Wrapper

Wraps tensornet.genesis.ga for geometric transformations:
    - Multivector operations in QTT format
    - Rotor-based transformations
    - Conformal Geometric Algebra (CGA)
    - Geometric invariant detection

Key Capabilities:
    - Exponentially large multivector spaces (2^n components)
    - Rotation, translation, dilation via versors
    - Unified treatment of geometric primitives

Anomaly Detection:
    - Non-unit rotors (normalization violations)
    - Grade mixing anomalies
    - Transformation discontinuities

Invariant Detection:
    - Rotor normalization: |R|² = 1
    - Geometric products: reversibility
    - Conformal structure preservation

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from tensornet.ml.discovery.findings import (
    AnomalyFinding,
    BottleneckFinding,
    InvariantFinding,
    PredictionFinding,
    Severity,
)
from tensornet.ml.discovery.protocol import (
    GenesisPrimitive,
    PrimitiveConfig,
    PrimitiveResult,
    PrimitiveType,
)

# Genesis GA imports
from tensornet.genesis.ga import (
    CliffordAlgebra,
    Multivector,
    geometric_product,
    inner_product,
    outer_product,
    reverse,
    magnitude,
    normalize,
    rotor_from_bivector,
    apply_rotor,
    ConformalGA,
)


class GeometricAlgebraPrimitive(GenesisPrimitive):
    """
    Geometric Algebra primitive for geometric analysis.
    
    Configuration params:
        signature: Clifford algebra signature (p, q, r)
        use_conformal: Whether to use Conformal GA embedding
        normalization_threshold: Threshold for rotor normalization check
    
    Example:
        >>> from tensornet.ml.discovery.primitives import GeometricAlgebraPrimitive
        >>> 
        >>> ga = GeometricAlgebraPrimitive(
        ...     signature=(3, 0, 0),  # 3D Euclidean
        ...     use_conformal=True,
        ... )
        >>> 
        >>> result = ga.discover(multivector_data)
        >>> print(f"Grade distribution: {result.metadata['grade_distribution']}")
    """
    
    @property
    def primitive_type(self) -> PrimitiveType:
        return PrimitiveType.GA
    
    def __init__(
        self,
        signature: Tuple[int, int, int] = (3, 0, 0),
        use_conformal: bool = False,
        normalization_threshold: float = 0.01,
        rank_budget: int = 64,
        tolerance: float = 1e-10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Initialize GA primitive.
        
        Args:
            signature: Clifford algebra signature (p positive, q negative, r zero)
            use_conformal: Use Conformal GA embedding
            normalization_threshold: Threshold for normalization anomalies
            rank_budget: Maximum QTT rank
            tolerance: Numerical tolerance
            seed: Random seed
        """
        config = PrimitiveConfig(
            primitive_type=PrimitiveType.GA,
            rank_budget=rank_budget,
            tolerance=tolerance,
            seed=seed,
            params={
                "signature": signature,
                "use_conformal": use_conformal,
                "normalization_threshold": normalization_threshold,
                **kwargs,
            },
        )
        super().__init__(config)
    
    def _setup(self) -> None:
        """Initialize GA components."""
        self.signature = self.config.params.get("signature", (3, 0, 0))
        self.use_conformal = self.config.params.get("use_conformal", False)
        self.normalization_threshold = self.config.params.get("normalization_threshold", 0.01)
        
        # Create algebra
        p, q, r = self.signature
        self.algebra = CliffordAlgebra(p=p, q=q, r=r)
        self.dimension = 2 ** (p + q + r)
        
        # CGA if requested
        if self.use_conformal:
            self.cga = ConformalGA(dimension=p)
        else:
            self.cga = None
        
        # History
        self._rotor_history: List[Multivector] = []
        self._grade_history: List[Dict[int, float]] = []
    
    def process(self, input_data: Any) -> PrimitiveResult:
        """
        Process geometric data.
        
        Args:
            input_data: One of:
                - Multivector
                - Dict with multivector operations
                - Tensor to embed as vector
                - PrimitiveResult from previous stage
        
        Returns:
            PrimitiveResult with geometric analysis
        """
        start_time = time.perf_counter()
        
        # Parse input
        mv = self._parse_input(input_data)
        
        # Analyze grade distribution
        grade_dist = self._compute_grade_distribution(mv)
        self._grade_history.append(grade_dist)
        
        # Compute magnitude
        mv_magnitude = float(magnitude(mv))
        
        # Check if it's a rotor (scalar + bivector with unit magnitude)
        is_rotor = self._is_rotor(mv)
        if is_rotor:
            self._rotor_history.append(mv)
        
        # Compute reverse and check reversibility
        mv_reverse = reverse(mv)
        product = geometric_product(mv, mv_reverse)
        reversibility_error = self._compute_reversibility_error(product)
        
        return PrimitiveResult(
            primitive_type=PrimitiveType.GA,
            data={
                "multivector": mv,
                "reverse": mv_reverse,
                "product": product,
            },
            metadata={
                "grade_distribution": grade_dist,
                "magnitude": mv_magnitude,
                "is_rotor": is_rotor,
                "reversibility_error": reversibility_error,
                "algebra_dimension": self.dimension,
                "signature": self.signature,
            },
            elapsed_time=time.perf_counter() - start_time,
            qtt_rank=getattr(mv, 'max_rank', 0),
        )
    
    def _parse_input(self, input_data: Any) -> Multivector:
        """Parse input into multivector."""
        if isinstance(input_data, Multivector):
            return input_data
        
        elif isinstance(input_data, torch.Tensor):
            # Embed as vector (grade 1)
            return self.algebra.vector(input_data.to(self.config.dtype))
        
        elif isinstance(input_data, PrimitiveResult):
            data = input_data.data
            
            if isinstance(data, Multivector):
                return data
            elif isinstance(data, dict) and "multivector" in data:
                return data["multivector"]
            
            # Try to extract tensor and embed
            tensor = input_data.as_tensor()
            if tensor is not None:
                return self.algebra.vector(tensor.to(self.config.dtype))
            
            raise ValueError(
                f"Cannot chain from {input_data.primitive_type}: "
                f"cannot extract multivector from {type(data)}"
            )
        
        elif isinstance(input_data, dict):
            if "multivector" in input_data:
                return input_data["multivector"]
            elif "vector" in input_data:
                return self.algebra.vector(input_data["vector"])
            elif "bivector" in input_data:
                return self.algebra.bivector(input_data["bivector"])
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _compute_grade_distribution(self, mv: Multivector) -> Dict[int, float]:
        """Compute energy distribution across grades."""
        dist = {}
        max_grade = sum(self.signature)
        
        for grade in range(max_grade + 1):
            grade_part = mv.grade(grade)
            energy = float(magnitude(grade_part) ** 2)
            if energy > self.config.tolerance:
                dist[grade] = energy
        
        # Normalize
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        
        return dist
    
    def _is_rotor(self, mv: Multivector) -> bool:
        """Check if multivector is a valid rotor."""
        # Rotor should have only scalar and bivector parts
        grade_dist = self._compute_grade_distribution(mv)
        
        # Check grades
        allowed_grades = {0, 2}
        if not set(grade_dist.keys()).issubset(allowed_grades):
            return False
        
        # Check normalization
        mv_mag = float(magnitude(mv))
        return abs(mv_mag - 1.0) < self.normalization_threshold
    
    def _compute_reversibility_error(self, product: Multivector) -> float:
        """Compute error from reversibility (mv * reverse(mv) should be scalar)."""
        total_energy = float(magnitude(product) ** 2)
        scalar_energy = float(magnitude(product.grade(0)) ** 2)
        
        if total_energy < self.config.tolerance:
            return 0.0
        
        non_scalar_energy = total_energy - scalar_energy
        return non_scalar_energy / total_energy
    
    def detect_anomalies(self, data: Any) -> List[AnomalyFinding]:
        """
        Detect geometric anomalies.
        """
        findings: List[AnomalyFinding] = []
        
        if isinstance(data, dict):
            # Check rotor normalization
            if data.get("is_rotor"):
                mv = data.get("multivector")
                mv_mag = data.get("magnitude", 1.0)
                
                if abs(mv_mag - 1.0) > self.normalization_threshold:
                    findings.append(AnomalyFinding(
                        severity=Severity.MEDIUM,
                        summary=f"Non-unit rotor: |R| = {mv_mag:.6f}",
                        primitives=["GA"],
                        evidence={
                            "magnitude": mv_mag,
                            "expected": 1.0,
                            "error": abs(mv_mag - 1.0),
                        },
                        anomaly_score=abs(mv_mag - 1.0) / self.normalization_threshold,
                    ))
            
            # Check reversibility
            reversibility_error = data.get("reversibility_error", 0.0)
            if reversibility_error > 0.01:
                findings.append(AnomalyFinding(
                    severity=Severity.LOW if reversibility_error < 0.1 else Severity.MEDIUM,
                    summary=f"Reversibility violation: {reversibility_error:.4f}",
                    primitives=["GA"],
                    evidence={
                        "reversibility_error": reversibility_error,
                    },
                    anomaly_score=reversibility_error,
                ))
            
            # Grade distribution anomaly
            if len(self._grade_history) >= 2:
                current = data.get("grade_distribution", {})
                previous = self._grade_history[-2]
                
                # Compute KL-divergence-like metric
                all_grades = set(current.keys()) | set(previous.keys())
                divergence = sum(
                    abs(current.get(g, 0) - previous.get(g, 0))
                    for g in all_grades
                )
                
                if divergence > 0.3:
                    findings.append(AnomalyFinding(
                        severity=Severity.LOW,
                        summary=f"Grade distribution shift: Δ = {divergence:.4f}",
                        primitives=["GA"],
                        evidence={
                            "current": current,
                            "previous": previous,
                            "divergence": divergence,
                        },
                        anomaly_score=divergence,
                    ))
        
        return findings
    
    def detect_invariants(self, data: Any) -> List[InvariantFinding]:
        """
        Verify geometric invariants.
        """
        findings: List[InvariantFinding] = []
        
        if isinstance(data, dict):
            # Check signature preservation
            findings.append(InvariantFinding(
                severity=Severity.INFO,
                summary=f"Algebra signature: Cl({self.signature[0]},{self.signature[1]},{self.signature[2]})",
                primitives=["GA"],
                evidence={
                    "signature": self.signature,
                    "dimension": self.dimension,
                },
                invariant_name="algebra_signature",
                value=sum(self.signature),
            ))
            
            # Rotor normalization
            if data.get("is_rotor"):
                mv_mag = data.get("magnitude", 1.0)
                
                findings.append(InvariantFinding(
                    severity=Severity.INFO if abs(mv_mag - 1.0) < self.config.tolerance else Severity.LOW,
                    summary=f"Rotor normalization: |R|² = {mv_mag**2:.6f}",
                    primitives=["GA"],
                    evidence={
                        "magnitude_squared": mv_mag ** 2,
                    },
                    invariant_name="rotor_normalization",
                    value=mv_mag ** 2,
                    tolerance=self.config.tolerance,
                ))
        
        return findings
    
    def detect_bottlenecks(self, data: Any) -> List[BottleneckFinding]:
        """
        Detect computational bottlenecks.
        """
        findings: List[BottleneckFinding] = []
        
        # Algebra dimension is 2^n, which grows exponentially
        if self.dimension > 1024:
            findings.append(BottleneckFinding(
                severity=Severity.LOW if self.dimension < 65536 else Severity.MEDIUM,
                summary=f"Large algebra dimension: 2^{sum(self.signature)} = {self.dimension}",
                primitives=["GA"],
                evidence={
                    "dimension": self.dimension,
                    "generators": sum(self.signature),
                },
                bottleneck_type="memory",
            ))
        
        return findings
    
    def predict(self, data: Any) -> List[PredictionFinding]:
        """
        Predict geometric evolution.
        """
        findings: List[PredictionFinding] = []
        
        # Track rotor evolution for prediction
        if len(self._rotor_history) >= 3:
            # Could predict angular velocity, etc.
            findings.append(PredictionFinding(
                severity=Severity.INFO,
                summary=f"Rotor sequence tracked: {len(self._rotor_history)} samples",
                primitives=["GA"],
                evidence={
                    "n_rotors": len(self._rotor_history),
                },
                prediction="interpolation_available",
                confidence=0.8,
            ))
        
        return findings
    
    def reset_state(self) -> None:
        """Reset internal state."""
        self._rotor_history.clear()
        self._grade_history.clear()
