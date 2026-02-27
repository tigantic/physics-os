"""
SPECTRAL WAVELET PRIMITIVE — Layer 21 Discovery Wrapper

Wraps tensornet.genesis.sgw for multi-scale signal analysis:
    - Graph Laplacian construction in QTT format
    - Spectral wavelet transforms
    - Multi-scale anomaly detection
    - Frequency-domain feature extraction

Key Capabilities:
    - Trillion-node graph analysis (O(r³ K log N))
    - Multi-scale wavelet decomposition
    - Chebyshev polynomial approximation
    - Graph signal filtering

Anomaly Detection:
    - Scale-specific energy anomalies
    - Spectral outliers (unusual frequency content)
    - Localized defects (wavelet coefficient spikes)

Invariant Detection:
    - Total energy conservation
    - Parseval's identity
    - Graph symmetries

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

# Genesis SGW imports
from tensornet.genesis.sgw import (
    QTTLaplacian,
    QTTSignal,
    QTTGraphWavelet,
    ChebyshevApproximator,
    LowPassFilter,
    HighPassFilter,
    BandPassFilter,
)


class SpectralWaveletPrimitive(GenesisPrimitive):
    """
    Spectral Graph Wavelet primitive for multi-scale analysis.
    
    Configuration params:
        scales: Wavelet scales for multi-resolution analysis
        chebyshev_order: Order of Chebyshev approximation (default: 50)
        graph_type: Type of underlying graph ('grid_1d', 'grid_2d', 'custom')
        energy_threshold: Threshold for energy anomaly detection
    
    Example:
        >>> from tensornet.ml.discovery.primitives import SpectralWaveletPrimitive
        >>> 
        >>> sgw = SpectralWaveletPrimitive(
        ...     scales=[1, 2, 4, 8, 16],
        ...     graph_type='grid_1d',
        ... )
        >>> 
        >>> result = sgw.discover(signal_data)
        >>> for scale, energy in result.metadata['scale_energies'].items():
        ...     print(f"Scale {scale}: {energy:.4f}")
    """
    
    @property
    def primitive_type(self) -> PrimitiveType:
        return PrimitiveType.SGW
    
    def __init__(
        self,
        scales: Optional[List[float]] = None,
        chebyshev_order: int = 50,
        graph_type: str = "grid_1d",
        energy_threshold: float = 3.0,  # sigma
        rank_budget: int = 64,
        tolerance: float = 1e-10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Initialize SGW primitive.
        
        Args:
            scales: Wavelet scales (default: [1, 2, 4, 8])
            chebyshev_order: Polynomial order for spectral approximation
            graph_type: Underlying graph structure
            energy_threshold: Z-score threshold for energy anomalies
            rank_budget: Maximum QTT rank
            tolerance: Numerical tolerance
            seed: Random seed
        """
        if scales is None:
            scales = [1.0, 2.0, 4.0, 8.0]
        
        config = PrimitiveConfig(
            primitive_type=PrimitiveType.SGW,
            rank_budget=rank_budget,
            tolerance=tolerance,
            seed=seed,
            params={
                "scales": scales,
                "chebyshev_order": chebyshev_order,
                "graph_type": graph_type,
                "energy_threshold": energy_threshold,
                **kwargs,
            },
        )
        super().__init__(config)
    
    def _setup(self) -> None:
        """Initialize wavelet transform components."""
        self.scales = self.config.params.get("scales", [1.0, 2.0, 4.0, 8.0])
        self.chebyshev_order = self.config.params.get("chebyshev_order", 50)
        self.graph_type = self.config.params.get("graph_type", "grid_1d")
        self.energy_threshold = self.config.params.get("energy_threshold", 3.0)
        
        # Laplacian will be created lazily based on input size
        self._laplacian: Optional[QTTLaplacian] = None
        self._wavelet: Optional[QTTGraphWavelet] = None
        
        # History for temporal analysis
        self._energy_history: Dict[float, List[float]] = {s: [] for s in self.scales}
    
    def _get_or_create_laplacian(self, n_nodes: int) -> QTTLaplacian:
        """Create Laplacian if not exists or size changed."""
        if self._laplacian is None or self._laplacian.num_nodes != n_nodes:
            if self.graph_type == "grid_1d":
                self._laplacian = QTTLaplacian.grid_1d(n_nodes)
            elif self.graph_type == "grid_2d":
                # Assume square grid
                side = int(n_nodes ** 0.5)
                self._laplacian = QTTLaplacian.grid_2d(side, side)
            else:
                raise ValueError(f"Unknown graph type: {self.graph_type}")
            
            # Create wavelet transform
            self._wavelet = QTTGraphWavelet(
                laplacian=self._laplacian,
                scales=self.scales,
                kernel_type="meyer",
                chebyshev_order=self.chebyshev_order,
            )
        
        return self._laplacian
    
    def process(self, input_data: Any) -> PrimitiveResult:
        """
        Compute multi-scale wavelet transform.
        
        Args:
            input_data: One of:
                - QTTSignal
                - torch.Tensor (1D signal)
                - PrimitiveResult from previous stage
        
        Returns:
            PrimitiveResult with wavelet coefficients at all scales
        """
        start_time = time.perf_counter()
        
        # Parse input
        signal = self._parse_input(input_data)
        
        # Ensure Laplacian exists - must be power of 2
        n_qubits = signal.num_qubits
        n_nodes = 2 ** n_qubits  # QTTSignal.num_qubits = log2(size)
        self._get_or_create_laplacian(n_nodes)
        
        # Compute wavelet transform
        wavelet_result = self._wavelet.transform(signal)
        
        # Extract energies from WaveletResult (energy_per_scale is a method)
        energies = wavelet_result.energy_per_scale()
        scale_energies = {}
        for i, scale in enumerate(self.scales):
            energy = energies[i] if i < len(energies) else 0.0
            scale_energies[scale] = float(energy)
            self._energy_history[scale].append(float(energy))
        
        # Total energy
        total_energy = float(wavelet_result.total_energy()) if hasattr(wavelet_result, 'total_energy') else sum(scale_energies.values())
        
        return PrimitiveResult(
            primitive_type=PrimitiveType.SGW,
            data={
                "signal": signal,
                "wavelet_result": wavelet_result,
                "laplacian": self._laplacian,
            },
            metadata={
                "scale_energies": scale_energies,
                "total_energy": total_energy,
                "num_qubits": signal.num_qubits,
                "scales": self.scales,
                "chebyshev_order": self.chebyshev_order,
            },
            elapsed_time=time.perf_counter() - start_time,
            qtt_rank=signal.max_rank,
        )
    
    def _parse_input(self, input_data: Any) -> QTTSignal:
        """Parse input into QTTSignal."""
        if isinstance(input_data, QTTSignal):
            return input_data
        
        elif isinstance(input_data, torch.Tensor):
            return QTTSignal.from_dense(
                input_data.flatten().to(self.config.dtype),
                max_rank=self.config.rank_budget,
            )
        
        elif isinstance(input_data, PrimitiveResult):
            # Try to extract signal from previous stage
            data = input_data.data
            
            if isinstance(data, QTTSignal):
                return data
            elif isinstance(data, dict):
                # Handle OT output: extract target tensor
                if "target_tensor" in data and data["target_tensor"] is not None:
                    return QTTSignal.from_dense(
                        data["target_tensor"].flatten().to(self.config.dtype),
                        max_rank=self.config.rank_budget,
                    )
                elif "signal" in data:
                    return data["signal"]
                elif "tensor" in data:
                    return QTTSignal.from_dense(
                        data["tensor"].flatten().to(self.config.dtype),
                        max_rank=self.config.rank_budget,
                    )
            elif hasattr(data, 'to_signal'):
                return data.to_signal()
            
            # Fallback: try to convert to tensor
            tensor = input_data.as_tensor()
            if tensor is not None:
                return QTTSignal.from_dense(tensor.flatten(), max_rank=self.config.rank_budget)
            raise ValueError(
                f"Cannot chain from {input_data.primitive_type}: "
                f"cannot extract signal from {type(data)}"
            )
        
        elif isinstance(input_data, dict):
            if "signal" in input_data:
                return self._parse_input(input_data["signal"])
            elif "tensor" in input_data:
                return self._parse_input(input_data["tensor"])
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def detect_anomalies(self, data: Any) -> List[AnomalyFinding]:
        """
        Detect scale-specific energy anomalies.
        """
        findings: List[AnomalyFinding] = []
        
        if isinstance(data, dict) and "coefficients" in data:
            coefficients = data["coefficients"]
            
            for i, scale in enumerate(self.scales):
                coef = coefficients[i]
                energy = float(coef.norm() ** 2)
                
                # Check against history for this scale
                history = self._energy_history.get(scale, [])
                if len(history) >= 3:
                    mean_energy = sum(history[:-1]) / len(history[:-1])
                    std_energy = (sum((e - mean_energy) ** 2 for e in history[:-1]) / len(history[:-1])) ** 0.5
                    
                    if std_energy > 0:
                        z_score = (energy - mean_energy) / std_energy
                        
                        if abs(z_score) > self.energy_threshold:
                            findings.append(AnomalyFinding(
                                severity=self._severity_from_zscore(z_score),
                                summary=f"Energy anomaly at scale {scale}: z = {z_score:.2f}",
                                primitives=["SGW"],
                                evidence={
                                    "scale": scale,
                                    "energy": energy,
                                    "mean_energy": mean_energy,
                                    "std_energy": std_energy,
                                    "z_score": z_score,
                                },
                                anomaly_score=abs(z_score),
                                baseline=mean_energy,
                                deviation=z_score,
                            ))
        
        return findings
    
    def _severity_from_zscore(self, z: float) -> Severity:
        """Map z-score to severity."""
        z = abs(z)
        if z > 6:
            return Severity.CRITICAL
        elif z > 4:
            return Severity.HIGH
        elif z > 3:
            return Severity.MEDIUM
        elif z > 2:
            return Severity.LOW
        return Severity.INFO
    
    def detect_invariants(self, data: Any) -> List[InvariantFinding]:
        """
        Verify Parseval's identity (energy conservation).
        """
        findings: List[InvariantFinding] = []
        
        if isinstance(data, dict):
            signal = data.get("signal")
            coefficients = data.get("coefficients")
            
            if signal is not None and coefficients is not None:
                # Signal energy
                signal_energy = float(signal.norm() ** 2)
                
                # Total wavelet energy
                wavelet_energy = sum(float(c.norm() ** 2) for c in coefficients)
                
                # Parseval's identity: energies should match (approximately)
                # Note: This is approximate for wavelets, exact for orthonormal bases
                energy_ratio = wavelet_energy / (signal_energy + 1e-10)
                
                findings.append(InvariantFinding(
                    severity=Severity.INFO,
                    summary=f"Energy ratio (wavelet/signal): {energy_ratio:.4f}",
                    primitives=["SGW"],
                    evidence={
                        "signal_energy": signal_energy,
                        "wavelet_energy": wavelet_energy,
                        "ratio": energy_ratio,
                    },
                    invariant_name="parseval_ratio",
                    value=energy_ratio,
                    tolerance=0.1,  # Allow 10% deviation for non-orthonormal wavelets
                ))
        
        return findings
    
    def detect_bottlenecks(self, data: Any) -> List[BottleneckFinding]:
        """
        Detect computational bottlenecks in wavelet transform.
        """
        findings: List[BottleneckFinding] = []
        
        # Check rank budget utilization
        if isinstance(data, dict) and "coefficients" in data:
            max_rank = max(getattr(c, 'max_rank', 0) for c in data["coefficients"])
            utilization = max_rank / self.config.rank_budget
            
            if utilization > 0.9:
                findings.append(BottleneckFinding(
                    severity=Severity.MEDIUM,
                    summary=f"QTT rank approaching budget: {max_rank}/{self.config.rank_budget}",
                    primitives=["SGW"],
                    evidence={
                        "max_rank": max_rank,
                        "rank_budget": self.config.rank_budget,
                    },
                    bottleneck_type="memory",
                    capacity=float(self.config.rank_budget),
                    utilization=utilization,
                ))
        
        return findings
    
    def predict(self, data: Any) -> List[PredictionFinding]:
        """
        Predict future energy patterns.
        """
        findings: List[PredictionFinding] = []
        
        # Predict energy trends for each scale
        for scale in self.scales:
            history = self._energy_history.get(scale, [])
            
            if len(history) >= 5:
                # Simple linear trend
                recent = history[-5:]
                trend = (recent[-1] - recent[0]) / 4
                
                if abs(trend) > 0.1 * recent[-1]:  # Significant trend
                    direction = "increasing" if trend > 0 else "decreasing"
                    predicted = recent[-1] + trend
                    
                    findings.append(PredictionFinding(
                        severity=Severity.INFO,
                        summary=f"Scale {scale} energy {direction}: trend = {trend:.4f}",
                        primitives=["SGW"],
                        evidence={
                            "scale": scale,
                            "history": recent,
                            "trend": trend,
                            "predicted_next": predicted,
                        },
                        prediction=predicted,
                        confidence=0.6,
                        horizon="next_step",
                    ))
        
        return findings
    
    def reset_state(self) -> None:
        """Reset internal state for fresh analysis."""
        self._energy_history = {s: [] for s in self.scales}
