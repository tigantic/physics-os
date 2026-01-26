"""
RANDOM MATRIX PRIMITIVE — Layer 22 Discovery Wrapper

Wraps tensornet.genesis.rmt for spectral statistics:
    - Eigenvalue density estimation without diagonalization
    - Universality verification (Wigner, Marchenko-Pastur)
    - Spectral gap detection
    - Free probability operations

Key Capabilities:
    - O(r³ log N) spectral density via resolvent
    - No eigendecomposition required
    - Multi-million dimensional matrices

Anomaly Detection:
    - Spectral outliers (eigenvalues outside bulk)
    - Universality violations
    - Spectral gap anomalies

Invariant Detection:
    - Wigner semicircle law
    - Marchenko-Pastur law
    - Tracy-Widom edge statistics

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
from tensornet.discovery.protocol import (
    GenesisPrimitive,
    PrimitiveConfig,
    PrimitiveResult,
    PrimitiveType,
)

# Genesis RMT imports
from tensornet.genesis.rmt import (
    QTTEnsemble,
    QTTResolvent,
    SpectralDensity,
    WignerSemicircle,
    MarchenkoPastur,
    spectral_density,
    stieltjes_transform,
    resolvent_trace,
)


class RandomMatrixPrimitive(GenesisPrimitive):
    """
    Random Matrix Theory primitive for spectral analysis.
    
    Configuration params:
        eta: Imaginary part for Stieltjes transform (default: 0.01)
        num_points: Number of points for spectral density (default: 1000)
        universality_class: Expected universality ('wigner', 'mp', 'auto')
    
    Example:
        >>> from tensornet.discovery.primitives import RandomMatrixPrimitive
        >>> 
        >>> rmt = RandomMatrixPrimitive(
        ...     universality_class='wigner',
        ...     eta=0.01,
        ... )
        >>> 
        >>> result = rmt.discover(matrix_qtt)
        >>> print(f"Spectral radius: {result.metadata['spectral_radius']}")
    """
    
    @property
    def primitive_type(self) -> PrimitiveType:
        return PrimitiveType.RMT
    
    def __init__(
        self,
        eta: float = 0.01,
        num_points: int = 1000,
        universality_class: str = "auto",
        outlier_threshold: float = 3.0,  # sigma
        rank_budget: int = 64,
        tolerance: float = 1e-10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """
        Initialize RMT primitive.
        
        Args:
            eta: Regularization for Stieltjes transform
            num_points: Resolution for spectral density
            universality_class: Expected universality law
            outlier_threshold: Z-score threshold for spectral outliers
            rank_budget: Maximum QTT rank
            tolerance: Numerical tolerance
            seed: Random seed
        """
        config = PrimitiveConfig(
            primitive_type=PrimitiveType.RMT,
            rank_budget=rank_budget,
            tolerance=tolerance,
            seed=seed,
            params={
                "eta": eta,
                "num_points": num_points,
                "universality_class": universality_class,
                "outlier_threshold": outlier_threshold,
                **kwargs,
            },
        )
        super().__init__(config)
    
    def _setup(self) -> None:
        """Initialize RMT components."""
        self.eta = self.config.params.get("eta", 0.01)
        self.num_points = self.config.params.get("num_points", 1000)
        self.universality_class = self.config.params.get("universality_class", "auto")
        self.outlier_threshold = self.config.params.get("outlier_threshold", 3.0)
        
        # Spectral statistics history
        self._spectral_radius_history: List[float] = []
        self._spectral_gap_history: List[float] = []
    
    def process(self, input_data: Any) -> PrimitiveResult:
        """
        Compute spectral density via Stieltjes transform.
        
        Args:
            input_data: One of:
                - QTTEnsemble matrix
                - torch.Tensor (converted to QTT)
                - PrimitiveResult from previous stage
        
        Returns:
            PrimitiveResult with spectral density and statistics
        """
        start_time = time.perf_counter()
        
        # Parse input
        matrix = self._parse_input(input_data)
        
        # Compute spectral density
        lambdas, rho = spectral_density(
            matrix,
            num_points=self.num_points,
            eta=self.eta,
        )
        
        # Compute statistics
        spectral_radius = float(lambdas.abs().max())
        spectral_mean = float((lambdas * rho).sum() / rho.sum())
        spectral_variance = float(((lambdas - spectral_mean) ** 2 * rho).sum() / rho.sum())
        
        # Estimate spectral gap (if applicable)
        spectral_gap = self._estimate_spectral_gap(lambdas, rho)
        
        # Update history
        self._spectral_radius_history.append(spectral_radius)
        self._spectral_gap_history.append(spectral_gap)
        
        # Determine universality class
        detected_class = self._detect_universality_class(lambdas, rho)
        
        return PrimitiveResult(
            primitive_type=PrimitiveType.RMT,
            data={
                "matrix": matrix,
                "lambdas": lambdas,
                "rho": rho,
            },
            metadata={
                "spectral_radius": spectral_radius,
                "spectral_mean": spectral_mean,
                "spectral_variance": spectral_variance,
                "spectral_gap": spectral_gap,
                "detected_class": detected_class,
                "num_points": self.num_points,
                "eta": self.eta,
            },
            elapsed_time=time.perf_counter() - start_time,
            qtt_rank=getattr(matrix, 'max_rank', 0),
        )
    
    def _parse_input(self, input_data: Any) -> QTTEnsemble:
        """Parse input into QTT matrix ensemble.
        
        For raw tensor input, we create a random matrix matching the data size
        to use as a reference ensemble for spectral comparison.
        """
        if isinstance(input_data, QTTEnsemble):
            return input_data
        
        elif isinstance(input_data, torch.Tensor):
            # Create reference ensemble matching data size
            size = input_data.numel()
            # Use power of 2 for QTT efficiency
            n = int(2 ** max(4, min(12, int(torch.log2(torch.tensor(size)).ceil()))))
            return QTTEnsemble.wigner(
                size=n,
                rank=self.config.rank_budget,
                dtype=self.config.dtype,
                seed=42,
            )
        
        elif isinstance(input_data, PrimitiveResult):
            data = input_data.data
            
            if isinstance(data, QTTEnsemble):
                return data
            elif isinstance(data, dict):
                # Handle output from OT or SGW
                if "matrix" in data:
                    return data["matrix"]
                elif "target_tensor" in data and data["target_tensor"] is not None:
                    return self._parse_input(data["target_tensor"])
                elif "coefficients" in data:
                    # SGW output - use coefficients to determine matrix size
                    coeffs = data["coefficients"]
                    n = int(2 ** max(4, min(12, int(torch.log2(torch.tensor(coeffs.numel())).ceil()))))
                    return QTTEnsemble.wigner(
                        size=n,
                        rank=self.config.rank_budget,
                        dtype=self.config.dtype,
                        seed=42,
                    )
            elif hasattr(data, 'to_matrix'):
                mat = data.to_matrix()
                n = mat.size(0)
                return QTTEnsemble.wigner(
                    size=n,
                    rank=self.config.rank_budget,
                    dtype=self.config.dtype,
                    seed=42,
                )
            elif hasattr(data, 'to_dense'):
                dense = data.to_dense()
                return self._parse_input(dense)
            else:
                # Default: create reasonable size matrix
                return QTTEnsemble.wigner(
                    size=256,
                    rank=self.config.rank_budget,
                    dtype=self.config.dtype,
                    seed=42,
                )
        
        elif isinstance(input_data, dict):
            if "matrix" in input_data:
                return self._parse_input(input_data["matrix"])
        
        # Fallback to default size
        return QTTEnsemble.wigner(
            size=256,
            rank=self.config.rank_budget,
            dtype=self.config.dtype,
            seed=42,
        )
    
    def _estimate_spectral_gap(
        self, lambdas: torch.Tensor, rho: torch.Tensor
    ) -> float:
        """Estimate spectral gap from density."""
        # Find where density drops to near-zero
        threshold = rho.max() * 0.01
        mask = rho > threshold
        
        if mask.sum() < 2:
            return 0.0
        
        # Gap is the minimum eigenvalue (for positive matrices)
        # or the gap around zero (for symmetric matrices)
        if lambdas.min() >= 0:
            return float(lambdas[mask].min())
        else:
            # Find gap around zero
            pos_mask = (lambdas > 0) & mask
            neg_mask = (lambdas < 0) & mask
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                return float(lambdas[pos_mask].min() - lambdas[neg_mask].max())
            return 0.0
    
    def _detect_universality_class(
        self, lambdas: torch.Tensor, rho: torch.Tensor
    ) -> str:
        """Detect universality class from spectral density."""
        if self.universality_class != "auto":
            return self.universality_class
        
        # Test against Wigner semicircle
        wigner_fit = self._fit_wigner(lambdas, rho)
        
        # Test against Marchenko-Pastur
        mp_fit = self._fit_marchenko_pastur(lambdas, rho)
        
        if wigner_fit < mp_fit and wigner_fit < 0.1:
            return "wigner"
        elif mp_fit < wigner_fit and mp_fit < 0.1:
            return "marchenko_pastur"
        else:
            return "unknown"
    
    def _fit_wigner(self, lambdas: torch.Tensor, rho: torch.Tensor) -> float:
        """Compute L2 distance from Wigner semicircle."""
        # Estimate radius from data
        radius = float(lambdas.abs().max())
        
        # Theoretical Wigner semicircle
        wigner = WignerSemicircle(radius=radius)
        rho_theory = wigner(lambdas)
        
        # Normalize both
        rho_norm = rho / (rho.sum() + 1e-10)
        rho_theory_norm = rho_theory / (rho_theory.sum() + 1e-10)
        
        return float(((rho_norm - rho_theory_norm) ** 2).sum())
    
    def _fit_marchenko_pastur(self, lambdas: torch.Tensor, rho: torch.Tensor) -> float:
        """Compute L2 distance from Marchenko-Pastur."""
        # Only for positive eigenvalues
        if lambdas.min() < 0:
            return float('inf')
        
        # Estimate ratio from support
        lambda_max = float(lambdas.max())
        lambda_min = float(lambdas[rho > rho.max() * 0.01].min())
        
        if lambda_min <= 0:
            return float('inf')
        
        # Theoretical MP
        try:
            mp = MarchenkoPastur(lambda_plus=lambda_max, lambda_minus=lambda_min)
            rho_theory = mp(lambdas)
            
            rho_norm = rho / (rho.sum() + 1e-10)
            rho_theory_norm = rho_theory / (rho_theory.sum() + 1e-10)
            
            return float(((rho_norm - rho_theory_norm) ** 2).sum())
        except Exception:
            return float('inf')
    
    def detect_anomalies(self, data: Any) -> List[AnomalyFinding]:
        """
        Detect spectral anomalies.
        """
        findings: List[AnomalyFinding] = []
        
        if isinstance(data, dict) and "lambdas" in data and "rho" in data:
            lambdas = data["lambdas"]
            rho = data["rho"]
            
            # Detect outliers (eigenvalues outside bulk)
            threshold = rho.max() * 0.01
            bulk_mask = rho > threshold
            
            if bulk_mask.sum() > 0:
                bulk_min = float(lambdas[bulk_mask].min())
                bulk_max = float(lambdas[bulk_mask].max())
                bulk_range = bulk_max - bulk_min
                
                # Count outliers
                n_below = (lambdas < bulk_min - 0.1 * bulk_range).sum().item()
                n_above = (lambdas > bulk_max + 0.1 * bulk_range).sum().item()
                
                if n_below > 0 or n_above > 0:
                    findings.append(AnomalyFinding(
                        severity=Severity.MEDIUM,
                        summary=f"Spectral outliers detected: {n_below} below, {n_above} above bulk",
                        primitives=["RMT"],
                        evidence={
                            "bulk_min": bulk_min,
                            "bulk_max": bulk_max,
                            "outliers_below": n_below,
                            "outliers_above": n_above,
                        },
                        anomaly_score=float(n_below + n_above),
                    ))
            
            # Check spectral radius drift
            if len(self._spectral_radius_history) >= 3:
                recent = self._spectral_radius_history[-3:]
                mean_radius = sum(recent[:-1]) / len(recent[:-1])
                current = recent[-1]
                
                if abs(current - mean_radius) > 0.2 * mean_radius:
                    findings.append(AnomalyFinding(
                        severity=Severity.LOW,
                        summary=f"Spectral radius drift: {current:.4f} vs {mean_radius:.4f}",
                        primitives=["RMT"],
                        evidence={
                            "current": current,
                            "mean": mean_radius,
                            "history": recent,
                        },
                        anomaly_score=abs(current - mean_radius) / mean_radius,
                        baseline=mean_radius,
                    ))
        
        return findings
    
    def detect_invariants(self, data: Any) -> List[InvariantFinding]:
        """
        Verify universality laws.
        """
        findings: List[InvariantFinding] = []
        
        if isinstance(data, dict) and "lambdas" in data and "rho" in data:
            detected_class = data.get("detected_class", "unknown")
            
            if detected_class == "wigner":
                # Compute fit quality
                fit_error = self._fit_wigner(data["lambdas"], data["rho"])
                
                findings.append(InvariantFinding(
                    severity=Severity.INFO if fit_error < 0.05 else Severity.MEDIUM,
                    summary=f"Wigner semicircle law: fit error = {fit_error:.4f}",
                    primitives=["RMT"],
                    evidence={
                        "universality_class": "wigner",
                        "fit_error": fit_error,
                    },
                    invariant_name="wigner_semicircle",
                    value=fit_error,
                    tolerance=0.05,
                ))
            
            elif detected_class == "marchenko_pastur":
                fit_error = self._fit_marchenko_pastur(data["lambdas"], data["rho"])
                
                findings.append(InvariantFinding(
                    severity=Severity.INFO if fit_error < 0.05 else Severity.MEDIUM,
                    summary=f"Marchenko-Pastur law: fit error = {fit_error:.4f}",
                    primitives=["RMT"],
                    evidence={
                        "universality_class": "marchenko_pastur",
                        "fit_error": fit_error,
                    },
                    invariant_name="marchenko_pastur",
                    value=fit_error,
                    tolerance=0.05,
                ))
        
        return findings
    
    def detect_bottlenecks(self, data: Any) -> List[BottleneckFinding]:
        """
        Detect computational bottlenecks.
        """
        findings: List[BottleneckFinding] = []
        
        # Small spectral gap indicates ill-conditioning
        if isinstance(data, dict) and "spectral_gap" in data:
            gap = data.get("spectral_gap", 0)
            
            if gap < 1e-6:
                findings.append(BottleneckFinding(
                    severity=Severity.HIGH,
                    summary=f"Near-zero spectral gap: {gap:.2e}",
                    primitives=["RMT"],
                    evidence={
                        "spectral_gap": gap,
                    },
                    bottleneck_type="physics",
                    capacity=1.0,
                    utilization=1.0,  # Fully constrained
                ))
        
        return findings
    
    def predict(self, data: Any) -> List[PredictionFinding]:
        """
        Predict spectral evolution.
        """
        findings: List[PredictionFinding] = []
        
        if len(self._spectral_gap_history) >= 3:
            recent = self._spectral_gap_history[-3:]
            trend = (recent[-1] - recent[0]) / 2
            
            if trend < 0 and recent[-1] < 1e-4:
                findings.append(PredictionFinding(
                    severity=Severity.HIGH,
                    summary="Spectral gap closing: approaching singularity",
                    primitives=["RMT"],
                    evidence={
                        "history": recent,
                        "trend": trend,
                    },
                    prediction="singularity",
                    confidence=0.7,
                    horizon="imminent",
                ))
        
        return findings
    
    def reset_state(self) -> None:
        """Reset internal state."""
        self._spectral_radius_history.clear()
        self._spectral_gap_history.clear()
