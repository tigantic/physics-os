# Copyright (c) 2025 Tigantic
# Phase 18: Entanglement Analysis
"""
Entanglement analysis for tensor network states.

Provides comprehensive tools for analyzing entanglement properties including
von Neumann entropy, mutual information, Schmidt spectra, and area law scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from scipy import optimize


class ScalingType(Enum):
    """Type of entanglement scaling."""
    
    AREA_LAW = auto()      # S ~ L^(d-1) boundary scaling
    VOLUME_LAW = auto()    # S ~ L^d volume scaling
    LOG_CORRECTED = auto() # S ~ L^(d-1) log(L) CFT correction
    UNKNOWN = auto()


@dataclass
class EntanglementSpectrum:
    """Entanglement spectrum from Schmidt decomposition.
    
    Attributes:
        singular_values: Schmidt coefficients (singular values)
        probabilities: Schmidt probabilities (λ²)
        entropy: Von Neumann entropy
        renyi_2: Rényi-2 entropy
        bond_index: Index of the bipartition bond
        effective_rank: Effective rank (participation ratio)
    """
    
    singular_values: torch.Tensor
    probabilities: torch.Tensor
    entropy: float
    renyi_2: float
    bond_index: int
    effective_rank: float
    
    @classmethod
    def from_singular_values(
        cls,
        sv: torch.Tensor,
        bond_index: int = 0,
    ) -> "EntanglementSpectrum":
        """Create spectrum from singular values.
        
        Args:
            sv: Singular values from SVD
            bond_index: Index of the bipartition
            
        Returns:
            EntanglementSpectrum instance
        """
        # Normalize
        sv = sv.clone()
        norm = torch.sqrt(torch.sum(sv ** 2))
        if norm > 1e-15:
            sv = sv / norm
        
        # Compute probabilities
        probs = sv ** 2
        
        # Von Neumann entropy: S = -sum p log(p)
        valid_probs = probs[probs > 1e-15]
        if len(valid_probs) > 0:
            entropy = -float(torch.sum(valid_probs * torch.log(valid_probs)))
        else:
            entropy = 0.0
        
        # Rényi-2 entropy: S_2 = -log(sum p²)
        purity = float(torch.sum(probs ** 2))
        renyi_2 = -math.log(purity) if purity > 1e-15 else 0.0
        
        # Effective rank (participation ratio): 1/sum(p²)
        effective_rank = 1.0 / purity if purity > 1e-15 else len(probs)
        
        return cls(
            singular_values=sv,
            probabilities=probs,
            entropy=entropy,
            renyi_2=renyi_2,
            bond_index=bond_index,
            effective_rank=effective_rank,
        )
    
    def get_entanglement_gap(self) -> float:
        """Get the entanglement gap (ratio of first two Schmidt values).
        
        Returns:
            Entanglement gap λ₀/λ₁
        """
        if len(self.singular_values) < 2:
            return float('inf')
        
        sv = self.singular_values
        if sv[1] < 1e-15:
            return float('inf')
        
        return float(sv[0] / sv[1])
    
    def get_truncation_error(self, chi: int) -> float:
        """Compute truncation error for given bond dimension.
        
        Args:
            chi: Target bond dimension
            
        Returns:
            Truncation error (discarded weight)
        """
        if chi >= len(self.probabilities):
            return 0.0
        
        return float(torch.sum(self.probabilities[chi:]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entropy": self.entropy,
            "renyi_2": self.renyi_2,
            "effective_rank": self.effective_rank,
            "bond_index": self.bond_index,
            "spectrum_size": len(self.singular_values),
            "entanglement_gap": self.get_entanglement_gap(),
        }


@dataclass
class AreaLawScaling:
    """Result of area law scaling analysis.
    
    Attributes:
        scaling_type: Detected scaling type
        exponent: Fitted scaling exponent
        coefficient: Prefactor coefficient
        log_correction: Log correction coefficient (for CFT)
        r_squared: Fit quality (R²)
        residuals: Fit residuals
        boundary_sizes: Boundary sizes used in fit
        entropies: Entropies at each boundary size
    """
    
    scaling_type: ScalingType
    exponent: float
    coefficient: float
    log_correction: float
    r_squared: float
    residuals: List[float]
    boundary_sizes: List[int]
    entropies: List[float]
    
    def predict(self, boundary_size: int) -> float:
        """Predict entropy for a given boundary size.
        
        Args:
            boundary_size: Size of the boundary
            
        Returns:
            Predicted entanglement entropy
        """
        L = boundary_size
        
        if self.scaling_type == ScalingType.AREA_LAW:
            return self.coefficient * (L ** self.exponent)
        
        elif self.scaling_type == ScalingType.LOG_CORRECTED:
            base = self.coefficient * (L ** self.exponent)
            log_term = self.log_correction * math.log(L) if L > 0 else 0
            return base + log_term
        
        elif self.scaling_type == ScalingType.VOLUME_LAW:
            return self.coefficient * (L ** self.exponent)
        
        return self.coefficient
    
    def is_area_law(self, tolerance: float = 0.5) -> bool:
        """Check if scaling is consistent with area law.
        
        For 1D systems, area law means S ~ const (exponent ≈ 0).
        For 2D systems, area law means S ~ L (exponent ≈ 1).
        
        Args:
            tolerance: Tolerance for exponent comparison
            
        Returns:
            True if consistent with area law
        """
        # For 1D: area law exponent should be ~0 (constant)
        # For 2D: area law exponent should be ~1 (linear in boundary)
        return self.scaling_type == ScalingType.AREA_LAW and self.r_squared > 0.9


class AreaLawAnalyzer:
    """Analyze area law scaling of entanglement entropy.
    
    Fits entanglement entropy vs boundary size to detect whether
    the system obeys area law, volume law, or logarithmic corrections.
    """
    
    def __init__(self, dimension: int = 1) -> None:
        """Initialize analyzer.
        
        Args:
            dimension: Spatial dimension of the system
        """
        self.dimension = dimension
    
    def analyze(
        self,
        boundary_sizes: List[int],
        entropies: List[float],
        include_log_correction: bool = True,
    ) -> AreaLawScaling:
        """Analyze entropy scaling with boundary size.
        
        Args:
            boundary_sizes: List of boundary sizes
            entropies: Corresponding entanglement entropies
            include_log_correction: Whether to try log-corrected fit
            
        Returns:
            AreaLawScaling result
        """
        L = np.array(boundary_sizes, dtype=float)
        S = np.array(entropies, dtype=float)
        
        if len(L) < 2:
            return AreaLawScaling(
                scaling_type=ScalingType.UNKNOWN,
                exponent=0.0,
                coefficient=float(np.mean(S)) if len(S) > 0 else 0.0,
                log_correction=0.0,
                r_squared=0.0,
                residuals=[],
                boundary_sizes=list(boundary_sizes),
                entropies=list(entropies),
            )
        
        results = []
        
        # Try power law fit: S = a * L^b
        try:
            power_result = self._fit_power_law(L, S)
            results.append(("power", power_result))
        except Exception:
            pass
        
        # Try log-corrected fit: S = a * L^b + c * log(L)
        if include_log_correction:
            try:
                log_result = self._fit_log_corrected(L, S)
                results.append(("log", log_result))
            except Exception:
                pass
        
        if not results:
            return AreaLawScaling(
                scaling_type=ScalingType.UNKNOWN,
                exponent=0.0,
                coefficient=float(np.mean(S)),
                log_correction=0.0,
                r_squared=0.0,
                residuals=[],
                boundary_sizes=list(boundary_sizes),
                entropies=list(entropies),
            )
        
        # Select best fit by R²
        best_type, best_result = max(results, key=lambda x: x[1]["r_squared"])
        
        # Determine scaling type
        exponent = best_result["exponent"]
        if best_type == "log" and abs(best_result["log_correction"]) > 0.01:
            scaling_type = ScalingType.LOG_CORRECTED
        elif exponent < 0.5:
            scaling_type = ScalingType.AREA_LAW
        elif exponent > self.dimension - 0.5:
            scaling_type = ScalingType.VOLUME_LAW
        else:
            scaling_type = ScalingType.AREA_LAW
        
        return AreaLawScaling(
            scaling_type=scaling_type,
            exponent=exponent,
            coefficient=best_result["coefficient"],
            log_correction=best_result.get("log_correction", 0.0),
            r_squared=best_result["r_squared"],
            residuals=best_result["residuals"],
            boundary_sizes=list(boundary_sizes),
            entropies=list(entropies),
        )
    
    def _fit_power_law(
        self,
        L: np.ndarray,
        S: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit power law S = a * L^b.
        
        Args:
            L: Boundary sizes
            S: Entropies
            
        Returns:
            Fit parameters dictionary
        """
        # Use log-linear fit for power law
        valid = (L > 0) & (S > 0)
        if np.sum(valid) < 2:
            raise ValueError("Insufficient valid data points")
        
        log_L = np.log(L[valid])
        log_S = np.log(S[valid])
        
        # Linear regression on log data
        n = len(log_L)
        sum_x = np.sum(log_L)
        sum_y = np.sum(log_S)
        sum_xy = np.sum(log_L * log_S)
        sum_x2 = np.sum(log_L ** 2)
        
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-15:
            raise ValueError("Singular data")
        
        b = (n * sum_xy - sum_x * sum_y) / denom  # exponent
        log_a = (sum_y - b * sum_x) / n
        a = np.exp(log_a)  # coefficient
        
        # Compute R²
        S_pred = a * L ** b
        ss_res = np.sum((S - S_pred) ** 2)
        ss_tot = np.sum((S - np.mean(S)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        residuals = list(S - S_pred)
        
        return {
            "exponent": float(b),
            "coefficient": float(a),
            "r_squared": float(r_squared),
            "residuals": residuals,
        }
    
    def _fit_log_corrected(
        self,
        L: np.ndarray,
        S: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit log-corrected form S = a * L^b + c * log(L).
        
        Args:
            L: Boundary sizes
            S: Entropies
            
        Returns:
            Fit parameters dictionary
        """
        def model(x, a, b, c):
            return a * x ** b + c * np.log(x + 1)
        
        # Initial guess from power law
        try:
            power_result = self._fit_power_law(L, S)
            p0 = [power_result["coefficient"], power_result["exponent"], 0.0]
        except Exception:
            p0 = [1.0, 0.0, 0.1]
        
        # Fit using scipy
        try:
            popt, _ = optimize.curve_fit(
                model, L, S, p0=p0,
                bounds=([0, -2, -10], [1000, 3, 10]),
                maxfev=5000,
            )
            a, b, c = popt
        except Exception:
            # Fall back to power law
            power_result = self._fit_power_law(L, S)
            return {**power_result, "log_correction": 0.0}
        
        # Compute R²
        S_pred = model(L, a, b, c)
        ss_res = np.sum((S - S_pred) ** 2)
        ss_tot = np.sum((S - np.mean(S)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return {
            "exponent": float(b),
            "coefficient": float(a),
            "log_correction": float(c),
            "r_squared": float(r_squared),
            "residuals": list(S - S_pred),
        }


@dataclass
class EntanglementEntropy:
    """Entanglement entropy calculator and container.
    
    Attributes:
        value: Entropy value
        bond_index: Bond index (bipartition point)
        subsystem_size: Size of left subsystem
        method: Method used for calculation
    """
    
    value: float
    bond_index: int
    subsystem_size: int
    method: str = "von_neumann"
    
    @classmethod
    def from_reduced_density_matrix(
        cls,
        rho: torch.Tensor,
        bond_index: int = 0,
    ) -> "EntanglementEntropy":
        """Compute entropy from reduced density matrix.
        
        S = -Tr(ρ log ρ)
        
        Args:
            rho: Reduced density matrix
            bond_index: Bond index
            
        Returns:
            EntanglementEntropy instance
        """
        # Eigendecompose
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues.real
        
        # Filter positive eigenvalues
        valid = eigenvalues > 1e-15
        probs = eigenvalues[valid]
        
        # Entropy
        if len(probs) == 0:
            entropy = 0.0
        else:
            entropy = -float(torch.sum(probs * torch.log(probs)))
        
        return cls(
            value=entropy,
            bond_index=bond_index,
            subsystem_size=rho.shape[0],
            method="von_neumann",
        )
    
    @classmethod
    def from_singular_values(
        cls,
        sv: torch.Tensor,
        bond_index: int = 0,
        subsystem_size: int = 0,
    ) -> "EntanglementEntropy":
        """Compute entropy from Schmidt singular values.
        
        Args:
            sv: Singular values
            bond_index: Bond index
            subsystem_size: Size of left subsystem
            
        Returns:
            EntanglementEntropy instance
        """
        spectrum = EntanglementSpectrum.from_singular_values(sv, bond_index)
        
        return cls(
            value=spectrum.entropy,
            bond_index=bond_index,
            subsystem_size=subsystem_size,
            method="schmidt",
        )


@dataclass
class MutualInformation:
    """Mutual information between two subsystems.
    
    I(A:B) = S(A) + S(B) - S(AB)
    
    Attributes:
        value: Mutual information value
        entropy_a: Entropy of subsystem A
        entropy_b: Entropy of subsystem B
        entropy_ab: Entropy of combined system AB
        region_a: Indices of region A
        region_b: Indices of region B
    """
    
    value: float
    entropy_a: float
    entropy_b: float
    entropy_ab: float
    region_a: Tuple[int, int]
    region_b: Tuple[int, int]
    
    @classmethod
    def compute(
        cls,
        entropies: List[float],
        region_a: Tuple[int, int],
        region_b: Tuple[int, int],
        total_entropy: float = 0.0,
    ) -> "MutualInformation":
        """Compute mutual information from entropy profile.
        
        For non-overlapping regions A and B:
        I(A:B) = S(A) + S(B) - S(A∪B)
        
        Args:
            entropies: Entropy at each bond
            region_a: Start and end of region A
            region_b: Start and end of region B
            total_entropy: Total system entropy (for pure states = 0)
            
        Returns:
            MutualInformation instance
        """
        # For an MPS, entropy at bond i gives S(1...i | i+1...L)
        # S(A) for region [a1, a2] is approximately S at the cut a1-1
        
        a1, a2 = region_a
        b1, b2 = region_b
        
        # Approximate entropies from bond entropy profile
        if a1 > 0 and a1 - 1 < len(entropies):
            s_a = entropies[a1 - 1]
        else:
            s_a = 0.0
        
        if b1 > 0 and b1 - 1 < len(entropies):
            s_b = entropies[b1 - 1]
        else:
            s_b = 0.0
        
        # For non-overlapping regions, S(AB) ≈ S at the boundary
        # This is a simplification; exact calculation requires contractions
        s_ab = max(s_a, s_b)  # Upper bound approximation
        
        mi = s_a + s_b - s_ab
        mi = max(0.0, mi)  # Ensure non-negative
        
        return cls(
            value=mi,
            entropy_a=s_a,
            entropy_b=s_b,
            entropy_ab=s_ab,
            region_a=region_a,
            region_b=region_b,
        )


# Convenience functions

def compute_entanglement_entropy(
    singular_values: torch.Tensor,
    alpha: float = 1.0,
) -> float:
    """Compute Rényi-α entanglement entropy from singular values.
    
    S_α = (1/(1-α)) log(Tr(ρ^α))
    
    For α → 1, this gives von Neumann entropy: S = -Tr(ρ log ρ)
    
    Args:
        singular_values: Schmidt singular values
        alpha: Rényi parameter (default 1 = von Neumann)
        
    Returns:
        Rényi-α entropy
    """
    sv = singular_values.clone()
    
    # Normalize
    norm = torch.sqrt(torch.sum(sv ** 2))
    if norm < 1e-15:
        return 0.0
    sv = sv / norm
    
    probs = sv ** 2
    valid = probs > 1e-15
    probs = probs[valid]
    
    if len(probs) == 0:
        return 0.0
    
    if abs(alpha - 1.0) < 1e-10:
        # Von Neumann entropy
        return -float(torch.sum(probs * torch.log(probs)))
    else:
        # Rényi-α entropy
        return float(torch.log(torch.sum(probs ** alpha))) / (1 - alpha)


def compute_mutual_information(
    entropies_profile: List[float],
    region_a: Tuple[int, int],
    region_b: Tuple[int, int],
) -> float:
    """Compute mutual information between two regions.
    
    Args:
        entropies_profile: Entropy at each bond
        region_a: (start, end) indices of region A
        region_b: (start, end) indices of region B
        
    Returns:
        Mutual information I(A:B)
    """
    mi = MutualInformation.compute(entropies_profile, region_a, region_b)
    return mi.value


def analyze_area_law(
    system_sizes: List[int],
    entropies: List[float],
    dimension: int = 1,
) -> AreaLawScaling:
    """Analyze area law scaling.
    
    Args:
        system_sizes: List of system/boundary sizes
        entropies: Corresponding entropies
        dimension: Spatial dimension
        
    Returns:
        AreaLawScaling result
    """
    analyzer = AreaLawAnalyzer(dimension=dimension)
    return analyzer.analyze(system_sizes, entropies)


def compute_schmidt_spectrum(
    tensor: torch.Tensor,
    bond_index: int = 0,
    normalize: bool = True,
) -> EntanglementSpectrum:
    """Compute Schmidt spectrum from a tensor.
    
    Reshapes the tensor into a matrix at the specified bond and
    computes the singular values.
    
    Args:
        tensor: Input tensor
        bond_index: Bond at which to bipartition
        normalize: Whether to normalize singular values
        
    Returns:
        EntanglementSpectrum at the specified bond
    """
    # Reshape tensor to matrix at bond_index
    shape = tensor.shape
    left_dim = 1
    for i in range(bond_index + 1):
        left_dim *= shape[i]
    right_dim = tensor.numel() // left_dim
    
    matrix = tensor.reshape(left_dim, right_dim)
    
    # SVD
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    
    if normalize:
        norm = torch.sqrt(torch.sum(S ** 2))
        if norm > 1e-15:
            S = S / norm
    
    return EntanglementSpectrum.from_singular_values(S, bond_index)
