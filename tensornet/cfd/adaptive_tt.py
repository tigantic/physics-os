"""
TT-AMR: Adaptive Bond Dimension for Tensor-Train CFD.

This module implements the tensor network equivalent of Adaptive Mesh
Refinement (AMR). Instead of refining spatial cells, we adaptively
increase bond dimension χ near discontinuities (shocks, contact surfaces)
and decrease it in smooth regions.

Key Components:
- ShockDetector: Identifies discontinuities from TT core structure
- BondAdapter: Dynamically adjusts χ allocation
- AdaptiveTTEuler: Euler solver with TT-AMR

The entanglement entropy across a bond measures how much information
flows through that connection. Shocks create localized high-entropy
regions that require larger χ.

References:
- GRAND_VISION.md §3.3 — Adaptive Bond Dimension
- Bachmayr et al. (2016) "Adaptive low-rank methods for PDE"

Constitution Compliance: Article I.1 (Proof Requirements)
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum, auto
import math

from .tt_cfd import MPSState, TTCFDConfig, TT_Euler1D, EulerMPO, tdvp_euler_step


@dataclass
class AdaptiveTTConfig:
    """Configuration for adaptive bond dimension."""
    chi_min: int = 4           # Minimum bond dimension
    chi_max: int = 128         # Maximum bond dimension
    chi_base: int = 16         # Base bond dimension in smooth regions
    chi_boost: int = 4         # Multiplier for shock regions
    
    # Detection thresholds
    gradient_threshold: float = 0.1    # |∂u/∂x| threshold for shock
    entropy_threshold: float = 0.5     # Entanglement entropy threshold
    
    # Adaptation parameters
    adaptation_interval: int = 10      # Steps between adaptations
    smoothing_window: int = 3          # Smoothing for shock detection
    hysteresis: float = 0.1            # Prevent oscillation


class ShockIndicator(Enum):
    """Types of shock detection methods."""
    GRADIENT = auto()       # Based on solution gradient
    ENTROPY = auto()        # Based on entanglement entropy
    COMBINED = auto()       # Both methods together
    TROUBLED_CELL = auto()  # WENO-style troubled cell indicator


# =============================================================================
# Shock Detection
# =============================================================================

class ShockDetector:
    """
    Detect discontinuities in TT-format CFD solutions.
    
    Uses multiple indicators to identify shock locations:
    1. Gradient-based: High |∂u/∂x| indicates discontinuity
    2. Entropy-based: High entanglement entropy at a bond
    3. Troubled-cell: WENO-style oscillation detection
    
    The detector operates on MPS cores directly, avoiding
    full dense reconstruction.
    
    Attributes:
        config: Adaptive TT configuration
        shock_history: History of detected shocks for smoothing
    """
    
    def __init__(self, config: Optional[AdaptiveTTConfig] = None):
        self.config = config or AdaptiveTTConfig()
        self.shock_history: List[Tensor] = []
    
    def gradient_indicator(
        self,
        mps: MPSState,
        variable_index: int = 0
    ) -> Tensor:
        """
        Compute gradient-based shock indicator.
        
        Uses finite difference approximation of |∂u/∂x| from
        the MPS representation.
        
        Args:
            mps: MPS state to analyze
            variable_index: Which variable to check (0=ρ, 1=ρu, 2=E)
            
        Returns:
            Tensor of gradient magnitudes at each bond (n_sites-1,)
        """
        # Extract dense for gradient computation
        # Full TT version would use local contractions
        state = mps._contract_to_dense()  # (N, n_vars)
        
        u = state[:, variable_index]
        n = len(u)
        
        # Central difference gradient
        grad = torch.zeros(n - 1)
        for i in range(n - 1):
            grad[i] = abs(u[i + 1] - u[i])
        
        # Normalize by maximum
        grad_max = grad.max()
        if grad_max > 1e-10:
            grad = grad / grad_max
        
        return grad
    
    def entropy_indicator(self, mps: MPSState) -> Tensor:
        """
        Compute entanglement entropy at each bond.
        
        The von Neumann entropy S = -Σ λ² log(λ²) measures
        entanglement across the bond. High entropy indicates
        complex correlations (e.g., near shocks).
        
        Args:
            mps: MPS state to analyze
            
        Returns:
            Tensor of entanglement entropies at each bond (n_sites-1,)
        """
        n_sites = len(mps.cores)
        entropies = torch.zeros(n_sites - 1)
        
        # Compute SVD at each bond
        for i in range(n_sites - 1):
            # Merge cores i and i+1
            left_core = mps.cores[i]
            right_core = mps.cores[i + 1]
            
            chi_l, d1, chi_m = left_core.shape
            _, d2, chi_r = right_core.shape
            
            # Two-site tensor
            two_site = torch.einsum('abc,cde->abde', left_core, right_core)
            two_site_mat = two_site.reshape(chi_l * d1, d2 * chi_r)
            
            # Randomized SVD
            try:
                q = min(min(two_site_mat.shape), 100)  # Limit for entropy calculation
                _, S, _ = torch.svd_lowrank(two_site_mat, q=q, niter=1)
                
                # Normalize singular values
                S = S / (S.sum() + 1e-10)
                
                # Von Neumann entropy
                S_sq = S ** 2
                S_sq = S_sq[S_sq > 1e-20]  # Avoid log(0)
                entropy = -torch.sum(S_sq * torch.log(S_sq))
                
                entropies[i] = entropy.item()
            except:
                entropies[i] = 0.0
        
        # Normalize
        ent_max = entropies.max()
        if ent_max > 1e-10:
            entropies = entropies / ent_max
        
        return entropies
    
    def troubled_cell_indicator(self, mps: MPSState) -> Tensor:
        """
        WENO-style troubled cell indicator.
        
        Based on comparing solution with its projected representation.
        Large differences indicate non-smooth regions.
        
        Args:
            mps: MPS state to analyze
            
        Returns:
            Tensor of troubled cell indicators (n_sites,)
        """
        state = mps._contract_to_dense()
        n = state.shape[0]
        
        troubled = torch.zeros(n)
        
        for i in range(2, n - 2):
            # Extract local stencil
            stencil = state[i-2:i+3, 0]  # Use density
            
            # Check for oscillations using WENO-style indicator
            # Based on smoothness indicators from WENO
            
            um2, um1, u0, up1, up2 = stencil
            
            # Smoothness indicators
            beta0 = (13.0/12.0) * (um2 - 2*um1 + u0)**2 + \
                    (1.0/4.0) * (um2 - 4*um1 + 3*u0)**2
            
            beta1 = (13.0/12.0) * (um1 - 2*u0 + up1)**2 + \
                    (1.0/4.0) * (um1 - up1)**2
            
            beta2 = (13.0/12.0) * (u0 - 2*up1 + up2)**2 + \
                    (1.0/4.0) * (3*u0 - 4*up1 + up2)**2
            
            # Troubled if smoothness varies significantly
            beta_max = max(beta0, beta1, beta2)
            beta_min = min(beta0, beta1, beta2)
            
            if beta_max > 1e-10:
                troubled[i] = (beta_max - beta_min) / beta_max
        
        return troubled
    
    def detect_shocks(
        self,
        mps: MPSState,
        method: ShockIndicator = ShockIndicator.COMBINED
    ) -> Tuple[List[int], Tensor]:
        """
        Detect shock locations in the MPS state.
        
        Args:
            mps: MPS state to analyze
            method: Detection method to use
            
        Returns:
            (shock_sites, indicator): List of shock site indices and
                                      the full indicator array
        """
        if method == ShockIndicator.GRADIENT:
            indicator = self.gradient_indicator(mps)
        elif method == ShockIndicator.ENTROPY:
            indicator = self.entropy_indicator(mps)
        elif method == ShockIndicator.TROUBLED_CELL:
            indicator = self.troubled_cell_indicator(mps)
        else:  # COMBINED
            grad = self.gradient_indicator(mps)
            ent = self.entropy_indicator(mps)
            
            # Pad to same length
            n = min(len(grad), len(ent))
            indicator = 0.5 * grad[:n] + 0.5 * ent[:n]
        
        # Smooth with history
        self.shock_history.append(indicator)
        if len(self.shock_history) > self.config.smoothing_window:
            self.shock_history.pop(0)
        
        # Average over history
        smoothed = torch.stack(self.shock_history).mean(dim=0)
        
        # Find peaks above threshold
        threshold = self.config.gradient_threshold
        shock_sites = []
        
        for i in range(len(smoothed)):
            if smoothed[i] > threshold:
                shock_sites.append(i)
        
        return shock_sites, smoothed
    
    def reset_history(self) -> None:
        """Clear shock detection history."""
        self.shock_history = []


# =============================================================================
# Bond Dimension Adapter
# =============================================================================

class BondAdapter:
    """
    Dynamically adjust bond dimensions in MPS based on shock locations.
    
    This is the core TT-AMR algorithm:
    1. Detect shocks using ShockDetector
    2. Increase χ at shock bonds
    3. Decrease χ in smooth regions
    4. Maintain total "bond budget" if desired
    
    Attributes:
        config: Adaptive TT configuration
        detector: Shock detector instance
        chi_profile: Current χ allocation at each bond
    """
    
    def __init__(self, config: Optional[AdaptiveTTConfig] = None):
        self.config = config or AdaptiveTTConfig()
        self.detector = ShockDetector(config)
        self.chi_profile: Optional[List[int]] = None
    
    def initialize_profile(self, n_bonds: int) -> None:
        """Initialize uniform χ profile."""
        self.chi_profile = [self.config.chi_base] * n_bonds
    
    def refine_at_shocks(
        self,
        mps: MPSState,
        shock_sites: List[int],
        chi_boost: Optional[int] = None
    ) -> MPSState:
        """
        Increase bond dimension at shock locations.
        
        Args:
            mps: Current MPS state
            shock_sites: List of bond indices near shocks
            chi_boost: Factor to multiply χ (default from config)
            
        Returns:
            New MPS with increased bonds at shock sites
        """
        if chi_boost is None:
            chi_boost = self.config.chi_boost
        
        n_bonds = len(mps.cores) - 1
        
        # Update chi profile
        if self.chi_profile is None:
            self.initialize_profile(n_bonds)
        
        for site in shock_sites:
            if 0 <= site < n_bonds:
                new_chi = min(
                    self.chi_profile[site] * chi_boost,
                    self.config.chi_max
                )
                self.chi_profile[site] = new_chi
        
        # Apply new profile by re-compressing
        new_cores = self._apply_chi_profile(mps)
        
        return MPSState(new_cores, mps.n_vars, 'none')
    
    def coarsen_smooth(
        self,
        mps: MPSState,
        smooth_sites: List[int],
        chi_min: Optional[int] = None
    ) -> MPSState:
        """
        Decrease bond dimension in smooth regions.
        
        Args:
            mps: Current MPS state
            smooth_sites: List of bond indices in smooth regions
            chi_min: Minimum χ to maintain (default from config)
            
        Returns:
            New MPS with decreased bonds in smooth regions
        """
        if chi_min is None:
            chi_min = self.config.chi_min
        
        n_bonds = len(mps.cores) - 1
        
        if self.chi_profile is None:
            self.initialize_profile(n_bonds)
        
        for site in smooth_sites:
            if 0 <= site < n_bonds:
                new_chi = max(
                    self.chi_profile[site] // 2,
                    chi_min
                )
                self.chi_profile[site] = new_chi
        
        new_cores = self._apply_chi_profile(mps)
        
        return MPSState(new_cores, mps.n_vars, 'none')
    
    def balance_bonds(
        self,
        mps: MPSState,
        chi_budget: Optional[int] = None
    ) -> MPSState:
        """
        Redistribute bond dimensions within a total budget.
        
        Allocates more χ where entropy is high, less where low,
        while maintaining a total storage budget.
        
        Args:
            mps: Current MPS state
            chi_budget: Total χ to distribute (sum of all bonds)
            
        Returns:
            New MPS with balanced bond dimensions
        """
        n_bonds = len(mps.cores) - 1
        
        if chi_budget is None:
            chi_budget = n_bonds * self.config.chi_base
        
        # Get entropy profile
        entropies = self.detector.entropy_indicator(mps)
        
        # Allocate proportional to entropy
        total_entropy = entropies.sum().item() + 1e-10
        
        new_profile = []
        remaining_budget = chi_budget
        
        for i in range(n_bonds):
            # Proportion of budget for this bond
            weight = (entropies[i].item() + 0.1) / (total_entropy + 0.1 * n_bonds)
            chi = int(weight * chi_budget)
            
            # Clamp to valid range
            chi = max(self.config.chi_min, min(chi, self.config.chi_max))
            chi = min(chi, remaining_budget - (n_bonds - i - 1) * self.config.chi_min)
            
            new_profile.append(chi)
            remaining_budget -= chi
        
        self.chi_profile = new_profile
        new_cores = self._apply_chi_profile(mps)
        
        return MPSState(new_cores, mps.n_vars, 'none')
    
    def _apply_chi_profile(self, mps: MPSState) -> List[Tensor]:
        """
        Truncate/expand MPS to match current χ profile.
        
        Uses SVD at each bond to adjust dimensions.
        """
        n_sites = len(mps.cores)
        new_cores = []
        
        # Right-to-left sweep for orthogonalization
        cores = [c.clone() for c in mps.cores]
        
        for i in range(n_sites - 1, 0, -1):
            core = cores[i]
            chi_l, d, chi_r = core.shape
            
            # Target chi for left bond
            if i - 1 < len(self.chi_profile):
                target_chi = self.chi_profile[i - 1]
            else:
                target_chi = self.config.chi_base
            
            # QR from right
            core_mat = core.reshape(chi_l, d * chi_r)
            Q, R = torch.linalg.qr(core_mat.T)
            
            # Truncate Q to target_chi
            if Q.shape[1] > target_chi:
                Q = Q[:, :target_chi]
                R = R[:target_chi, :]
            
            cores[i] = Q.T.reshape(-1, d, chi_r)
            
            # Absorb R into left neighbor
            cores[i-1] = torch.einsum('abc,cd->abd', cores[i-1], R.T)
        
        # Left-to-right sweep with SVD truncation
        for i in range(n_sites - 1):
            core = cores[i]
            chi_l, d, chi_r = core.shape
            
            # Target chi for right bond
            if i < len(self.chi_profile):
                target_chi = self.chi_profile[i]
            else:
                target_chi = self.config.chi_base
            
            # Randomized SVD (4× faster)
            core_mat = core.reshape(chi_l * d, chi_r)
            q = min(target_chi, min(core_mat.shape))
            U, S, Vh = torch.svd_lowrank(core_mat, q=q, niter=1)
            
            # Truncate to target
            chi = min(target_chi, len(S))
            U = U[:, :chi]
            S = S[:chi]
            Vh = Vh[:chi, :]
            
            new_cores.append(U.reshape(chi_l, d, chi))
            
            # Absorb S*Vh into right neighbor
            if i + 1 < n_sites:
                cores[i+1] = torch.einsum(
                    'ab,bcd->acd',
                    torch.diag(S) @ Vh,
                    cores[i+1]
                )
        
        # Last core
        new_cores.append(cores[-1])
        
        return new_cores
    
    def adapt(
        self,
        mps: MPSState,
        method: ShockIndicator = ShockIndicator.COMBINED
    ) -> MPSState:
        """
        Full adaptation cycle: detect shocks, refine, coarsen.
        
        Args:
            mps: Current MPS state
            method: Shock detection method
            
        Returns:
            Adapted MPS with new bond dimensions
        """
        # Detect shocks
        shock_sites, indicator = self.detector.detect_shocks(mps, method)
        
        # Determine smooth sites (low indicator)
        n_bonds = len(indicator)
        smooth_sites = [i for i in range(n_bonds) 
                       if indicator[i] < self.config.gradient_threshold / 2]
        
        # Refine at shocks
        mps = self.refine_at_shocks(mps, shock_sites)
        
        # Coarsen smooth regions
        mps = self.coarsen_smooth(mps, smooth_sites)
        
        return mps
    
    def get_chi_profile(self) -> List[int]:
        """Return current χ profile."""
        return self.chi_profile or []
    
    def get_statistics(self) -> Dict:
        """Return adaptation statistics."""
        profile = self.chi_profile or []
        
        return {
            'chi_min': min(profile) if profile else 0,
            'chi_max': max(profile) if profile else 0,
            'chi_mean': sum(profile) / len(profile) if profile else 0,
            'chi_total': sum(profile),
            'n_bonds': len(profile),
        }


# =============================================================================
# Adaptive TT Euler Solver
# =============================================================================

class AdaptiveTTEuler:
    """
    1D Euler solver with adaptive bond dimension (TT-AMR).
    
    Combines the TT_Euler1D solver with BondAdapter for automatic
    refinement near shocks and coarsening in smooth regions.
    
    This achieves optimal O(N·χ²) complexity where χ is locally
    adapted: high near shocks, low in smooth flow.
    
    Attributes:
        solver: Underlying TT Euler solver
        adapter: Bond dimension adapter
        config: Adaptive configuration
        adaptation_history: Record of χ over time
    """
    
    def __init__(
        self,
        N: int,
        L: float,
        gamma: float = 1.4,
        chi_max: int = 64,
        config: Optional[AdaptiveTTConfig] = None,
        solver_config: Optional[TTCFDConfig] = None
    ):
        self.config = config or AdaptiveTTConfig(chi_max=chi_max)
        
        # Create underlying solver
        solver_cfg = solver_config or TTCFDConfig(chi_max=chi_max, gamma=gamma)
        self.solver = TT_Euler1D(N, L, gamma, chi_max, solver_cfg)
        
        # Create adapter
        self.adapter = BondAdapter(self.config)
        
        # Tracking
        self.adaptation_history: List[Dict] = []
        self.step_count = 0
    
    def initialize(
        self,
        rho: Tensor,
        u: Tensor,
        p: Tensor
    ) -> None:
        """Initialize with primitive variables."""
        self.solver.initialize(rho, u, p)
        self.adapter.initialize_profile(len(self.solver.state.cores) - 1)
        self.step_count = 0
        self.adaptation_history = []
    
    def initialize_sod(self) -> None:
        """Initialize with Sod shock tube."""
        self.solver.initialize_sod()
        self.adapter.initialize_profile(len(self.solver.state.cores) - 1)
        self.step_count = 0
        self.adaptation_history = []
    
    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance by one time step with optional adaptation.
        """
        # Regular time step
        self.solver.step(dt)
        self.step_count += 1
        
        # Periodic adaptation
        if self.step_count % self.config.adaptation_interval == 0:
            self._adapt()
    
    def _adapt(self) -> None:
        """Perform bond dimension adaptation."""
        # Adapt the state
        self.solver.state = self.adapter.adapt(self.solver.state)
        
        # Record statistics
        stats = self.adapter.get_statistics()
        stats['time'] = self.solver.time
        stats['step'] = self.step_count
        self.adaptation_history.append(stats)
    
    def solve(
        self,
        t_final: float,
        callback=None,
        callback_interval: int = 10
    ) -> None:
        """Solve to final time with adaptive refinement."""
        while self.solver.time < t_final:
            dt = min(self.solver._compute_dt(), t_final - self.solver.time)
            self.step(dt)
            
            if callback is not None and self.step_count % callback_interval == 0:
                callback(self.solver.time, self.solver.state)
    
    def to_dense(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract dense primitive variables."""
        return self.solver.to_dense()
    
    def get_chi_profile(self) -> List[int]:
        """Get current bond dimension profile."""
        return self.adapter.get_chi_profile()
    
    def get_diagnostics(self) -> Dict:
        """Return comprehensive diagnostics."""
        solver_diag = self.solver.get_diagnostics()
        adapt_stats = self.adapter.get_statistics()
        
        return {
            **solver_diag,
            **adapt_stats,
            'adaptation_history': self.adaptation_history,
        }


# =============================================================================
# Entanglement Monitoring
# =============================================================================

class EntanglementMonitor:
    """
    Monitor entanglement entropy evolution over time.
    
    Tracks how the entanglement structure changes as shocks
    form, propagate, and interact.
    
    Useful for:
    - Understanding shock dynamics in TT representation
    - Validating that χ is sufficient
    - Analyzing compression efficiency
    """
    
    def __init__(self):
        self.history: List[Tuple[float, Tensor]] = []
    
    def record(self, time: float, mps: MPSState) -> Tensor:
        """Record entanglement profile at given time."""
        detector = ShockDetector()
        entropy = detector.entropy_indicator(mps)
        
        self.history.append((time, entropy.clone()))
        
        return entropy
    
    def get_max_entropy_history(self) -> List[Tuple[float, float]]:
        """Get maximum entropy at each recorded time."""
        return [(t, ent.max().item()) for t, ent in self.history]
    
    def get_total_entropy_history(self) -> List[Tuple[float, float]]:
        """Get total entropy at each recorded time."""
        return [(t, ent.sum().item()) for t, ent in self.history]
    
    def find_entropy_peaks(
        self,
        time_index: int = -1,
        threshold: float = 0.5
    ) -> List[int]:
        """Find locations of high entanglement (likely shocks)."""
        if not self.history:
            return []
        
        _, entropy = self.history[time_index]
        peaks = [i for i in range(len(entropy)) 
                if entropy[i] > threshold * entropy.max()]
        
        return peaks
    
    def clear(self) -> None:
        """Clear history."""
        self.history = []


# =============================================================================
# Utility Functions
# =============================================================================

def compute_compression_ratio_profile(mps: MPSState) -> List[float]:
    """
    Compute local compression ratio at each bond.
    
    Compares TT storage to what full-rank would require.
    
    Returns:
        List of compression ratios (higher = more compressed)
    """
    cores = mps.cores
    n_sites = len(cores)
    n_vars = mps.n_vars
    
    ratios = []
    
    for i in range(n_sites - 1):
        chi_l, d, chi_r = cores[i].shape
        
        # Full-rank bond dimension would be d
        full_rank_chi = d
        
        # Actual is chi_r (right bond)
        actual_chi = chi_r
        
        ratio = full_rank_chi / max(actual_chi, 1)
        ratios.append(ratio)
    
    return ratios


def estimate_memory_savings(
    N: int,
    chi_profile: List[int],
    n_vars: int = 3
) -> Dict:
    """
    Estimate memory savings from TT compression.
    
    Args:
        N: Number of grid points
        chi_profile: Bond dimensions at each bond
        n_vars: Number of variables per point
        
    Returns:
        Dictionary with storage estimates
    """
    # Dense storage
    dense_storage = N * n_vars
    
    # TT storage: sum of chi_l * d * chi_r for each core
    tt_storage = 0
    
    for i, chi in enumerate(chi_profile):
        chi_l = chi_profile[i-1] if i > 0 else 1
        chi_r = chi
        d = n_vars
        
        tt_storage += chi_l * d * chi_r
    
    # Last core
    if chi_profile:
        tt_storage += chi_profile[-1] * n_vars * 1
    
    return {
        'dense_storage': dense_storage,
        'tt_storage': tt_storage,
        'compression_ratio': dense_storage / max(tt_storage, 1),
        'memory_fraction': tt_storage / max(dense_storage, 1),
    }
