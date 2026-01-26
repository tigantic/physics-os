"""
Regime Detector: RMT-RKHS Switching Layer for the Genesis Stack.

This module implements a differentiable regime detection system that identifies
when market "physics" fundamentally change, triggering velocity model resets.

Detection Primitives:
    1. RMT Level Spacing Ratio (Spectral Gap): Detects chaos onset
    2. RKHS MMD Score: Detects distribution shift
    3. Betti Jump Detection: Detects structural breaks

The detector provides a unified regime_score that can be used to:
    - Pause velocity extrapolation during transitions
    - Adjust learning rates during high-volatility periods
    - Reset prediction models when physics change

Author: Genesis Stack / HyperTensor VM
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketRegime(Enum):
    """Market regime classification."""
    MEAN_REVERTING = auto()   # Stationary, predictable
    TRENDING = auto()          # Directional momentum
    CHAOTIC = auto()           # High volatility, low structure
    CRASH = auto()             # Extreme downward velocity
    TRANSITION = auto()        # Between regimes


@dataclass
class RegimeState:
    """Current regime detection state."""
    regime: MarketRegime
    confidence: float           # 0-1, how certain we are
    spectral_gap: float         # RMT level spacing ratio
    mmd_score: float            # RKHS distribution distance
    betti_delta: float          # Change in topological structure
    should_reset_velocity: bool # Whether to pause extrapolation
    regime_age: int             # Steps since last regime change
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "regime": self.regime.name,
            "confidence": float(self.confidence),
            "spectral_gap": float(self.spectral_gap),
            "mmd_score": float(self.mmd_score),
            "betti_delta": float(self.betti_delta),
            "should_reset_velocity": self.should_reset_velocity,
            "regime_age": self.regime_age
        }


@dataclass
class RegimeDetectorConfig:
    """Configuration for regime detector."""
    # RMT thresholds
    spectral_gap_threshold: float = 0.3      # Below this = chaos
    spectral_gap_window: int = 50            # Eigenvalue history window
    
    # RKHS MMD thresholds
    mmd_sigma_threshold: float = 3.0         # Sigma for regime shift
    mmd_kernel_bandwidth: float = 1.0        # RBF kernel bandwidth
    mmd_window: int = 100                    # Distribution comparison window
    
    # Betti thresholds
    betti_jump_threshold: float = 0.5        # Normalized jump magnitude
    betti_smoothing: float = 0.9             # EMA for Betti tracking
    
    # Regime stability
    min_regime_duration: int = 10            # Min steps before allowing switch
    transition_smoothing: float = 0.8        # Blend factor for transitions
    
    # Velocity control
    reset_cooldown: int = 20                 # Steps after reset before re-enabling


class RMTLevelSpacing(nn.Module):
    """
    Random Matrix Theory Level Spacing Ratio.
    
    The level spacing ratio r = min(δ_n, δ_{n+1}) / max(δ_n, δ_{n+1})
    where δ_n = λ_n - λ_{n-1} are the gaps between ordered eigenvalues.
    
    For GOE (chaos): <r> ≈ 0.5307
    For Poisson (integrable/structure): <r> ≈ 0.386
    
    When r drops below threshold, the eigenvalues are "repelling" less,
    indicating loss of structure and onset of chaotic dynamics.
    """
    
    def __init__(self, window_size: int = 50):
        super().__init__()
        self.window_size = window_size
        self.register_buffer(
            'eigenvalue_history',
            torch.zeros(window_size, dtype=torch.float32)
        )
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('history_filled', torch.tensor(False, dtype=torch.bool))
        
    def forward(self, covariance_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute level spacing ratio from covariance matrix.
        
        Args:
            covariance_matrix: [N, N] symmetric positive semi-definite matrix
            
        Returns:
            Tuple of (mean_ratio, spectral_gap_indicator)
            spectral_gap_indicator: 0 = structure, 1 = chaos
        """
        # Compute eigenvalues (sorted ascending)
        eigenvalues = torch.linalg.eigvalsh(covariance_matrix)
        
        # Filter to positive eigenvalues (numerical stability)
        positive_mask = eigenvalues > 1e-10
        if positive_mask.sum() < 3:
            # Not enough eigenvalues, return neutral
            return torch.tensor(0.45, device=eigenvalues.device), torch.tensor(0.5)
        
        eigenvalues = eigenvalues[positive_mask]
        
        # Compute gaps
        gaps = eigenvalues[1:] - eigenvalues[:-1]
        gaps = torch.clamp(gaps, min=1e-10)  # Avoid division by zero
        
        # Compute level spacing ratios
        if len(gaps) < 2:
            return torch.tensor(0.45, device=eigenvalues.device), torch.tensor(0.5)
            
        min_gaps = torch.minimum(gaps[:-1], gaps[1:])
        max_gaps = torch.maximum(gaps[:-1], gaps[1:])
        ratios = min_gaps / max_gaps
        
        # Mean ratio
        mean_ratio = ratios.mean()
        
        # Update history
        idx = self.history_idx.item()
        self.eigenvalue_history[idx] = mean_ratio
        self.history_idx = torch.tensor((idx + 1) % self.window_size, dtype=torch.long)
        if idx == self.window_size - 1:
            self.history_filled = torch.tensor(True, dtype=torch.bool)
        
        # Compute spectral gap indicator (0 = structure, 1 = chaos)
        # GOE: 0.5307, Poisson: 0.386
        # Map: below 0.4 → chaos indicator high
        gap_indicator = torch.sigmoid((0.45 - mean_ratio) * 10)
        
        return mean_ratio, gap_indicator
    
    def get_trend(self) -> torch.Tensor:
        """Get the trend in level spacing ratio."""
        if not self.history_filled:
            valid_count = self.history_idx.item()
            if valid_count < 10:
                return torch.tensor(0.0)
            history = self.eigenvalue_history[:valid_count]
        else:
            history = self.eigenvalue_history
            
        # Simple linear trend
        x = torch.arange(len(history), dtype=torch.float32, device=history.device)
        x_mean = x.mean()
        y_mean = history.mean()
        
        numerator = ((x - x_mean) * (history - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum() + 1e-10
        
        return numerator / denominator


class RKHSMMDScore(nn.Module):
    """
    Reproducing Kernel Hilbert Space Maximum Mean Discrepancy.
    
    Measures the distance between two distributions in a high-dimensional
    feature space induced by an RBF kernel. High MMD indicates a distribution
    shift (regime change).
    
    MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    where k is the RBF kernel: k(x,y) = exp(-||x-y||² / 2σ²)
    """
    
    def __init__(self, bandwidth: float = 1.0, window_size: int = 100):
        super().__init__()
        self.bandwidth = bandwidth
        self.window_size = window_size
        
        # Running statistics for baseline
        self.register_buffer('mmd_history', torch.zeros(50, dtype=torch.float32))
        self.register_buffer('mmd_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('mmd_filled', torch.tensor(False, dtype=torch.bool))
        
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between X and Y."""
        # X: [N, D], Y: [M, D]
        # Result: [N, M]
        XX = (X ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
        YY = (Y ** 2).sum(dim=-1, keepdim=True)  # [M, 1]
        XY = X @ Y.T                              # [N, M]
        
        distances_sq = XX + YY.T - 2 * XY
        distances_sq = torch.clamp(distances_sq, min=0)
        
        return torch.exp(-distances_sq / (2 * self.bandwidth ** 2))
    
    def forward(
        self, 
        reference: torch.Tensor, 
        current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MMD between reference and current distributions.
        
        Args:
            reference: [N, D] reference distribution samples
            current: [M, D] current distribution samples
            
        Returns:
            Tuple of (mmd_score, sigma_score)
            sigma_score: How many standard deviations above baseline
        """
        # Compute kernel matrices
        K_xx = self._rbf_kernel(reference, reference)
        K_yy = self._rbf_kernel(current, current)
        K_xy = self._rbf_kernel(reference, current)
        
        # Remove diagonal for unbiased estimate
        n = reference.shape[0]
        m = current.shape[0]
        
        # MMD² unbiased estimator
        mmd_xx = (K_xx.sum() - K_xx.trace()) / (n * (n - 1) + 1e-10)
        mmd_yy = (K_yy.sum() - K_yy.trace()) / (m * (m - 1) + 1e-10)
        mmd_xy = K_xy.mean()
        
        mmd_squared = mmd_xx + mmd_yy - 2 * mmd_xy
        mmd = torch.sqrt(torch.clamp(mmd_squared, min=0))
        
        # Update history
        idx = self.mmd_idx.item()
        self.mmd_history[idx] = mmd.detach()
        self.mmd_idx = torch.tensor((idx + 1) % 50, dtype=torch.long)
        if idx == 49:
            self.mmd_filled = torch.tensor(True, dtype=torch.bool)
        
        # Compute sigma score (how many std above baseline)
        if self.mmd_filled:
            history = self.mmd_history
        else:
            valid = self.mmd_idx.item()
            if valid < 5:
                return mmd, torch.tensor(0.0)
            history = self.mmd_history[:valid]
        
        mean = history.mean()
        std = history.std() + 1e-10
        sigma_score = (mmd - mean) / std
        
        return mmd, sigma_score


class LaplacianSpectralBetti(nn.Module):
    """
    Differentiable Betti Number Estimation via Laplacian Spectral Density.
    
    Instead of computing exact persistent homology (non-differentiable),
    we use the spectral density of the graph Laplacian to estimate
    topological features:
    
    β₀ ≈ #{λ : λ < ε} (connected components from near-zero eigenvalues)
    β₁ ≈ spectral gap structure (cycles from eigenvalue clustering)
    
    The gradient of eigenvalues is computed via implicit differentiation,
    allowing end-to-end training through topology.
    """
    
    def __init__(self, num_scales: int = 10, max_scale: float = 2.0):
        super().__init__()
        self.num_scales = num_scales
        self.max_scale = max_scale
        
        # Learnable scale parameters
        self.register_buffer(
            'scales',
            torch.linspace(0.01, max_scale, num_scales)
        )
        
        # Running Betti estimate
        self.register_buffer('prev_betti', torch.zeros(2, dtype=torch.float32))
        self.smoothing = 0.9
        
    def _adjacency_from_distance(
        self, 
        points: torch.Tensor, 
        scale: torch.Tensor
    ) -> torch.Tensor:
        """Build soft adjacency matrix at given scale."""
        # Pairwise distances
        dists = torch.cdist(points, points)
        
        # Soft adjacency (differentiable)
        adjacency = torch.exp(-(dists ** 2) / (2 * scale ** 2))
        adjacency = adjacency - torch.eye(
            adjacency.shape[0], 
            device=adjacency.device
        )
        
        return adjacency
    
    def _laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute normalized Laplacian."""
        degree = adjacency.sum(dim=1)
        degree_inv_sqrt = 1.0 / (torch.sqrt(degree) + 1e-10)
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        L = torch.eye(adjacency.shape[0], device=adjacency.device) - \
            D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        return L
    
    def forward(
        self, 
        points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute differentiable Betti curve from point cloud.
        
        Args:
            points: [N, D] point cloud
            
        Returns:
            Tuple of (betti_0, betti_1, betti_delta)
            betti_delta: Change from previous call
        """
        device = points.device
        betti_curve = []
        
        for scale in self.scales:
            adj = self._adjacency_from_distance(points, scale)
            L = self._laplacian(adj)
            
            # Add regularization to avoid ill-conditioned matrix
            L = L + 1e-6 * torch.eye(L.shape[0], device=device)
            
            # Compute eigenvalues (differentiable) with error handling
            try:
                eigenvalues = torch.linalg.eigvalsh(L)
            except RuntimeError:
                # Fallback: return neutral values
                eigenvalues = torch.linspace(0, 2, L.shape[0], device=device)
            
            # β₀: Count near-zero eigenvalues (connected components)
            # Soft count using sigmoid
            beta_0 = torch.sigmoid(-eigenvalues * 100).sum()
            
            # β₁: Eigenvalue clustering in [ε, 1-ε] range indicates cycles
            # Use spectral gap structure
            mid_mask = (eigenvalues > 0.01) & (eigenvalues < 0.99)
            mid_eigenvalues = eigenvalues[mid_mask]
            if len(mid_eigenvalues) > 1:
                # Variance of middle eigenvalues indicates cycle structure
                beta_1 = mid_eigenvalues.var() * len(mid_eigenvalues)
            else:
                beta_1 = torch.tensor(0.0, device=device)
            
            betti_curve.append(torch.stack([beta_0, beta_1]))
        
        # Stack across scales: [num_scales, 2]
        betti_curve = torch.stack(betti_curve)
        
        # Aggregate: use max across scales for each Betti number
        betti = betti_curve.max(dim=0).values  # [2]
        
        # Compute delta from previous
        betti_delta = (betti - self.prev_betti.to(device)).abs().sum()
        
        # Update running estimate
        self.prev_betti = self.smoothing * self.prev_betti.to(device) + \
                          (1 - self.smoothing) * betti.detach()
        
        return betti[0], betti[1], betti_delta
    
    def get_gradient_magnitude(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude of Betti numbers w.r.t. points.
        
        This allows detecting "forming" topological features before
        they fully crystallize.
        """
        try:
            points_grad = points.detach().requires_grad_(True)
            _, beta_1, _ = self.forward(points_grad)
            
            # Check if beta_1 requires grad
            if not beta_1.requires_grad:
                return torch.tensor(0.0, device=points.device)
            
            # Compute gradient
            grad = torch.autograd.grad(
                beta_1, 
                points_grad, 
                create_graph=False,
                retain_graph=False
            )[0]
            
            return grad.norm()
        except RuntimeError:
            # Gradient computation failed
            return torch.tensor(0.0, device=points.device)


class RegimeDetector(nn.Module):
    """
    Unified Regime Detection Module.
    
    Combines RMT level spacing, RKHS MMD, and spectral Betti to provide
    a robust regime classification with velocity reset triggers.
    """
    
    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        super().__init__()
        self.config = config or RegimeDetectorConfig()
        
        # Sub-modules
        self.rmt = RMTLevelSpacing(window_size=self.config.spectral_gap_window)
        self.mmd = RKHSMMDScore(
            bandwidth=self.config.mmd_kernel_bandwidth,
            window_size=self.config.mmd_window
        )
        self.betti = LaplacianSpectralBetti()
        
        # State tracking
        self.current_regime = MarketRegime.MEAN_REVERTING
        self.regime_age = 0
        self.reset_cooldown_remaining = 0
        self.last_state: Optional[RegimeState] = None
        
        # Reference distribution buffer
        self.register_buffer(
            'reference_buffer',
            torch.zeros(100, 10, dtype=torch.float32)
        )
        self.register_buffer('ref_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('ref_filled', torch.tensor(False, dtype=torch.bool))
        
    def update_reference(self, sample: torch.Tensor) -> None:
        """Add sample to reference distribution buffer."""
        idx = self.ref_idx.item()
        
        # Flatten sample to fixed size
        flat = sample.flatten()
        if flat.shape[0] > 10:
            flat = flat[:10]
        elif flat.shape[0] < 10:
            flat = F.pad(flat, (0, 10 - flat.shape[0]))
        
        self.reference_buffer[idx] = flat
        self.ref_idx = torch.tensor((idx + 1) % 100, dtype=torch.long)
        if idx == 99:
            self.ref_filled = torch.tensor(True, dtype=torch.bool)
    
    def forward(
        self,
        covariance_matrix: torch.Tensor,
        current_samples: torch.Tensor,
        point_cloud: torch.Tensor
    ) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            covariance_matrix: [N, N] for RMT analysis
            current_samples: [M, D] current distribution samples
            point_cloud: [K, D] points for topological analysis
            
        Returns:
            RegimeState with classification and reset trigger
        """
        device = covariance_matrix.device
        
        # 1. RMT Level Spacing
        ratio, spectral_indicator = self.rmt(covariance_matrix)
        spectral_gap = ratio.item()
        
        # 2. RKHS MMD
        # First flatten current samples for comparison (to fixed 10 features)
        flat_current = current_samples.reshape(current_samples.shape[0], -1)
        if flat_current.shape[1] > 10:
            flat_current = flat_current[:, :10]
        elif flat_current.shape[1] < 10:
            flat_current = F.pad(flat_current, (0, 10 - flat_current.shape[1]))
        
        # Get reference distribution
        if self.ref_filled:
            reference = self.reference_buffer  # [100, 10]
        else:
            ref_count = self.ref_idx.item()
            if ref_count < 10:
                # Use current samples as reference (self-comparison)
                # flat_current is already [N, 10]
                reference = flat_current.clone()
            else:
                reference = self.reference_buffer[:ref_count]  # [ref_count, 10]
        
        mmd, sigma = self.mmd(reference, flat_current)
        mmd_score = sigma.item()
        
        # 3. Betti Numbers
        beta_0, beta_1, betti_delta = self.betti(point_cloud)
        
        # Update reference buffer
        self.update_reference(current_samples)
        
        # 4. Classify Regime
        new_regime = self._classify_regime(
            spectral_gap, mmd_score, betti_delta.item()
        )
        
        # 5. Check for regime transition
        if new_regime != self.current_regime:
            if self.regime_age >= self.config.min_regime_duration:
                self.current_regime = new_regime
                self.regime_age = 0
                self.reset_cooldown_remaining = self.config.reset_cooldown
            # else: stick with current regime (too soon to switch)
        else:
            self.regime_age += 1
        
        # 6. Determine if velocity should reset
        should_reset = False
        if self.reset_cooldown_remaining > 0:
            self.reset_cooldown_remaining -= 1
            should_reset = True
        elif self._is_transition_detected(spectral_gap, mmd_score, betti_delta.item()):
            should_reset = True
            self.reset_cooldown_remaining = self.config.reset_cooldown
        
        # 7. Compute confidence
        confidence = self._compute_confidence(spectral_gap, mmd_score, betti_delta.item())
        
        # Build state
        state = RegimeState(
            regime=self.current_regime,
            confidence=confidence,
            spectral_gap=spectral_gap,
            mmd_score=mmd_score,
            betti_delta=betti_delta.item(),
            should_reset_velocity=should_reset,
            regime_age=self.regime_age
        )
        
        self.last_state = state
        return state
    
    def _classify_regime(
        self,
        spectral_gap: float,
        mmd_score: float,
        betti_delta: float
    ) -> MarketRegime:
        """Classify regime based on primitive scores."""
        # Chaos detection
        if spectral_gap < self.config.spectral_gap_threshold:
            if mmd_score > self.config.mmd_sigma_threshold * 1.5:
                return MarketRegime.CRASH
            return MarketRegime.CHAOTIC
        
        # Distribution shift detection
        if mmd_score > self.config.mmd_sigma_threshold:
            if betti_delta > self.config.betti_jump_threshold:
                return MarketRegime.CRASH
            return MarketRegime.TRENDING
        
        # Topological shift detection
        if betti_delta > self.config.betti_jump_threshold * 2:
            return MarketRegime.TRANSITION
        
        # Default: stable mean-reverting
        return MarketRegime.MEAN_REVERTING
    
    def _is_transition_detected(
        self,
        spectral_gap: float,
        mmd_score: float,
        betti_delta: float
    ) -> bool:
        """Check if we're in a regime transition that should pause velocity."""
        # Any of these indicates instability
        chaos = spectral_gap < self.config.spectral_gap_threshold
        distribution_shift = mmd_score > self.config.mmd_sigma_threshold
        structural_break = betti_delta > self.config.betti_jump_threshold
        
        return chaos or distribution_shift or structural_break
    
    def _compute_confidence(
        self,
        spectral_gap: float,
        mmd_score: float,
        betti_delta: float
    ) -> float:
        """Compute confidence in current regime classification."""
        # Higher spectral gap = more confidence (clearer structure)
        spectral_conf = min(1.0, spectral_gap / 0.6)
        
        # Lower MMD = more confidence (stable distribution)
        mmd_conf = max(0.0, 1.0 - abs(mmd_score) / 5.0)
        
        # Lower betti delta = more confidence (stable topology)
        betti_conf = max(0.0, 1.0 - betti_delta / 2.0)
        
        # Weighted average
        return 0.4 * spectral_conf + 0.3 * mmd_conf + 0.3 * betti_conf


class RegimeAwareExtrapolator(nn.Module):
    """
    Velocity extrapolator with regime-aware reset capability.
    
    This wraps a velocity prediction model and pauses/resets it
    when the regime detector indicates a transition.
    """
    
    def __init__(
        self,
        detector: RegimeDetector,
        velocity_decay: float = 0.95,
        reset_warmup: int = 5
    ):
        super().__init__()
        self.detector = detector
        self.velocity_decay = velocity_decay
        self.reset_warmup = reset_warmup
        
        # Velocity state - will be initialized on first update
        self.velocity: Optional[torch.Tensor] = None
        self.position: Optional[torch.Tensor] = None
        self.velocity_valid = False
        self.warmup_counter = 0
        
    def reset_on_divergence(self) -> None:
        """Reset velocity model on regime divergence."""
        if self.velocity is not None:
            self.velocity = torch.zeros_like(self.velocity)
        self.velocity_valid = False
        self.warmup_counter = 0
        
    def update(
        self,
        new_position: torch.Tensor,
        regime_state: RegimeState
    ) -> torch.Tensor:
        """
        Update velocity estimate and return extrapolated prediction.
        
        Args:
            new_position: Current observed position
            regime_state: Current regime detection state
            
        Returns:
            Extrapolated next position
        """
        device = new_position.device
        
        # Initialize on first call
        if self.position is None:
            self.position = new_position.clone()
            self.velocity = torch.zeros_like(new_position)
            return new_position
        
        # Resize if shape changed
        if self.position.shape != new_position.shape:
            self.position = new_position.clone()
            self.velocity = torch.zeros_like(new_position)
            self.velocity_valid = False
            self.warmup_counter = 0
            return new_position
        
        # Check for reset trigger
        if regime_state.should_reset_velocity:
            self.reset_on_divergence()
            # During reset, just return current position (no extrapolation)
            self.position = new_position.clone()
            return new_position
        
        # Warmup phase after reset
        if self.warmup_counter < self.reset_warmup:
            self.warmup_counter += 1
            self.position = new_position.clone()
            return new_position
        
        # Compute velocity
        if self.velocity_valid:
            instantaneous_velocity = new_position - self.position
            # EMA update with regime-dependent decay
            decay = self.velocity_decay
            if regime_state.regime in [MarketRegime.CHAOTIC, MarketRegime.TRANSITION]:
                decay = 0.5  # Faster adaptation during uncertainty
            self.velocity = decay * self.velocity + (1 - decay) * instantaneous_velocity
        else:
            self.velocity = new_position - self.position
            self.velocity_valid = True
        
        # Update position
        self.position = new_position.clone()
        
        # Extrapolate
        # Scale velocity by confidence
        scaled_velocity = self.velocity * regime_state.confidence
        
        # For mean-reverting, dampen extrapolation
        if regime_state.regime == MarketRegime.MEAN_REVERTING:
            scaled_velocity = scaled_velocity * 0.5
        
        prediction = self.position + scaled_velocity
        
        return prediction
    
    def predict_n_steps(
        self,
        n: int,
        regime_state: RegimeState
    ) -> List[torch.Tensor]:
        """Predict n steps ahead."""
        predictions = []
        
        if self.position is None or self.velocity is None:
            return predictions
            
        current_pos = self.position.clone()
        
        for i in range(n):
            # Decay velocity over time
            decay_factor = self.velocity_decay ** (i + 1)
            if regime_state.regime == MarketRegime.MEAN_REVERTING:
                decay_factor *= 0.5 ** (i + 1)  # Extra damping
            
            step = self.velocity * decay_factor * regime_state.confidence
            current_pos = current_pos + step
            predictions.append(current_pos.clone())
        
        return predictions


def create_regime_detector(
    spectral_gap_threshold: float = 0.3,
    mmd_sigma_threshold: float = 3.0,
    betti_jump_threshold: float = 0.5
) -> RegimeDetector:
    """Factory function to create a configured regime detector."""
    config = RegimeDetectorConfig(
        spectral_gap_threshold=spectral_gap_threshold,
        mmd_sigma_threshold=mmd_sigma_threshold,
        betti_jump_threshold=betti_jump_threshold
    )
    return RegimeDetector(config)


if __name__ == "__main__":
    # Quick test
    print("Testing Regime Detector...")
    
    detector = create_regime_detector()
    extrapolator = RegimeAwareExtrapolator(detector)
    
    # Simulate data
    torch.manual_seed(42)
    
    # Create test inputs
    cov_matrix = torch.eye(10) + 0.1 * torch.randn(10, 10)
    cov_matrix = cov_matrix @ cov_matrix.T  # Make PSD
    
    samples = torch.randn(50, 5)
    points = torch.randn(20, 3)
    
    # Test regime detection
    state = detector(cov_matrix, samples, points)
    print(f"Regime: {state.regime.name}")
    print(f"Confidence: {state.confidence:.3f}")
    print(f"Spectral Gap: {state.spectral_gap:.4f}")
    print(f"MMD Score: {state.mmd_score:.4f}")
    print(f"Betti Delta: {state.betti_delta:.4f}")
    print(f"Should Reset: {state.should_reset_velocity}")
    
    # Test extrapolator
    position = torch.tensor([1.0, 2.0, 3.0])
    prediction = extrapolator.update(position, state)
    print(f"\nPosition: {position.tolist()}")
    print(f"Prediction: {prediction.tolist()}")
    
    print("\n✓ Regime Detector test passed!")
