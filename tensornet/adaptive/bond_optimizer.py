# Copyright (c) 2025 Tigantic
# Phase 18: Adaptive Bond Dimension Optimizer
"""
Adaptive bond dimension management for tensor network time evolution.

Implements dynamic truncation strategies that adjust bond dimensions in real-time
based on entanglement growth, truncation error targets, and computational constraints.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    Iterator,
)

import torch
import numpy as np


class TruncationStrategy(Enum):
    """Truncation strategy selection."""
    
    FIXED = auto()           # Fixed bond dimension
    ERROR_TARGET = auto()    # Target truncation error
    ENTROPY_BASED = auto()   # Entropy-guided adaptation
    GRADIENT_BASED = auto()  # Gradient of observables
    HYBRID = auto()          # Combined strategies
    MEMORY_AWARE = auto()    # Memory-constrained adaptation


@dataclass
class AdaptiveBondConfig:
    """Configuration for adaptive bond dimension management.
    
    Attributes:
        chi_min: Minimum allowed bond dimension
        chi_max: Maximum allowed bond dimension
        target_truncation_error: Target cumulative truncation error
        entropy_threshold: Entropy threshold for adaptation trigger
        adaptation_rate: Rate of bond dimension change (0-1)
        memory_limit_mb: Memory limit in megabytes
        strategy: Truncation strategy to use
        check_interval: Steps between adaptation checks
        growth_factor: Maximum growth factor per adaptation
        shrink_factor: Minimum shrink factor per adaptation
        warmup_steps: Steps before adaptation begins
        enable_logging: Enable detailed logging
    """
    
    chi_min: int = 4
    chi_max: int = 512
    target_truncation_error: float = 1e-10
    entropy_threshold: float = 0.1
    adaptation_rate: float = 0.5
    memory_limit_mb: float = 4096.0
    strategy: TruncationStrategy = TruncationStrategy.ERROR_TARGET
    check_interval: int = 10
    growth_factor: float = 2.0
    shrink_factor: float = 0.5
    warmup_steps: int = 5
    enable_logging: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chi_min < 1:
            raise ValueError("chi_min must be at least 1")
        if self.chi_max < self.chi_min:
            raise ValueError("chi_max must be >= chi_min")
        if not 0 < self.target_truncation_error < 1:
            raise ValueError("target_truncation_error must be in (0, 1)")
        if not 0 < self.adaptation_rate <= 1:
            raise ValueError("adaptation_rate must be in (0, 1]")
        if self.growth_factor < 1:
            raise ValueError("growth_factor must be >= 1")
        if not 0 < self.shrink_factor <= 1:
            raise ValueError("shrink_factor must be in (0, 1]")


@dataclass
class TruncationRecord:
    """Record of a single truncation event.
    
    Attributes:
        step: Time step of truncation
        bond_index: Index of the bond being truncated
        chi_before: Bond dimension before truncation
        chi_after: Bond dimension after truncation
        truncation_error: Error from this truncation
        singular_values: Kept singular values (optional)
        discarded_weight: Weight of discarded singular values
        entropy_before: Entanglement entropy before truncation
        wall_time: Wall clock time for truncation
    """
    
    step: int
    bond_index: int
    chi_before: int
    chi_after: int
    truncation_error: float
    singular_values: Optional[torch.Tensor] = None
    discarded_weight: float = 0.0
    entropy_before: float = 0.0
    wall_time: float = 0.0


@dataclass
class AdaptationEvent:
    """Record of a bond dimension adaptation event.
    
    Attributes:
        step: Time step of adaptation
        trigger: What triggered the adaptation
        chi_old: Previous target bond dimension
        chi_new: New target bond dimension
        reason: Detailed reason for adaptation
        metrics: Associated metrics
    """
    
    step: int
    trigger: str
    chi_old: int
    chi_new: int
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


class BondDimensionTracker:
    """Track bond dimension evolution and statistics.
    
    Monitors how bond dimensions change over time, accumulates truncation
    errors, and provides diagnostics for adaptive algorithms.
    
    Attributes:
        config: Adaptive bond configuration
        current_chi: Current target bond dimension
        history: History of bond dimensions per site
        truncation_errors: Accumulated truncation errors
        adaptations: List of adaptation events
    """
    
    def __init__(
        self,
        config: AdaptiveBondConfig,
        num_sites: int,
        initial_chi: Optional[int] = None,
    ) -> None:
        """Initialize bond dimension tracker.
        
        Args:
            config: Configuration for adaptive management
            num_sites: Number of sites in the system
            initial_chi: Initial target bond dimension
        """
        self.config = config
        self.num_sites = num_sites
        self.current_chi = initial_chi or config.chi_min
        
        # Per-bond tracking
        self.bond_dimensions: List[int] = [self.current_chi] * (num_sites - 1)
        self.history: List[List[int]] = []
        self.truncation_records: List[TruncationRecord] = []
        self.adaptations: List[AdaptationEvent] = []
        
        # Accumulated statistics
        self.total_truncation_error: float = 0.0
        self.step_count: int = 0
        self.peak_chi: int = self.current_chi
        self._start_time: float = time.perf_counter()
    
    def record_truncation(
        self,
        bond_index: int,
        chi_before: int,
        chi_after: int,
        truncation_error: float,
        singular_values: Optional[torch.Tensor] = None,
        entropy: float = 0.0,
    ) -> TruncationRecord:
        """Record a truncation event.
        
        Args:
            bond_index: Index of the truncated bond
            chi_before: Bond dimension before truncation
            chi_after: Bond dimension after truncation
            truncation_error: Error from truncation
            singular_values: Retained singular values
            entropy: Entanglement entropy at this bond
            
        Returns:
            TruncationRecord for this event
        """
        wall_time = time.perf_counter() - self._start_time
        
        # Compute discarded weight
        discarded_weight = 0.0
        if singular_values is not None and chi_before > chi_after:
            all_weight = float(torch.sum(singular_values ** 2))
            kept_weight = float(torch.sum(singular_values[:chi_after] ** 2))
            discarded_weight = all_weight - kept_weight if all_weight > 0 else 0.0
        
        record = TruncationRecord(
            step=self.step_count,
            bond_index=bond_index,
            chi_before=chi_before,
            chi_after=chi_after,
            truncation_error=truncation_error,
            singular_values=singular_values,
            discarded_weight=discarded_weight,
            entropy_before=entropy,
            wall_time=wall_time,
        )
        
        self.truncation_records.append(record)
        self.total_truncation_error += truncation_error
        self.bond_dimensions[bond_index] = chi_after
        self.peak_chi = max(self.peak_chi, chi_after)
        
        return record
    
    def record_step(self) -> None:
        """Record completion of a time step."""
        self.history.append(self.bond_dimensions.copy())
        self.step_count += 1
    
    def record_adaptation(
        self,
        chi_old: int,
        chi_new: int,
        trigger: str,
        reason: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> AdaptationEvent:
        """Record a bond dimension adaptation.
        
        Args:
            chi_old: Previous target bond dimension
            chi_new: New target bond dimension
            trigger: What triggered the adaptation
            reason: Detailed reason string
            metrics: Associated metrics
            
        Returns:
            AdaptationEvent for this adaptation
        """
        event = AdaptationEvent(
            step=self.step_count,
            trigger=trigger,
            chi_old=chi_old,
            chi_new=chi_new,
            reason=reason,
            metrics=metrics or {},
        )
        
        self.adaptations.append(event)
        self.current_chi = chi_new
        
        return event
    
    def get_average_chi(self) -> float:
        """Get average bond dimension across all bonds."""
        if not self.bond_dimensions:
            return float(self.current_chi)
        return float(np.mean(self.bond_dimensions))
    
    def get_max_chi(self) -> int:
        """Get maximum current bond dimension."""
        return max(self.bond_dimensions) if self.bond_dimensions else self.current_chi
    
    def get_truncation_error_rate(self) -> float:
        """Get average truncation error per step."""
        if self.step_count == 0:
            return 0.0
        return self.total_truncation_error / self.step_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.
        
        Returns:
            Dictionary of tracking statistics
        """
        return {
            "current_chi": self.current_chi,
            "average_chi": self.get_average_chi(),
            "max_chi": self.get_max_chi(),
            "peak_chi": self.peak_chi,
            "total_truncation_error": self.total_truncation_error,
            "error_rate": self.get_truncation_error_rate(),
            "step_count": self.step_count,
            "num_adaptations": len(self.adaptations),
            "num_truncations": len(self.truncation_records),
            "bond_dimensions": self.bond_dimensions.copy(),
        }
    
    def should_adapt(self) -> bool:
        """Check if adaptation should be considered.
        
        Returns:
            True if we should check for adaptation
        """
        if self.step_count < self.config.warmup_steps:
            return False
        return self.step_count % self.config.check_interval == 0


class EntropyMonitor:
    """Monitor entanglement entropy across the system.
    
    Tracks von Neumann entanglement entropy at each bond to guide
    adaptive bond dimension management.
    
    Attributes:
        num_sites: Number of sites
        entropy_history: History of entropy at each bond
        current_entropies: Current entropy values
    """
    
    def __init__(self, num_sites: int, history_length: int = 100) -> None:
        """Initialize entropy monitor.
        
        Args:
            num_sites: Number of sites in the system
            history_length: Number of historical values to keep
        """
        self.num_sites = num_sites
        self.num_bonds = num_sites - 1
        self.history_length = history_length
        
        self.current_entropies: List[float] = [0.0] * self.num_bonds
        self.entropy_history: List[List[float]] = []
        self.peak_entropies: List[float] = [0.0] * self.num_bonds
    
    def compute_entropy(
        self,
        singular_values: torch.Tensor,
        normalize: bool = True,
    ) -> float:
        """Compute von Neumann entropy from singular values.
        
        The entanglement entropy is S = -sum_i λ_i² log(λ_i²)
        where λ_i are the singular values.
        
        Args:
            singular_values: Singular values of the bipartition
            normalize: Whether to normalize singular values
            
        Returns:
            Von Neumann entanglement entropy
        """
        sv = singular_values.clone()
        
        # Normalize if requested
        if normalize:
            norm = torch.sqrt(torch.sum(sv ** 2))
            if norm > 1e-15:
                sv = sv / norm
        
        # Compute probabilities (Schmidt coefficients squared)
        probs = sv ** 2
        probs = probs[probs > 1e-15]  # Filter near-zero values
        
        if len(probs) == 0:
            return 0.0
        
        # Entropy: S = -sum p_i log(p_i)
        entropy = -float(torch.sum(probs * torch.log(probs)))
        
        return entropy
    
    def update(
        self,
        bond_index: int,
        singular_values: torch.Tensor,
    ) -> float:
        """Update entropy at a specific bond.
        
        Args:
            bond_index: Index of the bond
            singular_values: Singular values at this bond
            
        Returns:
            Computed entropy value
        """
        entropy = self.compute_entropy(singular_values)
        self.current_entropies[bond_index] = entropy
        self.peak_entropies[bond_index] = max(
            self.peak_entropies[bond_index], entropy
        )
        return entropy
    
    def record_step(self) -> None:
        """Record current entropies to history."""
        self.entropy_history.append(self.current_entropies.copy())
        
        # Trim history if needed
        if len(self.entropy_history) > self.history_length:
            self.entropy_history = self.entropy_history[-self.history_length:]
    
    def get_max_entropy(self) -> float:
        """Get maximum current entropy across all bonds."""
        return max(self.current_entropies) if self.current_entropies else 0.0
    
    def get_average_entropy(self) -> float:
        """Get average entropy across all bonds."""
        if not self.current_entropies:
            return 0.0
        return float(np.mean(self.current_entropies))
    
    def get_entropy_growth_rate(self) -> float:
        """Estimate entropy growth rate from history.
        
        Returns:
            Estimated entropy growth rate (entropy/step)
        """
        if len(self.entropy_history) < 2:
            return 0.0
        
        # Use max entropy at each step
        max_entropies = [max(h) for h in self.entropy_history[-10:]]
        
        if len(max_entropies) < 2:
            return 0.0
        
        # Linear fit to estimate growth
        x = np.arange(len(max_entropies))
        y = np.array(max_entropies)
        
        # Simple slope calculation
        slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0
        
        return float(slope)
    
    def estimate_chi_for_entropy(
        self,
        target_entropy: float,
        safety_factor: float = 1.5,
    ) -> int:
        """Estimate required bond dimension for target entropy.
        
        For a maximally entangled state of dimension χ, the maximum
        entropy is log(χ). We invert this relationship.
        
        Args:
            target_entropy: Target entanglement entropy
            safety_factor: Safety factor for the estimate
            
        Returns:
            Estimated required bond dimension
        """
        # S_max = log(chi) => chi = exp(S)
        chi_estimate = int(np.ceil(safety_factor * np.exp(target_entropy)))
        return max(2, chi_estimate)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get entropy monitoring diagnostics.
        
        Returns:
            Dictionary of diagnostic information
        """
        return {
            "current_entropies": self.current_entropies.copy(),
            "max_entropy": self.get_max_entropy(),
            "average_entropy": self.get_average_entropy(),
            "peak_entropies": self.peak_entropies.copy(),
            "growth_rate": self.get_entropy_growth_rate(),
            "history_length": len(self.entropy_history),
        }


class TruncationScheduler:
    """Schedule adaptive truncation based on various strategies.
    
    Determines when and how to adapt bond dimensions based on the
    configured strategy and observed dynamics.
    
    Attributes:
        config: Adaptive bond configuration
        tracker: Bond dimension tracker
        entropy_monitor: Entropy monitor
    """
    
    def __init__(
        self,
        config: AdaptiveBondConfig,
        tracker: BondDimensionTracker,
        entropy_monitor: EntropyMonitor,
    ) -> None:
        """Initialize truncation scheduler.
        
        Args:
            config: Configuration for adaptive management
            tracker: Bond dimension tracker
            entropy_monitor: Entropy monitor
        """
        self.config = config
        self.tracker = tracker
        self.entropy_monitor = entropy_monitor
        
        # Strategy-specific state
        self._error_accumulator: float = 0.0
        self._adaptation_cooldown: int = 0
    
    def compute_target_chi(self) -> Tuple[int, str]:
        """Compute target bond dimension based on strategy.
        
        Returns:
            Tuple of (target_chi, reason_string)
        """
        strategy = self.config.strategy
        current_chi = self.tracker.current_chi
        
        if strategy == TruncationStrategy.FIXED:
            return current_chi, "fixed strategy"
        
        elif strategy == TruncationStrategy.ERROR_TARGET:
            return self._error_based_target()
        
        elif strategy == TruncationStrategy.ENTROPY_BASED:
            return self._entropy_based_target()
        
        elif strategy == TruncationStrategy.MEMORY_AWARE:
            return self._memory_aware_target()
        
        elif strategy == TruncationStrategy.HYBRID:
            return self._hybrid_target()
        
        else:
            return current_chi, "unknown strategy"
    
    def _error_based_target(self) -> Tuple[int, str]:
        """Compute target based on truncation error.
        
        Returns:
            Tuple of (target_chi, reason)
        """
        error_rate = self.tracker.get_truncation_error_rate()
        target_error = self.config.target_truncation_error
        current_chi = self.tracker.current_chi
        
        if error_rate > target_error * 10:
            # Error too high, increase chi
            new_chi = min(
                int(current_chi * self.config.growth_factor),
                self.config.chi_max
            )
            return new_chi, f"error rate {error_rate:.2e} > 10x target"
        
        elif error_rate < target_error * 0.01 and current_chi > self.config.chi_min:
            # Error very low, can decrease chi
            new_chi = max(
                int(current_chi * self.config.shrink_factor),
                self.config.chi_min
            )
            return new_chi, f"error rate {error_rate:.2e} < 0.01x target"
        
        return current_chi, f"error rate {error_rate:.2e} acceptable"
    
    def _entropy_based_target(self) -> Tuple[int, str]:
        """Compute target based on entanglement entropy.
        
        Returns:
            Tuple of (target_chi, reason)
        """
        max_entropy = self.entropy_monitor.get_max_entropy()
        growth_rate = self.entropy_monitor.get_entropy_growth_rate()
        current_chi = self.tracker.current_chi
        
        # Estimate required chi for current entropy
        required_chi = self.entropy_monitor.estimate_chi_for_entropy(max_entropy)
        
        # Predict future entropy (next check_interval steps)
        predicted_entropy = max_entropy + growth_rate * self.config.check_interval
        predicted_chi = self.entropy_monitor.estimate_chi_for_entropy(predicted_entropy)
        
        # Use the larger of current requirement and prediction
        target_chi = max(required_chi, predicted_chi)
        target_chi = max(self.config.chi_min, min(target_chi, self.config.chi_max))
        
        reason = f"entropy={max_entropy:.3f}, growth={growth_rate:.4f}/step"
        return target_chi, reason
    
    def _memory_aware_target(self) -> Tuple[int, str]:
        """Compute target respecting memory constraints.
        
        Returns:
            Tuple of (target_chi, reason)
        """
        # Estimate memory usage: O(L * chi^2 * d * sizeof(float))
        L = self.tracker.num_sites
        d = 2  # Assume spin-1/2
        bytes_per_element = 8  # float64
        
        # Memory for MPS: roughly L * chi^2 * d * 2 (for two copies)
        def estimate_memory_mb(chi: int) -> float:
            return (L * chi * chi * d * 2 * bytes_per_element) / (1024 * 1024)
        
        current_chi = self.tracker.current_chi
        memory_limit = self.config.memory_limit_mb
        
        # Find maximum chi that fits in memory
        max_feasible_chi = current_chi
        while estimate_memory_mb(max_feasible_chi + 1) < memory_limit * 0.8:
            max_feasible_chi += 1
            if max_feasible_chi >= self.config.chi_max:
                break
        
        # Get error-based target
        error_target, _ = self._error_based_target()
        
        # Respect memory limit
        target_chi = min(error_target, max_feasible_chi)
        target_chi = max(self.config.chi_min, target_chi)
        
        mem_est = estimate_memory_mb(target_chi)
        reason = f"memory-aware: {mem_est:.1f}MB / {memory_limit:.1f}MB limit"
        
        return target_chi, reason
    
    def _hybrid_target(self) -> Tuple[int, str]:
        """Compute target using hybrid strategy.
        
        Combines error-based and entropy-based approaches.
        
        Returns:
            Tuple of (target_chi, reason)
        """
        error_chi, error_reason = self._error_based_target()
        entropy_chi, entropy_reason = self._entropy_based_target()
        
        # Use the more conservative (larger) estimate
        if error_chi >= entropy_chi:
            target_chi = error_chi
            reason = f"hybrid (error-driven): {error_reason}"
        else:
            target_chi = entropy_chi
            reason = f"hybrid (entropy-driven): {entropy_reason}"
        
        # Apply memory constraint
        memory_chi, _ = self._memory_aware_target()
        if target_chi > memory_chi:
            target_chi = memory_chi
            reason = f"hybrid (memory-limited to {memory_chi})"
        
        return target_chi, reason
    
    def should_increase(self) -> Tuple[bool, str]:
        """Check if bond dimension should increase.
        
        Returns:
            Tuple of (should_increase, reason)
        """
        if self._adaptation_cooldown > 0:
            self._adaptation_cooldown -= 1
            return False, "in cooldown"
        
        target_chi, reason = self.compute_target_chi()
        
        if target_chi > self.tracker.current_chi:
            return True, reason
        return False, "no increase needed"
    
    def should_decrease(self) -> Tuple[bool, str]:
        """Check if bond dimension should decrease.
        
        Returns:
            Tuple of (should_decrease, reason)
        """
        if self._adaptation_cooldown > 0:
            return False, "in cooldown"
        
        target_chi, reason = self.compute_target_chi()
        
        if target_chi < self.tracker.current_chi:
            return True, reason
        return False, "no decrease needed"
    
    def apply_adaptation(self, new_chi: int, reason: str) -> None:
        """Apply a bond dimension adaptation.
        
        Args:
            new_chi: New target bond dimension
            reason: Reason for adaptation
        """
        old_chi = self.tracker.current_chi
        
        self.tracker.record_adaptation(
            chi_old=old_chi,
            chi_new=new_chi,
            trigger=self.config.strategy.name,
            reason=reason,
            metrics={
                "error_rate": self.tracker.get_truncation_error_rate(),
                "max_entropy": self.entropy_monitor.get_max_entropy(),
            },
        )
        
        # Set cooldown to prevent rapid oscillation
        self._adaptation_cooldown = self.config.check_interval // 2


class AdaptiveTruncator:
    """Main adaptive truncation engine.
    
    Orchestrates adaptive bond dimension management during tensor
    network time evolution.
    
    Attributes:
        config: Adaptive configuration
        tracker: Bond dimension tracker
        entropy_monitor: Entropy monitor
        scheduler: Truncation scheduler
    """
    
    def __init__(
        self,
        config: AdaptiveBondConfig,
        num_sites: int,
        initial_chi: Optional[int] = None,
    ) -> None:
        """Initialize adaptive truncator.
        
        Args:
            config: Configuration for adaptive management
            num_sites: Number of sites in the system
            initial_chi: Initial target bond dimension
        """
        self.config = config
        self.num_sites = num_sites
        
        self.tracker = BondDimensionTracker(config, num_sites, initial_chi)
        self.entropy_monitor = EntropyMonitor(num_sites)
        self.scheduler = TruncationScheduler(config, self.tracker, self.entropy_monitor)
    
    def truncate(
        self,
        tensor: torch.Tensor,
        bond_index: int,
        max_chi: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Perform adaptive SVD truncation.
        
        Args:
            tensor: Tensor to decompose (will be reshaped to matrix)
            bond_index: Index of the bond being truncated
            max_chi: Override maximum bond dimension
            
        Returns:
            Tuple of (U, S, Vh, truncation_error)
        """
        # Use current target if not specified
        target_chi = max_chi or self.tracker.current_chi
        target_chi = min(target_chi, self.config.chi_max)
        
        # Perform SVD
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
        
        # Update entropy monitor
        self.entropy_monitor.update(bond_index, S)
        
        # Determine truncation point
        chi_before = len(S)
        chi_after = min(chi_before, target_chi)
        
        # Compute truncation error
        if chi_after < chi_before:
            discarded = S[chi_after:]
            truncation_error = float(torch.sum(discarded ** 2))
        else:
            truncation_error = 0.0
        
        # Truncate
        U = U[:, :chi_after]
        S = S[:chi_after]
        Vh = Vh[:chi_after, :]
        
        # Record truncation
        self.tracker.record_truncation(
            bond_index=bond_index,
            chi_before=chi_before,
            chi_after=chi_after,
            truncation_error=truncation_error,
            singular_values=S,
            entropy=self.entropy_monitor.current_entropies[bond_index],
        )
        
        return U, S, Vh, truncation_error
    
    def step(self) -> Optional[AdaptationEvent]:
        """Complete a time step and check for adaptation.
        
        Returns:
            AdaptationEvent if adaptation occurred, None otherwise
        """
        self.tracker.record_step()
        self.entropy_monitor.record_step()
        
        if not self.tracker.should_adapt():
            return None
        
        # Check for adaptation
        target_chi, reason = self.scheduler.compute_target_chi()
        
        if target_chi != self.tracker.current_chi:
            self.scheduler.apply_adaptation(target_chi, reason)
            return self.tracker.adaptations[-1]
        
        return None
    
    def get_current_chi(self) -> int:
        """Get current target bond dimension."""
        return self.tracker.current_chi
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.
        
        Returns:
            Dictionary of statistics from all components
        """
        return {
            "tracker": self.tracker.get_statistics(),
            "entropy": self.entropy_monitor.get_diagnostics(),
            "config": {
                "strategy": self.config.strategy.name,
                "chi_range": (self.config.chi_min, self.config.chi_max),
                "target_error": self.config.target_truncation_error,
            },
        }


def estimate_optimal_chi(
    entanglement_entropy: float,
    target_truncation_error: float = 1e-10,
    safety_factor: float = 2.0,
) -> int:
    """Estimate optimal bond dimension from entropy and error target.
    
    Uses the relationship between entanglement entropy and required
    bond dimension, with corrections for truncation error.
    
    Args:
        entanglement_entropy: Maximum entanglement entropy
        target_truncation_error: Target cumulative truncation error
        safety_factor: Safety factor for the estimate
        
    Returns:
        Estimated optimal bond dimension
    """
    # Base estimate from entropy: chi ~ exp(S)
    chi_entropy = math.exp(entanglement_entropy)
    
    # Error correction: need more chi for lower error
    # Rough scaling: chi ~ 1/sqrt(epsilon) for error epsilon
    error_factor = 1.0 / math.sqrt(target_truncation_error)
    
    # Combine estimates
    chi_estimate = int(math.ceil(safety_factor * chi_entropy * math.log(error_factor + 1)))
    
    return max(4, chi_estimate)


def adapt_during_evolution(
    truncator: AdaptiveTruncator,
    tensors: List[torch.Tensor],
    num_steps: int,
    callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run adaptive truncation during time evolution.
    
    This is a demonstration function showing how to use the adaptive
    truncator during a simulation.
    
    Args:
        truncator: Initialized adaptive truncator
        tensors: List of MPS tensors
        num_steps: Number of time steps
        callback: Optional callback(step, stats) for monitoring
        
    Returns:
        Dictionary of evolution results and statistics
    """
    results = {
        "steps": [],
        "chi_history": [],
        "error_history": [],
        "entropy_history": [],
        "adaptations": [],
    }
    
    for step in range(num_steps):
        # Simulate truncation at each bond (example)
        for bond_idx in range(len(tensors) - 1):
            # Create a dummy matrix for truncation demo
            chi_left = tensors[bond_idx].shape[-1] if tensors[bond_idx].dim() > 1 else 4
            chi_right = tensors[bond_idx + 1].shape[0] if tensors[bond_idx + 1].dim() > 1 else 4
            
            # Demo matrix
            matrix = torch.randn(chi_left, chi_right, dtype=torch.float64)
            
            # Truncate
            U, S, Vh, error = truncator.truncate(matrix, bond_idx)
        
        # Complete step
        adaptation = truncator.step()
        
        # Record results
        stats = truncator.get_statistics()
        results["steps"].append(step)
        results["chi_history"].append(truncator.get_current_chi())
        results["error_history"].append(stats["tracker"]["error_rate"])
        results["entropy_history"].append(stats["entropy"]["max_entropy"])
        
        if adaptation:
            results["adaptations"].append({
                "step": step,
                "chi_old": adaptation.chi_old,
                "chi_new": adaptation.chi_new,
                "reason": adaptation.reason,
            })
        
        if callback:
            callback(step, stats)
    
    results["final_statistics"] = truncator.get_statistics()
    
    return results
