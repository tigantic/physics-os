"""
Radiation Hardening: Triple Modular Redundancy (TMR)
=====================================================

Phase 23: Infrastructure for radiation-tolerant GPU computation.

Implements TMR (Triple Modular Redundancy) for critical computations
to detect and correct Single Event Upsets (SEUs) caused by cosmic rays.

Key Features:
- Triple execution with majority voting
- CUDA-accelerated voting kernel
- Conservation law watchdog
- Checkpoint/rollback capability

References:
    - Lyons & Vanderkulk, "TMR Design Techniques" (1962)
    - Normand, "Single Event Upset at Ground Level" IEEE TNS (1996)
    - NASA/TM-2006-214301, "Radiation Hardening Techniques for FPGAs"

Constitution Compliance: Article V (Mission Assurance)
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
from pathlib import Path
import copy


# =============================================================================
# Configuration
# =============================================================================

class VotingStrategy(Enum):
    """Voting strategies for TMR."""
    MAJORITY = 'majority'  # 2-of-3 match
    UNANIMOUS = 'unanimous'  # All 3 must match
    WEIGHTED = 'weighted'  # Weighted by confidence
    MEDIAN = 'median'  # Take median value (for continuous)


class RecoveryAction(Enum):
    """Actions on error detection."""
    ROLLBACK = 'rollback'  # Restore from checkpoint
    RECOMPUTE = 'recompute'  # Re-execute computation
    ALERT = 'alert'  # Log and continue
    HALT = 'halt'  # Stop execution


@dataclass
class TMRConfig:
    """
    Configuration for Triple Modular Redundancy.
    
    Attributes:
        enabled: Whether TMR is active
        voting_strategy: How to combine triple results
        tolerance_abs: Absolute tolerance for floating point comparison
        tolerance_rel: Relative tolerance for floating point comparison
        recovery_action: Action on error detection
        max_retries: Maximum recomputation attempts
        checkpoint_interval: Steps between checkpoints
        keep_checkpoints: Number of checkpoints to retain
    """
    enabled: bool = True
    voting_strategy: VotingStrategy = VotingStrategy.MEDIAN
    tolerance_abs: float = 1e-10
    tolerance_rel: float = 1e-8
    recovery_action: RecoveryAction = RecoveryAction.RECOMPUTE
    max_retries: int = 3
    checkpoint_interval: int = 100
    keep_checkpoints: int = 5


@dataclass
class SEUEvent:
    """
    Detected Single Event Upset.
    
    Attributes:
        timestamp: Detection time
        location: Tensor index where upset detected
        original_value: Value before correction
        corrected_value: Value after voting
        replica_values: Values from all three replicas
    """
    timestamp: float
    location: Tuple[int, ...]
    original_value: float
    corrected_value: float
    replica_values: Tuple[float, float, float]


# =============================================================================
# Majority Voter
# =============================================================================

class MajorityVoter:
    """
    Majority voting for tensor values.
    
    Compares three tensor replicas and outputs the majority value
    at each position. For floating-point, uses tolerance-based matching.
    """
    
    def __init__(
        self,
        tolerance_abs: float = 1e-10,
        tolerance_rel: float = 1e-8
    ):
        self.tolerance_abs = tolerance_abs
        self.tolerance_rel = tolerance_rel
        self._seu_log: List[SEUEvent] = []
    
    def vote_tensors(
        self,
        t1: Tensor,
        t2: Tensor,
        t3: Tensor
    ) -> Tensor:
        """
        Vote on three tensor replicas.
        
        Uses median for floating-point tensors with continuous values.
        
        Args:
            t1, t2, t3: Three tensor replicas
            
        Returns:
            Consensus tensor
        """
        # Stack and compute median
        stacked = torch.stack([t1, t2, t3], dim=0)
        consensus, _ = torch.median(stacked, dim=0)
        
        return consensus
    
    def vote_with_detection(
        self,
        t1: Tensor,
        t2: Tensor,
        t3: Tensor
    ) -> Tuple[Tensor, List[SEUEvent]]:
        """
        Vote and detect upsets.
        
        Returns consensus and list of detected SEUs.
        """
        events = []
        timestamp = time.time()
        
        # Stack tensors
        stacked = torch.stack([t1, t2, t3], dim=0)
        consensus, _ = torch.median(stacked, dim=0)
        
        # Detect disagreements
        diff_1 = torch.abs(t1 - consensus)
        diff_2 = torch.abs(t2 - consensus)
        diff_3 = torch.abs(t3 - consensus)
        
        threshold = self.tolerance_abs + self.tolerance_rel * torch.abs(consensus)
        
        upset_mask = (
            (diff_1 > threshold) | 
            (diff_2 > threshold) | 
            (diff_3 > threshold)
        )
        
        if upset_mask.any():
            # Find upset locations
            upset_indices = torch.nonzero(upset_mask, as_tuple=False)
            
            for idx in upset_indices[:10]:  # Log first 10
                idx_tuple = tuple(idx.tolist())
                events.append(SEUEvent(
                    timestamp=timestamp,
                    location=idx_tuple,
                    original_value=max(t1[idx_tuple].item(), t2[idx_tuple].item(), t3[idx_tuple].item()),
                    corrected_value=consensus[idx_tuple].item(),
                    replica_values=(t1[idx_tuple].item(), t2[idx_tuple].item(), t3[idx_tuple].item())
                ))
        
        self._seu_log.extend(events)
        return consensus, events
    
    def detect_seu(
        self,
        t1: Tensor,
        t2: Tensor,
        t3: Tensor
    ) -> Optional[Tensor]:
        """
        Detect SEU locations without correcting.
        
        Returns tensor of boolean flags where disagreement occurred.
        """
        stacked = torch.stack([t1, t2, t3], dim=0)
        consensus, _ = torch.median(stacked, dim=0)
        
        diff_1 = torch.abs(t1 - consensus)
        diff_2 = torch.abs(t2 - consensus)
        diff_3 = torch.abs(t3 - consensus)
        
        threshold = self.tolerance_abs + self.tolerance_rel * torch.abs(consensus)
        
        upset_mask = (
            (diff_1 > threshold) | 
            (diff_2 > threshold) | 
            (diff_3 > threshold)
        )
        
        if upset_mask.any():
            return upset_mask
        return None
    
    @property
    def seu_log(self) -> List[SEUEvent]:
        """Get logged SEU events."""
        return self._seu_log
    
    def clear_log(self):
        """Clear SEU log."""
        self._seu_log.clear()


# =============================================================================
# TMR Executor
# =============================================================================

class TMRExecutor:
    """
    Triple Modular Redundancy executor for GPU kernels.
    
    Executes computation three times and votes on results.
    
    Attributes:
        kernel_fn: The function to execute in triplicate
        config: TMR configuration
        voter: Majority voting module
    """
    
    def __init__(
        self,
        kernel_fn: Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
        config: Optional[TMRConfig] = None
    ):
        self.kernel_fn = kernel_fn
        self.config = config or TMRConfig()
        self.voter = MajorityVoter(
            tolerance_abs=self.config.tolerance_abs,
            tolerance_rel=self.config.tolerance_rel
        )
        
        self._retry_count = 0
        self._seu_count = 0
    
    def execute(
        self,
        *args,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Execute with TMR protection.
        
        Args:
            *args, **kwargs: Arguments to kernel function
            
        Returns:
            Voted output tensor(s)
        """
        if not self.config.enabled:
            return self.kernel_fn(*args, **kwargs)
        
        for attempt in range(self.config.max_retries):
            try:
                # Execute triple
                out1, out2, out3 = self._launch_triple(*args, **kwargs)
                
                # Vote
                result, events = self._vote(out1, out2, out3)
                
                if events:
                    self._seu_count += len(events)
                
                return result
                
            except Exception as e:
                self._retry_count += 1
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"TMR execution failed after {self.config.max_retries} retries: {e}")
        
        raise RuntimeError("TMR execution failed")
    
    def _launch_triple(
        self,
        *args,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """
        Execute kernel three times.
        
        For true radiation tolerance, these would run on different
        hardware cores. Here we simulate with repeated execution.
        """
        # Clone inputs to ensure independent execution
        def clone_arg(x):
            if isinstance(x, Tensor):
                return x.clone()
            return x
        
        args1 = tuple(clone_arg(a) for a in args)
        args2 = tuple(clone_arg(a) for a in args)
        args3 = tuple(clone_arg(a) for a in args)
        
        out1 = self.kernel_fn(*args1, **kwargs)
        out2 = self.kernel_fn(*args2, **kwargs)
        out3 = self.kernel_fn(*args3, **kwargs)
        
        return out1, out2, out3
    
    def _vote(
        self,
        out1: Any,
        out2: Any,
        out3: Any
    ) -> Tuple[Any, List[SEUEvent]]:
        """
        Apply voting to triple outputs.
        
        Handles both single tensors and tuples of tensors.
        """
        all_events = []
        
        if isinstance(out1, Tensor):
            result, events = self.voter.vote_with_detection(out1, out2, out3)
            return result, events
        
        elif isinstance(out1, tuple):
            results = []
            for o1, o2, o3 in zip(out1, out2, out3):
                if isinstance(o1, Tensor):
                    r, events = self.voter.vote_with_detection(o1, o2, o3)
                    results.append(r)
                    all_events.extend(events)
                else:
                    # Non-tensor, just pass through (e.g., int, float)
                    results.append(o1)
            return tuple(results), all_events
        
        else:
            # Non-tensor output
            return out1, []
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return {
            'retry_count': self._retry_count,
            'seu_count': self._seu_count
        }


# =============================================================================
# Conservation Watchdog
# =============================================================================

class ConservationWatchdog:
    """
    Physics-based sanity checks for numerical solutions.
    
    Monitors conservation laws (mass, momentum, energy) to detect
    non-physical states that may indicate computational errors.
    
    Attributes:
        energy_threshold: Maximum allowed energy change per step (fraction)
        mass_threshold: Maximum allowed mass change per step (fraction)
        momentum_threshold: Maximum allowed momentum change (fraction)
    """
    
    def __init__(
        self,
        energy_threshold: float = 0.01,
        mass_threshold: float = 1e-6,
        momentum_threshold: float = 0.01,
        gamma: float = 1.4
    ):
        self.energy_threshold = energy_threshold
        self.mass_threshold = mass_threshold
        self.momentum_threshold = momentum_threshold
        self.gamma = gamma
        
        self._anomaly_log: List[Dict[str, Any]] = []
    
    def compute_mass(self, rho: Tensor, dx: float = 1.0) -> float:
        """Compute total mass from density field."""
        return (rho * dx).sum().item()
    
    def compute_energy(
        self,
        rho: Tensor,
        rhou: Tensor,
        E: Tensor,
        dx: float = 1.0
    ) -> float:
        """Compute total energy from conservative variables."""
        return (E * dx).sum().item()
    
    def compute_momentum(
        self,
        rhou: Tensor,
        dx: float = 1.0
    ) -> float:
        """Compute total momentum."""
        return (rhou * dx).sum().item()
    
    def check_energy(
        self,
        E_current: Tensor,
        E_prev: Tensor,
        dx: float = 1.0
    ) -> float:
        """
        Check energy conservation.
        
        Returns anomaly score (0 = perfect, >1 = violation).
        """
        total_current = (E_current * dx).sum().item()
        total_prev = (E_prev * dx).sum().item()
        
        if abs(total_prev) < 1e-10:
            return 0.0
        
        change = abs(total_current - total_prev) / abs(total_prev)
        return change / self.energy_threshold
    
    def check_mass(
        self,
        rho_current: Tensor,
        rho_prev: Tensor,
        dx: float = 1.0
    ) -> float:
        """
        Check mass conservation.
        
        Returns anomaly score (0 = perfect, >1 = violation).
        """
        mass_current = (rho_current * dx).sum().item()
        mass_prev = (rho_prev * dx).sum().item()
        
        if abs(mass_prev) < 1e-10:
            return 0.0
        
        change = abs(mass_current - mass_prev) / abs(mass_prev)
        return change / self.mass_threshold
    
    def check_positivity(self, rho: Tensor, p: Tensor) -> bool:
        """Check that density and pressure are positive."""
        return (rho > 0).all().item() and (p > 0).all().item()
    
    def check_state(
        self,
        state: Dict[str, Tensor],
        prev_state: Dict[str, Tensor],
        dx: float = 1.0
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Comprehensive state check.
        
        Args:
            state: Current state {'rho': ..., 'rhou': ..., 'E': ...}
            prev_state: Previous state
            dx: Grid spacing
            
        Returns:
            (is_valid, anomaly_scores)
        """
        scores = {}
        
        # Mass conservation
        if 'rho' in state and 'rho' in prev_state:
            scores['mass'] = self.check_mass(state['rho'], prev_state['rho'], dx)
        
        # Energy conservation
        if 'E' in state and 'E' in prev_state:
            scores['energy'] = self.check_energy(state['E'], prev_state['E'], dx)
        
        # Positivity
        if 'rho' in state and 'p' in state:
            scores['positivity'] = 0.0 if self.check_positivity(state['rho'], state['p']) else 10.0
        
        # Check for NaN/Inf
        for key, tensor in state.items():
            if isinstance(tensor, Tensor):
                if not torch.isfinite(tensor).all():
                    scores['finite'] = 10.0
                    break
        else:
            scores['finite'] = 0.0
        
        is_valid = all(s < 1.0 for s in scores.values())
        
        if not is_valid:
            self._anomaly_log.append({
                'timestamp': time.time(),
                'scores': scores.copy()
            })
        
        return is_valid, scores
    
    def rollback_if_anomaly(
        self,
        state: Dict[str, Tensor],
        checkpoint: Dict[str, Tensor],
        prev_state: Dict[str, Tensor],
        dx: float = 1.0
    ) -> Dict[str, Tensor]:
        """
        Rollback to checkpoint if anomaly detected.
        
        Returns the state to use (current if valid, checkpoint if not).
        """
        is_valid, _ = self.check_state(state, prev_state, dx)
        
        if is_valid:
            return state
        else:
            # Deep copy checkpoint
            return {k: v.clone() for k, v in checkpoint.items()}
    
    @property
    def anomaly_log(self) -> List[Dict[str, Any]]:
        """Get logged anomalies."""
        return self._anomaly_log


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Periodic state checkpointing for rollback capability.
    
    Saves snapshots of computational state at regular intervals
    to enable recovery from detected errors.
    
    Attributes:
        checkpoint_dir: Directory for checkpoint files
        keep_last_n: Number of checkpoints to retain
    """
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        keep_last_n: int = 5
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: Dict[int, Path] = {}
    
    def save(
        self,
        state: Dict[str, Tensor],
        step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            state: State dictionary to save
            step: Step number
            metadata: Optional metadata
            
        Returns:
            Path to saved checkpoint
        """
        filename = f"checkpoint_{step:08d}.pt"
        filepath = self.checkpoint_dir / filename
        
        save_data = {
            'step': step,
            'state': {k: v.cpu().clone() for k, v in state.items() if isinstance(v, Tensor)},
            'metadata': metadata or {}
        }
        
        torch.save(save_data, filepath)
        self._checkpoints[step] = filepath
        
        # Prune old checkpoints
        self.prune()
        
        return filepath
    
    def load(self, step: int) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """
        Load checkpoint.
        
        Args:
            step: Step number to load
            
        Returns:
            (state, metadata)
        """
        if step not in self._checkpoints:
            raise KeyError(f"No checkpoint at step {step}")
        
        filepath = self._checkpoints[step]
        data = torch.load(filepath)
        
        return data['state'], data['metadata']
    
    def load_latest(self) -> Tuple[Dict[str, Tensor], Dict[str, Any], int]:
        """
        Load most recent checkpoint.
        
        Returns:
            (state, metadata, step)
        """
        if not self._checkpoints:
            raise RuntimeError("No checkpoints available")
        
        latest_step = max(self._checkpoints.keys())
        state, metadata = self.load(latest_step)
        
        return state, metadata, latest_step
    
    def prune(self):
        """Remove old checkpoints, keeping only last N."""
        steps = sorted(self._checkpoints.keys())
        
        while len(steps) > self.keep_last_n:
            old_step = steps.pop(0)
            old_path = self._checkpoints.pop(old_step)
            
            if old_path.exists():
                old_path.unlink()
    
    def clear(self):
        """Remove all checkpoints."""
        for path in self._checkpoints.values():
            if path.exists():
                path.unlink()
        self._checkpoints.clear()
    
    @property
    def available_steps(self) -> List[int]:
        """List available checkpoint steps."""
        return sorted(self._checkpoints.keys())


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'VotingStrategy',
    'RecoveryAction',
    # Config
    'TMRConfig',
    'SEUEvent',
    # Classes
    'MajorityVoter',
    'TMRExecutor',
    'ConservationWatchdog',
    'CheckpointManager',
]
