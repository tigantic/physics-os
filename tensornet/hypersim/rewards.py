"""
Reward Functions
================

Physics-based reward signals for RL training.

Types:
    Sparse: +1 at goal, 0 otherwise
    Dense: Continuous shaping toward goal
    Shaped: Potential-based shaping (provably optimal-preserving)
    Composite: Weighted combination of multiple rewards
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable, Union


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    scale: float = 1.0
    clip_min: float = -10.0
    clip_max: float = 10.0
    normalize: bool = False
    gamma: float = 0.99  # For potential-based shaping


class RewardFunction(ABC):
    """Base class for reward functions."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
        # For normalization
        self._running_mean = 0.0
        self._running_var = 1.0
        self._count = 0
    
    @abstractmethod
    def compute(self, field: 'Field', step: int) -> float:
        """Compute raw reward."""
        pass
    
    def __call__(self, field: 'Field', step: int) -> float:
        """Compute reward with scaling and clipping."""
        raw = self.compute(field, step)
        
        # Scale
        reward = raw * self.config.scale
        
        # Normalize (running mean/std)
        if self.config.normalize:
            self._count += 1
            delta = reward - self._running_mean
            self._running_mean += delta / self._count
            delta2 = reward - self._running_mean
            self._running_var += delta * delta2
            
            std = np.sqrt(self._running_var / max(1, self._count - 1)) + 1e-8
            reward = (reward - self._running_mean) / std
        
        # Clip
        reward = np.clip(reward, self.config.clip_min, self.config.clip_max)
        
        return float(reward)
    
    def reset(self):
        """Reset any internal state."""
        pass


# =============================================================================
# REWARD TYPES
# =============================================================================

class SparseReward(RewardFunction):
    """
    Sparse reward: +1 when goal achieved, 0 otherwise.
    
    Good for well-defined goals, hard for exploration.
    """
    
    def __init__(
        self,
        goal_fn: Callable[['Field'], bool],
        success_reward: float = 1.0,
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.goal_fn = goal_fn
        self.success_reward = success_reward
    
    def compute(self, field: 'Field', step: int) -> float:
        if self.goal_fn(field):
            return self.success_reward
        return 0.0


class DenseReward(RewardFunction):
    """
    Dense reward: continuous signal at every step.
    
    Provides gradient toward goal but may cause reward hacking.
    """
    
    def __init__(
        self,
        reward_fn: Callable[['Field', int], float],
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.reward_fn = reward_fn
    
    def compute(self, field: 'Field', step: int) -> float:
        return self.reward_fn(field, step)


class ShapedReward(RewardFunction):
    """
    Potential-based reward shaping.
    
    R' = R + γΦ(s') - Φ(s)
    
    Provably preserves optimal policy while accelerating learning.
    """
    
    def __init__(
        self,
        base_reward: RewardFunction,
        potential_fn: Callable[['Field'], float],
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.base_reward = base_reward
        self.potential_fn = potential_fn
        self._prev_potential = None
    
    def compute(self, field: 'Field', step: int) -> float:
        # Base reward
        base = self.base_reward.compute(field, step)
        
        # Potential shaping
        current_potential = self.potential_fn(field)
        
        if self._prev_potential is None:
            shaping = 0.0
        else:
            shaping = self.config.gamma * current_potential - self._prev_potential
        
        self._prev_potential = current_potential
        
        return base + shaping
    
    def reset(self):
        self._prev_potential = None
        self.base_reward.reset()


class CompositeReward(RewardFunction):
    """
    Weighted combination of multiple reward functions.
    
    R = Σ w_i * R_i
    """
    
    def __init__(
        self,
        rewards: List[Tuple[RewardFunction, float]],
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.rewards = rewards  # [(reward_fn, weight), ...]
    
    def compute(self, field: 'Field', step: int) -> float:
        total = 0.0
        for reward_fn, weight in self.rewards:
            total += weight * reward_fn.compute(field, step)
        return total
    
    def reset(self):
        for reward_fn, _ in self.rewards:
            reward_fn.reset()


# =============================================================================
# PHYSICS-BASED REWARDS
# =============================================================================

class TargetMatchReward(RewardFunction):
    """
    Reward for matching a target field.
    
    R = -||f - f_target||²
    
    Computed efficiently in QTT format.
    """
    
    def __init__(
        self,
        target: 'Field',
        metric: str = 'l2',  # 'l2', 'l1', 'linf'
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.target = target
        self.metric = metric
    
    def compute(self, field: 'Field', step: int) -> float:
        # Compute difference in core space (approximate but fast)
        total_diff = 0.0
        
        for f_core, t_core in zip(field.cores, self.target.cores):
            diff = f_core - t_core.to(f_core.device)
            
            if self.metric == 'l2':
                total_diff += (diff ** 2).sum().item()
            elif self.metric == 'l1':
                total_diff += diff.abs().sum().item()
            elif self.metric == 'linf':
                total_diff = max(total_diff, diff.abs().max().item())
        
        return -total_diff


class VorticityReward(RewardFunction):
    """
    Reward for vorticity magnitude.
    
    Useful for creating/maintaining vortical structures.
    """
    
    def __init__(
        self,
        target_vorticity: float = 1.0,
        maximize: bool = True,
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.target_vorticity = target_vorticity
        self.maximize = maximize
    
    def compute(self, field: 'Field', step: int) -> float:
        # Approximate vorticity from QTT cores
        vorticity = 0.0
        
        for i, core in enumerate(field.cores):
            # Vorticity ~ antisymmetric part
            if core.shape[1] >= 2:
                vort = (core[:, 0, :] - core[:, 1, :]).abs().sum().item()
                vorticity += vort
        
        if self.maximize:
            return vorticity
        else:
            # Match target
            return -abs(vorticity - self.target_vorticity)


class EnergyReward(RewardFunction):
    """
    Reward based on field energy.
    
    E = Σ ||core||²
    """
    
    def __init__(
        self,
        target_energy: Optional[float] = None,
        minimize: bool = True,
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.target_energy = target_energy
        self.minimize = minimize
    
    def compute(self, field: 'Field', step: int) -> float:
        energy = sum(c.norm().item() ** 2 for c in field.cores)
        
        if self.target_energy is not None:
            # Match target energy
            return -abs(energy - self.target_energy)
        elif self.minimize:
            return -energy
        else:
            return energy


class DissipationReward(RewardFunction):
    """
    Reward for energy dissipation rate.
    
    Useful for damping control tasks.
    """
    
    def __init__(
        self,
        target_rate: float = 0.0,
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.target_rate = target_rate
        self._prev_energy = None
    
    def compute(self, field: 'Field', step: int) -> float:
        current_energy = sum(c.norm().item() ** 2 for c in field.cores)
        
        if self._prev_energy is None:
            dissipation = 0.0
        else:
            dissipation = self._prev_energy - current_energy
        
        self._prev_energy = current_energy
        
        if self.target_rate == 0.0:
            # Maximize dissipation
            return dissipation
        else:
            # Match target rate
            return -abs(dissipation - self.target_rate)
    
    def reset(self):
        self._prev_energy = None


class BoundaryPenalty(RewardFunction):
    """
    Penalty for violating boundary conditions.
    
    Encourages flow to stay within valid regions.
    """
    
    def __init__(
        self,
        boundary_type: str = 'dirichlet',  # 'dirichlet', 'neumann', 'obstacle'
        penalty: float = 1.0,
        config: Optional[RewardConfig] = None,
    ):
        super().__init__(config)
        self.boundary_type = boundary_type
        self.penalty = penalty
    
    def compute(self, field: 'Field', step: int) -> float:
        # Check boundary cores
        violation = 0.0
        
        # First and last cores correspond to boundaries
        first_core = field.cores[0]
        last_core = field.cores[-1]
        
        if self.boundary_type == 'dirichlet':
            # Should be zero at boundary
            violation += first_core[:, 0, :].abs().sum().item()
            violation += last_core[:, 1, :].abs().sum().item()
        
        return -violation * self.penalty


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def make_reward(
    reward_type: str,
    **kwargs,
) -> RewardFunction:
    """
    Factory for creating reward functions.
    
    Args:
        reward_type: 'energy', 'vorticity', 'dissipation', 'boundary'
        **kwargs: Reward-specific arguments
        
    Returns:
        RewardFunction instance
    """
    config = RewardConfig(
        scale=kwargs.pop('scale', 1.0),
        clip_min=kwargs.pop('clip_min', -10.0),
        clip_max=kwargs.pop('clip_max', 10.0),
    )
    
    if reward_type == 'energy':
        return EnergyReward(config=config, **kwargs)
    elif reward_type == 'vorticity':
        return VorticityReward(config=config, **kwargs)
    elif reward_type == 'dissipation':
        return DissipationReward(config=config, **kwargs)
    elif reward_type == 'boundary':
        return BoundaryPenalty(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
