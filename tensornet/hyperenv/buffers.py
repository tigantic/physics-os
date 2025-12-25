"""
Experience Buffers
==================

Replay buffers and rollout storage for RL training.

Features:
- Replay buffer for off-policy
- Prioritized experience replay
- Rollout buffer for on-policy
- Trajectory storage
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Iterator, NamedTuple


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Experience(NamedTuple):
    """Single experience tuple."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    info: Optional[Dict[str, Any]] = None


@dataclass
class Trajectory:
    """
    Sequence of experiences forming an episode.
    
    Used for:
    - Episode storage
    - Return computation
    - Advantage estimation
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)
    
    # Computed values
    returns: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    log_probs: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return len(self.rewards)
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Optional[Dict] = None,
    ):
        """Add step to trajectory."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info or {})
    
    @property
    def total_reward(self) -> float:
        """Sum of rewards in trajectory."""
        return sum(self.rewards)
    
    def compute_returns(
        self,
        gamma: float = 0.99,
        last_value: float = 0.0,
    ) -> np.ndarray:
        """Compute discounted returns."""
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        running_return = last_value
        
        for t in reversed(range(len(self.rewards))):
            running_return = self.rewards[t] + gamma * running_return * (1 - self.dones[t])
            returns[t] = running_return
        
        self.returns = returns
        return returns
    
    def compute_gae(
        self,
        values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        self.values = values
        
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        running_gae = 0.0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - values[t]
            running_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * running_gae
            advantages[t] = running_gae
        
        self.advantages = advantages
        self.returns = advantages + values
        return advantages
    
    def to_batch(self) -> Dict[str, np.ndarray]:
        """Convert to batch dictionary."""
        batch = {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }
        
        if self.returns is not None:
            batch["returns"] = self.returns
        if self.advantages is not None:
            batch["advantages"] = self.advantages
        if self.values is not None:
            batch["values"] = self.values
        if self.log_probs is not None:
            batch["log_probs"] = self.log_probs
        
        return batch


@dataclass
class Batch:
    """
    Batch of experiences for training.
    
    All arrays have shape (batch_size, ...)
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    
    # Optional
    returns: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    log_probs: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None  # For prioritized replay
    indices: Optional[np.ndarray] = None  # For priority updates
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def to_torch(
        self,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        result = {
            "observations": torch.from_numpy(self.observations).float().to(device),
            "actions": torch.from_numpy(self.actions).float().to(device),
            "rewards": torch.from_numpy(self.rewards).float().to(device),
            "next_observations": torch.from_numpy(self.next_observations).float().to(device),
            "dones": torch.from_numpy(self.dones).float().to(device),
        }
        
        if self.returns is not None:
            result["returns"] = torch.from_numpy(self.returns).float().to(device)
        if self.advantages is not None:
            result["advantages"] = torch.from_numpy(self.advantages).float().to(device)
        if self.values is not None:
            result["values"] = torch.from_numpy(self.values).float().to(device)
        if self.log_probs is not None:
            result["log_probs"] = torch.from_numpy(self.log_probs).float().to(device)
        if self.weights is not None:
            result["weights"] = torch.from_numpy(self.weights).float().to(device)
        
        return result


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Standard replay buffer for off-policy algorithms.
    
    Features:
    - Circular buffer with fixed capacity
    - Uniform random sampling
    - Numpy storage for efficiency
    
    Example:
        buffer = ReplayBuffer(capacity=100000)
        
        buffer.add(obs, action, reward, next_obs, done)
        
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        observation_shape: Optional[Tuple[int, ...]] = None,
        action_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        self._ptr = 0
        self._size = 0
        
        # Storage (lazy initialization)
        self._observations: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._next_observations: Optional[np.ndarray] = None
        self._dones: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return self._size
    
    def _init_storage(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ):
        """Initialize storage arrays."""
        obs_shape = observation.shape
        act_shape = action.shape if len(action.shape) > 0 else (1,)
        
        self._observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((self.capacity, *act_shape), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        
        self.observation_shape = obs_shape
        self.action_shape = act_shape
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        **kwargs,  # Ignore extra args like info
    ):
        """Add experience to buffer."""
        # Lazy init
        if self._observations is None:
            self._init_storage(observation, action)
        
        # Store
        self._observations[self._ptr] = observation
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_observations[self._ptr] = next_observation
        self._dones[self._ptr] = float(done)
        
        # Update pointer
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Batch:
        """Sample random batch."""
        indices = np.random.randint(0, self._size, size=batch_size)
        
        return Batch(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_observations[indices],
            dones=self._dones[indices],
            indices=indices,
        )
    
    def reset(self):
        """Clear buffer."""
        self._ptr = 0
        self._size = 0


# =============================================================================
# PRIORITIZED REPLAY BUFFER
# =============================================================================

class SumTree:
    """Sum tree for efficient priority sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_ptr = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample with priority sum s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Total priority."""
        return self.tree[0]
    
    def add(self, priority: float, data_idx: int):
        """Add priority."""
        idx = data_idx + self.capacity - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, int]:
        """Get data index for priority sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.
    
    Samples experiences based on TD error priority.
    
    Features:
    - Sum tree for efficient sampling
    - Importance sampling weights
    - Priority updates
    
    Example:
        buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
        
        buffer.add(obs, action, reward, next_obs, done, priority=1.0)
        
        batch = buffer.sample(batch_size, beta=0.4)
        # batch.weights contains importance sampling weights
        
        buffer.update_priorities(batch.indices, new_priorities)
    """
    
    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,  # Small constant for priorities
        **kwargs,
    ):
        super().__init__(capacity, **kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self._tree = SumTree(capacity)
        self._max_priority = 1.0
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
        **kwargs,
    ):
        """Add experience with priority."""
        # Store experience
        super().add(observation, action, reward, next_observation, done)
        
        # Add priority
        if priority is None:
            priority = self._max_priority
        
        self._tree.add(priority ** self.alpha, self._ptr - 1)
    
    def sample(self, batch_size: int, beta: Optional[float] = None) -> Batch:
        """Sample batch with priorities."""
        beta = beta or self.beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Sample from tree
        segment = self._tree.total() / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            _, priority, data_idx = self._tree.get(s)
            indices[i] = data_idx
            priorities[i] = priority
        
        # Compute importance sampling weights
        min_prob = np.min(priorities) / self._tree.total()
        max_weight = (self._size * min_prob) ** (-beta)
        
        probs = priorities / self._tree.total()
        weights = (self._size * probs) ** (-beta) / max_weight
        
        return Batch(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_observations[indices],
            dones=self._dones[indices],
            weights=weights.astype(np.float32),
            indices=indices,
        )
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            priority = max(priority, self.epsilon)
            self._max_priority = max(self._max_priority, priority)
            self._tree.add(priority ** self.alpha, idx)


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms.
    
    Stores full rollouts and computes returns/advantages.
    
    Features:
    - Fixed-size rollout storage
    - GAE computation
    - Mini-batch iteration
    
    Example:
        buffer = RolloutBuffer(capacity=2048, gamma=0.99)
        
        for step in range(2048):
            buffer.add(obs, action, reward, done, value, log_prob)
        
        buffer.compute_returns_and_advantages(last_value)
        
        for batch in buffer.get_batches(batch_size=64):
            # Train on batch
            pass
        
        buffer.reset()
    """
    
    def __init__(
        self,
        capacity: int = 2048,
        observation_shape: Optional[Tuple[int, ...]] = None,
        action_shape: Optional[Tuple[int, ...]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self._ptr = 0
        
        # Storage (lazy initialization)
        self._observations: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None
        self._dones: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._log_probs: Optional[np.ndarray] = None
        
        # Computed
        self._returns: Optional[np.ndarray] = None
        self._advantages: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return self._ptr
    
    def _init_storage(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ):
        """Initialize storage arrays."""
        obs_shape = observation.shape
        act_shape = action.shape if len(action.shape) > 0 else (1,)
        
        self._observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((self.capacity, *act_shape), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._values = np.zeros(self.capacity, dtype=np.float32)
        self._log_probs = np.zeros(self.capacity, dtype=np.float32)
        self._returns = np.zeros(self.capacity, dtype=np.float32)
        self._advantages = np.zeros(self.capacity, dtype=np.float32)
        
        self.observation_shape = obs_shape
        self.action_shape = act_shape
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float = 0.0,
        log_prob: float = 0.0,
    ):
        """Add experience to buffer."""
        if self._observations is None:
            self._init_storage(observation, action)
        
        if self._ptr >= self.capacity:
            raise ValueError("Buffer full, call reset()")
        
        self._observations[self._ptr] = observation
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._values[self._ptr] = value
        self._log_probs[self._ptr] = log_prob
        
        self._ptr += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
    ):
        """Compute returns and GAE advantages."""
        # GAE
        running_gae = 0.0
        for t in reversed(range(self._ptr)):
            if t == self._ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self._dones[t]
            else:
                next_value = self._values[t + 1]
                next_non_terminal = 1.0 - self._dones[t]
            
            delta = (self._rewards[t] + 
                     self.gamma * next_value * next_non_terminal - 
                     self._values[t])
            
            running_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * running_gae
            self._advantages[t] = running_gae
        
        # Returns = advantages + values
        self._returns[:self._ptr] = self._advantages[:self._ptr] + self._values[:self._ptr]
        
        # Normalize advantages
        self._advantages[:self._ptr] = (
            (self._advantages[:self._ptr] - np.mean(self._advantages[:self._ptr])) /
            (np.std(self._advantages[:self._ptr]) + 1e-8)
        )
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[Batch]:
        """Iterate over mini-batches."""
        indices = np.arange(self._ptr)
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self._ptr, batch_size):
            end = min(start + batch_size, self._ptr)
            batch_indices = indices[start:end]
            
            yield Batch(
                observations=self._observations[batch_indices],
                actions=self._actions[batch_indices],
                rewards=self._rewards[batch_indices],
                next_observations=self._observations[batch_indices],  # Not used in on-policy
                dones=self._dones[batch_indices],
                returns=self._returns[batch_indices],
                advantages=self._advantages[batch_indices],
                values=self._values[batch_indices],
                log_probs=self._log_probs[batch_indices],
            )
    
    def reset(self):
        """Clear buffer."""
        self._ptr = 0


# =============================================================================
# TRAJECTORY BUFFER
# =============================================================================

class TrajectoryBuffer:
    """
    Buffer that stores complete trajectories.
    
    Useful for:
    - Episodic algorithms
    - Hindsight experience replay
    - Goal-conditioned learning
    
    Example:
        buffer = TrajectoryBuffer(max_trajectories=1000)
        
        trajectory = Trajectory()
        for step in episode:
            trajectory.add(obs, action, reward, done)
        buffer.add_trajectory(trajectory)
        
        sampled = buffer.sample_trajectories(n=10)
    """
    
    def __init__(
        self,
        max_trajectories: int = 1000,
        max_length: Optional[int] = None,
    ):
        self.max_trajectories = max_trajectories
        self.max_length = max_length
        
        self._trajectories: List[Trajectory] = []
    
    def __len__(self) -> int:
        return len(self._trajectories)
    
    def add_trajectory(self, trajectory: Trajectory):
        """Add complete trajectory."""
        if self.max_length and len(trajectory) > self.max_length:
            # Truncate
            trajectory = Trajectory(
                observations=trajectory.observations[:self.max_length],
                actions=trajectory.actions[:self.max_length],
                rewards=trajectory.rewards[:self.max_length],
                dones=trajectory.dones[:self.max_length],
                infos=trajectory.infos[:self.max_length],
            )
        
        self._trajectories.append(trajectory)
        
        # Remove old if over capacity
        while len(self._trajectories) > self.max_trajectories:
            self._trajectories.pop(0)
    
    def sample_trajectories(self, n: int = 1) -> List[Trajectory]:
        """Sample n trajectories."""
        if n >= len(self._trajectories):
            return list(self._trajectories)
        
        indices = np.random.choice(len(self._trajectories), size=n, replace=False)
        return [self._trajectories[i] for i in indices]
    
    def sample_transitions(self, batch_size: int) -> Batch:
        """Sample random transitions from all trajectories."""
        # Build flat index
        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        all_dones = []
        
        for traj in self._trajectories:
            for i in range(len(traj) - 1):
                all_obs.append(traj.observations[i])
                all_actions.append(traj.actions[i])
                all_rewards.append(traj.rewards[i])
                all_next_obs.append(traj.observations[i + 1])
                all_dones.append(traj.dones[i])
        
        if not all_obs:
            raise ValueError("No transitions in buffer")
        
        # Sample
        indices = np.random.randint(0, len(all_obs), size=batch_size)
        
        return Batch(
            observations=np.array([all_obs[i] for i in indices]),
            actions=np.array([all_actions[i] for i in indices]),
            rewards=np.array([all_rewards[i] for i in indices], dtype=np.float32),
            next_observations=np.array([all_next_obs[i] for i in indices]),
            dones=np.array([all_dones[i] for i in indices], dtype=np.float32),
        )
    
    def reset(self):
        """Clear buffer."""
        self._trajectories = []
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self._trajectories:
            return {}
        
        lengths = [len(t) for t in self._trajectories]
        rewards = [t.total_reward for t in self._trajectories]
        
        return {
            "n_trajectories": len(self._trajectories),
            "mean_length": np.mean(lengths),
            "mean_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
        }
