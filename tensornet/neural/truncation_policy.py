"""
Neural Truncation Policy Module
===============================

Reinforcement learning-based truncation policies for adaptive
bond dimension management in tensor network algorithms.

The policy learns to balance accuracy vs computational cost
by observing truncation error, entropy, and resource usage.
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


class PolicyAction(Enum):
    """Actions available to the truncation policy."""
    
    DECREASE_LARGE = auto()   # Decrease χ by 50%
    DECREASE_SMALL = auto()   # Decrease χ by 10%
    MAINTAIN = auto()         # Keep current χ
    INCREASE_SMALL = auto()   # Increase χ by 10%
    INCREASE_LARGE = auto()   # Increase χ by 50%
    
    @classmethod
    def from_index(cls, index: int) -> "PolicyAction":
        """Get action from index."""
        actions = list(cls)
        return actions[index % len(actions)]
    
    def to_index(self) -> int:
        """Get index of action."""
        return list(PolicyAction).index(self)
    
    def apply(self, current_chi: int, chi_min: int, chi_max: int) -> int:
        """Apply action to get new chi value."""
        if self == PolicyAction.DECREASE_LARGE:
            new_chi = int(current_chi * 0.5)
        elif self == PolicyAction.DECREASE_SMALL:
            new_chi = int(current_chi * 0.9)
        elif self == PolicyAction.MAINTAIN:
            new_chi = current_chi
        elif self == PolicyAction.INCREASE_SMALL:
            new_chi = int(current_chi * 1.1)
        elif self == PolicyAction.INCREASE_LARGE:
            new_chi = int(current_chi * 1.5)
        else:
            new_chi = current_chi
        
        return max(chi_min, min(chi_max, new_chi))


@dataclass
class PolicyState:
    """State observation for the truncation policy.
    
    Captures the current state of the tensor network for
    policy decision making.
    
    Attributes:
        current_chi: Current bond dimension
        truncation_error: Recent truncation error
        entropy: Current entanglement entropy
        entropy_gradient: Rate of entropy change
        step: Current simulation step
        memory_usage: Current memory usage fraction
        target_error: Target truncation error
        chi_min: Minimum allowed chi
        chi_max: Maximum allowed chi
    """
    
    current_chi: int
    truncation_error: float
    entropy: float
    entropy_gradient: float
    step: int
    memory_usage: float
    target_error: float
    chi_min: int
    chi_max: int
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to normalized tensor for neural network."""
        # Normalize features to [0, 1] or reasonable range
        chi_normalized = (self.current_chi - self.chi_min) / max(1, self.chi_max - self.chi_min)
        error_log = -math.log10(max(self.truncation_error, 1e-16))
        target_log = -math.log10(max(self.target_error, 1e-16))
        
        features = torch.tensor([
            chi_normalized,
            min(error_log / 16.0, 1.0),  # Normalize log error
            min(self.entropy / 10.0, 1.0),  # Normalize entropy
            np.tanh(self.entropy_gradient),  # Bounded gradient
            min(self.step / 1000.0, 1.0),  # Normalize step
            self.memory_usage,
            min(target_log / 16.0, 1.0),  # Normalize target
            float(self.current_chi) / float(self.chi_max),  # Chi ratio
        ], dtype=torch.float32)
        
        return features
    
    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        chi_min: int,
        chi_max: int,
        target_error: float,
    ) -> "PolicyState":
        """Reconstruct state from tensor (approximate)."""
        chi_normalized = float(tensor[0])
        current_chi = int(chi_min + chi_normalized * (chi_max - chi_min))
        
        return cls(
            current_chi=current_chi,
            truncation_error=10 ** (-float(tensor[1]) * 16),
            entropy=float(tensor[2]) * 10,
            entropy_gradient=float(np.arctanh(np.clip(tensor[3], -0.99, 0.99))),
            step=int(float(tensor[4]) * 1000),
            memory_usage=float(tensor[5]),
            target_error=target_error,
            chi_min=chi_min,
            chi_max=chi_max,
        )


class PolicyNetwork(nn.Module):
    """Neural network for truncation policy.
    
    Actor-critic architecture for policy gradient learning.
    
    Attributes:
        state_dim: Dimension of state features
        action_dim: Number of possible actions
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 5,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize policy network.
        
        Args:
            state_dim: Dimension of state features
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[PolicyAction, torch.Tensor, torch.Tensor]:
        """Get action from policy.
        
        Args:
            state: State tensor
            deterministic: If True, take argmax action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        action_logits, value = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action_idx = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
        
        log_prob = F.log_softmax(action_logits, dim=-1)
        action_log_prob = log_prob.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
        
        action = PolicyAction.from_index(int(action_idx.item()))
        
        return action, action_log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training.
        
        Args:
            states: Batch of state tensors
            actions: Batch of action indices
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        action_logits, values = self.forward(states)
        
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_probs, values.squeeze(-1), entropy


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: float
    value: float


@dataclass
class TruncationPolicy:
    """Learned truncation policy for adaptive bond dimension.
    
    Wraps a PolicyNetwork with configuration and provides
    a simple interface for getting truncation decisions.
    
    Attributes:
        network: The neural network
        chi_min: Minimum bond dimension
        chi_max: Maximum bond dimension
        target_error: Target truncation error
        device: Computation device
    """
    
    network: PolicyNetwork
    chi_min: int = 4
    chi_max: int = 512
    target_error: float = 1e-10
    device: str = "cpu"
    
    def __post_init__(self) -> None:
        """Move network to device."""
        self.network = self.network.to(self.device)
        self.network.eval()
    
    def get_chi(
        self,
        current_chi: int,
        truncation_error: float,
        entropy: float,
        entropy_gradient: float = 0.0,
        step: int = 0,
        memory_usage: float = 0.5,
        deterministic: bool = True,
    ) -> int:
        """Get recommended bond dimension.
        
        Args:
            current_chi: Current bond dimension
            truncation_error: Recent truncation error
            entropy: Current entanglement entropy
            entropy_gradient: Rate of entropy change
            step: Current simulation step
            memory_usage: Current memory usage fraction
            deterministic: If True, use deterministic policy
            
        Returns:
            Recommended bond dimension
        """
        state = PolicyState(
            current_chi=current_chi,
            truncation_error=truncation_error,
            entropy=entropy,
            entropy_gradient=entropy_gradient,
            step=step,
            memory_usage=memory_usage,
            target_error=self.target_error,
            chi_min=self.chi_min,
            chi_max=self.chi_max,
        )
        
        state_tensor = state.to_tensor().to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.network.get_action(
                state_tensor, deterministic=deterministic
            )
        
        return action.apply(current_chi, self.chi_min, self.chi_max)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save policy to file."""
        path = Path(path)
        torch.save({
            "network_state": self.network.state_dict(),
            "chi_min": self.chi_min,
            "chi_max": self.chi_max,
            "target_error": self.target_error,
            "state_dim": self.network.state_dim,
            "action_dim": self.network.action_dim,
            "hidden_dim": self.network.hidden_dim,
        }, path)
    
    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "TruncationPolicy":
        """Load policy from file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        
        network = PolicyNetwork(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        network.load_state_dict(checkpoint["network_state"])
        
        return cls(
            network=network,
            chi_min=checkpoint["chi_min"],
            chi_max=checkpoint["chi_max"],
            target_error=checkpoint["target_error"],
            device=device,
        )


class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 10000) -> None:
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer: deque = deque(maxlen=capacity)
    
    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()


@dataclass
class RLTruncationAgent:
    """Reinforcement learning agent for truncation policy.
    
    Uses PPO (Proximal Policy Optimization) for stable training.
    
    Attributes:
        policy: The truncation policy
        optimizer: Network optimizer
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Gradient clipping norm
    """
    
    policy: TruncationPolicy
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    optimizer: optim.Optimizer = field(init=False)
    buffer: ReplayBuffer = field(init=False)
    
    def __post_init__(self) -> None:
        """Initialize optimizer and buffer."""
        self.optimizer = optim.Adam(
            self.policy.network.parameters(),
            lr=self.learning_rate,
        )
        self.buffer = ReplayBuffer()
        self.policy.network.train()
    
    def compute_reward(
        self,
        truncation_error: float,
        target_error: float,
        chi: int,
        chi_max: int,
        memory_usage: float,
    ) -> float:
        """Compute reward for truncation decision.
        
        Reward balances accuracy (low error) vs efficiency (low chi).
        
        Args:
            truncation_error: Achieved truncation error
            target_error: Target truncation error
            chi: Current bond dimension
            chi_max: Maximum bond dimension
            memory_usage: Current memory usage
            
        Returns:
            Reward value
        """
        # Accuracy reward (positive if below target)
        if truncation_error <= target_error:
            accuracy_reward = 1.0 + math.log10(target_error / max(truncation_error, 1e-16))
        else:
            accuracy_reward = -math.log10(truncation_error / target_error)
        
        # Efficiency reward (prefer smaller chi)
        efficiency_reward = 1.0 - (chi / chi_max)
        
        # Memory penalty
        memory_penalty = -max(0, memory_usage - 0.8) * 10
        
        # Combined reward
        reward = 0.7 * accuracy_reward + 0.2 * efficiency_reward + 0.1 * memory_penalty
        
        return reward
    
    def store_experience(
        self,
        state: PolicyState,
        action: PolicyAction,
        reward: float,
        next_state: PolicyState,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store experience in replay buffer."""
        experience = Experience(
            state=state.to_tensor(),
            action=action.to_index(),
            reward=reward,
            next_state=next_state.to_tensor(),
            done=done,
            log_prob=log_prob,
            value=value,
        )
        self.buffer.push(experience)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of final next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0.0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self, batch_size: int = 64, epochs: int = 4) -> Dict[str, float]:
        """Update policy using collected experiences.
        
        Args:
            batch_size: Mini-batch size
            epochs: Number of PPO epochs
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Convert buffer to tensors
        experiences = list(self.buffer.buffer)
        
        states = torch.stack([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences])
        rewards = [e.reward for e in experiences]
        dones = [e.done for e in experiences]
        old_log_probs = torch.tensor([e.log_prob for e in experiences])
        values = [e.value for e in experiences]
        
        # Compute returns and advantages
        with torch.no_grad():
            _, next_value = self.policy.network(experiences[-1].next_state.unsqueeze(0))
            next_value = float(next_value.squeeze())
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        device = self.policy.device
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        for _ in range(epochs):
            # Random permutation for mini-batches
            indices = torch.randperm(len(experiences))
            
            for start in range(0, len(experiences), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.network.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.network.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Clear buffer after update
        self.buffer.clear()
        
        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }


def train_truncation_policy(
    environment_fn: Callable[[], Any],
    num_episodes: int = 1000,
    max_steps: int = 100,
    chi_min: int = 4,
    chi_max: int = 512,
    target_error: float = 1e-10,
    device: str = "cpu",
    callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> TruncationPolicy:
    """Train a truncation policy using reinforcement learning.
    
    Args:
        environment_fn: Factory function for training environment
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        chi_min: Minimum bond dimension
        chi_max: Maximum bond dimension
        target_error: Target truncation error
        device: Computation device
        callback: Optional callback(episode, metrics)
        
    Returns:
        Trained TruncationPolicy
    """
    # Initialize policy and agent
    network = PolicyNetwork()
    policy = TruncationPolicy(
        network=network,
        chi_min=chi_min,
        chi_max=chi_max,
        target_error=target_error,
        device=device,
    )
    agent = RLTruncationAgent(policy=policy)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        env = environment_fn()
        
        # Initial state
        current_chi = (chi_min + chi_max) // 2
        truncation_error = 1e-6
        entropy = 1.0
        entropy_gradient = 0.0
        memory_usage = 0.3
        
        state = PolicyState(
            current_chi=current_chi,
            truncation_error=truncation_error,
            entropy=entropy,
            entropy_gradient=entropy_gradient,
            step=0,
            memory_usage=memory_usage,
            target_error=target_error,
            chi_min=chi_min,
            chi_max=chi_max,
        )
        
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Get action from policy
            state_tensor = state.to_tensor().to(device)
            action, log_prob, value = agent.policy.network.get_action(state_tensor)
            
            # Apply action
            new_chi = action.apply(state.current_chi, chi_min, chi_max)
            
            # Simulate environment step (simplified)
            # In real use, this would interface with actual TN simulation
            new_truncation_error = truncation_error * (state.current_chi / max(new_chi, 1)) ** 2
            new_truncation_error = max(1e-16, min(1.0, new_truncation_error))
            new_entropy = entropy + 0.01 * random.gauss(0, 1)
            new_memory_usage = memory_usage * (new_chi / state.current_chi) ** 1.5
            new_memory_usage = min(1.0, max(0.0, new_memory_usage))
            
            # Compute reward
            reward = agent.compute_reward(
                new_truncation_error,
                target_error,
                new_chi,
                chi_max,
                new_memory_usage,
            )
            
            # Create next state
            next_state = PolicyState(
                current_chi=new_chi,
                truncation_error=new_truncation_error,
                entropy=new_entropy,
                entropy_gradient=new_entropy - entropy,
                step=step + 1,
                memory_usage=new_memory_usage,
                target_error=target_error,
                chi_min=chi_min,
                chi_max=chi_max,
            )
            
            done = step == max_steps - 1 or new_memory_usage > 0.99
            
            # Store experience
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=float(log_prob),
                value=float(value),
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update policy
        metrics = agent.update()
        metrics["episode_reward"] = episode_reward
        episode_rewards.append(episode_reward)
        
        if callback is not None:
            callback(episode, metrics)
    
    # Set to eval mode
    policy.network.eval()
    
    return policy


def load_truncation_policy(
    path: Union[str, Path],
    device: str = "cpu",
) -> TruncationPolicy:
    """Load a trained truncation policy.
    
    Args:
        path: Path to saved policy
        device: Computation device
        
    Returns:
        Loaded TruncationPolicy
    """
    return TruncationPolicy.load(path, device)


def create_default_policy(
    chi_min: int = 4,
    chi_max: int = 512,
    target_error: float = 1e-10,
    device: str = "cpu",
) -> TruncationPolicy:
    """Create a default (untrained) truncation policy.
    
    Args:
        chi_min: Minimum bond dimension
        chi_max: Maximum bond dimension
        target_error: Target truncation error
        device: Computation device
        
    Returns:
        Default TruncationPolicy
    """
    network = PolicyNetwork()
    return TruncationPolicy(
        network=network,
        chi_min=chi_min,
        chi_max=chi_max,
        target_error=target_error,
        device=device,
    )
