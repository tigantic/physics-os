"""
Agent Interface
================

Base agent classes for interacting with environments.

Agents encapsulate:
- Observation processing
- Action selection
- State management
- Checkpointing
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str = "agent"
    observation_shape: Tuple[int, ...] = (64, 64, 3)
    action_dim: int = 4
    device: str = "cpu"
    
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    
    # Exploration
    exploration_fraction: float = 0.1
    exploration_initial: float = 1.0
    exploration_final: float = 0.05
    
    # Network
    hidden_sizes: Tuple[int, ...] = (256, 256)
    activation: str = "relu"


@dataclass
class AgentState:
    """
    Serializable agent state for checkpointing.
    
    Includes:
    - Policy parameters
    - Optimizer state
    - Training progress
    - Exploration state
    """
    step: int = 0
    episode: int = 0
    
    # Exploration
    epsilon: float = 1.0
    
    # Statistics
    total_reward: float = 0.0
    best_reward: float = float('-inf')
    
    # Network weights (as dict of numpy arrays)
    policy_state: Optional[Dict[str, np.ndarray]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    
    def save(self, path: str):
        """Save state to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'AgentState':
        """Load state from file."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


# =============================================================================
# BASE AGENT
# =============================================================================

class Agent(ABC):
    """
    Base agent interface.
    
    Agents are policies that can:
    - Select actions given observations
    - Learn from experience
    - Save/load checkpoints
    
    Example:
        agent = MyAgent(config)
        
        for episode in range(1000):
            obs = env.reset()
            done = False
            
            while not done:
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                agent.observe(obs, action, reward, next_obs, done)
                obs = next_obs
            
            agent.end_episode()
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.state = AgentState()
        self._training = True
    
    @property
    def name(self) -> str:
        """Agent name."""
        return self.config.name
    
    @property
    def training(self) -> bool:
        """Whether agent is in training mode."""
        return self._training
    
    def train(self):
        """Set agent to training mode."""
        self._training = True
    
    def eval(self):
        """Set agent to evaluation mode."""
        self._training = False
    
    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            observation: Current observation
            deterministic: If True, select best action without exploration
            
        Returns:
            Selected action
        """
        pass
    
    def observe(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        """
        Observe transition for learning.
        
        Default implementation does nothing.
        Override in learning agents.
        """
        pass
    
    def end_episode(self, episode_reward: float = 0.0):
        """
        Called at end of episode.
        
        Default implementation updates statistics.
        """
        self.state.episode += 1
        self.state.total_reward += episode_reward
        if episode_reward > self.state.best_reward:
            self.state.best_reward = episode_reward
    
    def learn(self) -> Dict[str, float]:
        """
        Perform learning update.
        
        Returns:
            Dictionary of training metrics
        """
        return {}
    
    def get_state(self) -> AgentState:
        """Get agent state for checkpointing."""
        return self.state
    
    def set_state(self, state: AgentState):
        """Restore agent from checkpoint."""
        self.state = state
    
    def save(self, path: str):
        """Save agent to file."""
        self.state.save(path)
    
    def load(self, path: str):
        """Load agent from file."""
        self.state = AgentState.load(path)
    
    def _get_epsilon(self) -> float:
        """Get current exploration epsilon."""
        cfg = self.config
        progress = min(1.0, self.state.step / (cfg.exploration_fraction * 1_000_000))
        return cfg.exploration_initial + progress * (cfg.exploration_final - cfg.exploration_initial)


# =============================================================================
# SIMPLE AGENTS
# =============================================================================

class RandomAgent(Agent):
    """
    Agent that takes random actions.
    
    Useful for:
    - Baseline comparison
    - Initial data collection
    - Testing environments
    """
    
    def __init__(
        self,
        action_space: Any = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(config)
        self.action_space = action_space
    
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        if self.action_space is not None:
            return self.action_space.sample()
        else:
            # Default: random continuous action
            return np.random.uniform(-1, 1, size=(self.config.action_dim,)).astype(np.float32)


class ConstantAgent(Agent):
    """
    Agent that always takes the same action.
    
    Useful for testing and debugging.
    """
    
    def __init__(
        self,
        action: np.ndarray,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(config)
        self.constant_action = action
    
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        return self.constant_action


class ScriptedAgent(Agent):
    """
    Agent that follows a scripted policy.
    
    Useful for:
    - Demonstration data collection
    - Expert policies
    - Testing specific action sequences
    """
    
    def __init__(
        self,
        script: Callable[[np.ndarray, int], np.ndarray],
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(config)
        self.script = script
    
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        return self.script(observation, self.state.step)


# =============================================================================
# POLICY WRAPPERS
# =============================================================================

class EpsilonGreedyAgent(Agent):
    """
    Wraps an agent with epsilon-greedy exploration.
    """
    
    def __init__(
        self,
        base_agent: Agent,
        action_space: Any = None,
        epsilon: float = 0.1,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(config or base_agent.config)
        self.base_agent = base_agent
        self.action_space = action_space
        self._epsilon = epsilon
    
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        if not deterministic and self.training and np.random.random() < self._get_epsilon():
            if self.action_space is not None:
                return self.action_space.sample()
            return np.random.uniform(-1, 1, size=(self.config.action_dim,)).astype(np.float32)
        
        return self.base_agent.act(observation, deterministic=True)
    
    def observe(self, *args, **kwargs):
        self.base_agent.observe(*args, **kwargs)
    
    def learn(self) -> Dict[str, float]:
        return self.base_agent.learn()


class GaussianNoiseAgent(Agent):
    """
    Wraps an agent with Gaussian action noise.
    
    Useful for continuous action spaces.
    """
    
    def __init__(
        self,
        base_agent: Agent,
        noise_scale: float = 0.1,
        noise_decay: float = 0.9999,
        min_noise: float = 0.01,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(config or base_agent.config)
        self.base_agent = base_agent
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self._current_noise = noise_scale
    
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        action = self.base_agent.act(observation, deterministic=True)
        
        if not deterministic and self.training:
            noise = np.random.normal(0, self._current_noise, size=action.shape)
            action = action + noise
            self._current_noise = max(self.min_noise, 
                                       self._current_noise * self.noise_decay)
        
        return action.astype(np.float32)
    
    def observe(self, *args, **kwargs):
        self.base_agent.observe(*args, **kwargs)
    
    def learn(self) -> Dict[str, float]:
        return self.base_agent.learn()


# =============================================================================
# FACTORY
# =============================================================================

def make_agent(
    agent_type: str,
    **kwargs,
) -> Agent:
    """
    Factory for creating agents.
    
    Args:
        agent_type: 'random', 'constant', 'scripted', etc.
        **kwargs: Agent-specific arguments
        
    Returns:
        Agent instance
    """
    if agent_type == 'random':
        return RandomAgent(**kwargs)
    elif agent_type == 'constant':
        action = kwargs.pop('action', np.zeros(4))
        return ConstantAgent(action=action, **kwargs)
    elif agent_type == 'scripted':
        script = kwargs.pop('script')
        return ScriptedAgent(script=script, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
