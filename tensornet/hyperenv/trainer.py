"""
Training Infrastructure
=======================

Training loops and utilities for RL agents.

Features:
- Flexible training loop
- Distributed training support
- Logging and checkpointing
- Resume from checkpoint
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable, Iterator
from pathlib import Path

from .agent import Agent, AgentConfig, AgentState
from .buffers import ReplayBuffer, RolloutBuffer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for trainer."""
    
    # Training
    total_timesteps: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 1000
    train_freq: int = 1
    gradient_steps: int = 1
    
    # Evaluation
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    
    # Logging
    log_freq: int = 1000
    verbose: int = 1
    
    # Checkpointing
    save_freq: int = 50_000
    save_path: Optional[str] = None
    
    # Environment
    n_envs: int = 1
    
    # Optimization
    max_grad_norm: float = 0.5
    
    # Seed
    seed: Optional[int] = None


@dataclass
class TrainingState:
    """State of training for resumption."""
    
    timestep: int = 0
    episode: int = 0
    
    # Best model
    best_mean_reward: float = float('-inf')
    best_model_path: Optional[str] = None
    
    # Statistics
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    
    # Timing
    start_time: float = 0.0
    total_train_time: float = 0.0
    
    # Learning
    updates: int = 0
    losses: List[float] = field(default_factory=list)
    
    def save(self, path: str):
        """Save training state."""
        import json
        data = {
            'timestep': self.timestep,
            'episode': self.episode,
            'best_mean_reward': self.best_mean_reward,
            'best_model_path': self.best_model_path,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'start_time': self.start_time,
            'total_train_time': self.total_train_time,
            'updates': self.updates,
            'losses': self.losses,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingState':
        """Load training state."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        state = cls()
        state.timestep = data.get('timestep', 0)
        state.episode = data.get('episode', 0)
        state.best_mean_reward = data.get('best_mean_reward', float('-inf'))
        state.best_model_path = data.get('best_model_path')
        state.episode_rewards = data.get('episode_rewards', [])
        state.episode_lengths = data.get('episode_lengths', [])
        state.start_time = data.get('start_time', 0.0)
        state.total_train_time = data.get('total_train_time', 0.0)
        state.updates = data.get('updates', 0)
        state.losses = data.get('losses', [])
        return state


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    Training loop for RL agents.
    
    Handles:
    - Environment interaction
    - Experience collection
    - Agent learning
    - Logging and checkpointing
    - Evaluation
    
    Example:
        trainer = Trainer(
            agent=my_agent,
            env=my_env,
            config=TrainerConfig(total_timesteps=1_000_000)
        )
        
        trainer.train()
    """
    
    def __init__(
        self,
        agent: Agent,
        env: Any,
        config: Optional[TrainerConfig] = None,
        eval_env: Optional[Any] = None,
        callbacks: Optional[List['Callback']] = None,
        buffer: Optional[Union[ReplayBuffer, RolloutBuffer]] = None,
    ):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env or env
        self.config = config or TrainerConfig()
        self.callbacks = callbacks or []
        
        # Initialize buffer
        if buffer is not None:
            self.buffer = buffer
        else:
            self.buffer = ReplayBuffer(capacity=100_000)
        
        # Training state
        self.state = TrainingState()
        self._should_stop = False
        
        # Seed
        if self.config.seed is not None:
            self._set_seed(self.config.seed)
    
    def _set_seed(self, seed: int):
        """Set random seeds."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
    
    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional['Callback'] = None,
        log_interval: Optional[int] = None,
        reset_num_timesteps: bool = True,
    ) -> 'Trainer':
        """
        Train the agent.
        
        Args:
            total_timesteps: Override config total timesteps
            callback: Additional callback
            log_interval: Override log frequency
            reset_num_timesteps: Reset timestep counter
            
        Returns:
            Self for chaining
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        log_interval = log_interval or self.config.log_freq
        
        # Setup
        if reset_num_timesteps:
            self.state = TrainingState()
        
        self.state.start_time = time.time()
        self._should_stop = False
        
        # Add extra callback if provided
        callbacks = list(self.callbacks)
        if callback is not None:
            callbacks.append(callback)
        
        # Notify callbacks
        self._on_training_start(callbacks)
        
        # Reset environment
        observation = self._reset_env()
        episode_reward = 0.0
        episode_length = 0
        
        # Training loop
        while self.state.timestep < total_timesteps and not self._should_stop:
            # Collect experience
            action = self._collect_action(observation)
            
            next_observation, reward, done, info = self._step_env(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Store in buffer
            self.buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
            
            # Update observation
            observation = next_observation
            self.state.timestep += 1
            
            # Handle episode end
            if done:
                self.state.episode += 1
                self.state.episode_rewards.append(episode_reward)
                self.state.episode_lengths.append(episode_length)
                
                # Callback
                self._on_episode_end(callbacks, episode_reward, episode_length)
                
                # Reset
                observation = self._reset_env()
                episode_reward = 0.0
                episode_length = 0
            
            # Learn
            if self.state.timestep >= self.config.learning_starts:
                if self.state.timestep % self.config.train_freq == 0:
                    self._learn(callbacks)
            
            # Log
            if self.state.timestep % log_interval == 0:
                self._log_progress()
            
            # Evaluate
            if self.state.timestep % self.config.eval_freq == 0:
                self._evaluate(callbacks)
            
            # Checkpoint
            if self.state.timestep % self.config.save_freq == 0:
                self._checkpoint(callbacks)
            
            # Callback
            self._on_step(callbacks)
        
        # Cleanup
        self.state.total_train_time = time.time() - self.state.start_time
        self._on_training_end(callbacks)
        
        return self
    
    def _reset_env(self) -> np.ndarray:
        """Reset environment and return observation."""
        result = self.env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def _step_env(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment."""
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            done = done or truncated
        else:
            obs, reward, done, info = result
        return obs, reward, done, info
    
    def _collect_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from agent."""
        return self.agent.act(observation, deterministic=False)
    
    def _learn(self, callbacks: List['Callback']):
        """Perform learning update."""
        for _ in range(self.config.gradient_steps):
            if len(self.buffer) >= self.config.batch_size:
                # Sample batch
                batch = self.buffer.sample(self.config.batch_size)
                
                # Learn (agent-specific)
                metrics = self.agent.learn()
                
                self.state.updates += 1
                if 'loss' in metrics:
                    self.state.losses.append(metrics['loss'])
    
    def _evaluate(self, callbacks: List['Callback']):
        """Evaluate current policy."""
        self.agent.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(self.config.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action = self.agent.act(obs, deterministic=True)
                result = self.eval_env.step(action)
                
                if len(result) == 5:
                    obs, reward, done, truncated, info = result
                    done = done or truncated
                else:
                    obs, reward, done, info = result
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # Update best
        if mean_reward > self.state.best_mean_reward:
            self.state.best_mean_reward = mean_reward
            if self.config.save_path:
                self.state.best_model_path = os.path.join(
                    self.config.save_path, "best_model"
                )
                self.agent.save(self.state.best_model_path)
        
        self.agent.train()
        
        # Callback
        for cb in callbacks:
            cb.on_evaluation(self, mean_reward, std_reward)
        
        if self.config.verbose >= 1:
            print(f"Eval: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    
    def _checkpoint(self, callbacks: List['Callback']):
        """Save checkpoint."""
        if self.config.save_path:
            path = os.path.join(
                self.config.save_path,
                f"checkpoint_{self.state.timestep}"
            )
            os.makedirs(path, exist_ok=True)
            
            self.agent.save(os.path.join(path, "agent.pkl"))
            self.state.save(os.path.join(path, "training_state.pkl"))
            
            for cb in callbacks:
                cb.on_checkpoint(self, path)
    
    def _log_progress(self):
        """Log training progress."""
        if self.config.verbose >= 1:
            elapsed = time.time() - self.state.start_time
            fps = self.state.timestep / elapsed if elapsed > 0 else 0
            
            recent_rewards = self.state.episode_rewards[-100:]
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0
            
            print(
                f"Step: {self.state.timestep} | "
                f"Episodes: {self.state.episode} | "
                f"FPS: {fps:.0f} | "
                f"Mean reward: {mean_reward:.2f}"
            )
    
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def _on_training_start(self, callbacks: List['Callback']):
        for cb in callbacks:
            cb.on_training_start(self)
    
    def _on_training_end(self, callbacks: List['Callback']):
        for cb in callbacks:
            cb.on_training_end(self)
    
    def _on_step(self, callbacks: List['Callback']):
        for cb in callbacks:
            if cb.on_step(self) is False:
                self._should_stop = True
    
    def _on_episode_end(
        self,
        callbacks: List['Callback'],
        episode_reward: float,
        episode_length: int,
    ):
        for cb in callbacks:
            cb.on_episode_end(self, episode_reward, episode_length)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def resume(self, checkpoint_path: str) -> 'Trainer':
        """Resume training from checkpoint."""
        agent_path = os.path.join(checkpoint_path, "agent.pkl")
        state_path = os.path.join(checkpoint_path, "training_state.pkl")
        
        self.agent.load(agent_path)
        self.state = TrainingState.load(state_path)
        
        return self
    
    def stop(self):
        """Stop training."""
        self._should_stop = True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "timesteps": self.state.timestep,
            "episodes": self.state.episode,
            "updates": self.state.updates,
            "best_reward": self.state.best_mean_reward,
            "mean_episode_reward": np.mean(self.state.episode_rewards[-100:]) if self.state.episode_rewards else 0,
            "mean_episode_length": np.mean(self.state.episode_lengths[-100:]) if self.state.episode_lengths else 0,
            "training_time": self.state.total_train_time,
        }


# =============================================================================
# CALLBACK (forward declaration)
# =============================================================================

class Callback(ABC):
    """Base callback class (full implementation in callbacks.py)."""
    
    def on_training_start(self, trainer: Trainer):
        pass
    
    def on_training_end(self, trainer: Trainer):
        pass
    
    def on_step(self, trainer: Trainer) -> Optional[bool]:
        pass
    
    def on_episode_end(
        self,
        trainer: Trainer,
        episode_reward: float,
        episode_length: int,
    ):
        pass
    
    def on_evaluation(
        self,
        trainer: Trainer,
        mean_reward: float,
        std_reward: float,
    ):
        pass
    
    def on_checkpoint(self, trainer: Trainer, path: str):
        pass


# =============================================================================
# SPECIALIZED TRAINERS
# =============================================================================

class OnPolicyTrainer(Trainer):
    """
    Trainer for on-policy algorithms (PPO, A2C, etc.).
    
    Uses rollout buffer instead of replay buffer.
    """
    
    def __init__(
        self,
        agent: Agent,
        env: Any,
        config: Optional[TrainerConfig] = None,
        n_steps: int = 2048,
        **kwargs,
    ):
        buffer = RolloutBuffer(capacity=n_steps)
        super().__init__(agent, env, config, buffer=buffer, **kwargs)
        self.n_steps = n_steps
    
    def _learn(self, callbacks: List[Callback]):
        """Learn from rollout buffer."""
        if len(self.buffer) >= self.n_steps:
            # Compute returns and advantages
            # (This would be algorithm-specific)
            
            # Learn
            metrics = self.agent.learn()
            self.state.updates += 1
            
            # Clear buffer for next rollout
            self.buffer.reset()


class OffPolicyTrainer(Trainer):
    """
    Trainer for off-policy algorithms (DQN, SAC, TD3, etc.).
    
    Uses replay buffer with prioritized experience replay support.
    """
    
    def __init__(
        self,
        agent: Agent,
        env: Any,
        config: Optional[TrainerConfig] = None,
        buffer_size: int = 1_000_000,
        prioritized: bool = False,
        **kwargs,
    ):
        from .buffers import PrioritizedReplayBuffer
        
        if prioritized:
            buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            buffer = ReplayBuffer(capacity=buffer_size)
        
        super().__init__(agent, env, config, buffer=buffer, **kwargs)


# =============================================================================
# FACTORY
# =============================================================================

def make_trainer(
    agent: Agent,
    env: Any,
    trainer_type: str = "offpolicy",
    **kwargs,
) -> Trainer:
    """
    Factory for creating trainers.
    
    Args:
        agent: Agent to train
        env: Environment
        trainer_type: 'offpolicy', 'onpolicy', 'base'
        
    Returns:
        Trainer instance
    """
    if trainer_type == "offpolicy":
        return OffPolicyTrainer(agent, env, **kwargs)
    elif trainer_type == "onpolicy":
        return OnPolicyTrainer(agent, env, **kwargs)
    else:
        return Trainer(agent, env, **kwargs)
