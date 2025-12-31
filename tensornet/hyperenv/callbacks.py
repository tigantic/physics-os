"""
Training Callbacks
==================

Modular callbacks for customizing training behavior.

Features:
- Checkpointing
- Evaluation
- Logging
- Early stopping
- Custom hooks
"""

from __future__ import annotations

import os
import time
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer


# =============================================================================
# BASE CALLBACK
# =============================================================================

class Callback(ABC):
    """
    Base callback for training hooks.
    
    Override methods to customize training behavior.
    
    Example:
        class MyCallback(Callback):
            def on_step(self, trainer):
                if trainer.state.timestep % 100 == 0:
                    print(f"Step {trainer.state.timestep}")
    """
    
    def on_training_start(self, trainer: 'Trainer'):
        """Called when training starts."""
        pass
    
    def on_training_end(self, trainer: 'Trainer'):
        """Called when training ends."""
        pass
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        """
        Called after each environment step.
        
        Return False to stop training.
        """
        pass
    
    def on_episode_end(
        self,
        trainer: 'Trainer',
        episode_reward: float,
        episode_length: int,
    ):
        """Called at end of each episode."""
        pass
    
    def on_evaluation(
        self,
        trainer: 'Trainer',
        mean_reward: float,
        std_reward: float,
    ):
        """Called after evaluation."""
        pass
    
    def on_checkpoint(self, trainer: 'Trainer', path: str):
        """Called after saving checkpoint."""
        pass


# =============================================================================
# CALLBACK LIST
# =============================================================================

class CallbackList(Callback):
    """
    Composite callback that wraps multiple callbacks.
    
    Example:
        callbacks = CallbackList([
            CheckpointCallback(save_freq=1000),
            EvalCallback(eval_freq=5000),
            LoggingCallback(),
        ])
        
        trainer = Trainer(agent, env, callbacks=[callbacks])
    """
    
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
    
    # L-024 NOTE: Callback dispatch loops - typically <10 callbacks total (offline RL)
    def on_training_start(self, trainer: 'Trainer'):
        for cb in self.callbacks:
            cb.on_training_start(trainer)
    
    def on_training_end(self, trainer: 'Trainer'):
        for cb in self.callbacks:
            cb.on_training_end(trainer)
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        for cb in self.callbacks:
            if cb.on_step(trainer) is False:
                return False
        return None
    
    def on_episode_end(
        self,
        trainer: 'Trainer',
        episode_reward: float,
        episode_length: int,
    ):
        for cb in self.callbacks:
            cb.on_episode_end(trainer, episode_reward, episode_length)
    
    def on_evaluation(
        self,
        trainer: 'Trainer',
        mean_reward: float,
        std_reward: float,
    ):
        for cb in self.callbacks:
            cb.on_evaluation(trainer, mean_reward, std_reward)
    
    def on_checkpoint(self, trainer: 'Trainer', path: str):
        for cb in self.callbacks:
            cb.on_checkpoint(trainer, path)


# =============================================================================
# CHECKPOINT CALLBACK
# =============================================================================

class CheckpointCallback(Callback):
    """
    Save model checkpoints during training.
    
    Saves:
    - Agent state
    - Training state
    - Buffer (optional)
    
    Example:
        callback = CheckpointCallback(
            save_freq=10000,
            save_path="./checkpoints",
            save_buffer=True,
        )
    """
    
    def __init__(
        self,
        save_freq: int = 10_000,
        save_path: str = "./checkpoints",
        name_prefix: str = "checkpoint",
        save_buffer: bool = False,
        save_best_only: bool = False,
        verbose: int = 1,
    ):
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_buffer = save_buffer
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self._best_reward = float('-inf')
    
    def on_training_start(self, trainer: 'Trainer'):
        os.makedirs(self.save_path, exist_ok=True)
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        if trainer.state.timestep % self.save_freq == 0:
            self._save(trainer)
    
    def on_evaluation(
        self,
        trainer: 'Trainer',
        mean_reward: float,
        std_reward: float,
    ):
        if self.save_best_only:
            if mean_reward > self._best_reward:
                self._best_reward = mean_reward
                self._save(trainer, prefix="best")
    
    def _save(self, trainer: 'Trainer', prefix: Optional[str] = None):
        """Save checkpoint."""
        prefix = prefix or self.name_prefix
        path = os.path.join(
            self.save_path,
            f"{prefix}_{trainer.state.timestep}"
        )
        os.makedirs(path, exist_ok=True)
        
        # Save agent
        trainer.agent.save(os.path.join(path, "agent.pkl"))
        
        # Save training state
        trainer.state.save(os.path.join(path, "training_state.pkl"))
        
        # Save buffer if requested (use torch.save for tensor-safe serialization)
        if self.save_buffer and hasattr(trainer, 'buffer'):
            torch.save(trainer.buffer, os.path.join(path, "buffer.pt"))
        
        if self.verbose >= 1:
            print(f"Saved checkpoint to {path}")


# =============================================================================
# EVALUATION CALLBACK
# =============================================================================

class EvalCallback(Callback):
    """
    Evaluate policy during training.
    
    Example:
        callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=5000,
            n_eval_episodes=10,
            best_model_save_path="./best_model",
        )
    """
    
    def __init__(
        self,
        eval_env: Any = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        verbose: int = 1,
    ):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.verbose = verbose
        
        self._best_mean_reward = float('-inf')
        self._evaluations: List[Dict] = []
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        if trainer.state.timestep % self.eval_freq == 0:
            self._evaluate(trainer)
    
    def on_training_end(self, trainer: 'Trainer'):
        if self.log_path:
            self._save_evaluations()
    
    def _evaluate(self, trainer: 'Trainer'):
        """Run evaluation."""
        env = self.eval_env or trainer.env
        agent = trainer.agent
        
        agent.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        # L-024 NOTE: Offline RL evaluation loop - configurable n_eval_episodes
        for _ in range(self.n_eval_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action = agent.act(obs, deterministic=self.deterministic)
                result = env.step(action)
                
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
        
        # Log evaluation
        self._evaluations.append({
            "timestep": trainer.state.timestep,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
        })
        
        # Save best model
        if mean_reward > self._best_mean_reward:
            self._best_mean_reward = mean_reward
            if self.best_model_save_path:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                agent.save(os.path.join(self.best_model_save_path, "best_model.pkl"))
                if self.verbose >= 1:
                    print(f"New best model: {mean_reward:.2f}")
        
        if self.verbose >= 1:
            print(f"Eval @ {trainer.state.timestep}: "
                  f"{mean_reward:.2f} +/- {std_reward:.2f}")
        
        agent.train()
    
    def _save_evaluations(self):
        """Save evaluation log."""
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w') as f:
                json.dump(self._evaluations, f, indent=2)


# =============================================================================
# LOGGING CALLBACK
# =============================================================================

class LoggingCallback(Callback):
    """
    Log training metrics.
    
    Supports:
    - Console logging
    - JSON file logging
    - TensorBoard (optional)
    
    Example:
        callback = LoggingCallback(
            log_freq=100,
            log_path="./logs/training.json",
        )
    """
    
    def __init__(
        self,
        log_freq: int = 1000,
        log_path: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
    ):
        self.log_freq = log_freq
        self.log_path = log_path
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        
        self._logs: List[Dict] = []
        self._writer = None
        self._start_time = 0.0
    
    def on_training_start(self, trainer: 'Trainer'):
        self._start_time = time.time()
        
        # Initialize tensorboard if requested
        if self.tensorboard_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(self.tensorboard_log)
            except ImportError:
                print("TensorBoard not available")
    
    def on_training_end(self, trainer: 'Trainer'):
        if self._writer:
            self._writer.close()
        
        if self.log_path:
            self._save_logs()
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        if trainer.state.timestep % self.log_freq == 0:
            self._log(trainer)
    
    def _log(self, trainer: 'Trainer'):
        """Log metrics."""
        elapsed = time.time() - self._start_time
        fps = trainer.state.timestep / elapsed if elapsed > 0 else 0
        
        recent_rewards = trainer.state.episode_rewards[-100:]
        mean_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        log_entry = {
            "timestep": trainer.state.timestep,
            "episode": trainer.state.episode,
            "mean_reward": float(mean_reward),
            "fps": fps,
            "elapsed": elapsed,
        }
        
        self._logs.append(log_entry)
        
        # Console
        if self.verbose >= 1:
            print(f"Step: {trainer.state.timestep:>8} | "
                  f"Episode: {trainer.state.episode:>5} | "
                  f"Mean Reward: {mean_reward:>8.2f} | "
                  f"FPS: {fps:>6.0f}")
        
        # TensorBoard
        if self._writer:
            self._writer.add_scalar("train/mean_reward", mean_reward, trainer.state.timestep)
            self._writer.add_scalar("train/episode", trainer.state.episode, trainer.state.timestep)
            self._writer.add_scalar("train/fps", fps, trainer.state.timestep)
    
    def _save_logs(self):
        """Save logs to file."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self._logs, f, indent=2)


# =============================================================================
# EARLY STOPPING CALLBACK
# =============================================================================

class EarlyStoppingCallback(Callback):
    """
    Stop training when performance stops improving.
    
    Example:
        callback = EarlyStoppingCallback(
            patience=10,
            min_delta=0.01,
            check_freq=10000,
        )
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        check_freq: int = 10_000,
        verbose: int = 1,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.verbose = verbose
        
        self._best_reward = float('-inf')
        self._patience_counter = 0
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        if trainer.state.timestep % self.check_freq == 0:
            return self._check(trainer)
    
    def _check(self, trainer: 'Trainer') -> Optional[bool]:
        """Check for early stopping."""
        recent_rewards = trainer.state.episode_rewards[-100:]
        if not recent_rewards:
            return None
        
        mean_reward = np.mean(recent_rewards)
        
        if mean_reward > self._best_reward + self.min_delta:
            self._best_reward = mean_reward
            self._patience_counter = 0
        else:
            self._patience_counter += 1
            
            if self.verbose >= 1:
                print(f"No improvement for {self._patience_counter} checks")
            
            if self._patience_counter >= self.patience:
                if self.verbose >= 1:
                    print(f"Early stopping at step {trainer.state.timestep}")
                return False
        
        return None


# =============================================================================
# PROGRESS CALLBACK
# =============================================================================

class ProgressCallback(Callback):
    """
    Display progress bar during training.
    
    Requires tqdm.
    """
    
    def __init__(self, total_timesteps: int):
        self.total_timesteps = total_timesteps
        self._pbar = None
    
    def on_training_start(self, trainer: 'Trainer'):
        try:
            from tqdm import tqdm
            self._pbar = tqdm(total=self.total_timesteps, desc="Training")
        except ImportError:
            print("tqdm not available for progress bar")
    
    def on_training_end(self, trainer: 'Trainer'):
        if self._pbar:
            self._pbar.close()
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        if self._pbar:
            self._pbar.update(1)
            
            # Update description
            recent = trainer.state.episode_rewards[-10:]
            if recent:
                mean_r = np.mean(recent)
                self._pbar.set_postfix({"reward": f"{mean_r:.2f}"})


# =============================================================================
# CUSTOM CALLBACK
# =============================================================================

class LambdaCallback(Callback):
    """
    Callback with lambda functions for quick customization.
    
    Example:
        callback = LambdaCallback(
            on_step=lambda t: print(f"Step {t.state.timestep}") if t.state.timestep % 1000 == 0 else None
        )
    """
    
    def __init__(
        self,
        on_training_start: Optional[Callable[['Trainer'], None]] = None,
        on_training_end: Optional[Callable[['Trainer'], None]] = None,
        on_step: Optional[Callable[['Trainer'], Optional[bool]]] = None,
        on_episode_end: Optional[Callable[['Trainer', float, int], None]] = None,
        on_evaluation: Optional[Callable[['Trainer', float, float], None]] = None,
    ):
        self._on_training_start = on_training_start
        self._on_training_end = on_training_end
        self._on_step = on_step
        self._on_episode_end = on_episode_end
        self._on_evaluation = on_evaluation
    
    def on_training_start(self, trainer: 'Trainer'):
        if self._on_training_start:
            self._on_training_start(trainer)
    
    def on_training_end(self, trainer: 'Trainer'):
        if self._on_training_end:
            self._on_training_end(trainer)
    
    def on_step(self, trainer: 'Trainer') -> Optional[bool]:
        if self._on_step:
            return self._on_step(trainer)
    
    def on_episode_end(
        self,
        trainer: 'Trainer',
        episode_reward: float,
        episode_length: int,
    ):
        if self._on_episode_end:
            self._on_episode_end(trainer, episode_reward, episode_length)
    
    def on_evaluation(
        self,
        trainer: 'Trainer',
        mean_reward: float,
        std_reward: float,
    ):
        if self._on_evaluation:
            self._on_evaluation(trainer, mean_reward, std_reward)
