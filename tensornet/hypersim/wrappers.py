"""
Environment Wrappers
====================

Standard Gymnasium wrappers for FluidEnv customization.

Wrappers:
    - FrameStack: Stack consecutive observations
    - ActionRepeat: Repeat actions multiple times
    - RewardScaling: Scale/clip rewards
    - TimeLimit: Truncate after max steps
    - RecordEpisode: Record trajectories for replay
    - NormalizeObservation: Running normalization
"""

from __future__ import annotations

import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any, SupportsFloat

# Make gymnasium optional
try:
    import gymnasium
    from gymnasium import Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    # Create stub classes
    class Wrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = getattr(env, 'observation_space', None)
            self.action_space = getattr(env, 'action_space', None)
        
        def reset(self, **kwargs):
            return self.env.reset(**kwargs)
        
        def step(self, action):
            return self.env.step(action)
    
    class ObservationWrapper(Wrapper):
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info
        
        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(action)
            return self.observation(obs), reward, term, trunc, info
        
        def observation(self, observation):
            return observation
    
    class ActionWrapper(Wrapper):
        def step(self, action):
            return self.env.step(self.action(action))
        
        def action(self, action):
            return action
    
    class RewardWrapper(Wrapper):
        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(action)
            return obs, self.reward(reward), term, trunc, info
        
        def reward(self, reward):
            return reward


# Helper to create observation space
def _make_box_space(shape, low=-np.inf, high=np.inf, dtype=np.float32):
    """Create a box observation/action space."""
    if HAS_GYMNASIUM:
        return gymnasium.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    else:
        class BoxSpace:
            def __init__(self):
                self.shape = shape
                self.low = low
                self.high = high
                self.dtype = dtype
            
            def sample(self):
                return np.random.randn(*shape).astype(dtype)
        return BoxSpace()


# =============================================================================
# OBSERVATION WRAPPERS
# =============================================================================

class FrameStack(ObservationWrapper):
    """
    Stack consecutive observations for temporal context.
    
    Useful for:
        - Inferring velocity from position frames
        - Providing temporal context for recurrence-free policies
    
    Example:
        env = FrameStack(env, num_frames=4)
        # obs.shape: (4 * channels, H, W)
    """
    
    def __init__(self, env, num_frames: int = 4):
        super().__init__(env)
        self.num_frames = num_frames
        self._frames = deque(maxlen=num_frames)
        
        # Update observation space
        old_shape = env.observation_space.shape
        new_shape = (old_shape[0] * num_frames,) + old_shape[1:]
        
        self.observation_space = _make_box_space(new_shape)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Fill buffer with initial observation
        for _ in range(self.num_frames):
            self._frames.append(obs)
        
        return self._get_observation(), info
    
    def observation(self, observation):
        self._frames.append(observation)
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)


class NormalizeObservation(ObservationWrapper):
    """
    Normalize observations using running mean and std.
    
    obs_normalized = (obs - running_mean) / running_std
    """
    
    def __init__(
        self,
        env,
        epsilon: float = 1e-8,
        clip: float = 10.0,
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        
        shape = env.observation_space.shape
        self._mean = np.zeros(shape, dtype=np.float64)
        self._var = np.ones(shape, dtype=np.float64)
        self._count = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def observation(self, observation):
        self._update_stats(observation)
        return self._normalize(observation)
    
    def _update_stats(self, obs: np.ndarray):
        """Update running mean and variance."""
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += (delta * delta2 - self._var) / self._count
    
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(self._var + self.epsilon)
        normalized = (obs - self._mean) / std
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)


class DownsampleObservation(ObservationWrapper):
    """
    Downsample spatial dimensions of observation.
    
    Reduces computational cost for large fields.
    """
    
    def __init__(
        self,
        env,
        factor: int = 2,
        method: str = 'average',  # 'average', 'max', 'nearest'
    ):
        super().__init__(env)
        self.factor = factor
        self.method = method
        
        old_shape = env.observation_space.shape
        new_shape = (old_shape[0],) + tuple(s // factor for s in old_shape[1:])
        
        self.observation_space = _make_box_space(
            low=-np.inf,
            high=np.inf,
            shape=new_shape,
            dtype=np.float32,
        )
    
    def observation(self, observation):
        if self.method == 'average':
            return self._average_pool(observation)
        elif self.method == 'max':
            return self._max_pool(observation)
        else:
            return observation[..., ::self.factor, ::self.factor]
    
    def _average_pool(self, obs: np.ndarray) -> np.ndarray:
        """Average pooling."""
        c, h, w = obs.shape[-3:]
        new_h, new_w = h // self.factor, w // self.factor
        
        # Reshape for pooling
        obs_reshaped = obs.reshape(
            c, new_h, self.factor, new_w, self.factor
        )
        return obs_reshaped.mean(axis=(2, 4)).astype(np.float32)
    
    def _max_pool(self, obs: np.ndarray) -> np.ndarray:
        """Max pooling."""
        c, h, w = obs.shape[-3:]
        new_h, new_w = h // self.factor, w // self.factor
        
        obs_reshaped = obs.reshape(
            c, new_h, self.factor, new_w, self.factor
        )
        return obs_reshaped.max(axis=(2, 4)).astype(np.float32)


# =============================================================================
# ACTION WRAPPERS
# =============================================================================

class ActionRepeat(Wrapper):
    """
    Repeat each action multiple times.
    
    Useful for:
        - Reducing effective decision frequency
        - Simulating slower control systems
    
    Example:
        env = ActionRepeat(env, repeat=4)
        # Each step executes 4 physics steps with same action
    """
    
    def __init__(self, env, repeat: int = 4):
        super().__init__(env)
        self.repeat = repeat
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info


class ClipAction(ActionWrapper):
    """
    Clip actions to valid range.
    
    Ensures actions stay within bounds even with
    exploration noise.
    """
    
    def __init__(self, env):
        super().__init__(env)
        assert hasattr(env.action_space, "low")
        self._low = env.action_space.low
        self._high = env.action_space.high
    
    def action(self, action):
        return np.clip(action, self._low, self._high)


class RescaleAction(ActionWrapper):
    """
    Rescale actions from [-1, 1] to environment bounds.
    
    Convenient for policies that output normalized actions.
    """
    
    def __init__(
        self,
        env,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__(env)
        assert hasattr(env.action_space, "low")
        
        self.min_action = min_action
        self.max_action = max_action
        self._low = env.action_space.low
        self._high = env.action_space.high
    
    def action(self, action):
        # Map [-1, 1] to [low, high]
        action = np.clip(action, self.min_action, self.max_action)
        normalized = (action - self.min_action) / (self.max_action - self.min_action)
        return self._low + normalized * (self._high - self._low)


class StickyAction(Wrapper):
    """
    Sticky actions with probability.
    
    With probability p, repeat previous action instead
    of executing new one. Tests robustness to action delay.
    """
    
    def __init__(self, env, sticky_prob: float = 0.1):
        super().__init__(env)
        self.sticky_prob = sticky_prob
        self._last_action = None
    
    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs)
    
    def step(self, action):
        if self._last_action is not None and np.random.random() < self.sticky_prob:
            action = self._last_action
        
        self._last_action = action
        return self.env.step(action)


# =============================================================================
# REWARD WRAPPERS
# =============================================================================

class RewardScaling(RewardWrapper):
    """
    Scale rewards by a constant factor.
    
    Useful for matching reward magnitudes across tasks.
    """
    
    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale
    
    def reward(self, reward):
        return reward * self.scale


class RewardClipping(RewardWrapper):
    """
    Clip rewards to a range.
    
    Prevents extreme rewards from destabilizing learning.
    """
    
    def __init__(
        self,
        env,
        min_reward: float = -10.0,
        max_reward: float = 10.0,
    ):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


class RewardNormalization(RewardWrapper):
    """
    Normalize rewards using running statistics.
    
    reward_normalized = reward / running_std
    """
    
    def __init__(
        self,
        env,
        epsilon: float = 1e-8,
        gamma: float = 0.99,
    ):
        super().__init__(env)
        self.epsilon = epsilon
        self.gamma = gamma
        
        self._return = 0.0
        self._return_var = 1.0
        self._count = 0
    
    def reset(self, **kwargs):
        self._return = 0.0
        return self.env.reset(**kwargs)
    
    def reward(self, reward):
        # Update return estimate
        self._return = self._return * self.gamma + reward
        
        # Update variance (Welford's online algorithm)
        self._count += 1
        delta = self._return - 0  # Assume zero mean
        self._return_var += (delta ** 2 - self._return_var) / self._count
        
        # Normalize by std
        std = np.sqrt(self._return_var + self.epsilon)
        return reward / std


# =============================================================================
# TIME/EPISODE WRAPPERS
# =============================================================================

class TimeLimit(Wrapper):
    """
    Truncate episode after maximum steps.
    
    Standard Gymnasium TimeLimit wrapper.
    """
    
    def __init__(self, env, max_steps: int = 200):
        super().__init__(env)
        self.max_steps = max_steps
        self._elapsed_steps = 0
    
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._elapsed_steps += 1
        
        if self._elapsed_steps >= self.max_steps:
            truncated = True
            info['TimeLimit.truncated'] = True
        
        return obs, reward, terminated, truncated, info


class AutoReset(Wrapper):
    """
    Automatically reset environment on termination.
    
    Returns first observation of new episode instead
    of terminal observation.
    """
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            info['terminal_observation'] = obs
            obs, reset_info = self.env.reset()
            info.update(reset_info)
        
        return obs, reward, terminated, truncated, info


# =============================================================================
# RECORDING/LOGGING WRAPPERS
# =============================================================================

@dataclass
class EpisodeRecord:
    """Record of a single episode."""
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    total_reward: float = 0.0
    length: int = 0


class RecordEpisode(Wrapper):
    """
    Record episode trajectories for replay/analysis.
    
    Stores observations, actions, rewards, and info dicts
    for later review or imitation learning.
    """
    
    def __init__(
        self,
        env,
        buffer_size: int = 100,
        record_video: bool = False,
    ):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.record_video = record_video
        
        self._episodes: deque = deque(maxlen=buffer_size)
        self._current_episode: Optional[EpisodeRecord] = None
    
    def reset(self, **kwargs):
        # Finalize previous episode
        if self._current_episode is not None:
            self._episodes.append(self._current_episode)
        
        obs, info = self.env.reset(**kwargs)
        
        # Start new episode
        self._current_episode = EpisodeRecord(
            observations=[obs.copy()],
            actions=[],
            rewards=[],
            infos=[info.copy()],
        )
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self._current_episode is not None:
            self._current_episode.observations.append(obs.copy())
            self._current_episode.actions.append(np.array(action))
            self._current_episode.rewards.append(reward)
            self._current_episode.infos.append(info.copy())
            self._current_episode.total_reward += reward
            self._current_episode.length += 1
        
        return obs, reward, terminated, truncated, info
    
    @property
    def episodes(self) -> List[EpisodeRecord]:
        """Get recorded episodes."""
        return list(self._episodes)
    
    def get_latest(self, n: int = 10) -> List[EpisodeRecord]:
        """Get latest n episodes."""
        return list(self._episodes)[-n:]
    
    def clear(self):
        """Clear episode buffer."""
        self._episodes.clear()
    
    def save(self, path: str):
        """Save episodes to file."""
        import json
        episodes_data = [
            {
                'observations': [obs.tolist() if hasattr(obs, 'tolist') else obs for obs in ep.observations],
                'actions': [act.tolist() if hasattr(act, 'tolist') else act for act in ep.actions],
                'rewards': ep.rewards,
                'total_reward': ep.total_reward,
                'length': ep.length,
                'info': ep.info,
            }
            for ep in self._episodes
        ]
        with open(path, 'w') as f:
            json.dump(episodes_data, f)
    
    def load(self, path: str):
        """Load episodes from file."""
        import json
        import numpy as np
        with open(path, 'r') as f:
            episodes_data = json.load(f)
        episodes = [
            EpisodeRecord(
                observations=[np.array(obs) for obs in ep['observations']],
                actions=[np.array(act) if isinstance(act, list) else act for act in ep['actions']],
                rewards=ep['rewards'],
                total_reward=ep['total_reward'],
                length=ep['length'],
                info=ep.get('info', {}),
            )
            for ep in episodes_data
        ]
        self._episodes = deque(episodes, maxlen=self.buffer_size)


class LogMetrics(Wrapper):
    """
    Log episode metrics for monitoring.
    
    Tracks rewards, lengths, success rates, and custom metrics.
    """
    
    def __init__(
        self,
        env,
        log_frequency: int = 100,
        logger: Optional[Any] = None,  # tensorboard, wandb, etc.
    ):
        super().__init__(env)
        self.log_frequency = log_frequency
        self.logger = logger
        
        self._episode_count = 0
        self._step_count = 0
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        
        self._current_reward = 0.0
        self._current_length = 0
    
    def reset(self, **kwargs):
        # Log previous episode
        if self._current_length > 0:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._current_length)
            self._episode_count += 1
            
            if self._episode_count % self.log_frequency == 0:
                self._log_metrics()
        
        # Reset counters
        self._current_reward = 0.0
        self._current_length = 0
        
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._current_reward += reward
        self._current_length += 1
        self._step_count += 1
        
        return obs, reward, terminated, truncated, info
    
    def _log_metrics(self):
        """Log aggregated metrics."""
        recent_rewards = self._episode_rewards[-100:]
        recent_lengths = self._episode_lengths[-100:]
        
        metrics = {
            'episode': self._episode_count,
            'steps': self._step_count,
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_length': np.mean(recent_lengths),
            'max_reward': np.max(recent_rewards),
        }
        
        if self.logger is not None:
            for key, value in metrics.items():
                self.logger.log({key: value}, step=self._step_count)
        else:
            print(f"[Episode {self._episode_count}] "
                  f"Mean reward: {metrics['mean_reward']:.2f}, "
                  f"Mean length: {metrics['mean_length']:.1f}")


# =============================================================================
# WRAPPER UTILITIES
# =============================================================================

def make_wrapped_env(
    env,
    frame_stack: int = 1,
    action_repeat: int = 1,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    reward_scale: float = 1.0,
    time_limit: Optional[int] = None,
    record: bool = False,
):
    """
    Apply standard wrapper stack to environment.
    
    Args:
        env: Base environment
        frame_stack: Number of frames to stack (1 = no stacking)
        action_repeat: Number of times to repeat each action
        normalize_obs: Apply observation normalization
        normalize_reward: Apply reward normalization
        reward_scale: Reward scaling factor
        time_limit: Maximum episode steps (None = use env default)
        record: Record episodes
        
    Returns:
        Wrapped environment
    """
    if record:
        env = RecordEpisode(env)
    
    if time_limit is not None:
        env = TimeLimit(env, max_steps=time_limit)
    
    if action_repeat > 1:
        env = ActionRepeat(env, repeat=action_repeat)
    
    if normalize_obs:
        env = NormalizeObservation(env)
    
    if frame_stack > 1:
        env = FrameStack(env, num_frames=frame_stack)
    
    if reward_scale != 1.0:
        env = RewardScaling(env, scale=reward_scale)
    
    if normalize_reward:
        env = RewardNormalization(env)
    
    return env
