"""
Curriculum Learning
===================

Automatic difficulty scaling and domain randomization
for robust policy training.

Components:
    - Curriculum: Multi-stage training progression
    - DomainRandomizer: Physical parameter variation
    - DifficultyScheduler: Adaptive difficulty adjustment
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# =============================================================================
# CURRICULUM STAGES
# =============================================================================


@dataclass
class CurriculumStage:
    """
    A single stage in the curriculum.

    Defines the conditions, parameters, and advancement criteria
    for one phase of training.
    """

    name: str

    # Stage parameters (override env defaults)
    viscosity: float = 0.01
    dt: float = 0.01
    max_steps: int = 100
    grid_bits: int = 6

    # Difficulty modifiers
    noise_scale: float = 0.0
    perturbation_freq: float = 0.0
    obstacle_count: int = 0

    # Reward shaping
    reward_scale: float = 1.0
    bonus_for_completion: float = 0.0

    # Advancement criteria
    success_threshold: float = 0.8  # Success rate to advance
    min_episodes: int = 100
    max_episodes: int = 1000

    # Randomization ranges (for domain randomization)
    viscosity_range: tuple[float, float] = (0.01, 0.01)
    dt_range: tuple[float, float] = (0.01, 0.01)

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample randomized parameters for an episode."""
        return {
            "viscosity": rng.uniform(*self.viscosity_range),
            "dt": rng.uniform(*self.dt_range),
            "noise_scale": self.noise_scale,
            "perturbation_freq": self.perturbation_freq,
            "obstacle_count": self.obstacle_count,
        }


class AdvancementPolicy(Enum):
    """Policy for curriculum advancement."""

    SUCCESS_RATE = "success_rate"  # Advance when success rate exceeds threshold
    REWARD_THRESHOLD = "reward"  # Advance when mean reward exceeds threshold
    EPISODE_COUNT = "episodes"  # Advance after fixed number of episodes
    MANUAL = "manual"  # Only advance when explicitly triggered


# =============================================================================
# CURRICULUM
# =============================================================================


class Curriculum:
    """
    Multi-stage curriculum for progressive training.

    Automatically advances through stages based on
    agent performance.

    Example:
        curriculum = Curriculum([
            CurriculumStage("easy", viscosity=0.1, max_steps=50),
            CurriculumStage("medium", viscosity=0.05, max_steps=100),
            CurriculumStage("hard", viscosity=0.01, max_steps=200),
        ])

        for episode in range(10000):
            env.reset(options=curriculum.get_options())
            ...
            curriculum.record_episode(success=done and reward > 0)
    """

    def __init__(
        self,
        stages: list[CurriculumStage],
        advancement_policy: AdvancementPolicy = AdvancementPolicy.SUCCESS_RATE,
        allow_regression: bool = False,
        seed: int | None = None,
    ):
        self.stages = stages
        self.advancement_policy = advancement_policy
        self.allow_regression = allow_regression

        self._current_stage = 0
        self._episode_count = 0
        self._success_count = 0
        self._reward_history: list[float] = []
        self._rng = np.random.default_rng(seed)

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self._current_stage]

    @property
    def stage_index(self) -> int:
        """Get current stage index."""
        return self._current_stage

    @property
    def is_final_stage(self) -> bool:
        """Check if at final stage."""
        return self._current_stage >= len(self.stages) - 1

    @property
    def progress(self) -> float:
        """Overall curriculum progress [0, 1]."""
        stage_progress = self._current_stage / max(1, len(self.stages) - 1)
        within_stage = min(1.0, self._episode_count / self.current_stage.min_episodes)
        return stage_progress + within_stage / len(self.stages)

    def get_options(self) -> dict[str, Any]:
        """
        Get environment options for current curriculum stage.

        Returns:
            Dict to pass to env.reset(options=...)
        """
        stage = self.current_stage
        params = stage.sample_params(self._rng)

        return {
            "curriculum_stage": stage.name,
            "curriculum_progress": self.progress,
            **params,
        }

    def record_episode(
        self,
        success: bool = False,
        reward: float = 0.0,
    ):
        """
        Record episode outcome and potentially advance stage.

        Args:
            success: Whether episode was successful
            reward: Total episode reward
        """
        self._episode_count += 1
        if success:
            self._success_count += 1
        self._reward_history.append(reward)

        # Check advancement
        if self._should_advance():
            self._advance()
        elif self.allow_regression and self._should_regress():
            self._regress()

    def _should_advance(self) -> bool:
        """Check if should advance to next stage."""
        if self.is_final_stage:
            return False

        stage = self.current_stage

        if self._episode_count < stage.min_episodes:
            return False

        if self.advancement_policy == AdvancementPolicy.SUCCESS_RATE:
            success_rate = self._success_count / max(1, self._episode_count)
            return success_rate >= stage.success_threshold

        elif self.advancement_policy == AdvancementPolicy.REWARD_THRESHOLD:
            mean_reward = np.mean(self._reward_history[-100:])
            return mean_reward >= stage.success_threshold

        elif self.advancement_policy == AdvancementPolicy.EPISODE_COUNT:
            return self._episode_count >= stage.min_episodes

        return False

    def _should_regress(self) -> bool:
        """Check if should regress to previous stage."""
        if self._current_stage == 0:
            return False

        # Regress if success rate drops significantly
        if self._episode_count > 50:
            success_rate = self._success_count / self._episode_count
            return success_rate < 0.3

        return False

    def _advance(self):
        """Advance to next stage."""
        if self._current_stage < len(self.stages) - 1:
            self._current_stage += 1
            self._reset_stats()

    def _regress(self):
        """Regress to previous stage."""
        if self._current_stage > 0:
            self._current_stage -= 1
            self._reset_stats()

    def _reset_stats(self):
        """Reset episode statistics for new stage."""
        self._episode_count = 0
        self._success_count = 0
        self._reward_history.clear()

    def set_stage(self, stage_index: int):
        """Manually set curriculum stage."""
        self._current_stage = max(0, min(stage_index, len(self.stages) - 1))
        self._reset_stats()

    def state_dict(self) -> dict[str, Any]:
        """Get curriculum state for checkpointing."""
        return {
            "stage_index": self._current_stage,
            "episode_count": self._episode_count,
            "success_count": self._success_count,
            "reward_history": self._reward_history.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]):
        """Load curriculum state from checkpoint."""
        self._current_stage = state["stage_index"]
        self._episode_count = state["episode_count"]
        self._success_count = state["success_count"]
        self._reward_history = state["reward_history"].copy()


# =============================================================================
# DOMAIN RANDOMIZATION
# =============================================================================


@dataclass
class RandomizationRange:
    """Range for a single randomized parameter."""

    name: str
    low: float
    high: float
    log_scale: bool = False  # Use log-uniform sampling

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from the range."""
        if self.log_scale:
            log_low, log_high = np.log(self.low), np.log(self.high)
            return np.exp(rng.uniform(log_low, log_high))
        return rng.uniform(self.low, self.high)


class DomainRandomizer:
    """
    Domain randomization for robust policy training.

    Varies physical parameters, initial conditions, and
    disturbances to prevent overfitting.

    Example:
        randomizer = DomainRandomizer([
            RandomizationRange('viscosity', 0.001, 0.1, log_scale=True),
            RandomizationRange('dt', 0.005, 0.02),
            RandomizationRange('noise_scale', 0.0, 0.1),
        ])

        params = randomizer.sample()
        env.reset(options={'physics_params': params})
    """

    def __init__(
        self,
        ranges: list[RandomizationRange],
        seed: int | None = None,
    ):
        self.ranges = {r.name: r for r in ranges}
        self._rng = np.random.default_rng(seed)

    def sample(self) -> dict[str, float]:
        """Sample all randomized parameters."""
        return {name: r.sample(self._rng) for name, r in self.ranges.items()}

    def sample_subset(self, names: list[str]) -> dict[str, float]:
        """Sample a subset of parameters."""
        return {
            name: self.ranges[name].sample(self._rng)
            for name in names
            if name in self.ranges
        }

    def set_range(self, name: str, low: float, high: float):
        """Update a parameter range."""
        if name in self.ranges:
            self.ranges[name].low = low
            self.ranges[name].high = high

    def narrow_ranges(self, factor: float = 0.5):
        """Narrow all ranges (for curriculum)."""
        for r in self.ranges.values():
            midpoint = (r.low + r.high) / 2
            half_width = (r.high - r.low) / 2 * factor
            r.low = midpoint - half_width
            r.high = midpoint + half_width


class AdaptiveRandomizer(DomainRandomizer):
    """
    Domain randomizer that adapts ranges based on performance.

    Automatically expands ranges where agent succeeds and
    narrows where it fails.
    """

    def __init__(
        self,
        ranges: list[RandomizationRange],
        adaptation_rate: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(ranges, seed)
        self.adaptation_rate = adaptation_rate
        self._success_by_param: dict[str, list[tuple[float, bool]]] = {
            r.name: [] for r in ranges
        }

    def record_outcome(
        self,
        params: dict[str, float],
        success: bool,
    ):
        """Record episode outcome with parameters used."""
        for name, value in params.items():
            if name in self._success_by_param:
                self._success_by_param[name].append((value, success))

                # Keep only recent history
                if len(self._success_by_param[name]) > 1000:
                    self._success_by_param[name].pop(0)

        self._adapt_ranges()

    def _adapt_ranges(self):
        """Adapt ranges based on performance."""
        for name, history in self._success_by_param.items():
            if len(history) < 50:
                continue

            r = self.ranges[name]

            # Find success rate at range extremes
            low_success = self._success_rate_near(history, r.low)
            high_success = self._success_rate_near(history, r.high)

            # Expand where succeeding, narrow where failing
            if low_success > 0.7:
                r.low *= 1 - self.adaptation_rate
            elif low_success < 0.3:
                r.low = r.low + (r.high - r.low) * self.adaptation_rate * 0.5

            if high_success > 0.7:
                r.high *= 1 + self.adaptation_rate
            elif high_success < 0.3:
                r.high = r.high - (r.high - r.low) * self.adaptation_rate * 0.5

    def _success_rate_near(
        self,
        history: list[tuple[float, bool]],
        value: float,
        threshold: float = 0.2,
    ) -> float:
        """Compute success rate near a value."""
        r = self.ranges.get(list(self.ranges.keys())[0])
        if r is None:
            return 0.5

        range_width = r.high - r.low
        near_episodes = [
            success
            for v, success in history
            if abs(v - value) < threshold * range_width
        ]

        if not near_episodes:
            return 0.5

        return sum(near_episodes) / len(near_episodes)


# =============================================================================
# DIFFICULTY SCHEDULER
# =============================================================================


class DifficultyScheduler:
    """
    Adaptive difficulty adjustment based on performance.

    Smoothly interpolates between easy and hard settings
    based on rolling success rate.
    """

    def __init__(
        self,
        easy_params: dict[str, float],
        hard_params: dict[str, float],
        target_success_rate: float = 0.7,
        adaptation_rate: float = 0.01,
    ):
        self.easy_params = easy_params
        self.hard_params = hard_params
        self.target_success_rate = target_success_rate
        self.adaptation_rate = adaptation_rate

        self._difficulty = 0.0  # 0 = easy, 1 = hard
        self._success_history: list[bool] = []

    @property
    def difficulty(self) -> float:
        """Current difficulty level [0, 1]."""
        return self._difficulty

    def get_params(self) -> dict[str, float]:
        """Get current parameters based on difficulty."""
        params = {}
        for key in self.easy_params:
            easy = self.easy_params[key]
            hard = self.hard_params.get(key, easy)
            params[key] = easy + (hard - easy) * self._difficulty
        return params

    def record_episode(self, success: bool):
        """Record episode outcome and adjust difficulty."""
        self._success_history.append(success)

        # Keep rolling window
        if len(self._success_history) > 100:
            self._success_history.pop(0)

        # Compute success rate
        success_rate = sum(self._success_history) / len(self._success_history)

        # Adjust difficulty
        if success_rate > self.target_success_rate + 0.05:
            self._difficulty = min(1.0, self._difficulty + self.adaptation_rate)
        elif success_rate < self.target_success_rate - 0.05:
            self._difficulty = max(0.0, self._difficulty - self.adaptation_rate)

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "difficulty": self._difficulty,
            "success_history": self._success_history.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]):
        """Load state from checkpoint."""
        self._difficulty = state["difficulty"]
        self._success_history = state["success_history"].copy()


# =============================================================================
# PRESET CURRICULA
# =============================================================================


def make_fluid_curriculum(difficulty: str = "standard") -> Curriculum:
    """
    Create a preset curriculum for fluid control tasks.

    Args:
        difficulty: 'easy', 'standard', or 'hard'

    Returns:
        Configured Curriculum
    """
    if difficulty == "easy":
        stages = [
            CurriculumStage(
                name="beginner",
                viscosity=0.1,
                dt=0.02,
                max_steps=50,
                success_threshold=0.7,
                min_episodes=50,
            ),
            CurriculumStage(
                name="intermediate",
                viscosity=0.05,
                dt=0.01,
                max_steps=100,
                success_threshold=0.7,
                min_episodes=100,
            ),
        ]
    elif difficulty == "hard":
        stages = [
            CurriculumStage(
                name="warmup",
                viscosity=0.1,
                dt=0.02,
                max_steps=50,
                success_threshold=0.8,
                min_episodes=100,
            ),
            CurriculumStage(
                name="beginner",
                viscosity=0.05,
                dt=0.01,
                max_steps=100,
                noise_scale=0.01,
                success_threshold=0.8,
                min_episodes=200,
            ),
            CurriculumStage(
                name="intermediate",
                viscosity=0.02,
                dt=0.01,
                max_steps=150,
                noise_scale=0.02,
                perturbation_freq=0.1,
                success_threshold=0.75,
                min_episodes=300,
            ),
            CurriculumStage(
                name="advanced",
                viscosity=0.01,
                dt=0.005,
                max_steps=200,
                noise_scale=0.05,
                perturbation_freq=0.2,
                obstacle_count=2,
                success_threshold=0.7,
                min_episodes=500,
            ),
            CurriculumStage(
                name="expert",
                viscosity=0.005,
                dt=0.005,
                max_steps=300,
                noise_scale=0.1,
                perturbation_freq=0.3,
                obstacle_count=4,
                viscosity_range=(0.001, 0.01),
                dt_range=(0.002, 0.01),
                success_threshold=0.6,
                min_episodes=1000,
            ),
        ]
    else:  # standard
        stages = [
            CurriculumStage(
                name="beginner",
                viscosity=0.1,
                dt=0.02,
                max_steps=50,
                success_threshold=0.75,
                min_episodes=100,
            ),
            CurriculumStage(
                name="intermediate",
                viscosity=0.05,
                dt=0.01,
                max_steps=100,
                noise_scale=0.01,
                success_threshold=0.75,
                min_episodes=200,
            ),
            CurriculumStage(
                name="advanced",
                viscosity=0.01,
                dt=0.01,
                max_steps=200,
                noise_scale=0.02,
                viscosity_range=(0.005, 0.02),
                success_threshold=0.7,
                min_episodes=500,
            ),
        ]

    return Curriculum(stages, allow_regression=True)
