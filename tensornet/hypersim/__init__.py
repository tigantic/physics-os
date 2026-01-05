"""
HyperSim - RL-Ready Simulation Environments
============================================

Layer 3 of HyperTensor architecture.
Gymnasium-compatible environments for learning fluid control.

Core Components:
    FluidEnv: Base environment for fluid dynamics control
    Rewards: Physics-based reward functions
    Spaces: Action/observation spaces with masking
    Curriculum: Progressive difficulty and domain randomization

Usage:
    from tensornet.hypersim import FluidEnv, make_env

    # Create environment
    env = make_env('fluid-control-v0')

    # Standard gym interface
    obs, info = env.reset(seed=42)
    for _ in range(1000):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
"""

from .curriculum import (
                         AdaptiveRandomizer,
                         AdvancementPolicy,
                         Curriculum,
                         CurriculumStage,
                         DifficultyScheduler,
                         DomainRandomizer,
                         RandomizationRange,
                         make_fluid_curriculum,
)
from .env import EnvState, FluidEnv, FluidEnvConfig, StepResult
from .registry import (
                         TASK_SPECS,
                         TaskSpec,
                         get_env_spec,
                         list_envs,
                         make_env,
                         make_fluid_env,
                         make_vectorized_env,
                         register_env,
                         register_with_gymnasium,
                         unregister_env,
)
from .rewards import (
                         BoundaryPenalty,
                         CompositeReward,
                         DenseReward,
                         DissipationReward,
                         EnergyReward,
                         RewardConfig,
                         RewardFunction,
                         ShapedReward,
                         SparseReward,
                         TargetMatchReward,
                         VorticityReward,
                         make_reward,
)
from .spaces import (
                         ActionConfig,
                         ActionMask,
                         ActionType,
                         ActuatorConfig,
                         FieldAction,
                         FieldObservation,
                         ObservationConfig,
                         ObservationType,
                         build_action_space,
                         build_observation_space,
)
from .wrappers import (
                         ActionRepeat,
                         AutoReset,
                         ClipAction,
                         DownsampleObservation,
                         EpisodeRecord,
                         FrameStack,
                         LogMetrics,
                         NormalizeObservation,
                         RecordEpisode,
                         RescaleAction,
                         RewardClipping,
                         RewardNormalization,
                         RewardScaling,
                         StickyAction,
                         TimeLimit,
                         make_wrapped_env,
)

__all__ = [
    # Environment
    "FluidEnv",
    "FluidEnvConfig",
    "EnvState",
    "StepResult",
    # Rewards
    "RewardFunction",
    "RewardConfig",
    "SparseReward",
    "DenseReward",
    "ShapedReward",
    "CompositeReward",
    "TargetMatchReward",
    "VorticityReward",
    "EnergyReward",
    "DissipationReward",
    "BoundaryPenalty",
    "make_reward",
    # Spaces
    "FieldObservation",
    "FieldAction",
    "ActionMask",
    "ObservationConfig",
    "ActionConfig",
    "ObservationType",
    "ActionType",
    "ActuatorConfig",
    "build_observation_space",
    "build_action_space",
    # Curriculum
    "Curriculum",
    "CurriculumStage",
    "AdvancementPolicy",
    "DomainRandomizer",
    "AdaptiveRandomizer",
    "DifficultyScheduler",
    "RandomizationRange",
    "make_fluid_curriculum",
    # Wrappers
    "FrameStack",
    "ActionRepeat",
    "RewardScaling",
    "RewardClipping",
    "RewardNormalization",
    "TimeLimit",
    "AutoReset",
    "RecordEpisode",
    "EpisodeRecord",
    "LogMetrics",
    "NormalizeObservation",
    "DownsampleObservation",
    "ClipAction",
    "RescaleAction",
    "StickyAction",
    "make_wrapped_env",
    # Registry
    "make_env",
    "register_env",
    "unregister_env",
    "list_envs",
    "get_env_spec",
    "make_fluid_env",
    "make_vectorized_env",
    "register_with_gymnasium",
    "TaskSpec",
    "TASK_SPECS",
]

__version__ = "0.1.0"
