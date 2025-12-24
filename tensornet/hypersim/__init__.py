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

from .env import (
    FluidEnv,
    FluidEnvConfig,
    EnvState,
    StepResult,
)

from .rewards import (
    RewardFunction,
    RewardConfig,
    SparseReward,
    DenseReward,
    ShapedReward,
    CompositeReward,
    TargetMatchReward,
    VorticityReward,
    EnergyReward,
    DissipationReward,
    BoundaryPenalty,
    make_reward,
)

from .spaces import (
    FieldObservation,
    FieldAction,
    ActionMask,
    ObservationConfig,
    ActionConfig,
    ObservationType,
    ActionType,
    ActuatorConfig,
    build_observation_space,
    build_action_space,
)

from .curriculum import (
    Curriculum,
    CurriculumStage,
    AdvancementPolicy,
    DomainRandomizer,
    AdaptiveRandomizer,
    DifficultyScheduler,
    RandomizationRange,
    make_fluid_curriculum,
)

from .wrappers import (
    FrameStack,
    ActionRepeat,
    RewardScaling,
    RewardClipping,
    RewardNormalization,
    TimeLimit,
    AutoReset,
    RecordEpisode,
    EpisodeRecord,
    LogMetrics,
    NormalizeObservation,
    DownsampleObservation,
    ClipAction,
    RescaleAction,
    StickyAction,
    make_wrapped_env,
)

from .registry import (
    make_env,
    register_env,
    unregister_env,
    list_envs,
    get_env_spec,
    make_fluid_env,
    make_vectorized_env,
    register_with_gymnasium,
    TaskSpec,
    TASK_SPECS,
)

__all__ = [
    # Environment
    'FluidEnv',
    'FluidEnvConfig',
    'EnvState',
    'StepResult',
    
    # Rewards
    'RewardFunction',
    'RewardConfig',
    'SparseReward',
    'DenseReward',
    'ShapedReward',
    'CompositeReward',
    'TargetMatchReward',
    'VorticityReward',
    'EnergyReward',
    'DissipationReward',
    'BoundaryPenalty',
    'make_reward',
    
    # Spaces
    'FieldObservation',
    'FieldAction',
    'ActionMask',
    'ObservationConfig',
    'ActionConfig',
    'ObservationType',
    'ActionType',
    'ActuatorConfig',
    'build_observation_space',
    'build_action_space',
    
    # Curriculum
    'Curriculum',
    'CurriculumStage',
    'AdvancementPolicy',
    'DomainRandomizer',
    'AdaptiveRandomizer',
    'DifficultyScheduler',
    'RandomizationRange',
    'make_fluid_curriculum',
    
    # Wrappers
    'FrameStack',
    'ActionRepeat',
    'RewardScaling',
    'RewardClipping',
    'RewardNormalization',
    'TimeLimit',
    'AutoReset',
    'RecordEpisode',
    'EpisodeRecord',
    'LogMetrics',
    'NormalizeObservation',
    'DownsampleObservation',
    'ClipAction',
    'RescaleAction',
    'StickyAction',
    'make_wrapped_env',
    
    # Registry
    'make_env',
    'register_env',
    'unregister_env',
    'list_envs',
    'get_env_spec',
    'make_fluid_env',
    'make_vectorized_env',
    'register_with_gymnasium',
    'TaskSpec',
    'TASK_SPECS',
]

__version__ = '0.1.0'
