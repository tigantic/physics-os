"""
HyperEnv - Multi-Agent RL Training Infrastructure
==================================================

Layer 4 of the HyperTensor platform.

Builds on HyperSim to provide:
- Multi-agent coordination
- Training APIs for popular RL frameworks
- Evaluation and benchmarking utilities
- Environment factories for common scenarios

Components:
    Agent           - Base agent interface
    MultiAgentEnv   - Multi-agent environment wrapper
    Trainer         - Training loop abstraction
    Evaluator       - Policy evaluation
    Callbacks       - Training callbacks
    Policies        - Policy implementations
    Buffers         - Experience replay buffers

Example:
    from tensornet.hyperenv import make_trainer, make_evaluator
    
    trainer = make_trainer('ppo', env='fluid-control-v0')
    trainer.train(total_timesteps=1_000_000)
    
    evaluator = make_evaluator(trainer.policy)
    results = evaluator.evaluate(n_episodes=100)
"""

from __future__ import annotations

# Agents
from .agent import (
    Agent,
    AgentConfig,
    AgentState,
    RandomAgent,
    ConstantAgent,
)

# Multi-agent
from .multiagent import (
    MultiAgentEnv,
    MultiAgentConfig,
    TeamConfig,
    AgentRole,
)

# Training
from .trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
    make_trainer,
)

# Evaluation
from .evaluator import (
    Evaluator,
    EvaluationResult,
    BenchmarkSuite,
    make_evaluator,
)

# Callbacks
from .callbacks import (
    Callback,
    CheckpointCallback,
    EvalCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    CallbackList,
    LambdaCallback,
    ProgressCallback,
)

# Buffers
from .buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    RolloutBuffer,
    Trajectory,
    Experience,
    Batch,
    TrajectoryBuffer,
)

__all__ = [
    # Agents
    'Agent',
    'AgentConfig',
    'AgentState',
    'RandomAgent',
    'ConstantAgent',
    
    # Multi-agent
    'MultiAgentEnv',
    'MultiAgentConfig',
    'TeamConfig',
    'AgentRole',
    
    # Training
    'Trainer',
    'TrainerConfig',
    'TrainingState',
    'make_trainer',
    
    # Evaluation
    'Evaluator',
    'EvaluationResult',
    'BenchmarkSuite',
    'make_evaluator',
    
    # Callbacks
    'Callback',
    'CheckpointCallback',
    'EvalCallback',
    'LoggingCallback',
    'EarlyStoppingCallback',
    'CallbackList',
    'LambdaCallback',
    'ProgressCallback',
    
    # Buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'RolloutBuffer',
    'Trajectory',
    'Experience',
    'Batch',
    'TrajectoryBuffer',
]

__version__ = '0.1.0'
