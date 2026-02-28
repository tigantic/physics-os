"""
HyperEnv - Multi-Agent RL Training Infrastructure
==================================================

Layer 4 of the Ontic Engine platform.

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
    from ontic.infra.hyperenv import make_trainer, make_evaluator

    trainer = make_trainer('ppo', env='fluid-control-v0')
    trainer.train(total_timesteps=1_000_000)

    evaluator = make_evaluator(trainer.policy)
    results = evaluator.evaluate(n_episodes=100)
"""

from __future__ import annotations

# Agents
from .agent import Agent, AgentConfig, AgentState, ConstantAgent, RandomAgent

# Buffers
from .buffers import (
                      Batch,
                      Experience,
                      PrioritizedReplayBuffer,
                      ReplayBuffer,
                      RolloutBuffer,
                      Trajectory,
                      TrajectoryBuffer,
)

# Callbacks
from .callbacks import (
                      Callback,
                      CallbackList,
                      CheckpointCallback,
                      EarlyStoppingCallback,
                      EvalCallback,
                      LambdaCallback,
                      LoggingCallback,
                      ProgressCallback,
)

# Evaluation
from .evaluator import BenchmarkSuite, EvaluationResult, Evaluator, make_evaluator

# Phase 4: Hypersonic RL Environment
from .hypersonic_env import (
                      AircraftState,
                      HypersonicEnv,
                      HypersonicEnvConfig,
                      TrajectoryTube,
                      make_hypersonic_env,
)

# Multi-agent
from .multiagent import AgentRole, MultiAgentConfig, MultiAgentEnv, TeamConfig

# Training
from .trainer import Trainer, TrainerConfig, TrainingState, make_trainer

__all__ = [
    # Agents
    "Agent",
    "AgentConfig",
    "AgentState",
    "RandomAgent",
    "ConstantAgent",
    # Multi-agent
    "MultiAgentEnv",
    "MultiAgentConfig",
    "TeamConfig",
    "AgentRole",
    # Training
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "make_trainer",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "BenchmarkSuite",
    "make_evaluator",
    # Callbacks
    "Callback",
    "CheckpointCallback",
    "EvalCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
    "CallbackList",
    "LambdaCallback",
    "ProgressCallback",
    # Buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "RolloutBuffer",
    "Trajectory",
    "Experience",
    "Batch",
    "TrajectoryBuffer",
    # Phase 4: Hypersonic RL
    "HypersonicEnv",
    "HypersonicEnvConfig",
    "AircraftState",
    "TrajectoryTube",
    "make_hypersonic_env",
]

__version__ = "0.1.0"
