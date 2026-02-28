"""
Tests for HyperEnv - Layer 4: Multi-Agent Training Infrastructure
===================================================================

Tests cover:
- Agent interface and implementations
- Multi-agent environments
- Training loops
- Evaluation
- Callbacks
- Experience buffers
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# =============================================================================
# AGENT TESTS
# =============================================================================


class TestAgent:
    """Tests for Agent base class and implementations."""

    def test_agent_config_defaults(self):
        """Test AgentConfig has sensible defaults."""
        from ontic.infra.hyperenv import AgentConfig

        config = AgentConfig()
        assert config.name == "agent"
        assert config.action_dim == 4
        assert config.learning_rate > 0
        assert config.gamma > 0 and config.gamma < 1

    def test_agent_state_serialization(self):
        """Test AgentState can be saved and loaded."""
        from ontic.infra.hyperenv import AgentState

        state = AgentState(step=100, episode=5, epsilon=0.5, total_reward=500.0)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            state.save(path)
            loaded = AgentState.load(path)

            assert loaded.step == 100
            assert loaded.episode == 5
            assert loaded.epsilon == 0.5
            assert loaded.total_reward == 500.0
        finally:
            os.unlink(path)

    def test_random_agent(self):
        """Test RandomAgent produces valid actions."""
        from ontic.infra.hyperenv import AgentConfig, RandomAgent

        config = AgentConfig(action_dim=4)
        agent = RandomAgent(config=config)

        obs = np.zeros((64, 64, 3), dtype=np.float32)
        action = agent.act(obs)

        assert action.shape == (4,)
        assert action.dtype == np.float32

    def test_constant_agent(self):
        """Test ConstantAgent returns same action."""
        from ontic.infra.hyperenv import ConstantAgent

        constant_action = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
        agent = ConstantAgent(action=constant_action)

        obs = np.zeros((64, 64, 3), dtype=np.float32)

        # Should always return same action
        for _ in range(10):
            action = agent.act(obs)
            np.testing.assert_array_equal(action, constant_action)

    def test_agent_train_eval_mode(self):
        """Test agent training/eval mode switching."""
        from ontic.infra.hyperenv import RandomAgent

        agent = RandomAgent()

        assert agent.training is True

        agent.eval()
        assert agent.training is False

        agent.train()
        assert agent.training is True

    def test_agent_end_episode(self):
        """Test agent episode tracking."""
        from ontic.infra.hyperenv import RandomAgent

        agent = RandomAgent()

        assert agent.state.episode == 0

        agent.end_episode(episode_reward=100.0)
        assert agent.state.episode == 1
        assert agent.state.total_reward == 100.0
        assert agent.state.best_reward == 100.0

        agent.end_episode(episode_reward=50.0)
        assert agent.state.episode == 2
        assert agent.state.total_reward == 150.0
        assert agent.state.best_reward == 100.0  # Should keep best


# =============================================================================
# MULTI-AGENT TESTS
# =============================================================================


class TestMultiAgent:
    """Tests for multi-agent environments."""

    def test_multiagent_config(self):
        """Test MultiAgentConfig creation."""
        from ontic.infra.hyperenv import AgentRole, MultiAgentConfig, TeamConfig

        config = MultiAgentConfig(
            teams=[
                TeamConfig(name="team_a", size=2, role=AgentRole.COOPERATIVE),
                TeamConfig(name="team_b", size=1, role=AgentRole.COMPETITIVE),
            ]
        )

        assert len(config.teams) == 2
        assert config.teams[0].size == 2
        assert config.teams[1].role == AgentRole.COMPETITIVE

    def test_multiagent_env_creation(self):
        """Test MultiAgentEnv wrapping."""
        from ontic.infra.hyperenv import (MultiAgentConfig, MultiAgentEnv,
                                        TeamConfig)

        # Mock base environment
        base_env = MagicMock()
        base_env.reset.return_value = (np.zeros((64, 64, 3)), {})
        base_env.step.return_value = (np.zeros((64, 64, 3)), 1.0, False, False, {})

        config = MultiAgentConfig(teams=[TeamConfig(name="team", size=2)])

        multi_env = MultiAgentEnv(base_env, config)

        assert multi_env.num_agents == 2
        assert len(multi_env.agent_ids) == 2

    def test_multiagent_reset(self):
        """Test multi-agent reset returns per-agent observations."""
        from ontic.infra.hyperenv import (MultiAgentConfig, MultiAgentEnv,
                                        TeamConfig)

        base_env = MagicMock()
        base_env.reset.return_value = (np.ones((64, 64, 3)), {})

        config = MultiAgentConfig(teams=[TeamConfig(name="team", size=3)])

        multi_env = MultiAgentEnv(base_env, config)
        observations = multi_env.reset()

        assert len(observations) == 3
        for obs in observations.values():
            assert obs.shape == (64, 64, 3)

    def test_multiagent_step(self):
        """Test multi-agent step processes all agents."""
        from ontic.infra.hyperenv import (AgentRole, MultiAgentConfig,
                                        MultiAgentEnv, TeamConfig)

        base_env = MagicMock()
        base_env.reset.return_value = (np.zeros((64, 64, 3)), {})
        base_env.step.return_value = (np.zeros((64, 64, 3)), 10.0, False, False, {})

        config = MultiAgentConfig(
            teams=[TeamConfig(name="team", size=2, role=AgentRole.COOPERATIVE)]
        )

        multi_env = MultiAgentEnv(base_env, config)
        multi_env.reset()

        actions = {agent_id: np.zeros(4) for agent_id in multi_env.agent_ids}

        obs, rewards, dones, truncated, infos = multi_env.step(actions)

        assert len(obs) == 2
        assert len(rewards) == 2
        assert len(dones) == 2

    def test_competitive_rewards(self):
        """Test competitive agents get opposite rewards."""
        from ontic.infra.hyperenv import (AgentRole, MultiAgentConfig,
                                        MultiAgentEnv, TeamConfig)

        base_env = MagicMock()
        base_env.reset.return_value = (np.zeros((64, 64, 3)), {})
        base_env.step.return_value = (np.zeros((64, 64, 3)), 10.0, False, False, {})

        config = MultiAgentConfig(
            teams=[
                TeamConfig(name="ego", size=1, role=AgentRole.COOPERATIVE),
                TeamConfig(name="adversary", size=1, role=AgentRole.ADVERSARY),
            ]
        )

        multi_env = MultiAgentEnv(base_env, config)
        multi_env.reset()

        actions = {aid: np.zeros(4) for aid in multi_env.agent_ids}
        obs, rewards, dones, truncated, infos = multi_env.step(actions)

        # Cooperative gets positive, adversary gets negative
        ego_reward = rewards[[k for k in rewards.keys() if "ego" in k][0]]
        adv_reward = rewards[[k for k in rewards.keys() if "adversary" in k][0]]

        assert ego_reward > 0
        assert adv_reward < 0


# =============================================================================
# TRAINER TESTS
# =============================================================================


class TestTrainer:
    """Tests for training infrastructure."""

    def test_trainer_config_defaults(self):
        """Test TrainerConfig has sensible defaults."""
        from ontic.infra.hyperenv import TrainerConfig

        config = TrainerConfig()
        assert config.total_timesteps > 0
        assert config.batch_size > 0
        assert config.eval_freq > 0

    def test_training_state_serialization(self):
        """Test TrainingState save/load."""
        from ontic.infra.hyperenv import TrainingState

        state = TrainingState(timestep=50000, episode=100)
        state.episode_rewards = [1.0, 2.0, 3.0]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            state.save(path)
            loaded = TrainingState.load(path)

            assert loaded.timestep == 50000
            assert loaded.episode == 100
            assert len(loaded.episode_rewards) == 3
        finally:
            os.unlink(path)

    def test_trainer_creation(self):
        """Test Trainer can be created."""
        from ontic.infra.hyperenv import RandomAgent, Trainer, TrainerConfig

        agent = RandomAgent()
        env = MagicMock()

        trainer = Trainer(agent, env, TrainerConfig(total_timesteps=100))

        assert trainer.agent == agent
        assert trainer.env == env

    def test_trainer_train_short(self):
        """Test short training run."""
        from ontic.infra.hyperenv import RandomAgent, Trainer, TrainerConfig

        agent = RandomAgent()

        # Mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})
        env.step.return_value = (np.zeros(10), 1.0, False, False, {})

        config = TrainerConfig(
            total_timesteps=50,
            learning_starts=10,
            log_freq=100,  # Suppress logging
            eval_freq=1000,  # Skip eval
            save_freq=1000,  # Skip save
        )

        trainer = Trainer(agent, env, config)
        trainer.train()

        assert trainer.state.timestep >= 50

    def test_trainer_stop(self):
        """Test trainer can be stopped."""
        from ontic.infra.hyperenv import (LambdaCallback, RandomAgent, Trainer,
                                        TrainerConfig)

        agent = RandomAgent()
        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})
        env.step.return_value = (np.zeros(10), 1.0, False, False, {})

        # Stop after 25 steps
        stop_callback = LambdaCallback(
            on_step=lambda t: False if t.state.timestep >= 25 else None
        )

        config = TrainerConfig(total_timesteps=1000)
        trainer = Trainer(agent, env, config, callbacks=[stop_callback])
        trainer.train()

        assert trainer.state.timestep < 1000


# =============================================================================
# EVALUATOR TESTS
# =============================================================================


class TestEvaluator:
    """Tests for evaluation infrastructure."""

    def test_evaluation_result(self):
        """Test EvaluationResult creation."""
        from ontic.infra.hyperenv import EvaluationResult

        result = EvaluationResult(
            mean_reward=100.0,
            std_reward=10.0,
            n_episodes=10,
        )

        assert result.mean_reward == 100.0
        assert result.n_episodes == 10

    def test_evaluation_result_serialization(self):
        """Test EvaluationResult save/load."""
        from ontic.infra.hyperenv import EvaluationResult

        result = EvaluationResult(mean_reward=42.0, std_reward=5.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            result.save(path)
            loaded = EvaluationResult.load(path)

            assert loaded.mean_reward == 42.0
            assert loaded.std_reward == 5.0
        finally:
            os.unlink(path)

    def test_evaluator_evaluate(self):
        """Test Evaluator.evaluate runs episodes."""
        from ontic.infra.hyperenv import Evaluator, RandomAgent

        agent = RandomAgent()

        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})

        # Episode ends after 5 steps
        step_count = [0]

        def mock_step(action):
            step_count[0] += 1
            done = step_count[0] % 5 == 0
            return np.zeros(10), 1.0, done, False, {}

        env.step.side_effect = mock_step

        evaluator = Evaluator(env, n_episodes=3)
        result = evaluator.evaluate(agent)

        assert result.n_episodes == 3
        assert len(result.episode_rewards) == 3
        assert all(r == 5.0 for r in result.episode_rewards)  # 5 steps * 1.0 reward

    def test_benchmark_suite(self):
        """Test BenchmarkSuite runs all environments."""
        from ontic.infra.hyperenv import BenchmarkSuite, RandomAgent

        agent = RandomAgent()

        # Mock environments
        def make_mock_env():
            env = MagicMock()
            env.reset.return_value = (np.zeros(10), {})
            env.step.return_value = (
                np.zeros(10),
                1.0,
                True,
                False,
                {},
            )  # Immediate done
            return env

        suite = BenchmarkSuite(n_episodes_per_env=2)
        suite.add_env("env_a", make_mock_env())
        suite.add_env("env_b", make_mock_env())

        results = suite.run(agent, verbose=False)

        assert "env_a" in results
        assert "env_b" in results
        assert results["env_a"].n_episodes == 2


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestCallbacks:
    """Tests for training callbacks."""

    def test_callback_list(self):
        """Test CallbackList composes callbacks."""
        from ontic.infra.hyperenv import Callback, CallbackList

        calls = []

        class TrackerCallback(Callback):
            def __init__(self, name):
                self.name = name

            def on_training_start(self, trainer):
                calls.append(f"{self.name}_start")

            def on_training_end(self, trainer):
                calls.append(f"{self.name}_end")

        cb_list = CallbackList(
            [
                TrackerCallback("a"),
                TrackerCallback("b"),
            ]
        )

        cb_list.on_training_start(None)
        cb_list.on_training_end(None)

        assert calls == ["a_start", "b_start", "a_end", "b_end"]

    def test_early_stopping_callback(self):
        """Test EarlyStoppingCallback stops training."""
        from ontic.infra.hyperenv import EarlyStoppingCallback, TrainingState

        callback = EarlyStoppingCallback(patience=2, check_freq=1, verbose=0)

        # Mock trainer
        trainer = MagicMock()
        trainer.state = TrainingState(timestep=0)
        trainer.state.episode_rewards = [10.0] * 100  # No improvement

        # First check - no improvement but within patience
        trainer.state.timestep = 1
        result = callback.on_step(trainer)
        assert result is None

        # Second check - still no improvement
        trainer.state.timestep = 2
        result = callback.on_step(trainer)
        assert result is None

        # Third check - patience exhausted
        trainer.state.timestep = 3
        result = callback.on_step(trainer)
        assert result is False  # Should stop

    def test_lambda_callback(self):
        """Test LambdaCallback with custom functions."""
        from ontic.infra.hyperenv import LambdaCallback

        steps = []

        callback = LambdaCallback(on_step=lambda t: steps.append(t.state.timestep))

        trainer = MagicMock()
        trainer.state.timestep = 1
        callback.on_step(trainer)

        trainer.state.timestep = 2
        callback.on_step(trainer)

        assert steps == [1, 2]


# =============================================================================
# BUFFER TESTS
# =============================================================================


class TestBuffers:
    """Tests for experience buffers."""

    def test_experience_namedtuple(self):
        """Test Experience tuple."""
        from ontic.infra.hyperenv import Experience

        exp = Experience(
            observation=np.zeros(10),
            action=np.zeros(4),
            reward=1.0,
            next_observation=np.zeros(10),
            done=False,
        )

        assert exp.reward == 1.0
        assert exp.done is False

    def test_trajectory(self):
        """Test Trajectory storage."""
        from ontic.infra.hyperenv import Trajectory

        traj = Trajectory()

        for i in range(10):
            traj.add(
                observation=np.zeros(10),
                action=np.zeros(4),
                reward=1.0,
                done=i == 9,
            )

        assert len(traj) == 10
        assert traj.total_reward == 10.0

    def test_trajectory_compute_returns(self):
        """Test Trajectory return computation."""
        from ontic.infra.hyperenv import Trajectory

        traj = Trajectory()
        traj.rewards = [1.0, 1.0, 1.0]
        traj.dones = [False, False, True]

        returns = traj.compute_returns(gamma=0.99)

        # R_2 = 1.0, R_1 = 1 + 0.99*1 = 1.99, R_0 = 1 + 0.99*1.99 = 2.9701
        assert len(returns) == 3
        assert abs(returns[2] - 1.0) < 0.01
        assert abs(returns[1] - 1.99) < 0.01
        assert abs(returns[0] - 2.9701) < 0.01

    def test_replay_buffer_add_sample(self):
        """Test ReplayBuffer add and sample."""
        from ontic.infra.hyperenv import ReplayBuffer

        buffer = ReplayBuffer(capacity=100)

        # Add experiences
        for i in range(50):
            buffer.add(
                observation=np.ones(10) * i,
                action=np.zeros(4),
                reward=float(i),
                next_observation=np.ones(10) * (i + 1),
                done=False,
            )

        assert len(buffer) == 50

        # Sample batch
        batch = buffer.sample(batch_size=16)

        assert batch.observations.shape == (16, 10)
        assert batch.actions.shape == (16, 4)
        assert batch.rewards.shape == (16,)

    def test_replay_buffer_circular(self):
        """Test ReplayBuffer wraps around."""
        from ontic.infra.hyperenv import ReplayBuffer

        buffer = ReplayBuffer(capacity=10)

        # Add more than capacity
        for i in range(25):
            buffer.add(
                observation=np.ones(5) * i,
                action=np.zeros(2),
                reward=float(i),
                next_observation=np.ones(5) * (i + 1),
                done=False,
            )

        # Should only have last 10
        assert len(buffer) == 10

        # Rewards should be 15-24 (last 10)
        batch = buffer.sample(10)
        assert all(r >= 15 for r in batch.rewards)

    def test_prioritized_replay_buffer(self):
        """Test PrioritizedReplayBuffer sampling."""
        from ontic.infra.hyperenv import PrioritizedReplayBuffer

        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

        # Add with different priorities
        for i in range(50):
            buffer.add(
                observation=np.ones(10),
                action=np.zeros(4),
                reward=float(i),
                next_observation=np.ones(10),
                done=False,
                priority=float(i + 1),  # Higher priority for larger indices
            )

        # Sample should have weights
        batch = buffer.sample(16)

        assert batch.weights is not None
        assert len(batch.weights) == 16
        assert all(w > 0 for w in batch.weights)

    def test_rollout_buffer(self):
        """Test RolloutBuffer for on-policy."""
        from ontic.infra.hyperenv import RolloutBuffer

        buffer = RolloutBuffer(capacity=100, gamma=0.99)

        # Add experiences
        for i in range(50):
            buffer.add(
                observation=np.zeros(10),
                action=np.zeros(4),
                reward=1.0,
                done=i == 49,
                value=0.5,
                log_prob=-0.5,
            )

        assert len(buffer) == 50

        # Compute returns
        buffer.compute_returns_and_advantages(last_value=0.0)

        # Iterate batches
        batches = list(buffer.get_batches(batch_size=16))

        # 50 / 16 = 3 full batches + 1 partial
        assert len(batches) >= 3
        assert batches[0].returns is not None
        assert batches[0].advantages is not None

    def test_batch_to_torch(self):
        """Test Batch conversion to PyTorch."""
        from ontic.infra.hyperenv import Batch

        batch = Batch(
            observations=np.zeros((16, 10)),
            actions=np.zeros((16, 4)),
            rewards=np.ones(16),
            next_observations=np.zeros((16, 10)),
            dones=np.zeros(16),
        )

        tensors = batch.to_torch(device="cpu")

        assert isinstance(tensors["observations"], torch.Tensor)
        assert tensors["observations"].shape == (16, 10)
        assert tensors["rewards"].dtype == torch.float32


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for HyperEnv components."""

    def test_agent_to_trainer_pipeline(self):
        """Test complete agent-trainer pipeline."""
        from ontic.infra.hyperenv import (RandomAgent, ReplayBuffer, Trainer,
                                        TrainerConfig)

        agent = RandomAgent()

        # Simple environment
        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})
        env.step.return_value = (np.zeros(10), 1.0, False, False, {})

        buffer = ReplayBuffer(capacity=1000)
        config = TrainerConfig(
            total_timesteps=100,
            learning_starts=50,
            log_freq=200,
            eval_freq=500,
            save_freq=500,
        )

        trainer = Trainer(agent, env, config, buffer=buffer)
        trainer.train()

        assert trainer.state.timestep >= 100
        assert len(buffer) > 0

    def test_multiagent_with_random_agents(self):
        """Test multi-agent environment with random agents."""
        from ontic.infra.hyperenv import (MultiAgentConfig, MultiAgentEnv,
                                        RandomAgent, TeamConfig)

        # Base environment
        base_env = MagicMock()
        base_env.reset.return_value = (np.zeros((64, 64, 3)), {})
        base_env.step.return_value = (np.zeros((64, 64, 3)), 1.0, False, False, {})

        config = MultiAgentConfig(teams=[TeamConfig(name="team", size=2)])

        multi_env = MultiAgentEnv(base_env, config)

        # Create agents
        agents = {agent_id: RandomAgent() for agent_id in multi_env.agent_ids}

        # Run episode
        observations = multi_env.reset()

        for _ in range(10):
            actions = {
                agent_id: agents[agent_id].act(obs)
                for agent_id, obs in observations.items()
            }
            observations, rewards, dones, truncated, infos = multi_env.step(actions)

    def test_evaluate_trained_agent(self):
        """Test evaluating agent after training."""
        from ontic.infra.hyperenv import (Evaluator, RandomAgent, Trainer,
                                        TrainerConfig)

        agent = RandomAgent()

        # Mock environment
        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})

        step_count = [0]

        def mock_step(action):
            step_count[0] += 1
            done = step_count[0] % 10 == 0
            return np.zeros(10), 1.0, done, False, {}

        env.step.side_effect = mock_step

        # Train
        config = TrainerConfig(
            total_timesteps=50,
            learning_starts=10,
            log_freq=100,
            eval_freq=1000,
            save_freq=1000,
        )
        trainer = Trainer(agent, env, config)
        trainer.train()

        # Evaluate
        evaluator = Evaluator(env, n_episodes=5)
        result = evaluator.evaluate(agent)

        assert result.n_episodes == 5
        assert result.mean_reward > 0

    def test_callback_with_trainer(self):
        """Test custom callback during training."""
        from ontic.infra.hyperenv import (LambdaCallback, RandomAgent, Trainer,
                                        TrainerConfig)

        episode_rewards = []

        callback = LambdaCallback(
            on_episode_end=lambda t, r, l: episode_rewards.append(r)
        )

        agent = RandomAgent()

        env = MagicMock()
        env.reset.return_value = (np.zeros(10), {})

        # Episode ends every 10 steps
        step_count = [0]

        def mock_step(action):
            step_count[0] += 1
            done = step_count[0] % 10 == 0
            return np.zeros(10), 1.0, done, False, {}

        env.step.side_effect = mock_step

        config = TrainerConfig(
            total_timesteps=50,
            log_freq=100,
            eval_freq=1000,
            save_freq=1000,
        )

        trainer = Trainer(agent, env, config, callbacks=[callback])
        trainer.train()

        # Should have at least some episodes
        assert len(episode_rewards) > 0
        assert all(r == 10.0 for r in episode_rewards)  # 10 steps * 1.0
