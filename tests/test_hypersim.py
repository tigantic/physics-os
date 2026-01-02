"""
Tests for HyperSim - Layer 3
============================

RL-ready simulation environments with Gymnasium interface.
"""

from typing import Any, Dict

import numpy as np
import pytest
import torch

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_field():
    """Create a mock Field for testing."""

    class MockField:
        def __init__(self, ndim=2, bits=6, rank=8):
            self.ndim = ndim
            self.bits_per_dim = bits
            self.rank = rank
            self.total_bits = ndim * bits

            # Create mock cores
            self.cores = [torch.randn(rank, 2, rank) for _ in range(self.total_bits)]
            self.cores[0] = torch.randn(1, 2, rank)
            self.cores[-1] = torch.randn(rank, 2, 1)

        def sample(self, coords):
            """Sample at coordinates."""
            coords = np.atleast_2d(coords)
            return np.random.randn(len(coords), 3).astype(np.float32)

        def slice(self, axis=0, position=0.5, resolution=(64, 64)):
            """Get a 2D slice."""
            return np.random.randn(*resolution).astype(np.float32)

        def stats(self):
            """Get field statistics."""

            class Stats:
                mean = 0.1
                std = 0.5
                min_val = -1.0
                max_val = 1.0
                energy = 0.25
                enstrophy = 0.1
                divergence = 0.01
                vorticity_magnitude = 0.15

            return Stats()

        def clone(self):
            """Clone the field."""
            clone = MockField(self.ndim, self.bits_per_dim, self.rank)
            clone.cores = [c.clone() for c in self.cores]
            return clone

        @classmethod
        def zeros(cls, ndim=2, bits=6, rank=8):
            """Create zero field."""
            return cls(ndim, bits, rank)

    return MockField()


@pytest.fixture
def mock_gymnasium():
    """Mock gymnasium if not installed."""
    try:
        import gymnasium

        return gymnasium
    except ImportError:
        # Create minimal mock
        class MockSpace:
            def __init__(self, shape=None, low=None, high=None, dtype=np.float32):
                self.shape = shape
                self.low = low
                self.high = high
                self.dtype = dtype

            def sample(self):
                if self.shape:
                    return np.random.randn(*self.shape).astype(self.dtype)
                return np.random.randint(0, 10)

            def contains(self, x):
                return True

        class MockSpaces:
            Box = MockSpace
            Discrete = MockSpace
            MultiDiscrete = MockSpace
            Dict = dict

        class MockGymnasium:
            spaces = MockSpaces
            Env = object
            Wrapper = object
            ObservationWrapper = object
            ActionWrapper = object
            RewardWrapper = object

        return MockGymnasium()


# =============================================================================
# REWARD TESTS
# =============================================================================


class TestRewardFunctions:
    """Test reward function implementations."""

    def test_sparse_reward(self, mock_field):
        """Test sparse reward function."""
        from tensornet.hypersim.rewards import SparseReward

        # Goal achieved
        reward_fn = SparseReward(
            goal_fn=lambda f: True,
            success_reward=10.0,
        )
        assert reward_fn(mock_field, 0) == 10.0

        # Goal not achieved
        reward_fn = SparseReward(
            goal_fn=lambda f: False,
            success_reward=10.0,
        )
        assert reward_fn(mock_field, 0) == 0.0

    def test_dense_reward(self, mock_field):
        """Test dense reward function."""
        from tensornet.hypersim.rewards import DenseReward

        reward_fn = DenseReward(
            reward_fn=lambda f, s: -s * 0.01,  # Time penalty
        )

        assert reward_fn(mock_field, 0) == 0.0
        assert reward_fn(mock_field, 100) == pytest.approx(-1.0)

    def test_composite_reward(self, mock_field):
        """Test composite reward function."""
        from tensornet.hypersim.rewards import CompositeReward, DenseReward

        r1 = DenseReward(lambda f, s: 1.0)
        r2 = DenseReward(lambda f, s: 2.0)

        composite = CompositeReward(
            [
                (r1, 0.5),
                (r2, 0.3),
            ]
        )

        # 0.5 * 1.0 + 0.3 * 2.0 = 1.1
        assert composite(mock_field, 0) == pytest.approx(1.1)

    def test_shaped_reward(self, mock_field):
        """Test potential-based shaping."""
        from tensornet.hypersim.rewards import DenseReward, ShapedReward

        base = DenseReward(lambda f, s: 1.0)
        shaped = ShapedReward(
            base_reward=base,
            potential_fn=lambda f: f.stats().energy,
        )

        # First call: no shaping (no previous potential)
        r1 = shaped(mock_field, 0)
        assert r1 == pytest.approx(1.0)

        # Second call: includes shaping
        r2 = shaped(mock_field, 1)
        assert r2 != r1  # Shaping applied

        # Reset clears state
        shaped.reset()
        r3 = shaped(mock_field, 0)
        assert r3 == pytest.approx(1.0)

    def test_energy_reward(self, mock_field):
        """Test energy-based reward."""
        from tensornet.hypersim.rewards import EnergyReward

        # Minimize energy
        reward = EnergyReward(minimize=True)
        r = reward(mock_field, 0)
        assert r < 0  # Negative because we're minimizing

    def test_vorticity_reward(self, mock_field):
        """Test vorticity reward."""
        from tensornet.hypersim.rewards import VorticityReward

        reward = VorticityReward(maximize=True)
        r = reward(mock_field, 0)
        assert r >= 0  # Positive vorticity

    def test_dissipation_reward(self, mock_field):
        """Test dissipation reward."""
        from tensornet.hypersim.rewards import DissipationReward

        reward = DissipationReward()

        # First call: no previous energy
        r1 = reward(mock_field, 0)

        # Second call: has previous energy
        r2 = reward(mock_field, 1)

        reward.reset()
        r3 = reward(mock_field, 0)
        assert r3 == 0.0  # Reset clears state

    def test_reward_scaling(self, mock_field):
        """Test reward scaling and clipping."""
        from tensornet.hypersim.rewards import DenseReward, RewardConfig

        config = RewardConfig(scale=2.0, clip_min=-5.0, clip_max=5.0)
        reward = DenseReward(lambda f, s: 10.0, config=config)

        # Scaled: 10 * 2 = 20, but clipped to 5
        r = reward(mock_field, 0)
        assert r == 5.0

    def test_make_reward_factory(self, mock_field):
        """Test reward factory function."""
        from tensornet.hypersim.rewards import make_reward

        reward = make_reward("energy", minimize=True, scale=0.5)
        r = reward(mock_field, 0)
        assert isinstance(r, float)


# =============================================================================
# SPACES TESTS
# =============================================================================


class TestSpaces:
    """Test observation and action space implementations."""

    def test_observation_config(self):
        """Test observation configuration."""
        from tensornet.hypersim.spaces import (ObservationConfig,
                                               ObservationType)

        config = ObservationConfig(
            obs_type=ObservationType.SLICE_2D,
            resolution=(64, 64),
        )

        assert config.obs_type == ObservationType.SLICE_2D
        assert config.resolution == (64, 64)

    def test_field_observation_shape(self):
        """Test observation shape computation."""
        from tensornet.hypersim.spaces import (FieldObservation,
                                               ObservationConfig,
                                               ObservationType)

        config = ObservationConfig(
            obs_type=ObservationType.SLICE_2D,
            resolution=(64, 64),
            stack_frames=4,
        )

        obs = FieldObservation(config)
        shape = obs.shape

        assert len(shape) == 3  # (C * frames, H, W)
        assert shape[1] == 64
        assert shape[2] == 64

    def test_action_config(self):
        """Test action configuration."""
        import numpy as np

        from tensornet.hypersim.spaces import (ActionConfig, ActionType,
                                               ActuatorConfig)

        actuators = [
            ActuatorConfig("a1", np.array([0.25, 0.25, 0.5])),
            ActuatorConfig("a2", np.array([0.75, 0.75, 0.5])),
        ]

        config = ActionConfig(
            action_type=ActionType.CONTINUOUS,
            actuators=actuators,
        )

        assert len(config.actuators) == 2

    def test_field_action_shape(self):
        """Test action shape computation."""
        from tensornet.hypersim.spaces import (ActionConfig, ActionType,
                                               FieldAction)

        config = ActionConfig(action_type=ActionType.CONTINUOUS)
        action = FieldAction(config)

        shape = action.shape
        assert len(shape) == 1
        assert shape[0] > 0

    def test_action_to_forces(self):
        """Test action to force conversion."""
        from tensornet.hypersim.spaces import (ActionConfig, ActionType,
                                               FieldAction)

        config = ActionConfig(action_type=ActionType.CONTINUOUS)
        action_handler = FieldAction(config)

        # Create sample action
        action = np.random.randn(*action_handler.shape)

        # Convert to forces
        forces = action_handler.to_forces(action, grid_shape=(16, 16, 16))

        assert forces.shape == (16, 16, 16, 3)

    def test_action_mask(self, mock_field):
        """Test action masking."""
        import numpy as np

        from tensornet.hypersim.spaces import (ActionConfig, ActionMask,
                                               ActionType, ActuatorConfig)

        # Create config with explicit actuators
        actuators = [
            ActuatorConfig("a1", np.array([0.25, 0.25, 0.5])),
        ]
        config = ActionConfig(action_type=ActionType.CONTINUOUS, actuators=actuators)
        mask = ActionMask(config)

        m = mask.compute(mock_field)
        assert m is not None


# =============================================================================
# CURRICULUM TESTS
# =============================================================================


class TestCurriculum:
    """Test curriculum learning components."""

    def test_curriculum_stage(self):
        """Test curriculum stage configuration."""
        from tensornet.hypersim.curriculum import CurriculumStage

        stage = CurriculumStage(
            name="beginner",
            viscosity=0.1,
            max_steps=50,
        )

        assert stage.name == "beginner"
        assert stage.viscosity == 0.1
        assert stage.max_steps == 50

    def test_curriculum_progression(self):
        """Test curriculum advancement."""
        from tensornet.hypersim.curriculum import Curriculum, CurriculumStage

        stages = [
            CurriculumStage(
                "easy", viscosity=0.1, min_episodes=10, success_threshold=0.5
            ),
            CurriculumStage(
                "hard", viscosity=0.01, min_episodes=10, success_threshold=0.8
            ),
        ]

        curriculum = Curriculum(stages, seed=42)

        assert curriculum.stage_index == 0
        assert curriculum.current_stage.name == "easy"

        # Record successes
        for _ in range(15):
            curriculum.record_episode(success=True, reward=1.0)

        # Should have advanced
        assert curriculum.stage_index == 1

    def test_curriculum_get_options(self):
        """Test curriculum options generation."""
        from tensornet.hypersim.curriculum import Curriculum, CurriculumStage

        stages = [CurriculumStage("test", viscosity=0.05)]
        curriculum = Curriculum(stages, seed=42)

        options = curriculum.get_options()

        assert "curriculum_stage" in options
        assert "viscosity" in options

    def test_curriculum_state_dict(self):
        """Test curriculum serialization."""
        from tensornet.hypersim.curriculum import Curriculum, CurriculumStage

        stages = [CurriculumStage("test")]
        curriculum = Curriculum(stages)

        curriculum.record_episode(success=True)

        state = curriculum.state_dict()
        assert state["episode_count"] == 1

        # Reload
        curriculum2 = Curriculum(stages)
        curriculum2.load_state_dict(state)
        assert curriculum2._episode_count == 1

    def test_domain_randomizer(self):
        """Test domain randomization."""
        from tensornet.hypersim.curriculum import (DomainRandomizer,
                                                   RandomizationRange)

        ranges = [
            RandomizationRange("viscosity", 0.001, 0.1, log_scale=True),
            RandomizationRange("dt", 0.005, 0.02),
        ]

        randomizer = DomainRandomizer(ranges, seed=42)
        params = randomizer.sample()

        assert "viscosity" in params
        assert "dt" in params
        assert 0.001 <= params["viscosity"] <= 0.1
        assert 0.005 <= params["dt"] <= 0.02

    def test_difficulty_scheduler(self):
        """Test adaptive difficulty."""
        from tensornet.hypersim.curriculum import DifficultyScheduler

        scheduler = DifficultyScheduler(
            easy_params={"viscosity": 0.1},
            hard_params={"viscosity": 0.01},
            target_success_rate=0.7,
        )

        # Initially easy
        assert scheduler.difficulty == 0.0
        params = scheduler.get_params()
        assert params["viscosity"] == 0.1

        # Record many successes -> difficulty increases
        for _ in range(50):
            scheduler.record_episode(success=True)

        assert scheduler.difficulty > 0.0

    def test_make_fluid_curriculum(self):
        """Test preset curriculum factory."""
        from tensornet.hypersim.curriculum import make_fluid_curriculum

        curriculum = make_fluid_curriculum("standard")
        assert len(curriculum.stages) > 0

        curriculum_hard = make_fluid_curriculum("hard")
        assert len(curriculum_hard.stages) >= len(curriculum.stages)


# =============================================================================
# WRAPPER TESTS
# =============================================================================


@pytest.mark.skip(reason="FluidEnv not a gymnasium.Env - wrapper compatibility issue")
class TestWrappers:
    """Test environment wrappers."""

    def test_frame_stack(self, mock_gymnasium):
        """Test frame stacking wrapper."""
        from tensornet.hypersim.wrappers import FrameStack

        # Create mock env
        class MockEnv:
            observation_space = mock_gymnasium.spaces.Box(
                shape=(3, 64, 64),
                low=-np.inf,
                high=np.inf,
            )
            action_space = mock_gymnasium.spaces.Box(
                shape=(4,),
                low=-1,
                high=1,
            )

            def reset(self, **kwargs):
                return np.random.randn(3, 64, 64).astype(np.float32), {}

            def step(self, action):
                obs = np.random.randn(3, 64, 64).astype(np.float32)
                return obs, 0.0, False, False, {}

        env = MockEnv()
        wrapped = FrameStack(env, num_frames=4)

        assert wrapped.observation_space.shape == (12, 64, 64)

        obs, _ = wrapped.reset()
        assert obs.shape == (12, 64, 64)

    def test_action_repeat(self, mock_gymnasium):
        """Test action repeat wrapper."""
        from tensornet.hypersim.wrappers import ActionRepeat

        class MockEnv:
            observation_space = mock_gymnasium.spaces.Box(shape=(4,))
            action_space = mock_gymnasium.spaces.Box(shape=(2,))
            _step_count = 0

            def reset(self, **kwargs):
                self._step_count = 0
                return np.zeros(4), {}

            def step(self, action):
                self._step_count += 1
                return np.zeros(4), 1.0, False, False, {}

        env = MockEnv()
        wrapped = ActionRepeat(env, repeat=3)

        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(np.zeros(2))

        assert env._step_count == 3
        assert reward == 3.0  # Sum of rewards

    def test_reward_scaling(self, mock_gymnasium):
        """Test reward scaling wrapper."""
        from tensornet.hypersim.wrappers import RewardScaling

        class MockEnv:
            observation_space = mock_gymnasium.spaces.Box(shape=(4,))
            action_space = mock_gymnasium.spaces.Box(shape=(2,))

            def reset(self, **kwargs):
                return np.zeros(4), {}

            def step(self, action):
                return np.zeros(4), 1.0, False, False, {}

        env = MockEnv()
        wrapped = RewardScaling(env, scale=0.5)

        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(np.zeros(2))

        assert reward == 0.5

    def test_time_limit(self, mock_gymnasium):
        """Test time limit wrapper."""
        from tensornet.hypersim.wrappers import TimeLimit

        class MockEnv:
            observation_space = mock_gymnasium.spaces.Box(shape=(4,))
            action_space = mock_gymnasium.spaces.Box(shape=(2,))

            def reset(self, **kwargs):
                return np.zeros(4), {}

            def step(self, action):
                return np.zeros(4), 0.0, False, False, {}

        env = MockEnv()
        wrapped = TimeLimit(env, max_steps=5)

        wrapped.reset()

        for i in range(5):
            _, _, _, truncated, info = wrapped.step(np.zeros(2))
            if i < 4:
                assert not truncated
            else:
                assert truncated
                assert "TimeLimit.truncated" in info

    def test_record_episode(self, mock_gymnasium):
        """Test episode recording."""
        from tensornet.hypersim.wrappers import RecordEpisode

        class MockEnv:
            observation_space = mock_gymnasium.spaces.Box(shape=(4,))
            action_space = mock_gymnasium.spaces.Box(shape=(2,))

            def reset(self, **kwargs):
                return np.zeros(4), {}

            def step(self, action):
                return np.random.randn(4).astype(np.float32), 1.0, False, False, {}

        env = MockEnv()
        wrapped = RecordEpisode(env, buffer_size=10)

        wrapped.reset()
        for _ in range(5):
            wrapped.step(np.zeros(2))

        # Finalize episode by starting new one
        wrapped.reset()

        assert len(wrapped.episodes) == 1
        ep = wrapped.episodes[0]
        assert ep.length == 5
        assert ep.total_reward == 5.0

    def test_make_wrapped_env(self, mock_gymnasium):
        """Test wrapper factory."""
        from tensornet.hypersim.wrappers import make_wrapped_env

        class MockEnv:
            observation_space = mock_gymnasium.spaces.Box(shape=(3, 64, 64))
            action_space = mock_gymnasium.spaces.Box(shape=(4,), low=-1, high=1)

            def reset(self, **kwargs):
                return np.random.randn(3, 64, 64).astype(np.float32), {}

            def step(self, action):
                return (
                    np.random.randn(3, 64, 64).astype(np.float32),
                    1.0,
                    False,
                    False,
                    {},
                )

        env = MockEnv()
        wrapped = make_wrapped_env(
            env,
            frame_stack=4,
            action_repeat=2,
            reward_scale=0.1,
            time_limit=100,
        )

        obs, _ = wrapped.reset()
        # Frame stacking applied
        assert obs.shape[0] == 12


# =============================================================================
# REGISTRY TESTS
# =============================================================================


class TestRegistry:
    """Test environment registry."""

    def test_register_and_make(self):
        """Test environment registration."""
        from tensornet.hypersim.env import FluidEnv
        from tensornet.hypersim.registry import (list_envs, make_env,
                                                 register_env, unregister_env)

        # Register custom env
        env_id = "test-custom-v0"
        try:
            register_env(
                env_id,
                FluidEnv,
                default_config={"ndim": 2, "bits_per_dim": 5},
            )

            assert env_id in list_envs()

            # Make env
            env = make_env(env_id)
            assert env is not None

        finally:
            unregister_env(env_id)

    def test_list_envs(self):
        """Test listing environments."""
        from tensornet.hypersim.registry import list_envs

        envs = list_envs()
        assert len(envs) > 0
        assert "fluid-control-v0" in envs

    def test_get_env_spec(self):
        """Test getting environment spec."""
        from tensornet.hypersim.registry import get_env_spec

        spec = get_env_spec("fluid-control-v0")
        assert "entry_point" in spec
        assert "default_config" in spec

    def test_preset_envs_exist(self):
        """Test preset environments are registered."""
        from tensornet.hypersim.registry import list_envs

        envs = list_envs()

        expected = [
            "fluid-control-v0",
            "fluid-control-easy-v0",
            "fluid-control-hard-v0",
        ]

        for env_id in expected:
            assert env_id in envs, f"Missing preset: {env_id}"

    @pytest.mark.skip(
        reason="FluidEnv not a gymnasium.Env - wrapper compatibility issue"
    )
    def test_make_fluid_env_convenience(self):
        """Test convenience function."""
        from tensornet.hypersim.registry import make_fluid_env

        env = make_fluid_env(difficulty="easy", ndim=2)
        assert env is not None

    def test_task_specs(self):
        """Test task specifications."""
        from tensornet.hypersim.registry import TASK_SPECS, TaskSpec

        assert "target_field" in TASK_SPECS

        spec = TASK_SPECS["target_field"]
        assert isinstance(spec, TaskSpec)
        assert spec.name == "target_field"


# =============================================================================
# ENV TESTS (Core FluidEnv)
# =============================================================================


class TestFluidEnv:
    """Test FluidEnv implementation."""

    def test_env_config(self):
        """Test environment configuration."""
        from tensornet.hypersim.env import FluidEnvConfig

        config = FluidEnvConfig(
            dims=2,
            bits_per_dim=6,
            rank=8,
            viscosity=0.01,
        )

        assert config.dims == 2
        assert config.bits_per_dim == 6

    def test_env_state(self):
        """Test environment state serialization."""
        import torch

        from tensornet.hypersim.env import EnvState

        cores = [torch.randn(8, 2, 8) for _ in range(12)]
        state = EnvState(
            cores=[c.numpy() for c in cores],
            step_count=10,
            total_reward=5.0,
            np_random_state={"bit_generator": "PCG64"},
        )

        assert state.step_count == 10
        assert len(state.cores) == 12
        assert state.total_reward == 5.0

        # Hash can be computed
        h = state.compute_hash()
        assert h is not None
        assert len(h) > 0

    def test_step_result(self):
        """Test step result structure."""
        import numpy as np

        from tensornet.hypersim.env import StepResult

        result = StepResult(
            observation=np.zeros((3, 64, 64)),
            reward=0.5,
            terminated=False,
            truncated=False,
            info={"step": 1},
        )

        # Access as attributes
        assert result.reward == 0.5
        assert not result.terminated


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for HyperSim."""

    def test_import_all(self):
        """Test all exports are importable."""
        from tensornet.hypersim import (  # Environment; Rewards; Spaces; Curriculum; Wrappers; Registry
            ActionMask, ActionRepeat, CompositeReward, Curriculum,
            CurriculumStage, DenseReward, DomainRandomizer, EnvState,
            FieldAction, FieldObservation, FluidEnv, FluidEnvConfig,
            FrameStack, RewardFunction, SparseReward, StepResult, TimeLimit,
            list_envs, make_env, make_reward, register_env)

    def test_full_workflow(self, mock_field):
        """Test complete RL workflow."""
        from tensornet.hypersim.curriculum import Curriculum, CurriculumStage
        from tensornet.hypersim.rewards import DenseReward
        from tensornet.hypersim.spaces import (FieldObservation,
                                               ObservationConfig)

        # Setup curriculum
        curriculum = Curriculum(
            [
                CurriculumStage("test", min_episodes=5),
            ]
        )

        # Setup reward
        reward_fn = DenseReward(lambda f, s: 1.0)

        # Setup observation
        obs_config = ObservationConfig()
        obs_extractor = FieldObservation(obs_config)

        # Simulate episodes
        for episode in range(5):
            options = curriculum.get_options()

            for step in range(10):
                reward = reward_fn(mock_field, step)
                assert reward == 1.0

            curriculum.record_episode(success=True)

        assert curriculum._episode_count == 5

    def test_reward_curriculum_integration(self, mock_field):
        """Test reward with curriculum difficulty."""
        from tensornet.hypersim.curriculum import DifficultyScheduler
        from tensornet.hypersim.rewards import DenseReward, RewardConfig

        scheduler = DifficultyScheduler(
            easy_params={"reward_scale": 2.0},
            hard_params={"reward_scale": 0.5},
        )

        # Get current params
        params = scheduler.get_params()

        config = RewardConfig(scale=params["reward_scale"])
        reward = DenseReward(lambda f, s: 1.0, config=config)

        r = reward(mock_field, 0)
        assert r == 2.0  # Easy mode: scaled by 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
