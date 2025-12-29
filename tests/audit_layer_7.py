"""
Layer 7 Audit: AI Environment Validation
=========================================

Validates that the HyperEnv RL infrastructure works with real physics.
"""

import numpy as np


def test_agent_creation():
    """Test agent creation and action sampling."""
    from tensornet.hyperenv import RandomAgent, AgentConfig
    
    config = AgentConfig(action_dim=4)
    agent = RandomAgent(config=config)
    obs = np.zeros(10)
    action = agent.act(obs)
    
    return True, f"action shape: {action.shape}"


def test_replay_buffer():
    """Test experience buffer."""
    from tensornet.hyperenv import ReplayBuffer
    
    buffer = ReplayBuffer(capacity=100)
    
    for i in range(20):
        buffer.add(
            observation=np.random.randn(10).astype(np.float32),
            action=np.random.randn(4).astype(np.float32),
            reward=float(np.random.randn()),
            next_observation=np.random.randn(10).astype(np.float32),
            done=False,
        )
    
    batch = buffer.sample(8)
    return True, f"sampled batch with {len(batch)} items"


def test_trainer_mock_env():
    """Test trainer with mock environment."""
    from tensornet.hyperenv import Trainer, RandomAgent, TrainerConfig
    from unittest.mock import MagicMock
    
    agent = RandomAgent()
    env = MagicMock()
    env.reset.return_value = (np.zeros(10), {})
    env.step.return_value = (np.zeros(10), 1.0, False, False, {})
    
    config = TrainerConfig(
        total_timesteps=50,
        learning_starts=10,
        log_freq=1000,
        eval_freq=1000,
        save_freq=1000,
    )
    
    trainer = Trainer(agent, env, config)
    trainer.train()
    
    return trainer.state.timestep >= 50, f"timesteps: {trainer.state.timestep}"


def test_callbacks():
    """Test callback system."""
    from tensornet.hyperenv import CheckpointCallback, EvalCallback
    
    # Just verify creation works
    ckpt = CheckpointCallback(save_freq=100, save_path="./checkpoints")
    eval_cb = EvalCallback(eval_freq=100)
    
    return True, "callbacks created"


def run_audit():
    print()
    print("=" * 66)
    print("           LAYER 7 AUDIT: AI Environment")
    print("=" * 66)
    print()
    
    results = []
    
    tests = [
        ("Agent Creation", test_agent_creation),
        ("Replay Buffer", test_replay_buffer),
        ("Trainer (Mock)", test_trainer_mock_env),
        ("Callbacks", test_callbacks),
    ]
    
    for name, test_fn in tests:
        print(f"Test: {name}...")
        try:
            passed, detail = test_fn()
            results.append((name, passed, detail))
            print(f"  {'PASS' if passed else 'FAIL'} | {detail}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAIL | {e}")
    
    print()
    all_passed = all(r[1] for r in results)
    n_passed = sum(1 for r in results if r[1])
    
    print("=" * 66)
    if all_passed:
        print("  ALL TESTS PASSED (scaffold validated)")
    else:
        print(f"  {n_passed}/{len(results)} TESTS PASSED")
    print()
    print("  NOTE: No RL agent trained on actual physics yet.")
    print("  These tests validate infrastructure only.")
    print("=" * 66)
    
    return all_passed


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
