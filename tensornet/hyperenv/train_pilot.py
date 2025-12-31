"""
Sovereign Pilot Training Script
================================

Train a PPO agent to fly the hypersonic "Safety Tube" at Mach 10.

This is the DOJO - where the AI learns to survive physics
that would kill a human pilot.

Usage:
    python train_pilot.py                    # Default training (1M steps)
    python train_pilot.py --timesteps 100000 # Quick test
    python train_pilot.py --eval             # Evaluate saved model

Requirements:
    pip install stable-baselines3[extra] tensorboard gymnasium
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add tensornet to path
TENSORNET_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TENSORNET_ROOT))


def train_wingman(
    total_timesteps: int = 1_000_000,
    save_path: str = "sovereign_pilot_v1",
    log_dir: str = "./logs/",
    seed: int = 42,
    verbose: int = 1,
):
    """
    Train a PPO agent to fly the hypersonic trajectory.
    
    Args:
        total_timesteps: Number of training steps
        save_path: Path to save the trained model
        log_dir: TensorBoard log directory
        seed: Random seed
        verbose: Verbosity level
    """
    # Import here to allow --help without deps
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            EvalCallback,
            CheckpointCallback,
            CallbackList,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3[extra]")
        sys.exit(1)
    
    from tensornet.hyperenv.hypersonic_env import HypersonicEnv
    
    print("=" * 60)
    print("SOVEREIGN PILOT TRAINING")
    print("=" * 60)
    print(f"Target: Survive Mach 10 flight for {total_timesteps:,} steps")
    print(f"Model: PPO (Proximal Policy Optimization)")
    print(f"Log dir: {log_dir}")
    print()
    
    # Create environment
    def make_env(rank: int = 0):
        def _init():
            env = HypersonicEnv(
                config={
                    "mach": 10.0,
                    "turbulence_level": "high",
                    "max_steps": 1000,
                }
            )
            env = Monitor(env)
            return env
        return _init
    
    # Vectorized environment for efficiency
    print("[DOJO] Initializing training environment...")
    env = DummyVecEnv([make_env(i) for i in range(4)])  # 4 parallel envs
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create the agent
    print("[DOJO] Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=verbose,
        tensorboard_log=log_dir,
        seed=seed,
        device="auto",  # Use GPU if available
    )
    
    print(f"[DOJO] Policy network: {model.policy}")
    print(f"[DOJO] Using device: {model.device}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="sovereign_pilot",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train!
    print()
    print("=" * 60)
    print("[DOJO] Training started.")
    print(f"[DOJO] Target: Survival at Mach 10 in high turbulence.")
    print(f"[DOJO] Total timesteps: {total_timesteps:,}")
    print()
    print("Run `tensorboard --logdir ./logs` to monitor training.")
    print("=" * 60)
    print()
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    # Save final model
    model.save(save_path)
    env.save(f"{save_path}_vecnormalize.pkl")
    
    print()
    print("=" * 60)
    print("[DOJO] Training complete!")
    print(f"[DOJO] Duration: {training_duration}")
    print(f"[DOJO] Model saved to: {save_path}.zip")
    print(f"[DOJO] Normalization saved to: {save_path}_vecnormalize.pkl")
    print("=" * 60)
    
    return model


def evaluate_pilot(
    model_path: str = "sovereign_pilot_v1",
    n_episodes: int = 10,
    render: bool = False,
):
    """
    Evaluate a trained pilot.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        sys.exit(1)
    
    from tensornet.hyperenv.hypersonic_env import HypersonicEnv
    
    print("=" * 60)
    print("SOVEREIGN PILOT EVALUATION")
    print("=" * 60)
    
    # Load model
    print(f"[EVAL] Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create eval environment
    env = DummyVecEnv([lambda: HypersonicEnv(config={"mach": 10.0, "turbulence_level": "high"})])
    
    # Try to load normalization stats
    norm_path = f"{model_path}_vecnormalize.pkl"
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    survival_times = []
    crash_reasons = {}
    
    print(f"\n[EVAL] Running {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if render:
                env.render()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        survival_times.append(steps * 0.1)  # dt = 0.1
        
        # Track crash reasons
        reason = info[0].get('episode', {}).get('crash_reason', 'completed')
        crash_reasons[reason] = crash_reasons.get(reason, 0) + 1
        
        print(f"  Episode {episode + 1}: Reward={total_reward:.1f}, Steps={steps}, Reason={reason}")
    
    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {sum(episode_rewards) / n_episodes:.2f}")
    print(f"Mean steps: {sum(episode_lengths) / n_episodes:.1f}")
    print(f"Mean survival time: {sum(survival_times) / n_episodes:.1f}s")
    print(f"\nCrash reasons:")
    for reason, count in crash_reasons.items():
        print(f"  {reason}: {count}")
    
    env.close()
    
    return {
        'mean_reward': sum(episode_rewards) / n_episodes,
        'mean_steps': sum(episode_lengths) / n_episodes,
        'crash_reasons': crash_reasons,
    }


def quick_test():
    """Run a quick test to verify environment works."""
    from tensornet.hyperenv.hypersonic_env import HypersonicEnv
    
    print("=" * 60)
    print("QUICK TEST: Random Agent in HypersonicEnv")
    print("=" * 60)
    
    env = HypersonicEnv(config={"mach": 10.0, "turbulence_level": "high"})
    
    print(f"\nObservation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    step_rewards = []
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_rewards.append(reward)
        
        if terminated or truncated:
            print(f"\nEpisode terminated at step {step + 1}")
            print(f"Reason: {info['episode'].get('crash_reason', 'timeout')}")
            break
    
    print(f"\n{'=' * 40}")
    print("RANDOM AGENT RESULTS")
    print(f"{'=' * 40}")
    print(f"Steps: {len(step_rewards)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Mean step reward: {total_reward / len(step_rewards):.4f}")
    print(f"Final altitude: {info['aircraft']['altitude']:.0f} m")
    print(f"Final speed: {info['aircraft']['speed']:.0f} m/s")
    
    env.close()
    
    print("\n✓ Exit Gate: Random agent logged reward score")
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sovereign Pilot")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate saved model instead of training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test with random agent",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sovereign_pilot_v1",
        help="Model save/load path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    elif args.eval:
        evaluate_pilot(model_path=args.model)
    else:
        train_wingman(
            total_timesteps=args.timesteps,
            save_path=args.model,
            seed=args.seed,
        )
