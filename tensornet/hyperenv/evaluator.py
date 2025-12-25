"""
Evaluation Infrastructure
=========================

Systematic evaluation of trained policies.

Features:
- Standardized evaluation protocols
- Benchmark suites
- Statistical analysis
- Comparison utilities
"""

from __future__ import annotations

import os
import time
import json
import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from pathlib import Path

from .agent import Agent


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of policy evaluation."""
    
    # Core metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    
    # Episode statistics
    mean_length: float = 0.0
    std_length: float = 0.0
    n_episodes: int = 0
    
    # Success metrics
    success_rate: float = 0.0
    
    # Timing
    total_time: float = 0.0
    fps: float = 0.0
    
    # Raw data
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_successes: List[bool] = field(default_factory=list)
    
    # Metadata
    agent_name: str = ""
    env_name: str = ""
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "mean_length": self.mean_length,
            "std_length": self.std_length,
            "n_episodes": self.n_episodes,
            "success_rate": self.success_rate,
            "total_time": self.total_time,
            "fps": self.fps,
            "agent_name": self.agent_name,
            "env_name": self.env_name,
            "timestamp": self.timestamp,
        }
    
    def save(self, path: str):
        """Save result to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EvaluationResult':
        """Load result from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ComparisonResult:
    """Result of comparing multiple agents."""
    
    agent_names: List[str] = field(default_factory=list)
    results: Dict[str, EvaluationResult] = field(default_factory=dict)
    
    # Statistical comparisons
    rankings: Dict[str, int] = field(default_factory=dict)
    p_values: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    def get_best(self) -> str:
        """Get best agent by mean reward."""
        best = None
        best_reward = float('-inf')
        for name, result in self.results.items():
            if result.mean_reward > best_reward:
                best_reward = result.mean_reward
                best = name
        return best


# =============================================================================
# EVALUATOR
# =============================================================================

class Evaluator:
    """
    Policy evaluator for systematic testing.
    
    Features:
    - Multiple evaluation episodes
    - Statistical analysis
    - Video recording
    - Deterministic vs stochastic evaluation
    
    Example:
        evaluator = Evaluator(env, n_episodes=100)
        result = evaluator.evaluate(agent)
        print(f"Mean reward: {result.mean_reward:.2f}")
    """
    
    def __init__(
        self,
        env: Any,
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        deterministic: bool = True,
        render: bool = False,
        record_video: bool = False,
        video_path: Optional[str] = None,
        success_threshold: Optional[float] = None,
    ):
        self.env = env
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.render = render
        self.record_video = record_video
        self.video_path = video_path
        self.success_threshold = success_threshold
    
    def evaluate(
        self,
        agent: Agent,
        seed: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate agent on environment.
        
        Args:
            agent: Agent to evaluate
            seed: Random seed for reproducibility
            
        Returns:
            EvaluationResult with statistics
        """
        agent.eval()
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        
        total_steps = 0
        start_time = time.time()
        
        for ep in range(self.n_episodes):
            # Reset with seed
            ep_seed = seed + ep if seed is not None else None
            obs = self._reset_env(ep_seed)
            
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                # Get action
                action = agent.act(obs, deterministic=self.deterministic)
                
                # Step
                obs, reward, done, info = self._step_env(action)
                
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Check max steps
                if self.max_steps and episode_length >= self.max_steps:
                    done = True
                
                # Render
                if self.render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check success
            success = False
            if self.success_threshold is not None:
                success = episode_reward >= self.success_threshold
            elif 'is_success' in info:
                success = info['is_success']
            episode_successes.append(success)
        
        total_time = time.time() - start_time
        
        # Compute statistics
        result = EvaluationResult(
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            min_reward=float(np.min(episode_rewards)),
            max_reward=float(np.max(episode_rewards)),
            mean_length=float(np.mean(episode_lengths)),
            std_length=float(np.std(episode_lengths)),
            n_episodes=self.n_episodes,
            success_rate=float(np.mean(episode_successes)),
            total_time=total_time,
            fps=total_steps / total_time if total_time > 0 else 0,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            episode_successes=episode_successes,
            agent_name=agent.name,
            env_name=getattr(self.env, 'name', 'unknown'),
            timestamp=time.time(),
        )
        
        agent.train()
        return result
    
    def compare(
        self,
        agents: List[Agent],
        seed: Optional[int] = None,
    ) -> ComparisonResult:
        """
        Compare multiple agents.
        
        Args:
            agents: List of agents to compare
            seed: Random seed
            
        Returns:
            ComparisonResult with rankings
        """
        results = {}
        for agent in agents:
            results[agent.name] = self.evaluate(agent, seed=seed)
        
        # Rank by mean reward
        sorted_agents = sorted(
            results.keys(),
            key=lambda x: results[x].mean_reward,
            reverse=True,
        )
        rankings = {name: i + 1 for i, name in enumerate(sorted_agents)}
        
        return ComparisonResult(
            agent_names=[a.name for a in agents],
            results=results,
            rankings=rankings,
        )
    
    def _reset_env(self, seed: Optional[int]) -> np.ndarray:
        """Reset environment."""
        if seed is not None:
            result = self.env.reset(seed=seed)
        else:
            result = self.env.reset()
        
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def _step_env(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment."""
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            done = done or truncated
        else:
            obs, reward, done, info = result
        return obs, reward, done, info


# =============================================================================
# BENCHMARK SUITE
# =============================================================================

class BenchmarkSuite:
    """
    Collection of environments for systematic benchmarking.
    
    Example:
        suite = BenchmarkSuite()
        suite.add_env("smoke_easy", make_env("smoke-v0", difficulty=0.3))
        suite.add_env("smoke_hard", make_env("smoke-v0", difficulty=0.9))
        
        results = suite.run(agent)
        suite.print_summary(results)
    """
    
    def __init__(
        self,
        n_episodes_per_env: int = 10,
        deterministic: bool = True,
    ):
        self.n_episodes = n_episodes_per_env
        self.deterministic = deterministic
        
        self._envs: Dict[str, Any] = {}
        self._evaluators: Dict[str, Evaluator] = {}
    
    def add_env(
        self,
        name: str,
        env: Any,
        n_episodes: Optional[int] = None,
        success_threshold: Optional[float] = None,
    ):
        """Add environment to benchmark suite."""
        self._envs[name] = env
        self._evaluators[name] = Evaluator(
            env=env,
            n_episodes=n_episodes or self.n_episodes,
            deterministic=self.deterministic,
            success_threshold=success_threshold,
        )
    
    def run(
        self,
        agent: Agent,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, EvaluationResult]:
        """
        Run all benchmarks.
        
        Args:
            agent: Agent to evaluate
            seed: Random seed
            verbose: Print progress
            
        Returns:
            Dictionary of results per environment
        """
        results = {}
        
        for name, evaluator in self._evaluators.items():
            if verbose:
                print(f"Evaluating on {name}...")
            
            results[name] = evaluator.evaluate(agent, seed=seed)
            
            if verbose:
                print(f"  Mean reward: {results[name].mean_reward:.2f} "
                      f"+/- {results[name].std_reward:.2f}")
        
        return results
    
    def run_comparison(
        self,
        agents: List[Agent],
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, ComparisonResult]:
        """
        Compare agents across all benchmarks.
        
        Returns:
            Dictionary mapping env_name -> ComparisonResult
        """
        results = {}
        
        for name, evaluator in self._evaluators.items():
            if verbose:
                print(f"\nBenchmark: {name}")
            
            results[name] = evaluator.compare(agents, seed=seed)
            
            if verbose:
                for agent_name in results[name].agent_names:
                    r = results[name].results[agent_name]
                    rank = results[name].rankings[agent_name]
                    print(f"  #{rank} {agent_name}: {r.mean_reward:.2f}")
        
        return results
    
    def print_summary(
        self,
        results: Dict[str, EvaluationResult],
    ):
        """Print summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Mean Reward: {result.mean_reward:.2f} +/- {result.std_reward:.2f}")
            print(f"  Success Rate: {result.success_rate:.1%}")
            print(f"  Mean Length: {result.mean_length:.1f}")
            print(f"  FPS: {result.fps:.0f}")
        
        # Aggregate
        all_rewards = [r.mean_reward for r in results.values()]
        all_success = [r.success_rate for r in results.values()]
        
        print("\n" + "-" * 60)
        print(f"Overall Mean Reward: {np.mean(all_rewards):.2f}")
        print(f"Overall Success Rate: {np.mean(all_success):.1%}")
        print("=" * 60)
    
    def save_results(
        self,
        results: Dict[str, EvaluationResult],
        path: str,
    ):
        """Save all results to directory."""
        os.makedirs(path, exist_ok=True)
        
        for name, result in results.items():
            result.save(os.path.join(path, f"{name}.json"))
        
        # Save summary
        summary = {
            name: result.to_dict() for name, result in results.items()
        }
        with open(os.path.join(path, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# FACTORY
# =============================================================================

def make_evaluator(
    env: Any,
    **kwargs,
) -> Evaluator:
    """
    Factory for creating evaluators.
    
    Args:
        env: Environment to evaluate on
        **kwargs: Evaluator arguments
        
    Returns:
        Evaluator instance
    """
    return Evaluator(env, **kwargs)


def make_benchmark_suite(
    env_factories: Dict[str, Callable[[], Any]],
    **kwargs,
) -> BenchmarkSuite:
    """
    Factory for creating benchmark suites.
    
    Args:
        env_factories: Dictionary mapping name -> env factory function
        **kwargs: BenchmarkSuite arguments
        
    Returns:
        BenchmarkSuite with environments added
    """
    suite = BenchmarkSuite(**kwargs)
    
    for name, factory in env_factories.items():
        suite.add_env(name, factory())
    
    return suite
