"""
Multi-Agent Environments
========================

Multi-agent wrappers and coordination for HyperSim environments.

Supports:
- Cooperative teams
- Competitive scenarios  
- Mixed cooperative-competitive
- Communication channels
"""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from enum import Enum

from .agent import Agent, AgentConfig, AgentState


# =============================================================================
# TYPES
# =============================================================================

class AgentRole(Enum):
    """Role of an agent in multi-agent environment."""
    COOPERATIVE = "cooperative"     # Works with other agents
    COMPETITIVE = "competitive"     # Competes against other agents
    NEUTRAL = "neutral"             # Independent, neither helps nor hinders
    ADVERSARY = "adversary"         # Actively works against others


@dataclass
class TeamConfig:
    """Configuration for a team of agents."""
    name: str = "team_0"
    size: int = 1
    role: AgentRole = AgentRole.COOPERATIVE
    
    # Reward sharing
    share_rewards: bool = True
    reward_scale: float = 1.0
    
    # Communication
    can_communicate: bool = False
    message_dim: int = 8
    
    # Observation
    observe_teammates: bool = True
    observe_opponents: bool = False


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent environment."""
    
    # Teams
    teams: List[TeamConfig] = field(default_factory=lambda: [TeamConfig()])
    
    # Global settings
    max_steps: int = 1000
    simultaneous_actions: bool = True  # All agents act at same time
    
    # Observation
    centralized_obs: bool = False  # Whether to provide global observation
    
    # Communication
    broadcast_messages: bool = False
    message_buffer_size: int = 10


# =============================================================================
# MULTI-AGENT STATE
# =============================================================================

@dataclass
class MultiAgentState:
    """State of multi-agent environment."""
    step: int = 0
    episode: int = 0
    
    # Per-agent tracking
    agent_rewards: Dict[str, float] = field(default_factory=dict)
    agent_dones: Dict[str, bool] = field(default_factory=dict)
    
    # Communication
    messages: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    
    # Team statistics
    team_rewards: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# MULTI-AGENT ENVIRONMENT
# =============================================================================

class MultiAgentEnv:
    """
    Multi-agent environment wrapper.
    
    Wraps a base environment and manages multiple agents,
    handling observation/action routing and reward distribution.
    
    Example:
        env = FluidEnv(config)
        multi_env = MultiAgentEnv(env, MultiAgentConfig(
            teams=[
                TeamConfig("team_a", size=2, role=AgentRole.COOPERATIVE),
                TeamConfig("team_b", size=1, role=AgentRole.COMPETITIVE),
            ]
        ))
        
        observations = multi_env.reset()
        
        while not multi_env.all_done:
            actions = {
                agent_id: agent.act(obs)
                for agent_id, obs in observations.items()
            }
            observations, rewards, dones, infos = multi_env.step(actions)
    """
    
    def __init__(
        self,
        env: Any,  # Base environment
        config: Optional[MultiAgentConfig] = None,
    ):
        self.env = env
        self.config = config or MultiAgentConfig()
        self.state = MultiAgentState()
        
        # Initialize agents
        self._agents: Dict[str, Any] = {}
        self._agent_teams: Dict[str, str] = {}
        
        # Create agent IDs
        self._agent_ids = []
        for team in self.config.teams:
            for i in range(team.size):
                agent_id = f"{team.name}_agent_{i}"
                self._agent_ids.append(agent_id)
                self._agent_teams[agent_id] = team.name
        
        # Initialize state
        for agent_id in self._agent_ids:
            self.state.agent_rewards[agent_id] = 0.0
            self.state.agent_dones[agent_id] = False
            self.state.messages[agent_id] = []
        
        for team in self.config.teams:
            self.state.team_rewards[team.name] = 0.0
    
    @property
    def agent_ids(self) -> List[str]:
        """List of all agent IDs."""
        return self._agent_ids
    
    @property
    def num_agents(self) -> int:
        """Total number of agents."""
        return len(self._agent_ids)
    
    @property
    def all_done(self) -> bool:
        """Whether all agents are done."""
        return all(self.state.agent_dones.values())
    
    def get_team(self, agent_id: str) -> str:
        """Get team name for agent."""
        return self._agent_teams[agent_id]
    
    def get_team_config(self, team_name: str) -> TeamConfig:
        """Get configuration for team."""
        for team in self.config.teams:
            if team.name == team_name:
                return team
        raise ValueError(f"Unknown team: {team_name}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Reset environment and return initial observations.
        
        Returns:
            Dictionary mapping agent_id -> observation
        """
        # Reset base environment
        if hasattr(self.env, 'reset'):
            base_obs = self.env.reset(seed=seed, options=options)
            if isinstance(base_obs, tuple):
                base_obs = base_obs[0]
        else:
            base_obs = None
        
        # Reset state
        self.state = MultiAgentState()
        for agent_id in self._agent_ids:
            self.state.agent_rewards[agent_id] = 0.0
            self.state.agent_dones[agent_id] = False
            self.state.messages[agent_id] = []
        
        for team in self.config.teams:
            self.state.team_rewards[team.name] = 0.0
        
        # Generate per-agent observations
        return self._get_observations(base_obs)
    
    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],        # rewards
        Dict[str, bool],         # dones
        Dict[str, bool],         # truncated
        Dict[str, Dict],         # infos
    ]:
        """
        Step all agents.
        
        Args:
            actions: Dictionary mapping agent_id -> action
            
        Returns:
            observations, rewards, dones, truncated, infos
        """
        self.state.step += 1
        
        # Combine actions for base environment
        combined_action = self._combine_actions(actions)
        
        # Step base environment
        if hasattr(self.env, 'step'):
            result = self.env.step(combined_action)
            if len(result) == 5:
                base_obs, base_reward, base_done, base_truncated, base_info = result
            else:
                base_obs, base_reward, base_done, base_info = result
                base_truncated = False
        else:
            base_obs = None
            base_reward = 0.0
            base_done = False
            base_truncated = False
            base_info = {}
        
        # Distribute rewards
        rewards = self._distribute_rewards(base_reward, actions)
        
        # Update dones
        dones = {}
        truncated = {}
        for agent_id in self._agent_ids:
            dones[agent_id] = base_done or self.state.step >= self.config.max_steps
            truncated[agent_id] = base_truncated
            self.state.agent_dones[agent_id] = dones[agent_id]
        
        # Generate observations
        observations = self._get_observations(base_obs)
        
        # Build infos
        infos = {}
        for agent_id in self._agent_ids:
            infos[agent_id] = {
                "team": self.get_team(agent_id),
                "step": self.state.step,
                **base_info,
            }
        
        return observations, rewards, dones, truncated, infos
    
    def _get_observations(
        self,
        base_obs: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Generate per-agent observations."""
        observations = {}
        
        for agent_id in self._agent_ids:
            team = self.get_team(agent_id)
            team_config = self.get_team_config(team)
            
            if base_obs is not None:
                # Start with base observation
                obs = base_obs.copy() if isinstance(base_obs, np.ndarray) else base_obs
            else:
                # Dynamic fallback: infer shape from observation space or use sensible default
                if hasattr(self, 'observation_space') and self.observation_space is not None:
                    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                elif hasattr(self, '_base_env') and hasattr(self._base_env, 'observation_space'):
                    obs = np.zeros(self._base_env.observation_space.shape, dtype=np.float32)
                else:
                    # Minimal default for unknown environments
                    obs = np.zeros((64, 64, 3), dtype=np.float32)
            
            # Add agent-specific info
            if isinstance(obs, np.ndarray) and obs.ndim == 3:
                # Could add agent position markers, etc.
                pass
            
            observations[agent_id] = obs
        
        return observations
    
    def _combine_actions(
        self,
        actions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine agent actions for base environment."""
        if not actions:
            return np.zeros(4, dtype=np.float32)
        
        # For cooperative agents, average actions
        # For competitive, could use different strategies
        combined = np.zeros_like(list(actions.values())[0], dtype=np.float32)
        
        cooperative_count = 0
        for agent_id, action in actions.items():
            team = self.get_team(agent_id)
            team_config = self.get_team_config(team)
            
            if team_config.role == AgentRole.COOPERATIVE:
                combined = combined + action
                cooperative_count += 1
        
        if cooperative_count > 0:
            combined = combined / cooperative_count
        
        return combined
    
    def _distribute_rewards(
        self,
        base_reward: float,
        actions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Distribute base reward to agents."""
        rewards = {}
        
        for agent_id in self._agent_ids:
            team = self.get_team(agent_id)
            team_config = self.get_team_config(team)
            
            if team_config.role == AgentRole.COOPERATIVE:
                # Cooperative agents share reward
                reward = base_reward * team_config.reward_scale
            elif team_config.role == AgentRole.COMPETITIVE:
                # Competitive agents get negative of base reward
                reward = -base_reward * team_config.reward_scale
            elif team_config.role == AgentRole.ADVERSARY:
                # Adversaries try to minimize base reward
                reward = -base_reward * team_config.reward_scale
            else:
                # Neutral agents get their own reward
                reward = base_reward * team_config.reward_scale
            
            rewards[agent_id] = reward
            self.state.agent_rewards[agent_id] += reward
            self.state.team_rewards[team] += reward
        
        return rewards
    
    # =========================================================================
    # COMMUNICATION
    # =========================================================================
    
    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: np.ndarray,
    ):
        """
        Send message between agents.
        
        Only works if agents can communicate.
        """
        from_team = self.get_team(from_agent)
        to_team = self.get_team(to_agent)
        
        from_config = self.get_team_config(from_team)
        to_config = self.get_team_config(to_team)
        
        # Check if communication is allowed
        can_send = from_config.can_communicate
        can_receive = to_config.can_communicate
        
        if from_team == to_team:
            # Same team communication
            allowed = can_send and can_receive
        else:
            # Cross-team communication (maybe restricted)
            allowed = can_send and can_receive and self.config.broadcast_messages
        
        if allowed:
            if len(self.state.messages[to_agent]) >= self.config.message_buffer_size:
                self.state.messages[to_agent].pop(0)
            self.state.messages[to_agent].append(message)
    
    def get_messages(self, agent_id: str) -> List[np.ndarray]:
        """Get messages for agent."""
        return self.state.messages.get(agent_id, [])
    
    def clear_messages(self, agent_id: str):
        """Clear message buffer for agent."""
        self.state.messages[agent_id] = []
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_agent_observation(
        self,
        agent_id: str,
        base_obs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get observation for specific agent."""
        return self._get_observations(base_obs).get(agent_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get multi-agent statistics."""
        return {
            "step": self.state.step,
            "episode": self.state.episode,
            "agent_rewards": dict(self.state.agent_rewards),
            "team_rewards": dict(self.state.team_rewards),
            "all_done": self.all_done,
        }


# =============================================================================
# SELF-PLAY ENVIRONMENT
# =============================================================================

class SelfPlayEnv(MultiAgentEnv):
    """
    Self-play environment where an agent plays against copies of itself.
    
    Useful for training robust policies.
    """
    
    def __init__(
        self,
        env: Any,
        num_opponents: int = 1,
        opponent_sample_rate: float = 0.5,  # Rate to sample old policies
        config: Optional[MultiAgentConfig] = None,
    ):
        # Create config with self-play teams
        if config is None:
            config = MultiAgentConfig(
                teams=[
                    TeamConfig("ego", size=1, role=AgentRole.COOPERATIVE),
                    TeamConfig("opponent", size=num_opponents, role=AgentRole.ADVERSARY),
                ]
            )
        
        super().__init__(env, config)
        self.opponent_sample_rate = opponent_sample_rate
        self._policy_pool: List[AgentState] = []
    
    def add_to_pool(self, policy_state: AgentState):
        """Add policy to opponent pool."""
        self._policy_pool.append(policy_state)
    
    def sample_opponent(self) -> Optional[AgentState]:
        """Sample opponent from policy pool."""
        if not self._policy_pool:
            return None
        return np.random.choice(self._policy_pool)


# =============================================================================
# FACTORY
# =============================================================================

def make_multi_agent(
    env: Any,
    num_agents: int = 2,
    mode: str = "cooperative",
    **kwargs,
) -> MultiAgentEnv:
    """
    Factory for creating multi-agent environments.
    
    Args:
        env: Base environment
        num_agents: Number of agents
        mode: 'cooperative', 'competitive', 'mixed'
        
    Returns:
        MultiAgentEnv instance
    """
    if mode == "cooperative":
        config = MultiAgentConfig(
            teams=[TeamConfig("team", size=num_agents, role=AgentRole.COOPERATIVE)]
        )
    elif mode == "competitive":
        # Each agent is their own team
        teams = [
            TeamConfig(f"agent_{i}", size=1, role=AgentRole.COMPETITIVE)
            for i in range(num_agents)
        ]
        config = MultiAgentConfig(teams=teams)
    elif mode == "mixed":
        # Half cooperative, half adversary
        half = num_agents // 2
        config = MultiAgentConfig(
            teams=[
                TeamConfig("team_a", size=half, role=AgentRole.COOPERATIVE),
                TeamConfig("team_b", size=num_agents - half, role=AgentRole.ADVERSARY),
            ]
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return MultiAgentEnv(env, config)
