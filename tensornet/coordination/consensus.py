# Copyright (c) 2025 Tigantic
# Phase 18: Consensus Protocols
"""
Distributed consensus protocols for multi-vehicle coordination.

Provides algorithms for achieving agreement in distributed systems
including average consensus, max consensus, and leader election.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from tensornet.coordination.swarm import SwarmTopology, TopologyType


class ConsensusState(Enum):
    """Consensus algorithm state."""

    INITIALIZING = auto()  # Setting up consensus
    RUNNING = auto()  # Iterating toward consensus
    CONVERGED = auto()  # Consensus reached
    FAILED = auto()  # Failed to converge


@dataclass
class ConsensusConfig:
    """Consensus configuration.

    Attributes:
        max_iterations: Maximum iterations
        convergence_threshold: Convergence threshold
        step_size: Iteration step size
        async_updates: Allow asynchronous updates
        timeout: Timeout in seconds
    """

    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    step_size: float = 0.1
    async_updates: bool = False
    timeout: float = 30.0

    def validate(self) -> bool:
        """Validate configuration."""
        if self.max_iterations < 1:
            return False
        if self.convergence_threshold <= 0:
            return False
        if not 0 < self.step_size <= 1:
            return False
        return True


@dataclass
class ConsensusResult:
    """Result of consensus algorithm.

    Attributes:
        state: Final consensus state
        values: Final values per agent
        consensus_value: Agreed-upon value (if converged)
        iterations: Number of iterations
        elapsed_time: Time taken
        convergence_history: History of max disagreement
    """

    state: ConsensusState
    values: dict[str, np.ndarray]
    consensus_value: np.ndarray | None = None
    iterations: int = 0
    elapsed_time: float = 0.0
    convergence_history: list[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """Check if consensus was reached."""
        return self.state == ConsensusState.CONVERGED


class ConsensusProtocol:
    """Base class for consensus protocols.

    Provides framework for distributed consensus algorithms
    over arbitrary graph topologies.

    Attributes:
        config: Consensus configuration
        topology: Communication topology
    """

    def __init__(
        self,
        config: ConsensusConfig | None = None,
        topology: SwarmTopology | None = None,
    ) -> None:
        """Initialize consensus protocol.

        Args:
            config: Consensus configuration
            topology: Communication topology
        """
        self.config = config or ConsensusConfig()
        self.topology = topology or SwarmTopology(TopologyType.FULLY_CONNECTED)

        self._state = ConsensusState.INITIALIZING
        self._values: dict[str, np.ndarray] = {}
        self._history: list[float] = []

    @property
    def state(self) -> ConsensusState:
        """Get current consensus state."""
        return self._state

    def initialize(self, initial_values: dict[str, np.ndarray]) -> None:
        """Initialize consensus with values.

        Args:
            initial_values: Initial values per agent
        """
        self._values = {k: np.array(v) for k, v in initial_values.items()}
        self.topology.build(list(self._values.keys()))
        self._state = ConsensusState.RUNNING
        self._history = []

    def update_rule(
        self,
        agent_id: str,
        current_value: np.ndarray,
        neighbor_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute update for an agent.

        To be overridden by subclasses.

        Args:
            agent_id: Agent identifier
            current_value: Current agent value
            neighbor_values: Values from neighbors

        Returns:
            Updated value
        """
        from tensornet.core.phase_deferred import PhaseDeferredError

        raise PhaseDeferredError(
            phase="25",
            reason="ConsensusProtocol._compute_update - protocol-specific update rule",
            depends_on=["average consensus", "max consensus", "formation protocols"],
        )

    def iterate(self) -> float:
        """Perform one consensus iteration.

        Returns:
            Maximum disagreement
        """
        new_values = {}

        for agent_id, current_value in self._values.items():
            neighbors = self.topology.get_neighbors(agent_id)
            neighbor_values = {
                n: self._values[n] for n in neighbors if n in self._values
            }

            new_values[agent_id] = self.update_rule(
                agent_id, current_value, neighbor_values
            )

        # Compute disagreement before updating
        max_disagreement = 0.0
        for agent_id in self._values:
            diff = np.linalg.norm(new_values[agent_id] - self._values[agent_id])
            max_disagreement = max(max_disagreement, diff)

        self._values = new_values
        self._history.append(max_disagreement)

        return max_disagreement

    def run(
        self,
        initial_values: dict[str, np.ndarray],
    ) -> ConsensusResult:
        """Run consensus algorithm to convergence.

        Args:
            initial_values: Initial values per agent

        Returns:
            ConsensusResult
        """
        self.initialize(initial_values)

        start_time = time.time()
        iteration = 0

        while iteration < self.config.max_iterations:
            if time.time() - start_time > self.config.timeout:
                self._state = ConsensusState.FAILED
                break

            disagreement = self.iterate()
            iteration += 1

            if disagreement < self.config.convergence_threshold:
                self._state = ConsensusState.CONVERGED
                break

        elapsed = time.time() - start_time

        # Compute consensus value
        consensus_value = None
        if self._state == ConsensusState.CONVERGED and self._values:
            values = list(self._values.values())
            consensus_value = np.mean(values, axis=0)

        return ConsensusResult(
            state=self._state,
            values=dict(self._values),
            consensus_value=consensus_value,
            iterations=iteration,
            elapsed_time=elapsed,
            convergence_history=self._history,
        )

    def get_value(self, agent_id: str) -> np.ndarray | None:
        """Get current value for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Current value or None
        """
        return self._values.get(agent_id)


class AverageConsensus(ConsensusProtocol):
    """Average consensus protocol.

    Agents converge to the average of initial values
    using local averaging updates.
    """

    def update_rule(
        self,
        agent_id: str,
        current_value: np.ndarray,
        neighbor_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute average consensus update.

        x_i(k+1) = x_i(k) + ε * Σ_j∈N(i) (x_j(k) - x_i(k))

        Args:
            agent_id: Agent identifier
            current_value: Current agent value
            neighbor_values: Values from neighbors

        Returns:
            Updated value
        """
        if not neighbor_values:
            return current_value

        epsilon = self.config.step_size

        # Compute weighted sum of differences
        delta = np.zeros_like(current_value)
        for neighbor_value in neighbor_values.values():
            delta = delta + (neighbor_value - current_value)

        return current_value + epsilon * delta


class MaxConsensus(ConsensusProtocol):
    """Max consensus protocol.

    Agents converge to the maximum of initial values.
    """

    def update_rule(
        self,
        agent_id: str,
        current_value: np.ndarray,
        neighbor_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute max consensus update.

        x_i(k+1) = max(x_i(k), max_j∈N(i)(x_j(k)))

        Args:
            agent_id: Agent identifier
            current_value: Current agent value
            neighbor_values: Values from neighbors

        Returns:
            Updated value
        """
        result = current_value.copy()

        for neighbor_value in neighbor_values.values():
            result = np.maximum(result, neighbor_value)

        return result


class MinConsensus(ConsensusProtocol):
    """Min consensus protocol.

    Agents converge to the minimum of initial values.
    """

    def update_rule(
        self,
        agent_id: str,
        current_value: np.ndarray,
        neighbor_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute min consensus update.

        x_i(k+1) = min(x_i(k), min_j∈N(i)(x_j(k)))

        Args:
            agent_id: Agent identifier
            current_value: Current agent value
            neighbor_values: Values from neighbors

        Returns:
            Updated value
        """
        result = current_value.copy()

        for neighbor_value in neighbor_values.values():
            result = np.minimum(result, neighbor_value)

        return result


class WeightedConsensus(ConsensusProtocol):
    """Weighted average consensus.

    Agents converge to weighted average based on confidence.
    """

    def __init__(
        self,
        config: ConsensusConfig | None = None,
        topology: SwarmTopology | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize weighted consensus.

        Args:
            config: Consensus configuration
            topology: Communication topology
            weights: Per-agent confidence weights
        """
        super().__init__(config, topology)
        self._weights = weights or {}

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set agent weights.

        Args:
            weights: Per-agent weights
        """
        self._weights = dict(weights)

    def update_rule(
        self,
        agent_id: str,
        current_value: np.ndarray,
        neighbor_values: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute weighted consensus update.

        Args:
            agent_id: Agent identifier
            current_value: Current agent value
            neighbor_values: Values from neighbors

        Returns:
            Updated value
        """
        if not neighbor_values:
            return current_value

        epsilon = self.config.step_size
        my_weight = self._weights.get(agent_id, 1.0)

        delta = np.zeros_like(current_value)
        total_weight = 0.0

        for neighbor_id, neighbor_value in neighbor_values.items():
            neighbor_weight = self._weights.get(neighbor_id, 1.0)
            # Weight by relative confidence
            relative_weight = neighbor_weight / (my_weight + neighbor_weight)
            delta = delta + relative_weight * (neighbor_value - current_value)
            total_weight += relative_weight

        if total_weight > 0:
            delta = delta / total_weight

        return current_value + epsilon * delta


class LeaderElection:
    """Distributed leader election protocol.

    Elects a leader in the network based on agent IDs or priorities.
    """

    def __init__(self, topology: SwarmTopology | None = None) -> None:
        """Initialize leader election.

        Args:
            topology: Communication topology
        """
        self.topology = topology or SwarmTopology(TopologyType.FULLY_CONNECTED)
        self._leader: str | None = None
        self._priorities: dict[str, float] = {}

    def elect(
        self,
        agent_ids: list[str],
        priorities: dict[str, float] | None = None,
    ) -> str:
        """Elect a leader.

        Args:
            agent_ids: List of agent IDs
            priorities: Optional priority values (higher = more likely)

        Returns:
            Elected leader ID
        """
        if not agent_ids:
            raise ValueError("No agents to elect from")

        self._priorities = priorities or {aid: hash(aid) % 1000 for aid in agent_ids}

        # Build topology
        self.topology.build(agent_ids)

        # Simple max-based election using consensus
        # Each agent starts with its priority, converge to max
        initial_values = {
            aid: np.array([self._priorities.get(aid, 0.0)]) for aid in agent_ids
        }

        config = ConsensusConfig(
            max_iterations=len(agent_ids) * 2,  # O(diameter) rounds
            convergence_threshold=1e-10,
        )

        max_consensus = MaxConsensus(config, self.topology)
        result = max_consensus.run(initial_values)

        # Find agent with max priority
        if result.consensus_value is not None:
            max_priority = result.consensus_value[0]
            for aid, priority in self._priorities.items():
                if abs(priority - max_priority) < 1e-10:
                    self._leader = aid
                    return aid

        # Fallback: highest priority
        self._leader = max(agent_ids, key=lambda a: self._priorities.get(a, 0))
        return self._leader

    @property
    def leader(self) -> str | None:
        """Get current leader."""
        return self._leader


def run_consensus(
    initial_values: dict[str, np.ndarray],
    protocol: str = "average",
    config: ConsensusConfig | None = None,
    topology_type: TopologyType = TopologyType.FULLY_CONNECTED,
) -> ConsensusResult:
    """Run a consensus protocol.

    Args:
        initial_values: Initial values per agent
        protocol: Protocol type ("average", "max", "min")
        config: Consensus configuration
        topology_type: Communication topology type

    Returns:
        ConsensusResult
    """
    config = config or ConsensusConfig()
    topology = SwarmTopology(topology_type)

    if protocol == "average":
        consensus = AverageConsensus(config, topology)
    elif protocol == "max":
        consensus = MaxConsensus(config, topology)
    elif protocol == "min":
        consensus = MinConsensus(config, topology)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return consensus.run(initial_values)
