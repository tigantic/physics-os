"""
Mission Planner Module
======================

High-level autonomous mission planning for
tensor network swarm operations.

Supports:
- Multi-phase mission planning
- Resource allocation
- Constraint satisfaction
- Replanning on failure
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class MissionStatus(Enum):
    """Status of a mission."""

    PENDING = auto()
    PLANNING = auto()
    READY = auto()
    EXECUTING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


class MissionPhaseType(Enum):
    """Types of mission phases."""

    INITIALIZE = auto()
    COMPUTE = auto()
    SYNCHRONIZE = auto()
    OPTIMIZE = auto()
    VALIDATE = auto()
    FINALIZE = auto()
    CUSTOM = auto()


@dataclass
class MissionConstraints:
    """Constraints for mission execution.

    Attributes:
        max_time: Maximum time allowed (seconds)
        max_memory: Maximum memory (GB)
        min_accuracy: Minimum accuracy required
        max_agents: Maximum agents to use
        priority: Priority level
        dependencies: Required prior missions
    """

    max_time: float = float("inf")
    max_memory: float = 16.0
    min_accuracy: float = 0.99
    max_agents: int = 100
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)


@dataclass
class MissionPhase:
    """A phase within a mission.

    Attributes:
        phase_id: Unique identifier
        phase_type: Type of phase
        name: Human-readable name
        duration_estimate: Estimated duration
        resource_requirement: Required resources
        status: Current status
        start_time: When phase started
        end_time: When phase ended
        result: Phase result
    """

    phase_id: int
    phase_type: MissionPhaseType
    name: str
    duration_estimate: float = 0.0
    resource_requirement: float = 1.0
    status: MissionStatus = MissionStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    result: Any = None

    @property
    def actual_duration(self) -> float | None:
        """Get actual duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def start(self) -> None:
        """Mark phase as started."""
        self.status = MissionStatus.EXECUTING
        self.start_time = time.perf_counter()

    def complete(self, result: Any = None) -> None:
        """Mark phase as completed."""
        self.status = MissionStatus.COMPLETED
        self.end_time = time.perf_counter()
        self.result = result

    def fail(self, reason: str = "") -> None:
        """Mark phase as failed."""
        self.status = MissionStatus.FAILED
        self.end_time = time.perf_counter()
        self.result = {"error": reason}


@dataclass
class Mission:
    """A complete mission specification.

    Attributes:
        mission_id: Unique identifier
        name: Mission name
        description: Detailed description
        phases: List of phases
        constraints: Mission constraints
        status: Current status
        created_time: Creation timestamp
        start_time: Execution start
        end_time: Execution end
    """

    mission_id: str
    name: str
    description: str = ""
    phases: list[MissionPhase] = field(default_factory=list)
    constraints: MissionConstraints = field(default_factory=MissionConstraints)
    status: MissionStatus = MissionStatus.PENDING
    created_time: float = field(default_factory=time.perf_counter)
    start_time: float | None = None
    end_time: float | None = None

    @property
    def total_estimate(self) -> float:
        """Total estimated duration."""
        return sum(p.duration_estimate for p in self.phases)

    @property
    def progress(self) -> float:
        """Current progress (0-1)."""
        if not self.phases:
            return 0.0
        completed = sum(1 for p in self.phases if p.status == MissionStatus.COMPLETED)
        return completed / len(self.phases)

    def get_current_phase(self) -> MissionPhase | None:
        """Get currently executing phase."""
        for phase in self.phases:
            if phase.status == MissionStatus.EXECUTING:
                return phase
        return None

    def get_next_phase(self) -> MissionPhase | None:
        """Get next pending phase."""
        for phase in self.phases:
            if phase.status == MissionStatus.PENDING:
                return phase
        return None


@dataclass
class MissionResult:
    """Result of mission execution.

    Attributes:
        mission_id: Mission identifier
        success: Whether mission succeeded
        total_time: Total execution time
        phase_results: Results per phase
        resource_usage: Resources consumed
        metrics: Performance metrics
    """

    mission_id: str
    success: bool
    total_time: float
    phase_results: list[Any] = field(default_factory=list)
    resource_usage: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class MissionPlanner:
    """Autonomous mission planner.

    Plans and coordinates multi-phase missions
    for tensor network operations.
    """

    def __init__(
        self,
        max_concurrent_missions: int = 4,
        replan_on_failure: bool = True,
    ) -> None:
        """Initialize planner.

        Args:
            max_concurrent_missions: Max concurrent
            replan_on_failure: Attempt replan on failure
        """
        self.max_concurrent = max_concurrent_missions
        self.replan_on_failure = replan_on_failure

        self.missions: dict[str, Mission] = {}
        self.active_missions: set[str] = set()
        self.completed_missions: list[str] = []

        self._mission_counter = 0

    def create_mission(
        self,
        name: str,
        description: str = "",
        constraints: MissionConstraints | None = None,
    ) -> Mission:
        """Create a new mission.

        Args:
            name: Mission name
            description: Description
            constraints: Constraints

        Returns:
            Created mission
        """
        mission_id = f"mission_{self._mission_counter:04d}"
        self._mission_counter += 1

        mission = Mission(
            mission_id=mission_id,
            name=name,
            description=description,
            constraints=constraints or MissionConstraints(),
        )

        self.missions[mission_id] = mission
        return mission

    def add_phase(
        self,
        mission: Mission,
        phase_type: MissionPhaseType,
        name: str,
        duration_estimate: float = 1.0,
        resource_requirement: float = 1.0,
    ) -> MissionPhase:
        """Add phase to mission.

        Args:
            mission: Target mission
            phase_type: Type of phase
            name: Phase name
            duration_estimate: Estimated duration
            resource_requirement: Resource requirement

        Returns:
            Created phase
        """
        phase = MissionPhase(
            phase_id=len(mission.phases),
            phase_type=phase_type,
            name=name,
            duration_estimate=duration_estimate,
            resource_requirement=resource_requirement,
        )

        mission.phases.append(phase)
        return phase

    def plan_computation_mission(
        self,
        problem_size: int,
        target_accuracy: float = 1e-6,
        max_time: float = 300.0,
    ) -> Mission:
        """Create a computation mission.

        Args:
            problem_size: Size of problem
            target_accuracy: Target accuracy
            max_time: Maximum time

        Returns:
            Planned mission
        """
        mission = self.create_mission(
            name="Tensor Network Computation",
            description=f"Compute TN with {problem_size} sites",
            constraints=MissionConstraints(
                max_time=max_time,
                min_accuracy=1.0 - target_accuracy,
            ),
        )

        # Add phases based on problem size
        self.add_phase(
            mission,
            MissionPhaseType.INITIALIZE,
            "Initialize MPS",
            duration_estimate=0.1 * problem_size,
        )

        self.add_phase(
            mission,
            MissionPhaseType.COMPUTE,
            "DMRG Sweeps",
            duration_estimate=1.0 * problem_size,
            resource_requirement=2.0,
        )

        self.add_phase(
            mission,
            MissionPhaseType.OPTIMIZE,
            "Compression",
            duration_estimate=0.2 * problem_size,
        )

        self.add_phase(
            mission,
            MissionPhaseType.VALIDATE,
            "Verify Results",
            duration_estimate=0.1 * problem_size,
        )

        self.add_phase(
            mission,
            MissionPhaseType.FINALIZE,
            "Finalize",
            duration_estimate=0.05 * problem_size,
        )

        mission.status = MissionStatus.READY
        return mission

    def plan_swarm_mission(
        self,
        num_agents: int,
        task_count: int,
        coordination_level: float = 0.5,
    ) -> Mission:
        """Create a swarm coordination mission.

        Args:
            num_agents: Number of agents
            task_count: Number of tasks
            coordination_level: How tightly coordinated

        Returns:
            Planned mission
        """
        mission = self.create_mission(
            name="Swarm Coordination",
            description=f"Coordinate {num_agents} agents for {task_count} tasks",
            constraints=MissionConstraints(max_agents=num_agents),
        )

        self.add_phase(
            mission,
            MissionPhaseType.INITIALIZE,
            "Deploy Agents",
            duration_estimate=num_agents * 0.01,
        )

        self.add_phase(
            mission,
            MissionPhaseType.SYNCHRONIZE,
            "Establish Formation",
            duration_estimate=coordination_level * num_agents * 0.1,
        )

        for i in range(min(task_count, 10)):
            self.add_phase(
                mission,
                MissionPhaseType.COMPUTE,
                f"Execute Task {i+1}",
                duration_estimate=1.0,
            )

        self.add_phase(
            mission,
            MissionPhaseType.FINALIZE,
            "Return to Base",
            duration_estimate=num_agents * 0.01,
        )

        mission.status = MissionStatus.READY
        return mission

    def start_mission(self, mission_id: str) -> bool:
        """Start executing a mission.

        Args:
            mission_id: Mission to start

        Returns:
            Success
        """
        if mission_id not in self.missions:
            return False

        if len(self.active_missions) >= self.max_concurrent:
            return False

        mission = self.missions[mission_id]
        if mission.status != MissionStatus.READY:
            return False

        mission.status = MissionStatus.EXECUTING
        mission.start_time = time.perf_counter()
        self.active_missions.add(mission_id)

        # Start first phase
        next_phase = mission.get_next_phase()
        if next_phase:
            next_phase.start()

        return True

    def advance_mission(
        self,
        mission_id: str,
        phase_result: Any = None,
    ) -> tuple[bool, MissionPhase | None]:
        """Advance mission to next phase.

        Args:
            mission_id: Mission to advance
            phase_result: Result of current phase

        Returns:
            (success, next_phase)
        """
        if mission_id not in self.missions:
            return False, None

        mission = self.missions[mission_id]
        current = mission.get_current_phase()

        if current:
            current.complete(phase_result)

        next_phase = mission.get_next_phase()

        if next_phase:
            next_phase.start()
            return True, next_phase
        else:
            # Mission complete
            self._complete_mission(mission_id, success=True)
            return True, None

    def fail_phase(
        self,
        mission_id: str,
        reason: str = "",
    ) -> bool:
        """Mark current phase as failed.

        Args:
            mission_id: Mission ID
            reason: Failure reason

        Returns:
            Whether replanning was attempted
        """
        if mission_id not in self.missions:
            return False

        mission = self.missions[mission_id]
        current = mission.get_current_phase()

        if current:
            current.fail(reason)

        if self.replan_on_failure:
            return self._attempt_replan(mission_id)
        else:
            self._complete_mission(mission_id, success=False)
            return False

    def _attempt_replan(self, mission_id: str) -> bool:
        """Attempt to replan after failure.

        Args:
            mission_id: Mission to replan

        Returns:
            Success
        """
        mission = self.missions[mission_id]

        # Find failed phase
        failed_phase = None
        for phase in mission.phases:
            if phase.status == MissionStatus.FAILED:
                failed_phase = phase
                break

        if failed_phase is None:
            return False

        # Simple replan: retry phase
        failed_phase.status = MissionStatus.PENDING
        failed_phase.start()

        return True

    def _complete_mission(
        self,
        mission_id: str,
        success: bool,
    ) -> None:
        """Mark mission as complete.

        Args:
            mission_id: Mission ID
            success: Whether successful
        """
        mission = self.missions[mission_id]
        mission.status = MissionStatus.COMPLETED if success else MissionStatus.FAILED
        mission.end_time = time.perf_counter()

        self.active_missions.discard(mission_id)
        self.completed_missions.append(mission_id)

    def abort_mission(self, mission_id: str) -> bool:
        """Abort a mission.

        Args:
            mission_id: Mission to abort

        Returns:
            Success
        """
        if mission_id not in self.missions:
            return False

        mission = self.missions[mission_id]
        mission.status = MissionStatus.ABORTED
        mission.end_time = time.perf_counter()

        self.active_missions.discard(mission_id)
        return True

    def get_mission_result(self, mission_id: str) -> MissionResult | None:
        """Get result of completed mission.

        Args:
            mission_id: Mission ID

        Returns:
            MissionResult or None
        """
        if mission_id not in self.missions:
            return None

        mission = self.missions[mission_id]

        if mission.status not in (
            MissionStatus.COMPLETED,
            MissionStatus.FAILED,
            MissionStatus.ABORTED,
        ):
            return None

        total_time = (mission.end_time or 0) - (mission.start_time or 0)

        return MissionResult(
            mission_id=mission_id,
            success=mission.status == MissionStatus.COMPLETED,
            total_time=total_time,
            phase_results=[p.result for p in mission.phases],
            metrics={
                "num_phases": len(mission.phases),
                "progress": mission.progress,
                "estimated_time": mission.total_estimate,
            },
        )

    def get_status_report(self) -> dict[str, Any]:
        """Get planner status report.

        Returns:
            Status dictionary
        """
        return {
            "total_missions": len(self.missions),
            "active_missions": len(self.active_missions),
            "completed_missions": len(self.completed_missions),
            "active_ids": list(self.active_missions),
            "pending": sum(
                1 for m in self.missions.values() if m.status == MissionStatus.PENDING
            ),
        }


def plan_mission(
    problem_type: str,
    **kwargs,
) -> Mission:
    """Convenience function to plan a mission.

    Args:
        problem_type: Type ('computation' or 'swarm')
        **kwargs: Additional parameters

    Returns:
        Planned mission
    """
    planner = MissionPlanner()

    if problem_type == "computation":
        return planner.plan_computation_mission(
            problem_size=kwargs.get("problem_size", 20),
            target_accuracy=kwargs.get("target_accuracy", 1e-6),
            max_time=kwargs.get("max_time", 300.0),
        )
    elif problem_type == "swarm":
        return planner.plan_swarm_mission(
            num_agents=kwargs.get("num_agents", 10),
            task_count=kwargs.get("task_count", 5),
            coordination_level=kwargs.get("coordination_level", 0.5),
        )
    else:
        return planner.create_mission(
            name=kwargs.get("name", "Custom Mission"),
            description=kwargs.get("description", ""),
        )


def execute_mission(
    mission: Mission,
    phase_executor: Callable[[MissionPhase], Any] | None = None,
) -> MissionResult:
    """Execute a mission.

    Args:
        mission: Mission to execute
        phase_executor: Function to execute phases

    Returns:
        MissionResult
    """
    planner = MissionPlanner()
    planner.missions[mission.mission_id] = mission

    # Ensure mission is ready
    if mission.status == MissionStatus.PENDING:
        mission.status = MissionStatus.READY

    planner.start_mission(mission.mission_id)

    # Execute phases
    while True:
        current = mission.get_current_phase()
        if current is None:
            break

        # Execute phase
        if phase_executor:
            try:
                result = phase_executor(current)
            except Exception as e:
                planner.fail_phase(mission.mission_id, str(e))
                continue
        else:
            result = None

        # Advance
        success, _ = planner.advance_mission(mission.mission_id, result)
        if not success:
            break

    return planner.get_mission_result(mission.mission_id) or MissionResult(
        mission_id=mission.mission_id,
        success=False,
        total_time=0.0,
    )
