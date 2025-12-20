"""
Autonomy Module for Tensor Network Operations
==============================================

Autonomous mission planning, decision making, and path planning
for tensor network computations and swarm coordination.

Components:
- MissionPlanner: High-level mission planning
- PathPlanner: Optimal path finding algorithms
- ObstacleAvoidance: Dynamic obstacle handling
- DecisionMaker: Autonomous decision making
"""

from tensornet.autonomy.mission_planner import (
    MissionPlanner,
    Mission,
    MissionPhase,
    MissionStatus,
    MissionConstraints,
    MissionResult,
    plan_mission,
    execute_mission,
)

from tensornet.autonomy.path_planning import (
    PathPlanner,
    Path,
    Waypoint,
    PathPlannerConfig,
    PlanningAlgorithm,
    plan_path,
    smooth_path,
)

from tensornet.autonomy.obstacle_avoidance import (
    ObstacleAvoidance,
    Obstacle,
    ObstacleType,
    AvoidanceStrategy,
    AvoidanceResult,
    detect_obstacles,
    compute_avoidance_vector,
)

from tensornet.autonomy.decision_making import (
    DecisionMaker,
    Decision,
    DecisionType,
    StateEstimate,
    ActionSpace,
    make_decision,
    evaluate_options,
)

__all__ = [
    # Mission planner
    "MissionPlanner",
    "Mission",
    "MissionPhase",
    "MissionStatus",
    "MissionConstraints",
    "MissionResult",
    "plan_mission",
    "execute_mission",
    # Path planning
    "PathPlanner",
    "Path",
    "Waypoint",
    "PathPlannerConfig",
    "PlanningAlgorithm",
    "plan_path",
    "smooth_path",
    # Obstacle avoidance
    "ObstacleAvoidance",
    "Obstacle",
    "ObstacleType",
    "AvoidanceStrategy",
    "AvoidanceResult",
    "detect_obstacles",
    "compute_avoidance_vector",
    # Decision making
    "DecisionMaker",
    "Decision",
    "DecisionType",
    "StateEstimate",
    "ActionSpace",
    "make_decision",
    "evaluate_options",
]
