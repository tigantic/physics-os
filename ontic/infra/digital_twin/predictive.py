"""
Predictive maintenance for hypersonic vehicle digital twins.

This module provides remaining useful life (RUL) estimation,
maintenance scheduling optimization, and reliability analysis
for hypersonic vehicle subsystems.

Key capabilities:
    - RUL estimation using physics-based and data-driven methods
    - Optimal maintenance scheduling considering mission constraints
    - Reliability and risk quantification
    - Component-level and system-level analysis

Author: HyperTensor Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class FailureMode(Enum):
    """Failure modes for vehicle components."""

    FATIGUE = auto()
    THERMAL_DEGRADATION = auto()
    OXIDATION = auto()
    ABLATION = auto()
    DELAMINATION = auto()
    CREEP = auto()
    CORROSION = auto()
    WEAR = auto()


class ComponentType(Enum):
    """Types of vehicle components."""

    TPS_TILE = auto()
    LEADING_EDGE = auto()
    CONTROL_SURFACE = auto()
    BEARING = auto()
    SEAL = auto()
    ACTUATOR = auto()
    SENSOR = auto()
    STRUCTURAL = auto()


class MaintenanceAction(Enum):
    """Types of maintenance actions."""

    INSPECTION = auto()
    MINOR_REPAIR = auto()
    MAJOR_REPAIR = auto()
    REPLACEMENT = auto()
    OVERHAUL = auto()


@dataclass
class MaintenanceConfig:
    """Configuration for predictive maintenance."""

    # RUL estimation
    rul_confidence_level: float = 0.95
    min_rul_threshold: float = 10.0  # hours

    # Maintenance scheduling
    planning_horizon: float = 1000.0  # hours
    min_maintenance_interval: float = 24.0  # hours

    # Cost parameters
    cost_inspection: float = 1000.0
    cost_minor_repair: float = 10000.0
    cost_major_repair: float = 100000.0
    cost_replacement: float = 500000.0
    cost_failure: float = 10000000.0  # Cost of in-flight failure

    # Risk thresholds
    acceptable_failure_probability: float = 1e-6
    warning_failure_probability: float = 1e-7


@dataclass
class ComponentState:
    """State of a vehicle component."""

    component_id: str
    component_type: ComponentType

    # Age and usage
    age_hours: float = 0.0
    cycles: int = 0

    # Damage state
    damage_index: float = 0.0
    failure_modes: list[FailureMode] = field(default_factory=list)

    # RUL estimate
    rul_mean: float = float("inf")
    rul_lower: float = float("inf")
    rul_upper: float = float("inf")

    # Maintenance
    last_maintenance: float | None = None
    next_scheduled: float | None = None
    maintenance_count: int = 0


@dataclass
class MaintenanceSchedule:
    """Optimized maintenance schedule."""

    component_id: str
    scheduled_time: float
    action: MaintenanceAction
    priority: int  # 1=highest
    estimated_cost: float
    estimated_downtime: float  # hours
    reason: str


class RULEstimator:
    """
    Remaining Useful Life estimator.

    Combines physics-based degradation models with data-driven
    approaches for accurate RUL prediction with uncertainty.
    """

    def __init__(self, config: MaintenanceConfig):
        self.config = config

        # Degradation model parameters
        self.degradation_models: dict[ComponentType, dict[str, Any]] = {
            ComponentType.TPS_TILE: {
                "failure_threshold": 1.0,
                "degradation_rate": 1e-4,  # per hour at nominal
                "temperature_factor": 0.1,  # acceleration per 100K above design
            },
            ComponentType.LEADING_EDGE: {
                "failure_threshold": 1.0,
                "degradation_rate": 5e-5,
                "temperature_factor": 0.2,
            },
            ComponentType.CONTROL_SURFACE: {
                "failure_threshold": 1.0,
                "degradation_rate": 1e-5,
                "cycle_factor": 1e-6,  # per cycle
            },
            ComponentType.BEARING: {
                "failure_threshold": 1.0,
                "degradation_rate": 1e-6,
                "cycle_factor": 1e-7,
            },
            ComponentType.ACTUATOR: {
                "failure_threshold": 1.0,
                "degradation_rate": 5e-6,
                "cycle_factor": 5e-7,
            },
            ComponentType.SENSOR: {
                "failure_threshold": 1.0,
                "degradation_rate": 1e-6,
                "drift_rate": 1e-5,  # calibration drift
            },
        }

        # Weibull parameters for different failure modes
        self.weibull_params: dict[FailureMode, tuple[float, float]] = {
            FailureMode.FATIGUE: (3.0, 1000.0),  # shape, scale
            FailureMode.THERMAL_DEGRADATION: (2.5, 500.0),
            FailureMode.OXIDATION: (2.0, 800.0),
            FailureMode.ABLATION: (1.5, 200.0),
            FailureMode.CREEP: (2.0, 1500.0),
            FailureMode.WEAR: (2.5, 2000.0),
        }

    def estimate_rul(
        self,
        component: ComponentState,
        operating_conditions: dict[str, float] | None = None,
    ) -> tuple[float, float, float]:
        """
        Estimate remaining useful life for a component.

        Args:
            component: Current component state
            operating_conditions: Current/expected operating conditions

        Returns:
            Tuple of (mean RUL, lower bound, upper bound) in hours
        """
        if operating_conditions is None:
            operating_conditions = {}

        # Get degradation model
        model = self.degradation_models.get(
            component.component_type,
            {"failure_threshold": 1.0, "degradation_rate": 1e-5},
        )

        # Base degradation rate
        rate = model["degradation_rate"]

        # Apply condition modifiers
        if "temperature" in operating_conditions and "temperature_factor" in model:
            temp_excess = max(0, operating_conditions["temperature"] - 2000) / 100
            rate *= 1 + model["temperature_factor"] * temp_excess

        if "cycles_per_hour" in operating_conditions and "cycle_factor" in model:
            rate += model["cycle_factor"] * operating_conditions["cycles_per_hour"]

        # Current remaining damage capacity
        remaining = model["failure_threshold"] - component.damage_index

        if remaining <= 0:
            return 0.0, 0.0, 0.0

        if rate <= 0:
            return float("inf"), float("inf"), float("inf")

        # Mean RUL
        rul_mean = remaining / rate

        # Uncertainty (increases with damage and age)
        uncertainty_factor = 0.2 + 0.3 * component.damage_index
        rul_std = rul_mean * uncertainty_factor

        # Confidence bounds
        z = 1.96  # 95% confidence
        rul_lower = max(0, rul_mean - z * rul_std)
        rul_upper = rul_mean + z * rul_std

        return rul_mean, rul_lower, rul_upper

    def estimate_failure_probability(
        self, component: ComponentState, time_horizon: float
    ) -> float:
        """
        Estimate probability of failure within time horizon.

        Uses Weibull distribution based on failure modes and age.

        Args:
            component: Component state
            time_horizon: Time horizon in hours

        Returns:
            Probability of failure
        """
        if not component.failure_modes:
            failure_modes = [FailureMode.FATIGUE]  # Default
        else:
            failure_modes = component.failure_modes

        # Combine failure probabilities (series system)
        survival_prob = 1.0

        for mode in failure_modes:
            shape, scale = self.weibull_params.get(mode, (2.0, 1000.0))

            # Adjust scale based on damage
            adjusted_scale = scale * (1 - component.damage_index * 0.5)

            # Weibull CDF: P(T <= t) = 1 - exp(-(t/scale)^shape)
            t = component.age_hours + time_horizon
            failure_prob = 1 - np.exp(-((t / adjusted_scale) ** shape))

            survival_prob *= 1 - failure_prob

        return 1 - survival_prob


class MaintenanceScheduler:
    """
    Optimal maintenance scheduling.

    Determines optimal maintenance timing and actions based on
    component health, RUL estimates, cost, and constraints.
    """

    def __init__(self, config: MaintenanceConfig):
        self.config = config
        self.rul_estimator = RULEstimator(config)

    def schedule_maintenance(
        self,
        components: list[ComponentState],
        constraints: dict[str, Any] | None = None,
    ) -> list[MaintenanceSchedule]:
        """
        Generate optimal maintenance schedule.

        Args:
            components: List of component states
            constraints: Scheduling constraints

        Returns:
            List of scheduled maintenance actions
        """
        if constraints is None:
            constraints = {}

        schedule = []

        for component in components:
            # Estimate RUL
            rul_mean, rul_lower, _ = self.rul_estimator.estimate_rul(component)

            # Update component RUL
            component.rul_mean = rul_mean
            component.rul_lower = rul_lower

            # Determine if maintenance needed
            action, priority, reason = self._determine_action(component)

            if action is not None:
                # Calculate optimal timing
                scheduled_time = self._optimal_timing(component, action)

                # Estimate cost and downtime
                cost = self._estimate_cost(action)
                downtime = self._estimate_downtime(action)

                schedule.append(
                    MaintenanceSchedule(
                        component_id=component.component_id,
                        scheduled_time=scheduled_time,
                        action=action,
                        priority=priority,
                        estimated_cost=cost,
                        estimated_downtime=downtime,
                        reason=reason,
                    )
                )

        # Sort by priority
        schedule.sort(key=lambda x: (x.priority, x.scheduled_time))

        return schedule

    def _determine_action(
        self, component: ComponentState
    ) -> tuple[MaintenanceAction | None, int, str]:
        """Determine required maintenance action."""

        # Critical damage
        if component.damage_index > 0.9:
            return MaintenanceAction.REPLACEMENT, 1, "Critical damage level"

        # Low RUL
        if component.rul_lower < self.config.min_rul_threshold:
            return MaintenanceAction.MAJOR_REPAIR, 1, "RUL below minimum threshold"

        # High damage
        if component.damage_index > 0.7:
            return MaintenanceAction.MAJOR_REPAIR, 2, "High damage accumulation"

        # Moderate damage
        if component.damage_index > 0.5:
            return MaintenanceAction.MINOR_REPAIR, 3, "Moderate damage"

        # Periodic inspection due
        if component.last_maintenance is not None:
            time_since = component.age_hours - component.last_maintenance
            if time_since > 500:  # Inspection every 500 hours
                return MaintenanceAction.INSPECTION, 4, "Periodic inspection due"

        return None, 0, ""

    def _optimal_timing(
        self, component: ComponentState, action: MaintenanceAction
    ) -> float:
        """Calculate optimal timing for maintenance."""

        if action in [MaintenanceAction.REPLACEMENT, MaintenanceAction.MAJOR_REPAIR]:
            # Schedule as soon as possible but not immediately
            return component.age_hours + min(24.0, component.rul_lower * 0.5)

        if action == MaintenanceAction.MINOR_REPAIR:
            # Some flexibility
            return component.age_hours + min(100.0, component.rul_lower * 0.7)

        if action == MaintenanceAction.INSPECTION:
            # Flexible timing
            return component.age_hours + min(200.0, component.rul_mean * 0.3)

        return component.age_hours + 100.0

    def _estimate_cost(self, action: MaintenanceAction) -> float:
        """Estimate cost of maintenance action."""
        costs = {
            MaintenanceAction.INSPECTION: self.config.cost_inspection,
            MaintenanceAction.MINOR_REPAIR: self.config.cost_minor_repair,
            MaintenanceAction.MAJOR_REPAIR: self.config.cost_major_repair,
            MaintenanceAction.REPLACEMENT: self.config.cost_replacement,
            MaintenanceAction.OVERHAUL: self.config.cost_major_repair * 2,
        }
        return costs.get(action, self.config.cost_inspection)

    def _estimate_downtime(self, action: MaintenanceAction) -> float:
        """Estimate downtime for maintenance action (hours)."""
        downtimes = {
            MaintenanceAction.INSPECTION: 2.0,
            MaintenanceAction.MINOR_REPAIR: 8.0,
            MaintenanceAction.MAJOR_REPAIR: 48.0,
            MaintenanceAction.REPLACEMENT: 72.0,
            MaintenanceAction.OVERHAUL: 168.0,
        }
        return downtimes.get(action, 4.0)


class PredictiveMaintenance:
    """
    Complete predictive maintenance system.

    Integrates RUL estimation, reliability analysis, and
    maintenance scheduling for vehicle fleet management.
    """

    def __init__(self, config: MaintenanceConfig):
        self.config = config
        self.rul_estimator = RULEstimator(config)
        self.scheduler = MaintenanceScheduler(config)

        # Fleet state
        self.components: dict[str, ComponentState] = {}
        self.maintenance_history: list[dict[str, Any]] = []

    def register_component(
        self, component_id: str, component_type: ComponentType
    ) -> ComponentState:
        """Register a new component for monitoring."""
        state = ComponentState(
            component_id=component_id,
            component_type=component_type,
        )
        self.components[component_id] = state
        return state

    def update_component(
        self,
        component_id: str,
        delta_hours: float = 0.0,
        delta_cycles: int = 0,
        damage_increment: float = 0.0,
        operating_conditions: dict[str, float] | None = None,
    ):
        """Update component state with new usage data."""
        if component_id not in self.components:
            raise ValueError(f"Unknown component: {component_id}")

        component = self.components[component_id]
        component.age_hours += delta_hours
        component.cycles += delta_cycles
        component.damage_index = min(1.0, component.damage_index + damage_increment)

        # Update RUL estimate
        rul = self.rul_estimator.estimate_rul(component, operating_conditions)
        component.rul_mean, component.rul_lower, component.rul_upper = rul

    def record_maintenance(
        self,
        component_id: str,
        action: MaintenanceAction,
        timestamp: float,
        notes: str = "",
    ):
        """Record completed maintenance action."""
        if component_id not in self.components:
            raise ValueError(f"Unknown component: {component_id}")

        component = self.components[component_id]

        # Update component state based on action
        if action == MaintenanceAction.REPLACEMENT:
            component.damage_index = 0.0
            component.age_hours = 0.0
            component.cycles = 0
        elif action == MaintenanceAction.MAJOR_REPAIR:
            component.damage_index *= 0.2  # 80% damage reduction
        elif action == MaintenanceAction.MINOR_REPAIR:
            component.damage_index *= 0.7  # 30% damage reduction

        component.last_maintenance = timestamp
        component.maintenance_count += 1

        # Record in history
        self.maintenance_history.append(
            {
                "component_id": component_id,
                "action": action.name,
                "timestamp": timestamp,
                "damage_before": component.damage_index,
                "notes": notes,
            }
        )

    def get_fleet_health(self) -> dict[str, Any]:
        """Get overall fleet health summary."""
        if not self.components:
            return {"status": "NO_COMPONENTS"}

        damages = [c.damage_index for c in self.components.values()]
        ruls = [
            c.rul_mean for c in self.components.values() if c.rul_mean < float("inf")
        ]

        critical_count = sum(1 for d in damages if d > 0.9)
        warning_count = sum(1 for d in damages if 0.7 < d <= 0.9)

        return {
            "total_components": len(self.components),
            "critical_count": critical_count,
            "warning_count": warning_count,
            "mean_damage": np.mean(damages),
            "max_damage": np.max(damages),
            "min_rul": min(ruls) if ruls else float("inf"),
            "mean_rul": np.mean(ruls) if ruls else float("inf"),
            "status": (
                "CRITICAL"
                if critical_count > 0
                else ("WARNING" if warning_count > 0 else "NOMINAL")
            ),
        }

    def generate_maintenance_plan(self) -> list[MaintenanceSchedule]:
        """Generate maintenance plan for all components."""
        return self.scheduler.schedule_maintenance(list(self.components.values()))


def estimate_remaining_life(
    damage: float, damage_rate: float, threshold: float = 1.0
) -> float:
    """
    Simple RUL estimation from current damage and rate.

    Args:
        damage: Current damage index (0-1)
        damage_rate: Damage accumulation rate per hour
        threshold: Failure threshold

    Returns:
        Remaining life in hours
    """
    remaining = threshold - damage
    if remaining <= 0:
        return 0.0
    if damage_rate <= 0:
        return float("inf")
    return remaining / damage_rate


def compute_reliability(failure_rate: float, time: float) -> float:
    """
    Compute reliability (survival probability) assuming constant failure rate.

    R(t) = exp(-lambda * t)

    Args:
        failure_rate: Failure rate (per hour)
        time: Time horizon (hours)

    Returns:
        Reliability (0-1)
    """
    return np.exp(-failure_rate * time)


def optimize_maintenance_schedule(
    components: list[ComponentState], config: MaintenanceConfig, objective: str = "cost"
) -> list[MaintenanceSchedule]:
    """
    Optimize maintenance schedule for minimum cost or maximum availability.

    Args:
        components: List of component states
        config: Maintenance configuration
        objective: 'cost' or 'availability'

    Returns:
        Optimized maintenance schedule
    """
    scheduler = MaintenanceScheduler(config)
    base_schedule = scheduler.schedule_maintenance(components)

    if objective == "availability":
        # Prioritize actions that maximize availability
        # (minimize downtime)
        base_schedule.sort(key=lambda x: x.estimated_downtime)
    else:
        # Default: minimize cost while meeting constraints
        base_schedule.sort(key=lambda x: x.estimated_cost)

    return base_schedule


def test_predictive_maintenance():
    """Test predictive maintenance module."""
    print("Testing Predictive Maintenance Module...")

    # Create configuration
    config = MaintenanceConfig()

    # Create predictive maintenance system
    pm = PredictiveMaintenance(config)

    # Register components
    tps_tile = pm.register_component("TPS-001", ComponentType.TPS_TILE)
    leading_edge = pm.register_component("LE-001", ComponentType.LEADING_EDGE)
    actuator = pm.register_component("ACT-001", ComponentType.ACTUATOR)

    print("\n  Registered components:")
    for cid in pm.components:
        print(f"    - {cid}")

    # Simulate usage
    print("\n  Simulating 500 hours of operation...")
    for i in range(500):
        # TPS tile degrades with temperature
        pm.update_component(
            "TPS-001",
            delta_hours=1.0,
            damage_increment=0.001,
            operating_conditions={"temperature": 2100},
        )

        # Leading edge degrades faster
        pm.update_component(
            "LE-001",
            delta_hours=1.0,
            damage_increment=0.0015,
            operating_conditions={"temperature": 2200},
        )

        # Actuator cycles
        pm.update_component(
            "ACT-001", delta_hours=1.0, delta_cycles=10, damage_increment=0.0005
        )

    # Check component states
    print("\n  Component states after 500 hours:")
    for cid, comp in pm.components.items():
        print(f"    {cid}:")
        print(f"      Damage: {comp.damage_index*100:.1f}%")
        print(
            f"      RUL: {comp.rul_mean:.0f} hrs (95% CI: [{comp.rul_lower:.0f}, {comp.rul_upper:.0f}])"
        )

    # Get fleet health
    health = pm.get_fleet_health()
    print(f"\n  Fleet Health: {health['status']}")
    print(f"    Mean damage: {health['mean_damage']*100:.1f}%")
    print(f"    Min RUL: {health['min_rul']:.0f} hrs")

    # Generate maintenance plan
    plan = pm.generate_maintenance_plan()
    print(f"\n  Maintenance Plan ({len(plan)} actions):")
    for item in plan[:5]:  # Show first 5
        print(
            f"    {item.component_id}: {item.action.name} at t={item.scheduled_time:.0f}h"
        )
        print(f"      Priority: {item.priority}, Cost: ${item.estimated_cost:,.0f}")
        print(f"      Reason: {item.reason}")

    # Test RUL estimator directly
    print("\n  Testing RUL estimator...")
    estimator = RULEstimator(config)

    test_comp = ComponentState(
        component_id="TEST-001",
        component_type=ComponentType.TPS_TILE,
        damage_index=0.5,
        age_hours=250,
    )

    rul_mean, rul_lower, rul_upper = estimator.estimate_rul(test_comp)
    print(f"    RUL for 50% damage TPS: {rul_mean:.0f} hrs")

    # Test failure probability
    fail_prob = estimator.estimate_failure_probability(test_comp, time_horizon=100)
    print(f"    Failure probability (100 hrs): {fail_prob*100:.2f}%")

    # Test helper functions
    rul = estimate_remaining_life(damage=0.6, damage_rate=0.001)
    print(f"\n  Simple RUL (60% damage, rate=0.001): {rul:.0f} hrs")

    rel = compute_reliability(failure_rate=1e-4, time=1000)
    print(f"  Reliability (λ=1e-4, t=1000): {rel*100:.2f}%")

    # Record maintenance
    pm.record_maintenance(
        "LE-001",
        MaintenanceAction.MAJOR_REPAIR,
        timestamp=500,
        notes="High damage repair",
    )

    print("\n  After major repair of LE-001:")
    print(f"    New damage: {pm.components['LE-001'].damage_index*100:.1f}%")

    print("\nPredictive Maintenance: All tests passed!")


if __name__ == "__main__":
    test_predictive_maintenance()
