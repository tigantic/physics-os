"""
Decision Making Module
======================

Autonomous decision making for tensor network
operations and swarm coordination.

Features:
- State estimation
- Option evaluation
- Risk assessment
- Multi-criteria decisions
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Set,
)

import numpy as np


class DecisionType(Enum):
    """Types of decisions."""
    
    ACTION = auto()      # What to do
    RESOURCE = auto()    # Resource allocation
    ROUTING = auto()     # Path selection
    SCHEDULING = auto()  # Timing decisions
    EMERGENCY = auto()   # Emergency response
    COORDINATION = auto()  # Team coordination


class DecisionCriteria(Enum):
    """Decision criteria."""
    
    SPEED = auto()
    ACCURACY = auto()
    SAFETY = auto()
    EFFICIENCY = auto()
    COST = auto()
    RELIABILITY = auto()


@dataclass
class StateEstimate:
    """Estimated state of the system.
    
    Attributes:
        position: Estimated position
        velocity: Estimated velocity
        uncertainty: Uncertainty covariance
        timestamp: When estimated
        source: Source of estimate
        confidence: Confidence level
    """
    
    position: Tuple[float, ...] = (0.0, 0.0)
    velocity: Tuple[float, ...] = (0.0, 0.0)
    uncertainty: float = 0.1
    timestamp: float = field(default_factory=time.perf_counter)
    source: str = "unknown"
    confidence: float = 1.0
    
    @property
    def speed(self) -> float:
        """Current speed."""
        return math.sqrt(sum(v ** 2 for v in self.velocity))
    
    def predict(self, dt: float) -> StateEstimate:
        """Predict future state.
        
        Args:
            dt: Time delta
            
        Returns:
            Predicted state
        """
        new_pos = tuple(p + v * dt for p, v in zip(self.position, self.velocity))
        new_uncertainty = self.uncertainty * (1.0 + 0.1 * dt)
        
        return StateEstimate(
            position=new_pos,
            velocity=self.velocity,
            uncertainty=new_uncertainty,
            timestamp=self.timestamp + dt,
            source=self.source,
            confidence=self.confidence * math.exp(-0.1 * dt),
        )


@dataclass
class ActionOption:
    """An action option to evaluate.
    
    Attributes:
        action_id: Unique identifier
        name: Action name
        parameters: Action parameters
        expected_outcome: Expected result
        risk_level: Risk (0-1)
        cost: Resource cost
        time_estimate: Time to complete
    """
    
    action_id: int
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None
    risk_level: float = 0.0
    cost: float = 0.0
    time_estimate: float = 0.0
    
    def __hash__(self) -> int:
        return hash(self.action_id)


@dataclass
class ActionSpace:
    """Available action space.
    
    Attributes:
        options: Available actions
        constraints: Constraints on actions
        history: Previous actions
    """
    
    options: List[ActionOption] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    history: List[int] = field(default_factory=list)
    
    def add_option(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        risk: float = 0.0,
        cost: float = 0.0,
        time: float = 0.0,
    ) -> ActionOption:
        """Add action option.
        
        Args:
            name: Action name
            parameters: Parameters
            risk: Risk level
            cost: Cost
            time: Time estimate
            
        Returns:
            Created option
        """
        option = ActionOption(
            action_id=len(self.options),
            name=name,
            parameters=parameters or {},
            risk_level=risk,
            cost=cost,
            time_estimate=time,
        )
        self.options.append(option)
        return option
    
    def filter_by_constraints(self) -> List[ActionOption]:
        """Get options satisfying constraints.
        
        Returns:
            Valid options
        """
        valid = []
        
        max_risk = self.constraints.get("max_risk", 1.0)
        max_cost = self.constraints.get("max_cost", float('inf'))
        max_time = self.constraints.get("max_time", float('inf'))
        
        for option in self.options:
            if (option.risk_level <= max_risk and
                option.cost <= max_cost and
                option.time_estimate <= max_time):
                valid.append(option)
        
        return valid


@dataclass
class Decision:
    """A decision result.
    
    Attributes:
        decision_type: Type of decision
        selected_action: Chosen action
        alternatives: Other options considered
        confidence: Decision confidence
        reasoning: Why selected
        timestamp: When decided
        evaluation_scores: Scores for each option
    """
    
    decision_type: DecisionType
    selected_action: Optional[ActionOption] = None
    alternatives: List[ActionOption] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""
    timestamp: float = field(default_factory=time.perf_counter)
    evaluation_scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class DecisionMakerConfig:
    """Configuration for decision maker.
    
    Attributes:
        criteria_weights: Weights for criteria
        risk_tolerance: Risk tolerance (0-1)
        min_confidence: Minimum confidence
        exploration_rate: Exploration vs exploitation
    """
    
    criteria_weights: Dict[DecisionCriteria, float] = field(default_factory=dict)
    risk_tolerance: float = 0.5
    min_confidence: float = 0.5
    exploration_rate: float = 0.1
    
    def __post_init__(self):
        if not self.criteria_weights:
            self.criteria_weights = {
                DecisionCriteria.SPEED: 0.2,
                DecisionCriteria.ACCURACY: 0.3,
                DecisionCriteria.SAFETY: 0.25,
                DecisionCriteria.EFFICIENCY: 0.15,
                DecisionCriteria.COST: 0.1,
            }


class DecisionMaker:
    """Autonomous decision maker.
    
    Evaluates options and makes decisions
    based on multiple criteria.
    """
    
    def __init__(
        self,
        config: Optional[DecisionMakerConfig] = None,
    ) -> None:
        """Initialize decision maker.
        
        Args:
            config: Configuration
        """
        self.config = config or DecisionMakerConfig()
        
        self.state: Optional[StateEstimate] = None
        self.action_space: Optional[ActionSpace] = None
        self.decision_history: List[Decision] = []
    
    def update_state(self, state: StateEstimate) -> None:
        """Update current state.
        
        Args:
            state: New state estimate
        """
        self.state = state
    
    def set_action_space(self, action_space: ActionSpace) -> None:
        """Set available actions.
        
        Args:
            action_space: Action space
        """
        self.action_space = action_space
    
    def evaluate_option(
        self,
        option: ActionOption,
        state: Optional[StateEstimate] = None,
    ) -> float:
        """Evaluate single option.
        
        Args:
            option: Option to evaluate
            state: Current state
            
        Returns:
            Score (higher is better)
        """
        state = state or self.state
        score = 0.0
        weights = self.config.criteria_weights
        
        # Speed: inverse of time
        if DecisionCriteria.SPEED in weights:
            speed_score = 1.0 / (option.time_estimate + 1.0)
            score += weights[DecisionCriteria.SPEED] * speed_score
        
        # Safety: inverse of risk
        if DecisionCriteria.SAFETY in weights:
            safety_score = 1.0 - option.risk_level
            score += weights[DecisionCriteria.SAFETY] * safety_score
        
        # Efficiency: value / cost
        if DecisionCriteria.EFFICIENCY in weights:
            efficiency_score = 1.0 / (option.cost + 1.0)
            score += weights[DecisionCriteria.EFFICIENCY] * efficiency_score
        
        # Cost: inverse of cost
        if DecisionCriteria.COST in weights:
            cost_score = 1.0 / (option.cost + 1.0)
            score += weights[DecisionCriteria.COST] * cost_score
        
        # Accuracy placeholder
        if DecisionCriteria.ACCURACY in weights:
            accuracy_score = 0.8  # Would be based on historical data
            score += weights[DecisionCriteria.ACCURACY] * accuracy_score
        
        # Apply risk tolerance
        if option.risk_level > self.config.risk_tolerance:
            penalty = (option.risk_level - self.config.risk_tolerance) * 2.0
            score *= max(0.1, 1.0 - penalty)
        
        return score
    
    def make_decision(
        self,
        decision_type: DecisionType = DecisionType.ACTION,
        force_exploration: bool = False,
    ) -> Decision:
        """Make a decision.
        
        Args:
            decision_type: Type of decision
            force_exploration: Force exploration
            
        Returns:
            Decision
        """
        if self.action_space is None:
            return Decision(
                decision_type=decision_type,
                confidence=0.0,
                reasoning="No action space defined",
            )
        
        # Get valid options
        valid_options = self.action_space.filter_by_constraints()
        
        if not valid_options:
            return Decision(
                decision_type=decision_type,
                confidence=0.0,
                reasoning="No valid options",
            )
        
        # Evaluate all options
        scores: Dict[int, float] = {}
        for option in valid_options:
            scores[option.action_id] = self.evaluate_option(option)
        
        # Exploration vs exploitation
        if force_exploration or np.random.random() < self.config.exploration_rate:
            # Random exploration
            selected = np.random.choice(valid_options)
            reasoning = "Exploration"
        else:
            # Greedy selection
            best_id = max(scores, key=lambda x: scores[x])
            selected = next(o for o in valid_options if o.action_id == best_id)
            reasoning = f"Best score: {scores[best_id]:.3f}"
        
        # Compute confidence
        if len(scores) > 1:
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, 0.5 + margin)
        else:
            confidence = 0.8
        
        # Create decision
        alternatives = [o for o in valid_options if o.action_id != selected.action_id]
        
        decision = Decision(
            decision_type=decision_type,
            selected_action=selected,
            alternatives=alternatives[:3],
            confidence=confidence,
            reasoning=reasoning,
            evaluation_scores=scores,
        )
        
        self.decision_history.append(decision)
        
        if self.action_space:
            self.action_space.history.append(selected.action_id)
        
        return decision
    
    def make_emergency_decision(
        self,
        threat_level: float,
    ) -> Decision:
        """Make emergency decision.
        
        Args:
            threat_level: Threat level (0-1)
            
        Returns:
            Emergency decision
        """
        if self.action_space is None:
            # Default emergency action
            emergency_action = ActionOption(
                action_id=-1,
                name="emergency_stop",
                risk_level=0.0,
            )
            return Decision(
                decision_type=DecisionType.EMERGENCY,
                selected_action=emergency_action,
                confidence=1.0,
                reasoning="No action space - default emergency stop",
            )
        
        # Find safest option
        valid_options = self.action_space.options
        
        safest = min(valid_options, key=lambda o: o.risk_level)
        
        return Decision(
            decision_type=DecisionType.EMERGENCY,
            selected_action=safest,
            confidence=1.0 - threat_level,
            reasoning=f"Emergency response to threat level {threat_level:.2f}",
        )
    
    def evaluate_past_decision(
        self,
        decision: Decision,
        actual_outcome: Dict[str, Any],
    ) -> float:
        """Evaluate past decision.
        
        Args:
            decision: Past decision
            actual_outcome: What happened
            
        Returns:
            Quality score
        """
        if decision.selected_action is None:
            return 0.0
        
        # Compare expected vs actual
        expected_time = decision.selected_action.time_estimate
        actual_time = actual_outcome.get("time", expected_time)
        
        time_error = abs(actual_time - expected_time) / (expected_time + 1.0)
        
        success = actual_outcome.get("success", True)
        
        if success:
            quality = 1.0 - 0.5 * min(1.0, time_error)
        else:
            quality = 0.2
        
        return quality
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.decision_history:
            return {"num_decisions": 0}
        
        total = len(self.decision_history)
        avg_confidence = sum(d.confidence for d in self.decision_history) / total
        
        action_counts: Dict[str, int] = {}
        for decision in self.decision_history:
            if decision.selected_action:
                name = decision.selected_action.name
                action_counts[name] = action_counts.get(name, 0) + 1
        
        return {
            "num_decisions": total,
            "average_confidence": avg_confidence,
            "action_distribution": action_counts,
            "decision_types": [d.decision_type.name for d in self.decision_history[-5:]],
        }


def make_decision(
    options: List[Dict[str, Any]],
    criteria_weights: Optional[Dict[str, float]] = None,
) -> Decision:
    """Convenience function for making decisions.
    
    Args:
        options: List of option dictionaries
        criteria_weights: Optional weights
        
    Returns:
        Decision
    """
    config = DecisionMakerConfig()
    
    if criteria_weights:
        config.criteria_weights = {
            DecisionCriteria[k.upper()]: v 
            for k, v in criteria_weights.items()
            if k.upper() in DecisionCriteria.__members__
        }
    
    maker = DecisionMaker(config)
    
    action_space = ActionSpace()
    for opt in options:
        action_space.add_option(
            name=opt.get("name", "unnamed"),
            parameters=opt.get("parameters", {}),
            risk=opt.get("risk", 0.0),
            cost=opt.get("cost", 0.0),
            time=opt.get("time", 1.0),
        )
    
    maker.set_action_space(action_space)
    return maker.make_decision()


def evaluate_options(
    options: List[Dict[str, Any]],
) -> List[Tuple[str, float]]:
    """Evaluate and rank options.
    
    Args:
        options: Options to evaluate
        
    Returns:
        List of (name, score) sorted by score
    """
    maker = DecisionMaker()
    
    results = []
    for opt in options:
        action = ActionOption(
            action_id=len(results),
            name=opt.get("name", "unnamed"),
            risk_level=opt.get("risk", 0.0),
            cost=opt.get("cost", 0.0),
            time_estimate=opt.get("time", 1.0),
        )
        score = maker.evaluate_option(action)
        results.append((action.name, score))
    
    return sorted(results, key=lambda x: -x[1])
