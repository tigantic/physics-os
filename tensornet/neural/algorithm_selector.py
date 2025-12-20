"""
Algorithm Selector Module
=========================

Neural network-based selection of optimal tensor network
algorithms based on problem characteristics.

Learns to recommend DMRG, TEBD, TDVP, etc. based on
system size, target accuracy, and available resources.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AlgorithmType(Enum):
    """Available tensor network algorithms."""
    
    DMRG = auto()          # Density Matrix Renormalization Group
    DMRG_X = auto()        # DMRG for excited states
    TEBD = auto()          # Time-Evolving Block Decimation
    TDVP_1 = auto()        # One-site TDVP
    TDVP_2 = auto()        # Two-site TDVP
    IMAGINARY_TEBD = auto()  # Imaginary time TEBD
    VARIATIONAL = auto()   # Variational optimization
    LANCZOS = auto()       # Lanczos diagonalization
    POWER_METHOD = auto()  # Power iteration
    
    @classmethod
    def from_index(cls, index: int) -> "AlgorithmType":
        """Get algorithm from index."""
        algorithms = list(cls)
        return algorithms[index % len(algorithms)]
    
    def to_index(self) -> int:
        """Get index of algorithm."""
        return list(AlgorithmType).index(self)
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            AlgorithmType.DMRG: "DMRG for ground states",
            AlgorithmType.DMRG_X: "DMRG for excited states",
            AlgorithmType.TEBD: "TEBD for real-time evolution",
            AlgorithmType.TDVP_1: "One-site TDVP (fixed chi)",
            AlgorithmType.TDVP_2: "Two-site TDVP (adaptive chi)",
            AlgorithmType.IMAGINARY_TEBD: "Imaginary TEBD for ground states",
            AlgorithmType.VARIATIONAL: "Direct variational optimization",
            AlgorithmType.LANCZOS: "Lanczos for few eigenvalues",
            AlgorithmType.POWER_METHOD: "Power iteration for largest eigenvalue",
        }
        return descriptions.get(self, "Unknown algorithm")


class SelectionCriteria(Enum):
    """Criteria for algorithm selection."""
    
    ACCURACY = auto()      # Prioritize accuracy
    SPEED = auto()         # Prioritize speed
    MEMORY = auto()        # Prioritize memory efficiency
    BALANCED = auto()      # Balanced trade-off
    REALTIME = auto()      # Real-time constraints
    
    def get_weights(self) -> Tuple[float, float, float]:
        """Get (accuracy, speed, memory) weights."""
        weights = {
            SelectionCriteria.ACCURACY: (0.8, 0.1, 0.1),
            SelectionCriteria.SPEED: (0.1, 0.8, 0.1),
            SelectionCriteria.MEMORY: (0.1, 0.1, 0.8),
            SelectionCriteria.BALANCED: (0.33, 0.34, 0.33),
            SelectionCriteria.REALTIME: (0.2, 0.7, 0.1),
        }
        return weights.get(self, (0.33, 0.34, 0.33))


@dataclass
class ProblemFeatures:
    """Features describing a tensor network problem.
    
    Attributes:
        num_sites: Number of sites in the system
        local_dim: Local Hilbert space dimension
        target_accuracy: Target accuracy (e.g., 1e-10)
        max_bond_dim: Maximum bond dimension allowed
        is_ground_state: Whether seeking ground state
        is_dynamics: Whether doing time evolution
        evolution_time: Total evolution time (if dynamics)
        num_eigenvalues: Number of eigenvalues needed
        is_periodic: Whether boundary conditions are periodic
        has_long_range: Whether Hamiltonian has long-range terms
        memory_limit_gb: Memory limit in GB
        time_limit_seconds: Time limit in seconds
    """
    
    num_sites: int
    local_dim: int = 2
    target_accuracy: float = 1e-10
    max_bond_dim: int = 256
    is_ground_state: bool = True
    is_dynamics: bool = False
    evolution_time: float = 0.0
    num_eigenvalues: int = 1
    is_periodic: bool = False
    has_long_range: bool = False
    memory_limit_gb: float = 8.0
    time_limit_seconds: float = 3600.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to feature tensor."""
        return torch.tensor([
            math.log(self.num_sites) / 10.0,
            math.log(self.local_dim) / 5.0,
            -math.log10(self.target_accuracy) / 16.0,
            math.log(self.max_bond_dim) / 10.0,
            float(self.is_ground_state),
            float(self.is_dynamics),
            math.log(1 + self.evolution_time) / 5.0,
            math.log(1 + self.num_eigenvalues) / 3.0,
            float(self.is_periodic),
            float(self.has_long_range),
            math.log(self.memory_limit_gb) / 5.0,
            math.log(self.time_limit_seconds) / 10.0,
        ], dtype=torch.float32)
    
    def estimate_difficulty(self) -> float:
        """Estimate problem difficulty (0-1)."""
        # Size contribution
        size_factor = min(1.0, self.num_sites / 100.0)
        
        # Accuracy contribution
        accuracy_factor = min(1.0, -math.log10(self.target_accuracy) / 16.0)
        
        # Entanglement estimate
        entanglement_factor = 0.5
        if self.is_dynamics:
            entanglement_factor = min(1.0, self.evolution_time / 10.0)
        
        # Periodic boundaries are harder
        periodic_factor = 0.2 if self.is_periodic else 0.0
        
        # Long-range interactions are harder
        long_range_factor = 0.3 if self.has_long_range else 0.0
        
        difficulty = (
            0.3 * size_factor
            + 0.3 * accuracy_factor
            + 0.2 * entanglement_factor
            + 0.1 * periodic_factor
            + 0.1 * long_range_factor
        )
        
        return min(1.0, difficulty)


@dataclass
class AlgorithmRecommendation:
    """Recommendation for algorithm selection.
    
    Attributes:
        algorithm: Recommended algorithm
        confidence: Confidence in recommendation
        estimated_time: Estimated runtime
        estimated_memory: Estimated memory usage
        fallback: Fallback algorithm if primary fails
        parameters: Recommended algorithm parameters
        reasoning: Explanation for recommendation
    """
    
    algorithm: AlgorithmType
    confidence: float
    estimated_time: float
    estimated_memory: float
    fallback: Optional[AlgorithmType]
    parameters: Dict[str, Any]
    reasoning: str
    
    @classmethod
    def from_scores(
        cls,
        scores: torch.Tensor,
        problem: ProblemFeatures,
        criteria: SelectionCriteria,
    ) -> "AlgorithmRecommendation":
        """Create from algorithm scores.
        
        Args:
            scores: Score tensor for each algorithm
            problem: Problem features
            criteria: Selection criteria
            
        Returns:
            AlgorithmRecommendation instance
        """
        # Apply softmax for confidence
        probs = F.softmax(scores, dim=-1)
        
        # Get top algorithm
        top_idx = int(torch.argmax(probs))
        algorithm = AlgorithmType.from_index(top_idx)
        confidence = float(probs[top_idx])
        
        # Get fallback (second best)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        fallback_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx
        fallback = AlgorithmType.from_index(fallback_idx)
        
        # Estimate runtime and memory
        difficulty = problem.estimate_difficulty()
        base_time = 10.0 * (1 + difficulty * 10)
        base_memory = 0.1 * problem.num_sites * (problem.max_bond_dim ** 2) / 1e9
        
        # Algorithm-specific adjustments
        time_factors = {
            AlgorithmType.DMRG: 1.0,
            AlgorithmType.DMRG_X: 1.5,
            AlgorithmType.TEBD: 0.5 * (1 + problem.evolution_time),
            AlgorithmType.TDVP_1: 0.7,
            AlgorithmType.TDVP_2: 1.2,
            AlgorithmType.IMAGINARY_TEBD: 1.3,
            AlgorithmType.VARIATIONAL: 2.0,
            AlgorithmType.LANCZOS: 0.3,
            AlgorithmType.POWER_METHOD: 0.1,
        }
        
        memory_factors = {
            AlgorithmType.DMRG: 1.0,
            AlgorithmType.DMRG_X: 1.2,
            AlgorithmType.TEBD: 0.8,
            AlgorithmType.TDVP_1: 0.9,
            AlgorithmType.TDVP_2: 1.1,
            AlgorithmType.IMAGINARY_TEBD: 0.8,
            AlgorithmType.VARIATIONAL: 0.7,
            AlgorithmType.LANCZOS: 1.5,
            AlgorithmType.POWER_METHOD: 0.5,
        }
        
        estimated_time = base_time * time_factors.get(algorithm, 1.0)
        estimated_memory = base_memory * memory_factors.get(algorithm, 1.0)
        
        # Generate parameters
        parameters = _generate_parameters(algorithm, problem)
        
        # Generate reasoning
        reasoning = _generate_reasoning(algorithm, problem, criteria, confidence)
        
        return cls(
            algorithm=algorithm,
            confidence=confidence,
            estimated_time=estimated_time,
            estimated_memory=estimated_memory,
            fallback=fallback if fallback != algorithm else None,
            parameters=parameters,
            reasoning=reasoning,
        )


def _generate_parameters(
    algorithm: AlgorithmType,
    problem: ProblemFeatures,
) -> Dict[str, Any]:
    """Generate recommended parameters for algorithm."""
    params: Dict[str, Any] = {}
    
    if algorithm in (AlgorithmType.DMRG, AlgorithmType.DMRG_X):
        params["num_sweeps"] = max(10, int(50 * problem.estimate_difficulty()))
        params["chi_max"] = problem.max_bond_dim
        params["cutoff"] = problem.target_accuracy / 10
        
    elif algorithm == AlgorithmType.TEBD:
        params["dt"] = min(0.1, 1.0 / problem.num_sites)
        params["num_steps"] = max(10, int(problem.evolution_time / params["dt"]))
        params["chi_max"] = problem.max_bond_dim
        params["order"] = 2
        
    elif algorithm in (AlgorithmType.TDVP_1, AlgorithmType.TDVP_2):
        params["dt"] = min(0.1, 1.0 / problem.num_sites)
        params["chi_max"] = problem.max_bond_dim
        params["cutoff"] = problem.target_accuracy / 10
        
    elif algorithm == AlgorithmType.IMAGINARY_TEBD:
        params["beta"] = max(10.0, 50.0 * problem.estimate_difficulty())
        params["d_beta"] = 0.1
        params["chi_max"] = problem.max_bond_dim
        
    elif algorithm == AlgorithmType.VARIATIONAL:
        params["max_iterations"] = 1000
        params["learning_rate"] = 1e-3
        params["tolerance"] = problem.target_accuracy
        
    elif algorithm == AlgorithmType.LANCZOS:
        params["num_eigenvalues"] = problem.num_eigenvalues
        params["max_iterations"] = 100
        params["tolerance"] = problem.target_accuracy
        
    elif algorithm == AlgorithmType.POWER_METHOD:
        params["max_iterations"] = 1000
        params["tolerance"] = problem.target_accuracy
    
    return params


def _generate_reasoning(
    algorithm: AlgorithmType,
    problem: ProblemFeatures,
    criteria: SelectionCriteria,
    confidence: float,
) -> str:
    """Generate explanation for algorithm recommendation."""
    parts = []
    
    parts.append(f"Selected {algorithm.name} with {confidence:.1%} confidence.")
    
    if problem.is_ground_state and algorithm == AlgorithmType.DMRG:
        parts.append("DMRG is optimal for ground state problems with short-range interactions.")
    elif problem.is_dynamics and algorithm == AlgorithmType.TEBD:
        parts.append("TEBD is efficient for real-time dynamics with nearest-neighbor terms.")
    elif problem.is_dynamics and algorithm in (AlgorithmType.TDVP_1, AlgorithmType.TDVP_2):
        parts.append("TDVP preserves the variational principle during time evolution.")
    
    if problem.has_long_range:
        parts.append("Long-range interactions may require higher bond dimensions.")
    
    if problem.is_periodic:
        parts.append("Periodic boundaries increase computational cost.")
    
    difficulty = problem.estimate_difficulty()
    if difficulty > 0.7:
        parts.append(f"High difficulty ({difficulty:.2f}): expect longer runtime.")
    
    weights = criteria.get_weights()
    if weights[1] > 0.5:  # Speed priority
        parts.append("Speed prioritized in selection.")
    elif weights[0] > 0.5:  # Accuracy priority
        parts.append("Accuracy prioritized in selection.")
    
    return " ".join(parts)


class AlgorithmSelectorNetwork(nn.Module):
    """Neural network for algorithm selection."""
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_algorithms: int = 9,
    ) -> None:
        """Initialize network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_algorithms: Number of algorithm choices
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_algorithms = num_algorithms
        
        self.network = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # +3 for criteria weights
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_algorithms),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        criteria_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Problem features
            criteria_weights: Selection criteria weights
            
        Returns:
            Algorithm scores
        """
        x = torch.cat([features, criteria_weights], dim=-1)
        return self.network(x)


class AlgorithmSelector:
    """Algorithm selector using neural network.
    
    Recommends optimal tensor network algorithm based on
    problem characteristics and selection criteria.
    
    Attributes:
        network: The selection network
        device: Computation device
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        """Initialize selector.
        
        Args:
            hidden_dim: Hidden layer dimension
            device: Computation device
        """
        self.device = device
        self.network = AlgorithmSelectorNetwork(hidden_dim=hidden_dim).to(device)
        self.network.eval()
        
        # Training data
        self.training_data: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self.optimizer: Optional[optim.Optimizer] = None
    
    def select(
        self,
        problem: ProblemFeatures,
        criteria: SelectionCriteria = SelectionCriteria.BALANCED,
    ) -> AlgorithmRecommendation:
        """Select optimal algorithm.
        
        Args:
            problem: Problem features
            criteria: Selection criteria
            
        Returns:
            AlgorithmRecommendation
        """
        features = problem.to_tensor().to(self.device)
        weights = torch.tensor(criteria.get_weights(), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            scores = self.network(features.unsqueeze(0), weights.unsqueeze(0)).squeeze(0)
        
        return AlgorithmRecommendation.from_scores(scores, problem, criteria)
    
    def select_with_heuristics(
        self,
        problem: ProblemFeatures,
        criteria: SelectionCriteria = SelectionCriteria.BALANCED,
    ) -> AlgorithmRecommendation:
        """Select algorithm using rule-based heuristics.
        
        Falls back to heuristics for interpretability.
        
        Args:
            problem: Problem features
            criteria: Selection criteria
            
        Returns:
            AlgorithmRecommendation
        """
        # Build score tensor based on heuristics
        scores = torch.zeros(len(AlgorithmType))
        
        # Ground state problems
        if problem.is_ground_state and not problem.is_dynamics:
            scores[AlgorithmType.DMRG.to_index()] += 3.0
            scores[AlgorithmType.IMAGINARY_TEBD.to_index()] += 2.0
            scores[AlgorithmType.VARIATIONAL.to_index()] += 1.0
            
            if problem.num_eigenvalues > 1:
                scores[AlgorithmType.DMRG_X.to_index()] += 2.0
                scores[AlgorithmType.LANCZOS.to_index()] += 1.5
        
        # Dynamics problems
        if problem.is_dynamics:
            scores[AlgorithmType.TEBD.to_index()] += 3.0
            scores[AlgorithmType.TDVP_2.to_index()] += 2.5
            scores[AlgorithmType.TDVP_1.to_index()] += 2.0
            
            if problem.has_long_range:
                scores[AlgorithmType.TDVP_2.to_index()] += 1.0
                scores[AlgorithmType.TEBD.to_index()] -= 1.0
        
        # Small systems
        if problem.num_sites < 20:
            scores[AlgorithmType.LANCZOS.to_index()] += 1.0
        
        # Speed criteria
        weights = criteria.get_weights()
        if weights[1] > 0.5:  # Speed priority
            scores[AlgorithmType.TEBD.to_index()] += 0.5
            scores[AlgorithmType.POWER_METHOD.to_index()] += 0.5
            scores[AlgorithmType.VARIATIONAL.to_index()] -= 0.5
        
        # Memory criteria
        if weights[2] > 0.5:  # Memory priority
            scores[AlgorithmType.TDVP_1.to_index()] += 0.5
            scores[AlgorithmType.TEBD.to_index()] += 0.5
            scores[AlgorithmType.LANCZOS.to_index()] -= 1.0
        
        return AlgorithmRecommendation.from_scores(scores, problem, criteria)
    
    def add_training_sample(
        self,
        problem: ProblemFeatures,
        criteria: SelectionCriteria,
        best_algorithm: AlgorithmType,
    ) -> None:
        """Add training sample.
        
        Args:
            problem: Problem features
            criteria: Selection criteria
            best_algorithm: Known best algorithm
        """
        features = problem.to_tensor()
        weights = torch.tensor(criteria.get_weights(), dtype=torch.float32)
        target = best_algorithm.to_index()
        
        self.training_data.append((features, weights, target))
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train the selector.
        
        Args:
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if len(self.training_data) < batch_size:
            return {"loss": [], "accuracy": []}
        
        self.network.train()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        history: Dict[str, List[float]] = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            import random
            random.shuffle(self.training_data)
            
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i + batch_size]
                if len(batch) < 2:
                    continue
                
                features = torch.stack([x[0] for x in batch]).to(self.device)
                weights = torch.stack([x[1] for x in batch]).to(self.device)
                targets = torch.tensor([x[2] for x in batch]).to(self.device)
                
                # Forward
                scores = self.network(features, weights)
                loss = F.cross_entropy(scores, targets)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                correct += (scores.argmax(dim=-1) == targets).sum().item()
                total += len(batch)
            
            history["loss"].append(epoch_loss / max(1, len(self.training_data) // batch_size))
            history["accuracy"].append(correct / max(1, total))
        
        self.network.eval()
        
        return history
    
    def save(self, path: Union[str, Path]) -> None:
        """Save selector to file."""
        path = Path(path)
        torch.save({
            "network_state": self.network.state_dict(),
            "input_dim": self.network.input_dim,
            "hidden_dim": self.network.hidden_dim,
            "num_algorithms": self.network.num_algorithms,
        }, path)
    
    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "AlgorithmSelector":
        """Load selector from file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        
        selector = cls(hidden_dim=checkpoint["hidden_dim"], device=device)
        selector.network.load_state_dict(checkpoint["network_state"])
        
        return selector


def select_algorithm(
    problem: ProblemFeatures,
    criteria: SelectionCriteria = SelectionCriteria.BALANCED,
    selector: Optional[AlgorithmSelector] = None,
    use_heuristics: bool = True,
) -> AlgorithmRecommendation:
    """Select optimal algorithm for problem.
    
    Args:
        problem: Problem features
        criteria: Selection criteria
        selector: Optional trained selector
        use_heuristics: Whether to use heuristic fallback
        
    Returns:
        AlgorithmRecommendation
    """
    if selector is None:
        selector = AlgorithmSelector()
        if use_heuristics:
            return selector.select_with_heuristics(problem, criteria)
    
    return selector.select(problem, criteria)


@dataclass
class BenchmarkResult:
    """Result of algorithm benchmarking."""
    
    algorithm: AlgorithmType
    runtime: float
    memory_usage: float
    accuracy: float
    success: bool
    error_message: Optional[str] = None


def benchmark_algorithms(
    problem: ProblemFeatures,
    algorithms: Optional[List[AlgorithmType]] = None,
    run_fn: Optional[Callable[[AlgorithmType, ProblemFeatures], BenchmarkResult]] = None,
) -> List[BenchmarkResult]:
    """Benchmark multiple algorithms on a problem.
    
    Args:
        problem: Problem to benchmark
        algorithms: List of algorithms to benchmark
        run_fn: Function to run each algorithm
        
    Returns:
        List of BenchmarkResult
    """
    if algorithms is None:
        algorithms = list(AlgorithmType)
    
    results = []
    
    for algorithm in algorithms:
        if run_fn is not None:
            result = run_fn(algorithm, problem)
        else:
            # Simulated benchmark
            difficulty = problem.estimate_difficulty()
            runtime = 1.0 + difficulty * 10 * np.random.uniform(0.5, 1.5)
            memory = 0.1 + difficulty * np.random.uniform(0.5, 1.5)
            accuracy = 1e-10 * (1 + difficulty * 100)
            
            result = BenchmarkResult(
                algorithm=algorithm,
                runtime=runtime,
                memory_usage=memory,
                accuracy=accuracy,
                success=True,
            )
        
        results.append(result)
    
    return results
