"""
Topology of Aging — Persistent homology of aging trajectories.

TENSOR GENESIS Protocol — Layer 27 (Aging)
Phase 21 of the Civilization Stack

The aging trajectory lives on a manifold in TT-rank space. As a cell ages,
its state traces a path through the space of biological configurations.
Persistent homology captures the topological features of this trajectory:

- H₀ (connected components): distinct aging regimes (pre-senescence,
  senescence, post-senescence)
- H₁ (loops): cyclic regulatory patterns that persist across ages
  (circadian, cell cycle, seasonal)
- H₂ (voids): irreversible transitions (points of no return in aging)

Key applications:
1. Detect phase transitions in aging (e.g., onset of senescence)
2. Identify topological barriers to rejuvenation
3. Compare aging trajectories across cell types
4. Find the "shortest path" back to youth (geodesic in state space)

Integration with QTT-PH:
    The distance matrix between aging trajectory points is computed in
    TT format using the CellStateTensor.distance() method. This feeds
    into the Vietoris-Rips complex construction from QTT-PH (Layer 25).

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ontic.genesis.aging.cell_state import (
    AgingHallmark,
    BiologicalMode,
    CellStateTensor,
    _tt_inner,
    _tt_norm,
)
from ontic.genesis.aging.dynamics import AgingTrajectory


# ---------------------------------------------------------------------------
# Aging Phase Detection
# ---------------------------------------------------------------------------

@dataclass
class AgingPhase:
    """
    A detected phase in the aging trajectory.

    Phases are contiguous age ranges where the aging dynamics have
    consistent topological character (stable rank growth rate,
    consistent mode dominance).

    Attributes
    ----------
    start_age : float
        Beginning of this phase (years).
    end_age : float
        End of this phase (years).
    label : str
        Human-readable phase name.
    mean_rank_rate : float
        Average rank growth rate in this phase.
    dominant_hallmarks : list of AgingHallmark
        Which hallmarks dominate this phase.
    persistence : float
        Topological persistence of this phase (robustness).
    """
    start_age: float
    end_age: float
    label: str
    mean_rank_rate: float
    dominant_hallmarks: List[AgingHallmark]
    persistence: float

    @property
    def duration(self) -> float:
        """Phase duration in years."""
        return self.end_age - self.start_age


@dataclass
class TopologicalBarrier:
    """
    A topological barrier to rejuvenation.

    Barriers are points in the aging trajectory where the state space
    geometry changes irreversibly — the homology of the accessible
    states changes, creating a topological obstruction to reversal.

    In biological terms: an irreversible commitment to senescence,
    terminal differentiation, or apoptotic pathway.

    Attributes
    ----------
    age : float
        Age at which the barrier occurs.
    barrier_type : str
        Description of the topological change.
    persistence : float
        How robust the barrier is (higher = harder to overcome).
    dimension : int
        Homological dimension (0 = disconnection, 1 = loop closure,
        2 = void creation).
    modes_involved : list of BiologicalMode
        Which biological modes participate in the barrier.
    """
    age: float
    barrier_type: str
    persistence: float
    dimension: int
    modes_involved: List[BiologicalMode]


@dataclass
class AgingTopology:
    """
    Complete topological analysis of an aging trajectory.

    Attributes
    ----------
    phases : list of AgingPhase
        Detected aging phases.
    barriers : list of TopologicalBarrier
        Detected topological barriers.
    betti_numbers : list of list of int
        Betti numbers at each timepoint: betti_numbers[t][k] = β_k(t).
    distance_matrix : np.ndarray
        Pairwise distance matrix between trajectory points.
    persistence_pairs : list of tuple
        (birth, death, dimension) persistence pairs.
    total_persistence : float
        Sum of all persistence values.
    """
    phases: List[AgingPhase]
    barriers: List[TopologicalBarrier]
    betti_numbers: List[List[int]]
    distance_matrix: np.ndarray
    persistence_pairs: List[Tuple[float, float, int]]
    total_persistence: float


# ---------------------------------------------------------------------------
# Topological Analysis Engine
# ---------------------------------------------------------------------------

class AgingTopologyAnalyzer:
    """
    Analyze the topology of aging trajectories.

    Uses persistent homology concepts to detect phase transitions,
    topological barriers, and aging regimes in the cell state space.

    The analyzer works with the pairwise distance matrix between
    trajectory points and constructs a filtration to compute
    persistent features.

    Parameters
    ----------
    max_dimension : int
        Maximum homological dimension to compute (0, 1, or 2).
    persistence_threshold : float
        Minimum persistence to report a feature.
    """

    __slots__ = ("max_dimension", "persistence_threshold")

    def __init__(
        self,
        max_dimension: int = 1,
        persistence_threshold: float = 0.05,
    ) -> None:
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold

    def analyze(self, trajectory: AgingTrajectory) -> AgingTopology:
        """
        Perform complete topological analysis of an aging trajectory.

        Parameters
        ----------
        trajectory : AgingTrajectory
            The aging trajectory to analyze.

        Returns
        -------
        AgingTopology
            Complete topological analysis.
        """
        # Step 1: Compute distance matrix
        D = trajectory.distance_matrix()
        n = len(trajectory.states)

        # Step 2: Compute persistence pairs via Vietoris-Rips filtration
        persistence_pairs = self._compute_persistence(D)

        # Step 3: Detect aging phases from H₀ components
        phases = self._detect_phases(trajectory, D, persistence_pairs)

        # Step 4: Detect topological barriers from dying features
        barriers = self._detect_barriers(trajectory, persistence_pairs)

        # Step 5: Compute Betti numbers at each timepoint
        betti_numbers = self._compute_betti_curve(D, persistence_pairs, n)

        # Step 6: Total persistence
        total_pers = sum(
            death - birth
            for birth, death, dim in persistence_pairs
            if death < float("inf")
        )

        return AgingTopology(
            phases=phases,
            barriers=barriers,
            betti_numbers=betti_numbers,
            distance_matrix=D,
            persistence_pairs=persistence_pairs,
            total_persistence=total_pers,
        )

    def _compute_persistence(
        self, D: np.ndarray
    ) -> List[Tuple[float, float, int]]:
        """
        Compute persistence pairs from a distance matrix using
        a simplified Vietoris-Rips filtration.

        For aging trajectories, the distance matrix has a natural ordering
        (time), so the filtration is particularly well-structured.

        Implementation: incremental construction following the filtration
        order, tracking connected components (H₀) and cycles (H₁).
        """
        n = D.shape[0]
        pairs: List[Tuple[float, float, int]] = []

        # ---- H₀: Connected components ----
        # Union-Find for tracking components
        parent = list(range(n))
        rank_uf = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank_uf[rx] < rank_uf[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank_uf[rx] == rank_uf[ry]:
                rank_uf[rx] += 1
            return True

        # Sort edges by distance
        edges: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((D[i, j], i, j))
        edges.sort()

        # Process edges — each component merge creates a death event
        # All points are born at filtration value 0
        component_births: Dict[int, float] = {i: 0.0 for i in range(n)}
        cycle_candidates: List[Tuple[float, int, int]] = []

        for dist, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                # Merge: the younger component dies
                younger = rj if component_births.get(rj, 0) >= component_births.get(ri, 0) else ri
                birth = component_births.get(younger, 0.0)
                death = dist
                if death - birth >= self.persistence_threshold:
                    pairs.append((birth, death, 0))
                union(i, j)
            else:
                # Same component — potential cycle (H₁ candidate)
                if self.max_dimension >= 1:
                    cycle_candidates.append((dist, i, j))

        # The longest-living H₀ component (the one that persists to infinity)
        pairs.append((0.0, float("inf"), 0))

        # ---- H₁: Cycles (simplified) ----
        if self.max_dimension >= 1 and cycle_candidates:
            # Heuristic: cycles are born when a same-component edge appears
            # and die when a shorter alternative path is available.
            # For aging trajectories, we detect cycles in the rank dynamics.
            for cycle_birth, ci, cj in cycle_candidates[:20]:  # Limit to top 20
                # Estimate cycle death: when a later edge closes the cycle
                # Use the shortest path distance change as a proxy
                if ci < n - 1 and cj < n - 1:
                    # Death ≈ max distance in the cycle region
                    region_max = max(
                        D[ci, cj],
                        D[min(ci + 1, n - 1), min(cj + 1, n - 1)]
                        if ci + 1 < n and cj + 1 < n
                        else D[ci, cj],
                    )
                    cycle_death = region_max * 1.5
                    if cycle_death - cycle_birth >= self.persistence_threshold:
                        pairs.append((cycle_birth, cycle_death, 1))

        return pairs

    def _detect_phases(
        self,
        trajectory: AgingTrajectory,
        D: np.ndarray,
        persistence_pairs: List[Tuple[float, float, int]],
    ) -> List[AgingPhase]:
        """
        Detect distinct aging phases from the trajectory topology.

        Phases are detected by:
        1. Segmenting the rank growth curve at inflection points
        2. Assigning labels based on dominant aging hallmarks
        3. Computing persistence for each phase
        """
        n = len(trajectory.states)
        if n < 3:
            return [
                AgingPhase(
                    start_age=trajectory.ages[0],
                    end_age=trajectory.ages[-1],
                    label="single_phase",
                    mean_rank_rate=trajectory.rank_growth_rate,
                    dominant_hallmarks=[],
                    persistence=trajectory.age_span,
                )
            ]

        # Compute rank growth rate at each timepoint
        rates: List[float] = [0.0]
        for i in range(1, n):
            dt = trajectory.ages[i] - trajectory.ages[i - 1]
            if dt > 1e-10:
                dr = float(trajectory.ranks[i] - trajectory.ranks[i - 1])
                rates.append(dr / dt)
            else:
                rates.append(rates[-1] if rates else 0.0)

        # Find phase boundaries via change-point detection
        # Simple approach: segment where rate changes by > 50%
        boundaries: List[int] = [0]
        running_rate = rates[1] if len(rates) > 1 else 0.0

        for i in range(2, n):
            if running_rate > 1e-10:
                change = abs(rates[i] - running_rate) / running_rate
                if change > 0.5:  # 50% change
                    boundaries.append(i)
                    running_rate = rates[i]
            else:
                running_rate = rates[i]
        boundaries.append(n - 1)

        # Build phases
        phases: List[AgingPhase] = []
        for b in range(len(boundaries) - 1):
            start_idx = boundaries[b]
            end_idx = boundaries[b + 1]
            start_age = trajectory.ages[start_idx]
            end_age = trajectory.ages[end_idx]

            # Mean rate in this phase
            phase_rates = rates[start_idx:end_idx + 1]
            mean_rate = float(np.mean(phase_rates)) if phase_rates else 0.0

            # Dominant hallmarks from the middle state
            mid_idx = (start_idx + end_idx) // 2
            sig = trajectory.states[mid_idx].aging_signature()
            sorted_hallmarks = sorted(
                sig.hallmark_scores.items(), key=lambda x: x[1], reverse=True
            )
            dominant = [h for h, s in sorted_hallmarks[:3] if s > 0.1]

            # Label based on age range and dominant features
            if start_age < 20:
                label = "development"
            elif mean_rate < 0.5:
                label = "maintenance"
            elif mean_rate < 2.0:
                label = "gradual_aging"
            elif mean_rate < 5.0:
                label = "accelerated_aging"
            else:
                label = "senescence_crisis"

            # Persistence = phase duration (simple proxy)
            persistence = end_age - start_age

            phases.append(AgingPhase(
                start_age=start_age,
                end_age=end_age,
                label=label,
                mean_rank_rate=mean_rate,
                dominant_hallmarks=dominant,
                persistence=persistence,
            ))

        return phases

    def _detect_barriers(
        self,
        trajectory: AgingTrajectory,
        persistence_pairs: List[Tuple[float, float, int]],
    ) -> List[TopologicalBarrier]:
        """
        Detect topological barriers from persistence pair deaths.

        A barrier occurs when a topological feature dies (H₀ component
        merges or H₁ cycle fills in), indicating an irreversible
        transition in the aging process.
        """
        barriers: List[TopologicalBarrier] = []

        # Each death in persistence corresponds to a topological barrier
        for birth, death, dim in persistence_pairs:
            if death == float("inf"):
                continue  # Essential feature, no barrier
            persistence = death - birth
            if persistence < self.persistence_threshold:
                continue

            # Map filtration value to age
            # The filtration value is a distance, not directly an age.
            # We use the birth/death indices to find the corresponding ages.
            age_at_barrier = self._filtration_to_age(death, trajectory)

            if dim == 0:
                barrier_type = "state_space_disconnection"
                modes = [BiologicalMode.GENE_EXPRESSION]
            elif dim == 1:
                barrier_type = "regulatory_cycle_collapse"
                modes = [BiologicalMode.SIGNALING, BiologicalMode.GENE_EXPRESSION]
            else:
                barrier_type = "irreversible_void"
                modes = [
                    BiologicalMode.METHYLATION,
                    BiologicalMode.CHROMATIN_ACCESS,
                ]

            barriers.append(TopologicalBarrier(
                age=age_at_barrier,
                barrier_type=barrier_type,
                persistence=persistence,
                dimension=dim,
                modes_involved=modes,
            ))

        # Sort by age
        barriers.sort(key=lambda b: b.age)
        return barriers

    def _compute_betti_curve(
        self,
        D: np.ndarray,
        persistence_pairs: List[Tuple[float, float, int]],
        n_timepoints: int,
    ) -> List[List[int]]:
        """
        Compute Betti numbers at each timepoint.

        β_k(t) = number of persistence pairs with birth ≤ t < death in dimension k.
        """
        # Use distance thresholds as filtration values
        max_dist = D.max() if D.size > 0 else 1.0
        thresholds = np.linspace(0, max_dist, n_timepoints)

        betti_curves: List[List[int]] = []
        for t_val in thresholds:
            betti = [0] * (self.max_dimension + 1)
            for birth, death, dim in persistence_pairs:
                if dim <= self.max_dimension and birth <= t_val < death:
                    betti[dim] += 1
            betti_curves.append(betti)

        return betti_curves

    def _filtration_to_age(
        self, filtration_value: float, trajectory: AgingTrajectory
    ) -> float:
        """
        Map a filtration value (distance) to an approximate age.

        Uses the trajectory's distance-to-age relationship.
        """
        if len(trajectory.ages) < 2:
            return trajectory.ages[0] if trajectory.ages else 0.0

        # Find the first pair of consecutive states whose distance
        # exceeds the filtration value
        D = trajectory.distance_matrix()
        cumulative_dist = 0.0
        for i in range(len(trajectory.ages) - 1):
            step_dist = D[i, i + 1]
            if cumulative_dist + step_dist >= filtration_value:
                # Interpolate
                frac = (
                    (filtration_value - cumulative_dist) / step_dist
                    if step_dist > 1e-10
                    else 0.0
                )
                return trajectory.ages[i] + frac * (
                    trajectory.ages[i + 1] - trajectory.ages[i]
                )
            cumulative_dist += step_dist

        return trajectory.ages[-1]


# ---------------------------------------------------------------------------
# Rejuvenation Path (Geodesic in State Space)
# ---------------------------------------------------------------------------

@dataclass
class RejuvenationPath:
    """
    A geodesic path from aged state to young state in the cell state space.

    This represents the optimal rejuvenation trajectory — the shortest
    path through state space that reverses aging while maintaining
    cell viability at every intermediate step.

    Attributes
    ----------
    waypoints : list of CellStateTensor
        Intermediate states along the path.
    ages : list of float
        Biological age at each waypoint.
    ranks : list of int
        TT rank at each waypoint.
    total_distance : float
        Total path length in state space.
    barriers_crossed : list of TopologicalBarrier
        Topological barriers that must be overcome.
    """
    waypoints: List[CellStateTensor]
    ages: List[float]
    ranks: List[int]
    total_distance: float
    barriers_crossed: List[TopologicalBarrier]


def compute_rejuvenation_path(
    aged_state: CellStateTensor,
    young_state: CellStateTensor,
    n_waypoints: int = 10,
) -> RejuvenationPath:
    """
    Compute an approximate geodesic rejuvenation path.

    Constructs a sequence of intermediate states that interpolate
    between aged and young by progressively reducing TT rank.

    The interpolation is NOT linear in state space (that would pass
    through unphysical regions). Instead, it follows a path of
    progressive rank reduction:

    ψ(α) = round(ψ_aged, rank = r_young + (1-α)(r_aged - r_young))

    for α ∈ [0, 1], which keeps the state on the low-rank manifold
    at every step.

    Parameters
    ----------
    aged_state : CellStateTensor
        Starting (aged) state.
    young_state : CellStateTensor
        Target (young) state.
    n_waypoints : int
        Number of intermediate waypoints.

    Returns
    -------
    RejuvenationPath
        The computed rejuvenation path.
    """
    r_aged = aged_state.max_rank
    r_young = young_state.max_rank

    waypoints: List[CellStateTensor] = [aged_state]
    ages: List[float] = []
    ranks: List[int] = [r_aged]

    for i in range(1, n_waypoints + 1):
        alpha = i / n_waypoints
        target_rank = max(
            r_young,
            int(r_aged + alpha * (r_young - r_aged)),
        )

        # Progressive compression
        intermediate = aged_state.compress(max_rank=target_rank, tol=1e-12)

        # Estimate biological age from rank
        sig = intermediate.aging_signature()
        ages.append(sig.biological_age)
        ranks.append(intermediate.max_rank)
        waypoints.append(intermediate)

    # Insert initial age
    ages.insert(0, aged_state.aging_signature().biological_age)

    # Compute total path distance
    total_dist = 0.0
    for i in range(len(waypoints) - 1):
        total_dist += waypoints[i].distance(waypoints[i + 1])

    # Detect barriers along the path
    analyzer = AgingTopologyAnalyzer()
    traj = AgingTrajectory(states=waypoints)
    topology = analyzer.analyze(traj)

    return RejuvenationPath(
        waypoints=waypoints,
        ages=ages,
        ranks=ranks,
        total_distance=total_dist,
        barriers_crossed=topology.barriers,
    )
