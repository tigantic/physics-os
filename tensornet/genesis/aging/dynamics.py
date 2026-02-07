"""
Aging Dynamics — Time evolution operators for biological aging.

TENSOR GENESIS Protocol — Layer 27 (Aging)
Phase 21 of the Civilization Stack

The aging operator A acts on a cell state ψ to produce the aged state:
    ψ(t + Δt) = A(Δt) · ψ(t)

In TT format, A is a Matrix Product Operator (MPO). Each mode has its
own aging dynamics:
    - Epigenetic drift: stochastic methylation changes (fastest clock)
    - Proteostatic collapse: misfolding accumulation
    - Telomere attrition: progressive shortening
    - Metabolic dysregulation: nutrient sensing drift
    - Genomic instability: mutation accumulation

The key insight: aging is NOT a single operator. It's a sum of
mode-specific perturbations, each increasing rank independently:

    A = I + Σ_k ε_k(t) · Δ_k

where Δ_k is the perturbation operator for mode k and ε_k(t) is
the age-dependent amplitude. The total rank grows as:

    rank(ψ(t)) ≤ rank(ψ(0)) + Σ_k rank(Δ_k · ψ(0))

This is why aging is multi-factorial but rank growth is bounded:
each perturbation adds at most rank(Δ_k) to the state.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from tensornet.genesis.aging.cell_state import (
    AgingHallmark,
    BiologicalMode,
    CellStateTensor,
    CellType,
    ModeSpec,
    _tt_add,
    _tt_inner,
    _tt_max_rank,
    _tt_norm,
    _tt_ranks,
    _tt_round,
    _tt_scale,
    _tt_svd,
)


# ---------------------------------------------------------------------------
# Aging Rate Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgingRateModel:
    """
    Parameters governing the rate of aging for each biological mode.

    Calibrated against:
    - Horvath 2013 (epigenetic clock, 353 CpG sites)
    - Hannum 2013 (blood methylation clock, 71 CpG sites)
    - Lehallier 2019 (plasma proteomics aging waves)
    - López-Otín 2023 (12 hallmarks of aging)
    - Gladyshev 2021 (multi-omics aging clocks)
    """
    # Base rate of rank growth per year for each mode
    epigenetic_rate: float = 0.025      # ~2.5% rank growth/year (fastest)
    proteostatic_rate: float = 0.018    # ~1.8% rank growth/year
    genomic_rate: float = 0.012         # ~1.2% rank growth/year
    telomere_rate: float = 0.008        # ~0.8% rank growth/year (steadiest)
    metabolic_rate: float = 0.015       # ~1.5% rank growth/year
    signaling_rate: float = 0.014       # ~1.4% rank growth/year
    histone_rate: float = 0.020         # ~2.0% rank growth/year
    chromatin_rate: float = 0.017       # ~1.7% rank growth/year

    # Non-linear acceleration parameters
    # Aging accelerates after ~50 years (Lehallier proteomics waves)
    acceleration_onset: float = 50.0    # Years before acceleration kicks in
    acceleration_factor: float = 1.8    # Fold increase in rate after onset

    # Stochastic noise amplitude (variance of rank perturbation)
    noise_scale: float = 0.1

    def rate_for_mode(self, mode: BiologicalMode) -> float:
        """Get the base aging rate for a specific biological mode."""
        rate_map: Dict[BiologicalMode, float] = {
            BiologicalMode.GENE_EXPRESSION: self.genomic_rate,
            BiologicalMode.PROTEIN_ABUNDANCE: self.proteostatic_rate,
            BiologicalMode.METHYLATION: self.epigenetic_rate,
            BiologicalMode.HISTONE_CODE: self.histone_rate,
            BiologicalMode.CHROMATIN_ACCESS: self.chromatin_rate,
            BiologicalMode.METABOLOME: self.metabolic_rate,
            BiologicalMode.SIGNALING: self.signaling_rate,
            BiologicalMode.TELOMERE_LENGTH: self.telomere_rate,
        }
        return rate_map.get(mode, 0.01)

    def effective_rate(self, mode: BiologicalMode, age: float) -> float:
        """
        Compute the effective aging rate at a given age, including
        non-linear acceleration.
        """
        base = self.rate_for_mode(mode)
        if age > self.acceleration_onset:
            excess = age - self.acceleration_onset
            # Sigmoid ramp-up over 10 years
            ramp = 1.0 / (1.0 + math.exp(-0.5 * (excess - 5.0)))
            acceleration = 1.0 + (self.acceleration_factor - 1.0) * ramp
            return base * acceleration
        return base


# ---------------------------------------------------------------------------
# Perturbation Operators (Mode-Specific Aging)
# ---------------------------------------------------------------------------

@dataclass
class ModePerturbation:
    """
    A rank-1 or low-rank perturbation applied to a specific biological mode.

    This represents one aging mechanism acting on one part of the cell state.
    The perturbation is a small TT added to the state, increasing rank.
    """
    mode: BiologicalMode
    site_range: Tuple[int, int]  # (start, end) indices into QTT sites
    amplitude: float
    cores: List[np.ndarray]

    @property
    def rank(self) -> int:
        """Rank of this perturbation."""
        return _tt_max_rank(self.cores)


def _build_epigenetic_drift(
    n_sites: int,
    site_range: Tuple[int, int],
    amplitude: float,
    rng: np.random.Generator,
) -> ModePerturbation:
    """
    Build an epigenetic drift perturbation.

    Models stochastic CpG methylation changes: each site has a small
    probability of flipping state. In TT format, this is a rank-1
    perturbation (random product state scaled by amplitude).
    """
    start, end = site_range
    n_mode_sites = end - start
    if n_mode_sites <= 0:
        raise ValueError(f"Invalid site range: {site_range}")

    cores: List[np.ndarray] = []
    for k in range(n_sites):
        if k < start or k >= end:
            # Identity pass-through for sites outside this mode
            core = np.zeros((1, 2, 1))
            core[0, 0, 0] = 1.0
            core[0, 1, 0] = 1.0
        else:
            # Stochastic flip: small perturbation to methylation state
            core = np.zeros((1, 2, 1))
            flip_prob = rng.uniform(0.001, 0.05)
            core[0, 0, 0] = 1.0 - flip_prob
            core[0, 1, 0] = flip_prob
        cores.append(core)

    cores = _tt_scale(cores, amplitude)

    return ModePerturbation(
        mode=BiologicalMode.METHYLATION,
        site_range=site_range,
        amplitude=amplitude,
        cores=cores,
    )


def _build_proteostatic_perturbation(
    n_sites: int,
    site_range: Tuple[int, int],
    amplitude: float,
    rng: np.random.Generator,
) -> ModePerturbation:
    """
    Build a proteostatic collapse perturbation.

    Models protein misfolding accumulation. Misfolded proteins occupy
    new states in the abundance tensor, increasing rank. The perturbation
    is a rank-1 product state with biased weights in the protein mode
    sites: abundance shifts toward lower levels (misfolding depletes
    native protein) and higher levels (misfolded aggregates accumulate).

    The rank growth comes from _tt_add: adding this rank-1 perturbation
    to the cell state increases the TT rank by 1 at each bond.
    """
    start, end = site_range

    cores: List[np.ndarray] = []
    for k in range(n_sites):
        if k < start or k >= end:
            core = np.zeros((1, 2, 1))
            core[0, 0, 0] = 1.0
            core[0, 1, 0] = 1.0
        else:
            # Rank-1 proteostatic perturbation:
            # Bias toward misfolded states (shift distribution)
            core = np.zeros((1, 2, 1))
            decay_rate = rng.uniform(0.001, 0.01)
            core[0, 0, 0] = 1.0 - decay_rate   # Native protein depletes
            core[0, 1, 0] = 1.0 + decay_rate   # Misfolded accumulates
        cores.append(core)

    cores = _tt_scale(cores, amplitude)

    return ModePerturbation(
        mode=BiologicalMode.PROTEIN_ABUNDANCE,
        site_range=site_range,
        amplitude=amplitude,
        cores=cores,
    )


def _build_telomere_perturbation(
    n_sites: int,
    site_range: Tuple[int, int],
    amplitude: float,
    rng: np.random.Generator,
) -> ModePerturbation:
    """
    Build a telomere attrition perturbation.

    Telomeres shorten deterministically with each cell division (~50-200 bp/division).
    In TT format: the telomere mode shifts toward shorter lengths, introducing
    rank growth as the distribution broadens from sharp (young) to diffuse (old).
    """
    start, end = site_range
    cores: List[np.ndarray] = []
    for k in range(n_sites):
        if k < start or k >= end:
            core = np.zeros((1, 2, 1))
            core[0, 0, 0] = 1.0
            core[0, 1, 0] = 1.0
        else:
            # Shift toward lower indices (shorter telomeres)
            core = np.zeros((1, 2, 1))
            shortening_bias = rng.uniform(0.01, 0.05)
            core[0, 0, 0] = 1.0 + shortening_bias  # Favor shorter
            core[0, 1, 0] = 1.0 - shortening_bias  # Disfavor longer
        cores.append(core)

    cores = _tt_scale(cores, amplitude)

    return ModePerturbation(
        mode=BiologicalMode.TELOMERE_LENGTH,
        site_range=site_range,
        amplitude=amplitude,
        cores=cores,
    )


def _build_generic_perturbation(
    mode: BiologicalMode,
    n_sites: int,
    site_range: Tuple[int, int],
    amplitude: float,
    rng: np.random.Generator,
) -> ModePerturbation:
    """
    Build a generic noise perturbation for any biological mode.

    Models mode-nonspecific stochastic damage as a rank-1 random
    product state perturbation.
    """
    start, end = site_range
    cores: List[np.ndarray] = []
    for k in range(n_sites):
        if k < start or k >= end:
            core = np.zeros((1, 2, 1))
            core[0, 0, 0] = 1.0
            core[0, 1, 0] = 1.0
        else:
            core = np.zeros((1, 2, 1))
            core[0, :, 0] = rng.standard_normal(2) * 0.1
            core[0, :, 0] += 1.0  # Keep near identity
        cores.append(core)

    cores = _tt_scale(cores, amplitude)

    return ModePerturbation(
        mode=mode,
        site_range=site_range,
        amplitude=amplitude,
        cores=cores,
    )


# ---------------------------------------------------------------------------
# Aging Operator
# ---------------------------------------------------------------------------

class AgingOperator:
    """
    Time evolution operator for biological aging.

    Advances a cell state forward in time by applying mode-specific
    perturbations that increase TT rank. The operator is the sum of
    an identity (state persistence) and mode-specific damage terms.

    The aging process is:
        ψ(t + Δt) = normalize(ψ(t) + Σ_k ε_k(t,Δt) · Δ_k(ψ(t)))

    where:
        ε_k(t,Δt) = rate_k(t) · Δt · (1 + noise)
        Δ_k is the perturbation for mode k
        rate_k(t) includes non-linear acceleration

    After each time step, the state is compressed to prevent unbounded
    rank growth (biological systems have finite information capacity).

    Parameters
    ----------
    rate_model : AgingRateModel
        Parameters governing aging rates per mode.
    max_rank_cap : int
        Maximum rank allowed (biological information capacity limit).
    compression_tol : float
        TT rounding tolerance after each step.
    seed : int, optional
        Random seed for reproducibility.
    """

    __slots__ = ("rate_model", "max_rank_cap", "compression_tol", "_rng")

    def __init__(
        self,
        rate_model: Optional[AgingRateModel] = None,
        max_rank_cap: int = 256,
        compression_tol: float = 1e-10,
        seed: Optional[int] = None,
    ) -> None:
        self.rate_model = rate_model or AgingRateModel()
        self.max_rank_cap = max_rank_cap
        self.compression_tol = compression_tol
        self._rng = np.random.default_rng(seed)

    def advance(
        self,
        state: CellStateTensor,
        dt_years: float,
    ) -> CellStateTensor:
        """
        Advance a cell state by dt_years of aging.

        Applies mode-specific perturbations proportional to dt and the
        mode's aging rate, then compresses to maintain bounded rank.

        Parameters
        ----------
        state : CellStateTensor
            Current cell state.
        dt_years : float
            Time step in years.

        Returns
        -------
        CellStateTensor
            Aged cell state.
        """
        if dt_years <= 0:
            return state

        current_age = state.chronological_age
        n_sites = state.n_sites

        # Build site ranges for each mode
        site_ranges = self._compute_site_ranges(state.mode_specs)

        # Accumulate perturbations
        perturbed_cores = [c.copy() for c in state.cores]

        for spec in state.mode_specs:
            if spec.mode_type not in site_ranges:
                continue

            site_range = site_ranges[spec.mode_type]
            rate = self.rate_model.effective_rate(spec.mode_type, current_age)
            amplitude = rate * dt_years

            # Add stochastic noise
            noise = 1.0 + self._rng.standard_normal() * self.rate_model.noise_scale
            amplitude *= max(noise, 0.0)

            if amplitude < 1e-15:
                continue

            # Build mode-specific perturbation
            perturbation = self._build_perturbation(
                spec.mode_type, n_sites, site_range, amplitude
            )

            # Add perturbation to state: ψ' = ψ + δψ
            perturbed_cores = _tt_add(perturbed_cores, perturbation.cores)

        # Compress to prevent unbounded growth
        perturbed_cores = _tt_round(
            perturbed_cores,
            max_rank=self.max_rank_cap,
            tol=self.compression_tol,
        )

        # Normalize
        norm_val = _tt_norm(perturbed_cores)
        if norm_val > 1e-300:
            perturbed_cores = _tt_scale(perturbed_cores, 1.0 / norm_val)

        new_age = current_age + dt_years
        result = CellStateTensor(
            cores=perturbed_cores,
            mode_specs=state.mode_specs,
            cell_type=state.cell_type,
            chronological_age=new_age,
            metadata={
                **state.metadata,
                "aged_by": dt_years,
                "aging_rate_model": type(self.rate_model).__name__,
            },
        )
        result._rank_history = list(state._rank_history) + [
            (new_age, result.max_rank)
        ]
        return result

    def evolve(
        self,
        state: CellStateTensor,
        target_age: float,
        dt_years: float = 1.0,
        callback: Optional[Callable[[CellStateTensor, int], None]] = None,
    ) -> Tuple[CellStateTensor, List[CellStateTensor]]:
        """
        Evolve a cell state from its current age to target_age.

        Parameters
        ----------
        state : CellStateTensor
            Initial cell state.
        target_age : float
            Target chronological age in years.
        dt_years : float
            Time step per iteration.
        callback : callable, optional
            Called after each step with (state, step_number).

        Returns
        -------
        final_state : CellStateTensor
            Cell state at target_age.
        trajectory : list of CellStateTensor
            All intermediate states.
        """
        if target_age <= state.chronological_age:
            return state, [state]

        trajectory: List[CellStateTensor] = [state]
        current = state
        step = 0

        while current.chronological_age < target_age - 1e-10:
            dt = min(dt_years, target_age - current.chronological_age)
            current = self.advance(current, dt)
            trajectory.append(current)
            step += 1

            if callback is not None:
                callback(current, step)

        return current, trajectory

    def _build_perturbation(
        self,
        mode: BiologicalMode,
        n_sites: int,
        site_range: Tuple[int, int],
        amplitude: float,
    ) -> ModePerturbation:
        """Build the appropriate perturbation for a biological mode."""
        if mode == BiologicalMode.METHYLATION:
            return _build_epigenetic_drift(
                n_sites, site_range, amplitude, self._rng
            )
        elif mode == BiologicalMode.PROTEIN_ABUNDANCE:
            return _build_proteostatic_perturbation(
                n_sites, site_range, amplitude, self._rng
            )
        elif mode == BiologicalMode.TELOMERE_LENGTH:
            return _build_telomere_perturbation(
                n_sites, site_range, amplitude, self._rng
            )
        else:
            return _build_generic_perturbation(
                mode, n_sites, site_range, amplitude, self._rng
            )

    def _compute_site_ranges(
        self, mode_specs: List[ModeSpec]
    ) -> Dict[BiologicalMode, Tuple[int, int]]:
        """Compute the QTT site range for each biological mode."""
        ranges: Dict[BiologicalMode, Tuple[int, int]] = {}
        site_idx = 0
        for spec in mode_specs:
            n_sites = spec.n_qtt_sites
            ranges[spec.mode_type] = (site_idx, site_idx + n_sites)
            site_idx += n_sites
        return ranges


# ---------------------------------------------------------------------------
# Aging Trajectory Analysis
# ---------------------------------------------------------------------------

@dataclass
class AgingTrajectory:
    """
    Complete aging trajectory: a sequence of cell states over time.

    Provides analysis of rank dynamics, mode-specific aging rates,
    and identification of critical transitions.
    """
    states: List[CellStateTensor]
    ages: List[float] = field(init=False)
    ranks: List[int] = field(init=False)
    entropies: List[float] = field(init=False)

    def __post_init__(self) -> None:
        self.ages = [s.chronological_age for s in self.states]
        self.ranks = [s.max_rank for s in self.states]
        self.entropies = [s.rank_entropy for s in self.states]

    @property
    def n_timepoints(self) -> int:
        """Number of timepoints in the trajectory."""
        return len(self.states)

    @property
    def age_span(self) -> float:
        """Total age span covered."""
        if len(self.ages) < 2:
            return 0.0
        return self.ages[-1] - self.ages[0]

    @property
    def rank_growth_rate(self) -> float:
        """
        Average rank growth rate (ranks per year).
        """
        if len(self.ages) < 2 or self.age_span < 1e-10:
            return 0.0
        return (self.ranks[-1] - self.ranks[0]) / self.age_span

    def rank_at_age(self, age: float) -> float:
        """
        Interpolate rank at a given age.
        """
        if age <= self.ages[0]:
            return float(self.ranks[0])
        if age >= self.ages[-1]:
            return float(self.ranks[-1])

        # Linear interpolation
        for i in range(len(self.ages) - 1):
            if self.ages[i] <= age <= self.ages[i + 1]:
                t = (age - self.ages[i]) / (self.ages[i + 1] - self.ages[i])
                return self.ranks[i] + t * (self.ranks[i + 1] - self.ranks[i])
        return float(self.ranks[-1])

    def mode_rank_trajectory(
        self, mode: BiologicalMode
    ) -> List[float]:
        """
        Extract the rank trajectory for a specific biological mode.
        """
        mode_ranks: List[float] = []
        for state in self.states:
            mode_rank_dict = state._compute_mode_ranks()
            mode_ranks.append(mode_rank_dict.get(mode, 1.0))
        return mode_ranks

    def detect_transitions(
        self, threshold: float = 2.0
    ) -> List[Tuple[float, float, str]]:
        """
        Detect critical aging transitions where rank growth accelerates.

        Returns list of (age, rank_change, description) tuples.

        A transition occurs when the rank growth rate exceeds `threshold`
        times the running average, indicating a phase change in aging
        dynamics (e.g., onset of senescence, proteostatic collapse).
        """
        if len(self.ages) < 3:
            return []

        transitions: List[Tuple[float, float, str]] = []
        running_avg = 0.0
        n_avg = 0

        for i in range(1, len(self.ages)):
            dt = self.ages[i] - self.ages[i - 1]
            if dt < 1e-10:
                continue
            dr = float(self.ranks[i] - self.ranks[i - 1])
            rate = dr / dt

            if n_avg > 2 and running_avg > 0:
                if rate > threshold * running_avg:
                    transitions.append((
                        self.ages[i],
                        rate,
                        f"Rank acceleration at age {self.ages[i]:.1f}: "
                        f"rate={rate:.2f}/yr (avg={running_avg:.2f}/yr)"
                    ))

            # Update running average
            running_avg = (running_avg * n_avg + rate) / (n_avg + 1)
            n_avg += 1

        return transitions

    def distance_matrix(self) -> np.ndarray:
        """
        Compute pairwise distances between all states in the trajectory.

        Returns an (n × n) matrix where D[i,j] = ||ψ_i - ψ_j||.
        """
        n = len(self.states)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.states[i].distance(self.states[j])
                D[i, j] = d
                D[j, i] = d
        return D
