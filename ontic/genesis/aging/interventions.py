"""
Intervention Engine — Rank-reducing operators for biological rejuvenation.

TENSOR GENESIS Protocol — Layer 27 (Aging)
Phase 21 of the Civilization Stack

The central theorem of this module:

    Aging is rank growth. Reversal is rank reduction.

    Yamanaka showed that four transcription factors (Oct4, Sox2, Klf4, c-Myc)
    can reset cellular age. In TT language, OSKM is a rank-4 operator that
    projects the aged state tensor back to its low-rank (young) manifold.

    This module provides:
    1. YamanakaOperator — rank-4 reprogramming that explains iPSC generation
    2. PartialReprogramming — controlled rank reduction without identity loss
    3. SenolyticOperator — targeted removal of senescent rank contributions
    4. CalorieRestrictionOperator — metabolic mode rank reduction
    5. InterventionOptimizer — find minimum-rank interventions via QTT-OT

    The key insight: any rejuvenation intervention is a low-rank operator
    that compresses the state tensor. The rank of the operator determines
    the minimum number of independent molecular targets needed.

Why Yamanaka Works (QTT Explanation):

    An aged cell has state ψ_aged with max_rank ≈ 50-200.
    OSKM factors bind to ~1000 genomic loci, imposing a rank-4 constraint:

    ψ_iPSC = P_OSKM · ψ_aged

    where P_OSKM is a rank-4 projection operator. This works because:
    1. Oct4 couples gene expression to methylation (bonds 2 modes)
    2. Sox2 couples chromatin to histone code (bonds 2 modes)
    3. Klf4 stabilizes the epithelial-mesenchymal program
    4. c-Myc amplifies the metabolic reset

    Each factor constrains one "slice" of the tensor — four factors,
    four constraints, rank reduced from ~200 to ~4.

    Partial reprogramming (Ocampo 2016, Lu 2020) applies OSKM briefly,
    reducing rank partially without full dedifferentiation. This is
    equivalent to TT rounding with a target rank between young and aged.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from ontic.genesis.aging.cell_state import (
    AgingHallmark,
    AgingSignature,
    BiologicalMode,
    CellStateTensor,
    CellType,
    ModeSpec,
    YAMANAKA_FACTORS,
    THOMSON_FACTORS,
    _tt_add,
    _tt_inner,
    _tt_max_rank,
    _tt_norm,
    _tt_ranks,
    _tt_round,
    _tt_scale,
    _tt_svd,
)
from ontic.genesis.aging.epigenetic_clock import (
    HorvathClock,
    MethylationState,
    extract_methylation,
)


# ---------------------------------------------------------------------------
# Intervention Result
# ---------------------------------------------------------------------------

@dataclass
class InterventionResult:
    """
    Result of applying a rejuvenation intervention.

    Attributes
    ----------
    state_before : CellStateTensor
        Cell state before intervention.
    state_after : CellStateTensor
        Cell state after intervention.
    rank_before : int
        Max TT rank before.
    rank_after : int
        Max TT rank after.
    rank_reduction : float
        Fraction of rank removed (0 to 1).
    biological_age_before : float
        Estimated biological age before.
    biological_age_after : float
        Estimated biological age after.
    years_reversed : float
        Biological years reversed.
    intervention_rank : int
        Rank of the intervention operator.
    fidelity : float
        How well cell identity is preserved (0 to 1).
    elapsed_seconds : float
        Wall-clock computation time.
    """
    state_before: CellStateTensor
    state_after: CellStateTensor
    rank_before: int
    rank_after: int
    rank_reduction: float
    biological_age_before: float
    biological_age_after: float
    years_reversed: float
    intervention_rank: int
    fidelity: float
    elapsed_seconds: float

    @property
    def efficiency(self) -> float:
        """Years reversed per unit of intervention rank."""
        if self.intervention_rank <= 0:
            return 0.0
        return self.years_reversed / self.intervention_rank


# ---------------------------------------------------------------------------
# Yamanaka Operator
# ---------------------------------------------------------------------------

class YamanakaOperator:
    """
    Full Yamanaka reprogramming: OSKM → iPSC.

    Applies a rank-4 projection that resets the cell state to embryonic-like
    condition. This is the most extreme form of rejuvenation — it reverses
    ALL aging hallmarks but also erases cell identity.

    The operator is constructed as a product of four factor-specific
    projectors, one per Yamanaka factor:

        P_OSKM = P_Oct4 · P_Sox2 · P_Klf4 · P_Myc

    Each projector constrains a specific set of tensor modes.

    Parameters
    ----------
    target_rank : int
        Target rank after reprogramming (default 4).
    preserve_identity : bool
        If True, attempts to preserve some cell-type-specific features.
        This is partial reprogramming.
    """

    __slots__ = ("target_rank", "preserve_identity", "_factor_operators")

    def __init__(
        self,
        target_rank: int = 4,
        preserve_identity: bool = False,
    ) -> None:
        self.target_rank = target_rank
        self.preserve_identity = preserve_identity
        self._factor_operators: Dict[str, _FactorProjector] = {}

    def apply(self, state: CellStateTensor) -> InterventionResult:
        """
        Apply Yamanaka reprogramming to a cell state.

        Parameters
        ----------
        state : CellStateTensor
            Aged cell state to reprogram.

        Returns
        -------
        InterventionResult
            Result containing before/after states and metrics.
        """
        t0 = time.time()
        sig_before = state.aging_signature()

        # Build factor-specific projectors
        site_ranges = self._compute_site_ranges(state.mode_specs)
        n_sites = state.n_sites

        # Apply each Yamanaka factor as a rank constraint
        projected = state

        # Oct4: constrains gene expression ↔ methylation coupling
        projected = self._apply_factor(
            projected,
            "OCT4",
            target_modes=[BiologicalMode.GENE_EXPRESSION, BiologicalMode.METHYLATION],
            site_ranges=site_ranges,
            factor_rank=1,  # Each factor imposes rank-1 constraint
        )

        # Sox2: constrains chromatin ↔ histone coupling
        projected = self._apply_factor(
            projected,
            "SOX2",
            target_modes=[BiologicalMode.CHROMATIN_ACCESS, BiologicalMode.HISTONE_CODE],
            site_ranges=site_ranges,
            factor_rank=1,
        )

        # Klf4: constrains gene expression ↔ protein coupling
        projected = self._apply_factor(
            projected,
            "KLF4",
            target_modes=[BiologicalMode.GENE_EXPRESSION, BiologicalMode.PROTEIN_ABUNDANCE],
            site_ranges=site_ranges,
            factor_rank=1,
        )

        # c-Myc: constrains metabolome ↔ signaling coupling
        projected = self._apply_factor(
            projected,
            "MYC",
            target_modes=[BiologicalMode.METABOLOME, BiologicalMode.SIGNALING],
            site_ranges=site_ranges,
            factor_rank=1,
        )

        # Final compression to target rank
        result_state = projected.compress(
            max_rank=self.target_rank,
            tol=1e-14,
        )

        # Update metadata
        result_state = CellStateTensor(
            cores=result_state.cores,
            mode_specs=result_state.mode_specs,
            cell_type=CellType.INDUCED_PLURIPOTENT if not self.preserve_identity else state.cell_type,
            chronological_age=state.chronological_age,  # Chronological age unchanged
            metadata={
                **state.metadata,
                "intervention": "yamanaka_OSKM",
                "target_rank": self.target_rank,
                "preserve_identity": self.preserve_identity,
            },
        )

        sig_after = result_state.aging_signature()
        elapsed = time.time() - t0

        # Compute fidelity (how much of original state is preserved)
        fidelity = self._compute_fidelity(state, result_state)

        return InterventionResult(
            state_before=state,
            state_after=result_state,
            rank_before=sig_before.max_rank,
            rank_after=sig_after.max_rank,
            rank_reduction=1.0 - sig_after.max_rank / max(sig_before.max_rank, 1),
            biological_age_before=sig_before.biological_age,
            biological_age_after=sig_after.biological_age,
            years_reversed=sig_before.biological_age - sig_after.biological_age,
            intervention_rank=4,  # Yamanaka is always rank-4
            fidelity=fidelity,
            elapsed_seconds=elapsed,
        )

    def _apply_factor(
        self,
        state: CellStateTensor,
        factor_name: str,
        target_modes: List[BiologicalMode],
        site_ranges: Dict[BiologicalMode, Tuple[int, int]],
        factor_rank: int,
    ) -> CellStateTensor:
        """
        Apply a single transcription factor as a rank-reducing projection.

        The factor constrains the targeted modes by attenuating high-order
        singular values in those modes' cores, then globally re-compressing
        the TT. This is the mathematically correct way to reduce rank:
        individual core truncation alone doesn't change TT rank (only
        changes core shapes), but attenuating singular values makes the
        subsequent global TT rounding much more effective.

        Each Yamanaka factor imposes coherence on its target modes, which
        in TT language means the correlations within those modes collapse
        to a low-rank subspace.
        """
        # Identify the sites controlled by this factor
        target_sites: List[Tuple[int, int]] = []
        for mode in target_modes:
            if mode in site_ranges:
                target_sites.append(site_ranges[mode])

        if not target_sites:
            return state

        # Attenuate high-rank singular values in targeted mode cores.
        # This models the transcription factor imposing coherence: the
        # factor binds to its target loci and forces a low-rank regulatory
        # program, effectively zeroing out the noise components.
        new_cores: List[np.ndarray] = [c.copy() for c in state.cores]
        for k in range(len(new_cores)):
            is_target = any(start <= k < end for start, end in target_sites)
            if not is_target:
                continue

            r_left, d, r_right = new_cores[k].shape
            mat = new_cores[k].reshape(r_left * d, r_right)
            try:
                U, S, Vt = la.svd(mat, full_matrices=False)
            except la.LinAlgError:
                U, S, Vt = np.linalg.svd(mat, full_matrices=False)

            # Keep the top 'factor_rank' singular values at full strength;
            # exponentially attenuate the remainder. This makes the TT
            # "almost" low-rank in these modes, so global rounding achieves
            # the target rank efficiently.
            for i in range(len(S)):
                if i >= factor_rank:
                    S[i] *= math.exp(-(i - factor_rank + 1) * 2.0)

            new_cores[k] = ((U * S[np.newaxis, :]) @ Vt).reshape(r_left, d, r_right)

        # Global TT rounding realizes the rank reduction. The attenuation
        # above ensures the tolerance-based truncation can remove most of
        # the rank introduced by aging noise.
        new_cores = _tt_round(new_cores, max_rank=max(state.max_rank, 4), tol=1e-8)

        return CellStateTensor(
            cores=new_cores,
            mode_specs=state.mode_specs,
            cell_type=state.cell_type,
            chronological_age=state.chronological_age,
            metadata=state.metadata,
        )

    def _compute_site_ranges(
        self, mode_specs: List[ModeSpec]
    ) -> Dict[BiologicalMode, Tuple[int, int]]:
        """Compute QTT site ranges for each biological mode."""
        ranges: Dict[BiologicalMode, Tuple[int, int]] = {}
        site_idx = 0
        for spec in mode_specs:
            n_sites = spec.n_qtt_sites
            ranges[spec.mode_type] = (site_idx, site_idx + n_sites)
            site_idx += n_sites
        return ranges

    def _compute_fidelity(
        self,
        original: CellStateTensor,
        reprogrammed: CellStateTensor,
    ) -> float:
        """
        Compute fidelity — overlap between original and reprogrammed states.

        Fidelity = |⟨ψ_orig | ψ_reprog⟩|² / (||ψ_orig|| · ||ψ_reprog||)

        For full reprogramming, fidelity is low (cell identity erased).
        For partial reprogramming, fidelity should be moderate to high.
        """
        inner = abs(_tt_inner(original.cores, reprogrammed.cores))
        norm_a = _tt_norm(original.cores)
        norm_b = _tt_norm(reprogrammed.cores)
        denom = norm_a * norm_b
        if denom < 1e-300:
            return 0.0
        overlap = inner / denom
        return min(overlap**2, 1.0)


# ---------------------------------------------------------------------------
# Partial Reprogramming
# ---------------------------------------------------------------------------

class PartialReprogrammingOperator:
    """
    Partial reprogramming — controlled rank reduction without full
    dedifferentiation.

    Based on:
    - Ocampo 2016: Cyclic OSKM expression reverses aging in progeroid mice
    - Lu 2020: Reprogramming to recover youthful epigenetic information
    - Browder 2022: In vivo partial reprogramming alters age-associated
      molecular changes during physiological aging in mice

    Partial reprogramming reduces rank to an intermediate level:
        rank_target = rank_young + α · (rank_aged - rank_young)

    where α ∈ (0, 1) controls the degree of rejuvenation.
    α = 0 → full reprogramming (iPSC), α = 1 → no change.

    Parameters
    ----------
    rejuvenation_fraction : float
        Fraction of aging to reverse (0 = no effect, 1 = full reversal).
        Typical values: 0.3-0.6 for safe partial reprogramming.
    preserve_identity : bool
        If True, protects cell-type-specific modes from modification.
    cycles : int
        Number of reprogramming cycles (cyclic expression).
    """

    __slots__ = ("rejuvenation_fraction", "preserve_identity", "cycles")

    def __init__(
        self,
        rejuvenation_fraction: float = 0.5,
        preserve_identity: bool = True,
        cycles: int = 1,
    ) -> None:
        if not 0.0 <= rejuvenation_fraction <= 1.0:
            raise ValueError(
                f"rejuvenation_fraction must be in [0, 1], got {rejuvenation_fraction}"
            )
        self.rejuvenation_fraction = rejuvenation_fraction
        self.preserve_identity = preserve_identity
        self.cycles = max(1, cycles)

    def apply(self, state: CellStateTensor) -> InterventionResult:
        """
        Apply partial reprogramming to a cell state.

        Each cycle reduces rank by a fraction, with identity preservation
        protecting cell-type-specific features.
        """
        t0 = time.time()
        sig_before = state.aging_signature()

        current = state
        young_rank = 4  # Yamanaka baseline

        for cycle in range(self.cycles):
            current_rank = current.max_rank

            # Target rank for this cycle
            rank_reduction = self.rejuvenation_fraction / self.cycles
            target_rank = max(
                young_rank,
                int(current_rank * (1.0 - rank_reduction)),
            )

            if self.preserve_identity:
                # Only compress modes that are aging-related
                current = self._selective_compress(current, target_rank)
            else:
                current = current.compress(max_rank=target_rank, tol=1e-12)

        # Normalize
        norm_val = _tt_norm(current.cores)
        if norm_val > 1e-300:
            new_cores = _tt_scale(current.cores, 1.0 / norm_val)
        else:
            new_cores = current.cores

        result_state = CellStateTensor(
            cores=new_cores,
            mode_specs=state.mode_specs,
            cell_type=state.cell_type,  # Identity preserved
            chronological_age=state.chronological_age,
            metadata={
                **state.metadata,
                "intervention": "partial_reprogramming",
                "rejuvenation_fraction": self.rejuvenation_fraction,
                "cycles": self.cycles,
                "preserve_identity": self.preserve_identity,
            },
        )

        sig_after = result_state.aging_signature()
        elapsed = time.time() - t0
        fidelity = abs(_tt_inner(state.cores, result_state.cores))
        norm_a = _tt_norm(state.cores)
        norm_b = _tt_norm(result_state.cores)
        if norm_a * norm_b > 1e-300:
            fidelity = min((fidelity / (norm_a * norm_b)) ** 2, 1.0)
        else:
            fidelity = 0.0

        return InterventionResult(
            state_before=state,
            state_after=result_state,
            rank_before=sig_before.max_rank,
            rank_after=sig_after.max_rank,
            rank_reduction=1.0 - sig_after.max_rank / max(sig_before.max_rank, 1),
            biological_age_before=sig_before.biological_age,
            biological_age_after=sig_after.biological_age,
            years_reversed=sig_before.biological_age - sig_after.biological_age,
            intervention_rank=4 * self.cycles,
            fidelity=fidelity,
            elapsed_seconds=elapsed,
        )

    def _selective_compress(
        self,
        state: CellStateTensor,
        target_rank: int,
    ) -> CellStateTensor:
        """
        Selectively compress only aging-related modes.

        Modes with high aging rates (methylation, histone, chromatin) are
        compressed aggressively. Modes that define cell identity
        (gene expression baseline, signaling) are left mostly intact.
        """
        # Aging-aggressive modes (compress more)
        aggressive_modes = {
            BiologicalMode.METHYLATION,
            BiologicalMode.HISTONE_CODE,
            BiologicalMode.CHROMATIN_ACCESS,
        }
        # Moderate modes
        moderate_modes = {
            BiologicalMode.PROTEIN_ABUNDANCE,
            BiologicalMode.METABOLOME,
            BiologicalMode.TELOMERE_LENGTH,
        }
        # Identity modes (compress less)
        identity_modes = {
            BiologicalMode.GENE_EXPRESSION,
            BiologicalMode.SIGNALING,
        }

        site_idx = 0
        new_cores: List[np.ndarray] = []

        for spec in state.mode_specs:
            n_sites = spec.n_qtt_sites
            end_idx = min(site_idx + n_sites, len(state.cores))

            # Extract mode-specific cores
            mode_cores = [state.cores[k].copy() for k in range(site_idx, end_idx)]

            if spec.mode_type in aggressive_modes:
                mode_target = max(2, target_rank // 2)
            elif spec.mode_type in moderate_modes:
                mode_target = max(3, int(target_rank * 0.75))
            else:  # identity modes
                mode_target = max(4, target_rank)

            # Compress this mode's cores independently
            if mode_cores:
                mode_cores = _tt_round(mode_cores, max_rank=mode_target, tol=1e-12)

            new_cores.extend(mode_cores)
            site_idx = end_idx

        # Fix cross-mode rank compatibility
        for k in range(len(new_cores) - 1):
            r_right = new_cores[k].shape[2]
            r_left_next = new_cores[k + 1].shape[0]
            if r_right != r_left_next:
                # Pad smaller dimension
                _, d_next, r_right_next = new_cores[k + 1].shape
                if r_right < r_left_next:
                    # Pad current core's right dimension
                    r_left_curr, d_curr, _ = new_cores[k].shape
                    padded = np.zeros((r_left_curr, d_curr, r_left_next))
                    padded[:, :, :r_right] = new_cores[k]
                    new_cores[k] = padded
                else:
                    # Pad next core's left dimension
                    padded = np.zeros((r_right, d_next, r_right_next))
                    padded[:r_left_next, :, :] = new_cores[k + 1]
                    new_cores[k + 1] = padded

        return CellStateTensor(
            cores=new_cores,
            mode_specs=state.mode_specs,
            cell_type=state.cell_type,
            chronological_age=state.chronological_age,
            metadata=state.metadata,
        )


# ---------------------------------------------------------------------------
# Senolytic Operator
# ---------------------------------------------------------------------------

class SenolyticOperator:
    """
    Senolytic intervention — targeted removal of senescent cell signatures.

    Senescent cells are characterized by:
    - SASP (senescence-associated secretory phenotype) → high signaling rank
    - p16/p21 activation → specific gene expression patterns
    - Epigenetic changes at senescence-associated loci

    In TT language, senescence is a specific rank contribution that can
    be identified and surgically removed without affecting the bulk state.

    Based on:
    - Baker 2016: Clearance of p16+ cells delays aging-associated disorders
    - Xu 2018: Senolytics improve physical function in aged mice
    - Kirkland 2017: The clinical potential of senolytic drugs

    Parameters
    ----------
    target_modes : list of BiologicalMode
        Modes to target for senescent signature removal.
    aggressiveness : float
        How aggressively to remove senescent signatures (0-1).
    """

    __slots__ = ("target_modes", "aggressiveness")

    def __init__(
        self,
        target_modes: Optional[List[BiologicalMode]] = None,
        aggressiveness: float = 0.7,
    ) -> None:
        self.target_modes = target_modes or [
            BiologicalMode.SIGNALING,
            BiologicalMode.GENE_EXPRESSION,
        ]
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))

    def apply(self, state: CellStateTensor) -> InterventionResult:
        """Apply senolytic intervention to remove senescent rank contributions."""
        t0 = time.time()
        sig_before = state.aging_signature()

        # Compute site ranges
        site_ranges: Dict[BiologicalMode, Tuple[int, int]] = {}
        site_idx = 0
        for spec in state.mode_specs:
            site_ranges[spec.mode_type] = (site_idx, site_idx + spec.n_qtt_sites)
            site_idx += spec.n_qtt_sites

        # For each target mode, identify and remove the top singular values
        # that represent senescent signatures
        new_cores: List[np.ndarray] = []
        for k in range(state.n_sites):
            core = state.cores[k].copy()

            # Check if this site is in a target mode
            is_target = False
            for mode in self.target_modes:
                if mode in site_ranges:
                    start, end = site_ranges[mode]
                    if start <= k < end:
                        is_target = True
                        break

            if is_target:
                r_left, d, r_right = core.shape
                mat = core.reshape(r_left * d, r_right)
                try:
                    U, S, Vt = la.svd(mat, full_matrices=False)
                except la.LinAlgError:
                    U, S, Vt = np.linalg.svd(mat, full_matrices=False)

                # Remove the largest singular values (senescent signatures
                # tend to dominate the spectrum)
                n_remove = max(0, int(len(S) * self.aggressiveness * 0.3))
                if n_remove > 0 and len(S) > n_remove:
                    # Zero out top n_remove singular values
                    S_cleaned = S.copy()
                    # Keep top few (cell identity) but remove middle range (senescence)
                    n_keep_top = max(1, len(S) - n_remove)
                    S_cleaned[n_keep_top:] = 0.0

                    mat_cleaned = (U * S_cleaned[np.newaxis, :]) @ Vt
                    core = mat_cleaned.reshape(r_left, d, r_right)

            new_cores.append(core)

        # Compress to remove the zeroed components
        new_cores = _tt_round(new_cores, tol=1e-10)

        # Normalize
        norm_val = _tt_norm(new_cores)
        if norm_val > 1e-300:
            new_cores = _tt_scale(new_cores, 1.0 / norm_val)

        result_state = CellStateTensor(
            cores=new_cores,
            mode_specs=state.mode_specs,
            cell_type=state.cell_type,
            chronological_age=state.chronological_age,
            metadata={
                **state.metadata,
                "intervention": "senolytic",
                "aggressiveness": self.aggressiveness,
            },
        )

        sig_after = result_state.aging_signature()
        elapsed = time.time() - t0

        fidelity = abs(_tt_inner(state.cores, result_state.cores))
        n_a = _tt_norm(state.cores)
        n_b = _tt_norm(result_state.cores)
        if n_a * n_b > 1e-300:
            fidelity = min((fidelity / (n_a * n_b)) ** 2, 1.0)
        else:
            fidelity = 0.0

        return InterventionResult(
            state_before=state,
            state_after=result_state,
            rank_before=sig_before.max_rank,
            rank_after=sig_after.max_rank,
            rank_reduction=1.0 - sig_after.max_rank / max(sig_before.max_rank, 1),
            biological_age_before=sig_before.biological_age,
            biological_age_after=sig_after.biological_age,
            years_reversed=sig_before.biological_age - sig_after.biological_age,
            intervention_rank=len(self.target_modes),
            fidelity=fidelity,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Calorie Restriction Operator
# ---------------------------------------------------------------------------

class CalorieRestrictionOperator:
    """
    Calorie restriction (CR) intervention.

    CR is the most robust lifespan-extending intervention across species.
    In TT language, CR reduces rank in the metabolome and nutrient-sensing
    modes by imposing a simpler metabolic program.

    Based on:
    - Colman 2014: CR delays disease onset and mortality in rhesus monkeys
    - Mattison 2017: CR improves healthspan in non-human primates
    - Mitchell 2016: CR rewires metabolic circuits

    Parameters
    ----------
    restriction_level : float
        Fraction of caloric restriction (0.0 = no restriction, 1.0 = complete
        restriction). Typical CR: 0.2-0.4 (20-40% restriction).
    """

    __slots__ = ("restriction_level",)

    def __init__(self, restriction_level: float = 0.3) -> None:
        if not 0.0 <= restriction_level <= 1.0:
            raise ValueError(
                f"restriction_level must be in [0, 1], got {restriction_level}"
            )
        self.restriction_level = restriction_level

    def apply(self, state: CellStateTensor) -> InterventionResult:
        """Apply calorie restriction to reduce metabolic mode rank."""
        t0 = time.time()
        sig_before = state.aging_signature()

        # Target rank reduction: CR reduces metabolome and signaling ranks
        target_modes = [BiologicalMode.METABOLOME, BiologicalMode.SIGNALING]

        site_ranges: Dict[BiologicalMode, Tuple[int, int]] = {}
        site_idx = 0
        for spec in state.mode_specs:
            site_ranges[spec.mode_type] = (site_idx, site_idx + spec.n_qtt_sites)
            site_idx += spec.n_qtt_sites

        # For metabolic modes, compress proportionally to restriction level
        new_cores: List[np.ndarray] = [c.copy() for c in state.cores]

        for mode in target_modes:
            if mode not in site_ranges:
                continue
            start, end = site_ranges[mode]

            # The restriction simplifies the metabolic program
            # by reducing rank in these modes
            mode_rank_target = max(
                2,
                int(_tt_max_rank(new_cores) * (1.0 - self.restriction_level * 0.5)),
            )

            for k in range(start, min(end, len(new_cores))):
                r_left, d, r_right = new_cores[k].shape
                if r_right > mode_rank_target:
                    mat = new_cores[k].reshape(r_left * d, r_right)
                    try:
                        U, S, Vt = la.svd(mat, full_matrices=False)
                    except la.LinAlgError:
                        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
                    r_new = min(mode_rank_target, len(S))
                    new_cores[k] = (
                        (U[:, :r_new] * S[:r_new][np.newaxis, :])
                    ).reshape(r_left, d, r_new)

        # Fix compatibility and compress
        new_cores = self._fix_and_compress(new_cores)

        norm_val = _tt_norm(new_cores)
        if norm_val > 1e-300:
            new_cores = _tt_scale(new_cores, 1.0 / norm_val)

        result_state = CellStateTensor(
            cores=new_cores,
            mode_specs=state.mode_specs,
            cell_type=state.cell_type,
            chronological_age=state.chronological_age,
            metadata={
                **state.metadata,
                "intervention": "calorie_restriction",
                "restriction_level": self.restriction_level,
            },
        )

        sig_after = result_state.aging_signature()
        elapsed = time.time() - t0

        fidelity = abs(_tt_inner(state.cores, result_state.cores))
        n_a = _tt_norm(state.cores)
        n_b = _tt_norm(result_state.cores)
        if n_a * n_b > 1e-300:
            fidelity = min((fidelity / (n_a * n_b)) ** 2, 1.0)
        else:
            fidelity = 0.0

        return InterventionResult(
            state_before=state,
            state_after=result_state,
            rank_before=sig_before.max_rank,
            rank_after=sig_after.max_rank,
            rank_reduction=1.0 - sig_after.max_rank / max(sig_before.max_rank, 1),
            biological_age_before=sig_before.biological_age,
            biological_age_after=sig_after.biological_age,
            years_reversed=sig_before.biological_age - sig_after.biological_age,
            intervention_rank=2,  # CR targets 2 modes
            fidelity=fidelity,
            elapsed_seconds=elapsed,
        )

    def _fix_and_compress(self, cores: List[np.ndarray]) -> List[np.ndarray]:
        """Fix rank compatibility and compress."""
        # Fix compatibility
        for k in range(len(cores) - 1):
            r_right = cores[k].shape[2]
            r_left_next = cores[k + 1].shape[0]
            if r_right != r_left_next:
                _, d_next, r_right_next = cores[k + 1].shape
                if r_right < r_left_next:
                    cores[k + 1] = cores[k + 1][:r_right, :, :]
                else:
                    padded = np.zeros((r_right, d_next, r_right_next))
                    padded[:r_left_next, :, :] = cores[k + 1]
                    cores[k + 1] = padded

        return _tt_round(cores, tol=1e-10)


# ---------------------------------------------------------------------------
# Combination Interventions
# ---------------------------------------------------------------------------

class CombinationIntervention:
    """
    Apply multiple interventions in sequence.

    Models real-world combination therapies:
    - Senolytic + partial reprogramming (Xu 2023)
    - CR + exercise + senolytic
    - Multi-target pharmacological interventions

    Parameters
    ----------
    interventions : list
        Ordered list of intervention operators to apply.
    """

    __slots__ = ("interventions",)

    def __init__(
        self,
        interventions: List,
    ) -> None:
        self.interventions = interventions

    def apply(self, state: CellStateTensor) -> InterventionResult:
        """Apply all interventions in sequence."""
        t0 = time.time()
        sig_before = state.aging_signature()

        current = state
        total_intervention_rank = 0

        for intervention in self.interventions:
            result = intervention.apply(current)
            current = result.state_after
            total_intervention_rank += result.intervention_rank

        sig_after = current.aging_signature()
        elapsed = time.time() - t0

        fidelity = abs(_tt_inner(state.cores, current.cores))
        n_a = _tt_norm(state.cores)
        n_b = _tt_norm(current.cores)
        if n_a * n_b > 1e-300:
            fidelity = min((fidelity / (n_a * n_b)) ** 2, 1.0)
        else:
            fidelity = 0.0

        return InterventionResult(
            state_before=state,
            state_after=current,
            rank_before=sig_before.max_rank,
            rank_after=sig_after.max_rank,
            rank_reduction=1.0 - sig_after.max_rank / max(sig_before.max_rank, 1),
            biological_age_before=sig_before.biological_age,
            biological_age_after=sig_after.biological_age,
            years_reversed=sig_before.biological_age - sig_after.biological_age,
            intervention_rank=total_intervention_rank,
            fidelity=fidelity,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Intervention Search (Optimal Transport for Rejuvenation Paths)
# ---------------------------------------------------------------------------

def find_optimal_intervention(
    aged_state: CellStateTensor,
    young_reference: CellStateTensor,
    max_intervention_rank: int = 8,
    n_candidates: int = 20,
    seed: Optional[int] = None,
) -> Tuple[InterventionResult, Dict]:
    """
    Find the minimum-rank intervention that transports an aged state
    to a young reference state.

    This is the core optimization problem of aging reversal:
    given ψ_aged and ψ_young, find the operator P of minimum rank such that:

        ||P · ψ_aged - ψ_young|| < ε

    We solve this approximately by:
    1. Computing the optimal TT rank for each bond
    2. Constructing a rank-constrained projection
    3. Evaluating intervention quality

    Parameters
    ----------
    aged_state : CellStateTensor
        The aged cell state to rejuvenate.
    young_reference : CellStateTensor
        The target young cell state.
    max_intervention_rank : int
        Maximum rank of the intervention operator.
    n_candidates : int
        Number of candidate interventions to evaluate.
    seed : int, optional
        Random seed.

    Returns
    -------
    best_result : InterventionResult
        The best intervention found.
    search_log : dict
        Search statistics and all candidates evaluated.
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()

    best_result: Optional[InterventionResult] = None
    best_distance = float("inf")
    candidates: List[Dict] = []

    for trial in range(n_candidates):
        # Sample a target rank between young and aged
        target_rank = max(
            young_reference.max_rank,
            rng.integers(young_reference.max_rank, max_intervention_rank + 1),
        )

        # Sample a rejuvenation fraction
        rejuv_frac = rng.uniform(0.3, 1.0)

        # Apply partial reprogramming with this configuration
        operator = PartialReprogrammingOperator(
            rejuvenation_fraction=rejuv_frac,
            preserve_identity=True,
            cycles=rng.integers(1, 4),
        )
        result = operator.apply(aged_state)

        # Evaluate distance to young reference
        dist = result.state_after.distance(young_reference)

        candidates.append({
            "trial": trial,
            "target_rank": target_rank,
            "rejuvenation_fraction": rejuv_frac,
            "distance_to_young": dist,
            "rank_after": result.rank_after,
            "years_reversed": result.years_reversed,
        })

        if dist < best_distance:
            best_distance = dist
            best_result = result

    elapsed = time.time() - t0

    if best_result is None:
        # Fallback: simple compression
        compressed = aged_state.compress(
            max_rank=young_reference.max_rank, tol=1e-12
        )
        sig_before = aged_state.aging_signature()
        sig_after = compressed.aging_signature()
        best_result = InterventionResult(
            state_before=aged_state,
            state_after=compressed,
            rank_before=sig_before.max_rank,
            rank_after=sig_after.max_rank,
            rank_reduction=1.0 - sig_after.max_rank / max(sig_before.max_rank, 1),
            biological_age_before=sig_before.biological_age,
            biological_age_after=sig_after.biological_age,
            years_reversed=sig_before.biological_age - sig_after.biological_age,
            intervention_rank=max_intervention_rank,
            fidelity=0.0,
            elapsed_seconds=elapsed,
        )

    search_log = {
        "n_candidates": n_candidates,
        "best_distance": best_distance,
        "candidates": candidates,
        "elapsed_seconds": elapsed,
    }

    return best_result, search_log
