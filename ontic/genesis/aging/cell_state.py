"""
Cell State Tensor — The biological state of a cell in QTT format.

TENSOR GENESIS Protocol — Layer 27 (Aging)
Phase 21 of the Civilization Stack

Core Thesis:
    The biological state of a cell is a tensor over gene expression,
    protein abundance, and epigenetic marks. Aging is rank growth:
    stochastic damage, epigenetic drift, and proteostatic collapse
    increase the effective TT rank over time. Reversal is rank reduction:
    Yamanaka factors perform a rank-4 intervention that collapses the
    state back to its initial (young) condition.

    A young cell has low rank because its regulatory programs are coherent:
    gene expression → protein abundance → epigenetic state are tightly
    coupled through a small number of master regulatory circuits.
    Aging decouples these — noise accumulates, correlations break, rank grows.

Mathematical Framework:
    Cell state ψ ∈ ℝ^{d₁ × d₂ × ... × d_N} where:
        d₁ = gene expression levels (quantized)
        d₂ = protein abundances (quantized)
        d₃ = methylation state at CpG sites
        d₄ = histone modification combinatorial code
        d₅ = chromatin accessibility
        ...
        d_N = metabolite concentrations

    TT decomposition: ψ = G₁ · G₂ · ... · G_N
    where G_k has shape (r_{k-1}, d_k, r_k)

    Young cell:  max_rank(ψ_young) ≈ 4-8
    Aged cell:   max_rank(ψ_aged) ≈ 50-200
    Yamanaka:    max_rank(OSKM · ψ_aged) ≈ 4

    This explains *why* four transcription factors suffice:
    they impose rank-4 structure on the full state tensor.

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import scipy.linalg as la


# ---------------------------------------------------------------------------
# Biological Constants
# ---------------------------------------------------------------------------

# Human genome reference counts
NUM_PROTEIN_CODING_GENES: int = 20_000
NUM_PROTEINS: int = 100_000
NUM_CPG_SITES: int = 28_000_000
NUM_HISTONE_MARKS: int = 12  # Core combinatorial histone code
NUM_CHROMATIN_STATES: int = 15  # ChromHMM resolution
NUM_METABOLITES: int = 2_500

# Quantization depths (log₂ of levels per mode)
EXPRESSION_BITS: int = 8   # 256 expression levels per gene
PROTEIN_BITS: int = 8      # 256 abundance levels per protein
METHYLATION_BITS: int = 1  # Binary: methylated or not
HISTONE_BITS: int = 4      # 16 combinatorial histone states
CHROMATIN_BITS: int = 4    # 16 accessibility levels
METABOLITE_BITS: int = 6   # 64 concentration levels

# Yamanaka factor indices in gene expression space
YAMANAKA_FACTORS: Dict[str, int] = {
    "OCT4": 5460,   # POU5F1 - pluripotency master regulator
    "SOX2": 6657,   # SOX2 - neural/stem cell TF
    "KLF4": 9314,   # KLF4 - epithelial differentiation
    "MYC": 4609,    # MYC - proliferation/metabolism
}

# Thomson factors (alternative reprogramming cocktail)
THOMSON_FACTORS: Dict[str, int] = {
    "OCT4": 5460,
    "SOX2": 6657,
    "NANOG": 79923,
    "LIN28": 26013,
}

# Horvath clock CpG sites (353 sites from Horvath 2013)
HORVATH_CLOCK_SITES: int = 353

# Hannum clock CpG sites
HANNUM_CLOCK_SITES: int = 71


class CellType(Enum):
    """Major human cell types with distinct epigenetic landscapes."""
    EMBRYONIC_STEM = auto()
    INDUCED_PLURIPOTENT = auto()
    FIBROBLAST = auto()
    EPITHELIAL = auto()
    NEURON = auto()
    HEPATOCYTE = auto()
    CARDIOMYOCYTE = auto()
    T_CELL = auto()
    B_CELL = auto()
    MACROPHAGE = auto()
    ADIPOCYTE = auto()
    MYOCYTE = auto()
    OSTEOBLAST = auto()
    CHONDROCYTE = auto()
    KERATINOCYTE = auto()
    ENDOTHELIAL = auto()
    HEMATOPOIETIC_STEM = auto()
    NEURAL_STEM = auto()
    MESENCHYMAL_STEM = auto()
    INTESTINAL_STEM = auto()


class AgingHallmark(Enum):
    """López-Otín hallmarks of aging (2023 update, 12 hallmarks)."""
    GENOMIC_INSTABILITY = auto()
    TELOMERE_ATTRITION = auto()
    EPIGENETIC_ALTERATIONS = auto()
    LOSS_OF_PROTEOSTASIS = auto()
    DISABLED_MACROAUTOPHAGY = auto()
    DEREGULATED_NUTRIENT_SENSING = auto()
    MITOCHONDRIAL_DYSFUNCTION = auto()
    CELLULAR_SENESCENCE = auto()
    STEM_CELL_EXHAUSTION = auto()
    ALTERED_INTERCELLULAR_COMMUNICATION = auto()
    CHRONIC_INFLAMMATION = auto()
    DYSBIOSIS = auto()


class BiologicalMode(Enum):
    """Tensor modes for cell state representation."""
    GENE_EXPRESSION = auto()
    PROTEIN_ABUNDANCE = auto()
    METHYLATION = auto()
    HISTONE_CODE = auto()
    CHROMATIN_ACCESS = auto()
    METABOLOME = auto()
    SIGNALING = auto()
    TELOMERE_LENGTH = auto()


# ---------------------------------------------------------------------------
# QTT Core Operations (inlined for zero-dependency biology module)
# ---------------------------------------------------------------------------

def _tt_svd(
    tensor: np.ndarray,
    shape: Sequence[int],
    max_rank: Optional[int] = None,
    tol: float = 1e-12,
) -> List[np.ndarray]:
    """
    TT-SVD decomposition (Oseledets 2011).

    Decomposes a tensor of shape (d₁, d₂, ..., d_N) into a train of cores
    G_k of shape (r_{k-1}, d_k, r_k) with r_0 = r_N = 1.

    Parameters
    ----------
    tensor : np.ndarray
        Flattened tensor data of length prod(shape).
    shape : sequence of int
        The mode dimensions (d₁, d₂, ..., d_N).
    max_rank : int, optional
        Maximum TT rank. If None, determined by tolerance.
    tol : float
        Truncation tolerance (relative to Frobenius norm).

    Returns
    -------
    cores : list of np.ndarray
        TT cores, each of shape (r_{k-1}, d_k, r_k).
    """
    n_modes = len(shape)
    total_size = int(np.prod(shape))
    if tensor.size != total_size:
        raise ValueError(
            f"Tensor size {tensor.size} != product of shape {shape} = {total_size}"
        )

    tensor = tensor.ravel().astype(np.float64)
    frobenius_norm = np.linalg.norm(tensor)
    if frobenius_norm < 1e-300:
        # Zero tensor: return rank-1 zero cores
        return [np.zeros((1, d, 1)) for d in shape]

    # Truncation threshold per bond (distributed equally)
    delta = (tol * frobenius_norm) / math.sqrt(n_modes - 1) if n_modes > 1 else 0.0

    cores: List[np.ndarray] = []
    remainder = tensor.copy()
    r_prev = 1

    for k in range(n_modes - 1):
        d_k = shape[k]
        cols = remainder.size // (r_prev * d_k)
        if cols < 1:
            raise ValueError(
                f"Degenerate reshape at mode {k}: "
                f"r_prev={r_prev}, d_k={d_k}, remainder size={remainder.size}"
            )

        mat = remainder.reshape(r_prev * d_k, cols)

        # SVD with truncation
        try:
            U, S, Vt = la.svd(mat, full_matrices=False)
        except la.LinAlgError:
            # Fallback to numpy SVD
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)

        # Determine rank from tolerance
        if delta > 0:
            cumulative_sq = np.cumsum(S[::-1] ** 2)[::-1]
            rank_tol = np.searchsorted(-cumulative_sq, -delta**2) + 1
            rank_tol = min(rank_tol, len(S))
        else:
            rank_tol = len(S)

        # Apply max_rank constraint
        r_k = rank_tol
        if max_rank is not None:
            r_k = min(r_k, max_rank)
        r_k = max(r_k, 1)

        # Truncate
        U = U[:, :r_k]
        S = S[:r_k]
        Vt = Vt[:r_k, :]

        # Core shape: (r_{k-1}, d_k, r_k)
        core = U.reshape(r_prev, d_k, r_k)
        cores.append(core)

        # Pass S into remainder
        remainder = np.diag(S) @ Vt
        r_prev = r_k

    # Last core
    d_last = shape[-1]
    last_core = remainder.reshape(r_prev, d_last, 1)
    cores.append(last_core)

    return cores


def _tt_round(
    cores: List[np.ndarray],
    max_rank: Optional[int] = None,
    tol: float = 1e-12,
) -> List[np.ndarray]:
    """
    TT rounding: compress a TT decomposition to lower rank.

    Performs left-orthogonalization via QR, then right-to-left SVD truncation.

    Parameters
    ----------
    cores : list of np.ndarray
        TT cores, each of shape (r_{k-1}, d_k, r_k).
    max_rank : int, optional
        Maximum TT rank after rounding.
    tol : float
        Relative truncation tolerance.

    Returns
    -------
    rounded : list of np.ndarray
        Compressed TT cores.
    """
    n = len(cores)
    if n == 0:
        return []
    if n == 1:
        return [cores[0].copy()]

    # Copy cores
    work = [c.copy() for c in cores]

    # Left-to-right QR sweep (left-orthogonalize)
    for k in range(n - 1):
        r_left, d_k, r_right = work[k].shape
        mat = work[k].reshape(r_left * d_k, r_right)
        Q, R = np.linalg.qr(mat)
        new_rank = Q.shape[1]
        work[k] = Q.reshape(r_left, d_k, new_rank)
        # Absorb R into next core
        r_next_left, d_next, r_next_right = work[k + 1].shape
        work[k + 1] = np.einsum("ij,jkl->ikl", R, work[k + 1])

    # Compute norm for tolerance
    last = work[-1]
    total_norm = np.linalg.norm(last.ravel())
    if total_norm < 1e-300:
        return [np.zeros_like(c) for c in cores]

    delta = tol * total_norm / math.sqrt(max(n - 1, 1))

    # Right-to-left SVD sweep (truncate)
    for k in range(n - 1, 0, -1):
        r_left, d_k, r_right = work[k].shape
        mat = work[k].reshape(r_left, d_k * r_right)
        try:
            U, S, Vt = la.svd(mat, full_matrices=False)
        except la.LinAlgError:
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)

        # Rank from tolerance
        if delta > 0:
            cumulative_sq = np.cumsum(S[::-1] ** 2)[::-1]
            rank_tol = np.searchsorted(-cumulative_sq, -delta**2) + 1
            rank_tol = min(rank_tol, len(S))
        else:
            rank_tol = len(S)

        r_new = rank_tol
        if max_rank is not None:
            r_new = min(r_new, max_rank)
        r_new = max(r_new, 1)

        U = U[:, :r_new]
        S = S[:r_new]
        Vt = Vt[:r_new, :]

        work[k] = Vt.reshape(r_new, d_k, r_right)
        # Absorb U @ diag(S) into left neighbor
        r_left_prev, d_prev, _ = work[k - 1].shape
        work[k - 1] = np.einsum("ijk,kl->ijl", work[k - 1], U * S[np.newaxis, :])

    return work


def _tt_add(
    cores_a: List[np.ndarray],
    cores_b: List[np.ndarray],
) -> List[np.ndarray]:
    """
    TT addition via block-diagonal core concatenation.

    rank(A + B) ≤ rank(A) + rank(B).
    """
    n = len(cores_a)
    if n != len(cores_b):
        raise ValueError("Incompatible number of cores")

    result: List[np.ndarray] = []
    for k in range(n):
        ra_left, d_a, ra_right = cores_a[k].shape
        rb_left, d_b, rb_right = cores_b[k].shape
        if d_a != d_b:
            raise ValueError(f"Mode dimension mismatch at core {k}: {d_a} vs {d_b}")
        d = d_a

        if k == 0:
            # First core: horizontal concatenation [A | B]
            core = np.zeros((1, d, ra_right + rb_right))
            core[0, :, :ra_right] = cores_a[k][0, :, :]
            core[0, :, ra_right:] = cores_b[k][0, :, :]
        elif k == n - 1:
            # Last core: vertical concatenation [A; B]
            core = np.zeros((ra_left + rb_left, d, 1))
            core[:ra_left, :, 0] = cores_a[k][:, :, 0]
            core[ra_left:, :, 0] = cores_b[k][:, :, 0]
        else:
            # Interior core: block diagonal [[A, 0], [0, B]]
            core = np.zeros((ra_left + rb_left, d, ra_right + rb_right))
            core[:ra_left, :, :ra_right] = cores_a[k]
            core[ra_left:, :, ra_right:] = cores_b[k]
        result.append(core)

    return result


def _tt_inner(
    cores_a: List[np.ndarray],
    cores_b: List[np.ndarray],
) -> float:
    """
    TT inner product via left-to-right contraction.

    Complexity: O(N · d · r²) without densification.
    """
    n = len(cores_a)
    # Initialize transfer matrix
    transfer = np.ones((1, 1))

    for k in range(n):
        # Contract: transfer_{i,j} @ A_{i,d,i'} @ B_{j,d,j'} -> transfer_{i',j'}
        ga = cores_a[k]  # (ra_left, d, ra_right)
        gb = cores_b[k]  # (rb_left, d, rb_right)
        # transfer @ ga -> (rb_left, d, ra_right) summed over ra_left
        tmp = np.einsum("ij,idk->jdk", transfer, ga)
        # contract with gb over d and rb_left
        transfer = np.einsum("jdk,jdl->kl", tmp, gb)

    return float(transfer[0, 0])


def _tt_norm(cores: List[np.ndarray]) -> float:
    """TT Frobenius norm via inner product."""
    return math.sqrt(max(_tt_inner(cores, cores), 0.0))


def _tt_scale(cores: List[np.ndarray], scalar: float) -> List[np.ndarray]:
    """Scale a TT by a scalar (applied to first core only)."""
    result = [c.copy() for c in cores]
    if len(result) > 0:
        result[0] = result[0] * scalar
    return result


def _tt_ranks(cores: List[np.ndarray]) -> List[int]:
    """Extract TT ranks from cores."""
    if not cores:
        return []
    ranks = [cores[0].shape[0]]
    for c in cores:
        ranks.append(c.shape[2])
    return ranks


def _tt_max_rank(cores: List[np.ndarray]) -> int:
    """Maximum TT rank across all bonds."""
    if not cores:
        return 0
    return max(c.shape[2] for c in cores[:-1]) if len(cores) > 1 else 1


# ---------------------------------------------------------------------------
# Cell State Tensor
# ---------------------------------------------------------------------------

@dataclass
class ModeSpec:
    """Specification for one tensor mode (biological dimension)."""
    name: str
    mode_type: BiologicalMode
    n_features: int
    bits_per_feature: int
    description: str

    @property
    def local_dim(self) -> int:
        """Quantization levels per site in this mode's QTT unfolding."""
        return 2 ** self.bits_per_feature

    @property
    def n_qtt_sites(self) -> int:
        """Number of QTT sites needed for this mode."""
        return self.n_features * self.bits_per_feature


@dataclass
class AgingSignature:
    """Quantitative aging signature extracted from cell state."""
    chronological_age: float
    biological_age: float
    max_rank: int
    mean_rank: float
    rank_trajectory: List[int]
    hallmark_scores: Dict[AgingHallmark, float]
    epigenetic_age: float
    proteostatic_burden: float
    telomere_index: float
    entropy: float
    timestamp: float = field(default_factory=time.time)

    @property
    def age_acceleration(self) -> float:
        """Difference between biological and chronological age."""
        return self.biological_age - self.chronological_age

    @property
    def rejuvenation_potential(self) -> float:
        """
        Fraction of rank that is compressible — how much rank can be removed
        by an ideal intervention. 1.0 = fully reversible, 0.0 = irreversible.
        """
        if self.max_rank <= 1:
            return 0.0
        # Young baseline rank is ~4 (Yamanaka dimensionality)
        baseline_rank = 4
        if self.max_rank <= baseline_rank:
            return 0.0
        return 1.0 - (baseline_rank / self.max_rank)


class CellStateTensor:
    """
    The biological state of a cell as a Quantized Tensor Train.

    This is the central data structure of the aging framework. A cell's state
    spans gene expression, protein abundance, epigenetic marks, chromatin
    accessibility, and metabolite concentrations — all encoded as a single
    tensor train with quantized indices.

    Aging = rank growth. Young cells have coherent regulatory programs
    (low rank). Aging decouples these programs through stochastic damage,
    epigenetic drift, and proteostatic collapse (rank grows). Reversal
    is rank reduction: find operators that restore coherence.

    Attributes
    ----------
    cores : list of np.ndarray
        TT cores, each of shape (r_{k-1}, 2, r_k) for binary QTT.
    mode_specs : list of ModeSpec
        Specification of each biological dimension.
    cell_type : CellType
        The cell type this state represents.
    chronological_age : float
        Chronological age in years.
    metadata : dict
        Additional metadata (tissue source, donor ID, etc.).
    """

    __slots__ = (
        "cores",
        "mode_specs",
        "cell_type",
        "chronological_age",
        "metadata",
        "_rank_history",
        "_creation_time",
    )

    def __init__(
        self,
        cores: List[np.ndarray],
        mode_specs: List[ModeSpec],
        cell_type: CellType,
        chronological_age: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        if not cores:
            raise ValueError("Cell state requires at least one TT core")
        # Validate core shapes
        for i, core in enumerate(cores):
            if core.ndim != 3:
                raise ValueError(
                    f"Core {i} has {core.ndim} dimensions, expected 3 "
                    f"(r_left, d, r_right)"
                )
            if i == 0 and core.shape[0] != 1:
                raise ValueError(f"First core left rank must be 1, got {core.shape[0]}")
            if i == len(cores) - 1 and core.shape[2] != 1:
                raise ValueError(f"Last core right rank must be 1, got {core.shape[2]}")
            if i > 0 and cores[i - 1].shape[2] != core.shape[0]:
                raise ValueError(
                    f"Rank mismatch between core {i-1} right rank "
                    f"({cores[i-1].shape[2]}) and core {i} left rank "
                    f"({core.shape[0]})"
                )

        self.cores = cores
        self.mode_specs = mode_specs
        self.cell_type = cell_type
        self.chronological_age = chronological_age
        self.metadata = metadata or {}
        self._rank_history: List[Tuple[float, int]] = [
            (chronological_age, self.max_rank)
        ]
        self._creation_time = time.time()

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

    @property
    def n_sites(self) -> int:
        """Total number of QTT sites."""
        return len(self.cores)

    @property
    def local_dims(self) -> List[int]:
        """Physical dimension at each site."""
        return [c.shape[1] for c in self.cores]

    @property
    def ranks(self) -> List[int]:
        """TT ranks (bond dimensions) at each bond."""
        return _tt_ranks(self.cores)

    @property
    def max_rank(self) -> int:
        """Maximum TT rank — the key aging biomarker."""
        return _tt_max_rank(self.cores)

    @property
    def mean_rank(self) -> float:
        """Mean TT rank across all bonds."""
        ranks = self.ranks[1:-1]  # Exclude boundary ranks (always 1)
        if not ranks:
            return 1.0
        return float(np.mean(ranks))

    @property
    def rank_entropy(self) -> float:
        """
        Shannon entropy of the singular value spectrum across all bonds.

        Higher entropy = more disorder = more aged.
        A perfectly coherent (young) state has low entropy.
        """
        total_entropy = 0.0
        n_bonds = 0
        for k in range(len(self.cores) - 1):
            r_left, d_k, r_right = self.cores[k].shape
            mat = self.cores[k].reshape(r_left * d_k, r_right)
            try:
                S = la.svdvals(mat)
            except la.LinAlgError:
                S = np.linalg.svd(mat, compute_uv=False)
            S = S[S > 1e-300]
            if len(S) > 0:
                p = S**2 / np.sum(S**2)
                total_entropy -= float(np.sum(p * np.log(p + 1e-300)))
            n_bonds += 1
        return total_entropy / max(n_bonds, 1)

    @property
    def total_dim(self) -> float:
        """
        Total dimension of the uncompressed state space.

        Returns float because the true dimension (product of local_dims)
        can vastly exceed integer and even float64 representable range
        for biological-scale tensors.
        """
        log2_dim = sum(math.log2(d) for d in self.local_dims)
        if log2_dim > 1023:  # float64 exponent limit
            return float("inf")
        return 2.0 ** log2_dim

    @property
    def memory_bytes(self) -> int:
        """Memory usage of the TT representation."""
        return sum(c.nbytes for c in self.cores)

    @property
    def compression_ratio(self) -> float:
        """
        Ratio of uncompressed to compressed storage.

        Computed in log-domain to avoid integer overflow when the
        uncompressed state space is astronomically large (e.g., 2^1056).
        """
        mem = max(self.memory_bytes, 1)
        # log2(dense_size) = log2(total_dim) + log2(8)
        log2_total = sum(math.log2(d) for d in self.local_dims)
        log2_dense = log2_total + 3.0  # 8 bytes per float64 = 2^3
        log2_mem = math.log2(mem)
        exponent = log2_dense - log2_mem
        if exponent > 1023:
            return float("inf")
        if exponent < -1023:
            return 0.0
        return 2.0 ** exponent

    @property
    def norm(self) -> float:
        """Frobenius norm of the cell state tensor."""
        return _tt_norm(self.cores)

    # -------------------------------------------------------------------
    # TT Arithmetic
    # -------------------------------------------------------------------

    def add(self, other: "CellStateTensor") -> "CellStateTensor":
        """
        Add two cell states. Rank grows: rank(A+B) ≤ rank(A) + rank(B).
        This models accumulation of perturbations.
        """
        if self.n_sites != other.n_sites:
            raise ValueError("Cannot add cell states with different site counts")
        new_cores = _tt_add(self.cores, other.cores)
        return CellStateTensor(
            cores=new_cores,
            mode_specs=self.mode_specs,
            cell_type=self.cell_type,
            chronological_age=self.chronological_age,
            metadata={**self.metadata, "operation": "add"},
        )

    def scale(self, factor: float) -> "CellStateTensor":
        """Scale the cell state by a factor."""
        new_cores = _tt_scale(self.cores, factor)
        return CellStateTensor(
            cores=new_cores,
            mode_specs=self.mode_specs,
            cell_type=self.cell_type,
            chronological_age=self.chronological_age,
            metadata=self.metadata,
        )

    def compress(
        self,
        max_rank: Optional[int] = None,
        tol: float = 1e-12,
    ) -> "CellStateTensor":
        """
        Compress the cell state to lower rank via TT rounding.
        This is the mathematical operation underlying rejuvenation.
        """
        new_cores = _tt_round(self.cores, max_rank=max_rank, tol=tol)
        result = CellStateTensor(
            cores=new_cores,
            mode_specs=self.mode_specs,
            cell_type=self.cell_type,
            chronological_age=self.chronological_age,
            metadata={**self.metadata, "compressed": True},
        )
        return result

    def inner(self, other: "CellStateTensor") -> float:
        """Inner product between two cell states (without densification)."""
        return _tt_inner(self.cores, other.cores)

    def distance(self, other: "CellStateTensor") -> float:
        """
        Frobenius distance between two cell states.
        ||A - B||² = ||A||² + ||B||² - 2⟨A,B⟩
        """
        norm_a_sq = _tt_inner(self.cores, self.cores)
        norm_b_sq = _tt_inner(other.cores, other.cores)
        inner_ab = _tt_inner(self.cores, other.cores)
        dist_sq = norm_a_sq + norm_b_sq - 2.0 * inner_ab
        return math.sqrt(max(dist_sq, 0.0))

    # -------------------------------------------------------------------
    # Aging Signature
    # -------------------------------------------------------------------

    def aging_signature(self) -> AgingSignature:
        """
        Extract a comprehensive aging signature from the cell state.

        The signature captures:
        - Rank-based age estimation (biological age from TT structure)
        - Hallmark decomposition (which aging mechanisms dominate)
        - Epigenetic age (Horvath-style from methylation modes)
        - Proteostatic burden (from protein abundance mode ranks)
        - Telomere state (from telomere mode)
        - Information entropy (disorder measure)
        """
        # Biological age estimated from rank structure
        # Calibration: rank 4 ≈ age 0 (embryonic), rank 200 ≈ age 120
        bio_age = self._estimate_biological_age()

        # Decompose aging into hallmark contributions
        hallmark_scores = self._compute_hallmark_scores()

        # Epigenetic age from methylation mode ranks
        epi_age = self._estimate_epigenetic_age()

        # Proteostatic burden from protein mode ranks
        proteo_burden = self._compute_proteostatic_burden()

        # Telomere index
        telo_idx = self._compute_telomere_index()

        return AgingSignature(
            chronological_age=self.chronological_age,
            biological_age=bio_age,
            max_rank=self.max_rank,
            mean_rank=self.mean_rank,
            rank_trajectory=[r for _, r in self._rank_history],
            hallmark_scores=hallmark_scores,
            epigenetic_age=epi_age,
            proteostatic_burden=proteo_burden,
            telomere_index=telo_idx,
            entropy=self.rank_entropy,
        )

    def _estimate_biological_age(self) -> float:
        """
        Estimate biological age from TT rank structure.

        Model: age = α · log(max_rank / r₀) where r₀ = baseline rank (4).
        Calibrated so rank 4 → age 0, rank 200 → age 120.
        """
        r0 = 4.0  # Yamanaka baseline rank
        r_max = float(self.max_rank)
        if r_max <= r0:
            return 0.0

        # Calibration: 120 = α · log(200/4) → α = 120 / log(50) ≈ 30.67
        alpha = 120.0 / math.log(200.0 / r0)
        age = alpha * math.log(r_max / r0)
        return max(age, 0.0)

    def _compute_hallmark_scores(self) -> Dict[AgingHallmark, float]:
        """
        Decompose aging into hallmark contributions based on mode-specific ranks.

        Each hallmark maps to specific tensor modes. The rank growth in those
        modes relative to baseline indicates hallmark severity.
        """
        # Map hallmarks to modes
        hallmark_mode_map: Dict[AgingHallmark, List[BiologicalMode]] = {
            AgingHallmark.GENOMIC_INSTABILITY: [BiologicalMode.GENE_EXPRESSION],
            AgingHallmark.TELOMERE_ATTRITION: [BiologicalMode.TELOMERE_LENGTH],
            AgingHallmark.EPIGENETIC_ALTERATIONS: [
                BiologicalMode.METHYLATION,
                BiologicalMode.HISTONE_CODE,
                BiologicalMode.CHROMATIN_ACCESS,
            ],
            AgingHallmark.LOSS_OF_PROTEOSTASIS: [BiologicalMode.PROTEIN_ABUNDANCE],
            AgingHallmark.DISABLED_MACROAUTOPHAGY: [BiologicalMode.PROTEIN_ABUNDANCE],
            AgingHallmark.DEREGULATED_NUTRIENT_SENSING: [
                BiologicalMode.METABOLOME,
                BiologicalMode.SIGNALING,
            ],
            AgingHallmark.MITOCHONDRIAL_DYSFUNCTION: [BiologicalMode.METABOLOME],
            AgingHallmark.CELLULAR_SENESCENCE: [
                BiologicalMode.GENE_EXPRESSION,
                BiologicalMode.SIGNALING,
            ],
            AgingHallmark.STEM_CELL_EXHAUSTION: [BiologicalMode.GENE_EXPRESSION],
            AgingHallmark.ALTERED_INTERCELLULAR_COMMUNICATION: [
                BiologicalMode.SIGNALING
            ],
            AgingHallmark.CHRONIC_INFLAMMATION: [BiologicalMode.SIGNALING],
            AgingHallmark.DYSBIOSIS: [BiologicalMode.METABOLOME],
        }

        # Compute mode-specific rank contributions
        mode_ranks = self._compute_mode_ranks()
        baseline_rank = 4.0

        scores: Dict[AgingHallmark, float] = {}
        for hallmark, modes in hallmark_mode_map.items():
            mode_rank_sum = 0.0
            count = 0
            for mode in modes:
                if mode in mode_ranks:
                    mode_rank_sum += mode_ranks[mode]
                    count += 1
            if count > 0:
                avg_rank = mode_rank_sum / count
                # Score: normalized rank excess above baseline
                scores[hallmark] = max(0.0, (avg_rank - baseline_rank) / baseline_rank)
            else:
                scores[hallmark] = 0.0

        return scores

    def _compute_mode_ranks(self) -> Dict[BiologicalMode, float]:
        """
        Compute the effective rank contribution from each biological mode.

        Groups consecutive QTT sites by their mode assignment and computes
        the maximum rank within each mode's site range.
        """
        mode_ranks: Dict[BiologicalMode, float] = {}

        # Build site-to-mode mapping
        site_idx = 0
        for spec in self.mode_specs:
            n_sites = spec.n_qtt_sites
            end_idx = min(site_idx + n_sites, len(self.cores))

            # Max rank within this mode's range
            max_r = 1
            for k in range(site_idx, min(end_idx, len(self.cores))):
                r_right = self.cores[k].shape[2]
                if r_right > max_r:
                    max_r = r_right
            mode_ranks[spec.mode_type] = float(max_r)
            site_idx = end_idx

        return mode_ranks

    def _estimate_epigenetic_age(self) -> float:
        """
        Estimate epigenetic age from methylation mode rank structure.

        In the Horvath model, 353 CpG sites predict age. In QTT, this
        corresponds to the rank structure of the methylation sub-tensor.
        Higher rank = more epigenetic drift = older.
        """
        methylation_rank = 4.0  # default young
        for spec in self.mode_specs:
            if spec.mode_type == BiologicalMode.METHYLATION:
                ranks = self._compute_mode_ranks()
                methylation_rank = ranks.get(BiologicalMode.METHYLATION, 4.0)
                break

        # Horvath calibration: methylation rank 4 → age 0, rank 80 → age 120
        r0 = 4.0
        if methylation_rank <= r0:
            return 0.0
        alpha = 120.0 / math.log(80.0 / r0)
        return alpha * math.log(methylation_rank / r0)

    def _compute_proteostatic_burden(self) -> float:
        """
        Compute proteostatic burden from protein abundance mode.

        Higher rank in protein mode = more misfolded/aggregated protein states
        = higher burden. Normalized to [0, 1].
        """
        protein_rank = 4.0
        for spec in self.mode_specs:
            if spec.mode_type == BiologicalMode.PROTEIN_ABUNDANCE:
                ranks = self._compute_mode_ranks()
                protein_rank = ranks.get(BiologicalMode.PROTEIN_ABUNDANCE, 4.0)
                break

        # Normalize: rank 4 → 0.0, rank 200 → 1.0
        return min(1.0, max(0.0, (protein_rank - 4.0) / 196.0))

    def _compute_telomere_index(self) -> float:
        """
        Compute telomere length index from telomere mode.

        Young: telomere rank = 1 (uniform length), index = 1.0
        Aged: telomere rank grows (variable length), index → 0.0
        """
        telo_rank = 1.0
        for spec in self.mode_specs:
            if spec.mode_type == BiologicalMode.TELOMERE_LENGTH:
                ranks = self._compute_mode_ranks()
                telo_rank = ranks.get(BiologicalMode.TELOMERE_LENGTH, 1.0)
                break

        # Index: 1/rank (higher rank = shorter/more variable telomeres)
        return 1.0 / max(telo_rank, 1.0)

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize cell state to dictionary (cores stored as lists)."""
        return {
            "cores": [c.tolist() for c in self.cores],
            "mode_specs": [
                {
                    "name": s.name,
                    "mode_type": s.mode_type.name,
                    "n_features": s.n_features,
                    "bits_per_feature": s.bits_per_feature,
                    "description": s.description,
                }
                for s in self.mode_specs
            ],
            "cell_type": self.cell_type.name,
            "chronological_age": self.chronological_age,
            "metadata": self.metadata,
            "rank_history": self._rank_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CellStateTensor":
        """Deserialize cell state from dictionary."""
        cores = [np.array(c) for c in data["cores"]]
        mode_specs = [
            ModeSpec(
                name=s["name"],
                mode_type=BiologicalMode[s["mode_type"]],
                n_features=s["n_features"],
                bits_per_feature=s["bits_per_feature"],
                description=s["description"],
            )
            for s in data["mode_specs"]
        ]
        cell_type = CellType[data["cell_type"]]
        cst = cls(
            cores=cores,
            mode_specs=mode_specs,
            cell_type=cell_type,
            chronological_age=data.get("chronological_age", 0.0),
            metadata=data.get("metadata", {}),
        )
        cst._rank_history = [
            (age, rank) for age, rank in data.get("rank_history", [])
        ]
        return cst

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CellStateTensor("
            f"type={self.cell_type.name}, "
            f"age={self.chronological_age:.1f}y, "
            f"sites={self.n_sites}, "
            f"max_rank={self.max_rank}, "
            f"mean_rank={self.mean_rank:.1f}, "
            f"entropy={self.rank_entropy:.4f}, "
            f"memory={self.memory_bytes:,}B"
            f")"
        )


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def _default_mode_specs() -> List[ModeSpec]:
    """
    Default mode specifications for a human cell state.

    Site budget: ~88 total QTT sites — small enough for numerically stable
    TT arithmetic in float64 while large enough to capture the biological
    structure. Each mode is quantized to 2-bit depth per feature by default,
    giving 4 levels per feature. Methylation uses 1-bit (binary).

    The full-genome version would use thousands of sites; this is the
    computationally tractable representation that preserves the rank
    structure essential to the aging theory.
    """
    return [
        ModeSpec(
            name="gene_expression",
            mode_type=BiologicalMode.GENE_EXPRESSION,
            n_features=8,
            bits_per_feature=2,
            description="Quantized gene expression levels (top 8 variable genes)",
        ),
        ModeSpec(
            name="protein_abundance",
            mode_type=BiologicalMode.PROTEIN_ABUNDANCE,
            n_features=8,
            bits_per_feature=2,
            description="Quantized protein abundances (top 8 variable proteins)",
        ),
        ModeSpec(
            name="methylation",
            mode_type=BiologicalMode.METHYLATION,
            n_features=16,
            bits_per_feature=1,
            description="CpG methylation state (16 clock sites)",
        ),
        ModeSpec(
            name="histone_code",
            mode_type=BiologicalMode.HISTONE_CODE,
            n_features=4,
            bits_per_feature=2,
            description="Histone modification combinatorial code",
        ),
        ModeSpec(
            name="chromatin_access",
            mode_type=BiologicalMode.CHROMATIN_ACCESS,
            n_features=4,
            bits_per_feature=2,
            description="Chromatin accessibility (ATAC-seq)",
        ),
        ModeSpec(
            name="metabolome",
            mode_type=BiologicalMode.METABOLOME,
            n_features=4,
            bits_per_feature=2,
            description="Metabolite concentrations",
        ),
        ModeSpec(
            name="signaling",
            mode_type=BiologicalMode.SIGNALING,
            n_features=4,
            bits_per_feature=2,
            description="Signaling pathway activities",
        ),
        ModeSpec(
            name="telomere",
            mode_type=BiologicalMode.TELOMERE_LENGTH,
            n_features=4,
            bits_per_feature=2,
            description="Telomere length distribution",
        ),
    ]


def young_cell(
    cell_type: CellType = CellType.FIBROBLAST,
    max_rank: int = 4,
    seed: Optional[int] = None,
) -> CellStateTensor:
    """
    Generate a young cell state with low TT rank.

    A young cell has coherent regulatory programs — gene expression,
    protein abundance, and epigenetic state are tightly coupled through
    a small number of master regulatory circuits. This manifests as
    low TT rank.

    Parameters
    ----------
    cell_type : CellType
        The cell type to generate.
    max_rank : int
        Maximum TT rank (default 4, matching Yamanaka dimensionality).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    CellStateTensor
        A young cell state with rank ≈ max_rank.
    """
    rng = np.random.default_rng(seed)
    specs = _default_mode_specs()

    # Total QTT sites
    n_sites = sum(s.n_qtt_sites for s in specs)

    # Generate low-rank cores via left-orthogonal construction.
    # At each site, draw a random matrix and QR-factorize to get
    # an orthonormal left factor. This ensures numerical stability
    # even for many sites.
    cores: List[np.ndarray] = []
    r_prev = 1
    for k in range(n_sites):
        d = 2  # binary QTT
        r_right = 1 if k == n_sites - 1 else min(max_rank, r_prev * d)

        A = rng.standard_normal((r_prev * d, r_right))
        Q, R = np.linalg.qr(A)
        actual_r = min(Q.shape[1], r_right)
        core = Q[:, :actual_r].reshape(r_prev, d, actual_r)
        cores.append(core)
        r_prev = actual_r

    # Round to enforce max_rank exactly
    cores = _tt_round(cores, max_rank=max_rank, tol=1e-14)

    # Normalize
    norm_val = _tt_norm(cores)
    if norm_val > 1e-300:
        cores = _tt_scale(cores, 1.0 / norm_val)

    return CellStateTensor(
        cores=cores,
        mode_specs=specs,
        cell_type=cell_type,
        chronological_age=0.0,
        metadata={"generated": "young_cell", "seed": seed},
    )


def embryonic_stem_cell(seed: Optional[int] = None) -> CellStateTensor:
    """
    Generate an embryonic stem cell state.

    ESCs have the lowest rank of any cell type — they are maximally
    coherent, with pluripotency factors coordinating the entire state.
    """
    return young_cell(
        cell_type=CellType.EMBRYONIC_STEM,
        max_rank=3,
        seed=seed,
    )


def aged_cell(
    cell_type: CellType = CellType.FIBROBLAST,
    chronological_age: float = 70.0,
    seed: Optional[int] = None,
) -> CellStateTensor:
    """
    Generate an aged cell state with high TT rank.

    An aged cell has accumulated stochastic damage that decouples
    regulatory programs. Gene expression drifts, epigenetic marks erode,
    protein homeostasis breaks down — each process independently,
    increasing the effective rank of the state tensor.

    Parameters
    ----------
    cell_type : CellType
        The cell type.
    chronological_age : float
        Age in years (0-120). Determines rank growth.
    seed : int, optional
        Random seed.

    Returns
    -------
    CellStateTensor
        An aged cell state with rank proportional to age.
    """
    rng = np.random.default_rng(seed)
    specs = _default_mode_specs()
    n_sites = sum(s.n_qtt_sites for s in specs)

    # Age-dependent max rank
    # Calibration: age 0 → rank 4, age 120 → rank 200
    r0 = 4
    r_max_age = int(r0 * math.exp(chronological_age * math.log(200.0 / r0) / 120.0))
    r_max_age = max(r0, min(r_max_age, 256))

    # Generate higher-rank cores using left-orthogonal construction
    # with mode-specific rank scaling. Use global index k for rank
    # dimensions to avoid mismatches at mode boundaries.
    cores: List[np.ndarray] = []
    r_prev = 1
    site_idx = 0

    # Pre-compute mode assignment for each site (which mode owns this site)
    mode_for_site: List[BiologicalMode] = []
    for spec in specs:
        mode_for_site.extend([spec.mode_type] * spec.n_qtt_sites)

    for k in range(n_sites):
        d = 2  # binary QTT
        mode = mode_for_site[k]
        mode_aging_rate = _mode_aging_rate(mode)
        mode_max_rank = int(r0 + (r_max_age - r0) * mode_aging_rate)
        mode_max_rank = max(2, min(mode_max_rank, r_max_age))

        # Use global k for rank bounds to prevent mode-boundary mismatches
        r_right = 1 if k == n_sites - 1 else min(mode_max_rank, r_prev * d)

        # Generate left-orthogonal core via QR
        noise_amplitude = max(0.1, 0.1 + 0.2 * (chronological_age / 120.0) * mode_aging_rate)
        A = rng.standard_normal((r_prev * d, r_right)) * noise_amplitude
        Q, R = np.linalg.qr(A)
        actual_r = min(Q.shape[1], r_right)
        core = Q[:, :actual_r].reshape(r_prev, d, actual_r)
        cores.append(core)
        r_prev = actual_r

    site_idx = n_sites

    # Round to realistic rank (not exactly r_max_age, but close)
    cores = _tt_round(cores, max_rank=r_max_age, tol=1e-14)

    # Normalize
    norm_val = _tt_norm(cores)
    if norm_val > 1e-300:
        cores = _tt_scale(cores, 1.0 / norm_val)

    cst = CellStateTensor(
        cores=cores,
        mode_specs=specs,
        cell_type=cell_type,
        chronological_age=chronological_age,
        metadata={"generated": "aged_cell", "seed": seed},
    )

    return cst


def _mode_aging_rate(mode: BiologicalMode) -> float:
    """
    Relative aging rate for each biological mode.

    Based on empirical observations:
    - Epigenetic marks drift fastest (Horvath clock)
    - Protein homeostasis degrades progressively
    - Gene expression changes are partly compensatory
    - Telomeres shorten steadily
    - Metabolome reflects downstream integration
    """
    rates: Dict[BiologicalMode, float] = {
        BiologicalMode.GENE_EXPRESSION: 0.6,
        BiologicalMode.PROTEIN_ABUNDANCE: 0.8,
        BiologicalMode.METHYLATION: 1.0,       # Fastest — epigenetic clock
        BiologicalMode.HISTONE_CODE: 0.9,
        BiologicalMode.CHROMATIN_ACCESS: 0.85,
        BiologicalMode.METABOLOME: 0.7,
        BiologicalMode.SIGNALING: 0.75,
        BiologicalMode.TELOMERE_LENGTH: 0.5,   # Steady but slow
    }
    return rates.get(mode, 0.5)
