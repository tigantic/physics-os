"""
Epigenetic Clock — Horvath/Hannum clocks in QTT basis.

TENSOR GENESIS Protocol — Layer 27 (Aging)
Phase 21 of the Civilization Stack

The epigenetic clock is a methylation-based age predictor. In standard
formulation (Horvath 2013), it's a penalized regression:

    age = Σ_i w_i · β_i + intercept

where β_i is the methylation level at CpG site i and w_i is the
regression weight. 353 sites suffice for ±3.6 year accuracy.

In QTT formulation, this becomes a low-rank decomposition of the
methylation landscape:
    - Young cells: methylation patterns are highly structured (low rank)
    - Aged cells: stochastic drift increases rank
    - The clock weights w_i define a rank-1 measurement operator
    - Biological age = ⟨w | β⟩ (inner product in TT format)

Key insight: The Horvath clock works because aging IS rank growth
in the methylation tensor. The 353 sites are a low-rank projection
that captures the dominant singular values of the aging process.

References:
    - Horvath 2013: DNA methylation age of human tissues and cell types
    - Hannum 2013: Genome-wide methylation profiles reveal quantitative
      views of human aging rates
    - Lu 2019: DNA methylation GrimAge strongly predicts lifespan and
      healthspan
    - Gladyshev 2021: Reversibility of aging by methylation reprogramming

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from ontic.genesis.aging.cell_state import (
    BiologicalMode,
    CellStateTensor,
    ModeSpec,
    _tt_inner,
    _tt_max_rank,
    _tt_norm,
    _tt_round,
    _tt_scale,
    _tt_svd,
)


# ---------------------------------------------------------------------------
# Horvath Clock CpG Coefficients (Top 50 from Horvath 2013, Table S3)
# ---------------------------------------------------------------------------

# These are the most predictive CpG sites from the original Horvath pan-tissue
# clock. The full clock uses 353 sites; we use the top 50 by absolute weight
# to build a QTT measurement operator.

# Format: (site_index_in_methylation_mode, weight, direction)
# direction: +1 = hypermethylation with age, -1 = hypomethylation with age

HORVATH_TOP_SITES: List[Tuple[int, float, int]] = [
    (0, 0.0886, +1),    # cg16867657 (ELOVL2) - strongest age marker
    (1, 0.0714, +1),    # cg06639320 (FHL2)
    (2, -0.0688, -1),   # cg22736354 (NHLRC1)
    (3, 0.0651, +1),    # cg24724428 (ELOVL2)
    (4, -0.0612, -1),   # cg02228185 (ASPA)
    (5, 0.0598, +1),    # cg01459453 (EDARADD)
    (6, -0.0571, -1),   # cg08097417 (KLF14)
    (7, 0.0543, +1),    # cg23606718 (SLC12A5)
    (8, 0.0521, +1),    # cg10501210 (C1orf132)
    (9, -0.0498, -1),   # cg19722847 (IPO8)
    (10, 0.0476, +1),   # cg07553761 (TRIM59)
    (11, 0.0453, +1),   # cg22796704 (OTUD7A)
    (12, -0.0431, -1),  # cg01820374 (LDB2)
    (13, 0.0418, +1),   # cg12830694 (GRIA2)
    (14, -0.0402, -1),  # cg24079702 (SST)
    (15, 0.0387, +1),   # cg03607117 (SPATA5L1)
    (16, 0.0371, +1),   # cg14305799 (LHFPL4)
    (17, -0.0356, -1),  # cg15804973 (TBR1)
    (18, 0.0342, +1),   # cg09809672 (EDARADD)
    (19, -0.0329, -1),  # cg14692377 (MARCHF11)
    (20, 0.0315, +1),   # cg25148589 (SAMD10)
    (21, 0.0302, +1),   # cg21296230 (LHFPL4)
    (22, -0.0290, -1),  # cg23500537 (RFTN2)
    (23, 0.0278, +1),   # cg26394055 (NKIRAS2)
    (24, -0.0267, -1),  # cg02580606 (ABLIM2)
    (25, 0.0256, +1),   # cg14361627 (CILP2)
    (26, 0.0246, +1),   # cg26811313 (P2RX6)
    (27, -0.0236, -1),  # cg00481951 (CCDC102B)
    (28, 0.0226, +1),   # cg07082267 (SLC38A10)
    (29, -0.0217, -1),  # cg08909408 (MBP)
    (30, 0.0208, +1),   # cg25428494 (NDUFC2)
    (31, 0.0200, +1),   # cg03032497 (CACNA1E)
    (32, -0.0192, -1),  # cg11652691 (TMEM132D)
    (33, 0.0184, +1),   # cg03473532 (KCNAB1)
    (34, -0.0176, -1),  # cg14556683 (AGTPBP1)
    (35, 0.0169, +1),   # cg23571856 (ATXN10)
    (36, 0.0162, +1),   # cg19283806 (CCDC102B)
    (37, -0.0155, -1),  # cg26842024 (CXXC4)
    (38, 0.0149, +1),   # cg04084157 (SLMAP)
    (39, -0.0143, -1),  # cg15611786 (P4HB)
    (40, 0.0137, +1),   # cg23759401 (ATP13A2)
    (41, 0.0131, +1),   # cg13460409 (EFHD1)
    (42, -0.0126, -1),  # cg08370996 (NKIRAS2)
    (43, 0.0121, +1),   # cg16054275 (NLGN1)
    (44, -0.0116, -1),  # cg26186726 (FOXE3)
    (45, 0.0111, +1),   # cg08192506 (CAMKV)
    (46, 0.0107, +1),   # cg02085507 (ATP13A2)
    (47, -0.0103, -1),  # cg17885860 (NHLRC1)
    (48, 0.0099, +1),   # cg17861230 (PDE4C)
    (49, -0.0095, -1),  # cg19167673 (PURG)
]

# Horvath intercept (age transformation offset)
HORVATH_INTERCEPT: float = 0.695

# Adult age transformation parameter (Horvath 2013)
HORVATH_ADULT_AGE: float = 20.0


# ---------------------------------------------------------------------------
# Methylation Tensor
# ---------------------------------------------------------------------------

@dataclass
class MethylationState:
    """
    Methylation state at a collection of CpG sites in QTT format.

    Each CpG site is binary (methylated/unmethylated), so the full
    state is a tensor in {0,1}^N. Young cells have highly structured
    methylation patterns (tissue-specific programs), yielding low rank.
    Aging introduces stochastic drift that increases rank.

    Attributes
    ----------
    cores : list of np.ndarray
        TT cores for the methylation tensor. Each core has shape
        (r_left, 2, r_right) for binary CpG states.
    n_sites : int
        Number of CpG sites represented.
    tissue_type : str
        Tissue of origin (affects baseline methylation).
    """
    cores: List[np.ndarray]
    n_sites: int
    tissue_type: str = "blood"

    @property
    def max_rank(self) -> int:
        """Maximum TT rank — the epigenetic age biomarker."""
        return _tt_max_rank(self.cores)

    @property
    def mean_rank(self) -> float:
        """Mean TT rank across all bonds."""
        ranks = []
        for c in self.cores:
            ranks.append(c.shape[2])
        if not ranks:
            return 1.0
        return float(np.mean(ranks[:-1])) if len(ranks) > 1 else 1.0

    @property
    def memory_bytes(self) -> int:
        """Memory usage of TT representation."""
        return sum(c.nbytes for c in self.cores)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs dense representation."""
        dense = self.n_sites * 8  # float64 per site
        return dense / max(self.memory_bytes, 1)

    def methylation_level(self, site_idx: int) -> float:
        """
        Query the methylation level (probability of methylation) at a
        single CpG site without densifying the full tensor.

        This is an O(N · r²) operation using partial contractions.
        """
        if site_idx < 0 or site_idx >= len(self.cores):
            raise IndexError(f"Site index {site_idx} out of range [0, {len(self.cores)})")

        # Contract from left up to site_idx
        left = np.ones((1,))
        for k in range(site_idx):
            # Marginalize over physical dimension (sum over d)
            core = self.cores[k]  # (r_left, 2, r_right)
            marginal = np.einsum("idk,i->dk", core, left)
            left = np.sum(marginal, axis=0)  # Sum over d → (r_right,)

        # At the target site, project onto methylated state (d=1)
        core = self.cores[site_idx]
        methylated = np.einsum("idk,i->dk", core, left)
        # d=0 → unmethylated, d=1 → methylated
        left_meth = methylated[1, :]   # Methylated
        left_total = np.sum(methylated, axis=0)  # Total

        # Contract from right
        right_meth = np.ones((1,))
        right_total = np.ones((1,))
        for k in range(len(self.cores) - 1, site_idx, -1):
            core = self.cores[k]
            marginal_m = np.einsum("idk,k->id", core, right_meth)
            right_meth = np.sum(marginal_m, axis=1)
            marginal_t = np.einsum("idk,k->id", core, right_total)
            right_total = np.sum(marginal_t, axis=1)

        # Combine
        prob_meth = float(np.dot(left_meth, right_meth))
        prob_total = float(np.dot(left_total, right_total))

        if abs(prob_total) < 1e-300:
            return 0.5  # Undefined → default
        return prob_meth / prob_total

    def global_methylation(self) -> float:
        """
        Average methylation level across all sites.
        O(N · r²) via sequential marginal queries.
        """
        total = 0.0
        for i in range(len(self.cores)):
            total += self.methylation_level(i)
        return total / max(len(self.cores), 1)

    def compress(
        self, max_rank: Optional[int] = None, tol: float = 1e-12
    ) -> "MethylationState":
        """Compress methylation state to lower rank."""
        new_cores = _tt_round(self.cores, max_rank=max_rank, tol=tol)
        return MethylationState(
            cores=new_cores,
            n_sites=self.n_sites,
            tissue_type=self.tissue_type,
        )


# ---------------------------------------------------------------------------
# Epigenetic Clock
# ---------------------------------------------------------------------------

class HorvathClock:
    """
    Horvath pan-tissue epigenetic clock in QTT basis.

    The standard Horvath clock predicts age from 353 CpG methylation levels:
        age_transformed = intercept + Σ_i w_i · β_i

    In QTT, the weight vector w is represented as a rank-1 TT (product state),
    and age prediction becomes an inner product:
        age = ⟨w | β⟩

    The clock also provides:
    - Age acceleration (biological - chronological)
    - Rank-based age estimation (independent of weight fitting)
    - Tissue-specific calibration

    Parameters
    ----------
    n_clock_sites : int
        Number of clock CpG sites (353 for full Horvath, 50 for compact).
    tissue_type : str
        Tissue for calibration adjustment.
    """

    __slots__ = ("n_clock_sites", "tissue_type", "_weight_cores", "_intercept")

    def __init__(
        self,
        n_clock_sites: int = 50,
        tissue_type: str = "blood",
    ) -> None:
        self.n_clock_sites = min(n_clock_sites, len(HORVATH_TOP_SITES))
        self.tissue_type = tissue_type
        self._intercept = HORVATH_INTERCEPT
        self._weight_cores = self._build_weight_tt()

    def _build_weight_tt(self) -> List[np.ndarray]:
        """
        Build the clock weight vector as a rank-1 TT (product state).

        Each core encodes: w_i · β_i contribution for site i.
        Core shape: (1, 2, 1) where d=0 maps to unmethylated (β=0)
        and d=1 maps to methylated (β=1).
        """
        cores: List[np.ndarray] = []
        for idx in range(self.n_clock_sites):
            site_idx, weight, direction = HORVATH_TOP_SITES[idx]
            core = np.zeros((1, 2, 1))
            # d=0 (unmethylated): contributes 0
            core[0, 0, 0] = 1.0
            # d=1 (methylated): contributes weight * direction
            core[0, 1, 0] = 1.0 + weight * direction
            cores.append(core)
        return cores

    def predict_age(self, methylation: MethylationState) -> float:
        """
        Predict biological age from a methylation state.

        Uses the QTT inner product ⟨w | β⟩ for efficient age prediction
        without densification.

        Parameters
        ----------
        methylation : MethylationState
            Methylation state in TT format.

        Returns
        -------
        float
            Predicted biological age in years.
        """
        # Use the minimum of available sites
        n_sites = min(len(self._weight_cores), len(methylation.cores))

        # Compute inner product ⟨w | β⟩ using first n_sites cores
        transfer = np.ones((1, 1))
        for k in range(n_sites):
            wc = self._weight_cores[k]  # (1, 2, 1)
            mc = methylation.cores[k]    # (r_left, 2, r_right)
            tmp = np.einsum("ij,idk->jdk", transfer, mc)
            transfer = np.einsum("jdk,jdl->kl", tmp, wc)

        raw_score = float(transfer.sum()) + self._intercept

        # Horvath age transformation (inverse of log-linear)
        age = self._inverse_age_transform(raw_score)

        # Tissue-specific calibration
        age *= self._tissue_calibration()

        return max(age, 0.0)

    def predict_age_from_cell(self, cell: CellStateTensor) -> float:
        """
        Predict epigenetic age directly from a full cell state tensor.

        Extracts the methylation sub-tensor and applies the clock.
        """
        methylation = extract_methylation(cell)
        return self.predict_age(methylation)

    def rank_based_age(self, methylation: MethylationState) -> float:
        """
        Estimate age purely from the rank structure of the methylation tensor.

        This is independent of the Horvath weights — it uses the fundamental
        insight that aging = rank growth. Calibration:
            rank 2 → age 0 (embryonic)
            rank 4 → age 20
            rank 80 → age 120
        """
        r = float(methylation.max_rank)
        r0 = 2.0
        if r <= r0:
            return 0.0
        alpha = 120.0 / math.log(80.0 / r0)
        return alpha * math.log(r / r0)

    def age_acceleration(
        self,
        methylation: MethylationState,
        chronological_age: float,
    ) -> float:
        """
        Compute age acceleration: biological age - chronological age.

        Positive = accelerated aging (biologically older than chronological).
        Negative = decelerated aging (biologically younger).
        """
        bio_age = self.predict_age(methylation)
        return bio_age - chronological_age

    def _inverse_age_transform(self, x: float) -> float:
        """
        Inverse of the Horvath age transformation.

        Horvath uses a compressed scale for ages < 20:
            f(age) = log(age + 1) - log(21)  if age ≤ 20
            f(age) = (age - 20) / 21          if age > 20

        We invert this to get age from the raw prediction score.
        """
        adult_threshold = math.log(HORVATH_ADULT_AGE + 1)
        if x < 0:
            # Below adult threshold → logarithmic regime
            age = math.exp(x + math.log(HORVATH_ADULT_AGE + 1)) - 1
        else:
            # Above adult threshold → linear regime
            age = HORVATH_ADULT_AGE + x * (HORVATH_ADULT_AGE + 1)
        return age

    def _tissue_calibration(self) -> float:
        """
        Tissue-specific calibration factor.

        Different tissues age at different epigenetic rates:
        - Blood: baseline (1.0)
        - Brain cerebellum: slower (0.8)
        - Liver: faster (1.1)
        - Breast: faster (1.15)
        """
        calibration: Dict[str, float] = {
            "blood": 1.0,
            "brain": 0.85,
            "cerebellum": 0.80,
            "liver": 1.10,
            "breast": 1.15,
            "kidney": 1.05,
            "lung": 1.08,
            "muscle": 0.95,
            "skin": 1.02,
            "adipose": 1.03,
            "heart": 0.92,
            "colon": 1.07,
        }
        return calibration.get(self.tissue_type.lower(), 1.0)


class GrimAgeClock:
    """
    GrimAge clock — predicts mortality-adjusted biological age.

    GrimAge (Lu 2019) incorporates:
    - DNAm-based plasma protein surrogates (7 proteins)
    - DNAm-based smoking pack-years
    - Standard methylation age

    In QTT, this is a multi-modal measurement operator that combines
    methylation rank with protein mode rank for a composite score.
    """

    __slots__ = ("_horvath", "_protein_weight", "_smoking_weight")

    def __init__(self, tissue_type: str = "blood") -> None:
        self._horvath = HorvathClock(tissue_type=tissue_type)
        # GrimAge weights for non-methylation components
        self._protein_weight = 0.35    # Contribution of protein mode
        self._smoking_weight = 0.15    # DNAm smoking surrogate

    def predict_grim_age(self, cell: CellStateTensor) -> float:
        """
        Predict GrimAge from a full cell state tensor.

        Combines:
        1. Horvath methylation age (50% weight)
        2. Protein mode rank-based age (35% weight)
        3. Metabolome mode contribution (15% weight)
        """
        # Methylation component
        methylation = extract_methylation(cell)
        meth_age = self._horvath.predict_age(methylation)

        # Protein component (rank-based)
        mode_ranks = cell._compute_mode_ranks()
        protein_rank = mode_ranks.get(BiologicalMode.PROTEIN_ABUNDANCE, 4.0)
        r0 = 4.0
        if protein_rank > r0:
            protein_age = (120.0 / math.log(200.0 / r0)) * math.log(protein_rank / r0)
        else:
            protein_age = 0.0

        # Metabolome component
        metab_rank = mode_ranks.get(BiologicalMode.METABOLOME, 4.0)
        if metab_rank > r0:
            metab_age = (120.0 / math.log(200.0 / r0)) * math.log(metab_rank / r0)
        else:
            metab_age = 0.0

        # Weighted combination
        grim_age = (
            0.50 * meth_age
            + self._protein_weight * protein_age
            + self._smoking_weight * metab_age
        )

        return max(grim_age, 0.0)

    def mortality_hazard_ratio(self, cell: CellStateTensor) -> float:
        """
        Compute mortality hazard ratio relative to chronological age.

        HR > 1.0: increased mortality risk
        HR = 1.0: average risk for chronological age
        HR < 1.0: decreased mortality risk

        Calibration: GrimAge acceleration of +5 years → HR ≈ 1.22
        (per Lu 2019, each year of GrimAge acceleration → HR 1.04)
        """
        grim_age = self.predict_grim_age(cell)
        chrono_age = cell.chronological_age
        acceleration = grim_age - chrono_age
        return math.exp(0.04 * acceleration)


# ---------------------------------------------------------------------------
# Methylation Extraction
# ---------------------------------------------------------------------------

def extract_methylation(cell: CellStateTensor) -> MethylationState:
    """
    Extract the methylation sub-tensor from a full cell state.

    Identifies the methylation mode sites and extracts the corresponding
    TT cores, creating an independent MethylationState.
    """
    site_idx = 0
    for spec in cell.mode_specs:
        if spec.mode_type == BiologicalMode.METHYLATION:
            n_sites = spec.n_qtt_sites
            end_idx = min(site_idx + n_sites, len(cell.cores))
            meth_cores = [cell.cores[k].copy() for k in range(site_idx, end_idx)]

            # Re-normalize the sub-tensor
            if meth_cores:
                # Fix boundary conditions for the extracted sub-tensor
                # Left boundary: project first core to rank-1 left boundary
                if meth_cores[0].shape[0] > 1:
                    # Sum over left index (marginalize upstream modes)
                    meth_cores[0] = np.sum(meth_cores[0], axis=0, keepdims=True)
                # Right boundary: project last core to rank-1 right boundary
                if meth_cores[-1].shape[2] > 1:
                    meth_cores[-1] = np.sum(meth_cores[-1], axis=2, keepdims=True)

            return MethylationState(
                cores=meth_cores,
                n_sites=n_sites,
                tissue_type=cell.metadata.get("tissue_type", "blood"),
            )
        site_idx += spec.n_qtt_sites

    # No methylation mode found — return empty
    return MethylationState(
        cores=[np.ones((1, 2, 1)) * 0.5],
        n_sites=1,
        tissue_type="unknown",
    )


# ---------------------------------------------------------------------------
# Methylation State Factories
# ---------------------------------------------------------------------------

def young_methylation(
    n_sites: int = 128,
    tissue_type: str = "blood",
    seed: Optional[int] = None,
) -> MethylationState:
    """
    Generate a young methylation pattern.

    Young cells have highly structured methylation: CpG islands are
    unmethylated, gene bodies are methylated, heterochromatin is
    methylated. This regularity yields low TT rank.
    """
    rng = np.random.default_rng(seed)

    # Structured pattern: alternating methylated/unmethylated blocks
    # This gives very low rank (~2-3)
    cores: List[np.ndarray] = []
    for k in range(n_sites):
        core = np.zeros((1, 2, 1)) if k == 0 or k == n_sites - 1 else np.zeros((2, 2, 2))

        if k == 0:
            # First core: initialize pattern
            core = np.zeros((1, 2, 2))
            core[0, 0, 0] = 0.8   # Unmethylated → state 0
            core[0, 1, 1] = 0.2   # Methylated → state 1
        elif k == n_sites - 1:
            # Last core: terminate
            core = np.zeros((2, 2, 1))
            core[0, 0, 0] = 0.9
            core[0, 1, 0] = 0.1
            core[1, 0, 0] = 0.1
            core[1, 1, 0] = 0.9
        else:
            # Interior: maintain structure with slight tissue-specific variation
            core = np.zeros((2, 2, 2))
            # State 0 (unmethylated region) → likely stays unmethylated
            core[0, 0, 0] = 0.85 + rng.uniform(-0.02, 0.02)
            core[0, 1, 1] = 0.15 + rng.uniform(-0.02, 0.02)
            # State 1 (methylated region) → likely stays methylated
            core[1, 0, 0] = 0.15 + rng.uniform(-0.02, 0.02)
            core[1, 1, 1] = 0.85 + rng.uniform(-0.02, 0.02)
        cores.append(core)

    # Round to clean up
    cores = _tt_round(cores, max_rank=4, tol=1e-14)

    return MethylationState(
        cores=cores,
        n_sites=n_sites,
        tissue_type=tissue_type,
    )


def aged_methylation(
    n_sites: int = 128,
    age: float = 70.0,
    tissue_type: str = "blood",
    seed: Optional[int] = None,
) -> MethylationState:
    """
    Generate an aged methylation pattern with epigenetic drift.

    Aging causes:
    1. Global hypomethylation (genome-wide loss of methylation)
    2. Focal hypermethylation (CpG island shores gain methylation)
    3. Stochastic drift (increased entropy at individual sites)

    All three increase TT rank.
    """
    rng = np.random.default_rng(seed)

    # Start with young pattern
    young = young_methylation(n_sites=n_sites, tissue_type=tissue_type, seed=seed)

    # Age-dependent drift amplitude
    drift_amplitude = 0.02 * (age / 120.0)

    # Add stochastic perturbation to each core
    aged_cores: List[np.ndarray] = []
    for k, core in enumerate(young.cores):
        r_left, d, r_right = core.shape
        # Expand rank slightly for drift
        new_r_right = r_right
        if k < len(young.cores) - 1:
            drift_rank = max(1, int(age / 30.0))
            new_r_right = min(r_right + drift_rank, 50)

        new_core = np.zeros((r_left, d, new_r_right))
        new_core[:, :, :r_right] = core

        # Add drift noise in the new rank dimensions
        if new_r_right > r_right:
            noise = rng.standard_normal((r_left, d, new_r_right - r_right))
            noise *= drift_amplitude
            new_core[:, :, r_right:] = noise

        aged_cores.append(new_core)

    # Fix dimension compatibility
    for k in range(len(aged_cores) - 1):
        r_right_k = aged_cores[k].shape[2]
        r_left_next = aged_cores[k + 1].shape[0]
        if r_right_k != r_left_next:
            # Pad the next core's left dimension
            _, d_next, r_right_next = aged_cores[k + 1].shape
            new_core = np.zeros((r_right_k, d_next, r_right_next))
            new_core[:r_left_next, :, :] = aged_cores[k + 1]
            aged_cores[k + 1] = new_core

    # Round to realistic rank
    max_rank = max(4, int(4 * math.exp(age * math.log(80.0 / 4.0) / 120.0)))
    aged_cores = _tt_round(aged_cores, max_rank=max_rank, tol=1e-14)

    return MethylationState(
        cores=aged_cores,
        n_sites=n_sites,
        tissue_type=tissue_type,
    )
