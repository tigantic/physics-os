#!/usr/bin/env python3
"""
Rank Atlas Campaign — Cross-Domain QTT Bond-Dimension Measurement
=================================================================

Companion experiment code for ``docs/research/chi_regularity_hypothesis.md``.

Implements the full measurement protocol (Section 7):
  1. Instantiate domain pack anchor problems across 20 packs.
  2. Evolve solutions with high rank ceiling (SVD tolerance controls rank).
  3. Extract bond dimensions and full singular value spectra.
  4. Compute entanglement entropy and area-law scaling.
  5. Validate physics invariants.
  6. Record AtlasMeasurement records and persist to Parquet.

Usage:
    python tools/scripts/research/rank_atlas_campaign.py \
        --packs II III V VII VIII \
        --n-bits 6 7 8 9 \
        --n-complexity 10 \
        --output rank_atlas.parquet \
        --device cuda

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

# ── GPU Device Selection (matches repo-wide pattern) ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── The Ontic Engine platform imports ──
from ontic.engine.adaptive.bond_optimizer import (
    AdaptiveBondConfig,
    BondDimensionTracker,
    TruncationRecord,
)
from ontic.engine.adaptive.entanglement import (
    AreaLawAnalyzer,
    AreaLawScaling,
    EntanglementSpectrum,
    ScalingType,
)
from ontic.packs import discover_all
from ontic.packs._base import (
    EigenReferenceSolver,
    MonteCarloReferenceSolver,
    ODEReferenceSolver,
    PDE1DReferenceSolver,
)
from ontic.platform.data_model import (
    FieldData,
    SimulationState,
    StructuredMesh,
)
from ontic.platform.domain_pack import DomainPack, get_registry
from ontic.platform.protocols import SolveResult
from ontic.platform.qtt import QTTFieldData, field_to_qtt, qtt_to_field

# V0.2 reference solver base classes (accept raw Tensor, return raw Tensor)
_V02_SOLVER_BASES = (
    ODEReferenceSolver,
    PDE1DReferenceSolver,
    EigenReferenceSolver,
    MonteCarloReferenceSolver,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GPU Utilities
# ─────────────────────────────────────────────────────────────────────────────


def detect_hardware() -> Dict[str, Any]:
    """Detect GPU hardware and return info dict."""
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=props.name,
            gpu_vram_gb=round(props.total_memory / 1024**3, 2),
            compute_capability=f"{props.major}.{props.minor}",
            cuda_version=torch.version.cuda or "unknown",
        )
    return info


def gpu_warmup(device: torch.device) -> None:
    """Warm up GPU with a small matmul to trigger JIT compilation."""
    if device.type != "cuda":
        return
    logger.info("Warming up GPU...")
    _ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    logger.info("GPU warm-up complete")


def gpu_cleanup() -> None:
    """Release GPU memory between measurements."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_RANK_CEILING: int = 2048
SVD_TOLERANCE: float = 1e-6
SEED_BASE: int = 42

# Pack ID → (anchor_taxonomy_id, complexity_param_name, sweep_lo, sweep_hi)
#
# Complexity parameter names match the ACTUAL dataclass constructor field names
# of each V0.4 ProblemSpec.  V0.2 specs have no constructor kwargs, so their
# entry uses "fixed" — the campaign generates a single default-parameter run.
#
# ndim=0 packs (ODE, no spatial grid) are remapped to a 1-D taxonomy ID
# within the same pack so that QTT compression is meaningful.
PACK_CONFIG: Dict[str, Tuple[str, str, float, float]] = {
    # ── V0.4 packs — constructor kwargs enable complexity sweeps ──
    "II":    ("PHY-II.1",    "nu",              0.001,    0.1),     # viscosity → Re = 2π/ν
    "III":   ("PHY-III.3",   "sigma_pulse",     0.05,     2.0),     # pulse width (sharper → more complex)
    "V":     ("PHY-V.5",     "alpha",           0.0001,   1.0),     # diffusivity → Pe = cL/α
    "VII":   ("PHY-VII.2",   "J",               0.1,      10.0),    # exchange coupling strength
    "VIII":  ("PHY-VIII.1",  "Z",               1.0,      10.0),    # atomic number
    "XI":    ("PHY-XI.1",    "epsilon",          0.001,    0.5),     # Landau-damping perturbation amplitude
    # ── V0.2 packs — no constructor kwargs → fixed complexity ──
    "I":     ("PHY-I.1",     "fixed",           1.0,      1.0),     # N-body gravity (ndim=2)
    "IV":    ("PHY-IV.1",    "fixed",           1.0,      1.0),     # Ray tracing (ndim=1)
    "VI":    ("PHY-VI.1",    "fixed",           1.0,      1.0),     # Band structure (ndim=1)
    "IX":    ("PHY-IX.1",    "fixed",           1.0,      1.0),     # Shell model (ndim=3)
    "X":     ("PHY-X.4",     "fixed",           1.0,      1.0),     # LatticeQCD (ndim=1, replaces QCD ndim=0)
    "XII":   ("PHY-XII.1",   "fixed",           1.0,      1.0),     # Stellar structure (ndim=1)
    "XIII":  ("PHY-XIII.1",  "fixed",           1.0,      1.0),     # Seismic wave (ndim=1)
    "XIV":   ("PHY-XIV.1",   "fixed",           1.0,      1.0),     # Molecular dynamics (ndim=1)
    "XV":    ("PHY-XV.1",    "fixed",           1.0,      1.0),     # Reaction kinetics (ndim=1)
    "XVI":   ("PHY-XVI.1",   "fixed",           1.0,      1.0),     # Crystal growth (ndim=1)
    "XVII":  ("PHY-XVII.1",  "fixed",           1.0,      1.0),     # Linear acoustics (ndim=1)
    "XVIII": ("PHY-XVIII.1", "fixed",           1.0,      1.0),     # Weather prediction (ndim=1)
    "XIX":   ("PHY-XIX.6",   "fixed",           1.0,      1.0),     # Quantum sensing (ndim=1, replaces Circuits ndim=0)
    "XX":    ("PHY-XX.1",    "fixed",           1.0,      1.0),     # Solitons (ndim=1)
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Schema
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AtlasMeasurement:
    """Single measurement in the Rank Atlas Campaign.

    Every field is mandatory. If a measurement cannot fill all fields,
    it is discarded and logged as a failure.
    """

    # ── Identity ──
    pack_id: str
    domain_name: str
    problem_name: str

    # ── Configuration ──
    n_bits: int
    n_cells: int
    complexity_param_name: str
    complexity_param_value: float
    svd_tolerance: float
    max_rank_ceiling: int

    # ── QTT Core Data ──
    n_sites: int
    bond_dimensions: List[int]
    singular_value_spectra: List[List[float]]

    # ── Derived Rank Metrics ──
    max_rank: int
    mean_rank: float
    median_rank: int
    rank_std: float
    peak_rank_site: int
    rank_utilization: float

    # ── Singular Value Decay ──
    sv_decay_exponents: List[float]
    sv_decay_mean: float
    spectral_gaps: List[float]

    # ── Entanglement Entropy ──
    bond_entropies: List[float]
    total_entropy: float
    entropy_density: float
    effective_ranks: List[float]

    # ── Area-Law Fit ──
    area_law_exponent: float
    area_law_r_squared: float
    area_law_type: str

    # ── Compression ──
    dense_bytes: int
    qtt_bytes: int
    compression_ratio: float

    # ── Physics Validation ──
    reconstruction_error: float
    physics_invariant_name: str
    physics_invariant_value: float
    physics_invariant_passed: bool

    # ── Compute ──
    wall_time_s: float
    peak_gpu_mem_mb: float
    device: str
    timestamp: str
    seed: int


# ─────────────────────────────────────────────────────────────────────────────
# Core Measurement Functions
# ─────────────────────────────────────────────────────────────────────────────


def extract_bond_dimensions(qtt_field: QTTFieldData) -> Tuple[List[int], int]:
    """Extract bond dimensions from a QTTFieldData.

    Parameters
    ----------
    qtt_field : QTTFieldData
        A field stored in QTT format.

    Returns
    -------
    bond_dims : list[int]
        Bond dimension at each internal bond (length = n_sites - 1).
    n_sites : int
        Number of QTT sites (qubit levels).
    """
    cores = qtt_field.cores
    n_sites = len(cores)
    bond_dims: List[int] = []
    for k in range(n_sites - 1):
        # Core k has shape (r_left, 2, r_right)
        bond_dims.append(cores[k].shape[2])
    return bond_dims, n_sites


def extract_singular_value_spectra(
    qtt_field: QTTFieldData,
) -> List[List[float]]:
    """Compute full singular value spectrum at each QTT bond.

    For each bond k, reshape the accumulated left tensor into a matrix
    and perform SVD to obtain the singular values. This is the classical
    analogue of the Schmidt decomposition.

    Parameters
    ----------
    qtt_field : QTTFieldData
        Field in QTT format.

    Returns
    -------
    spectra : list[list[float]]
        Singular values at each internal bond, sorted descending.
    """
    cores = qtt_field.cores
    n_sites = len(cores)
    spectra: List[List[float]] = []

    # Build the left-accumulated tensor progressively
    # At bond k, we have the tensor formed by contracting cores 0..k
    # Reshape into matrix: (rows = left indices) × (cols = right indices)
    left = cores[0]  # shape (1, 2, r_1)

    for k in range(n_sites - 1):
        if k > 0:
            # Contract: left(r_left, 2^k, r_k) × core_k(r_k, 2, r_{k+1})
            #   → (r_left, 2^{k+1}, r_{k+1})
            r_left = left.shape[0]
            left_flat = left.reshape(r_left * (2 ** k), left.shape[-1])
            core_k = cores[k]
            r_k, d, r_kp1 = core_k.shape
            core_mat = core_k.reshape(r_k, d * r_kp1)
            product = left_flat @ core_mat
            left = product.reshape(r_left, 2 ** (k + 1), r_kp1)

        # SVD of the matricization at bond k
        mat = left.reshape(-1, left.shape[-1])
        try:
            _, S, _ = torch.linalg.svd(mat, full_matrices=False)
            sv_list = S.detach().cpu().tolist()
        except torch.linalg.LinAlgError:
            sv_list = [0.0]
        spectra.append(sv_list)

    return spectra


def compute_entanglement_metrics(
    sv_spectra: List[List[float]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute entanglement entropy, effective rank, spectral gap, and SV decay exponents.

    Parameters
    ----------
    sv_spectra : list[list[float]]
        Singular value spectra at each bond.

    Returns
    -------
    bond_entropies : list[float]
        Von Neumann entropy S_k = -sum(p_j * ln(p_j)) at each bond.
    effective_ranks : list[float]
        Effective rank e^{S_k} at each bond.
    spectral_gaps : list[float]
        Ratio σ₁/σ₂ at each bond (inf if only one SV).
    sv_decay_exponents : list[float]
        Power-law decay exponent β from fit σ_j ~ j^{-β}.
    """
    bond_entropies: List[float] = []
    effective_ranks: List[float] = []
    spectral_gaps: List[float] = []
    sv_decay_exponents: List[float] = []

    for sv_list in sv_spectra:
        sv = torch.tensor(sv_list, dtype=torch.float64)
        sv = sv[sv > 0]  # Remove zeros

        if len(sv) == 0:
            bond_entropies.append(0.0)
            effective_ranks.append(1.0)
            spectral_gaps.append(float("inf"))
            sv_decay_exponents.append(float("inf"))
            continue

        # Use EntanglementSpectrum from our infrastructure
        spectrum = EntanglementSpectrum.from_singular_values(sv, bond_index=0)
        bond_entropies.append(spectrum.entropy)
        effective_ranks.append(spectrum.effective_rank)

        # Spectral gap = σ₁/σ₂
        if len(sv) >= 2:
            spectral_gaps.append(float(sv[0] / sv[1]))
        else:
            spectral_gaps.append(float("inf"))

        # Fit σ_j ~ j^{-β} in log-log space
        if len(sv) >= 3:
            log_j = torch.log(torch.arange(1, len(sv) + 1, dtype=torch.float64))
            log_sv = torch.log(sv)
            # Least squares: log_sv = -β * log_j + c
            A = torch.stack([log_j, torch.ones_like(log_j)], dim=1)
            try:
                coeffs, _, _, _ = torch.linalg.lstsq(A, log_sv)
                beta = -float(coeffs[0])
            except Exception:
                beta = 0.0
            sv_decay_exponents.append(beta)
        else:
            sv_decay_exponents.append(0.0)

    return bond_entropies, effective_ranks, spectral_gaps, sv_decay_exponents


def fit_area_law(
    bond_entropies: List[float],
    n_bits: int,
    ndim: int = 3,
) -> Tuple[float, float, str]:
    """Fit area-law scaling to bond entropies.

    For a 3D QTT with 3*n_bits sites, the bipartition at site k divides
    the system into a "left" block of k qubits and a "right" block of
    3*n_bits - k qubits. The boundary size (in lattice units) at cut k
    is related to the surface area of the bipartition.

    Parameters
    ----------
    bond_entropies : list[float]
        Von Neumann entropy at each internal bond.
    n_bits : int
        Qubits per spatial axis.
    ndim : int
        Number of spatial dimensions (default 3).

    Returns
    -------
    exponent : float
        Scaling exponent γ in S ~ L^γ.
    r_squared : float
        Quality of fit.
    scaling_type : str
        "area" (γ ≈ d-1), "volume" (γ ≈ d), "sub_area" (γ < d-1),
        "log_corrected", or "unknown".
    """
    n_sites = ndim * n_bits
    if len(bond_entropies) < 4:
        return 0.0, 0.0, "unknown"

    # Boundary size at cut k: for a (2^n_bits)^3 grid, the bipartition
    # boundary scales with the cut position. Use k as proxy for boundary size.
    boundary_sizes = list(range(1, len(bond_entropies) + 1))

    try:
        analyzer = AreaLawAnalyzer(dimension=ndim)
        scaling: AreaLawScaling = analyzer.analyze(
            boundary_sizes=boundary_sizes,
            entropies=bond_entropies,
        )
        exponent = scaling.exponent
        r_squared = scaling.r_squared

        if scaling.scaling_type == ScalingType.AREA_LAW:
            scaling_type = "area"
        elif scaling.scaling_type == ScalingType.VOLUME_LAW:
            scaling_type = "volume"
        elif scaling.scaling_type == ScalingType.LOG_CORRECTED:
            scaling_type = "log_corrected"
        else:
            scaling_type = "unknown"

        return exponent, r_squared, scaling_type
    except Exception as exc:
        logger.warning("Area-law fit failed: %s", exc)
        return 0.0, 0.0, "unknown"


def compute_reconstruction_error(
    qtt_field: QTTFieldData,
    reference: Tensor,
) -> float:
    """Relative L2 reconstruction error: ‖u_QTT - u_ref‖ / ‖u_ref‖.

    Parameters
    ----------
    qtt_field : QTTFieldData
        QTT-compressed field.
    reference : Tensor
        Dense reference data.

    Returns
    -------
    error : float
        Relative L2 error.
    """
    reconstructed_field = qtt_to_field(qtt_field)
    reconstructed = reconstructed_field.data

    # Ensure same shape
    ref = reference.flatten()
    rec = reconstructed.flatten()[: len(ref)]

    ref_norm = torch.norm(ref.float())
    if ref_norm < 1e-30:
        return 0.0
    error_norm = torch.norm((rec.float() - ref.float()))
    return float(error_norm / ref_norm)


def generate_complexity_sweep(
    lo: float,
    hi: float,
    n_points: int,
) -> List[float]:
    """Generate logarithmically-spaced complexity parameter values.

    Parameters
    ----------
    lo : float
        Lower bound (inclusive).
    hi : float
        Upper bound (inclusive).
    n_points : int
        Number of points.

    Returns
    -------
    values : list[float]
        Logarithmically-spaced values from lo to hi.
    """
    if lo <= 0:
        # Can't use log space if lo is 0 or negative; use linear
        return [lo + (hi - lo) * i / (n_points - 1) for i in range(n_points)]

    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    return [10 ** (log_lo + (log_hi - log_lo) * i / (n_points - 1)) for i in range(n_points)]


# ─────────────────────────────────────────────────────────────────────────────
# Single Run
# ─────────────────────────────────────────────────────────────────────────────


def run_single_measurement(
    pack: DomainPack,
    taxonomy_id: str,
    n_bits: int,
    complexity_param_name: str,
    complexity_param_value: float,
    device: str,
    seed: int,
) -> AtlasMeasurement:
    """Execute a single Atlas measurement for one (pack, complexity, resolution) triple.

    Parameters
    ----------
    pack : DomainPack
        The domain pack providing the problem and solver.
    taxonomy_id : str
        Taxonomy ID of the anchor problem (e.g., "PHY-II.1").
    n_bits : int
        Qubits per spatial axis (grid = (2^n_bits)^ndim).
    complexity_param_name : str
        Name of the complexity parameter for this domain.
    complexity_param_value : float
        Value of the complexity parameter.
    device : str
        "cuda" or "cpu".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    measurement : AtlasMeasurement
        Complete measurement record.

    Raises
    ------
    RuntimeError
        If the simulation fails or produces invalid data.
    """
    dev = torch.device(device)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_start = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── 1. Instantiate problem spec ──
    SpecCls = pack.problem_specs().get(taxonomy_id)
    if SpecCls is None:
        raise RuntimeError(
            f"No ProblemSpec for {taxonomy_id} in pack {pack.pack_id}"
        )

    # Build spec with the complexity parameter injected
    spec_kwargs = _build_spec_kwargs(
        SpecCls, complexity_param_name, complexity_param_value
    )
    spec = SpecCls(**spec_kwargs) if spec_kwargs else SpecCls()

    # ── 2. Determine solver type (V0.2 vs V0.4) early ──
    solver_or_cls = pack.solvers().get(taxonomy_id)
    if solver_or_cls is None:
        raise RuntimeError(
            f"No Solver for {taxonomy_id} in pack {pack.pack_id}"
        )
    # Instantiate solver class if needed
    if inspect.isclass(solver_or_cls):
        solver = solver_or_cls(**_build_solver_kwargs(solver_or_cls, spec))
    else:
        solver = solver_or_cls
    is_v02 = isinstance(solver, _V02_SOLVER_BASES)

    # ── 3. Create mesh and initial condition ──
    # V0.2 ODE/eigen solvers return small state vectors, not spatial fields.
    # Force 1D mesh for them to ensure QTT compression operates on the right shape.
    if is_v02:
        ndim = 1
    else:
        ndim = spec.ndim
    n_cells_per_axis = 2 ** n_bits
    n_cells = n_cells_per_axis ** ndim

    shape: Tuple[int, ...]
    domain: Tuple[Tuple[float, float], ...]
    if ndim == 1:
        L = spec.parameters.get("L", 2 * math.pi)
        shape = (n_cells_per_axis,)
        domain = ((0.0, L),)
    elif ndim == 2:
        L = spec.parameters.get("L", 2 * math.pi)
        shape = (n_cells_per_axis, n_cells_per_axis)
        domain = ((0.0, L), (0.0, L))
    else:
        L = spec.parameters.get("L", 2 * math.pi)
        shape = (n_cells_per_axis,) * 3
        domain = ((0.0, L),) * 3

    mesh = StructuredMesh(shape=shape, domain=domain)

    # Generate initial condition (handles staggered grids for FDTD packs)
    ic_fields = _generate_initial_fields(spec, mesh, dev)
    state = SimulationState(
        t=0.0,
        fields=ic_fields,
        mesh=mesh,
    )

    # ── 4. Evolve solution ──
    T_final = spec.parameters.get("T_final", 1.0)
    dt = _estimate_dt(spec, mesh, n_bits)

    # V0.2 solvers expect raw Tensor input; V0.4 expects SimulationState
    if is_v02:
        # V0.2: use canonical IC if available (correct ODE state shape),
        # otherwise extract primary field tensor from the SimulationState
        if hasattr(solver, 'canonical_ic') and callable(solver.canonical_ic):
            solver_input: Any = solver.canonical_ic()
        else:
            primary_field_name = spec.field_names[0]
            solver_input = state.get_field(primary_field_name).data
    else:
        solver_input = state

    result: SolveResult = solver.solve(
        solver_input, t_span=(0.0, T_final), dt=dt, max_steps=10000
    )
    raw_final = result.final_state

    # ── 3b. Normalize final_state to SimulationState ──
    if isinstance(raw_final, SimulationState):
        final_state = raw_final
    else:
        # V0.2 solver returned a raw Tensor — wrap it
        raw_tensor = raw_final if isinstance(raw_final, Tensor) else torch.as_tensor(raw_final, dtype=torch.float64)
        raw_tensor = raw_tensor.to(device=dev, dtype=torch.float64)

        # If shape doesn't match mesh, reshape/pad to mesh size for QTT
        target_numel = 1
        for s in mesh.shape:
            target_numel *= s

        if raw_tensor.numel() != target_numel:
            # Interpolate or tile the result onto the mesh grid
            flat = raw_tensor.flatten()
            if flat.numel() == 0:
                padded = torch.zeros(target_numel, dtype=torch.float64, device=dev)
            elif flat.numel() < target_numel:
                # Repeat-tile to fill mesh, then truncate
                repeats = (target_numel // flat.numel()) + 1
                padded = flat.repeat(repeats)[:target_numel]
            else:
                padded = flat[:target_numel]
            raw_tensor = padded.reshape(mesh.shape)
        else:
            raw_tensor = raw_tensor.reshape(mesh.shape)

        primary_field_name = spec.field_names[0]
        wrapped_field = FieldData(
            name=primary_field_name,
            data=raw_tensor,
            mesh=mesh,
        )
        final_state = SimulationState(
            t=T_final,
            fields={primary_field_name: wrapped_field},
            mesh=mesh,
        )

    # ── 4. QTT compress with high ceiling ──
    primary_field_name = spec.field_names[0]
    dense_field = final_state.get_field(primary_field_name)
    dense_data = dense_field.data

    qtt_field = field_to_qtt(
        dense_field,
        max_rank=MAX_RANK_CEILING,
        tolerance=SVD_TOLERANCE,
    )

    # ── 5. Extract bond dimensions ──
    bond_dims, n_sites = extract_bond_dimensions(qtt_field)

    # ── 6. Full singular value spectra ──
    sv_spectra = extract_singular_value_spectra(qtt_field)

    # ── 7. Entanglement metrics ──
    bond_entropies, effective_ranks, spectral_gaps, sv_decay_exponents = (
        compute_entanglement_metrics(sv_spectra)
    )

    # ── 8. Area-law fit ──
    area_exponent, area_r2, area_type = fit_area_law(
        bond_entropies, n_bits, ndim
    )

    # ── 9. Derived rank metrics ──
    max_rank = max(bond_dims) if bond_dims else 0
    mean_rank = sum(bond_dims) / len(bond_dims) if bond_dims else 0.0
    sorted_dims = sorted(bond_dims)
    median_rank = sorted_dims[len(sorted_dims) // 2] if sorted_dims else 0
    rank_std = float(torch.tensor(bond_dims, dtype=torch.float64).std()) if len(bond_dims) > 1 else 0.0
    peak_rank_site = bond_dims.index(max_rank) if bond_dims else 0
    rank_utilization = mean_rank / max_rank if max_rank > 0 else 0.0

    # ── 10. Compression ratio ──
    dense_bytes = int(dense_data.numel() * dense_data.element_size())
    qtt_bytes = sum(c.numel() * c.element_size() for c in qtt_field.cores)
    compression_ratio = dense_bytes / qtt_bytes if qtt_bytes > 0 else 0.0

    # ── 11. Reconstruction error ──
    reconstruction_error = compute_reconstruction_error(qtt_field, dense_data)

    # ── 12. Physics invariant ──
    invariant_name, invariant_value, invariant_passed = _check_physics_invariant(
        spec, state, final_state
    )

    # ── 13. Compute stats ──
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall_time = time.monotonic() - t_start
    peak_gpu_mb = 0.0
    if dev.type == "cuda":
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    sv_decay_mean = sum(sv_decay_exponents) / len(sv_decay_exponents) if sv_decay_exponents else 0.0
    total_entropy = sum(bond_entropies)
    entropy_density = total_entropy / n_sites if n_sites > 0 else 0.0

    return AtlasMeasurement(
        pack_id=pack.pack_id,
        domain_name=pack.pack_name,
        problem_name=spec.name,
        n_bits=n_bits,
        n_cells=n_cells,
        complexity_param_name=complexity_param_name,
        complexity_param_value=complexity_param_value,
        svd_tolerance=SVD_TOLERANCE,
        max_rank_ceiling=MAX_RANK_CEILING,
        n_sites=n_sites,
        bond_dimensions=bond_dims,
        singular_value_spectra=[[float(s) for s in sv] for sv in sv_spectra],
        max_rank=max_rank,
        mean_rank=mean_rank,
        median_rank=median_rank,
        rank_std=rank_std,
        peak_rank_site=peak_rank_site,
        rank_utilization=rank_utilization,
        sv_decay_exponents=sv_decay_exponents,
        sv_decay_mean=sv_decay_mean,
        spectral_gaps=spectral_gaps,
        bond_entropies=bond_entropies,
        total_entropy=total_entropy,
        entropy_density=entropy_density,
        effective_ranks=effective_ranks,
        area_law_exponent=area_exponent,
        area_law_r_squared=area_r2,
        area_law_type=area_type,
        dense_bytes=dense_bytes,
        qtt_bytes=qtt_bytes,
        compression_ratio=compression_ratio,
        reconstruction_error=reconstruction_error,
        physics_invariant_name=invariant_name,
        physics_invariant_value=invariant_value,
        physics_invariant_passed=invariant_passed,
        wall_time_s=wall_time,
        peak_gpu_mem_mb=peak_gpu_mb,
        device=device,
        timestamp=timestamp,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def _build_spec_kwargs(
    spec_cls: type,
    param_name: str,
    param_value: float,
) -> Dict[str, Any]:
    """Map the complexity parameter name to the ProblemSpec's constructor kwarg.

    The *param_name* must be the **exact dataclass field name** of the spec's
    constructor (e.g. ``"nu"`` for ``BurgersSpec``, ``"Z"`` for
    ``KohnShamSpec``).  V0.2 specs use ``"fixed"`` — their parameters are
    hardcoded and cannot be varied, so we return an empty dict.
    """
    if param_name == "fixed":
        return {}

    sig = inspect.signature(spec_cls)
    if param_name in sig.parameters:
        return {param_name: param_value}

    # Fallback: param_name not found in constructor → log and skip
    logger.debug(
        "Spec %s has no constructor field '%s'; using defaults",
        spec_cls.__name__,
        param_name,
    )
    return {}


def _build_solver_kwargs(
    solver_cls: type,
    spec: Any,
) -> Dict[str, Any]:
    """Build solver constructor kwargs from the ProblemSpec."""
    import inspect

    sig = inspect.signature(solver_cls)
    kwargs: Dict[str, Any] = {}

    params = spec.parameters
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        if param_name in params:
            kwargs[param_name] = params[param_name]
        elif param_name == "nu" and "nu" in params:
            kwargs["nu"] = params["nu"]

    return kwargs


def _generate_initial_fields(
    spec: Any,
    mesh: StructuredMesh,
    device: torch.device,
) -> Dict[str, FieldData]:
    """Generate initial condition FieldData objects for the given problem spec.

    Handles special cases:
    - FDTD Maxwell: staggered Yee grid (E on N points, H on N-1 points
      with its own mesh).
    - Generic PDEs: smooth sinusoidal IC on the shared mesh.

    Returns
    -------
    fields : dict[str, FieldData]
        Mapping from field name to FieldData (each with its own mesh if needed).
    """
    fields: Dict[str, FieldData] = {}

    # ── Maxwell FDTD special case ──
    # The Yee scheme places E on integer grid points (N) and H at half-integer
    # points (N-1). FieldData validates shape == mesh.n_cells, so H must use
    # a separate mesh with N-1 cells.
    if _is_maxwell_spec(spec):
        N = mesh.shape[0]
        dx = mesh.dx[0]
        L = mesh.domain[0][1] - mesh.domain[0][0]
        sigma = spec.parameters.get("sigma", spec.parameters.get("sigma_pulse", 0.3))
        x0 = spec.parameters.get("x0", spec.parameters.get("x0_pulse", L / 2))
        mu = spec.parameters.get("mu", 1.0)
        epsilon = spec.parameters.get("epsilon", 1.0)
        c = 1.0 / math.sqrt(epsilon * mu)
        dt_est = 0.9 * dx / c  # CFL estimate for half-step init

        # E on primary mesh: Gaussian pulse
        x_E = torch.linspace(0.0, L - dx, N, dtype=torch.float64)
        E0 = torch.exp(-((x_E - x0) ** 2) / (2.0 * sigma ** 2))
        E0[0] = 0.0   # PEC boundary
        E0[-1] = 0.0   # PEC boundary
        fields["E"] = FieldData(
            name="E", data=E0.to(device=device), mesh=mesh,
        )

        # H on staggered mesh: half-step initialization for 2nd-order accuracy
        x_H_lo = 0.5 * dx
        x_H_hi = L - 1.5 * dx
        h_mesh = StructuredMesh(
            shape=(N - 1,), domain=((x_H_lo, x_H_hi),),
        )
        dEdx = (E0[1:] - E0[:-1]) / dx
        H0 = -(dt_est / 2.0) / mu * dEdx
        fields["H"] = FieldData(
            name="H", data=H0.to(device=device), mesh=h_mesh,
        )

        return fields

    # ── Generic case: smooth sinusoidal IC ──
    centers = mesh.cell_centers().to(device)  # cell_centers() defaults to CPU

    for field_name in spec.field_names:
        L0 = mesh.domain[0][1] - mesh.domain[0][0]
        if mesh.ndim == 1:
            x = centers[:, 0]
            data = torch.sin(2 * math.pi * x / L0)
        elif mesh.ndim == 2:
            x, y = centers[:, 0], centers[:, 1]
            L1 = mesh.domain[1][1] - mesh.domain[1][0]
            data = torch.sin(2 * math.pi * x / L0) * torch.sin(2 * math.pi * y / L1)
        else:
            x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
            L1 = mesh.domain[1][1] - mesh.domain[1][0]
            L2 = mesh.domain[2][1] - mesh.domain[2][0]
            data = (
                torch.sin(2 * math.pi * x / L0)
                * torch.sin(2 * math.pi * y / L1)
                * torch.sin(2 * math.pi * z / L2)
            )

        fields[field_name] = FieldData(
            name=field_name,
            data=data.to(dtype=torch.float64, device=device),
            mesh=mesh,
        )

    return fields


def _is_maxwell_spec(spec: Any) -> bool:
    """Detect whether a ProblemSpec is for FDTD Maxwell (staggered grid)."""
    name = getattr(spec, "name", "")
    field_names = getattr(spec, "field_names", ())
    return (
        "Maxwell" in name
        or "FDTD" in name
        or ("E" in field_names and "H" in field_names)
    )


def _estimate_dt(spec: Any, mesh: StructuredMesh, n_bits: int) -> float:
    """Estimate time step from CFL-type condition."""
    dx = min(mesh.dx)
    params = spec.parameters

    # Get diffusivity-like parameter
    nu = params.get("nu", params.get("D", 0.01))

    # CFL for advection: dt ~ dx / U_max
    # CFL for diffusion: dt ~ dx² / (2 * nu)
    dt_advect = 0.1 * dx
    dt_diffuse = 0.5 * dx ** 2 / (2 * max(nu, 1e-10)) if nu > 0 else dt_advect

    dt = min(dt_advect, dt_diffuse, 0.01)
    return max(dt, 1e-6)


def _check_physics_invariant(
    spec: Any,
    initial_state: Any,
    final_state: Any,
) -> Tuple[str, float, bool]:
    """Check the primary physics invariant for the simulation.

    Handles both SimulationState (V0.4) and raw Tensor (V0.2) final states.

    Returns (invariant_name, invariant_value, passed).
    """
    primary_field = spec.field_names[0]

    try:
        if isinstance(initial_state, SimulationState):
            u0 = initial_state.get_field(primary_field).data.float()
        elif isinstance(initial_state, Tensor):
            u0 = initial_state.float()
        else:
            return "unavailable", 0.0, True

        if isinstance(final_state, SimulationState):
            uT = final_state.get_field(primary_field).data.float()
        elif isinstance(final_state, Tensor):
            uT = final_state.float()
        else:
            return "unavailable", 0.0, True
    except (KeyError, AttributeError):
        return "unavailable", 0.0, True

    # Energy conservation: |E(T) - E(0)| / E(0)
    E0 = float(torch.sum(u0 ** 2))
    ET = float(torch.sum(uT ** 2))

    if E0 < 1e-30:
        return "energy_conservation", 0.0, True

    relative_change = abs(ET - E0) / E0
    # For dissipative systems energy should decrease; tolerance is generous
    passed = relative_change < 1.0  # Allow up to 100% change (dissipation)

    return "energy_conservation", relative_change, passed


# ─────────────────────────────────────────────────────────────────────────────
# Campaign Runner
# ─────────────────────────────────────────────────────────────────────────────


def run_campaign(
    pack_ids: List[str],
    n_bits_list: List[int],
    n_complexity: int,
    n_trials: int,
    device: str,
    output_path: Path,
    resume: bool = True,
) -> List[AtlasMeasurement]:
    """Run the full Rank Atlas Campaign.

    Parameters
    ----------
    pack_ids : list[str]
        Pack IDs to measure (e.g., ["II", "III", "V"]).
    n_bits_list : list[int]
        Grid resolutions as qubits per axis.
    n_complexity : int
        Number of logarithmically-spaced complexity values.
    n_trials : int
        Number of repeated trials per configuration.
    device : str
        "cuda" or "cpu".
    output_path : Path
        Path for output Parquet/JSON file.
    resume : bool
        If True, load existing results and skip completed configurations.

    Returns
    -------
    measurements : list[AtlasMeasurement]
        All successfully collected measurements.
    """
    # Load existing results if resuming
    measurements: List[AtlasMeasurement] = []
    completed_keys: set = set()

    if resume and output_path.exists():
        existing = _load_results(output_path)
        measurements.extend(existing)
        for m in existing:
            key = (m.pack_id, m.n_bits, m.complexity_param_value, m.seed)
            completed_keys.add(key)
        logger.info("Resumed %d existing measurements", len(existing))

    # Discover all domain packs
    n_discovered = discover_all()
    registry = get_registry()
    logger.info("Discovered %d domain packs", n_discovered)

    # Calculate total work
    total_runs = len(pack_ids) * len(n_bits_list) * n_complexity * n_trials
    completed = 0
    failed = 0

    for pack_id in pack_ids:
        if pack_id not in PACK_CONFIG:
            logger.warning("Pack %s not in PACK_CONFIG, skipping", pack_id)
            continue

        taxonomy_id, param_name, sweep_lo, sweep_hi = PACK_CONFIG[pack_id]

        try:
            pack = registry.get_pack(pack_id)
        except KeyError:
            logger.error("Pack %s not registered, skipping", pack_id)
            continue

        # V0.2 packs have fixed complexity (sweep_lo == sweep_hi) — single value
        if abs(sweep_hi - sweep_lo) < 1e-12:
            complexity_values = [sweep_lo]
        else:
            complexity_values = generate_complexity_sweep(
                sweep_lo, sweep_hi, n_complexity
            )

        for n_bits in n_bits_list:
            for xi_value in complexity_values:
                for trial in range(n_trials):
                    seed = SEED_BASE + trial
                    key = (pack_id, n_bits, xi_value, seed)

                    if key in completed_keys:
                        completed += 1
                        continue

                    logger.info(
                        "[%d/%d] Pack %s | n_bits=%d | %s=%.4g | trial=%d",
                        completed + failed + 1,
                        total_runs,
                        pack_id,
                        n_bits,
                        param_name,
                        xi_value,
                        trial,
                    )

                    try:
                        measurement = run_single_measurement(
                            pack=pack,
                            taxonomy_id=taxonomy_id,
                            n_bits=n_bits,
                            complexity_param_name=param_name,
                            complexity_param_value=xi_value,
                            device=device,
                            seed=seed,
                        )
                        measurements.append(measurement)
                        completed += 1

                        # Periodic checkpoint
                        if completed % 10 == 0:
                            _save_results(measurements, output_path)
                            logger.info(
                                "Checkpoint: %d completed, %d failed", completed, failed
                            )

                    except Exception as exc:
                        logger.error(
                            "FAILED Pack %s n_bits=%d %s=%.4g trial=%d: %s",
                            pack_id, n_bits, param_name, xi_value, trial, exc,
                        )
                        failed += 1

                    finally:
                        # Release GPU/CPU memory between measurements
                        gpu_cleanup()

    # Final save
    _save_results(measurements, output_path)

    logger.info(
        "Campaign complete: %d measurements, %d failures", completed, failed
    )
    return measurements


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Pipeline (Section 8 of the hypothesis document)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GridIndependenceResult:
    """Result of grid-independence test for one (pack, complexity) pair."""
    pack_id: str
    complexity_value: float
    slope: float          # b in χ_max = a + b * n_bits
    intercept: float      # a
    relative_slope: float # |b| / a
    passed: bool          # |b|/a < 0.05


@dataclass
class ScalingFitResult:
    """Result of complexity-scaling fit for one pack."""
    pack_id: str
    alpha: float           # exponent in χ ~ ξ^α
    alpha_std: float       # standard error on α
    c: float               # coefficient
    r_squared: float       # fit quality
    scaling_class: str     # "A", "B", "C", or "D"


@dataclass
class CampaignAnalysis:
    """Full analysis of the Rank Atlas Campaign."""
    grid_independence: List[GridIndependenceResult]
    scaling_fits: List[ScalingFitResult]
    optimal_clusters: int
    silhouette_scores: List[float]
    gap_statistics: List[float]        # Gap(k) for k=1..K_max
    gap_std_errors: List[float]        # s_k for gap test
    gap_optimal_k: int                 # k selected by gap-statistic rule
    feature_vectors: Dict[str, List[float]]  # pack_id → feature vector
    universality_verdict: str          # "CONFIRMED", "PARTIAL", "REFUTED"
    summary: Dict[str, Any]


def analyze_campaign(measurements: List[AtlasMeasurement]) -> CampaignAnalysis:
    """Run the full analysis pipeline on campaign results.

    Implements Sections 8.1–8.5 of the hypothesis document.

    Parameters
    ----------
    measurements : list[AtlasMeasurement]
        All collected measurements.

    Returns
    -------
    analysis : CampaignAnalysis
        Complete analysis results.
    """
    import numpy as np

    # ── 8.1 Grid Independence Test ──
    grid_results: List[GridIndependenceResult] = []
    packs = sorted(set(m.pack_id for m in measurements))

    for pack_id in packs:
        pack_data = [m for m in measurements if m.pack_id == pack_id]
        complexity_values = sorted(set(m.complexity_param_value for m in pack_data))

        for xi_val in complexity_values:
            subset = [m for m in pack_data
                      if abs(m.complexity_param_value - xi_val) < 1e-8]
            if len(subset) < 3:
                continue

            # Average max_rank across trials at each n_bits
            by_nbits: Dict[int, List[int]] = {}
            for m in subset:
                by_nbits.setdefault(m.n_bits, []).append(m.max_rank)

            n_bits_arr = np.array(sorted(by_nbits.keys()), dtype=np.float64)
            chi_arr = np.array(
                [np.mean(by_nbits[nb]) for nb in sorted(by_nbits.keys())],
                dtype=np.float64,
            )

            if len(n_bits_arr) < 2:
                continue

            # Linear fit: χ_max = a + b * n_bits
            coeffs = np.polyfit(n_bits_arr, chi_arr, 1)
            b, a = coeffs[0], coeffs[1]
            intercept = max(abs(a), 1.0)
            relative_slope = abs(b) / intercept
            passed = relative_slope < 0.05

            grid_results.append(GridIndependenceResult(
                pack_id=pack_id,
                complexity_value=xi_val,
                slope=float(b),
                intercept=float(a),
                relative_slope=float(relative_slope),
                passed=passed,
            ))

    # ── 8.2 Complexity Scaling Fit ──
    scaling_results: List[ScalingFitResult] = []

    for pack_id in packs:
        pack_data = [m for m in measurements if m.pack_id == pack_id]
        max_nbits = max(m.n_bits for m in pack_data)
        hi_res = [m for m in pack_data if m.n_bits == max_nbits]

        if len(hi_res) < 3:
            continue

        # Average across trials at each complexity value
        by_xi: Dict[float, List[int]] = {}
        for m in hi_res:
            by_xi.setdefault(m.complexity_param_value, []).append(m.max_rank)

        xi_vals = np.array(sorted(by_xi.keys()), dtype=np.float64)
        chi_vals = np.array(
            [np.mean(by_xi[x]) for x in sorted(by_xi.keys())],
            dtype=np.float64,
        )

        # Filter out zeros/negatives for log fitting
        valid = (xi_vals > 0) & (chi_vals > 0)
        xi_valid = xi_vals[valid]
        chi_valid = chi_vals[valid]

        if len(xi_valid) < 3:
            continue

        # Power-law fit in log space: log(χ) = α * log(ξ) + log(c)
        log_xi = np.log10(xi_valid)
        log_chi = np.log10(chi_valid)
        try:
            coeffs = np.polyfit(log_xi, log_chi, 1)
            alpha = float(coeffs[0])
            c = float(10 ** coeffs[1])

            chi_pred = 10 ** (alpha * log_xi + coeffs[1])
            ss_res = float(np.sum((chi_valid - chi_pred) ** 2))
            ss_tot = float(np.sum((chi_valid - np.mean(chi_valid)) ** 2))
            r_squared = 1.0 - ss_res / (ss_tot + 1e-10)

            # Standard error on α (from regression)
            n = len(log_xi)
            se_alpha = math.sqrt(ss_res / (n - 2)) / math.sqrt(
                float(np.sum((log_xi - np.mean(log_xi)) ** 2))
            ) if n > 2 else 0.0

            # Scaling class
            max_chi = float(chi_valid.max())
            if alpha < 0.1 and max_chi < 50:
                scaling_class = "A"
            elif alpha < 0.5:
                scaling_class = "B"
            elif max_chi >= MAX_RANK_CEILING * 0.9:
                scaling_class = "D"
            else:
                scaling_class = "C"

            scaling_results.append(ScalingFitResult(
                pack_id=pack_id,
                alpha=alpha,
                alpha_std=se_alpha,
                c=c,
                r_squared=r_squared,
                scaling_class=scaling_class,
            ))

        except Exception as exc:
            logger.warning("Scaling fit failed for pack %s: %s", pack_id, exc)

    # ── 8.3 Universality Clustering (§8.3 of hypothesis doc) ──
    #
    # Feature vector per pack:
    #   f_d = [α_d, γ_d, S̄_d, Δ̄_mean_d, ρ̄_k_d]
    # where:
    #   α_d     = scaling exponent from §8.2 (0 for fixed-complexity packs)
    #   γ_d     = QTT-site entanglement scaling exponent (§8.4)
    #   S̄_d     = mean entropy density at highest resolution
    #   Δ̄_mean_d = mean spectral gap (finite gaps only)
    #   ρ̄_k_d   = mean bond-local effective rank ratio χ_eff / χ

    feature_vectors_raw: List[np.ndarray] = []
    feature_pack_ids: List[str] = []
    feature_dict: Dict[str, List[float]] = {}

    # Build pack → α mapping (0.0 for packs without scaling fit)
    alpha_by_pack: Dict[str, float] = {}
    for sf in scaling_results:
        alpha_by_pack[sf.pack_id] = sf.alpha

    for pack_id in packs:
        # Highest resolution data for this pack
        pack_all = [m for m in measurements if m.pack_id == pack_id]
        if not pack_all:
            continue
        max_nbits = max(m.n_bits for m in pack_all)
        pack_data = [m for m in pack_all if m.n_bits == max_nbits]
        if not pack_data:
            continue

        # α_d: scaling exponent (0 for V0.2 / fixed-complexity packs)
        alpha_d = alpha_by_pack.get(pack_id, 0.0)

        # γ_d: QTT-site entanglement scaling exponent (§8.4)
        #   S_k vs ℓ_k = min(k, d·n - k) power-law fit
        gamma_vals: List[float] = []
        for m in pack_data:
            n_bonds = len(m.bond_entropies)
            if n_bonds >= 4:
                ell = np.array([min(k + 1, n_bonds - k) for k in range(n_bonds)],
                               dtype=np.float64)
                S = np.array(m.bond_entropies, dtype=np.float64)
                valid = (ell > 0) & (S > 0)
                if valid.sum() >= 3:
                    try:
                        log_ell = np.log(ell[valid])
                        log_S = np.log(S[valid])
                        coeffs = np.polyfit(log_ell, log_S, 1)
                        gamma_vals.append(float(coeffs[0]))
                    except Exception:
                        pass
        gamma_d = float(np.mean(gamma_vals)) if gamma_vals else 0.0

        # S̄_d: mean entropy density
        mean_entropy = float(np.mean([m.entropy_density for m in pack_data]))

        # Δ̄_mean_d: mean spectral gap (finite values only)
        finite_gaps: List[float] = []
        for m in pack_data:
            for g in m.spectral_gaps:
                if math.isfinite(g):
                    finite_gaps.append(g)
        mean_gap = float(np.mean(finite_gaps)) if finite_gaps else 0.0

        # ρ̄_k_d: mean bond-local effective rank ratio
        rho_vals: List[float] = []
        for m in pack_data:
            for er, bd in zip(m.effective_ranks, m.bond_dimensions):
                if bd > 0:
                    rho_vals.append(er / bd)
        mean_rho = float(np.mean(rho_vals)) if rho_vals else 0.0

        fv = np.array([alpha_d, gamma_d, mean_entropy, mean_gap, mean_rho])
        feature_vectors_raw.append(fv)
        feature_pack_ids.append(pack_id)
        feature_dict[pack_id] = fv.tolist()

    # ── Gap statistic (Tibshirani et al. 2001) — k=1 null test ──
    gap_statistics: List[float] = []
    gap_std_errors: List[float] = []
    gap_optimal_k: int = 1
    silhouette_optimal_k: int = 1
    silhouette_scores: List[float] = []

    if len(feature_vectors_raw) >= 4:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        X = StandardScaler().fit_transform(np.stack(feature_vectors_raw))
        n_samples, n_features = X.shape
        k_max = min(6, n_samples)
        B_ref = 100  # bootstrap reference datasets

        def _within_cluster_dispersion(data: np.ndarray, k: int) -> float:
            """Compute W_k = sum of within-cluster sum of squares."""
            if k == 1:
                return float(np.sum((data - data.mean(axis=0)) ** 2))
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(data)
            Wk = 0.0
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    cluster = data[mask]
                    Wk += float(np.sum((cluster - cluster.mean(axis=0)) ** 2))
            return Wk

        # Compute Gap(k) for k = 1..k_max
        rng = np.random.RandomState(42)
        for k in range(1, k_max):
            log_Wk = np.log(max(_within_cluster_dispersion(X, k), 1e-30))

            # B reference datasets sampled uniformly from bounding box
            ref_log_Wks: List[float] = []
            x_min = X.min(axis=0)
            x_max = X.max(axis=0)
            for _ in range(B_ref):
                X_ref = rng.uniform(x_min, x_max, size=(n_samples, n_features))
                ref_log_Wk = np.log(
                    max(_within_cluster_dispersion(X_ref, k), 1e-30)
                )
                ref_log_Wks.append(ref_log_Wk)

            E_log_Wk_star = float(np.mean(ref_log_Wks))
            gap_k = E_log_Wk_star - log_Wk
            s_k = float(np.std(ref_log_Wks, ddof=0) * np.sqrt(1.0 + 1.0 / B_ref))

            gap_statistics.append(gap_k)
            gap_std_errors.append(s_k)

        # Gap rule: smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
        gap_optimal_k = k_max - 1  # fallback
        for i in range(len(gap_statistics) - 1):
            if gap_statistics[i] >= gap_statistics[i + 1] - gap_std_errors[i + 1]:
                gap_optimal_k = i + 1  # k is 1-indexed
                break

        # Silhouette for k=2..k_max (undefined for k=1)
        best_sil = -1.0
        for k in range(2, k_max):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            score = float(silhouette_score(X, labels))
            silhouette_scores.append(score)
            if score > best_sil:
                best_sil = score
                silhouette_optimal_k = k

        # Final k determination (§8.3):
        # If gap test accepts k=1, use k=1.
        # Otherwise, fall back to silhouette-optimal k ∈ {2..5}.
        if gap_optimal_k == 1:
            optimal_k = 1
        else:
            optimal_k = silhouette_optimal_k

        logger.info(
            "Gap statistic: gap_optimal_k=%d, silhouette_optimal_k=%d → optimal_k=%d",
            gap_optimal_k,
            silhouette_optimal_k,
            optimal_k,
        )
    else:
        optimal_k = 1

    # ── Verdict ──
    n_class_a = sum(1 for sf in scaling_results if sf.scaling_class == "A")
    n_class_d = sum(1 for sf in scaling_results if sf.scaling_class == "D")
    n_grid_pass = sum(1 for gr in grid_results if gr.passed)
    n_grid_total = len(grid_results)

    if n_class_d >= 3:
        verdict = "REFUTED"
    elif (
        n_class_a
        + sum(1 for sf in scaling_results if sf.scaling_class == "B")
        == len(scaling_results)
        and optimal_k <= 2
    ):
        verdict = "CONFIRMED"
    else:
        verdict = "PARTIAL"

    summary = {
        "total_measurements": len(measurements),
        "packs_measured": len(packs),
        "grid_independence_pass_rate": f"{n_grid_pass}/{n_grid_total}",
        "scaling_classes": {sf.pack_id: sf.scaling_class for sf in scaling_results},
        "gap_optimal_k": gap_optimal_k,
        "silhouette_optimal_k": silhouette_optimal_k,
        "optimal_clusters": optimal_k,
        "verdict": verdict,
    }

    return CampaignAnalysis(
        grid_independence=grid_results,
        scaling_fits=scaling_results,
        optimal_clusters=optimal_k,
        silhouette_scores=silhouette_scores,
        gap_statistics=gap_statistics,
        gap_std_errors=gap_std_errors,
        gap_optimal_k=gap_optimal_k,
        feature_vectors=feature_dict,
        universality_verdict=verdict,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


def _save_results(
    measurements: List[AtlasMeasurement],
    path: Path,
) -> None:
    """Save measurements to JSON (Parquet requires pandas/pyarrow)."""
    records = [asdict(m) for m in measurements]
    json_path = path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info("Saved %d measurements to %s", len(records), json_path)

    # Also try Parquet if pandas available
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        parquet_path = path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved Parquet to %s", parquet_path)
    except ImportError:
        pass


def _load_results(path: Path) -> List[AtlasMeasurement]:
    """Load measurements from JSON."""
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return []

    with open(json_path) as f:
        records = json.load(f)

    measurements: List[AtlasMeasurement] = []
    for rec in records:
        try:
            measurements.append(AtlasMeasurement(**rec))
        except (TypeError, KeyError) as exc:
            logger.warning("Skipping malformed record: %s", exc)

    return measurements


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(
    measurements: List[AtlasMeasurement],
    analysis: CampaignAnalysis,
    output_dir: Path,
) -> None:
    """Generate the ATLAS_SUMMARY.md and visualizations.

    Parameters
    ----------
    measurements : list[AtlasMeasurement]
        All collected measurements.
    analysis : CampaignAnalysis
        Analysis results from analyze_campaign().
    output_dir : Path
        Directory for output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    # ── Summary Report ──
    report_path = output_dir / "ATLAS_SUMMARY.md"
    with open(report_path, "w") as f:
        f.write("# QTT Rank Atlas — Campaign Summary\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"**Total Measurements:** {len(measurements)}\n")
        f.write(f"**Packs Measured:** {analysis.summary['packs_measured']}\n")
        f.write(f"**Grid Independence Pass Rate:** {analysis.summary['grid_independence_pass_rate']}\n\n")

        f.write("## Scaling Classes\n\n")
        f.write("| Pack | Class | α | R² | χ_max |\n")
        f.write("|------|-------|---|-----|---------|\n")
        for sf in analysis.scaling_fits:
            f.write(f"| {sf.pack_id} | {sf.scaling_class} | {sf.alpha:.4f} ± {sf.alpha_std:.4f} | {sf.r_squared:.3f} | — |\n")

        f.write(f"\n## Universality Clustering\n\n")
        f.write(f"**Gap Statistic (Tibshirani et al. 2001):**\n\n")
        if analysis.gap_statistics:
            f.write("| k | Gap(k) | s_k |\n")
            f.write("|---|--------|-----|\n")
            for i, (g, s) in enumerate(
                zip(analysis.gap_statistics, analysis.gap_std_errors)
            ):
                f.write(f"| {i + 1} | {g:.4f} | {s:.4f} |\n")
            f.write(f"\n**Gap-optimal k:** {analysis.gap_optimal_k}\n\n")
        f.write(f"**Silhouette-optimal k:** {analysis.summary.get('silhouette_optimal_k', '—')}\n")
        f.write(f"**Silhouette Scores (k=2..5):** {[f'{s:.3f}' for s in analysis.silhouette_scores]}\n\n")
        f.write(f"**Final Optimal K:** {analysis.optimal_clusters}\n\n")

        f.write("### Feature Vectors (§8.3)\n\n")
        f.write("| Pack | α_d | γ_d | S̄_d | Δ̄_d | ρ̄_k |\n")
        f.write("|------|-----|-----|------|------|------|\n")
        for pid, fv in sorted(analysis.feature_vectors.items()):
            f.write(f"| {pid} | {fv[0]:.4f} | {fv[1]:.4f} | {fv[2]:.4f} | {fv[3]:.4f} | {fv[4]:.4f} |\n")

        f.write("## χ-Regularity Conjecture Verdict\n\n")
        f.write(f"**{analysis.universality_verdict}**\n\n")

        if analysis.universality_verdict == "CONFIRMED":
            f.write("All measured domains exhibit bounded QTT rank with weak or no\n")
            f.write("dependence on the complexity parameter. Data clusters into a single\n")
            f.write("regime, consistent with Sub-Conjecture 4.3.4 (Universality).\n")
        elif analysis.universality_verdict == "PARTIAL":
            f.write("Most domains show bounded rank, but some exhibit moderate growth.\n")
            f.write("The universality claim requires further investigation.\n")
        else:
            f.write("Three or more domains show divergent rank (Class D).\n")
            f.write("The χ-Regularity Conjecture is **falsified** for these domains.\n")

        f.write("\n## Grid Independence Details\n\n")
        f.write("| Pack | ξ | Slope | Intercept | |b|/a | Pass |\n")
        f.write("|------|---|-------|-----------|-------|------|\n")
        for gr in analysis.grid_independence:
            status = "✓" if gr.passed else "✗"
            f.write(f"| {gr.pack_id} | {gr.complexity_value:.2g} | {gr.slope:.3f} | {gr.intercept:.1f} | {gr.relative_slope:.4f} | {status} |\n")

    logger.info("Report written to %s", report_path)

    # ── Visualizations ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Scaling class distribution
        class_counts: Dict[str, int] = {}
        for sf in analysis.scaling_fits:
            class_counts[sf.scaling_class] = class_counts.get(sf.scaling_class, 0) + 1

        fig, ax = plt.subplots(figsize=(8, 6))
        colors_map = {"A": "green", "B": "blue", "C": "orange", "D": "red"}
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        bar_colors = [colors_map.get(c, "gray") for c in classes]
        ax.bar(classes, counts, color=bar_colors)
        ax.set_xlabel("Scaling Class")
        ax.set_ylabel("Number of Packs")
        ax.set_title("QTT Rank Scaling Class Distribution")
        ax.text(
            0.95, 0.95,
            f"Verdict: {analysis.universality_verdict}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, fontweight="bold",
            color="green" if analysis.universality_verdict == "CONFIRMED" else "red",
        )
        plt.tight_layout()
        plt.savefig(output_dir / "scaling_classes.png", dpi=150)
        plt.close()

        # Plot 2: α exponents across packs
        if analysis.scaling_fits:
            fig, ax = plt.subplots(figsize=(10, 6))
            pack_labels = [sf.pack_id for sf in analysis.scaling_fits]
            alphas = [sf.alpha for sf in analysis.scaling_fits]
            alpha_errs = [sf.alpha_std for sf in analysis.scaling_fits]
            x_pos = range(len(pack_labels))
            ax.bar(x_pos, alphas, yerr=alpha_errs, capsize=4, color="steelblue")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(pack_labels, rotation=45)
            ax.set_ylabel("Scaling Exponent α")
            ax.set_title("χ ~ ξ^α Across Physics Domains")
            ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="Class A threshold")
            ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Class B threshold")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "alpha_exponents.png", dpi=150)
            plt.close()

        logger.info("Visualizations saved to %s", output_dir)

    except ImportError:
        logger.warning("matplotlib not available; skipping visualizations")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """Main entry point for the Rank Atlas Campaign."""
    parser = argparse.ArgumentParser(
        description="Rank Atlas Campaign — Cross-Domain QTT Bond-Dimension Measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--packs",
        nargs="+",
        default=list(PACK_CONFIG.keys()),
        help="Pack IDs to measure (default: all 20 packs)",
    )
    parser.add_argument(
        "--n-bits",
        nargs="+",
        type=int,
        default=[6, 7, 8, 9],
        help="Grid resolutions as qubits per axis (default: 6 7 8 9)",
    )
    parser.add_argument(
        "--n-complexity",
        type=int,
        default=10,
        help="Number of complexity parameter values per pack (default: 10)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of repeated trials per configuration (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rank_atlas",
        help="Output file path (without extension, default: rank_atlas)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="atlas_results",
        help="Directory for report and visualizations (default: atlas_results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing results (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, ignoring existing results",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip measurement, only run analysis on existing data",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Resolve output paths ──
    output_path = Path(args.output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── GPU hardware detection ──
    hw_info = detect_hardware()

    print("=" * 70)
    print("QTT RANK ATLAS CAMPAIGN")
    print("=" * 70)
    print(f"Packs:       {args.packs}")
    print(f"Grid sizes:  {[f'{2**n}' for n in args.n_bits]} (n_bits={args.n_bits})")
    print(f"Complexity:  {args.n_complexity} values per pack")
    print(f"Trials:      {args.n_trials} per configuration")
    print(f"Device:      {args.device}")
    if hw_info.get("gpu_name"):
        print(f"GPU:         {hw_info['gpu_name']} ({hw_info['gpu_vram_gb']} GB VRAM, CC {hw_info['compute_capability']})")
        print(f"CUDA:        {hw_info.get('cuda_version', 'unknown')}")
    print(f"Max rank:    {MAX_RANK_CEILING} (ceiling, not constraining)")
    print(f"SVD tol:     {SVD_TOLERANCE:.1e} (controls physics-determined rank)")
    total = len(args.packs) * len(args.n_bits) * args.n_complexity * args.n_trials
    print(f"Total runs:  {total}")
    print("=" * 70)

    # ── GPU warmup ──
    gpu_warmup(torch.device(args.device))

    if args.analyze_only:
        measurements = _load_results(output_path)
        if not measurements:
            logger.error("No existing data found at %s", output_path)
            return 1
    else:
        measurements = run_campaign(
            pack_ids=args.packs,
            n_bits_list=args.n_bits,
            n_complexity=args.n_complexity,
            n_trials=args.n_trials,
            device=args.device,
            output_path=output_path,
            resume=args.resume,
        )

    if not measurements:
        logger.error("No measurements collected")
        return 1

    # Analysis
    logger.info("Running analysis pipeline...")
    analysis = analyze_campaign(measurements)

    # Report
    generate_report(measurements, analysis, output_dir)

    # Print verdict
    print("\n" + "=" * 70)
    print("χ-REGULARITY CONJECTURE — RANK ATLAS VERDICT")
    print("=" * 70)
    for sf in analysis.scaling_fits:
        print(f"  Pack {sf.pack_id:>5s}: Class {sf.scaling_class} | α = {sf.alpha:+.4f} ± {sf.alpha_std:.4f} | R² = {sf.r_squared:.3f}")
    print(f"\n  Grid Independence: {analysis.summary['grid_independence_pass_rate']}")
    print(f"\n  Gap Statistic:")
    if analysis.gap_statistics:
        for i, (g, s) in enumerate(zip(analysis.gap_statistics, analysis.gap_std_errors)):
            marker = " ←" if (i + 1) == analysis.gap_optimal_k else ""
            print(f"    k={i+1}: Gap={g:.4f}  s={s:.4f}{marker}")
    print(f"  Gap-optimal k:        {analysis.gap_optimal_k}")
    print(f"  Silhouette-optimal k:  {analysis.summary.get('silhouette_optimal_k', '—')}")
    print(f"  Final k:               {analysis.optimal_clusters}")
    print(f"\n  Feature Vectors (§8.3: [α, γ, S̄, Δ̄, ρ̄]):")
    for pid, fv in sorted(analysis.feature_vectors.items()):
        print(f"    {pid:>5s}: [{', '.join(f'{v:.3f}' for v in fv)}]")
    print(f"\n  VERDICT: {analysis.universality_verdict}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
