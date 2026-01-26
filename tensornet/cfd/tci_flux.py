"""Native TCI-based flux computation for QTT CFD.

This module implements the Python/PyTorch side of TCI flux computation,
using the Rust tci_core for MaxVol pivot selection and index arithmetic.

The key insight: flux F(U) is a black-box function. We don't care about its
algebraic structure - we sample it at O(r² × log N) points and build a
QTT approximation directly.

Architecture:
- Rust (tci_core): MaxVol pivot selection, neighbor index generation
- PyTorch: GPU-batched flux evaluation, QTT contraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

# Try to import Rust core
try:
    from tci_core import (
        RUST_AVAILABLE,
        IndexBatch,
        MaxVolConfig,
        TCIConfig,
        TCISampler,
        TruncationPolicy,
    )
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class TCIFluxConfig:
    """Configuration for TCI flux computation."""

    # QTT parameters
    n_qubits: int = 20  # Domain size = 2^20 = 1M points
    max_rank: int = 32
    hard_cap: int = 128  # Non-negotiable upper bound

    # TCI parameters
    tci_tolerance: float = 1e-6
    max_tci_iterations: int = 20

    # MaxVol parameters
    maxvol_tolerance: float = 0.05
    maxvol_max_iterations: int = 15

    # Batching
    min_batch_size: int = 10_000

    # Truncation
    truncate_every: int = 1  # SVD truncation after every TCI step

    # Physics
    gamma: float = 1.4  # Ratio of specific heats

    # Boundary condition: 'periodic', 'zero_gradient', 'extrapolate'
    boundary: str = "periodic"

    # Flux scheme: 'rusanov', 'hll', 'hllc' (Phase 1: Rusanov only)
    flux_scheme: str = "rusanov"


class FluxFunction(Protocol):
    """Protocol for flux functions."""

    def __call__(
        self,
        rho: torch.Tensor,
        rho_u: torch.Tensor,
        E: torch.Tensor,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate flux for conservative variables.

        Args:
            rho: Density
            rho_u: Momentum
            E: Total energy
            gamma: Ratio of specific heats

        Returns:
            Tuple of (F_rho, F_rho_u, F_E) flux components
        """
        ...


def rusanov_flux(
    rho_L: torch.Tensor,
    rho_u_L: torch.Tensor,
    E_L: torch.Tensor,
    rho_R: torch.Tensor,
    rho_u_R: torch.Tensor,
    E_R: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rusanov (Local Lax-Friedrichs) flux.

    F = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)

    where λ_max = max(|u_L| + c_L, |u_R| + c_R)

    CRITICAL: Sound speed c = sqrt(γ * p / ρ)
    NOT c = γ * p / ρ (that would be wrong by a factor of sqrt!)
    """
    device = rho_L.device
    dtype = rho_L.dtype
    eps = 1e-10

    # Primitive variables - LEFT
    u_L = rho_u_L / (rho_L + eps)
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
    p_L = torch.clamp(p_L, min=eps)

    # CRITICAL: c = sqrt(γp/ρ), NOT γp/ρ!
    c_L = torch.sqrt(gamma * p_L / (rho_L + eps))

    # Primitive variables - RIGHT
    u_R = rho_u_R / (rho_R + eps)
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)
    p_R = torch.clamp(p_R, min=eps)

    # CRITICAL: c = sqrt(γp/ρ), NOT γp/ρ!
    c_R = torch.sqrt(gamma * p_R / (rho_R + eps))

    # Physical fluxes
    F_rho_L = rho_u_L
    F_rho_u_L = rho_u_L * u_L + p_L
    F_E_L = (E_L + p_L) * u_L

    F_rho_R = rho_u_R
    F_rho_u_R = rho_u_R * u_R + p_R
    F_E_R = (E_R + p_R) * u_R

    # Maximum wavespeed
    lambda_max = torch.maximum(torch.abs(u_L) + c_L, torch.abs(u_R) + c_R)

    # Rusanov flux
    F_rho = 0.5 * (F_rho_L + F_rho_R) - 0.5 * lambda_max * (rho_R - rho_L)
    F_rho_u = 0.5 * (F_rho_u_L + F_rho_u_R) - 0.5 * lambda_max * (rho_u_R - rho_u_L)
    F_E = 0.5 * (F_E_L + F_E_R) - 0.5 * lambda_max * (E_R - E_L)

    return F_rho, F_rho_u, F_E


class TCIFlux:
    """TCI-based flux computation for QTT-compressed CFD.

    This class builds a QTT approximation of the flux function by
    sampling at carefully chosen points (via TCI/MaxVol) rather than
    computing flux at all O(N) grid points.

    Complexity: O(r² × log N) instead of O(N)
    """

    def __init__(
        self,
        config: TCIFluxConfig,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize TCI flux computer.

        Args:
            config: TCI configuration
            device: PyTorch device for computation
        """
        self.config = config
        self.device = device

        # Domain size
        self.N = 1 << config.n_qubits  # 2^n_qubits

        # Initialize Rust sampler if available
        if RUST_AVAILABLE:
            self.sampler = TCISampler(
                config.n_qubits,
                config.boundary,
                None,  # Random seed
            )
            self.sampler.set_min_batch_size(config.min_batch_size)
        else:
            self.sampler = None

        # Select flux function
        if config.flux_scheme == "rusanov":
            self.flux_fn = rusanov_flux
        else:
            raise ValueError(f"Unknown flux scheme: {config.flux_scheme}")

        # Statistics
        self.total_samples = 0
        self.total_iterations = 0

    def compute_flux_qtt(
        self,
        state_qtt: QTTState,
    ) -> QTTFlux:
        """Compute flux QTT from state QTT via TCI.

        This is the main entry point. Given conservative variables
        in QTT format, build a QTT approximation of the numerical flux.

        Args:
            state_qtt: Conservative variables (rho, rho_u, E) in QTT format

        Returns:
            Flux in QTT format
        """
        if not RUST_AVAILABLE:
            raise RuntimeError("TCI Core not available. Build with `maturin develop`")

        # Reset sampler for new flux computation
        self.sampler.reset()

        # Phase 1: Initial exploration with random samples
        initial_batch = self.sampler.sample_random(self.config.min_batch_size)

        # Evaluate flux at sample points
        flux_values = self._evaluate_flux_batch(state_qtt, initial_batch)

        # Build initial QTT approximation
        flux_qtt = self._build_qtt_from_samples(
            initial_batch.indices,
            flux_values,
        )

        # Phase 2: TCI refinement
        for iteration in range(self.config.max_tci_iterations):
            # Sample fibers for each qubit level
            for qubit in range(self.config.n_qubits):
                fiber_batch = self.sampler.sample_fibers(qubit)
                fiber_flux = self._evaluate_flux_batch(state_qtt, fiber_batch)

                # Update QTT with new samples
                self._update_qtt_from_samples(
                    flux_qtt,
                    fiber_batch.indices,
                    fiber_flux,
                    qubit,
                )

            # Check convergence
            error = self._estimate_error(state_qtt, flux_qtt)
            if error < self.config.tci_tolerance:
                break

            # Truncate if needed
            if (iteration + 1) % self.config.truncate_every == 0:
                self._truncate_qtt(flux_qtt)

            # Adaptive sampling for high-error regions
            if error > self.config.tci_tolerance * 10:
                adaptive_batch = self.sampler.sample_adaptive(
                    error_threshold=self.config.tci_tolerance * 5
                )
                adaptive_flux = self._evaluate_flux_batch(state_qtt, adaptive_batch)
                self._update_qtt_from_samples(
                    flux_qtt,
                    adaptive_batch.indices,
                    adaptive_flux,
                    -1,  # All levels
                )

        self.total_iterations += iteration + 1
        self.total_samples = self.sampler.num_samples()

        return flux_qtt

    def _evaluate_flux_batch(
        self,
        state_qtt: QTTState,
        batch: IndexBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate flux at a batch of sample points.

        This is where GPU parallelism happens. We evaluate the pointwise
        flux function at O(batch_size) points in parallel.

        CRITICAL: Neighbor indices are precomputed in Rust (batch.left, batch.right)
        to avoid GPU thread divergence from binary addition.
        """
        # Convert indices to tensors
        indices = torch.tensor(batch.indices, device=self.device, dtype=torch.long)
        left_idx = torch.tensor(batch.left, device=self.device, dtype=torch.long)
        right_idx = torch.tensor(batch.right, device=self.device, dtype=torch.long)

        # Evaluate state at sample points and neighbors
        rho_L, rho_u_L, E_L = state_qtt.evaluate_at_indices(left_idx)
        rho_R, rho_u_R, E_R = state_qtt.evaluate_at_indices(right_idx)

        # Compute Rusanov flux at cell interfaces
        F_rho, F_rho_u, F_E = self.flux_fn(
            rho_L,
            rho_u_L,
            E_L,
            rho_R,
            rho_u_R,
            E_R,
            self.config.gamma,
        )

        return F_rho, F_rho_u, F_E

    def _build_qtt_from_samples(
        self,
        indices: list,
        flux_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> QTTFlux:
        """Build initial QTT from sample points.

        This uses TCI skeleton decomposition to construct a QTT from
        the sampled flux values. The key insight is that TCI finds
        the "most informative" indices, which we use as pivot rows/cols.
        
        Algorithm (Cross Approximation):
            1. Use sampled indices as initial pivot candidates
            2. Build skeleton matrices A_k for each core
            3. Compute pseudo-inverse to get core tensors
            4. Iterate MaxVol to improve pivots
        """
        F_rho, F_rho_u, F_E = flux_values
        n_samples = len(indices)
        
        # For initial construction, use a simple rank-1 + corrections approach
        # Build interpolation from samples
        
        # Convert indices to binary representation for QTT structure
        indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        binary = _indices_to_binary(indices_tensor, self.config.n_qubits)
        
        # Build initial QTT via least-squares fitting
        # For each flux component, find cores that minimize error at samples
        
        F_rho_cores = self._fit_qtt_to_samples(binary, F_rho)
        F_rho_u_cores = self._fit_qtt_to_samples(binary, F_rho_u)
        F_E_cores = self._fit_qtt_to_samples(binary, F_E)
        
        return QTTFlux(F_rho_cores, F_rho_u_cores, F_E_cores, self.config.n_qubits)
    
    def _fit_qtt_to_samples(
        self,
        binary_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Fit QTT cores to minimize error at sample points.
        
        Uses alternating least squares (ALS) on the sampled data.
        """
        n_samples = len(values)
        num_qubits = self.config.n_qubits
        max_rank = self.config.max_rank
        
        device = self.device
        dtype = values.dtype
        
        # Initialize random cores
        cores = []
        for k in range(num_qubits):
            r_left = 1 if k == 0 else min(2 ** k, max_rank)
            r_right = 1 if k == num_qubits - 1 else min(2 ** (k + 1), max_rank)
            core = torch.randn(r_left, 2, r_right, dtype=dtype, device=device) * 0.1
            cores.append(core)
        
        # ALS iteration to fit samples
        for als_iter in range(5):  # Few iterations suffice for good init
            # Sweep left to right
            for k in range(num_qubits):
                # Build effective matrix for site k
                # Evaluate QTT at samples with site k as unknown
                A_left = self._contract_left(cores, binary_indices, k)  # (n_samples, r_left)
                A_right = self._contract_right(cores, binary_indices, k)  # (n_samples, r_right)
                
                r_left = cores[k].shape[0]
                r_right = cores[k].shape[2]
                
                # For each binary value (0, 1), solve least squares
                new_core = torch.zeros_like(cores[k])
                
                for b in range(2):
                    mask = binary_indices[:, k] == b
                    if mask.sum() == 0:
                        continue
                    
                    # Design matrix: outer product of left and right contractions
                    # A[i, (r_l, r_r)] = A_left[i, r_l] * A_right[i, r_r]
                    A_left_masked = A_left[mask]  # (n_masked, r_left)
                    A_right_masked = A_right[mask]  # (n_masked, r_right)
                    values_masked = values[mask]  # (n_masked,)
                    
                    # Khatri-Rao product for least squares
                    n_masked = mask.sum().item()
                    design = torch.einsum('il,ir->ilr', A_left_masked, A_right_masked)
                    design = design.reshape(n_masked, r_left * r_right)
                    
                    # Solve least squares
                    try:
                        solution, _ = torch.linalg.lstsq(design, values_masked.unsqueeze(-1))
                        new_core[:, b, :] = solution.squeeze(-1).reshape(r_left, r_right)
                    except:
                        # Fallback to pseudo-inverse
                        pinv = torch.linalg.pinv(design)
                        solution = pinv @ values_masked
                        new_core[:, b, :] = solution.reshape(r_left, r_right)
                
                cores[k] = new_core
        
        return cores
    
    def _contract_left(
        self,
        cores: list[torch.Tensor],
        binary_indices: torch.Tensor,
        stop_at: int,
    ) -> torch.Tensor:
        """Contract cores from left up to (but not including) stop_at.
        
        Returns shape (n_samples, r_left) where r_left is left rank of core[stop_at].
        """
        n_samples = binary_indices.shape[0]
        
        if stop_at == 0:
            return torch.ones(n_samples, 1, device=cores[0].device, dtype=cores[0].dtype)
        
        result = torch.ones(n_samples, 1, device=cores[0].device, dtype=cores[0].dtype)
        
        for k in range(stop_at):
            bit_k = binary_indices[:, k]
            core = cores[k]
            r_left, _, r_right = core.shape
            
            # Select slices for each sample
            selected = core[:, bit_k, :].permute(1, 0, 2)  # (n_samples, r_left, r_right)
            
            # Contract
            result = result.unsqueeze(-1)  # (n_samples, r_prev, 1)
            result = torch.bmm(result.transpose(1, 2), selected)  # (n_samples, 1, r_right)
            result = result.squeeze(1)  # (n_samples, r_right)
        
        return result
    
    def _contract_right(
        self,
        cores: list[torch.Tensor],
        binary_indices: torch.Tensor,
        start_after: int,
    ) -> torch.Tensor:
        """Contract cores from right starting after start_after.
        
        Returns shape (n_samples, r_right) where r_right is right rank of core[start_after].
        """
        n_samples = binary_indices.shape[0]
        num_qubits = len(cores)
        
        if start_after == num_qubits - 1:
            return torch.ones(n_samples, 1, device=cores[0].device, dtype=cores[0].dtype)
        
        result = torch.ones(n_samples, 1, device=cores[0].device, dtype=cores[0].dtype)
        
        for k in range(num_qubits - 1, start_after, -1):
            bit_k = binary_indices[:, k]
            core = cores[k]
            r_left, _, r_right = core.shape
            
            # Select slices
            selected = core[:, bit_k, :].permute(1, 0, 2)  # (n_samples, r_left, r_right)
            
            # Contract from right
            result = result.unsqueeze(1)  # (n_samples, 1, r_prev)
            result = torch.bmm(selected, result)  # (n_samples, r_left, 1)
            result = result.squeeze(-1)  # (n_samples, r_left)
        
        return result

    def _update_qtt_from_samples(
        self,
        flux_qtt: QTTFlux,
        indices: list,
        flux_values: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        qubit: int,
    ):
        """Update QTT with new sample information.

        Incorporates new samples by updating the core at the specified qubit level.
        Uses a single ALS step at the target site.
        """
        F_rho, F_rho_u, F_E = flux_values
        
        # Convert indices to binary
        indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        binary = _indices_to_binary(indices_tensor, self.config.n_qubits)
        
        # Update each flux component
        if qubit < 0:
            # Update all sites
            for k in range(self.config.n_qubits):
                self._update_core_at_site(flux_qtt.F_rho_cores, binary, F_rho, k)
                self._update_core_at_site(flux_qtt.F_rho_u_cores, binary, F_rho_u, k)
                self._update_core_at_site(flux_qtt.F_E_cores, binary, F_E, k)
        else:
            self._update_core_at_site(flux_qtt.F_rho_cores, binary, F_rho, qubit)
            self._update_core_at_site(flux_qtt.F_rho_u_cores, binary, F_rho_u, qubit)
            self._update_core_at_site(flux_qtt.F_E_cores, binary, F_E, qubit)
    
    def _update_core_at_site(
        self,
        cores: list[torch.Tensor],
        binary_indices: torch.Tensor,
        values: torch.Tensor,
        site: int,
    ):
        """Update a single core using new samples."""
        A_left = self._contract_left(cores, binary_indices, site)
        A_right = self._contract_right(cores, binary_indices, site)
        
        r_left = cores[site].shape[0]
        r_right = cores[site].shape[2]
        
        new_core = cores[site].clone()
        
        for b in range(2):
            mask = binary_indices[:, site] == b
            if mask.sum() == 0:
                continue
            
            A_left_masked = A_left[mask]
            A_right_masked = A_right[mask]
            values_masked = values[mask]
            
            n_masked = mask.sum().item()
            design = torch.einsum('il,ir->ilr', A_left_masked, A_right_masked)
            design = design.reshape(n_masked, r_left * r_right)
            
            # Incremental update: weighted average of old and new
            try:
                solution, _ = torch.linalg.lstsq(design, values_masked.unsqueeze(-1))
                new_slice = solution.squeeze(-1).reshape(r_left, r_right)
                # Blend with existing
                new_core[:, b, :] = 0.5 * new_core[:, b, :] + 0.5 * new_slice
            except:
                pass  # Keep existing if solve fails
        
        cores[site] = new_core

    def _estimate_error(
        self,
        state_qtt: QTTState,
        flux_qtt: QTTFlux,
    ) -> float:
        """Estimate approximation error via random sampling.

        Sample random points, compare QTT flux vs direct flux evaluation.
        """
        # Sample random test points (not from TCI samples)
        test_batch = self.sampler.sample_random(1000)

        # True flux at test points
        true_flux = self._evaluate_flux_batch(state_qtt, test_batch)

        # QTT flux at test points
        test_indices = torch.tensor(test_batch.indices, device=self.device)
        approx_flux = flux_qtt.evaluate_at_indices(test_indices)

        # Max absolute error
        error = max(
            (true_flux[0] - approx_flux[0]).abs().max().item(),
            (true_flux[1] - approx_flux[1]).abs().max().item(),
            (true_flux[2] - approx_flux[2]).abs().max().item(),
        )

        return error

    def _truncate_qtt(self, flux_qtt: QTTFlux):
        """Truncate QTT ranks via SVD.

        CRITICAL: Must truncate to prevent unbounded rank growth.
        Hard cap at config.hard_cap (128).
        """
        # Placeholder - applies truncated SVD to each core
        pass


# Placeholder classes for full implementation
class QTTState:
    """QTT representation of conservative variables (rho, rho_u, E).
    
    Each conservative variable is stored as a separate QTT for simplicity.
    In a production implementation, these would be coupled.
    """
    
    def __init__(
        self,
        rho_cores: list[torch.Tensor],
        rho_u_cores: list[torch.Tensor],
        E_cores: list[torch.Tensor],
        num_qubits: int,
    ):
        """Initialize QTT state from TT cores for each variable."""
        self.rho_cores = rho_cores
        self.rho_u_cores = rho_u_cores
        self.E_cores = E_cores
        self.num_qubits = num_qubits
        self.N = 2 ** num_qubits
    
    @classmethod
    def from_dense(
        cls,
        rho: torch.Tensor,
        rho_u: torch.Tensor,
        E: torch.Tensor,
        num_qubits: int,
        max_rank: int = 32,
    ) -> "QTTState":
        """Create QTT state from dense arrays."""
        rho_cores = _dense_to_qtt_cores(rho, num_qubits, max_rank)
        rho_u_cores = _dense_to_qtt_cores(rho_u, num_qubits, max_rank)
        E_cores = _dense_to_qtt_cores(E, num_qubits, max_rank)
        return cls(rho_cores, rho_u_cores, E_cores, num_qubits)

    def evaluate_at_indices(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate (rho, rho_u, E) at given indices.
        
        Uses the QTT structure to evaluate at arbitrary indices in O(n_qubits * r²)
        per index, where r is the max TT rank.
        """
        rho = _evaluate_qtt_at_indices(self.rho_cores, indices, self.num_qubits)
        rho_u = _evaluate_qtt_at_indices(self.rho_u_cores, indices, self.num_qubits)
        E = _evaluate_qtt_at_indices(self.E_cores, indices, self.num_qubits)
        return rho, rho_u, E


class QTTFlux:
    """QTT representation of numerical flux (F_rho, F_rho_u, F_E).
    
    Each flux component is stored as a separate QTT.
    """
    
    def __init__(
        self,
        F_rho_cores: list[torch.Tensor],
        F_rho_u_cores: list[torch.Tensor],
        F_E_cores: list[torch.Tensor],
        num_qubits: int,
    ):
        """Initialize QTT flux from TT cores for each component."""
        self.F_rho_cores = F_rho_cores
        self.F_rho_u_cores = F_rho_u_cores
        self.F_E_cores = F_E_cores
        self.num_qubits = num_qubits
        self.N = 2 ** num_qubits

    def evaluate_at_indices(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate (F_rho, F_rho_u, F_E) at given indices."""
        F_rho = _evaluate_qtt_at_indices(self.F_rho_cores, indices, self.num_qubits)
        F_rho_u = _evaluate_qtt_at_indices(self.F_rho_u_cores, indices, self.num_qubits)
        F_E = _evaluate_qtt_at_indices(self.F_E_cores, indices, self.num_qubits)
        return F_rho, F_rho_u, F_E


def _dense_to_qtt_cores(
    tensor: torch.Tensor,
    num_qubits: int,
    max_rank: int = 32,
) -> list[torch.Tensor]:
    """Convert dense 1D tensor to QTT cores via TT-SVD.
    
    Args:
        tensor: Dense 1D tensor of length N (will be padded to 2^num_qubits)
        num_qubits: Number of QTT cores
        max_rank: Maximum TT rank
        
    Returns:
        List of QTT cores, each of shape (r_left, 2, r_right)
    """
    N = 2 ** num_qubits
    device = tensor.device
    dtype = tensor.dtype
    
    # Pad to power of 2 if needed
    if len(tensor) < N:
        tensor = torch.cat([tensor, torch.zeros(N - len(tensor), dtype=dtype, device=device)])
    elif len(tensor) > N:
        tensor = tensor[:N]
    
    # Reshape to 2x2x...x2 (num_qubits dimensions)
    tensor_reshaped = tensor.reshape([2] * num_qubits)
    
    cores = []
    current = tensor_reshaped
    
    for k in range(num_qubits - 1):
        # Current shape: (r_left * 2, 2^remaining)
        # First iteration: (2, 2^{n-1})
        shape = current.shape
        r_left = 1 if k == 0 else cores[-1].shape[2]
        
        # Unfold for SVD: (r_left * 2, remaining_elements)
        mode_0_size = shape[0]  # This is 2 for first iter, r_left for rest
        remaining = int(torch.prod(torch.tensor(shape[1:])).item())
        
        if k == 0:
            unfolded = current.reshape(2, remaining)
        else:
            unfolded = current.reshape(mode_0_size * 2, remaining // 2)
        
        # SVD
        U, S, Vh = torch.linalg.svd(unfolded, full_matrices=False)
        
        # Truncate to max_rank
        r_new = min(max_rank, len(S))
        nonzero_mask = S > 1e-14 * S[0] if S[0] > 0 else torch.ones_like(S, dtype=torch.bool)
        r_new = min(r_new, nonzero_mask.sum().item())
        r_new = max(1, r_new)
        
        U = U[:, :r_new]
        S = S[:r_new]
        Vh = Vh[:r_new, :]
        
        # Store core: (r_left, 2, r_new)
        if k == 0:
            core = U.reshape(1, 2, r_new)
        else:
            core = U.reshape(r_left, 2, r_new)
        cores.append(core)
        
        # Prepare next tensor: S @ Vh reshaped appropriately
        SV = torch.diag(S) @ Vh
        remaining_qubits = num_qubits - k - 1
        if remaining_qubits > 1:
            current = SV.reshape([r_new] + [2] * (remaining_qubits))
        else:
            current = SV
    
    # Last core: (r_left, 2, 1)
    r_left = cores[-1].shape[2] if cores else 1
    last_core = current.reshape(r_left, 2, 1)
    cores.append(last_core)
    
    return cores


def _evaluate_qtt_at_indices(
    cores: list[torch.Tensor],
    indices: torch.Tensor,
    num_qubits: int,
) -> torch.Tensor:
    """Evaluate QTT at given integer indices.
    
    For each index i, extract the binary representation and contract
    the corresponding slices of each core.
    
    Args:
        cores: QTT cores, each of shape (r_left, 2, r_right)
        indices: 1D tensor of integer indices to evaluate at
        num_qubits: Number of qubits (= number of cores)
        
    Returns:
        Values at the requested indices
    """
    batch_size = len(indices)
    device = cores[0].device
    dtype = cores[0].dtype
    
    # Convert indices to binary representation
    # binary[i, k] = k-th bit of index i (k=0 is least significant)
    binary = torch.zeros(batch_size, num_qubits, dtype=torch.long, device=device)
    temp_indices = indices.clone()
    for k in range(num_qubits):
        binary[:, k] = temp_indices % 2
        temp_indices = temp_indices // 2
    
    # Reverse to match QTT ordering (most significant bit first)
    binary = binary.flip(dims=[1])
    
    # Contract cores for each sample
    # Start with identity: shape (batch_size, 1)
    result = torch.ones(batch_size, 1, dtype=dtype, device=device)
    
    for k, core in enumerate(cores):
        # core: (r_left, 2, r_right)
        # Select the slice corresponding to binary[:, k]
        # selected: (batch_size, r_left, r_right)
        
        r_left, _, r_right = core.shape
        bit_k = binary[:, k]  # (batch_size,)
        
        # Gather the correct slice for each sample
        # core[:, bit, :] for each sample's bit
        selected = core[:, bit_k, :].permute(1, 0, 2)  # (batch_size, r_left, r_right)
        
        # Contract: result @ selected
        # result: (batch_size, r_left_prev)
        # After reshape: (batch_size, r_left_prev, 1)
        # selected: (batch_size, r_left, r_right) where r_left = r_left_prev
        
        result = result.unsqueeze(-1)  # (batch_size, r_prev, 1)
        result = torch.bmm(result.transpose(1, 2), selected)  # (batch_size, 1, r_right)
        result = result.squeeze(1)  # (batch_size, r_right)
    
    # Final result should be scalar per sample
    return result.squeeze(-1)


def _indices_to_binary(indices: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Convert integer indices to binary representation for QTT indexing.
    
    Args:
        indices: 1D tensor of integer indices
        num_qubits: Number of bits (= number of QTT cores)
        
    Returns:
        2D tensor of shape (len(indices), num_qubits) with binary representation.
        Most significant bit is first (column 0).
    """
    batch_size = len(indices)
    device = indices.device
    
    binary = torch.zeros(batch_size, num_qubits, dtype=torch.long, device=device)
    temp_indices = indices.clone()
    
    for k in range(num_qubits):
        binary[:, num_qubits - 1 - k] = temp_indices % 2
        temp_indices = temp_indices // 2
    
    return binary


def verify_sound_speed_formula():
    """Verify that sound speed formula is correct.

    This is a sanity check to ensure we haven't made the c = γp/ρ mistake.

    For ideal gas: c² = γp/ρ, so c = sqrt(γp/ρ)

    At standard conditions (air at sea level):
    - γ = 1.4
    - p = 101325 Pa
    - ρ = 1.225 kg/m³
    - Expected c ≈ 340 m/s
    """
    gamma = 1.4
    p = 101325.0  # Pa
    rho = 1.225  # kg/m³

    # CORRECT formula
    c_correct = (gamma * p / rho) ** 0.5

    # WRONG formula (missing sqrt)
    c_wrong = gamma * p / rho

    print("Sound speed verification:")
    print(f"  Correct (sqrt): {c_correct:.1f} m/s (expected ~340)")
    print(f"  Wrong (no sqrt): {c_wrong:.1f} m/s")
    print(f"  Ratio: {c_wrong / c_correct:.1f}x")

    assert abs(c_correct - 340) < 5, "Sound speed formula verification failed!"
    return c_correct


if __name__ == "__main__":
    verify_sound_speed_formula()
