"""
Quantized Tensor Train (QTT) for CFD Field Compression
=======================================================

This module implements the core thesis of Project HyperTensor:

    **Area Law Compression**: Turbulent flow fields satisfy an Area Law
    analogous to quantum entanglement—correlations scale with boundary
    area, not volume—enabling compression from O(N³) to O(N·D²) via
    Tensor Train decomposition.

The QTT format reshapes an N-dimensional vector into a 2×2×...×2 tensor
(log₂N indices), then applies TT-SVD. For smooth functions with bounded
derivatives, QTT achieves exponential compression.

Key Functions:
    field_to_qtt     - Compress 2D flow field to QTT/MPS format
    qtt_to_field     - Reconstruct 2D field from QTT/MPS
    euler_to_qtt     - Compress full Euler2DState (ρ, ρu, ρv, E)
    qtt_to_euler     - Reconstruct Euler2DState from QTT list

Theory:
    For a flow field u(x,y) on an N×N grid, the QTT representation:

    1. Flatten to vector: u ∈ ℝ^(N²)
    2. Reshape to tensor: U ∈ ℝ^(2×2×...×2) with 2log₂N indices
    3. TT decomposition: U ≈ A₁ · A₂ · ... · A_L with χ_max bonds

    Storage: O(L · χ² · 2) = O(log N · χ²) vs O(N²) classical

    For smooth fields: χ = O(1), giving exponential compression!
    For shocks: χ ~ O(N^ε), still sublinear compression.

References:
    [1] Oseledets, "Tensor-Train Decomposition", SIAM J. Sci. Comput. (2011)
    [2] Khoromskij, "O(d log N)-Quantics Approximation", Constr. Approx. (2011)
    [3] Gourianov et al., "A quantum-inspired approach to exploit turbulence
        structures", arXiv:2305.10784 (2023)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from tensornet.core.mps import MPS


@dataclass
class QTTCompressionResult:
    """Result container for QTT compression."""

    mps: MPS
    original_shape: tuple[int, ...]
    compression_ratio: float
    truncation_error: float
    num_qubits: int
    bond_dimensions: list[int]
    norm: float = 1.0  # Store original norm for reconstruction

    def __repr__(self) -> str:
        return (
            f"QTTCompressionResult(shape={self.original_shape}, "
            f"CR={self.compression_ratio:.2f}x, "
            f"error={self.truncation_error:.2e}, "
            f"χ_max={max(self.bond_dimensions)})"
        )


def _next_power_of_two(n: int) -> int:
    """Return smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def _pad_to_power_of_two(tensor: Tensor, target_size: int) -> Tensor:
    """Pad tensor to target size with zeros."""
    if tensor.numel() >= target_size:
        return tensor.flatten()[:target_size]

    padded = torch.zeros(target_size, dtype=tensor.dtype, device=tensor.device)
    padded[: tensor.numel()] = tensor.flatten()
    return padded


def tt_svd(
    tensor: Tensor,
    shape: tuple[int, ...],
    chi_max: int | None = None,
    tol: float = 1e-10,
    normalize: bool = True,
    rsvd_threshold: int = 256,
) -> tuple[list[Tensor], float, float]:
    """
    Tensor Train SVD decomposition using randomized SVD (rSVD) for large matrices.

    Decomposes a tensor T[i₁,i₂,...,i_L] into a train of tensors:
    T ≈ A₁[i₁] · A₂[i₂] · ... · A_L[i_L]

    where each A_k has shape (χ_{k-1}, d_k, χ_k).

    Uses rSVD (Halko-Martinsson-Tropp algorithm) for matrices larger than
    rsvd_threshold, which is O(m·n·k) instead of O(m·n·min(m,n)) for full SVD.

    Args:
        tensor: Input tensor (will be reshaped to `shape`)
        shape: Target shape for TT decomposition
        chi_max: Maximum bond dimension
        tol: Truncation tolerance (relative to Frobenius norm)
        normalize: Whether to normalize the result
        rsvd_threshold: Use rSVD for matrices with min dimension > this value

    Returns:
        Tuple of (list of TT cores, total truncation error, norm)
    """
    L = len(shape)
    dtype = tensor.dtype
    device = tensor.device

    if chi_max is None:
        chi_max = 2 ** (L // 2)  # Reasonable default

    # Reshape tensor to target shape then flatten
    T = tensor.reshape(shape)
    frobenius_norm = torch.norm(T).item()

    cores = []
    total_error = 0.0

    # Left-to-right sweep with SVD truncation
    current = T.flatten()
    chi_left = 1
    remaining_size = current.numel()

    for k in range(L - 1):
        d_k = shape[k]
        remaining_size = remaining_size // d_k
        
        # Reshape for SVD: (χ_left * d_k, remaining)
        current = current.reshape(chi_left * d_k, remaining_size)
        m, n = current.shape
        
        # Choose SVD algorithm based on matrix size
        # rSVD is faster for large matrices when we only need top-k singular values
        use_rsvd = min(m, n) > rsvd_threshold and chi_max < min(m, n) // 2
        
        if use_rsvd:
            # Randomized SVD (Halko-Martinsson-Tropp): O(m·n·k) instead of O(m·n·min(m,n))
            # Request slightly more than chi_max for better accuracy
            q = min(chi_max + 10, min(m, n) - 1)
            U, S, V = torch.svd_lowrank(current, q=q, niter=2)
            # svd_lowrank returns V (not Vt), so transpose
            Vt = V.T
        else:
            # Standard SVD for small matrices (more accurate)
            U, S, Vt = torch.linalg.svd(current, full_matrices=False)

        # Determine truncation using vectorized operation (no Python loop)
        if tol > 0 and len(S) > 1:
            # Compute cumulative sum of squared singular values from the end
            S_sq = S ** 2
            # Reverse cumsum: tail_sq[i] = sum of S[i:]^2
            tail_sq = torch.flip(torch.cumsum(torch.flip(S_sq, [0]), dim=0), [0])
            threshold = tol ** 2 * frobenius_norm ** 2
            # Find first index where tail_sq <= threshold (vectorized)
            mask = tail_sq > threshold
            keep = mask.sum().item()  # Count of values above threshold
            keep = max(1, min(keep, chi_max))
        else:
            keep = max(1, min(chi_max, len(S)))

        # Truncate
        U = U[:, :keep]
        S_kept = S[:keep]
        Vt = Vt[:keep, :]

        # Track error (from truncated singular values)
        if keep < len(S):
            error = torch.sqrt(torch.sum(S[keep:] ** 2)).item()
            total_error += error**2

        # Form core tensor: (χ_left, d_k, χ_right)
        core = U.reshape(chi_left, d_k, keep)
        cores.append(core)

        # Propagate S @ Vt for next iteration (vectorized, no loop)
        current = (S_kept.unsqueeze(1) * Vt).flatten()
        chi_left = keep

    # Last core
    d_last = shape[-1]
    last_core = current.reshape(chi_left, d_last, 1)
    cores.append(last_core)

    total_error = math.sqrt(total_error)

    # Compute norm for later reconstruction
    final_norm = torch.norm(cores[-1]).item()

    # Normalize if requested (but return the norm for rescaling)
    if normalize and final_norm > 0:
        cores[-1] = cores[-1] / final_norm
    else:
        final_norm = 1.0

    return cores, total_error, final_norm


def field_to_qtt(
    field: Tensor,
    chi_max: int = 32,
    tol: float = 1e-10,
    normalize: bool = False,  # Default False to preserve amplitude
) -> QTTCompressionResult:
    """
    Compress a 2D field to Quantized Tensor Train (QTT) format.

    The field is reshaped into a tensor with 2×2×...×2 dimensions,
    then decomposed via TT-SVD. This exploits the hierarchical
    structure in smooth fields for exponential compression.

    Args:
        field: 2D tensor of shape (Ny, Nx)
        chi_max: Maximum bond dimension
        tol: Truncation tolerance
        normalize: Normalize the MPS

    Returns:
        QTTCompressionResult with compressed MPS and metadata

    Example:
        >>> field = torch.randn(64, 64)
        >>> result = field_to_qtt(field, chi_max=16)
        >>> print(f"Compression: {result.compression_ratio:.1f}x")
        >>> reconstructed = qtt_to_field(result)
    """
    original_shape = tuple(field.shape)
    dtype = field.dtype
    device = field.device

    # Flatten and pad to power of 2
    flat = field.flatten()
    N = flat.numel()
    N_padded = _next_power_of_two(N)

    if N_padded > N:
        flat = _pad_to_power_of_two(flat, N_padded)

    # Determine number of qubits (each index is 2-dimensional)
    num_qubits = int(math.log2(N_padded))
    qtt_shape = tuple([2] * num_qubits)

    # TT-SVD decomposition
    cores, truncation_error, field_norm = tt_svd(
        flat, qtt_shape, chi_max=chi_max, tol=tol, normalize=normalize
    )

    # Create MPS from cores
    mps = MPS(cores)

    # Compute compression ratio
    original_params = N
    compressed_params = sum(c.numel() for c in cores)
    compression_ratio = original_params / compressed_params

    bond_dims = [cores[0].shape[0]] + [c.shape[2] for c in cores[:-1]]

    return QTTCompressionResult(
        mps=mps,
        original_shape=original_shape,
        compression_ratio=compression_ratio,
        truncation_error=truncation_error,
        num_qubits=num_qubits,
        bond_dimensions=bond_dims,
        norm=field_norm,
    )


def qtt_to_field(result: QTTCompressionResult) -> Tensor:
    """
    Reconstruct 2D field from QTT/MPS representation.

    Args:
        result: QTTCompressionResult from field_to_qtt

    Returns:
        Reconstructed 2D tensor with original shape
    """
    # Contract MPS to get full tensor
    mps = result.mps

    # Contract all cores to get the full vector
    # Start with first core
    vec = mps.tensors[0]  # (1, d, chi)

    for core in mps.tensors[1:]:
        # vec: (..., chi_left)
        # core: (chi_left, d, chi_right)
        # Contract last dim of vec with first dim of core
        vec = torch.tensordot(vec, core, dims=([-1], [0]))

    # vec is now (1, d1, d2, ..., dL, 1) - squeeze boundaries
    vec = vec.squeeze(0).squeeze(-1)

    # Flatten to 1D
    vec = vec.flatten()

    # Rescale by stored norm
    vec = vec * result.norm

    # Unpad to original size
    original_size = math.prod(result.original_shape)
    vec = vec[:original_size]

    # Reshape to original dimensions
    return vec.reshape(result.original_shape)


def euler_to_qtt(
    state,  # Euler2DState - avoid circular import
    chi_max: int = 32,
    tol: float = 1e-10,
    compress_together: bool = False,
) -> dict[str, QTTCompressionResult]:
    """
    Compress full Euler2DState to QTT format.

    Compresses all four conservative variables (ρ, ρu, ρv, E)
    to separate QTT representations.

    Args:
        state: Euler2DState with flow field data
        chi_max: Maximum bond dimension per field
        tol: Truncation tolerance
        compress_together: If True, stack fields and compress as one

    Returns:
        Dictionary mapping field names to QTTCompressionResult

    Example:
        >>> from tensornet.cfd.euler_2d import supersonic_wedge_ic
        >>> state = supersonic_wedge_ic(Nx=128, Ny=64, M_inf=5.0)
        >>> compressed = euler_to_qtt(state, chi_max=24)
        >>> for name, result in compressed.items():
        ...     print(f"{name}: {result.compression_ratio:.1f}x")
    """
    # Get conservative variables
    U = state.to_conservative()

    field_names = ["rho", "rho_u", "rho_v", "E"]
    results = {}

    for i, name in enumerate(field_names):
        field = U[i]
        results[name] = field_to_qtt(field, chi_max=chi_max, tol=tol)

    return results


def qtt_to_euler(compressed: dict[str, QTTCompressionResult], gamma: float = 1.4):
    """
    Reconstruct Euler2DState from QTT-compressed fields.

    Args:
        compressed: Dictionary from euler_to_qtt
        gamma: Ratio of specific heats

    Returns:
        Euler2DState reconstructed from compressed data
    """
    # Import here to avoid circular dependency
    from tensornet.cfd.euler_2d import Euler2DState

    # Reconstruct conservative variables
    rho = qtt_to_field(compressed["rho"])
    rho_u = qtt_to_field(compressed["rho_u"])
    rho_v = qtt_to_field(compressed["rho_v"])
    E = qtt_to_field(compressed["E"])

    # Convert to primitive
    u = rho_u / rho
    v = rho_v / rho
    ke = 0.5 * rho * (u**2 + v**2)
    p = (gamma - 1) * (E - ke)

    return Euler2DState(rho=rho, u=u, v=v, p=p, gamma=gamma)


def compression_analysis(
    state, chi_values: list[int] = [4, 8, 16, 32, 64, 128], verbose: bool = True
) -> dict:
    """
    Analyze compression quality vs bond dimension.

    Args:
        state: Euler2DState to analyze
        chi_values: List of bond dimensions to test
        verbose: Print results

    Returns:
        Dictionary with analysis results
    """
    results = {
        "chi": [],
        "compression_ratio": [],
        "rho_error": [],
        "u_error": [],
        "p_error": [],
    }

    # Get original fields
    rho_orig = state.rho.clone()
    u_orig = state.u.clone()
    p_orig = state.p.clone()

    if verbose:
        print("=" * 70)
        print("QTT COMPRESSION ANALYSIS")
        print("=" * 70)
        print(f"Field shape: {tuple(rho_orig.shape)}")
        print(f"Original size: {rho_orig.numel() * 4} elements (4 fields)")
        print()
        print(f"{'χ':>6} {'CR':>10} {'ρ error':>12} {'u error':>12} {'p error':>12}")
        print("-" * 70)

    for chi in chi_values:
        compressed = euler_to_qtt(state, chi_max=chi)
        reconstructed = qtt_to_euler(compressed, gamma=state.gamma)

        # Compute relative errors
        rho_err = torch.norm(reconstructed.rho - rho_orig) / torch.norm(rho_orig)
        u_err = torch.norm(reconstructed.u - u_orig) / (torch.norm(u_orig) + 1e-10)
        p_err = torch.norm(reconstructed.p - p_orig) / torch.norm(p_orig)

        # Average compression ratio
        cr = sum(r.compression_ratio for r in compressed.values()) / 4

        results["chi"].append(chi)
        results["compression_ratio"].append(cr)
        results["rho_error"].append(rho_err.item())
        results["u_error"].append(u_err.item())
        results["p_error"].append(p_err.item())

        if verbose:
            print(
                f"{chi:>6} {cr:>10.2f}x {rho_err.item():>12.2e} "
                f"{u_err.item():>12.2e} {p_err.item():>12.2e}"
            )

    if verbose:
        print("=" * 70)

    return results


def estimate_area_law_exponent(
    state,
    chi_values: list[int] = [4, 8, 16, 32, 64],
) -> dict:
    """
    Estimate the Area Law exponent from compression scaling.

    For true Area Law behavior, the error should scale as:
        ε(χ) ~ exp(-α · χ^β)

    with β = 1 for exact area law (1D entanglement).
    For volume law (generic states), β → 0.

    Args:
        state: Euler2DState to analyze
        chi_values: Bond dimensions to test

    Returns:
        Dictionary with scaling exponents
    """
    import math

    analysis = compression_analysis(state, chi_values, verbose=False)

    # Fit log-log for power law: log(ε) = -α * χ^β
    # Approximate: fit log(ε) vs log(χ)
    log_chi = [math.log(c) for c in chi_values]
    log_err = [math.log(e + 1e-16) for e in analysis["rho_error"]]

    # Simple linear regression
    n = len(log_chi)
    mean_x = sum(log_chi) / n
    mean_y = sum(log_err) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_chi, log_err))
    den = sum((x - mean_x) ** 2 for x in log_chi)

    slope = num / den if den > 0 else 0
    intercept = mean_y - slope * mean_x

    return {
        "slope": slope,  # Negative slope indicates compression
        "intercept": intercept,
        "interpretation": (
            "Area Law"
            if slope < -0.5
            else "Volume Law" if slope > -0.1 else "Intermediate"
        ),
        "raw_data": analysis,
    }
