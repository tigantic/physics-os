"""
Pure QTT Arithmetic Operations.

This module implements operations directly on QTT (Quantized Tensor-Train) cores
WITHOUT decompressing to the full grid. This enables handling of grids up to 2^60
points on a laptop.

The key insight: Derivatives, Laplacians, and convolutions can all be expressed
as MPO (Matrix Product Operator) actions on the MPS (Matrix Product State).

QTT -> Math (on cores) -> QTT

Not: QTT -> Decompress -> Math -> Compress -> QTT

References:
- Oseledets (2010): "Tensor-Train Decomposition"
- Kazeev & Schwab (2015): "Quantized Tensor-Train approach for PDEs"
- Bachmayr & Dahmen (2016): "Adaptive low-rank methods"
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class QTTCore:
    """A single core of a QTT decomposition."""

    tensor: torch.Tensor  # Shape: (r_left, 2, r_right) for 1D


@dataclass
class QTTState:
    """A full QTT state (MPS with physical dimension 2)."""

    cores: list[torch.Tensor]  # List of cores
    num_qubits: int  # Number of qubits = log2(grid_size)

    @property
    def grid_size(self) -> int:
        return 2**self.num_qubits

    @property
    def ranks(self) -> list[int]:
        """Bond dimensions between cores."""
        return [c.shape[2] for c in self.cores[:-1]]

    @property
    def max_rank(self) -> int:
        """Maximum bond dimension."""
        return max(c.shape[2] for c in self.cores) if self.cores else 1


@dataclass
class MPOCore:
    """A single core of an MPO (Matrix Product Operator)."""

    tensor: torch.Tensor  # Shape: (r_left, d_out, d_in, r_right)


@dataclass
class MPO:
    """Matrix Product Operator for QTT operations."""

    cores: list[torch.Tensor]
    num_sites: int


# =============================================================================
# MPO ARITHMETIC — O(n_qubits) operations
# =============================================================================

def mpo_scale(mpo: MPO, scalar: float) -> MPO:
    """
    Scale an MPO by a scalar: scalar * O.
    
    Complexity: O(1) - only modifies first core.
    """
    if len(mpo.cores) == 0:
        return mpo
    
    new_cores = [c.clone() for c in mpo.cores]
    new_cores[0] = new_cores[0] * scalar
    return MPO(cores=new_cores, num_sites=mpo.num_sites)


def mpo_add(mpo1: MPO, mpo2: MPO) -> MPO:
    """
    Add two MPOs: O1 + O2.
    
    Result has bond dimension r1 + r2 (block diagonal structure).
    Complexity: O(n_qubits) - one pass through cores.
    
    Args:
        mpo1: First MPO
        mpo2: Second MPO (must have same num_sites)
        
    Returns:
        MPO representing mpo1 + mpo2
    """
    if mpo1.num_sites != mpo2.num_sites:
        raise ValueError(f"MPO site count mismatch: {mpo1.num_sites} vs {mpo2.num_sites}")
    
    n = mpo1.num_sites
    new_cores = []
    
    for i in range(n):
        c1 = mpo1.cores[i]  # (r1_left, d_out, d_in, r1_right)
        c2 = mpo2.cores[i]  # (r2_left, d_out, d_in, r2_right)
        
        r1L, d_out, d_in, r1R = c1.shape
        r2L, _, _, r2R = c2.shape
        
        device = c1.device
        dtype = c1.dtype
        
        if i == 0:
            # First core: concatenate along right bond
            # Result: (1, d_out, d_in, r1R + r2R)
            new_core = torch.zeros(1, d_out, d_in, r1R + r2R, device=device, dtype=dtype)
            new_core[0, :, :, :r1R] = c1[0]
            new_core[0, :, :, r1R:] = c2[0]
        elif i == n - 1:
            # Last core: concatenate along left bond
            # Result: (r1L + r2L, d_out, d_in, 1)
            new_core = torch.zeros(r1L + r2L, d_out, d_in, 1, device=device, dtype=dtype)
            new_core[:r1L, :, :, 0] = c1[:, :, :, 0]
            new_core[r1L:, :, :, 0] = c2[:, :, :, 0]
        else:
            # Middle cores: block diagonal
            # Result: (r1L + r2L, d_out, d_in, r1R + r2R)
            new_core = torch.zeros(r1L + r2L, d_out, d_in, r1R + r2R, device=device, dtype=dtype)
            new_core[:r1L, :, :, :r1R] = c1
            new_core[r1L:, :, :, r1R:] = c2
        
        new_cores.append(new_core)
    
    return MPO(cores=new_cores, num_sites=n)


def mpo_negate(mpo: MPO) -> MPO:
    """Negate an MPO: -O."""
    return mpo_scale(mpo, -1.0)


def mpo_subtract(mpo1: MPO, mpo2: MPO) -> MPO:
    """Subtract two MPOs: O1 - O2."""
    return mpo_add(mpo1, mpo_negate(mpo2))


def identity_mpo(
    num_qubits: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu"
) -> MPO:
    """
    Create the identity MPO.

    I = ⊗ᵢ [[1, 0], [0, 1]]
    
    Args:
        num_qubits: Number of qubits (grid = 2^n)
        dtype: Tensor dtype (torch.float32 or torch.float64)
        device: Device to create tensors on
    """
    cores = []
    I = torch.eye(2, dtype=dtype, device=device)

    for i in range(num_qubits):
        # Shape: (1, 2, 2, 1) - trivial bond dimensions
        core = I.unsqueeze(0).unsqueeze(-1)
        cores.append(core)

    return MPO(cores=cores, num_sites=num_qubits)


def shift_mpo(
    num_qubits: int,
    direction: int = 1,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu"
) -> MPO:
    """
    Create the shift operator S in MPO form.

    S|x⟩ = |x+1 mod 2^n⟩  (forward)
    S|x⟩ = |x-1 mod 2^n⟩  (backward)

    This is a building block for derivative operators.
    The shift uses carry propagation with bounded bond dimension.
    
    QTT bit ordering: core[0] = MSB, core[n-1] = LSB
    For increment: carry propagates from LSB to MSB

    Args:
        num_qubits: Number of qubits (grid = 2^n)
        direction: +1 for forward shift, -1 for backward
        dtype: Tensor dtype (torch.float32 or torch.float64)
        device: Device to create tensors on
    """
    cores = []

    # For forward shift (+1):
    # Start at LSB (rightmost core, i=n-1): increment with carry_in=1
    # Propagate carry leftward toward MSB
    #
    # MPO core layout: (r_left, d_out, d_in, r_right)
    # Bond carries the "carry" signal: 0 = no carry, 1 = carry
    
    # Process from LSB to MSB (i = n-1 down to 0)
    for i in range(num_qubits - 1, -1, -1):
        if i == num_qubits - 1:
            # LSB (rightmost): always increment (carry_in = 1)
            # r_left=2, d_out=2, d_in=2, r_right=1
            core = torch.zeros(2, 2, 2, 1, dtype=dtype, device=device)
            if direction == 1:
                # |0⟩ + 1 → |1⟩, carry_out=0
                core[0, 1, 0, 0] = 1.0
                # |1⟩ + 1 → |0⟩, carry_out=1
                core[1, 0, 1, 0] = 1.0
            else:
                # Decrement: borrow logic
                # |0⟩ - 1 → |1⟩, borrow_out=1
                core[1, 1, 0, 0] = 1.0
                # |1⟩ - 1 → |0⟩, borrow_out=0
                core[0, 0, 1, 0] = 1.0
        elif i == 0:
            # MSB (leftmost): absorb carry, no outgoing
            # r_left=1, d_out=2, d_in=2, r_right=2
            core = torch.zeros(1, 2, 2, 2, dtype=dtype, device=device)
            if direction == 1:
                # carry_in=0: identity
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
                # carry_in=1: increment (wraps at MSB)
                core[0, 1, 0, 1] = 1.0  # |0⟩+1 → |1⟩
                core[0, 0, 1, 1] = 1.0  # |1⟩+1 → |0⟩ (wrap)
            else:
                # carry_in=0: identity
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
                # borrow_in=1: decrement
                core[0, 1, 0, 1] = 1.0  # |0⟩-1 → |1⟩ (wrap)
                core[0, 0, 1, 1] = 1.0  # |1⟩-1 → |0⟩
        else:
            # Middle sites: propagate carry
            # r_left=2, d_out=2, d_in=2, r_right=2
            core = torch.zeros(2, 2, 2, 2, dtype=dtype, device=device)
            if direction == 1:
                # carry_in=0: identity, carry_out=0
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
                # carry_in=1: increment
                core[0, 1, 0, 1] = 1.0  # |0⟩+1 → |1⟩, carry_out=0
                core[1, 0, 1, 1] = 1.0  # |1⟩+1 → |0⟩, carry_out=1
            else:
                # borrow_in=0: identity
                core[0, 0, 0, 0] = 1.0
                core[0, 1, 1, 0] = 1.0
                # borrow_in=1: decrement
                core[1, 1, 0, 1] = 1.0  # |0⟩-1 → |1⟩, borrow_out=1
                core[0, 0, 1, 1] = 1.0  # |1⟩-1 → |0⟩, borrow_out=0

        cores.append(core)
    
    # Reverse to get MSB-first order
    cores = cores[::-1]

    return MPO(cores=cores, num_sites=num_qubits)


def _shift_plus_mpo(num_qubits: int) -> MPO:
    """
    Create the forward shift operator S⁺ in MPO form: S⁺|x⟩ = |x+1 mod 2^n⟩

    Uses O(n) ripple-carry construction with bond dimension 2.
    NO DENSE MATRICES - works for any num_qubits.
    
    The ripple-carry adder logic:
    - LSB (site n-1): always increment, propagate carry if bit was 1
    - Middle sites: if carry_in, flip bit and propagate; else identity
    - MSB (site 0): absorb carry (periodic BC)
    
    Bond dimension = 2 encodes carry state.
    """
    # Delegate to the correct O(n) implementation
    return shift_mpo(num_qubits, direction=+1)


def _shift_minus_mpo(num_qubits: int) -> MPO:
    """
    Create backward shift S⁻|x⟩ = |x-1 mod N⟩.
    
    Uses O(n) ripple-borrow construction with bond dimension 2.
    NO DENSE MATRICES.
    """
    return shift_mpo(num_qubits, direction=-1)


def _dense_matrix_to_mpo(mat: torch.Tensor, num_qubits: int, max_bond: int = 64) -> MPO:
    """
    Convert a dense 2^n × 2^n matrix to MPO form via SVD.

    The matrix M[y, x] is viewed as a tensor M[y_0,..,y_{n-1}, x_0,..,x_{n-1}]
    where y = sum_i y_i * 2^{n-1-i} (MSB first).

    We then decompose into MPO cores O^i[r_l, y_i, x_i, r_r].
    """
    N = 2**num_qubits
    assert mat.shape == (N, N), f"Matrix shape {mat.shape} != ({N}, {N})"

    # Reshape to tensor with 2n indices: [y_0, y_1, ..., y_{n-1}, x_0, ..., x_{n-1}]
    T = mat.reshape([2] * num_qubits + [2] * num_qubits)

    # Reorder to interleaved: [y_0, x_0, y_1, x_1, ..., y_{n-1}, x_{n-1}]
    perm = []
    for i in range(num_qubits):
        perm.append(i)  # y_i
        perm.append(num_qubits + i)  # x_i
    T = T.permute(perm)

    # T now has shape [2, 2, 2, 2, ..., 2, 2] (2n indices)
    # Each pair (y_i, x_i) will become one MPO site with d_out=2, d_in=2

    # Sequential SVD to extract MPO cores
    cores = []
    current = T.reshape(4, -1)  # First pair → (4, 4^{n-1})
    r_left = 1

    for i in range(num_qubits):
        if i < num_qubits - 1:
            # current shape: (r_left * 4, remaining)
            # We want to factor out one (4,) from the left

            # Reshape to (r_left * 4, remaining)
            mat_2d = current.reshape(-1, current.shape[-1])

            q = min(max_bond, min(mat_2d.shape))
            U, S, Vh = torch.svd_lowrank(mat_2d, q=q, niter=1)

            # Determine rank
            rank = min(len(S), max_bond)
            # Also truncate small singular values
            if len(S) > 1:
                rel_cutoff = 1e-14 * S[0]
                rank = min(rank, (S > rel_cutoff).sum().item())
            rank = max(1, rank)

            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]

            # Core shape: (r_left, 2, 2, rank)
            if i == 0:
                core = U.reshape(1, 2, 2, rank)
            else:
                core = U.reshape(r_left, 2, 2, rank)
            cores.append(core)

            # Prepare for next iteration
            current = torch.diag(S) @ Vh  # (rank, remaining)
            r_left = rank

            # Reshape current for next site: need to extract next (2,2) pair
            # remaining = 4^{n-i-1}
            remaining_pairs = num_qubits - i - 1
            if remaining_pairs > 1:
                # current: (rank, 4^{remaining_pairs})
                # reshape to (rank * 4, 4^{remaining_pairs - 1})
                current = current.reshape(r_left * 4, -1)
            else:
                # Last iteration: current is (rank, 4)
                current = current.reshape(r_left * 4, 1)
        else:
            # Last site: just reshape what's left
            # current: (r_left * 4, 1) or (r_left, 4)
            core = current.reshape(r_left, 2, 2, 1)
            cores.append(core)

    return MPO(cores=cores, num_sites=num_qubits)


def derivative_mpo(
    num_qubits: int,
    dx: float,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu"
) -> MPO:
    """
    Create the first derivative operator D in MPO form (central difference).

    Central difference: D f[i] = (f[i+1] - f[i-1]) / (2*dx)
    
    Operator semantics:
    - S⁺|x⟩ = |x+1⟩ so (S⁺ f)[i] = f[i-1] (shift values right)
    - S⁻|x⟩ = |x-1⟩ so (S⁻ f)[i] = f[i+1] (shift values left)
    
    Therefore: D = (S⁻ - S⁺) / (2*dx)

    Uses O(n_qubits) MPO arithmetic instead of O(N²) dense matrix.
    Bond dimension of result: 4 (sum of two rank-2 shift MPOs).

    Args:
        num_qubits: log2(grid_size)
        dx: Grid spacing
        dtype: Tensor dtype (torch.float32 or torch.float64)
        device: Device to create tensors on

    Returns:
        MPO for derivative operator
    """
    # D = (S⁻ - S⁺) / (2*dx)
    # Because S⁻ f[i] = f[i+1] and S⁺ f[i] = f[i-1]
    scale = 1.0 / (2 * dx)
    
    S_plus = shift_mpo(num_qubits, direction=+1, dtype=dtype, device=device)
    S_minus = shift_mpo(num_qubits, direction=-1, dtype=dtype, device=device)
    
    # S⁻ - S⁺ gives (f[i+1] - f[i-1])
    diff = mpo_subtract(S_minus, S_plus)
    
    # Scale by 1/(2*dx)
    return mpo_scale(diff, scale)


def laplacian_mpo(
    num_qubits: int,
    dx: float,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu"
) -> MPO:
    """
    Create the Laplacian operator Δ = (S⁺ - 2I + S⁻) / dx² in MPO form.

    Uses O(n_qubits) MPO arithmetic instead of O(N²) dense matrix.
    Bond dimension of result: 5 (sum of three MPOs: rank-2 + rank-1 + rank-2).

    This is the standard second-order central difference:
    (d²f/dx²)(x) ≈ [f(x+dx) - 2f(x) + f(x-dx)] / dx²

    Args:
        num_qubits: log2(grid_size)
        dx: Grid spacing
        dtype: Tensor dtype (torch.float32 or torch.float64)
        device: Device to create tensors on

    Returns:
        MPO for Laplacian operator
    """
    # L = (S⁺ - 2I + S⁻) / dx²
    # Build using O(n_qubits) MPO arithmetic
    scale = 1.0 / (dx * dx)
    
    S_plus = shift_mpo(num_qubits, direction=+1, dtype=dtype, device=device)
    S_minus = shift_mpo(num_qubits, direction=-1, dtype=dtype, device=device)
    I = identity_mpo(num_qubits, dtype=dtype, device=device)
    
    # S⁺ + S⁻
    shifts_sum = mpo_add(S_plus, S_minus)
    
    # S⁺ + S⁻ - 2I
    minus_2I = mpo_scale(I, -2.0)
    stencil = mpo_add(shifts_sum, minus_2I)
    
    # Scale by 1/dx²
    return mpo_scale(stencil, scale)


def apply_mpo(mpo: MPO, qtt: QTTState, max_bond: int = 64) -> QTTState:
    """
    Apply an MPO to a QTT state: |ψ'⟩ = O|ψ⟩

    This is the core operation that enables pure QTT arithmetic.
    The result is a new QTT state (with possibly increased bond dimension).

    Contraction diagram:

        MPO core:   (rLo) --[ O ]-- (rRo)
                            |d_out
                            |d_in

        QTT core:   (rLp) --[ P ]-- (rRp)
                            |d_in

        Result:     (rLo*rLp) --[ R ]-- (rRo*rRp)
                               |d_out

    Args:
        mpo: Matrix Product Operator
        qtt: Input QTT state
        max_bond: Maximum bond dimension (truncate if exceeded)

    Returns:
        New QTT state = MPO @ QTT
    """
    assert mpo.num_sites == qtt.num_qubits, "MPO and QTT must have same number of sites"

    # Determine target dtype (use QTT's dtype, promote to float64 if mixed)
    target_dtype = qtt.cores[0].dtype

    new_cores = []

    for i in range(qtt.num_qubits):
        # Contract MPO core with QTT core over physical input index
        # MPO core O: (rLo, d_out, d_in, rRo)  - indices: o, a, b, r
        # QTT core P: (rLp, d_in, rRp)         - indices: p, b, q
        # Contract over b (d_in), output d_out (a)
        # Result: (rLo, rLp, d_out, rRo, rRp)  - indices: o, p, a, r, q

        O = mpo.cores[i].to(target_dtype)  # Convert MPO to match QTT dtype
        P = qtt.cores[i]  # (rLp, d_in, rRp)

        rLo, d_out, d_in, rRo = O.shape
        rLp, d_in_p, rRp = P.shape

        assert (
            d_in == d_in_p
        ), f"Physical dimension mismatch at site {i}: MPO has d_in={d_in}, QTT has d={d_in_p}"

        # Contract over physical input dimension (b)
        # O[o,a,b,r] @ P[p,b,q] -> result[o,p,a,r,q]
        result = torch.einsum("oabr,pbq->oparq", O, P)

        # Reshape to (rLo*rLp, d_out, rRo*rRp)
        result = result.reshape(rLo * rLp, d_out, rRo * rRp)

        new_cores.append(result)

    new_qtt = QTTState(cores=new_cores, num_qubits=qtt.num_qubits)

    # Truncate to control bond dimension
    new_qtt = truncate_qtt(new_qtt, max_bond=max_bond)

    return new_qtt


def truncate_qtt(qtt: QTTState, max_bond: int = 64, tol: float = 1e-10) -> QTTState:
    """
    Truncate QTT bond dimensions using SVD.

    This is the compression step that keeps the representation efficient.
    
    OPTIMIZED (January 2026):
    - Single right-to-left SVD sweep (no redundant QR pass)
    - Skips cores that are already within bounds
    - Uses rSVD (svd_lowrank) for O(r² max_bond) instead of O(r³)

    Args:
        qtt: Input QTT state
        max_bond: Maximum allowed bond dimension
        tol: Singular value threshold

    Returns:
        Compressed QTT state
    """
    # Fast path: check if truncation is needed at all
    # Must check BOTH left and right bonds (qtt_add inflates both)
    max_right_bond = max(c.shape[2] for c in qtt.cores[:-1]) if len(qtt.cores) > 1 else 1
    max_left_bond = max(c.shape[0] for c in qtt.cores[1:]) if len(qtt.cores) > 1 else 1
    max_current_bond = max(max_right_bond, max_left_bond)
    
    if max_current_bond <= max_bond:
        # Already within bounds - just clean NaN/Inf
        cores = [torch.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6) for c in qtt.cores]
        return QTTState(cores=cores, num_qubits=qtt.num_qubits)
    
    # Single right-to-left SVD sweep
    # This is sufficient and avoids redundant QR pass
    cores = [c.clone() for c in qtt.cores]
    n = len(cores)
    
    for i in range(n - 1, 0, -1):
        c = cores[i]
        r_left, d, r_right = c.shape
        
        # Skip if left bond already within bounds
        if r_left <= max_bond:
            continue

        # Reshape: (r_left, d * r_right) for left-side compression
        mat = c.reshape(r_left, d * r_right)
        
        # Clean matrix
        mat = torch.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)

        try:
            # rSVD: only compute top max_bond singular values
            q = min(max_bond, min(mat.shape))
            U, S, V = torch.svd_lowrank(mat, q=q, niter=1)
            
            # Truncate based on tolerance and max_bond
            if tol > 0 and len(S) > 0:
                mask = S > tol * S[0]
                new_rank = min(mask.sum().item(), max_bond)
            else:
                new_rank = min(len(S), max_bond)
            new_rank = max(1, new_rank)

            U = U[:, :new_rank]
            S = S[:new_rank]
            V = V[:, :new_rank]  # V is (n, k), not Vh

            # Update current core: V.T @ original = new core
            cores[i] = V.T.reshape(new_rank, d, r_right)
            
            # Absorb U @ S into previous core
            US = U * S.unsqueeze(0)  # (r_left, new_rank)
            cores[i - 1] = torch.einsum("ijk,kl->ijl", cores[i - 1], US)
            
        except (RuntimeError, torch.linalg.LinAlgError):
            # SVD failed - keep core as is
            continue

    return QTTState(cores=cores, num_qubits=qtt.num_qubits)


def qtt_add(qtt1: QTTState, qtt2: QTTState, max_bond: int = 64, truncate: bool = True, tol: float = 1e-10) -> QTTState:
    """
    Add two QTT states: |ψ⟩ = |ψ₁⟩ + |ψ₂⟩

    Bond dimension doubles, then truncate (if truncate=True).
    
    Args:
        qtt1, qtt2: QTT states to add
        max_bond: Maximum bond dimension after truncation
        truncate: If False, skip truncation (for batching multiple ops)
        tol: SVD truncation tolerance.  Singular values below tol × S_max
             are discarded.  Set higher (e.g. 1e-6) for adaptive rank
             control where the rank grows with solution complexity.
    
    Optimized:
    - Skips truncation if combined ranks already ≤ max_bond
    - Block diagonal assembly is vectorized
    - Set truncate=False for deferred truncation (batch multiple adds)
    """
    assert qtt1.num_qubits == qtt2.num_qubits

    cores = []
    n = qtt1.num_qubits

    # Determine target dtype and device (promote to higher precision if mixed)
    dtype = qtt1.cores[0].dtype
    device = qtt1.cores[0].device
    if qtt2.cores[0].dtype == torch.float64:
        dtype = torch.float64

    # Check if truncation will be needed
    max_combined_rank = 0
    
    for i in range(n):
        c1 = qtt1.cores[i].to(dtype=dtype, device=device)
        c2 = qtt2.cores[i].to(dtype=dtype, device=device)

        r1L, d, r1R = c1.shape
        r2L, _, r2R = c2.shape

        if i == 0:
            # First core: concatenate along right bond
            new_core = torch.cat([c1, c2], dim=2)
            max_combined_rank = max(max_combined_rank, r1R + r2R)
        elif i == n - 1:
            # Last core: concatenate along left bond
            new_core = torch.cat([c1, c2], dim=0)
            max_combined_rank = max(max_combined_rank, r1L + r2L)
        else:
            # Middle cores: block diagonal
            new_core = torch.zeros(r1L + r2L, d, r1R + r2R, dtype=dtype, device=device)
            new_core[:r1L, :, :r1R] = c1
            new_core[r1L:, :, r1R:] = c2
            max_combined_rank = max(max_combined_rank, r1L + r2L, r1R + r2R)

        cores.append(new_core)

    result = QTTState(cores=cores, num_qubits=n)
    
    # Skip truncation if already within bounds OR if deferred
    if not truncate or max_combined_rank <= max_bond:
        return result
    
    return truncate_qtt(result, max_bond=max_bond, tol=tol)


def qtt_sum(states: list[QTTState], max_bond: int = 64, weights: list[float] | None = None) -> QTTState:
    """
    Sum multiple QTT states in one fused operation: |ψ⟩ = Σᵢ wᵢ|ψᵢ⟩
    
    This is MUCH faster than chaining qtt_add() calls:
    - Single block-diagonal assembly for all N states
    - Single truncation sweep at the end
    - O(N) memory vs O(N²) for pairwise adds
    
    Args:
        states: List of QTT states to sum
        max_bond: Maximum bond dimension after truncation
        weights: Optional weights for each state (default: all 1.0)
    
    Returns:
        QTT state representing weighted sum
        
    Example:
        # Jacobi iteration: psi = (psi_xp + psi_xm + psi_yp + psi_ym + rhs) / D
        psi = qtt_sum([psi_xp, psi_xm, psi_yp, psi_ym, rhs], max_bond=24)
    """
    if len(states) == 0:
        raise ValueError("Need at least one state to sum")
    if len(states) == 1:
        return states[0] if weights is None else qtt_scale(states[0], weights[0])
    
    n = states[0].num_qubits
    for s in states[1:]:
        assert s.num_qubits == n, "All states must have same num_qubits"
    
    if weights is None:
        weights = [1.0] * len(states)
    
    # Determine dtype/device
    dtype = states[0].cores[0].dtype
    device = states[0].cores[0].device
    for s in states:
        if s.cores[0].dtype == torch.float64:
            dtype = torch.float64
    
    # Apply weights to first cores
    weighted_states = []
    for s, w in zip(states, weights):
        if w == 1.0:
            weighted_states.append(s)
        else:
            weighted_states.append(qtt_scale(s, w))
    
    # Fused block-diagonal assembly
    cores = []
    for i in range(n):
        all_cores = [s.cores[i].to(dtype=dtype, device=device) for s in weighted_states]
        
        if i == 0:
            # First: concatenate along right bond
            new_core = torch.cat(all_cores, dim=2)
        elif i == n - 1:
            # Last: concatenate along left bond
            new_core = torch.cat(all_cores, dim=0)
        else:
            # Middle: block diagonal
            total_left = sum(c.shape[0] for c in all_cores)
            total_right = sum(c.shape[2] for c in all_cores)
            d = all_cores[0].shape[1]
            new_core = torch.zeros(total_left, d, total_right, dtype=dtype, device=device)
            
            left_offset, right_offset = 0, 0
            for c in all_cores:
                rL, _, rR = c.shape
                new_core[left_offset:left_offset+rL, :, right_offset:right_offset+rR] = c
                left_offset += rL
                right_offset += rR
        
        cores.append(new_core)
    
    result = QTTState(cores=cores, num_qubits=n)
    return truncate_qtt(result, max_bond=max_bond)


def qtt_scale(qtt: QTTState, scalar: float) -> QTTState:
    """Scale a QTT state by a scalar."""
    cores = [c.clone() for c in qtt.cores]
    cores[0] = cores[0] * scalar
    return QTTState(cores=cores, num_qubits=qtt.num_qubits)


def qtt_hadamard(qtt1: QTTState, qtt2: QTTState, max_bond: int = 64, truncate: bool = True) -> QTTState:
    """
    Element-wise (Hadamard) product of two QTT states: |ψ⟩ = |ψ₁⟩ ⊙ |ψ₂⟩

    The Hadamard product in TT format is computed by taking the Kronecker
    product of corresponding cores at each site:

    C_new[i] = C1[i] ⊗ C2[i]  (in the bond indices)

    Resulting bond dimension = r1 × r2.

    Args:
        qtt1: First QTT state
        qtt2: Second QTT state
        max_bond: Maximum bond dimension after truncation
        truncate: If False, skip truncation (for batching multiple ops)

    Returns:
        QTT state representing element-wise product
    """
    assert qtt1.num_qubits == qtt2.num_qubits, "QTT dimensions must match"

    n = qtt1.num_qubits
    
    # Build Kronecker product of all cores
    cores = []
    for i in range(n):
        c1 = qtt1.cores[i]  # (r1L, d, r1R)
        c2 = qtt2.cores[i]  # (r2L, d, r2R)
        r1L, d, r1R = c1.shape
        r2L, _, r2R = c2.shape
        
        # Kronecker in bond dimensions: (r1L, d, r1R) ⊗ (r2L, d, r2R) -> (r1L*r2L, d, r1R*r2R)
        kron = torch.einsum("adb,cde->acdbe", c1, c2)
        cores.append(kron.reshape(r1L * r2L, d, r1R * r2R))
    
    result = QTTState(cores=cores, num_qubits=n)
    
    # Skip truncation if deferred
    if not truncate:
        return result
    
    # Truncate if needed (optimized single-sweep rSVD)
    max_right_bond = max(c.shape[2] for c in cores[:-1]) if n > 1 else 1
    max_left_bond = max(c.shape[0] for c in cores[1:]) if n > 1 else 1
    if max(max_right_bond, max_left_bond) > max_bond:
        result = truncate_qtt(result, max_bond=max_bond)
    
    return result


def qtt_inner_product(qtt1: QTTState, qtt2: QTTState) -> float:
    """
    Compute ⟨ψ₁|ψ₂⟩ in O(n·d²·r³) time.

    This stays in compressed format - no decompression needed.
    """
    assert qtt1.num_qubits == qtt2.num_qubits

    # Contract from left to right
    # Start with trivial left boundary (on same device as cores)
    device = qtt1.cores[0].device
    dtype = qtt1.cores[0].dtype
    left = torch.ones(1, 1, device=device, dtype=dtype)

    for i in range(qtt1.num_qubits):
        c1 = qtt1.cores[i]  # (r1L, d, r1R)
        c2 = qtt2.cores[i]  # (r2L, d, r2R)

        # Contract: left[r1L, r2L] @ c1[r1L, d, r1R].conj() @ c2[r2L, d, r2R]
        # = new_left[r1R, r2R]

        # Step 1: contract left with c1
        temp = torch.einsum("ij,idk->jdk", left, c1)  # (r2L, d, r1R)

        # Step 2: contract with c2 over physical index
        left = torch.einsum("jdk,jdl->kl", temp, c2)  # (r1R, r2R)

    return left.item()


def qtt_norm(qtt: QTTState) -> float:
    """Compute ||ψ|| = sqrt(⟨ψ|ψ⟩)."""
    return np.sqrt(qtt_inner_product(qtt, qtt))


# =============================================================================
# HIGH-LEVEL OPERATIONS FOR CFD
# =============================================================================


def apply_derivative_qtt(
    qtt: QTTState, axis: int, dx: float, max_bond: int = 64
) -> QTTState:
    """
    Apply derivative operator to QTT state along specified axis.

    For 3D fields, we have separate QTT for each axis.

    Args:
        qtt: Input QTT state
        axis: Axis for differentiation (0, 1, or 2)
        dx: Grid spacing
        max_bond: Maximum bond dimension

    Returns:
        QTT of derivative
    """
    D = derivative_mpo(qtt.num_qubits, dx)
    return apply_mpo(D, qtt, max_bond=max_bond)


def apply_laplacian_qtt(qtt: QTTState, dx: float, max_bond: int = 64) -> QTTState:
    """
    Apply Laplacian to QTT state (1D version).

    For 3D: Δ = Δ_x + Δ_y + Δ_z, apply to each axis and add.
    """
    L = laplacian_mpo(qtt.num_qubits, dx)
    return apply_mpo(L, qtt, max_bond=max_bond)


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================


def dense_to_qtt(tensor: torch.Tensor, max_bond: int = 64) -> QTTState:
    """
    Convert dense tensor to QTT format.

    For a 1D tensor of size 2^n, creates n cores with physical dim 2.
    """
    numel = tensor.numel()
    if numel < 2:
        raise ValueError(f"Tensor must have at least 2 elements, got {numel}")
    
    n = int(np.log2(numel))
    if 2**n != numel:
        raise ValueError(f"Tensor size must be power of 2, got {numel}")

    # Reshape to [2, 2, ..., 2] (n dimensions)
    reshaped = tensor.reshape([2] * n)

    # Sequential SVD from left to right (TT-SVD algorithm)
    cores = []
    current = reshaped.reshape(1, -1)  # (1, 2^n)

    for i in range(n):
        r_left = current.shape[0]
        remaining_size = current.numel() // (r_left * 2)

        # Reshape: (r_left, 2, remaining)
        mat = (
            current.reshape(r_left * 2, remaining_size)
            if remaining_size > 0
            else current.reshape(r_left * 2, 1)
        )

        if i < n - 1:
            # Randomized SVD - note: svd_lowrank returns (U, S, V) not (U, S, Vh)
            q = min(max_bond, min(mat.shape))
            U, S, V = torch.svd_lowrank(mat, q=q, niter=1)

            # Truncate
            rank = min(len(S), max_bond, mat.shape[1])
            rank = max(1, min(rank, (S > 1e-14 * S[0]).sum().item()))

            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]  # V is (n, k), not Vh

            # Store core: (r_left, 2, rank)
            cores.append(U.reshape(r_left, 2, rank))

            # Prepare for next iteration
            current = torch.diag(S) @ V.T  # V.T to get Vh, (rank, remaining)
        else:
            # Last core: (r_left, 2, 1)
            cores.append(mat.reshape(r_left, 2, 1))

    return QTTState(cores=cores, num_qubits=n)


def qtt_to_dense(qtt: QTTState) -> torch.Tensor:
    """
    Convert QTT back to dense tensor.

    Warning: Only use for small tensors! This creates 2^n elements.
    """
    # Contract all cores
    result = qtt.cores[0]  # (1, 2, r1)

    for i in range(1, qtt.num_qubits):
        c = qtt.cores[i]  # (r_{i-1}, 2, r_i)
        # result: (..., r_{i-1}) @ c: (r_{i-1}, 2, r_i) -> (..., 2, r_i)
        result = torch.einsum("...i,ijk->...jk", result, c)

    # Final shape: (1, 2, 2, ..., 2, 1) -> (2, 2, ..., 2)
    return result.squeeze(0).squeeze(-1).reshape(-1)


if __name__ == "__main__":
    # Test pure QTT operations
    print("=" * 60)
    print("PURE QTT OPERATIONS TEST")
    print("=" * 60)

    # Create test function: sin wave
    n_qubits = 10  # 2^10 = 1024 points
    N = 2**n_qubits
    dx = 2 * np.pi / N
    x = torch.linspace(0, 2 * np.pi - dx, N)
    f = torch.sin(x)

    print(f"\nGrid size: {N} points ({n_qubits} qubits)")

    # Convert to QTT
    f_qtt = dense_to_qtt(f, max_bond=32)
    print(f"QTT ranks: {f_qtt.ranks}")

    # Check reconstruction
    f_reconstructed = qtt_to_dense(f_qtt)
    error = torch.norm(f - f_reconstructed) / torch.norm(f)
    print(f"Reconstruction error: {error:.2e}")

    # Test QTT arithmetic
    print("\nTesting QTT arithmetic...")

    # Addition
    g = torch.cos(x)
    g_qtt = dense_to_qtt(g, max_bond=32)
    sum_qtt = qtt_add(f_qtt, g_qtt, max_bond=32)
    sum_dense = qtt_to_dense(sum_qtt)
    sum_exact = f + g
    add_error = torch.norm(sum_dense - sum_exact) / torch.norm(sum_exact)
    print(f"  Addition error: {add_error:.2e}")

    # Scaling
    scaled_qtt = qtt_scale(f_qtt, 2.5)
    scaled_dense = qtt_to_dense(scaled_qtt)
    scale_error = torch.norm(scaled_dense - 2.5 * f) / torch.norm(2.5 * f)
    print(f"  Scaling error: {scale_error:.2e}")

    # Inner product
    ip = qtt_inner_product(f_qtt, g_qtt)
    ip_exact = torch.dot(f, g).item()
    ip_error = (
        abs(ip - ip_exact) / abs(ip_exact)
        if abs(ip_exact) > 1e-10
        else abs(ip - ip_exact)
    )
    print(f"  Inner product: {ip:.6f} (exact: {ip_exact:.6f}, error: {ip_error:.2e})")

    # Norm
    norm_qtt = qtt_norm(f_qtt)
    norm_exact = torch.norm(f).item()
    norm_error = abs(norm_qtt - norm_exact) / norm_exact
    print(f"  Norm: {norm_qtt:.6f} (exact: {norm_exact:.6f}, error: {norm_error:.2e})")

    # Test scaling to HUGE grids
    print("\n" + "=" * 60)
    print("SCALING TEST: HUGE GRIDS")
    print("=" * 60)

    for n_qubits in [20, 25, 30]:
        N = 2**n_qubits

        # Create a random low-rank function (can't store full grid!)
        # We'll create it directly in QTT form
        cores = []
        rank = 8  # Low rank representation

        for i in range(n_qubits):
            r_left = 1 if i == 0 else rank
            r_right = 1 if i == n_qubits - 1 else rank
            core = torch.randn(r_left, 2, r_right) * 0.1
            cores.append(core)

        huge_qtt = QTTState(cores=cores, num_qubits=n_qubits)

        # QTT operations still work!
        norm = qtt_norm(huge_qtt)

        # Create another and add
        cores2 = [torch.randn_like(c) * 0.1 for c in cores]
        huge_qtt2 = QTTState(cores=cores2, num_qubits=n_qubits)

        sum_huge = qtt_add(huge_qtt, huge_qtt2, max_bond=16)

        print(f"  N = 2^{n_qubits} = {N:,} points")
        print(f"    Memory if dense: {N * 8 / 1e9:.1f} GB")
        print(f"    QTT ranks: {sum_huge.ranks[:5]}... (max = 16)")
        print(
            f"    QTT memory: ~{sum(c.numel() for c in sum_huge.cores) * 8 / 1e3:.1f} KB"
        )
        print(f"    Norm computable: {norm:.6f}")
        print()

    print("★ Pure QTT operations complete!")
    print("  This enables 2^60 grids on a laptop (for smooth functions).")
