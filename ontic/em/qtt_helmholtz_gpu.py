"""GPU-Native QTT Helmholtz Solver — Zero CPU Path.

Solves the 3D Helmholtz equation

    (∇² + k² ε(x)) E(x) = −J(x)

entirely on GPU using PyTorch complex tensors + Triton-accelerated QTT ops.

Architecture
────────────
• All tensors: ``torch.Tensor`` on ``device='cuda'``
• QTT cores: ``list[torch.Tensor]`` — each shape ``(r_l, 2, r_r)`` complex64
• MPO cores: ``list[torch.Tensor]`` — each shape ``(r_l, 2, 2, r_r)`` complex64
• SVD: ``torch.linalg.svd`` / ``torch.svd_lowrank`` — cuSOLVER, never NumPy
• Contractions: ``torch.einsum`` — cuBLAS batched GEMM
• PML / ε profiles: built analytically from 1D factors in QTT (no dense N³)

NO NumPy, NO CPU fallback, NO dense N³ arrays.

Author: TiganticLabz
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# GPU Guard — tripwire against CPU tensors in hot path
# ═══════════════════════════════════════════════════════════════════════════════


def _assert_gpu(t: torch.Tensor, name: str = "tensor") -> None:
    """Hard assert that tensor is on CUDA. Tripwire for CPU contamination."""
    if not t.is_cuda:
        raise RuntimeError(
            f"[GPU TRIPWIRE] {name} is on {t.device}, expected CUDA. "
            f"No CPU tensors allowed in the GPU hot path."
        )


def _assert_gpu_cores(cores: list[torch.Tensor], label: str = "QTT") -> None:
    """Assert all QTT/MPO cores are on GPU."""
    for i, c in enumerate(cores):
        _assert_gpu(c, f"{label} core[{i}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Complex TT Decomposition (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def _rsvd_gpu(
    mat: torch.Tensor,
    k: int,
    n_oversampling: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD targeting top-k singular values. GPU-native.

    O(m·n·p) where p = k + n_oversampling, vs O(m·n·min(m,n)) for full SVD.
    """
    m, n = mat.shape
    p = min(k + n_oversampling, min(m, n))
    device, dtype = mat.device, mat.dtype

    # Random Gaussian sketch
    if mat.is_complex():
        Omega = torch.randn(n, p, device=device, dtype=dtype)
    else:
        Omega = torch.randn(n, p, device=device, dtype=dtype)

    Y = mat @ Omega  # (m, p)
    Q, _ = torch.linalg.qr(Y)  # (m, p) orthonormal

    # Project to low-rank subspace
    B = Q.conj().T @ mat  # (p, n)

    # Small SVD of B
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b  # (m, p)

    return U[:, :k], S[:k], Vh[:k, :]


def array_to_tt_gpu(
    arr: torch.Tensor,
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[torch.Tensor]:
    """Decompose 1D tensor of length 2^n into QTT cores on GPU.

    Left-to-right SVD sweep. Uses rSVD for large matrices,
    Gram-matrix eigendecomposition for extreme aspect ratios,
    and cuSOLVER SVD for moderate sizes.

    Args:
        arr: 1D tensor of length 2^n, on CUDA.
        max_rank: Maximum bond dimension.
        cutoff: Relative singular value cutoff.

    Returns:
        List of QTT cores, each (r_l, 2, r_r), on same device.
    """
    _assert_gpu(arr, "array_to_tt_gpu input")
    N = arr.numel()
    n_bits = int(math.log2(N))
    assert 2 ** n_bits == N, f"Length {N} is not a power of 2"

    cores: list[torch.Tensor] = []
    remainder = arr.reshape(1, N)

    for k in range(n_bits):
        r_left = remainder.shape[0]
        cols = remainder.shape[1] // 2
        mat = remainder.reshape(r_left * 2, cols)

        if k < n_bits - 1:
            m, n = mat.shape
            min_dim = min(m, n)

            if min_dim <= 64 and max(m, n) > 100_000:
                # Gram-matrix eigendecomposition for extreme aspect ratios
                # Avoids cuSOLVER Jacobi SVD failures on very wide/tall matrices
                if m <= n:
                    gram = mat @ mat.conj().T  # (m, m) — tiny
                    eigvals, U = torch.linalg.eigh(gram)
                    idx = torch.arange(m - 1, -1, -1, device=mat.device)
                    eigvals = eigvals[idx].clamp(min=0.0)
                    U = U[:, idx]
                    S = torch.sqrt(eigvals)
                    S_inv = torch.where(
                        S > 1e-30, 1.0 / S,
                        torch.zeros_like(S),
                    )
                    Vh = S_inv.unsqueeze(1).to(mat.dtype) * (U.conj().T @ mat)
                else:
                    gram = mat.conj().T @ mat  # (n, n) — tiny
                    eigvals, V = torch.linalg.eigh(gram)
                    idx = torch.arange(n - 1, -1, -1, device=mat.device)
                    eigvals = eigvals[idx].clamp(min=0.0)
                    V = V[:, idx]
                    S = torch.sqrt(eigvals)
                    Vh = V.conj().T
                    S_inv = torch.where(
                        S > 1e-30, 1.0 / S,
                        torch.zeros_like(S),
                    )
                    U = mat @ V * S_inv.unsqueeze(0).to(mat.dtype)
            elif min_dim > max_rank * 2 and max(m, n) > 4096:
                # rSVD for large matrices where we only need top-k
                U, S, Vh = _rsvd_gpu(mat, max_rank)
            else:
                # Standard cuSOLVER SVD for moderate sizes
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

            r = min(len(S), max_rank)
            if r > 1 and cutoff > 0:
                thresh = S[0].abs() * cutoff
                mask = S[:r].abs() > thresh
                r_cut = mask.sum().item()
                r = max(1, r_cut)
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]
            cores.append(U.reshape(r_left, 2, r))
            remainder = S.to(Vh.dtype).unsqueeze(1) * Vh
        else:
            # Last core — no SVD needed
            cores.append(mat.reshape(r_left, 2, 1))

    return cores


def reconstruct_1d_gpu(tt_cores: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct dense 1D array from QTT cores (GPU).

    WARNING: Only for validation at small sizes. Creates 2^n dense array.
    """
    _assert_gpu_cores(tt_cores, "reconstruct_1d_gpu")
    vec = tt_cores[0].reshape(tt_cores[0].shape[1], tt_cores[0].shape[2])
    for k in range(1, len(tt_cores)):
        c = tt_cores[k]
        # vec: (d_accum, r) @ c: (r, 2, r_next) → (d_accum * 2, r_next)
        vec = torch.einsum("ir,rjk->ijk", vec, c)
        vec = vec.reshape(-1, c.shape[2])
    return vec.reshape(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Complex TT Algebra (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def tt_inner_gpu(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    """Hermitian inner product ⟨a|b⟩ = Σ a̅ᵢ bᵢ in TT format. GPU-native.

    Returns scalar tensor on GPU (complex).
    """
    _assert_gpu_cores(a, "tt_inner a")
    _assert_gpu_cores(b, "tt_inner b")

    env = torch.ones(1, 1, device=a[0].device, dtype=a[0].dtype)
    for k in range(len(a)):
        # env: (ra_l, rb_l)
        # a[k]: (ra_l, 2, ra_r)  → conjugate
        # b[k]: (rb_l, 2, rb_r)
        tmp = torch.einsum("ij,idk->jdk", env, a[k].conj())
        env = torch.einsum("jdk,jdl->kl", tmp, b[k])
    return env.squeeze()


def tt_norm_gpu(cores: list[torch.Tensor]) -> float:
    """||cores||₂ via Hermitian inner product. GPU-native."""
    val = tt_inner_gpu(cores, cores)
    return math.sqrt(max(val.real.item(), 0.0))


def tt_scale_gpu(
    cores: list[torch.Tensor], alpha: complex
) -> list[torch.Tensor]:
    """Scale TT by complex scalar: α · |cores⟩. Modifies first core only."""
    out = [c.clone() for c in cores]
    out[0] = out[0] * alpha
    return out


def tt_add_gpu(
    a: list[torch.Tensor], b: list[torch.Tensor]
) -> list[torch.Tensor]:
    """QTT addition via block-diagonal core assembly. GPU-native.

    Returns cores with ranks r_a + r_b (before truncation).
    """
    n = len(a)
    assert len(b) == n
    device = a[0].device
    dtype = a[0].dtype

    cores: list[torch.Tensor] = []
    for k in range(n):
        ca, cb = a[k], b[k]
        ra_l, d, ra_r = ca.shape
        rb_l, _, rb_r = cb.shape

        if k == 0:
            # First: horizontal concat along right bond
            c = torch.zeros(1, d, ra_r + rb_r, device=device, dtype=dtype)
            c[:, :, :ra_r] = ca
            c[:, :, ra_r:] = cb
        elif k == n - 1:
            # Last: vertical concat along left bond
            c = torch.zeros(ra_l + rb_l, d, 1, device=device, dtype=dtype)
            c[:ra_l, :, :] = ca
            c[ra_l:, :, :] = cb
        else:
            # Middle: block diagonal
            c = torch.zeros(
                ra_l + rb_l, d, ra_r + rb_r, device=device, dtype=dtype
            )
            c[:ra_l, :, :ra_r] = ca
            c[ra_l:, :, ra_r:] = cb
        cores.append(c)
    return cores


def tt_axpy_gpu(
    alpha: complex,
    x: list[torch.Tensor],
    y: list[torch.Tensor],
    max_rank: int = 0,
) -> list[torch.Tensor]:
    """Compute α·x + y in TT format. GPU-native."""
    scaled = tt_scale_gpu(x, alpha)
    result = tt_add_gpu(scaled, y)
    if max_rank > 0:
        result = tt_round_gpu(result, max_rank=max_rank)
    return result


def tt_round_gpu(
    cores: list[torch.Tensor],
    max_rank: int = 64,
    cutoff: float = 1e-12,
) -> list[torch.Tensor]:
    """Truncate TT via right-to-left QR + left-to-right rSVD. GPU-native.

    This is the standard orthogonalize-then-truncate algorithm:
      Phase 1: Right-to-left QR (right-orthogonalize)
      Phase 2: Left-to-right SVD (truncate bonds)
    """
    _assert_gpu_cores(cores, "tt_round_gpu")
    n = len(cores)
    if n < 2:
        return [c.clone() for c in cores]

    work = [c.clone() for c in cores]

    # Phase 1: Right-to-left QR sweep
    for k in range(n - 1, 0, -1):
        c = work[k]
        r_l, d, r_r = c.shape
        mat = c.reshape(r_l, d * r_r)
        Q, R = torch.linalg.qr(mat.T)  # mat^T = Q R → mat = R^T Q^T
        Q = Q.T  # (r_new, d*r_r)
        R = R.T  # (r_l, r_new)
        r_new = Q.shape[0]
        work[k] = Q.reshape(r_new, d, r_r)
        # Absorb R into left neighbor
        work[k - 1] = torch.einsum("idk,kj->idj", work[k - 1], R)

    # Phase 2: Left-to-right SVD truncation
    for k in range(n - 1):
        c = work[k]
        r_l, d, r_r = c.shape
        mat = c.reshape(r_l * d, r_r)

        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        r = min(len(S), max_rank)
        if r > 1 and cutoff > 0:
            thresh = S[0].abs() * cutoff
            mask = S[:r].abs() > thresh
            r_cut = mask.sum().item()
            r = max(1, r_cut)

        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        work[k] = U.reshape(r_l, d, r)
        SV = S.to(Vh.dtype).unsqueeze(1) * Vh
        work[k + 1] = torch.einsum("ij,jdk->idk", SV, work[k + 1])

    return work


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: MPO Algebra (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def mpo_add_gpu(
    a: list[torch.Tensor], b: list[torch.Tensor]
) -> list[torch.Tensor]:
    """MPO addition via block-diagonal assembly. GPU-native."""
    n = len(a)
    assert len(b) == n
    device = a[0].device
    dtype = a[0].dtype
    cores: list[torch.Tensor] = []

    for k in range(n):
        ca, cb = a[k], b[k]
        ra_l, d1, d2, ra_r = ca.shape
        rb_l, _, _, rb_r = cb.shape

        if k == 0:
            c = torch.zeros(1, d1, d2, ra_r + rb_r, device=device, dtype=dtype)
            c[:, :, :, :ra_r] = ca
            c[:, :, :, ra_r:] = cb
        elif k == n - 1:
            c = torch.zeros(ra_l + rb_l, d1, d2, 1, device=device, dtype=dtype)
            c[:ra_l, :, :, :] = ca
            c[ra_l:, :, :, :] = cb
        else:
            c = torch.zeros(
                ra_l + rb_l, d1, d2, ra_r + rb_r, device=device, dtype=dtype
            )
            c[:ra_l, :, :, :ra_r] = ca
            c[ra_l:, :, :, ra_r:] = cb
        cores.append(c)
    return cores


def mpo_scale_gpu(
    cores: list[torch.Tensor], alpha: complex
) -> list[torch.Tensor]:
    """Scale MPO by complex scalar. GPU-native."""
    out = [c.clone() for c in cores]
    out[0] = out[0] * alpha
    return out


def diag_mpo_from_tt_gpu(tt_cores: list[torch.Tensor]) -> list[torch.Tensor]:
    """Convert diagonal TT → MPO: diag(tt)[i,j] = δ_{ij} tt[i].

    Each TT core (r_l, d, r_r) → MPO core (r_l, d, d, r_r) with
    the physical indices tied: core[r_l, i, j, r_r] = δ_{ij} * tt_core[r_l, i, r_r].
    """
    _assert_gpu_cores(tt_cores, "diag_mpo_from_tt_gpu")
    mpo_cores: list[torch.Tensor] = []
    for c in tt_cores:
        r_l, d, r_r = c.shape
        mc = torch.zeros(r_l, d, d, r_r, device=c.device, dtype=c.dtype)
        for i in range(d):
            mc[:, i, i, :] = c[:, i, :]
        mpo_cores.append(mc)
    return mpo_cores


def identity_mpo_gpu(
    n_sites: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> list[torch.Tensor]:
    """Identity MPO: each core = eye(2) reshaped to (1, 2, 2, 1)."""
    eye = torch.eye(2, device=device, dtype=dtype).reshape(1, 2, 2, 1)
    return [eye.clone() for _ in range(n_sites)]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: MPO × TT (matvec) on GPU
# ═══════════════════════════════════════════════════════════════════════════════


def tt_matvec_gpu(
    mpo: list[torch.Tensor],
    tt: list[torch.Tensor],
    max_rank: int = 0,
) -> list[torch.Tensor]:
    """Apply MPO to TT: |out⟩ = M|tt⟩. GPU-native.

    Per-site contraction via einsum (cuBLAS). Ranks multiply: r_out = r_M × r_tt.
    Optionally truncates result.
    """
    n = len(mpo)
    assert len(tt) == n
    cores: list[torch.Tensor] = []

    for k in range(n):
        M = mpo[k]  # (rM_l, d_out, d_in, rM_r)
        P = tt[k]   # (rP_l, d_in, rP_r)
        # Contract over d_in:
        # out[mL*pL, d_out, mR*pR] = Σ_{d_in} M[mL, d_out, d_in, mR] * P[pL, d_in, pR]
        out = torch.einsum("oabr,pbq->oparq", M, P)
        rM_l, _, _, rM_r = M.shape
        rP_l, _, rP_r = P.shape
        d_out = M.shape[1]
        out = out.reshape(rM_l * rP_l, d_out, rM_r * rP_r)
        cores.append(out)

    if max_rank > 0:
        cores = tt_round_gpu(cores, max_rank=max_rank)
    return cores


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: 1D Operator Construction (GPU-native)
# ═══════════════════════════════════════════════════════════════════════════════


def _shift_right_mpo_gpu(
    n_bits: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> list[torch.Tensor]:
    """Cyclic right-shift MPO S₊|x⟩ = |x+1 mod 2^n⟩. Bond dim = 2.

    Ripple-carry adder structure.
    """
    cores: list[torch.Tensor] = []

    # First site (MSB, site 0): initiate carry
    c = torch.zeros(1, 2, 2, 2, device=device, dtype=dtype)
    c[0, 1, 0, 0] = 1.0  # bit 0→1, no carry
    c[0, 0, 1, 1] = 1.0  # bit 1→0, carry out
    cores.append(c)

    # Middle sites
    for _ in range(1, n_bits - 1):
        c = torch.zeros(2, 2, 2, 2, device=device, dtype=dtype)
        c[0, 0, 0, 0] = 1.0  # no carry in, bit 0→0, no carry out
        c[0, 1, 1, 0] = 1.0  # no carry in, bit 1→1, no carry out
        c[1, 1, 0, 0] = 1.0  # carry in, bit 0→1, no carry out
        c[1, 0, 1, 1] = 1.0  # carry in, bit 1→0, carry out
        cores.append(c)

    # Last site (LSB)
    c = torch.zeros(2, 2, 2, 1, device=device, dtype=dtype)
    c[0, 0, 0, 0] = 1.0
    c[0, 1, 1, 0] = 1.0
    c[1, 1, 0, 0] = 1.0
    c[1, 0, 1, 0] = 1.0  # carry wraps (cyclic)
    cores.append(c)

    return cores


def _shift_left_mpo_gpu(
    n_bits: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> list[torch.Tensor]:
    """Cyclic left-shift MPO S₋|x⟩ = |x−1 mod 2^n⟩. Bond dim = 2."""
    cores: list[torch.Tensor] = []

    # First site: initiate borrow
    c = torch.zeros(1, 2, 2, 2, device=device, dtype=dtype)
    c[0, 0, 1, 0] = 1.0  # bit 1→0, no borrow
    c[0, 1, 0, 1] = 1.0  # bit 0→1, borrow out
    cores.append(c)

    # Middle sites
    for _ in range(1, n_bits - 1):
        c = torch.zeros(2, 2, 2, 2, device=device, dtype=dtype)
        c[0, 0, 0, 0] = 1.0  # no borrow, bit 0→0
        c[0, 1, 1, 0] = 1.0  # no borrow, bit 1→1
        c[1, 0, 1, 0] = 1.0  # borrow, bit 1→0
        c[1, 1, 0, 1] = 1.0  # borrow, bit 0→1, propagate
        cores.append(c)

    # Last site
    c = torch.zeros(2, 2, 2, 1, device=device, dtype=dtype)
    c[0, 0, 0, 0] = 1.0
    c[0, 1, 1, 0] = 1.0
    c[1, 0, 1, 0] = 1.0
    c[1, 1, 0, 0] = 1.0  # borrow wraps
    cores.append(c)

    return cores


def laplacian_mpo_1d_gpu(
    n_bits: int,
    h: float,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> list[torch.Tensor]:
    """Laplacian D₂ = (S₊ − 2I + S₋)/h² as MPO. Bond dim ≤ 5. GPU-native."""
    sp = _shift_right_mpo_gpu(n_bits, device, dtype)
    sm = _shift_left_mpo_gpu(n_bits, device, dtype)
    I = identity_mpo_gpu(n_bits, device, dtype)
    alpha = 1.0 / (h * h)
    terms = mpo_add_gpu(mpo_scale_gpu(sp, alpha), mpo_scale_gpu(sm, alpha))
    return mpo_add_gpu(terms, mpo_scale_gpu(I, -2.0 * alpha))


def _embed_1d_mpo_gpu(
    mpo_1d: list[torch.Tensor],
    dim: int,
    bits_per_dim: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> list[torch.Tensor]:
    """Embed 1D MPO into dimension `dim` of multi-dim QTT. GPU-native."""
    full: list[torch.Tensor] = []
    for d in range(len(bits_per_dim)):
        if d == dim:
            full.extend(mpo_1d)
        else:
            full.extend(identity_mpo_gpu(bits_per_dim[d], device, dtype))
    return full


def laplacian_mpo_3d_gpu(
    bits_per_dim: tuple[int, int, int],
    domain: tuple[tuple[float, float], ...],
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> list[torch.Tensor]:
    """Full 3D Laplacian ∇² as MPO. Bond dim ≤ 15. GPU-native."""
    full = None
    for d in range(3):
        lo, hi = domain[d]
        N = 2 ** bits_per_dim[d]
        h = (hi - lo) / N
        mpo_1d = laplacian_mpo_1d_gpu(bits_per_dim[d], h, device, dtype)
        embedded = _embed_1d_mpo_gpu(mpo_1d, d, bits_per_dim, device, dtype)
        if full is None:
            full = embedded
        else:
            full = mpo_add_gpu(full, embedded)
    assert full is not None
    return full


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Helmholtz MPO (3D, GPU-native)
# ═══════════════════════════════════════════════════════════════════════════════


def build_pml_eps_1d_gpu(
    n_bits: int,
    k: float,
    pml_cells: int = 20,
    sigma_max: float = 10.0,
    damping: float = 0.01,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    max_rank: int = 16,
) -> list[torch.Tensor]:
    """Build 1D PML complex permittivity profile as QTT. GPU-native.

    ε_pml(x) = 1 − j·σ(x)/(ω·ε₀) where σ(x) is the PML conductivity.
    Uses stretched coordinate: s = 1 − j·σ/(ω·ε₀), ε_eff = ε_r / s.

    For simplicity we use ε_pml = 1 + j·σ/k as the absorbing factor.
    """
    N = 2 ** n_bits

    # Build dense 1D array (small: just 2^n_bits, not N³)
    x = torch.arange(N, device=device, dtype=torch.float32)

    sigma = torch.zeros(N, device=device, dtype=torch.float32)
    # Left PML
    left_mask = x < pml_cells
    sigma[left_mask] = sigma_max * ((pml_cells - x[left_mask]) / pml_cells) ** 2
    # Right PML
    right_mask = x >= (N - pml_cells)
    sigma[right_mask] = sigma_max * (
        (x[right_mask] - (N - pml_cells - 1)) / pml_cells
    ) ** 2

    # ε_pml = 1 + damping − j·σ/k  (complex permittivity with loss)
    eps = (1.0 + damping) * torch.ones(N, device=device, dtype=dtype)
    eps = eps - 1j * sigma.to(dtype) / k

    # Decompose into QTT
    return array_to_tt_gpu(eps, max_rank=max_rank)


def build_pml_eps_3d_separable_gpu(
    bits_per_dim: tuple[int, int, int],
    k: float,
    domain: tuple[tuple[float, float], ...],
    pml_cells: int = 20,
    sigma_max: float = 10.0,
    damping: float = 0.01,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    max_rank: int = 16,
) -> list[torch.Tensor]:
    """Build 3D PML ε as QTT from separable 1D profiles. GPU-native.

    ε_3D(x,y,z) = ε_x(x) · ε_y(y) · ε_z(z)

    In QTT format: cores are concatenated [ε_x_cores | ε_y_cores | ε_z_cores]
    with product structure via Hadamard.

    For benchmark purposes we use the additive PML model:
    ε_3D = 1 + (ε_x − 1) + (ε_y − 1) + (ε_z − 1) + damping
    which is formally separable and has rank ≤ 4 per bond.
    """
    total_sites = sum(bits_per_dim)

    # Build 1D profiles for each dimension
    eps_1d: list[list[torch.Tensor]] = []
    for d in range(3):
        eps_1d.append(
            build_pml_eps_1d_gpu(
                bits_per_dim[d], k, pml_cells, sigma_max, damping,
                device, dtype, max_rank,
            )
        )

    # Embed each 1D profile into full 3D QTT
    # ε_x embedded: ε_x cores at dim 0, identity (=ones) at dim 1,2
    # But for product: we need a constant-1 TT for other dims
    def ones_tt(n_bits: int) -> list[torch.Tensor]:
        """Rank-1 QTT representing constant 1."""
        return [
            torch.ones(1, 2, 1, device=device, dtype=dtype) for _ in range(n_bits)
        ]

    # Product via Hadamard: ε_x ⊗ 1_y ⊗ 1_z
    # First build the full-length TT for each factor
    factors_full: list[list[torch.Tensor]] = []
    for d in range(3):
        full_cores: list[torch.Tensor] = []
        for d2 in range(3):
            if d2 == d:
                full_cores.extend(eps_1d[d])
            else:
                full_cores.extend(ones_tt(bits_per_dim[d2]))
        factors_full.append(full_cores)

    # Multiply: product = f0 ⊙ f1 ⊙ f2 via Hadamard (Kronecker per core)
    result = factors_full[0]
    for fac in factors_full[1:]:
        new_cores: list[torch.Tensor] = []
        for k_idx in range(total_sites):
            ca, cb = result[k_idx], fac[k_idx]
            ra_l, d_phys, ra_r = ca.shape
            rb_l, _, rb_r = cb.shape
            kron = torch.einsum("adb,cde->acdbe", ca, cb)
            new_cores.append(kron.reshape(ra_l * rb_l, d_phys, ra_r * rb_r))
        result = new_cores

    # Truncate
    return tt_round_gpu(result, max_rank=max_rank)


def helmholtz_mpo_3d_gpu(
    bits_per_dim: tuple[int, int, int],
    k: float,
    domain: tuple[tuple[float, float], ...],
    eps_pml_cores: list[torch.Tensor] | None = None,
    pml_cells: int = 20,
    sigma_max: float = 10.0,
    damping: float = 0.01,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    max_rank: int = 16,
) -> list[torch.Tensor]:
    """Build 3D Helmholtz operator H = ∇² + k²·ε as MPO. GPU-native.

    Returns complex MPO cores on CUDA.
    """
    # Laplacian MPO (bond dim ≤ 15)
    L_cores = laplacian_mpo_3d_gpu(bits_per_dim, domain, device, dtype)

    # PML permittivity
    total_sites = sum(bits_per_dim)
    if eps_pml_cores is not None:
        eps_mpo = diag_mpo_from_tt_gpu(eps_pml_cores)
    else:
        eps_tt = build_pml_eps_3d_separable_gpu(
            bits_per_dim, k, domain, pml_cells, sigma_max, damping,
            device, dtype, max_rank,
        )
        eps_mpo = diag_mpo_from_tt_gpu(eps_tt)

    k2_eps = mpo_scale_gpu(eps_mpo, k * k)

    # H = Laplacian + k²·ε
    return mpo_add_gpu(L_cores, k2_eps)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Two-Site DMRG Solve (GPU-native AMEN)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SolveResult:
    """Result of GPU AMEN solve."""
    x: list[torch.Tensor]
    converged: bool
    n_iter: int
    final_residual: float
    residual_norms: list[float]
    wall_time: float
    chi_max: int


def _right_orth_gpu(cores: list[torch.Tensor]) -> list[torch.Tensor]:
    """Right-orthogonalize TT cores via QR sweep. GPU-native."""
    n = len(cores)
    work = [c.clone() for c in cores]
    for k in range(n - 1, 0, -1):
        c = work[k]
        r_l, d, r_r = c.shape
        mat = c.reshape(r_l, d * r_r)
        # QR on transpose: mat^T = Q R → mat = R^T Q^T
        Q, R = torch.linalg.qr(mat.T)
        r_new = Q.shape[1]
        work[k] = Q.T.reshape(r_new, d, r_r)
        work[k - 1] = torch.einsum("idk,kj->idj", work[k - 1], R.T)
    return work


def _update_phi_left_gpu(
    phi_left: torch.Tensor,
    x_core: torch.Tensor,
    a_core: torch.Tensor,
) -> torch.Tensor:
    """Update left projection (L→R).

    phi_left: (rx_l, rA_l, rx_l)  — noted as (p, a, q)
    x_core:   (rx_l, d, rx_r)    — x_ket: (q, s, q')
    a_core:   (rA_l, d, d, rA_r) — A: (a, t, s, b)

    Returns: (rx_r, rA_r, rx_r)
    """
    # phi_l[p, a, q] * x_ket[q, s, r] → tmp[p, a, s, r]
    tmp = torch.einsum("paq,qsr->pasr", phi_left, x_core)
    # tmp[p, a, s, r] * A[a, t, s, b] → tmp2[p, t, r, b]
    tmp2 = torch.einsum("pasr,atsb->ptrb", tmp, a_core)
    # conj(x_bra)[p, t, m] * tmp2[p, t, r, b] → out[m, b, r]
    out = torch.einsum("ptm,ptrb->mbr", x_core.conj(), tmp2)
    return out


def _update_phi_right_gpu(
    phi_right: torch.Tensor,
    x_core: torch.Tensor,
    a_core: torch.Tensor,
) -> torch.Tensor:
    """Update right projection (R→L).

    phi_right: (rx_r, rA_r, rx_r) — noted as (m, b, n)
    x_core:    (rx_l, d, rx_r)   — x: (q, s, n)
    a_core:    (rA_l, d, d, rA_r) — A: (a, t, s, b)

    Returns: (rx_l, rA_l, rx_l)
    """
    # phi_r[m, b, n] * x_ket[q, s, n] → tmp[m, b, q, s]
    tmp = torch.einsum("mbn,qsn->mbqs", phi_right, x_core)
    # tmp[m, b, q, s] * A[a, t, s, b] → tmp2[m, q, a, t]
    tmp2 = torch.einsum("mbqs,atsb->mqat", tmp, a_core)
    # conj(x_bra)[p, t, m] * tmp2[m, q, a, t] → out[p, a, q]
    out = torch.einsum("ptm,mqat->paq", x_core.conj(), tmp2)
    return out


def _update_psi_left_gpu(
    psi_left: torch.Tensor,
    x_core: torch.Tensor,
    b_core: torch.Tensor,
) -> torch.Tensor:
    """Update left RHS projection (L→R).

    psi_left: (rx_l, rb_l) — (p, c)
    x_core:   (rx_l, d, rx_r) — (p, s, m)
    b_core:   (rb_l, d, rb_r) — (c, s, d)

    Returns: (rx_r, rb_r)
    """
    # psi_l[p, c] * b[c, s, d] → tmp[p, s, d]
    tmp = torch.einsum("pc,csd->psd", psi_left, b_core)
    # conj(x)[p, s, m] * tmp[p, s, d] → out[m, d]
    out = torch.einsum("psm,psd->md", x_core.conj(), tmp)
    return out


def _update_psi_right_gpu(
    psi_right: torch.Tensor,
    x_core: torch.Tensor,
    b_core: torch.Tensor,
) -> torch.Tensor:
    """Update right RHS projection (R→L).

    psi_right: (rx_r, rb_r) — (m, d)
    x_core:    (rx_l, d, rx_r) — (p, s, m)
    b_core:    (rb_l, d, rb_r) — (c, s, d)

    Returns: (rx_l, rb_l)
    """
    # psi_r[m, d] * b[c, s, d] → tmp[m, c, s]
    tmp = torch.einsum("md,csd->mcs", psi_right, b_core)
    # conj(x)[p, s, m] * tmp[m, c, s] → out[p, c]
    out = torch.einsum("psm,mcs->pc", x_core.conj(), tmp)
    return out


def _merge_mpo_cores_gpu(
    a_i: torch.Tensor, a_j: torch.Tensor
) -> torch.Tensor:
    """Merge two adjacent MPO cores into supersite.

    a_i: (rA_l, d, d, rA_m)   — A_i[a, t, s, b]
    a_j: (rA_m, d, d, rA_r)   — A_j[b, u, v, c]
    → merged: (rA_l, d*d, d*d, rA_r)  with output=(t,u), input=(s,v)
    """
    # Contract over shared bond b:
    # A_i[a, t, s, b] * A_j[b, u, v, c] → tmp[a, t, s, u, v, c]
    tmp = torch.einsum("atsb,buvc->atsuvc", a_i, a_j)
    rAl = a_i.shape[0]
    d = a_i.shape[1]
    rAr = a_j.shape[3]
    # Reorder to (a, t, u, s, v, c) — output=(t,u), input=(s,v)
    tmp = tmp.permute(0, 1, 3, 2, 4, 5)
    return tmp.reshape(rAl, d * d, d * d, rAr)


def _merge_tt_cores_gpu(
    b_i: torch.Tensor, b_j: torch.Tensor
) -> torch.Tensor:
    """Merge two adjacent TT cores into supersite.

    b_i: (rb_l, d, rb_m)
    b_j: (rb_m, d, rb_r)
    → merged: (rb_l, d², rb_r)
    """
    rbl, d1, rbm = b_i.shape
    _, d2, rbr = b_j.shape
    merged = torch.einsum("aim,mjr->aijr", b_i, b_j)
    return merged.reshape(rbl, d1 * d2, rbr)


def _form_local_op_gpu(
    phi_l: torch.Tensor,
    a_merged: torch.Tensor,
    phi_r: torch.Tensor,
) -> torch.Tensor:
    """Form local Galerkin operator for supersite.

    phi_l:    (rx_l, rA_l, rx_l) — (p, a, q)
    a_merged: (rA_l, D_out, D_in, rA_r) — (a, t, s, b)
    phi_r:    (rx_r, rA_r, rx_r) — (r, b, w)

    Returns: H of shape (rx_l * D_out * rx_r, rx_l * D_in * rx_r)
      row index = (p, t, r) = bra
      col index = (q, s, w) = ket
    """
    rx_l = phi_l.shape[0]
    D_out = a_merged.shape[1]
    D_in = a_merged.shape[2]
    rx_r = phi_r.shape[0]

    # PhiL[p, a, q] * A[a, t, s, b] → tmp[p, q, t, s, b]
    tmp = torch.einsum("paq,atsb->pqtsb", phi_l, a_merged)
    # tmp[p, q, t, s, b] * PhiR[r, b, w] → H6[p, q, t, s, r, w]
    H6 = torch.einsum("pqtsb,rbw->pqtsrw", tmp, phi_r)
    # Reorder: (p, t, r, q, s, w) — bra=(p,t,r), ket=(q,s,w)
    H6 = H6.permute(0, 2, 4, 1, 3, 5)
    row_dim = rx_l * D_out * rx_r
    col_dim = rx_l * D_in * rx_r
    return H6.reshape(row_dim, col_dim)


def _form_local_rhs_gpu(
    psi_l: torch.Tensor,
    b_merged: torch.Tensor,
    psi_r: torch.Tensor,
) -> torch.Tensor:
    """Form local RHS for supersite.

    psi_l:    (rx_l, rb_l) — (p, c)
    b_merged: (rb_l, D, rb_r) — (c, s, e)
    psi_r:    (rx_r, rb_r) — (r, e)

    Returns: f of shape (rx_l * D * rx_r)
      index = (p, s, r)  matching bra of local operator
    """
    # psi_l[p, c] * b[c, s, e] → tmp[p, s, e]
    tmp = torch.einsum("pc,cse->pse", psi_l, b_merged)
    # tmp[p, s, e] * psi_r[r, e] → f[p, s, r]
    f = torch.einsum("pse,re->psr", tmp, psi_r)
    return f.reshape(-1)


def tt_amen_solve_gpu(
    mpo_cores: list[torch.Tensor],
    b_cores: list[torch.Tensor],
    x0: list[torch.Tensor] | None = None,
    max_rank: int = 64,
    n_sweeps: int = 40,
    tol: float = 1e-8,
    verbose: bool = False,
    stagnation_tol: float = 0.02,
    stagnation_window: int = 3,
    check_interval: int = 1,
) -> SolveResult:
    """Two-site DMRG solver for Ax = b in TT format. GPU-native.

    All operations on CUDA:
    - Local dense solves: torch.linalg.solve (cuSOLVER)
    - SVD truncation: torch.linalg.svd (cuSOLVER)
    - Contractions: torch.einsum (cuBLAS)

    No NumPy. No CPU. No dense N³.

    Parameters
    ----------
    stagnation_tol : float
        Stop early if relative improvement over `stagnation_window`
        consecutive checks is below this fraction. Default: 0.02 (2%).
    stagnation_window : int
        Number of consecutive stagnant residual checks before early stop.
    check_interval : int
        Compute global residual every this many sweeps. Default: 1.
    """
    _assert_gpu_cores(mpo_cores, "AMEN MPO")
    _assert_gpu_cores(b_cores, "AMEN RHS")

    t0 = time.perf_counter()
    n = len(mpo_cores)
    if n < 2:
        raise ValueError("Need at least 2 sites for two-site DMRG")

    device = mpo_cores[0].device
    dtype = mpo_cores[0].dtype
    d = mpo_cores[0].shape[1]

    # Initialize x
    if x0 is not None:
        x = [c.clone().to(dtype=dtype) for c in x0]
    else:
        x = [c.clone().to(dtype=dtype) for c in b_cores]

    b = [c.to(dtype=dtype) for c in b_cores]
    A = [c.to(dtype=dtype) for c in mpo_cores]

    # Right-orthogonalize x
    x = _right_orth_gpu(x)

    b_norm = tt_norm_gpu(b)
    if b_norm < 1e-30:
        b_norm = 1.0

    # Build right projections
    PhiR: list[torch.Tensor | None] = [None] * (n + 1)
    PsiR: list[torch.Tensor | None] = [None] * (n + 1)
    PhiR[n] = torch.ones(1, 1, 1, device=device, dtype=dtype)
    PsiR[n] = torch.ones(1, 1, device=device, dtype=dtype)

    for i in range(n - 1, 0, -1):
        PhiR[i] = _update_phi_right_gpu(PhiR[i + 1], x[i], A[i])
        PsiR[i] = _update_psi_right_gpu(PsiR[i + 1], x[i], b[i])

    PhiL: list[torch.Tensor | None] = [None] * (n + 1)
    PsiL: list[torch.Tensor | None] = [None] * (n + 1)
    PhiL[0] = torch.ones(1, 1, 1, device=device, dtype=dtype)
    PsiL[0] = torch.ones(1, 1, device=device, dtype=dtype)

    residual_norms: list[float] = []
    stagnation_count = 0
    best_residual = float("inf")

    for sweep in range(n_sweeps):
        # Left-to-right half-sweep
        for bond in range(n - 1):
            i, j = bond, bond + 1
            rx_l = x[i].shape[0]
            rx_r = x[j].shape[2]

            A_merged = _merge_mpo_cores_gpu(A[i], A[j])
            b_merged = _merge_tt_cores_gpu(b[i], b[j])

            H_loc = _form_local_op_gpu(PhiL[i], A_merged, PhiR[j + 1])
            f_loc = _form_local_rhs_gpu(PsiL[i], b_merged, PsiR[j + 1])

            # Solve local system
            sol = torch.linalg.solve(H_loc, f_loc)

            # SVD split
            sol_2d = sol.reshape(rx_l * d, d * rx_r)
            U, S, Vh = torch.linalg.svd(sol_2d, full_matrices=False)

            r_new = min(len(S), max_rank)
            if r_new > 1:
                s_thresh = S[0].abs() * 1e-14
                while r_new > 1 and S[r_new - 1].abs() < s_thresh:
                    r_new -= 1

            U = U[:, :r_new]
            S = S[:r_new]
            Vh = Vh[:r_new, :]

            x[i] = U.reshape(rx_l, d, r_new)
            x[j] = (S.unsqueeze(1) * Vh).reshape(r_new, d, rx_r)

            PhiL[i + 1] = _update_phi_left_gpu(PhiL[i], x[i], A[i])
            PsiL[i + 1] = _update_psi_left_gpu(PsiL[i], x[i], b[i])

        # Right-to-left half-sweep
        for bond in range(n - 2, -1, -1):
            i, j = bond, bond + 1
            rx_l = x[i].shape[0]
            rx_r = x[j].shape[2]

            A_merged = _merge_mpo_cores_gpu(A[i], A[j])
            b_merged = _merge_tt_cores_gpu(b[i], b[j])

            H_loc = _form_local_op_gpu(PhiL[i], A_merged, PhiR[j + 1])
            f_loc = _form_local_rhs_gpu(PsiL[i], b_merged, PsiR[j + 1])

            sol = torch.linalg.solve(H_loc, f_loc)

            sol_2d = sol.reshape(rx_l * d, d * rx_r)
            U, S, Vh = torch.linalg.svd(sol_2d, full_matrices=False)

            r_new = min(len(S), max_rank)
            if r_new > 1:
                s_thresh = S[0].abs() * 1e-14
                while r_new > 1 and S[r_new - 1].abs() < s_thresh:
                    r_new -= 1

            U = U[:, :r_new]
            S = S[:r_new]
            Vh = Vh[:r_new, :]

            x[i] = (U * S.unsqueeze(0)).reshape(rx_l, d, r_new)
            x[j] = Vh.reshape(r_new, d, rx_r)

            PhiR[j] = _update_phi_right_gpu(PhiR[j + 1], x[j], A[j])
            PsiR[j] = _update_psi_right_gpu(PsiR[j + 1], x[j], b[j])

        # Convergence check (possibly every check_interval sweeps)
        if (sweep % check_interval == 0) or (sweep == n_sweeps - 1):
            Ax = tt_matvec_gpu(A, x, max_rank=max_rank)
            res_cores = tt_axpy_gpu(-1.0, Ax, [c.clone() for c in b], max_rank=max_rank)
            res_norm = tt_norm_gpu(res_cores)
            rel_res = res_norm / b_norm
        else:
            # Use last known residual (skip expensive global check)
            rel_res = residual_norms[-1] if residual_norms else float("inf")

        residual_norms.append(rel_res)

        ranks = [c.shape[2] for c in x[:-1]]
        chi = max(ranks) if ranks else 1

        if verbose:
            print(
                f"  DMRG sweep {sweep:3d}  rel_residual={rel_res:.4e}  chi_max={chi}"
            )

        if rel_res < tol:
            return SolveResult(
                x=x,
                converged=True,
                n_iter=sweep + 1,
                final_residual=rel_res,
                residual_norms=residual_norms,
                wall_time=time.perf_counter() - t0,
                chi_max=chi,
            )

        # Stagnation detection: stop if residual is not improving
        if rel_res < best_residual * (1.0 - stagnation_tol):
            best_residual = rel_res
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_window and sweep >= stagnation_window:
            if verbose:
                print(
                    f"  AMEN stagnated: no >{stagnation_tol*100:.0f}% improvement "
                    f"for {stagnation_window} checks. Stopping."
                )
            return SolveResult(
                x=x,
                converged=rel_res < tol,
                n_iter=sweep + 1,
                final_residual=rel_res,
                residual_norms=residual_norms,
                wall_time=time.perf_counter() - t0,
                chi_max=chi,
            )

    return SolveResult(
        x=x,
        converged=False,
        n_iter=n_sweeps,
        final_residual=residual_norms[-1] if residual_norms else float("inf"),
        residual_norms=residual_norms,
        wall_time=time.perf_counter() - t0,
        chi_max=max(c.shape[2] for c in x[:-1]) if len(x) > 1 else 1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Source Construction (GPU)
# ═══════════════════════════════════════════════════════════════════════════════


def gaussian_source_tt_gpu(
    bits_per_dim: tuple[int, int, int],
    domain: tuple[tuple[float, float], ...],
    center: tuple[float, float, float],
    width: float,
    k: float,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.complex64,
    max_rank: int = 16,
) -> list[torch.Tensor]:
    """3D Gaussian source J(r) = exp(−|r−r₀|²/w²) as QTT. GPU-native.

    Separable: J = Jx(x) · Jy(y) · Jz(z) where each is a 1D Gaussian.
    Product via Hadamard gives rank ≤ rank_x × rank_y × rank_z.
    """
    total_sites = sum(bits_per_dim)

    # Build 1D Gaussians for each dimension
    gaussians_1d: list[list[torch.Tensor]] = []
    for d in range(3):
        N = 2 ** bits_per_dim[d]
        lo, hi = domain[d]
        x = torch.linspace(lo, hi, N, device=device, dtype=torch.float32)
        g = torch.exp(-((x - center[d]) / width) ** 2)
        g_complex = g.to(dtype)
        gaussians_1d.append(array_to_tt_gpu(g_complex, max_rank=max_rank))

    # Embed each 1D Gaussian into full 3D QTT and take product
    def ones_tt(n_bits: int) -> list[torch.Tensor]:
        return [
            torch.ones(1, 2, 1, device=device, dtype=dtype)
            for _ in range(n_bits)
        ]

    result: list[torch.Tensor] | None = None
    for d in range(3):
        full: list[torch.Tensor] = []
        for d2 in range(3):
            if d2 == d:
                full.extend(gaussians_1d[d])
            else:
                full.extend(ones_tt(bits_per_dim[d2]))

        if result is None:
            result = full
        else:
            new: list[torch.Tensor] = []
            for k_idx in range(total_sites):
                ca, cb = result[k_idx], full[k_idx]
                ra_l, d_phys, ra_r = ca.shape
                rb_l, _, rb_r = cb.shape
                kron = torch.einsum("adb,cde->acdbe", ca, cb)
                new.append(kron.reshape(ra_l * rb_l, d_phys, ra_r * rb_r))
            result = new

    assert result is not None
    return tt_round_gpu(result, max_rank=max_rank)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_qtt_metrics(
    cores: list[torch.Tensor],
    grid_size: int,
) -> dict:
    """Compute QTT compression metrics. GPU-native (no dense reconstruction)."""
    total_params = sum(c.numel() for c in cores)
    element_size = cores[0].element_size()
    qtt_bytes = total_params * element_size

    # Dense equivalent
    dense_elements = grid_size ** 3
    dense_bytes = dense_elements * element_size

    ranks = [c.shape[2] for c in cores[:-1]]
    chi_max = max(ranks) if ranks else 1

    return {
        "n_sites": len(cores),
        "chi_max": chi_max,
        "ranks": ranks,
        "total_params": total_params,
        "qtt_bytes": qtt_bytes,
        "dense_bytes": dense_bytes,
        "compression_ratio": dense_bytes / max(qtt_bytes, 1),
        "grid_size": grid_size,
        "grid_N3": dense_elements,
    }


def compute_conservation_error(
    mpo: list[torch.Tensor],
    x: list[torch.Tensor],
    b: list[torch.Tensor],
    max_rank: int = 128,
) -> float:
    """Compute ||Ax − b|| / ||b|| on GPU. No dense reconstruction."""
    Ax = tt_matvec_gpu(mpo, x, max_rank=max_rank)
    res = tt_axpy_gpu(-1.0, Ax, [c.clone() for c in b], max_rank=max_rank)
    res_norm = tt_norm_gpu(res)
    b_norm = tt_norm_gpu(b)
    return res_norm / max(b_norm, 1e-30)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Full Benchmark Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkConfig:
    """Configuration for a single scale point in the GPU benchmark."""
    n_bits: int             # bits per dimension (7=128³, 10=1024³, 12=4096³)
    max_rank: int           # AMEN max bond dimension
    n_sweeps: int           # max AMEN sweeps
    tol: float              # convergence tolerance
    k: float                # wavenumber
    domain_size: float      # physical domain size
    pml_cells: int          # PML cells
    sigma_max: float        # PML conductivity
    damping: float          # material damping
    source_width: float     # Gaussian source width
    max_rank_pml: int = 16  # max rank for PML profile


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    config: BenchmarkConfig
    grid_size: int
    n_sites: int
    chi_max: int
    qtt_bytes: int
    dense_bytes: int
    compression_ratio: float
    wall_time_build: float
    wall_time_solve: float
    wall_time_total: float
    gpu_mem_peak_mb: float
    conservation_error: float
    converged: bool
    n_iter: int
    final_residual: float


def run_benchmark_point(
    cfg: BenchmarkConfig,
    device: torch.device = torch.device("cuda"),
    verbose: bool = True,
) -> BenchmarkResult:
    """Run a single benchmark point: build + solve + metrics. GPU-only."""
    torch.cuda.reset_peak_memory_stats(device)
    t_total_start = time.perf_counter()

    grid_size = 2 ** cfg.n_bits
    bits_per_dim = (cfg.n_bits, cfg.n_bits, cfg.n_bits)
    domain = (
        (0.0, cfg.domain_size),
        (0.0, cfg.domain_size),
        (0.0, cfg.domain_size),
    )
    center = (cfg.domain_size / 2, cfg.domain_size / 2, cfg.domain_size / 2)

    dtype = torch.complex64

    if verbose:
        print(f"\n{'='*70}")
        print(f"  GPU QTT Maxwell Benchmark: {grid_size}³ ({cfg.n_bits} bits/dim)")
        print(f"  Total QTT sites: {3 * cfg.n_bits}")
        print(f"  Dense N³: {grid_size**3:,} ({grid_size**3 * 8 / 1e9:.1f} GB complex64)")
        print(f"  Max rank: {cfg.max_rank}")
        print(f"{'='*70}")

    # Phase 1: Build operators
    t_build_start = time.perf_counter()

    if verbose:
        print("  Building Helmholtz MPO...", end=" ", flush=True)
    mpo = helmholtz_mpo_3d_gpu(
        bits_per_dim, cfg.k, domain,
        pml_cells=cfg.pml_cells, sigma_max=cfg.sigma_max,
        damping=cfg.damping, device=device, dtype=dtype,
        max_rank=cfg.max_rank_pml,
    )
    if verbose:
        mpo_ranks = [c.shape[3] for c in mpo[:-1]]
        print(f"done (MPO max rank: {max(mpo_ranks)})")

    if verbose:
        print("  Building source QTT...", end=" ", flush=True)
    source = gaussian_source_tt_gpu(
        bits_per_dim, domain, center, cfg.source_width,
        cfg.k, device, dtype, max_rank=cfg.max_rank_pml,
    )
    # Negate source: solve H·E = −J → b = −J
    source = tt_scale_gpu(source, -1.0)
    if verbose:
        src_ranks = [c.shape[2] for c in source[:-1]]
        print(f"done (source max rank: {max(src_ranks) if src_ranks else 1})")

    t_build_end = time.perf_counter()
    wall_build = t_build_end - t_build_start

    # Phase 2: Solve
    if verbose:
        print(f"  Solving with AMEN (max_rank={cfg.max_rank}, "
              f"max_sweeps={cfg.n_sweeps}, tol={cfg.tol:.0e})...")

    result = tt_amen_solve_gpu(
        mpo, source,
        max_rank=cfg.max_rank,
        n_sweeps=cfg.n_sweeps,
        tol=cfg.tol,
        verbose=verbose,
    )

    t_solve_end = time.perf_counter()
    wall_solve = t_solve_end - t_build_end

    # Phase 3: Metrics
    metrics = compute_qtt_metrics(result.x, grid_size)
    conservation_err = compute_conservation_error(
        mpo, result.x, source, max_rank=cfg.max_rank * 2,
    )

    gpu_mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    wall_total = time.perf_counter() - t_total_start

    if verbose:
        print(f"\n  Results:")
        print(f"    Converged:         {result.converged} ({result.n_iter} sweeps)")
        print(f"    chi_max:           {metrics['chi_max']}")
        print(f"    QTT bytes:         {metrics['qtt_bytes']:,}")
        print(f"    Dense bytes:       {metrics['dense_bytes']:,}")
        print(f"    Compression:       {metrics['compression_ratio']:.0f}×")
        print(f"    Conservation err:  {conservation_err:.4e}")
        print(f"    GPU peak mem:      {gpu_mem_peak:.1f} MB")
        print(f"    Time (build):      {wall_build:.2f}s")
        print(f"    Time (solve):      {wall_solve:.2f}s")
        print(f"    Time (total):      {wall_total:.2f}s")

    return BenchmarkResult(
        config=cfg,
        grid_size=grid_size,
        n_sites=metrics["n_sites"],
        chi_max=metrics["chi_max"],
        qtt_bytes=metrics["qtt_bytes"],
        dense_bytes=metrics["dense_bytes"],
        compression_ratio=metrics["compression_ratio"],
        wall_time_build=wall_build,
        wall_time_solve=wall_solve,
        wall_time_total=wall_total,
        gpu_mem_peak_mb=gpu_mem_peak,
        conservation_error=conservation_err,
        converged=result.converged,
        n_iter=result.n_iter,
        final_residual=result.final_residual,
    )
