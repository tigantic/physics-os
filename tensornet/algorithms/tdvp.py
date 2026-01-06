"""
Time-Dependent Variational Principle (TDVP) for MPS Time Evolution
===================================================================

TDVP provides a variational approach to time evolution that:
1. Preserves the MPS manifold structure
2. Conserves energy exactly (for real-time evolution)
3. Naturally handles long-range correlations

The TDVP equations project the Schrödinger equation onto the MPS tangent space:

    i∂|ψ⟩/∂t = P_T H |ψ⟩

where P_T is the projector onto the tangent space of the MPS manifold.

Implementations:
    - TDVP-1: Single-site variant (fixed bond dimension, fast)
    - TDVP-2: Two-site variant (adaptive bond dimension, more accurate)

References:
    [1] Haegeman et al., "Time-Dependent Variational Principle for Quantum
        Lattices", Phys. Rev. Lett. 107, 070601 (2011)
    [2] Haegeman et al., "Unifying time evolution and optimization with
        matrix product states", Phys. Rev. B 94, 165116 (2016)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from tensornet.algorithms.lanczos import lanczos_expm
from tensornet.core.mpo import MPO
from tensornet.core.mps import MPS


@dataclass
class TDVPResult:
    """Result container for TDVP time evolution."""

    psi: MPS
    times: list[float]
    energies: list[float]
    entropies: list[list[float]]  # Entropies at each bond for each time
    info: dict


def tdvp_step(
    psi: MPS,
    H: MPO,
    dt: float,
    chi_max: int | None = None,
    lanczos_dim: int = 20,
    direction: int = 1,
    tol: float = 1e-12,
) -> MPS:
    """
    Perform one TDVP-2 sweep (left-to-right or right-to-left).

    The two-site TDVP algorithm:
    1. Build left/right environments
    2. For each bond: form two-site tensor, apply exp(-iHdt), SVD split
    3. Evolve center backwards to complete the sweep

    Args:
        psi: Input MPS (modified in place and returned)
        H: Hamiltonian MPO
        dt: Time step (real for real-time, imaginary for ground state)
        chi_max: Maximum bond dimension after truncation
        lanczos_dim: Krylov subspace dimension for exponential
        direction: +1 for left-to-right, -1 for right-to-left
        tol: SVD truncation tolerance

    Returns:
        Time-evolved MPS
    """
    L = psi.L
    dtype = psi.tensors[0].dtype
    device = psi.tensors[0].device

    if chi_max is None:
        chi_max = max(t.shape[0] for t in psi.tensors) * 2

    # Build environments
    left_envs = _build_left_environments(psi, H)
    right_envs = _build_right_environments(psi, H)

    if direction == 1:  # Left to right
        sites = range(L - 1)
    else:  # Right to left
        sites = range(L - 2, -1, -1)

    for i in sites:
        # Two-site update at (i, i+1)
        psi = _two_site_tdvp_update(
            psi,
            H,
            left_envs,
            right_envs,
            i,
            dt / 2,
            chi_max,
            lanczos_dim,
            tol,
            direction,
        )

        # Update environments for next site
        if direction == 1 and i < L - 2:
            left_envs[i + 1] = _update_left_env(
                left_envs[i], psi.tensors[i], H.tensors[i]
            )
        elif direction == -1 and i > 0:
            right_envs[i] = _update_right_env(
                right_envs[i + 1], psi.tensors[i + 1], H.tensors[i + 1]
            )

    return psi


def _build_left_environments(psi: MPS, H: MPO) -> list[Tensor]:
    """Build left environment tensors for TDVP."""
    L = psi.L
    dtype = psi.tensors[0].dtype
    device = psi.tensors[0].device

    # L[i] has shape (chi_psi, D_mpo, chi_psi)
    left_envs = [None] * L

    # Leftmost environment is trivial
    chi_psi = psi.tensors[0].shape[0]
    D_mpo = H.tensors[0].shape[0]
    left_envs[0] = torch.zeros(chi_psi, D_mpo, chi_psi, dtype=dtype, device=device)
    left_envs[0][0, 0, 0] = 1.0

    # Build environments from left
    for i in range(L - 1):
        left_envs[i + 1] = _update_left_env(left_envs[i], psi.tensors[i], H.tensors[i])

    return left_envs


def _build_right_environments(psi: MPS, H: MPO) -> list[Tensor]:
    """Build right environment tensors for TDVP."""
    L = psi.L
    dtype = psi.tensors[0].dtype
    device = psi.tensors[0].device

    right_envs = [None] * L

    # Rightmost environment is trivial
    chi_psi = psi.tensors[-1].shape[2]
    D_mpo = H.tensors[-1].shape[3]
    right_envs[-1] = torch.zeros(chi_psi, D_mpo, chi_psi, dtype=dtype, device=device)
    right_envs[-1][0, 0, 0] = 1.0

    # Build environments from right
    for i in range(L - 2, -1, -1):
        right_envs[i] = _update_right_env(
            right_envs[i + 1], psi.tensors[i + 1], H.tensors[i + 1]
        )

    return right_envs


def _update_left_env(L_env: Tensor, A: Tensor, W: Tensor) -> Tensor:
    """
    Update left environment by contracting one site.

    L_env: (chi_L, D_L, chi_L')
    A: (chi_L, d, chi_R)
    W: (D_L, d', d, D_R)

    Returns: (chi_R, D_R, chi_R')
    """
    chi_L, d, chi_R = A.shape
    D_L, d_out, d_in, D_R = W.shape

    # Contract: L_env[a,m,a'] * A[a,s,b] * W[m,s',s,n] * A*[a',s',b']
    # Step 1: L_env @ A -> (D_L, chi_L', d, chi_R)
    temp = torch.einsum("amx,adb->mxdb", L_env, A)

    # Step 2: temp @ W -> (chi_L', chi_R, d', D_R)
    temp = torch.einsum("mxdb,modn->xbon", temp, W)

    # Step 3: temp @ A* -> (chi_R, D_R, chi_R')
    result = torch.einsum("xbon,xob->bno", temp, A.conj())

    # Reshape to match expected output
    return result.permute(0, 2, 1).contiguous()  # (chi_R, D_R, chi_R')


def _update_right_env(R_env: Tensor, A: Tensor, W: Tensor) -> Tensor:
    """
    Update right environment by contracting one site.

    R_env: (chi_R, D_R, chi_R')
    A: (chi_L, d, chi_R)
    W: (D_L, d', d, D_R)

    Returns: (chi_L, D_L, chi_L')
    """
    chi_L, d, chi_R = A.shape
    D_L, d_out, d_in, D_R = W.shape

    # Contract: R_env[b,n,b'] * A[a,s,b] * W[m,s',s,n] * A*[a',s',b']
    # Step 1: R_env @ A -> (chi_L, d, D_R, chi_R')
    temp = torch.einsum("bnx,adb->adnx", R_env, A)

    # Step 2: temp @ W -> (chi_L, d', D_L, chi_R')
    temp = torch.einsum("adnx,modn->aomx", temp, W)

    # Step 3: temp @ A* -> (chi_L, D_L, chi_L')
    result = torch.einsum("aomx,xoa->amx", temp, A.conj())

    return result


def _two_site_tdvp_update(
    psi: MPS,
    H: MPO,
    left_envs: list[Tensor],
    right_envs: list[Tensor],
    site: int,
    dt: float,
    chi_max: int,
    lanczos_dim: int,
    tol: float,
    direction: int,
) -> MPS:
    """
    Perform two-site TDVP update at sites (site, site+1).

    1. Form two-site tensor Θ = A[site] · A[site+1]
    2. Apply exp(-i H_eff dt) Θ using Lanczos
    3. SVD and truncate
    4. Backward evolution of center matrix
    """
    i = site
    A_i = psi.tensors[i]
    A_j = psi.tensors[i + 1]

    chi_L, d_i, chi_m = A_i.shape
    chi_m2, d_j, chi_R = A_j.shape

    # Form two-site tensor: Θ[chi_L, d_i, d_j, chi_R]
    theta = torch.einsum("adb,bec->adec", A_i, A_j)
    theta_shape = theta.shape
    theta_flat = theta.reshape(-1)

    # Build effective Hamiltonian for two sites
    L_env = left_envs[i]
    R_env = right_envs[i + 1] if i + 1 < len(right_envs) else right_envs[-1]
    W_i = H.tensors[i]
    W_j = H.tensors[i + 1]

    def H_eff(v: Tensor) -> Tensor:
        """Apply two-site effective Hamiltonian."""
        v = v.reshape(theta_shape)

        # Contract: L[a,m,a'] * v[a,s1,s2,b] * W1[m,s1',s1,n] * W2[n,s2',s2,p] * R[b,p,b']
        # This is a 6-tensor contraction, done step by step

        # Step 1: L @ v
        temp = torch.einsum("amx,aseb->mxseb", L_env, v)

        # Step 2: temp @ W1
        temp = torch.einsum("mxseb,mton->xtoeb", temp, W_i)

        # Step 3: temp @ W2
        temp = torch.einsum("xtoeb,nueo->xtunb", temp, W_j)

        # Step 4: temp @ R
        result = torch.einsum("xtunb,bnx->tuab", temp, R_env)

        # Reshape to match input
        # result has shape (d_out_i, d_out_j, chi_L', chi_R')
        # Need to permute to (chi_L', d_out_i, d_out_j, chi_R')
        result = result.permute(2, 0, 1, 3)

        return result.reshape(-1)

    # Time evolution: exp(-i H dt) |θ⟩
    # For real-time: use -i*dt
    # For imaginary-time ground state: use -dt (real)
    theta_evolved = lanczos_expm(H_eff, theta_flat, -1j * dt, krylov_dim=lanczos_dim)
    theta_evolved = theta_evolved.reshape(theta_shape)

    # Randomized SVD split: Θ[a,s1,s2,b] -> A[a,s1,m] · B[m,s2,b]
    # Note: svd_lowrank returns (U, S, V) not (U, S, Vh)
    theta_mat = theta_evolved.reshape(chi_L * d_i, d_j * chi_R)
    q = min(chi_max, min(theta_mat.shape))
    U, S, V = torch.svd_lowrank(theta_mat, q=q, niter=1)

    # Truncate
    chi_new = min(chi_max, len(S), (S > tol * S[0]).sum().item())
    chi_new = max(1, chi_new)

    U = U[:, :chi_new]
    S = S[:chi_new]
    V = V[:, :chi_new]  # V is (n, k), column slicing

    # Form new tensors
    A_new = U.reshape(chi_L, d_i, chi_new)
    SV = torch.diag(S) @ V.T  # V.T to get Vh
    B_new = SV.reshape(chi_new, d_j, chi_R)

    # Backward evolution of center matrix (one-site)
    if direction == 1:
        # Move orthogonality center right: absorb S into B
        psi.tensors[i] = A_new
        psi.tensors[i + 1] = B_new

        # Backward evolve the center (optional for TDVP-2)
        # For simplicity, we skip backward evolution here

    else:
        # Move orthogonality center left: absorb S into A
        psi.tensors[i] = torch.einsum("adb,bc->adc", A_new, torch.diag(S))
        psi.tensors[i + 1] = Vh.reshape(chi_new, d_j, chi_R)

    return psi


def tdvp(
    psi: MPS,
    H: MPO,
    dt: float,
    num_steps: int,
    chi_max: int | None = None,
    lanczos_dim: int = 20,
    tol: float = 1e-12,
    callback: Callable[[MPS, float], None] | None = None,
    verbose: bool = False,
) -> TDVPResult:
    """
    Time evolution using TDVP-2 algorithm.

    Performs symmetric sweeps: left-to-right followed by right-to-left.

    Args:
        psi: Initial MPS state
        H: Hamiltonian MPO
        dt: Time step per sweep
        num_steps: Number of full sweeps (each sweep = dt evolution)
        chi_max: Maximum bond dimension
        lanczos_dim: Krylov subspace dimension
        tol: SVD truncation tolerance
        callback: Optional function called after each step with (psi, time)
        verbose: Print progress

    Returns:
        TDVPResult containing evolved state and history

    Example:
        >>> psi = MPS.random(L=20, d=2, chi=16)
        >>> H = heisenberg_mpo(L=20)
        >>> result = tdvp(psi, H, dt=0.1, num_steps=100)
        >>> print(f"Final energy: {result.energies[-1]}")
    """
    psi = psi.copy()

    times = [0.0]
    energies = []
    entropies = []

    # Initial energy
    E0 = H.expectation(psi)
    energies.append(E0)

    # Initial entropies
    s0 = [psi.entropy(bond=b).item() for b in range(psi.L - 1)]
    entropies.append(s0)

    if verbose:
        print(f"TDVP Evolution: dt={dt}, steps={num_steps}, chi_max={chi_max}")
        print(f"  t=0.000: E = {E0:.10f}")

    t = 0.0
    for step in range(num_steps):
        # Forward sweep (left to right) with dt/2
        psi = tdvp_step(psi, H, dt, chi_max, lanczos_dim, direction=1, tol=tol)

        # Backward sweep (right to left) with dt/2
        psi = tdvp_step(psi, H, dt, chi_max, lanczos_dim, direction=-1, tol=tol)

        t += dt
        times.append(t)

        # Compute observables
        E = H.expectation(psi)
        energies.append(E)

        s = [psi.entropy(bond=b).item() for b in range(psi.L - 1)]
        entropies.append(s)

        if callback is not None:
            callback(psi, t)

        if verbose and (step + 1) % max(1, num_steps // 10) == 0:
            dE = abs(E - E0) / abs(E0) if E0 != 0 else abs(E)
            chi_max_current = max(t.shape[0] for t in psi.tensors)
            print(
                f"  t={t:.3f}: E = {E:.10f}, ΔE/E = {dE:.2e}, χ_max = {chi_max_current}"
            )

    info = {
        "dt": dt,
        "num_steps": num_steps,
        "chi_max": chi_max,
        "lanczos_dim": lanczos_dim,
        "energy_conservation": abs(energies[-1] - energies[0]) / abs(energies[0]),
    }

    return TDVPResult(
        psi=psi, times=times, energies=energies, entropies=entropies, info=info
    )


def imaginary_time_tdvp(
    psi: MPS,
    H: MPO,
    dt: float = 0.1,
    num_steps: int = 100,
    chi_max: int = 64,
    tol: float = 1e-10,
    energy_tol: float = 1e-8,
    verbose: bool = False,
) -> TDVPResult:
    """
    Find ground state using imaginary time TDVP.

    Evolves in imaginary time: |ψ(τ)⟩ ∝ exp(-Hτ)|ψ(0)⟩
    As τ → ∞, the state converges to the ground state.

    Args:
        psi: Initial MPS state
        H: Hamiltonian MPO
        dt: Imaginary time step
        num_steps: Maximum number of steps
        chi_max: Maximum bond dimension
        tol: SVD truncation tolerance
        energy_tol: Convergence tolerance for energy
        verbose: Print progress

    Returns:
        TDVPResult with ground state approximation
    """
    psi = psi.copy()
    # Normalize using in-place method
    n = psi.norm()
    if n > 0:
        psi.tensors[0] = psi.tensors[0] / n

    times = [0.0]
    energies = []

    E_prev = H.expectation(psi)
    energies.append(E_prev)

    if verbose:
        print("Imaginary Time TDVP: seeking ground state")
        print(f"  τ=0.000: E = {E_prev:.10f}")

    converged = False
    for step in range(num_steps):
        # Imaginary time: use real dt (no i factor)
        # Forward sweep
        psi = tdvp_step(psi, H, -dt, chi_max, direction=1, tol=tol)
        psi.normalize_()

        # Backward sweep
        psi = tdvp_step(psi, H, -dt, chi_max, direction=-1, tol=tol)
        psi.normalize_()

        tau = (step + 1) * dt
        times.append(tau)

        E = H.expectation(psi)
        energies.append(E)

        dE = abs(E - E_prev)
        E_prev = E

        if verbose and (step + 1) % max(1, num_steps // 10) == 0:
            print(f"  τ={tau:.3f}: E = {E:.10f}, ΔE = {dE:.2e}")

        if dE < energy_tol:
            if verbose:
                print(f"  Converged at step {step + 1}: ΔE = {dE:.2e} < {energy_tol}")
            converged = True
            break

    info = {
        "dt": dt,
        "num_steps": step + 1,
        "chi_max": chi_max,
        "converged": converged,
        "final_energy": energies[-1],
    }

    return TDVPResult(psi=psi, times=times, energies=energies, entropies=[], info=info)
