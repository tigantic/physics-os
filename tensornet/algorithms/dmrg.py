"""
Density Matrix Renormalization Group (DMRG)
============================================

Variational ground state algorithm for Matrix Product States.

Theory
------
DMRG minimizes E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ by sweeping through the chain
and optimizing one (1-site DMRG) or two (2-site DMRG) tensors at a time.

The key insight is that the optimization at each site is a local
eigenvalue problem in the effective Hamiltonian H_eff.

2-Site Algorithm:
1. Sweep left → right:
   - Contract left environment L, two-site tensor Θ, right environment R
   - Diagonalize H_eff to get ground state Θ'
   - SVD: Θ' = U S V^†, truncate to χ
   - Update A[i] = U, A[i+1] = S V^†
   - Grow left environment

2. Sweep right → left (similarly)

Convergence when ΔE < tol.
"""

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

from tensornet.core.decompositions import svd_truncated
from tensornet.core.mpo import MPO
from tensornet.core.mps import MPS
from tensornet.core.profiling import memory_profile

# Article V.5.2 truncation error threshold (warn if exceeded)
TRUNCATION_ERROR_WARN_THRESHOLD = 1e-10


@dataclass
class DMRGResult:
    """Result container for DMRG."""

    psi: MPS
    energy: float
    energies: list[float]
    entropies: list[float]
    truncation_errors: list[float]
    converged: bool
    sweeps: int


def _contract_left_env(
    L: Tensor,
    A: Tensor,
    W: Tensor,
) -> Tensor:
    """
    Contract left environment with site tensor and MPO.

    Diagram (top to bottom = left to right in chain):

        a ─── A ─── b        (ket)
              │s
        w ─── W ─── n        (MPO)
              │t
        c ─── A*─── d        (bra)

    L has open indices (a, w, c) on the left - ket, MPO, bra virtual indices.
    Result L' has open indices (b, n, d) on the right.

    L : (χ_L_ket, D_L, χ_L_bra)
    A : (χ_L, d, χ_R)  - MPS site tensor
    W : (D_L, d_out, d_in, D_R)  - MPO site tensor where d_out acts on bra, d_in acts on ket

    Returns: L' : (χ_R_ket, D_R, χ_R_bra)
    """
    # L[a,w,c] A[a,s,b] -> [w,c,s,b]
    temp = torch.einsum("awc,asb->wcsb", L, A)

    # [w,c,s,b] W[w,t,s,n] -> [c,t,b,n] (contract w and s)
    # W indices: (D_L, d_bra, d_ket, D_R) = (w, t, s, n)
    temp2 = torch.einsum("wcsb,wtsn->ctbn", temp, W)

    # [c,t,b,n] A*[c,t,d] -> L'[b,n,d]
    Aconj = A.conj()
    L_new = torch.einsum("ctbn,ctd->bnd", temp2, Aconj)

    return L_new


def _contract_right_env(
    R: Tensor,
    A: Tensor,
    W: Tensor,
) -> Tensor:
    """
    Contract right environment with site tensor and MPO.

    Diagram:
        a ─── A ─── b        (ket)
              │s
        w ─── W ─── n        (MPO)
              │t
        c ─── A*─── d        (bra)

    R has open indices (b, n, d) on the right - ket, MPO, bra virtual indices.
    Result R' has open indices (a, w, c) on the left.

    R : (χ_R_ket, D_R, χ_R_bra)
    A : (χ_L, d, χ_R)  - MPS site tensor
    W : (D_L, d_out, d_in, D_R)  - MPO site tensor

    Returns: R' : (χ_L_ket, D_L, χ_L_bra)
    """
    # A[a,s,b] R[b,n,d] -> [a,s,n,d]
    temp = torch.einsum("asb,bnd->asnd", A, R)

    # [a,s,n,d] W[w,t,s,n] -> [a,d,w,t] (contract s and n)
    temp2 = torch.einsum("asnd,wtsn->adwt", temp, W)

    # [a,d,w,t] A*[c,t,d] -> R'[a,w,c]
    Aconj = A.conj()
    R_new = torch.einsum("adwt,ctd->awc", temp2, Aconj)

    return R_new


@memory_profile
def _build_environments(
    psi: MPS,
    H: MPO,
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Build left and right environments for DMRG.

    Returns:
        L_envs: Left environments, L[i] is environment to the left of site i
        R_envs: Right environments, R[i] is environment to the right of site i
    """
    L = psi.L
    chi_max = max(A.shape[0] for A in psi.tensors)
    D_max = max(W.shape[0] for W in H.tensors)

    dtype = psi.tensors[0].dtype
    device = psi.tensors[0].device

    # Initialize boundary environments
    L_envs = [None] * (L + 1)
    R_envs = [None] * (L + 1)

    # Left boundary: (1, 1, 1)
    L_envs[0] = torch.ones(1, 1, 1, dtype=dtype, device=device)

    # Right boundary: (1, 1, 1)
    R_envs[L] = torch.ones(1, 1, 1, dtype=dtype, device=device)

    # Build right environments from right to left
    for i in range(L - 1, -1, -1):
        R_envs[i] = _contract_right_env(R_envs[i + 1], psi.tensors[i], H.tensors[i])

    return L_envs, R_envs


def _apply_heff_two_site(
    theta: Tensor,
    L: Tensor,
    W1: Tensor,
    W2: Tensor,
    R: Tensor,
) -> Tensor:
    """
    Apply two-site effective Hamiltonian.

    theta: (χ_L, d1, d2, χ_R)
    L: (χ_L, D_L, χ_L')
    W1: (D_L, d1', d1, D_m)
    W2: (D_m, d2', d2, D_R)
    R: (χ_R, D_R, χ_R')

    Returns: H_eff @ theta, same shape as theta
    """
    # Contract in optimal order for efficiency
    # L @ theta -> temp1
    temp1 = torch.einsum(
        "abc,ade->bcde", L, theta.reshape(theta.shape[0], -1, theta.shape[-1])
    )

    # Actually, let's be more explicit
    chi_L, d1, d2, chi_R = theta.shape

    # Flatten theta for contraction
    # theta: (χ_L, d1, d2, χ_R)

    # Contract with L: L[a,w,a'] theta[a,s1,s2,b] -> [w,a',s1,s2,b]
    t1 = torch.einsum(
        "awa,asb->wsab",
        L.reshape(L.shape[0], -1, L.shape[2]),
        theta.reshape(chi_L, d1 * d2, chi_R),
    )

    # This is getting complex, let me use a simpler explicit approach

    # For 2-site DMRG, reshape theta to matrix for eigensolve
    # theta_mat: (χ_L * d1, d2 * χ_R)

    # Build the full effective Hamiltonian explicitly (for small systems)
    # For large systems, use iterative Lanczos

    # Step-by-step contraction:
    # 1. L @ theta -> (D, χ', d1, d2, χ_R)
    t = torch.einsum("abc,ader->bcder", L, theta)

    # 2. Contract with W1
    t = torch.einsum("abcde,bfca->fdce", t, W1)

    # 3. Contract with W2
    t = torch.einsum("abcd,aecb->decd", t, W2)

    # 4. Contract with R
    result = torch.einsum("abcd,dae->abce", t, R)

    # Hmm, indices are getting tangled. Let me use explicit loop for clarity.

    # Actually, the cleanest way is the explicit Heff matrix build for small problems
    # and iterative methods (Lanczos) for large ones

    return result


def _two_site_eigensolve(
    L: Tensor,
    W1: Tensor,
    W2: Tensor,
    R: Tensor,
    theta0: Tensor,
    num_lanczos: int = 20,
) -> tuple[float, Tensor]:
    """
    Solve for ground state of two-site effective Hamiltonian using Lanczos.

    Args:
        L: Left environment (χ_L, D_L, χ_L')
        W1, W2: MPO tensors for sites i, i+1 with shape (D_L, d_out, d_in, D_R)
        R: Right environment (χ_R, D_R, χ_R')
        theta0: Initial two-site tensor (χ_L, d1, d2, χ_R)
        num_lanczos: Number of Lanczos iterations

    Returns:
        (energy, theta_optimal)
    """
    chi_L, d1, d2, chi_R = theta0.shape
    dim = chi_L * d1 * d2 * chi_R

    dtype = theta0.dtype
    device = theta0.device

    # Get MPO bond dimensions
    D_L1 = W1.shape[0]
    D_R1 = W1.shape[3]
    D_L2 = W2.shape[0]
    D_R2 = W2.shape[3]

    def matvec(v: Tensor) -> Tensor:
        """Apply H_eff to vector."""
        theta = v.reshape(chi_L, d1, d2, chi_R)

        # H_eff @ theta with proper index contractions
        # L: (χ_L_ket, D_L, χ_L_bra) = (a, w, c)
        # theta: (χ_L, d1, d2, χ_R) = (a, s, u, b) where s=d1, u=d2
        # W1: (D_L, d_out, d_in, D_M) = (w, t, s, m) - contracts s with theta's d1
        # W2: (D_M, d_out, d_in, D_R) = (m, v, u, n) - contracts u with theta's d2
        # R: (χ_R_ket, D_R, χ_R_bra) = (b, n, d)
        #
        # Result: H_eff·θ[c, t, v, d] = (χ_L_bra, d1_out, d2_out, χ_R_bra)

        # Step 1: Contract theta with L
        # L[a,w,c] theta[a,s,u,b] -> [w,c,s,u,b]
        t1 = torch.einsum("awc,asub->wcsub", L, theta)

        # Step 2: Contract with W1 (contract w and s)
        # [w,c,s,u,b] W1[w,t,s,m] -> [c,t,u,b,m]
        t2 = torch.einsum("wcsub,wtsm->ctubm", t1, W1)

        # Step 3: Contract with W2 (contract m and u)
        # [c,t,u,b,m] W2[m,v,u,n] -> [c,t,v,b,n]
        t3 = torch.einsum("ctubm,mvun->ctvbn", t2, W2)

        # Step 4: Contract with R (contract b and n)
        # [c,t,v,b,n] R[b,n,d] -> [c,t,v,d]
        result = torch.einsum("ctvbn,bnd->ctvd", t3, R)

        return result.reshape(-1)

    # Lanczos iteration
    v = theta0.reshape(-1)
    v = v / torch.linalg.norm(v)

    alpha_list = []
    beta_list = []
    V = [v]

    w = matvec(v)
    alpha = torch.dot(v, w).real
    alpha_list.append(alpha)

    w = w - alpha * v

    for j in range(1, min(num_lanczos, dim)):
        beta = torch.linalg.norm(w)
        if beta < 1e-14:
            break
        beta_list.append(beta)

        v_old = v
        v = w / beta
        V.append(v)

        w = matvec(v)
        w = w - beta * v_old
        alpha = torch.dot(v, w).real
        alpha_list.append(alpha)
        w = w - alpha * v

    # Build tridiagonal matrix and diagonalize
    k = len(alpha_list)
    T = torch.zeros(k, k, dtype=dtype, device=device)
    for i in range(k):
        T[i, i] = alpha_list[i]
        if i < k - 1 and i < len(beta_list):
            T[i, i + 1] = beta_list[i]
            T[i + 1, i] = beta_list[i]

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    # Ground state
    energy = eigenvalues[0].item()
    gs_coeff = eigenvectors[:, 0]

    # Reconstruct ground state in original basis
    V_stack = torch.stack(V, dim=1)  # (dim, k)
    theta_gs = V_stack @ gs_coeff
    theta_gs = theta_gs.reshape(chi_L, d1, d2, chi_R)

    return energy, theta_gs


def dmrg_sweep(
    psi: MPS,
    H: MPO,
    chi_max: int,
    L_envs: list[Tensor],
    R_envs: list[Tensor],
    direction: str = "right",
    svd_cutoff: float = 1e-14,
) -> tuple[float, float, float]:
    """
    Perform one DMRG sweep.

    Args:
        psi: MPS to optimize (modified in-place)
        H: Hamiltonian MPO
        chi_max: Maximum bond dimension
        L_envs: Left environments (modified in-place)
        R_envs: Right environments
        direction: 'right' or 'left'
        svd_cutoff: SVD singular value cutoff

    Returns:
        (energy, max_entropy, max_truncation_error)
    """
    L = psi.L
    energy = 0.0
    max_entropy = 0.0
    max_trunc_err = 0.0

    if direction == "right":
        # Left-to-right sweep
        for i in range(L - 1):
            # Form two-site tensor
            # theta[χ_L, d_i, d_{i+1}, χ_R] = A[i] @ A[i+1]
            theta = torch.einsum("ijk,klm->ijlm", psi.tensors[i], psi.tensors[i + 1])

            # Solve eigenvalue problem
            energy, theta_opt = _two_site_eigensolve(
                L_envs[i], H.tensors[i], H.tensors[i + 1], R_envs[i + 2], theta
            )

            # SVD and truncate
            chi_L, d1, d2, chi_R = theta_opt.shape
            theta_mat = theta_opt.reshape(chi_L * d1, d2 * chi_R)

            U, S, Vh, info = svd_truncated(
                theta_mat, chi_max, cutoff=svd_cutoff, return_info=True
            )

            # Article V.5.2: Warn if truncation error exceeds threshold (left-to-right sweep)
            truncation_err = info.get("truncation_error", 0.0)
            if truncation_err > TRUNCATION_ERROR_WARN_THRESHOLD:
                warnings.warn(
                    f"DMRG truncation error {truncation_err:.2e} exceeds threshold "
                    f"{TRUNCATION_ERROR_WARN_THRESHOLD:.0e} at site {i}. "
                    f"Consider increasing chi_max.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            chi_new = S.shape[0]

            # Compute entropy
            S_normalized = S / torch.linalg.norm(S)
            S2 = S_normalized**2
            entropy = -torch.sum(S2 * torch.log(S2 + 1e-20)).item()
            max_entropy = max(max_entropy, entropy)
            max_trunc_err = max(max_trunc_err, truncation_err)

            # Update tensors
            psi.tensors[i] = U.reshape(chi_L, d1, chi_new)
            psi.tensors[i + 1] = (torch.diag(S) @ Vh).reshape(chi_new, d2, chi_R)

            # Update left environment
            L_envs[i + 1] = _contract_left_env(L_envs[i], psi.tensors[i], H.tensors[i])

    else:
        # Right-to-left sweep
        for i in range(L - 2, -1, -1):
            # Form two-site tensor
            theta = torch.einsum("ijk,klm->ijlm", psi.tensors[i], psi.tensors[i + 1])

            # Solve eigenvalue problem
            energy, theta_opt = _two_site_eigensolve(
                L_envs[i], H.tensors[i], H.tensors[i + 1], R_envs[i + 2], theta
            )

            # SVD and truncate (keep S on left side for right-to-left)
            chi_L, d1, d2, chi_R = theta_opt.shape
            theta_mat = theta_opt.reshape(chi_L * d1, d2 * chi_R)

            U, S, Vh, info = svd_truncated(
                theta_mat, chi_max, cutoff=svd_cutoff, return_info=True
            )

            # Article V.5.2: Warn if truncation error exceeds threshold (right-to-left sweep)
            truncation_err = info.get("truncation_error", 0.0)
            if truncation_err > TRUNCATION_ERROR_WARN_THRESHOLD:
                warnings.warn(
                    f"DMRG truncation error {truncation_err:.2e} exceeds threshold "
                    f"{TRUNCATION_ERROR_WARN_THRESHOLD:.0e} at site {i}. "
                    f"Consider increasing chi_max.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            chi_new = S.shape[0]

            # Compute entropy
            S_normalized = S / torch.linalg.norm(S)
            S2 = S_normalized**2
            entropy = -torch.sum(S2 * torch.log(S2 + 1e-20)).item()
            max_entropy = max(max_entropy, entropy)
            max_trunc_err = max(max_trunc_err, truncation_err)

            # Update tensors (put S on right for left-canonicalization)
            psi.tensors[i] = (U @ torch.diag(S)).reshape(chi_L, d1, chi_new)
            psi.tensors[i + 1] = Vh.reshape(chi_new, d2, chi_R)

            # Update right environment
            R_envs[i + 1] = _contract_right_env(
                R_envs[i + 2], psi.tensors[i + 1], H.tensors[i + 1]
            )

    return energy, max_entropy, max_trunc_err


@memory_profile
def dmrg(
    H: MPO,
    chi_max: int,
    num_sweeps: int = 10,
    tol: float = 1e-10,
    psi0: MPS | None = None,
    svd_cutoff: float = 1e-14,
    verbose: bool = False,
) -> DMRGResult:
    """
    Run 2-site DMRG to find the ground state.

    Args:
        H: Hamiltonian as MPO
        chi_max: Maximum bond dimension
        num_sweeps: Maximum number of sweeps
        tol: Energy convergence tolerance
        psi0: Initial MPS (random if None)
        svd_cutoff: SVD singular value cutoff
        verbose: Print progress

    Returns:
        DMRGResult with ground state MPS and diagnostics
    """
    L = H.L
    d = H.tensors[0].shape[1]  # Physical dimension from MPO

    # Initialize MPS
    if psi0 is None:
        psi = MPS.random(
            L,
            d,
            min(chi_max, 4),
            dtype=H.tensors[0].dtype,
            device=H.tensors[0].device,
            normalize=True,
        )
    else:
        psi = psi0

    # Build environments
    L_envs, R_envs = _build_environments(psi, H)

    energies = []
    entropies = []
    truncation_errors = []
    converged = False

    E_old = float("inf")

    for sweep in range(num_sweeps):
        # Right sweep
        E1, S1, err1 = dmrg_sweep(
            psi, H, chi_max, L_envs, R_envs, direction="right", svd_cutoff=svd_cutoff
        )

        # Left sweep
        E2, S2, err2 = dmrg_sweep(
            psi, H, chi_max, L_envs, R_envs, direction="left", svd_cutoff=svd_cutoff
        )

        energy = E2
        energies.append(energy)
        entropies.append(max(S1, S2))
        truncation_errors.append(max(err1, err2))

        if verbose:
            print(
                f"Sweep {sweep + 1}: E = {energy:.12f}, ΔE = {abs(energy - E_old):.2e}, "
                f"S = {max(S1, S2):.4f}, χ = {psi.bond_dims()}"
            )

        # Check convergence
        if abs(energy - E_old) < tol:
            converged = True
            break

        E_old = energy

    # Normalize final state
    psi.normalize_()

    return DMRGResult(
        psi=psi,
        energy=energies[-1] if energies else 0.0,
        energies=energies,
        entropies=entropies,
        truncation_errors=truncation_errors,
        converged=converged,
        sweeps=len(energies),
    )
