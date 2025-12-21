"""
Time-Evolving Block Decimation (TEBD)
======================================

Real and imaginary time evolution for Matrix Product States.

Theory
------
TEBD applies the Suzuki-Trotter decomposition to approximate e^{-iHt}
(real time) or e^{-βH} (imaginary time) for nearest-neighbor Hamiltonians.

H = Σᵢ hᵢ,ᵢ₊₁

First-order Trotter: e^{-iHdt} ≈ Πᵢ e^{-ihᵢ,ᵢ₊₁ dt}
Second-order Trotter: 
    e^{-iHdt} ≈ Π_odd e^{-ihdt/2} · Π_even e^{-ihdt} · Π_odd e^{-ihdt/2}

Each e^{-ih_{i,i+1}dt} is a two-site gate applied via SVD truncation.

Error Scaling:
- First-order: O(dt²)
- Second-order: O(dt³)
- Fourth-order: O(dt⁵)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Union
import warnings
import torch
from torch import Tensor
import math

from tensornet.core.mps import MPS
from tensornet.core.decompositions import svd_truncated
from tensornet.core.profiling import memory_profile

# Article V.5.2 truncation error threshold (warn if exceeded)
TRUNCATION_ERROR_WARN_THRESHOLD = 1e-10


@dataclass
class TEBDResult:
    """Result container for TEBD time evolution."""
    psi: MPS
    times: List[float]
    energies: Optional[List[float]]
    entropies: List[List[float]]  # entropies[time_idx][bond_idx]
    truncation_errors: List[float]


def _make_two_site_gate(
    h_local: Tensor,
    dt: complex,
    d: int = 2,
) -> Tensor:
    """
    Create two-site time evolution gate.
    
    U = exp(-i * h_local * dt) for real time
    U = exp(-h_local * dt) for imaginary time (dt = -i * tau)
    
    Args:
        h_local: Two-site Hamiltonian (d², d²)
        dt: Time step (complex for imaginary time)
        d: Physical dimension
        
    Returns:
        Gate tensor (d, d, d, d)
    """
    # Exponentiate
    U = torch.linalg.matrix_exp(-1j * h_local * dt)
    
    # Reshape to (d, d, d, d)
    return U.reshape(d, d, d, d)


def _apply_two_site_gate(
    psi: MPS,
    gate: Tensor,
    site: int,
    chi_max: int,
    cutoff: float = 1e-14,
) -> float:
    """
    Apply a two-site gate at position (site, site+1) and truncate.
    
    Args:
        psi: MPS (modified in-place)
        gate: Two-site gate (d, d, d, d)
        site: Left site index
        chi_max: Maximum bond dimension after truncation
        cutoff: Singular value cutoff
        
    Returns:
        Truncation error
    """
    # Contract the two site tensors
    # A[site]: (χ_L, d, χ_m)
    # A[site+1]: (χ_m, d, χ_R)
    theta = torch.einsum('ijk,klm->ijlm', psi.tensors[site], psi.tensors[site+1])
    
    # Apply gate: gate[s1', s2', s1, s2] @ theta[χ_L, s1, s2, χ_R]
    # -> theta'[χ_L, s1', s2', χ_R]
    theta_new = torch.einsum('abcd,ecdf->eabf', gate, theta)
    
    # SVD truncate
    chi_L, d1, d2, chi_R = theta_new.shape
    theta_mat = theta_new.reshape(chi_L * d1, d2 * chi_R)
    
    U, S, Vh, info = svd_truncated(theta_mat, chi_max, cutoff=cutoff, return_info=True)
    
    # Article V.5.2: Warn if truncation error exceeds threshold
    truncation_err = info.get('truncation_error', 0.0)
    if truncation_err > TRUNCATION_ERROR_WARN_THRESHOLD:
        warnings.warn(
            f"TEBD truncation error {truncation_err:.2e} exceeds threshold "
            f"{TRUNCATION_ERROR_WARN_THRESHOLD:.0e} at site {site}. "
            f"Consider increasing chi_max or decreasing dt.",
            RuntimeWarning,
            stacklevel=2
        )
    
    chi_new = S.shape[0]
    
    # Update MPS tensors (left-canonical form)
    psi.tensors[site] = U.reshape(chi_L, d1, chi_new)
    psi.tensors[site+1] = (torch.diag(S) @ Vh).reshape(chi_new, d2, chi_R)
    
    return info.get('truncation_error', 0.0)


def build_gates_from_mpo(
    H: 'MPO',
    dt: complex,
) -> List[Tensor]:
    """
    Build TEBD gates from an MPO Hamiltonian.
    
    Extracts nearest-neighbor terms from the MPO.
    
    Note: This is a simplified extraction that works for standard
    nearest-neighbor Hamiltonians. For more complex MPOs, you may
    need to provide gates directly.
    
    Args:
        H: Hamiltonian MPO
        dt: Time step
        
    Returns:
        List of two-site gates
    """
    # This would require knowing the structure of the MPO
    # For now, we require gates to be passed directly
    raise NotImplementedError(
        "Automatic gate extraction from MPO not yet implemented. "
        "Please provide gates directly or use a Hamiltonian class."
    )


@memory_profile
def tebd_step(
    psi: MPS,
    gates_odd: List[Tensor],
    gates_even: List[Tensor],
    chi_max: int,
    cutoff: float = 1e-14,
    order: int = 2,
) -> float:
    """
    Perform one TEBD step (second-order Trotter).
    
    For second-order:
        e^{-iHdt} ≈ U_odd(dt/2) · U_even(dt) · U_odd(dt/2)
    
    Args:
        psi: MPS (modified in-place)
        gates_odd: Gates for odd bonds (0-1, 2-3, ...)
        gates_even: Gates for even bonds (1-2, 3-4, ...)
        chi_max: Maximum bond dimension
        cutoff: SVD cutoff
        order: Trotter order (1 or 2)
        
    Returns:
        Maximum truncation error in this step
    """
    L = psi.L
    max_error = 0.0
    
    if order == 1:
        # First-order Trotter: U_odd · U_even
        # Odd bonds
        for i, gate in enumerate(gates_odd):
            site = 2 * i
            if site + 1 < L:
                err = _apply_two_site_gate(psi, gate, site, chi_max, cutoff)
                max_error = max(max_error, err)
        
        # Even bonds
        for i, gate in enumerate(gates_even):
            site = 2 * i + 1
            if site + 1 < L:
                err = _apply_two_site_gate(psi, gate, site, chi_max, cutoff)
                max_error = max(max_error, err)
    
    elif order == 2:
        # Second-order Trotter: U_odd(dt/2) · U_even(dt) · U_odd(dt/2)
        
        # First half-step on odd bonds
        for i, gate in enumerate(gates_odd):
            site = 2 * i
            if site + 1 < L:
                # gates_odd should already be for dt/2
                err = _apply_two_site_gate(psi, gate, site, chi_max, cutoff)
                max_error = max(max_error, err)
        
        # Full step on even bonds
        for i, gate in enumerate(gates_even):
            site = 2 * i + 1
            if site + 1 < L:
                err = _apply_two_site_gate(psi, gate, site, chi_max, cutoff)
                max_error = max(max_error, err)
        
        # Second half-step on odd bonds
        for i, gate in enumerate(gates_odd):
            site = 2 * i
            if site + 1 < L:
                err = _apply_two_site_gate(psi, gate, site, chi_max, cutoff)
                max_error = max(max_error, err)
    
    else:
        raise ValueError(f"Trotter order {order} not implemented")
    
    return max_error


def build_heisenberg_gates(
    L: int,
    dt: complex,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
    order: int = 2,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Build TEBD gates for Heisenberg XXZ model.
    
    H = Σᵢ [Jx SˣᵢSˣᵢ₊₁ + Jy SʸᵢSʸᵢ₊₁ + Jz SᶻᵢSᶻᵢ₊₁] - h Σᵢ Sᶻᵢ
    
    Args:
        L: Number of sites
        dt: Time step
        Jx, Jy, Jz: Exchange couplings
        h: Magnetic field
        order: Trotter order
        dtype: Data type
        device: Device
        
    Returns:
        (gates_odd, gates_even) for TEBD
    """
    if device is None:
        device = torch.device('cpu')
    
    # Pauli matrices
    Sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device) / 2
    Sy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device) / 2
    Sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device) / 2
    I = torch.eye(2, dtype=dtype, device=device)
    
    # Two-site Hamiltonian
    def h_two_site(include_left_h: bool = True, include_right_h: bool = True):
        h2 = (
            Jx * torch.kron(Sx, Sx) +
            Jy * torch.kron(Sy, Sy) +
            Jz * torch.kron(Sz, Sz)
        )
        # Distribute magnetic field equally between sites
        if include_left_h:
            h2 = h2 - (h / 2) * torch.kron(Sz, I)
        if include_right_h:
            h2 = h2 - (h / 2) * torch.kron(I, Sz)
        return h2
    
    # Compute gates
    if order == 1:
        dt_odd = dt
        dt_even = dt
    elif order == 2:
        dt_odd = dt / 2  # Half step for odd
        dt_even = dt      # Full step for even
    else:
        raise ValueError(f"Order {order} not implemented")
    
    gates_odd = []
    gates_even = []
    
    # Odd bonds: 0-1, 2-3, 4-5, ...
    for i in range(0, L - 1, 2):
        h2 = h_two_site(
            include_left_h=(i == 0),  # Full field on leftmost site
            include_right_h=(i + 1 == L - 1 and (L - 1) % 2 == 1)  # If last site not covered by even
        )
        gate = _make_two_site_gate(h2, dt_odd, d=2)
        gates_odd.append(gate)
    
    # Even bonds: 1-2, 3-4, 5-6, ...
    for i in range(1, L - 1, 2):
        h2 = h_two_site(
            include_left_h=(len(gates_odd) == 0),  # If no odd bonds covered this
            include_right_h=(i + 1 == L - 1)  # Full field on rightmost site if last
        )
        gate = _make_two_site_gate(h2, dt_even, d=2)
        gates_even.append(gate)
    
    return gates_odd, gates_even


def build_tfim_gates(
    L: int,
    dt: complex,
    J: float = 1.0,
    g: float = 1.0,
    order: int = 2,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Build TEBD gates for Transverse Field Ising Model.
    
    H = -J Σᵢ SᶻᵢSᶻᵢ₊₁ - g Σᵢ Sˣᵢ
    
    Args:
        L: Number of sites
        dt: Time step
        J: Ising coupling
        g: Transverse field
        order: Trotter order
        dtype: Data type
        device: Device
        
    Returns:
        (gates_odd, gates_even) for TEBD
    """
    if device is None:
        device = torch.device('cpu')
    
    Sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device) / 2
    Sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device) / 2
    I = torch.eye(2, dtype=dtype, device=device)
    
    def h_two_site():
        h2 = -J * torch.kron(Sz, Sz)
        # Distribute transverse field
        h2 = h2 - (g / 2) * torch.kron(Sx, I) - (g / 2) * torch.kron(I, Sx)
        return h2
    
    if order == 2:
        dt_odd = dt / 2
        dt_even = dt
    else:
        dt_odd = dt
        dt_even = dt
    
    h2 = h_two_site()
    
    gates_odd = []
    gates_even = []
    
    # Odd bonds
    for i in range(0, L - 1, 2):
        gate = _make_two_site_gate(h2, dt_odd, d=2)
        gates_odd.append(gate)
    
    # Even bonds
    for i in range(1, L - 1, 2):
        gate = _make_two_site_gate(h2, dt_even, d=2)
        gates_even.append(gate)
    
    return gates_odd, gates_even


@memory_profile
def tebd(
    psi: MPS,
    gates_odd: List[Tensor],
    gates_even: List[Tensor],
    num_steps: int,
    dt: float,
    chi_max: int,
    order: int = 2,
    cutoff: float = 1e-14,
    compute_energy: Optional[Callable[[MPS], float]] = None,
    compute_every: int = 1,
    normalize_every: int = 1,
    verbose: bool = False,
) -> TEBDResult:
    """
    Run TEBD time evolution.
    
    Args:
        psi: Initial MPS (modified in-place)
        gates_odd: Gates for odd bonds
        gates_even: Gates for even bonds
        num_steps: Number of time steps
        dt: Time step (for logging, gates should already encode dt)
        chi_max: Maximum bond dimension
        order: Trotter order
        cutoff: SVD cutoff
        compute_energy: Optional function to compute energy
        compute_every: Compute observables every N steps
        normalize_every: Normalize every N steps
        verbose: Print progress
        
    Returns:
        TEBDResult with time evolution data
    """
    times = [0.0]
    energies = [compute_energy(psi)] if compute_energy else None
    entropies = [psi.entropy()]
    truncation_errors = []
    
    max_error = 0.0
    
    for step in range(1, num_steps + 1):
        # One TEBD step
        err = tebd_step(psi, gates_odd, gates_even, chi_max, cutoff, order)
        max_error = max(max_error, err)
        
        # Normalize periodically
        if step % normalize_every == 0:
            psi.normalize_()
        
        # Record observables
        if step % compute_every == 0:
            times.append(step * dt)
            
            if compute_energy:
                energies.append(compute_energy(psi))
            
            entropies.append(psi.entropy())
            truncation_errors.append(max_error)
            max_error = 0.0
            
            if verbose:
                E_str = f", E = {energies[-1]:.8f}" if energies else ""
                print(f"Step {step}/{num_steps}: t = {step * dt:.4f}{E_str}, "
                      f"χ = {psi.bond_dimensions}, S_max = {max(entropies[-1]):.4f}")
    
    return TEBDResult(
        psi=psi,
        times=times,
        energies=energies,
        entropies=entropies,
        truncation_errors=truncation_errors,
    )


def imaginary_time_evolution(
    psi: MPS,
    gates_odd: List[Tensor],
    gates_even: List[Tensor],
    num_steps: int,
    chi_max: int,
    order: int = 2,
    cutoff: float = 1e-14,
    normalize_every: int = 1,
    verbose: bool = False,
) -> MPS:
    """
    Imaginary time evolution for ground state preparation.
    
    |ψ(β)⟩ = e^{-βH} |ψ(0)⟩ / ||...||
    
    As β → ∞, |ψ⟩ → ground state (if overlap is non-zero).
    
    Note: Gates should be created with imaginary time step:
        dt = -1j * tau  (so e^{-iHdt} = e^{-Hτ})
    
    Args:
        psi: Initial MPS (modified in-place)
        gates_odd, gates_even: Imaginary time gates
        num_steps: Number of steps
        chi_max: Maximum bond dimension
        order: Trotter order
        cutoff: SVD cutoff
        normalize_every: Normalize every N steps
        verbose: Print progress
        
    Returns:
        Ground state MPS
    """
    for step in range(1, num_steps + 1):
        tebd_step(psi, gates_odd, gates_even, chi_max, cutoff, order)
        
        if step % normalize_every == 0:
            psi.normalize_()
        
        if verbose and step % 10 == 0:
            print(f"ITE step {step}/{num_steps}: χ = {psi.bond_dimensions}")
    
    psi.normalize_()
    return psi
