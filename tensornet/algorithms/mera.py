"""
Multi-scale Entanglement Renormalisation Ansatz (MERA)
=======================================================

Hierarchical tensor-network ansatz implementing real-space
renormalisation on a lattice.

Structure (binary MERA):
    - **Disentanglers** :math:`u` remove short-range entanglement.
    - **Isometries** :math:`w` coarse-grain two sites into one.
    - Layers are stacked: each layer halves the number of sites.
    - The top tensor encodes the long-range entanglement (fixed point).

Entropy scaling:
    - Ground states of 1D critical (gapless) systems follow
      :math:`S \\sim \\frac{c}{3} \\log L` (area law violation in 1D).
    - MERA naturally captures this logarithmic correction due to
      its causal-cone structure.

Supported types:
    - Binary MERA (coarse-grain factor 2)
    - Ternary MERA (coarse-grain factor 3)

References:
    [1] Vidal, PRL 99, 220405 (2007).
    [2] Vidal, PRL 101, 110501 (2008).
    [3] Evenbly & Vidal, PRB 79, 144108 (2009).

Domain IV.3 — Tensor-Network Algorithms / MERA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class MERAType(Enum):
    BINARY = auto()
    TERNARY = auto()


@dataclass
class MERALayer:
    """
    A single MERA layer consisting of disentanglers and isometries.

    Binary MERA (N sites → N/2 sites):
        - disentanglers: list of unitaries :math:`u` of shape ``(χ, χ, χ, χ)``
        - isometries: list of tensors :math:`w` of shape ``(χ, χ, χ_out)``

    Ternary MERA (N sites → N/3 sites):
        - disentanglers: :math:`u` of shape ``(χ, χ, χ, χ)``
        - isometries: :math:`w` of shape ``(χ, χ, χ, χ_out)``
    """
    disentanglers: list[NDArray]
    isometries: list[NDArray]


@dataclass
class MERAState:
    """
    Full MERA tensor network.

    Attributes:
        mera_type: Binary or ternary.
        chi: Bond dimension.
        d: Physical dimension of the bottom layer.
        layers: List of MERALayer from bottom (UV) to top (IR).
        top_tensor: The top-level density matrix or tensor.
        n_sites: Number of physical sites.
    """
    mera_type: MERAType
    chi: int
    d: int
    layers: list[MERALayer]
    top_tensor: NDArray
    n_sites: int

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def total_parameters(self) -> int:
        total = self.top_tensor.size
        for layer in self.layers:
            total += sum(u.size for u in layer.disentanglers)
            total += sum(w.size for w in layer.isometries)
        return total


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _random_unitary(n: int, rng: np.random.Generator) -> NDArray:
    """Random unitary via QR of Gaussian matrix."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    return Q


def _random_isometry(n_in: int, n_out: int, rng: np.random.Generator) -> NDArray:
    """Random isometry: V†V = I_{n_out}."""
    Q = _random_unitary(max(n_in, n_out), rng)
    return Q[:n_in, :n_out]


def random_binary_mera(
    n_sites: int, d: int = 2, chi: int = 4, seed: int = 0,
) -> MERAState:
    """
    Create a random binary MERA for ``n_sites`` (must be power of 2).

    Parameters:
        n_sites: Number of physical sites.
        d: Physical (bottom-layer) dimension.
        chi: Bond dimension for internal layers.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    if n_sites & (n_sites - 1) != 0:
        raise ValueError(f"n_sites must be a power of 2, got {n_sites}")

    layers: list[MERALayer] = []
    n = n_sites
    current_d = d

    while n > 2:
        n_pairs = n // 2
        # Disentanglers: n_pairs unitaries of dim (d², d²)
        disent = []
        for _ in range(n_pairs):
            U = _random_unitary(current_d ** 2, rng)
            disent.append(U.reshape(current_d, current_d, current_d, current_d))

        # Isometries: n_pairs isometries (d, d) → chi
        isom = []
        out_d = min(chi, current_d ** 2)
        for _ in range(n_pairs):
            W = _random_isometry(current_d ** 2, out_d, rng)
            isom.append(W.reshape(current_d, current_d, out_d))

        layers.append(MERALayer(disentanglers=disent, isometries=isom))
        n = n // 2
        current_d = out_d

    # Top tensor
    top = rng.standard_normal((current_d, current_d)) + 1j * rng.standard_normal((current_d, current_d))
    top = 0.5 * (top + top.conj().T)
    top /= np.trace(top)

    return MERAState(
        mera_type=MERAType.BINARY,
        chi=chi, d=d,
        layers=layers,
        top_tensor=top,
        n_sites=n_sites,
    )


# ---------------------------------------------------------------------------
# Ascending / descending super-operator
# ---------------------------------------------------------------------------

def ascending_superoperator(
    rho: NDArray,
    disentangler: NDArray,
    isometry: NDArray,
) -> NDArray:
    """
    One layer of the ascending super-operator (coarse-graining).

    Maps a two-site reduced density matrix ρ upward through one MERA
    layer: ρ' = w† u† ρ u w.

    Parameters:
        rho: Density matrix ``(d², d²)`` on two sites.
        disentangler: ``(d, d, d, d)`` unitary.
        isometry: ``(d, d, d')`` isometry.

    Returns:
        Coarse-grained density matrix ``(d', d')`` (or ``(d'², d'²)``).
    """
    d = disentangler.shape[0]
    d_out = isometry.shape[2]

    # Apply u: ρ_out = u† ρ u
    u_mat = disentangler.reshape(d * d, d * d)
    rho_u = u_mat.conj().T @ rho @ u_mat

    # Apply w ⊗ w: trace out and coarse-grain
    w_mat = isometry.reshape(d * d, d_out)
    rho_coarse = w_mat.conj().T @ rho_u @ w_mat

    return rho_coarse


def descending_superoperator(
    h_coarse: NDArray,
    disentangler: NDArray,
    isometry: NDArray,
) -> NDArray:
    """
    One layer of the descending super-operator (fine-graining).

    Maps a coarse-grained operator h downward: h_fine = u w h w† u†.

    Parameters:
        h_coarse: Operator ``(d', d')`` on the coarse level.
        disentangler: ``(d, d, d, d)`` unitary.
        isometry: ``(d, d, d')`` isometry.

    Returns:
        Fine-grained operator ``(d², d²)``.
    """
    d = disentangler.shape[0]
    d_out = isometry.shape[2]

    w_mat = isometry.reshape(d * d, d_out)
    u_mat = disentangler.reshape(d * d, d * d)

    h_fine = u_mat @ w_mat @ h_coarse @ w_mat.conj().T @ u_mat.conj().T
    return h_fine


# ---------------------------------------------------------------------------
# Energy evaluation
# ---------------------------------------------------------------------------

def evaluate_energy(
    state: MERAState,
    hamiltonian_2site: NDArray,
) -> float:
    """
    Evaluate the energy :math:`\\langle H \\rangle` for a
    translationally-invariant binary MERA.

    Uses the ascending super-operator to push the Hamiltonian up
    to the top tensor.

    Parameters:
        state: MERA state.
        hamiltonian_2site: Two-site Hamiltonian ``(d², d²)``.

    Returns:
        Energy per site.
    """
    h = hamiltonian_2site.copy()
    for layer in state.layers:
        if not layer.disentanglers or not layer.isometries:
            break
        u = layer.disentanglers[0]
        w = layer.isometries[0]
        h = ascending_superoperator(h, u, w)

    # Contract with top tensor
    energy = np.real(np.trace(h @ state.top_tensor))
    return float(energy / state.n_sites)


# ---------------------------------------------------------------------------
# Variational optimisation (energy minimisation)
# ---------------------------------------------------------------------------

def optimise_layer(
    state: MERAState,
    hamiltonian_2site: NDArray,
    layer_idx: int,
    n_sweeps: int = 5,
) -> MERAState:
    """
    Variationally optimise a single MERA layer to minimise energy.

    Uses the linearised environment technique:
        1. Compute the effective environment (descending from above,
           ascending from below).
        2. Optimise disentangler via polar decomposition of environment.
        3. Optimise isometry similarly.

    Parameters:
        state: MERA state.
        hamiltonian_2site: Nearest-neighbour Hamiltonian ``(d², d²)``.
        layer_idx: Which layer to optimise.
        n_sweeps: Number of alternating optimisation sweeps.

    Returns:
        Updated MERA state.
    """
    d = state.d
    layer = state.layers[layer_idx]

    for _ in range(n_sweeps):
        # Push Hamiltonian up to this layer
        h_eff = hamiltonian_2site.copy()
        for li in range(layer_idx):
            u = state.layers[li].disentanglers[0]
            w = state.layers[li].isometries[0]
            h_eff = ascending_superoperator(h_eff, u, w)

        # Push top tensor down to this layer
        rho_eff = state.top_tensor.copy()
        for li in range(state.n_layers - 1, layer_idx, -1):
            u = state.layers[li].disentanglers[0]
            w = state.layers[li].isometries[0]
            rho_eff = descending_superoperator(rho_eff, u, w)

        # Optimise disentangler via polar decomposition
        # Environment for u: E_u = ∂<H>/∂u†
        w = layer.isometries[0]
        d_layer = layer.disentanglers[0].shape[0]
        w_mat = w.reshape(d_layer * d_layer, w.shape[2])
        E_u = w_mat @ rho_eff @ w_mat.conj().T @ h_eff
        U_env, _, Vt_env = np.linalg.svd(E_u)
        u_opt = (Vt_env.conj().T @ U_env.conj().T)
        layer.disentanglers[0] = u_opt.reshape(d_layer, d_layer, d_layer, d_layer)

        # Optimise isometry via polar decomposition
        u_mat = u_opt.reshape(d_layer * d_layer, d_layer * d_layer)
        E_w = u_mat.conj().T @ h_eff @ u_mat
        U_w, _, Vt_w = np.linalg.svd(
            E_w[:, :w.shape[2]], full_matrices=False,
        )
        w_opt = U_w @ Vt_w
        layer.isometries[0] = w_opt.reshape(d_layer, d_layer, w.shape[2])

    return state
