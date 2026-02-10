"""
5.19 — Neural Closure Models
==============================

Learned closure terms for under-resolved PDE simulations.

In coarse-grained or RANS/LES CFD, sub-grid effects are modelled
by closure terms.  These neural closures learn the mapping from
resolved fields to unresolved stress/flux terms.

Components:
    * ReynoldsStressClosure — τ_ij(S_ij, Ω_ij) for RANS
    * SubgridFluxClosure — π_sgs for LES
    * TensorBasisClosure — Pope's tensor-basis expansion with learned coefficients
    * ClosureTrainer — online/offline training with DNS data
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Tensor invariant utilities ────────────────────────────────────

def strain_rate(grad_u: np.ndarray) -> np.ndarray:
    """Symmetric strain-rate tensor S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i).

    Parameters
    ----------
    grad_u : (..., 3, 3)  velocity gradient tensor

    Returns
    -------
    S : (..., 3, 3)
    """
    return 0.5 * (grad_u + np.swapaxes(grad_u, -2, -1))


def rotation_rate(grad_u: np.ndarray) -> np.ndarray:
    """Antisymmetric rotation-rate tensor Ω_ij = ½(∂u_i/∂x_j − ∂u_j/∂x_i)."""
    return 0.5 * (grad_u - np.swapaxes(grad_u, -2, -1))


def invariants_2d(S: np.ndarray, O: np.ndarray) -> np.ndarray:
    """Compute scalar invariants of S and Ω for 2-D/3-D flow.

    Returns (..., 5) tensor of the first 5 invariants:
        I1 = tr(S²), I2 = tr(Ω²), I3 = tr(S³),
        I4 = tr(Ω²S), I5 = tr(Ω²S²)
    """
    S2 = np.einsum("...ij,...jk->...ik", S, S)
    O2 = np.einsum("...ij,...jk->...ik", O, O)
    S3 = np.einsum("...ij,...jk->...ik", S2, S)

    I1 = np.einsum("...ii->...", S2)
    I2 = np.einsum("...ii->...", O2)
    I3 = np.einsum("...ii->...", S3)
    I4 = np.einsum("...ii->...", np.einsum("...ij,...jk->...ik", O2, S))
    I5 = np.einsum("...ii->...", np.einsum("...ij,...jk->...ik", O2, S2))

    return np.stack([I1, I2, I3, I4, I5], axis=-1)


def tensor_basis(S: np.ndarray, O: np.ndarray) -> List[np.ndarray]:
    """Pope's 10-term tensor basis T^(n) for symmetric deviatoric tensors.

    Returns list of 10 tensors each with shape (..., 3, 3).
    """
    I = np.eye(3, dtype=S.dtype)
    S2 = np.einsum("...ij,...jk->...ik", S, S)
    O2 = np.einsum("...ij,...jk->...ik", O, O)
    SO = np.einsum("...ij,...jk->...ik", S, O)
    OS = np.einsum("...ij,...jk->...ik", O, S)

    T = [
        S,
        np.einsum("...ij,...jk->...ik", S, O) - np.einsum("...ij,...jk->...ik", O, S),
        S2 - np.einsum("...ii->...", S2)[..., None, None] / 3 * I,
        O2 - np.einsum("...ii->...", O2)[..., None, None] / 3 * I,
        np.einsum("...ij,...jk->...ik", O, S2) - np.einsum("...ij,...jk->...ik", S2, O),
    ]
    # 5-10: higher-order (simplified for practical use)
    S2O = np.einsum("...ij,...jk->...ik", S2, O)
    OS2 = np.einsum("...ij,...jk->...ik", O, S2)
    T.append(O2 @ S + S @ O2 - 2.0/3.0 * np.einsum("...ii->...", O2 @ S)[..., None, None] * I)
    T.append(OS @ O2 - O2 @ SO)
    T.append(S @ O @ O2 - O2 @ O @ S)
    T.append(SO @ S2 - S2 @ OS)
    T.append(O @ S2 @ O2 - O2 @ S2 @ O)

    return T


# ── MLP layer ─────────────────────────────────────────────────────

class _MLP:
    """Small MLP for closure coefficient prediction."""

    def __init__(self, layer_sizes: List[int], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            s = math.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.weights.append(
                rng.normal(0, s, (layer_sizes[i], layer_sizes[i + 1])).astype(np.float32)
            )
            self.biases.append(np.zeros(layer_sizes[i + 1], dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(len(self.weights) - 1):
            x = np.tanh(x @ self.weights[i] + self.biases[i])
        return x @ self.weights[-1] + self.biases[-1]

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


# ── Reynolds Stress Closure ───────────────────────────────────────

@dataclass
class ReynoldsStressConfig:
    """Config for RANS Reynolds stress closure."""
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    n_invariants: int = 5
    n_basis: int = 10
    realizability: bool = True  # enforce positive-semi-definiteness


class ReynoldsStressClosure:
    """Neural closure for RANS Reynolds stress anisotropy tensor.

    Maps invariants of S and Ω → coefficients g^(n) in the
    tensor-basis expansion:
        a_ij = Σ_n g^(n)(λ) T^(n)_ij
    """

    def __init__(self, cfg: Optional[ReynoldsStressConfig] = None) -> None:
        self.cfg = cfg or ReynoldsStressConfig()
        layers = [self.cfg.n_invariants] + self.cfg.hidden_dims + [self.cfg.n_basis]
        self.net = _MLP(layers, seed=0)

    def predict_anisotropy(self, grad_u: np.ndarray) -> np.ndarray:
        """Predict Reynolds stress anisotropy a_ij.

        Parameters
        ----------
        grad_u : (N, 3, 3)  mean velocity gradient

        Returns
        -------
        a_ij : (N, 3, 3)  anisotropy tensor
        """
        S = strain_rate(grad_u)
        O = rotation_rate(grad_u)

        # Normalise
        S_norm = np.linalg.norm(S.reshape(len(S), -1), axis=1, keepdims=True).clip(1e-8)
        S_hat = S / S_norm[:, :, None] if S.ndim == 3 else S / S_norm
        O_hat = O / S_norm[:, :, None] if O.ndim == 3 else O / S_norm

        # Invariants
        inv = invariants_2d(S_hat, O_hat)  # (N, 5)

        # Predict coefficients
        g = self.net.forward(inv)  # (N, n_basis)

        # Tensor basis
        T_list = tensor_basis(S_hat, O_hat)  # list of (N, 3, 3)

        # Sum basis
        n_use = min(len(T_list), self.cfg.n_basis)
        a = np.zeros_like(S)
        for n in range(n_use):
            a += g[:, n:n + 1, None] * T_list[n]

        # Realizability: project to nearest PSD
        if self.cfg.realizability:
            for i in range(len(a)):
                eigvals, eigvecs = np.linalg.eigh(a[i])
                eigvals = np.clip(eigvals, -1.0 / 3.0, 2.0 / 3.0)
                a[i] = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return a

    @property
    def param_count(self) -> int:
        return self.net.param_count


# ── Sub-grid Flux Closure ─────────────────────────────────────────

@dataclass
class SubgridFluxConfig:
    """Config for LES sub-grid stress closure."""
    input_dim: int = 9          # flattened filter-width-normalised gradient
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    output_dim: int = 6         # 6 independent components of symmetric tensor


class SubgridFluxClosure:
    """Neural sub-grid stress model for LES.

    Maps resolved strain rate → sub-grid stress tensor τ_sgs.
    """

    def __init__(self, cfg: Optional[SubgridFluxConfig] = None) -> None:
        self.cfg = cfg or SubgridFluxConfig()
        layers = [self.cfg.input_dim] + self.cfg.hidden_dims + [self.cfg.output_dim]
        self.net = _MLP(layers, seed=7)

    def predict(self, grad_u_filtered: np.ndarray) -> np.ndarray:
        """Predict sub-grid stress.

        Parameters
        ----------
        grad_u_filtered : (N, 3, 3)  filtered velocity gradient

        Returns
        -------
        tau_sgs : (N, 3, 3)  symmetric sub-grid stress
        """
        N = grad_u_filtered.shape[0]
        flat = grad_u_filtered.reshape(N, 9)
        out = self.net.forward(flat)  # (N, 6)

        # Assemble symmetric tensor from 6 components
        tau = np.zeros((N, 3, 3), dtype=np.float32)
        tau[:, 0, 0] = out[:, 0]
        tau[:, 1, 1] = out[:, 1]
        tau[:, 2, 2] = out[:, 2]
        tau[:, 0, 1] = tau[:, 1, 0] = out[:, 3]
        tau[:, 0, 2] = tau[:, 2, 0] = out[:, 4]
        tau[:, 1, 2] = tau[:, 2, 1] = out[:, 5]

        return tau

    @property
    def param_count(self) -> int:
        return self.net.param_count


# ── Tensor-Basis Neural Network ──────────────────────────────────

class TensorBasisClosure:
    """Tensor-Basis Neural Network (TBNN) for Reynolds stress.

    Embeds Galilean invariance by construction: the network only
    operates on scalar invariants and linearly combines the
    tensor basis.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        n_basis: int = 5,
    ) -> None:
        self.n_basis = n_basis
        hidden_dims = hidden_dims or [64, 64]
        layers = [5] + hidden_dims + [n_basis]
        self.net = _MLP(layers, seed=42)

    def predict(self, grad_u: np.ndarray) -> np.ndarray:
        """Predict anisotropy tensor from velocity gradient."""
        S = strain_rate(grad_u)
        O = rotation_rate(grad_u)
        S_norm = np.linalg.norm(S.reshape(len(S), -1), axis=1, keepdims=True).clip(1e-8)
        S_hat = S / S_norm[:, :, None]
        O_hat = O / S_norm[:, :, None]

        inv = invariants_2d(S_hat, O_hat)
        g = self.net.forward(inv)

        T_list = tensor_basis(S_hat, O_hat)
        n_use = min(len(T_list), self.n_basis)
        a = np.zeros_like(S)
        for n in range(n_use):
            a += g[:, n:n + 1, None] * T_list[n]
        return a


# ── Closure Trainer ───────────────────────────────────────────────

@dataclass
class ClosureTrainingResult:
    losses: List[float]
    final_mse: float
    n_epochs: int


class ClosureTrainer:
    """Train closure models against DNS/high-fidelity data."""

    def __init__(self, closure: Any, lr: float = 1e-3) -> None:
        self.closure = closure
        self.lr = lr

    def train(
        self,
        grad_u_data: np.ndarray,    # (N, 3, 3)
        target_stress: np.ndarray,  # (N, 3, 3)
        n_epochs: int = 100,
    ) -> ClosureTrainingResult:
        """Train the closure model."""
        losses: List[float] = []

        for epoch in range(n_epochs):
            if hasattr(self.closure, "predict_anisotropy"):
                pred = self.closure.predict_anisotropy(grad_u_data)
            elif hasattr(self.closure, "predict"):
                pred = self.closure.predict(grad_u_data)
            else:
                break

            loss = float(np.mean((pred - target_stress) ** 2))
            losses.append(loss)

            # Update output layer via numerical gradient
            net = self.closure.net if hasattr(self.closure, "net") else None
            if net is not None and net.weights:
                W = net.weights[-1]
                b = net.biases[-1]
                eps = 1e-5

                for idx in range(min(W.size, 200)):
                    flat_idx = np.unravel_index(idx, W.shape)
                    old = float(W[flat_idx])
                    W[flat_idx] = old + eps
                    if hasattr(self.closure, "predict_anisotropy"):
                        p = self.closure.predict_anisotropy(grad_u_data)
                    else:
                        p = self.closure.predict(grad_u_data)
                    lp = float(np.mean((p - target_stress) ** 2))
                    W[flat_idx] = old - eps
                    if hasattr(self.closure, "predict_anisotropy"):
                        p = self.closure.predict_anisotropy(grad_u_data)
                    else:
                        p = self.closure.predict(grad_u_data)
                    lm = float(np.mean((p - target_stress) ** 2))
                    W[flat_idx] = old
                    grad = (lp - lm) / (2 * eps)
                    W[flat_idx] = old - self.lr * grad

        if hasattr(self.closure, "predict_anisotropy"):
            final_pred = self.closure.predict_anisotropy(grad_u_data)
        else:
            final_pred = self.closure.predict(grad_u_data)
        final_mse = float(np.mean((final_pred - target_stress) ** 2))

        return ClosureTrainingResult(
            losses=losses,
            final_mse=final_mse,
            n_epochs=n_epochs,
        )


__all__ = [
    "strain_rate",
    "rotation_rate",
    "invariants_2d",
    "tensor_basis",
    "ReynoldsStressConfig",
    "ReynoldsStressClosure",
    "SubgridFluxConfig",
    "SubgridFluxClosure",
    "TensorBasisClosure",
    "ClosureTrainer",
    "ClosureTrainingResult",
]
