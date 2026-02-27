"""
Machine Learning for Physics — Physics-Informed Neural Networks (PINN),
Fourier Neural Operator (FNO), SchNet-style NNP.

Domain XVII.3 — NEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Dense Layer (minimal NN building block)
# ---------------------------------------------------------------------------

class DenseLayer:
    """Simple dense (fully-connected) layer with Xavier init."""

    def __init__(self, n_in: int, n_out: int, activation: str = 'tanh',
                 seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        limit = math.sqrt(6 / (n_in + n_out))
        self.W = rng.uniform(-limit, limit, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.activation = activation

        self._input: Optional[NDArray] = None
        self._pre_act: Optional[NDArray] = None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: NDArray) -> NDArray:
        self._input = x
        z = x @ self.W + self.b
        self._pre_act = z
        if self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(z, 0)
        elif self.activation == 'linear':
            return z
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return z

    def backward(self, grad_out: NDArray) -> NDArray:
        """Backpropagate gradient."""
        if self.activation == 'tanh':
            da = 1 - np.tanh(self._pre_act)**2
        elif self.activation == 'relu':
            da = (self._pre_act > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(self._pre_act, -500, 500)))
            da = s * (1 - s)
        else:
            da = np.ones_like(self._pre_act)

        delta = grad_out * da
        self.dW = self._input.T @ delta
        self.db = np.sum(delta, axis=0)
        return delta @ self.W.T


# ---------------------------------------------------------------------------
#  Multi-Layer Perceptron
# ---------------------------------------------------------------------------

class MLP:
    """Simple MLP for use in PINN / FNO / NNP architectures."""

    def __init__(self, layer_sizes: List[int], activation: str = 'tanh',
                 seed: int = 42) -> None:
        self.layers: List[DenseLayer] = []
        for i in range(len(layer_sizes) - 1):
            act = activation if i < len(layer_sizes) - 2 else 'linear'
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i + 1],
                                           act, seed=seed + i))

    def forward(self, x: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: NDArray) -> NDArray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> List[Tuple[NDArray, NDArray]]:
        return [(l.W, l.b) for l in self.layers]

    def update(self, lr: float = 1e-3) -> None:
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db


# ---------------------------------------------------------------------------
#  Physics-Informed Neural Network (PINN)
# ---------------------------------------------------------------------------

class PINN:
    r"""
    Physics-Informed Neural Network (Raissi et al., 2019).

    Loss = MSE_data + λ · MSE_physics

    For PDE: $\mathcal{N}[u](x,t) = 0$ → residual loss.

    Example: 1D Burgers equation:
    $$u_t + uu_x = \nu u_{xx}$$

    Uses automatic differentiation via finite differences on the MLP.
    """

    def __init__(self, layer_sizes: List[int],
                 pde_residual: Callable,
                 lambda_physics: float = 1.0,
                 seed: int = 42) -> None:
        self.net = MLP(layer_sizes, 'tanh', seed)
        self.pde_residual = pde_residual
        self.lambda_physics = lambda_physics

    def predict(self, x: NDArray) -> NDArray:
        return self.net.forward(x)

    def compute_derivatives(self, x: NDArray, eps: float = 1e-4) -> Dict[str, NDArray]:
        """Numerical derivatives for physics residual.

        x: (N, d) input. Returns dict with 'u', 'u_x', 'u_t', 'u_xx', etc.
        Assumes x[:, 0] = spatial, x[:, 1] = temporal.
        """
        u = self.predict(x)
        d = x.shape[1]

        derivs: Dict[str, NDArray] = {'u': u}

        if d >= 1:
            x_p = x.copy(); x_p[:, 0] += eps
            x_m = x.copy(); x_m[:, 0] -= eps
            derivs['u_x'] = (self.predict(x_p) - self.predict(x_m)) / (2 * eps)
            derivs['u_xx'] = (self.predict(x_p) - 2 * u + self.predict(x_m)) / eps**2

        if d >= 2:
            x_tp = x.copy(); x_tp[:, 1] += eps
            x_tm = x.copy(); x_tm[:, 1] -= eps
            derivs['u_t'] = (self.predict(x_tp) - self.predict(x_tm)) / (2 * eps)

        return derivs

    def loss(self, x_data: NDArray, u_data: NDArray,
               x_colloc: NDArray) -> Tuple[float, float, float]:
        """Compute PINN loss.

        Returns (total_loss, data_loss, physics_loss).
        """
        u_pred = self.predict(x_data)
        data_loss = float(np.mean((u_pred - u_data)**2))

        derivs = self.compute_derivatives(x_colloc)
        residual = self.pde_residual(derivs)
        physics_loss = float(np.mean(residual**2))

        total = data_loss + self.lambda_physics * physics_loss
        return total, data_loss, physics_loss

    def train_step(self, x_data: NDArray, u_data: NDArray,
                     x_colloc: NDArray, lr: float = 1e-3) -> float:
        """One training step with gradient descent."""
        # Data loss gradient
        u_pred = self.net.forward(x_data)
        grad_data = 2 * (u_pred - u_data) / len(u_data)
        self.net.backward(grad_data)

        # Store data gradients
        data_grads = [(l.dW.copy(), l.db.copy()) for l in self.net.layers]

        # Physics loss gradient (via perturbation)
        derivs = self.compute_derivatives(x_colloc)
        residual = self.pde_residual(derivs)
        grad_phys = 2 * residual / len(x_colloc) * self.lambda_physics
        _ = self.net.forward(x_colloc)
        self.net.backward(grad_phys)

        # Combine gradients
        for i, layer in enumerate(self.net.layers):
            layer.dW += data_grads[i][0]
            layer.db += data_grads[i][1]

        self.net.update(lr)

        total, _, _ = self.loss(x_data, u_data, x_colloc)
        return total


# ---------------------------------------------------------------------------
#  Fourier Neural Operator (FNO) — 1D
# ---------------------------------------------------------------------------

class FourierNeuralOperator1D:
    r"""
    Fourier Neural Operator (Li et al., 2020) for learning operator mappings.

    Spectral convolution layer:
    $$(Kv)(x) = \mathcal{F}^{-1}\!\left[R_\phi\cdot\mathcal{F}[v]\right](x)$$

    Full layer: $v^{(l+1)} = \sigma(W v^{(l)} + K v^{(l)} + b)$.

    Learns a → u(a) mapping: input function to solution.
    Resolution-independent once trained.
    """

    def __init__(self, n_modes: int = 16, width: int = 32,
                 n_layers: int = 4, seed: int = 42) -> None:
        self.n_modes = n_modes
        self.width = width
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)

        # Spectral weights: complex R_phi[modes, width, width]
        self.R = []
        for _ in range(n_layers):
            R_real = rng.standard_normal((n_modes, width, width)) * 0.02
            R_imag = rng.standard_normal((n_modes, width, width)) * 0.02
            self.R.append(R_real + 1j * R_imag)

        # Linear bypass
        self.W = [rng.standard_normal((width, width)) * 0.02 for _ in range(n_layers)]
        self.b = [np.zeros(width) for _ in range(n_layers)]

        # Lifting and projection
        self.lift = rng.standard_normal((2, width)) * 0.1  # input: (a, x) → width
        self.proj = rng.standard_normal((width, 1)) * 0.1  # width → output

    def spectral_conv(self, v: NDArray, R: NDArray) -> NDArray:
        """Spectral convolution: F^{-1}(R · F(v)).

        v: (nx, width).
        R: (modes, width, width).
        """
        nx = v.shape[0]
        v_hat = np.fft.rfft(v, axis=0)  # (nx//2+1, width)

        n_modes = min(R.shape[0], v_hat.shape[0])
        out_hat = np.zeros_like(v_hat)

        for m in range(n_modes):
            out_hat[m] = v_hat[m] @ R[m]

        return np.fft.irfft(out_hat, n=nx, axis=0)

    def forward(self, a: NDArray, x: NDArray) -> NDArray:
        """Forward pass.

        a: (nx,) input function values.
        x: (nx,) grid points.
        Returns: (nx, 1) output.
        """
        nx = len(a)
        inp = np.stack([a, x], axis=-1)  # (nx, 2)
        v = inp @ self.lift  # (nx, width)

        for l in range(self.n_layers):
            v1 = self.spectral_conv(v, self.R[l])
            v2 = v @ self.W[l]
            v = np.tanh(v1 + v2 + self.b[l])

        return v @ self.proj  # (nx, 1)


# ---------------------------------------------------------------------------
#  SchNet-Style Neural Network Potential
# ---------------------------------------------------------------------------

class SchNetNNP:
    r"""
    SchNet-inspired neural network potential (Schütt et al., 2017).

    $E = \sum_i \text{MLP}(\mathbf{x}_i)$
    $\mathbf{x}_i = \sum_{j\neq i} W \cdot \text{RBF}(r_{ij})$

    Radial basis functions: $e_k(r) = \exp(-\gamma_k(r-\mu_k)^2)$

    Interatomic potential from learned representations.
    Forces via negative gradient: $F_i = -\partial E/\partial r_i$.
    """

    def __init__(self, n_rbf: int = 20, cutoff: float = 5.0,
                 n_hidden: int = 64, seed: int = 42) -> None:
        self.n_rbf = n_rbf
        self.cutoff = cutoff

        # RBF centres (uniformly spaced)
        self.mu = np.linspace(0.1, cutoff, n_rbf)
        self.gamma = np.ones(n_rbf) / (0.5 * (self.mu[1] - self.mu[0]))**2

        # Interaction network
        self.W_int = np.random.default_rng(seed).standard_normal((n_rbf, n_hidden)) * 0.1
        self.mlp = MLP([n_hidden, n_hidden, 1], 'tanh', seed)

    def rbf(self, r: float) -> NDArray:
        """Radial basis functions: e_k(r) = exp(−γ_k(r−μ_k)²)."""
        return np.exp(-self.gamma * (r - self.mu)**2)

    def cutoff_function(self, r: float) -> float:
        """Cosine cutoff: f_c(r) = ½(cos(πr/r_c)+1) for r < r_c, else 0."""
        if r >= self.cutoff:
            return 0.0
        return 0.5 * (math.cos(math.pi * r / self.cutoff) + 1)

    def energy(self, positions: NDArray, species: Optional[NDArray] = None) -> float:
        """Total energy for a set of atoms.

        positions: (N, 3).
        Returns scalar energy.
        """
        N = len(positions)
        E_total = 0.0

        for i in range(N):
            x_i = np.zeros(self.W_int.shape[1])

            for j in range(N):
                if i == j:
                    continue
                r_ij = float(np.linalg.norm(positions[j] - positions[i]))
                if r_ij >= self.cutoff:
                    continue

                fc = self.cutoff_function(r_ij)
                rbf_ij = self.rbf(r_ij)
                x_i += fc * (rbf_ij @ self.W_int)

            E_i = self.mlp.forward(x_i.reshape(1, -1))
            E_total += float(E_i[0, 0])

        return E_total

    def forces(self, positions: NDArray, eps: float = 1e-4) -> NDArray:
        """Forces via finite differences: F_i = −∂E/∂r_i."""
        N = len(positions)
        F = np.zeros((N, 3))

        for i in range(N):
            for d in range(3):
                pos_p = positions.copy()
                pos_m = positions.copy()
                pos_p[i, d] += eps
                pos_m[i, d] -= eps
                F[i, d] = -(self.energy(pos_p) - self.energy(pos_m)) / (2 * eps)

        return F
