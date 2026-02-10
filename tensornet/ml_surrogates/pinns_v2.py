"""
5.3 — PINNs 2.0: Causal, Separated, and Competitive variants
================================================================

Extends the existing PINN framework with three advanced training
strategies from the literature:

* **CausalPINN** — time-causal loss weighting (Krishnapriyan et al.);
  earlier time-steps are weighted more heavily to prevent backward
  information leakage.
* **SeparatedPINN** — separate sub-networks for each solution variable
  with shared physics coupling (Karniadakis group).
* **CompetitivePINN** — adversarial training where a discriminator
  network identifies high-residual regions and the generator
  (solver network) adapts (Zeng et al.).

All implement a uniform API:  ``train_step(coords, targets) → losses``.
Pure NumPy for the core logic; torch adapter provided when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Shared utilities ──────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


class DenseNet:
    """Minimal fully-connected network for PINN building blocks."""

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        seed: int = 0,
    ) -> None:
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = math.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(rng.normal(0, std, (fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return _tanh(x)
        elif self.activation == "relu":
            return _relu(x)
        elif self.activation == "sigmoid":
            return _sigmoid(x)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(len(self.weights) - 1):
            x = self._act(x @ self.weights[i] + self.biases[i])
        x = x @ self.weights[-1] + self.biases[-1]
        return x

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


# ── Finite-difference PDE residual ───────────────────────────────

def fd_gradient(net: DenseNet, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Approximate ∂u/∂xᵢ via central differences. Returns (N, out, in_dim)."""
    N, d = x.shape
    u0 = net.forward(x)  # (N, out)
    out_dim = u0.shape[-1]
    grads = np.zeros((N, out_dim, d), dtype=np.float32)
    for i in range(d):
        x_p = x.copy()
        x_m = x.copy()
        x_p[:, i] += eps
        x_m[:, i] -= eps
        grads[:, :, i] = (net.forward(x_p) - net.forward(x_m)) / (2 * eps)
    return grads


def fd_laplacian(net: DenseNet, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Approximate Laplacian ∇²u via central differences. Returns (N, out)."""
    N, d = x.shape
    u0 = net.forward(x)
    lap = np.zeros_like(u0)
    for i in range(d):
        x_p = x.copy()
        x_m = x.copy()
        x_p[:, i] += eps
        x_m[:, i] -= eps
        lap += (net.forward(x_p) - 2 * u0 + net.forward(x_m)) / (eps * eps)
    return lap


# ── SGD helper ────────────────────────────────────────────────────

def _sgd_step(params: List[np.ndarray], grads: List[np.ndarray], lr: float) -> None:
    """In-place SGD update."""
    for p, g in zip(params, grads):
        p -= lr * g


def _numerical_grads(
    loss_fn: Callable[[], float],
    params: List[np.ndarray],
    eps: float = 1e-5,
) -> List[np.ndarray]:
    """Numerical gradient w.r.t. each parameter for small-network training."""
    grads = []
    for p in params:
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = float(p[idx])
            p[idx] = old + eps
            loss_p = loss_fn()
            p[idx] = old - eps
            loss_m = loss_fn()
            g[idx] = (loss_p - loss_m) / (2 * eps)
            p[idx] = old
            it.iternext()
        grads.append(g)
    return grads


# ═══════════════════════════════════════════════════════════════════
# 1) CausalPINN
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalPINNConfig:
    """Configuration for Causal PINN."""
    input_dim: int = 3
    output_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"
    n_time_windows: int = 10
    causality_tolerance: float = 1.0
    epsilon: float = 1.0       # causal weight steepness

    @property
    def layer_sizes(self) -> List[int]:
        return [self.input_dim] + list(self.hidden_dims) + [self.output_dim]


class CausalPINN:
    """Time-causal PINN that weights physics loss by time window.

    Each time window t_k only receives full physics loss weight
    after the residual in t_{k-1} has dropped below a tolerance.
    This prevents the network from fitting later times with
    incorrect propagation from earlier times.
    """

    def __init__(self, cfg: Optional[CausalPINNConfig] = None) -> None:
        self.cfg = cfg or CausalPINNConfig()
        self.net = DenseNet(self.cfg.layer_sizes, self.cfg.activation, seed=42)
        self.time_weights = np.ones(self.cfg.n_time_windows, dtype=np.float32)

    def _assign_windows(self, t: np.ndarray) -> List[np.ndarray]:
        """Split collocation points into temporal windows."""
        t_min, t_max = float(t.min()), float(t.max())
        if t_max <= t_min:
            return [np.arange(len(t))]
        edges = np.linspace(t_min, t_max, self.cfg.n_time_windows + 1)
        windows = []
        for k in range(self.cfg.n_time_windows):
            mask = (t >= edges[k]) & (t < edges[k + 1])
            if k == self.cfg.n_time_windows - 1:
                mask = mask | (t == edges[k + 1])
            windows.append(np.where(mask)[0])
        return windows

    def update_causal_weights(self, residuals_per_window: List[float]) -> None:
        """Update time-window weights based on residual magnitudes."""
        for k in range(self.cfg.n_time_windows):
            prefix_loss = sum(residuals_per_window[:k + 1])
            self.time_weights[k] = float(np.exp(
                -self.cfg.epsilon * prefix_loss
            ))

    def compute_physics_residual(
        self,
        colloc: np.ndarray,
        pde_residual_fn: Callable[[DenseNet, np.ndarray], np.ndarray],
    ) -> Tuple[float, List[float]]:
        """Compute causally-weighted physics residual.

        Parameters
        ----------
        colloc : (N, input_dim) — include time as last column
        pde_residual_fn : f(net, coords) → (N,) residual

        Returns
        -------
        total_loss, per_window_residuals
        """
        t_col = colloc[:, -1]  # time is last coordinate
        windows = self._assign_windows(t_col)
        residual = pde_residual_fn(self.net, colloc)

        per_window: List[float] = []
        total = 0.0
        for k, idx in enumerate(windows):
            if len(idx) == 0:
                per_window.append(0.0)
                continue
            r_k = float(np.mean(residual[idx] ** 2))
            per_window.append(r_k)
            total += self.time_weights[k] * r_k

        self.update_causal_weights(per_window)
        return total, per_window

    def train_step(
        self,
        data_coords: np.ndarray,
        data_targets: np.ndarray,
        colloc: np.ndarray,
        pde_residual_fn: Callable[[DenseNet, np.ndarray], np.ndarray],
        lr: float = 1e-3,
        data_weight: float = 1.0,
        physics_weight: float = 1.0,
    ) -> Dict[str, float]:
        """Single training step with causal weighting."""
        pred = self.net.forward(data_coords)
        data_loss = _mse(pred, data_targets)
        phys_loss, per_win = self.compute_physics_residual(colloc, pde_residual_fn)

        def total_loss_fn() -> float:
            p = self.net.forward(data_coords)
            dl = _mse(p, data_targets)
            pl, _ = self.compute_physics_residual(colloc, pde_residual_fn)
            return data_weight * dl + physics_weight * pl

        grads = _numerical_grads(total_loss_fn, self.net.parameters())
        _sgd_step(self.net.parameters(), grads, lr)

        return {
            "data_loss": data_loss,
            "physics_loss": phys_loss,
            "total_loss": data_weight * data_loss + physics_weight * phys_loss,
            "causal_weights": self.time_weights.tolist(),
        }


# ═══════════════════════════════════════════════════════════════════
# 2) SeparatedPINN
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SeparatedPINNConfig:
    """Configuration for Separated PINN with per-variable sub-networks."""
    input_dim: int = 3
    n_variables: int = 4        # e.g. rho, u, v, p
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"
    coupling_weight: float = 1.0

    @property
    def subnet_sizes(self) -> List[int]:
        return [self.input_dim] + list(self.hidden_dims) + [1]


class SeparatedPINN:
    """Separated PINN: one sub-network per physical variable.

    Each variable has its own network with shared input, but physics
    coupling is enforced through cross-variable PDE residuals.
    """

    def __init__(self, cfg: Optional[SeparatedPINNConfig] = None) -> None:
        self.cfg = cfg or SeparatedPINNConfig()
        self.subnets: List[DenseNet] = [
            DenseNet(self.cfg.subnet_sizes, self.cfg.activation, seed=i)
            for i in range(self.cfg.n_variables)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass concatenating all sub-network outputs.

        Returns (N, n_variables).
        """
        outputs = [net.forward(x) for net in self.subnets]
        return np.concatenate(outputs, axis=-1)

    @property
    def param_count(self) -> int:
        return sum(s.param_count for s in self.subnets)

    def all_parameters(self) -> List[np.ndarray]:
        params: List[np.ndarray] = []
        for s in self.subnets:
            params.extend(s.parameters())
        return params

    def train_step(
        self,
        data_coords: np.ndarray,
        data_targets: np.ndarray,
        colloc: np.ndarray,
        pde_residual_fn: Callable[["SeparatedPINN", np.ndarray], np.ndarray],
        lr: float = 1e-3,
        data_weight: float = 1.0,
        physics_weight: float = 1.0,
    ) -> Dict[str, float]:
        """Training step for separated PINN."""
        pred = self.forward(data_coords)
        data_loss = _mse(pred, data_targets)
        residual = pde_residual_fn(self, colloc)
        phys_loss = float(np.mean(residual ** 2))

        def total_fn() -> float:
            p = self.forward(data_coords)
            dl = _mse(p, data_targets)
            r = pde_residual_fn(self, colloc)
            pl = float(np.mean(r ** 2))
            return data_weight * dl + physics_weight * pl

        grads = _numerical_grads(total_fn, self.all_parameters())
        _sgd_step(self.all_parameters(), grads, lr)

        return {
            "data_loss": data_loss,
            "physics_loss": phys_loss,
            "total_loss": data_weight * data_loss + physics_weight * phys_loss,
        }


# ═══════════════════════════════════════════════════════════════════
# 3) CompetitivePINN
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CompetitivePINNConfig:
    """Configuration for Competitive (adversarial) PINN."""
    input_dim: int = 3
    output_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    disc_hidden_dims: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = "tanh"
    disc_lr_ratio: float = 5.0  # discriminator learns faster
    n_disc_steps: int = 3       # discriminator steps per generator step

    @property
    def gen_sizes(self) -> List[int]:
        return [self.input_dim] + list(self.hidden_dims) + [self.output_dim]

    @property
    def disc_sizes(self) -> List[int]:
        # discriminator maps coords → weight (scalar)
        return [self.input_dim] + list(self.disc_hidden_dims) + [1]


class CompetitivePINN:
    """Adversarial PINN with a discriminator that finds hard regions.

    The discriminator learns a spatial weight function w(x) that
    up-weights collocation points where the PDE residual is high.
    The generator (solver network) then minimises the weighted
    residual, creating a minimax game.
    """

    def __init__(self, cfg: Optional[CompetitivePINNConfig] = None) -> None:
        self.cfg = cfg or CompetitivePINNConfig()
        self.generator = DenseNet(self.cfg.gen_sizes, self.cfg.activation, seed=0)
        self.discriminator = DenseNet(self.cfg.disc_sizes, "sigmoid", seed=99)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.generator.forward(x)

    def discriminator_weights(self, x: np.ndarray) -> np.ndarray:
        """Get spatial weights from discriminator (all positive via softplus)."""
        raw = self.discriminator.forward(x)
        return np.log1p(np.exp(np.clip(raw, -20, 20)))  # softplus

    def train_step(
        self,
        data_coords: np.ndarray,
        data_targets: np.ndarray,
        colloc: np.ndarray,
        pde_residual_fn: Callable[[DenseNet, np.ndarray], np.ndarray],
        lr: float = 1e-3,
        data_weight: float = 1.0,
        physics_weight: float = 1.0,
    ) -> Dict[str, float]:
        """Minimax training step.

        1. Update discriminator to maximise weighted residual.
        2. Update generator to minimise weighted residual.
        """
        residual = pde_residual_fn(self.generator, colloc)
        r_sq = residual ** 2

        # ── Discriminator steps ──
        for _ in range(self.cfg.n_disc_steps):
            def disc_loss_fn() -> float:
                w = self.discriminator_weights(colloc).squeeze()
                return -float(np.mean(w * r_sq))

            d_grads = _numerical_grads(disc_loss_fn, self.discriminator.parameters())
            _sgd_step(
                self.discriminator.parameters(),
                d_grads,
                lr * self.cfg.disc_lr_ratio,
            )

        # ── Generator step ──
        spatial_w = self.discriminator_weights(colloc).squeeze()
        weighted_phys = float(np.mean(spatial_w * r_sq))
        pred = self.generator.forward(data_coords)
        data_loss = _mse(pred, data_targets)

        def gen_loss_fn() -> float:
            p = self.generator.forward(data_coords)
            dl = _mse(p, data_targets)
            r = pde_residual_fn(self.generator, colloc)
            w = self.discriminator_weights(colloc).squeeze()
            pl = float(np.mean(w * r ** 2))
            return data_weight * dl + physics_weight * pl

        g_grads = _numerical_grads(gen_loss_fn, self.generator.parameters())
        _sgd_step(self.generator.parameters(), g_grads, lr)

        return {
            "data_loss": data_loss,
            "physics_loss": weighted_phys,
            "total_loss": data_weight * data_loss + physics_weight * weighted_phys,
            "disc_mean_weight": float(spatial_w.mean()),
        }


__all__ = [
    "DenseNet",
    "CausalPINNConfig",
    "CausalPINN",
    "SeparatedPINNConfig",
    "SeparatedPINN",
    "CompetitivePINNConfig",
    "CompetitivePINN",
    "fd_gradient",
    "fd_laplacian",
]
