"""
5.9 — Neural PDE Correctors
==============================

Learned residual correction networks that augment classical PDE
solvers.  The corrector maps (coarse solution, local features) →
correction field that is added to the coarse solution.

Components:
    * CorrectorNet — base residual-correction MLP
    * SpectralCorrector — correction in Fourier space
    * AdaptiveCorrector — learns where correction is needed
    * CorrectorTrainer — training loop with solver-in-the-loop
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Utility ───────────────────────────────────────────────────────

def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# ── Dense block ───────────────────────────────────────────────────

class DenseBlock:
    """Two-layer residual MLP block."""

    def __init__(self, dim: int, hidden_factor: int = 4, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        h = dim * hidden_factor
        s = math.sqrt(2.0 / (dim + h))
        self.W1 = rng.normal(0, s, (dim, h)).astype(np.float32)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = rng.normal(0, s, (h, dim)).astype(np.float32)
        self.b2 = np.zeros(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = _silu(x @ self.W1 + self.b1)
        return x + h @ self.W2 + self.b2

    @property
    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# ── CorrectorNet ─────────────────────────────────────────────────

@dataclass
class CorrectorConfig:
    """Configuration for neural PDE corrector."""
    input_dim: int = 8        # coarse solution fields
    hidden_dim: int = 128
    n_blocks: int = 4
    output_dim: int = 4       # correction fields
    residual_scale: float = 0.1  # initial scaling of correction


class CorrectorNet:
    """Residual correction network.

    Given a coarse PDE solution u_h, predicts a correction δu
    such that u_h + α·δu ≈ u_exact.
    """

    def __init__(self, cfg: Optional[CorrectorConfig] = None) -> None:
        self.cfg = cfg or CorrectorConfig()
        rng = np.random.default_rng(0)

        # Input projection
        s = math.sqrt(2.0 / (self.cfg.input_dim + self.cfg.hidden_dim))
        self.proj_in = rng.normal(
            0, s, (self.cfg.input_dim, self.cfg.hidden_dim)
        ).astype(np.float32)

        # Residual blocks
        self.blocks = [
            DenseBlock(self.cfg.hidden_dim, seed=i * 7)
            for i in range(self.cfg.n_blocks)
        ]

        # Output projection
        s2 = math.sqrt(2.0 / (self.cfg.hidden_dim + self.cfg.output_dim))
        self.proj_out = rng.normal(
            0, s2, (self.cfg.hidden_dim, self.cfg.output_dim)
        ).astype(np.float32)
        self.scale = self.cfg.residual_scale

    def forward(self, coarse_solution: np.ndarray) -> np.ndarray:
        """Predict correction.  coarse_solution: (..., input_dim) → (..., output_dim)."""
        h = coarse_solution @ self.proj_in
        for block in self.blocks:
            h = _layer_norm(block.forward(h))
        return self.scale * (h @ self.proj_out)

    def correct(self, coarse_solution: np.ndarray) -> np.ndarray:
        """Return corrected solution.

        Assumes output_dim ≤ input_dim and correction applies
        to the first output_dim channels.
        """
        delta = self.forward(coarse_solution)
        corrected = coarse_solution.copy()
        corrected[..., :self.cfg.output_dim] += delta
        return corrected

    @property
    def param_count(self) -> int:
        total = self.proj_in.size + self.proj_out.size
        total += sum(b.param_count for b in self.blocks)
        return total


# ── SpectralCorrector ─────────────────────────────────────────────

@dataclass
class SpectralCorrectorConfig:
    """Config for Fourier-space correction."""
    n_modes: int = 16
    n_channels: int = 4
    hidden_dim: int = 64
    n_layers: int = 3


class SpectralCorrector:
    """Correction applied in Fourier space.

    Learns a spectral filter that modifies the low-/mid-frequency
    content of the coarse solution to better match the fine solution.
    """

    def __init__(self, cfg: Optional[SpectralCorrectorConfig] = None) -> None:
        self.cfg = cfg or SpectralCorrectorConfig()
        rng = np.random.default_rng(42)

        # Spectral weights per mode
        self.spectral_weight = rng.normal(
            0, 0.02, (self.cfg.n_modes, self.cfg.n_channels, self.cfg.n_channels)
        ).astype(np.float32) + np.eye(self.cfg.n_channels, dtype=np.float32)

        # MLP for real-space residual
        layers = [self.cfg.n_channels] + [self.cfg.hidden_dim] * self.cfg.n_layers + [self.cfg.n_channels]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            s = math.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.weights.append(rng.normal(0, s, (layers[i], layers[i + 1])).astype(np.float32))
            self.biases.append(np.zeros(layers[i + 1], dtype=np.float32))

    def forward(self, u: np.ndarray) -> np.ndarray:
        """Spectral correction.  u: (N, n_channels) → (N, n_channels)."""
        N = u.shape[0]

        # 1) FFT along spatial axis
        u_hat = np.fft.rfft(u, axis=0)
        n_freq = u_hat.shape[0]
        modes = min(self.cfg.n_modes, n_freq)

        # 2) Apply learned spectral weights to low modes
        for k in range(modes):
            u_hat[k] = u_hat[k] @ self.spectral_weight[k]

        # 3) IFFT back
        spectral_correction = np.fft.irfft(u_hat, n=N, axis=0)

        # 4) Real-space residual MLP
        h = u
        for i in range(len(self.weights) - 1):
            h = _silu(h @ self.weights[i] + self.biases[i])
        real_correction = h @ self.weights[-1] + self.biases[-1]

        return spectral_correction + 0.1 * real_correction - u

    def correct(self, u: np.ndarray) -> np.ndarray:
        return u + self.forward(u)

    @property
    def param_count(self) -> int:
        total = self.spectral_weight.size
        total += sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
        return total


# ── AdaptiveCorrector ─────────────────────────────────────────────

@dataclass
class AdaptiveCorrectorConfig:
    """Config for spatially-adaptive correction."""
    input_dim: int = 8
    hidden_dim: int = 64
    output_dim: int = 4
    n_blocks: int = 3
    gate_threshold: float = 0.1


class AdaptiveCorrector:
    """Spatially-adaptive corrector with a learned gating mask.

    A gating network predicts where correction is needed (σ ∈ [0,1]),
    and the correction is only applied in those regions.
    """

    def __init__(self, cfg: Optional[AdaptiveCorrectorConfig] = None) -> None:
        self.cfg = cfg or AdaptiveCorrectorConfig()
        self.corrector = CorrectorNet(CorrectorConfig(
            input_dim=cfg.input_dim if cfg else 8,
            hidden_dim=cfg.hidden_dim if cfg else 64,
            output_dim=cfg.output_dim if cfg else 4,
            n_blocks=cfg.n_blocks if cfg else 3,
        ))

        # Gating network → sigmoid → [0,1]
        rng = np.random.default_rng(99)
        s = math.sqrt(2.0 / (self.cfg.input_dim + self.cfg.hidden_dim))
        self.gate_W1 = rng.normal(
            0, s, (self.cfg.input_dim, self.cfg.hidden_dim)
        ).astype(np.float32)
        self.gate_b1 = np.zeros(self.cfg.hidden_dim, dtype=np.float32)
        self.gate_W2 = rng.normal(
            0, 0.02, (self.cfg.hidden_dim, self.cfg.output_dim)
        ).astype(np.float32)
        self.gate_b2 = np.zeros(self.cfg.output_dim, dtype=np.float32)

    def gate(self, x: np.ndarray) -> np.ndarray:
        """Predict gating mask σ(x) ∈ [0, 1]."""
        h = np.tanh(x @ self.gate_W1 + self.gate_b1)
        logits = h @ self.gate_W2 + self.gate_b2
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    def forward(self, coarse_solution: np.ndarray) -> np.ndarray:
        """Gated correction."""
        delta = self.corrector.forward(coarse_solution)
        mask = self.gate(coarse_solution)
        return delta * mask

    def correct(self, coarse_solution: np.ndarray) -> np.ndarray:
        corrected = coarse_solution.copy()
        corrected[..., :self.cfg.output_dim] += self.forward(coarse_solution)
        return corrected


# ── CorrectorTrainer ──────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Outcome of a corrector training run."""
    losses: List[float]
    final_l2_error: float
    improvement_ratio: float  # ||u_corrected - u_exact|| / ||u_coarse - u_exact||
    n_epochs: int


class CorrectorTrainer:
    """Train a corrector network with solver-in-the-loop.

    Given pairs (u_coarse, u_fine), trains the corrector to minimise
    ||u_coarse + δu - u_fine||².
    """

    def __init__(self, corrector: CorrectorNet, lr: float = 1e-3) -> None:
        self.corrector = corrector
        self.lr = lr

    def train(
        self,
        coarse_data: np.ndarray,   # (N, input_dim)
        fine_data: np.ndarray,     # (N, output_dim)  — target truth
        n_epochs: int = 100,
    ) -> TrainingResult:
        """Train the corrector."""
        losses: List[float] = []
        target = fine_data  # (N, output_dim)

        # Initial error (coarse vs fine)
        uncorrected_err = float(np.mean((
            coarse_data[..., :self.corrector.cfg.output_dim] - target
        ) ** 2))

        for epoch in range(n_epochs):
            delta = self.corrector.forward(coarse_data)
            pred = coarse_data[..., :self.corrector.cfg.output_dim] + delta
            loss = float(np.mean((pred - target) ** 2))
            losses.append(loss)

            # Numerical gradient on output projection (fast approx)
            eps = 1e-5
            for p_idx, p in enumerate([self.corrector.proj_out]):
                grad = np.zeros_like(p)
                for ii in range(min(p.size, 200)):
                    idx = np.unravel_index(ii, p.shape)
                    old = float(p[idx])
                    p[idx] = old + eps
                    l_p = float(np.mean((
                        coarse_data[..., :self.corrector.cfg.output_dim]
                        + self.corrector.forward(coarse_data) - target
                    ) ** 2))
                    p[idx] = old - eps
                    l_m = float(np.mean((
                        coarse_data[..., :self.corrector.cfg.output_dim]
                        + self.corrector.forward(coarse_data) - target
                    ) ** 2))
                    p[idx] = old
                    grad[idx] = (l_p - l_m) / (2 * eps)
                p -= self.lr * grad

        final_delta = self.corrector.forward(coarse_data)
        final_pred = coarse_data[..., :self.corrector.cfg.output_dim] + final_delta
        final_err = float(np.mean((final_pred - target) ** 2))

        return TrainingResult(
            losses=losses,
            final_l2_error=final_err,
            improvement_ratio=final_err / max(uncorrected_err, 1e-15),
            n_epochs=n_epochs,
        )


__all__ = [
    "CorrectorConfig",
    "CorrectorNet",
    "SpectralCorrectorConfig",
    "SpectralCorrector",
    "AdaptiveCorrectorConfig",
    "AdaptiveCorrector",
    "CorrectorTrainer",
    "TrainingResult",
]
