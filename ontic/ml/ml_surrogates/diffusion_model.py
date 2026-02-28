"""
5.6 — Score-Based Diffusion Model for Physics Fields
======================================================

Denoising diffusion probabilistic model (DDPM) adapted for
generating physically consistent field samples:

* Forward process: Gaussian noise schedule  q(x_t | x_{t-1})
* Reverse process: Learned score network  s_θ(x_t, t)
* Conditional generation: class / domain conditioning

Use-cases:
    - Turbulence field generation
    - Uncertainty ensemble sampling
    - Super-resolution for coarse CFD outputs
    - Stochastic forcing term synthesis

Pure NumPy core; torch adapter layer provided separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Noise schedule ────────────────────────────────────────────────

@dataclass
class NoiseSchedule:
    """Variance schedule for diffusion process."""
    n_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "linear"   # "linear" | "cosine"

    def __post_init__(self) -> None:
        if self.schedule_type == "linear":
            self.betas = np.linspace(
                self.beta_start, self.beta_end, self.n_timesteps, dtype=np.float32
            )
        elif self.schedule_type == "cosine":
            steps = np.arange(self.n_timesteps + 1, dtype=np.float64)
            alpha_bar = np.cos((steps / self.n_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = np.clip(betas, 0, 0.999).astype(np.float32)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas).astype(np.float32)
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar).astype(np.float32)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar).astype(np.float32)

    def q_sample(
        self, x0: np.ndarray, t: int, noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion: sample x_t given x_0.

        Returns (x_t, noise).
        """
        if noise is None:
            noise = np.random.randn(*x0.shape).astype(np.float32)
        x_t = (
            self.sqrt_alpha_bar[t] * x0
            + self.sqrt_one_minus_alpha_bar[t] * noise
        )
        return x_t, noise


# ── Score network (simple MLP) ────────────────────────────────────

def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


def _timestep_embedding(t: int, d: int) -> np.ndarray:
    """Sinusoidal time embedding."""
    half = d // 2
    freqs = np.exp(
        -math.log(10000.0) * np.arange(half, dtype=np.float32) / half
    )
    args = float(t) * freqs
    emb = np.concatenate([np.sin(args), np.cos(args)])
    if d % 2:
        emb = np.concatenate([emb, np.zeros(1, dtype=np.float32)])
    return emb


@dataclass
class ScoreNetConfig:
    """Configuration for the score estimation network."""
    field_dim: int = 64         # flattened spatial field dimension
    hidden_dim: int = 256
    n_layers: int = 4
    time_embed_dim: int = 128
    cond_dim: int = 0           # optional conditioning vector dim


class ScoreNet:
    """MLP-based score network  s_θ(x_t, t) ≈ ∇_x log p_t(x_t).

    For production, replace with a U-Net.  This MLP version is
    fully functional for 1-D and low-res 2-D fields.
    """

    def __init__(self, cfg: Optional[ScoreNetConfig] = None) -> None:
        self.cfg = cfg or ScoreNetConfig()
        rng = np.random.default_rng(0)

        in_dim = self.cfg.field_dim + self.cfg.time_embed_dim + self.cfg.cond_dim
        layers = [in_dim] + [self.cfg.hidden_dim] * self.cfg.n_layers + [self.cfg.field_dim]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            std = math.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.weights.append(rng.normal(0, std, (layers[i], layers[i + 1])).astype(np.float32))
            self.biases.append(np.zeros(layers[i + 1], dtype=np.float32))

    def forward(
        self,
        x_t: np.ndarray,        # (B, field_dim)
        t: int,
        cond: Optional[np.ndarray] = None,  # (B, cond_dim) or None
    ) -> np.ndarray:
        """Estimate noise/score. Returns (B, field_dim)."""
        B = x_t.shape[0]
        t_emb = np.tile(_timestep_embedding(t, self.cfg.time_embed_dim), (B, 1))
        parts = [x_t, t_emb]
        if cond is not None:
            parts.append(cond)
        h = np.concatenate(parts, axis=-1)

        for i in range(len(self.weights) - 1):
            h = _silu(h @ self.weights[i] + self.biases[i])
        h = h @ self.weights[-1] + self.biases[-1]
        return h

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


# ── Diffusion Model ──────────────────────────────────────────────

@dataclass
class DiffusionModelConfig:
    """Top-level configuration."""
    field_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 4
    time_embed_dim: int = 128
    cond_dim: int = 0
    n_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "linear"


class PhysicsDiffusionModel:
    """Denoising diffusion model for physics field generation.

    Training workflow:
        1. Sample x_0 from dataset
        2. Sample t ~ Uniform(0, T)
        3. Sample noise ε ~ N(0, I)
        4. Train network to predict ε from x_t

    Sampling:
        Reverse process from x_T ~ N(0, I) → x_0
    """

    def __init__(self, cfg: Optional[DiffusionModelConfig] = None) -> None:
        self.cfg = cfg or DiffusionModelConfig()
        self.schedule = NoiseSchedule(
            n_timesteps=self.cfg.n_timesteps,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            schedule_type=self.cfg.schedule_type,
        )
        self.score_net = ScoreNet(ScoreNetConfig(
            field_dim=self.cfg.field_dim,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.n_layers,
            time_embed_dim=self.cfg.time_embed_dim,
            cond_dim=self.cfg.cond_dim,
        ))

    def training_loss(
        self,
        x0: np.ndarray,         # (B, field_dim)
        cond: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute denoising loss for a random timestep.

        Returns {mse, timestep}.
        """
        B = x0.shape[0]
        t = int(np.random.randint(0, self.cfg.n_timesteps))
        noise = np.random.randn(*x0.shape).astype(np.float32)
        x_t, _ = self.schedule.q_sample(x0, t, noise)

        pred_noise = self.score_net.forward(x_t, t, cond)
        mse = float(np.mean((pred_noise - noise) ** 2))
        return {"mse": mse, "timestep": t}

    def sample(
        self,
        batch_size: int = 1,
        cond: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate samples via ancestral sampling (reverse process).

        Parameters
        ----------
        batch_size : number of fields to generate
        cond : optional conditioning  (batch_size, cond_dim)
        seed : optional RNG seed

        Returns
        -------
        x_0 : (batch_size, field_dim)
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((batch_size, self.cfg.field_dim)).astype(np.float32)

        for t in reversed(range(self.cfg.n_timesteps)):
            pred_noise = self.score_net.forward(x, t, cond)

            alpha = self.schedule.alphas[t]
            alpha_bar = self.schedule.alpha_bar[t]
            beta = self.schedule.betas[t]

            mean = (1.0 / math.sqrt(alpha)) * (
                x - (beta / math.sqrt(1.0 - alpha_bar)) * pred_noise
            )

            if t > 0:
                sigma = math.sqrt(beta)
                z = rng.standard_normal(x.shape).astype(np.float32)
                x = mean + sigma * z
            else:
                x = mean

        return x

    def sample_ddim(
        self,
        batch_size: int = 1,
        n_steps: int = 50,
        eta: float = 0.0,
        cond: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """DDIM (deterministic) sampling with fewer steps.

        Parameters
        ----------
        n_steps : number of denoising steps (T → 0)
        eta : stochasticity (0 = deterministic, 1 = DDPM)
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((batch_size, self.cfg.field_dim)).astype(np.float32)

        # Sub-sample timesteps
        step_size = self.cfg.n_timesteps // n_steps
        timesteps = list(range(0, self.cfg.n_timesteps, step_size))[::-1]

        for i, t in enumerate(timesteps):
            pred_noise = self.score_net.forward(x, t, cond)
            alpha_bar_t = self.schedule.alpha_bar[t]

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.schedule.alpha_bar[t_prev]
            else:
                alpha_bar_prev = 1.0

            # Predicted x_0
            x0_pred = (x - math.sqrt(1 - alpha_bar_t) * pred_noise) / math.sqrt(alpha_bar_t)

            sigma = eta * math.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )

            dir_xt = math.sqrt(max(1 - alpha_bar_prev - sigma ** 2, 0)) * pred_noise
            x = math.sqrt(alpha_bar_prev) * x0_pred + dir_xt

            if sigma > 0:
                x = x + sigma * rng.standard_normal(x.shape).astype(np.float32)

        return x

    def physics_guided_sample(
        self,
        batch_size: int,
        conservation_fn: Optional[callable] = None,
        n_correction_steps: int = 5,
        correction_lr: float = 0.01,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Physics-guided sampling with conservation correction.

        After each denoising step, optionally project the sample
        toward satisfying a conservation constraint.
        """
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((batch_size, self.cfg.field_dim)).astype(np.float32)

        for t in reversed(range(self.cfg.n_timesteps)):
            pred_noise = self.score_net.forward(x, t)
            alpha = self.schedule.alphas[t]
            alpha_bar = self.schedule.alpha_bar[t]
            beta = self.schedule.betas[t]

            mean = (1.0 / math.sqrt(alpha)) * (
                x - (beta / math.sqrt(1.0 - alpha_bar)) * pred_noise
            )

            if t > 0:
                sigma = math.sqrt(beta)
                z = rng.standard_normal(x.shape).astype(np.float32)
                x = mean + sigma * z
            else:
                x = mean

            # Conservation correction
            if conservation_fn is not None and t % 50 == 0:
                for _ in range(n_correction_steps):
                    violation = conservation_fn(x)
                    if isinstance(violation, (int, float)):
                        if abs(violation) < 1e-10:
                            break
                    elif np.max(np.abs(violation)) < 1e-10:
                        break
                    # Gradient-free correction: shift toward zero violation
                    eps = 1e-5
                    grad = np.zeros_like(x)
                    for d in range(x.shape[-1]):
                        x_p = x.copy()
                        x_p[..., d] += eps
                        v_p = conservation_fn(x_p)
                        if isinstance(v_p, (int, float)):
                            v_p = np.full(1, v_p)
                        v_0 = conservation_fn(x)
                        if isinstance(v_0, (int, float)):
                            v_0 = np.full(1, v_0)
                        grad[..., d] = np.mean(v_p - v_0) / eps
                    x = x - correction_lr * grad

        return x


__all__ = [
    "NoiseSchedule",
    "ScoreNetConfig",
    "ScoreNet",
    "DiffusionModelConfig",
    "PhysicsDiffusionModel",
]
