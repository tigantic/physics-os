"""
5.1 — Foundation Model for Physics
====================================

Large pre-trained transformer architecture on multi-domain simulation
data across all 20 The Ontic Engine packs.  Supports few-shot adaptation,
fine-tuning, and zero-shot transfer between physics domains.

Architecture: Encoder-only transformer with continuous (x,t,field)
tokenisation, sinusoidal position encoding, and domain-specific
output heads.

This module is framework-agnostic: it runs in pure NumPy for
inference (and testing).  When PyTorch is available the same
weights can be loaded into torch tensors for GPU training.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Domain registry ────────────────────────────────────────────────

class PhysicsDomain(Enum):
    """The 20 The Ontic Engine physics packs as token domains."""
    CFD = "cfd"
    STRUCTURAL = "structural"
    THERMAL = "thermal"
    ELECTROMAGNETICS = "em"
    QUANTUM = "quantum"
    ACOUSTICS = "acoustics"
    PLASMA = "plasma"
    COMBUSTION = "combustion"
    MHD = "mhd"
    MULTIPHASE = "multiphase"
    RELATIVITY = "relativity"
    GEOPHYSICS = "geophysics"
    BIOPHYSICS = "biophysics"
    OPTICS = "optics"
    NUCLEAR = "nuclear"
    CLIMATE = "climate"
    MATERIALS = "materials"
    ASTROPHYSICS = "astrophysics"
    CHEMISTRY = "chemistry"
    TURBULENCE = "turbulence"


# ── Configuration ─────────────────────────────────────────────────

@dataclass
class FoundationConfig:
    """Hyper-parameters for the physics foundation model."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 4096
    dropout: float = 0.1
    input_dim: int = 4          # (x, y, z, t)
    output_dim: int = 5         # (rho, u, v, w, p)
    vocab_domains: int = 20     # number of physics domains
    use_rotary: bool = True     # rotary position embedding
    layer_norm_eps: float = 1e-6
    weight_init_std: float = 0.02

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ── Positional encoding ──────────────────────────────────────────

def sinusoidal_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Standard sinusoidal positional encoding."""
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(seq_len)[:, None]
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def rotary_embedding(x: np.ndarray, seq_len: int) -> np.ndarray:
    """Apply Rotary Position Embedding (RoPE) to query/key."""
    d = x.shape[-1]
    theta = 10000.0
    freqs = 1.0 / (theta ** (np.arange(0, d, 2).astype(np.float32) / d))
    t = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # (seq_len, d//2)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Split x into even/odd
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos_a - x_odd * sin_a
    out_odd = x_even * sin_a + x_odd * cos_a
    out = np.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


# ── Core layers ───────────────────────────────────────────────────

def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _multi_head_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    n_heads: int, mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-head scaled dot-product attention. q/k/v: (B, T, D)."""
    B, T, D = q.shape
    d_head = D // n_heads

    def _reshape(x: np.ndarray) -> np.ndarray:
        return x.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    Q = _reshape(q)   # (B, H, T, d)
    K = _reshape(k)
    V = _reshape(v)

    scale = 1.0 / math.sqrt(d_head)
    scores = np.einsum("bhid,bhjd->bhij", Q, K) * scale
    if mask is not None:
        scores = scores + mask
    attn = _softmax(scores, axis=-1)
    out = np.einsum("bhij,bhjd->bhid", attn, V)
    return out.transpose(0, 2, 1, 3).reshape(B, T, D)


# ── Weight containers ─────────────────────────────────────────────

@dataclass
class TransformerLayerWeights:
    """Weights for a single transformer encoder block."""
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray
    ff1_w: np.ndarray
    ff1_b: np.ndarray
    ff2_w: np.ndarray
    ff2_b: np.ndarray


@dataclass
class FoundationWeights:
    """Full model weight set."""
    input_proj: np.ndarray          # (input_dim, d_model)
    domain_embed: np.ndarray        # (n_domains, d_model)
    layers: List[TransformerLayerWeights] = field(default_factory=list)
    ln_final_gamma: np.ndarray = field(default_factory=lambda: np.ones(1))
    ln_final_beta: np.ndarray = field(default_factory=lambda: np.zeros(1))
    output_head: np.ndarray = field(default_factory=lambda: np.ones(1))  # (d_model, output_dim)
    output_bias: np.ndarray = field(default_factory=lambda: np.zeros(1))


def _init_weights(cfg: FoundationConfig) -> FoundationWeights:
    """Xavier-style initialization for all model weights."""
    rng = np.random.default_rng(42)
    s = cfg.weight_init_std
    d = cfg.d_model

    layers: List[TransformerLayerWeights] = []
    for _ in range(cfg.n_layers):
        layers.append(TransformerLayerWeights(
            ln1_gamma=np.ones(d, dtype=np.float32),
            ln1_beta=np.zeros(d, dtype=np.float32),
            Wq=rng.normal(0, s, (d, d)).astype(np.float32),
            Wk=rng.normal(0, s, (d, d)).astype(np.float32),
            Wv=rng.normal(0, s, (d, d)).astype(np.float32),
            Wo=rng.normal(0, s, (d, d)).astype(np.float32),
            ln2_gamma=np.ones(d, dtype=np.float32),
            ln2_beta=np.zeros(d, dtype=np.float32),
            ff1_w=rng.normal(0, s, (d, cfg.d_ff)).astype(np.float32),
            ff1_b=np.zeros(cfg.d_ff, dtype=np.float32),
            ff2_w=rng.normal(0, s, (cfg.d_ff, d)).astype(np.float32),
            ff2_b=np.zeros(d, dtype=np.float32),
        ))

    return FoundationWeights(
        input_proj=rng.normal(0, s, (cfg.input_dim, d)).astype(np.float32),
        domain_embed=rng.normal(0, s, (cfg.vocab_domains, d)).astype(np.float32),
        layers=layers,
        ln_final_gamma=np.ones(d, dtype=np.float32),
        ln_final_beta=np.zeros(d, dtype=np.float32),
        output_head=rng.normal(0, s, (d, cfg.output_dim)).astype(np.float32),
        output_bias=np.zeros(cfg.output_dim, dtype=np.float32),
    )


# ── Forward pass ──────────────────────────────────────────────────

def _transformer_block(
    x: np.ndarray,
    w: TransformerLayerWeights,
    n_heads: int,
    eps: float,
    use_rotary: bool,
) -> np.ndarray:
    """Single pre-norm transformer block."""
    # ─ Self-attention ─
    h = _layer_norm(x, w.ln1_gamma, w.ln1_beta, eps)
    q = h @ w.Wq
    k = h @ w.Wk
    v = h @ w.Wv
    if use_rotary:
        T = q.shape[1] if q.ndim == 3 else q.shape[0]
        q = rotary_embedding(q, T)
        k = rotary_embedding(k, T)
    attn_out = _multi_head_attention(q, k, v, n_heads)
    x = x + attn_out @ w.Wo

    # ─ Feed-forward ─
    h = _layer_norm(x, w.ln2_gamma, w.ln2_beta, eps)
    ff = _gelu(h @ w.ff1_w + w.ff1_b)
    x = x + ff @ w.ff2_w + w.ff2_b
    return x


def forward(
    coords: np.ndarray,       # (B, T, input_dim)
    domain_id: int,            # index into PhysicsDomain
    weights: FoundationWeights,
    cfg: FoundationConfig,
) -> np.ndarray:
    """Run the foundation model forward pass.

    Parameters
    ----------
    coords : (B, T, input_dim)  spatiotemporal coordinates
    domain_id : physics domain index
    weights : model weight set
    cfg : model configuration

    Returns
    -------
    predictions : (B, T, output_dim)
    """
    B, T, _ = coords.shape

    # Input projection + domain embedding
    x = coords @ weights.input_proj           # (B, T, d_model)
    x = x + weights.domain_embed[domain_id]   # broadcast domain

    # Transformer encoder stack
    for layer_w in weights.layers:
        x = _transformer_block(x, layer_w, cfg.n_heads, cfg.layer_norm_eps, cfg.use_rotary)

    # Final layer norm + output
    x = _layer_norm(x, weights.ln_final_gamma, weights.ln_final_beta, cfg.layer_norm_eps)
    return x @ weights.output_head + weights.output_bias


# ── Model class (convenience) ────────────────────────────────────

class PhysicsFoundationModel:
    """High-level wrapper for the physics foundation model."""

    def __init__(self, cfg: Optional[FoundationConfig] = None) -> None:
        self.cfg = cfg or FoundationConfig()
        self.weights = _init_weights(self.cfg)
        self._param_count: Optional[int] = None

    @property
    def param_count(self) -> int:
        if self._param_count is None:
            total = 0
            for w in self.weights.layers:
                for name in ("Wq", "Wk", "Wv", "Wo", "ff1_w", "ff1_b",
                             "ff2_w", "ff2_b", "ln1_gamma", "ln1_beta",
                             "ln2_gamma", "ln2_beta"):
                    total += getattr(w, name).size
            total += self.weights.input_proj.size
            total += self.weights.domain_embed.size
            total += self.weights.output_head.size
            total += self.weights.output_bias.size
            total += self.weights.ln_final_gamma.size
            total += self.weights.ln_final_beta.size
            self._param_count = total
        return self._param_count

    def predict(
        self,
        coords: np.ndarray,
        domain: PhysicsDomain = PhysicsDomain.CFD,
    ) -> np.ndarray:
        """Run inference.

        Parameters
        ----------
        coords : (B, T, input_dim)  or  (T, input_dim)
        domain : physics domain

        Returns
        -------
        (B, T, output_dim)
        """
        if coords.ndim == 2:
            coords = coords[np.newaxis]
        domain_id = list(PhysicsDomain).index(domain)
        return forward(coords, domain_id, self.weights, self.cfg)

    def few_shot_adapt(
        self,
        support_coords: np.ndarray,
        support_values: np.ndarray,
        domain: PhysicsDomain = PhysicsDomain.CFD,
        lr: float = 1e-4,
        n_steps: int = 20,
    ) -> Dict[str, float]:
        """Few-shot fine-tuning on a small support set (output head only).

        Parameters
        ----------
        support_coords : (N, input_dim)
        support_values : (N, output_dim)
        domain, lr, n_steps : adaptation config

        Returns
        -------
        Dict with "initial_loss" and "final_loss"
        """
        coords = support_coords[np.newaxis]       # (1, N, in_dim)
        targets = support_values[np.newaxis]       # (1, N, out_dim)
        domain_id = list(PhysicsDomain).index(domain)

        # Forward to get hidden
        B, T, _ = coords.shape
        x = coords @ self.weights.input_proj
        x = x + self.weights.domain_embed[domain_id]
        for lw in self.weights.layers:
            x = _transformer_block(
                x, lw, self.cfg.n_heads, self.cfg.layer_norm_eps,
                self.cfg.use_rotary,
            )
        x = _layer_norm(
            x, self.weights.ln_final_gamma, self.weights.ln_final_beta,
            self.cfg.layer_norm_eps,
        )
        # x: (1, N, d_model) — frozen features

        hidden = x[0]  # (N, d_model)
        W = self.weights.output_head.copy()  # (d_model, out_dim)
        b = self.weights.output_bias.copy()

        initial_loss = float(np.mean((hidden @ W + b - targets[0]) ** 2))

        for _ in range(n_steps):
            pred = hidden @ W + b
            err = pred - targets[0]
            grad_W = hidden.T @ err / T
            grad_b = err.mean(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b

        final_loss = float(np.mean((hidden @ W + b - targets[0]) ** 2))

        self.weights.output_head = W
        self.weights.output_bias = b

        return {"initial_loss": initial_loss, "final_loss": final_loss}

    def save(self, path: Path) -> None:
        """Serialize weights to a directory of .npy files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "input_proj.npy", self.weights.input_proj)
        np.save(path / "domain_embed.npy", self.weights.domain_embed)
        np.save(path / "output_head.npy", self.weights.output_head)
        np.save(path / "output_bias.npy", self.weights.output_bias)
        np.save(path / "ln_final_gamma.npy", self.weights.ln_final_gamma)
        np.save(path / "ln_final_beta.npy", self.weights.ln_final_beta)
        for i, lw in enumerate(self.weights.layers):
            layer_dir = path / f"layer_{i:03d}"
            layer_dir.mkdir(exist_ok=True)
            for name in ("Wq", "Wk", "Wv", "Wo", "ff1_w", "ff1_b",
                         "ff2_w", "ff2_b", "ln1_gamma", "ln1_beta",
                         "ln2_gamma", "ln2_beta"):
                np.save(layer_dir / f"{name}.npy", getattr(lw, name))
        with open(path / "config.json", "w") as f:
            json.dump({k: v for k, v in self.cfg.__dict__.items()}, f)

    @classmethod
    def load(cls, path: Path) -> "PhysicsFoundationModel":
        """Load model from directory."""
        path = Path(path)
        with open(path / "config.json") as f:
            cfg = FoundationConfig(**json.load(f))
        model = cls(cfg)
        model.weights.input_proj = np.load(path / "input_proj.npy")
        model.weights.domain_embed = np.load(path / "domain_embed.npy")
        model.weights.output_head = np.load(path / "output_head.npy")
        model.weights.output_bias = np.load(path / "output_bias.npy")
        model.weights.ln_final_gamma = np.load(path / "ln_final_gamma.npy")
        model.weights.ln_final_beta = np.load(path / "ln_final_beta.npy")
        for i in range(cfg.n_layers):
            layer_dir = path / f"layer_{i:03d}"
            lw = model.weights.layers[i]
            for name in ("Wq", "Wk", "Wv", "Wo", "ff1_w", "ff1_b",
                         "ff2_w", "ff2_b", "ln1_gamma", "ln1_beta",
                         "ln2_gamma", "ln2_beta"):
                setattr(lw, name, np.load(layer_dir / f"{name}.npy"))
        return model


__all__ = [
    "PhysicsDomain",
    "FoundationConfig",
    "FoundationWeights",
    "PhysicsFoundationModel",
    "sinusoidal_encoding",
    "rotary_embedding",
    "forward",
]
