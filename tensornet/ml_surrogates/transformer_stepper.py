"""
5.13 — Transformer Time-Stepper
==================================

Autoregressive transformer for PDE time-stepping.

Takes a window of past solution snapshots as tokens and predicts
the next snapshot.  Supports:
    * Multi-variable fields  (e.g. ρ, u, v, p)
    * Spatial patch tokenisation
    * Causal attention mask (look at past only)
    * Positional encoding with physical time embedding

Pure NumPy implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────

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


def _causal_mask(T: int) -> np.ndarray:
    """Lower-triangular causal mask.  0 = attend, -inf = block."""
    mask = np.full((T, T), -1e9, dtype=np.float32)
    for i in range(T):
        mask[i, :i + 1] = 0.0
    return mask


# ── Configuration ─────────────────────────────────────────────────

@dataclass
class TimeStepperConfig:
    """Hyper-parameters for the transformer time-stepper."""
    n_fields: int = 4          # number of solution variables
    patch_size: int = 8        # spatial patch size (1-D)
    n_patches: int = 16        # patches per snapshot
    window_size: int = 8       # number of past snapshots
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_time: float = 10.0     # maximum physical time for embedding

    @property
    def token_dim(self) -> int:
        """Dimension of a single token = patch_size × n_fields."""
        return self.patch_size * self.n_fields

    @property
    def seq_len(self) -> int:
        """Total sequence length = n_patches × window_size."""
        return self.n_patches * self.window_size

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ── Layer weights ─────────────────────────────────────────────────

@dataclass
class _BlockWeights:
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    ff1_w: np.ndarray
    ff1_b: np.ndarray
    ff2_w: np.ndarray
    ff2_b: np.ndarray


def _init_block(d: int, d_ff: int, rng: np.random.Generator) -> _BlockWeights:
    s = 0.02
    return _BlockWeights(
        ln1_g=np.ones(d, dtype=np.float32),
        ln1_b=np.zeros(d, dtype=np.float32),
        Wq=rng.normal(0, s, (d, d)).astype(np.float32),
        Wk=rng.normal(0, s, (d, d)).astype(np.float32),
        Wv=rng.normal(0, s, (d, d)).astype(np.float32),
        Wo=rng.normal(0, s, (d, d)).astype(np.float32),
        ln2_g=np.ones(d, dtype=np.float32),
        ln2_b=np.zeros(d, dtype=np.float32),
        ff1_w=rng.normal(0, s, (d, d_ff)).astype(np.float32),
        ff1_b=np.zeros(d_ff, dtype=np.float32),
        ff2_w=rng.normal(0, s, (d_ff, d)).astype(np.float32),
        ff2_b=np.zeros(d, dtype=np.float32),
    )


# ── Forward ───────────────────────────────────────────────────────

def _attn(q: np.ndarray, k: np.ndarray, v: np.ndarray,
          n_heads: int, mask: Optional[np.ndarray]) -> np.ndarray:
    B, T, D = q.shape
    d_h = D // n_heads

    def reshape(x: np.ndarray) -> np.ndarray:
        return x.reshape(B, T, n_heads, d_h).transpose(0, 2, 1, 3)

    Q, K, V = reshape(q), reshape(k), reshape(v)
    scores = np.einsum("bhid,bhjd->bhij", Q, K) / math.sqrt(d_h)
    if mask is not None:
        scores = scores + mask
    attn = _softmax(scores, axis=-1)
    out = np.einsum("bhij,bhjd->bhid", attn, V)
    return out.transpose(0, 2, 1, 3).reshape(B, T, D)


def _block_forward(x: np.ndarray, w: _BlockWeights,
                   n_heads: int, mask: Optional[np.ndarray]) -> np.ndarray:
    h = _layer_norm(x, w.ln1_g, w.ln1_b)
    q, k, v = h @ w.Wq, h @ w.Wk, h @ w.Wv
    x = x + _attn(q, k, v, n_heads, mask) @ w.Wo
    h = _layer_norm(x, w.ln2_g, w.ln2_b)
    x = x + _gelu(h @ w.ff1_w + w.ff1_b) @ w.ff2_w + w.ff2_b
    return x


# ── Model ─────────────────────────────────────────────────────────

class TransformerTimeStepper:
    """Autoregressive transformer for PDE time-stepping.

    Input:  window of past snapshots  (B, window_size, n_patches, patch_size*n_fields)
    Output: next snapshot prediction   (B, n_patches, patch_size*n_fields)
    """

    def __init__(self, cfg: Optional[TimeStepperConfig] = None) -> None:
        self.cfg = cfg or TimeStepperConfig()
        rng = np.random.default_rng(42)
        c = self.cfg

        # Token embedding
        s = math.sqrt(2.0 / (c.token_dim + c.d_model))
        self.token_embed = rng.normal(0, s, (c.token_dim, c.d_model)).astype(np.float32)

        # Positional embedding (spatial + temporal)
        self.pos_embed = rng.normal(0, 0.02, (c.seq_len, c.d_model)).astype(np.float32)

        # Transformer blocks
        self.blocks = [
            _init_block(c.d_model, c.d_ff, rng) for _ in range(c.n_layers)
        ]

        # Final norm
        self.ln_final_g = np.ones(c.d_model, dtype=np.float32)
        self.ln_final_b = np.zeros(c.d_model, dtype=np.float32)

        # Output projection
        s2 = math.sqrt(2.0 / (c.d_model + c.token_dim))
        self.out_proj = rng.normal(0, s2, (c.d_model, c.token_dim)).astype(np.float32)

        # Causal mask
        self._mask = _causal_mask(c.seq_len)

    def _tokenise(self, snapshots: np.ndarray) -> np.ndarray:
        """(B, W, P, F) → (B, W*P, F)."""
        B, W, P, F = snapshots.shape
        return snapshots.reshape(B, W * P, F)

    def forward(self, snapshots: np.ndarray) -> np.ndarray:
        """Predict next snapshot.

        Parameters
        ----------
        snapshots : (B, window_size, n_patches, token_dim)

        Returns
        -------
        next_snapshot : (B, n_patches, token_dim)
        """
        B = snapshots.shape[0]
        tokens = self._tokenise(snapshots)  # (B, seq_len, token_dim)
        T = tokens.shape[1]

        # Embed
        x = tokens @ self.token_embed + self.pos_embed[:T]
        x = x.reshape(B, T, -1)  # ensure (B, T, d_model)

        # Transformer
        mask = self._mask[:T, :T]
        for blk in self.blocks:
            x = _block_forward(x, blk, self.cfg.n_heads, mask)

        # Final norm + project
        x = _layer_norm(x, self.ln_final_g, self.ln_final_b)

        # Take last n_patches tokens → next snapshot
        last_tokens = x[:, -self.cfg.n_patches:]  # (B, P, d_model)
        return last_tokens @ self.out_proj          # (B, P, token_dim)

    def autoregressive_rollout(
        self,
        initial_window: np.ndarray,   # (B, W, P, F)
        n_steps: int,
    ) -> np.ndarray:
        """Roll out predictions autoregressively.

        Returns (B, n_steps, n_patches, token_dim).
        """
        B, W, P, F = initial_window.shape
        window = initial_window.copy()
        predictions: List[np.ndarray] = []

        for _ in range(n_steps):
            next_snap = self.forward(window)       # (B, P, F)
            predictions.append(next_snap)
            # Shift window
            window = np.concatenate(
                [window[:, 1:], next_snap[:, np.newaxis]], axis=1,
            )

        return np.stack(predictions, axis=1)       # (B, steps, P, F)

    @property
    def param_count(self) -> int:
        total = self.token_embed.size + self.pos_embed.size
        total += self.out_proj.size
        total += self.ln_final_g.size + self.ln_final_b.size
        for b in self.blocks:
            for attr in ("Wq", "Wk", "Wv", "Wo", "ff1_w", "ff1_b",
                         "ff2_w", "ff2_b", "ln1_g", "ln1_b", "ln2_g", "ln2_b"):
                total += getattr(b, attr).size
        return total


__all__ = [
    "TimeStepperConfig",
    "TransformerTimeStepper",
]
