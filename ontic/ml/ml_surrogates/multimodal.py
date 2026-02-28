"""
5.20 — Multi-Modal Physics AI
================================

Fuses heterogeneous data modalities (fields, point clouds, images,
time-series, text annotations) into a unified physics representation.

Components:
    * FieldEncoder — encode 1-D/2-D fields
    * PointCloudEncoder — encode unstructured point clouds
    * TimeSeriesEncoder — encode temporal signals
    * TextEncoder — encode physics text descriptions
    * MultiModalFusion — cross-modal attention fusion
    * MultiModalPhysicsAI — end-to-end multi-modal model
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Modality types ────────────────────────────────────────────────

class Modality(Enum):
    FIELD = auto()
    POINT_CLOUD = auto()
    TIME_SERIES = auto()
    IMAGE = auto()
    TEXT = auto()


# ── Base MLP ──────────────────────────────────────────────────────

class _MLP:
    def __init__(self, layers: List[int], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            s = math.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.W.append(rng.normal(0, s, (layers[i], layers[i + 1])).astype(np.float32))
            self.b.append(np.zeros(layers[i + 1], dtype=np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for i in range(len(self.W) - 1):
            x = np.tanh(x @ self.W[i] + self.b[i])
        return x @ self.W[-1] + self.b[-1]

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)


# ── Modality encoders ────────────────────────────────────────────

@dataclass
class EncoderConfig:
    """Shared encoder configuration."""
    input_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 64
    n_layers: int = 3


class FieldEncoder:
    """Encode 1-D or flattened 2-D physics fields."""

    def __init__(self, cfg: Optional[EncoderConfig] = None) -> None:
        self.cfg = cfg or EncoderConfig()
        layers = [self.cfg.input_dim] + [self.cfg.hidden_dim] * self.cfg.n_layers + [self.cfg.latent_dim]
        self.net = _MLP(layers, seed=0)
        self.modality = Modality.FIELD

    def encode(self, field: np.ndarray) -> np.ndarray:
        """field: (B, input_dim) → (B, latent_dim)."""
        return self.net(field)


class PointCloudEncoder:
    """Encode unstructured point clouds via DeepSets-style mean pooling."""

    def __init__(self, cfg: Optional[EncoderConfig] = None) -> None:
        self.cfg = cfg or EncoderConfig(input_dim=3)
        point_layers = [self.cfg.input_dim] + [self.cfg.hidden_dim] * 2 + [self.cfg.latent_dim]
        self.point_net = _MLP(point_layers, seed=1)
        self.modality = Modality.POINT_CLOUD

    def encode(self, points: np.ndarray) -> np.ndarray:
        """points: (B, N_points, 3) → (B, latent_dim)."""
        if points.ndim == 2:
            points = points[np.newaxis]
        B, N, D = points.shape
        # Per-point encoding then mean pool
        flat = points.reshape(B * N, D)
        h = self.point_net(flat).reshape(B, N, -1)
        return h.mean(axis=1)


class TimeSeriesEncoder:
    """Encode temporal signals via 1-D sliding window + pooling."""

    def __init__(
        self,
        cfg: Optional[EncoderConfig] = None,
        window_size: int = 16,
    ) -> None:
        self.cfg = cfg or EncoderConfig()
        self.window_size = window_size
        layers = [window_size * self.cfg.input_dim] + \
                 [self.cfg.hidden_dim] * self.cfg.n_layers + [self.cfg.latent_dim]
        self.net = _MLP(layers, seed=2)
        self.modality = Modality.TIME_SERIES

    def encode(self, ts: np.ndarray) -> np.ndarray:
        """ts: (B, T, n_channels) → (B, latent_dim).

        Pads/truncates to window_size and flattens.
        """
        if ts.ndim == 2:
            ts = ts[np.newaxis]
        B, T, C = ts.shape
        # Take last window_size steps
        if T >= self.window_size:
            window = ts[:, -self.window_size:]
        else:
            pad = np.zeros((B, self.window_size - T, C), dtype=ts.dtype)
            window = np.concatenate([pad, ts], axis=1)
        flat = window.reshape(B, -1)
        # Pad or truncate to expected input dim
        expected = self.window_size * self.cfg.input_dim
        if flat.shape[1] < expected:
            flat = np.pad(flat, ((0, 0), (0, expected - flat.shape[1])))
        elif flat.shape[1] > expected:
            flat = flat[:, :expected]
        return self.net(flat)


class TextEncoder:
    """Encode physics text descriptions via bag-of-words + MLP."""

    def __init__(self, cfg: Optional[EncoderConfig] = None, vocab_dim: int = 256) -> None:
        self.cfg = cfg or EncoderConfig()
        self.vocab_dim = vocab_dim
        layers = [vocab_dim] + [self.cfg.hidden_dim] * 2 + [self.cfg.latent_dim]
        self.net = _MLP(layers, seed=3)
        self.modality = Modality.TEXT

    def _text_to_bow(self, text: str) -> np.ndarray:
        """Hash-based bag of words."""
        vec = np.zeros(self.vocab_dim, dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.vocab_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-10)

    def encode(self, texts: List[str]) -> np.ndarray:
        """texts: list of B strings → (B, latent_dim)."""
        bows = np.array([self._text_to_bow(t) for t in texts])
        return self.net(bows)


# ── Cross-modal attention fusion ──────────────────────────────────

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class MultiModalFusion:
    """Cross-modal attention to fuse encodings from different modalities."""

    def __init__(self, latent_dim: int = 64, n_heads: int = 4) -> None:
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        rng = np.random.default_rng(42)
        d = latent_dim
        s = 0.02
        self.Wq = rng.normal(0, s, (d, d)).astype(np.float32)
        self.Wk = rng.normal(0, s, (d, d)).astype(np.float32)
        self.Wv = rng.normal(0, s, (d, d)).astype(np.float32)
        self.Wo = rng.normal(0, s, (d, d)).astype(np.float32)

    def fuse(self, embeddings: Dict[Modality, np.ndarray]) -> np.ndarray:
        """Fuse multi-modal embeddings via cross-attention.

        Parameters
        ----------
        embeddings : {Modality: (B, latent_dim)}

        Returns
        -------
        fused : (B, latent_dim)
        """
        if not embeddings:
            raise ValueError("No embeddings to fuse")

        # Stack as sequence: (B, M, d)
        modalities = sorted(embeddings.keys(), key=lambda m: m.value)
        emb_list = [embeddings[m] for m in modalities]

        # Ensure same batch size
        B = emb_list[0].shape[0]
        M = len(emb_list)
        seq = np.stack(emb_list, axis=1)  # (B, M, d)

        d = self.latent_dim
        d_h = d // self.n_heads

        Q = seq @ self.Wq  # (B, M, d)
        K = seq @ self.Wk
        V = seq @ self.Wv

        # Multi-head attention
        Q = Q.reshape(B, M, self.n_heads, d_h).transpose(0, 2, 1, 3)
        K = K.reshape(B, M, self.n_heads, d_h).transpose(0, 2, 1, 3)
        V = V.reshape(B, M, self.n_heads, d_h).transpose(0, 2, 1, 3)

        scores = np.einsum("bhid,bhjd->bhij", Q, K) / math.sqrt(d_h)
        attn = _softmax(scores, axis=-1)
        out = np.einsum("bhij,bhjd->bhid", attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, M, d)
        out = out @ self.Wo

        # Mean-pool across modalities
        return out.mean(axis=1)

    @property
    def param_count(self) -> int:
        return self.Wq.size + self.Wk.size + self.Wv.size + self.Wo.size


# ── End-to-end multi-modal model ─────────────────────────────────

@dataclass
class MultiModalConfig:
    """Configuration for multi-modal physics AI."""
    field_dim: int = 64
    point_dim: int = 3
    ts_channels: int = 4
    ts_window: int = 16
    vocab_dim: int = 256
    latent_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 8
    n_layers: int = 3
    n_heads: int = 4


class MultiModalPhysicsAI:
    """End-to-end multi-modal physics AI.

    Accepts any combination of field, point-cloud, time-series,
    and text inputs, fuses them via cross-modal attention, and
    produces predictions.
    """

    def __init__(self, cfg: Optional[MultiModalConfig] = None) -> None:
        self.cfg = cfg or MultiModalConfig()
        c = self.cfg

        self.field_enc = FieldEncoder(EncoderConfig(
            input_dim=c.field_dim, hidden_dim=c.hidden_dim,
            latent_dim=c.latent_dim, n_layers=c.n_layers,
        ))
        self.pc_enc = PointCloudEncoder(EncoderConfig(
            input_dim=c.point_dim, hidden_dim=c.hidden_dim,
            latent_dim=c.latent_dim, n_layers=c.n_layers,
        ))
        self.ts_enc = TimeSeriesEncoder(
            EncoderConfig(
                input_dim=c.ts_channels, hidden_dim=c.hidden_dim,
                latent_dim=c.latent_dim, n_layers=c.n_layers,
            ),
            window_size=c.ts_window,
        )
        self.text_enc = TextEncoder(
            EncoderConfig(
                input_dim=c.vocab_dim, hidden_dim=c.hidden_dim,
                latent_dim=c.latent_dim, n_layers=c.n_layers,
            ),
            vocab_dim=c.vocab_dim,
        )

        self.fusion = MultiModalFusion(c.latent_dim, c.n_heads)

        # Output head
        rng = np.random.default_rng(99)
        s = math.sqrt(2.0 / (c.latent_dim + c.output_dim))
        self.out_W = rng.normal(0, s, (c.latent_dim, c.output_dim)).astype(np.float32)
        self.out_b = np.zeros(c.output_dim, dtype=np.float32)

    def predict(
        self,
        field: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        time_series: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Multi-modal prediction.

        At least one modality must be provided.
        Returns (B, output_dim).
        """
        embeddings: Dict[Modality, np.ndarray] = {}

        if field is not None:
            embeddings[Modality.FIELD] = self.field_enc.encode(field)
        if points is not None:
            embeddings[Modality.POINT_CLOUD] = self.pc_enc.encode(points)
        if time_series is not None:
            embeddings[Modality.TIME_SERIES] = self.ts_enc.encode(time_series)
        if texts is not None:
            embeddings[Modality.TEXT] = self.text_enc.encode(texts)

        if not embeddings:
            raise ValueError("At least one modality must be provided")

        fused = self.fusion.fuse(embeddings)  # (B, latent_dim)
        return fused @ self.out_W + self.out_b

    @property
    def param_count(self) -> int:
        return (
            self.field_enc.net.param_count
            + self.pc_enc.point_net.param_count
            + self.ts_enc.net.param_count
            + self.text_enc.net.param_count
            + self.fusion.param_count
            + self.out_W.size + self.out_b.size
        )


__all__ = [
    "Modality",
    "FieldEncoder",
    "PointCloudEncoder",
    "TimeSeriesEncoder",
    "TextEncoder",
    "MultiModalFusion",
    "MultiModalConfig",
    "MultiModalPhysicsAI",
]
