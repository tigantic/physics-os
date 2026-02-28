"""
5.16 — Self-Supervised Pre-Training for Physics
=================================================

Contrastive and masked-autoencoding pre-training strategies for
learning transferable physics representations without labels.

Components:
    * MaskedFieldAutoencoder — MAE that reconstructs masked patches
    * ContrastiveLearner — SimCLR-style contrastive on augmented fields
    * PhysicsAugmentation — domain-aware augmentation transforms
    * PreTrainingPipeline — orchestrates pre-training workflow
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Augmentations ─────────────────────────────────────────────────

class PhysicsAugmentation:
    """Domain-aware augmentation transforms for physics fields.

    These preserve physical invariance while providing diversity for
    contrastive learning.
    """

    @staticmethod
    def add_noise(field: np.ndarray, sigma: float = 0.01,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Additive Gaussian noise."""
        rng = rng or np.random.default_rng()
        return field + sigma * rng.standard_normal(field.shape).astype(field.dtype)

    @staticmethod
    def random_crop(field: np.ndarray, crop_ratio: float = 0.8,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Random spatial crop (1-D fields)."""
        rng = rng or np.random.default_rng()
        N = field.shape[0]
        crop_size = max(1, int(N * crop_ratio))
        start = int(rng.integers(0, max(N - crop_size, 1)))
        cropped = field[start:start + crop_size]
        # Resize back to original length via linear interpolation
        indices = np.linspace(0, len(cropped) - 1, N)
        if field.ndim == 1:
            return np.interp(indices, np.arange(len(cropped)), cropped).astype(field.dtype)
        # Multi-channel
        result = np.zeros_like(field)
        for c in range(field.shape[-1]):
            result[:, c] = np.interp(indices, np.arange(len(cropped)), cropped[:, c])
        return result.astype(field.dtype)

    @staticmethod
    def galilean_shift(field: np.ndarray, shift: int = 1) -> np.ndarray:
        """Galilean invariance: circular spatial shift."""
        return np.roll(field, shift, axis=0)

    @staticmethod
    def scale_intensity(field: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """Scale field values (physical scaling symmetry)."""
        return field * factor

    @staticmethod
    def flip_spatial(field: np.ndarray) -> np.ndarray:
        """Spatial reflection (parity symmetry)."""
        return field[::-1].copy()


# ── Encoder backbone ─────────────────────────────────────────────

class EncoderMLP:
    """Patch encoder for pre-training."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 n_layers: int = 3, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        layers = [input_dim] + [hidden_dim] * n_layers + [latent_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            s = math.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.weights.append(rng.normal(0, s, (layers[i], layers[i + 1])).astype(np.float32))
            self.biases.append(np.zeros(layers[i + 1], dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(len(self.weights) - 1):
            x = np.tanh(x @ self.weights[i] + self.biases[i])
        return x @ self.weights[-1] + self.biases[-1]

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


class DecoderMLP:
    """Patch decoder for MAE reconstruction."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 3, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        layers = [latent_dim] + [hidden_dim] * n_layers + [output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            s = math.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.weights.append(rng.normal(0, s, (layers[i], layers[i + 1])).astype(np.float32))
            self.biases.append(np.zeros(layers[i + 1], dtype=np.float32))

    def forward(self, z: np.ndarray) -> np.ndarray:
        for i in range(len(self.weights) - 1):
            z = np.tanh(z @ self.weights[i] + self.biases[i])
        return z @ self.weights[-1] + self.biases[-1]

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases


# ── Masked Field Autoencoder ─────────────────────────────────────

@dataclass
class MAEConfig:
    """Configuration for Masked Autoencoder."""
    patch_dim: int = 16         # dimension per patch
    n_patches: int = 32         # patches per field
    mask_ratio: float = 0.75    # fraction of patches to mask
    latent_dim: int = 64
    hidden_dim: int = 128
    n_encoder_layers: int = 3
    n_decoder_layers: int = 2


class MaskedFieldAutoencoder:
    """Masked Autoencoder for physics field pre-training.

    Randomly masks a fraction of spatial patches and trains the
    network to reconstruct them from the visible patches.
    """

    def __init__(self, cfg: Optional[MAEConfig] = None) -> None:
        self.cfg = cfg or MAEConfig()
        self.encoder = EncoderMLP(
            self.cfg.patch_dim, self.cfg.hidden_dim, self.cfg.latent_dim,
            self.cfg.n_encoder_layers, seed=0,
        )
        self.decoder = DecoderMLP(
            self.cfg.latent_dim, self.cfg.hidden_dim, self.cfg.patch_dim,
            self.cfg.n_decoder_layers, seed=1,
        )
        # Mask token
        self.mask_token = np.zeros(self.cfg.latent_dim, dtype=np.float32)

    def mask_patches(
        self, patches: np.ndarray, rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly mask patches.

        Parameters
        ----------
        patches : (B, n_patches, patch_dim)

        Returns
        -------
        visible_patches : (B, n_visible, patch_dim)
        mask_indices : (B, n_masked)
        visible_indices: (B, n_visible)
        """
        rng = rng or np.random.default_rng()
        B, P, D = patches.shape
        n_mask = int(P * self.cfg.mask_ratio)
        n_vis = P - n_mask

        all_mask = np.zeros((B, n_mask), dtype=np.int64)
        all_vis = np.zeros((B, n_vis), dtype=np.int64)
        vis_patches = np.zeros((B, n_vis, D), dtype=np.float32)

        for b in range(B):
            perm = rng.permutation(P)
            mask_idx = np.sort(perm[:n_mask])
            vis_idx = np.sort(perm[n_mask:])
            all_mask[b] = mask_idx
            all_vis[b] = vis_idx
            vis_patches[b] = patches[b, vis_idx]

        return vis_patches, all_mask, all_vis

    def forward(
        self, patches: np.ndarray, rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Full MAE forward: mask → encode → decode → reconstruct.

        Returns dict with reconstruction loss and details.
        """
        rng = rng or np.random.default_rng()
        B, P, D = patches.shape
        vis_patches, mask_idx, vis_idx = self.mask_patches(patches, rng)

        # Encode visible patches
        B2, V, _ = vis_patches.shape
        vis_flat = vis_patches.reshape(B2 * V, D)
        latents = self.encoder.forward(vis_flat).reshape(B2, V, -1)

        # Decode all patches (visible = encoded, masked = mask_token)
        n_mask = mask_idx.shape[1]
        all_latents = np.tile(
            self.mask_token, (B, P, 1)
        )
        for b in range(B):
            all_latents[b, vis_idx[b]] = latents[b]

        # Reconstruct
        all_flat = all_latents.reshape(B * P, -1)
        recon = self.decoder.forward(all_flat).reshape(B, P, D)

        # Loss on masked patches only
        total_loss = 0.0
        for b in range(B):
            pred_masked = recon[b, mask_idx[b]]
            true_masked = patches[b, mask_idx[b]]
            total_loss += float(np.mean((pred_masked - true_masked) ** 2))
        total_loss /= B

        return {
            "loss": total_loss,
            "reconstruction": recon,
            "n_masked": n_mask,
            "n_visible": P - n_mask,
        }

    @property
    def param_count(self) -> int:
        enc = sum(w.size for w in self.encoder.weights) + sum(b.size for b in self.encoder.biases)
        dec = sum(w.size for w in self.decoder.weights) + sum(b.size for b in self.decoder.biases)
        return enc + dec + self.mask_token.size


# ── Contrastive Learner ──────────────────────────────────────────

@dataclass
class ContrastiveConfig:
    """Configuration for contrastive pre-training."""
    patch_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    projection_dim: int = 32
    temperature: float = 0.1
    n_encoder_layers: int = 3


class ContrastiveLearner:
    """SimCLR-style contrastive learning for physics fields.

    Two augmented views of each field are encoded, and the model
    learns to bring them together in embedding space while pushing
    apart embeddings from different fields.
    """

    def __init__(self, cfg: Optional[ContrastiveConfig] = None) -> None:
        self.cfg = cfg or ContrastiveConfig()
        self.encoder = EncoderMLP(
            self.cfg.patch_dim, self.cfg.hidden_dim, self.cfg.latent_dim,
            self.cfg.n_encoder_layers, seed=10,
        )
        # Projection head
        rng = np.random.default_rng(20)
        s = math.sqrt(2.0 / (self.cfg.latent_dim + self.cfg.projection_dim))
        self.proj_w = rng.normal(
            0, s, (self.cfg.latent_dim, self.cfg.projection_dim)
        ).astype(np.float32)
        self.proj_b = np.zeros(self.cfg.projection_dim, dtype=np.float32)

        self.augmentations = PhysicsAugmentation()

    def project(self, z: np.ndarray) -> np.ndarray:
        """L2-normalised projection."""
        h = z @ self.proj_w + self.proj_b
        norm = np.linalg.norm(h, axis=-1, keepdims=True).clip(1e-8)
        return h / norm

    def nt_xent_loss(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Normalised temperature-scaled cross-entropy loss."""
        B = z1.shape[0]
        p1 = self.project(z1)
        p2 = self.project(z2)

        # Similarity matrix
        z = np.concatenate([p1, p2], axis=0)  # (2B, proj_dim)
        sim = z @ z.T / self.cfg.temperature   # (2B, 2B)

        # Mask out self-similarity
        mask = np.eye(2 * B, dtype=bool)
        sim[mask] = -1e9

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = np.concatenate([
            np.arange(B, 2 * B),
            np.arange(0, B),
        ])

        # Softmax cross-entropy
        exp_sim = np.exp(sim - sim.max(axis=1, keepdims=True))
        probs = exp_sim / exp_sim.sum(axis=1, keepdims=True)

        loss = 0.0
        for i in range(2 * B):
            loss -= math.log(max(probs[i, labels[i]], 1e-10))
        return loss / (2 * B)

    def training_step(
        self, fields: np.ndarray, rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """One contrastive training step.

        Parameters
        ----------
        fields : (B, patch_dim)

        Returns
        -------
        {"loss": float}
        """
        rng = rng or np.random.default_rng()
        # Create two augmented views
        view1 = self.augmentations.add_noise(fields, sigma=0.02, rng=rng)
        view2 = self.augmentations.scale_intensity(
            self.augmentations.add_noise(fields, sigma=0.01, rng=rng),
            factor=1.0 + 0.1 * rng.standard_normal(),
        )

        z1 = self.encoder.forward(view1)
        z2 = self.encoder.forward(view2)
        loss = self.nt_xent_loss(z1, z2)
        return {"loss": loss}


# ── Pre-Training Pipeline ────────────────────────────────────────

class PreTrainingPipeline:
    """Orchestrates self-supervised pre-training."""

    def __init__(
        self,
        mae: Optional[MaskedFieldAutoencoder] = None,
        contrastive: Optional[ContrastiveLearner] = None,
    ) -> None:
        self.mae = mae or MaskedFieldAutoencoder()
        self.contrastive = contrastive or ContrastiveLearner()

    def pretrain_mae(
        self,
        data: np.ndarray,      # (N, n_patches, patch_dim)
        n_epochs: int = 10,
        batch_size: int = 32,
    ) -> List[float]:
        """Run MAE pre-training epochs."""
        losses: List[float] = []
        rng = np.random.default_rng(0)
        N = data.shape[0]

        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                batch = data[perm[start:start + batch_size]]
                if len(batch) == 0:
                    continue
                result = self.mae.forward(batch, rng)
                epoch_loss += result["loss"]
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        return losses

    def pretrain_contrastive(
        self,
        data: np.ndarray,      # (N, patch_dim)
        n_epochs: int = 10,
        batch_size: int = 32,
    ) -> List[float]:
        """Run contrastive pre-training epochs."""
        losses: List[float] = []
        rng = np.random.default_rng(1)
        N = data.shape[0]

        for epoch in range(n_epochs):
            perm = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                batch = data[perm[start:start + batch_size]]
                if len(batch) < 2:
                    continue
                result = self.contrastive.training_step(batch, rng)
                epoch_loss += result["loss"]
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        return losses


__all__ = [
    "PhysicsAugmentation",
    "MAEConfig",
    "MaskedFieldAutoencoder",
    "ContrastiveConfig",
    "ContrastiveLearner",
    "PreTrainingPipeline",
]
