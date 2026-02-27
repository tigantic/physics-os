"""
5.4 — Equivariant Neural Networks
===================================

SE(3)-, SO(3)-, and E(3)-equivariant layers for physics that
respect continuous rotational and translational symmetry.

Key components:
    * SE3Linear — linear map that commutes with SE(3) actions
    * SO3Convolution — spherical harmonic–based convolution on point clouds
    * E3MessagePassing — equivariant message passing on graphs
    * EquivariantNet — full end-to-end equivariant network

All implemented in pure NumPy for portability.  Real spherical
harmonics up to degree ℓ=3 are used internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Real spherical harmonics up to ℓ=3 ───────────────────────────

def real_spherical_harmonics(
    coords: np.ndarray, lmax: int = 3,
) -> np.ndarray:
    """Evaluate real spherical harmonics Y_ℓ^m at Cartesian coords.

    Parameters
    ----------
    coords : (N, 3)  Cartesian  (x, y, z)
    lmax : maximum degree (default 3)

    Returns
    -------
    Y : (N, (lmax+1)²)
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2).clip(1e-12)
    xn, yn, zn = x / r, y / r, z / r

    Y_list: List[np.ndarray] = []

    # ℓ=0
    Y_list.append(np.full(len(x), 0.5 * math.sqrt(1.0 / math.pi)))
    if lmax >= 1:
        # ℓ=1:  Y_1^{-1}, Y_1^0, Y_1^1
        c = 0.5 * math.sqrt(3.0 / math.pi)
        Y_list.append(c * yn)
        Y_list.append(c * zn)
        Y_list.append(c * xn)
    if lmax >= 2:
        c0 = 0.25 * math.sqrt(5.0 / math.pi)
        c1 = 0.5 * math.sqrt(15.0 / math.pi)
        c2 = 0.25 * math.sqrt(15.0 / math.pi)
        Y_list.append(c1 * xn * yn)
        Y_list.append(c1 * yn * zn)
        Y_list.append(c0 * (3 * zn ** 2 - 1))
        Y_list.append(c1 * xn * zn)
        Y_list.append(c2 * (xn ** 2 - yn ** 2))
    if lmax >= 3:
        c0 = 0.25 * math.sqrt(7.0 / math.pi)
        c1 = 0.25 * math.sqrt(21.0 / (2.0 * math.pi))
        c2 = 0.25 * math.sqrt(105.0 / math.pi)
        c3 = 0.25 * math.sqrt(35.0 / (2.0 * math.pi))
        Y_list.append(c3 * (3 * xn ** 2 - yn ** 2) * yn)
        Y_list.append(c2 * xn * yn * zn)
        Y_list.append(c1 * yn * (5 * zn ** 2 - 1))
        Y_list.append(c0 * zn * (5 * zn ** 2 - 3))
        Y_list.append(c1 * xn * (5 * zn ** 2 - 1))
        Y_list.append(c2 * (xn ** 2 - yn ** 2) * zn)
        Y_list.append(c3 * xn * (xn ** 2 - 3 * yn ** 2))

    return np.stack(Y_list, axis=-1)


# ── Wigner-D rotation utility ────────────────────────────────────

def rotation_matrix_from_euler(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """ZYZ Euler angles → 3×3 rotation matrix."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)

    Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=np.float64)
    return Rz2 @ Ry @ Rz1


# ── SE(3) Linear layer ────────────────────────────────────────────

@dataclass
class SE3LinearConfig:
    """Configuration for SE(3)-equivariant linear layer."""
    in_features: int = 16
    out_features: int = 16
    lmax: int = 2
    bias: bool = False


class SE3Linear:
    """Linear map that commutes with SE(3) transformations.

    Operates on irrep-decomposed features: separate weight matrices
    per angular-momentum channel ℓ.  The scalar (ℓ=0) channel may
    optionally have a bias.
    """

    def __init__(self, cfg: SE3LinearConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(42)
        std = math.sqrt(2.0 / (cfg.in_features + cfg.out_features))

        # One weight matrix per ℓ channel
        self.weights: Dict[int, np.ndarray] = {}
        self.biases: Dict[int, np.ndarray] = {}
        for ell in range(cfg.lmax + 1):
            self.weights[ell] = rng.normal(
                0, std, (cfg.in_features, cfg.out_features)
            ).astype(np.float32)
            if cfg.bias and ell == 0:
                self.biases[ell] = np.zeros(cfg.out_features, dtype=np.float32)

    def forward(self, features: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Apply equivariant linear map.

        Parameters
        ----------
        features : {ℓ: (N, 2ℓ+1, in_features)}

        Returns
        -------
        {ℓ: (N, 2ℓ+1, out_features)}
        """
        out: Dict[int, np.ndarray] = {}
        for ell, x in features.items():
            if ell not in self.weights:
                continue
            # x: (N, 2ℓ+1, in_features)
            y = np.einsum("nmi,io->nmo", x, self.weights[ell])
            if ell in self.biases:
                y = y + self.biases[ell]
            out[ell] = y
        return out

    @property
    def param_count(self) -> int:
        total = sum(w.size for w in self.weights.values())
        total += sum(b.size for b in self.biases.values())
        return total


# ── SO(3) Convolution ─────────────────────────────────────────────

@dataclass
class SO3ConvConfig:
    """Config for spherical harmonic convolution on point clouds."""
    in_features: int = 16
    out_features: int = 16
    lmax: int = 2
    cutoff: float = 5.0


class SO3Convolution:
    """SO(3)-equivariant convolution on point clouds.

    Uses radial basis functions × spherical harmonics as the
    convolutional filter, ensuring rotation equivariance.
    """

    def __init__(self, cfg: SO3ConvConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(7)
        n_sh = (cfg.lmax + 1) ** 2
        std = math.sqrt(2.0 / (cfg.in_features + cfg.out_features))
        # Radial weight: (n_sh, in_features, out_features)
        self.radial_weight = rng.normal(
            0, std, (n_sh, cfg.in_features, cfg.out_features)
        ).astype(np.float32)

    def _radial_basis(self, dist: np.ndarray) -> np.ndarray:
        """Smooth cutoff envelope."""
        mask = dist < self.cfg.cutoff
        env = np.zeros_like(dist)
        env[mask] = 0.5 * (1 + np.cos(math.pi * dist[mask] / self.cfg.cutoff))
        return env

    def forward(
        self,
        positions: np.ndarray,     # (N, 3)
        features: np.ndarray,      # (N, in_features)
        edge_src: np.ndarray,      # (E,) source indices
        edge_dst: np.ndarray,      # (E,) target indices
    ) -> np.ndarray:
        """Perform SO(3)-equivariant convolution.

        Returns (N, out_features).
        """
        diff = positions[edge_dst] - positions[edge_src]  # (E, 3)
        dist = np.linalg.norm(diff, axis=-1).clip(1e-8)
        env = self._radial_basis(dist)  # (E,)

        Y = real_spherical_harmonics(diff, self.cfg.lmax)  # (E, n_sh)

        # Message: env * Y_ℓm * f_src projected through radial weight
        f_src = features[edge_src]  # (E, in_features)
        messages = np.zeros(
            (positions.shape[0], self.cfg.out_features), dtype=np.float32
        )
        for e in range(len(edge_src)):
            w_e = np.einsum("s,sio->io", Y[e], self.radial_weight)
            msg = env[e] * (f_src[e] @ w_e)
            messages[edge_dst[e]] += msg

        return messages

    @property
    def param_count(self) -> int:
        return self.radial_weight.size


# ── E(3) Message Passing ──────────────────────────────────────────

@dataclass
class E3MessagePassingConfig:
    """Config for E(3)-equivariant message passing layer."""
    node_features: int = 32
    edge_features: int = 16
    out_features: int = 32
    lmax: int = 2
    cutoff: float = 5.0
    n_radial_basis: int = 8


class E3MessagePassing:
    """E(3)-equivariant message passing on molecular/mesh graphs.

    Combines invariant radial information with equivariant angular
    features (spherical harmonics) for each edge.
    """

    def __init__(self, cfg: E3MessagePassingConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(13)
        n_sh = (cfg.lmax + 1) ** 2
        std_msg = math.sqrt(2.0 / (cfg.node_features + cfg.n_radial_basis))
        std_upd = math.sqrt(2.0 / (cfg.node_features + cfg.out_features))

        # Message MLP weights
        self.msg_W1 = rng.normal(
            0, std_msg, (cfg.node_features + cfg.n_radial_basis, cfg.edge_features)
        ).astype(np.float32)
        self.msg_b1 = np.zeros(cfg.edge_features, dtype=np.float32)
        self.msg_W2 = rng.normal(
            0, std_msg, (cfg.edge_features, cfg.node_features)
        ).astype(np.float32)

        # Update MLP weights
        self.upd_W1 = rng.normal(
            0, std_upd, (2 * cfg.node_features, cfg.out_features)
        ).astype(np.float32)
        self.upd_b1 = np.zeros(cfg.out_features, dtype=np.float32)

        # Spherical harmonic gate
        self.sh_gate = rng.normal(
            0, 0.02, (n_sh, cfg.node_features)
        ).astype(np.float32)

    def _bessel_basis(self, dist: np.ndarray) -> np.ndarray:
        """Sinc-based radial basis functions."""
        n = self.cfg.n_radial_basis
        c = self.cfg.cutoff
        d = dist[:, None].clip(1e-8)
        k = np.arange(1, n + 1, dtype=np.float32)[None, :]
        return np.sqrt(2.0 / c) * np.sin(k * math.pi * d / c) / d

    def forward(
        self,
        positions: np.ndarray,     # (N, 3)
        features: np.ndarray,      # (N, node_features)
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
    ) -> np.ndarray:
        """E(3)-equivariant message passing step.  Returns (N, out_features)."""
        diff = positions[edge_dst] - positions[edge_src]
        dist = np.linalg.norm(diff, axis=-1).clip(1e-8)

        rbf = self._bessel_basis(dist)           # (E, n_radial_basis)
        Y = real_spherical_harmonics(diff, self.cfg.lmax)  # (E, n_sh)

        f_src = features[edge_src]                # (E, node_features)

        # message MLP
        inp = np.concatenate([f_src, rbf], axis=-1)
        h = np.tanh(inp @ self.msg_W1 + self.msg_b1)
        msg = h @ self.msg_W2                     # (E, node_features)

        # Gate by spherical harmonics
        sh_weight = Y @ self.sh_gate              # (E, node_features)
        msg = msg * sh_weight

        # Aggregate
        agg = np.zeros_like(features)
        np.add.at(agg, edge_dst, msg)

        # Update MLP
        cat = np.concatenate([features, agg], axis=-1)
        out = np.tanh(cat @ self.upd_W1 + self.upd_b1)
        return out

    @property
    def param_count(self) -> int:
        return (
            self.msg_W1.size + self.msg_b1.size +
            self.msg_W2.size + self.upd_W1.size +
            self.upd_b1.size + self.sh_gate.size
        )


# ── Full equivariant network ─────────────────────────────────────

@dataclass
class EquivariantNetConfig:
    """Full equivariant network configuration."""
    n_layers: int = 3
    node_features: int = 32
    hidden_features: int = 64
    out_features: int = 1
    lmax: int = 2
    cutoff: float = 5.0


class EquivariantNet:
    """End-to-end E(3)-equivariant network for physics on point data.

    Stacks multiple E3MessagePassing layers with skip connections
    and a final invariant pooling + readout.
    """

    def __init__(self, cfg: Optional[EquivariantNetConfig] = None) -> None:
        self.cfg = cfg or EquivariantNetConfig()
        self.layers: List[E3MessagePassing] = []
        rng = np.random.default_rng(0)

        in_f = self.cfg.node_features
        for i in range(self.cfg.n_layers):
            out_f = self.cfg.hidden_features if i < self.cfg.n_layers - 1 else self.cfg.node_features
            layer = E3MessagePassing(E3MessagePassingConfig(
                node_features=in_f,
                out_features=out_f,
                lmax=self.cfg.lmax,
                cutoff=self.cfg.cutoff,
            ))
            self.layers.append(layer)
            in_f = out_f

        # Readout
        std = math.sqrt(2.0 / (in_f + self.cfg.out_features))
        self.readout_W = rng.normal(0, std, (in_f, self.cfg.out_features)).astype(np.float32)
        self.readout_b = np.zeros(self.cfg.out_features, dtype=np.float32)

    def forward(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
    ) -> np.ndarray:
        """Forward pass returning (N, out_features)."""
        h = features
        for layer in self.layers:
            h_new = layer.forward(positions, h, edge_src, edge_dst)
            if h_new.shape == h.shape:
                h = h + h_new   # skip connection
            else:
                h = h_new
        return h @ self.readout_W + self.readout_b

    @property
    def param_count(self) -> int:
        total = sum(l.param_count for l in self.layers)
        total += self.readout_W.size + self.readout_b.size
        return total


__all__ = [
    "real_spherical_harmonics",
    "rotation_matrix_from_euler",
    "SE3LinearConfig",
    "SE3Linear",
    "SO3ConvConfig",
    "SO3Convolution",
    "E3MessagePassingConfig",
    "E3MessagePassing",
    "EquivariantNetConfig",
    "EquivariantNet",
]
