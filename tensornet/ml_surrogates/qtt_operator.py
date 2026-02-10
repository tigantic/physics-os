"""
5.15 — Operator Learning on QTT Representations
=================================================

Maps QTT (Quantics Tensor Train) compressed fields to QTT fields
directly — never decompressing to full grids.

Components:
    * QTTOperatorLayer — learned contraction between QTT cores
    * QTTOperatorNet — stacked operator for multi-step prediction
    * qtt_inner_product — inner product in QTT space
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── QTT core representation ──────────────────────────────────────

@dataclass
class QTTField:
    """A field stored in QTT (Quantics Tensor Train) format.

    cores[k] has shape (r_{k-1}, n_k, r_k) where n_k is the local
    mode dimension (typically 2 for binary quantics) and r_k are
    bond dimensions.
    """
    cores: List[np.ndarray]

    @property
    def n_sites(self) -> int:
        return len(self.cores)

    @property
    def bond_dims(self) -> List[int]:
        return [c.shape[2] for c in self.cores[:-1]]

    @property
    def total_params(self) -> int:
        return sum(c.size for c in self.cores)

    def to_full(self) -> np.ndarray:
        """Contract to full tensor (for small test cases only)."""
        result = self.cores[0]  # (1, n0, r0)
        for c in self.cores[1:]:
            # (r_{prev}, n_{prev}, r_mid) × (r_mid, n_k, r_k)
            result = np.einsum("...i,ijk->...jk", result, c)
        return result.squeeze()


def random_qtt(
    n_sites: int = 10,
    mode_dim: int = 2,
    bond_dim: int = 4,
    seed: int = 0,
) -> QTTField:
    """Create a random QTT field for testing."""
    rng = np.random.default_rng(seed)
    cores: List[np.ndarray] = []
    for k in range(n_sites):
        r_left = 1 if k == 0 else bond_dim
        r_right = 1 if k == n_sites - 1 else bond_dim
        cores.append(
            rng.normal(0, 0.1, (r_left, mode_dim, r_right)).astype(np.float32)
        )
    return QTTField(cores=cores)


def qtt_inner_product(a: QTTField, b: QTTField) -> float:
    """Compute ⟨a|b⟩ in QTT space without full decompression."""
    if a.n_sites != b.n_sites:
        raise ValueError("QTT fields must have same number of sites")

    # Contract site-by-site
    # E_{k} shape: (r_a_k, r_b_k)
    E = np.einsum("inj,ink->jk", a.cores[0], b.cores[0])
    for k in range(1, a.n_sites):
        # E: (r_a_{k-1}, r_b_{k-1}), cores: (r_left, mode, r_right)
        # Contract left bonds (a,b) and mode index (n); propagate right bonds (c,d)
        M = np.einsum("ab,anc,bnd->cd", E, a.cores[k], b.cores[k])
        E = M
    return float(E.squeeze())


def qtt_norm(a: QTTField) -> float:
    """||a||₂ in QTT space."""
    return math.sqrt(max(qtt_inner_product(a, a), 0.0))


# ── Operator layer ────────────────────────────────────────────────

@dataclass
class QTTOperatorConfig:
    """Configuration for a single QTT operator layer."""
    n_sites: int = 10
    mode_dim: int = 2
    bond_dim_in: int = 4
    bond_dim_out: int = 4
    bond_dim_op: int = 4


class QTTOperatorLayer:
    """Learned operator that maps QTT → QTT via site-local transformations.

    For each site k, a 4-index tensor
        W_k[r_op_left, n_in, n_out, r_op_right]
    acts on the input core via contraction over n_in, producing
    a new core in the output QTT.
    """

    def __init__(self, cfg: QTTOperatorConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(0)

        self.op_cores: List[np.ndarray] = []
        for k in range(cfg.n_sites):
            rl = 1 if k == 0 else cfg.bond_dim_op
            rr = 1 if k == cfg.n_sites - 1 else cfg.bond_dim_op
            std = math.sqrt(2.0 / (cfg.mode_dim * (rl + rr)))
            W = rng.normal(0, std, (rl, cfg.mode_dim, cfg.mode_dim, rr)).astype(np.float32)
            # Initialise near identity
            for m in range(min(cfg.mode_dim, cfg.mode_dim)):
                W[:, m, m, :] += np.eye(rl, rr, dtype=np.float32) * 0.5
            self.op_cores.append(W)

    def forward(self, qtt_in: QTTField) -> QTTField:
        """Apply operator to QTT field.

        Output bond dimension = bond_dim_in × bond_dim_op (Kronecker).
        """
        out_cores: List[np.ndarray] = []
        for k in range(self.cfg.n_sites):
            # input core: (r_in_L, n_in, r_in_R)
            C = qtt_in.cores[k]
            # operator core: (r_op_L, n_in, n_out, r_op_R)
            W = self.op_cores[k]

            # Contract over n_in → (r_in_L, r_op_L, n_out, r_in_R, r_op_R)
            result = np.einsum("inj,knml->iknlj", C, W)
            # Reshape Kronecker: (r_in_L*r_op_L, n_out, r_in_R*r_op_R)
            ri_l, ro_l, n_out, ri_r, ro_r = result.shape
            out_core = result.reshape(ri_l * ro_l, n_out, ri_r * ro_r)
            out_cores.append(out_core)

        return QTTField(cores=out_cores)

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.op_cores)


# ── Stacked operator network ─────────────────────────────────────

@dataclass
class QTTOperatorNetConfig:
    """Config for multi-layer QTT operator network."""
    n_sites: int = 10
    mode_dim: int = 2
    bond_dim: int = 4
    n_layers: int = 3
    truncation_threshold: float = 1e-6


class QTTOperatorNet:
    """Stacked QTT operator layers with bond-dimension truncation.

    After each operator application, the output bond dimension
    grows multiplicatively.  We truncate via SVD to keep it bounded.
    """

    def __init__(self, cfg: Optional[QTTOperatorNetConfig] = None) -> None:
        self.cfg = cfg or QTTOperatorNetConfig()
        self.layers: List[QTTOperatorLayer] = []
        for i in range(self.cfg.n_layers):
            self.layers.append(QTTOperatorLayer(QTTOperatorConfig(
                n_sites=self.cfg.n_sites,
                mode_dim=self.cfg.mode_dim,
                bond_dim_in=self.cfg.bond_dim,
                bond_dim_out=self.cfg.bond_dim,
                bond_dim_op=self.cfg.bond_dim,
            )))

    @staticmethod
    def truncate(qtt: QTTField, max_bond: int, eps: float = 1e-6) -> QTTField:
        """Truncate QTT bond dimensions via left-to-right SVD sweep."""
        cores = [c.copy() for c in qtt.cores]
        L = len(cores)

        for k in range(L - 1):
            r_l, n_k, r_r = cores[k].shape
            mat = cores[k].reshape(r_l * n_k, r_r)
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)

            # Truncate
            mask = S > eps
            keep = min(int(mask.sum()), max_bond)
            keep = max(keep, 1)

            U = U[:, :keep]
            S = S[:keep]
            Vt = Vt[:keep, :]

            cores[k] = U.reshape(r_l, n_k, keep)
            SV = np.diag(S) @ Vt
            # Absorb into next core
            cores[k + 1] = np.einsum("ij,jnk->ink", SV, cores[k + 1])

        return QTTField(cores=cores)

    def forward(self, qtt_in: QTTField) -> QTTField:
        """Apply stacked operators with intermediate truncation."""
        x = qtt_in
        for layer in self.layers:
            x = layer.forward(x)
            x = self.truncate(x, self.cfg.bond_dim, self.cfg.truncation_threshold)
        return x

    @property
    def param_count(self) -> int:
        return sum(l.param_count for l in self.layers)


__all__ = [
    "QTTField",
    "random_qtt",
    "qtt_inner_product",
    "qtt_norm",
    "QTTOperatorConfig",
    "QTTOperatorLayer",
    "QTTOperatorNetConfig",
    "QTTOperatorNet",
]
