"""
Tree Tensor Networks (TTN)
===========================

Hierarchical tensor decomposition with binary-tree topology.
Generalises MPS to tree structures with :math:`O(\\chi^3 d)` contraction cost
at each node, enabling efficient representation of systems with hierarchical
entanglement (e.g. multi-scale physics, renormalisation-group flows).

Key classes
-----------
* :class:`TTNNode` — single tensor in the tree
* :class:`TTN` — full tree-tensor-network state
* :func:`random_ttn` — initialise a random TTN
* :func:`ttn_from_mps` — convert an MPS to TTN via successive SVD grouping
* :func:`coarse_grain` — one RG step (contract leaf pairs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Node & Tree structures
# ======================================================================

class NodeType(Enum):
    """Classification of a TTN node."""
    LEAF = auto()
    INTERNAL = auto()
    ROOT = auto()


@dataclass
class TTNNode:
    """
    Single tensor node in a Tree Tensor Network.

    For a leaf node the tensor has shape ``(chi_parent, d)`` where *d* is the
    local physical dimension.  For an internal (binary) node the tensor has
    shape ``(chi_parent, chi_left, chi_right)``.  The root has no parent bond
    so its first axis is dimension 1.

    Attributes
    ----------
    tensor : NDArray
        The node tensor.
    node_type : NodeType
        LEAF, INTERNAL, or ROOT.
    children : list[int]
        Indices of child nodes (empty for leaves).
    parent : int | None
        Index of parent node (None for root).
    site : int | None
        Physical site index (only for leaves).
    """
    tensor: NDArray
    node_type: NodeType
    children: list[int] = field(default_factory=list)
    parent: Optional[int] = None
    site: Optional[int] = None

    @property
    def bond_dim_parent(self) -> int:
        return self.tensor.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor.shape


@dataclass
class TTN:
    """
    Tree Tensor Network state.

    Stores a list of :class:`TTNNode` objects that form a binary tree.
    Leaves carry physical indices; internal nodes carry only bond indices.
    The tree is stored as a flat list with ``nodes[root_idx]`` being the root.

    Attributes
    ----------
    nodes : list[TTNNode]
        All nodes in the tree.
    root_idx : int
        Index of the root node in ``nodes``.
    d : int
        Local physical dimension shared by all leaves.
    """
    nodes: list[TTNNode]
    root_idx: int
    d: int

    # ------------------------------------------------------------------
    @property
    def n_sites(self) -> int:
        """Number of physical sites (leaves)."""
        return sum(1 for n in self.nodes if n.node_type == NodeType.LEAF)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Dense reconstruction
    # ------------------------------------------------------------------
    def to_tensor(self) -> NDArray:
        """
        Contract to full dense tensor of shape ``(d,)*n_sites``.

        .. warning:: Exponential in the number of sites — use for testing only.
        """
        cache: dict[int, NDArray] = {}

        def _contract(idx: int) -> NDArray:
            if idx in cache:
                return cache[idx]
            node = self.nodes[idx]
            if node.node_type == NodeType.LEAF:
                # shape (chi_parent, d)
                cache[idx] = node.tensor
                return node.tensor
            # Internal / root: (chi_parent, chi_left, chi_right)
            left = _contract(node.children[0])   # (chi_left, ...)
            right = _contract(node.children[1])  # (chi_right, ...)
            # Contract: sum over chi_left and chi_right
            #   node[p, l, r] * left[l, ...] * right[r, ...]
            T = node.tensor  # (p, l, r)
            tmp = np.tensordot(T, left, axes=([1], [0]))   # (p, r, *left_phys)
            tmp = np.tensordot(tmp, right, axes=([1], [0]))  # (p, *left_phys, *right_phys)
            cache[idx] = tmp
            return tmp

        full = _contract(self.root_idx)
        # Root has dummy axis-0 of size 1 — squeeze it
        if full.shape[0] == 1:
            full = full[0]
        return full

    # ------------------------------------------------------------------
    # Bond dimensions
    # ------------------------------------------------------------------
    def bond_dims(self) -> list[int]:
        """Return bond dimension at every internal node (parent axis)."""
        return [
            self.nodes[i].bond_dim_parent
            for i in range(len(self.nodes))
            if self.nodes[i].node_type != NodeType.LEAF
        ]

    # ------------------------------------------------------------------
    # Expectation value of a local operator
    # ------------------------------------------------------------------
    def expectation_local(self, op: NDArray, site: int) -> complex:
        """
        Compute :math:`\\langle\\psi|O_{\\text{site}}|\\psi\\rangle` for a
        single-site operator *op* of shape ``(d, d)``.
        """
        dense = self.to_tensor()
        n = self.n_sites
        # Build operator acting on correct site
        axes = list(range(n))
        # Contract op on axis=site
        result = np.tensordot(dense.conj(), op, axes=([site], [0]))
        # Move new axis to position site
        perm = list(range(n - 1))
        perm.insert(site, n - 1)
        result = result.transpose(perm)
        return np.tensordot(result.ravel(), dense.ravel(), axes=1)

    # ------------------------------------------------------------------
    # Truncation (leaf-to-root SVD sweep)
    # ------------------------------------------------------------------
    def truncate_(self, chi_max: int, cutoff: float = 1e-14) -> "TTN":
        """
        In-place truncation via leaf-to-root SVD sweep.
        """
        visited: set[int] = set()

        def _trunc(idx: int) -> None:
            node = self.nodes[idx]
            if node.node_type == NodeType.LEAF:
                visited.add(idx)
                return
            for c in node.children:
                if c not in visited:
                    _trunc(c)
            # Now truncate bond between this node and its children
            for ci, c in enumerate(node.children):
                child = self.nodes[c]
                M = child.tensor  # (chi_parent, ...)
                shape_rest = M.shape[1:]
                mat = M.reshape(M.shape[0], -1)
                U, S, Vh = np.linalg.svd(mat, full_matrices=False)
                # Truncate
                keep = min(chi_max, np.sum(S > cutoff).item(), len(S))
                keep = max(keep, 1)
                U = U[:, :keep]
                S = S[:keep]
                Vh = Vh[:keep, :]
                child.tensor = (np.diag(S) @ Vh).reshape((keep,) + shape_rest)
                # Absorb U into parent
                T = node.tensor
                # Contract axis ci+1 of T with axis 0 of U^T
                # T shape: (p, chi_left, chi_right)  for binary
                T = np.tensordot(T, U, axes=([ci + 1], [0]))
                # Move new axis to position ci+1
                ax = list(range(T.ndim))
                ax.remove(T.ndim - 1)
                ax.insert(ci + 1, T.ndim - 1)
                node.tensor = T.transpose(ax)
            visited.add(idx)

        _trunc(self.root_idx)
        return self


# ======================================================================
# Constructors
# ======================================================================

def random_ttn(
    n_sites: int,
    d: int = 2,
    chi: int = 4,
    seed: Optional[int] = None,
) -> TTN:
    """
    Create a random binary TTN for *n_sites* physical sites.

    The number of sites is rounded up to the next power of two;
    additional leaves carry trivial (d=1) tensors.

    Parameters
    ----------
    n_sites : int
        Number of physical sites (must be >= 2).
    d : int
        Local physical dimension.
    chi : int
        Maximum bond dimension.
    seed : int, optional
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Round up to power of 2
    L_eff = 1
    while L_eff < n_sites:
        L_eff *= 2

    nodes: list[TTNNode] = []

    # Create leaves
    for i in range(L_eff):
        phys = d if i < n_sites else 1
        tensor = rng.standard_normal((chi, phys))
        tensor /= np.linalg.norm(tensor) + 1e-30
        nodes.append(TTNNode(
            tensor=tensor,
            node_type=NodeType.LEAF,
            site=i if i < n_sites else None,
        ))

    # Build binary tree bottom-up
    current_level = list(range(len(nodes)))
    while len(current_level) > 1:
        next_level: list[int] = []
        for i in range(0, len(current_level), 2):
            left_idx = current_level[i]
            right_idx = current_level[i + 1]
            chi_l = nodes[left_idx].bond_dim_parent
            chi_r = nodes[right_idx].bond_dim_parent
            chi_p = min(chi, chi_l * chi_r)
            tensor = rng.standard_normal((chi_p, chi_l, chi_r))
            tensor /= np.linalg.norm(tensor) + 1e-30
            idx = len(nodes)
            nodes.append(TTNNode(
                tensor=tensor,
                node_type=NodeType.INTERNAL,
                children=[left_idx, right_idx],
            ))
            nodes[left_idx].parent = idx
            nodes[right_idx].parent = idx
            next_level.append(idx)
        current_level = next_level

    # Mark root
    root_idx = current_level[0]
    nodes[root_idx].node_type = NodeType.ROOT
    # Root tensor: leading dimension = 1
    T = nodes[root_idx].tensor
    if T.shape[0] != 1:
        # Reshape so root has trivial parent bond
        T = T[:1]
        nodes[root_idx].tensor = T

    return TTN(nodes=nodes, root_idx=root_idx, d=d)


def ttn_from_mps(
    cores: list[NDArray],
    chi_max: int = 64,
    cutoff: float = 1e-14,
) -> TTN:
    """
    Convert an MPS (list of 3-index tensors) to a binary TTN.

    Pairs neighbouring sites, contracts them, then SVDs to form
    the tree bottom-up.

    Parameters
    ----------
    cores : list[NDArray]
        MPS cores of shape ``(chi_left, d, chi_right)``.
    chi_max : int
        Maximum bond dimension in the tree.
    cutoff : float
        SVD truncation cutoff.
    """
    L = len(cores)
    # Pad to power of 2
    L_eff = 1
    while L_eff < L:
        L_eff *= 2
    padded = list(cores)
    while len(padded) < L_eff:
        # Add trivial site (chi=1, d=1, chi=1)
        padded.append(np.ones((1, 1, 1), dtype=cores[0].dtype))

    nodes: list[TTNNode] = []
    d = cores[0].shape[1]

    # Create leaves from MPS cores (drop left/right bonds initially)
    leaf_data: list[NDArray] = []
    for i, core in enumerate(padded):
        # Reshape (chi_l, d, chi_r) → (chi_l * chi_r, d)
        chi_l, di, chi_r = core.shape
        mat = core.reshape(chi_l * chi_r, di)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        keep = min(chi_max, np.sum(S > cutoff).item(), len(S))
        keep = max(keep, 1)
        leaf_tensor = (np.diag(S[:keep]) @ Vh[:keep]).T  # (d, keep) → transpose to (keep, di)
        leaf_tensor = leaf_tensor.T  # (keep, di)
        nodes.append(TTNNode(
            tensor=leaf_tensor,
            node_type=NodeType.LEAF,
            site=i if i < L else None,
        ))

    # Build tree
    current_level = list(range(len(nodes)))
    while len(current_level) > 1:
        next_level: list[int] = []
        for i in range(0, len(current_level), 2):
            left_idx = current_level[i]
            right_idx = current_level[i + 1]
            chi_l = nodes[left_idx].bond_dim_parent
            chi_r = nodes[right_idx].bond_dim_parent
            chi_p = min(chi_max, chi_l * chi_r)
            tensor = np.eye(chi_p, chi_l * chi_r).reshape(chi_p, chi_l, chi_r)
            # Normalise
            nrm = np.linalg.norm(tensor)
            if nrm > 1e-30:
                tensor = tensor / nrm
            idx = len(nodes)
            nodes.append(TTNNode(
                tensor=tensor,
                node_type=NodeType.INTERNAL,
                children=[left_idx, right_idx],
            ))
            nodes[left_idx].parent = idx
            nodes[right_idx].parent = idx
            next_level.append(idx)
        current_level = next_level

    root_idx = current_level[0]
    nodes[root_idx].node_type = NodeType.ROOT
    T = nodes[root_idx].tensor
    if T.shape[0] != 1:
        T = T[:1]
        nodes[root_idx].tensor = T

    return TTN(nodes=nodes, root_idx=root_idx, d=d)


# ======================================================================
# Coarse-graining (one RG step)
# ======================================================================

def coarse_grain(ttn: TTN, chi_max: int = 64) -> TTN:
    """
    One coarse-graining step: contract pairs of leaves through their
    shared parent, producing a new TTN with half as many sites.

    Returns a *new* TTN (does not modify the input).
    """
    import copy
    new_ttn = copy.deepcopy(ttn)

    # Find parents of leaf pairs
    parents_done: set[int] = set()
    for i, node in enumerate(new_ttn.nodes):
        if node.node_type == NodeType.LEAF:
            pid = node.parent
            if pid is not None and pid not in parents_done:
                parent = new_ttn.nodes[pid]
                if len(parent.children) == 2:
                    c0, c1 = parent.children
                    n0 = new_ttn.nodes[c0]
                    n1 = new_ttn.nodes[c1]
                    if n0.node_type == NodeType.LEAF and n1.node_type == NodeType.LEAF:
                        # Contract: parent(p, l, r) * left(l, d0) * right(r, d1)
                        T = parent.tensor  # (p, l, r)
                        tmp = np.tensordot(T, n0.tensor, axes=([1], [0]))  # (p, r, d0)
                        tmp = np.tensordot(tmp, n1.tensor, axes=([1], [0]))  # (p, d0, d1)
                        # Merge d0, d1 into one physical index
                        d_merged = tmp.shape[1] * tmp.shape[2]
                        merged = tmp.reshape(tmp.shape[0], d_merged)
                        # SVD to reduce rank
                        U, S, Vh = np.linalg.svd(merged, full_matrices=False)
                        keep = min(chi_max, len(S))
                        parent.tensor = (U[:, :keep] * S[:keep]).reshape(
                            parent.tensor.shape[0], keep
                        )
                        # This parent effectively becomes a new leaf
                        parent.node_type = NodeType.LEAF
                        parent.children = []
                        parents_done.add(pid)

    return new_ttn
