"""
Virtual Element Method (VEM)
=============================

Generalisation of FEM to arbitrary polygonal/polyhedral meshes.
The local basis functions are NOT computed explicitly — only their
degrees of freedom and projection operators are needed.

VEM key ingredients:
    1. DOF: nodal values at vertices of each polygon.
    2. Projection Πᵖ: maps the virtual basis to polynomial space.
    3. Consistency + Stability decomposition of the bilinear form:
       a^h(u,v) = a^Π(Πu, Πv) + S^h((I - Π)u, (I - Π)v)

For the Poisson problem (-∇²u = f):
    K_e = Π^T K_consistency Π + α_stab (I - Π)^T (I - Π)

    where α_stab = tr(K_consistency) / n_dof.

Advantages:
    - Handles arbitrary polygons (Voronoi, agglomerated, cut cells)
    - Exact satisfaction of polynomial patch test
    - Natural for adaptive mesh coarsening/refinement

References:
    [1] Beirão da Veiga et al., "Basic Principles of Virtual Element
        Methods", Math. Models Methods Appl. Sci. 23, 2013.
    [2] Beirão da Veiga et al., "The Hitchhiker's Guide to the
        Virtual Element Method", Math. Models Methods Appl. Sci. 24, 2014.
    [3] Ahmad, Alsaedi, Brezzi, Marini & Russo, "Equivalent Projectors
        for Virtual Element Methods",
        Comput. Math. Appl. 66, 2013.

Domain III.1 — Solid Mechanics / Virtual Element Method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Polygon utilities
# ---------------------------------------------------------------------------

def polygon_area(vertices: NDArray) -> float:
    """Signed area of a 2D polygon (shoelace formula)."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return 0.5 * area


def polygon_centroid(vertices: NDArray)  -> NDArray:
    """Centroid of a 2D polygon."""
    n = len(vertices)
    A = polygon_area(vertices)
    if abs(A) < 1e-30:
        return np.mean(vertices, axis=0)
    cx, cy = 0.0, 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = (vertices[i, 0] * vertices[j, 1]
                 - vertices[j, 0] * vertices[i, 1])
        cx += (vertices[i, 0] + vertices[j, 0]) * cross
        cy += (vertices[i, 1] + vertices[j, 1]) * cross
    return np.array([cx, cy], dtype=np.float64) / (6.0 * A)


def polygon_diameter(vertices: NDArray) -> float:
    """Maximum edge length of the polygon."""
    n = len(vertices)
    h = 0.0
    for i in range(n):
        j = (i + 1) % n
        h = max(h, np.linalg.norm(vertices[j] - vertices[i]))
    return h


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class VEMState:
    """VEM solution state — nodal values at polygon vertices."""
    u: NDArray  # (n_nodes,)

    @property
    def n_nodes(self) -> int:
        return self.u.shape[0]


# ---------------------------------------------------------------------------
# 2D Scalar VEM (order 1)
# ---------------------------------------------------------------------------

class VEMSolver:
    r"""
    First-order Virtual Element Method for 2D Poisson equation.

    Solves :math:`-\nabla^2 u = f` on arbitrary polygonal meshes.

    The element stiffness is decomposed as:

    .. math::
        \mathbf{K}_E = \boldsymbol\Pi^T \mathbf{G}^{-1} \boldsymbol\Pi
            + \alpha_s (\mathbf{I} - \mathbf{D}\boldsymbol\Pi)^T
              (\mathbf{I} - \mathbf{D}\boldsymbol\Pi)

    where:
        - **Π** projects DOFs onto the gradient of the linear polynomial
        - **G** is the polynomial Gram matrix
        - **D** is the DOF evaluation operator
        - :math:`\alpha_s` is the stabilisation parameter

    Parameters:
        nodes: Vertex coordinates ``(n_nodes, 2)``.
        elements: List of element connectivity (variable-length polygons).

    Example::

        solver = VEMSolver(nodes, elements)
        state = solver.solve_poisson(lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y))
    """

    def __init__(
        self,
        nodes: NDArray,
        elements: list[NDArray],
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = [np.asarray(e, dtype=np.intp) for e in elements]
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = len(self.elements)

    def _element_stiffness(self, elem_idx: int) -> Tuple[NDArray, NDArray]:
        """
        Compute the VEM element stiffness matrix for a polygon.

        Returns ``(K_e, dofs)`` where dofs are global node indices.
        """
        nids = self.elements[elem_idx]
        n_v = len(nids)
        verts = self.nodes[nids]  # (n_v, 2)

        A_E = abs(polygon_area(verts))
        if A_E < 1e-30:
            return np.zeros((n_v, n_v)), nids

        xc = polygon_centroid(verts)
        h_E = polygon_diameter(verts)

        # Scaled monomial basis for linear VEM: {1, (x-xc)/h, (y-yc)/h}
        # D matrix: evaluation of monomials at vertices  (n_v, 3)
        D = np.zeros((n_v, 3), dtype=np.float64)
        D[:, 0] = 1.0
        D[:, 1] = (verts[:, 0] - xc[0]) / h_E
        D[:, 2] = (verts[:, 1] - xc[1]) / h_E

        # B matrix: (3, n_v) — from integration by parts
        # B[0,:] = 1/n_v  (constant)
        # B[1,i] and B[2,i] from boundary integrals of ∇m_α · n
        B = np.zeros((3, n_v), dtype=np.float64)
        B[0, :] = 1.0 / n_v

        for i in range(n_v):
            ip1 = (i + 1) % n_v
            im1 = (i - 1 + n_v) % n_v
            # Outward normals of edges meeting at vertex i
            # Edge i-1 → i: normal = (y_i - y_{i-1}, -(x_i - x_{i-1}))
            # Edge i → i+1: normal = (y_{i+1} - y_i, -(x_{i+1} - x_i))
            # Average contribution
            n_prev = np.array([
                verts[i, 1] - verts[im1, 1],
                -(verts[i, 0] - verts[im1, 0]),
            ])
            n_next = np.array([
                verts[ip1, 1] - verts[i, 1],
                -(verts[ip1, 0] - verts[i, 0]),
            ])
            # B[1,i] = (1/2A) ∫_∂E N_i (n_x/h) ds
            # Using trapezoidal: midpoint of edges × edge normal
            B[1, i] = 0.5 * (n_prev[0] + n_next[0]) / (2.0 * A_E) * h_E
            B[2, i] = 0.5 * (n_prev[1] + n_next[1]) / (2.0 * A_E) * h_E

        # More standard derivation for the first-order VEM:
        # G = B D  (should be I for consistency)
        # But let's use the direct approach.
        # Projection: Πk = G^{-1} B  with G = B @ D
        G = B @ D  # (3, 3)
        G_inv = np.linalg.inv(G + 1e-14 * np.eye(3))
        Pi_star = G_inv @ B  # (3, n_v) — projection coefficients
        Pi = D @ Pi_star     # (n_v, n_v) — projector in DOF space

        # Consistency part: K_c = Π*^T G̃ Π*
        # G̃ = ∫_E ∇m_α · ∇m_β dΩ = A_E / h_E² [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
        G_tilde = (A_E / h_E ** 2) * np.array([
            [0, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float64)
        K_c = Pi_star.T @ G_tilde @ Pi_star  # (n_v, n_v)

        # Stability part: S = α (I - Pi)^T (I - Pi)
        I_minus_Pi = np.eye(n_v) - Pi
        alpha_stab = max(np.trace(K_c), 1e-10) / n_v
        K_s = alpha_stab * I_minus_Pi.T @ I_minus_Pi

        K_e = K_c + K_s
        return K_e, nids

    def assemble(self) -> NDArray:
        """Assemble global stiffness matrix."""
        K = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        for e in range(self.n_elements):
            K_e, nids = self._element_stiffness(e)
            for i, gi in enumerate(nids):
                for j, gj in enumerate(nids):
                    K[gi, gj] += K_e[i, j]
        return K

    def assemble_load(self, f_func: Callable) -> NDArray:
        """Assemble load vector using polygon centroid quadrature."""
        F = np.zeros(self.n_nodes, dtype=np.float64)
        for e in range(self.n_elements):
            nids = self.elements[e]
            verts = self.nodes[nids]
            n_v = len(nids)
            A_E = abs(polygon_area(verts))
            xc = polygon_centroid(verts)
            f_c = f_func(xc[0], xc[1])
            # Distribute equally to vertices (linear VEM)
            for i in range(n_v):
                F[nids[i]] += f_c * A_E / n_v
        return F

    def solve_poisson(
        self,
        f_func: Callable,
        bc_nodes: Optional[list[int]] = None,
        bc_values: Optional[NDArray] = None,
    ) -> VEMState:
        """Solve -∇²u = f with Dirichlet BCs."""
        K = self.assemble()
        F = self.assemble_load(f_func)

        if bc_nodes is not None and bc_values is not None:
            for i, nid in enumerate(bc_nodes):
                K[nid, :] = 0.0
                K[nid, nid] = 1.0
                F[nid] = bc_values[i]
        elif bc_nodes is None:
            # Default: all boundary nodes = 0
            boundary = self._boundary_nodes()
            for nid in boundary:
                K[nid, :] = 0.0
                K[nid, nid] = 1.0
                F[nid] = 0.0

        u = np.linalg.solve(K, F)
        return VEMState(u=u)

    def _boundary_nodes(self) -> set[int]:
        """Identify boundary nodes (appear in only one element face)."""
        edge_count: dict[Tuple[int, int], int] = {}
        for e in range(self.n_elements):
            nids = self.elements[e]
            n_v = len(nids)
            for i in range(n_v):
                j = (i + 1) % n_v
                edge = (min(nids[i], nids[j]), max(nids[i], nids[j]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        boundary = set()
        for (a, b), count in edge_count.items():
            if count == 1:
                boundary.add(a)
                boundary.add(b)
        return boundary

    def l2_error(
        self,
        state: VEMState,
        exact: Callable,
    ) -> float:
        """Approximate L² error using centroid evaluation."""
        err2 = 0.0
        for e in range(self.n_elements):
            nids = self.elements[e]
            verts = self.nodes[nids]
            A_E = abs(polygon_area(verts))
            xc = polygon_centroid(verts)
            # Approximate u_h at centroid from DOF average
            u_h = np.mean(state.u[nids])
            u_ex = exact(xc[0], xc[1])
            err2 += (u_h - u_ex) ** 2 * A_E
        return float(np.sqrt(err2))
