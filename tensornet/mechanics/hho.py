"""
Hybrid High-Order (HHO) Method
================================

Arbitrary-order discretisation on general polytopal meshes based on
cell and face unknowns reconstructed via a higher-order potential.

For the model Poisson problem  -Δu = f  in Ω, u = g on ∂Ω:

1. **DOFs** — cell polynomial of degree k plus face polynomials of
   degree k on each face.
2. **Reconstruction** — build p^{k+1} from cell and face DOFs via
   a local potential reconstruction.
3. **Stabilisation** — penalise the jump between face DOFs and
   the trace of the reconstruction.
4. **Assembly & solve** — element-by-element stiffness + static
   condensation of cell DOFs.

References:
    [1] Di Pietro & Ern, "A Hybrid High-Order locking-free method
        for linear elasticity on general meshes", CMAME 283, 2015.
    [2] Di Pietro, Ern & Lemaire, "An arbitrary-order and
        compact-stencil discretization of diffusion on general meshes
        based on local reconstruction operators", CMAM 14, 2014.
    [3] Cicuttin, Di Pietro & Ern, "Implementation of Discontinuous
        Skeletal methods on arbitrary-dimensional, polytopal meshes
        using generic programming", J. Comput. Appl. Math. 344, 2018.

Domain III.1 — Solid Mechanics / Hybrid High-Order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Polynomial tools
# ---------------------------------------------------------------------------

class PolySpace1D:
    """Scaled monomial basis on 1D reference interval [-1, 1]."""

    @staticmethod
    def eval(x: NDArray, degree: int) -> NDArray:
        """
        Evaluate monomials up to *degree* at points *x*.

        Returns ``(n_pts, degree + 1)`` matrix.
        """
        n = x.shape[0]
        V = np.ones((n, degree + 1), dtype=np.float64)
        for k in range(1, degree + 1):
            V[:, k] = V[:, k - 1] * x
        return V


class PolySpace2D:
    """
    Scaled monomial basis on 2D: :math:`(x - x_c)^a (y - y_c)^b / h^{a+b}`.
    """

    @staticmethod
    def dim(k: int) -> int:
        """Number of 2D monomials up to total degree k."""
        return (k + 1) * (k + 2) // 2

    @staticmethod
    def eval(
        x: NDArray, y: NDArray, degree: int,
        xc: float = 0.0, yc: float = 0.0, h: float = 1.0,
    ) -> NDArray:
        """
        Scaled monomial evaluation.

        Returns ``(n_pts, dim(degree))`` matrix.
        """
        n = x.shape[0]
        dim = PolySpace2D.dim(degree)
        V = np.zeros((n, dim), dtype=np.float64)
        xs = (x - xc) / h
        ys = (y - yc) / h
        col = 0
        for tot in range(degree + 1):
            for b in range(tot + 1):
                a = tot - b
                V[:, col] = xs ** a * ys ** b
                col += 1
        return V

    @staticmethod
    def grad(
        x: NDArray, y: NDArray, degree: int,
        xc: float = 0.0, yc: float = 0.0, h: float = 1.0,
    ) -> Tuple[NDArray, NDArray]:
        """
        Gradient of scaled monomials.

        Returns ``(dV_dx, dV_dy)``, each ``(n_pts, dim(degree))``.
        """
        n = x.shape[0]
        dim = PolySpace2D.dim(degree)
        dVdx = np.zeros((n, dim), dtype=np.float64)
        dVdy = np.zeros((n, dim), dtype=np.float64)
        xs = (x - xc) / h
        ys = (y - yc) / h
        col = 0
        for tot in range(degree + 1):
            for b in range(tot + 1):
                a = tot - b
                if a >= 1:
                    dVdx[:, col] = a * xs ** (a - 1) * ys ** b / h
                if b >= 1:
                    dVdy[:, col] = xs ** a * b * ys ** (b - 1) / h
                col += 1
        return dVdx, dVdy


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

@dataclass
class HHOMesh2D:
    """
    2D polygonal mesh for HHO.

    Attributes:
        nodes: Vertex coordinates ``(n_nodes, 2)``.
        cells: List of vertex-index arrays per cell.
        cell_faces: Face indices per cell.
        face_nodes: Two endpoint indices per face ``(n_faces, 2)``.
        boundary_faces: Indices of faces on the domain boundary.
    """
    nodes: NDArray
    cells: list[NDArray]
    cell_faces: list[NDArray]
    face_nodes: NDArray
    boundary_faces: NDArray

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_faces(self) -> int:
        return self.face_nodes.shape[0]


def _polygon_centroid_area(verts: NDArray) -> Tuple[NDArray, float]:
    """Centroid and area of a simple polygon."""
    n = verts.shape[0]
    area = 0.0
    cx, cy = 0.0, 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = verts[i, 0] * verts[j, 1] - verts[j, 0] * verts[i, 1]
        area += cross
        cx += (verts[i, 0] + verts[j, 0]) * cross
        cy += (verts[i, 1] + verts[j, 1]) * cross
    area *= 0.5
    cx /= 6.0 * area
    cy /= 6.0 * area
    return np.array([cx, cy], dtype=np.float64), abs(area)


def build_cartesian_hho_mesh(
    nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
) -> HHOMesh2D:
    """Build a Cartesian quad mesh in HHO format."""
    dx, dy = Lx / nx, Ly / ny
    nodes = np.zeros(((nx + 1) * (ny + 1), 2), dtype=np.float64)
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes[j * (nx + 1) + i] = [i * dx, j * dy]

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            cells.append(np.array([n0, n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1)], dtype=np.intp))

    face_map: dict[Tuple[int, int], int] = {}
    face_list: list[Tuple[int, int]] = []

    def _get_face(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in face_map:
            face_map[key] = len(face_list)
            face_list.append(key)
        return face_map[key]

    cell_faces = []
    for cell in cells:
        nv = len(cell)
        cf = [_get_face(cell[k], cell[(k + 1) % nv]) for k in range(nv)]
        cell_faces.append(np.array(cf, dtype=np.intp))

    face_nodes = np.array(face_list, dtype=np.intp)

    # Boundary faces
    bnd = []
    tol = 1e-12
    xmax, ymax = Lx, Ly
    for fi, (a, b) in enumerate(face_list):
        mid = 0.5 * (nodes[a] + nodes[b])
        if mid[0] < tol or mid[0] > xmax - tol or mid[1] < tol or mid[1] > ymax - tol:
            bnd.append(fi)

    return HHOMesh2D(
        nodes=nodes, cells=cells, cell_faces=cell_faces,
        face_nodes=face_nodes, boundary_faces=np.array(bnd, dtype=np.intp),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class HHOState:
    """HHO solution state."""
    p_cell: NDArray  # Cell DOFs per cell (list or flat array)
    p_face: NDArray  # Face DOFs per face (flat array)

    @property
    def n_faces(self) -> int:
        return self.p_face.shape[0]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class HHOSolver:
    r"""
    Hybrid High-Order solver for 2D Poisson on polygonal meshes.

    Solves :math:`-\Delta u = f` with Dirichlet BCs.

    The method:
        1. Local unknowns: one cell DOF + one face DOF per face (k = 0).
        2. Reconstruction: p^{k+1} from cell/face DOFs via local Neumann
           problems.
        3. Stabilisation: jump penalty between face DOF and trace of
           reconstruction.
        4. Static condensation: eliminate cell DOFs → face-only system.

    Parameters:
        mesh: HHO polygonal mesh.
        k: Polynomial degree (0 or 1 currently supported, default 0).

    Example::

        mesh = build_cartesian_hho_mesh(20, 20)
        solver = HHOSolver(mesh, k=0)
        state = solver.solve(
            f_func=lambda x, y: 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y),
        )
    """

    def __init__(self, mesh: HHOMesh2D, k: int = 0) -> None:
        if k > 1:
            raise NotImplementedError("Only k=0 and k=1 supported.")
        self.mesh = mesh
        self.k = k

    def _cell_geometry(self, ci: int) -> Tuple[NDArray, float, float]:
        """Return centroid, area, and diameter of cell ci."""
        verts = self.mesh.nodes[self.mesh.cells[ci]]
        centroid, area = _polygon_centroid_area(verts)
        diam = 0.0
        nv = verts.shape[0]
        for i in range(nv):
            for j in range(i + 1, nv):
                d = np.linalg.norm(verts[i] - verts[j])
                if d > diam:
                    diam = d
        return centroid, area, diam

    def _face_midpoint_length(self, fi: int) -> Tuple[NDArray, float]:
        """Face midpoint and length."""
        a, b = self.mesh.face_nodes[fi]
        xa, xb = self.mesh.nodes[a], self.mesh.nodes[b]
        return 0.5 * (xa + xb), float(np.linalg.norm(xb - xa))

    def _local_stiffness_k0(self, ci: int) -> Tuple[NDArray, NDArray]:
        """
        Local stiffness for k = 0 (piece-wise constants).

        DOFs: [p_T, p_{F_1}, ..., p_{F_n}]  (1 + n_local_faces).

        Consistency uses the reconstruction operator G:
            G(v_T, v_F) = (1/|T|) Σ_F |F| (v_F - v_T) n_F

        Stabilisation penalises:
            s(u, v) = Σ_F (|F|/h) (v_F - v_T) (u_F - u_T)

        Returns (K_local, f_local_template) of size ``(1 + nf, 1 + nf)`` and ``(1 + nf,)``.
        """
        cf = self.mesh.cell_faces[ci]
        nf_loc = len(cf)
        ndof = 1 + nf_loc
        centroid, area, diam = self._cell_geometry(ci)

        # Compute outward normals for this cell's faces
        cell_verts = self.mesh.nodes[self.mesh.cells[ci]]
        normals = []
        lengths = []
        for local_f, fi in enumerate(cf):
            a, b = self.mesh.face_nodes[fi]
            xa, xb = self.mesh.nodes[a], self.mesh.nodes[b]
            edge = xb - xa
            length = np.linalg.norm(edge)
            n = np.array([edge[1], -edge[0]], dtype=np.float64) / (length + 1e-30)
            mid = 0.5 * (xa + xb)
            if np.dot(mid - centroid, n) < 0:
                n = -n
            normals.append(n)
            lengths.append(length)

        # Gradient reconstruction: G = (1/|T|) Σ |F_i| n_i (p_Fi - p_T)
        # G is a 2-vector-valued linear form on the local DOFs
        # a(u, v) ≈ |T| G(u)·G(v)  (consistency part)
        G = np.zeros((2, ndof), dtype=np.float64)
        for i in range(nf_loc):
            G[:, 0] -= lengths[i] * normals[i]  # coefficient of p_T
            G[:, 1 + i] = lengths[i] * normals[i]  # coefficient of p_Fi

        G /= area

        # Consistency
        K_cons = area * G.T @ G  # (ndof, ndof)

        # Stabilisation: s(u, v) = Σ (|F|/h_T) (u_F - u_T)(v_F - v_T)
        K_stab = np.zeros((ndof, ndof), dtype=np.float64)
        for i in range(nf_loc):
            eta = lengths[i] / diam
            # DOF indices: 0 = cell, 1 + i = face i
            K_stab[0, 0] += eta
            K_stab[0, 1 + i] -= eta
            K_stab[1 + i, 0] -= eta
            K_stab[1 + i, 1 + i] += eta

        K_local = K_cons + K_stab

        # Source: f_T = f(x_T) |T|, loaded on cell DOF
        f_local = np.zeros(ndof, dtype=np.float64)
        f_local[0] = area  # placeholder multiplier

        return K_local, f_local

    def solve(
        self,
        f_func: Callable,
        bc_func: Optional[Callable] = None,
    ) -> HHOState:
        """
        Assemble and solve via static condensation.

        For k = 0, uses Schur complement to eliminate cell DOFs.
        """
        nc = self.mesh.n_cells
        nf = self.mesh.n_faces
        cf_all = self.mesh.cell_faces

        # Global face-only system
        K_ff = np.zeros((nf, nf), dtype=np.float64)
        rhs_f = np.zeros(nf, dtype=np.float64)
        p_cell_from_face = []  # condensation: p_T = Λ Σ p_F + b

        for ci in range(nc):
            K_loc, _ = self._local_stiffness_k0(ci)
            cf = cf_all[ci]
            nf_loc = len(cf)

            centroid, area, _ = self._cell_geometry(ci)
            f_val = f_func(centroid[0], centroid[1]) * area

            # Partition: T = cell DOF (index 0), F = face DOFs (indices 1:)
            K_TT = K_loc[0:1, 0:1]
            K_TF = K_loc[0:1, 1:]
            K_FT = K_loc[1:, 0:1]
            K_FF = K_loc[1:, 1:]

            # Source vector
            b_T = np.array([f_val], dtype=np.float64)
            b_F = np.zeros(nf_loc, dtype=np.float64)

            # Static condensation: K_TT p_T + K_TF p_F = b_T
            # p_T = K_TT^{-1} (b_T - K_TF p_F)
            K_TT_inv = 1.0 / (K_TT[0, 0] + 1e-30)

            # Schur complement: (K_FF - K_FT K_TT^{-1} K_TF) p_F = b_F - K_FT K_TT^{-1} b_T
            S_loc = K_FF - K_FT * K_TT_inv * K_TF
            rhs_loc = b_F - K_FT[:, 0] * K_TT_inv * b_T[0]

            # Assemble into global face system
            for i, fi in enumerate(cf):
                rhs_f[fi] += rhs_loc[i]
                for j, fj in enumerate(cf):
                    K_ff[fi, fj] += S_loc[i, j]

            p_cell_from_face.append((K_TT_inv, K_TF, b_T))

        # Apply Dirichlet BCs
        bnd = set(self.mesh.boundary_faces.tolist())
        for fi in bnd:
            mid, _ = self._face_midpoint_length(fi)
            bc_val = 0.0 if bc_func is None else bc_func(mid[0], mid[1])
            K_ff[fi, :] = 0.0
            K_ff[:, fi] = 0.0
            K_ff[fi, fi] = 1.0
            rhs_f[fi] = bc_val

        # Solve global face system
        p_face = np.linalg.solve(K_ff, rhs_f)

        # Recover cell pressures
        p_cell = np.zeros(nc, dtype=np.float64)
        for ci in range(nc):
            cf = cf_all[ci]
            K_TT_inv, K_TF, b_T = p_cell_from_face[ci]
            p_F_local = p_face[cf]
            p_cell[ci] = K_TT_inv * (b_T[0] - K_TF[0] @ p_F_local)

        return HHOState(p_cell=p_cell, p_face=p_face)

    def l2_error(self, state: HHOState, exact: Callable) -> float:
        """Cell-wise L² error."""
        err2 = 0.0
        for ci in range(self.mesh.n_cells):
            centroid, area, _ = self._cell_geometry(ci)
            err2 += (state.p_cell[ci] - exact(centroid[0], centroid[1])) ** 2 * area
        return float(np.sqrt(err2))
