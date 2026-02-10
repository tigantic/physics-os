"""
Mimetic Finite Differences (MFD)
==================================

Structure-preserving discretisation that mimics fundamental properties
of continuous differential operators: exact sequence, integration by
parts (Green's formula), and symmetry/positivity of constitutive laws.

For diffusion -∇·(K ∇p) = f on a polyhedral mesh:
    - Pressures at cell centres (p_c)
    - Fluxes at faces (u_f)
    - Discrete divergence: DIV @ u = rhs
    - Discrete gradient: GRAD p approximated via flux consistency
    - Inner product:  (K ∇p, ∇p) ≈ u^T M u

The mimetic inner product M on each cell E is built so that:
    M N = R                (linear exactness)

where N maps fluxes to face normals and R evaluates K.

References:
    [1] Brezzi, Lipnikov & Simoncini, "A Family of Mimetic Finite
        Difference Methods on Polygonal and Polyhedral Meshes",
        Math. Models Methods Appl. Sci. 15, 2005.
    [2] Lipnikov, Manzini & Shashkov, "Mimetic Finite Difference
        Method", J. Comp. Phys. 257, 2014.
    [3] Beirão da Veiga, Lipnikov & Manzini, "The Mimetic Finite
        Difference Method for Elliptic Problems", Springer, 2014.

Domain III.1 — Solid Mechanics / Mimetic Finite Differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Mesh representation
# ---------------------------------------------------------------------------

@dataclass
class MFDMesh2D:
    """
    2D polygonal mesh for MFD.

    Attributes:
        nodes: Vertex coordinates ``(n_nodes, 2)``.
        cells: List of cell connectivity (vertex indices per cell).
        faces: Edge list ``(n_faces, 2)`` — endpoints of each edge.
        cell_faces: Face indices per cell (list of arrays).
        face_normals: Outward face normals ``(n_faces, 2)``.
        face_areas: Face lengths ``(n_faces,)``.
        cell_volumes: Cell areas ``(n_cells,)``.
        cell_centres: Cell centroids ``(n_cells, 2)``.
    """
    nodes: NDArray
    cells: list[NDArray]
    faces: NDArray
    cell_faces: list[NDArray]
    face_normals: NDArray
    face_areas: NDArray
    cell_volumes: NDArray
    cell_centres: NDArray

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]


def build_cartesian_mfd_mesh(
    nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
) -> MFDMesh2D:
    """
    Build MFD mesh data from a Cartesian quad grid.

    Returns a MFDMesh2D with proper topology and geometry.
    """
    dx = Lx / nx
    dy = Ly / ny
    n_nodes = (nx + 1) * (ny + 1)
    n_cells = nx * ny

    # Nodes
    nodes = np.zeros((n_nodes, 2), dtype=np.float64)
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = j * (nx + 1) + i
            nodes[nid] = [i * dx, j * dy]

    # Cells (quad connectivity)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            cells.append(np.array([n0, n1, n2, n3], dtype=np.intp))

    # Faces (edges): horizontal then vertical
    face_list = []
    face_map: dict[Tuple[int, int], int] = {}

    def _add_face(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in face_map:
            face_map[key] = len(face_list)
            face_list.append(key)
        return face_map[key]

    cell_faces_list = []
    for cell in cells:
        n_v = len(cell)
        cf = []
        for k in range(n_v):
            a, b = cell[k], cell[(k + 1) % n_v]
            cf.append(_add_face(a, b))
        cell_faces_list.append(np.array(cf, dtype=np.intp))

    faces = np.array(face_list, dtype=np.intp)
    n_faces = len(faces)

    # Face normals and areas
    face_normals = np.zeros((n_faces, 2), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)
    for fi, (a, b) in enumerate(face_list):
        edge = nodes[b] - nodes[a]
        length = np.linalg.norm(edge)
        face_areas[fi] = length
        # Normal (rotated tangent)
        face_normals[fi] = np.array([edge[1], -edge[0]], dtype=np.float64) / (length + 1e-30)

    # Cell volumes and centres
    cell_volumes = np.zeros(n_cells, dtype=np.float64)
    cell_centres = np.zeros((n_cells, 2), dtype=np.float64)
    for ci, cell in enumerate(cells):
        verts = nodes[cell]
        cell_centres[ci] = np.mean(verts, axis=0)
        # Shoelace area
        n_v = len(cell)
        area = 0.0
        for k in range(n_v):
            kn = (k + 1) % n_v
            area += verts[k, 0] * verts[kn, 1] - verts[kn, 0] * verts[k, 1]
        cell_volumes[ci] = abs(0.5 * area)

    return MFDMesh2D(
        nodes=nodes, cells=cells, faces=faces,
        cell_faces=cell_faces_list,
        face_normals=face_normals, face_areas=face_areas,
        cell_volumes=cell_volumes, cell_centres=cell_centres,
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class MFDState:
    """MFD solution state."""
    p: NDArray  # Cell-centre pressures (n_cells,)
    u: Optional[NDArray] = None  # Face fluxes (n_faces,)

    @property
    def n_cells(self) -> int:
        return self.p.shape[0]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class MimeticFDSolver:
    r"""
    Mimetic Finite Difference solver for 2D diffusion.

    Solves:
    .. math::
        -\nabla \cdot (K \nabla p) = f

    using the mixed formulation:
        u = -K ∇p
        ∇·u = f

    The mimetic inner product ensures:
        - Linear exactness: exact for linear solutions
        - Symmetry and positive-definiteness

    Parameters:
        mesh: MFD polygonal mesh.
        K: Diffusion tensor or scalar.

    Example::

        mesh = build_cartesian_mfd_mesh(20, 20)
        solver = MimeticFDSolver(mesh, K=1.0)
        state = solver.solve(f_func=lambda x, y: 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y))
    """

    def __init__(
        self,
        mesh: MFDMesh2D,
        K: float | NDArray = 1.0,
    ) -> None:
        self.mesh = mesh
        if isinstance(K, (int, float)):
            self.K = float(K) * np.eye(2)
        else:
            self.K = np.asarray(K, dtype=np.float64)

    def _cell_inner_product(self, ci: int) -> NDArray:
        """
        Build mimetic inner product matrix M_E for cell ci.

        For each face f of the cell, the consistency condition is:
            M N = R

        where N_fα = n_f,α |f| and R_fα = K_αβ (x_f - x_c)_β |f|.

        The full matrix is:
            M = M_consistency + M_stability
            M_consistency = R (N^T R)^{-1} R^T / |E|
            M_stability = α_s (I - R (N^T R)^{-1} N^T)^T (...) (scaled identity)
        """
        cf = self.mesh.cell_faces[ci]
        n_f = len(cf)
        vol = self.mesh.cell_volumes[ci]
        xc = self.mesh.cell_centres[ci]

        N = np.zeros((n_f, 2), dtype=np.float64)
        R = np.zeros((n_f, 2), dtype=np.float64)

        for local_f, fi in enumerate(cf):
            face_area = self.mesh.face_areas[fi]
            normal = self.mesh.face_normals[fi]

            # Face midpoint
            a, b = self.mesh.faces[fi]
            x_f = 0.5 * (self.mesh.nodes[a] + self.mesh.nodes[b])

            # Orient normal outward from cell
            if np.dot(x_f - xc, normal) < 0:
                normal = -normal

            N[local_f] = normal * face_area
            R[local_f] = self.K @ (x_f - xc) * face_area

        # Consistency: M_c = (1/vol) R @ inv(N^T R) @ R^T
        NtR = N.T @ R  # (2, 2)
        NtR_inv = np.linalg.inv(NtR + 1e-14 * np.eye(2))
        M_c = (1.0 / vol) * R @ NtR_inv @ R.T

        # Stability: M_s = alpha * (I - R NtR^{-1} N^T)^T @ (I - R NtR^{-1} N^T)
        proj = R @ NtR_inv @ N.T  # (n_f, n_f)
        I_minus_P = np.eye(n_f) - proj
        alpha_s = max(np.trace(M_c) / n_f, 1e-10)
        M_s = alpha_s * I_minus_P.T @ I_minus_P

        return M_c + M_s

    def assemble(self) -> Tuple[NDArray, NDArray]:
        """
        Assemble the saddle-point system:
            [M  B^T] [u]   [0]
            [B  0  ] [p] = [f]

        Returns ``(A, rhs_template)`` for the Schur complement formulation.
        """
        nc = self.mesh.n_cells
        nf = self.mesh.n_faces

        M = np.zeros((nf, nf), dtype=np.float64)
        DIV = np.zeros((nc, nf), dtype=np.float64)

        for ci in range(nc):
            cf = self.mesh.cell_faces[ci]
            M_local = self._cell_inner_product(ci)
            xc = self.mesh.cell_centres[ci]

            for i, fi in enumerate(cf):
                for j, fj in enumerate(cf):
                    M[fi, fj] += M_local[i, j]
                # Divergence
                a, b = self.mesh.faces[fi]
                x_f = 0.5 * (self.mesh.nodes[a] + self.mesh.nodes[b])
                normal = self.mesh.face_normals[fi]
                sign = 1.0 if np.dot(x_f - xc, normal) >= 0 else -1.0
                DIV[ci, fi] = sign * self.mesh.face_areas[fi]

        return M, DIV

    def solve(
        self,
        f_func: Callable,
        bc_type: str = "dirichlet",
        bc_func: Optional[Callable] = None,
    ) -> MFDState:
        """
        Solve the diffusion equation via Schur complement.

        The mixed system is reduced to:
            B M^{-1} B^T p = f
        """
        M, DIV = self.assemble()
        nc = self.mesh.n_cells

        # RHS
        rhs = np.zeros(nc, dtype=np.float64)
        for ci in range(nc):
            xc = self.mesh.cell_centres[ci]
            rhs[ci] = f_func(xc[0], xc[1]) * self.mesh.cell_volumes[ci]

        # Schur complement: S = DIV @ M^{-1} @ DIV^T
        M_inv = np.linalg.inv(M + 1e-14 * np.eye(M.shape[0]))
        S = DIV @ M_inv @ DIV.T  # (nc, nc)

        # Apply Dirichlet BC via penalty
        if bc_type == "dirichlet":
            penalty = 1e10
            for ci in range(nc):
                xc = self.mesh.cell_centres[ci]
                # Check if cell touches boundary
                cf = self.mesh.cell_faces[ci]
                for fi in cf:
                    a, b = self.mesh.faces[fi]
                    xa, xb = self.mesh.nodes[a], self.mesh.nodes[b]
                    mid = 0.5 * (xa + xb)
                    tol = 1e-10
                    on_bnd = (
                        abs(mid[0]) < tol or abs(mid[0] - self.mesh.nodes[:, 0].max()) < tol
                        or abs(mid[1]) < tol or abs(mid[1] - self.mesh.nodes[:, 1].max()) < tol
                    )
                    if on_bnd:
                        if bc_func is not None:
                            bc_val = bc_func(xc[0], xc[1])
                        else:
                            bc_val = 0.0
                        S[ci, ci] += penalty
                        rhs[ci] += penalty * bc_val
                        break

        p = np.linalg.solve(S, rhs)

        # Recover fluxes
        u = -M_inv @ DIV.T @ p

        return MFDState(p=p, u=u)

    def l2_error(
        self,
        state: MFDState,
        exact: Callable,
    ) -> float:
        """Cell-wise L² error."""
        err2 = 0.0
        for ci in range(self.mesh.n_cells):
            xc = self.mesh.cell_centres[ci]
            vol = self.mesh.cell_volumes[ci]
            err2 += (state.p[ci] - exact(xc[0], xc[1])) ** 2 * vol
        return float(np.sqrt(err2))
