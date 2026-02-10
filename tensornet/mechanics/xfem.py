"""
Extended Finite Element Method (XFEM)
======================================

Enriches the standard FEM approximation space with discontinuous and
near-tip singular functions to model cracks, interfaces, and other
discontinuities without conforming meshes.

Approximation:
    u^h(x) = Σ_I N_I(x) u_I                     — standard FEM
            + Σ_J N_J(x) H(x) a_J               — Heaviside enrichment
            + Σ_K N_K(x) Σ_α F_α(x) b_{Kα}      — crack-tip enrichment

Heaviside function:
    H(x) = sign(φ(x)) where φ is the signed distance to the crack.

Near-tip enrichment functions (Westergaard):
    F_α = { √r sin(θ/2), √r cos(θ/2),
            √r sin(θ/2) sin θ, √r cos(θ/2) sin θ }

Level-set crack description:
    φ(x) — signed distance (normal to crack).
    ψ(x) — signed distance (tangent to crack, locates tip).

References:
    [1] Moës, Dolbow & Belytschko, "A Finite Element Method for
        Crack Growth without Remeshing",
        Int. J. Numer. Meth. Eng. 46, 1999.
    [2] Belytschko & Black, "Elastic Crack Growth in Finite Elements
        with Minimal Remeshing",
        Int. J. Numer. Meth. Eng. 45, 1999.
    [3] Fries & Belytschko, "The Extended/Generalized Finite Element
        Method: An Overview of the Method and its Applications",
        Int. J. Numer. Meth. Eng. 84, 2010.

Domain III.1 — Solid Mechanics / Extended Finite Element Method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enrichment functions
# ---------------------------------------------------------------------------

def heaviside(phi: NDArray) -> NDArray:
    """Generalised Heaviside: +1 above crack, -1 below."""
    return np.sign(phi).astype(np.float64)


def crack_tip_enrichments(r: NDArray, theta: NDArray) -> list[NDArray]:
    r"""
    Westergaard near-tip enrichment functions:

    .. math::
        \{F_\alpha\} = \left\{
            \sqrt{r}\sin\frac\theta2,\;
            \sqrt{r}\cos\frac\theta2,\;
            \sqrt{r}\sin\frac\theta2\sin\theta,\;
            \sqrt{r}\cos\frac\theta2\sin\theta
        \right\}
    """
    sqrt_r = np.sqrt(np.maximum(r, 0.0))
    ht = theta / 2.0
    return [
        sqrt_r * np.sin(ht),
        sqrt_r * np.cos(ht),
        sqrt_r * np.sin(ht) * np.sin(theta),
        sqrt_r * np.cos(ht) * np.sin(theta),
    ]


# ---------------------------------------------------------------------------
# 2D Q4 Element (bi-linear quad)
# ---------------------------------------------------------------------------

def _q4_shape(xi: float, eta: float) -> NDArray:
    """4-node quad shape functions at reference point (ξ, η)."""
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ], dtype=np.float64)


def _q4_shape_deriv(xi: float, eta: float) -> NDArray:
    """Shape function derivatives: ``(2, 4)`` = [dN/dξ; dN/dη]."""
    dNdxi = 0.25 * np.array([
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
    ], dtype=np.float64)
    return dNdxi


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class XFEMState:
    """
    XFEM solution state.

    Attributes:
        u_std: Standard DOF — ``(n_nodes, 2)`` displacement.
        a_hev: Heaviside enrichment DOF — ``(n_enriched_hev, 2)``.
        b_tip: Crack-tip enrichment DOF — ``(n_enriched_tip, 4, 2)``.
    """
    u_std: NDArray
    a_hev: NDArray
    b_tip: NDArray

    @property
    def n_nodes(self) -> int:
        return self.u_std.shape[0]


@dataclass
class CrackGeometry:
    """
    Level-set crack description.

    Attributes:
        phi: Signed distance to crack surface — ``(n_nodes,)``.
        psi: Signed distance along crack (tip locator) — ``(n_nodes,)``.
        tip: Crack tip coordinates ``(2,)``.
    """
    phi: NDArray
    psi: NDArray
    tip: NDArray


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class XFEMSolver:
    r"""
    2D XFEM solver for linear elastic fracture.

    Solves the enriched system:

    .. math::
        \mathbf{K}^{enr} \mathbf{d} = \mathbf{f}

    where :math:`\mathbf{K}^{enr}` includes standard, Heaviside,
    and tip enrichment contributions.

    Parameters:
        nodes: Nodal coordinates ``(n_nodes, 2)``.
        elements: Element connectivity ``(n_elem, 4)`` (Q4).
        E: Young's modulus.
        nu: Poisson's ratio.
        plane: ``'stress'`` or ``'strain'``.

    Example::

        solver = XFEMSolver(nodes, elements, E=210e9, nu=0.3)
        crack = CrackGeometry(phi, psi, tip)
        solver.set_crack(crack)
        state = solver.solve(f_ext)
        K_I, K_II = solver.compute_sif(state, crack)
    """

    def __init__(
        self,
        nodes: NDArray,
        elements: NDArray,
        E: float = 210e9,
        nu: float = 0.3,
        plane: str = "stress",
    ) -> None:
        self.nodes = nodes.astype(np.float64)
        self.elements = elements.astype(np.intp)
        self.n_nodes = nodes.shape[0]
        self.n_elements = elements.shape[0]
        self.E = E
        self.nu = nu

        # Constitutive matrix
        if plane == "stress":
            factor = E / (1.0 - nu ** 2)
            self.D = factor * np.array([
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ], dtype=np.float64)
        else:
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.D = factor * np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2 * nu) / 2],
            ], dtype=np.float64)

        self._crack: Optional[CrackGeometry] = None
        self._hev_nodes: list[int] = []
        self._tip_nodes: list[int] = []

        # Gauss quadrature 2×2
        gp = 1.0 / np.sqrt(3.0)
        self._gauss_pts = np.array([
            [-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp],
        ], dtype=np.float64)
        self._gauss_wts = np.ones(4, dtype=np.float64)

    def set_crack(self, crack: CrackGeometry) -> None:
        """Identify enriched nodes from level-set crack representation."""
        self._crack = crack

        # Heaviside-enriched: elements cut by crack (φ changes sign)
        hev_set = set()
        tip_set = set()
        for e in range(self.n_elements):
            nids = self.elements[e]
            phi_e = crack.phi[nids]
            psi_e = crack.psi[nids]

            cut = np.min(phi_e) * np.max(phi_e) < 0
            contains_tip = np.min(psi_e) * np.max(psi_e) < 0

            if cut and not contains_tip:
                hev_set.update(nids.tolist())
            if contains_tip:
                tip_set.update(nids.tolist())

        self._hev_nodes = sorted(hev_set)
        self._tip_nodes = sorted(tip_set)

    @property
    def n_hev(self) -> int:
        return len(self._hev_nodes)

    @property
    def n_tip(self) -> int:
        return len(self._tip_nodes)

    @property
    def total_dof(self) -> int:
        return 2 * self.n_nodes + 2 * self.n_hev + 8 * self.n_tip

    def _hev_local(self, node_id: int) -> int:
        """Map global node id to Heaviside DOF index."""
        return self._hev_nodes.index(node_id)

    def _tip_local(self, node_id: int) -> int:
        return self._tip_nodes.index(node_id)

    def _elem_stiffness(self, elem_idx: int) -> Tuple[NDArray, NDArray]:
        """
        Compute element stiffness matrix and DOF mapping,
        including enrichment contributions.
        """
        nids = self.elements[elem_idx]
        xe = self.nodes[nids]  # (4, 2)

        # Determine enriched nodes in this element
        hev_in_elem = [n for n in nids if n in self._hev_nodes]
        tip_in_elem = [n for n in nids if n in self._tip_nodes]

        n_std = 8
        n_hev_dof = 2 * len(hev_in_elem)
        n_tip_dof = 8 * len(tip_in_elem)
        n_local = n_std + n_hev_dof + n_tip_dof

        K_local = np.zeros((n_local, n_local), dtype=np.float64)

        for gp, gw in zip(self._gauss_pts, self._gauss_wts):
            xi, eta = gp
            N = _q4_shape(xi, eta)
            dN = _q4_shape_deriv(xi, eta)

            # Jacobian
            J = dN @ xe
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-30:
                continue
            dNdx = np.linalg.solve(J, dN)  # (2, 4) = dN/dx

            # Physical point
            x_phys = N @ xe  # (2,)

            # Standard B-matrix
            B_std = np.zeros((3, n_std), dtype=np.float64)
            for i in range(4):
                B_std[0, 2 * i] = dNdx[0, i]
                B_std[1, 2 * i + 1] = dNdx[1, i]
                B_std[2, 2 * i] = dNdx[1, i]
                B_std[2, 2 * i + 1] = dNdx[0, i]

            # Build enriched B columns
            B_all = B_std
            col_offset = n_std

            # Heaviside enrichment
            if hev_in_elem and self._crack is not None:
                phi_gp = float(N @ self._crack.phi[nids])
                H_gp = np.sign(phi_gp)
                for node in hev_in_elem:
                    li = list(nids).index(node)
                    H_node = np.sign(self._crack.phi[node])
                    psi_val = N[li] * (H_gp - H_node)
                    dpsi_dx = dNdx[0, li] * (H_gp - H_node)
                    dpsi_dy = dNdx[1, li] * (H_gp - H_node)

                    B_hev = np.zeros((3, 2), dtype=np.float64)
                    B_hev[0, 0] = dpsi_dx
                    B_hev[1, 1] = dpsi_dy
                    B_hev[2, 0] = dpsi_dy
                    B_hev[2, 1] = dpsi_dx
                    B_all = np.hstack([B_all, B_hev])

            # Crack-tip enrichment
            if tip_in_elem and self._crack is not None:
                dx = x_phys - self._crack.tip
                r_gp = np.linalg.norm(dx)
                theta_gp = np.arctan2(dx[1], dx[0])
                F_vals = crack_tip_enrichments(
                    np.array([r_gp]), np.array([theta_gp])
                )
                for node in tip_in_elem:
                    li = list(nids).index(node)
                    for alpha in range(4):
                        F_a = float(F_vals[alpha][0])
                        dx_n = self.nodes[node] - self._crack.tip
                        r_n = np.linalg.norm(dx_n)
                        th_n = np.arctan2(dx_n[1], dx_n[0])
                        F_n = float(crack_tip_enrichments(
                            np.array([r_n]), np.array([th_n])
                        )[alpha][0])

                        enrichment = N[li] * (F_a - F_n)
                        d_enrichment_dx = dNdx[0, li] * (F_a - F_n)
                        d_enrichment_dy = dNdx[1, li] * (F_a - F_n)

                        B_tip = np.zeros((3, 2), dtype=np.float64)
                        B_tip[0, 0] = d_enrichment_dx
                        B_tip[1, 1] = d_enrichment_dy
                        B_tip[2, 0] = d_enrichment_dy
                        B_tip[2, 1] = d_enrichment_dx
                        B_all = np.hstack([B_all, B_tip])

            # Accumulate stiffness
            n_actual = B_all.shape[1]
            K_gp = B_all.T @ self.D @ B_all * detJ * gw
            K_local[:n_actual, :n_actual] += K_gp

        # Build DOF map  (global DOF indices)
        dof_map = []
        for i in range(4):
            dof_map.extend([2 * nids[i], 2 * nids[i] + 1])
        base = 2 * self.n_nodes
        for node in hev_in_elem:
            li = self._hev_local(node)
            dof_map.extend([base + 2 * li, base + 2 * li + 1])
        base2 = base + 2 * self.n_hev
        for node in tip_in_elem:
            li = self._tip_local(node)
            for alpha in range(4):
                dof_map.extend([base2 + 8 * li + 2 * alpha,
                                base2 + 8 * li + 2 * alpha + 1])

        return K_local[:len(dof_map), :len(dof_map)], np.array(dof_map, dtype=np.intp)

    def assemble(self) -> NDArray:
        """Assemble global enriched stiffness matrix."""
        n_dof = self.total_dof
        K = np.zeros((n_dof, n_dof), dtype=np.float64)
        for e in range(self.n_elements):
            K_e, dof_map = self._elem_stiffness(e)
            for i, gi in enumerate(dof_map):
                for j, gj in enumerate(dof_map):
                    K[gi, gj] += K_e[i, j]
        return K

    def solve(
        self,
        f_ext: NDArray,
        bc_nodes: Optional[list[int]] = None,
        bc_values: Optional[NDArray] = None,
    ) -> XFEMState:
        """
        Solve the enriched linear system Ku = f.

        Args:
            f_ext: External force vector ``(total_dof,)``.
            bc_nodes: Node indices with Dirichlet BCs.
            bc_values: Prescribed displacements ``(len(bc_nodes), 2)``.
        """
        K = self.assemble()
        rhs = f_ext.copy()

        # Apply Dirichlet BCs
        if bc_nodes is not None and bc_values is not None:
            for i, nid in enumerate(bc_nodes):
                for d in range(2):
                    dof = 2 * nid + d
                    K[dof, :] = 0.0
                    K[dof, dof] = 1.0
                    rhs[dof] = bc_values[i, d]

        d = np.linalg.solve(K, rhs)

        u_std = d[:2 * self.n_nodes].reshape(self.n_nodes, 2)
        a_hev = d[2 * self.n_nodes:2 * self.n_nodes + 2 * self.n_hev].reshape(-1, 2)
        b_tip = d[2 * self.n_nodes + 2 * self.n_hev:].reshape(-1, 4, 2)

        return XFEMState(u_std=u_std, a_hev=a_hev, b_tip=b_tip)

    def compute_sif(
        self,
        state: XFEMState,
        crack: CrackGeometry,
        r_domain: float = 0.1,
    ) -> Tuple[float, float]:
        r"""
        Compute stress intensity factors K_I, K_II via the
        interaction integral (J-integral domain form).

        Approximate from tip enrichment DOFs using Westergaard
        analytical relation:

        .. math::
            K_I \approx \sqrt{2\pi} E' b_1^{(2)}, \quad
            K_II \approx \sqrt{2\pi} E' b_1^{(1)}

        where :math:`E' = E` (plane stress) or :math:`E/(1-\nu^2)`.
        """
        if state.b_tip.size == 0:
            return 0.0, 0.0

        E_prime = self.E
        # Leading tip enrichment coefficients
        b0 = state.b_tip[0]  # (4, 2) — first enriched node
        # Mode I from cos(θ/2) enrichment (α=1), y-component
        K_I = np.sqrt(2.0 * np.pi) * E_prime * b0[1, 1]
        # Mode II from sin(θ/2) enrichment (α=0), x-component
        K_II = np.sqrt(2.0 * np.pi) * E_prime * b0[0, 0]

        return float(K_I), float(K_II)
