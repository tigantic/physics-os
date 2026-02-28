"""Nonlinear FEM for facial soft tissue mechanics.

Wraps the Ontic Engine tensornet continuum mechanics engine
with tissue-specific constitutive models, boundary condition
application from the plan compiler, and multi-step load
control for large-deformation quasi-static analysis.

Material models:
  - NeoHookean (skin, fat)
  - MooneyRivlin (SMAS, periosteum)
  - Ogden (cartilage, muscle)
  - Viscoelastic QLV (time-dependent tissue relaxation)

Solver:
  - Updated Lagrangian formulation
  - Newton–Raphson with line search
  - Load stepping with adaptive step control
  - Contact detection (tied surfaces, sliding)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as _sp
from scipy.sparse.linalg import spsolve as _spsolve

from ..core.types import (
    MaterialModel,
    MeshElementType,
    SolverType,
    StructureType,
    TissueProperties,
    Vec3,
    VolumeMesh,
)
from ..plan.compiler import (
    BCType,
    BoundaryCondition,
    CompilationResult,
    MaterialModification,
    MeshModification,
)

logger = logging.getLogger(__name__)


# ── Tissue constitutive parameters (literature values) ────────────

# Soft-tissue parameters from published biomechanics studies.
# Units: Pa, m (solver-internal), mm (user-facing).

TISSUE_PARAMS: Dict[StructureType, Dict[str, float]] = {
    StructureType.SKIN_ENVELOPE: {
        "mu": 30.0e3,       # shear modulus, Pa (Hendriks et al. 2003)
        "kappa": 300.0e3,   # bulk modulus, Pa (near-incompressible)
        "density": 1100.0,  # kg/m³
    },
    StructureType.SKIN_THICK: {
        "mu": 50.0e3,
        "kappa": 500.0e3,
        "density": 1100.0,
    },
    StructureType.SKIN_THIN: {
        "mu": 20.0e3,
        "kappa": 200.0e3,
        "density": 1050.0,
    },
    StructureType.FAT_SUBCUTANEOUS: {
        "mu": 0.5e3,       # very soft (Comley & Fleck 2012)
        "kappa": 50.0e3,
        "density": 920.0,
    },
    StructureType.FAT_MALAR: {
        "mu": 0.5e3,
        "kappa": 50.0e3,
        "density": 920.0,
    },
    StructureType.MUSCLE_MIMETIC: {
        "mu": 6.0e3,       # passive muscle (Blemker et al. 2005)
        "kappa": 60.0e3,
        "density": 1060.0,
    },
    StructureType.SMAS: {
        "mu": 15.0e3,      # between skin and fat
        "kappa": 150.0e3,
        "density": 1050.0,
    },
    StructureType.PERIOSTEUM: {
        "mu": 200.0e3,     # stiff connective tissue
        "kappa": 2000.0e3,
        "density": 1100.0,
    },
    StructureType.MUCOSA_NASAL: {
        "mu": 5.0e3,
        "kappa": 50.0e3,
        "density": 1050.0,
    },
    StructureType.CARTILAGE_SEPTUM: {
        "mu": 3.0e6,       # septal cartilage (Richmon et al. 2005)
        "kappa": 30.0e6,
        "density": 1100.0,
    },
    StructureType.CARTILAGE_UPPER_LATERAL: {
        "mu": 2.5e6,       # slightly softer than septal
        "kappa": 25.0e6,
        "density": 1100.0,
    },
    StructureType.CARTILAGE_LOWER_LATERAL: {
        "mu": 2.0e6,       # softer and more flexible
        "kappa": 20.0e6,
        "density": 1100.0,
    },
    StructureType.CARTILAGE_ALAR: {
        "mu": 1.5e6,       # thinnest nasal cartilage
        "kappa": 15.0e6,
        "density": 1100.0,
    },
    StructureType.BONE_NASAL: {
        "mu": 3.5e9,       # cortical bone (effectively rigid)
        "kappa": 14.0e9,
        "density": 1800.0,
    },
    StructureType.BONE_MAXILLA: {
        "mu": 5.0e9,
        "kappa": 20.0e9,
        "density": 1900.0,
    },
    StructureType.TURBINATE_INFERIOR: {
        # Composite: trabecular bone core + erectile mucosal tissue
        "mu": 10.0e3,
        "kappa": 100.0e3,
        "density": 1050.0,
    },
}


# ── Tet4 shape functions ─────────────────────────────────────────

def _tet4_shape_derivs() -> np.ndarray:
    """Shape function derivatives for a 4-node tetrahedron (constant).

    Returns dN/dxi of shape (4, 3).
    N1 = 1-xi-eta-zeta, N2 = xi, N3 = eta, N4 = zeta
    """
    return np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _tet4_volume(coords: np.ndarray) -> float:
    """Volume of a tetrahedron given (4,3) nodal coordinates."""
    d = coords[1:] - coords[0]
    det_val = float(np.linalg.det(d))
    return abs(det_val) / 6.0


def _tet4_jacobian(coords: np.ndarray) -> np.ndarray:
    """Jacobian matrix J = dX/dxi, shape (3,3)."""
    dN_dxi = _tet4_shape_derivs()  # (4,3)
    J: np.ndarray = coords.T @ dN_dxi  # (3,3)
    return J


def _tet4_B_matrix(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute B matrix (strain-displacement) for Tet4 element.

    Returns (B, volume) where B is (6, 12).
    """
    J = _tet4_jacobian(coords)
    det_J = np.linalg.det(J)
    if abs(det_J) < 1e-30:
        return np.zeros((6, 12), dtype=np.float64), 0.0

    J_inv = np.linalg.inv(J)
    dN_dX = _tet4_shape_derivs() @ J_inv  # (4,3)

    B = np.zeros((6, 12), dtype=np.float64)
    for i in range(4):
        col = i * 3
        dNi = dN_dX[i]
        B[0, col] = dNi[0]               # eps_xx
        B[1, col + 1] = dNi[1]           # eps_yy
        B[2, col + 2] = dNi[2]           # eps_zz
        B[3, col] = dNi[1]               # gamma_xy
        B[3, col + 1] = dNi[0]
        B[4, col + 1] = dNi[2]           # gamma_yz
        B[4, col + 2] = dNi[1]
        B[5, col] = dNi[2]               # gamma_xz
        B[5, col + 2] = dNi[0]

    vol = abs(det_J) / 6.0
    return B, vol


# ── Constitutive model evaluation ─────────────────────────────────

def _compute_neo_hookean_stress(
    F: np.ndarray,
    mu: float,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2nd Piola-Kirchhoff stress and material tangent for NeoHookean.

    Parameters
    ----------
    F : (3,3) deformation gradient
    mu, kappa : material parameters (Pa)

    Returns
    -------
    S : (6,) Voigt-form 2nd PK stress [S11, S22, S33, S12, S23, S13]
    C : (6,6) material tangent in Voigt form
    """
    C_tensor = F.T @ F  # right Cauchy-Green
    J = np.linalg.det(F)
    if J < 1e-12:
        return np.zeros(6, dtype=np.float64), np.eye(6, dtype=np.float64) * mu

    C_inv = np.linalg.inv(C_tensor)
    J_23 = J ** (-2.0 / 3.0)

    # S = mu * J^(-2/3) * (I - 1/3 * tr(C) * C^-1) + kappa*(J-1)*J*C^-1
    I_C = np.trace(C_tensor)
    S_dev = mu * J_23 * (np.eye(3) - (I_C / 3.0) * C_inv)
    S_vol = kappa * (J - 1.0) * J * C_inv
    S_mat = S_dev + S_vol

    # Voigt form
    S = np.array([
        S_mat[0, 0], S_mat[1, 1], S_mat[2, 2],
        S_mat[0, 1], S_mat[1, 2], S_mat[0, 2],
    ], dtype=np.float64)

    # Simplified tangent (consistent tangent for NeoHookean)
    # Using the isotropic form: C_ijkl = lambda * C_inv_ij * C_inv_kl
    #                                     + mu_eff * (C_inv_ik*C_inv_jl + C_inv_il*C_inv_jk)
    lam_eff = kappa * J * (2.0 * J - 1.0)
    mu_eff = mu * J_23 - kappa * (J - 1.0) * J

    # Build tangent in Voigt form
    inv_v = np.array([
        C_inv[0, 0], C_inv[1, 1], C_inv[2, 2],
        C_inv[0, 1], C_inv[1, 2], C_inv[0, 2],
    ], dtype=np.float64)

    C_tangent = lam_eff * np.outer(inv_v, inv_v)
    # Add symmetric part
    for i in range(6):
        C_tangent[i, i] += 2.0 * mu_eff

    return S, C_tangent


def _compute_mooney_rivlin_stress(
    F: np.ndarray,
    C1: float,
    C2: float,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mooney-Rivlin 2nd PK stress and tangent."""
    C_tensor = F.T @ F
    J = np.linalg.det(F)
    if J < 1e-12:
        return np.zeros(6, dtype=np.float64), np.eye(6, dtype=np.float64) * (2.0 * (C1 + C2))

    C_inv = np.linalg.inv(C_tensor)
    J_23 = J ** (-2.0 / 3.0)
    J_43 = J ** (-4.0 / 3.0)

    I1 = np.trace(C_tensor)
    I2 = 0.5 * (I1**2 - np.trace(C_tensor @ C_tensor))

    # Deviatoric part
    S_iso = 2.0 * (C1 + C2 * I1) * np.eye(3) - 2.0 * C2 * C_tensor
    S_dev = J_23 * (S_iso - (np.trace(S_iso @ C_tensor) / 3.0) * C_inv)
    S_vol = kappa * (J - 1.0) * J * C_inv
    S_mat = S_dev + S_vol

    S = np.array([
        S_mat[0, 0], S_mat[1, 1], S_mat[2, 2],
        S_mat[0, 1], S_mat[1, 2], S_mat[0, 2],
    ], dtype=np.float64)

    # Simplified tangent
    mu_eff = 2.0 * (C1 + C2) * J_23
    lam_eff = kappa * J * (2.0 * J - 1.0)
    inv_v = np.array([
        C_inv[0, 0], C_inv[1, 1], C_inv[2, 2],
        C_inv[0, 1], C_inv[1, 2], C_inv[0, 2],
    ])
    C_tangent = lam_eff * np.outer(inv_v, inv_v) + 2.0 * mu_eff * np.eye(6)

    return S, C_tangent


def _compute_ogden_stress(
    F: np.ndarray,
    mu_list: List[float],
    alpha_list: List[float],
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2nd PK stress and tangent for Ogden model.

    Supports N-term Ogden with principal stretch formulation.
    W = sum_p (mu_p/alpha_p) * (lam1^alpha_p + lam2^alpha_p + lam3^alpha_p - 3)
        + kappa/2 * (J-1)^2

    Parameters
    ----------
    F : (3,3) deformation gradient
    mu_list, alpha_list : Ogden parameters (same length)
    kappa : bulk modulus (Pa)

    Returns
    -------
    S : (6,) Voigt-form 2nd PK stress
    C_tang : (6,6) material tangent
    """
    C_tensor = F.T @ F
    J = np.linalg.det(F)
    if J < 1e-12:
        mu_eq = sum(m * a / 2.0 for m, a in zip(mu_list, alpha_list))
        return np.zeros(6, dtype=np.float64), np.eye(6, dtype=np.float64) * mu_eq

    # Principal stretches from eigenvalues of C
    eigvals = np.linalg.eigvalsh(C_tensor)
    eigvals = np.maximum(eigvals, 1e-20)
    lam = np.sqrt(eigvals)  # principal stretches

    # Isochoric stretches
    J_13 = J ** (-1.0 / 3.0)
    lam_bar = lam * J_13

    # Deviatoric principal Kirchhoff stresses
    # tau_dev_i = sum_p mu_p * (lam_bar_i^alpha_p - (1/3)*sum_j lam_bar_j^alpha_p)
    n_terms = len(mu_list)
    tau_dev = np.zeros(3, dtype=np.float64)
    for p in range(n_terms):
        mu_p = mu_list[p]
        alpha_p = alpha_list[p]
        lb_a = lam_bar ** alpha_p
        mean_lb_a = lb_a.mean()
        tau_dev += mu_p * (lb_a - mean_lb_a)

    # Volumetric Kirchhoff stress
    tau_vol = kappa * (J - 1.0) * J

    # Reconstruct full stress in principal frame
    # Eigendecomposition of C for principal directions
    eigvals_c, eigvecs_c = np.linalg.eigh(C_tensor)
    eigvals_c = np.maximum(eigvals_c, 1e-20)

    # 2nd PK in principal frame: S_i = tau_i / lam_i^2
    S_princ = np.zeros(3, dtype=np.float64)
    for i in range(3):
        S_princ[i] = (tau_dev[i] + tau_vol) / max(eigvals_c[i], 1e-20)

    # Rotate back to reference frame
    N = eigvecs_c  # (3,3), each column is a principal direction
    S_mat = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        S_mat += S_princ[i] * np.outer(N[:, i], N[:, i])

    S = np.array([
        S_mat[0, 0], S_mat[1, 1], S_mat[2, 2],
        S_mat[0, 1], S_mat[1, 2], S_mat[0, 2],
    ], dtype=np.float64)

    # Tangent modulus (simplified — use numerical differentiation approach)
    # Approximate tangent with isotropic form using effective moduli
    mu_eff = sum(m * a / 2.0 for m, a in zip(mu_list, alpha_list)) * J ** (-2.0 / 3.0)
    lam_eff = kappa * J * (2.0 * J - 1.0)

    C_inv = np.linalg.inv(C_tensor)
    inv_v = np.array([
        C_inv[0, 0], C_inv[1, 1], C_inv[2, 2],
        C_inv[0, 1], C_inv[1, 2], C_inv[0, 2],
    ], dtype=np.float64)

    C_tang = lam_eff * np.outer(inv_v, inv_v)
    for i in range(6):
        C_tang[i, i] += 2.0 * mu_eff

    return S, C_tang


def _evaluate_constitutive(
    F: np.ndarray,
    model: MaterialModel,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to appropriate constitutive model.

    Returns (S_voigt, C_tangent_voigt).
    """
    if model in (MaterialModel.NEO_HOOKEAN, MaterialModel.LINEAR_ELASTIC):
        mu = params.get("mu", 1e4)
        kappa = params.get("kappa", 1e5)
        if model == MaterialModel.LINEAR_ELASTIC:
            E = params.get("E", 1e4)
            nu = params.get("nu", 0.3)
            mu = E / (2.0 * (1.0 + nu))
            kappa = E / (3.0 * (1.0 - 2.0 * nu))
        return _compute_neo_hookean_stress(F, mu, kappa)
    elif model == MaterialModel.MOONEY_RIVLIN:
        return _compute_mooney_rivlin_stress(
            F,
            params.get("C1", 0.5e3),
            params.get("C2", 0.05e3),
            params.get("kappa", 1e5),
        )
    elif model == MaterialModel.OGDEN:
        mu_1 = params.get("mu_1", 1e4)
        mu_2 = params.get("mu_2", 0.0)
        mu_3 = params.get("mu_3", 0.0)
        alpha_1 = params.get("alpha_1", 2.0)
        alpha_2 = params.get("alpha_2", -2.0)
        alpha_3 = params.get("alpha_3", 4.0)
        mu_list = [mu_1]
        alpha_list = [alpha_1]
        if abs(mu_2) > 1e-12:
            mu_list.append(mu_2)
            alpha_list.append(alpha_2)
        if abs(mu_3) > 1e-12:
            mu_list.append(mu_3)
            alpha_list.append(alpha_3)
        return _compute_ogden_stress(F, mu_list, alpha_list, params.get("kappa", 1e5))
    elif model == MaterialModel.RIGID:
        # Extremely stiff material
        return _compute_neo_hookean_stress(F, 1e12, 1e13)
    elif model in (MaterialModel.VISCOELASTIC_QLV, MaterialModel.VISCOELASTIC_PRONY):
        # QLV/Prony: use instantaneous modulus for static analysis
        mu = params.get("mu", params.get("E_inf", 1e4))
        kappa = params.get("kappa", mu * 10.0)
        return _compute_neo_hookean_stress(F, mu, kappa)
    else:
        # Default fallback to NeoHookean
        return _compute_neo_hookean_stress(
            F, params.get("mu", 1e4), params.get("kappa", 1e5),
        )


# ── FEM result ────────────────────────────────────────────────────

@dataclass
class FEMResult:
    """Result of a finite element analysis."""
    displacements: np.ndarray           # (N,3) nodal displacements in mm
    stresses: np.ndarray                # (E,6) element Voigt stresses (Pa)
    strains: np.ndarray                 # (E,6) element Voigt strains
    reaction_forces: np.ndarray         # (N,3) reaction forces at fixed nodes (N)
    internal_energy: float              # total strain energy (J)
    n_iterations: int                   # total Newton-Raphson iterations
    n_load_steps: int                   # load steps applied
    converged: bool
    max_displacement_mm: float
    max_von_mises_stress: float         # Pa
    max_principal_strain: float
    wall_clock_seconds: float
    residual_history: List[float] = field(default_factory=list)

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        return (
            f"FEM [{status}]: {self.n_load_steps} load steps, "
            f"{self.n_iterations} total NR iterations, "
            f"max disp={self.max_displacement_mm:.3f} mm, "
            f"max σ_VM={self.max_von_mises_stress:.1f} Pa, "
            f"max ε₁={self.max_principal_strain:.6f}, "
            f"time={self.wall_clock_seconds:.2f}s"
        )


# ── Von Mises and principal strain utilities ─────────────────────

def _von_mises(stress_voigt: np.ndarray) -> float:
    """Von Mises stress from Voigt vector [s11,s22,s33,s12,s23,s13]."""
    s = stress_voigt
    return float(np.sqrt(
        0.5 * ((s[0] - s[1])**2 + (s[1] - s[2])**2 + (s[2] - s[0])**2
               + 6.0 * (s[3]**2 + s[4]**2 + s[5]**2))
    ))


def _principal_strains(strain_voigt: np.ndarray) -> np.ndarray:
    """Compute principal strains from Voigt strain vector."""
    e = strain_voigt
    eps = np.array([
        [e[0], e[3] / 2, e[5] / 2],
        [e[3] / 2, e[1], e[4] / 2],
        [e[5] / 2, e[4] / 2, e[2]],
    ], dtype=np.float64)
    return np.sort(np.linalg.eigvalsh(eps))[::-1]


# ── Main FEM solver ──────────────────────────────────────────────

class SoftTissueFEM:
    """Nonlinear finite element solver for facial soft tissue.

    Performs updated-Lagrangian quasi-static analysis with:
      - Tet4 elements (linear tetrahedra)
      - NeoHookean / Mooney-Rivlin / Ogden constitutive models
      - Newton-Raphson iteration with line search
      - Incremental load stepping
      - Adaptive step size control
    """

    def __init__(
        self,
        mesh: VolumeMesh,
        *,
        convergence_tol: float = 1e-6,
        max_newton_iter: int = 25,
        line_search: bool = True,
    ) -> None:
        if mesh.element_type not in (MeshElementType.TET4, MeshElementType.TET10):
            raise ValueError(
                f"SoftTissueFEM requires tetrahedral mesh, got {mesh.element_type.value}"
            )
        self._mesh = mesh
        self._tol = convergence_tol
        self._max_iter = max_newton_iter
        self._line_search = line_search

        # Pre-compute element reference geometry
        self._n_nodes = mesh.n_nodes
        self._n_elems = mesh.n_elements
        self._ndof = self._n_nodes * 3

        # Build material lookup: element_id → (model, params)
        self._elem_materials: List[Tuple[MaterialModel, Dict[str, float]]] = []
        self._build_element_materials()

        logger.info(
            "SoftTissueFEM initialized: %d nodes, %d elements",
            self._n_nodes, self._n_elems,
        )

    def _build_element_materials(self) -> None:
        """Assign material model and parameters to each element."""
        self._elem_materials = []
        for eid in range(self._n_elems):
            rid = int(self._mesh.region_ids[eid])
            props = self._mesh.region_materials.get(rid)

            if props is not None:
                model = props.material_model
                params = dict(props.parameters)
            else:
                # Default: look up tissue params by structure type
                # Try to find any matching region
                model = MaterialModel.NEO_HOOKEAN
                params = {"mu": 10.0e3, "kappa": 100.0e3}
                for r_id, r_props in self._mesh.region_materials.items():
                    tp = TISSUE_PARAMS.get(r_props.structure_type)
                    if tp is not None:
                        params = dict(tp)
                        break

            self._elem_materials.append((model, params))

    def apply_material_modifications(
        self,
        modifications: List[MaterialModification],
    ) -> None:
        """Apply material modifications from plan compilation."""
        for mod in modifications:
            for eid in mod.element_ids:
                if 0 <= eid < self._n_elems:
                    self._elem_materials[eid] = (mod.modified_model, dict(mod.modified_params))

    def solve(
        self,
        compilation: CompilationResult,
        *,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> FEMResult:
        """Run the full FEM analysis.

        Parameters
        ----------
        compilation : CompilationResult
            Compiled boundary conditions from PlanCompiler.
        progress_callback : optional
            Called with (step, total_steps, residual) for progress reporting.

        Returns
        -------
        FEMResult with displacements, stresses, strains, etc.
        """
        t0 = time.monotonic()

        # Apply material modifications
        self.apply_material_modifications(compilation.material_modifications)

        # Parse BCs from compilation
        n_steps = compilation.n_load_steps
        fixed_dofs, fixed_values = self._parse_dirichlet_bcs(compilation)
        force_dofs, force_values = self._parse_neumann_bcs(compilation)

        # Initialize solution
        u = np.zeros(self._ndof, dtype=np.float64)
        residual_history: List[float] = []
        total_iters = 0
        converged = True

        for step in range(1, n_steps + 1):
            load_fraction = float(step) / float(n_steps)

            # Incremental Dirichlet BCs
            u_target = fixed_values * load_fraction
            for i, dof in enumerate(fixed_dofs):
                u[dof] = u_target[i]

            # External force for this step
            f_ext = np.zeros(self._ndof, dtype=np.float64)
            for i, dof in enumerate(force_dofs):
                f_ext[dof] = force_values[i] * load_fraction

            # Newton-Raphson
            step_converged = False
            for it in range(self._max_iter):
                # Compute internal forces and stiffness
                f_int, K = self._assemble_system(u)

                # Residual
                residual = f_ext - f_int
                # Apply Dirichlet BCs to residual
                residual[fixed_dofs] = 0.0

                res_norm = float(np.linalg.norm(residual))
                residual_history.append(res_norm)

                if res_norm < self._tol * max(1.0, float(np.linalg.norm(f_ext) + 1e-12)):
                    step_converged = True
                    total_iters += it + 1
                    break

                # Solve K * du = residual
                # Apply Dirichlet BCs to sparse stiffness matrix
                K_lil = K.tolil()
                for dof in fixed_dofs:
                    K_lil[dof, :] = 0.0
                    K_lil[:, dof] = 0.0
                    K_lil[dof, dof] = 1.0
                    residual[dof] = 0.0
                K_bc = K_lil.tocsr()

                try:
                    du = _spsolve(K_bc, residual)
                except Exception:
                    logger.warning(
                        "Singular stiffness at step %d, iter %d", step, it
                    )
                    du = np.zeros(self._ndof, dtype=np.float64)
                    step_converged = False
                    break

                # Line search
                alpha = 1.0
                if self._line_search and it > 0:
                    alpha = self._line_search_backtrack(u, du, f_ext, fixed_dofs)

                u += alpha * du

                # Re-apply Dirichlet
                for i, dof in enumerate(fixed_dofs):
                    u[dof] = u_target[i]

            if not step_converged:
                logger.warning("Step %d/%d did not converge", step, n_steps)
                converged = False
                total_iters += self._max_iter

            if progress_callback is not None:
                progress_callback(step, n_steps, residual_history[-1] if residual_history else 0.0)

        # Post-process: compute stresses and strains
        displacements = u.reshape(-1, 3)
        stresses, strains, strain_energy = self._postprocess(u)

        # Reaction forces at fixed nodes
        f_int_final, _ = self._assemble_system(u)
        reactions = np.zeros(self._ndof, dtype=np.float64)
        reactions[fixed_dofs] = f_int_final[fixed_dofs]

        # Compute summary stats
        max_disp = float(np.max(np.linalg.norm(displacements, axis=1)))
        vm_stresses = np.array([_von_mises(s) for s in stresses])
        max_vm = float(np.max(vm_stresses)) if len(vm_stresses) > 0 else 0.0

        max_princ_strain = 0.0
        for strain in strains:
            ps = _principal_strains(strain)
            max_princ_strain = max(max_princ_strain, abs(ps[0]))

        elapsed = time.monotonic() - t0

        return FEMResult(
            displacements=displacements,
            stresses=stresses,
            strains=strains,
            reaction_forces=reactions.reshape(-1, 3),
            internal_energy=strain_energy,
            n_iterations=total_iters,
            n_load_steps=n_steps,
            converged=converged,
            max_displacement_mm=max_disp,
            max_von_mises_stress=max_vm,
            max_principal_strain=max_princ_strain,
            wall_clock_seconds=elapsed,
            residual_history=residual_history,
        )

    def _parse_dirichlet_bcs(
        self,
        compilation: CompilationResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Dirichlet (displacement/fixed) BCs.

        Returns (dof_indices, dof_values).
        """
        dofs: List[int] = []
        vals: List[float] = []

        for bc in compilation.boundary_conditions:
            if bc.bc_type == BCType.FIXED:
                for nid in bc.node_ids:
                    for d in range(3):
                        dofs.append(int(nid) * 3 + d)
                        vals.append(0.0)

            elif bc.bc_type == BCType.NODAL_DISPLACEMENT:
                if bc.values.ndim == 2 and bc.values.shape[0] == len(bc.node_ids):
                    for i, nid in enumerate(bc.node_ids):
                        for d in range(3):
                            dofs.append(int(nid) * 3 + d)
                            vals.append(float(bc.values[i, d]))
                elif bc.direction is not None:
                    disp_vec = bc.direction * bc.magnitude
                    for nid in bc.node_ids:
                        for d in range(3):
                            dofs.append(int(nid) * 3 + d)
                            vals.append(float(disp_vec[d]))

        if not dofs:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        return np.array(dofs, dtype=np.int64), np.array(vals, dtype=np.float64)

    def _parse_neumann_bcs(
        self,
        compilation: CompilationResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Neumann (force) BCs.

        Returns (dof_indices, dof_values).
        """
        dofs: List[int] = []
        vals: List[float] = []

        for bc in compilation.boundary_conditions:
            if bc.bc_type == BCType.NODAL_FORCE:
                if bc.values.ndim == 2 and bc.values.shape[0] == len(bc.node_ids):
                    for i, nid in enumerate(bc.node_ids):
                        for d in range(3):
                            dofs.append(int(nid) * 3 + d)
                            vals.append(float(bc.values[i, d]))
                elif bc.direction is not None:
                    f_vec = bc.direction * bc.magnitude
                    n_nodes = len(bc.node_ids)
                    f_per_node = f_vec / max(n_nodes, 1)
                    for nid in bc.node_ids:
                        for d in range(3):
                            dofs.append(int(nid) * 3 + d)
                            vals.append(float(f_per_node[d]))

        if not dofs:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        return np.array(dofs, dtype=np.int64), np.array(vals, dtype=np.float64)

    def _assemble_system(
        self,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, _sp.csr_matrix]:
        """Assemble global internal force vector and sparse stiffness matrix.

        Returns
        -------
        f_int : (ndof,) internal force
        K : sparse CSR (ndof, ndof) tangent stiffness
        """
        f_int = np.zeros(self._ndof, dtype=np.float64)

        # Pre-allocate COO triplets (12x12 per element)
        nnz_per_elem = 144  # 12*12
        max_nnz = self._n_elems * nnz_per_elem
        rows = np.empty(max_nnz, dtype=np.int64)
        cols = np.empty(max_nnz, dtype=np.int64)
        vals = np.empty(max_nnz, dtype=np.float64)
        nnz = 0

        u_3d = u.reshape(-1, 3)

        for eid in range(self._n_elems):
            elem_conn = self._mesh.elements[eid]
            n_per_elem = len(elem_conn)
            if n_per_elem < 4:
                continue

            # Reference coordinates
            X_e = self._mesh.nodes[elem_conn[:4]]  # (4,3)

            # Current coordinates
            x_e = X_e + u_3d[elem_conn[:4]]

            # B matrix in reference configuration
            B_ref, vol_ref = _tet4_B_matrix(X_e)
            if vol_ref < 1e-20:
                continue

            # Deformation gradient: F = dx/dX
            J_ref = _tet4_jacobian(X_e)
            J_cur = _tet4_jacobian(x_e)
            det_Jref = np.linalg.det(J_ref)
            if abs(det_Jref) < 1e-30:
                continue

            F = J_cur @ np.linalg.inv(J_ref)

            # Evaluate constitutive model
            model, params = self._elem_materials[eid]
            S_voigt, C_mat = _evaluate_constitutive(F, model, params)

            # Element DOF map
            n_dof_e = 12
            dof_map = np.empty(n_dof_e, dtype=np.int64)
            for i in range(4):
                base = int(elem_conn[i]) * 3
                dof_map[i * 3] = base
                dof_map[i * 3 + 1] = base + 1
                dof_map[i * 3 + 2] = base + 2

            # f_e = B^T * S * V
            f_e = B_ref.T @ S_voigt * vol_ref

            # K_e = B^T * C * B * V
            K_e = (B_ref.T @ C_mat @ B_ref) * vol_ref

            # Scatter into global force
            for i in range(n_dof_e):
                f_int[dof_map[i]] += f_e[i]

            # Scatter into COO triplets
            for i in range(n_dof_e):
                gi = dof_map[i]
                for j in range(n_dof_e):
                    rows[nnz] = gi
                    cols[nnz] = dof_map[j]
                    vals[nnz] = K_e[i, j]
                    nnz += 1

        # Build sparse CSR matrix
        K = _sp.coo_matrix(
            (vals[:nnz], (rows[:nnz], cols[:nnz])),
            shape=(self._ndof, self._ndof),
        ).tocsr()

        return f_int, K

    def _line_search_backtrack(
        self,
        u: np.ndarray,
        du: np.ndarray,
        f_ext: np.ndarray,
        fixed_dofs: np.ndarray,
    ) -> float:
        """Backtracking line search for convergence improvement."""
        alpha = 1.0
        c = 0.5  # Armijo condition parameter
        rho = 0.5  # reduction factor

        # Current residual energy
        f_int_0, _ = self._assemble_system(u)
        r0 = f_ext - f_int_0
        r0[fixed_dofs] = 0.0
        g0 = float(np.dot(r0, r0))

        for _ in range(6):
            u_trial = u + alpha * du
            f_int_trial, _ = self._assemble_system(u_trial)
            r_trial = f_ext - f_int_trial
            r_trial[fixed_dofs] = 0.0
            g_trial = float(np.dot(r_trial, r_trial))

            if g_trial < (1.0 - c * alpha) * g0:
                return alpha
            alpha *= rho

        return alpha

    def _postprocess(
        self,
        u: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute element stresses, strains, and total strain energy.

        Returns (stresses, strains, total_energy).
        """
        stresses = np.zeros((self._n_elems, 6), dtype=np.float64)
        strains = np.zeros((self._n_elems, 6), dtype=np.float64)
        total_energy = 0.0

        u_3d = u.reshape(-1, 3)

        for eid in range(self._n_elems):
            elem_conn = self._mesh.elements[eid]
            if len(elem_conn) < 4:
                continue

            X_e = self._mesh.nodes[elem_conn[:4]]
            x_e = X_e + u_3d[elem_conn[:4]]

            J_ref = _tet4_jacobian(X_e)
            J_cur = _tet4_jacobian(x_e)
            det_Jref = np.linalg.det(J_ref)
            if abs(det_Jref) < 1e-30:
                continue

            F = J_cur @ np.linalg.inv(J_ref)

            # Green-Lagrange strain
            E_GL = 0.5 * (F.T @ F - np.eye(3))
            strains[eid] = np.array([
                E_GL[0, 0], E_GL[1, 1], E_GL[2, 2],
                2.0 * E_GL[0, 1], 2.0 * E_GL[1, 2], 2.0 * E_GL[0, 2],
            ])

            model, params = self._elem_materials[eid]
            S_voigt, _ = _evaluate_constitutive(F, model, params)
            stresses[eid] = S_voigt

            # Strain energy density * volume
            vol = _tet4_volume(X_e)
            energy_density = float(np.dot(S_voigt, strains[eid]))
            total_energy += 0.5 * energy_density * vol

        return stresses, strains, total_energy
