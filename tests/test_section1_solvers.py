"""
Tests for §1.1 — Solver Advances
==================================

Covers: LBM, SPH, DG, SEM, Space-Time DG, Peridynamics, MPM, PFC,
XFEM, IGA, VEM, Mimetic FD, HHO, ILES mode, AMR.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── LBM ─────────────────────────────────────────────────────────────
class TestLBM:
    def test_import_and_config(self):
        from tensornet.cfd.lbm import LatticeModel, CollisionModel, LBMState
        assert LatticeModel.D2Q9 is not None
        assert CollisionModel.BGK is not None

    def test_solver_exists(self):
        from tensornet.cfd.lbm import LatticeBoltzmannSolver
        assert callable(LatticeBoltzmannSolver)


# ── SPH ─────────────────────────────────────────────────────────────
class TestSPH:
    def test_import(self):
        from tensornet.cfd.sph import KernelType, SPHState, SPHSolver, TaitEOS

    def test_tait_eos(self):
        from tensornet.cfd.sph import TaitEOS
        eos = TaitEOS(rho0=1000.0, c0=50.0, gamma=7.0)
        p = eos.pressure(np.array([1000.0]))
        # At rho == rho0, pressure = B * (1^gamma - 1) = 0
        assert abs(p[0]) < 1e-10


# ── DG ──────────────────────────────────────────────────────────────
class TestDG:
    def test_import(self):
        from tensornet.cfd.dg import DGSolver1D, DGState, FluxType

    def test_gauss_legendre(self):
        from tensornet.cfd.dg import gauss_legendre
        x, w = gauss_legendre(5)
        assert len(x) == 5
        assert abs(sum(w) - 2.0) < 1e-12


# ── SEM ─────────────────────────────────────────────────────────────
class TestSEM:
    def test_import(self):
        from tensornet.cfd.sem import SEMSolver1D, SEMState

    def test_gll_points(self):
        from tensornet.cfd.sem import gll_points
        x, w = gll_points(6)
        assert len(x) == 6
        assert abs(x[0] - (-1.0)) < 1e-12
        assert abs(x[-1] - 1.0) < 1e-12


# ── Space-Time DG ──────────────────────────────────────────────────
class TestSpaceTimeDG:
    def test_import(self):
        from tensornet.cfd.space_time_dg import SpaceTimeDGSolver, STDGState


# ── Peridynamics ───────────────────────────────────────────────────
class TestPeridynamics:
    def test_import(self):
        from tensornet.mechanics.peridynamics import (
            PeridynamicsModel,
            PeridynamicsSolver,
            Material,
            PeridynamicState,
        )


# ── MPM ─────────────────────────────────────────────────────────────
class TestMPM:
    def test_import(self):
        from tensornet.mechanics.mpm import (
            MPMSolver,
            MPMParticleState,
            ShapeFunction,
            ConstitutiveModel,
        )


# ── PFC ─────────────────────────────────────────────────────────────
class TestPFC:
    def test_import(self):
        from tensornet.phase_field.pfc import PFCState, PFCSolver


# ── XFEM ────────────────────────────────────────────────────────────
class TestXFEM:
    def test_import(self):
        from tensornet.mechanics.xfem import XFEMSolver, XFEMState, CrackGeometry

    def test_heaviside(self):
        from tensornet.mechanics.xfem import heaviside
        # heaviside returns sign: +1 above, -1 below
        vals = heaviside(np.array([1.0, -1.0, 0.0]))
        assert vals[0] == 1.0
        assert vals[1] == -1.0
        assert vals[2] == 0.0


# ── IGA ─────────────────────────────────────────────────────────────
class TestIGA:
    def test_import(self):
        from tensornet.mechanics.iga import IGASolver1D, IGAState

    def test_bspline_partition_of_unity(self):
        from tensornet.mechanics.iga import basis_funs, find_span
        # Knot vector, degree, param value
        U = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype=float)
        p = 2
        n = len(U) - p - 2  # n = number of basis fns - 1
        xi = 0.3
        # find_span(n, p, u, U)
        span = find_span(n, p, xi, U)
        # basis_funs(i, u, p, U)
        N = basis_funs(span, xi, p, U)
        assert abs(sum(N) - 1.0) < 1e-12


# ── VEM ─────────────────────────────────────────────────────────────
class TestVEM:
    def test_import(self):
        from tensornet.mechanics.vem import VEMSolver, VEMState


# ── Mimetic FD ──────────────────────────────────────────────────────
class TestMimeticFD:
    def test_import(self):
        from tensornet.mechanics.mimetic import (
            MimeticFDSolver,
            MFDMesh2D,
            build_cartesian_mfd_mesh,
        )

    def test_mesh_build(self):
        from tensornet.mechanics.mimetic import build_cartesian_mfd_mesh
        mesh = build_cartesian_mfd_mesh(4, 4)
        assert mesh.n_cells > 0


# ── HHO ─────────────────────────────────────────────────────────────
class TestHHO:
    def test_import(self):
        from tensornet.mechanics.hho import HHOSolver, HHOState, HHOMesh2D


# ── LES/ILES ────────────────────────────────────────────────────────
class TestILES:
    def test_iles_mode_exists(self):
        from tensornet.cfd.les import LESModel
        assert hasattr(LESModel, "ILES")
        assert LESModel.ILES.value == "iles"


# ── AMR ─────────────────────────────────────────────────────────────
class TestAMR:
    def test_octree_amr(self):
        from tensornet.mesh_amr import OctreeAMR
        amr = OctreeAMR(Lx=1.0, Ly=1.0, Lz=1.0, max_level=3)
        assert amr.max_level == 3

    def test_sfc_load_balancer(self):
        from tensornet.mesh_amr import SFCLoadBalancer
        assert callable(SFCLoadBalancer)
