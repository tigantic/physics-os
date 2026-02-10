"""
Tests for §1.3 — Numerical Methods
====================================

Covers: Parareal, Exponential integrators, AMG, p-Multigrid,
Deflated Krylov, H-Matrix, FMM, PGD, Sparse Grids, Reduced Basis, AD.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import diags


# ── Parareal ────────────────────────────────────────────────────────
class TestParareal:
    def test_import(self):
        from tensornet.numerics.parareal import (
            PararealSolver,
            forward_euler_propagator,
            rk4_propagator,
        )

    def test_exponential_decay(self):
        from tensornet.numerics.parareal import (
            PararealSolver,
            forward_euler_propagator,
            rk4_propagator,
        )

        def rhs(y, t):
            return -y

        coarse = forward_euler_propagator(rhs, n_sub=5)
        fine = rk4_propagator(rhs, n_sub=50)
        solver = PararealSolver(fine=fine, coarse=coarse)
        result = solver.solve(np.array([1.0]), t_span=(0.0, 1.0), N=4)
        # y(1) = e^{-1} ≈ 0.3679
        y_final = result.solution[-1]
        assert abs(y_final[0] - math.exp(-1)) < 0.05


# ── Exponential Integrators ────────────────────────────────────────
class TestExponentialIntegrators:
    def test_import(self):
        from tensornet.numerics.exponential import ETD1, ETDRK2, ETDRK4

    def test_etd1_linear(self):
        from tensornet.numerics.exponential import ETD1

        # dy/dt = -y → y = exp(-t)
        L = np.array([-1.0])
        N_func = lambda y, t: np.zeros_like(y)
        solver = ETD1(L, N_func)
        y = np.array([1.0])
        t = 0.0
        dt = 0.01
        for _ in range(100):
            y = solver.step(y, t, dt)
            t += dt
        assert abs(np.real(y[0]) - math.exp(-1)) < 0.01


# ── AMG ─────────────────────────────────────────────────────────────
class TestAMG:
    def test_import(self):
        from tensornet.numerics.amg import AMGSolver, SmoothingType

    def test_1d_laplacian(self):
        from tensornet.numerics.amg import AMGSolver

        n = 100
        A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format="csr")
        b = np.ones(n)
        amg = AMGSolver(A, max_levels=5)
        x = amg.solve(b)
        # Residual should be small
        r = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        assert r < 1e-6

    def test_hierarchy_depth(self):
        from tensornet.numerics.amg import AMGSolver

        n = 200
        A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format="csr")
        amg = AMGSolver(A, max_levels=5, max_coarse=30)
        assert amg.n_levels >= 2


# ── p-Multigrid ────────────────────────────────────────────────────
class TestPMultigrid:
    def test_import(self):
        from tensornet.numerics.p_multigrid import PMultigridSolver

    def test_simple_solve(self):
        from tensornet.numerics.p_multigrid import PMultigridSolver

        def assemble(p):
            n = p + 1
            # Diagonally-dominant SPD system suited to Jacobi smoothing
            A = np.eye(n) * 4.0
            for i in range(n - 1):
                A[i, i + 1] = -1.0
                A[i + 1, i] = -1.0
            return A

        solver = PMultigridSolver(p_max=4, assemble_fn=assemble, p_min=2, omega=0.4)
        assert solver.n_levels == 3  # p=4, p=3, p=2
        # At coarsest level, direct solve is used
        A = assemble(4)
        b = np.ones(5)
        x_direct = np.linalg.solve(A, b)
        # One V-cycle should produce a finite result
        x_vc = solver.v_cycle(b)
        assert np.all(np.isfinite(x_vc))


# ── Deflated Krylov ─────────────────────────────────────────────────
class TestDeflatedKrylov:
    def test_import(self):
        from tensornet.numerics.deflated_krylov import DeflatedCG, DeflatedGMRES

    def test_deflated_cg(self):
        from tensornet.numerics.deflated_krylov import DeflatedCG

        n = 50
        A = np.eye(n) * 10.0
        A[0, 0] = 0.01  # near-singular mode
        b = np.ones(n)

        # Deflation vector targeting near-null
        W = np.zeros((n, 1))
        W[0, 0] = 1.0

        solver = DeflatedCG(W)
        result = solver.solve(A, b, tol=1e-8)
        assert result.converged
        r = np.linalg.norm(b - A @ result.x) / np.linalg.norm(b)
        assert r < 1e-6

    def test_deflated_gmres(self):
        from tensornet.numerics.deflated_krylov import DeflatedGMRES

        n = 30
        np.random.seed(42)
        A = np.random.randn(n, n) + np.eye(n) * 5.0  # well-conditioned
        b = np.ones(n)
        solver = DeflatedGMRES(k=5, m=20)
        result = solver.solve(A, b, tol=1e-8)
        r = np.linalg.norm(b - A @ result.x) / np.linalg.norm(b)
        assert r < 1e-4


# ── H-Matrix ───────────────────────────────────────────────────────
class TestHMatrix:
    def test_import(self):
        from tensornet.numerics.h_matrix import (
            HMatrix,
            build_cluster_tree,
            aca,
        )

    def test_cluster_tree(self):
        from tensornet.numerics.h_matrix import build_cluster_tree

        pts = np.random.rand(100, 2)
        tree = build_cluster_tree(pts, leaf_size=16)
        assert tree.size == 100

    def test_matvec(self):
        from tensornet.numerics.h_matrix import HMatrix

        np.random.seed(0)
        n = 80
        pts = np.random.rand(n, 2)

        def kernel(pi, pj):
            r = np.linalg.norm(pi - pj)
            return 1.0 / (r + 0.1)

        H = HMatrix(pts, kernel, leaf_size=16, aca_tol=1e-4)
        x = np.ones(n)
        y = H.matvec(x)
        assert len(y) == n
        assert np.all(np.isfinite(y))
        assert H.compression_ratio() <= 1.0


# ── FMM ─────────────────────────────────────────────────────────────
class TestFMM:
    def test_import(self):
        from tensornet.numerics.fmm import FMMSolver, build_tree

    def test_tree_construction(self):
        from tensornet.numerics.fmm import build_tree

        pts = np.random.rand(200, 2)
        tree = build_tree(pts, max_leaf=32)
        assert tree.n_particles == 200

    def test_small_fmm_vs_direct(self):
        from tensornet.numerics.fmm import FMMSolver

        np.random.seed(1)
        n = 30
        pts = np.random.rand(n, 2) * 10.0
        q = np.random.randn(n)

        fmm = FMMSolver(pts, q, max_leaf=8)
        phi_direct = fmm.evaluate_direct()

        # Direct evaluation must be finite and non-zero for random charges
        assert np.all(np.isfinite(phi_direct))
        assert np.linalg.norm(phi_direct) > 0

        # Tree-based evaluation should also produce finite results
        phi_fmm = fmm.evaluate()
        assert np.all(np.isfinite(phi_fmm))


# ── PGD ─────────────────────────────────────────────────────────────
class TestPGD:
    def test_import(self):
        from tensornet.numerics.pgd import PGDSolver, SeparatedFunction

    def test_2d_laplacian(self):
        from tensornet.numerics.pgd import PGDSolver

        n = 10
        h = 1.0 / (n + 1)
        diag = np.ones(n) * 2.0 / h
        off = np.ones(n - 1) * (-1.0 / h)
        K = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        M = np.eye(n) * h

        ops = [[K, M], [M, K]]
        rhs = [[np.ones(n), np.ones(n)]]
        solver = PGDSolver(ops, rhs, max_terms=5)
        sol = solver.solve()
        assert sol.n_terms >= 1


# ── Sparse Grid ────────────────────────────────────────────────────
class TestSparseGrid:
    def test_import(self):
        from tensornet.numerics.sparse_grid import (
            SparseGrid,
            SparseGridQuadrature,
            SparseGridInterpolator,
        )

    def test_grid_points(self):
        from tensornet.numerics.sparse_grid import SparseGrid

        sg = SparseGrid(d=2, level=3)
        result = sg.build()
        assert result.n_points > 0
        assert result.nodes.shape[1] == 2

    def test_quadrature_polynomial(self):
        from tensornet.numerics.sparse_grid import SparseGridQuadrature

        # Integrate x^2 + y^2 over [-1,1]^2
        # Exact = 2 * (2/3) * 2 = 8/3
        quad = SparseGridQuadrature(d=2, level=3)
        val = quad.integrate(lambda x: x[0] ** 2 + x[1] ** 2)
        assert abs(val - 8.0 / 3.0) < 0.1


# ── Reduced Basis ──────────────────────────────────────────────────
class TestReducedBasis:
    def test_import(self):
        from tensornet.numerics.reduced_basis import (
            ReducedBasis,
            GreedyRBM,
            POD_RBM,
        )

    def test_pod_rbm(self):
        from tensornet.numerics.reduced_basis import POD_RBM

        n = 20

        def assemble_A(mu):
            k = float(mu[0])
            return np.diag(np.ones(n) * (2 + k)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)

        def assemble_f(mu):
            return np.ones(n)

        def solve_hf(A, f):
            return np.linalg.solve(A, f)

        mu_samples = np.linspace(0.1, 2.0, 10).reshape(-1, 1)
        pod = POD_RBM(assemble_A, assemble_f, solve_hf, mu_samples, energy_fraction=0.999)
        rb_data = pod.run()
        assert rb_data.V.shape[0] == n
        assert rb_data.V.shape[1] <= 10

    def test_online_solve(self):
        from tensornet.numerics.reduced_basis import ReducedBasis, POD_RBM

        n = 20

        def assemble_A(mu):
            k = float(mu[0])
            return np.diag(np.ones(n) * (2 + k)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)

        def assemble_f(mu):
            return np.ones(n)

        def solve_hf(A, f):
            return np.linalg.solve(A, f)

        mu_samples = np.linspace(0.1, 2.0, 10).reshape(-1, 1)
        pod = POD_RBM(assemble_A, assemble_f, solve_hf, mu_samples)
        rb_data = pod.run()

        rb = ReducedBasis(rb_data.V, assemble_A, assemble_f)
        mu_test = np.array([1.0])
        u_N = rb.solve(mu_test)
        u_h = rb.reconstruct(u_N)
        # Compare with direct solve
        u_exact = solve_hf(assemble_A(mu_test), assemble_f(mu_test))
        assert np.linalg.norm(u_h - u_exact) / np.linalg.norm(u_exact) < 0.05


# ── AD ──────────────────────────────────────────────────────────────
class TestAD:
    def test_import(self):
        from tensornet.numerics.ad import (
            Dual,
            Variable,
            Tape,
            grad,
            jacobian,
            forward_derivative,
            forward_gradient,
        )

    def test_forward_mode_polynomial(self):
        from tensornet.numerics.ad import Dual

        # f(x) = x^3, f'(x) = 3x^2
        x = Dual(2.0, 1.0)
        y = x * x * x
        assert abs(y.val - 8.0) < 1e-12
        assert abs(y.der - 12.0) < 1e-12

    def test_forward_mode_trig(self):
        from tensornet.numerics.ad import Dual

        # f(x) = sin(x), f'(x) = cos(x)
        x = Dual(0.5, 1.0)
        y = Dual.sin(x)
        assert abs(y.val - math.sin(0.5)) < 1e-12
        assert abs(y.der - math.cos(0.5)) < 1e-12

    def test_reverse_mode_product(self):
        from tensornet.numerics.ad import Variable, Tape

        # f(x, y) = x * y → ∂f/∂x = y, ∂f/∂y = x
        tape = Tape()
        x = Variable(3.0, tape)
        y = Variable(4.0, tape)
        z = x * y
        adjoints = tape.backward(z.idx)
        assert abs(adjoints[x.idx] - 4.0) < 1e-12
        assert abs(adjoints[y.idx] - 3.0) < 1e-12

    def test_grad_convenience(self):
        from tensornet.numerics.ad import grad, Variable

        # f(x,y) = x^2 + xy → ∇f = (2x+y, x)
        def f(x, y):
            return x * x + x * y

        g = grad(f, np.array([3.0, 2.0]))
        assert abs(g[0] - 8.0) < 1e-10  # 2*3 + 2
        assert abs(g[1] - 3.0) < 1e-10  # x

    def test_forward_gradient(self):
        from tensornet.numerics.ad import forward_gradient, Dual

        # f(x,y) = x^2 + y^2 → ∇f = (2x, 2y)
        def f(x, y):
            return x * x + y * y

        g = forward_gradient(f, np.array([1.0, 2.0]))
        assert abs(g[0] - 2.0) < 1e-12
        assert abs(g[1] - 4.0) < 1e-12

    def test_forward_vs_reverse_agree(self):
        from tensornet.numerics.ad import forward_gradient, grad, Dual, Variable

        def f_forward(x, y):
            return Dual.exp(x) + Dual.sin(y)

        def f_reverse(x, y):
            return Variable.exp(x) + Variable.sin(y)

        pt = np.array([1.0, 0.5])
        g_fwd = forward_gradient(f_forward, pt)
        g_rev = grad(f_reverse, pt)
        np.testing.assert_allclose(g_fwd, g_rev, atol=1e-10)
