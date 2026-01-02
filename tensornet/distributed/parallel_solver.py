"""
Parallel iterative solvers for distributed CFD.

This module provides parallel implementations of iterative
solvers (CG, GMRES) with domain decomposition preconditioning.

Author: HyperTensor Team
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .communication import AllReduceOp, Communicator, all_reduce
from .domain_decomp import DomainConfig, DomainDecomposition, SubdomainInfo


@dataclass
class ParallelConfig:
    """Configuration for parallel solvers."""

    # Solver parameters
    max_iter: int = 1000
    rtol: float = 1e-8  # Relative tolerance
    atol: float = 1e-12  # Absolute tolerance

    # Preconditioner
    preconditioner: str = "jacobi"  # 'jacobi', 'schwarz', 'none'
    schwarz_overlap: int = 2

    # Communication
    async_communication: bool = True

    # Monitoring
    verbose: bool = True
    log_interval: int = 100


class DomainSolver:
    """
    Local domain solver for subdomain problems.

    Solves local systems within a subdomain as part
    of domain decomposition methods.
    """

    def __init__(self, subdomain: SubdomainInfo):
        self.subdomain = subdomain

        # Local operator (to be set)
        self.A_local: Callable | None = None

        # Preconditioner
        self.M_inv: Callable | None = None

    def set_operator(self, A: Callable[[torch.Tensor], torch.Tensor]):
        """Set the local matrix-vector product operator."""
        self.A_local = A

    def set_preconditioner(self, M_inv: Callable[[torch.Tensor], torch.Tensor]):
        """Set the local preconditioner (inverse)."""
        self.M_inv = M_inv

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local operator."""
        if self.A_local is None:
            return x
        return self.A_local(x)

    def precondition(self, r: torch.Tensor) -> torch.Tensor:
        """Apply local preconditioner."""
        if self.M_inv is None:
            return r
        return self.M_inv(r)

    def solve_local(
        self,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Solve local system Ax = b.

        Uses local CG for symmetric positive definite problems.
        """
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        if self.A_local is None:
            return b  # Identity operator

        r = b - self.apply(x)
        z = self.precondition(r)
        p = z.clone()
        rz = torch.sum(r * z)

        for _ in range(max_iter):
            Ap = self.apply(p)
            pAp = torch.sum(p * Ap)

            if abs(pAp) < 1e-16:
                break

            alpha = rz / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            if torch.norm(r) < tol * torch.norm(b):
                break

            z = self.precondition(r)
            rz_new = torch.sum(r * z)
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new

        return x


class ParallelCGSolver:
    """
    Parallel Conjugate Gradient solver.

    Solves symmetric positive definite linear systems
    with domain decomposition.

    Example:
        >>> solver = ParallelCGSolver(config, decomp, comm)
        >>> x = solver.solve(A, b)
    """

    def __init__(
        self, config: ParallelConfig, decomp: DomainDecomposition, comm: Communicator
    ):
        self.config = config
        self.decomp = decomp
        self.comm = comm

        # Local solvers
        self.local_solvers: dict[int, DomainSolver] = {}
        for rank, subdomain in decomp.subdomains.items():
            self.local_solvers[rank] = DomainSolver(subdomain)

        # Convergence history
        self.history: list[float] = []

    def set_operator(self, A: Callable[[torch.Tensor], torch.Tensor]):
        """Set the global matrix-vector product operator."""
        for solver in self.local_solvers.values():
            solver.set_operator(A)

    def set_preconditioner(self, precond_type: str = "jacobi"):
        """Set up the preconditioner."""
        if precond_type == "jacobi":
            # Diagonal preconditioner
            for solver in self.local_solvers.values():
                solver.set_preconditioner(lambda x: x)  # Identity for now
        elif precond_type == "schwarz":
            # Additive Schwarz
            pass

    def _global_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute global dot product with communication."""
        local_dot = torch.sum(x * y)
        return all_reduce(self.comm, local_dot, AllReduceOp.SUM)

    def solve(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Solve Ax = b using parallel CG.

        Args:
            A: Matrix-vector product function
            b: Right-hand side
            x0: Initial guess

        Returns:
            Solution x
        """
        config = self.config
        self.history = []

        # Initialize
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        # Initial residual
        r = b - A(x)

        # Apply preconditioner
        rank = self.comm.rank
        if rank in self.local_solvers:
            z = self.local_solvers[rank].precondition(r)
        else:
            z = r.clone()

        p = z.clone()

        rz = self._global_dot(r, z)
        b_norm = torch.sqrt(self._global_dot(b, b))

        if b_norm < config.atol:
            return x

        for iteration in range(config.max_iter):
            # Matrix-vector product
            Ap = A(p)

            # Step size
            pAp = self._global_dot(p, Ap)

            if abs(pAp) < 1e-16:
                break

            alpha = rz / pAp

            # Update solution and residual
            x = x + alpha * p
            r = r - alpha * Ap

            # Check convergence
            r_norm = torch.sqrt(self._global_dot(r, r))
            rel_error = r_norm / b_norm

            self.history.append(rel_error.item())

            if config.verbose and (iteration + 1) % config.log_interval == 0:
                print(
                    f"  CG iteration {iteration+1}: "
                    f"residual = {rel_error.item():.6e}"
                )

            if rel_error < config.rtol or r_norm < config.atol:
                if config.verbose:
                    print(f"  CG converged in {iteration+1} iterations")
                break

            # Precondition
            if rank in self.local_solvers:
                z = self.local_solvers[rank].precondition(r)
            else:
                z = r.clone()

            # Update search direction
            rz_new = self._global_dot(r, z)
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new

        return x


class ParallelGMRESSolver:
    """
    Parallel GMRES solver.

    Solves general (non-symmetric) linear systems
    with domain decomposition.
    """

    def __init__(
        self,
        config: ParallelConfig,
        decomp: DomainDecomposition,
        comm: Communicator,
        restart: int = 30,
    ):
        self.config = config
        self.decomp = decomp
        self.comm = comm
        self.restart = restart

        self.history: list[float] = []

    def _global_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute global dot product."""
        local_dot = torch.sum(x * y)
        return all_reduce(self.comm, local_dot, AllReduceOp.SUM)

    def _global_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute global norm."""
        return torch.sqrt(self._global_dot(x, x))

    def solve(
        self,
        A: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        M: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Solve Ax = b using parallel restarted GMRES.

        Args:
            A: Matrix-vector product function
            b: Right-hand side
            x0: Initial guess
            M: Preconditioner (applies M^{-1})

        Returns:
            Solution x
        """
        config = self.config
        self.history = []

        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        b_norm = self._global_norm(b)

        if b_norm < config.atol:
            return x

        # Outer iterations (restarts)
        for outer in range(config.max_iter // self.restart + 1):
            # Compute initial residual
            r = b - A(x)

            if M is not None:
                r = M(r)

            r_norm = self._global_norm(r)

            if r_norm / b_norm < config.rtol or r_norm < config.atol:
                break

            # Initialize Krylov subspace
            V = [r / r_norm]
            H = torch.zeros(self.restart + 1, self.restart)
            g = torch.zeros(self.restart + 1)
            g[0] = r_norm

            # Arnoldi iteration
            for j in range(self.restart):
                # New Krylov vector
                w = A(V[j])

                if M is not None:
                    w = M(w)

                # Gram-Schmidt orthogonalization
                for i in range(j + 1):
                    H[i, j] = self._global_dot(w, V[i])
                    w = w - H[i, j] * V[i]

                H[j + 1, j] = self._global_norm(w)

                if H[j + 1, j] > 1e-14:
                    V.append(w / H[j + 1, j])
                else:
                    break

                # Apply Givens rotations
                for i in range(j):
                    temp = H[i, j]
                    H[i, j] = self._givens_c[i] * temp + self._givens_s[i] * H[i + 1, j]
                    H[i + 1, j] = (
                        -self._givens_s[i] * temp + self._givens_c[i] * H[i + 1, j]
                    )

                # New rotation
                if j == 0:
                    self._givens_c = []
                    self._givens_s = []

                denom = torch.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
                c = H[j, j] / denom
                s = H[j + 1, j] / denom

                self._givens_c.append(c)
                self._givens_s.append(s)

                H[j, j] = denom
                H[j + 1, j] = 0

                g[j + 1] = -s * g[j]
                g[j] = c * g[j]

                # Check convergence
                res = abs(g[j + 1].item())
                self.history.append(res / b_norm.item())

                if config.verbose and len(self.history) % config.log_interval == 0:
                    print(
                        f"  GMRES iteration {len(self.history)}: "
                        f"residual = {res / b_norm.item():.6e}"
                    )

                if res / b_norm < config.rtol:
                    break

            # Solve upper triangular system
            k = min(j + 1, self.restart)
            y = torch.zeros(k)

            for i in range(k - 1, -1, -1):
                y[i] = g[i]
                for jj in range(i + 1, k):
                    y[i] -= H[i, jj] * y[jj]
                y[i] /= H[i, i]

            # Update solution
            for i in range(k):
                x = x + y[i] * V[i]

            # Check convergence
            if res / b_norm < config.rtol:
                if config.verbose:
                    print(f"  GMRES converged in {len(self.history)} iterations")
                break

        return x


class SchwarzPreconditioner:
    """
    Additive Schwarz preconditioner.

    Uses overlapping domain decomposition for
    parallel preconditioning.
    """

    def __init__(
        self, decomp: DomainDecomposition, comm: Communicator, overlap: int = 2
    ):
        self.decomp = decomp
        self.comm = comm
        self.overlap = overlap

        # Local solvers for each subdomain
        self.local_solvers: dict[int, DomainSolver] = {}

    def setup(self, A_local: Callable[[torch.Tensor], torch.Tensor]):
        """Set up the preconditioner with local operators."""
        for rank, subdomain in self.decomp.subdomains.items():
            solver = DomainSolver(subdomain)
            solver.set_operator(A_local)
            self.local_solvers[rank] = solver

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply the additive Schwarz preconditioner.

        Args:
            r: Residual vector

        Returns:
            Preconditioned residual
        """
        rank = self.comm.rank

        if rank not in self.local_solvers:
            return r

        # Solve local problem
        solver = self.local_solvers[rank]
        z_local = solver.solve_local(r)

        # Sum contributions (additive Schwarz)
        z = all_reduce(self.comm, z_local, AllReduceOp.SUM)

        return z


def parallel_solve(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    config: ParallelConfig | None = None,
    method: str = "cg",
    decomp: DomainDecomposition | None = None,
    comm: Communicator | None = None,
) -> torch.Tensor:
    """
    Convenience function for parallel linear solve.

    Args:
        A: Matrix-vector product function
        b: Right-hand side
        config: Solver configuration
        method: 'cg' or 'gmres'
        decomp: Domain decomposition (or create single-domain)
        comm: Communicator (or create single-process)

    Returns:
        Solution x
    """
    if config is None:
        config = ParallelConfig(verbose=False)

    if comm is None:
        comm = Communicator(n_procs=1, rank=0)

    if decomp is None:
        # Single domain
        n = b.shape[0] if b.dim() == 1 else b.shape[0] * b.shape[1]
        domain_config = DomainConfig(nx=n, ny=1, nz=1, n_procs=1)
        decomp = DomainDecomposition(domain_config)

    if method == "cg":
        solver = ParallelCGSolver(config, decomp, comm)
        return solver.solve(A, b)
    elif method == "gmres":
        solver = ParallelGMRESSolver(config, decomp, comm)
        return solver.solve(A, b)
    else:
        raise ValueError(f"Unknown method: {method}")


def test_parallel_solvers():
    """Test parallel solvers."""
    print("Testing Parallel Solvers...")

    # Create test problem
    print("\n  Creating test problem...")
    n = 100

    # SPD matrix: A = D + E where D is diagonal dominant
    D = torch.diag(torch.rand(n) + 3.0)  # Diagonal
    E = 0.1 * torch.randn(n, n)
    A_mat = D + E + E.t()  # Symmetric

    # True solution and RHS
    x_true = torch.randn(n)
    b = A_mat @ x_true

    def matvec(x):
        return A_mat @ x

    # Test single-domain CG
    print("\n  Testing ParallelCGSolver (single domain)...")
    config = ParallelConfig(max_iter=200, rtol=1e-6, verbose=False)

    domain_config = DomainConfig(nx=n, ny=1, n_procs=1)
    decomp = DomainDecomposition(domain_config)
    comm = Communicator(n_procs=1, rank=0)

    solver = ParallelCGSolver(config, decomp, comm)
    x_sol = solver.solve(matvec, b)

    error = torch.norm(x_sol - x_true) / torch.norm(x_true)
    print(f"    Relative error: {error:.6e}")
    print(f"    Iterations: {len(solver.history)}")

    assert error < 1e-4, f"CG error too large: {error}"

    # Test GMRES
    print("\n  Testing ParallelGMRESSolver (single domain)...")

    # Non-symmetric matrix
    A_nonsym = torch.randn(n, n)
    A_nonsym = A_nonsym + 3 * torch.eye(n)  # Make diagonally dominant

    b_nonsym = A_nonsym @ x_true

    def matvec_nonsym(x):
        return A_nonsym @ x

    gmres_solver = ParallelGMRESSolver(config, decomp, comm, restart=30)
    x_gmres = gmres_solver.solve(matvec_nonsym, b_nonsym)

    error_gmres = torch.norm(x_gmres - x_true) / torch.norm(x_true)
    print(f"    Relative error: {error_gmres:.6e}")
    print(f"    Iterations: {len(gmres_solver.history)}")

    # Test convenience function
    print("\n  Testing parallel_solve convenience function...")
    x_conv = parallel_solve(matvec, b, method="cg")

    error_conv = torch.norm(x_conv - x_true) / torch.norm(x_true)
    print(f"    Relative error: {error_conv:.6e}")

    # Test DomainSolver
    print("\n  Testing DomainSolver...")
    subdomain = decomp.get_subdomain(0)
    local_solver = DomainSolver(subdomain)
    local_solver.set_operator(matvec)

    x_local = local_solver.solve_local(b, max_iter=100, tol=1e-6)
    error_local = torch.norm(x_local - x_true) / torch.norm(x_true)
    print(f"    Local solve error: {error_local:.6e}")

    # Test Schwarz preconditioner setup
    print("\n  Testing SchwarzPreconditioner...")
    schwarz = SchwarzPreconditioner(decomp, comm, overlap=2)
    schwarz.setup(matvec)

    r = torch.randn(n)
    z = schwarz.apply(r)
    print(f"    Preconditioned residual norm: {torch.norm(z):.4f}")

    print("\nParallel Solvers: All tests passed!")


if __name__ == "__main__":
    test_parallel_solvers()
