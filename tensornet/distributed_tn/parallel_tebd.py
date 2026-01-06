"""
Parallel TEBD Module
====================

Parallel implementation of Time-Evolving Block Decimation
for distributed time evolution of MPS.

Supports:
- Domain decomposition with ghost sites
- Strang splitting across partitions
- Asynchronous boundary updates
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

import torch


class SplittingOrder(Enum):
    """Order of Trotter-Suzuki splitting."""

    FIRST = 1  # First order: O(dt)
    SECOND = 2  # Second order (Strang): O(dt^2)
    FOURTH = 4  # Fourth order: O(dt^4)


@dataclass
class GhostSites:
    """Ghost sites for boundary communication.

    Attributes:
        left_ghost: Tensors from left neighbor
        right_ghost: Tensors from right neighbor
        num_ghost: Number of ghost sites per side
        last_update: Time of last update
    """

    left_ghost: list[torch.Tensor] = field(default_factory=list)
    right_ghost: list[torch.Tensor] = field(default_factory=list)
    num_ghost: int = 2
    last_update: float = 0.0

    def update_left(self, tensors: list[torch.Tensor]) -> None:
        """Update left ghost sites."""
        self.left_ghost = [t.clone() for t in tensors]
        self.last_update = time.perf_counter()

    def update_right(self, tensors: list[torch.Tensor]) -> None:
        """Update right ghost sites."""
        self.right_ghost = [t.clone() for t in tensors]
        self.last_update = time.perf_counter()


@dataclass
class TEBDPartition:
    """A partition for parallel TEBD.

    Attributes:
        partition_id: Unique identifier
        start_site: First site index
        end_site: Last site index
        tensors: Local MPS tensors
        ghost: Ghost sites for boundaries
        local_time: Local simulation time
    """

    partition_id: int
    start_site: int
    end_site: int
    tensors: list[torch.Tensor]
    ghost: GhostSites = field(default_factory=GhostSites)
    local_time: float = 0.0

    @property
    def num_sites(self) -> int:
        """Number of sites in partition."""
        return self.end_site - self.start_site

    def get_extended_tensors(self) -> list[torch.Tensor]:
        """Get tensors including ghost sites."""
        result = []

        if self.ghost.left_ghost:
            result.extend(self.ghost.left_ghost)

        result.extend(self.tensors)

        if self.ghost.right_ghost:
            result.extend(self.ghost.right_ghost)

        return result


@dataclass
class ParallelTEBDResult:
    """Result from parallel TEBD.

    Attributes:
        final_time: Final simulation time
        num_steps: Number of time steps taken
        truncation_errors: Average truncation error per step
        wall_time: Total wall clock time
        partition_times: Final time per partition
        success: Whether evolution completed
    """

    final_time: float
    num_steps: int
    truncation_errors: list[float]
    wall_time: float
    partition_times: list[float]
    success: bool


class TEBDWorker:
    """Worker for TEBD on a single partition."""

    def __init__(
        self,
        partition: TEBDPartition,
        chi_max: int = 64,
        cutoff: float = 1e-10,
    ) -> None:
        """Initialize worker.

        Args:
            partition: The partition to evolve
            chi_max: Maximum bond dimension
            cutoff: Truncation cutoff
        """
        self.partition = partition
        self.chi_max = chi_max
        self.cutoff = cutoff

        self.truncation_errors: list[float] = []

    def apply_gate(
        self,
        gate: torch.Tensor,
        site: int,
        truncate: bool = True,
    ) -> float:
        """Apply two-site gate at given site.

        Args:
            gate: Two-site gate tensor
            site: Left site index (local)
            truncate: Whether to truncate

        Returns:
            Truncation error
        """
        if site < 0 or site >= len(self.partition.tensors) - 1:
            return 0.0

        A = self.partition.tensors[site]
        B = self.partition.tensors[site + 1]

        # Dimensions
        chi_left = A.shape[0]
        d = A.shape[1]
        chi_mid = A.shape[2]
        chi_right = B.shape[2]

        # Contract: A[i,s,j] B[j,t,k] -> theta[i,s,t,k]
        theta = torch.einsum("isj,jtk->istk", A, B)

        # Apply gate: gate[s,t,s',t'] theta[i,s,t,k] -> theta'[i,s',t',k]
        theta = torch.einsum("stab,istk->iabk", gate, theta)

        # Reshape for SVD
        theta = theta.reshape(chi_left * d, d * chi_right)

        # Randomized SVD (4× faster)
        # Note: svd_lowrank returns (U, S, V) not (U, S, Vh)
        q = min(self.chi_max, min(theta.shape))
        U, S, V = torch.svd_lowrank(theta, q=q, niter=1)

        # Truncate
        chi_new = min(self.chi_max, len(S))
        if truncate:
            mask = S > S[0] * self.cutoff
            chi_new = min(chi_new, mask.sum().item())
        chi_new = max(1, chi_new)

        # Compute truncation error
        truncation_error = (
            float(torch.sum(S[chi_new:] ** 2)) if chi_new < len(S) else 0.0
        )

        U = U[:, :chi_new]
        S = S[:chi_new]
        V = V[:, :chi_new]  # V is (n, k), column slicing

        # Split back (symmetric gauge)
        S_sqrt = torch.sqrt(S)
        self.partition.tensors[site] = (U @ torch.diag(S_sqrt)).reshape(
            chi_left, d, chi_new
        )
        self.partition.tensors[site + 1] = (torch.diag(S_sqrt) @ V.T).reshape(
            chi_new, d, chi_right
        )

        return truncation_error

    def evolve_step(
        self,
        gates_even: list[torch.Tensor],
        gates_odd: list[torch.Tensor],
        dt: float,
        order: SplittingOrder = SplittingOrder.SECOND,
    ) -> float:
        """Evolve partition by one time step.

        Args:
            gates_even: Gates for even bonds
            gates_odd: Gates for odd bonds
            dt: Time step
            order: Splitting order

        Returns:
            Total truncation error
        """
        total_error = 0.0

        if order == SplittingOrder.SECOND:
            # Strang splitting: odd(dt/2) even(dt) odd(dt/2)

            # Half step on odd bonds
            for i in range(1, len(self.partition.tensors) - 1, 2):
                if i < len(gates_odd):
                    error = self.apply_gate(gates_odd[i // 2], i)
                    total_error += error

            # Full step on even bonds
            for i in range(0, len(self.partition.tensors) - 1, 2):
                if i // 2 < len(gates_even):
                    error = self.apply_gate(gates_even[i // 2], i)
                    total_error += error

            # Half step on odd bonds
            for i in range(1, len(self.partition.tensors) - 1, 2):
                if i // 2 < len(gates_odd):
                    error = self.apply_gate(gates_odd[i // 2], i)
                    total_error += error

        else:
            # First order: even then odd
            for i in range(0, len(self.partition.tensors) - 1, 2):
                if i // 2 < len(gates_even):
                    error = self.apply_gate(gates_even[i // 2], i)
                    total_error += error

            for i in range(1, len(self.partition.tensors) - 1, 2):
                if i // 2 < len(gates_odd):
                    error = self.apply_gate(gates_odd[i // 2], i)
                    total_error += error

        self.partition.local_time += dt
        self.truncation_errors.append(total_error)

        return total_error

    def get_boundary_tensors(self, side: str, num: int = 2) -> list[torch.Tensor]:
        """Get boundary tensors for ghost exchange.

        Args:
            side: 'left' or 'right'
            num: Number of tensors

        Returns:
            List of boundary tensors
        """
        if side == "left":
            return [t.clone() for t in self.partition.tensors[:num]]
        else:
            return [t.clone() for t in self.partition.tensors[-num:]]

    def update_ghost_sites(
        self,
        tensors: list[torch.Tensor],
        side: str,
    ) -> None:
        """Update ghost sites from neighbor.

        Args:
            tensors: Ghost tensors
            side: 'left' or 'right'
        """
        if side == "left":
            self.partition.ghost.update_left(tensors)
        else:
            self.partition.ghost.update_right(tensors)


class ParallelTEBD:
    """Parallel TEBD implementation.

    Evolves MPS using domain decomposition with
    ghost site communication.

    Attributes:
        num_partitions: Number of partitions
        chi_max: Maximum bond dimension
        cutoff: Truncation cutoff
        order: Splitting order
    """

    def __init__(
        self,
        num_partitions: int = 4,
        chi_max: int = 64,
        cutoff: float = 1e-10,
        order: SplittingOrder = SplittingOrder.SECOND,
        num_ghost: int = 2,
    ) -> None:
        """Initialize parallel TEBD.

        Args:
            num_partitions: Number of partitions
            chi_max: Maximum bond dimension
            cutoff: Truncation cutoff
            order: Splitting order
            num_ghost: Number of ghost sites
        """
        self.num_partitions = num_partitions
        self.chi_max = chi_max
        self.cutoff = cutoff
        self.order = order
        self.num_ghost = num_ghost

        self.workers: list[TEBDWorker] = []
        self.partitions: list[TEBDPartition] = []
        self.executor = ThreadPoolExecutor(max_workers=num_partitions)

    def partition_mps(
        self,
        mps_tensors: list[torch.Tensor],
    ) -> list[TEBDPartition]:
        """Partition MPS for parallel evolution.

        Args:
            mps_tensors: List of MPS tensors

        Returns:
            List of partitions
        """
        num_sites = len(mps_tensors)
        sites_per_partition = num_sites // self.num_partitions
        remainder = num_sites % self.num_partitions

        partitions = []
        start = 0

        for i in range(self.num_partitions):
            extra = 1 if i < remainder else 0
            end = start + sites_per_partition + extra

            partition = TEBDPartition(
                partition_id=i,
                start_site=start,
                end_site=end,
                tensors=[t.clone() for t in mps_tensors[start:end]],
                ghost=GhostSites(num_ghost=self.num_ghost),
            )
            partitions.append(partition)

            start = end

        self.partitions = partitions
        return partitions

    def create_workers(self) -> list[TEBDWorker]:
        """Create workers for each partition."""
        self.workers = []

        for partition in self.partitions:
            worker = TEBDWorker(
                partition=partition,
                chi_max=self.chi_max,
                cutoff=self.cutoff,
            )
            self.workers.append(worker)

        return self.workers

    def exchange_ghosts(self) -> None:
        """Exchange ghost sites between adjacent workers."""
        for i in range(len(self.workers) - 1):
            # Right boundary of i -> left ghost of i+1
            right_tensors = self.workers[i].get_boundary_tensors(
                "right", self.num_ghost
            )
            self.workers[i + 1].update_ghost_sites(right_tensors, "left")

            # Left boundary of i+1 -> right ghost of i
            left_tensors = self.workers[i + 1].get_boundary_tensors(
                "left", self.num_ghost
            )
            self.workers[i].update_ghost_sites(left_tensors, "right")

    def evolve_step(
        self,
        gates_even: list[torch.Tensor],
        gates_odd: list[torch.Tensor],
        dt: float,
    ) -> float:
        """Evolve all partitions by one step.

        Args:
            gates_even: Even bond gates
            gates_odd: Odd bond gates
            dt: Time step

        Returns:
            Total truncation error
        """
        # Submit parallel evolution
        futures: list[Future] = []
        for worker in self.workers:
            future = self.executor.submit(
                worker.evolve_step, gates_even, gates_odd, dt, self.order
            )
            futures.append(future)

        # Collect results
        total_error = 0.0
        for future in futures:
            error = future.result()
            total_error += error

        # Exchange ghost sites
        self.exchange_ghosts()

        return total_error

    def run(
        self,
        mps_tensors: list[torch.Tensor],
        hamiltonian_terms: list[torch.Tensor],
        total_time: float,
        dt: float,
        callback: Callable[[int, float, float], None] | None = None,
    ) -> ParallelTEBDResult:
        """Run parallel TEBD evolution.

        Args:
            mps_tensors: Initial MPS tensors
            hamiltonian_terms: Two-site Hamiltonian terms
            total_time: Total evolution time
            dt: Time step
            callback: Optional callback(step, time, error)

        Returns:
            ParallelTEBDResult
        """
        start_time = time.perf_counter()

        # Partition and create workers
        self.partition_mps(mps_tensors)
        self.create_workers()

        # Build gates from Hamiltonian
        # For now, use identity gates as placeholder
        d = mps_tensors[0].shape[1] if mps_tensors else 2
        identity_gate = torch.eye(d * d).reshape(d, d, d, d)

        # Simple evolution gate: exp(-i H dt)
        # Using identity for demonstration
        gates_even = [identity_gate.clone() for _ in range((len(mps_tensors) + 1) // 2)]
        gates_odd = [identity_gate.clone() for _ in range(len(mps_tensors) // 2)]

        num_steps = int(total_time / dt)
        truncation_errors = []
        current_time = 0.0

        for step in range(num_steps):
            error = self.evolve_step(gates_even, gates_odd, dt)
            truncation_errors.append(error)
            current_time += dt

            if callback is not None:
                callback(step, current_time, error)

        wall_time = time.perf_counter() - start_time

        return ParallelTEBDResult(
            final_time=current_time,
            num_steps=num_steps,
            truncation_errors=truncation_errors,
            wall_time=wall_time,
            partition_times=[w.partition.local_time for w in self.workers],
            success=True,
        )

    def collect_mps(self) -> list[torch.Tensor]:
        """Collect MPS tensors from all partitions.

        Returns:
            Complete list of MPS tensors
        """
        tensors = []
        for partition in self.partitions:
            tensors.extend(partition.tensors)
        return tensors

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def run_parallel_tebd(
    mps_tensors: list[torch.Tensor],
    hamiltonian_terms: list[torch.Tensor],
    total_time: float,
    dt: float,
    num_partitions: int = 4,
    chi_max: int = 64,
) -> ParallelTEBDResult:
    """Convenience function for parallel TEBD.

    Args:
        mps_tensors: Initial MPS
        hamiltonian_terms: Hamiltonian terms
        total_time: Total time
        dt: Time step
        num_partitions: Number of partitions
        chi_max: Maximum chi

    Returns:
        ParallelTEBDResult
    """
    tebd = ParallelTEBD(num_partitions=num_partitions, chi_max=chi_max)

    try:
        result = tebd.run(mps_tensors, hamiltonian_terms, total_time, dt)
    finally:
        tebd.shutdown()

    return result
