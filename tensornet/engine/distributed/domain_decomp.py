"""
Domain decomposition for parallel CFD simulations.

This module implements domain decomposition strategies for
distributing CFD grids across multiple processors/GPUs.

Author: HyperTensor Team
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch


class DecompType(Enum):
    """Domain decomposition type."""

    SLAB = auto()  # 1D slabs
    PENCIL = auto()  # 2D pencils
    BLOCK = auto()  # 3D blocks
    HILBERT = auto()  # Space-filling curve


@dataclass
class DomainConfig:
    """Configuration for domain decomposition."""

    # Global domain
    nx: int = 64
    ny: int = 64
    nz: int = 1

    # Decomposition
    decomp_type: DecompType = DecompType.BLOCK
    n_procs: int = 4  # Number of processors

    # Ghost zones
    n_ghost: int = 2  # Ghost cells per boundary

    # Periodicity
    periodic_x: bool = False
    periodic_y: bool = False
    periodic_z: bool = False

    # Load balancing
    load_balance: bool = True

    # Physical domain
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0


@dataclass
class SubdomainInfo:
    """Information about a subdomain."""

    rank: int

    # Global indices (excluding ghost)
    i_start: int
    i_end: int
    j_start: int
    j_end: int
    k_start: int
    k_end: int

    # Local size (including ghost)
    local_nx: int
    local_ny: int
    local_nz: int

    # Neighbors
    neighbor_left: int | None = None
    neighbor_right: int | None = None
    neighbor_bottom: int | None = None
    neighbor_top: int | None = None
    neighbor_back: int | None = None
    neighbor_front: int | None = None

    # Physical coordinates
    x_local: torch.Tensor | None = None
    y_local: torch.Tensor | None = None
    z_local: torch.Tensor | None = None


class DomainDecomposition:
    """
    Domain decomposition manager.

    Handles partitioning of CFD grids across multiple
    processors and management of ghost zones.

    Example:
        >>> config = DomainConfig(nx=128, ny=128, n_procs=4)
        >>> decomp = DomainDecomposition(config)
        >>> subdomain = decomp.get_subdomain(rank=0)
    """

    def __init__(self, config: DomainConfig):
        self.config = config

        # Compute processor grid
        self.proc_dims = self._compute_proc_dims()

        # Create subdomains
        self.subdomains: dict[int, SubdomainInfo] = {}
        self._create_subdomains()

    def _compute_proc_dims(self) -> tuple[int, int, int]:
        """Compute processor grid dimensions."""
        config = self.config
        n_procs = config.n_procs

        if config.nz == 1:
            # 2D decomposition - find factors close to square root
            nx_procs = int(np.sqrt(n_procs))
            while n_procs % nx_procs != 0:
                nx_procs -= 1
            ny_procs = n_procs // nx_procs
            return (nx_procs, ny_procs, 1)
        else:
            # 3D decomposition - prefer cubic decomposition
            nz_procs = int(np.cbrt(n_procs))
            while n_procs % nz_procs != 0:
                nz_procs -= 1

            remaining = n_procs // nz_procs
            nx_procs = int(np.sqrt(remaining))
            while remaining % nx_procs != 0:
                nx_procs -= 1
            ny_procs = remaining // nx_procs

            return (nx_procs, ny_procs, nz_procs)

    def _get_neighbor_rank(
        self, pi: int, pj: int, pk: int, di: int, dj: int, dk: int
    ) -> int | None:
        """
        Compute neighbor rank with periodic boundary handling.

        Args:
            pi, pj, pk: Current processor indices
            di, dj, dk: Direction offsets (-1, 0, or 1)

        Returns:
            Neighbor rank or None if no neighbor
        """
        config = self.config
        px, py, pz = self.proc_dims

        ni, nj, nk = pi + di, pj + dj, pk + dk

        # X direction
        if config.periodic_x:
            ni = ni % px
        elif ni < 0 or ni >= px:
            return None

        # Y direction
        if config.periodic_y:
            nj = nj % py
        elif nj < 0 or nj >= py:
            return None

        # Z direction
        if pz > 1:
            if config.periodic_z:
                nk = nk % pz
            elif nk < 0 or nk >= pz:
                return None

        return ni * py * pz + nj * pz + nk

    def _compute_ghost_cells(
        self, pi: int, pj: int, pk: int
    ) -> tuple[int, int, int, int, int, int]:
        """
        Compute ghost cell counts for each boundary.

        Returns:
            Tuple of (left, right, bottom, top, back, front) ghost counts
        """
        config = self.config
        px, py, pz = self.proc_dims
        n_ghost = config.n_ghost

        ghost_left = n_ghost if (pi > 0 or config.periodic_x) else 0
        ghost_right = n_ghost if (pi < px - 1 or config.periodic_x) else 0
        ghost_bottom = n_ghost if (pj > 0 or config.periodic_y) else 0
        ghost_top = n_ghost if (pj < py - 1 or config.periodic_y) else 0
        ghost_back = n_ghost if (pk > 0 or config.periodic_z) and pz > 1 else 0
        ghost_front = n_ghost if (pk < pz - 1 or config.periodic_z) and pz > 1 else 0

        return ghost_left, ghost_right, ghost_bottom, ghost_top, ghost_back, ghost_front

    def _create_subdomain(
        self,
        rank: int,
        pi: int,
        pj: int,
        pk: int,
        i_offset: int,
        j_offset: int,
        k_offset: int,
        local_nx: int,
        local_ny: int,
        local_nz: int,
    ) -> SubdomainInfo:
        """Create a single subdomain with all computed properties."""
        config = self.config
        px, py, pz = self.proc_dims

        # Get ghost cell counts
        ghosts = self._compute_ghost_cells(pi, pj, pk)
        ghost_left, ghost_right, ghost_bottom, ghost_top, ghost_back, ghost_front = (
            ghosts
        )

        # Create subdomain
        subdomain = SubdomainInfo(
            rank=rank,
            i_start=i_offset,
            i_end=i_offset + local_nx,
            j_start=j_offset,
            j_end=j_offset + local_ny,
            k_start=k_offset,
            k_end=k_offset + local_nz,
            local_nx=local_nx + ghost_left + ghost_right,
            local_ny=local_ny + ghost_bottom + ghost_top,
            local_nz=local_nz + ghost_back + ghost_front if pz > 1 else 1,
            neighbor_left=self._get_neighbor_rank(pi, pj, pk, -1, 0, 0),
            neighbor_right=self._get_neighbor_rank(pi, pj, pk, 1, 0, 0),
            neighbor_bottom=self._get_neighbor_rank(pi, pj, pk, 0, -1, 0),
            neighbor_top=self._get_neighbor_rank(pi, pj, pk, 0, 1, 0),
            neighbor_back=self._get_neighbor_rank(pi, pj, pk, 0, 0, -1),
            neighbor_front=self._get_neighbor_rank(pi, pj, pk, 0, 0, 1),
        )

        # Create local coordinates
        subdomain.x_local = self._create_local_coords(
            i_offset,
            local_nx,
            ghost_left,
            ghost_right,
            config.x_min,
            config.x_max,
            config.nx,
        )
        subdomain.y_local = self._create_local_coords(
            j_offset,
            local_ny,
            ghost_bottom,
            ghost_top,
            config.y_min,
            config.y_max,
            config.ny,
        )
        if pz > 1:
            subdomain.z_local = self._create_local_coords(
                k_offset,
                local_nz,
                ghost_back,
                ghost_front,
                config.z_min,
                config.z_max,
                config.nz,
            )

        return subdomain

    def _create_subdomains(self):
        """Create subdomain information for all processors."""
        config = self.config
        px, py, pz = self.proc_dims

        # Compute base sizes and remainders for load balancing
        base_nx, rem_nx = divmod(config.nx, px)
        base_ny, rem_ny = divmod(config.ny, py)
        base_nz = config.nz // pz if config.nz > 1 else 1
        rem_nz = config.nz % pz if config.nz > 1 else 0

        rank = 0
        i_offset = 0

        for pi in range(px):
            local_nx = base_nx + (1 if pi < rem_nx else 0)
            j_offset = 0

            for pj in range(py):
                local_ny = base_ny + (1 if pj < rem_ny else 0)
                k_offset = 0

                for pk in range(pz):
                    local_nz = base_nz + (1 if pk < rem_nz else 0)

                    self.subdomains[rank] = self._create_subdomain(
                        rank,
                        pi,
                        pj,
                        pk,
                        i_offset,
                        j_offset,
                        k_offset,
                        local_nx,
                        local_ny,
                        local_nz,
                    )

                    rank += 1
                    k_offset += local_nz

                j_offset += local_ny
            i_offset += local_nx

    def _create_local_coords(
        self,
        offset: int,
        size: int,
        ghost_left: int,
        ghost_right: int,
        x_min: float,
        x_max: float,
        n_global: int,
    ) -> torch.Tensor:
        """Create local coordinate array including ghost zones."""
        dx = (x_max - x_min) / n_global

        i_start = offset - ghost_left
        i_end = offset + size + ghost_right

        return torch.linspace(
            x_min + (i_start + 0.5) * dx,
            x_min + (i_end - 0.5) * dx,
            size + ghost_left + ghost_right,
        )

    def get_subdomain(self, rank: int) -> SubdomainInfo:
        """Get subdomain information for a processor."""
        return self.subdomains[rank]

    def get_neighbors(self, rank: int) -> dict[str, int | None]:
        """Get neighbor ranks for a processor."""
        sub = self.subdomains[rank]
        return {
            "left": sub.neighbor_left,
            "right": sub.neighbor_right,
            "bottom": sub.neighbor_bottom,
            "top": sub.neighbor_top,
            "back": sub.neighbor_back,
            "front": sub.neighbor_front,
        }

    def global_to_local(
        self, rank: int, i: int, j: int, k: int = 0
    ) -> tuple[int, int, int]:
        """Convert global index to local index."""
        sub = self.subdomains[rank]
        n_ghost = self.config.n_ghost

        i_local = i - sub.i_start + n_ghost
        j_local = j - sub.j_start + n_ghost
        k_local = k - sub.k_start + (n_ghost if self.config.nz > 1 else 0)

        return (i_local, j_local, k_local)

    def local_to_global(
        self, rank: int, i: int, j: int, k: int = 0
    ) -> tuple[int, int, int]:
        """Convert local index to global index."""
        sub = self.subdomains[rank]
        n_ghost = self.config.n_ghost

        i_global = i - n_ghost + sub.i_start
        j_global = j - n_ghost + sub.j_start
        k_global = k - (n_ghost if self.config.nz > 1 else 0) + sub.k_start

        return (i_global, j_global, k_global)


def decompose_domain(config: DomainConfig) -> DomainDecomposition:
    """
    Convenience function to create domain decomposition.

    Args:
        config: Domain configuration

    Returns:
        Domain decomposition object
    """
    return DomainDecomposition(config)


def compute_ghost_zones(
    data: torch.Tensor, subdomain: SubdomainInfo, n_ghost: int
) -> dict[str, torch.Tensor]:
    """
    Extract ghost zone data to send to neighbors.

    Args:
        data: Local field data [ny, nx] or [nz, ny, nx]
        subdomain: Subdomain information
        n_ghost: Number of ghost cells

    Returns:
        Dictionary of ghost zone data per direction
    """
    ghost_data = {}

    if data.dim() == 2:
        ny, nx = data.shape

        # Left boundary (send to left neighbor)
        if subdomain.neighbor_left is not None:
            ghost_data["left"] = data[:, n_ghost : 2 * n_ghost].clone()

        # Right boundary
        if subdomain.neighbor_right is not None:
            ghost_data["right"] = data[:, -2 * n_ghost : -n_ghost].clone()

        # Bottom boundary
        if subdomain.neighbor_bottom is not None:
            ghost_data["bottom"] = data[n_ghost : 2 * n_ghost, :].clone()

        # Top boundary
        if subdomain.neighbor_top is not None:
            ghost_data["top"] = data[-2 * n_ghost : -n_ghost, :].clone()

    elif data.dim() == 3:
        nz, ny, nx = data.shape

        # X-direction
        if subdomain.neighbor_left is not None:
            ghost_data["left"] = data[:, :, n_ghost : 2 * n_ghost].clone()
        if subdomain.neighbor_right is not None:
            ghost_data["right"] = data[:, :, -2 * n_ghost : -n_ghost].clone()

        # Y-direction
        if subdomain.neighbor_bottom is not None:
            ghost_data["bottom"] = data[:, n_ghost : 2 * n_ghost, :].clone()
        if subdomain.neighbor_top is not None:
            ghost_data["top"] = data[:, -2 * n_ghost : -n_ghost, :].clone()

        # Z-direction
        if subdomain.neighbor_back is not None:
            ghost_data["back"] = data[n_ghost : 2 * n_ghost, :, :].clone()
        if subdomain.neighbor_front is not None:
            ghost_data["front"] = data[-2 * n_ghost : -n_ghost, :, :].clone()

    return ghost_data


def exchange_ghost_data(
    data: torch.Tensor,
    subdomain: SubdomainInfo,
    received: dict[str, torch.Tensor],
    n_ghost: int,
) -> torch.Tensor:
    """
    Fill ghost zones with received data.

    Args:
        data: Local field data
        subdomain: Subdomain information
        received: Dictionary of received ghost data
        n_ghost: Number of ghost cells

    Returns:
        Updated data with filled ghost zones
    """
    result = data.clone()

    if data.dim() == 2:
        # Left ghost zone
        if "left" in received:
            result[:, :n_ghost] = received["left"]

        # Right ghost zone
        if "right" in received:
            result[:, -n_ghost:] = received["right"]

        # Bottom ghost zone
        if "bottom" in received:
            result[:n_ghost, :] = received["bottom"]

        # Top ghost zone
        if "top" in received:
            result[-n_ghost:, :] = received["top"]

    elif data.dim() == 3:
        # X-direction
        if "left" in received:
            result[:, :, :n_ghost] = received["left"]
        if "right" in received:
            result[:, :, -n_ghost:] = received["right"]

        # Y-direction
        if "bottom" in received:
            result[:, :n_ghost, :] = received["bottom"]
        if "top" in received:
            result[:, -n_ghost:, :] = received["top"]

        # Z-direction
        if "back" in received:
            result[:n_ghost, :, :] = received["back"]
        if "front" in received:
            result[-n_ghost:, :, :] = received["front"]

    return result


def test_domain_decomposition():
    """Test domain decomposition."""
    print("Testing Domain Decomposition...")

    # Test 2D decomposition
    print("\n  Testing 2D decomposition (4 processors)...")
    config = DomainConfig(nx=64, ny=64, nz=1, n_procs=4, n_ghost=2)
    decomp = DomainDecomposition(config)

    print(f"    Processor grid: {decomp.proc_dims}")

    for rank in range(4):
        sub = decomp.get_subdomain(rank)
        print(
            f"    Rank {rank}: global[{sub.i_start}:{sub.i_end}, {sub.j_start}:{sub.j_end}] "
            f"local[{sub.local_nx}, {sub.local_ny}]"
        )

    # Test neighbor connectivity
    print("\n  Testing neighbor connectivity...")
    for rank in range(4):
        neighbors = decomp.get_neighbors(rank)
        print(
            f"    Rank {rank}: L={neighbors['left']} R={neighbors['right']} "
            f"B={neighbors['bottom']} T={neighbors['top']}"
        )

    # Test periodic boundaries
    print("\n  Testing periodic boundaries...")
    config_periodic = DomainConfig(
        nx=32, ny=32, n_procs=4, periodic_x=True, periodic_y=True
    )
    decomp_periodic = DomainDecomposition(config_periodic)

    # All ranks should have 4 neighbors in periodic case
    for rank in range(4):
        neighbors = decomp_periodic.get_neighbors(rank)
        n_neighbors = sum(1 for v in neighbors.values() if v is not None)
        assert n_neighbors == 4, f"Rank {rank} should have 4 neighbors"
    print("    Periodic connectivity verified")

    # Test ghost zone extraction
    print("\n  Testing ghost zone operations...")
    sub = decomp.get_subdomain(0)
    data = torch.randn(sub.local_ny, sub.local_nx)

    ghost_data = compute_ghost_zones(data, sub, config.n_ghost)
    print(f"    Extracted ghost zones: {list(ghost_data.keys())}")

    # Simulate receiving ghost data
    received = {}
    if sub.neighbor_right is not None:
        received["right"] = torch.ones(sub.local_ny, config.n_ghost)
    if sub.neighbor_top is not None:
        received["top"] = torch.ones(config.n_ghost, sub.local_nx)

    data_updated = exchange_ghost_data(data, sub, received, config.n_ghost)
    print("    Ghost exchange completed")

    # Test coordinate generation
    print("\n  Testing coordinate generation...")
    assert sub.x_local is not None
    assert len(sub.x_local) == sub.local_nx
    print(f"    X coordinates: [{sub.x_local[0]:.4f}, {sub.x_local[-1]:.4f}]")

    # Test 3D decomposition
    print("\n  Testing 3D decomposition (8 processors)...")
    config_3d = DomainConfig(nx=32, ny=32, nz=32, n_procs=8, n_ghost=2)
    decomp_3d = DomainDecomposition(config_3d)

    print(f"    Processor grid: {decomp_3d.proc_dims}")

    # Verify total cells match
    total_cells = 0
    for rank in range(8):
        sub = decomp_3d.get_subdomain(rank)
        # Subtract ghost cells
        cells = (
            (sub.i_end - sub.i_start)
            * (sub.j_end - sub.j_start)
            * (sub.k_end - sub.k_start)
        )
        total_cells += cells

    assert total_cells == config_3d.nx * config_3d.ny * config_3d.nz
    print(f"    Total cells match: {total_cells}")

    print("\nDomain Decomposition: All tests passed!")


if __name__ == "__main__":
    test_domain_decomposition()
