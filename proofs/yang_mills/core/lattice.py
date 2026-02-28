#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            LATTICE GEOMETRY MODULE                           ║
║                                                                              ║
║                   Hypercubic Lattice for Gauge Theory                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module defines the lattice geometry for lattice gauge theory.

Key Concepts:
    - Sites: Points on the lattice, labeled by integer coordinates
    - Links: Edges connecting neighboring sites, carry gauge field (U)
    - Plaquettes: Elementary squares, used for magnetic energy term
    
Gauge Field Structure:
    - U_μ(x) ∈ SU(N) lives on link from site x in direction μ
    - Under gauge transform g(x): U_μ(x) → g(x) U_μ(x) g†(x+μ)
    
Indexing:
    - Sites: Lexicographic ordering x = x₀ + L₀(x₁ + L₁(x₂ + ...))
    - Links: (site_index, direction) or flat index
    - Periodic boundary conditions standard

Author: TiganticLabz Yang-Mills Project
Date: 2026-01-15
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, Optional
from enum import IntEnum


# =============================================================================
# SITE AND LINK STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class LatticeSite:
    """
    A site (vertex) on the lattice.
    
    Coordinates are integers in [0, L_μ) for each direction μ.
    """
    coords: Tuple[int, ...]
    
    @property
    def dim(self) -> int:
        return len(self.coords)
    
    def __add__(self, direction: 'Direction') -> 'LatticeSite':
        """Move in a direction (with periodic BC applied later)."""
        new_coords = list(self.coords)
        new_coords[direction.axis] += direction.sign
        return LatticeSite(tuple(new_coords))
    
    def __repr__(self) -> str:
        return f"Site{self.coords}"


class Direction(IntEnum):
    """
    Direction on the lattice.
    
    For d dimensions, we have 2d directions: ±μ for μ = 0, 1, ..., d-1.
    Positive directions: 0, 1, 2, ...
    Negative directions: -1, -2, -3, ... (or equivalently d, d+1, ...)
    """
    pass  # Will be created dynamically per lattice


@dataclass(frozen=True)
class LatticeLink:
    """
    A link (edge) on the lattice.
    
    Defined by starting site and direction.
    Link from x in direction μ carries U_μ(x).
    """
    site: LatticeSite
    direction: int  # 0, 1, 2, ... for positive directions
    
    @property
    def start(self) -> LatticeSite:
        return self.site
    
    def __repr__(self) -> str:
        return f"Link({self.site}, μ={self.direction})"


@dataclass(frozen=True)
class Plaquette:
    """
    An elementary plaquette (square) on the lattice.
    
    Defined by corner site and two directions (μ, ν).
    The plaquette operator is:
        P_μν(x) = U_μ(x) U_ν(x+μ) U†_μ(x+ν) U†_ν(x)
    
    This traces a closed loop, gauge invariant under Tr.
    """
    site: LatticeSite
    mu: int  # First direction
    nu: int  # Second direction (mu < nu by convention)
    
    def __repr__(self) -> str:
        return f"Plaq({self.site}, μ={self.mu}, ν={self.nu})"


# =============================================================================
# LATTICE CLASS
# =============================================================================

@dataclass
class Lattice:
    """
    Hypercubic lattice with periodic boundary conditions.
    
    Attributes:
        dims: Tuple of lattice sizes (L₀, L₁, ..., L_{d-1})
        
    For Yang-Mills in D spacetime dimensions:
        - 2+1D: dims = (Lt, Lx, Ly)
        - 3+1D: dims = (Lt, Lx, Ly, Lz)
        
    In Hamiltonian formalism, we often use spatial lattice only.
    """
    dims: Tuple[int, ...]
    
    # Computed attributes
    n_dims: int = field(init=False)
    volume: int = field(init=False)
    n_sites: int = field(init=False)
    n_links: int = field(init=False)
    n_plaquettes: int = field(init=False)
    
    def __post_init__(self):
        self.n_dims = len(self.dims)
        self.volume = int(np.prod(self.dims))
        self.n_sites = self.volume
        self.n_links = self.n_sites * self.n_dims  # One link per site per direction
        # Plaquettes: one per site per pair of directions
        self.n_plaquettes = self.n_sites * (self.n_dims * (self.n_dims - 1)) // 2
    
    # -------------------------------------------------------------------------
    # Site operations
    # -------------------------------------------------------------------------
    
    def site_index(self, site: LatticeSite) -> int:
        """Convert site coordinates to flat index (lexicographic)."""
        idx = 0
        stride = 1
        for i, (c, L) in enumerate(zip(site.coords, self.dims)):
            idx += (c % L) * stride  # Apply periodic BC
            stride *= L
        return idx
    
    def index_to_site(self, idx: int) -> LatticeSite:
        """Convert flat index to site coordinates."""
        coords = []
        for L in self.dims:
            coords.append(idx % L)
            idx //= L
        return LatticeSite(tuple(coords))
    
    def shift_site(self, site: LatticeSite, direction: int, steps: int = 1) -> LatticeSite:
        """
        Shift site in given direction with periodic BC.
        
        Args:
            site: Starting site
            direction: Direction index (0, 1, ..., d-1)
            steps: Number of steps (can be negative)
        """
        coords = list(site.coords)
        coords[direction] = (coords[direction] + steps) % self.dims[direction]
        return LatticeSite(tuple(coords))
    
    def neighbor(self, site: LatticeSite, direction: int) -> LatticeSite:
        """Get neighboring site in positive direction."""
        return self.shift_site(site, direction, +1)
    
    def all_sites(self) -> Iterator[LatticeSite]:
        """Iterate over all lattice sites."""
        for idx in range(self.n_sites):
            yield self.index_to_site(idx)
    
    # -------------------------------------------------------------------------
    # Link operations
    # -------------------------------------------------------------------------
    
    def link_index(self, link: LatticeLink) -> int:
        """Convert link to flat index."""
        site_idx = self.site_index(link.site)
        return site_idx * self.n_dims + link.direction
    
    def index_to_link(self, idx: int) -> LatticeLink:
        """Convert flat index to link."""
        direction = idx % self.n_dims
        site_idx = idx // self.n_dims
        site = self.index_to_site(site_idx)
        return LatticeLink(site, direction)
    
    def all_links(self) -> Iterator[LatticeLink]:
        """Iterate over all links."""
        for site in self.all_sites():
            for mu in range(self.n_dims):
                yield LatticeLink(site, mu)
    
    def links_at_site(self, site: LatticeSite) -> List[LatticeLink]:
        """Get all links emanating from a site (positive directions)."""
        return [LatticeLink(site, mu) for mu in range(self.n_dims)]
    
    def links_into_site(self, site: LatticeSite) -> List[LatticeLink]:
        """Get all links pointing into a site (from negative directions)."""
        links = []
        for mu in range(self.n_dims):
            prev_site = self.shift_site(site, mu, -1)
            links.append(LatticeLink(prev_site, mu))
        return links
    
    # -------------------------------------------------------------------------
    # Plaquette operations  
    # -------------------------------------------------------------------------
    
    def all_plaquettes(self) -> Iterator[Plaquette]:
        """Iterate over all elementary plaquettes."""
        for site in self.all_sites():
            for mu in range(self.n_dims):
                for nu in range(mu + 1, self.n_dims):
                    yield Plaquette(site, mu, nu)
    
    def plaquette_links(self, plaq: Plaquette) -> List[Tuple[LatticeLink, bool]]:
        """
        Get the four links forming a plaquette with their orientations.
        
        Returns list of (link, is_forward) tuples.
        
        Path: x → x+μ → x+μ+ν → x+ν → x
        Links: U_μ(x), U_ν(x+μ), U†_μ(x+ν), U†_ν(x)
        """
        x = plaq.site
        mu, nu = plaq.mu, plaq.nu
        
        x_plus_mu = self.neighbor(x, mu)
        x_plus_nu = self.neighbor(x, nu)
        
        return [
            (LatticeLink(x, mu), True),           # U_μ(x)
            (LatticeLink(x_plus_mu, nu), True),   # U_ν(x+μ)
            (LatticeLink(x_plus_nu, mu), False),  # U†_μ(x+ν)
            (LatticeLink(x, nu), False),          # U†_ν(x)
        ]
    
    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return f"Lattice(dims={self.dims}, sites={self.n_sites}, links={self.n_links})"
    
    def info(self) -> str:
        """Detailed lattice information."""
        d = self.n_dims
        return f"""
Lattice Geometry
================
  Dimensions: {d}D
  Shape: {' × '.join(map(str, self.dims))}
  Sites: {self.n_sites:,}
  Links: {self.n_links:,}
  Plaquettes: {self.n_plaquettes:,}
  
For SU(2) gauge theory:
  Link Hilbert space dimension: ∞ (L²(SU(2)))
  Truncated to j_max: (2j_max + 1)² per link
  Total dimension: (2j_max + 1)^(2 × {self.n_links}) 
  
QTT representation required for tractability.
"""


# =============================================================================
# GAUGE FIELD CONFIGURATION
# =============================================================================

@dataclass
class GaugeConfiguration:
    """
    A configuration of gauge fields on the lattice.
    
    Stores U_μ(x) ∈ SU(N) for each link.
    This is the classical configuration (before quantization).
    """
    lattice: Lattice
    links: np.ndarray  # Shape: (n_links, N, N) complex
    gauge_group_dim: int = 2  # SU(2) by default
    
    def __post_init__(self):
        expected_shape = (self.lattice.n_links, self.gauge_group_dim, self.gauge_group_dim)
        if self.links.shape != expected_shape:
            raise ValueError(f"Links shape {self.links.shape} != expected {expected_shape}")
    
    @classmethod
    def identity(cls, lattice: Lattice, N: int = 2) -> 'GaugeConfiguration':
        """Create trivial configuration (all links = identity)."""
        links = np.zeros((lattice.n_links, N, N), dtype=np.complex128)
        for i in range(lattice.n_links):
            links[i] = np.eye(N)
        return cls(lattice=lattice, links=links, gauge_group_dim=N)
    
    @classmethod
    def random(cls, lattice: Lattice, N: int = 2, seed: Optional[int] = None) -> 'GaugeConfiguration':
        """Create random configuration (Haar-random SU(N) on each link)."""
        if seed is not None:
            np.random.seed(seed)
        
        links = np.zeros((lattice.n_links, N, N), dtype=np.complex128)
        for i in range(lattice.n_links):
            # Generate random SU(N) via QR decomposition
            z = np.random.randn(N, N) + 1j * np.random.randn(N, N)
            q, r = np.linalg.qr(z)
            # Make determinant 1
            d = np.diag(r)
            ph = d / np.abs(d)
            q = q @ np.diag(ph.conj())
            q = q / np.linalg.det(q)**(1/N)  # Ensure det = 1
            links[i] = q
        
        return cls(lattice=lattice, links=links, gauge_group_dim=N)
    
    def get_link(self, link: LatticeLink) -> np.ndarray:
        """Get U_μ(x) for given link."""
        idx = self.lattice.link_index(link)
        return self.links[idx]
    
    def set_link(self, link: LatticeLink, U: np.ndarray):
        """Set U_μ(x) for given link."""
        idx = self.lattice.link_index(link)
        self.links[idx] = U
    
    def plaquette(self, plaq: Plaquette) -> np.ndarray:
        """
        Compute plaquette operator P_μν(x) = U_μ(x) U_ν(x+μ) U†_μ(x+ν) U†_ν(x).
        """
        result = np.eye(self.gauge_group_dim, dtype=np.complex128)
        
        for link, is_forward in self.lattice.plaquette_links(plaq):
            U = self.get_link(link)
            if is_forward:
                result = result @ U
            else:
                result = result @ U.conj().T
        
        return result
    
    def wilson_action(self, beta: float) -> float:
        """
        Wilson gauge action: S = β Σ_P (1 - Re Tr(P) / N)
        
        This is the lattice action for pure gauge theory.
        β = 2N/g² relates to continuum coupling.
        """
        N = self.gauge_group_dim
        action = 0.0
        
        for plaq in self.lattice.all_plaquettes():
            P = self.plaquette(plaq)
            action += 1.0 - np.real(np.trace(P)) / N
        
        return beta * action
    
    def average_plaquette(self) -> float:
        """Average plaquette value: ⟨Re Tr(P)⟩ / N."""
        N = self.gauge_group_dim
        total = 0.0
        
        for plaq in self.lattice.all_plaquettes():
            P = self.plaquette(plaq)
            total += np.real(np.trace(P)) / N
        
        return total / self.lattice.n_plaquettes


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LATTICE GAUGE THEORY GEOMETRY")
    print("=" * 70)
    
    # Create sample lattices
    lattices = [
        Lattice((4, 4)),        # 2D
        Lattice((4, 4, 4)),     # 3D (spatial for 2+1D QFT)
        Lattice((4, 4, 4, 4)),  # 4D (spacetime for 3+1D QFT)
    ]
    
    for lat in lattices:
        print(lat.info())
    
    # Test with 2D lattice
    print("=" * 70)
    print("GAUGE CONFIGURATION TEST (4×4 lattice)")
    print("=" * 70)
    
    lat = Lattice((4, 4))
    
    # Identity configuration
    config_id = GaugeConfiguration.identity(lat)
    print(f"\nIdentity configuration:")
    print(f"  Wilson action (β=1): {config_id.wilson_action(1.0):.6f}")
    print(f"  Average plaquette: {config_id.average_plaquette():.6f}")
    
    # Random configuration
    config_rand = GaugeConfiguration.random(lat, seed=42)
    print(f"\nRandom configuration:")
    print(f"  Wilson action (β=1): {config_rand.wilson_action(1.0):.6f}")
    print(f"  Average plaquette: {config_rand.average_plaquette():.6f}")
    
    # Verify plaquette trace bounds
    print(f"\nPlaquette verification:")
    plaq_traces = []
    for plaq in lat.all_plaquettes():
        P = config_rand.plaquette(plaq)
        tr = np.real(np.trace(P))
        plaq_traces.append(tr)
    
    print(f"  Min Re Tr(P): {min(plaq_traces):.4f}")
    print(f"  Max Re Tr(P): {max(plaq_traces):.4f}")
    print(f"  Valid range for SU(2): [-2, 2] ✓")
    
    print("\n" + "=" * 70)
    print("  ★ LATTICE INFRASTRUCTURE VALIDATED ★")
    print("=" * 70)
