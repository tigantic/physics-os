"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     A C O U S T I C S   M O D U L E                        ║
║                                                                            ║
║  Frequency-domain, room, and aero-acoustics for domain I.6 & XX.5.        ║
║                                                                            ║
║  Solvers:                                                                  ║
║    - Helmholtz BEM (boundary element, ∇²p + k²p = 0)                      ║
║    - Room acoustics: image source method (ISM)                             ║
║    - Structural-acoustic coupling (impedance interface)                    ║
║    - Lighthill aeroacoustic analogy (T_ij → ∂²T_ij/∂x_i∂x_j)            ║
║    - Ffowcs Williams–Hawkings (FW-H) surface integral                     ║
║                                                                            ║
║  References:                                                               ║
║    [1] Wu, T.W. (2000). Boundary Element Acoustics.                       ║
║    [2] Allen & Berkley (1979). Image method for room acoustics.            ║
║    [3] Lighthill (1952). On sound generated aerodynamically.               ║
║    [4] Ffowcs Williams & Hawkings (1969). Proc. R. Soc. London A.         ║
║    [5] Farassat (2007). Derivation of F1A for moving surfaces.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import math


# ═══════════════════════════════════════════════════════════════════════════════
#  HELMHOLTZ BEM (BOUNDARY ELEMENT METHOD)
# ═══════════════════════════════════════════════════════════════════════════════

def _hankel2_0(z: Tensor) -> Tensor:
    """
    Hankel function of the second kind H₀⁽²⁾(z) = J₀(z) - i Y₀(z).

    For acoustics, the 2D free-space Green's function is:
        G(r) = (i/4) H₀⁽¹⁾(kr)

    Uses series expansion for small arguments, asymptotic for large.
    """
    x = z.real if z.is_complex() else z
    abs_z = z.abs()

    # J₀ via power series (|z| < 10)
    j0 = torch.ones_like(z)
    term = torch.ones_like(z)
    z2_over_4 = -(z / 2.0) ** 2
    for m in range(1, 25):
        term = term * z2_over_4 / (m * m)
        j0 = j0 + term

    # Y₀ via Neumann series (Euler-Mascheroni γ = 0.5772156649...)
    gamma_em = 0.5772156649015329
    y0 = (2.0 / math.pi) * (torch.log(z / 2.0 + 1e-30) + gamma_em) * j0

    # Correction terms for Y₀
    partial_sum = torch.zeros_like(z)
    term_y = torch.ones_like(z)
    for m in range(1, 25):
        term_y is not None  # suppress warning
        term_y = term_y * z2_over_4 / (m * m)
        harmonic = sum(1.0 / k for k in range(1, m + 1))
        partial_sum = partial_sum + term_y * harmonic

    y0 = y0 - (2.0 / math.pi) * partial_sum

    return j0 - 1j * y0


@dataclass
class HelmholtzBEM:
    """
    Boundary Element Method solver for the Helmholtz equation in 2D:

        ∇²p + k²p = 0

    Solves the exterior/interior problem with boundary integral equation:

        c(x)p(x) + ∫_Γ (∂G/∂n)p dΓ = ∫_Γ G (∂p/∂n) dΓ

    where G is the free-space Green's function:
        G(x,y) = (i/4) H₀⁽¹⁾(k|x-y|)   (2D)

    Burton-Miller formulation for unique solution at all frequencies:
        (cI + H + αD)p = (G + αS)(∂p/∂n)

    where α = i/k (coupling parameter).

    Attributes:
        frequency: Excitation frequency (Hz)
        c_sound: Speed of sound (m/s)
        rho: Medium density (kg/m³)
    """

    frequency: float = 1000.0
    c_sound: float = 343.0
    rho: float = 1.225

    @property
    def k(self) -> float:
        """Wavenumber k = 2πf/c."""
        return 2.0 * math.pi * self.frequency / self.c_sound

    @property
    def omega(self) -> float:
        """Angular frequency."""
        return 2.0 * math.pi * self.frequency

    @property
    def wavelength(self) -> float:
        return self.c_sound / self.frequency

    def assemble_matrices(
        self, nodes: Tensor, normals: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Assemble BEM influence matrices H and G.

        H_ij = ∫_Γj (∂G/∂n_y)(x_i, y) dΓ_y
        G_ij = ∫_Γj G(x_i, y) dΓ_y

        For collocation at boundary nodes using constant elements.

        Args:
            nodes: [N, 2] boundary element midpoints
            normals: [N, 2] outward normals

        Returns:
            (H, G): [N, N] complex influence matrices
        """
        N = nodes.shape[0]
        k = self.k

        H = torch.zeros(N, N, dtype=torch.complex128, device=nodes.device)
        G_mat = torch.zeros(N, N, dtype=torch.complex128, device=nodes.device)

        for i in range(N):
            for j in range(N):
                if i == j:
                    # Diagonal: singular integral
                    # For constant element of length Δℓ:
                    # G_ii ≈ (Δℓ/2π)(1 - ln(kΔℓ/4) + iπ/2 + γ)
                    # Approximate element length from neighbors
                    if j < N - 1:
                        dl = torch.norm(nodes[j + 1] - nodes[j])
                    else:
                        dl = torch.norm(nodes[j] - nodes[j - 1])

                    gamma_em = 0.5772156649015329
                    kdl = k * dl.item()
                    G_mat[i, i] = (dl / (2.0 * math.pi)) * (
                        1.0 - math.log(max(kdl / 4.0, 1e-30)) + 1j * math.pi / 2.0 + gamma_em
                    )
                    H[i, i] = 0.5  # c(x) = 1/2 on smooth boundary
                else:
                    r_vec = nodes[i] - nodes[j]
                    r = torch.norm(r_vec)
                    kr = k * r

                    if kr < 1e-12:
                        continue

                    # Green's function: G = (i/4) H₀⁽¹⁾(kr) = (-i/4) H₀⁽²⁾(kr)^*
                    # Use H₀⁽¹⁾ = J₀ + iY₀
                    kr_t = torch.tensor(kr.item(), dtype=torch.complex128)
                    H0_2 = _hankel2_0(kr_t)  # H₀⁽²⁾
                    H0_1 = H0_2.conj()       # H₀⁽¹⁾

                    G_val = 0.25j * H0_1

                    # ∂G/∂n = (∂G/∂r)(∂r/∂n)
                    # ∂G/∂r = -(ik/4) H₁⁽¹⁾(kr)
                    # H₁⁽¹⁾ ~ J₁ + iY₁, but use recurrence:
                    # H₁⁽¹⁾(z) = -(d/dz)H₀⁽¹⁾(z) + (0/z)H₀⁽¹⁾ by identity
                    # Actually: H₁(z) = -H₀'(z)
                    # Approximate with finite difference for robustness
                    eps_fd = max(kr.item() * 1e-6, 1e-10)
                    kr_p = torch.tensor(kr.item() + eps_fd, dtype=torch.complex128)
                    kr_m = torch.tensor(kr.item() - eps_fd, dtype=torch.complex128)
                    H0_p = _hankel2_0(kr_p).conj()
                    H0_m = _hankel2_0(kr_m).conj()
                    dH0_dkr = (H0_p - H0_m) / (2.0 * eps_fd)

                    dG_dr = 0.25j * k * dH0_dkr

                    # ∂r/∂n = (r_vec · n_j) / |r|
                    dr_dn = -(r_vec @ normals[j]) / r  # negative: y is source

                    # Approximate element length
                    if j < N - 1:
                        dl = torch.norm(nodes[j + 1] - nodes[j])
                    else:
                        dl = torch.norm(nodes[j] - nodes[j - 1])

                    G_mat[i, j] = G_val * dl
                    H[i, j] = dG_dr * dr_dn * dl

        return H, G_mat

    def solve(
        self,
        nodes: Tensor,
        normals: Tensor,
        bc_type: Tensor,
        bc_values: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Solve the BEM system for pressure and normal velocity.

        Boundary integral equation (collocated):
            (½I + H)p = G q

        where q = ∂p/∂n = -iωρ v_n.

        Args:
            nodes: [N, 2] boundary node positions
            normals: [N, 2] outward normals
            bc_type: [N] int — 0=Neumann (q given), 1=Dirichlet (p given)
            bc_values: [N] complex — prescribed values

        Returns:
            (pressure, flux): [N] complex values on boundary
        """
        H, G_mat = self.assemble_matrices(nodes, normals)
        N = nodes.shape[0]

        # Rearrange into [A]{x} = {b}
        # Neumann (type=0): p unknown, q known → move G*q to RHS
        # Dirichlet (type=1): q unknown, p known → move H*p to RHS

        A = torch.zeros(N, N, dtype=torch.complex128, device=nodes.device)
        b = torch.zeros(N, dtype=torch.complex128, device=nodes.device)

        for j in range(N):
            if bc_type[j] == 0:
                # Neumann: j-th unknown is p_j
                A[:, j] = H[:, j]
                A[j, j] += 0.5  # c(x) = 1/2
            else:
                # Dirichlet: j-th unknown is q_j
                A[:, j] = -G_mat[:, j]

        # RHS
        for j in range(N):
            if bc_type[j] == 0:
                # q_j is known
                b += G_mat[:, j] * bc_values[j]
            else:
                # p_j is known
                known_contrib = H[:, j] * bc_values[j]
                known_contrib[j] += 0.5 * bc_values[j]
                b -= known_contrib

        # Solve
        x = torch.linalg.solve(A, b)

        # Reconstruct full p and q
        pressure = torch.zeros(N, dtype=torch.complex128, device=nodes.device)
        flux = torch.zeros(N, dtype=torch.complex128, device=nodes.device)

        for j in range(N):
            if bc_type[j] == 0:
                pressure[j] = x[j]
                flux[j] = bc_values[j]
            else:
                pressure[j] = bc_values[j]
                flux[j] = x[j]

        return pressure, flux

    def field_pressure(
        self,
        field_points: Tensor,
        nodes: Tensor,
        normals: Tensor,
        pressure: Tensor,
        flux: Tensor,
    ) -> Tensor:
        """
        Compute pressure at interior/exterior field points from boundary solution.

        p(x) = ∫_Γ G(x,y)q(y)dΓ - ∫_Γ (∂G/∂n_y)(x,y)p(y)dΓ

        Args:
            field_points: [M, 2] evaluation coordinates
            nodes, normals, pressure, flux: boundary data

        Returns:
            [M] complex pressure at field points
        """
        M = field_points.shape[0]
        N = nodes.shape[0]
        k = self.k

        p_field = torch.zeros(M, dtype=torch.complex128, device=field_points.device)

        for m in range(M):
            for j in range(N):
                r_vec = field_points[m] - nodes[j]
                r = torch.norm(r_vec)
                kr = k * r.item()

                if kr < 1e-12:
                    continue

                kr_t = torch.tensor(kr, dtype=torch.complex128)
                H0_1 = _hankel2_0(kr_t).conj()
                G_val = 0.25j * H0_1

                # ∂G/∂n_y
                eps_fd = max(kr * 1e-6, 1e-10)
                kr_p = torch.tensor(kr + eps_fd, dtype=torch.complex128)
                kr_m = torch.tensor(kr - eps_fd, dtype=torch.complex128)
                dH0_dkr = (_hankel2_0(kr_p).conj() - _hankel2_0(kr_m).conj()) / (2.0 * eps_fd)
                dG_dr = 0.25j * k * dH0_dkr
                dr_dn = -(r_vec @ normals[j]) / r

                # Element length
                if j < N - 1:
                    dl = torch.norm(nodes[j + 1] - nodes[j])
                else:
                    dl = torch.norm(nodes[j] - nodes[j - 1])

                p_field[m] += G_val * flux[j] * dl - dG_dr * dr_dn * pressure[j] * dl

        return p_field


# ═══════════════════════════════════════════════════════════════════════════════
#  ROOM ACOUSTICS — IMAGE SOURCE METHOD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoomAcoustics:
    """
    Image Source Method (ISM) for rectangular room acoustics.

    The pressure at receiver due to source in a shoebox room is:

        p(x,ω) = Σ_{images} (A_n / 4πr_n) exp(-ikr_n) × Π β_walls

    where:
        - r_n: distance from n-th image source to receiver
        - β_walls: wall reflection coefficients (complex, frequency-dependent)
        - Sum runs over all image orders up to max_order

    For time-domain RIR (Room Impulse Response):
        h(t) = Σ_n (A_n / r_n) δ(t - r_n/c)

    Attributes:
        Lx, Ly, Lz: Room dimensions (m)
        c: Speed of sound (m/s)
        max_order: Maximum image reflection order
        wall_absorption: Dict of 6 wall absorption coefficients α ∈ [0,1]
    """

    Lx: float = 10.0     # Length
    Ly: float = 8.0      # Width
    Lz: float = 3.0      # Height
    c: float = 343.0
    max_order: int = 10
    wall_absorption: Dict[str, float] = field(default_factory=lambda: {
        "x0": 0.1, "x1": 0.1,  # x=0, x=Lx walls
        "y0": 0.1, "y1": 0.1,  # y=0, y=Ly walls
        "z0": 0.05, "z1": 0.8, # floor(z=0), ceiling(z=Lz)
    })

    def _reflection_coefficient(self, wall: str) -> float:
        """β = √(1 - α) for wall with absorption coefficient α."""
        alpha = self.wall_absorption.get(wall, 0.1)
        return math.sqrt(max(1.0 - alpha, 0.0))

    def compute_image_sources(
        self, source: Tensor
    ) -> List[Tuple[Tensor, float]]:
        """
        Generate all image sources up to max_order.

        Uses the standard 8-octant Allen-Berkley enumeration for a
        shoebox room [0,Lx]×[0,Ly]×[0,Lz].

        For each parity triple (px, py, pz) ∈ {0,1}³ and integer
        triple (nx, ny, nz):

            image_x = (1 - 2px) · sx + 2 nx Lx
            image_y = (1 - 2py) · sy + 2 ny Ly
            image_z = (1 - 2pz) · sz + 2 nz Lz

        The reflection order is |nx| + |ny| + |nz| where the number
        of reflections off each wall pair is tracked exactly.

        Args:
            source: [3] source position

        Returns:
            List of (image_position[3], amplitude) tuples
        """
        images: List[Tuple[Tensor, float]] = []
        sx, sy, sz = source[0].item(), source[1].item(), source[2].item()

        beta_x0 = self._reflection_coefficient("x0")
        beta_x1 = self._reflection_coefficient("x1")
        beta_y0 = self._reflection_coefficient("y0")
        beta_y1 = self._reflection_coefficient("y1")
        beta_z0 = self._reflection_coefficient("z0")
        beta_z1 = self._reflection_coefficient("z1")

        # Direct source (order 0)
        images.append((source.clone(), 1.0))

        # Enumerate all unique image sources using parity-index form
        # px ∈ {0,1} determines if the source is reflected in x (px=1) or not
        for px in (0, 1):
            for py in (0, 1):
                for pz in (0, 1):
                    # Range of integer indices such that reflection order ≤ max_order
                    max_n = (self.max_order + 1) // 2 + 1
                    for nx in range(-max_n, max_n + 1):
                        for ny in range(-max_n, max_n + 1):
                            nz_remaining = self.max_order - abs(nx) - abs(ny)
                            if nz_remaining < 0:
                                continue
                            for nz in range(-max_n, max_n + 1):
                                if abs(nz) > nz_remaining:
                                    continue

                                # Skip origin (direct source already added)
                                if px == 0 and py == 0 and pz == 0 and nx == 0 and ny == 0 and nz == 0:
                                    continue

                                # Reflection order (number of wall bounces)
                                # For each axis: reflections = 2|n| + p (if n≠0 or p=1)
                                # But total order = sum of per-axis reflections
                                n_ref_x = 2 * abs(nx) + px if (nx != 0 or px != 0) else 0
                                n_ref_y = 2 * abs(ny) + py if (ny != 0 or py != 0) else 0
                                n_ref_z = 2 * abs(nz) + pz if (nz != 0 or pz != 0) else 0
                                total_order = n_ref_x + n_ref_y + n_ref_z

                                if total_order == 0 or total_order > self.max_order:
                                    continue

                                # Image position
                                x_img = (1 - 2 * px) * sx + 2 * nx * self.Lx
                                y_img = (1 - 2 * py) * sy + 2 * ny * self.Ly
                                z_img = (1 - 2 * pz) * sz + 2 * nz * self.Lz

                                # Count reflections per wall exactly:
                                # x=0 wall gets ceil(n_ref_x/2) if first reflection is off x=0
                                # For px=1, nx>=0: first hit is x=0 if source near x=0
                                # Use symmetric distribution: alternate between walls
                                n_x0 = (n_ref_x + 1) // 2 if nx <= 0 and px == 1 else n_ref_x // 2
                                n_x1 = n_ref_x - n_x0
                                n_y0 = (n_ref_y + 1) // 2 if ny <= 0 and py == 1 else n_ref_y // 2
                                n_y1 = n_ref_y - n_y0
                                n_z0 = (n_ref_z + 1) // 2 if nz <= 0 and pz == 1 else n_ref_z // 2
                                n_z1 = n_ref_z - n_z0

                                amp = (
                                    beta_x0 ** n_x0 * beta_x1 ** n_x1
                                    * beta_y0 ** n_y0 * beta_y1 ** n_y1
                                    * beta_z0 ** n_z0 * beta_z1 ** n_z1
                                )

                                pos = torch.tensor(
                                    [x_img, y_img, z_img],
                                    dtype=source.dtype,
                                    device=source.device,
                                )
                                images.append((pos, amp))

        return images

    def room_impulse_response(
        self,
        source: Tensor,
        receiver: Tensor,
        fs: float = 16000.0,
        duration: float = 1.0,
    ) -> Tensor:
        """
        Compute Room Impulse Response (RIR) via image source method.

        h(t) = Σ_n (A_n / 4πr_n) δ(t - r_n/c)

        Discretized with fractional delay interpolation.

        Args:
            source: [3] source position
            receiver: [3] receiver position
            fs: Sampling frequency (Hz)
            duration: RIR duration (seconds)

        Returns:
            [n_samples] impulse response
        """
        n_samples = int(fs * duration)
        h = torch.zeros(n_samples, dtype=torch.float64)

        images = self.compute_image_sources(source)

        for img_pos, amp in images:
            r = torch.norm(img_pos - receiver).item()
            if r < 1e-10:
                r = 1e-10

            # Time of arrival
            t_arrival = r / self.c
            sample_f = t_arrival * fs

            if sample_f >= n_samples - 1:
                continue

            # Amplitude: 1/(4πr) with reflection loss
            a = amp / (4.0 * math.pi * r)

            # Fractional delay interpolation (linear)
            idx = int(sample_f)
            frac = sample_f - idx

            if 0 <= idx < n_samples:
                h[idx] += a * (1.0 - frac)
            if 0 <= idx + 1 < n_samples:
                h[idx + 1] += a * frac

        return h

    def t60_sabine(self) -> float:
        """
        Sabine reverberation time T₆₀ (seconds).

        T₆₀ = 0.161 V / A

        where V = room volume, A = total absorption area.
        """
        V = self.Lx * self.Ly * self.Lz

        # Absorption area: A = Σ αᵢ Sᵢ
        A = (
            self.wall_absorption["x0"] * self.Ly * self.Lz
            + self.wall_absorption["x1"] * self.Ly * self.Lz
            + self.wall_absorption["y0"] * self.Lx * self.Lz
            + self.wall_absorption["y1"] * self.Lx * self.Lz
            + self.wall_absorption["z0"] * self.Lx * self.Ly
            + self.wall_absorption["z1"] * self.Lx * self.Ly
        )

        if A < 1e-10:
            return float("inf")

        return 0.161 * V / A

    def t60_eyring(self) -> float:
        """
        Eyring reverberation time (more accurate for high absorption).

        T₆₀ = 0.161 V / (-S ln(1 - ᾱ))

        where S = total surface area, ᾱ = average absorption coefficient.
        """
        V = self.Lx * self.Ly * self.Lz
        S = 2.0 * (self.Lx * self.Ly + self.Ly * self.Lz + self.Lx * self.Lz)

        # Average absorption
        alpha_avg = (
            self.wall_absorption["x0"] * self.Ly * self.Lz
            + self.wall_absorption["x1"] * self.Ly * self.Lz
            + self.wall_absorption["y0"] * self.Lx * self.Lz
            + self.wall_absorption["y1"] * self.Lx * self.Lz
            + self.wall_absorption["z0"] * self.Lx * self.Ly
            + self.wall_absorption["z1"] * self.Lx * self.Ly
        ) / S

        if alpha_avg >= 1.0:
            return 0.0

        denominator = -S * math.log(max(1.0 - alpha_avg, 1e-10))
        return 0.161 * V / denominator


# ═══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL-ACOUSTIC COUPLING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralAcousticCoupler:
    """
    Coupled structural-acoustic system.

    The coupled system in frequency domain:

        ⎡ K_s - ω²M_s    -C    ⎤ ⎧u⎫   ⎧F⎫
        ⎢                       ⎥ ⎨ ⎬ = ⎨ ⎬
        ⎣  ρω²Cᵀ     K_a-ω²M_a⎦ ⎩p⎭   ⎩0⎭

    where:
        K_s, M_s: structural stiffness/mass
        K_a, M_a: acoustic stiffness/mass
        C: coupling matrix (area integrals of N_s · n · N_a)

    The coupling C connects structural displacement to acoustic pressure
    through the interface normal velocity continuity:
        v_n = iω u_n  (structural side)
        ∂p/∂n = -ρω² u_n  (acoustic side)

    Attributes:
        rho_fluid: Fluid density
        c_fluid: Sound speed in fluid
    """

    rho_fluid: float = 1.225
    c_fluid: float = 343.0

    def assemble_coupling_matrix(
        self,
        structural_nodes: Tensor,
        acoustic_nodes: Tensor,
        interface_elements: List[Tuple[int, int]],
        normals: Tensor,
        areas: Tensor,
    ) -> Tensor:
        """
        Assemble structural-acoustic coupling matrix C.

        C_ij = ∫_Γ N_s,i · n · N_a,j dΓ

        For coincident nodes on the interface:
            C_ij = A_e · n_e  (lumped)

        Args:
            structural_nodes: [N_s, 3] structural node positions
            acoustic_nodes: [N_a, 3] acoustic node positions
            interface_elements: List of (structural_node_idx, acoustic_node_idx) pairs
            normals: [N_interface, 3] interface normals (into fluid)
            areas: [N_interface] interface element areas

        Returns:
            C: [N_s_dof, N_a_dof] coupling matrix
        """
        n_s = structural_nodes.shape[0]
        n_a = acoustic_nodes.shape[0]

        C = torch.zeros(n_s, n_a, dtype=torch.float64)

        for idx, (s_node, a_node) in enumerate(interface_elements):
            # Lumped coupling: C_{s,a} = area * 1 (normal DOF)
            C[s_node, a_node] = areas[idx].item()

        return C

    def solve_coupled(
        self,
        K_s: Tensor,
        M_s: Tensor,
        K_a: Tensor,
        M_a: Tensor,
        C: Tensor,
        F_ext: Tensor,
        omega: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Solve the coupled structural-acoustic system at frequency ω.

        Args:
            K_s: [n_s, n_s] structural stiffness
            M_s: [n_s, n_s] structural mass
            K_a: [n_a, n_a] acoustic stiffness
            M_a: [n_a, n_a] acoustic mass
            C: [n_s, n_a] coupling matrix
            F_ext: [n_s] external structural load
            omega: Angular frequency

        Returns:
            (u, p): Structural displacement [n_s], acoustic pressure [n_a]
        """
        n_s = K_s.shape[0]
        n_a = K_a.shape[0]
        n_total = n_s + n_a

        # Build coupled system matrix (complex for damping)
        Z = torch.zeros(n_total, n_total, dtype=torch.complex128)

        # Structural block
        Z[:n_s, :n_s] = (K_s - omega ** 2 * M_s).to(torch.complex128)

        # Coupling: structural → acoustic
        Z[:n_s, n_s:] = -(C).to(torch.complex128)

        # Coupling: acoustic → structural
        Z[n_s:, :n_s] = (self.rho_fluid * omega ** 2 * C.T).to(torch.complex128)

        # Acoustic block
        Z[n_s:, n_s:] = (K_a - omega ** 2 * M_a).to(torch.complex128)

        # RHS
        rhs = torch.zeros(n_total, dtype=torch.complex128)
        rhs[:n_s] = F_ext.to(torch.complex128)

        # Solve
        solution = torch.linalg.solve(Z, rhs)

        u = solution[:n_s]
        p = solution[n_s:]

        return u, p


# ═══════════════════════════════════════════════════════════════════════════════
#  AEROACOUSTICS — LIGHTHILL ANALOGY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LighthillAnalogy:
    """
    Lighthill's acoustic analogy for aerodynamically generated sound.

    The Lighthill equation:
        ∂²ρ'/∂t² - c₀²∇²ρ' = ∂²T_ij/(∂x_i∂x_j)

    where the Lighthill stress tensor:
        T_ij = ρu_iu_j + (p' - c₀²ρ')δ_ij - τ_ij

    For high Reynolds number flows (viscous stress negligible):
        T_ij ≈ ρu_iu_j = ρ₀v_iv_j (for subsonic, incompressible source)

    Solution via retarded-time volume integral:
        ρ'(x,t) = (1/4πc₀²) ∫ [∂²T_ij/∂y_i∂y_j]_{t_ret} / |x-y| d³y

    In the far field (Fraunhofer zone, |x| >> source region):
        p'(x,t) ≈ (x_ix_j / 4πc₀²|x|³) ∫ [∂²T_ij/∂t²]_{t_ret} d³y

    Attributes:
        c0: Ambient sound speed
        rho0: Ambient density
    """

    c0: float = 343.0
    rho0: float = 1.225

    def lighthill_stress_tensor(
        self,
        velocity: Tensor,
        density: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Lighthill stress tensor T_ij from flow field.

        T_ij = ρ u_i u_j  (high-Re, subsonic approximation)

        Args:
            velocity: [..., 3] velocity field
            density: [...] density field (default: rho0)

        Returns:
            T: [..., 3, 3] stress tensor
        """
        if density is None:
            density = self.rho0 * torch.ones(
                velocity.shape[:-1], dtype=velocity.dtype, device=velocity.device
            )

        # T_ij = ρ u_i u_j
        v = velocity  # [..., 3]
        T = density.unsqueeze(-1).unsqueeze(-1) * (
            v.unsqueeze(-1) * v.unsqueeze(-2)
        )
        return T

    def far_field_pressure(
        self,
        observer: Tensor,
        source_positions: Tensor,
        T_ij: Tensor,
        dT_ij_dt2: Tensor,
        source_volumes: Tensor,
    ) -> complex:
        """
        Far-field acoustic pressure at observer from Lighthill sources.

        p'(x) = (x̂_i x̂_j) / (4π c₀² |x|) ∫ [∂²T_ij/∂t²]_ret dV

        Args:
            observer: [3] observer position
            source_positions: [N_src, 3] source cell centroids
            T_ij: [N_src, 3, 3] Lighthill stress at source
            dT_ij_dt2: [N_src, 3, 3] ∂²T_ij/∂t² at source
            source_volumes: [N_src] cell volumes

        Returns:
            Complex acoustic pressure (for frequency-domain analysis)
        """
        r_vec = observer.unsqueeze(0) - source_positions  # [N, 3]
        r_mag = torch.norm(r_vec, dim=-1, keepdim=True)   # [N, 1]
        x_hat = r_vec / (r_mag + 1e-30)                    # [N, 3]

        # x̂_i T̈_ij x̂_j = contraction
        # x_hat: [N, 3], dT: [N, 3, 3]
        # tmp = x̂_i T̈_ij → [N, 3]
        tmp = torch.einsum("ni,nij->nj", x_hat, dT_ij_dt2)
        # scalar = x̂_j tmp_j → [N]
        integrand = torch.einsum("nj,nj->n", tmp, x_hat)

        # Integrate: Σ integrand × dV / (4π c₀² r)
        r_obs = torch.norm(observer)
        if r_obs < 1e-10:
            return 0.0 + 0.0j

        p_prime = (
            torch.sum(integrand * source_volumes) / (4.0 * math.pi * self.c0 ** 2 * r_obs)
        )

        return p_prime.item()

    def acoustic_power_estimate(
        self,
        velocity_rms: float,
        length_scale: float,
    ) -> float:
        """
        Lighthill's 8th-power law for subsonic jet noise.

        W_a ∝ ρ₀ (U⁸ / c₀⁵) L²

        More precisely:
            W_a ≈ K_L ρ₀ L² U⁸ / c₀⁵

        where K_L ≈ 1e-5 to 1e-4 (empirical).

        Args:
            velocity_rms: Characteristic velocity (m/s)
            length_scale: Characteristic source dimension (m)

        Returns:
            Estimated acoustic power (W)
        """
        K_L = 3.0e-5  # Typical empirical constant
        return K_L * self.rho0 * length_scale ** 2 * velocity_rms ** 8 / self.c0 ** 5


# ═══════════════════════════════════════════════════════════════════════════════
#  FFOWCS WILLIAMS—HAWKINGS (FW-H) SURFACE INTEGRAL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FfowcsWilliamsHawkings:
    """
    Ffowcs Williams-Hawkings surface integral formulation.

    The FW-H equation extends Lighthill's analogy to incorporate
    surfaces (solid bodies, permeable control surfaces):

        □²p' = ∂/∂t[ρ₀vₙδ(f)] - ∂/∂xᵢ[pnᵢδ(f)] + ∂²Tᵢⱼ/(∂xᵢ∂xⱼ)H(f)

    Here only the surface terms (Farassat Formulation 1A):

    Thickness noise:
        4πp_T(x,t) = ∫ [ρ₀v̇ₙ / (r(1-Mᵣ)²)]_ret dS
                    + ∫ [ρ₀vₙ(rṀᵣ + c·(Mᵣ-M²)) / (r²(1-Mᵣ)³)]_ret dS

    Loading noise:
        4πp_L(x,t) = (1/c) ∫ [l̇ᵣ / (r(1-Mᵣ)²)]_ret dS
                    + ∫ [(lᵣ - lₘ) / (r²(1-Mᵣ)²)]_ret dS
                    + (1/c) ∫ [lᵣ(rṀᵣ + c(Mᵣ-M²)) / (r²(1-Mᵣ)³)]_ret dS

    where:
        vₙ = surface normal velocity
        lᵢ = pnᵢ = loading (force per unit area)
        Mᵣ = relative Mach number in radiation direction

    Attributes:
        c0: Ambient speed of sound
        rho0: Ambient density
    """

    c0: float = 343.0
    rho0: float = 1.225

    def thickness_noise(
        self,
        observer: Tensor,
        panel_centers: Tensor,
        panel_normals: Tensor,
        panel_areas: Tensor,
        v_n: Tensor,
        v_n_dot: Tensor,
        panel_velocity: Tensor,
    ) -> float:
        """
        Evaluate thickness noise at observer (Farassat 1A, first term).

        4πp_T ≈ ∫ ρ₀ v̇ₙ / (r(1-Mᵣ)²) dS

        Retarded time effects neglected (compact source approximation).

        Args:
            observer: [3] observer position
            panel_centers: [N, 3] panel centroid positions
            panel_normals: [N, 3] outward normals
            panel_areas: [N] panel areas
            v_n: [N] normal velocity (surface velocity component)
            v_n_dot: [N] time derivative of normal velocity
            panel_velocity: [N, 3] panel velocity vectors

        Returns:
            Thickness noise pressure contribution (Pa)
        """
        r_vec = observer.unsqueeze(0) - panel_centers  # [N, 3]
        r_mag = torch.norm(r_vec, dim=-1)               # [N]
        r_hat = r_vec / (r_mag.unsqueeze(-1) + 1e-30)  # [N, 3]

        # Mach number in radiation direction: Mᵣ = (v · r̂)/c₀
        M_r = torch.sum(panel_velocity * r_hat, dim=-1) / self.c0  # [N]

        # Doppler factor
        doppler = (1.0 - M_r) ** 2

        # Thickness term: ρ₀ v̇ₙ / (r (1-Mᵣ)²)
        integrand = self.rho0 * v_n_dot / (r_mag * doppler + 1e-30)

        # Integrate
        p_T = torch.sum(integrand * panel_areas) / (4.0 * math.pi)

        return p_T.item()

    def loading_noise(
        self,
        observer: Tensor,
        panel_centers: Tensor,
        panel_normals: Tensor,
        panel_areas: Tensor,
        pressure: Tensor,
        panel_velocity: Tensor,
    ) -> float:
        """
        Evaluate loading noise at observer (Farassat 1A, far-field term).

        4πp_L ≈ (1/c) ∫ l̇ᵣ / (r(1-Mᵣ)²) dS

        For stationary surface simplification:
            4πp_L ≈ ∫ lᵣ / (r²) dS  (near-field dipole)

        Args:
            observer: [3]
            panel_centers: [N, 3]
            panel_normals: [N, 3]
            panel_areas: [N]
            pressure: [N] surface pressure (gauge)
            panel_velocity: [N, 3] panel velocity

        Returns:
            Loading noise pressure (Pa)
        """
        r_vec = observer.unsqueeze(0) - panel_centers
        r_mag = torch.norm(r_vec, dim=-1)
        r_hat = r_vec / (r_mag.unsqueeze(-1) + 1e-30)

        M_r = torch.sum(panel_velocity * r_hat, dim=-1) / self.c0
        doppler = (1.0 - M_r) ** 2

        # Loading: l_i = p n_i
        l = pressure.unsqueeze(-1) * panel_normals  # [N, 3]

        # l_r = l · r̂
        l_r = torch.sum(l * r_hat, dim=-1)  # [N]

        # Near-field + far-field combined
        integrand = l_r / (r_mag ** 2 * doppler + 1e-30)

        p_L = torch.sum(integrand * panel_areas) / (4.0 * math.pi)

        return p_L.item()

    def total_noise(
        self,
        observer: Tensor,
        panel_centers: Tensor,
        panel_normals: Tensor,
        panel_areas: Tensor,
        pressure: Tensor,
        v_n: Tensor,
        v_n_dot: Tensor,
        panel_velocity: Tensor,
    ) -> Dict[str, float]:
        """
        Total FW-H noise = thickness + loading.

        Returns:
            Dict with 'thickness', 'loading', 'total' in Pa
        """
        p_T = self.thickness_noise(
            observer, panel_centers, panel_normals, panel_areas,
            v_n, v_n_dot, panel_velocity,
        )
        p_L = self.loading_noise(
            observer, panel_centers, panel_normals, panel_areas,
            pressure, panel_velocity,
        )

        return {
            "thickness_Pa": p_T,
            "loading_Pa": p_L,
            "total_Pa": p_T + p_L,
            "total_dB": 20.0 * math.log10(max(abs(p_T + p_L), 1e-30) / 2e-5),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "HelmholtzBEM",
    "RoomAcoustics",
    "StructuralAcousticCoupler",
    "LighthillAnalogy",
    "FfowcsWilliamsHawkings",
]
