"""
Analytical QTT Construction for Separable Functions.

This module constructs QTT (Quantized Tensor Train) representations of trigonometric
functions DIRECTLY from their mathematical definitions, with ZERO dense memory allocation.

Key Insight:
-----------
Trigonometric functions like sin(kx) and cos(kx) have exact, analytical QTT representations
with rank 2. This allows initialization of arbitrarily large grids (1024³, 4096³, etc.)
without hitting memory limits.

The Taylor-Green vortex:
    u = A * sin(ax) * cos(by) * cos(cz)
    v = -A * cos(ax) * sin(by) * cos(cz)
    w = 0

is a product of separable 1D functions, each representable as rank-2 QTT.

Mathematical Foundation:
-----------------------
For a 1D grid with N = 2^n points:
    sin(k * x_i) where x_i = i * dx, i ∈ [0, N-1]

Using the binary representation i = Σ_{j=0}^{n-1} b_j * 2^j:
    x_i = dx * Σ_{j=0}^{n-1} b_j * 2^j

The QTT cores encode the recurrence:
    exp(i*k*x) = Π_{j=0}^{n-1} exp(i*k*dx*2^j*b_j)

Each core is 2x2 (rank 2) encoding the phase accumulation.

Author: TiganticLabz
Date: 2026-02-03
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnalyticalQTTConfig:
    """Configuration for analytical QTT construction."""
    n_bits: int          # Number of bits per dimension (N = 2^n_bits)
    L: float = 2 * np.pi # Domain size
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32


def _exp_qtt_cores_1d(
    k: float,
    n_bits: int,
    L: float,
    device: str,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """
    Construct QTT cores for exp(i*k*x) on [0, L) with N = 2^n_bits points.
    
    Returns complex-valued cores of shape (r_left, 2, r_right).
    For exp(ikx), rank is exactly 2 throughout (except boundaries).
    
    Mathematical derivation:
    -----------------------
    x_i = (i / N) * L for i = 0, 1, ..., N-1
    
    With binary i = b_0 + 2*b_1 + 4*b_2 + ... + 2^(n-1)*b_{n-1}:
    
    exp(i*k*x_i) = exp(i*k*L/N * Σ_j b_j * 2^j)
                 = Π_j exp(i*k*L/N * 2^j * b_j)
    
    Each factor is 1 if b_j=0, or exp(i*k*L*2^j/N) if b_j=1.
    
    This gives rank-1 representation, but we need rank-2 for real/imag separation.
    """
    N = 1 << n_bits
    dx = L / N
    
    # Phase increment for each bit position
    # For bit j, the contribution is k * dx * 2^j when b_j = 1
    cores = []
    
    for j in range(n_bits):
        phase = k * dx * (1 << j)
        
        # Core shape: (r_left, 2, r_right)
        # For exp(ikx), we track [Re, Im] as a 2-component vector
        
        if j == 0:
            # First core: (1, 2, 2)
            # b=0: multiply by 1 (no phase)
            # b=1: multiply by exp(i*phase)
            core = torch.zeros(1, 2, 2, device=device, dtype=dtype)
            # b=0: [1, 0] (real=1, imag=0)
            core[0, 0, 0] = 1.0  # Re -> Re
            core[0, 0, 1] = 0.0  # Re -> Im
            # b=1: [cos(phase), sin(phase)]
            core[0, 1, 0] = np.cos(phase)
            core[0, 1, 1] = np.sin(phase)
        elif j == n_bits - 1:
            # Last core: (2, 2, 1) - contract to scalar output
            core = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            # b=0: pass through
            core[0, 0, 0] = 1.0  # Re stays Re
            core[1, 0, 0] = 0.0  # Im contributes 0 to final Re
            # b=1: rotate by phase
            c, s = np.cos(phase), np.sin(phase)
            core[0, 1, 0] = c   # Re * cos
            core[1, 1, 0] = s   # Im * sin (but we want Re part only for now)
        else:
            # Middle cores: (2, 2, 2)
            # Propagate [Re, Im] with phase rotation
            core = torch.zeros(2, 2, 2, device=device, dtype=dtype)
            # b=0: identity
            core[0, 0, 0] = 1.0  # Re -> Re
            core[1, 0, 1] = 1.0  # Im -> Im
            # b=1: rotation by phase
            c, s = np.cos(phase), np.sin(phase)
            core[0, 1, 0] = c   # Re -> Re (cos)
            core[0, 1, 1] = s   # Re -> Im (sin)
            core[1, 1, 0] = -s  # Im -> Re (-sin)
            core[1, 1, 1] = c   # Im -> Im (cos)
        
        cores.append(core)
    
    return cores


def sin_qtt_cores_1d(
    k: float,
    n_bits: int,
    L: float = 2 * np.pi,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Construct exact QTT cores for sin(k*x) on [0, L) with N = 2^n_bits points.
    
    Returns rank-2 cores (except boundaries which are rank 1).
    Total memory: O(n_bits) instead of O(2^n_bits).
    
    Parameters
    ----------
    k : float
        Wavenumber
    n_bits : int
        Number of bits (grid has 2^n_bits points)
    L : float
        Domain size
    device : str
        Torch device
    dtype : torch.dtype
        Data type
        
    Returns
    -------
    List[torch.Tensor]
        QTT cores of shape (r_left, 2, r_right)
    """
    N = 1 << n_bits
    dx = L / N
    
    cores = []
    
    for j in range(n_bits):
        phase = k * dx * (1 << j)
        c, s = np.cos(phase), np.sin(phase)
        
        if j == 0:
            # First core: (1, 2, 2) - start with [cos, sin] basis
            core = torch.zeros(1, 2, 2, device=device, dtype=dtype)
            # For sin: we want Im(exp(ikx))
            # State vector is [cos(accumulated), sin(accumulated)]
            # b=0: phase=0, so [1, 0]
            core[0, 0, 0] = 1.0  # cos(0) = 1
            core[0, 0, 1] = 0.0  # sin(0) = 0
            # b=1: phase=k*dx, so [cos(phase), sin(phase)]
            core[0, 1, 0] = c
            core[0, 1, 1] = s
            
        elif j == n_bits - 1:
            # Last core: (2, 2, 1) - extract sin component
            core = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            # b=0: extract sin from state [cos, sin] -> sin
            core[0, 0, 0] = 0.0  # cos contributes 0 to sin
            core[1, 0, 0] = 1.0  # sin contributes 1 to sin
            # b=1: rotate then extract sin
            # [cos, sin] -> [cos*c - sin*s, cos*s + sin*c] -> extract second
            core[0, 1, 0] = s   # cos -> sin (via rotation)
            core[1, 1, 0] = c   # sin -> sin (via rotation)
            
        else:
            # Middle cores: (2, 2, 2) - rotation matrix
            core = torch.zeros(2, 2, 2, device=device, dtype=dtype)
            # b=0: identity
            core[0, 0, 0] = 1.0
            core[1, 0, 1] = 1.0
            # b=1: rotation by phase
            # [cos, sin] -> [cos*c - sin*s, cos*s + sin*c]
            core[0, 1, 0] = c
            core[0, 1, 1] = s
            core[1, 1, 0] = -s
            core[1, 1, 1] = c
        
        cores.append(core)
    
    return cores


def cos_qtt_cores_1d(
    k: float,
    n_bits: int,
    L: float = 2 * np.pi,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Construct exact QTT cores for cos(k*x) on [0, L) with N = 2^n_bits points.
    
    Same structure as sin, but extracts cos component at the end.
    """
    N = 1 << n_bits
    dx = L / N
    
    cores = []
    
    for j in range(n_bits):
        phase = k * dx * (1 << j)
        c, s = np.cos(phase), np.sin(phase)
        
        if j == 0:
            # First core: (1, 2, 2)
            core = torch.zeros(1, 2, 2, device=device, dtype=dtype)
            core[0, 0, 0] = 1.0  # cos(0) = 1
            core[0, 0, 1] = 0.0  # sin(0) = 0
            core[0, 1, 0] = c
            core[0, 1, 1] = s
            
        elif j == n_bits - 1:
            # Last core: (2, 2, 1) - extract cos component
            core = torch.zeros(2, 2, 1, device=device, dtype=dtype)
            # b=0: extract cos from state [cos, sin] -> cos
            core[0, 0, 0] = 1.0  # cos contributes 1 to cos
            core[1, 0, 0] = 0.0  # sin contributes 0 to cos
            # b=1: rotate then extract cos
            core[0, 1, 0] = c   # cos -> cos (via rotation)
            core[1, 1, 0] = -s  # sin -> cos (via rotation)
            
        else:
            # Middle cores: (2, 2, 2) - rotation matrix
            core = torch.zeros(2, 2, 2, device=device, dtype=dtype)
            core[0, 0, 0] = 1.0
            core[1, 0, 1] = 1.0
            core[0, 1, 0] = c
            core[0, 1, 1] = s
            core[1, 1, 0] = -s
            core[1, 1, 1] = c
        
        cores.append(core)
    
    return cores


def constant_qtt_cores_1d(
    value: float,
    n_bits: int,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Construct QTT cores for a constant function f(x) = value.
    
    Rank 1 throughout.
    """
    cores = []
    
    for j in range(n_bits):
        if j == 0:
            # First core: (1, 2, 1) - apply value once
            core = torch.ones(1, 2, 1, device=device, dtype=dtype)
            core[0, 0, 0] = value ** (1.0 / n_bits)  # Distribute value across cores
            core[0, 1, 0] = value ** (1.0 / n_bits)
        else:
            # Other cores: (1, 2, 1) - just pass through
            core = torch.ones(1, 2, 1, device=device, dtype=dtype)
        
        cores.append(core)
    
    return cores


def hadamard_qtt_cores(
    cores_a: List[torch.Tensor],
    cores_b: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Compute Hadamard (element-wise) product of two QTT representations.
    
    If A has rank r_a and B has rank r_b, result has rank r_a * r_b.
    For sin/cos (rank 2), the product is rank 4.
    
    This is exact - no truncation.
    """
    assert len(cores_a) == len(cores_b)
    
    result = []
    for ca, cb in zip(cores_a, cores_b):
        # ca: (r_a_l, 2, r_a_r)
        # cb: (r_b_l, 2, r_b_r)
        # out: (r_a_l * r_b_l, 2, r_a_r * r_b_r)
        
        r_a_l, _, r_a_r = ca.shape
        r_b_l, _, r_b_r = cb.shape
        
        # Hadamard via Kronecker on rank dimensions
        # out[i*r_b_l + j, d, k*r_b_r + l] = ca[i, d, k] * cb[j, d, l]
        out = torch.einsum('idk,jdl->ijdkl', ca, cb)
        out = out.reshape(r_a_l * r_b_l, 2, r_a_r * r_b_r)
        
        result.append(out)
    
    return result


def scale_qtt_cores(
    cores: List[torch.Tensor],
    scalar: float,
) -> List[torch.Tensor]:
    """Scale QTT by a scalar (applied to first core)."""
    result = [c.clone() for c in cores]
    result[0] = result[0] * scalar
    return result


def separable_3d_qtt(
    cores_x: List[torch.Tensor],
    cores_y: List[torch.Tensor],
    cores_z: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Construct 3D QTT for separable function f(x,y,z) = g(x) * h(y) * k(z).
    
    Uses the correct 3D QTT site ordering: [x0, y0, z0, x1, y1, z1, ...]
    
    For separable functions, the 3D QTT is constructed as:
    - Concatenate all 1D cores sequentially
    - The rank at the junction between dimensions is 1 (or the rank of the 1D QTT)
    
    But with interleaved ordering, we use Kronecker products at each level.
    
    Simpler approach: For separable f(x)*g(y)*h(z), we can directly build
    cores where each "level" contains the product structure.
    
    Parameters
    ----------
    cores_x, cores_y, cores_z : List[torch.Tensor]
        1D QTT cores for each dimension (same n_bits)
        
    Returns
    -------
    List[torch.Tensor]
        3D QTT cores with 3*n_bits sites
    """
    n_bits = len(cores_x)
    assert len(cores_y) == n_bits and len(cores_z) == n_bits
    
    device = cores_x[0].device
    dtype = cores_x[0].dtype
    
    # For interleaved 3D QTT of separable function:
    # Site order: x_0, y_0, z_0, x_1, y_1, z_1, ...
    #
    # The value at (i,j,k) with binary i = Σ i_b 2^b, j = Σ j_b 2^b, k = Σ k_b 2^b is:
    # f(x_i) * g(y_j) * h(z_k)
    #
    # Since f, g, h are each encoded as QTT, and they are independent,
    # we can encode the 3D function as a tensor product.
    #
    # For each level b, we have three sites: x_b, y_b, z_b
    # The core for x_b only depends on i_b (the x-bit)
    # The core for y_b only depends on j_b (the y-bit)  
    # The core for z_b only depends on k_b (the z-bit)
    #
    # Key insight: At each level, the x,y,z bits are INDEPENDENT.
    # So the cores at a level form a tensor product structure.
    
    result = []
    
    for level in range(n_bits):
        cx = cores_x[level]  # (rx_l, 2, rx_r)
        cy = cores_y[level]  # (ry_l, 2, ry_r)
        cz = cores_z[level]  # (rz_l, 2, rz_r)
        
        rx_l, _, rx_r = cx.shape
        ry_l, _, ry_r = cy.shape
        rz_l, _, rz_r = cz.shape
        
        if level == 0:
            # First level: input rank is 1
            # x-site: (1, 2, rx_r)
            result.append(cx.clone())
            
            # y-site: (rx_r, 2, rx_r * ry_r)
            # Connect x output to y input while keeping y's structure
            # The y core should be independent of x, but ranks multiply
            cy_3d = torch.zeros(rx_r, 2, rx_r * ry_r, device=device, dtype=dtype)
            for i in range(rx_r):
                # For each x output channel, replicate y structure
                cy_3d[i, :, i*ry_r:(i+1)*ry_r] = cy[0, :, :]  # cy[0] since ry_l=1 at level 0
            result.append(cy_3d)
            
            # z-site: (rx_r * ry_r, 2, rx_r * ry_r * rz_r)
            rxy = rx_r * ry_r
            cz_3d = torch.zeros(rxy, 2, rxy * rz_r, device=device, dtype=dtype)
            for i in range(rxy):
                cz_3d[i, :, i*rz_r:(i+1)*rz_r] = cz[0, :, :]  # cz[0] since rz_l=1 at level 0
            result.append(cz_3d)
            
        elif level == n_bits - 1:
            # Last level: output rank should be 1
            # Ranks: rx_l, ry_l, rz_l coming in; rx_r=ry_r=rz_r=1 going out
            
            prev_rank = result[-1].shape[2]  # This is rx_r * ry_r * rz_r from previous level
            
            # x-site: (prev_rank, 2, ry_l * rz_l)
            # After x, we still need to apply y and z which have input ranks ry_l, rz_l
            cx_3d = torch.zeros(prev_rank, 2, ry_l * rz_l, device=device, dtype=dtype)
            # prev_rank = rx_l * ry_l * rz_l (from previous level's z output)
            for i in range(prev_rank):
                # Decompose i into (ix, iy, iz) where i = ix * ry_l * rz_l + iy * rz_l + iz
                ix = i // (ry_l * rz_l)
                remainder = i % (ry_l * rz_l)
                # For x, we apply cx[ix, :, 0] (since rx_r = 1)
                # Output index is just (iy, iz) = remainder
                if ix < rx_l:
                    cx_3d[i, :, remainder] = cx[ix, :, 0]
            result.append(cx_3d)
            
            # y-site: (ry_l * rz_l, 2, rz_l)
            cy_3d = torch.zeros(ry_l * rz_l, 2, rz_l, device=device, dtype=dtype)
            for i in range(ry_l * rz_l):
                iy = i // rz_l
                iz = i % rz_l
                if iy < ry_l:
                    cy_3d[i, :, iz] = cy[iy, :, 0]  # ry_r = 1
            result.append(cy_3d)
            
            # z-site: (rz_l, 2, 1)
            cz_3d = torch.zeros(rz_l, 2, 1, device=device, dtype=dtype)
            for i in range(rz_l):
                cz_3d[i, :, 0] = cz[i, :, 0]  # rz_r = 1
            result.append(cz_3d)
            
        else:
            # Middle levels
            prev_rank = result[-1].shape[2]
            
            # x-site: (prev_rank, 2, new_x_rank)
            # prev_rank = rx_l * ry_l * rz_l (from previous z output)
            # new_x_rank = rx_r * ry_l * rz_l (x updated, y/z unchanged)
            new_x_rank = rx_r * ry_l * rz_l
            cx_3d = torch.zeros(prev_rank, 2, new_x_rank, device=device, dtype=dtype)
            for i in range(prev_rank):
                ix = i // (ry_l * rz_l)
                remainder = i % (ry_l * rz_l)
                if ix < rx_l:
                    # Map to output: ox * ry_l * rz_l + remainder
                    for ox in range(rx_r):
                        out_idx = ox * ry_l * rz_l + remainder
                        cx_3d[i, :, out_idx] = cx[ix, :, ox]
            result.append(cx_3d)
            
            # y-site: (new_x_rank, 2, new_y_rank)
            # new_y_rank = rx_r * ry_r * rz_l
            new_y_rank = rx_r * ry_r * rz_l
            cy_3d = torch.zeros(new_x_rank, 2, new_y_rank, device=device, dtype=dtype)
            for i in range(new_x_rank):
                ix = i // (ry_l * rz_l)
                remainder = i % (ry_l * rz_l)
                iy = remainder // rz_l
                iz = remainder % rz_l
                if iy < ry_l:
                    for oy in range(ry_r):
                        out_idx = ix * ry_r * rz_l + oy * rz_l + iz
                        cy_3d[i, :, out_idx] = cy[iy, :, oy]
            result.append(cy_3d)
            
            # z-site: (new_y_rank, 2, new_z_rank)
            # new_z_rank = rx_r * ry_r * rz_r
            new_z_rank = rx_r * ry_r * rz_r
            cz_3d = torch.zeros(new_y_rank, 2, new_z_rank, device=device, dtype=dtype)
            for i in range(new_y_rank):
                ix = i // (ry_r * rz_l)
                remainder = i % (ry_r * rz_l)
                iy = remainder // rz_l
                iz = remainder % rz_l
                if iz < rz_l:
                    for oz in range(rz_r):
                        out_idx = ix * ry_r * rz_r + iy * rz_r + oz
                        cz_3d[i, :, out_idx] = cz[iz, :, oz]
            result.append(cz_3d)
    
    return result


def interleave_3d_cores(
    cores_x: List[torch.Tensor],
    cores_y: List[torch.Tensor],
    cores_z: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Alias for separable_3d_qtt for backward compatibility."""
    return separable_3d_qtt(cores_x, cores_y, cores_z)


def taylor_green_analytical_3d(
    n_bits: int,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    L: float = 2 * np.pi,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """
    Construct Taylor-Green vortex velocity and vorticity fields analytically.
    
    NO DENSE MEMORY ALLOCATION.
    
    Taylor-Green vortex:
        u =  sin(x) * cos(y) * cos(z)
        v = -cos(x) * sin(y) * cos(z)
        w = 0
        
    Vorticity ω = ∇ × u:
        ω_x = ∂w/∂y - ∂v/∂z = sin(x) * sin(y) * sin(z)
        ω_y = ∂u/∂z - ∂w/∂x = sin(x) * sin(y) * sin(z)  
        ω_z = ∂v/∂x - ∂u/∂y = 0
        
    Wait, let me recalculate:
        u = sin(x)cos(y)cos(z)
        v = -cos(x)sin(y)cos(z)
        w = 0
        
        ∂u/∂y = -sin(x)sin(y)cos(z)
        ∂u/∂z = -sin(x)cos(y)sin(z)
        ∂v/∂x = sin(x)sin(y)cos(z)
        ∂v/∂z = cos(x)sin(y)sin(z)
        
        ω_x = ∂w/∂y - ∂v/∂z = 0 - cos(x)sin(y)sin(z) = -cos(x)sin(y)sin(z)
        ω_y = ∂u/∂z - ∂w/∂x = -sin(x)cos(y)sin(z) - 0 = -sin(x)cos(y)sin(z)
        ω_z = ∂v/∂x - ∂u/∂y = sin(x)sin(y)cos(z) - (-sin(x)sin(y)cos(z)) = 2*sin(x)sin(y)cos(z)
    
    Parameters
    ----------
    n_bits : int
        Grid resolution is 2^n_bits per dimension
    device : str
        Torch device
    dtype : torch.dtype
        Data type
    L : float
        Domain size (default 2π for standard Taylor-Green)
        
    Returns
    -------
    u_cores : List[List[torch.Tensor]]
        [u_x_cores, u_y_cores, u_z_cores]
    omega_cores : List[List[torch.Tensor]]
        [omega_x_cores, omega_y_cores, omega_z_cores]
    """
    k = 2 * np.pi / L  # wavenumber = 1 for standard Taylor-Green
    
    # Build 1D basis functions
    sin_x = sin_qtt_cores_1d(k, n_bits, L, device, dtype)
    cos_x = cos_qtt_cores_1d(k, n_bits, L, device, dtype)
    sin_y = sin_qtt_cores_1d(k, n_bits, L, device, dtype)
    cos_y = cos_qtt_cores_1d(k, n_bits, L, device, dtype)
    sin_z = sin_qtt_cores_1d(k, n_bits, L, device, dtype)
    cos_z = cos_qtt_cores_1d(k, n_bits, L, device, dtype)
    
    # Build velocity components as products
    # u = sin(x) * cos(y) * cos(z)
    u_x_cores = interleave_3d_cores(sin_x, cos_y, cos_z)
    
    # v = -cos(x) * sin(y) * cos(z)
    v_cores_unscaled = interleave_3d_cores(cos_x, sin_y, cos_z)
    u_y_cores = scale_qtt_cores(v_cores_unscaled, -1.0)
    
    # w = 0 (zero field - rank 1 with zeros)
    zero_1d = constant_qtt_cores_1d(0.0, n_bits, device, dtype)
    u_z_cores = interleave_3d_cores(zero_1d, zero_1d, zero_1d)
    
    # Build vorticity components
    # ω_x = -cos(x) * sin(y) * sin(z)
    omega_x_unscaled = interleave_3d_cores(cos_x, sin_y, sin_z)
    omega_x_cores = scale_qtt_cores(omega_x_unscaled, -1.0)
    
    # ω_y = -sin(x) * cos(y) * sin(z)
    omega_y_unscaled = interleave_3d_cores(sin_x, cos_y, sin_z)
    omega_y_cores = scale_qtt_cores(omega_y_unscaled, -1.0)
    
    # ω_z = 2 * sin(x) * sin(y) * cos(z)
    omega_z_unscaled = interleave_3d_cores(sin_x, sin_y, cos_z)
    omega_z_cores = scale_qtt_cores(omega_z_unscaled, 2.0)
    
    return (
        [u_x_cores, u_y_cores, u_z_cores],
        [omega_x_cores, omega_y_cores, omega_z_cores]
    )


def verify_sin_qtt(n_bits: int, k: float = 1.0, L: float = 2 * np.pi) -> float:
    """Verify sin QTT construction against dense computation."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    N = 1 << n_bits
    x = torch.linspace(0, L * (N-1) / N, N, device=device, dtype=dtype)
    
    # Dense reference
    sin_dense = torch.sin(k * x)
    
    # QTT reconstruction
    cores = sin_qtt_cores_1d(k, n_bits, L, device, dtype)
    
    # Contract QTT to get values
    # For each index i with binary representation b_0...b_{n-1}
    # value[i] = cores[0][:, b_0, :] @ cores[1][:, b_1, :] @ ... @ cores[n-1][:, b_{n-1}, :]
    
    sin_qtt = torch.zeros(N, device=device, dtype=dtype)
    
    for i in range(N):
        val = cores[0][0, (i >> 0) & 1, :]  # Shape: (r,)
        for j in range(1, n_bits):
            bit = (i >> j) & 1
            val = val @ cores[j][:, bit, :]  # (r,) @ (r, r') -> (r',)
        sin_qtt[i] = val.item() if val.numel() == 1 else val[0].item()
    
    error = torch.abs(sin_dense - sin_qtt).max().item()
    return error


def memory_estimate_analytical(n_bits: int) -> dict:
    """
    Estimate memory usage for analytical vs dense Taylor-Green initialization.
    """
    N = 1 << n_bits
    
    # Dense: 3 velocity + 3 vorticity components, each N^3 floats
    dense_bytes = 6 * (N ** 3) * 4  # float32
    
    # Analytical QTT: 6 fields, each with 3*n_bits cores
    # Each core is at most (4, 2, 4) = 32 floats for rank-4 (after Hadamard)
    # Actually for Taylor-Green, rank is about 8 due to 3-way products
    max_rank = 8
    qtt_bytes = 6 * 3 * n_bits * (max_rank * 2 * max_rank) * 4
    
    return {
        'n_bits': n_bits,
        'grid_size': f'{N}³',
        'cells': N ** 3,
        'dense_GB': dense_bytes / 1e9,
        'qtt_MB': qtt_bytes / 1e6,
        'compression': dense_bytes / qtt_bytes,
    }


if __name__ == '__main__':
    print("Analytical QTT Construction Test")
    print("=" * 60)
    
    # Test sin QTT accuracy
    for n_bits in [4, 6, 8, 10]:
        error = verify_sin_qtt(n_bits)
        print(f"sin QTT n_bits={n_bits:2d} (N={1<<n_bits:4d}): max_error = {error:.2e}")
    
    print()
    print("Memory Estimates for Taylor-Green:")
    print("-" * 60)
    
    for n_bits in [8, 9, 10, 11, 12]:
        est = memory_estimate_analytical(n_bits)
        print(f"{est['grid_size']:>8s}: Dense={est['dense_GB']:.2f} GB, QTT={est['qtt_MB']:.3f} MB, "
              f"Compression={est['compression']:.0f}x")
