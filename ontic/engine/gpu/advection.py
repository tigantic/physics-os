"""
Phase 2C-6: CUDA Accelerated Advection Module
==============================================

Production-ready advection module with automatic GPU detection
and fallback to PyTorch CPU/GPU implementation.

Usage:
    from ontic.engine.gpu.advection import advect_2d, advect_velocity_2d

    # Automatically uses CUDA kernel if available
    result = advect_2d(density, velocity, dt)
"""

import os
import sys

import torch
from torch import Tensor

# ═══════════════════════════════════════════════════════════════════════════
# CUDA Extension Loading
# ═══════════════════════════════════════════════════════════════════════════

_CUDA_AVAILABLE = False
_tensornet_cuda = None


def _load_cuda_extension():
    """Attempt to load the CUDA extension."""
    global _CUDA_AVAILABLE, _tensornet_cuda

    if not torch.cuda.is_available():
        return False

    try:
        # Add cuda directory to path
        cuda_dir = os.path.join(os.path.dirname(__file__), "..", "cuda")
        if cuda_dir not in sys.path:
            sys.path.insert(0, cuda_dir)

        import tensornet_cuda

        _tensornet_cuda = tensornet_cuda
        _CUDA_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"[WARNING] CUDA extension not available: {e}")
        print("  Falling back to PyTorch implementation")
        return False


# Try to load on module import
_load_cuda_extension()


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available."""
    return _CUDA_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Reference Implementations (Fallback)
# ═══════════════════════════════════════════════════════════════════════════


def _advect_2d_pytorch(
    density: Tensor,
    velocity: Tensor,
    dt: float,
) -> Tensor:
    """
    PyTorch implementation of 2D Semi-Lagrangian advection.
    Works on both CPU and GPU.
    """
    height, width = density.shape
    device = density.device

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )

    # Extract velocity components
    u_vel = velocity[0]
    v_vel = velocity[1]

    # Backtrace
    src_x = x_coords - u_vel * dt
    src_y = y_coords - v_vel * dt

    # Clamp
    src_x = torch.clamp(src_x, 0, width - 1.001)
    src_y = torch.clamp(src_y, 0, height - 1.001)

    # Bilinear interpolation
    x0 = src_x.long()
    y0 = src_y.long()
    x1 = torch.clamp(x0 + 1, 0, width - 1)
    y1 = torch.clamp(y0 + 1, 0, height - 1)

    dx = src_x - x0.float()
    dy = src_y - y0.float()

    v00 = density[y0, x0]
    v10 = density[y0, x1]
    v01 = density[y1, x0]
    v11 = density[y1, x1]

    top = v00 * (1 - dx) + v10 * dx
    bot = v01 * (1 - dx) + v11 * dx

    return top * (1 - dy) + bot * dy


def _advect_velocity_2d_pytorch(
    velocity: Tensor,
    dt: float,
) -> Tensor:
    """
    PyTorch implementation of velocity self-advection.
    """
    result = torch.zeros_like(velocity)
    result[0] = _advect_2d_pytorch(velocity[0], velocity, dt)
    result[1] = _advect_2d_pytorch(velocity[1], velocity, dt)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Public API - Auto-dispatches to CUDA or PyTorch
# ═══════════════════════════════════════════════════════════════════════════


def advect_2d(
    density: Tensor,
    velocity: Tensor,
    dt: float,
    force_pytorch: bool = False,
) -> Tensor:
    """
    Advect a 2D scalar field using Semi-Lagrangian method.

    Automatically uses CUDA kernel if available and data is on GPU.

    Args:
        density: Scalar field [H, W], float32
        velocity: Velocity field [2, H, W], float32
        dt: Time step
        force_pytorch: Force PyTorch implementation (for debugging)

    Returns:
        Advected density field [H, W]

    Performance:
        CUDA kernel: ~0.01 ms for 512x512
        PyTorch GPU: ~0.3 ms for 512x512
        Speedup: ~30x
    """
    # Validate inputs
    if density.dim() != 2:
        raise ValueError(f"density must be 2D, got {density.dim()}D")
    if velocity.dim() != 3 or velocity.size(0) != 2:
        raise ValueError(f"velocity must be [2, H, W], got {velocity.shape}")

    # Convert to float32 if needed
    if density.dtype != torch.float32:
        density = density.float()
    if velocity.dtype != torch.float32:
        velocity = velocity.float()

    # Dispatch to appropriate implementation
    use_cuda = (
        _CUDA_AVAILABLE and not force_pytorch and density.is_cuda and velocity.is_cuda
    )

    if use_cuda:
        return _tensornet_cuda.advect_2d(density, velocity, dt)
    else:
        return _advect_2d_pytorch(density, velocity, dt)


def advect_velocity_2d(
    velocity: Tensor,
    dt: float,
    force_pytorch: bool = False,
) -> Tensor:
    """
    Advect a 2D velocity field by itself (self-advection).

    Used for: u^{n+1} = u^n - dt * (u · ∇)u

    Args:
        velocity: Velocity field [2, H, W], float32
        dt: Time step
        force_pytorch: Force PyTorch implementation

    Returns:
        Advected velocity field [2, H, W]
    """
    if velocity.dim() != 3 or velocity.size(0) != 2:
        raise ValueError(f"velocity must be [2, H, W], got {velocity.shape}")

    if velocity.dtype != torch.float32:
        velocity = velocity.float()

    use_cuda = _CUDA_AVAILABLE and not force_pytorch and velocity.is_cuda

    if use_cuda:
        return _tensornet_cuda.advect_velocity_2d(velocity, dt)
    else:
        return _advect_velocity_2d_pytorch(velocity, dt)


def advect_3d(
    density: Tensor,
    velocity: Tensor,
    dt: float,
    force_pytorch: bool = False,
) -> Tensor:
    """
    Advect a 3D scalar field using Semi-Lagrangian method.

    Args:
        density: Scalar field [D, H, W], float32
        velocity: Velocity field [3, D, H, W], float32
        dt: Time step
        force_pytorch: Force PyTorch implementation

    Returns:
        Advected density field [D, H, W]
    """
    if density.dim() != 3:
        raise ValueError(f"density must be 3D, got {density.dim()}D")
    if velocity.dim() != 4 or velocity.size(0) != 3:
        raise ValueError(f"velocity must be [3, D, H, W], got {velocity.shape}")

    if density.dtype != torch.float32:
        density = density.float()
    if velocity.dtype != torch.float32:
        velocity = velocity.float()

    use_cuda = (
        _CUDA_AVAILABLE and not force_pytorch and density.is_cuda and velocity.is_cuda
    )

    if use_cuda:
        return _tensornet_cuda.advect_3d(density, velocity, dt)
    else:
        # PyTorch 3D fallback using semi-Lagrangian advection
        return _advect_3d_pytorch(density, velocity, dt)


def _advect_3d_pytorch(
    density: Tensor,
    velocity: Tensor,
    dt: float,
) -> Tensor:
    """
    PyTorch implementation of 3D semi-Lagrangian advection.
    
    Uses trilinear interpolation for backtraced positions.
    """
    depth, height, width = density.shape
    device = density.device
    
    # Create coordinate grids
    z_coords, y_coords, x_coords = torch.meshgrid(
        torch.arange(depth, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    
    # Extract velocity components
    u_vel = velocity[0]  # x velocity
    v_vel = velocity[1]  # y velocity
    w_vel = velocity[2]  # z velocity
    
    # Backtrace to find source positions
    src_x = x_coords - u_vel * dt
    src_y = y_coords - v_vel * dt
    src_z = z_coords - w_vel * dt
    
    # Clamp to valid range
    src_x = torch.clamp(src_x, 0, width - 1.001)
    src_y = torch.clamp(src_y, 0, height - 1.001)
    src_z = torch.clamp(src_z, 0, depth - 1.001)
    
    # Trilinear interpolation indices
    x0 = src_x.long()
    y0 = src_y.long()
    z0 = src_z.long()
    
    x1 = torch.clamp(x0 + 1, 0, width - 1)
    y1 = torch.clamp(y0 + 1, 0, height - 1)
    z1 = torch.clamp(z0 + 1, 0, depth - 1)
    
    # Interpolation weights
    dx = src_x - x0.float()
    dy = src_y - y0.float()
    dz = src_z - z0.float()
    
    # Sample 8 corner values
    v000 = density[z0, y0, x0]
    v100 = density[z0, y0, x1]
    v010 = density[z0, y1, x0]
    v110 = density[z0, y1, x1]
    v001 = density[z1, y0, x0]
    v101 = density[z1, y0, x1]
    v011 = density[z1, y1, x0]
    v111 = density[z1, y1, x1]
    
    # Trilinear interpolation
    # First interpolate along x
    c00 = v000 * (1 - dx) + v100 * dx
    c10 = v010 * (1 - dx) + v110 * dx
    c01 = v001 * (1 - dx) + v101 * dx
    c11 = v011 * (1 - dx) + v111 * dx
    
    # Then interpolate along y
    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy
    
    # Finally interpolate along z
    return c0 * (1 - dz) + c1 * dz


# ═══════════════════════════════════════════════════════════════════════════
# GPU Diagnostics
# ═══════════════════════════════════════════════════════════════════════════


def print_gpu_status():
    """Print GPU acceleration status."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║              GPU ACCELERATION STATUS                       ║")
    print("╚════════════════════════════════════════════════════════════╝")

    print(f"  PyTorch CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU Device:   {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Compute Cap:  {torch.cuda.get_device_capability()}")

    print(f"  CUDA Kernels: {'✓ LOADED' if _CUDA_AVAILABLE else '✗ NOT LOADED'}")

    if _CUDA_AVAILABLE:
        print("  Functions:    advect_2d, advect_velocity_2d, advect_3d")


if __name__ == "__main__":
    print_gpu_status()
