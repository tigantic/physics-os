"""
Phase 3A-2: Trajectory Optimizer
================================

Finds optimal flight path through the hazard field using gradient-based
optimization. The trajectory is parameterized as a smooth curve and
optimized to minimize integrated cost while respecting vehicle dynamics.

Methods:
    1. Gradient Descent: Direct optimization of waypoint positions
    2. Tensor Train Path: Represent trajectory in TT format for global search
    3. Fast Marching: Find shortest path in cost-weighted space

Reference:
    - Betts, Practical Methods for Optimal Control (2010)
    - Sethian, Level Set Methods and Fast Marching Methods (1999)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tensornet.physics.hypersonic import HazardField

# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Waypoint:
    """Single trajectory waypoint."""

    lat: float  # Latitude (degrees)
    lon: float  # Longitude (degrees)
    alt: float  # Altitude (meters)
    time: float  # Time since trajectory start (seconds)

    def to_tensor(self, device: torch.device = None) -> Tensor:
        return torch.tensor([self.lat, self.lon, self.alt], device=device)

    @classmethod
    def from_tensor(cls, t: Tensor, time: float = 0.0) -> "Waypoint":
        return cls(lat=t[0].item(), lon=t[1].item(), alt=t[2].item(), time=time)


@dataclass
class Trajectory:
    """Complete flight trajectory."""

    waypoints: list[Waypoint]
    total_cost: float
    path_length: float
    computation_time_ms: float
    converged: bool
    iterations: int

    def to_tensor(self, device: torch.device = None) -> Tensor:
        """Convert to [N, 3] tensor."""
        return torch.stack([w.to_tensor(device) for w in self.waypoints])

    @classmethod
    def from_tensor(cls, path: Tensor, dt: float = 1.0, **kwargs) -> "Trajectory":
        """Create from [N, D] tensor (D=2 or D=3)."""
        waypoints = []
        ndim = path.shape[1]
        for i, p in enumerate(path):
            waypoints.append(
                Waypoint(
                    lat=p[0].item(),
                    lon=p[1].item() if ndim > 1 else 0.0,
                    alt=p[2].item() if ndim > 2 else 0.0,
                    time=i * dt,
                )
            )
        return cls(waypoints=waypoints, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Cost Sampling
# ═══════════════════════════════════════════════════════════════════════════


def sample_cost_along_path(
    path: Tensor,
    cost_field: Tensor,
    bounds: tuple[tuple[float, float], ...],
) -> Tensor:
    """
    Sample cost field values along a path using trilinear interpolation.

    Args:
        path: [N, 3] or [N, 2] trajectory points (normalized to [0, 1])
        cost_field: [D, H, W] or [H, W] cost tensor
        bounds: ((x_min, x_max), (y_min, y_max), ...) world coordinates

    Returns:
        [N] cost values along path
    """
    device = path.device
    ndim = path.shape[-1]

    # Normalize path to [-1, 1] for grid_sample
    path_normalized = path.clone()
    for i in range(ndim):
        lo, hi = bounds[i]
        path_normalized[..., i] = 2.0 * (path[..., i] - lo) / (hi - lo) - 1.0

    if ndim == 2:
        # 2D case
        grid = path_normalized.view(1, -1, 1, 2)  # [1, N, 1, 2]
        cost_4d = cost_field.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        sampled = F.grid_sample(
            cost_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled.view(-1)
    else:
        # 3D case
        grid = path_normalized.view(1, 1, -1, 1, 3)  # [1, 1, N, 1, 3]
        cost_5d = cost_field.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        sampled = F.grid_sample(
            cost_5d, grid, mode="trilinear", padding_mode="border", align_corners=True
        )
        return sampled.view(-1)


def compute_path_cost(
    path: Tensor,
    cost_field: Tensor,
    bounds: tuple[tuple[float, float], ...],
    smoothness_weight: float = 0.1,
) -> Tensor:
    """
    Compute total cost for a path.

    Cost = ∫ C(x) ds + λ * ∫ κ² ds

    Where:
        - C(x) = hazard cost at position x
        - κ = path curvature (smoothness penalty)

    Args:
        path: [N, D] trajectory waypoints
        cost_field: [shape] cost tensor
        bounds: World coordinate bounds
        smoothness_weight: Weight for curvature penalty

    Returns:
        Scalar total cost
    """
    # Sample cost along path
    costs = sample_cost_along_path(path, cost_field, bounds)

    # Integrated cost (trapezoidal rule)
    segment_lengths = torch.norm(path[1:] - path[:-1], dim=-1)
    integrated_cost = torch.sum((costs[:-1] + costs[1:]) / 2 * segment_lengths)

    # Curvature penalty (second derivative)
    if path.shape[0] > 2:
        d1 = path[1:] - path[:-1]  # First derivative
        d2 = d1[1:] - d1[:-1]  # Second derivative
        curvature_sq = torch.sum(d2**2, dim=-1)
        curvature_penalty = torch.sum(curvature_sq)
    else:
        curvature_penalty = torch.tensor(0.0, device=path.device)

    return integrated_cost + smoothness_weight * curvature_penalty


# ═══════════════════════════════════════════════════════════════════════════
# Gradient-Based Optimizer
# ═══════════════════════════════════════════════════════════════════════════


def optimize_trajectory_gradient(
    start: Tensor,
    end: Tensor,
    cost_field: Tensor,
    bounds: tuple[tuple[float, float], ...],
    num_waypoints: int = 100,
    max_iterations: int = 500,
    learning_rate: float = 0.01,
    smoothness_weight: float = 0.1,
    convergence_tol: float = 1e-4,
    verbose: bool = False,
) -> Trajectory:
    """
    Optimize trajectory using gradient descent.

    The path is initialized as a straight line from start to end,
    then iteratively adjusted to minimize cost while maintaining smoothness.

    Args:
        start: [D] starting point
        end: [D] ending point
        cost_field: [shape] hazard cost tensor
        bounds: World coordinate bounds
        num_waypoints: Number of waypoints in trajectory
        max_iterations: Maximum optimization iterations
        learning_rate: Gradient descent step size
        smoothness_weight: Curvature penalty weight
        convergence_tol: Stop when cost change < tol
        verbose: Print progress

    Returns:
        Optimized Trajectory
    """
    import time

    t_start = time.perf_counter()

    device = cost_field.device
    ndim = start.shape[0]

    # Initialize as straight line
    t = torch.linspace(0, 1, num_waypoints, device=device).unsqueeze(1)
    path = start.unsqueeze(0) + t * (end - start).unsqueeze(0)

    # Make interior points optimizable (keep endpoints fixed)
    interior = path[1:-1].clone().requires_grad_(True)

    optimizer = torch.optim.Adam([interior], lr=learning_rate)

    costs_history = []
    prev_cost = float("inf")

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Reconstruct full path
        full_path = torch.cat([start.unsqueeze(0), interior, end.unsqueeze(0)], dim=0)

        # Compute cost
        cost = compute_path_cost(full_path, cost_field, bounds, smoothness_weight)

        # Backward
        cost.backward()

        # Update
        optimizer.step()

        cost_val = cost.item()
        costs_history.append(cost_val)

        if verbose and iteration % 50 == 0:
            print(f"  Iter {iteration:4d}: Cost = {cost_val:.4f}")

        # Check convergence
        if abs(prev_cost - cost_val) < convergence_tol:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break

        prev_cost = cost_val

    # Final path
    with torch.no_grad():
        final_path = torch.cat([start.unsqueeze(0), interior, end.unsqueeze(0)], dim=0)
        path_length = torch.sum(torch.norm(final_path[1:] - final_path[:-1], dim=-1))

    t_end = time.perf_counter()

    return Trajectory.from_tensor(
        final_path.detach(),
        dt=1.0,
        total_cost=costs_history[-1] if costs_history else float("inf"),
        path_length=path_length.item(),
        computation_time_ms=(t_end - t_start) * 1000,
        converged=iteration < max_iterations - 1,
        iterations=iteration + 1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fast Marching Method (Eikonal Solver)
# ═══════════════════════════════════════════════════════════════════════════


def fast_marching_2d(
    cost_field: Tensor,
    start_idx: tuple[int, int],
    end_idx: tuple[int, int],
) -> tuple[Tensor, list[tuple[int, int]]]:
    """
    Find shortest path through cost field using Fast Marching.

    Solves the Eikonal equation: |∇T| = C(x)
    Where T is the arrival time and C is the cost.

    Args:
        cost_field: [H, W] cost tensor (higher = slower)
        start_idx: (row, col) starting cell
        end_idx: (row, col) ending cell

    Returns:
        (arrival_time_field, path_indices)
    """
    device = cost_field.device
    H, W = cost_field.shape

    # Initialize arrival time (infinity everywhere except start)
    T = torch.full((H, W), float("inf"), device=device)
    T[start_idx] = 0.0

    # Frozen mask (True = finalized)
    frozen = torch.zeros(H, W, dtype=torch.bool, device=device)

    # Narrow band (active cells)
    import heapq

    heap = [(0.0, start_idx[0], start_idx[1])]

    # Neighbor offsets
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        t_curr, r, c = heapq.heappop(heap)

        if frozen[r, c]:
            continue

        frozen[r, c] = True

        # Check if we reached the end
        if (r, c) == end_idx:
            break

        # Update neighbors
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc

            if 0 <= nr < H and 0 <= nc < W and not frozen[nr, nc]:
                # Compute new arrival time
                cost = cost_field[nr, nc].item()
                if cost == float("inf") or cost < 0:
                    continue

                # Simple update: T_new = T_curr + cost
                # (Full Eikonal uses quadratic update)
                t_new = T[r, c].item() + cost

                if t_new < T[nr, nc].item():
                    T[nr, nc] = t_new
                    heapq.heappush(heap, (t_new, nr, nc))

    # Backtrack from end to start
    path = [end_idx]
    r, c = end_idx

    while (r, c) != start_idx:
        best_t = float("inf")
        best_n = None

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if T[nr, nc].item() < best_t:
                    best_t = T[nr, nc].item()
                    best_n = (nr, nc)

        if best_n is None:
            break  # No valid path

        path.append(best_n)
        r, c = best_n

    path.reverse()
    return T, path


def optimize_trajectory_fast_marching(
    start: Tensor,
    end: Tensor,
    cost_field: Tensor,
    bounds: tuple[tuple[float, float], ...],
    num_waypoints: int = 100,
) -> Trajectory:
    """
    Find optimal path using Fast Marching Method.

    Faster than gradient descent for finding global optimum,
    but gives grid-aligned path that needs smoothing.

    Args:
        start: [D] starting point (in world coordinates)
        end: [D] ending point
        cost_field: [H, W] cost tensor
        bounds: World coordinate bounds
        num_waypoints: Resample to this many waypoints

    Returns:
        Optimized Trajectory
    """
    import time

    t_start = time.perf_counter()

    device = cost_field.device
    H, W = cost_field.shape[:2]

    # Convert world coordinates to grid indices
    def world_to_grid(p: Tensor) -> tuple[int, int]:
        r = int((p[0].item() - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) * (H - 1))
        c = int((p[1].item() - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) * (W - 1))
        return (max(0, min(H - 1, r)), max(0, min(W - 1, c)))

    def grid_to_world(r: int, c: int) -> Tensor:
        y = bounds[0][0] + r / (H - 1) * (bounds[0][1] - bounds[0][0])
        x = bounds[1][0] + c / (W - 1) * (bounds[1][1] - bounds[1][0])
        return torch.tensor([y, x], device=device)

    start_idx = world_to_grid(start)
    end_idx = world_to_grid(end)

    # Run fast marching
    T, path_indices = fast_marching_2d(cost_field, start_idx, end_idx)

    # Convert to world coordinates
    path_world = torch.stack([grid_to_world(r, c) for r, c in path_indices])

    # Resample to desired number of waypoints
    if len(path_indices) > 2:
        # Interpolate along path using numpy (torch.interp not available)
        import numpy as np

        t_orig = np.linspace(0, 1, len(path_indices))
        t_new = np.linspace(0, 1, num_waypoints)

        path_np = path_world.cpu().numpy()
        path_resampled_np = np.zeros((num_waypoints, 2))
        for d in range(2):
            path_resampled_np[:, d] = np.interp(t_new, t_orig, path_np[:, d])
        path_resampled = torch.tensor(
            path_resampled_np, device=device, dtype=torch.float32
        )
    else:
        path_resampled = path_world

    # Compute path cost and length
    with torch.no_grad():
        costs = sample_cost_along_path(path_resampled, cost_field, bounds)
        total_cost = costs.sum().item()
        path_length = torch.sum(
            torch.norm(path_resampled[1:] - path_resampled[:-1], dim=-1)
        )

    t_end = time.perf_counter()

    return Trajectory.from_tensor(
        path_resampled,
        dt=1.0,
        total_cost=total_cost,
        path_length=path_length.item(),
        computation_time_ms=(t_end - t_start) * 1000,
        converged=True,
        iterations=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# High-Level API
# ═══════════════════════════════════════════════════════════════════════════


def find_optimal_trajectory(
    hazard: HazardField,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    bounds: tuple[tuple[float, float], ...],
    method: str = "gradient",
    num_waypoints: int = 100,
    **kwargs,
) -> Trajectory:
    """
    Find optimal trajectory through hazard field.

    This is the main entry point for trajectory optimization.

    Args:
        hazard: HazardField from calculate_hazard_field
        start: (lat, lon, alt) starting point
        end: (lat, lon, alt) ending point
        bounds: ((lat_min, lat_max), (lon_min, lon_max), (alt_min, alt_max))
        method: 'gradient' or 'fast_marching'
        num_waypoints: Number of output waypoints
        **kwargs: Additional optimizer arguments

    Returns:
        Optimized Trajectory

    Example:
        >>> hazard = calculate_hazard_field(density, wind_u, wind_v)
        >>> trajectory = find_optimal_trajectory(
        ...     hazard,
        ...     start=(30.0, -100.0, 25000.0),
        ...     end=(35.0, -90.0, 30000.0),
        ...     bounds=((25, 40), (-110, -85), (20000, 40000)),
        ... )
        >>> print(f"Path cost: {trajectory.total_cost:.2f}")
    """
    device = hazard.device
    cost_field = hazard.total_cost

    # Convert to tensors
    ndim = len(start)
    start_t = torch.tensor(
        start[: cost_field.dim()], device=device, dtype=torch.float32
    )
    end_t = torch.tensor(end[: cost_field.dim()], device=device, dtype=torch.float32)

    if method == "gradient":
        return optimize_trajectory_gradient(
            start_t,
            end_t,
            cost_field,
            bounds[: cost_field.dim()],
            num_waypoints=num_waypoints,
            **kwargs,
        )
    elif method == "fast_marching":
        return optimize_trajectory_fast_marching(
            start_t,
            end_t,
            cost_field,
            bounds[: cost_field.dim()],
            num_waypoints=num_waypoints,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════


def demo_trajectory_optimizer():
    """Demonstrate trajectory optimization through synthetic weather."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create synthetic hazard field
    H, W = 128, 256

    # Base cost: low everywhere
    cost = torch.ones(H, W, device=device) * 0.1

    # Add storm regions (high cost)
    x = torch.linspace(0, 1, W, device=device)
    y = torch.linspace(0, 1, H, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Storm 1: blocks direct path
    storm1 = 50.0 * torch.exp(-((X - 0.4) ** 2 + (Y - 0.5) ** 2) / 0.02)

    # Storm 2: secondary obstacle
    storm2 = 30.0 * torch.exp(-((X - 0.7) ** 2 + (Y - 0.3) ** 2) / 0.03)

    # Jet stream corridor (low cost)
    corridor = -0.05 * torch.exp(-((Y - 0.7) ** 2) / 0.01)

    cost = cost + storm1 + storm2 + corridor
    cost = torch.clamp(cost, 0.01, 100.0)

    # Create mock hazard field
    hazard = HazardField(
        total_cost=cost,
        q_cost=cost,
        thermal_cost=cost * 0,
        shear_cost=cost * 0,
        dynamic_pressure=cost * 1000,
        stagnation_temp=cost * 100,
        wind_shear=cost * 0.01,
        grid_shape=(H, W),
        device=device,
    )

    # Define start and end points
    start = (0.1, 0.1)  # Bottom-left
    end = (0.9, 0.9)  # Top-right
    bounds = ((0.0, 1.0), (0.0, 1.0))

    print("\n" + "═" * 60)
    print("TRAJECTORY OPTIMIZATION TEST")
    print("═" * 60)

    # Method 1: Gradient descent
    print("\n1. Gradient Descent Optimizer:")
    traj_grad = find_optimal_trajectory(
        hazard,
        start + (0.0,),
        end + (0.0,),
        bounds + ((0, 1),),
        method="gradient",
        num_waypoints=50,
        max_iterations=200,
        verbose=True,
    )
    print(f"   Cost: {traj_grad.total_cost:.4f}")
    print(f"   Path Length: {traj_grad.path_length:.4f}")
    print(f"   Time: {traj_grad.computation_time_ms:.2f} ms")
    print(f"   Converged: {traj_grad.converged} ({traj_grad.iterations} iterations)")

    # Method 2: Fast marching
    print("\n2. Fast Marching Method:")
    traj_fm = find_optimal_trajectory(
        hazard,
        start + (0.0,),
        end + (0.0,),
        bounds + ((0, 1),),
        method="fast_marching",
        num_waypoints=50,
    )
    print(f"   Cost: {traj_fm.total_cost:.4f}")
    print(f"   Path Length: {traj_fm.path_length:.4f}")
    print(f"   Time: {traj_fm.computation_time_ms:.2f} ms")

    # Output waypoints
    print("\n" + "═" * 60)
    print("SAMPLE WAYPOINTS (first 10):")
    print("═" * 60)
    print(f"{'Idx':>4} │ {'Lat':>8} │ {'Lon':>8} │ {'Time':>6}")
    print("─" * 35)
    for i, wp in enumerate(traj_grad.waypoints[:10]):
        print(f"{i:>4} │ {wp.lat:>8.4f} │ {wp.lon:>8.4f} │ {wp.time:>6.1f}")

    return traj_grad, traj_fm


if __name__ == "__main__":
    demo_trajectory_optimizer()
