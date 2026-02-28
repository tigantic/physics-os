# Tutorial: Compressible Flow Simulation with The Physics OS

This tutorial demonstrates how to use The Physics OS's CFD module to simulate compressible flows, including shock waves and supersonic aerodynamics.

## Overview

The Physics OS provides GPU-accelerated solvers for the compressible Euler equations:

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = 0$$

where U = (rho, rho*u, E) are the conserved variables.

## Quick Start: Sod Shock Tube

The Sod shock tube is a classic 1D Riemann problem:

```python
import torch
from ontic.cfd.euler_1d import Euler1D, EulerState

# Create solver
Nx = 400
solver = Euler1D(N=Nx, x_min=0.0, x_max=1.0, gamma=1.4, cfl=0.4)

# Initial condition: left and right states
x = solver.x_cell
rho = torch.where(x < 0.5, torch.ones_like(x), 0.125 * torch.ones_like(x))
u = torch.zeros_like(x)
p = torch.where(x < 0.5, torch.ones_like(x), 0.1 * torch.ones_like(x))

state = EulerState.from_primitive(rho, u, p, gamma=1.4)
solver.set_initial_condition(state)

# Run simulation
t_final = 0.2
while solver.t < t_final:
    solver.step()

print(f"Final time: {solver.t:.4f}")
```

### Visualizing Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(solver.x_cell, solver.state.rho)
axes[0, 0].set_ylabel('Density')

axes[0, 1].plot(solver.x_cell, solver.state.u)
axes[0, 1].set_ylabel('Velocity')

axes[1, 0].plot(solver.x_cell, solver.state.p)
axes[1, 0].set_ylabel('Pressure')

axes[1, 1].plot(solver.x_cell, solver.state.M)
axes[1, 1].set_ylabel('Mach Number')

for ax in axes.flat:
    ax.set_xlabel('x')
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## Riemann Solvers

The Physics OS provides several flux schemes:

```python
from ontic.cfd.godunov import exact_riemann, hll_flux, hllc_flux, roe_flux

# Exact Riemann solver (most accurate, slower)
rho, u, p = exact_riemann(rho_L, u_L, p_L, rho_R, u_R, p_R, x_over_t=0.0)

# Approximate solvers (faster)
F_hll = hll_flux(U_L, U_R, gamma=1.4)
F_hllc = hllc_flux(U_L, U_R, gamma=1.4)
F_roe = roe_flux(U_L, U_R, gamma=1.4)
```

## Oblique Shock Relations

For supersonic flow over a wedge, compute shock properties analytically:

```python
from ontic.cfd.euler_2d import oblique_shock_exact
import math

# Mach 5 flow over 15 degree wedge
M1 = 5.0
theta = math.radians(15.0)

result = oblique_shock_exact(M1=M1, theta=theta)

print(f"Shock angle: {math.degrees(result['beta']):.2f} deg")
print(f"Post-shock Mach: {result['M2']:.3f}")
print(f"Pressure ratio: {result['p2_p1']:.3f}")
print(f"Density ratio: {result['rho2_rho1']:.3f}")
```

Output:
```
Shock angle: 24.32 deg
Post-shock Mach: 3.504
Pressure ratio: 4.781
Density ratio: 2.753
```

## 2D Euler Solver

For 2D simulations with complex geometries:

```python
from ontic.cfd.euler_2d import Euler2D, Euler2DState, BCType
from ontic.cfd.geometry import WedgeGeometry, ImmersedBoundary

# Domain setup
Nx, Ny = 200, 100
Lx, Ly = 2.0, 1.0

# Initial uniform supersonic flow
gamma = 1.4
M_inf = 3.0

rho = torch.ones(Ny, Nx, dtype=torch.float64)
u = M_inf * torch.ones(Ny, Nx, dtype=torch.float64)
v = torch.zeros(Ny, Nx, dtype=torch.float64)
p = (1.0 / gamma) * torch.ones(Ny, Nx, dtype=torch.float64)

state = Euler2DState(rho, u, v, p)

# Create solver with boundary conditions
solver = Euler2D(
    state,
    dx=Lx/Nx,
    dy=Ly/Ny,
    bc_left=BCType.SUPERSONIC_INFLOW,
    bc_right=BCType.OUTFLOW,
    bc_bottom=BCType.REFLECTIVE,
    bc_top=BCType.OUTFLOW
)
solver.inflow_state = state

# Time stepping
for step in range(100):
    dt = solver.step(cfl=0.4)
```

## Immersed Boundary Method

For flow over solid bodies:

```python
import math
from ontic.cfd.geometry import WedgeGeometry, ImmersedBoundary

# Define wedge geometry
wedge = WedgeGeometry(
    x_leading_edge=0.3,
    y_leading_edge=0.5,
    half_angle=math.radians(15.0),
    length=0.5
)

# Create coordinate mesh
x = torch.linspace(0, Lx, Nx, dtype=torch.float64)
y = torch.linspace(0, Ly, Ny, dtype=torch.float64)
Y, X = torch.meshgrid(y, x, indexing='ij')

# Initialize immersed boundary
ib = ImmersedBoundary(wedge, X, Y)

# In time loop, apply IB after each step:
U = solver.state.to_conservative()
U = ib.apply(U)
```

## Boundary Conditions

Available boundary condition types:

| BCType | Description |
|--------|-------------|
| `PERIODIC` | Periodic (wrap-around) |
| `OUTFLOW` | Zero-gradient extrapolation |
| `REFLECTIVE` | Solid wall (velocity reflected) |
| `SUPERSONIC_INFLOW` | Fixed inflow state |
| `SUPERSONIC_OUTFLOW` | Zero-gradient (supersonic exit) |

## Slope Limiters

For high-resolution schemes:

```python
from ontic.cfd.limiters import minmod, superbee, van_leer, mc_limiter

# Minmod (most diffusive, most stable)
phi = minmod(r)

# Superbee (least diffusive)
phi = superbee(r)

# Van Leer (good balance)
phi = van_leer(r)
```

## Performance Tips

1. **GPU acceleration**: Use `device='cuda'` for large grids
2. **CFL number**: Start with CFL=0.4, reduce if unstable
3. **Grid resolution**: Start coarse, refine until solution converges
4. **Double precision**: Use `dtype=torch.float64` for shock-capturing

## References

1. Toro, E.F. "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009)
2. Anderson, J.D. "Modern Compressible Flow" (2003)
3. LeVeque, R.J. "Finite Volume Methods for Hyperbolic Problems" (2002)

## Next Steps

- See [wedge_flow_demo.py](../../scripts/wedge_flow_demo.py) for a working example
- Explore the [CFD API Reference](../api/cfd.euler_1d.md)
- Check [benchmarks/](../../Physics/benchmarks/) for validation against analytical solutions
