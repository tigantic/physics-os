# Module `cfd.godunov`

Godunov-Type Riemann Solvers ============================

Numerical flux functions for the 1D Euler equations based on
solving the Riemann problem at cell interfaces.

Solvers implemented:
- Roe: Linearized Riemann solver with entropy fix
- HLL: Harten-Lax-van Leer two-wave approximation
- HLLC: HLL with contact wave restoration
- Exact: Newton iteration exact Riemann solver

For the Euler equations, the Riemann problem is:
    ∂U/∂t + ∂F/∂x = 0
with piecewise constant initial data U_L, U_R.

The solution consists of three waves:
1. Left-going wave (shock or rarefaction)
2. Contact discontinuity (entropy wave)
3. Right-going wave (shock or rarefaction)

**Contents:**

- [Functions](#functions)

## Functions

### `conserved_to_primitive`

```python
def conserved_to_primitive(U: torch.Tensor, gamma: float = 1.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Convert conserved (ρ, ρu, E) to primitive (ρ, u, p).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py:65](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py#L65)*

### `euler_flux`

```python
def euler_flux(U: torch.Tensor, gamma: float = 1.4) -> torch.Tensor
```

Physical flux for 1D Euler equations.

F = [ρu, ρu² + p, (E + p)u]ᵀ

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py:80](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py#L80)*

### `exact_riemann`

```python
def exact_riemann(rho_L: float, u_L: float, p_L: float, rho_R: float, u_R: float, p_R: float, gamma: float = 1.4, x: Optional[torch.Tensor] = None, t: float = 1.0, x0: float = 0.5, tol: float = 1e-08, max_iter: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Exact Riemann solver using Newton-Raphson iteration.

Solves for the star-state pressure p* at the contact,
then constructs the full solution.

**Parameters:**

- **gamma** (`<class 'float'>`): Ratio of specific heats
- **x** (`typing.Optional[torch.Tensor]`): Spatial coordinates for sampling solution
- **t** (`<class 'float'>`): Time at which to sample solution
- **x0** (`<class 'float'>`): Initial discontinuity location
- **tol** (`<class 'float'>`): Newton tolerance
- **max_iter** (`<class 'int'>`): Maximum Newton iterations

**Returns**: `typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` - (rho, u, p) sampled at x coordinates

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py:385](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py#L385)*

### `hll_flux`

```python
def hll_flux(U_L: torch.Tensor, U_R: torch.Tensor, gamma: float = 1.4) -> torch.Tensor
```

HLL (Harten-Lax-van Leer) approximate Riemann solver.

Uses two-wave approximation, ignoring the contact discontinuity.
Robust but diffusive for contact waves.

Wave speed estimates (Davis):
    S_L = min(u_L - a_L, u_R - a_R)
    S_R = max(u_L + a_L, u_R + a_R)

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py:248](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py#L248)*

### `hllc_flux`

```python
def hllc_flux(U_L: torch.Tensor, U_R: torch.Tensor, gamma: float = 1.4) -> torch.Tensor
```

HLLC (HLL-Contact) approximate Riemann solver.

Extends HLL by restoring the contact discontinuity,
giving exact resolution of isolated contact waves.

Three-wave structure: S_L | S_* | S_R

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py:301](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py#L301)*

### `primitive_to_conserved`

```python
def primitive_to_conserved(rho: torch.Tensor, u: torch.Tensor, p: torch.Tensor, gamma: float = 1.4) -> torch.Tensor
```

Convert primitive (ρ, u, p) to conserved (ρ, ρu, E).

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py:53](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py#L53)*

### `roe_flux`

```python
def roe_flux(U_L: torch.Tensor, U_R: torch.Tensor, gamma: float = 1.4, entropy_fix: bool = True, epsilon: float = 0.1) -> torch.Tensor
```

Roe's linearized Riemann solver.

Uses Roe-averaged quantities to construct an approximate
Jacobian with exact eigenstructure.

Roe averages:
    ρ̂ = √(ρ_L ρ_R)
    û = (√ρ_L u_L + √ρ_R u_R) / (√ρ_L + √ρ_R)
    Ĥ = (√ρ_L H_L + √ρ_R H_R) / (√ρ_L + √ρ_R)

**Parameters:**

- **U_L** (`<class 'torch.Tensor'>`): Left state (batch, 3) - conserved variables [ρ, ρu, E]
- **U_R** (`<class 'torch.Tensor'>`): Right state (batch, 3) - conserved variables [ρ, ρu, E]
- **gamma** (`<class 'float'>`): Ratio of specific heats (default 1.4 for air) Default: `1`.
- **entropy_fix** (`<class 'bool'>`): Apply Harten's entropy fix for expansion shocks
- **epsilon** (`<class 'float'>`): Entropy fix parameter (unused, kept for API compatibility)

**Returns**: `<class 'torch.Tensor'>` - Numerical flux (batch, 3)

**Raises:**

- `ValueError`: If U_L and U_R have different shapes

**Examples:**

```python
U_L = torch.tensor([[1.0, 0.0, 2.5]])
U_R = torch.tensor([[0.125, 0.0, 0.25]])
F = roe_flux(U_L, U_R)
print(f"Flux shape: {F.shape}")
```

*Source: [C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py:100](C:\TiganticLabz\Main_Projects\The Physics OS\ontic\cfd\godunov.py#L100)*
