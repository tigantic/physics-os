# Module `cfd.limiters`

Slope Limiters for High-Resolution Schemes ==========================================

Limiters prevent spurious oscillations near discontinuities while
maintaining high-order accuracy in smooth regions.

TVD (Total Variation Diminishing) limiters satisfy:
    TV(u^{n+1}) ≤ TV(u^n)

where TV(u) = Σ|u_{i+1} - u_i|.

For a ratio r = (u_i - u_{i-1}) / (u_{i+1} - u_i):
- φ(r) = limiter function
- Limited slope: Δu = φ(r) * (u_{i+1} - u_i)

Sweby's TVD region: max(0, min(2r, 1)) ≤ φ(r) ≤ max(0, min(r, 2))

**Contents:**

- [Classes](#classes)
- [Functions](#functions)

## Classes

### class `MUSCL`

MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws).

Provides second-order reconstruction with slope limiting.

u_{i+1/2,L} = u_i + (1-κ)/4 * φ(r_i) * (u_{i+1} - u_i) + (1+κ)/4 * φ(1/r_i) * (u_i - u_{i-1})
u_{i+1/2,R} = u_{i+1} - (1+κ)/4 * φ(r_{i+1}) * (u_{i+2} - u_{i+1}) - (1-κ)/4 * φ(1/r_{i+1}) * (u_{i+1} - u_i)

κ = -1: Fully upwind (second-order upwind)
κ = 0: Fromm's scheme
κ = 1/3: Third-order upwind-biased
κ = 1: Central

#### Methods

##### `__init__`

```python
def __init__(self, limiter: str = 'van_leer', kappa: float = 0.3333333333333333)
```

Initialize MUSCL reconstruction.

**Parameters:**

- **limiter** (`<class 'str'>`): Slope limiter to use
- **kappa** (`<class 'float'>`): Interpolation parameter

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:248](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L248)*

##### `reconstruct`

```python
def reconstruct(self, u: torch.Tensor) -> tuple
```

Perform MUSCL reconstruction.

**Parameters:**

- **u** (`<class 'torch.Tensor'>`): Cell-centered values (N,) or (N, n_vars)

**Returns**: `<class 'tuple'>` - Left and right interface values at N-1 faces

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:270](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L270)*

## Functions

### `apply_limiter`

```python
def apply_limiter(u: torch.Tensor, limiter: str = 'minmod') -> torch.Tensor
```

Apply slope limiter to get limited gradients.

**Parameters:**

- **u** (`<class 'torch.Tensor'>`): Cell values (N,) or (N, n_vars)
- **limiter** (`<class 'str'>`): Limiter name ('minmod', 'superbee', 'van_leer', 'mc')

**Returns**: `<class 'torch.Tensor'>` - Limited slopes for reconstruction

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:175](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L175)*

### `compute_slope_ratio`

```python
def compute_slope_ratio(u: torch.Tensor, i: int) -> torch.Tensor
```

Compute slope ratio at cell i for limiter application.

r_i = (u_i - u_{i-1}) / (u_{i+1} - u_i)

Handles boundary cases with one-sided differences.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:147](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L147)*

### `koren`

```python
def koren(r: torch.Tensor) -> torch.Tensor
```

Koren limiter (third-order accurate in smooth regions).

φ(r) = max(0, min(2r, (2 + r)/3, 2))

Optimized for third-order upstream-biased interpolation.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:118](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L118)*

### `mc_limiter`

```python
def mc_limiter(r: torch.Tensor) -> torch.Tensor
```

MC (Monotonized Central) limiter.

φ(r) = max(0, min(2r, (1+r)/2, 2))

Symmetric limiter that uses the central difference when possible.
Good balance of accuracy and stability.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:100](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L100)*

### `minmod`

```python
def minmod(r: torch.Tensor) -> torch.Tensor
```

Minmod limiter.

φ(r) = max(0, min(1, r))

Most diffusive TVD limiter. Very robust but only first-order
at smooth extrema.

**Parameters:**

- **r** (`<class 'torch.Tensor'>`): Ratio of consecutive gradients

**Returns**: `<class 'torch.Tensor'>` - Limiter value

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:25](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L25)*

### `minmod_3`

```python
def minmod_3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor
```

Three-argument minmod function.

minmod(a, b, c) = sign(a) * min(|a|, |b|, |c|) if signs agree, else 0

Used in higher-order reconstructions.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:43](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L43)*

### `ospre`

```python
def ospre(r: torch.Tensor) -> torch.Tensor
```

OSPRE limiter (Waterson & Deconinck, 2007).

φ(r) = 1.5 * (r² + r) / (r² + r + 1)

Smooth, symmetric limiter with good convergence properties.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:135](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L135)*

### `superbee`

```python
def superbee(r: torch.Tensor) -> torch.Tensor
```

Superbee limiter (Roe, 1985).

φ(r) = max(0, min(2r, 1), min(r, 2))

Most compressive TVD limiter. Excellent for contact discontinuities
but can steepen smooth waves too aggressively.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:58](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L58)*

### `van_albada`

```python
def van_albada(r: torch.Tensor, epsilon: float = 1e-06) -> torch.Tensor
```

Van Albada limiter.

φ(r) = (r² + r) / (r² + 1)

Differentiable limiter, useful for implicit schemes.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:88](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L88)*

### `van_leer`

```python
def van_leer(r: torch.Tensor) -> torch.Tensor
```

Van Leer limiter.

φ(r) = (r + |r|) / (1 + |r|)

Smooth limiter with good balance between accuracy and
oscillation suppression.

*Source: [C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py:76](C:\TiganticLabz\Main_Projects\Project HyperTensor\tensornet\cfd\limiters.py#L76)*
