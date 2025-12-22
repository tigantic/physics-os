# Technical Notes

## Mathematical Derivations for NS-Millennium

Working notes on the mathematical foundations of the χ-regularity approach.

---

## 1. QTT Representation of 3D Velocity Fields

### 1.1 Discretization Setup

Consider the velocity field $\mathbf{u}(\mathbf{x}) = (u, v, w)$ on the domain $\Omega = [0, L]^3$.

Discretize with $N = 2^n$ points per dimension:
- Total grid points: $N^3 = 2^{3n}$
- Degrees of freedom: $3 \cdot N^3$ (three velocity components)

### 1.2 QTT Encoding

Each component $u(x, y, z)$ is a 3D tensor with indices $(i, j, k)$ where $i, j, k \in \{0, \ldots, N-1\}$.

Binary encoding: $i = \sum_{l=1}^{n} i_l 2^{l-1}$ with $i_l \in \{0, 1\}$.

The QTT representation:
$$u_{i,j,k} = \sum_{\alpha} G^{(1)}_{\alpha_0, i_1, \alpha_1} G^{(2)}_{\alpha_1, i_2, \alpha_2} \cdots G^{(3n)}_{\alpha_{3n-1}, k_n, \alpha_{3n}}$$

**Storage:** $O(3n \cdot \chi^2) = O(\log N \cdot d \cdot \chi^2)$ where $d = 3$ spatial dimensions.

### 1.3 Physical Interpretation

The bond dimension $\chi$ at each cut represents the entanglement between length scales:
- Low $\chi$: solution separable across scales (smooth)
- High $\chi$: strong cross-scale correlations (turbulent/singular)

---

## 2. Incompressibility Constraint in QTT

### 2.1 The Constraint

Incompressible flow requires:
$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

### 2.2 Challenge in QTT Format

The divergence involves derivatives, which in QTT format become MPO applications:
$$\frac{\partial}{\partial x} \leftrightarrow D_x \text{ (differentiation MPO)}$$

Enforcing $\nabla \cdot \mathbf{u} = 0$ exactly in QTT is non-trivial.

### 2.3 Approaches

**Option A: Projection Method**
1. Evolve velocity (may violate incompressibility)
2. Solve Poisson equation: $\nabla^2 \phi = \nabla \cdot \mathbf{u}$
3. Project: $\mathbf{u} \leftarrow \mathbf{u} - \nabla \phi$

Challenge: Poisson solve in QTT format.

**Option B: Stream Function / Vorticity**
- Work with $\boldsymbol{\omega} = \nabla \times \mathbf{u}$
- Recover $\mathbf{u}$ via Biot-Savart
- Automatically divergence-free

Challenge: Biot-Savart is nonlocal.

**Option C: Penalty Method**
- Add term $-\lambda \nabla(\nabla \cdot \mathbf{u})$ to momentum equation
- Drives divergence to zero for large $\lambda$

Challenge: Stiff system; may require implicit solver.

**Decision: `[DECISION-005]` — Projection Method Selected**

Rationale: Penalty method would contaminate χ(t) signal at extreme Re with artificial dissipation. 
Cannot distinguish physical singularity from numerical artifact. Projection gives clean, 
interpretable χ(t) growth. See DECISION_LOG.md for full analysis.

---

## 3. Navier-Stokes Operators in QTT

### 3.1 Advection Term

$$(\mathbf{u} \cdot \nabla) \mathbf{u}$$

This is the nonlinear term. In component form:
$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z}$$

**QTT Implementation:**
1. Compute $\partial u / \partial x$ via differentiation MPO
2. Multiply $u \cdot (\partial u / \partial x)$ — pointwise product in QTT
3. Sum contributions

Pointwise product of two QTT tensors increases bond dimension: $\chi_{result} \leq \chi_1 \cdot \chi_2$

Must truncate back to target $\chi$ after each operation.

### 3.2 Diffusion Term

$$\nu \nabla^2 \mathbf{u}$$

Laplacian is sum of second derivatives:
$$\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$$

Each is an MPO application. Sum of MPOs is an MPO with $\chi$ summed.

### 3.3 Pressure Gradient

$$-\nabla p$$

Requires solving for pressure from incompressibility constraint. See Section 2.

---

## 4. χ-Regularity Theory

### 4.1 Conjecture Formalization

**Definition (χ-representability):** A function $f \in L^2(\Omega)$ is $\chi$-representable at tolerance $\epsilon$ if there exists a QTT decomposition with bond dimension $\leq \chi$ such that $\|f - f_{QTT}\|_2 \leq \epsilon$.

**Definition (χ-regularity):** A time-dependent solution $u(t)$ is $\chi$-regular on $[0, T]$ if $u(t)$ is $\chi$-representable at tolerance $\epsilon$ for all $t \in [0, T]$.

**Conjecture:** If $u(t)$ solving 3D incompressible NS is $\chi$-regular on $[0, T]$, then $u(t) \in H^s(\Omega)$ for some $s > 5/2$ on $[0, T]$.

### 4.2 Heuristic Argument

Smooth functions have rapidly decaying Fourier coefficients:
$$|\hat{f}_k| \lesssim |k|^{-s}$$ for $f \in H^s$

QTT efficiently represents functions with low-rank structure in Fourier space. High Sobolev regularity $\Rightarrow$ low effective rank $\Rightarrow$ low $\chi$.

Conversely: If $\chi$ stays bounded, the solution maintains structure consistent with smoothness.

### 4.3 What Would Constitute Proof?

To rigorously prove χ-regularity implies Sobolev regularity:

1. **Error analysis:** Bound $\|u - u_{QTT}\|_{H^s}$ in terms of $\chi$ and truncation error
2. **Stability:** Show NS evolution preserves QTT representability
3. **Connection:** Derive $H^s$ bound from QTT approximation bound

This is **hard**. Likely requires collaboration with PDE analysts.

---

## 5. Numerical Considerations

### 5.1 Truncation Strategy

After each QTT operation, truncate to target $\chi$ via SVD:
1. Left-orthogonalize the TT
2. Sweep right, truncating singular values below threshold
3. Resulting $\chi$ is data-dependent

**Fixed $\chi$ vs Adaptive $\chi$:**
- Fixed: Easier to analyze, but may lose accuracy
- Adaptive: Maintains accuracy, $\chi$ becomes the observable

For χ-regularity research, **adaptive $\chi$ with fixed tolerance** is preferred.

### 5.2 Conservation in QTT

Truncation does not preserve conservation laws exactly. Must monitor:
- Mass: $\int \rho \, dV$
- Momentum: $\int \rho \mathbf{u} \, dV$
- Energy: $\int \frac{1}{2} |\mathbf{u}|^2 \, dV$

If conservation drift exceeds tolerance, flag and investigate.

### 5.3 Time Integration

Options for evolving QTT state:
1. **TDVP (Time-Dependent Variational Principle):** Evolves on TT manifold, preserves structure
2. **Runge-Kutta with truncation:** Standard RK, truncate after each stage
3. **Splitting methods:** Separate advection/diffusion, each in QTT

TDVP preferred for structure preservation; RK simpler to implement.

---

## 6. Relationship to Hou-Luo

### 6.1 Their Setup

Hou-Luo (2014) studied axisymmetric Euler (no viscosity) in a cylinder with:
- Adaptive mesh refinement near potential singularity
- Found candidate blowup at the boundary

### 6.2 Our Differences

| Aspect | Hou-Luo | NS-Millennium |
|--------|---------|---------------|
| Equations | Euler (inviscid) | Navier-Stokes (viscous) |
| Representation | Adaptive mesh | QTT (fixed log N structure) |
| Observable | Vorticity growth | Bond dimension growth |
| Geometry | Axisymmetric | Full 3D (eventually) |

### 6.3 Potential Synergy

Could apply QTT to Hou-Luo's initial conditions:
- Does χ(t) diverge as they approach blowup time?
- If yes: supports χ-regularity conjecture
- If no: QTT may be regularizing (risk R2)

---

*Last updated: 2025-12-22*
