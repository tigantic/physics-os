# Yang-Mills Phase II: Tensor Network Implementation

**Goal:** Access weak coupling regime ($g \to 0$) to prove continuum mass gap

---

## Why Phase I Failed

| Phase I (Exact Diag) | Phase II (Tensor Networks) |
|---------------------|---------------------------|
| Store full $|\psi\rangle$ | Store compressed MPS |
| $O(2^N)$ memory | $O(N \cdot \chi^2 \cdot d)$ memory |
| Limited to $g \gtrsim 1$ | Can reach $g \ll 1$ |
| Product state vacuum | Entangled vacuum |
| 4.77 TiB explosion | ~GB manageable |

---

## Architecture: MPS for Yang-Mills

### State Representation

```
Full vector:       |ψ⟩ = Σ c_{i₁i₂...iₙ} |i₁⟩|i₂⟩...|iₙ⟩
                   Memory: d^N (exponential)

MPS:               |ψ⟩ = Σ A¹[i₁] A²[i₂] ... Aⁿ[iₙ] |i₁⟩|i₂⟩...|iₙ⟩
                   Memory: N × χ² × d (polynomial)
```

Where:
- $d$ = local Hilbert space dimension (2j+1 for each link)
- $\chi$ = bond dimension (entanglement cutoff)
- $N$ = number of sites (links in our case)

### Why This Works for Yang-Mills

1. **Area law**: Ground states of gapped systems have bounded entanglement
2. **If we find a gap**: MPS is efficient representation
3. **If no gap**: MPS entanglement grows → we'd detect it

---

## Implementation Plan

### Step 1: MPO Hamiltonian

Convert our Hamiltonian to Matrix Product Operator form:

```python
class YangMillsMPO:
    """Yang-Mills Hamiltonian as MPO."""
    
    def __init__(self, N_links: int, g: float, j_max: float):
        self.g = g
        self.N = N_links
        self.d = int(2 * j_max + 1)  # Local dimension
        
        # Build local tensors
        self.W = self._build_mpo_tensors()
    
    def _build_mpo_tensors(self):
        """
        MPO for H = (g²/2) Σ E² + (1/g²) Σ (1 - Re Tr U_P)
        
        W tensor structure for electric term:
        W[α,β,i,j] where α,β are bond indices, i,j physical
        """
        # Electric term: diagonal in j-basis
        E_squared = self._electric_casimir()
        
        # Magnetic term: plaquette coupling
        U_plaq = self._plaquette_operator()
        
        return self._combine_terms(E_squared, U_plaq)
```

### Step 2: MPS Ground State

```python
class YangMillsMPS:
    """Ground state as Matrix Product State."""
    
    def __init__(self, N_links: int, chi: int, d: int):
        self.N = N_links
        self.chi = chi  # Bond dimension
        self.d = d      # Local dimension
        
        # Initialize random MPS tensors
        self.A = [self._random_tensor(i) for i in range(N)]
    
    def _random_tensor(self, site: int):
        """Create random MPS tensor at site."""
        chi_l = 1 if site == 0 else self.chi
        chi_r = 1 if site == self.N - 1 else self.chi
        return np.random.randn(chi_l, self.d, chi_r)
```

### Step 3: DMRG Optimization

```python
def dmrg_sweep(mps: YangMillsMPS, mpo: YangMillsMPO, direction: str):
    """
    One DMRG sweep: optimize each tensor while holding others fixed.
    
    Key insight: Local optimization is eigenvalue problem
    H_eff |A_i⟩ = E |A_i⟩
    """
    for site in sweep_order(mps.N, direction):
        # Contract environment
        L = left_environment(mps, mpo, site)
        R = right_environment(mps, mpo, site)
        
        # Build effective Hamiltonian
        H_eff = build_effective_H(L, mpo.W[site], R)
        
        # Solve local eigenproblem (small!)
        E, A_opt = eigsh(H_eff, k=1, which='SA')
        
        # Update tensor
        mps.A[site] = A_opt.reshape(mps.A[site].shape)
        
        # SVD for canonicalization
        mps.canonicalize(site, direction)
    
    return mps, E
```

### Step 4: Measure Gap

```python
def measure_mass_gap(mps_gs: YangMillsMPS, mpo: YangMillsMPO):
    """
    Compute gap by finding first excited state.
    
    Method: Project out ground state, run DMRG again
    """
    # Ground state energy
    E0 = expectation_value(mps_gs, mpo, mps_gs)
    
    # Find excited state (orthogonal to ground)
    mps_ex = dmrg_with_projection(mpo, mps_gs)
    E1 = expectation_value(mps_ex, mpo, mps_ex)
    
    gap = E1 - E0
    return gap
```

---

## Critical Test: Scaling at Weak Coupling

```python
def test_continuum_limit():
    """
    THE critical test for Millennium Prize.
    
    If gap ~ g² : FAILS (strong coupling only)
    If gap ~ const : POSSIBLE (dimensional transmutation)
    If gap ~ exp(-c/g²) : INTERESTING (instanton effects)
    """
    results = []
    
    for g in [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]:
        # Use larger bond dimension for smaller g
        chi = int(50 / g)  # Scale with entanglement
        
        mpo = YangMillsMPO(N_links=4, g=g, j_max=2.0)
        mps = YangMillsMPS(N_links=4, chi=chi, d=mpo.d)
        
        # Run DMRG
        mps_gs, E0 = dmrg(mps, mpo, sweeps=100)
        gap = measure_mass_gap(mps_gs, mpo)
        
        results.append({
            'g': g,
            'gap': gap,
            'gap_over_g2': gap / g**2,
            'chi': chi,
            'entanglement': compute_entropy(mps_gs)
        })
    
    # Analyze scaling
    analyze_polynomial_vs_exponential(results)
```

---

## Expected Outcomes

### Scenario A: Gap ~ g² (Strong Coupling Persists)
- MPS confirms exact diagonalization
- No dimensional transmutation
- Prize NOT solved

### Scenario B: Gap Deviates from g²
- Something interesting at weak coupling
- Check for exp(-c/g²) behavior
- Possible confinement mechanism

### Scenario C: Entanglement Explodes
- Bond dimension insufficient
- Need MERA for scale invariance
- Indicates critical/conformal behavior

---

## Implementation Priority

1. **Week 1**: MPO builder for Yang-Mills H
2. **Week 2**: MPS class with canonicalization
3. **Week 3**: DMRG sweeps
4. **Week 4**: Gap measurement + scaling test

---

## Libraries

```python
# Option 1: TenPy (mature, well-tested)
from tenpy.models.model import MPOModel
from tenpy.algorithms import dmrg

# Option 2: Custom PyTorch (GPU acceleration)
import torch
from torch_tensornetwork import MPS, MPO, contract

# Option 3: ITensor-style (pure Python)
from yangmills.tensor_network import MPS, MPO, DMRG
```

---

## Success Criteria

| Metric | Strong Coupling (Current) | Weak Coupling (Target) |
|--------|--------------------------|------------------------|
| Coupling g | 1.0 | 0.01 - 0.1 |
| Gap/g² | 1.5 (constant) | ? (should deviate) |
| Entanglement S | 0 | > 0 (growing?) |
| Memory | MB | GB (bounded by χ) |
| Physical gap | → 0 | → Λ_QCD > 0 |

---

## References

1. Schollwöck, U. (2011). DMRG review. *Annals of Physics*
2. Orús, R. (2014). Tensor networks review. *Annals of Physics*
3. Bañuls, M.C. et al. (2020). Tensor networks for LGT. *EPJD*
4. Buyens, B. et al. (2017). MPS for gauge theories. *PRX*

---

*HyperTensor Phase II — The Real Challenge Begins*
