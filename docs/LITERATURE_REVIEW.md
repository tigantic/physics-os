# Literature Review

## Navier-Stokes Regularity and Related Work

Comprehensive bibliography of prior approaches to the NS Millennium Problem and tensor network methods in PDEs.

---

## Status: IN PROGRESS

This review will be expanded as research proceeds. Initial entries based on common knowledge; citations to be verified.

---

## 1. Navier-Stokes Regularity — Classical Results

### 1.1 Existence and Uniqueness

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| Leray | 1934 | Weak solutions exist globally in 3D | Foundational; uniqueness unknown |
| Hopf | 1951 | Extended Leray's results | Energy inequality |
| Ladyzhenskaya | 1969 | 2D global regularity | Shows 2D is solved |
| Temam | 1977 | Navier-Stokes Equations (book) | Standard reference |

### 1.2 Regularity Criteria

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| Beale-Kato-Majda | 1984 | Blowup iff vorticity integral diverges | Key criterion |
| Constantin-Fefferman | 1993 | Geometric regularity conditions | Direction of vorticity |
| Escauriaza-Seregin-Šverák | 2003 | Blowup implies L³ blowup | Strongest known criterion |
| Seregin | 2012 | L_{3,∞} criterion | Refined bounds |

### 1.3 Partial Regularity

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| Caffarelli-Kohn-Nirenberg | 1982 | Singular set has measure zero | Doesn't prove smoothness |
| Lin | 1998 | Simplified CKN proof | Pedagogical |

---

## 2. Potential Singularity Research

### 2.1 Numerical Blowup Studies

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| Kerr | 1993 | Possible blowup in Euler | Controversial; later questioned |
| Hou-Li | 2006 | No blowup found in Euler | Contradicts Kerr |
| **Hou-Luo** | **2014** | Candidate blowup on boundary | **Key paper — axisymmetric flow** |
| Brenner et al. | 2016 | Analysis of Hou-Luo scenario | Supporting evidence |

**Note:** Hou-Luo 2014 is closest prior work to our goal. They found potential blowup via adaptive mesh DNS. We seek same via QTT.

### 2.2 Theoretical Blowup Constructions

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| **Tao** | **2016** | Blowup for averaged NS | **Shows blowup possible in modified equations** |
| Elgindi | 2019 | Blowup for Euler with corners | Singular domain |
| Chen-Hou | 2023 | Computer-assisted blowup for Euler | Rigorous numerics |

---

## 3. Tensor Network Methods in PDEs

### 3.1 Tensor Train / QTT for PDEs

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| Oseledets | 2011 | TT decomposition | Foundational algorithm |
| Khoromskij | 2011 | QTT for multivariate functions | O(log N) representation |
| Dolgov-Savostyanov | 2014 | TT for parabolic PDEs | Time-dependent problems |
| Bachmayr et al. | 2016 | Low-rank methods for PDEs | Survey paper |

### 3.2 Tensor Networks in Fluid Dynamics

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| Gourianov et al. | 2022 | MPS for CFD | Direct application |
| Dektor-Venturi | 2021 | Tensor methods for Burgers | Nonlinear hyperbolic |
| Ye-Luo | 2023 | TT for incompressible flow | **Closest to our approach** |

---

## 4. χ-Regularity Related Concepts

### 4.1 Tensor Rank and Function Smoothness

| Author(s) | Year | Result | Relevance |
|-----------|------|--------|-----------|
| Schneider-Uschmajew | 2014 | Approximation in TT format | Error bounds |
| Griebel-Knapek | 2009 | Sparse grids and smoothness | Related compression |
| Hackbusch | 2012 | Tensor Spaces book | Mathematical foundations |

### 4.2 Entanglement and Physical Complexity

| Concept | Field | Relevance |
|---------|-------|-----------|
| Area law | Quantum physics | Low entanglement ⟹ low χ |
| Kolmogorov complexity | Information theory | Compressibility and regularity |
| Sobolev embedding | Analysis | Smoothness hierarchies |

---

## 5. Gap in Literature

**Key Observation:** No prior work examines tensor network rank as a regularity indicator for Navier-Stokes.

The connection between:
- χ (bond dimension) from tensor networks
- s (Sobolev index) from PDE regularity theory

is **unexplored** in the literature. This is our unique contribution angle.

---

## 6. To Be Reviewed

Papers to read and add:

- [ ] Hou-Luo 2014 (full paper, not just summary)
- [ ] Tao 2016 averaged NS construction
- [ ] Chen-Hou 2023 computer-assisted Euler blowup
- [ ] Ye-Luo 2023 TT for incompressible flow
- [ ] Recent arXiv on NS regularity (2023-2025)

---

## References (BibTeX)

```bibtex
@article{hou2014potentially,
  title={Potentially singular solutions of the 3D axisymmetric Euler equations},
  author={Hou, Thomas Y and Luo, Guo},
  journal={Proceedings of the National Academy of Sciences},
  year={2014}
}

@article{tao2016finite,
  title={Finite time blowup for an averaged three-dimensional Navier-Stokes equation},
  author={Tao, Terence},
  journal={Journal of the American Mathematical Society},
  year={2016}
}

@article{oseledets2011tensor,
  title={Tensor-train decomposition},
  author={Oseledets, Ivan V},
  journal={SIAM Journal on Scientific Computing},
  year={2011}
}
```

---

*Last updated: 2025-12-22*
