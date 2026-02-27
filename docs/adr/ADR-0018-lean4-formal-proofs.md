# ADR-0018: Lean 4 for Formal Proofs

| Field | Value |
|-------|-------|
| **Status** | accepted |
| **Date** | 2026-02-25 |
| **Author** | @core-team |

## Context

HyperTensor claims mathematical correctness for conservation laws (mass, momentum, energy), PDE discretization operators, and tensor decomposition properties. Marketing these claims to aerospace, defense, and nuclear clients requires more than unit tests — it requires **formal verification**: machine-checked proofs that the mathematical statements are logically valid.

Candidates evaluated:

| System | Language | Ecosystem | Automation | Maturity |
|--------|----------|-----------|------------|----------|
| Coq | Gallina | Mature, large stdlib | Ltac/Ltac2 | 35+ years |
| Lean 4 | Lean | Mathlib (growing fast) | Aesop, simp, omega | 5 years, accelerating |
| Isabelle | Isar | Very large stdlib | Sledgehammer | 30+ years |
| Agda | Agda | Smaller, type-theory focused | Limited | 20+ years |

Lean 4 was selected because: (1) Mathlib already formalizes real analysis, linear algebra, and measure theory needed for our proofs; (2) Lean 4's metaprogramming framework enables custom tactics for tensor algebra; (3) active community growth and backing by Microsoft Research; (4) native code compilation for proof checking performance.

## Decision

**All formal mathematical proofs use Lean 4 with Mathlib.** Specifically:

1. Proof source files live in `proofs/yang_mills/lean/` and `proofs/conservation/*/lean/`.
2. Each proof targets a specific theorem stated in `PLATFORM_SPECIFICATION.md` §8.
3. Proofs are checked by `lean4` in CI (`audit-gates.yml`).
4. Proof artifacts (`.olean` files) are cached but not committed.
5. New conservation law claims require a companion Lean proof before the claim enters the platform specification.

## Consequences

- **Easier:** Machine-checked proofs eliminate human proof-review bottleneck for mathematical claims.
- **Easier:** Mathlib provides lemmas for Hilbert spaces, Sobolev norms, and Navier-Stokes weak formulations.
- **Harder:** Lean 4 learning curve for engineers without formal methods background.
- **Harder:** Proof maintenance as Mathlib evolves (breaking API changes).
- **Risk:** Some theorems may be infeasible to formalize completely (e.g., turbulence regularity). Documented as open conjectures, not claims.
