# Architecture

This section describes the high-level architecture of the The Physics OS
monorepo, the three-tier package design, and the tensor network engine
internals.

## Repository Structure

```
HyperTensor-VM/
├── tensornet/          # Tier 1: Physics engine (Python)
├── hypertensor/        # Tier 2: Execution fabric (Python)
├── crates/             # Tier 3: Rust performance substrate
├── apps/               # Standalone applications
├── products/           # Commercial products
├── contracts/          # Solidity smart contracts
├── proofs/             # Formal verification proofs
├── tests/              # Cross-package integration tests
├── tools/              # Build and maintenance scripts
├── docs/               # This documentation
├── deploy/             # Deployment configurations
└── integrations/       # Third-party integrations
```
