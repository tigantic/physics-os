# Security Model

- ProofPackage artifacts are content-addressed and hash-verified prior to render.
- DomainPacks are data-only and must validate against DomainPackSchema.
- Verification status is derived from artifact hash integrity and merkle root validation.
