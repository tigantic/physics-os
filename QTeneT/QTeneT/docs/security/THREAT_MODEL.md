# Threat Model — QTeneT

**Classification:** Proprietary & Confidential  
**Scope:** QTeneT governance + packaging layer (`qtenet`), plus integration surfaces to upstream monorepo components.  
**Non-scope (by ADR):** The_Compressor as a product is a separate deployment/monetization path; QTeneT may integrate but does not subsume it.

---

## 1. Executive Summary

QTeneT’s primary near-term risk is not exploitability of complex math kernels; it is **enterprise exposure risk** arising from:
- artifact/container handling (untrusted inputs),
- future CLI and service surfaces,
- supply chain and dependency integrity,
- inadvertent IP leakage via logs/docs/packaging,
- densification leading to resource exhaustion and denial-of-service.

This threat model defines **assets**, **trust boundaries**, **attack surfaces**, and **required mitigations** to reach an externally distributable posture.

---

## 2. Assets (What must be protected)

### 2.1 Confidential IP
- QTT algorithms, operator libraries, solver pipelines, and genesis primitives.
- Canonicalization maps and capability packaging metadata.

### 2.2 Customer / Partner Data
- Inputs to compression/query/solver pipelines.
- Any datasets used in demos/benchmarks when QTeneT becomes distributable.

### 2.3 Integrity of Results
- Correctness and reproducibility of tensor outputs.
- Provenance/attestation artifacts and run manifests.

### 2.4 Availability
- CLI runs must not allow trivially-triggered OOM/DoS.
- Future services must be resilient to malformed requests.

---

## 3. Trust Boundaries

### TB1 — Local Developer Machine
- Trusted operator, but accidental misconfiguration is common.

### TB2 — CI/CD
- Must treat PRs as untrusted and isolate secrets.

### TB3 — External Inputs
- Any file/container passed to `qtenet inspect/query/reconstruct` is **untrusted**.

### TB4 — Optional Network Services (future)
- Any HTTP/gRPC query service is internet-exposed unless explicitly isolated.

---

## 4. Attack Surfaces

### 4.1 CLI
- `qtenet inspect`
- `qtenet query`
- `qtenet reconstruct` (explicit dense escape hatch)

Risks:
- path traversal, arbitrary file reads
- resource exhaustion (decompress-to-dense)
- unsafe deserialization (pickle, eval)

### 4.2 Container/Artifact Parsing
Risks:
- zip bombs / NPZ bombs
- malformed headers triggering code paths with quadratic behavior
- memory mapping edge cases

### 4.3 Dependency/Supply Chain
Risks:
- malicious dependency upgrade
- compromised wheel/SDist artifacts

### 4.4 Documentation & Distribution
Risks:
- leaking confidential file paths or internal architecture in public docs
- shipping internal inventories that reveal sensitive capabilities

---

## 5. Threats (STRIDE-style)

### S — Spoofing
- Fake artifacts masquerading as “official” QTeneT outputs.

Mitigations:
- signed manifests/attestations (hashes)
- optional signature verification mode for artifacts

### T — Tampering
- Modifying containers/manifests to misrepresent results.

Mitigations:
- content hashes in manifests
- immutable output directories in pipelines (optional)

### R — Repudiation
- “We didn’t run that” / “those parameters weren’t used.”

Mitigations:
- run manifest schema (seed, operator identity, eps/max_rank)
- deterministic mode flags

### I — Information Disclosure
- Accidentally leaking proprietary internals or user data.

Mitigations:
- strict log redaction policy
- avoid printing full paths by default
- never embed raw data in manifests unless requested

### D — Denial of Service
- Dense reconstruction used to trigger OOM.
- Malformed inputs trigger worst-case behavior.

Mitigations:
- `--allow-dense` required for reconstruct
- hard caps: max output size, max ranks, max eval budget
- timeouts and memory guards in service mode

### E — Elevation of Privilege
- Any dynamic import, eval, plugin system, or shell execution.

Mitigations:
- no `eval`, no pickle
- no shelling out by default
- isolate service runtime (container, least privilege)

---

## 6. Security Requirements (Release Gates)

### Phase 1 (internal dev)
- [ ] Ban unsafe deserialization in QTeneT surface (`pickle`, `eval`).
- [ ] `qtenet reconstruct` requires `--allow-dense` and enforces size caps.
- [ ] Manifest schema exists and is emitted by CLI commands.

### Phase 2 (private distribution)
- [ ] Artifact signature verification option.
- [ ] Threat model reviewed annually or per major release.
- [ ] CI: dependency pinning + SBOM generation.

### Phase 3 (external distribution)
- [ ] Service mode hardened: authn/z, rate limiting, input validation.
- [ ] Security contact verified and incident workflow tested.

---

## 7. Minimal Security Controls (Non-negotiable)

- **Never Go Dense** enforced by default.
- Explicit dense escape hatch with caps.
- No pickle/eval in artifact formats.
- Deterministic provenance manifests for all runs.
- CI secrets never available to untrusted PRs.

---

## 8. Appendix: Suggested Manifest Fields

```json
{
  "tool": "qtenet",
  "version": "0.1.0-dev",
  "timestamp": "2026-01-31T00:00:00Z",
  "git": {"sha": "..."},
  "env": {"python": "...", "torch": "...", "cuda": true},
  "inputs": [{"path": "...", "sha256": "..."}],
  "operators": [{"name": "laplacian", "scheme": "cd2", "version": "v1"}],
  "rank_control": {"eps": 1e-6, "max_rank": 64},
  "outputs": [{"path": "...", "sha256": "..."}]
}
```
