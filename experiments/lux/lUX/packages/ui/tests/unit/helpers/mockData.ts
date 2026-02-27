import type { ProofPackage, DomainPack } from "@luxury/core";

/**
 * Minimal valid ProofPackage for unit tests.
 * Not loaded from disk — constructed in-memory so tests don't depend on fixtures.
 */
export function makeProof(overrides?: Partial<ProofPackage>): ProofPackage {
  const base: ProofPackage = {
    schema_version: "1.0.0",
    meta: {
      id: "test-uuid",
      project_id: "proj-001",
      domain_id: "com.physics.vlasov",
      solver: { name: "TestSolver", version: "1.0.0", commit_hash: "abcdef01" },
      environment: {
        container_digest: "sha256:" + "a".repeat(64),
        seed: 42,
        arch: "x86_64",
      },
      timestamp: "2026-02-16T00:00:00Z",
    },
    verdict: { status: "PASS", reason: "All gates passed", quality_score: 0.95 },
    claims: {
      "C-001": {
        id: "C-001",
        statement: "Conservation residual stays below tolerance",
        category: "stability",
        tags: ["physics"],
        gate_ids: ["G-001"],
        evidence_refs: ["A-ts"],
      },
    },
    gate_manifests: {
      "M-001": {
        id: "M-001",
        version: "1.0.0",
        gates: {
          "G-001": {
            id: "G-001",
            metric_id: "conservation_residual",
            operator: "lt",
            threshold: 1e-6,
            aggregation: "max",
            validity: { requires_artifacts: [], requires_metrics: [], domain_assumptions: [] },
          },
        },
      },
    },
    gate_results: {
      "G-001": {
        gate_id: "G-001",
        metric_id: "conservation_residual",
        value: { status: "ok", value: 1e-8 },
        passed: { status: "ok", value: true },
        margin: { status: "ok", value: 0.99 },
      },
    },
    timeline: {
      step_count: 2,
      steps: [
        {
          step_index: 0,
          state_hash: "sha256:" + "0".repeat(64),
          metrics: {
            conservation_residual: { status: "ok", value: 1e-9 },
            l2_drift: { status: "ok", value: 2e-7 },
          },
          artifact_refs: [],
        },
        {
          step_index: 1,
          state_hash: "sha256:" + "1".repeat(64),
          metrics: {
            conservation_residual: { status: "ok", value: 1e-8 },
            l2_drift: { status: "ok", value: 3e-7 },
          },
          artifact_refs: [],
        },
      ],
    },
    artifacts: {
      "A-ts": {
        id: "A-ts",
        type: "time_series",
        mime_type: "text/csv",
        hash: "sha256:" + "b".repeat(64),
        size_bytes: 1024,
        uri: "artifacts/timeseries.csv",
        metadata: {},
      },
    },
    attestation: {
      merkle_root: "sha256:" + "c".repeat(64),
      signatures: [
        {
          key_id: "key-1",
          algorithm: "ed25519",
          signature: "base64sig==",
          timestamp: "2026-02-16T00:00:00Z",
        },
      ],
    },
    verification: {
      status: "VERIFIED",
      verifier_version: "0.1.0",
      failures: [],
    },
  };
  return { ...base, ...overrides };
}

/** Minimal valid DomainPack for unit tests. */
export function makeDomain(overrides?: Partial<DomainPack>): DomainPack {
  const base: DomainPack = {
    id: "com.physics.vlasov",
    version: "1.0.0",
    metrics: {
      conservation_residual: {
        label: "Conservation Residual",
        unit: "",
        format: "scientific",
        precision: 3,
      },
      l2_drift: {
        label: "L2 Drift",
        unit: "",
        format: "scientific",
        precision: 4,
      },
    },
    gate_packs: {
      "GP-001": {
        label: "Stability Pack",
        manifest_ref: "M-001",
        highlight_metrics: ["conservation_residual"],
      },
    },
    viewers: [
      {
        when: { artifact_type: "time_series" },
        component: "TimeSeriesViewer",
        default_config: {},
        priority: 0,
      },
    ],
    templates: {
      executive_summary_metric_ids: ["conservation_residual", "l2_drift"],
      publication_sections: [],
      citation_format: "bibtex",
    },
  };
  return { ...base, ...overrides };
}
