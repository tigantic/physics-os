/**
 * Shared Storybook fixture data for components requiring ProofPackage / DomainPack.
 *
 * This module provides production-realistic fixture data that mirrors the actual
 * @luxury/core schema exactly. Used by all screen, rail, and composite stories.
 */
import type { ProofPackage, DomainPack, ProofMode } from "@luxury/core";

/* ── Proof Package fixtures ─────────────────────────────────────────── */

export const FIXTURE_PROOF_PASS: ProofPackage = {
  schema_version: "1.0.0",
  meta: {
    id: "hydro-sim-002-2026-02-16",
    project_id: "hydro-sim",
    domain_id: "fluid_dynamics",
    solver: { name: "HydroSolve", version: "4.2.1", commit_hash: "a1b2c3d4e5f6" },
    environment: {
      container_digest: "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
      seed: 42,
      arch: "x86_64",
    },
    timestamp: "2026-02-16T14:30:00Z",
  },
  verdict: {
    status: "PASS",
    reason: "All 12 gates passed with quality score 0.94",
    quality_score: 0.94,
    quality_breakdown: {
      accuracy: 0.97,
      stability: 0.92,
      convergence: 0.93,
    },
  },
  verification: {
    status: "VERIFIED",
    verifier_version: "0.3.1",
    verified_at: "2026-02-16T14:35:00Z",
    failures: [],
  },
  attestation: {
    merkle_root: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    signatures: [
      {
        key_id: "fp:ed25519:a1b2c3...d4e5f6",
        algorithm: "ed25519",
        signature: "MEUCIQDBase64EncodedSignatureDataHere==",
        timestamp: "2026-02-16T14:35:01Z",
      },
    ],
  },
  claims: {
    "claim-convergence": {
      id: "claim-convergence",
      statement: "Reynolds-averaged simulation converges within 500 iterations",
      category: "convergence",
      tags: ["RANS", "steady-state"],
      gate_ids: ["gate-residual-l2", "gate-convergence-rate"],
      evidence_refs: ["artifact-convergence-plot"],
    },
    "claim-accuracy": {
      id: "claim-accuracy",
      statement: "Drag coefficient within 2% of experimental reference",
      category: "accuracy",
      tags: ["drag", "validation"],
      gate_ids: ["gate-cd-tolerance"],
      evidence_refs: ["artifact-cd-comparison"],
    },
  },
  gate_manifests: {
    "manifest-fluid-v1": {
      id: "manifest-fluid-v1",
      version: "1.0.0",
      gates: {
        "gate-residual-l2": {
          id: "gate-residual-l2",
          metric_id: "residual_l2",
          operator: "lt",
          threshold: 1e-6,
          aggregation: "final",
          validity: { requires_artifacts: [], requires_metrics: ["residual_l2"], domain_assumptions: [] },
        },
        "gate-cd-tolerance": {
          id: "gate-cd-tolerance",
          metric_id: "cd_error_pct",
          operator: "lt",
          threshold: 2.0,
          aggregation: "final",
          validity: { requires_artifacts: [], requires_metrics: ["cd_error_pct"], domain_assumptions: [] },
        },
        "gate-convergence-rate": {
          id: "gate-convergence-rate",
          metric_id: "convergence_rate",
          operator: "gt",
          threshold: 0.8,
          aggregation: "mean",
          validity: { requires_artifacts: [], requires_metrics: ["convergence_rate"], domain_assumptions: [] },
        },
      },
    },
  },
  gate_results: {
    "gate-residual-l2": {
      gate_id: "gate-residual-l2",
      metric_id: "residual_l2",
      value: { status: "ok", value: 3.2e-7 },
      passed: { status: "ok", value: true },
      margin: { status: "ok", value: 0.68 },
    },
    "gate-cd-tolerance": {
      gate_id: "gate-cd-tolerance",
      metric_id: "cd_error_pct",
      value: { status: "ok", value: 1.3 },
      passed: { status: "ok", value: true },
      margin: { status: "ok", value: 0.35 },
    },
    "gate-convergence-rate": {
      gate_id: "gate-convergence-rate",
      metric_id: "convergence_rate",
      value: { status: "ok", value: 0.92 },
      passed: { status: "ok", value: true },
      margin: { status: "ok", value: 0.15 },
    },
  },
  artifacts: {
    "artifact-convergence-plot": {
      id: "artifact-convergence-plot",
      type: "time_series",
      mime_type: "image/png",
      hash: "sha256:convergenceplot1234567890abcdef",
      size_bytes: 45_200,
      uri: "artifacts/convergence-plot.png",
      metadata: { title: "Residual Convergence", x_label: "Iteration", y_label: "L2 Norm" },
    },
    "artifact-cd-comparison": {
      id: "artifact-cd-comparison",
      type: "table",
      mime_type: "text/csv",
      hash: "sha256:cdcomparison1234567890abcdef",
      size_bytes: 1_280,
      uri: "artifacts/cd-comparison.csv",
      metadata: { title: "Drag Coefficient Comparison" },
    },
  },
  timeline: {
    step_count: 3,
    steps: [
      {
        step_index: 0,
        state_hash: "sha256:step0hash",
        metrics: {
          residual_l2: { status: "ok", value: 1.2e-2 },
          cd_error_pct: { status: "ok", value: 8.5 },
          convergence_rate: { status: "ok", value: 0.45 },
        },
        artifact_refs: [],
      },
      {
        step_index: 1,
        state_hash: "sha256:step1hash",
        metrics: {
          residual_l2: { status: "ok", value: 5.1e-5 },
          cd_error_pct: { status: "ok", value: 3.2 },
          convergence_rate: { status: "ok", value: 0.78 },
        },
        artifact_refs: [],
      },
      {
        step_index: 2,
        state_hash: "sha256:step2hash",
        metrics: {
          residual_l2: { status: "ok", value: 3.2e-7 },
          cd_error_pct: { status: "ok", value: 1.3 },
          convergence_rate: { status: "ok", value: 0.92 },
        },
        artifact_refs: ["artifact-convergence-plot", "artifact-cd-comparison"],
      },
    ],
  },
};

export const FIXTURE_PROOF_FAIL: ProofPackage = {
  ...FIXTURE_PROOF_PASS,
  meta: { ...FIXTURE_PROOF_PASS.meta, id: "hydro-sim-003-fail" },
  verdict: {
    status: "FAIL",
    reason: "Gate gate-cd-tolerance failed: 4.7% > 2.0% threshold",
    quality_score: 0.61,
  },
  verification: {
    status: "BROKEN_CHAIN",
    verifier_version: "0.3.1",
    failures: [
      { code: "SIG_MISMATCH", message: "Signature does not match recomputed merkle root" },
    ],
  },
  gate_results: {
    ...FIXTURE_PROOF_PASS.gate_results,
    "gate-cd-tolerance": {
      gate_id: "gate-cd-tolerance",
      metric_id: "cd_error_pct",
      value: { status: "ok", value: 4.7 },
      passed: { status: "ok", value: false },
      margin: { status: "ok", value: -1.35 },
    },
  },
};

/* ── Domain Pack fixture ────────────────────────────────────────────── */

export const FIXTURE_DOMAIN: DomainPack = {
  id: "fluid_dynamics",
  version: "1.0.0",
  metrics: {
    residual_l2: {
      label: "Residual L₂",
      symbol_latex: "\\|r\\|_2",
      unit: "",
      dimension: "scalar",
      format: "scientific",
      precision: 2,
      validity_range: [0, 1],
      description: "L2 norm of the residual vector",
    },
    cd_error_pct: {
      label: "Cᴅ Error",
      symbol_latex: "\\epsilon_{C_D}",
      unit: "%",
      format: "fixed",
      precision: 1,
      validity_range: [0, 100],
      description: "Percentage error in drag coefficient vs. experimental reference",
    },
    convergence_rate: {
      label: "Convergence Rate",
      symbol_latex: "\\rho",
      unit: "",
      format: "fixed",
      precision: 2,
      validity_range: [0, 1],
      description: "Asymptotic convergence rate (0=diverging, 1=instant convergence)",
    },
  },
  gate_packs: {
    "manifest-fluid-v1": {
      label: "Fluid Dynamics v1",
      manifest_ref: "manifest-fluid-v1",
      highlight_metrics: ["residual_l2", "cd_error_pct"],
    },
  },
  viewers: [
    {
      when: { artifact_type: "time_series", mime_type: "image/png" },
      component: "TimeSeriesViewer",
      default_config: {},
      priority: 1,
    },
    {
      when: { metric_id: "residual_l2" },
      component: "TableViewer",
      default_config: { scale: "log" },
      priority: 2,
    },
  ],
  templates: {
    executive_summary_metric_ids: ["residual_l2", "cd_error_pct", "convergence_rate"],
    publication_sections: ["abstract", "methodology", "results", "conclusion"],
    citation_format: "bibtex",
  },
};

/* ── Helpers ─────────────────────────────────────────────────────────── */

export const ALL_MODES: ProofMode[] = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"];
export const DEFAULT_MODE: ProofMode = "REVIEW";
