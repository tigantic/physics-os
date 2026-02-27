import { describe, it, expect, vi } from "vitest";
import React from "react";
import type { ProofPackage, DomainPack, ProofMode } from "@luxury/core";

// Mock all screen components and viewers to isolate modeComposer logic
vi.mock("@/features/screens/Summary", () => ({
  SummaryScreen: (props: Record<string, unknown>) => (
    <div data-testid="summary-screen" data-mode={String(props.mode)} />
  ),
}));
vi.mock("@/features/screens/Timeline", () => ({
  TimelineScreen: () => <div data-testid="timeline-screen" />,
}));
vi.mock("@/features/screens/Gates", () => ({
  GatesScreen: () => <div data-testid="gates-screen" />,
}));
vi.mock("@/features/screens/Evidence", () => ({
  EvidenceScreen: () => <div data-testid="evidence-screen" />,
}));
vi.mock("@/features/screens/Integrity", () => ({
  IntegrityScreen: () => <div data-testid="integrity-screen" />,
}));
vi.mock("@/features/screens/Compare", () => ({
  CompareScreen: () => <div data-testid="compare-screen" />,
}));
vi.mock("@/features/viewers/PrimaryViewer", () => ({
  PrimaryViewer: () => <div data-testid="primary-viewer" />,
}));

import { renderCenterScreens } from "@/features/proof/modeComposer";

const mockProof: ProofPackage = {
  schema_version: "1.0.0",
  meta: {
    id: "run-001",
    project_id: "hydro-sim",
    domain_id: "fluid_dynamics",
    solver: { name: "TestSolver", version: "1.0.0", commit_hash: "abc123" },
    environment: { container_digest: "sha256:deadbeef", seed: 42, arch: "x86_64" },
    timestamp: "2026-02-16T00:00:00Z",
  },
  verdict: { status: "PASS", reason: "All gates passed", quality_score: 0.95 },
  verification: { status: "VERIFIED", verifier_version: "0.1.0", failures: [] },
  attestation: {
    merkle_root: "sha256:merkleroot",
    signatures: [{ key_id: "k1", algorithm: "ed25519", signature: "sig", timestamp: "2026-02-16T00:00:00Z" }],
  },
  claims: {},
  gate_results: {},
  gate_manifests: {},
  artifacts: {},
  timeline: { step_count: 0, steps: [] },
};

const mockDomain: DomainPack = {
  id: "fluid_dynamics",
  version: "1.0.0",
  metrics: {},
  gate_packs: {},
  viewers: [],
  templates: {
    executive_summary_metric_ids: [],
    publication_sections: [],
    citation_format: "bibtex",
  },
};

function makeCtx(mode: ProofMode) {
  return { proof: mockProof, domain: mockDomain, packageId: "pass", mode };
}

describe("renderCenterScreens", () => {
  it("returns elements for EXECUTIVE mode", () => {
    const elements = renderCenterScreens(makeCtx("EXECUTIVE"));
    // EXECUTIVE center: ["HeroMetrics", "ExecutiveNarrative"] => 2 elements
    expect(Array.isArray(elements)).toBe(true);
    expect(elements).toHaveLength(2);
  });

  it("returns elements for REVIEW mode", () => {
    const elements = renderCenterScreens(makeCtx("REVIEW"));
    // REVIEW center: ["Timeline", "ClaimCards", "PrimaryViewer"] => 3 elements
    expect(Array.isArray(elements)).toBe(true);
    expect(elements).toHaveLength(3);
  });

  it("returns elements for AUDIT mode", () => {
    const elements = renderCenterScreens(makeCtx("AUDIT"));
    // AUDIT center: ["RawArtifactViewer", "ManifestViewer", "DiffViewer"] => 3 elements
    expect(Array.isArray(elements)).toBe(true);
    expect(elements).toHaveLength(3);
  });

  it("returns elements for PUBLICATION mode", () => {
    const elements = renderCenterScreens(makeCtx("PUBLICATION"));
    // PUBLICATION center: ["PaperView", "FigureStaging"] => 2 elements
    expect(Array.isArray(elements)).toBe(true);
    expect(elements).toHaveLength(2);
  });

  it("AUDIT mode includes ManifestViewer with gate manifests", () => {
    const proofWithManifests: ProofPackage = {
      ...mockProof,
      gate_manifests: {
        "audit-v1": {
          id: "audit-v1",
          version: "1.0.0",
          gates: {},
        },
      },
    };
    const ctx = { proof: proofWithManifests, domain: mockDomain, packageId: "pass", mode: "AUDIT" as ProofMode };
    const elements = renderCenterScreens(ctx);
    expect(elements.length).toBe(3);
  });

  it("all elements have unique keys", () => {
    for (const mode of ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as ProofMode[]) {
      const elements = renderCenterScreens(makeCtx(mode));
      const keys = elements.map((el) => el.key);
      const unique = new Set(keys);
      expect(unique.size).toBe(keys.length);
    }
  });
});
