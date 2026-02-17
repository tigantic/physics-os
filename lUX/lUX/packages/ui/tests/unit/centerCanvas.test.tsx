import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import type { ProofPackage, DomainPack, ProofMode } from "@luxury/core";

// Mock all screen components and viewers so CenterCanvas renders without side effects
vi.mock("@/features/screens/Summary", () => ({
  SummaryScreen: () => <div data-testid="summary-screen">Summary</div>,
}));
vi.mock("@/features/screens/Timeline", () => ({
  TimelineScreen: () => <div data-testid="timeline-screen">Timeline</div>,
}));
vi.mock("@/features/screens/Gates", () => ({
  GatesScreen: () => <div data-testid="gates-screen">Gates</div>,
}));
vi.mock("@/features/screens/Evidence", () => ({
  EvidenceScreen: () => <div data-testid="evidence-screen">Evidence</div>,
}));
vi.mock("@/features/screens/Integrity", () => ({
  IntegrityScreen: () => <div data-testid="integrity-screen">Integrity</div>,
}));
vi.mock("@/features/screens/Compare", () => ({
  CompareScreen: () => <div data-testid="compare-screen">Compare</div>,
}));
vi.mock("@/features/viewers/PrimaryViewer", () => ({
  PrimaryViewer: () => <div data-testid="primary-viewer">PrimaryViewer</div>,
}));

import { CenterCanvas } from "@/features/proof/CenterCanvas";

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

describe("CenterCanvas", () => {
  it('has role="tabpanel"', () => {
    render(<CenterCanvas proof={mockProof} domain={mockDomain} mode={"REVIEW" as ProofMode} bundleDir="/tmp/bundle" />);
    expect(screen.getByRole("tabpanel")).toBeInTheDocument();
  });

  it("has correct aria-labelledby linking to mode tab", () => {
    render(<CenterCanvas proof={mockProof} domain={mockDomain} mode={"AUDIT" as ProofMode} bundleDir="/tmp/bundle" />);
    const panel = screen.getByRole("tabpanel");
    expect(panel).toHaveAttribute("aria-labelledby", "mode-tab-AUDIT");
  });

  it("renders content from modeComposer", () => {
    render(<CenterCanvas proof={mockProof} domain={mockDomain} mode={"REVIEW" as ProofMode} bundleDir="/tmp/bundle" />);
    // REVIEW mode renders Timeline, ClaimCards, PrimaryViewer
    expect(screen.getByTestId("timeline-screen")).toBeInTheDocument();
  });
});
