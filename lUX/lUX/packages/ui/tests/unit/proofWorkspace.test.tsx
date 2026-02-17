import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import type { ProofPackage, DomainPack, ProofMode } from "@luxury/core";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams("mode=REVIEW&id=pass"),
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/packages/pass",
}));

// Mock all screen components and viewers to isolate ProofWorkspace
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

import { ProofWorkspace } from "@/features/proof/ProofWorkspace";

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
  claims: {
    "claim-1": {
      id: "claim-1",
      statement: "Energy conserved",
      category: "stability",
      tags: [],
      gate_ids: [],
      evidence_refs: [],
    },
  },
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

describe("ProofWorkspace", () => {
  const defaultProps = {
    proof: mockProof,
    domain: mockDomain,
    fixture: "pass",
    mode: "REVIEW" as ProofMode,
    packageId: "pass",
  };

  it("renders IdentityStrip with project heading", () => {
    render(<ProofWorkspace {...defaultProps} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("hydro-sim · fluid_dynamics");
  });

  it('renders id="main-content"', () => {
    const { container } = render(<ProofWorkspace {...defaultProps} />);
    expect(container.querySelector("#main-content")).toBeInTheDocument();
  });

  it("contains Suspense boundary (renders without crashing)", () => {
    expect(() => render(<ProofWorkspace {...defaultProps} />)).not.toThrow();
  });

  it("renders package label", () => {
    render(<ProofWorkspace {...defaultProps} />);
    expect(screen.getByText("pass")).toBeInTheDocument();
  });

  it("renders ModeDial tablist", () => {
    render(<ProofWorkspace {...defaultProps} />);
    expect(screen.getByRole("tablist")).toBeInTheDocument();
  });

  it("renders VerdictSeal status", () => {
    render(<ProofWorkspace {...defaultProps} />);
    expect(screen.getByText("PASS")).toBeInTheDocument();
    // "VERIFIED" appears in multiple components (VerdictSeal, IdentityStrip, RightRail)
    const verifiedEls = screen.getAllByText("VERIFIED");
    expect(verifiedEls.length).toBeGreaterThanOrEqual(1);
  });

  it("renders CenterCanvas tabpanel", () => {
    render(<ProofWorkspace {...defaultProps} />);
    expect(screen.getByRole("tabpanel")).toBeInTheDocument();
  });

  it("renders RightRail integrity section", () => {
    render(<ProofWorkspace {...defaultProps} />);
    expect(screen.getByText("Integrity")).toBeInTheDocument();
  });
});
