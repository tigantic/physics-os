import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import type { ProofPackage } from "@luxury/core";
import { IdentityStrip } from "@/features/proof/IdentityStrip";

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

describe("IdentityStrip", () => {
  it("renders h1 with project_id and domain_id", () => {
    render(<IdentityStrip proof={mockProof} />);
    const heading = screen.getByRole("heading", { level: 1 });
    expect(heading).toHaveTextContent("hydro-sim · fluid_dynamics");
  });

  it("renders VerdictSeal with verdict status", () => {
    render(<IdentityStrip proof={mockProof} />);
    expect(screen.getByText("PASS")).toBeInTheDocument();
    expect(screen.getByText("VERIFIED")).toBeInTheDocument();
  });

  it('contains "lUX Proof Viewer" text', () => {
    render(<IdentityStrip proof={mockProof} />);
    expect(screen.getByText("lUX Proof Viewer")).toBeInTheDocument();
  });
});
