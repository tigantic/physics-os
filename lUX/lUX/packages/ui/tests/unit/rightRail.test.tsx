import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import type { ProofPackage } from "@luxury/core";
import { RightRail } from "@/features/proof/RightRail";

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

describe("RightRail", () => {
  it('renders "Integrity" heading', () => {
    render(<RightRail proof={mockProof} />);
    expect(screen.getByText("Integrity")).toBeInTheDocument();
  });

  it("renders Merkle Root CopyField", () => {
    render(<RightRail proof={mockProof} />);
    expect(screen.getByText("Merkle Root")).toBeInTheDocument();
    expect(screen.getByText("sha256:merkleroot")).toBeInTheDocument();
  });

  it("renders Container Digest CopyField", () => {
    render(<RightRail proof={mockProof} />);
    expect(screen.getByText("Container Digest")).toBeInTheDocument();
    expect(screen.getByText("sha256:deadbeef")).toBeInTheDocument();
  });

  it("shows verification status", () => {
    render(<RightRail proof={mockProof} />);
    expect(screen.getByText("VERIFIED")).toBeInTheDocument();
  });

  it("renders commit hash CopyField", () => {
    render(<RightRail proof={mockProof} />);
    expect(screen.getByText("Commit")).toBeInTheDocument();
    expect(screen.getByText("abc123")).toBeInTheDocument();
  });

  it("renders aside with aria-label", () => {
    const { container } = render(<RightRail proof={mockProof} />);
    const aside = container.querySelector("aside");
    expect(aside).toBeTruthy();
    expect(aside?.getAttribute("aria-label")).toBe("Integrity details");
  });

  it("renders failures list when verification has failures", () => {
    const failProof: ProofPackage = {
      ...mockProof,
      verification: {
        status: "BROKEN_CHAIN",
        verifier_version: "0.1.0",
        failures: [
          { code: "DIGEST_MISMATCH", artifact_id: "A-ts", message: "Hash mismatch" },
          { code: "MISSING_ARTIFACT", message: "Artifact not found" },
        ],
      },
    };
    render(<RightRail proof={failProof} />);
    expect(screen.getByText("Failures")).toBeInTheDocument();
    expect(screen.getByText(/DIGEST_MISMATCH/)).toBeInTheDocument();
    expect(screen.getByText(/\(A-ts\)/)).toBeInTheDocument();
    expect(screen.getByText(/MISSING_ARTIFACT/)).toBeInTheDocument();
  });

  it("does not render failures section when empty", () => {
    render(<RightRail proof={mockProof} />);
    expect(screen.queryByText("Failures")).not.toBeInTheDocument();
  });

  it("falls back to UNVERIFIED when no verification object", () => {
    const noVerify: ProofPackage = {
      ...mockProof,
      verification: undefined as unknown as ProofPackage["verification"],
    };
    render(<RightRail proof={noVerify} />);
    expect(screen.getByText("UNVERIFIED")).toBeInTheDocument();
  });
});
