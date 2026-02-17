import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import type { ProofPackage } from "@luxury/core";

// TimeSeriesViewer imports server-only; stub it out
vi.mock("server-only", () => ({}));

import { PrimaryViewer } from "@/features/viewers/PrimaryViewer";

function makeProof(overrides?: Partial<ProofPackage>): ProofPackage {
  const base: ProofPackage = {
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
  return { ...base, ...overrides } as ProofPackage;
}

describe("PrimaryViewer", () => {
  it("BROKEN_CHAIN status shows failure chip", () => {
    const proof = makeProof({
      verification: { status: "BROKEN_CHAIN", verifier_version: "0.1.0", failures: [] },
    });
    render(<PrimaryViewer proof={proof} bundleDir="/tmp/bundle" />);
    expect(screen.getByText("BROKEN_CHAIN")).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it('no time_series artifact shows "Data Unavailable"', () => {
    const proof = makeProof({ artifacts: {} });
    render(<PrimaryViewer proof={proof} bundleDir="/tmp/bundle" />);
    expect(screen.getByText("Data Unavailable")).toBeInTheDocument();
  });
});
