import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import type { ProofPackage, ProofMode } from "@luxury/core";
import { LeftRail } from "@/features/proof/LeftRail";

vi.mock("next/link", () => ({
  default: ({ children, ...props }: React.PropsWithChildren<React.AnchorHTMLAttributes<HTMLAnchorElement>>) => (
    <a {...props}>{children}</a>
  ),
}));

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
    "claim-2": {
      id: "claim-2",
      statement: "Momentum conserved",
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

describe("LeftRail", () => {
  describe("EXECUTIVE mode", () => {
    it("renders 5 fixture links", () => {
      render(<LeftRail proof={mockProof} fixture="pass" mode={"EXECUTIVE" as ProofMode} />);
      const links = screen.getAllByRole("link");
      expect(links).toHaveLength(5);
    });

    it("active fixture shows 'Active' chip", () => {
      render(<LeftRail proof={mockProof} fixture="pass" mode={"EXECUTIVE" as ProofMode} />);
      expect(screen.getByText("Active")).toBeInTheDocument();
    });

    it("active link has aria-current='page'", () => {
      render(<LeftRail proof={mockProof} fixture="warn" mode={"EXECUTIVE" as ProofMode} />);
      const activeLink = screen.getByRole("link", { current: "page" });
      expect(activeLink).toBeInTheDocument();
      expect(activeLink).toHaveAttribute("href", "/packages/warn?mode=EXECUTIVE");
    });
  });

  describe("REVIEW mode", () => {
    it("renders claims list", () => {
      render(<LeftRail proof={mockProof} fixture="pass" mode={"REVIEW" as ProofMode} />);
      expect(screen.getByText("Energy conserved")).toBeInTheDocument();
      expect(screen.getByText("Momentum conserved")).toBeInTheDocument();
    });

    it("shows claim count", () => {
      render(<LeftRail proof={mockProof} fixture="pass" mode={"REVIEW" as ProofMode} />);
      expect(screen.getByText("2 total")).toBeInTheDocument();
    });
  });
});
