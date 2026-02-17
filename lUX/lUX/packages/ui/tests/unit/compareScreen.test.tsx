import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import React from "react";
import type { ProofPackage, DomainPack } from "@luxury/core";

// Mock heavy child components to isolate CompareScreen logic
vi.mock("@/features/proof/DataValueView", () => ({
  DataValueNumberView: ({ metricId }: { metricId: string }) => (
    <span data-testid={`dv-${metricId}`}>{metricId}</span>
  ),
}));

import { CompareScreen } from "@/features/screens/Compare";

/* ── Fixtures ────────────────────────────────────────────────── */

function makeProof(id: string, metricValue = 42): ProofPackage {
  return {
    schema_version: "1.0.0",
    meta: {
      id,
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
    timeline: {
      step_count: 1,
      steps: [
        {
          step_index: 0,
          state_hash: "sha256:abc123",
          metrics: {
            accuracy: { status: "ok" as const, value: metricValue },
            loss: { status: "ok" as const, value: 0.01 },
          },
          artifact_refs: [],
        },
      ],
    },
  };
}

const mockDomain: DomainPack = {
  id: "fluid_dynamics",
  version: "1.0.0",
  metrics: {
    accuracy: {
      label: "Accuracy",
      unit: "%",
      format: "fixed",
      precision: 2,
    },
    loss: {
      label: "Loss",
      unit: "",
      format: "scientific",
      precision: 4,
    },
  },
  gate_packs: {},
  viewers: [],
  templates: {
    executive_summary_metric_ids: ["accuracy", "loss"],
    publication_sections: [],
    citation_format: "bibtex",
  },
};

/* ── Tests ───────────────────────────────────────────────────── */

describe("CompareScreen", () => {
  const proof = makeProof("current-run");
  const baseline = makeProof("baseline-run", 38);

  describe("without baseline", () => {
    it("renders empty state", () => {
      render(<CompareScreen proof={proof} domain={mockDomain} />);
      expect(screen.getByText("No baseline")).toBeInTheDocument();
      expect(
        screen.getByText("Select a baseline proof to enable comparison."),
      ).toBeInTheDocument();
    });

    it("renders Data Unavailable chip", () => {
      render(<CompareScreen proof={proof} domain={mockDomain} />);
      expect(screen.getByText("Data Unavailable")).toBeInTheDocument();
    });

    it("renders Compare header text", () => {
      render(<CompareScreen proof={proof} domain={mockDomain} />);
      expect(screen.getByText("Compare")).toBeInTheDocument();
    });
  });

  describe("with baseline", () => {
    it("renders the comparison table", () => {
      render(<CompareScreen proof={proof} baseline={baseline} domain={mockDomain} />);
      expect(screen.getByText("Compare")).toBeInTheDocument();
      expect(screen.getByText("Metric comparison: current vs baseline")).toBeInTheDocument();
    });

    it("renders metric rows from executive_summary_metric_ids", () => {
      render(<CompareScreen proof={proof} baseline={baseline} domain={mockDomain} />);
      // Each metric appears twice (current + baseline columns)
      expect(screen.getAllByTestId("dv-accuracy")).toHaveLength(2);
      expect(screen.getAllByTestId("dv-loss")).toHaveLength(2);
    });

    it("renders KeyValueGrid with proof IDs", () => {
      render(<CompareScreen proof={proof} baseline={baseline} domain={mockDomain} />);
      expect(screen.getByText("current-run")).toBeInTheDocument();
      expect(screen.getByText("baseline-run")).toBeInTheDocument();
    });

    it("renders table column headers", () => {
      render(<CompareScreen proof={proof} baseline={baseline} domain={mockDomain} />);
      expect(screen.getByText("Metric")).toBeInTheDocument();
      // "Current" and "Baseline" also appear in KeyValueGrid, so use getAllByText
      expect(screen.getAllByText("Current").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Baseline").length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("BaselineSelector", () => {
    const availableIds = ["run-001", "run-002", "run-003"];

    it("renders selector when availableBaselineIds is non-empty (no baseline)", () => {
      render(
        <CompareScreen
          proof={proof}
          domain={mockDomain}
          availableBaselineIds={availableIds}
          onBaselineSelect={vi.fn()}
        />,
      );
      const select = screen.getByRole("combobox");
      expect(select).toBeInTheDocument();
    });

    it("renders selector when availableBaselineIds is non-empty (with baseline)", () => {
      render(
        <CompareScreen
          proof={proof}
          baseline={baseline}
          domain={mockDomain}
          availableBaselineIds={availableIds}
          onBaselineSelect={vi.fn()}
        />,
      );
      const select = screen.getByRole("combobox");
      expect(select).toBeInTheDocument();
    });

    it("does not render selector when availableBaselineIds is empty", () => {
      render(
        <CompareScreen
          proof={proof}
          domain={mockDomain}
          availableBaselineIds={[]}
          onBaselineSelect={vi.fn()}
        />,
      );
      expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    });

    it("calls onBaselineSelect when user selects an option", () => {
      const onSelect = vi.fn();
      render(
        <CompareScreen
          proof={proof}
          domain={mockDomain}
          availableBaselineIds={availableIds}
          onBaselineSelect={onSelect}
        />,
      );
      const select = screen.getByRole("combobox");
      fireEvent.change(select, { target: { value: "run-002" } });
      expect(onSelect).toHaveBeenCalledWith("run-002");
    });

    it("renders all available IDs as options", () => {
      render(
        <CompareScreen
          proof={proof}
          domain={mockDomain}
          availableBaselineIds={availableIds}
          onBaselineSelect={vi.fn()}
        />,
      );
      for (const id of availableIds) {
        expect(screen.getByText(id)).toBeInTheDocument();
      }
    });

    it("renders 'Select baseline…' placeholder option", () => {
      render(
        <CompareScreen
          proof={proof}
          domain={mockDomain}
          availableBaselineIds={availableIds}
          onBaselineSelect={vi.fn()}
        />,
      );
      expect(screen.getByText("Select baseline…")).toBeInTheDocument();
    });

    it("shows baseline label", () => {
      render(
        <CompareScreen
          proof={proof}
          domain={mockDomain}
          availableBaselineIds={availableIds}
          onBaselineSelect={vi.fn()}
        />,
      );
      expect(screen.getByText("Baseline")).toBeInTheDocument();
    });
  });

  describe("empty domain metrics", () => {
    it("renders empty table state when domain has no metric ids", () => {
      const emptyDomain: DomainPack = {
        ...mockDomain,
        templates: {
          ...mockDomain.templates,
          executive_summary_metric_ids: [],
        },
      };
      render(
        <CompareScreen proof={proof} baseline={baseline} domain={emptyDomain} />,
      );
      expect(screen.getByText("No metrics")).toBeInTheDocument();
      expect(
        screen.getByText("Domain has no metrics configured."),
      ).toBeInTheDocument();
    });
  });
});
