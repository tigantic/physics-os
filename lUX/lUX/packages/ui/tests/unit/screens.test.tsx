import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { SummaryScreen } from "@/features/screens/Summary";
import { TimelineScreen } from "@/features/screens/Timeline";
import { EvidenceScreen } from "@/features/screens/Evidence";
import { GatesScreen } from "@/features/screens/Gates";
import { IntegrityScreen } from "@/features/screens/Integrity";
import { CompareScreen } from "@/features/screens/Compare";
import { ReproduceScreen } from "@/features/screens/Reproduce";
import { makeProof, makeDomain } from "./helpers/mockData";

// MathJax does not work in jsdom — stub the MathBlock component
vi.mock("@/features/math/MathBlock", () => ({
  MathBlock: ({ latex }: { latex: string }) =>
    React.createElement("div", { "data-testid": "math-block" }, `[LaTeX: ${latex}]`),
}));

const proof = makeProof();
const domain = makeDomain();

// ---------------------------------------------------------------------------
// SummaryScreen
// ---------------------------------------------------------------------------
describe("SummaryScreen", () => {
  it("renders overview heading and metric labels", () => {
    render(<SummaryScreen proof={proof} domain={domain} mode="REVIEW" />);
    expect(screen.getByText("Overview")).toBeInTheDocument();
    expect(screen.getByText("Conservation Residual")).toBeInTheDocument();
    expect(screen.getByText("L2 Drift")).toBeInTheDocument();
  });

  it("shows Paper View card in PUBLICATION mode", () => {
    render(<SummaryScreen proof={proof} domain={domain} mode="PUBLICATION" />);
    expect(screen.getByText("Paper View")).toBeInTheDocument();
    expect(screen.getByTestId("math-block")).toBeInTheDocument();
  });

  it("hides Paper View in non-PUBLICATION modes", () => {
    render(<SummaryScreen proof={proof} domain={domain} mode="EXECUTIVE" />);
    expect(screen.queryByText("Paper View")).not.toBeInTheDocument();
    expect(screen.queryByTestId("math-block")).not.toBeInTheDocument();
  });

  it("shows verdict reason when present", () => {
    render(<SummaryScreen proof={proof} domain={domain} mode="REVIEW" />);
    expect(screen.getAllByText("All gates passed").length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// TimelineScreen
// ---------------------------------------------------------------------------
describe("TimelineScreen", () => {
  it("renders step count and step rows", () => {
    render(<TimelineScreen proof={proof} domain={domain} />);
    expect(screen.getByText("Timeline")).toBeInTheDocument();
    expect(screen.getByText("2 steps")).toBeInTheDocument();
    expect(screen.getByText("Step 0")).toBeInTheDocument();
    expect(screen.getByText("Step 1")).toBeInTheDocument();
  });

  it("displays state hashes", () => {
    render(<TimelineScreen proof={proof} domain={domain} />);
    expect(screen.getAllByText("sha256:" + "0".repeat(64)).length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// EvidenceScreen
// ---------------------------------------------------------------------------
describe("EvidenceScreen", () => {
  it("renders artifact count and artifact IDs", () => {
    render(<EvidenceScreen proof={proof} />);
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.getByText("1 artifacts")).toBeInTheDocument();
    expect(screen.getByText("A-ts · time_series")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// GatesScreen
// ---------------------------------------------------------------------------
describe("GatesScreen", () => {
  it("renders gate count and gate details", () => {
    render(<GatesScreen proof={proof} domain={domain} />);
    expect(screen.getByText("Gates")).toBeInTheDocument();
    expect(screen.getByText("1 evaluated")).toBeInTheDocument();
    expect(screen.getByText(/G-001/)).toBeInTheDocument();
  });

  it("shows PASS chip for passing gate", () => {
    render(<GatesScreen proof={proof} domain={domain} />);
    expect(screen.getAllByText("PASS").length).toBeGreaterThanOrEqual(1);
  });

  it("shows FAIL chip for failing gate", () => {
    const failProof = makeProof({
      gate_results: {
        "G-001": {
          gate_id: "G-001",
          metric_id: "conservation_residual",
          value: { status: "ok", value: 1e-3 },
          passed: { status: "ok", value: false },
          margin: { status: "ok", value: -0.5 },
        },
      },
    });
    render(<GatesScreen proof={failProof} domain={domain} />);
    expect(screen.getAllByText("FAIL").length).toBeGreaterThanOrEqual(1);
  });

  it("shows Data Unavailable for missing gate status", () => {
    const missingProof = makeProof({
      gate_results: {
        "G-001": {
          gate_id: "G-001",
          metric_id: "conservation_residual",
          value: { status: "missing", reason: "no data" },
          passed: { status: "missing", reason: "no data" },
          margin: { status: "missing", reason: "no data" },
        },
      },
    });
    render(<GatesScreen proof={missingProof} domain={domain} />);
    expect(screen.getAllByText("Data Unavailable").length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// IntegrityScreen
// ---------------------------------------------------------------------------
describe("IntegrityScreen", () => {
  it("renders verification status and merkle root", () => {
    render(<IntegrityScreen proof={proof} />);
    expect(screen.getByText("Verification")).toBeInTheDocument();
    expect(screen.getByText("VERIFIED")).toBeInTheDocument();
    expect(screen.getByText("Merkle Root")).toBeInTheDocument();
    expect(screen.getByText("sha256:" + "c".repeat(64))).toBeInTheDocument();
  });

  it("shows Chain Intact when no failures", () => {
    render(<IntegrityScreen proof={proof} />);
    expect(screen.getAllByText("Chain Intact").length).toBeGreaterThanOrEqual(1);
  });

  it("shows failure codes when verification has failures", () => {
    const brokenProof = makeProof({
      verification: {
        status: "BROKEN_CHAIN",
        verifier_version: "0.1.0",
        failures: [
          { code: "MERKLE_MISMATCH", message: "Root doesn't match" },
          { code: "SIG_INVALID", message: "Bad signature" },
        ],
      },
    });
    render(<IntegrityScreen proof={brokenProof} />);
    expect(screen.getAllByText("MERKLE_MISMATCH").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("SIG_INVALID").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("Chain Intact")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// CompareScreen
// ---------------------------------------------------------------------------
describe("CompareScreen", () => {
  it("shows Data Unavailable when no baseline", () => {
    render(<CompareScreen proof={proof} domain={domain} />);
    expect(screen.getAllByText("Compare").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("No baseline selected")).toBeInTheDocument();
    expect(screen.getAllByText("Data Unavailable").length).toBeGreaterThanOrEqual(1);
  });

  it("shows side-by-side metrics when baseline present", () => {
    render(<CompareScreen proof={proof} baseline={proof} domain={domain} />);
    expect(screen.getAllByText("Compare").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Baseline/)).toBeInTheDocument();
    // Metric labels appear in each comparison row
    expect(screen.getAllByText("Conservation Residual").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("L2 Drift").length).toBeGreaterThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// ReproduceScreen
// ---------------------------------------------------------------------------
describe("ReproduceScreen", () => {
  it("renders reproduce heading and deterministic command", () => {
    render(<ReproduceScreen proof={proof} />);
    expect(screen.getByText("Reproduce")).toBeInTheDocument();
    expect(screen.getByText("Deterministic command")).toBeInTheDocument();
  });

  it("contains the docker command with correct digest and seed", () => {
    const { container } = render(<ReproduceScreen proof={proof} />);
    const text = container.textContent ?? "";
    expect(text).toContain("sha256:" + "a".repeat(64));
    expect(text).toContain("--seed 42");
  });

  it("renders invalid metadata chip when digest is malformed", () => {
    const bad = makeProof({
      meta: {
        ...proof.meta,
        environment: {
          ...proof.meta.environment,
          container_digest: "EVIL; rm -rf /" as never,
        },
      },
    });
    render(<ReproduceScreen proof={bad} />);
    expect(screen.getByText("Invalid reproduction metadata")).toBeInTheDocument();
  });
});
