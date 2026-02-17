import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import React from "react";
import { DataValueNumberView } from "@/features/proof/DataValueView";
import type { DataValue, DomainPack } from "@luxury/core";

function makeDomain(overrides: Partial<DomainPack["metrics"][string]> = {}): DomainPack {
  return {
    id: "com.physics.test",
    version: "1.0.0",
    metrics: {
      energy: {
        label: "Energy",
        unit: "J",
        format: "scientific",
        precision: 4,
        ...overrides,
      },
    },
    gate_packs: {},
    viewers: [],
    templates: {
      executive_summary_metric_ids: [],
      publication_sections: [],
      citation_format: "bibtex",
    },
  };
}

describe("DataValueNumberView", () => {
  it("renders scientific notation per metric config", () => {
    const dv: DataValue<number> = { status: "ok", value: 1.23456e-7 };
    const domain = makeDomain({ format: "scientific", precision: 4 });
    render(<DataValueNumberView dv={dv} metricId="energy" domain={domain} />);
    expect(screen.getByText("1.2346e-7")).toBeInTheDocument();
  });

  it("renders fixed notation", () => {
    const dv: DataValue<number> = { status: "ok", value: 3.14159 };
    const domain = makeDomain({ format: "fixed", precision: 2 });
    render(<DataValueNumberView dv={dv} metricId="energy" domain={domain} />);
    expect(screen.getByText("3.14")).toBeInTheDocument();
  });

  it("renders engineering notation", () => {
    const dv: DataValue<number> = { status: "ok", value: 1500 };
    const domain = makeDomain({ format: "engineering", precision: 2 });
    render(<DataValueNumberView dv={dv} metricId="energy" domain={domain} />);
    expect(screen.getByText("1.50e3")).toBeInTheDocument();
  });

  it("shows Data Unavailable chip for missing data", () => {
    const dv: DataValue<number> = { status: "missing", reason: "Not computed" };
    const domain = makeDomain();
    render(<DataValueNumberView dv={dv} metricId="energy" domain={domain} />);
    expect(screen.getByText("Data Unavailable")).toBeInTheDocument();
  });

  it("shows Invalid chip for invalid data", () => {
    const dv: DataValue<number> = { status: "invalid", reason: "NaN" };
    const domain = makeDomain();
    render(<DataValueNumberView dv={dv} metricId="energy" domain={domain} />);
    expect(screen.getByText("Invalid")).toBeInTheDocument();
  });

  it("shows Invalid for out-of-range values when validity_range is set", () => {
    const dv: DataValue<number> = { status: "ok", value: 2.0 };
    const domain = makeDomain({ validity_range: [0, 1] });
    const { container } = render(<DataValueNumberView dv={dv} metricId="energy" domain={domain} />);
    expect(within(container).getByText("Invalid")).toBeInTheDocument();
  });

  it("renders normally for in-range values", () => {
    const dv: DataValue<number> = { status: "ok", value: 0.5 };
    const domain = makeDomain({ format: "fixed", precision: 1, validity_range: [0, 1] });
    render(<DataValueNumberView dv={dv} metricId="energy" domain={domain} />);
    expect(screen.getByText("0.5")).toBeInTheDocument();
  });

  it("falls back to precision 4 for unknown metric", () => {
    const dv: DataValue<number> = { status: "ok", value: 1.23456789 };
    const domain = makeDomain();
    render(<DataValueNumberView dv={dv} metricId="unknown_metric" domain={domain} />);
    expect(screen.getByText("1.2346")).toBeInTheDocument();
  });
});
