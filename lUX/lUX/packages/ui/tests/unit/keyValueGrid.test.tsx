import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { KeyValueGrid } from "@/ds/components/KeyValueGrid";

describe("KeyValueGrid", () => {
  const entries = [
    { label: "Project", value: "hydro-sim" },
    { label: "Hash", value: "sha256:abc123", mono: true },
    { label: "Status", value: "Verified" },
  ];

  it("renders all label-value pairs", () => {
    render(<KeyValueGrid entries={entries} />);
    expect(screen.getByText("Project")).toBeInTheDocument();
    expect(screen.getByText("hydro-sim")).toBeInTheDocument();
    expect(screen.getByText("Hash")).toBeInTheDocument();
    expect(screen.getByText("sha256:abc123")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("Verified")).toBeInTheDocument();
  });

  it("applies mono font to mono entries", () => {
    const { container } = render(<KeyValueGrid entries={entries} />);
    const dds = container.querySelectorAll("dd");
    const hashDd = Array.from(dds).find((dd) => dd.textContent === "sha256:abc123");
    expect(hashDd?.className).toContain("font-mono");
  });

  it("renders as dl element", () => {
    const { container } = render(<KeyValueGrid entries={entries} />);
    expect(container.querySelector("dl")).toBeInTheDocument();
  });
});
