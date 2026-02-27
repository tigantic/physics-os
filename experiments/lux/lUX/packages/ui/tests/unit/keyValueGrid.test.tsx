import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
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
    const hashDd = Array.from(dds).find((dd) => dd.textContent?.includes("sha256:abc123"));
    expect(hashDd?.className).toContain("font-mono");
  });

  it("renders as dl element", () => {
    const { container } = render(<KeyValueGrid entries={entries} />);
    expect(container.querySelector("dl")).toBeInTheDocument();
  });

  it("shows copy button when copyable is true", () => {
    render(
      <KeyValueGrid
        entries={[{ label: "Hash", value: "sha256:abc", mono: true, copyable: true }]}
      />,
    );
    expect(screen.getByRole("button", { name: /Copy Hash/ })).toBeInTheDocument();
  });

  it("copies value to clipboard on copy button click", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    render(
      <KeyValueGrid
        entries={[{ label: "Hash", value: "sha256:abc", mono: true, copyable: true }]}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Copy Hash/ }));
    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("sha256:abc");
    });
  });

  it("does not show copy button when copyable is false or omitted", () => {
    render(<KeyValueGrid entries={entries} />);
    expect(screen.queryByRole("button", { name: /Copy/ })).not.toBeInTheDocument();
  });

  it("adds title attribute to string values for truncated text tooltip", () => {
    const { container } = render(<KeyValueGrid entries={entries} />);
    const spans = container.querySelectorAll("span[title]");
    const hashSpan = Array.from(spans).find((s) => s.getAttribute("title") === "sha256:abc123");
    expect(hashSpan).toBeDefined();
  });
});
