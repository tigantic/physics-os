import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

// Dynamic import path uses [id] — import directly from the file
import PackageError from "@/app/packages/[id]/error";

describe("PackageError", () => {
  const mockReset = () => {};

  it("renders error message", () => {
    const error = new Error("Package not found");
    render(<PackageError error={error} reset={mockReset} />);
    expect(screen.getByText("Package not found")).toBeInTheDocument();
  });

  it("renders heading and label", () => {
    const error = new Error("fail");
    render(<PackageError error={error} reset={mockReset} />);
    expect(screen.getByText("Package Load Error")).toBeInTheDocument();
    expect(screen.getByText("Render Halted")).toBeInTheDocument();
  });

  it("has role=alert for assistive tech", () => {
    const error = new Error("fail");
    render(<PackageError error={error} reset={mockReset} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("renders retry button and back link", () => {
    const error = new Error("fail");
    render(<PackageError error={error} reset={mockReset} />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Back to Packages" })).toHaveAttribute("href", "/packages");
  });

  it("renders digest when present", () => {
    const error = Object.assign(new Error("fail"), { digest: "d1g3st" });
    render(<PackageError error={error} reset={mockReset} />);
    expect(screen.getByText(/Digest: d1g3st/)).toBeInTheDocument();
  });

  it("does not render digest when absent", () => {
    const error = new Error("no digest");
    render(<PackageError error={error} reset={mockReset} />);
    expect(screen.queryByText(/Digest:/)).not.toBeInTheDocument();
  });
});
