import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import PackagesListError from "@/app/packages/error";

describe("PackagesListError", () => {
  const mockReset = () => {};

  it("renders error message", () => {
    const error = new Error("Provider unavailable");
    render(<PackagesListError error={error} reset={mockReset} />);
    expect(screen.getByText("Provider unavailable")).toBeInTheDocument();
  });

  it("renders heading and label", () => {
    const error = new Error("fail");
    render(<PackagesListError error={error} reset={mockReset} />);
    expect(screen.getByText("Package List Error")).toBeInTheDocument();
    expect(screen.getByText("Render Halted")).toBeInTheDocument();
  });

  it("has role=alert for assistive tech", () => {
    const error = new Error("fail");
    render(<PackagesListError error={error} reset={mockReset} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("renders retry button", () => {
    const error = new Error("fail");
    render(<PackagesListError error={error} reset={mockReset} />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders digest when present", () => {
    const error = Object.assign(new Error("fail"), { digest: "abc123" });
    render(<PackagesListError error={error} reset={mockReset} />);
    expect(screen.getByText(/Digest: abc123/)).toBeInTheDocument();
  });

  it("does not render digest when absent", () => {
    const error = new Error("no digest");
    render(<PackagesListError error={error} reset={mockReset} />);
    expect(screen.queryByText(/Digest:/)).not.toBeInTheDocument();
  });
});
