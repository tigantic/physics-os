import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import RootError from "@/app/error";

describe("RootError", () => {
  const defaultProps = {
    error: new Error("Unexpected token in JSON"),
    reset: vi.fn(),
  };

  it("has role=alert for assistive tech", () => {
    console.error = vi.fn();
    render(<RootError {...defaultProps} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("has aria-live=assertive", () => {
    console.error = vi.fn();
    render(<RootError {...defaultProps} />);
    expect(screen.getByRole("alert")).toHaveAttribute("aria-live", "assertive");
  });

  it("renders h1 heading", () => {
    console.error = vi.fn();
    render(<RootError {...defaultProps} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Something went wrong");
  });

  it("renders error message", () => {
    console.error = vi.fn();
    render(<RootError {...defaultProps} />);
    expect(screen.getByText("Unexpected token in JSON")).toBeInTheDocument();
  });

  it("renders Application Error label", () => {
    console.error = vi.fn();
    render(<RootError {...defaultProps} />);
    expect(screen.getByText("Application Error")).toBeInTheDocument();
  });

  it("renders retry button", () => {
    console.error = vi.fn();
    render(<RootError {...defaultProps} />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders digest when present", () => {
    console.error = vi.fn();
    const err = new Error("fail") as Error & { digest?: string };
    err.digest = "abc123";
    render(<RootError error={err} reset={vi.fn()} />);
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
  });

  it("does not render digest when absent", () => {
    console.error = vi.fn();
    render(<RootError {...defaultProps} />);
    expect(screen.queryByText(/Digest:/)).not.toBeInTheDocument();
  });
});
