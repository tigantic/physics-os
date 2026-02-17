import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import GlobalError from "@/app/global-error";

describe("GlobalError", () => {
  const defaultProps = {
    error: new Error("Layout crash"),
    reset: vi.fn(),
  };

  it("has role=alert for assistive tech", () => {
    console.error = vi.fn();
    render(<GlobalError {...defaultProps} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("has aria-live=assertive", () => {
    console.error = vi.fn();
    render(<GlobalError {...defaultProps} />);
    expect(screen.getByRole("alert")).toHaveAttribute("aria-live", "assertive");
  });

  it("renders h1 heading", () => {
    console.error = vi.fn();
    render(<GlobalError {...defaultProps} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Viewer Unrecoverable");
  });

  it("renders error message", () => {
    console.error = vi.fn();
    render(<GlobalError {...defaultProps} />);
    expect(screen.getByText("Layout crash")).toBeInTheDocument();
  });

  it("renders Fatal Render Error label", () => {
    console.error = vi.fn();
    render(<GlobalError {...defaultProps} />);
    expect(screen.getByText("Fatal Render Error")).toBeInTheDocument();
  });

  it("renders retry button", () => {
    console.error = vi.fn();
    render(<GlobalError {...defaultProps} />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders digest when present", () => {
    console.error = vi.fn();
    const err = new Error("crash") as Error & { digest?: string };
    err.digest = "digest-xyz";
    render(<GlobalError error={err} reset={vi.fn()} />);
    expect(screen.getByText(/digest-xyz/)).toBeInTheDocument();
  });

  it("does not render digest when absent", () => {
    console.error = vi.fn();
    render(<GlobalError {...defaultProps} />);
    expect(screen.queryByText(/Digest:/)).not.toBeInTheDocument();
  });
});
