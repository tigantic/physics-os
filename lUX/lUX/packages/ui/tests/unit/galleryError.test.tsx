import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import GalleryError from "@/app/gallery/error";

describe("GalleryError", () => {
  const mockReset = () => {};

  it("renders error message", () => {
    const error = new Error("Test error message");
    render(<GalleryError error={error} reset={mockReset} />);
    expect(screen.getByText("Test error message")).toBeInTheDocument();
  });

  it("renders Viewer Error heading", () => {
    const error = new Error("fail");
    render(<GalleryError error={error} reset={mockReset} />);
    expect(screen.getByText("Viewer Error")).toBeInTheDocument();
  });

  it("renders Render Halted label", () => {
    const error = new Error("fail");
    render(<GalleryError error={error} reset={mockReset} />);
    expect(screen.getByText("Render Halted")).toBeInTheDocument();
  });

  it("has role=alert for assistive tech", () => {
    const error = new Error("fail");
    render(<GalleryError error={error} reset={mockReset} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("renders retry button", () => {
    const error = new Error("fail");
    render(<GalleryError error={error} reset={mockReset} />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders digest when present", () => {
    const error = Object.assign(new Error("fail"), { digest: "abc123" });
    render(<GalleryError error={error} reset={mockReset} />);
    expect(screen.getByText(/Digest: abc123/)).toBeInTheDocument();
  });

  it("does not render digest when absent", () => {
    const error = new Error("no digest");
    render(<GalleryError error={error} reset={mockReset} />);
    expect(screen.queryByText(/Digest:/)).not.toBeInTheDocument();
  });
});
