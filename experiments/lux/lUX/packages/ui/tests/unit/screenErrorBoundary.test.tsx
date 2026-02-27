import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ScreenErrorBoundary } from "@/ds/components/ScreenErrorBoundary";

/* Utility: a component that throws on render */
function ThrowingChild({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error("Screen exploded");
  return <div>Healthy content</div>;
}

describe("ScreenErrorBoundary", () => {
  beforeEach(() => {
    // Suppress React error boundary console.error noise
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("renders children when no error", () => {
    render(
      <ScreenErrorBoundary screenName="Test">
        <div>Normal content</div>
      </ScreenErrorBoundary>,
    );
    expect(screen.getByText("Normal content")).toBeDefined();
  });

  it("shows error card when child throws", () => {
    render(
      <ScreenErrorBoundary screenName="Timeline">
        <ThrowingChild shouldThrow />
      </ScreenErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toBeDefined();
    expect(screen.getByText("Screen Error")).toBeDefined();
    expect(screen.getByText("Timeline")).toBeDefined();
    expect(screen.getByText("Screen exploded")).toBeDefined();
  });

  it("provides retry button that resets the boundary", () => {
    // First render: child throws → error card
    let shouldThrow = true;
    function ConditionalChild() {
      if (shouldThrow) throw new Error("Screen exploded");
      return <div>Healthy content</div>;
    }

    const { rerender } = render(
      <ScreenErrorBoundary screenName="Gates">
        <ConditionalChild />
      </ScreenErrorBoundary>,
    );

    expect(screen.getByText("Screen Error")).toBeDefined();

    // Fix the child, then retry — boundary resets, child renders normally
    shouldThrow = false;
    fireEvent.click(screen.getByText("Retry"));

    rerender(
      <ScreenErrorBoundary screenName="Gates">
        <ConditionalChild />
      </ScreenErrorBoundary>,
    );
    expect(screen.getByText("Healthy content")).toBeDefined();
  });
});
