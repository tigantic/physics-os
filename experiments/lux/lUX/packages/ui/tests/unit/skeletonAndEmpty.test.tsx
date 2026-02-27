import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { Skeleton } from "@/ds/components/Skeleton";
import { EmptyState } from "@/ds/components/EmptyState";

describe("Skeleton", () => {
  it("renders a single skeleton bar", () => {
    const { container } = render(<Skeleton />);
    const el = container.firstChild as HTMLElement;
    expect(el.getAttribute("role")).toBe("status");
    expect(el.getAttribute("aria-label")).toBe("Loading");
    expect(el.className).toContain("lux-shimmer-bg");
  });

  it("renders multiple skeleton rows", () => {
    const { container } = render(<Skeleton rows={3} />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.children.length).toBe(3);
  });

  it("applies custom height", () => {
    const { container } = render(<Skeleton heightClass="h-8" />);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("h-8");
  });
});

describe("EmptyState", () => {
  it("renders title", () => {
    render(<EmptyState title="No results" />);
    expect(screen.getByText("No results")).toBeInTheDocument();
  });

  it("renders description when provided", () => {
    render(<EmptyState title="Empty" description="Try a different query" />);
    expect(screen.getByText("Try a different query")).toBeInTheDocument();
  });

  it("renders action when provided", () => {
    render(<EmptyState title="Empty" action={<button type="button">Retry</button>} />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("renders icon when provided", () => {
    render(<EmptyState title="Empty" icon={<span data-testid="icon">📦</span>} />);
    expect(screen.getByTestId("icon")).toBeInTheDocument();
  });
});
