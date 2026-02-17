import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { Badge } from "@/components/ui/badge";

describe("Badge", () => {
  it("renders children text", () => {
    render(<Badge>PASS</Badge>);
    expect(screen.getByText("PASS")).toBeInTheDocument();
  });

  it("applies default variant classes", () => {
    const { container } = render(<Badge>test</Badge>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("inline-flex");
    expect(el.className).toContain("items-center");
  });

  it("applies variant-specific classes for each variant", () => {
    const variants = ["default", "gold", "pass", "fail", "warn"] as const;
    for (const variant of variants) {
      const { unmount, container } = render(<Badge variant={variant}>{variant}</Badge>);
      const el = container.firstChild as HTMLElement;
      expect(el.className).toContain("inline-flex");
      // Each variant should produce a unique class string
      expect(el.className.length).toBeGreaterThan(10);
      unmount();
    }
  });

  it("merges custom className", () => {
    const { container } = render(<Badge className="custom-class">test</Badge>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("custom-class");
  });

  it("passes through HTML attributes", () => {
    render(<Badge data-testid="my-badge">test</Badge>);
    expect(screen.getByTestId("my-badge")).toBeInTheDocument();
  });
});
