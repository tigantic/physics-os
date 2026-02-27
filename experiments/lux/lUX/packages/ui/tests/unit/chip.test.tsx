import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { Chip } from "@/ds/components/Chip";

describe("Chip", () => {
  it("renders children text", () => {
    render(<Chip>Label</Chip>);
    expect(screen.getByText("Label")).toBeInTheDocument();
  });

  it("renders default tone styling", () => {
    const { container } = render(<Chip>Default</Chip>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("inline-flex");
    expect(el.className).toContain("rounded-md");
  });

  it("renders gold tone", () => {
    const { container } = render(<Chip tone="gold">Gold</Chip>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("color-accent");
  });

  it("renders fail tone", () => {
    const { container } = render(<Chip tone="fail">Fail</Chip>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("color-status-fail");
  });

  it("renders warn tone", () => {
    const { container } = render(<Chip tone="warn">Warn</Chip>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("color-status-warn");
  });

  it("renders as span element", () => {
    const { container } = render(<Chip>Test</Chip>);
    expect(container.firstChild?.nodeName).toBe("SPAN");
  });
});
