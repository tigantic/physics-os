import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { Card, CardHeader, CardContent } from "@/ds/components/Card";

describe("Card", () => {
  it("renders children", () => {
    render(<Card>Card content</Card>);
    expect(screen.getByText("Card content")).toBeInTheDocument();
  });

  it("applies default styling classes", () => {
    const { container } = render(<Card>Test</Card>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("rounded-");
    expect(el.className).toContain("border");
  });

  it("merges custom className", () => {
    const { container } = render(<Card className="my-class">Test</Card>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("my-class");
  });
});

describe("CardHeader", () => {
  it("renders children", () => {
    render(<CardHeader>Header</CardHeader>);
    expect(screen.getByText("Header")).toBeInTheDocument();
  });

  it("applies padding classes", () => {
    const { container } = render(<CardHeader>Header</CardHeader>);
    const el = container.firstChild as HTMLElement;
    expect(el.className).toContain("px-6");
  });
});

describe("CardContent", () => {
  it("renders children", () => {
    render(<CardContent>Content</CardContent>);
    expect(screen.getByText("Content")).toBeInTheDocument();
  });

  it("composes with Card", () => {
    const { container } = render(
      <Card>
        <CardHeader>Title</CardHeader>
        <CardContent>Body</CardContent>
      </Card>,
    );
    expect(container.querySelectorAll("div").length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText("Title")).toBeInTheDocument();
    expect(screen.getByText("Body")).toBeInTheDocument();
  });
});
