import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { CodeBlock } from "@/ds/components/CodeBlock";

describe("CodeBlock", () => {
  const code = `function hello() {\n  return "world";\n}`;

  it("renders code content", () => {
    render(<CodeBlock>{code}</CodeBlock>);
    expect(screen.getByText(/function hello/)).toBeInTheDocument();
    expect(screen.getByText(/return "world"/)).toBeInTheDocument();
  });

  it("renders language label when provided", () => {
    render(<CodeBlock language="typescript">{code}</CodeBlock>);
    expect(screen.getByText("typescript")).toBeInTheDocument();
  });

  it("renders copy button by default", () => {
    render(<CodeBlock>{code}</CodeBlock>);
    expect(screen.getByRole("button", { name: "Copy code" })).toBeInTheDocument();
  });

  it("hides copy button when copyable=false", () => {
    render(<CodeBlock copyable={false}>{code}</CodeBlock>);
    expect(screen.queryByRole("button", { name: "Copy code" })).not.toBeInTheDocument();
  });

  it("renders line numbers when enabled", () => {
    const { container } = render(<CodeBlock lineNumbers>{code}</CodeBlock>);
    expect(container.textContent).toContain("1");
    expect(container.textContent).toContain("2");
    expect(container.textContent).toContain("3");
  });

  it("has accessible code region", () => {
    render(<CodeBlock language="json">{code}</CodeBlock>);
    expect(screen.getByRole("region", { name: "json code block" })).toBeInTheDocument();
  });
});
