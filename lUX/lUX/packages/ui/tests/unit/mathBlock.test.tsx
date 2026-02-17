import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

vi.mock("@luxury/core", async () => {
  const actual = await vi.importActual("@luxury/core");
  return { ...actual, renderLatexToSvg: () => "<svg><text>mocked</text></svg>" };
});

import { MathBlock } from "@/features/math/MathBlock";

describe("MathBlock", () => {
  it('renders with role="img"', () => {
    render(<MathBlock latex="E = mc^2" />);
    expect(screen.getByRole("img")).toBeInTheDocument();
  });

  it("has aria-label containing the latex string", () => {
    render(<MathBlock latex="E = mc^2" />);
    const el = screen.getByRole("img");
    expect(el).toHaveAttribute("aria-label", "Math equation: E = mc^2");
  });

  it("renders SVG content via dangerouslySetInnerHTML", () => {
    render(<MathBlock latex="\\int_0^1 f(x) dx" />);
    const el = screen.getByRole("img");
    expect(el.innerHTML).toContain("<svg>");
    expect(el.innerHTML).toContain("mocked");
  });
});
