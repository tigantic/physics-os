import { describe, it, expect } from "vitest";
import { renderLatexToSvg } from "../src/util/renderLatexToSvg.js";

describe("renderLatexToSvg", () => {
  it("renders valid LaTeX to an SVG string", () => {
    const svg = renderLatexToSvg("E = mc^2");
    expect(svg).toContain("<svg");
    expect(svg).toContain("</svg>");
  });

  it("renders in display mode by default", () => {
    const svg = renderLatexToSvg("\\int_0^1 f(x) dx", true);
    expect(svg).toContain("<svg");
  });

  it("handles inline mode", () => {
    const svg = renderLatexToSvg("x^2", false);
    expect(svg).toContain("<svg");
  });

  it("handles empty latex string without throwing", () => {
    const svg = renderLatexToSvg("");
    expect(typeof svg).toBe("string");
  });

  it("returns sanitized output (no script tags)", () => {
    const svg = renderLatexToSvg("a + b");
    expect(svg.toLowerCase()).not.toContain("<script");
    expect(svg.toLowerCase()).not.toContain("onclick");
    expect(svg.toLowerCase()).not.toContain("onerror");
  });

  it("returns normalized output (no \\r\\n)", () => {
    const svg = renderLatexToSvg("\\frac{a}{b}");
    expect(svg).not.toContain("\r\n");
  });
});
