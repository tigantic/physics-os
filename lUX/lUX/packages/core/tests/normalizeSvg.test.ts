import { describe, it, expect } from "vitest";
import { normalizeSvg } from "../src/util/normalizeSvg.js";

describe("normalizeSvg", () => {
  it("replaces \\r\\n with \\n", () => {
    expect(normalizeSvg("<svg>\r\n<rect/>\r\n</svg>")).toBe("<svg><rect/></svg>");
  });

  it("collapses multiple whitespace into single space", () => {
    expect(normalizeSvg("<svg>   <rect/>   </svg>")).toBe("<svg><rect/></svg>");
  });

  it("strips whitespace between > and <", () => {
    expect(normalizeSvg("<svg>  <g>  <rect/>  </g>  </svg>")).toBe("<svg><g><rect/></g></svg>");
  });

  it("trims leading and trailing whitespace", () => {
    expect(normalizeSvg("  <svg/>  ")).toBe("<svg/>");
  });

  it("handles empty string", () => {
    expect(normalizeSvg("")).toBe("");
  });

  it("handles already-normalized svg", () => {
    const svg = '<svg><rect width="10" height="10"/></svg>';
    expect(normalizeSvg(svg)).toBe(svg);
  });

  it("handles mixed whitespace (tabs, newlines, spaces)", () => {
    expect(normalizeSvg("<svg>\t\n  <circle/>\n\t</svg>")).toBe("<svg><circle/></svg>");
  });
});
