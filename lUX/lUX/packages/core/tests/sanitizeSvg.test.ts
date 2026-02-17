import { describe, it, expect } from "vitest";
import { sanitizeSvg } from "../src/util/sanitizeSvg.js";

describe("sanitizeSvg", () => {
  it("passes clean SVG through unchanged", () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100"/></svg>';
    expect(sanitizeSvg(svg)).toBe(svg);
  });

  it("strips <script> elements", () => {
    const input = "<svg><script>alert(1)</script><rect/></svg>";
    expect(sanitizeSvg(input)).not.toContain("<script");
    expect(sanitizeSvg(input)).toContain("<rect/>");
  });

  it("strips self-closing <script/> elements", () => {
    const input = '<svg><script src="evil.js"/><rect/></svg>';
    expect(sanitizeSvg(input)).not.toContain("<script");
  });

  it("strips on* event handler attributes", () => {
    const input = '<svg><rect onclick="alert(1)" onload="fetch(\'x\')" width="10"/></svg>';
    const result = sanitizeSvg(input);
    expect(result).not.toContain("onclick");
    expect(result).not.toContain("onload");
    expect(result).toContain("width");
  });

  it("strips javascript: protocol in href", () => {
    const input = '<svg><a href="javascript:alert(1)"><text>x</text></a></svg>';
    expect(sanitizeSvg(input)).not.toContain("javascript:");
    expect(sanitizeSvg(input)).toContain("#blocked:");
  });

  it("strips javascript: protocol in xlink:href", () => {
    const input = '<svg><a xlink:href="javascript:alert(1)"><text>x</text></a></svg>';
    expect(sanitizeSvg(input)).not.toContain("javascript:");
  });

  it("strips vbscript: protocol in href", () => {
    const input = '<svg><a href="vbscript:MsgBox(1)"><text>x</text></a></svg>';
    expect(sanitizeSvg(input)).not.toContain("vbscript:");
  });

  it("strips data: URIs in href (non-image)", () => {
    const input = '<svg><a href="data:text/html,<h1>XSS</h1>"><text>x</text></a></svg>';
    expect(sanitizeSvg(input)).toContain("#blocked:");
  });

  it("preserves data: image URIs in href", () => {
    const input = '<svg><image href="data:image/png;base64,abc"/></svg>';
    expect(sanitizeSvg(input)).toContain("data:image/png;base64,abc");
  });

  it("strips <foreignObject> elements", () => {
    const input = "<svg><foreignObject><body><script>alert(1)</script></body></foreignObject><rect/></svg>";
    expect(sanitizeSvg(input)).not.toContain("foreignObject");
    expect(sanitizeSvg(input)).toContain("<rect/>");
  });

  it("strips <use> with external references", () => {
    const input = '<svg><use href="https://evil.com/sprite.svg#icon"/></svg>';
    expect(sanitizeSvg(input)).not.toContain("evil.com");
  });

  it("preserves <use> with local fragment references", () => {
    const input = '<svg><defs><g id="icon"><rect/></g></defs><use href="#icon"/></svg>';
    expect(sanitizeSvg(input)).toContain('href="#icon"');
  });

  it("handles multiple threats in a single SVG", () => {
    const input = [
      "<svg>",
      "<script>alert(1)</script>",
      '<rect onclick="hack()" width="10"/>',
      '<a href="javascript:void(0)"><text>link</text></a>',
      "<foreignObject><div>embedded HTML</div></foreignObject>",
      "</svg>",
    ].join("");
    const result = sanitizeSvg(input);
    expect(result).not.toContain("<script");
    expect(result).not.toContain("onclick");
    expect(result).not.toContain("javascript:");
    expect(result).not.toContain("foreignObject");
    expect(result).toContain("width");
  });
});
