import { describe, it, expect } from "vitest";
import { TOKENS } from "@/ds/tokens";

describe("TOKENS", () => {
  it("exports color.bg tokens", () => {
    expect(TOKENS.color.bg.base).toBe("#0B0C10");
    expect(TOKENS.color.bg.raised).toBe("#13141A");
    expect(TOKENS.color.bg.hover).toBe("#1B1C24");
    expect(TOKENS.color.bg.surface).toBe("#20212B");
  });

  it("exports color.text tokens", () => {
    expect(TOKENS.color.text.primary).toBe("#EAECF0");
    expect(TOKENS.color.text.secondary).toBe("#8A8EA0");
    expect(TOKENS.color.text.tertiary).toBe("#5A5E70");
  });

  it("exports accent base token (cobalt)", () => {
    expect(TOKENS.color.accent.base).toBe("#4B7BF5");
    expect(TOKENS.color.accent.strong).toBe("#6B96FF");
  });

  it("exports status tokens", () => {
    expect(TOKENS.color.status.pass).toBe("#34B870");
    expect(TOKENS.color.status.fail).toBe("#E05252");
    expect(TOKENS.color.status.warn).toBe("#E5A833");
  });

  it("exports radius tokens", () => {
    expect(TOKENS.radius.outer).toBe(12);
    expect(TOKENS.radius.inner).toBe(8);
    expect(TOKENS.radius.pill).toBe(9999);
    expect(TOKENS.radius.control).toBe(6);
  });

  it("exports motion tokens", () => {
    expect(TOKENS.motion.hoverMs).toBe(160);
    expect(TOKENS.motion.baseMs).toBe(220);
    expect(TOKENS.motion.easeOut).toContain("cubic-bezier");
  });

  it("exports type tokens", () => {
    expect(TOKENS.type.ui).toBe("Inter");
    expect(TOKENS.type.mono).toBe("JetBrainsMono");
    expect(TOKENS.type.math).toBe("SVG");
  });

  it("is deeply frozen (immutable at compile time via as const)", () => {
    const keys = Object.keys(TOKENS);
    expect(keys).toContain("color");
    expect(keys).toContain("radius");
    expect(keys).toContain("space");
    expect(keys).toContain("shadow");
    expect(keys).toContain("motion");
    expect(keys).toContain("type");
  });

  it("shadow tokens have correct format", () => {
    expect(TOKENS.shadow.raised).toContain("rgba");
    expect(TOKENS.shadow.floating).toContain("rgba");
  });

  it("space unit is 8", () => {
    expect(TOKENS.space.u).toBe(8);
  });
});
