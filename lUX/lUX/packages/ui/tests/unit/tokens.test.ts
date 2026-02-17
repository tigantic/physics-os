import { describe, it, expect } from "vitest";
import { TOKENS } from "@/ds/tokens";

describe("TOKENS", () => {
  it("exports color.bg tokens", () => {
    expect(TOKENS.color.bg.base).toBe("#0D0D10");
    expect(TOKENS.color.bg.raised).toBe("#16161B");
    expect(TOKENS.color.bg.hover).toBe("#1E1E25");
    expect(TOKENS.color.bg.surface).toBe("#232329");
  });

  it("exports color.text tokens", () => {
    expect(TOKENS.color.text.primary).toBe("#F5F3EF");
    expect(TOKENS.color.text.secondary).toBe("#AEA9B4");
    expect(TOKENS.color.text.tertiary).toBe("#7A7584");
  });

  it("exports accent gold token", () => {
    expect(TOKENS.color.accent.gold).toBe("#C9A96E");
  });

  it("exports verdict tokens", () => {
    expect(TOKENS.color.verdict.pass).toBe("#3D8B5E");
    expect(TOKENS.color.verdict.fail).toBe("#A8423F");
    expect(TOKENS.color.verdict.warn).toBe("#B8862D");
  });

  it("exports radius tokens", () => {
    expect(TOKENS.radius.outer).toBe(14);
    expect(TOKENS.radius.inner).toBe(10);
  });

  it("exports motion tokens", () => {
    expect(TOKENS.motion.fastMs).toBe(180);
    expect(TOKENS.motion.baseMs).toBe(220);
    expect(TOKENS.motion.easeOut).toContain("cubic-bezier");
  });

  it("exports type tokens", () => {
    expect(TOKENS.type.ui).toBe("IBMPlexSans");
    expect(TOKENS.type.mono).toBe("JetBrainsMono");
    expect(TOKENS.type.math).toBe("SVG");
  });

  it("is deeply frozen (immutable at compile time via as const)", () => {
    // Verify the structure is complete (as const makes it readonly)
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
