import { describe, it, expect } from "vitest";
import { TOKENS } from "@/ds/tokens";

/**
 * Validates that responsive/fluid design tokens exist and are well-formed.
 */
describe("responsive tokens", () => {
  describe("fluid type scale", () => {
    const FLUID_TOKENS = [
      { key: "fluidXs" as const, minRem: 0.6875, maxRem: 0.75 },
      { key: "fluidSm" as const, minRem: 0.75, maxRem: 0.875 },
      { key: "fluidBase" as const, minRem: 0.875, maxRem: 1 },
      { key: "fluidLg" as const, minRem: 1, maxRem: 1.125 },
      { key: "fluidXl" as const, minRem: 1.125, maxRem: 1.25 },
    ];

    for (const { key, minRem, maxRem } of FLUID_TOKENS) {
      it(`${key} is a valid clamp() value`, () => {
        const value = TOKENS.type[key];
        expect(value).toMatch(/^clamp\(.+\)$/);
      });

      it(`${key} min (${minRem}rem) < max (${maxRem}rem)`, () => {
        expect(minRem).toBeLessThan(maxRem);
      });

      it(`${key} contains viewport-relative unit (vw)`, () => {
        const value = TOKENS.type[key];
        expect(value).toContain("vw");
      });
    }
  });

  describe("motion tokens", () => {
    it("fastMs < baseMs", () => {
      expect(TOKENS.motion.fastMs).toBeLessThan(TOKENS.motion.baseMs);
    });

    it("fastMs >= 100ms for perceptibility", () => {
      expect(TOKENS.motion.fastMs).toBeGreaterThanOrEqual(100);
    });

    it("baseMs <= 400ms for responsiveness", () => {
      expect(TOKENS.motion.baseMs).toBeLessThanOrEqual(400);
    });
  });
});
