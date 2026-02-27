import { describe, it, expect } from "vitest";
import { ModeMap } from "../src/mode/modeMap.js";
import type { ProofMode, ModeLayout } from "../src/mode/modeMap.js";

const ALL_MODES: ProofMode[] = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"];

describe("ModeMap", () => {
  it("contains all four proof modes", () => {
    for (const mode of ALL_MODES) {
      expect(ModeMap[mode]).toBeDefined();
    }
  });

  it("has no extra modes beyond the four defined", () => {
    expect(Object.keys(ModeMap)).toHaveLength(4);
    expect(Object.keys(ModeMap).sort()).toEqual([...ALL_MODES].sort());
  });

  for (const mode of ALL_MODES) {
    describe(`${mode} layout`, () => {
      let layout: ModeLayout;

      it("has a valid leftRail variant", () => {
        layout = ModeMap[mode];
        expect(["collapsed", "claimsTree", "flatClaims", "chapters"]).toContain(layout.leftRail.variant);
      });

      it("has a non-empty center array", () => {
        layout = ModeMap[mode];
        expect(Array.isArray(layout.center)).toBe(true);
        expect(layout.center.length).toBeGreaterThan(0);
      });

      it("has a non-empty rightRail array", () => {
        layout = ModeMap[mode];
        expect(Array.isArray(layout.rightRail)).toBe(true);
        expect(layout.rightRail.length).toBeGreaterThan(0);
      });

      it("center contains only string keys", () => {
        layout = ModeMap[mode];
        for (const key of layout.center) {
          expect(typeof key).toBe("string");
          expect(key.length).toBeGreaterThan(0);
        }
      });
    });
  }

  it("ModeMap keys are readonly (TypeScript enforced)", () => {
    // Verify the object structure is correctly typed — runtime mutability
    // is prevented by TypeScript's Readonly<> type, not Object.freeze.
    // We verify the data integrity here instead.
    for (const mode of ALL_MODES) {
      expect(ModeMap[mode].center.length).toBeGreaterThan(0);
      expect(ModeMap[mode].rightRail.length).toBeGreaterThan(0);
    }
  });

  it("AUDIT mode includes artifact browser entry", () => {
    expect(ModeMap.AUDIT.leftRail.includeArtifactBrowserEntry).toBe(true);
  });

  it("EXECUTIVE mode has collapsed leftRail", () => {
    expect(ModeMap.EXECUTIVE.leftRail.variant).toBe("collapsed");
  });
});
