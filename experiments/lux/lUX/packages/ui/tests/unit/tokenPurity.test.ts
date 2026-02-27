import { describe, it, expect } from "vitest";
import fs from "node:fs";
import path from "node:path";

function scanDir(dir: string): string[] {
  const out: string[] = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...scanDir(p));
    else if (p.endsWith(".ts") || p.endsWith(".tsx") || p.endsWith(".css")) out.push(p);
  }
  return out;
}

describe("token purity", () => {
  it("contains no hardcoded colors in src (excluding generated token outputs)", () => {
    const srcDir = path.resolve(__dirname, "../../src");
    const files = scanDir(srcDir).filter(
      (f) =>
        !f.endsWith("tokens.css") &&
        !f.endsWith("tokens.ts") &&
        // global-error.tsx must use inline hex since root layout CSS may not be loaded
        !f.endsWith("global-error.tsx") &&
        // layout.tsx themeColor requires raw hex for Next.js Viewport metadata API
        !f.endsWith("layout.tsx") &&
        // CriticalCSS.tsx intentionally inlines raw hex tokens to prevent FOUC
        !f.endsWith("CriticalCSS.tsx"),
    );
    const bad: Array<{ file: string; match: string }> = [];
    const re = /(#[0-9a-fA-F]{3,8})|\brgb\(|\brgba\(|\bhsl\(|\bhsla\(/g;
    for (const f of files) {
      const s = fs.readFileSync(f, "utf8");
      const m = s.match(re);
      if (m) bad.push({ file: f, match: m[0] });
    }
    expect(bad, JSON.stringify(bad, null, 2)).toEqual([]);
  });
});
