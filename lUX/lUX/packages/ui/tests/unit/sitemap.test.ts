import { describe, it, expect } from "vitest";
import sitemap from "@/app/sitemap";

describe("sitemap", () => {
  it("returns array of URL entries", () => {
    const result = sitemap();
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeGreaterThan(0);
  });

  it("first entry is the base URL with priority 1", () => {
    const result = sitemap();
    const first = result[0];
    expect(first.priority).toBe(1);
    expect(first.url).toBe("http://localhost:3000");
  });

  it("contains entries for all package × mode combinations (5 × 4 = 20 + 1 base + 1 /packages index = 22 total)", () => {
    const result = sitemap();
    expect(result).toHaveLength(22);
  });

  it("all entries have lastModified date", () => {
    const result = sitemap();
    for (const entry of result) {
      expect(entry.lastModified).toBeInstanceOf(Date);
    }
  });
});
