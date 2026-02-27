import { describe, it, expect } from "vitest";
import robots from "@/app/robots";

describe("robots", () => {
  it('returns rules with userAgent "*"', () => {
    const result = robots();
    const rules = result.rules;
    if (Array.isArray(rules)) {
      expect(rules.some((r) => r.userAgent === "*")).toBe(true);
    } else {
      expect(rules.userAgent).toBe("*");
    }
  });

  it('returns allow "/"', () => {
    const result = robots();
    const rules = result.rules;
    if (Array.isArray(rules)) {
      const rule = rules.find((r) => r.userAgent === "*");
      expect(rule?.allow).toBe("/");
    } else {
      expect(rules.allow).toBe("/");
    }
  });

  it("has sitemap URL containing /sitemap.xml", () => {
    const result = robots();
    expect(result.sitemap).toContain("/sitemap.xml");
  });
});
