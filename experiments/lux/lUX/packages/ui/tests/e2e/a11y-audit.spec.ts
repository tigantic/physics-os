import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Automated accessibility audit using axe-core.
 * Scans every mode × fixture combination for WCAG 2.1 AA violations.
 * Zero violations enforced in CI.
 */

const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as const;
const FIXTURES = ["pass", "fail"] as const;

for (const fixture of FIXTURES) {
  for (const mode of MODES) {
    test(`axe scan: fixture=${fixture} mode=${mode}`, async ({ page }) => {
      await page.goto(`/gallery?fixture=${fixture}&mode=${mode}`);
      // Wait for meaningful content to render
      await page.waitForLoadState("networkidle");
      await expect(page.locator("body")).not.toBeEmpty();

      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze();

      const violations = results.violations.map((v) => ({
        id: v.id,
        impact: v.impact,
        description: v.description,
        nodes: v.nodes.length,
        targets: v.nodes.slice(0, 3).map((n) => n.target.join(" > ")),
      }));

      expect(violations, `axe violations in ${fixture}/${mode}`).toEqual([]);
    });
  }
}

test("axe scan: 404 page", async ({ page }) => {
  await page.goto("/this-page-does-not-exist");
  await page.waitForLoadState("networkidle");

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();

  expect(results.violations).toEqual([]);
});

test("axe scan: gallery index (no fixture)", async ({ page }) => {
  await page.goto("/gallery");
  await page.waitForLoadState("networkidle");

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();

  expect(results.violations).toEqual([]);
});
