/**
 * Comprehensive axe-core devtools verification — scans every screen × mode ×
 * fixture combination for accessibility violations.
 *
 * This replaces the need for manual axe-core Chrome DevTools verification by
 * automating the exact same audit engine (@axe-core/playwright) across ALL
 * permutations of the UI.
 *
 * Covers ROADMAP item:
 *   - Phase 1: "Verify with axe-core devtools on every screen × mode combo"
 *
 * Run:
 *   pnpm test:e2e -- axe-full-matrix
 */
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as const;
const FIXTURES = ["pass", "tampered", "incomplete"] as const;

const AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"];

test.describe("axe-core full matrix — all modes × fixtures", () => {
  for (const fixture of FIXTURES) {
    for (const mode of MODES) {
      test(`${fixture} / ${mode} — zero a11y violations`, async ({ page }) => {
        await page.goto(`/gallery?fixture=${fixture}&mode=${mode}`, {
          waitUntil: "networkidle",
        });
        await page.waitForTimeout(300);

        const results = await new AxeBuilder({ page })
          .withTags(AXE_TAGS)
          .analyze();

        const violations = results.violations.map((v) => ({
          id: v.id,
          impact: v.impact,
          description: v.description,
          nodes: v.nodes.length,
          targets: v.nodes.slice(0, 3).map((n) => n.target.join(" > ")),
        }));

        if (violations.length > 0) {
          // eslint-disable-next-line no-console
          console.error(
            `[a11y] ${fixture}/${mode}: ${violations.length} violation(s)`,
            JSON.stringify(violations, null, 2),
          );
        }

        expect(
          results.violations,
          `Expected zero axe violations for ${fixture}/${mode}`,
        ).toHaveLength(0);
      });
    }
  }
});

test.describe("axe-core full matrix — static pages", () => {
  const STATIC_PAGES = [
    { name: "packages list", url: "/packages" },
    { name: "404 page", url: "/not-a-real-route" },
    { name: "gallery redirect", url: "/gallery" },
  ];

  for (const pg of STATIC_PAGES) {
    test(`${pg.name} — zero a11y violations`, async ({ page }) => {
      await page.goto(pg.url, { waitUntil: "networkidle" });
      await page.waitForTimeout(300);

      const results = await new AxeBuilder({ page })
        .withTags(AXE_TAGS)
        .analyze();

      expect(
        results.violations,
        `Expected zero axe violations on ${pg.name}`,
      ).toHaveLength(0);
    });
  }
});

test.describe("axe-core full matrix — mobile viewports", () => {
  for (const mode of MODES) {
    test(`mobile (375px) ${mode} — zero a11y violations`, async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 });
      await page.goto(`/gallery?fixture=pass&mode=${mode}`, {
        waitUntil: "networkidle",
      });
      await page.waitForTimeout(300);

      const results = await new AxeBuilder({ page })
        .withTags(AXE_TAGS)
        .analyze();

      expect(
        results.violations,
        `Expected zero axe violations for mobile ${mode}`,
      ).toHaveLength(0);
    });
  }
});

test.describe("axe-core — interactive states", () => {
  test("disclosure open state has no violations", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=AUDIT", {
      waitUntil: "networkidle",
    });
    await page.waitForTimeout(300);

    // Open all visible disclosure panels
    const showButtons = page.getByRole("button", { name: /show/i });
    const count = await showButtons.count();
    for (let i = 0; i < Math.min(count, 5); i++) {
      await showButtons.nth(i).click();
    }
    await page.waitForTimeout(200);

    const results = await new AxeBuilder({ page })
      .withTags(AXE_TAGS)
      .analyze();

    expect(
      results.violations,
      "Expected zero axe violations with disclosures open",
    ).toHaveLength(0);
  });

  test("mobile drawer open state has no violations", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/gallery?fixture=pass&mode=EXECUTIVE", {
      waitUntil: "networkidle",
    });
    await page.waitForTimeout(300);

    const hamburger = page
      .getByRole("button", { name: /menu|navigation/i })
      .first();
    if (await hamburger.isVisible()) {
      await hamburger.click();
      await page.waitForTimeout(300);
    }

    const results = await new AxeBuilder({ page })
      .withTags(AXE_TAGS)
      .analyze();

    expect(
      results.violations,
      "Expected zero axe violations with mobile drawer open",
    ).toHaveLength(0);
  });
});
