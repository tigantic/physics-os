/**
 * Visual regression tests — baseline screenshots for every mode at every
 * supported viewport breakpoint, plus reduced-motion variants.
 *
 * Run with `--update-snapshots` to regenerate baselines:
 *   pnpm test:e2e -- --update-snapshots visual-regression
 *
 * Covers ROADMAP items:
 *   - Phase 1: "Visual regression: update all Playwright screenshot baselines"
 *   - Phase 2: "Update all Playwright screenshots (animations disabled via reduced-motion)"
 */
import { test, expect } from "@playwright/test";

const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as const;

const VIEWPORTS = [
  { label: "mobile-xs", width: 375, height: 812 },
  { label: "mobile-sm", width: 428, height: 926 },
  { label: "sm", width: 640, height: 900 },
  { label: "md", width: 768, height: 1024 },
  { label: "lg", width: 1024, height: 768 },
  { label: "xl", width: 1280, height: 900 },
  { label: "xxl", width: 1440, height: 900 },
  { label: "ultrawide", width: 1728, height: 900 },
  { label: "4k", width: 2560, height: 1440 },
] as const;

const FIXTURES = ["pass", "tampered", "incomplete"] as const;

test.describe("visual regression — all modes × viewports", () => {
  for (const vp of VIEWPORTS) {
    for (const mode of MODES) {
      test(`${mode} @ ${vp.label} (${vp.width}×${vp.height})`, async ({ page }) => {
        await page.setViewportSize({ width: vp.width, height: vp.height });
        await page.goto(`/gallery?fixture=pass&mode=${mode}`, {
          waitUntil: "networkidle",
        });

        // Wait for any CSS transitions to settle (reduced-motion is on by default
        // in playwright.config.ts, but give 200ms for any layout shifts)
        await page.waitForTimeout(200);

        await expect(page).toHaveScreenshot(
          `${mode.toLowerCase()}-${vp.label}.png`,
          { fullPage: true, maxDiffPixelRatio: 0.01 },
        );
      });
    }
  }
});

test.describe("visual regression — fixture variants", () => {
  for (const fixture of FIXTURES) {
    for (const mode of MODES) {
      test(`${fixture}/${mode} @ lg`, async ({ page }) => {
        await page.setViewportSize({ width: 1280, height: 900 });
        await page.goto(`/gallery?fixture=${fixture}&mode=${mode}`, {
          waitUntil: "networkidle",
        });
        await page.waitForTimeout(200);

        await expect(page).toHaveScreenshot(
          `fixture-${fixture}-${mode.toLowerCase()}.png`,
          { fullPage: true, maxDiffPixelRatio: 0.01 },
        );
      });
    }
  }
});

test.describe("visual regression — reduced-motion confirmation", () => {
  // Confirm animations are properly disabled across all viewports.
  // All projects already set reducedMotion: "reduce" in playwright.config.ts —
  // these tests explicitly verify the visual stability under that mode.
  for (const vp of [VIEWPORTS[0], VIEWPORTS[4], VIEWPORTS[6]] as const) {
    test(`reduced-motion stable @ ${vp.label}`, async ({ page }) => {
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.goto("/gallery?fixture=pass&mode=EXECUTIVE", {
        waitUntil: "networkidle",
      });
      await page.waitForTimeout(200);

      await expect(page).toHaveScreenshot(
        `reduced-motion-${vp.label}.png`,
        { fullPage: true, maxDiffPixelRatio: 0.01 },
      );
    });
  }
});

test.describe("visual regression — interactive states", () => {
  test("disclosure open/closed states", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/gallery?fixture=pass&mode=AUDIT", {
      waitUntil: "networkidle",
    });
    await page.waitForTimeout(200);

    // Screenshot with disclosures closed
    await expect(page).toHaveScreenshot("disclosure-closed.png", {
      fullPage: true,
      maxDiffPixelRatio: 0.01,
    });

    // Open first disclosure if present
    const showBtn = page.getByRole("button", { name: /show/i }).first();
    if (await showBtn.isVisible()) {
      await showBtn.click();
      await page.waitForTimeout(200);
      await expect(page).toHaveScreenshot("disclosure-open.png", {
        fullPage: true,
        maxDiffPixelRatio: 0.01,
      });
    }
  });

  test("hover states on cards", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/gallery?fixture=pass&mode=EXECUTIVE", {
      waitUntil: "networkidle",
    });
    await page.waitForTimeout(200);

    const card = page.locator('[class*="Card"]').first();
    if (await card.isVisible()) {
      await card.hover();
      await page.waitForTimeout(200);
      await expect(page).toHaveScreenshot("card-hover.png", {
        fullPage: true,
        maxDiffPixelRatio: 0.02,
      });
    }
  });

  test("mobile drawer open", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/gallery?fixture=pass&mode=EXECUTIVE", {
      waitUntil: "networkidle",
    });
    await page.waitForTimeout(200);

    // Open the hamburger menu / mobile drawer
    const hamburger = page
      .getByRole("button", { name: /menu|navigation/i })
      .first();
    if (await hamburger.isVisible()) {
      await hamburger.click();
      await page.waitForTimeout(300);
      await expect(page).toHaveScreenshot("mobile-drawer-open.png", {
        fullPage: true,
        maxDiffPixelRatio: 0.01,
      });
    }
  });
});

test.describe("visual regression — error & empty states", () => {
  test("404 page", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/nonexistent-route", { waitUntil: "networkidle" });
    await page.waitForTimeout(200);

    await expect(page).toHaveScreenshot("404-page.png", {
      fullPage: true,
      maxDiffPixelRatio: 0.01,
    });
  });

  test("packages list", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/packages", { waitUntil: "networkidle" });
    await page.waitForTimeout(200);

    await expect(page).toHaveScreenshot("packages-list.png", {
      fullPage: true,
      maxDiffPixelRatio: 0.01,
    });
  });
});
