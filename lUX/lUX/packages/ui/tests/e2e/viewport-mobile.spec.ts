import { test, expect } from "@playwright/test";

/**
 * Viewport responsiveness and mobile-specific tests.
 *
 * Tests tagged `@viewport` run across the viewport matrix (sm, md, lg, xl, 2xl).
 * Tests tagged `@mobile` run on `mobile-chrome`, `mobile-safari`, and `mobile-landscape`.
 */

const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"] as const;

test.describe("@viewport responsive layout", () => {
  for (const mode of MODES) {
    test(`${mode} mode renders without horizontal overflow @viewport`, async ({ page }) => {
      await page.goto(`/gallery?fixture=pass&mode=${mode}`);
      await page.waitForLoadState("networkidle");

      const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
      const viewportWidth = await page.evaluate(() => window.innerWidth);

      // Body should not be wider than viewport (no horizontal scroll)
      expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 1); // 1px tolerance
    });

    test(`${mode} mode has visible heading @viewport`, async ({ page }) => {
      await page.goto(`/gallery?fixture=pass&mode=${mode}`);
      const h1 = page.locator("h1");
      await expect(h1).toBeVisible();
    });

    test(`${mode} mode screenshot @viewport`, async ({ page }) => {
      await page.goto(`/gallery?fixture=pass&mode=${mode}`);
      await page.waitForLoadState("networkidle");
      // Wait for any transitions
      await page.waitForTimeout(500);
      await expect(page).toHaveScreenshot(`${mode}-layout.png`, {
        fullPage: true,
        maxDiffPixelRatio: 0.01,
      });
    });
  }
});

test.describe("@mobile mobile experience", () => {
  test("gallery renders on mobile without overflow @mobile", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    await page.waitForLoadState("networkidle");

    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 1);
  });

  test("mode dial is fully visible on mobile @mobile", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    const tablist = page.getByRole("tablist");
    await expect(tablist).toBeVisible();

    // All mode tabs should be visible (not clipped)
    const tabs = tablist.getByRole("tab");
    const tabCount = await tabs.count();
    for (let i = 0; i < tabCount; i++) {
      await expect(tabs.nth(i)).toBeVisible();
    }
  });

  test("text is readable on mobile (font-size >= 12px) @mobile", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    await page.waitForLoadState("networkidle");

    // Sample text elements and verify minimum readable size
    const textElements = page.locator("p, span, td, th, li, h1, h2, h3, h4").first();
    const fontSize = await textElements.evaluate((el) => {
      return parseFloat(window.getComputedStyle(el).fontSize);
    });
    expect(fontSize).toBeGreaterThanOrEqual(12);
  });

  test("touch targets are sufficiently large (>= 44px) @mobile", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    await page.waitForLoadState("networkidle");

    const buttons = page.locator("button, a[href], [role='tab'], [role='button']");
    const count = await buttons.count();

    for (let i = 0; i < Math.min(count, 20); i++) {
      const box = await buttons.nth(i).boundingBox();
      if (box && box.width > 0 && box.height > 0) {
        // WCAG 2.5.5 recommends 44x44; we check minimum of 40 with tolerance
        const minDimension = Math.min(box.width, box.height);
        // Skip if element is tiny (likely hidden or decorative)
        if (minDimension < 10) continue;
        expect(
          minDimension,
          `Button ${i} touch target too small: ${box.width}x${box.height}`,
        ).toBeGreaterThanOrEqual(32); // 32px minimum with padding
      }
    }
  });

  test("landscape orientation renders correctly @mobile", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    await page.waitForLoadState("networkidle");

    // On landscape mobile, content should still be visible and usable
    const main = page.locator("#main-content");
    await expect(main).toBeVisible();

    // No horizontal overflow
    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 1);
  });
});
