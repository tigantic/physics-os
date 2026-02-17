import { test, expect } from "@playwright/test";

const MODES = ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"];

test.describe("gallery", () => {
  for (const width of [1440, 1728, 2560]) {
    test(`renders stable across modes at width ${width}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      for (const mode of MODES) {
        await page.goto(`/gallery?fixture=pass&mode=${mode}`);
        await expect(page.getByText("Luxury Physics Viewer")).toBeVisible();
        await expect(page).toHaveScreenshot(`gallery-${mode}-${width}.png`, {
          fullPage: true,
          maxDiffPixelRatio: 0.01,
        });
      }
    });

    test(`reduced motion stable at width ${width}`, async ({ page }) => {
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`/gallery?fixture=pass&mode=REVIEW`);
      await expect(page).toHaveScreenshot(`gallery-reduced-${width}.png`, {
        fullPage: true,
        maxDiffPixelRatio: 0.01,
      });
    });
  }

  test("tampered fixture shows BROKEN_CHAIN", async ({ page }) => {
    await page.goto("/gallery?fixture=tampered&mode=AUDIT");
    await expect(page.getByText("BROKEN_CHAIN")).toBeVisible();
  });

  test("missing values render Data Unavailable chip", async ({ page }) => {
    await page.goto("/gallery?fixture=incomplete&mode=REVIEW");
    await expect(page.getByText("Data Unavailable").first()).toBeVisible();
  });
});
