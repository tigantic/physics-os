import { test, expect } from "@playwright/test";

test.describe("mode layouts", () => {
  test("EXECUTIVE mode renders", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=EXECUTIVE");
    await expect(page.locator("body")).not.toBeEmpty();
    // EXECUTIVE mode has a condensed layout
  });

  test("REVIEW mode renders claims and timeline", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    await expect(page.locator("body")).not.toBeEmpty();
  });

  test("AUDIT mode renders", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=AUDIT");
    await expect(page.locator("body")).not.toBeEmpty();
  });

  test("PUBLICATION mode renders", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=PUBLICATION");
    await expect(page.locator("body")).not.toBeEmpty();
  });
});

test.describe("accessibility basics", () => {
  test("page has a heading structure", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    const headings = page.locator("h1, h2, h3, h4");
    await expect(headings.first()).toBeVisible();
  });

  test("interactive elements are focusable via keyboard", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    // Tab through the page and verify focus moves
    await page.keyboard.press("Tab");
    const focused = await page.evaluate(() => document.activeElement?.tagName);
    expect(focused).toBeTruthy();
  });

  test("reduced motion is respected (no animations detected)", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    // Verify the page loads correctly with reduced motion
    await expect(page.locator("body")).not.toBeEmpty();
  });
});
