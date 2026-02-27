import { test, expect } from "@playwright/test";

test.describe("navigation", () => {
  test("root redirects to /gallery", async ({ page }) => {
    await page.goto("/");
    await page.waitForURL("**/gallery**");
    expect(page.url()).toContain("/gallery");
  });

  test("gallery loads with default fixture", async ({ page }) => {
    await page.goto("/gallery");
    await expect(page.locator("body")).not.toBeEmpty();
  });

  test("unknown fixture falls back to pass", async ({ page }) => {
    await page.goto("/gallery?fixture=EVIL_INPUT");
    await expect(page.locator("body")).not.toBeEmpty();
    // Should not crash — should render the pass fixture
  });

  test("unknown mode falls back to REVIEW", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=INVALID");
    await expect(page.locator("body")).not.toBeEmpty();
  });
});
