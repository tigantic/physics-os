import { test, expect } from "@playwright/test";

test.describe("accessibility", () => {
  test("skip-to-content link moves focus to main", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    // Tab into the page — skip link should be the first focusable element
    await page.keyboard.press("Tab");
    const skipLink = page.getByRole("link", { name: /skip to/i });
    await expect(skipLink).toBeFocused();
    await skipLink.click();
    const main = page.locator("#main-content");
    await expect(main).toBeFocused();
  });

  test("mode dial uses tab navigation pattern", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    const tablist = page.getByRole("tablist");
    await expect(tablist).toBeVisible();
    const tabs = tablist.getByRole("tab");
    await expect(tabs).toHaveCount(4);
    // Only the selected tab should have tabIndex 0
    const selectedTab = tablist.getByRole("tab", { selected: true });
    await expect(selectedTab).toHaveAttribute("tabindex", "0");
  });

  test("heading hierarchy starts at h1", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    const h1 = page.locator("h1");
    await expect(h1).toHaveCount(1);
    await expect(h1).toBeVisible();
  });
});
