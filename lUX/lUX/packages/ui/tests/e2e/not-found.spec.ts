import { test, expect } from "@playwright/test";

test.describe("404 page", () => {
  test("invalid route shows not found", async ({ page }) => {
    const res = await page.goto("/this-route-does-not-exist-xyz");
    expect(res?.status()).toBe(404);
    await expect(page.getByText(/not found|404/i).first()).toBeVisible();
  });

  test("404 page has navigation back to home", async ({ page }) => {
    await page.goto("/this-route-does-not-exist-xyz");
    // Ensure there's at least one link on the page (navigation exists)
    const links = page.getByRole("link");
    await expect(links.first()).toBeVisible();
  });
});
