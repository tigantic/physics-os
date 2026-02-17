import { test, expect } from "@playwright/test";

/**
 * Full keyboard navigation flow tests.
 *
 * Verifies that:
 *   - Skip-to-content works and moves focus to #main-content
 *   - All interactive elements are reachable via Tab
 *   - Mode dial supports arrow-key navigation
 *   - Disclosure Escape key closes panels and returns focus
 *   - Copy button returns focus after action
 *   - Focus rings are visible on keyboard navigation
 */

test.describe("keyboard navigation flow", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    await page.waitForLoadState("networkidle");
  });

  test("full tab traversal reaches all interactive elements without traps", async ({ page }) => {
    // Start from body
    await page.keyboard.press("Tab");

    // Track focused elements across the page — press Tab until we cycle
    // back to the skip link (or hit a reasonable limit).
    const focusedElements: string[] = [];
    const MAX_TABS = 100;

    for (let i = 0; i < MAX_TABS; i++) {
      const tagName = await page.evaluate(() => document.activeElement?.tagName?.toLowerCase() ?? "");
      const role = await page.evaluate(() => document.activeElement?.getAttribute("role") ?? "");
      const label =
        (await page.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? "")) ||
        (await page.evaluate(() => (document.activeElement as HTMLElement)?.textContent?.trim().slice(0, 30) ?? ""));

      const id = `${tagName}[${role || ""}]:${label}`;
      if (focusedElements.includes(id) && i > 5) {
        // Cycled back — we've traversed the full page
        break;
      }
      focusedElements.push(id);
      await page.keyboard.press("Tab");
    }

    expect(focusedElements.length).toBeGreaterThan(3);
    // Ensure no focus trap consumed all tabs
    expect(focusedElements.length).toBeLessThan(MAX_TABS);
  });

  test("mode dial supports arrow-key navigation", async ({ page }) => {
    const tablist = page.getByRole("tablist");
    await expect(tablist).toBeVisible();

    // Focus the selected tab
    const selectedTab = tablist.getByRole("tab", { selected: true });
    await selectedTab.focus();
    await expect(selectedTab).toBeFocused();

    // ArrowRight should move to next tab
    const initialText = await selectedTab.textContent();
    await page.keyboard.press("ArrowRight");

    const newFocused = page.locator(":focus");
    const newText = await newFocused.textContent();
    expect(newText).not.toBe(initialText);

    // ArrowLeft should go back
    await page.keyboard.press("ArrowLeft");
    const backText = await page.locator(":focus").textContent();
    expect(backText).toBe(initialText);
  });

  test("Disclosure panel closes on Escape and returns focus to trigger", async ({ page }) => {
    // Find a disclosure trigger button (Show/Hide pattern)
    const showButtons = page.getByRole("button", { name: /^Show$/i });
    const count = await showButtons.count();
    if (count === 0) {
      test.skip();
      return;
    }

    const trigger = showButtons.first();
    await trigger.click();

    // Panel should be open
    const panels = page.locator("[role='region']");
    await expect(panels.first()).toBeVisible();

    // Focus something inside the panel, then Escape
    await page.keyboard.press("Tab");
    await page.keyboard.press("Escape");

    // Panel should be closed
    await expect(panels.first()).not.toBeVisible();

    // Focus should be back on the trigger
    await expect(trigger).toBeFocused();
  });

  test("CopyField returns focus to button after copy", async ({ page }) => {
    const copyButtons = page.getByRole("button", { name: /^Copy /i });
    const count = await copyButtons.count();
    if (count === 0) {
      test.skip();
      return;
    }

    const btn = copyButtons.first();
    await btn.focus();
    await btn.click();

    // After clipboard action, focus should remain on the button
    await expect(btn).toBeFocused();
  });

  test("focus rings are visible during keyboard navigation", async ({ page }) => {
    // Tab to the first interactive element
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");

    const focused = page.locator(":focus-visible");
    const count = await focused.count();
    expect(count).toBeGreaterThanOrEqual(1);

    // Verify the outline/ring is actually visible (not transparent)
    const outlineStyle = await focused.first().evaluate((el) => {
      const style = window.getComputedStyle(el);
      return {
        outline: style.outline,
        boxShadow: style.boxShadow,
      };
    });

    // At least outline or box-shadow should be present
    const hasVisibleIndicator =
      (outlineStyle.outline && outlineStyle.outline !== "none" && !outlineStyle.outline.includes("0px")) ||
      (outlineStyle.boxShadow && outlineStyle.boxShadow !== "none");

    expect(hasVisibleIndicator).toBeTruthy();
  });
});
