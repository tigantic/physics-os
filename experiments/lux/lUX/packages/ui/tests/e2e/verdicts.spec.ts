import { test, expect } from "@playwright/test";

test.describe("verdict rendering", () => {
  test("pass fixture shows PASS + VERIFIED", async ({ page }) => {
    await page.goto("/gallery?fixture=pass&mode=REVIEW");
    await expect(page.getByText("PASS")).toBeVisible();
    await expect(page.getByText("VERIFIED")).toBeVisible();
  });

  test("fail fixture shows FAIL verdict", async ({ page }) => {
    await page.goto("/gallery?fixture=fail&mode=REVIEW");
    await expect(page.getByText("FAIL")).toBeVisible();
  });

  test("warn fixture shows WARN verdict", async ({ page }) => {
    await page.goto("/gallery?fixture=warn&mode=REVIEW");
    await expect(page.getByText("WARN")).toBeVisible();
  });

  test("incomplete fixture shows INCOMPLETE", async ({ page }) => {
    await page.goto("/gallery?fixture=incomplete&mode=REVIEW");
    await expect(page.getByText("INCOMPLETE")).toBeVisible();
  });

  test("tampered fixture shows BROKEN_CHAIN in all modes", async ({ page }) => {
    for (const mode of ["EXECUTIVE", "REVIEW", "AUDIT", "PUBLICATION"]) {
      await page.goto(`/gallery?fixture=tampered&mode=${mode}`);
      await expect(page.getByText("BROKEN_CHAIN")).toBeVisible();
    }
  });
});
