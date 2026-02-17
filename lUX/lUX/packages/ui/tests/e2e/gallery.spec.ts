import { test, expect } from "@playwright/test";
import crypto from "node:crypto";

function hash(buf: Buffer) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

const MODES = ["EXECUTIVE","REVIEW","AUDIT","PUBLICATION"];

test.describe("gallery", () => {
  for (const width of [1440, 1728, 2560]) {
    test(`renders stable across modes at width ${width}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      for (const mode of MODES) {
        await page.goto(`/gallery?fixture=pass&mode=${mode}`);
        await expect(page.getByText("Luxury Physics Viewer")).toBeVisible();
        const shot1 = await page.screenshot({ fullPage: true });
        await page.reload();
        const shot2 = await page.screenshot({ fullPage: true });
        expect(hash(Buffer.from(shot1))).toBe(hash(Buffer.from(shot2)));
      }
    });

    test(`reduced motion stable at width ${width}`, async ({ page }) => {
      await page.emulateMedia({ reducedMotion: "reduce" });
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`/gallery?fixture=pass&mode=REVIEW`);
      const s1 = await page.screenshot({ fullPage: true });
      await page.reload();
      const s2 = await page.screenshot({ fullPage: true });
      expect(hash(Buffer.from(s1))).toBe(hash(Buffer.from(s2)));
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
