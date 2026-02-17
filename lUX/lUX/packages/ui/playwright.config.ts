import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: process.env.CI ? [["html", { open: "never" }], ["list"]] : [["list"]],
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    viewport: { width: 1280, height: 720 },
    launchOptions: {
      args: ["--disable-gpu", "--use-gl=swiftshader", "--disable-dev-shm-usage", "--no-sandbox"],
    },
  },
  projects: [
    /* ── Desktop browsers ──────────────────────────────────────────── */
    { name: "chromium", use: { ...devices["Desktop Chrome"], reducedMotion: "reduce" } },
    { name: "firefox", use: { ...devices["Desktop Firefox"], reducedMotion: "reduce" } },

    /* ── Mobile / touch ────────────────────────────────────────────── */
    { name: "mobile-chrome", use: { ...devices["Pixel 5"], reducedMotion: "reduce" } },
    { name: "mobile-safari", use: { ...devices["iPhone 13"], reducedMotion: "reduce" } },

    /* ── Viewport matrix (chromium only, for screenshot / layout tests) */
    {
      name: "viewport-sm",
      use: { ...devices["Desktop Chrome"], viewport: { width: 640, height: 480 }, reducedMotion: "reduce" },
      grep: /@viewport/,
    },
    {
      name: "viewport-md",
      use: { ...devices["Desktop Chrome"], viewport: { width: 768, height: 1024 }, reducedMotion: "reduce" },
      grep: /@viewport/,
    },
    {
      name: "viewport-lg",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1024, height: 768 }, reducedMotion: "reduce" },
      grep: /@viewport/,
    },
    {
      name: "viewport-xl",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" },
      grep: /@viewport/,
    },
    {
      name: "viewport-2xl",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1920, height: 1080 }, reducedMotion: "reduce" },
      grep: /@viewport/,
    },

    /* ── Landscape mobile ──────────────────────────────────────────── */
    {
      name: "mobile-landscape",
      use: {
        ...devices["Pixel 5"],
        viewport: { width: 851, height: 393 },
        reducedMotion: "reduce",
      },
      grep: /@mobile/,
    },
  ],
  webServer: {
    command: "pnpm dev",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
