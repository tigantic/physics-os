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
    { name: "chromium", use: { ...devices["Desktop Chrome"], reducedMotion: "reduce" } },
    { name: "firefox", use: { ...devices["Desktop Firefox"], reducedMotion: "reduce" } },
    { name: "mobile-chrome", use: { ...devices["Pixel 5"], reducedMotion: "reduce" } },
  ],
  webServer: {
    command: "pnpm dev",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
