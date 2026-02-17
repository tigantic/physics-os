import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    viewport: { width: 1280, height: 720 },
    launchOptions: {
      args: [
        "--disable-gpu",
        "--use-gl=swiftshader",
        "--disable-dev-shm-usage",
        "--no-sandbox"
      ]
    }
  },
  projects: [
    { name: "reduced-motion", use: { ...devices["Desktop Chrome"], reducedMotion: "reduce" } },
    { name: "normal-motion", use: { ...devices["Desktop Chrome"], reducedMotion: "no-preference" } }
  ],
  webServer: {
    command: "pnpm dev",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000
  }
});
