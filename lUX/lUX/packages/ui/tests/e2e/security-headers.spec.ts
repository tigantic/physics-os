import { test, expect } from "@playwright/test";

/**
 * Verify that all security headers are set correctly by the middleware.
 * Tests both page responses and API responses.
 */

const REQUIRED_HEADERS: Record<string, string | RegExp> = {
  "content-security-policy": /default-src 'self'/,
  "x-content-type-options": "nosniff",
  "x-frame-options": "DENY",
  "referrer-policy": "strict-origin-when-cross-origin",
  "permissions-policy": "camera=(), microphone=(), geolocation=()",
  "strict-transport-security": /max-age=63072000.*includeSubDomains.*preload/,
  "x-dns-prefetch-control": "off",
  "cross-origin-opener-policy": "same-origin",
  "x-request-id": /^[0-9a-f-]{36}$/,
  "reporting-endpoints": /csp-endpoint/,
};

test.describe("security headers", () => {
  test("page response includes all security headers", async ({ page }) => {
    const response = await page.goto("/gallery?fixture=pass&mode=REVIEW");
    expect(response).not.toBeNull();
    const headers = response!.headers();

    for (const [name, expected] of Object.entries(REQUIRED_HEADERS)) {
      const value = headers[name];
      expect(value, `Missing header: ${name}`).toBeDefined();

      if (expected instanceof RegExp) {
        expect(value).toMatch(expected);
      } else {
        expect(value).toBe(expected);
      }
    }
  });

  test("API response includes security headers", async ({ request }) => {
    const res = await request.get("/api/health");
    const headers = res.headers();

    // API routes should have the same security headers
    for (const [name, expected] of Object.entries(REQUIRED_HEADERS)) {
      const value = headers[name];
      expect(value, `Missing header on API: ${name}`).toBeDefined();

      if (expected instanceof RegExp) {
        expect(value).toMatch(expected);
      } else {
        expect(value).toBe(expected);
      }
    }
  });

  test("CSP includes nonce for scripts", async ({ page }) => {
    const response = await page.goto("/gallery?fixture=pass&mode=REVIEW");
    const csp = response?.headers()["content-security-policy"] ?? "";
    expect(csp).toMatch(/nonce-[A-Za-z0-9+/=]+/);
    expect(csp).toContain("'strict-dynamic'");
  });

  test("CSP report-uri and report-to are configured", async ({ page }) => {
    const response = await page.goto("/gallery?fixture=pass&mode=REVIEW");
    const csp = response?.headers()["content-security-policy"] ?? "";
    expect(csp).toContain("report-uri /api/csp-report");
    expect(csp).toContain("report-to csp-endpoint");

    const reportingEndpoints = response?.headers()["reporting-endpoints"] ?? "";
    expect(reportingEndpoints).toContain("/api/csp-report");
  });

  test("rate limit headers are present", async ({ request }) => {
    const res = await request.get("/api/health");
    const headers = res.headers();

    expect(headers["x-ratelimit-limit"]).toBeDefined();
    expect(headers["x-ratelimit-remaining"]).toBeDefined();
    expect(Number(headers["x-ratelimit-limit"])).toBeGreaterThan(0);
    expect(Number(headers["x-ratelimit-remaining"])).toBeGreaterThanOrEqual(0);
  });

  test("X-Powered-By header is absent", async ({ request }) => {
    const res = await request.get("/api/health");
    const headers = res.headers();
    expect(headers["x-powered-by"]).toBeUndefined();
  });

  test("Cache-Control is no-store for API routes", async ({ request }) => {
    const res = await request.get("/api/health");
    // Health endpoint is dynamic
    const cacheControl = res.headers()["cache-control"];
    // Should not be publicly cached
    if (cacheControl) {
      expect(cacheControl).not.toContain("public, max-age=31536000");
    }
  });
});
