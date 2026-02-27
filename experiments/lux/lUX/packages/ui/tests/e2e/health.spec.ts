import { test, expect } from "@playwright/test";

test.describe("health endpoint", () => {
  test("GET /api/health returns 200 with status ok", async ({ request }) => {
    const res = await request.get("/api/health");
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.service).toBe("lux-proof-viewer");
    expect(typeof body.timestamp).toBe("string");
    expect(typeof body.uptime).toBe("number");
  });

  test("health response has correct content-type", async ({ request }) => {
    const res = await request.get("/api/health");
    const ct = res.headers()["content-type"] ?? "";
    expect(ct).toContain("application/json");
  });
});
