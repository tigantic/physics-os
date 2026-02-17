import { describe, it, expect, vi } from "vitest";

// Mock server-only (required by provider.ts imported from health route)
vi.mock("server-only", () => ({}));

vi.mock("@/config/provider", () => ({
  isProviderReady: () => true,
}));

import { GET } from "@/app/api/health/route";

describe("GET /api/health", () => {
  it("returns 200 status", async () => {
    const response = GET();
    expect(response.status).toBe(200);
  });

  it("response body has status: 'ok'", async () => {
    const response = GET();
    const body = await response.json();
    expect(body.status).toBe("ok");
  });

  it("response body has service name", async () => {
    const response = GET();
    const body = await response.json();
    expect(body.service).toBe("lux-proof-viewer");
  });

  it("response body has timestamp", async () => {
    const response = GET();
    const body = await response.json();
    expect(typeof body.timestamp).toBe("string");
    expect(() => new Date(body.timestamp)).not.toThrow();
  });

  it("response body has uptime", async () => {
    const response = GET();
    const body = await response.json();
    expect(typeof body.uptime).toBe("number");
    expect(body.uptime).toBeGreaterThanOrEqual(0);
  });

  it("response body has version and commitSha", async () => {
    const response = GET();
    const body = await response.json();
    expect(body.version).toBeDefined();
    expect(body.commitSha).toBeDefined();
  });

  it("response body has provider readiness", async () => {
    const response = GET();
    const body = await response.json();
    expect(body.provider.ready).toBe(true);
  });

  it("response body has memory stats", async () => {
    const response = GET();
    const body = await response.json();
    expect(typeof body.memory.heapUsed).toBe("number");
    expect(typeof body.memory.rss).toBe("number");
  });
});
